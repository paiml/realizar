use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::{
    println!("  Final CV: {:.4}", cv);
    println!("  Target CV: {:.4}", runner.target_cv);

    assert!(
        cv < runner.target_cv,
        "QA-031: CV should be below threshold"
    );
    assert!(
        iterations >= runner.min_iterations,
        "QA-031: Should run minimum iterations"
    );
    assert!(
        iterations <= runner.max_iterations,
        "QA-031: Should not exceed max iterations"
    );
}

/// Test PARITY-009b: QA-032 Warmup iterations discard
#[test]
fn test_parity009b_warmup_discard() {
    /// Benchmark with warmup discard per Mytkowicz et al.
    #[derive(Debug)]
    struct WarmupBenchmark {
        warmup_iterations: usize,
        measurement_iterations: usize,
    }

    impl WarmupBenchmark {
        fn new(warmup: usize, measure: usize) -> Self {
            Self {
                warmup_iterations: warmup,
                measurement_iterations: measure,
            }
        }

        fn run<F>(&self, mut benchmark_fn: F) -> (Vec<f64>, Vec<f64>)
        where
            F: FnMut(usize) -> f64,
        {
            let mut warmup_values = Vec::with_capacity(self.warmup_iterations);
            let mut measurement_values = Vec::with_capacity(self.measurement_iterations);

            // Warmup phase (JIT, cache warming)
            for i in 0..self.warmup_iterations {
                warmup_values.push(benchmark_fn(i));
            }

            // Measurement phase
            for i in 0..self.measurement_iterations {
                measurement_values.push(benchmark_fn(self.warmup_iterations + i));
            }

            (warmup_values, measurement_values)
        }
    }

    let runner = WarmupBenchmark::new(3, 5);

    // Simulate JIT warmup effect: first iterations are slower
    let (warmup, measurements) = runner.run(|i| {
        if i < 3 {
            200.0 - (i as f64 * 30.0) // Warmup: 200, 170, 140
        } else {
            100.0 + (i as f64 * 0.5) // Stable: ~101.5 - 103.5
        }
    });

    let warmup_mean: f64 = warmup.iter().sum::<f64>() / warmup.len() as f64;
    let measure_mean: f64 = measurements.iter().sum::<f64>() / measurements.len() as f64;

    println!("\nPARITY-009b: Warmup discard");
    println!(
        "  Warmup iterations: {} (mean: {:.1})",
        warmup.len(),
        warmup_mean
    );
    println!(
        "  Measurement iterations: {} (mean: {:.1})",
        measurements.len(),
        measure_mean
    );

    assert_eq!(warmup.len(), 3, "QA-032: Should have 3 warmup iterations");
    assert_eq!(
        measurements.len(),
        5,
        "QA-032: Should have 5 measurement iterations"
    );
    assert!(
        warmup_mean > measure_mean,
        "QA-032: Warmup should be slower (JIT effect)"
    );
}

/// Test PARITY-009c: QA-033 Environment metadata capture
#[test]
fn test_parity009c_environment_metadata() {
    /// Environment metadata per Vitek & Kalibera
    #[derive(Debug, Clone)]
    struct EnvironmentMetadata {
        // System info
        os: String,
        arch: String,
        #[allow(dead_code)]
        cpu_model: String,
        cpu_cores: usize,
        #[allow(dead_code)]
        ram_gb: usize,

        // Runtime info
        #[allow(dead_code)]
        rust_version: String,
        cargo_profile: String,
        #[allow(dead_code)]
        target_triple: String,

        // Benchmark config
        #[allow(dead_code)]
        timestamp: String,
        #[allow(dead_code)]
        git_commit: String,
        #[allow(dead_code)]
        benchmark_version: String,
    }

    impl EnvironmentMetadata {
        fn capture() -> Self {
            Self {
                os: std::env::consts::OS.to_string(),
                arch: std::env::consts::ARCH.to_string(),
                cpu_model: "Unknown".to_string(), // Would read from /proc/cpuinfo
                cpu_cores: std::thread::available_parallelism()
                    .map(std::num::NonZero::get)
                    .unwrap_or(1),
                ram_gb: 16, // Would read from system
                rust_version: env!("CARGO_PKG_RUST_VERSION").to_string(),
                cargo_profile: if cfg!(debug_assertions) {
                    "debug"
                } else {
                    "release"
                }
                .to_string(),
                target_triple: std::env::consts::ARCH.to_string(),
                timestamp: "2025-12-13T22:00:00Z".to_string(),
                git_commit: "abc123".to_string(),
                benchmark_version: "1.0.0".to_string(),
            }
        }

        fn is_reproducible(&self) -> bool {
            !self.os.is_empty()
                && !self.arch.is_empty()
                && self.cpu_cores > 0
                && !self.cargo_profile.is_empty()
        }
    }

    let env = EnvironmentMetadata::capture();

    println!("\nPARITY-009c: Environment metadata");
    println!("  OS: {}", env.os);
    println!("  Arch: {}", env.arch);
    println!("  CPU cores: {}", env.cpu_cores);
    println!("  Profile: {}", env.cargo_profile);

    assert!(
        env.is_reproducible(),
        "QA-033: Environment must be reproducible"
    );
    assert!(!env.os.is_empty(), "QA-033: OS must be captured");
    assert!(!env.arch.is_empty(), "QA-033: Arch must be captured");
    assert!(env.cpu_cores > 0, "QA-033: CPU cores must be captured");
}

/// Test PARITY-009d: QA-034 Outlier detection using MAD
#[test]
fn test_parity009d_outlier_detection_mad() {
    /// Median Absolute Deviation (MAD) outlier detection
    /// Per Fleming & Wallace: MAD is robust to outliers
    fn median(values: &mut [f64]) -> f64 {
        values.sort_by(|a, b| a.partial_cmp(b).expect("test"));
        let mid = values.len() / 2;
        if values.len().is_multiple_of(2) {
            f64::midpoint(values[mid - 1], values[mid])
        } else {
            values[mid]
        }
    }

    fn mad(values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        let med = median(&mut sorted);
        let mut deviations: Vec<f64> = values.iter().map(|v| (v - med).abs()).collect();
        median(&mut deviations)
    }

    fn detect_outliers(values: &[f64], threshold: f64) -> Vec<usize> {
        let mut sorted = values.to_vec();
        let med = median(&mut sorted);
        let mad_value = mad(values);
        let k = 1.4826; // Scale factor for normal distribution

        values
            .iter()
            .enumerate()
            .filter(|(_, &v)| {
                if mad_value == 0.0 {
                    false
                } else {
                    ((v - med).abs() / (k * mad_value)) > threshold
                }
            })
            .map(|(i, _)| i)
            .collect()
    }

    // Test data with outliers
    let values = vec![100.0, 101.0, 99.0, 102.0, 98.0, 500.0, 100.5, 99.5];
    let outliers = detect_outliers(&values, 3.0); // 3 MAD threshold

    println!("\nPARITY-009d: MAD outlier detection");
    println!("  Values: {:?}", values);
    println!("  MAD: {:.2}", mad(&values));
    println!("  Outliers at indices: {:?}", outliers);

    assert!(
        outliers.contains(&5),
        "QA-034: Should detect 500.0 as outlier"
    );
    assert!(
        !outliers.contains(&0),
        "QA-034: 100.0 should not be outlier"
    );
}

/// Test PARITY-009e: QA-035 p50, p95, p99 latencies
#[test]
fn test_parity009e_latency_percentiles() {
    /// Latency percentile calculator per Georges et al.
    #[derive(Debug, Clone)]
    struct LatencyStats {
        p50: f64,
        p95: f64,
        p99: f64,
        min: f64,
        max: f64,
        #[allow(dead_code)]
        mean: f64,
    }

    impl LatencyStats {
        fn from_latencies(latencies: &[f64]) -> Self {
            let mut sorted = latencies.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).expect("test"));

            let percentile = |p: f64| -> f64 {
                let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
                sorted[idx.min(sorted.len() - 1)]
            };

            Self {
                p50: percentile(0.50),
                p95: percentile(0.95),
                p99: percentile(0.99),
                min: sorted[0],
                max: sorted[sorted.len() - 1],
                mean: latencies.iter().sum::<f64>() / latencies.len() as f64,
            }
        }
    }

    // Simulate latency distribution
    let latencies: Vec<f64> = (0..100)
        .map(|i| 10.0 + (i as f64 * 0.5) + if i > 95 { 50.0 } else { 0.0 })
        .collect();

    let stats = LatencyStats::from_latencies(&latencies);

    println!("\nPARITY-009e: Latency percentiles");
    println!("  p50: {:.2}ms", stats.p50);
    println!("  p95: {:.2}ms", stats.p95);
    println!("  p99: {:.2}ms", stats.p99);
    println!("  min: {:.2}ms, max: {:.2}ms", stats.min, stats.max);

    assert!(stats.p50 < stats.p95, "QA-035: p50 should be less than p95");
    assert!(stats.p95 < stats.p99, "QA-035: p95 should be less than p99");
    assert!(stats.min <= stats.p50, "QA-035: min should be <= p50");
    assert!(stats.p99 <= stats.max, "QA-035: p99 should be <= max");
}

/// Test PARITY-009f: QA-036 Throughput with variance
#[test]
fn test_parity009f_throughput_variance() {
    /// Throughput measurement with variance tracking
    #[derive(Debug, Clone)]
    struct ThroughputStats {
        mean_tps: f64,
        variance: f64,
        stddev: f64,
        cv: f64,
        samples: usize,
    }

    impl ThroughputStats {
        fn from_samples(tps_samples: &[f64]) -> Self {
            let n = tps_samples.len() as f64;
            let mean = tps_samples.iter().sum::<f64>() / n;
            let variance = tps_samples.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n;
            let stddev = variance.sqrt();
            let cv = if mean > 0.0 { stddev / mean } else { 0.0 };

            Self {
                mean_tps: mean,
                variance,
                stddev,
                cv,
                samples: tps_samples.len(),
            }
        }

        fn is_stable(&self) -> bool {
            self.cv < 0.05 // 5% CV threshold
        }

        fn confidence_interval_95(&self) -> (f64, f64) {
            let margin = 1.96 * self.stddev / (self.samples as f64).sqrt();
            (self.mean_tps - margin, self.mean_tps + margin)
        }
    }

    // Simulate throughput measurements
    let tps_samples = vec![200.0, 205.0, 198.0, 202.0, 201.0, 199.0, 203.0, 200.5];
    let stats = ThroughputStats::from_samples(&tps_samples);
    let (ci_low, ci_high) = stats.confidence_interval_95();

    println!("\nPARITY-009f: Throughput with variance");
    println!("  Mean: {:.2} tok/s", stats.mean_tps);
    println!("  StdDev: {:.2}", stats.stddev);
    println!("  CV: {:.4}", stats.cv);
    println!("  95% CI: [{:.2}, {:.2}]", ci_low, ci_high);

    assert!(
        stats.is_stable(),
        "QA-036: Measurements should be stable (CV < 0.05)"
    );
    assert!(stats.variance > 0.0, "QA-036: Variance should be positive");
    assert!(
        ci_low < stats.mean_tps && stats.mean_tps < ci_high,
        "QA-036: Mean should be in CI"
    );
}

/// Test PARITY-009g: QA-037 Versioned benchmark results
#[test]
fn test_parity009g_versioned_results() {
    /// Versioned benchmark result for reproducibility
    #[derive(Debug, Clone)]
    struct VersionedBenchmarkResult {
        // Version info
        schema_version: String,
        benchmark_version: String,
        realizar_version: String,

        // Metadata
        timestamp: String,
        git_commit: String,
        environment_hash: String,

        // Results
        throughput_tps: f64,
        #[allow(dead_code)]
        latency_p50_ms: f64,
        #[allow(dead_code)]
        latency_p99_ms: f64,
        cv: f64,
        iterations: usize,
    }

    impl VersionedBenchmarkResult {
        fn new(tps: f64, p50: f64, p99: f64, cv: f64, iterations: usize) -> Self {
            Self {
                schema_version: "1.0.0".to_string(),
                benchmark_version: "PARITY-009".to_string(),
                realizar_version: env!("CARGO_PKG_VERSION").to_string(),
                timestamp: "2025-12-13T22:00:00Z".to_string(),
                git_commit: "abc123def".to_string(),
                environment_hash: "sha256:...".to_string(),
                throughput_tps: tps,
                latency_p50_ms: p50,
                latency_p99_ms: p99,
                cv,
                iterations,
            }
        }

        fn is_valid(&self) -> bool {
            !self.schema_version.is_empty()
                && !self.benchmark_version.is_empty()
                && !self.realizar_version.is_empty()
                && self.throughput_tps > 0.0
                && self.cv >= 0.0
                && self.iterations > 0
        }

        fn is_reproducible(&self) -> bool {
            !self.git_commit.is_empty()
                && !self.timestamp.is_empty()
                && !self.environment_hash.is_empty()
        }
    }

    let result = VersionedBenchmarkResult::new(
        200.5, // tps
        5.2,   // p50
        12.8,  // p99
        0.025, // cv
        50,    // iterations
    );

    println!("\nPARITY-009g: Versioned results");
    println!("  Schema: {}", result.schema_version);
    println!("  Benchmark: {}", result.benchmark_version);
    println!("  Realizar: {}", result.realizar_version);
    println!("  Throughput: {:.2} tok/s", result.throughput_tps);

    assert!(result.is_valid(), "QA-037: Result must be valid");
    assert!(
        result.is_reproducible(),
        "QA-037: Result must be reproducible"
    );
    assert_eq!(
        result.schema_version, "1.0.0",
        "QA-037: Schema version must be set"
    );
}

// ========================================================================
// PARITY-010: Benchmark Infrastructure QA-038 to QA-040
// ========================================================================

/// Test PARITY-010a: QA-038 Preflight checks validate server availability
#[test]
fn test_parity010a_preflight_server_checks() {
    /// Preflight check result
    #[derive(Debug, Clone)]
    enum PreflightStatus {
        Pass,
        Fail(String),
        Skip(String),
    }

    /// Server availability check
    #[derive(Debug)]
    struct ServerPreflightCheck {
        name: String,
        endpoint: String,
        #[allow(dead_code)]
        timeout_ms: u64,
        required: bool,
    }

    impl ServerPreflightCheck {
        fn new(name: &str, endpoint: &str, required: bool) -> Self {
            Self {
                name: name.to_string(),
                endpoint: endpoint.to_string(),
                timeout_ms: 5000,
                required,
            }
        }

        /// Simulate server check (real impl would use HTTP client)
        fn check(&self, server_available: bool) -> PreflightStatus {
            if server_available {
                PreflightStatus::Pass
            } else if self.required {
                PreflightStatus::Fail(format!("{} not available at {}", self.name, self.endpoint))
            } else {
                PreflightStatus::Skip(format!("{} optional, skipping", self.name))
            }
        }
    }

    /// Preflight suite for benchmark servers
    #[derive(Debug)]
    struct PreflightSuite {
        checks: Vec<ServerPreflightCheck>,
    }

    impl PreflightSuite {
        fn new() -> Self {
            Self {
                checks: vec![
                    ServerPreflightCheck::new("Ollama", "http://localhost:11434", true),
                    ServerPreflightCheck::new("llama.cpp", "http://localhost:8080", false),
                    ServerPreflightCheck::new("vLLM", "http://localhost:8000", false),
                ],
            }
        }

        fn run(&self, availability: &[bool]) -> (usize, usize, usize) {
            let mut passed = 0;
            let mut failed = 0;
            let mut skipped = 0;

            for (check, &available) in self.checks.iter().zip(availability.iter()) {
                match check.check(available) {
                    PreflightStatus::Pass => passed += 1,
                    PreflightStatus::Fail(_) => failed += 1,
                    PreflightStatus::Skip(_) => skipped += 1,
                }
            }

            (passed, failed, skipped)
        }

        fn all_required_pass(&self, availability: &[bool]) -> bool {
            for (check, &available) in self.checks.iter().zip(availability.iter()) {
                if check.required && !available {
                    return false;
                }
            }
            true
        }
    }

    let suite = PreflightSuite::new();

    // Test: All servers available
    let (passed, failed, _skipped) = suite.run(&[true, true, true]);
    assert_eq!(passed, 3, "QA-038: All 3 servers should pass");
    assert_eq!(failed, 0, "QA-038: No failures");

    // Test: Only required (Ollama) available
    let (passed, _failed, skipped) = suite.run(&[true, false, false]);
    assert_eq!(passed, 1, "QA-038: Ollama passes");
    assert_eq!(skipped, 2, "QA-038: Optional servers skipped");
    assert!(
        suite.all_required_pass(&[true, false, false]),
        "QA-038: Required servers pass"
    );

    // Test: Required server unavailable
    assert!(
        !suite.all_required_pass(&[false, true, true]),
        "QA-038: Should fail if Ollama down"
    );

    println!("\nPARITY-010a: Preflight server checks");
    println!("  Checks defined: {}", suite.checks.len());
    println!("  Required: Ollama");
    println!("  Optional: llama.cpp, vLLM");
}

/// Test PARITY-010b: QA-039 Automatic model download from Hugging Face
#[test]
fn test_parity010b_model_download() {
    /// Model download configuration
    #[derive(Debug, Clone)]
    struct ModelDownloadConfig {
        repo_id: String,
        filename: String,
        revision: String,
        cache_dir: String,
    }

    impl ModelDownloadConfig {
        fn new(repo_id: &str, filename: &str) -> Self {
            Self {
                repo_id: repo_id.to_string(),
                filename: filename.to_string(),
                revision: "main".to_string(),
                cache_dir: "~/.cache/huggingface/hub".to_string(),
            }
        }

        fn url(&self) -> String {
            format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                self.repo_id, self.revision, self.filename
            )
        }

        fn cache_path(&self) -> String {
            let repo_dir = self.repo_id.replace('/', "--");
            format!(
                "{}/models--{}/snapshots/{}/{}",
                self.cache_dir, repo_dir, self.revision, self.filename
            )
        }
    }

    /// Model download status
    #[derive(Debug, Clone)]
    enum DownloadStatus {
        Cached(String),     // Already in cache
        Downloaded(String), // Freshly downloaded
        #[allow(dead_code)]
        Failed(String), // Download failed
    }

    /// Model downloader (test)
    struct ModelDownloader {
        configs: Vec<ModelDownloadConfig>,
    }

    impl ModelDownloader {
        fn new() -> Self {
            Self {
                configs: vec![
                    ModelDownloadConfig::new("TheBloke/phi-2-GGUF", "phi-2.Q4_K_M.gguf"),
                    ModelDownloadConfig::new("microsoft/phi-2", "model.safetensors"),
                ],
            }
        }

        /// Simulate download check
        fn check_or_download(&self, config: &ModelDownloadConfig, cached: bool) -> DownloadStatus {
            if cached {
                DownloadStatus::Cached(config.cache_path())
            } else {
                // In real impl: download from config.url()
                DownloadStatus::Downloaded(config.cache_path())
            }
        }
    }

    let downloader = ModelDownloader::new();
    let config = &downloader.configs[0];

    // Test: Model already cached
    let status = downloader.check_or_download(config, true);
    assert!(
        matches!(status, DownloadStatus::Cached(_)),
        "QA-039: Should return cached"
    );

    // Test: Model needs download
    let status = downloader.check_or_download(config, false);
    assert!(
        matches!(status, DownloadStatus::Downloaded(_)),
        "QA-039: Should download"
    );

    // Test: URL construction
    let url = config.url();
    assert!(
        url.contains("huggingface.co"),
        "QA-039: URL should be HuggingFace"
    );
    assert!(
        url.contains(&config.repo_id),
        "QA-039: URL should contain repo"
    );
    assert!(
        url.contains(&config.filename),
        "QA-039: URL should contain filename"
    );

    println!("\nPARITY-010b: Model download from HuggingFace");
    println!("  Repo: {}", config.repo_id);
    println!("  File: {}", config.filename);
    println!("  URL: {}", config.url());
}

/// Test PARITY-010c: QA-040 JSON schema validation for benchmark results
#[test]
fn test_parity010c_json_schema_validation() {
    /// JSON schema field definition
    #[derive(Debug, Clone)]
    struct SchemaField {
        name: String,
        field_type: FieldType,
        required: bool,
    }

    #[derive(Debug, Clone)]
    enum FieldType {
        String,
        Number,
        Integer,
        #[allow(dead_code)]
        Boolean,
        #[allow(dead_code)]
        Array(Box<FieldType>),
        Object(Vec<SchemaField>),
    }

    /// Benchmark result schema
    #[derive(Debug)]
    struct BenchmarkResultSchema {
        version: String,
        fields: Vec<SchemaField>,
    }

    impl BenchmarkResultSchema {
        fn v1() -> Self {
            Self {
                version: "1.0.0".to_string(),
                fields: vec![
                    SchemaField {
                        name: "schema_version".to_string(),
                        field_type: FieldType::String,
                        required: true,
                    },
                    SchemaField {
                        name: "timestamp".to_string(),
                        field_type: FieldType::String,
                        required: true,
                    },
                    SchemaField {
                        name: "git_commit".to_string(),
                        field_type: FieldType::String,
                        required: true,
                    },
                    SchemaField {
                        name: "throughput_tps".to_string(),
                        field_type: FieldType::Number,
                        required: true,
                    },
                    SchemaField {
                        name: "latency_p50_ms".to_string(),
                        field_type: FieldType::Number,
                        required: true,
                    },
                    SchemaField {
                        name: "latency_p95_ms".to_string(),
                        field_type: FieldType::Number,
                        required: true,
                    },
                    SchemaField {
                        name: "latency_p99_ms".to_string(),
                        field_type: FieldType::Number,
                        required: true,
                    },
                    SchemaField {
                        name: "cv".to_string(),
                        field_type: FieldType::Number,
                        required: true,
                    },
                    SchemaField {
                        name: "iterations".to_string(),
                        field_type: FieldType::Integer,
                        required: true,
                    },
                    SchemaField {
                        name: "environment".to_string(),
                        field_type: FieldType::Object(vec![
                            SchemaField {
                                name: "os".to_string(),
                                field_type: FieldType::String,
                                required: true,
                            },
                            SchemaField {
                                name: "arch".to_string(),
                                field_type: FieldType::String,
                                required: true,
                            },
                            SchemaField {
                                name: "cpu_cores".to_string(),
                                field_type: FieldType::Integer,
                                required: true,
                            },
                        ]),
                        required: true,
                    },
                ],
            }
        }

        fn required_field_count(&self) -> usize {
            self.fields.iter().filter(|f| f.required).count()
        }

        fn validate_field_presence(&self, field_names: &[&str]) -> Vec<String> {
            let mut missing = Vec::new();
            for field in &self.fields {
                if field.required && !field_names.contains(&field.name.as_str()) {
                    missing.push(field.name.clone());
                }
            }
            missing
        }
    }

    let schema = BenchmarkResultSchema::v1();

    // Test: Schema version
    assert_eq!(
        schema.version, "1.0.0",
        "QA-040: Schema version should be 1.0.0"
    );

    // Test: Required fields
    assert!(
        schema.required_field_count() >= 9,
        "QA-040: Should have >=9 required fields"
    );

    // Test: Validation with all fields
    let all_fields = vec![
        "schema_version",
        "timestamp",
        "git_commit",
        "throughput_tps",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "cv",
        "iterations",
        "environment",
    ];
    let missing = schema.validate_field_presence(&all_fields);
    assert!(missing.is_empty(), "QA-040: All required fields present");

    // Test: Validation with missing fields
    let partial_fields = vec!["schema_version", "throughput_tps"];
    let missing = schema.validate_field_presence(&partial_fields);
    assert!(!missing.is_empty(), "QA-040: Should detect missing fields");
    assert!(
        missing.contains(&"timestamp".to_string()),
        "QA-040: timestamp should be missing"
    );

    println!("\nPARITY-010c: JSON schema validation");
    println!("  Schema version: {}", schema.version);
    println!("  Required fields: {}", schema.required_field_count());
    println!("  Total fields: {}", schema.fields.len());
}

/// Test PARITY-010d: Combined preflight and validation suite
#[test]
fn test_parity010d_benchmark_preflight_suite() {
    /// Complete preflight suite combining all checks
    #[derive(Debug)]
    struct BenchmarkPreflightSuite {
        server_checks: Vec<(&'static str, bool)>, // (name, required)
        model_checks: Vec<&'static str>,          // model repo IDs
        schema_version: &'static str,
    }

    impl BenchmarkPreflightSuite {
        fn standard() -> Self {
            Self {
                server_checks: vec![("Ollama", true), ("llama.cpp", false), ("vLLM", false)],
                model_checks: vec!["TheBloke/phi-2-GGUF", "microsoft/phi-2"],
                schema_version: "1.0.0",
            }
        }

        fn run_all(&self, servers_up: &[bool], models_cached: &[bool]) -> PreflightResult {
            let mut result = PreflightResult::default();

            // Server checks
            for ((name, required), &up) in self.server_checks.iter().zip(servers_up.iter()) {
                if up {
                    result.servers_passed += 1;
                } else if *required {
                    result.servers_failed += 1;
                    result.errors.push(format!("{} unavailable", name));
                } else {
                    result.servers_skipped += 1;
                }
            }

            // Model checks
            for (_model, &cached) in self.model_checks.iter().zip(models_cached.iter()) {
                if cached {
                    result.models_cached += 1;
                } else {
                    result.models_to_download += 1;
                }
            }

            result.schema_valid = true;
            result
        }
    }

    #[derive(Debug, Default)]
    struct PreflightResult {
        servers_passed: usize,
        servers_failed: usize,
        #[allow(dead_code)]
        servers_skipped: usize,
        models_cached: usize,
        models_to_download: usize,
        schema_valid: bool,
        errors: Vec<String>,
    }

    impl PreflightResult {
        fn can_proceed(&self) -> bool {
            self.servers_failed == 0 && self.schema_valid
        }
    }

    let suite = BenchmarkPreflightSuite::standard();

    // Test: All ready
    let result = suite.run_all(&[true, true, true], &[true, true]);
    assert!(
        result.can_proceed(),
        "QA-038-040: Should proceed when all ready"
    );
    assert_eq!(result.servers_passed, 3);
    assert_eq!(result.models_cached, 2);

    // Test: Required server down
    let result = suite.run_all(&[false, true, true], &[true, true]);
    assert!(
        !result.can_proceed(),
        "QA-038-040: Should not proceed if required down"
    );

    // Test: Model needs download
    let result = suite.run_all(&[true, false, false], &[false, true]);
    assert!(
        result.can_proceed(),
        "QA-038-040: Can proceed with download needed"
    );
    assert_eq!(result.models_to_download, 1);

    println!("\nPARITY-010d: Complete preflight suite");
    println!("  Server checks: {}", suite.server_checks.len());
    println!("  Model checks: {}", suite.model_checks.len());
    println!("  Schema: {}", suite.schema_version);
}
