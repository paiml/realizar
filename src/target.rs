//! Multi-Target Deployment Support
//!
//! Per `docs/specifications/serve-deploy-apr.md` Section 6 and §11.4:
//! Support for Lambda, Docker, WASM Edge, and bare metal deployments.
//!
//! ## Supported Targets
//!
//! | Target | Features | Use Case |
//! |--------|----------|----------|
//! | Native | Full (SIMD, GPU, threads) | Bare metal, EC2 |
//! | Lambda | SIMD, no GPU | AWS Lambda ARM64 |
//! | Docker | Full | Container deployments |
//! | WASM | CPU-only, no threads | Cloudflare Workers |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use realizar::target::{DeployTarget, TargetCapabilities};
//!
//! let target = DeployTarget::detect();
//! let caps = target.capabilities();
//!
//! if caps.supports_simd {
//!     // Use SIMD-accelerated inference
//! }
//! ```

use serde::{Deserialize, Serialize};

/// Deployment target enumeration
///
/// Per spec §6: Multi-target deployment support
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeployTarget {
    /// Native binary (bare metal, EC2, etc.)
    Native,
    /// AWS Lambda (ARM64 Graviton or `x86_64`)
    Lambda,
    /// Docker container
    Docker,
    /// WebAssembly (Cloudflare Workers, browser)
    Wasm,
}

impl DeployTarget {
    /// Detect current deployment target at runtime
    ///
    /// Detection heuristics:
    /// - WASM: `cfg!(target_arch = "wasm32")`
    /// - Lambda: `AWS_LAMBDA_FUNCTION_NAME` env var
    /// - Docker: `/.dockerenv` file exists
    /// - Native: fallback
    #[must_use]
    pub fn detect() -> Self {
        // WASM detection at compile time
        #[cfg(target_arch = "wasm32")]
        {
            return Self::Wasm;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Lambda detection via environment
            if std::env::var("AWS_LAMBDA_FUNCTION_NAME").is_ok() {
                return Self::Lambda;
            }

            // Docker detection via /.dockerenv
            if std::path::Path::new("/.dockerenv").exists() {
                return Self::Docker;
            }

            // Default to native
            Self::Native
        }
    }

    /// Get capabilities for this target
    #[must_use]
    pub const fn capabilities(&self) -> TargetCapabilities {
        match self {
            // Native and Docker have full capabilities
            Self::Native | Self::Docker => TargetCapabilities {
                supports_simd: true,
                supports_gpu: true,
                supports_threads: true,
                supports_filesystem: true,
                supports_async_io: true,
                max_memory_mb: 0, // Unlimited (container limit applies for Docker)
            },
            Self::Lambda => TargetCapabilities {
                supports_simd: true,
                supports_gpu: false,
                supports_threads: true,
                supports_filesystem: false, // /tmp only
                supports_async_io: true,
                max_memory_mb: 10240, // Max Lambda memory
            },
            Self::Wasm => TargetCapabilities {
                supports_simd: false, // WASM SIMD limited
                supports_gpu: false,
                supports_threads: false,    // No rayon
                supports_filesystem: false, // Must use include_bytes!
                supports_async_io: false,   // Limited
                max_memory_mb: 128,         // Typical worker limit
            },
        }
    }

    /// Get target name as string
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Lambda => "lambda",
            Self::Docker => "docker",
            Self::Wasm => "wasm",
        }
    }

    /// Check if target supports a specific feature
    #[must_use]
    pub const fn supports(&self, feature: TargetFeature) -> bool {
        let caps = self.capabilities();
        match feature {
            TargetFeature::Simd => caps.supports_simd,
            TargetFeature::Gpu => caps.supports_gpu,
            TargetFeature::Threads => caps.supports_threads,
            TargetFeature::Filesystem => caps.supports_filesystem,
            TargetFeature::AsyncIo => caps.supports_async_io,
        }
    }
}

impl std::fmt::Display for DeployTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Target capabilities
///
/// Per spec §6.3: WASM Limitations table
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)] // Capability flags are naturally boolean
pub struct TargetCapabilities {
    /// SIMD acceleration available (AVX2, NEON, etc.)
    pub supports_simd: bool,
    /// GPU acceleration available (wgpu)
    pub supports_gpu: bool,
    /// Multi-threading available (rayon)
    pub supports_threads: bool,
    /// Filesystem access available
    pub supports_filesystem: bool,
    /// Async I/O available
    pub supports_async_io: bool,
    /// Maximum memory in MB (0 = unlimited)
    pub max_memory_mb: u32,
}

impl TargetCapabilities {
    /// Check if all required features are available
    #[must_use]
    pub const fn has_all(&self, required: &[TargetFeature]) -> bool {
        let mut i = 0;
        while i < required.len() {
            let has = match required[i] {
                TargetFeature::Simd => self.supports_simd,
                TargetFeature::Gpu => self.supports_gpu,
                TargetFeature::Threads => self.supports_threads,
                TargetFeature::Filesystem => self.supports_filesystem,
                TargetFeature::AsyncIo => self.supports_async_io,
            };
            if !has {
                return false;
            }
            i += 1;
        }
        true
    }
}

/// Target feature flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetFeature {
    /// SIMD acceleration
    Simd,
    /// GPU acceleration
    Gpu,
    /// Multi-threading
    Threads,
    /// Filesystem access
    Filesystem,
    /// Async I/O
    AsyncIo,
}

/// Docker build configuration
///
/// Per spec §6.2: Multi-stage Dockerfile patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerConfig {
    /// Base image for builder stage
    pub builder_image: String,
    /// Base image for runtime stage
    pub runtime_image: String,
    /// Target triple for cross-compilation
    pub target_triple: String,
    /// Whether to strip binary
    pub strip_binary: bool,
    /// Exposed port
    pub expose_port: u16,
}

impl Default for DockerConfig {
    fn default() -> Self {
        Self {
            builder_image: "rust:1.83".to_string(),
            runtime_image: "gcr.io/distroless/static-debian12:latest".to_string(),
            target_triple: "x86_64-unknown-linux-musl".to_string(),
            strip_binary: true,
            expose_port: 8080,
        }
    }
}

impl DockerConfig {
    /// Create config for ARM64 (Graviton)
    #[must_use]
    pub fn arm64() -> Self {
        Self {
            target_triple: "aarch64-unknown-linux-musl".to_string(),
            ..Self::default()
        }
    }

    /// Create config for minimal scratch image
    #[must_use]
    pub fn scratch() -> Self {
        Self {
            runtime_image: "scratch".to_string(),
            ..Self::default()
        }
    }

    /// Generate Dockerfile content
    #[must_use]
    pub fn generate_dockerfile(&self) -> String {
        format!(
            r#"# Auto-generated by realizar target module
# Per docs/specifications/serve-deploy-apr.md Section 6.2

# Stage 1: Build
FROM {builder} AS builder
WORKDIR /build

# Cache dependencies
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {{}}" > src/main.rs
RUN cargo build --release --target {target}
RUN rm -rf src

# Build actual binary
COPY src ./src
RUN cargo build --release --target {target}
{strip}

# Stage 2: Runtime
FROM {runtime}
COPY --from=builder /build/target/{target}/release/realizar /serve
EXPOSE {port}
ENTRYPOINT ["/serve"]
"#,
            builder = self.builder_image,
            runtime = self.runtime_image,
            target = self.target_triple,
            strip = if self.strip_binary {
                format!(
                    "RUN strip target/{}/release/realizar || true",
                    self.target_triple
                )
            } else {
                String::new()
            },
            port = self.expose_port
        )
    }

    /// Estimated image size in MB
    #[must_use]
    pub fn estimated_size_mb(&self) -> u32 {
        let base_size = if self.runtime_image.contains("scratch") {
            0
        } else if self.runtime_image.contains("distroless/static") {
            2
        } else if self.runtime_image.contains("distroless/cc") {
            20
        } else {
            50 // Generic estimate
        };

        // Binary size estimate: ~10MB for release build
        let binary_size = if self.strip_binary { 5 } else { 10 };

        base_size + binary_size
    }
}

/// WASM build configuration
///
/// Per spec §6.3: WASM Edge deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmConfig {
    /// wasm-pack target (web, bundler, nodejs)
    pub target: WasmTarget,
    /// Output directory
    pub out_dir: String,
    /// Enable WASM SIMD (experimental)
    pub enable_simd: bool,
}

/// WASM build targets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WasmTarget {
    /// Browser via ES modules
    Web,
    /// Bundler (webpack, etc.)
    Bundler,
    /// Node.js
    NodeJs,
}

impl WasmTarget {
    /// Get wasm-pack target flag
    #[must_use]
    pub const fn flag(&self) -> &'static str {
        match self {
            Self::Web => "web",
            Self::Bundler => "bundler",
            Self::NodeJs => "nodejs",
        }
    }
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            target: WasmTarget::Web,
            out_dir: "pkg".to_string(),
            enable_simd: false,
        }
    }
}

impl WasmConfig {
    /// Generate build command
    #[must_use]
    pub fn build_command(&self) -> String {
        let mut cmd = format!(
            "wasm-pack build --target {} --release --out-dir {}",
            self.target.flag(),
            self.out_dir
        );

        if self.enable_simd {
            cmd.push_str(" -- -C target-feature=+simd128");
        }

        cmd
    }

    /// Generate Cloudflare Worker template
    #[must_use]
    pub fn cloudflare_worker_template(&self) -> String {
        format!(
            r"// Auto-generated Cloudflare Worker
// Per docs/specifications/serve-deploy-apr.md Section 6.3

import init, {{ predict }} from './{}/realizar.js';

let initialized = false;

export default {{
  async fetch(request, env) {{
    if (!initialized) {{
      await init();
      initialized = true;
    }}

    if (request.method !== 'POST') {{
      return new Response('Method not allowed', {{ status: 405 }});
    }}

    try {{
      const body = await request.json();
      const result = predict(body.features);

      return new Response(JSON.stringify(result), {{
        headers: {{ 'Content-Type': 'application/json' }}
      }});
    }} catch (e) {{
      return new Response(JSON.stringify({{ error: e.message }}), {{
        status: 500,
        headers: {{ 'Content-Type': 'application/json' }}
      }});
    }}
  }}
}};
",
            self.out_dir
        )
    }
}

/// Build manifest for CI/CD
///
/// Per spec §11.4: Multi-target build support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildManifest {
    /// Version from Cargo.toml
    pub version: String,
    /// Git commit hash
    pub git_hash: Option<String>,
    /// Build timestamp
    pub build_time: String,
    /// Targets to build
    pub targets: Vec<BuildTarget>,
}

/// Individual build target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildTarget {
    /// Target name
    pub name: String,
    /// Rust target triple
    pub triple: String,
    /// Deploy target type
    pub deploy_target: DeployTarget,
    /// Build features to enable
    pub features: Vec<String>,
}

impl BuildManifest {
    /// Create default manifest with all targets
    #[must_use]
    pub fn default_all_targets() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            git_hash: None,
            build_time: String::new(),
            targets: vec![
                BuildTarget {
                    name: "linux-x86_64".to_string(),
                    triple: "x86_64-unknown-linux-musl".to_string(),
                    deploy_target: DeployTarget::Docker,
                    features: vec!["server".to_string()],
                },
                BuildTarget {
                    name: "linux-arm64".to_string(),
                    triple: "aarch64-unknown-linux-musl".to_string(),
                    deploy_target: DeployTarget::Lambda,
                    features: vec!["lambda".to_string()],
                },
                BuildTarget {
                    name: "wasm".to_string(),
                    triple: "wasm32-unknown-unknown".to_string(),
                    deploy_target: DeployTarget::Wasm,
                    features: vec![],
                },
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // RED PHASE: Failing tests for multi-target support
    // Per EXTREME TDD: Write tests FIRST
    // ==========================================================================

    // --------------------------------------------------------------------------
    // Test: DeployTarget detection and capabilities
    // --------------------------------------------------------------------------

    #[test]
    fn test_deploy_target_detect_native() {
        // In test environment without Lambda/Docker env vars, should detect native
        // Note: This test may behave differently in CI with Docker
        let target = DeployTarget::detect();
        // Should be Native, Lambda, or Docker depending on environment
        assert!(
            matches!(
                target,
                DeployTarget::Native | DeployTarget::Lambda | DeployTarget::Docker
            ),
            "Should detect a valid non-WASM target"
        );
    }

    #[test]
    fn test_deploy_target_names() {
        assert_eq!(DeployTarget::Native.name(), "native");
        assert_eq!(DeployTarget::Lambda.name(), "lambda");
        assert_eq!(DeployTarget::Docker.name(), "docker");
        assert_eq!(DeployTarget::Wasm.name(), "wasm");
    }

    #[test]
    fn test_deploy_target_display() {
        assert_eq!(format!("{}", DeployTarget::Native), "native");
        assert_eq!(format!("{}", DeployTarget::Wasm), "wasm");
    }

    #[test]
    fn test_native_capabilities() {
        let caps = DeployTarget::Native.capabilities();
        assert!(caps.supports_simd);
        assert!(caps.supports_gpu);
        assert!(caps.supports_threads);
        assert!(caps.supports_filesystem);
        assert!(caps.supports_async_io);
        assert_eq!(caps.max_memory_mb, 0); // Unlimited
    }

    #[test]
    fn test_lambda_capabilities() {
        let caps = DeployTarget::Lambda.capabilities();
        assert!(caps.supports_simd);
        assert!(!caps.supports_gpu); // No GPU in Lambda
        assert!(caps.supports_threads);
        assert!(!caps.supports_filesystem); // /tmp only
        assert!(caps.supports_async_io);
        assert_eq!(caps.max_memory_mb, 10240);
    }

    #[test]
    fn test_wasm_capabilities() {
        let caps = DeployTarget::Wasm.capabilities();
        assert!(!caps.supports_simd); // Limited SIMD
        assert!(!caps.supports_gpu);
        assert!(!caps.supports_threads); // No rayon
        assert!(!caps.supports_filesystem); // Must use include_bytes!
        assert!(!caps.supports_async_io);
        assert_eq!(caps.max_memory_mb, 128);
    }

    #[test]
    fn test_deploy_target_supports_feature() {
        assert!(DeployTarget::Native.supports(TargetFeature::Simd));
        assert!(DeployTarget::Native.supports(TargetFeature::Gpu));

        assert!(DeployTarget::Lambda.supports(TargetFeature::Simd));
        assert!(!DeployTarget::Lambda.supports(TargetFeature::Gpu));

        assert!(!DeployTarget::Wasm.supports(TargetFeature::Threads));
        assert!(!DeployTarget::Wasm.supports(TargetFeature::Filesystem));
    }

    #[test]
    fn test_capabilities_has_all() {
        let native_caps = DeployTarget::Native.capabilities();
        assert!(native_caps.has_all(&[TargetFeature::Simd, TargetFeature::Gpu]));
        assert!(native_caps.has_all(&[TargetFeature::Threads, TargetFeature::Filesystem]));

        let wasm_caps = DeployTarget::Wasm.capabilities();
        assert!(!wasm_caps.has_all(&[TargetFeature::Simd]));
        assert!(!wasm_caps.has_all(&[TargetFeature::Threads]));
        assert!(wasm_caps.has_all(&[])); // Empty requirements
    }

    // --------------------------------------------------------------------------
    // Test: Docker configuration
    // --------------------------------------------------------------------------

    #[test]
    fn test_docker_config_default() {
        let config = DockerConfig::default();
        assert_eq!(config.builder_image, "rust:1.83");
        assert!(config.runtime_image.contains("distroless"));
        assert_eq!(config.target_triple, "x86_64-unknown-linux-musl");
        assert!(config.strip_binary);
        assert_eq!(config.expose_port, 8080);
    }

    #[test]
    fn test_docker_config_arm64() {
        let config = DockerConfig::arm64();
        assert!(config.target_triple.contains("aarch64"));
    }

    #[test]
    fn test_docker_config_scratch() {
        let config = DockerConfig::scratch();
        assert_eq!(config.runtime_image, "scratch");
    }

    #[test]
    fn test_docker_generate_dockerfile() {
        let config = DockerConfig::default();
        let dockerfile = config.generate_dockerfile();

        assert!(dockerfile.contains("FROM rust:1.83 AS builder"));
        assert!(dockerfile.contains("distroless"));
        assert!(dockerfile.contains("x86_64-unknown-linux-musl"));
        assert!(dockerfile.contains("EXPOSE 8080"));
        assert!(dockerfile.contains("strip"));
    }

    #[test]
    fn test_docker_estimated_size() {
        let distroless = DockerConfig::default();
        assert!(distroless.estimated_size_mb() < 20); // ~7MB

        let scratch = DockerConfig::scratch();
        assert!(scratch.estimated_size_mb() < 10); // ~5MB (binary only)
    }

    // --------------------------------------------------------------------------
    // Test: WASM configuration
    // --------------------------------------------------------------------------

    #[test]
    fn test_wasm_config_default() {
        let config = WasmConfig::default();
        assert_eq!(config.target, WasmTarget::Web);
        assert_eq!(config.out_dir, "pkg");
        assert!(!config.enable_simd);
    }

    #[test]
    fn test_wasm_target_flags() {
        assert_eq!(WasmTarget::Web.flag(), "web");
        assert_eq!(WasmTarget::Bundler.flag(), "bundler");
        assert_eq!(WasmTarget::NodeJs.flag(), "nodejs");
    }

    #[test]
    fn test_wasm_build_command() {
        let config = WasmConfig::default();
        let cmd = config.build_command();

        assert!(cmd.contains("wasm-pack build"));
        assert!(cmd.contains("--target web"));
        assert!(cmd.contains("--release"));
        assert!(cmd.contains("--out-dir pkg"));
    }

    #[test]
    fn test_wasm_build_command_with_simd() {
        let config = WasmConfig {
            enable_simd: true,
            ..WasmConfig::default()
        };
        let cmd = config.build_command();

        assert!(cmd.contains("simd128"));
    }

    #[test]
    fn test_wasm_cloudflare_worker_template() {
        let config = WasmConfig::default();
        let template = config.cloudflare_worker_template();

        assert!(template.contains("import init"));
        assert!(template.contains("predict"));
        assert!(template.contains("async fetch"));
        assert!(template.contains("application/json"));
    }

    // --------------------------------------------------------------------------
    // Test: Build manifest
    // --------------------------------------------------------------------------

    #[test]
    fn test_build_manifest_default() {
        let manifest = BuildManifest::default_all_targets();

        assert!(!manifest.version.is_empty());
        assert_eq!(manifest.targets.len(), 3);

        // Check target names
        let names: Vec<_> = manifest.targets.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"linux-x86_64"));
        assert!(names.contains(&"linux-arm64"));
        assert!(names.contains(&"wasm"));
    }

    #[test]
    fn test_build_manifest_targets() {
        let manifest = BuildManifest::default_all_targets();

        let x86_target = manifest
            .targets
            .iter()
            .find(|t| t.name == "linux-x86_64")
            .unwrap();
        assert_eq!(x86_target.deploy_target, DeployTarget::Docker);

        let arm_target = manifest
            .targets
            .iter()
            .find(|t| t.name == "linux-arm64")
            .unwrap();
        assert_eq!(arm_target.deploy_target, DeployTarget::Lambda);

        let wasm_target = manifest.targets.iter().find(|t| t.name == "wasm").unwrap();
        assert_eq!(wasm_target.deploy_target, DeployTarget::Wasm);
    }

    // --------------------------------------------------------------------------
    // Test: Serialization
    // --------------------------------------------------------------------------

    #[test]
    fn test_deploy_target_serialization() {
        let target = DeployTarget::Lambda;
        let json = serde_json::to_string(&target).unwrap();
        assert!(json.contains("Lambda"));

        let deserialized: DeployTarget = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, DeployTarget::Lambda);
    }

    #[test]
    fn test_docker_config_serialization() {
        let config = DockerConfig::default();
        let json = serde_json::to_string(&config).unwrap();

        assert!(json.contains("rust:1.83"));
        assert!(json.contains("distroless"));

        let deserialized: DockerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.expose_port, 8080);
    }

    #[test]
    fn test_wasm_config_serialization() {
        let config = WasmConfig::default();
        let json = serde_json::to_string(&config).unwrap();

        assert!(json.contains("Web"));
        assert!(json.contains("pkg"));

        let deserialized: WasmConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.target, WasmTarget::Web);
    }
}
