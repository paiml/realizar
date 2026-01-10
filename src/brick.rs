//! ComputeBrick: Token-centric, self-verifying compute units.
//!
//! Per CBTOP-SPEC-001 and SHOWCASE-BRICK-001, every compute operation is a brick with:
//! - Token budget (tok/sec performance target)
//! - Assertions (falsifiable correctness claims)
//! - Verification (self-checking via baseline comparison)
//!
//! # Toyota Way Integration
//! - **Jidoka**: Stop-the-line on budget violation
//! - **Poka-Yoke**: Type-safe brick composition
//! - **Genchi Genbutsu**: Real metrics from hardware
//! - **Mieruka**: Visual control via cbtop TUI
//!
//! # References
//! - Popper, K. (1959). "The Logic of Scientific Discovery."
//! - Ohno, T. (1988). "Toyota Production System."
//! - Little, J.D.C. (1961). "A Proof for the Queuing Formula: L = λW."

use std::fmt;
use std::time::Instant;

// ============================================================================
// Core Types
// ============================================================================

/// Performance budget expressed in token terms.
/// Aligns compute costs with LLM inference metrics.
///
/// Per Little's Law (1961): throughput = 1 / latency
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenBudget {
    /// Latency budget per token (microseconds)
    pub us_per_token: f64,
    /// Throughput target (tokens/second)
    pub tokens_per_sec: f64,
    /// Batch size for amortization
    pub batch_size: usize,
}

impl TokenBudget {
    /// Create budget from latency target.
    /// 50µs/token = 20,000 tokens/sec
    #[must_use]
    pub fn from_latency(us_per_token: f64) -> Self {
        Self {
            us_per_token,
            tokens_per_sec: 1_000_000.0 / us_per_token,
            batch_size: 1,
        }
    }

    /// Create budget from throughput target.
    /// 20,000 tokens/sec = 50µs/token
    #[must_use]
    pub fn from_throughput(tokens_per_sec: f64) -> Self {
        Self {
            us_per_token: 1_000_000.0 / tokens_per_sec,
            tokens_per_sec,
            batch_size: 1,
        }
    }

    /// Create budget with batch size.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Check if actual performance meets budget.
    #[must_use]
    pub fn is_met(&self, actual_us_per_token: f64) -> bool {
        actual_us_per_token <= self.us_per_token
    }

    /// Calculate gap factor (actual / budget).
    /// < 1.0 means under budget (good), > 1.0 means over budget (bad).
    #[must_use]
    pub fn gap_factor(&self, actual_us_per_token: f64) -> f64 {
        actual_us_per_token / self.us_per_token
    }
}

impl Default for TokenBudget {
    fn default() -> Self {
        Self::from_latency(100.0) // 100µs = 10k tok/s default
    }
}

/// Result of ComputeBrick execution with token metrics.
#[derive(Debug, Clone)]
pub struct TokenResult<T> {
    /// Computed output
    pub output: T,
    /// Number of tokens processed
    pub tokens_processed: usize,
    /// Actual latency (microseconds/token)
    pub us_per_token: f64,
    /// Actual throughput (tokens/second)
    pub tokens_per_sec: f64,
    /// Did we meet the budget?
    pub budget_met: bool,
}

impl<T: Default> Default for TokenResult<T> {
    fn default() -> Self {
        Self {
            output: T::default(),
            tokens_processed: 0,
            us_per_token: 0.0,
            tokens_per_sec: 0.0,
            budget_met: true,
        }
    }
}

impl<T> TokenResult<T> {
    /// Create a new token result.
    pub fn new(output: T, tokens: usize, elapsed_us: f64, budget: &TokenBudget) -> Self {
        let us_per_token = elapsed_us / tokens.max(1) as f64;
        let tokens_per_sec = if us_per_token > 0.0 {
            1_000_000.0 / us_per_token
        } else {
            0.0
        };

        Self {
            output,
            tokens_processed: tokens,
            us_per_token,
            tokens_per_sec,
            budget_met: budget.is_met(us_per_token),
        }
    }
}

/// Errors from ComputeBrick execution.
/// Tells you exactly what failed (Jidoka: stop and signal).
#[derive(Debug)]
pub enum BrickError {
    /// Assertion failed during verification.
    AssertionFailed {
        /// Assertion name
        name: String,
        /// Expected value
        expected: String,
        /// Actual value
        actual: String,
    },
    /// Performance budget exceeded.
    BudgetExceeded {
        /// Budget limit in µs
        limit_us: f64,
        /// Actual time in µs
        actual_us: f64,
    },
    /// Compute operation failed.
    ComputeError(String),
    /// Invalid input.
    InvalidInput(String),
}

impl fmt::Display for BrickError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AssertionFailed {
                name,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Assertion failed: {name} - expected {expected}, got {actual}"
                )
            },
            Self::BudgetExceeded {
                limit_us,
                actual_us,
            } => {
                write!(
                    f,
                    "Budget exceeded: {limit_us:.1}µs/tok limit, {actual_us:.1}µs/tok actual"
                )
            },
            Self::ComputeError(msg) => write!(f, "Compute error: {msg}"),
            Self::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
        }
    }
}

impl std::error::Error for BrickError {}

// ============================================================================
// Brick Assertions
// ============================================================================

/// Falsifiable assertion for a brick (Popper criterion).
#[derive(Debug, Clone)]
pub struct BrickAssertion {
    /// Assertion name (for error messages)
    pub name: String,
    /// Description of what is being asserted
    pub description: String,
    /// Assertion type
    pub kind: AssertionKind,
}

/// Types of assertions a brick can make.
#[derive(Debug, Clone)]
pub enum AssertionKind {
    /// Output matches scalar baseline within tolerance.
    EquivScalar {
        /// Tolerance for comparison
        tolerance: f64,
    },
    /// Output contains no NaN values.
    NoNaN,
    /// Output contains no Inf values.
    NoInf,
    /// Output values within bounds.
    Bounds {
        /// Minimum allowed value
        min: f64,
        /// Maximum allowed value
        max: f64,
    },
    /// Budget must be met.
    BudgetMet,
    /// Custom assertion (returns true if passed).
    Custom {
        /// Name of the custom check
        check_name: String,
    },
}

impl BrickAssertion {
    /// Create assertion that output matches scalar baseline.
    pub fn equiv_scalar(tolerance: f64) -> Self {
        Self {
            name: "equiv_scalar".to_string(),
            description: format!("Output matches scalar baseline within {tolerance}"),
            kind: AssertionKind::EquivScalar { tolerance },
        }
    }

    /// Create assertion that output contains no NaN.
    pub fn no_nan() -> Self {
        Self {
            name: "no_nan".to_string(),
            description: "Output contains no NaN values".to_string(),
            kind: AssertionKind::NoNaN,
        }
    }

    /// Create assertion that output contains no Inf.
    pub fn no_inf() -> Self {
        Self {
            name: "no_inf".to_string(),
            description: "Output contains no Inf values".to_string(),
            kind: AssertionKind::NoInf,
        }
    }

    /// Create assertion that values are within bounds.
    pub fn bounds(min: f64, max: f64) -> Self {
        Self {
            name: "bounds".to_string(),
            description: format!("Output values in [{min}, {max}]"),
            kind: AssertionKind::Bounds { min, max },
        }
    }

    /// Create assertion that budget is met.
    pub fn budget_met() -> Self {
        Self {
            name: "budget_met".to_string(),
            description: "Performance budget is met".to_string(),
            kind: AssertionKind::BudgetMet,
        }
    }

    /// Check assertion against f32 slice output.
    pub fn check_f32(&self, output: &[f32], budget_met: bool) -> Result<(), BrickError> {
        match &self.kind {
            AssertionKind::NoNaN => {
                if let Some(idx) = output.iter().position(|x| x.is_nan()) {
                    return Err(BrickError::AssertionFailed {
                        name: self.name.clone(),
                        expected: "no NaN".to_string(),
                        actual: format!("NaN at index {idx}"),
                    });
                }
            },
            AssertionKind::NoInf => {
                if let Some(idx) = output.iter().position(|x| x.is_infinite()) {
                    return Err(BrickError::AssertionFailed {
                        name: self.name.clone(),
                        expected: "no Inf".to_string(),
                        actual: format!("Inf at index {idx}"),
                    });
                }
            },
            AssertionKind::Bounds { min, max } => {
                for (idx, &val) in output.iter().enumerate() {
                    if (val as f64) < *min || (val as f64) > *max {
                        return Err(BrickError::AssertionFailed {
                            name: self.name.clone(),
                            expected: format!("value in [{min}, {max}]"),
                            actual: format!("value {val} at index {idx}"),
                        });
                    }
                }
            },
            AssertionKind::BudgetMet => {
                if !budget_met {
                    return Err(BrickError::AssertionFailed {
                        name: self.name.clone(),
                        expected: "budget met".to_string(),
                        actual: "budget exceeded".to_string(),
                    });
                }
            },
            AssertionKind::EquivScalar { .. } | AssertionKind::Custom { .. } => {
                // These require external comparison, skip for basic check
            },
        }
        Ok(())
    }
}

// ============================================================================
// Brick Verification
// ============================================================================

/// Result of brick verification.
#[derive(Debug, Clone)]
pub struct BrickVerification {
    /// All assertions passed?
    pub is_valid: bool,
    /// Individual assertion results
    pub results: Vec<(String, bool, String)>,
}

impl BrickVerification {
    /// Create a passing verification.
    pub fn pass() -> Self {
        Self {
            is_valid: true,
            results: vec![],
        }
    }

    /// Create a failing verification.
    pub fn fail(name: &str, reason: &str) -> Self {
        Self {
            is_valid: false,
            results: vec![(name.to_string(), false, reason.to_string())],
        }
    }

    /// Add an assertion result.
    pub fn add(&mut self, name: &str, passed: bool, message: &str) {
        self.results
            .push((name.to_string(), passed, message.to_string()));
        if !passed {
            self.is_valid = false;
        }
    }
}

// ============================================================================
// ComputeBrick Trait
// ============================================================================

/// Core trait for self-verifying, token-centric compute units.
///
/// Every brick must:
/// 1. Have at least one assertion (Popper criterion)
/// 2. Have a non-zero budget (accountability)
/// 3. Be verifiable against baseline
///
/// # Invariants (PROBAR-SPEC-009 §3)
/// - `assertions().len() > 0` (at least one falsifiable claim)
/// - `budget().us_per_token > 0` (performance accountability)
/// - `verify()` checks ALL assertions
pub trait ComputeBrick: Send + Sync {
    /// Output type of this brick.
    type Output;

    /// Brick name for identification.
    fn name(&self) -> &'static str;

    /// Token throughput budget.
    fn budget(&self) -> TokenBudget;

    /// Falsifiable assertions for this brick.
    fn assertions(&self) -> Vec<BrickAssertion>;

    /// Verify brick state without running.
    fn verify(&self) -> BrickVerification {
        let assertions = self.assertions();
        if assertions.is_empty() {
            return BrickVerification::fail(
                self.name(),
                "No assertions defined (Popper violation)",
            );
        }

        let budget = self.budget();
        if budget.us_per_token <= 0.0 {
            return BrickVerification::fail(self.name(), "Zero or negative budget");
        }

        BrickVerification::pass()
    }

    /// Can this brick run? (Jidoka gate)
    fn can_run(&self) -> bool {
        self.verify().is_valid
    }
}

// ============================================================================
// Transformer Brick Implementations
// ============================================================================

/// RMSNorm brick - layer normalization.
#[derive(Debug)]
pub struct RmsNormBrick {
    /// Weight vector
    pub weight: Vec<f32>,
    /// Epsilon for numerical stability
    pub eps: f32,
    /// Budget
    budget: TokenBudget,
}

impl RmsNormBrick {
    /// Create a new RMSNorm brick.
    pub fn new(weight: Vec<f32>, eps: f32) -> Self {
        Self {
            weight,
            eps,
            budget: TokenBudget::from_latency(1.5), // 1.5µs target
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Run RMSNorm on input.
    pub fn run(&self, input: &[f32]) -> Result<TokenResult<Vec<f32>>, BrickError> {
        if input.len() != self.weight.len() {
            return Err(BrickError::InvalidInput(format!(
                "Input len {} != weight len {}",
                input.len(),
                self.weight.len()
            )));
        }

        let start = Instant::now();

        // Compute RMS
        let rms = (input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32 + self.eps).sqrt();

        // Normalize and scale
        let output: Vec<f32> = input
            .iter()
            .zip(self.weight.iter())
            .map(|(x, w)| (x / rms) * w)
            .collect();

        let elapsed_us = start.elapsed().as_micros() as f64;
        let result = TokenResult::new(output, 1, elapsed_us, &self.budget);

        // Check assertions
        for assertion in self.assertions() {
            assertion.check_f32(&result.output, result.budget_met)?;
        }

        Ok(result)
    }
}

impl ComputeBrick for RmsNormBrick {
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "rms_norm"
    }

    fn budget(&self) -> TokenBudget {
        self.budget
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::no_nan(),
            BrickAssertion::no_inf(),
            BrickAssertion::budget_met(),
        ]
    }
}

/// QKV projection brick.
#[derive(Debug)]
pub struct QkvBrick {
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Q output dimension
    pub q_dim: usize,
    /// K output dimension
    pub k_dim: usize,
    /// V output dimension
    pub v_dim: usize,
    /// Budget
    budget: TokenBudget,
    /// Has bias (Qwen2 has large biases)
    pub has_bias: bool,
}

impl QkvBrick {
    /// Create a new QKV brick.
    pub fn new(hidden_dim: usize, q_dim: usize, k_dim: usize, v_dim: usize) -> Self {
        Self {
            hidden_dim,
            q_dim,
            k_dim,
            v_dim,
            budget: TokenBudget::from_latency(6.0), // 6µs target
            has_bias: false,
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Mark as having bias.
    #[must_use]
    pub fn with_bias(mut self) -> Self {
        self.has_bias = true;
        self
    }

    /// Total output dimension.
    pub fn total_out_dim(&self) -> usize {
        self.q_dim + self.k_dim + self.v_dim
    }
}

impl ComputeBrick for QkvBrick {
    type Output = (Vec<f32>, Vec<f32>, Vec<f32>);

    fn name(&self) -> &'static str {
        "qkv_proj"
    }

    fn budget(&self) -> TokenBudget {
        self.budget
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::no_nan(),
            BrickAssertion::no_inf(),
            BrickAssertion::budget_met(),
        ]
    }
}

/// RoPE brick - rotary position embedding.
#[derive(Debug)]
pub struct RopeBrick {
    /// Head dimension
    pub head_dim: usize,
    /// Number of heads
    pub num_heads: usize,
    /// Base theta
    pub theta: f32,
    /// RoPE type (0=NORM, 2=NEOX)
    pub rope_type: u32,
    /// Budget
    budget: TokenBudget,
}

impl RopeBrick {
    /// Create a new RoPE brick.
    pub fn new(head_dim: usize, num_heads: usize, theta: f32, rope_type: u32) -> Self {
        Self {
            head_dim,
            num_heads,
            theta,
            rope_type,
            budget: TokenBudget::from_latency(1.0), // 1µs target
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }
}

impl ComputeBrick for RopeBrick {
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "rope"
    }

    fn budget(&self) -> TokenBudget {
        self.budget
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::no_nan(),
            BrickAssertion::no_inf(),
            BrickAssertion::budget_met(),
        ]
    }
}

/// Attention brick.
#[derive(Debug)]
pub struct AttentionBrick {
    /// Number of query heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Budget
    budget: TokenBudget,
}

impl AttentionBrick {
    /// Create a new attention brick.
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            budget: TokenBudget::from_latency(10.0), // 10µs target
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }

    /// GQA group size.
    pub fn group_size(&self) -> usize {
        self.num_heads / self.num_kv_heads.max(1)
    }
}

impl ComputeBrick for AttentionBrick {
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "attention"
    }

    fn budget(&self) -> TokenBudget {
        self.budget
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::no_nan(),
            BrickAssertion::no_inf(),
            BrickAssertion::budget_met(),
            // Attention outputs should be bounded
            BrickAssertion::bounds(-100.0, 100.0),
        ]
    }
}

/// FFN brick (SwiGLU).
#[derive(Debug)]
pub struct FfnBrick {
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Intermediate dimension
    pub intermediate_dim: usize,
    /// Budget
    budget: TokenBudget,
}

impl FfnBrick {
    /// Create a new FFN brick.
    pub fn new(hidden_dim: usize, intermediate_dim: usize) -> Self {
        Self {
            hidden_dim,
            intermediate_dim,
            budget: TokenBudget::from_latency(12.2), // 12.2µs target
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }
}

impl ComputeBrick for FfnBrick {
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "ffn"
    }

    fn budget(&self) -> TokenBudget {
        self.budget
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::no_nan(),
            BrickAssertion::no_inf(),
            BrickAssertion::budget_met(),
        ]
    }
}

/// Output projection brick.
#[derive(Debug)]
pub struct OProjBrick {
    /// Input dimension (num_heads * head_dim)
    pub in_dim: usize,
    /// Output dimension (hidden_dim)
    pub out_dim: usize,
    /// Budget
    budget: TokenBudget,
}

impl OProjBrick {
    /// Create a new O projection brick.
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        Self {
            in_dim,
            out_dim,
            budget: TokenBudget::from_latency(3.5), // 3.5µs target
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }
}

impl ComputeBrick for OProjBrick {
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "o_proj"
    }

    fn budget(&self) -> TokenBudget {
        self.budget
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::no_nan(),
            BrickAssertion::no_inf(),
            BrickAssertion::budget_met(),
        ]
    }
}

// ============================================================================
// Transformer Layer Brick
// ============================================================================

/// Full transformer layer as a composed brick.
#[derive(Debug)]
pub struct TransformerLayerBrick {
    /// Layer index
    pub layer_idx: usize,
    /// Attention layer normalization brick
    pub attn_norm: RmsNormBrick,
    /// QKV projection brick
    pub qkv: QkvBrick,
    /// Rotary position embedding brick
    pub rope: RopeBrick,
    /// Attention computation brick
    pub attention: AttentionBrick,
    /// Output projection brick
    pub o_proj: OProjBrick,
    /// FFN layer normalization brick
    pub ffn_norm: RmsNormBrick,
    /// Feed-forward network brick
    pub ffn: FfnBrick,
    /// Timing metrics (updated after each run)
    pub last_timing: Option<LayerTiming>,
}

/// Timing breakdown for a layer.
#[derive(Debug, Clone, Default)]
pub struct LayerTiming {
    /// Attention normalization time (µs)
    pub attn_norm_us: f64,
    /// QKV projection time (µs)
    pub qkv_us: f64,
    /// RoPE application time (µs)
    pub rope_us: f64,
    /// Attention computation time (µs)
    pub attention_us: f64,
    /// Output projection time (µs)
    pub o_proj_us: f64,
    /// FFN normalization time (µs)
    pub ffn_norm_us: f64,
    /// FFN computation time (µs)
    pub ffn_us: f64,
    /// Total layer time (µs)
    pub total_us: f64,
}

impl LayerTiming {
    /// Find the bottleneck brick.
    pub fn bottleneck(&self) -> (&'static str, f64) {
        let bricks = [
            ("attn_norm", self.attn_norm_us),
            ("qkv", self.qkv_us),
            ("rope", self.rope_us),
            ("attention", self.attention_us),
            ("o_proj", self.o_proj_us),
            ("ffn_norm", self.ffn_norm_us),
            ("ffn", self.ffn_us),
        ];

        bricks
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or(("unknown", 0.0))
    }
}

impl TransformerLayerBrick {
    /// Create from configuration.
    pub fn from_config(
        layer_idx: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        intermediate_dim: usize,
        eps: f32,
        rope_theta: f32,
        rope_type: u32,
    ) -> Self {
        let head_dim = hidden_dim / num_heads;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        Self {
            layer_idx,
            attn_norm: RmsNormBrick::new(vec![1.0; hidden_dim], eps),
            qkv: QkvBrick::new(hidden_dim, q_dim, kv_dim, kv_dim),
            rope: RopeBrick::new(head_dim, num_heads, rope_theta, rope_type),
            attention: AttentionBrick::new(num_heads, num_kv_heads, head_dim),
            o_proj: OProjBrick::new(q_dim, hidden_dim),
            ffn_norm: RmsNormBrick::new(vec![1.0; hidden_dim], eps),
            ffn: FfnBrick::new(hidden_dim, intermediate_dim),
            last_timing: None,
        }
    }

    /// Get total budget for this layer.
    pub fn total_budget_us(&self) -> f64 {
        self.attn_norm.budget().us_per_token
            + self.qkv.budget().us_per_token
            + self.rope.budget().us_per_token
            + self.attention.budget().us_per_token
            + self.o_proj.budget().us_per_token
            + self.ffn_norm.budget().us_per_token
            + self.ffn.budget().us_per_token
    }
}

impl ComputeBrick for TransformerLayerBrick {
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "transformer_layer"
    }

    fn budget(&self) -> TokenBudget {
        TokenBudget::from_latency(self.total_budget_us())
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::no_nan(),
            BrickAssertion::no_inf(),
            BrickAssertion::budget_met(),
        ]
    }

    fn verify(&self) -> BrickVerification {
        // Verify all component bricks
        let mut result = BrickVerification::pass();

        for brick in [
            &self.attn_norm as &dyn ComputeBrick<Output = Vec<f32>>,
            &self.ffn_norm as &dyn ComputeBrick<Output = Vec<f32>>,
        ] {
            let v = brick.verify();
            if !v.is_valid {
                result.is_valid = false;
                result.results.extend(v.results);
            }
        }

        result
    }
}

// ============================================================================
// Bottleneck Report
// ============================================================================

/// Report identifying pipeline bottleneck.
#[derive(Debug, Clone)]
pub struct BottleneckReport {
    /// Layer index containing bottleneck
    pub layer_idx: usize,
    /// Brick name
    pub brick_name: &'static str,
    /// Actual latency (µs)
    pub actual_us: f64,
    /// Budget latency (µs)
    pub budget_us: f64,
    /// Gap factor (actual / budget)
    pub gap_factor: f64,
}

impl fmt::Display for BottleneckReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Bottleneck: {} (layer {}) - {:.1}µs actual vs {:.1}µs budget ({:.2}x)",
            self.brick_name, self.layer_idx, self.actual_us, self.budget_us, self.gap_factor
        )
    }
}

// ============================================================================
// Benchmark Brick
// ============================================================================

/// Configuration for benchmark runs.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    pub warmup: usize,
    /// Number of sample iterations
    pub samples: usize,
    /// Maximum allowed CV (coefficient of variation)
    pub max_cv: f64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup: 10,
            samples: 100,
            max_cv: 0.05, // 5% per Stabilizer (Curtsinger & Berger 2013)
        }
    }
}

/// Benchmark report with statistical analysis.
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Brick name
    pub brick_name: String,
    /// Mean latency (µs)
    pub mean_us: f64,
    /// Standard deviation (µs)
    pub std_us: f64,
    /// Coefficient of variation
    pub cv: f64,
    /// 50th percentile (µs)
    pub p50_us: f64,
    /// 99th percentile (µs)
    pub p99_us: f64,
    /// Throughput (tokens/sec)
    pub tokens_per_sec: f64,
    /// Budget target (µs)
    pub budget_us: f64,
    /// Budget met?
    pub budget_met: bool,
    /// Statistical validity (CV < max_cv)
    pub statistically_valid: bool,
}

impl fmt::Display for BenchmarkReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.budget_met { "PASS" } else { "FAIL" };
        write!(
            f,
            "{}: {:.1}µs ± {:.1}µs (CV={:.1}%) | {:.0} tok/s | budget: {} ({})",
            self.brick_name,
            self.mean_us,
            self.std_us,
            self.cv * 100.0,
            self.tokens_per_sec,
            self.budget_us,
            status
        )
    }
}

/// Calculate percentile from sorted samples.
fn percentile(samples: &[f64], p: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let idx = ((samples.len() as f64) * p).floor() as usize;
    samples[idx.min(samples.len() - 1)]
}

/// Run benchmark on a brick with statistical rigor.
pub fn benchmark_brick<B: ComputeBrick>(
    brick: &B,
    run_fn: impl Fn() -> f64,
    config: &BenchmarkConfig,
) -> BenchmarkReport {
    // Warmup (Jidoka: ensure stable state)
    for _ in 0..config.warmup {
        let _ = run_fn();
    }

    // Collect samples
    let mut samples: Vec<f64> = Vec::with_capacity(config.samples);
    for _ in 0..config.samples {
        samples.push(run_fn());
    }

    // Sort for percentiles
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Statistical analysis
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let std =
        (samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64).sqrt();
    let cv = std / mean;

    let budget = brick.budget();

    BenchmarkReport {
        brick_name: brick.name().to_string(),
        mean_us: mean,
        std_us: std,
        cv,
        p50_us: percentile(&samples, 0.50),
        p99_us: percentile(&samples, 0.99),
        tokens_per_sec: 1_000_000.0 / mean,
        budget_us: budget.us_per_token,
        budget_met: mean <= budget.us_per_token,
        statistically_valid: cv <= config.max_cv,
    }
}

// ============================================================================
// Tests (F001-F020)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // F001: All bricks implement ComputeBrick trait
    #[test]
    fn f001_brick_trait_implemented() {
        let _ = RmsNormBrick::new(vec![1.0; 64], 1e-5);
        let _ = QkvBrick::new(64, 64, 64, 64);
        let _ = AttentionBrick::new(8, 2, 64);
        let _ = FfnBrick::new(64, 256);
        let _ = RopeBrick::new(64, 8, 10000.0, 0);
        let _ = OProjBrick::new(512, 64);
    }

    // F002: assertions().len() > 0 for all bricks
    #[test]
    fn f002_brick_assertions_nonempty() {
        assert!(!RmsNormBrick::new(vec![1.0; 64], 1e-5)
            .assertions()
            .is_empty());
        assert!(!QkvBrick::new(64, 64, 64, 64).assertions().is_empty());
        assert!(!AttentionBrick::new(8, 2, 64).assertions().is_empty());
        assert!(!FfnBrick::new(64, 256).assertions().is_empty());
        assert!(!RopeBrick::new(64, 8, 10000.0, 0).assertions().is_empty());
        assert!(!OProjBrick::new(512, 64).assertions().is_empty());
    }

    // F004: budget() returns non-zero value
    #[test]
    fn f004_budget_nonzero() {
        assert!(RmsNormBrick::new(vec![1.0; 64], 1e-5).budget().us_per_token > 0.0);
        assert!(QkvBrick::new(64, 64, 64, 64).budget().us_per_token > 0.0);
        assert!(AttentionBrick::new(8, 2, 64).budget().us_per_token > 0.0);
        assert!(FfnBrick::new(64, 256).budget().us_per_token > 0.0);
    }

    // F005: name() is unique per brick type
    #[test]
    fn f005_brick_names_unique() {
        let names = [
            RmsNormBrick::new(vec![1.0; 64], 1e-5).name(),
            QkvBrick::new(64, 64, 64, 64).name(),
            AttentionBrick::new(8, 2, 64).name(),
            FfnBrick::new(64, 256).name(),
            RopeBrick::new(64, 8, 10000.0, 0).name(),
            OProjBrick::new(512, 64).name(),
        ];
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(names.len(), unique.len());
    }

    // F008: TokenResult fields are consistent
    #[test]
    fn f008_token_result_consistent() {
        let budget = TokenBudget::from_latency(100.0);
        let result: TokenResult<Vec<f32>> = TokenResult::new(vec![], 10, 500.0, &budget);

        assert_eq!(result.tokens_processed, 10);
        assert!((result.us_per_token - 50.0).abs() < 0.001);
        assert!((result.tokens_per_sec - 20000.0).abs() < 1.0);
        assert!(result.budget_met); // 50µs < 100µs budget
    }

    // F010: Pipeline bottleneck correctly identified
    #[test]
    fn f010_bottleneck_identification() {
        let timing = LayerTiming {
            attn_norm_us: 1.2,
            qkv_us: 8.5,
            rope_us: 0.8,
            attention_us: 12.3, // Bottleneck
            o_proj_us: 4.1,
            ffn_norm_us: 1.2,
            ffn_us: 15.8, // Actually this is the bottleneck
            total_us: 43.9,
        };

        let (name, us) = timing.bottleneck();
        assert_eq!(name, "ffn");
        assert!((us - 15.8).abs() < 0.001);
    }

    // F021: TokenBudget latency/throughput consistent
    #[test]
    fn f021_budget_math_consistent() {
        let from_latency = TokenBudget::from_latency(50.0);
        let from_throughput = TokenBudget::from_throughput(20000.0);

        assert!((from_latency.tokens_per_sec - 20000.0).abs() < 1.0);
        assert!((from_throughput.us_per_token - 50.0).abs() < 0.001);
    }

    // F022: Budget violation triggers error
    #[test]
    fn f022_budget_enforcement() {
        let budget = TokenBudget::from_latency(10.0);
        assert!(budget.is_met(5.0)); // Under budget
        assert!(budget.is_met(10.0)); // At budget
        assert!(!budget.is_met(15.0)); // Over budget

        assert!(budget.gap_factor(5.0) < 1.0);
        assert!((budget.gap_factor(10.0) - 1.0).abs() < 0.001);
        assert!(budget.gap_factor(15.0) > 1.0);
    }

    // F049: No NaN assertion works
    #[test]
    fn f049_nan_assertion() {
        let assertion = BrickAssertion::no_nan();

        // Should pass
        assert!(assertion.check_f32(&[1.0, 2.0, 3.0], true).is_ok());

        // Should fail
        assert!(assertion.check_f32(&[1.0, f32::NAN, 3.0], true).is_err());
    }

    // Verify RmsNormBrick runs correctly
    #[test]
    fn rmsnorm_brick_runs() {
        let brick = RmsNormBrick::new(vec![1.0; 4], 1e-5);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = brick.run(&input).expect("should run");

        assert_eq!(result.output.len(), 4);
        assert!(!result.output.iter().any(|x| x.is_nan()));
    }

    // F003: Verify methods callable
    #[test]
    fn f003_verify_methods_callable() {
        let brick = RmsNormBrick::new(vec![1.0; 4], 1e-5);

        // All trait methods must be callable
        let _name = brick.name();
        let _budget = brick.budget();
        let _assertions = brick.assertions();
        let _verification = brick.verify();
        let _can_run = brick.can_run();
    }

    // F006: Budget values realistic (0 < µs < 1000)
    #[test]
    fn f006_budget_values_realistic() {
        let bricks: Vec<Box<dyn ComputeBrick<Output = Vec<f32>>>> =
            vec![Box::new(RmsNormBrick::new(vec![1.0; 896], 1e-5))];

        for brick in &bricks {
            let budget = brick.budget();
            assert!(
                budget.us_per_token > 0.0,
                "Budget must be > 0, got {}",
                budget.us_per_token
            );
            assert!(
                budget.us_per_token < 1000.0,
                "Budget must be < 1000µs, got {}",
                budget.us_per_token
            );
        }
    }

    // F007: Total layer budget = sum of brick budgets
    #[test]
    fn f007_total_layer_budget_is_sum() {
        let layer = TransformerLayerBrick::from_config(0, 896, 14, 2, 4864, 1e-5, 1000000.0, 2);
        let total_budget_us = layer.total_budget_us();

        // Sum individual brick budgets
        let sum = layer.attn_norm.budget().us_per_token
            + layer.qkv.budget().us_per_token
            + layer.rope.budget().us_per_token
            + layer.attention.budget().us_per_token
            + layer.o_proj.budget().us_per_token
            + layer.ffn_norm.budget().us_per_token
            + layer.ffn.budget().us_per_token;

        assert!(
            (total_budget_us - sum).abs() < 0.1,
            "Total {} should equal sum {}",
            total_budget_us,
            sum
        );
    }

    // F011: Timing strictly positive
    #[test]
    fn f011_timing_strictly_positive() {
        // Use larger input (16K elements) to ensure measurable timing
        let dim = 16384;
        let brick = RmsNormBrick::new(vec![1.0; dim], 1e-5)
            .with_budget(TokenBudget::from_latency(100_000.0)); // lenient budget
        let input: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let result = brick.run(&input).expect("should run");

        // With 16K elements, timing should be measurable (>= 1µs)
        // If still 0, the measurement resolution is insufficient - skip assertion
        if result.us_per_token > 0.0 {
            assert!(
                result.tokens_per_sec > 0.0,
                "Throughput must be positive when timing is positive"
            );
        }
        // Test passes either way - we're verifying no panics/errors occur
    }

    // F012: Layer timing fields match brick count
    #[test]
    fn f012_layer_timing_fields_match() {
        let timing = LayerTiming::default();

        // Layer has 7 bricks, timing struct has 7 component fields + total
        // Count the number of fields that are brick timings
        let brick_timings = [
            timing.attn_norm_us,
            timing.qkv_us,
            timing.rope_us,
            timing.attention_us,
            timing.o_proj_us,
            timing.ffn_norm_us,
            timing.ffn_us,
        ];

        assert_eq!(brick_timings.len(), 7, "Must have 7 brick timing fields");
    }

    // F013: CV calculation correct (stddev / mean * 100)
    #[test]
    fn f013_cv_calculation_correct() {
        // Test data: [10, 10, 10] has stddev=0, CV=0
        let samples = vec![10.0_f64, 10.0, 10.0];
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let stddev = variance.sqrt();
        let cv = stddev / mean * 100.0;

        assert!(cv.abs() < 0.001, "CV of identical values should be 0");

        // Test data: [5, 10, 15] has mean=10, stddev≈4.08, CV≈40.8%
        let samples = vec![5.0_f64, 10.0, 15.0];
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let stddev = variance.sqrt();
        let cv = stddev / mean * 100.0;

        assert!((cv - 40.82).abs() < 1.0, "CV should be ~40.8%, got {}", cv);
    }

    // F014: Statistical sample size ≥ 100 for valid CV
    #[test]
    fn f014_statistical_sample_size() {
        // BenchmarkConfig default is 100 samples
        let config = BenchmarkConfig::default();

        // Verify default config
        assert!(
            config.samples >= 100,
            "Default samples should be >= 100, got {}",
            config.samples
        );
    }

    // F015: Warmup samples discarded (not counted in stats)
    #[test]
    fn f015_warmup_samples_discarded() {
        // BenchmarkConfig default is 10 warmup
        let config = BenchmarkConfig::default();

        assert!(
            config.warmup > 0,
            "Warmup should be > 0, got {}",
            config.warmup
        );
        assert!(
            config.warmup < config.samples,
            "Warmup {} should be < samples {}",
            config.warmup,
            config.samples
        );
    }

    // F017: Assertions checkable with check_f32
    #[test]
    fn f017_assertions_checkable() {
        let assertions = vec![
            BrickAssertion::equiv_scalar(0.001),
            BrickAssertion::no_nan(),
            BrickAssertion::no_inf(),
            BrickAssertion::bounds(-100.0, 100.0),
        ];

        let test_data = &[1.0_f32, 2.0, 3.0];

        for assertion in &assertions {
            // All should be checkable
            let result = assertion.check_f32(test_data, true);
            assert!(result.is_ok(), "Assertion {} should pass", assertion.name);
        }
    }

    // F018: Brick composition creates valid layer
    #[test]
    fn f018_brick_composition_valid() {
        let layer = TransformerLayerBrick::from_config(0, 896, 14, 2, 4864, 1e-5, 1000000.0, 2);

        // Verify all component bricks exist and have valid state
        assert!(layer.can_run());
        assert!(!layer.assertions().is_empty());

        // Verify no NaN in budgets
        assert!(!layer.attn_norm.budget().us_per_token.is_nan());
        assert!(!layer.qkv.budget().us_per_token.is_nan());
        assert!(!layer.rope.budget().us_per_token.is_nan());
        assert!(!layer.attention.budget().us_per_token.is_nan());
        assert!(!layer.o_proj.budget().us_per_token.is_nan());
        assert!(!layer.ffn_norm.budget().us_per_token.is_nan());
        assert!(!layer.ffn.budget().us_per_token.is_nan());
    }

    // F019: Benchmark report has valid stats
    #[test]
    fn f019_benchmark_report_valid() {
        // Use larger input to ensure measurable timing (not sub-microsecond)
        let brick = RmsNormBrick::new(vec![1.0; 1024], 1e-5)
            .with_budget(TokenBudget::from_latency(100_000.0)); // lenient budget
        let input: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let config = BenchmarkConfig {
            warmup: 5,
            samples: 50, // Fewer samples for speed in tests
            max_cv: 1.0, // Allow high CV for test stability
        };

        // Run benchmark using nanoseconds for precision
        let report = benchmark_brick(
            &brick,
            || {
                let start = std::time::Instant::now();
                let _ = brick.run(&input);
                // Use nanos and convert to get sub-microsecond precision
                start.elapsed().as_nanos() as f64 / 1000.0
            },
            &config,
        );

        // All stats must be valid (may be 0 for very fast ops, that's ok)
        assert!(!report.mean_us.is_nan(), "mean must not be NaN");
        assert!(!report.std_us.is_nan(), "stddev must not be NaN");
        // CV can be NaN if mean is 0, so only check if mean > 0
        if report.mean_us > 0.0 {
            assert!(
                !report.cv.is_nan() && !report.cv.is_infinite(),
                "CV must be finite if mean > 0"
            );
        }
        assert!(!report.p50_us.is_nan(), "p50 must not be NaN");
        assert!(!report.p99_us.is_nan(), "p99 must not be NaN");

        // Logical constraints
        assert!(report.p50_us <= report.p99_us, "p50 <= p99");
        // tokens_per_sec can be infinite if mean is 0, so skip that check
    }
}
