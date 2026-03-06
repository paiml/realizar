//! Auto-detected GPU kernel configuration.
//!
//! Replaces per-machine env var tuning (`DP4A_Q4K`, `HW_DP4A_Q4K`, `MWV_Q6K`, etc.)
//! with automatic detection based on `compute_capability()`.
//!
//! Env vars still work as overrides for experimentation, but the defaults are
//! now correct for each GPU — no forjar config drift.

use trueno_gpu::driver::CudaContext;

/// Kernel variant for Q4K GEMV dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q4kVariant {
    /// Legacy single-warp (32 threads), no DP4A. Fallback for sm < 7.5.
    Legacy,
    /// Wide: 128 threads per output row.
    Wide,
    /// Vectorized: 32 threads with vectorized loads.
    Vectorized,
    /// Multi-warp DP4A: 32 threads/super-block with shfl broadcast.
    MwvDp4a,
    /// Half-warp DP4A: 16 threads/super-block, direct scale loads. Best on sm_75+.
    HwDp4a,
    /// Multi-warp vectorized (no DP4A).
    Mwv,
}

/// Kernel variant for Q6K GEMV dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q6kVariant {
    /// Original single-warp Q6K (fallback).
    Legacy,
    /// Multi-warp vectorized Q6K (GH-118).
    Mwv,
    /// DP4A Q6K with Q8 pre-quantization.
    Dp4a,
}

/// Auto-detected GPU profile for kernel dispatch.
///
/// Computed once at executor init from `compute_capability()`.
/// All kernel dispatch reads from this instead of env vars.
#[derive(Debug, Clone)]
pub struct GpuProfile {
    /// Q4K GEMV kernel variant (auto-detected: HwDp4a on sm_75+).
    pub q4k: Q4kVariant,
    /// Q6K GEMV kernel variant (auto-detected: Dp4a on sm_75+).
    pub q6k: Q6kVariant,
    /// Multi-warp GEMV warp count (default: 3, override: MWV_WARPS env).
    pub mwv_warps: u32,
    /// Whether batched prefill is enabled (default: true, override: BATCHED_PREFILL=0).
    pub batched_prefill: bool,
    /// SM version for logging (e.g., "sm_89").
    pub sm_target: String,
}

impl GpuProfile {
    /// Detect optimal kernel configuration from GPU hardware.
    ///
    /// Priority: env var override > auto-detect from compute capability.
    /// This means `HW_DP4A_Q4K=1` still works for experimentation,
    /// but production deployments need zero env vars.
    pub fn detect(context: &CudaContext) -> Self {
        let (major, minor) = context.compute_capability().unwrap_or((7, 0));
        let sm_target = format!("sm_{major}{minor}");
        let has_dp4a = major > 7 || (major == 7 && minor >= 5); // sm_75+ (Turing)

        let q4k = Self::detect_q4k(has_dp4a);
        let q6k = Self::detect_q6k(has_dp4a);
        let mwv_warps = Self::detect_mwv_warps();
        let batched_prefill = Self::detect_batched_prefill();

        let profile = Self { q4k, q6k, mwv_warps, batched_prefill, sm_target };

        eprintln!(
            "[GpuProfile] {}: q4k={:?}, q6k={:?}, warps={}, batched_prefill={}",
            profile.sm_target, profile.q4k, profile.q6k, profile.mwv_warps, profile.batched_prefill,
        );

        profile
    }

    /// Q4K variant: env var override, else HwDp4a on sm_75+, else Mwv.
    fn detect_q4k(has_dp4a: bool) -> Q4kVariant {
        // Env var overrides (for experimentation only)
        if std::env::var("WIDE_Q4K_DISABLE").is_ok() { return Q4kVariant::Legacy; }
        if std::env::var("WIDE_Q4K").is_ok() { return Q4kVariant::Wide; }
        if std::env::var("VECTORIZED_Q4K").is_ok() { return Q4kVariant::Vectorized; }
        if std::env::var("HW_DP4A_Q4K").is_ok() { return Q4kVariant::HwDp4a; }
        if std::env::var("DP4A_Q4K").is_ok() { return Q4kVariant::MwvDp4a; }

        // Auto-detect: HW DP4A on any GPU with DP4A support (sm_75+)
        if has_dp4a {
            Q4kVariant::HwDp4a
        } else {
            Q4kVariant::Mwv
        }
    }

    /// Q6K variant: env var override, else Dp4a on sm_75+, else Mwv.
    fn detect_q6k(has_dp4a: bool) -> Q6kVariant {
        if std::env::var("DP4A_Q6K").is_ok() { return Q6kVariant::Dp4a; }
        if std::env::var("MWV_Q6K").is_ok() { return Q6kVariant::Mwv; }

        if has_dp4a {
            Q6kVariant::Dp4a
        } else {
            Q6kVariant::Mwv
        }
    }

    /// MWV warp count: env var override, else 3 (empirical optimum for both Orin and 4090).
    fn detect_mwv_warps() -> u32 {
        std::env::var("MWV_WARPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3)
    }

    /// Batched prefill: env var override, else always on.
    fn detect_batched_prefill() -> bool {
        // BATCHED_PREFILL=0 disables; any other value or absent = enabled
        std::env::var("BATCHED_PREFILL")
            .map(|v| v != "0")
            .unwrap_or(true)
    }
}
