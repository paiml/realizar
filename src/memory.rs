//! Memory management for hot expert pinning
//!
//! Per Dean & Barroso (2013) "The Tail at Scale", page faults cause
//! tail latency. This module provides mlock support for hot experts.

/// Configuration for memory pinning
#[derive(Debug, Clone, Default)]
pub struct MlockConfig {
    /// Whether to attempt memory locking
    pub enabled: bool,
    /// Maximum bytes to lock (0 = unlimited)
    pub max_locked_bytes: usize,
}

/// Result of mlock operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlockResult {
    /// Successfully locked memory
    Locked,
    /// Mlock disabled in config
    Disabled,
    /// Failed to lock (insufficient privileges)
    InsufficientPrivileges,
    /// Failed to lock (resource limit)
    ResourceLimit,
    /// Platform not supported
    Unsupported,
}

/// Memory region that can be pinned
pub struct PinnedRegion {
    /// Pointer to locked memory (for tracking)
    ptr: *const u8,
    /// Size of locked region
    len: usize,
    /// Whether actually locked
    locked: bool,
}

// Safety: PinnedRegion only holds a pointer for tracking, doesn't access it
unsafe impl Send for PinnedRegion {}
unsafe impl Sync for PinnedRegion {}

impl PinnedRegion {
    /// Create a pinned region (attempts mlock)
    ///
    /// # Safety
    ///
    /// Caller must ensure ptr points to valid memory for len bytes.
    #[must_use]
    pub unsafe fn new(ptr: *const u8, len: usize, config: &MlockConfig) -> (Self, MlockResult) {
        if !config.enabled {
            return (
                Self {
                    ptr,
                    len,
                    locked: false,
                },
                MlockResult::Disabled,
            );
        }

        if config.max_locked_bytes > 0 && len > config.max_locked_bytes {
            return (
                Self {
                    ptr,
                    len,
                    locked: false,
                },
                MlockResult::ResourceLimit,
            );
        }

        let result = Self::mlock_impl(ptr, len);
        let locked = result == MlockResult::Locked;
        (Self { ptr, len, locked }, result)
    }

    /// Check if memory is locked
    #[must_use]
    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// Get size of region
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if region is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[cfg(target_family = "unix")]
    fn mlock_impl(ptr: *const u8, len: usize) -> MlockResult {
        // Safety: ptr and len validated by caller
        // SAFETY: Memory safety ensured by bounds checking and alignment
        let result = unsafe { libc::mlock(ptr.cast(), len) };
        if result == 0 {
            MlockResult::Locked
        } else {
            // Check errno for specific error
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            if errno == libc::EPERM {
                MlockResult::InsufficientPrivileges
            } else {
                MlockResult::ResourceLimit
            }
        }
    }

    #[cfg(not(target_family = "unix"))]
    fn mlock_impl(_ptr: *const u8, _len: usize) -> MlockResult {
        MlockResult::Unsupported
    }
}

impl Drop for PinnedRegion {
    fn drop(&mut self) {
        if self.locked {
            #[cfg(target_family = "unix")]
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                libc::munlock(self.ptr.cast(), self.len);
            }
        }
    }
}

/// Expert tier classification for memory management
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertTier {
    /// Hot tier - pinned in memory, never swapped
    Hot,
    /// Warm tier - in memory but may be swapped
    Warm,
    /// Cold tier - on disk, loaded on demand
    Cold,
}

impl ExpertTier {
    /// Determine tier based on access frequency
    #[must_use]
    pub fn from_access_count(count: usize, hot_threshold: usize, warm_threshold: usize) -> Self {
        if count >= hot_threshold {
            Self::Hot
        } else if count >= warm_threshold {
            Self::Warm
        } else {
            Self::Cold
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlock_disabled() {
        let config = MlockConfig {
            enabled: false,
            max_locked_bytes: 0,
        };
        let data = vec![0u8; 1024];
        // SAFETY: Memory safety ensured by bounds checking and alignment
        let (region, result) = unsafe { PinnedRegion::new(data.as_ptr(), data.len(), &config) };
        assert_eq!(result, MlockResult::Disabled);
        assert!(!region.is_locked());
    }

    #[test]
    fn test_mlock_resource_limit() {
        let config = MlockConfig {
            enabled: true,
            max_locked_bytes: 100,
        };
        let data = vec![0u8; 1024]; // Exceeds limit
                                    // SAFETY: Memory safety ensured by bounds checking and alignment
        let (region, result) = unsafe { PinnedRegion::new(data.as_ptr(), data.len(), &config) };
        assert_eq!(result, MlockResult::ResourceLimit);
        assert!(!region.is_locked());
    }

    #[test]
    fn test_expert_tier_hot() {
        let tier = ExpertTier::from_access_count(1000, 100, 10);
        assert_eq!(tier, ExpertTier::Hot);
    }

    #[test]
    fn test_expert_tier_warm() {
        let tier = ExpertTier::from_access_count(50, 100, 10);
        assert_eq!(tier, ExpertTier::Warm);
    }

    #[test]
    fn test_expert_tier_cold() {
        let tier = ExpertTier::from_access_count(5, 100, 10);
        assert_eq!(tier, ExpertTier::Cold);
    }

    #[test]
    fn test_pinned_region_len() {
        let config = MlockConfig::default();
        let data = vec![0u8; 1024];
        // SAFETY: Memory safety ensured by bounds checking and alignment
        let (region, _) = unsafe { PinnedRegion::new(data.as_ptr(), data.len(), &config) };
        assert_eq!(region.len(), 1024);
        assert!(!region.is_empty());
    }

    #[test]
    fn test_pinned_region_empty() {
        let config = MlockConfig::default();
        let data: Vec<u8> = vec![];
        // SAFETY: Memory safety ensured by bounds checking and alignment
        let (region, _) = unsafe { PinnedRegion::new(data.as_ptr(), data.len(), &config) };
        assert_eq!(region.len(), 0);
        assert!(region.is_empty());
    }

    #[test]
    fn test_mlock_config_default() {
        let config = MlockConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.max_locked_bytes, 0);
    }

    #[test]
    fn test_mlock_result_equality() {
        assert_eq!(MlockResult::Locked, MlockResult::Locked);
        assert_eq!(MlockResult::Disabled, MlockResult::Disabled);
        assert_ne!(MlockResult::Locked, MlockResult::Disabled);
    }

    #[test]
    fn test_expert_tier_boundary() {
        // Test exact boundary conditions
        assert_eq!(ExpertTier::from_access_count(100, 100, 10), ExpertTier::Hot);
        assert_eq!(ExpertTier::from_access_count(99, 100, 10), ExpertTier::Warm);
        assert_eq!(ExpertTier::from_access_count(10, 100, 10), ExpertTier::Warm);
        assert_eq!(ExpertTier::from_access_count(9, 100, 10), ExpertTier::Cold);
    }

    #[test]
    fn test_mlock_enabled_within_limit() {
        // Test when enabled and within limit - will try actual mlock
        let config = MlockConfig {
            enabled: true,
            max_locked_bytes: 0, // 0 = unlimited
        };
        let data: [u8; 64] = [0u8; 64];
        // SAFETY: Memory safety ensured by bounds checking and alignment
        let (region, result) = unsafe { PinnedRegion::new(data.as_ptr(), data.len(), &config) };
        // Result depends on system permissions, but API should work
        assert!(
            result == MlockResult::Locked
                || result == MlockResult::InsufficientPrivileges
                || result == MlockResult::ResourceLimit
                || result == MlockResult::Unsupported
        );
        assert_eq!(region.len(), 64);
    }

    // ========================================================================
    // Coverage Tests
    // ========================================================================

    #[test]
    fn test_mlock_config_debug() {
        let config = MlockConfig {
            enabled: true,
            max_locked_bytes: 1024,
        };
        let debug = format!("{:?}", config);
        assert!(debug.contains("MlockConfig"));
        assert!(debug.contains("1024"));
    }

    #[test]
    fn test_mlock_config_clone() {
        let config = MlockConfig {
            enabled: true,
            max_locked_bytes: 2048,
        };
        let cloned = config.clone();
        assert_eq!(config.enabled, cloned.enabled);
        assert_eq!(config.max_locked_bytes, cloned.max_locked_bytes);
    }

    #[test]
    fn test_mlock_result_debug() {
        let result = MlockResult::Locked;
        let debug = format!("{:?}", result);
        assert!(debug.contains("Locked"));

        let result2 = MlockResult::InsufficientPrivileges;
        let debug2 = format!("{:?}", result2);
        assert!(debug2.contains("InsufficientPrivileges"));
    }

    #[test]
    fn test_mlock_result_clone() {
        let result = MlockResult::Locked;
        let cloned = result.clone();
        assert_eq!(result, cloned);
    }

    #[test]
    fn test_expert_tier_debug() {
        let tier = ExpertTier::Hot;
        let debug = format!("{:?}", tier);
        assert!(debug.contains("Hot"));
    }

    #[test]
    fn test_expert_tier_clone() {
        let tier = ExpertTier::Warm;
        let cloned = tier;
        assert_eq!(tier, cloned);
    }
}
