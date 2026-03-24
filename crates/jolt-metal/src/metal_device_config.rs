//! Metal device configuration with automatic hardware detection.
//!
//! [`MetalDeviceConfig`] replaces hardcoded constants (`REDUCE_GROUP_SIZE`,
//! `MAX_REDUCE_GROUPS`, `SIMD_SIZE`, `SPLIT_PASS_THRESHOLD`) with
//! values tuned per Apple GPU generation.

use metal::Device;

/// Metal-specific tuning parameters for kernel dispatch.
#[derive(Clone, Debug)]
pub struct MetalDeviceConfig {
    /// Threadgroup size for pairwise_reduce (sumcheck) kernels. Power of 2.
    /// Set to 128 by default -- smaller groups improve GPU scheduling
    /// for register-heavy kernels (D>=8). Benchmarks show +9% for BN254 D=8.
    pub reduce_group_size: usize,
    /// Threadgroup size for elementwise sum/dot_product kernels.
    /// These are lightweight kernels with low register pressure, so the
    /// full 256-thread groups work well and match the hardcoded
    /// `SUM_GROUP_SIZE` in elementwise.metal.
    pub elementwise_group_size: usize,
    /// Maximum threadgroups per reduce dispatch.
    pub max_reduce_groups: usize,
    /// SIMD width (threads per simdgroup).
    pub simd_size: usize,
    /// ProductSum D threshold for split-pass kernel generation.
    /// Kernels with `D >= split_pass_threshold` use multi-pass streaming.
    /// Currently disabled on all hardware (set very high).
    pub split_pass_threshold: usize,
}

impl Default for MetalDeviceConfig {
    fn default() -> Self {
        Self {
            // 128 threads = 4 simdgroups per threadgroup. Smaller groups
            // give the GPU more scheduling flexibility for register-heavy
            // kernels (D>=8). Benchmarks show +9% for BN254 D=8 vs 256.
            reduce_group_size: 128,
            elementwise_group_size: 256,
            max_reduce_groups: 256,
            simd_size: 32,
            split_pass_threshold: 1024,
        }
    }
}

impl MetalDeviceConfig {
    /// Detect optimal configuration from the Metal device.
    ///
    /// M4+ GPUs have a larger effective register file (dynamic caching)
    /// which could enable lower split-pass thresholds. Currently all
    /// generations use the same conservative defaults -- this provides
    /// the hook for future per-generation tuning.
    pub fn detect(device: &Device) -> Self {
        let name = device.name().to_string().to_lowercase();

        if name.contains("m4") || name.contains("m5") {
            Self {
                reduce_group_size: 128,
                elementwise_group_size: 256,
                max_reduce_groups: 256,
                simd_size: 32,
                split_pass_threshold: 1024,
            }
        } else {
            // M1/M2/M3: conservative defaults
            Self::default()
        }
    }

    /// Number of simdgroups per threadgroup.
    #[inline]
    pub fn num_simdgroups(&self) -> usize {
        self.reduce_group_size / self.simd_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_consistent() {
        let config = MetalDeviceConfig::default();
        assert_eq!(config.reduce_group_size % config.simd_size, 0);
        assert!(config.reduce_group_size.is_power_of_two());
        assert!(config.elementwise_group_size.is_power_of_two());
        assert!(config.max_reduce_groups > 0);
    }

    #[test]
    fn detect_returns_valid_config() {
        let device = Device::system_default().expect("no Metal device");
        let config = MetalDeviceConfig::detect(&device);
        assert_eq!(config.reduce_group_size % config.simd_size, 0);
        assert!(config.reduce_group_size.is_power_of_two());
    }
}
