//! Metal device configuration for kernel dispatch tuning.

/// Metal-specific tuning parameters for kernel dispatch.
#[derive(Clone, Debug)]
pub struct MetalDeviceConfig {
    /// Threadgroup size for reduce (sumcheck) kernels. Power of 2.
    /// 128 threads = 4 simdgroups. Smaller groups improve GPU scheduling
    /// for register-heavy kernels (D>=8). Benchmarks show +9% for BN254 D=8 vs 256.
    pub reduce_group_size: usize,
    /// Maximum threadgroups per reduce dispatch.
    pub max_reduce_groups: usize,
    /// SIMD width (threads per simdgroup). 32 on all Apple GPUs.
    pub simd_size: usize,
}

impl Default for MetalDeviceConfig {
    fn default() -> Self {
        Self {
            reduce_group_size: 128,
            max_reduce_groups: 256,
            simd_size: 32,
        }
    }
}

impl MetalDeviceConfig {
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
        assert!(config.max_reduce_groups > 0);
    }
}
