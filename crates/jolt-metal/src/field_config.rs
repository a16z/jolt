//! Runtime field configuration for Metal shader parameterization.
//!
//! [`FieldConfig`] validates safety invariants at construction time and
//! caches the generated MSL preamble for use by the shader compiler.

use jolt_field::GpuFieldConfig;

use crate::msl_field_gen;

/// Validated field configuration with pre-generated MSL.
pub struct FieldConfig {
    pub n_limbs: usize,
    pub acc_limbs: usize,
    pub byte_size: usize,
    /// Complete field arithmetic MSL (Fr struct + constants + all operations + WideAcc).
    pub msl_preamble: String,
    /// Complete test kernel MSL (mul, add, sub, sqr, neg, fmadd, from_u64).
    pub msl_test_kernels: String,
}

impl FieldConfig {
    /// Construct a validated `FieldConfig` from a [`GpuFieldConfig`] implementation.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `ACC_U32_LIMBS != 2 * NUM_U32_LIMBS + 2`
    /// - `FIELD_BYTE_SIZE != NUM_U32_LIMBS * 4`
    /// - The CIOS unreduced chaining invariant `4r²/R < 2r` is violated
    pub fn from_gpu_field<F: GpuFieldConfig>() -> Self {
        let n = F::NUM_U32_LIMBS;

        // Structural invariants
        assert_eq!(
            F::ACC_U32_LIMBS,
            2 * n + 2,
            "ACC_U32_LIMBS must be 2*NUM_U32_LIMBS + 2"
        );
        assert_eq!(
            F::FIELD_BYTE_SIZE,
            n * 4,
            "FIELD_BYTE_SIZE must be NUM_U32_LIMBS * 4"
        );

        // Verify CIOS unreduced chaining: 4r²/R < 2r ⟺ 2r < R.
        // For any Montgomery field with N u32 limbs, R = 2^(32N).
        // r < 2^(32N) by definition (it's an N-limb number).
        // If the top limb is < 2^31, then r < 2^(32N-1) = R/2, so 2r < R. ✓
        let modulus = F::modulus_u32();
        assert!(
            modulus[n - 1] < (1u32 << 31),
            "CIOS unreduced chaining requires top limb < 2^31 (i.e., 2r < R)"
        );

        // Verify limb slice lengths
        assert_eq!(modulus.len(), n);
        assert_eq!(F::r2_u32().len(), n);
        assert_eq!(F::one_u32().len(), n);

        Self {
            n_limbs: n,
            acc_limbs: F::ACC_U32_LIMBS,
            byte_size: F::FIELD_BYTE_SIZE,
            msl_preamble: msl_field_gen::generate_full_preamble::<F>(),
            msl_test_kernels: msl_field_gen::generate_test_kernels::<F>(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;

    #[test]
    fn bn254_field_config_validates() {
        let config = FieldConfig::from_gpu_field::<Fr>();
        assert_eq!(config.n_limbs, 8);
        assert_eq!(config.acc_limbs, 18);
        assert_eq!(config.byte_size, 32);
        assert!(!config.msl_preamble.is_empty());
        assert!(!config.msl_test_kernels.is_empty());
    }
}
