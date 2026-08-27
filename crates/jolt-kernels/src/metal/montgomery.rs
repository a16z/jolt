//! Montgomery field constants for the Metal shader generator.
//!
//! Provides the constants needed to generate field-arithmetic shaders for a
//! Montgomery-form prime field. Values are little-endian u32 limbs matching
//! the shader representation, derived from the arkworks `MontConfig` the
//! `jolt_field` BN254 newtypes wrap — no limb value is hand-written.
//!
//! # Safety invariants
//!
//! Implementations must guarantee the CIOS unreduced chaining property:
//! `4 * r^2 / R < 2r` where `R = 2^(32 * NUM_U32_LIMBS)`. This ensures that
//! intermediate products from `fr_mul_unreduced` remain in `[0, 2r)` and can
//! be safely fed into the next CIOS multiplication without explicit
//! reduction. BN254's moduli are < 2^254 = R/4, so it holds for both fields.

use ark_bn254::{FqConfig, FrConfig};
use ark_ff::MontConfig;
use jolt_field::{Fq, Fr};

/// Montgomery-form constants of a shader-resident prime field.
pub trait MontgomeryConstants: 'static {
    /// Number of 32-bit limbs in the Montgomery representation.
    /// 8 for BN254 (256-bit).
    const NUM_U32_LIMBS: usize;

    /// Number of 32-bit limbs in the wide accumulator: `2 * NUM_U32_LIMBS + 2`.
    /// Provides headroom for accumulating ~2^32 unreduced products.
    const ACC_U32_LIMBS: usize;

    /// Byte size of a single field element: `NUM_U32_LIMBS * 4`.
    const FIELD_BYTE_SIZE: usize;

    /// The field modulus `r` as little-endian u32 limbs (`NUM_U32_LIMBS` elements).
    fn modulus_u32() -> &'static [u32];

    /// `-r^{-1} mod 2^{32}` — the Montgomery reduction constant.
    fn inv32() -> u32;

    /// `R^2 mod r` as little-endian u32 limbs (`NUM_U32_LIMBS` elements),
    /// where `R = 2^(32 * NUM_U32_LIMBS)`.
    fn r2_u32() -> &'static [u32];

    /// `R mod r` as little-endian u32 limbs (`NUM_U32_LIMBS` elements) —
    /// the Montgomery representation of 1.
    fn one_u32() -> &'static [u32];
}

// The u32 limbs are derived from arkworks' FrConfig u64 limbs by splitting
// each u64 into (lo, hi) u32 pairs. This matches the little-endian byte layout
// on ARM64 — the same bytes that represent [u64; 4] on CPU are read as [u32; 8]
// by the Metal shader.

const MODULUS: [u64; 4] = <FrConfig as MontConfig<4>>::MODULUS.0;
const R: [u64; 4] = <FrConfig as MontConfig<4>>::R.0;
const R2: [u64; 4] = <FrConfig as MontConfig<4>>::R2.0;
const INV64: u64 = <FrConfig as MontConfig<4>>::INV;

const fn u64s_to_u32s(limbs: &[u64; 4]) -> [u32; 8] {
    [
        limbs[0] as u32,
        (limbs[0] >> 32) as u32,
        limbs[1] as u32,
        (limbs[1] >> 32) as u32,
        limbs[2] as u32,
        (limbs[2] >> 32) as u32,
        limbs[3] as u32,
        (limbs[3] >> 32) as u32,
    ]
}

static MODULUS_U32: [u32; 8] = u64s_to_u32s(&MODULUS);
static R2_U32: [u32; 8] = u64s_to_u32s(&R2);
static ONE_U32: [u32; 8] = u64s_to_u32s(&R);

/// `-r^{-1} mod 2^{32}` derived from the arkworks 64-bit `INV` value.
/// arkworks stores `-r^{-1} mod 2^{64}`; the low 32 bits give `mod 2^{32}`.
const INV32: u32 = INV64 as u32;

impl MontgomeryConstants for Fr {
    const NUM_U32_LIMBS: usize = 8;
    const ACC_U32_LIMBS: usize = 18; // 2*8 + 2
    const FIELD_BYTE_SIZE: usize = 32; // 8 * 4

    fn modulus_u32() -> &'static [u32] {
        &MODULUS_U32
    }

    fn inv32() -> u32 {
        INV32
    }

    fn r2_u32() -> &'static [u32] {
        &R2_U32
    }

    fn one_u32() -> &'static [u32] {
        &ONE_U32
    }
}

// The BN254 base field (G1 coordinate field), same limb derivation as Fr.

const FQ_MODULUS: [u64; 4] = <FqConfig as MontConfig<4>>::MODULUS.0;
const FQ_R: [u64; 4] = <FqConfig as MontConfig<4>>::R.0;
const FQ_R2: [u64; 4] = <FqConfig as MontConfig<4>>::R2.0;
const FQ_INV64: u64 = <FqConfig as MontConfig<4>>::INV;

static FQ_MODULUS_U32: [u32; 8] = u64s_to_u32s(&FQ_MODULUS);
static FQ_R2_U32: [u32; 8] = u64s_to_u32s(&FQ_R2);
static FQ_ONE_U32: [u32; 8] = u64s_to_u32s(&FQ_R);

const FQ_INV32: u32 = FQ_INV64 as u32;

impl MontgomeryConstants for Fq {
    const NUM_U32_LIMBS: usize = 8;
    const ACC_U32_LIMBS: usize = 18; // 2*8 + 2
    const FIELD_BYTE_SIZE: usize = 32; // 8 * 4

    fn modulus_u32() -> &'static [u32] {
        &FQ_MODULUS_U32
    }

    fn inv32() -> u32 {
        FQ_INV32
    }

    fn r2_u32() -> &'static [u32] {
        &FQ_R2_U32
    }

    fn one_u32() -> &'static [u32] {
        &FQ_ONE_U32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bn254_modulus_matches_shader() {
        // These are the constants from the original bn254_fr.metal shader.
        let expected: [u32; 8] = [
            0xf000_0001,
            0x43e1_f593,
            0x79b9_7091,
            0x2833_e848,
            0x8181_585d,
            0xb850_45b6,
            0xe131_a029,
            0x3064_4e72,
        ];
        assert_eq!(MODULUS_U32, expected);
    }

    #[test]
    fn bn254_inv32_matches_shader() {
        assert_eq!(INV32, 0xefff_ffff);
    }

    #[test]
    fn bn254_r2_matches_shader() {
        let expected: [u32; 8] = [
            0xae21_6da7,
            0x1bb8_e645,
            0xe35c_59e3,
            0x53fe_3ab1,
            0x53bb_8085,
            0x8c49_833d,
            0x7f4e_44a5,
            0x0216_d0b1,
        ];
        assert_eq!(R2_U32, expected);
    }

    #[test]
    fn bn254_one_matches_shader() {
        let expected: [u32; 8] = [
            0x4fff_fffb,
            0xac96_341c,
            0x9f60_cd29,
            0x36fc_7695,
            0x7879_462e,
            0x666e_a36f,
            0x9a07_df2f,
            0x0e0a_77c1,
        ];
        assert_eq!(ONE_U32, expected);
    }

    #[test]
    fn acc_limbs_invariant() {
        assert_eq!(
            <Fr as MontgomeryConstants>::ACC_U32_LIMBS,
            2 * <Fr as MontgomeryConstants>::NUM_U32_LIMBS + 2
        );
    }

    #[test]
    fn bn254_fq_modulus_known_value() {
        // q = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
        let expected: [u32; 8] = [
            0xd87c_fd47,
            0x3c20_8c16,
            0x6871_ca8d,
            0x9781_6a91,
            0x8181_585d,
            0xb850_45b6,
            0xe131_a029,
            0x3064_4e72,
        ];
        assert_eq!(FQ_MODULUS_U32, expected);
    }

    #[test]
    fn bn254_fq_inv32_is_negative_inverse() {
        // q · (-q^{-1}) ≡ -1 (mod 2^32)
        assert_eq!(FQ_MODULUS_U32[0].wrapping_mul(FQ_INV32), u32::MAX);
        assert_eq!(FQ_INV32, 0xe486_6389);
    }

    #[test]
    fn field_byte_size_invariant() {
        assert_eq!(
            <Fr as MontgomeryConstants>::FIELD_BYTE_SIZE,
            <Fr as MontgomeryConstants>::NUM_U32_LIMBS * 4
        );
    }
}
