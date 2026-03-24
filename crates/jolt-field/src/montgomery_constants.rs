/// Montgomery field constants
///
/// Provides the constants needed to generate field-arithmetic shaders for any
/// Montgomery-form prime field. Values are in little-endian u32 limbs matching
/// the shader representation.
///
/// # Safety invariants
///
/// Implementations must guarantee the CIOS unreduced chaining property:
/// `4 * r^2 / R < 2r` where `R = 2^(32 * NUM_U32_LIMBS)`. This ensures that
/// intermediate products from `fr_mul_unreduced` remain in `[0, 2r)` and can
/// be safely fed into the next CIOS multiplication without explicit reduction.
pub trait MontgomeryConstants: 'static {
    /// Number of 32-bit limbs in the Montgomery representation.
    /// 4 for 128-bit fields, 8 for BN254 (256-bit).
    const NUM_U32_LIMBS: usize;

    /// Number of 32-bit limbs in the wide accumulator: `2 * NUM_U32_LIMBS + 2`.
    /// Provides headroom for accumulating ~2^32 unreduced products.
    const ACC_U32_LIMBS: usize;

    /// Byte size of a single field element: `NUM_U32_LIMBS * 4`.
    const FIELD_BYTE_SIZE: usize;

    /// The field modulus `r` as little-endian u32 limbs.
    fn modulus_u32() -> &'static [u32];

    /// `-r^{-1} mod 2^{32}` — the Montgomery reduction constant.
    fn inv32() -> u32;

    /// `R^2 mod r` as little-endian u32 limbs, where `R = 2^(32 * NUM_U32_LIMBS)`.
    /// Used for converting standard-form integers to Montgomery form.
    fn r2_u32() -> &'static [u32];

    /// `R mod r` as little-endian u32 limbs — the Montgomery representation of 1.
    fn one_u32() -> &'static [u32];
}
