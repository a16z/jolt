//! Stable symbols for inspecting production A7F7 arithmetic kernels.

use jolt_field::Prime128OffsetA7F7;

/// Two 64-bit limbs returned through the platform C calling convention.
#[repr(C)]
pub struct Fp128Result {
    lo: u64,
    hi: u64,
}

/// Invoke public field addition for the proved A7F7 modulus.
///
/// The two input values must be canonical field representatives.
#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn jolt_fp128_add_production_witness(
    a_lo: u64,
    a_hi: u64,
    b_lo: u64,
    b_hi: u64,
) -> Fp128Result {
    // SAFETY: This inspection symbol has the same canonical input contract as
    // the machine theorem.
    let a = unsafe { Prime128OffsetA7F7::from_canonical_u128((a_hi as u128) << 64 | a_lo as u128) };
    // SAFETY: See the function contract above.
    let b = unsafe { Prime128OffsetA7F7::from_canonical_u128((b_hi as u128) << 64 | b_lo as u128) };
    let [lo, hi] = (a + b).to_limbs();
    Fp128Result { lo, hi }
}

/// Invoke public field subtraction for the proved A7F7 modulus.
///
/// The two input values must be canonical field representatives.
#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn jolt_fp128_sub_production_witness(
    a_lo: u64,
    a_hi: u64,
    b_lo: u64,
    b_hi: u64,
) -> Fp128Result {
    // SAFETY: This inspection symbol has the same canonical input contract as
    // the machine theorem.
    let a = unsafe { Prime128OffsetA7F7::from_canonical_u128((a_hi as u128) << 64 | a_lo as u128) };
    // SAFETY: See the function contract above.
    let b = unsafe { Prime128OffsetA7F7::from_canonical_u128((b_hi as u128) << 64 | b_lo as u128) };
    let [lo, hi] = (a - b).to_limbs();
    Fp128Result { lo, hi }
}

/// Invoke public field multiplication for the proved A7F7 modulus.
///
/// The two input values must be canonical field representatives.
#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn jolt_fp128_mul_production_witness(
    a_lo: u64,
    a_hi: u64,
    b_lo: u64,
    b_hi: u64,
) -> Fp128Result {
    // SAFETY: This inspection symbol has the same canonical input contract as
    // the machine theorem.
    let a = unsafe { Prime128OffsetA7F7::from_canonical_u128((a_hi as u128) << 64 | a_lo as u128) };
    // SAFETY: See the function contract above.
    let b = unsafe { Prime128OffsetA7F7::from_canonical_u128((b_hi as u128) << 64 | b_lo as u128) };
    let [lo, hi] = (a * b).to_limbs();
    Fp128Result { lo, hi }
}

fn main() {
    let add = std::hint::black_box(jolt_fp128_add_production_witness(1, 0, 1, 0));
    let sub = std::hint::black_box(jolt_fp128_sub_production_witness(1, 0, 1, 0));
    let mul = std::hint::black_box(jolt_fp128_mul_production_witness(1, 0, 1, 0));
    let _ = std::hint::black_box((add, sub, mul));
}
