//! Stable symbols for inspecting production scalar `2^64 - 59` arithmetic.

use jolt_field::Prime64Offset59;

#[inline(always)]
fn canonical(value: u64) -> Prime64Offset59 {
    // SAFETY: The machine theorem assumes that every witness input is below
    // the modulus before interpreting it as a field element.
    unsafe { Prime64Offset59::from_canonical_u64(value) }
}

/// Invoke public base-field addition on canonical representatives.
#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn jolt_fp64_add_production_witness(a: u64, b: u64) -> u64 {
    (canonical(a) + canonical(b)).to_canonical_u64()
}

/// Invoke public base-field subtraction on canonical representatives.
#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn jolt_fp64_sub_production_witness(a: u64, b: u64) -> u64 {
    (canonical(a) - canonical(b)).to_canonical_u64()
}

/// Invoke public base-field multiplication on canonical representatives.
#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn jolt_fp64_mul_production_witness(a: u64, b: u64) -> u64 {
    (canonical(a) * canonical(b)).to_canonical_u64()
}

fn main() {
    let result = std::hint::black_box((
        jolt_fp64_add_production_witness(1, 1),
        jolt_fp64_sub_production_witness(1, 1),
        jolt_fp64_mul_production_witness(1, 1),
    ));
    let _ = std::hint::black_box(result);
}
