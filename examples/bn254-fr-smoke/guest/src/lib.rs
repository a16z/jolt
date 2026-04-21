#![cfg_attr(feature = "guest", no_std)]

use jolt_inlines_bn254_fr::Fr;

/// BN254 Fr smoke: compute `(a + b) * a` using the native-field coprocessor.
/// Exercises FADD and FMUL in one call, with both operands flowing through
/// the limb-register ABI.
///
/// The guest is traced; the emitted `FieldOp` / `FMov{I2F,F2I}` instructions
/// land in the trace at their real cycle indices. Host-side R1CS witness
/// populates the FieldOp flag slots from `CycleRow::circuit_flags` and the
/// event overlay fills the operand columns. Spartan's widened outer uniskip
/// (task #63) enforces FADD gate 19 and FMUL gates 21-23.
#[jolt::provable(heap_size = 32768, max_trace_length = 65536)]
fn fr_add_mul(
    a_lo: u64,
    a_1: u64,
    a_2: u64,
    a_3: u64,
    b_lo: u64,
    b_1: u64,
    b_2: u64,
    b_3: u64,
) -> [u64; 4] {
    let a = Fr::from_limbs([a_lo, a_1, a_2, a_3]);
    let b = Fr::from_limbs([b_lo, b_1, b_2, b_3]);
    let sum = a.add(&b);
    let prod = sum.mul(&a);
    prod.to_limbs()
}
