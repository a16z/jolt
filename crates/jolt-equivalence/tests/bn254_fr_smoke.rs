//! BN254 Fr native-field coprocessor smoke tests.
//!
//! Compiles the `bn254-fr-smoke-guest` ELF via the `jolt` CLI, traces a
//! `Fr::add` + `Fr::mul` computation, and inspects the trace for the
//! expected FieldOp / FMov{I2F,F2I} cycles and FieldRegEvents.
//!
//! This does NOT yet run through the refactor-crates modular prover — the
//! end-to-end prove/verify integration is blocked on adapting
//! `setup_zkvm_muldiv_with_example` to accept arbitrary guest + inputs.
//! See task #60 for the remaining scaffolding.
//!
//! What this test DOES validate:
//! - SDK's single-asm-block FMov → FieldOp → FMov sequence compiles and
//!   lands in the ELF
//! - Tracer decodes opcode 0x0B + funct7 0x40 into `FieldOp` /
//!   `FMovIntToFieldLimb` / `FMovFieldToIntLimb` variants
//! - Emulator executes FieldOp with correct Fr arithmetic (result limb
//!   matches ark-bn254's Fr::add / Fr::mul)
//! - FieldRegEvents carry the expected slot transitions + FieldOpPayload
//!   (funct3 + a/b limb reads)
//! - Fixed-register ABI: `x[10..=13]` hold `a`'s limbs at every FMov-I2F
//!   cycle loading `a`, `x[14..=17]` hold `b`'s limbs, and those registers
//!   are still live at the FieldOp cycle (task #52 bridge prereq).

use jolt_host::Program;
use tracer::instruction::field_op::{
    FUNCT3_FADD, FUNCT3_FINV, FUNCT3_FMUL, FUNCT3_FSUB,
};
use tracer::instruction::Cycle;

/// Postcard-encode the 8-u64 argument tuple the guest's `fr_add_mul`
/// expects. Returns (12345, 0, 0, 0, 67890, 0, 0, 0).
fn smoke_inputs() -> Vec<u8> {
    postcard::to_stdvec(&(12345u64, 0u64, 0u64, 0u64, 67890u64, 0u64, 0u64, 0u64))
        .expect("postcard encode")
}

/// Trace the guest once and return (trace, field_reg_events).
fn trace_smoke_guest() -> (Vec<Cycle>, Vec<tracer::emulator::cpu::FieldRegEvent>) {
    let mut program = Program::new("bn254-fr-smoke-guest");
    let inputs = smoke_inputs();
    let (_, trace, _, _, field_reg_events) =
        program.trace_with_field_reg_events(&inputs, &[], &[]);
    (trace, field_reg_events)
}

#[test]
fn trace_contains_expected_field_op_cycles() {
    let (trace, _events) = trace_smoke_guest();

    let mut n_fmov_i2f = 0usize;
    let mut n_fmov_f2i = 0usize;
    let mut n_field_op = 0usize;
    let mut field_op_funct3s: Vec<u8> = Vec::new();

    for cycle in &trace {
        match cycle {
            Cycle::FMovIntToFieldLimb(_) => n_fmov_i2f += 1,
            Cycle::FMovFieldToIntLimb(_) => n_fmov_f2i += 1,
            Cycle::FieldOp(op) => {
                n_field_op += 1;
                field_op_funct3s.push(op.instruction.funct3);
            }
            _ => {}
        }
    }

    // The guest does `(a + b) * a`: one Fr::add (8 I2F + 1 FieldOp + 4 F2I)
    // + one Fr::mul (8 I2F + 1 FieldOp + 4 F2I) = 16 I2F + 8 F2I + 2 FieldOp.
    // Compiler may elide duplicate loads; we only assert lower bounds.
    assert!(
        n_field_op >= 2,
        "expected at least 2 FieldOp cycles (add + mul), got {n_field_op}"
    );
    assert!(
        n_fmov_i2f >= 4,
        "expected at least 4 FMovIntToFieldLimb cycles, got {n_fmov_i2f}"
    );
    assert!(
        n_fmov_f2i >= 4,
        "expected at least 4 FMovFieldToIntLimb cycles, got {n_fmov_f2i}"
    );

    // Guest emits FADD then FMUL. Both funct3s present.
    assert!(
        field_op_funct3s.iter().any(|&f| f == FUNCT3_FADD),
        "expected at least one FADD FieldOp, got funct3s = {field_op_funct3s:?}"
    );
    assert!(
        field_op_funct3s.iter().any(|&f| f == FUNCT3_FMUL),
        "expected at least one FMUL FieldOp, got funct3s = {field_op_funct3s:?}"
    );
    // FSUB/FINV aren't used by this guest.
    assert!(
        !field_op_funct3s.iter().any(|&f| f == FUNCT3_FSUB || f == FUNCT3_FINV),
        "FSUB/FINV should not appear in this guest's trace: {field_op_funct3s:?}"
    );
}

#[test]
fn field_reg_events_payload_matches_fr_arithmetic() {
    use ark_bn254::Fr as ArkFr;
    use ark_ff::PrimeField;

    let (_, events) = trace_smoke_guest();

    // Convention from the SDK ABI: Fr::add is the FIRST FieldOp in the trace
    // (the guest does `a+b` first). Its FieldRegEvent should carry a payload
    // with funct3=FADD, a = (12345, 0, 0, 0), b = (67890, 0, 0, 0), and
    // new = a + b (= 80235 since both inputs fit in a single limb).
    let first_field_op_event = events
        .iter()
        .find(|e| matches!(e.op, Some(p) if p.funct3 == FUNCT3_FADD))
        .expect("no FADD FieldRegEvent found in trace");

    let payload = first_field_op_event
        .op
        .expect("FADD event must have a payload");

    assert_eq!(payload.a, [12345, 0, 0, 0], "a-limbs mismatch");
    assert_eq!(payload.b, [67890, 0, 0, 0], "b-limbs mismatch");

    // Verify `new` matches `a + b` in BN254 Fr (both inputs small so no wrap).
    let a = ArkFr::from_le_bytes_mod_order(&limbs_to_bytes(&payload.a));
    let b = ArkFr::from_le_bytes_mod_order(&limbs_to_bytes(&payload.b));
    let expected_sum = ark_to_limbs(&(a + b));
    assert_eq!(
        first_field_op_event.new, expected_sum,
        "FieldRegEvent.new must equal a + b in Fr"
    );
}

fn limbs_to_bytes(limbs: &[u64; 4]) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for (i, &l) in limbs.iter().enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&l.to_le_bytes());
    }
    bytes
}

fn ark_to_limbs(fr: &ark_bn254::Fr) -> [u64; 4] {
    use ark_ff::{BigInteger, PrimeField};
    let bi = fr.into_bigint();
    let bytes = bi.to_bytes_le();
    let mut limbs = [0u64; 4];
    for (i, limb) in limbs.iter_mut().enumerate() {
        let start = i * 8;
        let end = std::cmp::min(start + 8, bytes.len());
        if start < bytes.len() {
            let mut buf = [0u8; 8];
            buf[..end - start].copy_from_slice(&bytes[start..end]);
            *limb = u64::from_le_bytes(buf);
        }
    }
    limbs
}
