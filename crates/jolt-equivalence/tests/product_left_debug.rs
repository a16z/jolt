//! Targeted diagnostic: compare ProductRemainder left polynomial values
//! between jolt-core's fused_left_right_at_r and jolt-zkvm's
//! DerivedSource::product_left() + lagrange_evals projection.
//!
//! Uses a fixed r0 to isolate the projection math from transcript state.
#![allow(non_snake_case, clippy::print_stderr)]

use jolt_field::{Field, Fr};
use jolt_host::{extract_trace, BytecodePreprocessing, Program};
use jolt_r1cs::constraints::rv64::{self, *};
use jolt_r1cs::R1csKey;
use num_traits::Zero;

use ark_ff::One as _;
use jolt_core::field::JoltField;
use jolt_core::poly::lagrange_poly::LagrangePolynomial;
use jolt_core::zkvm::r1cs::evaluation::ProductVirtualEval;
use jolt_core::zkvm::r1cs::inputs::ProductCycleInputs;

use jolt_zkvm::derived::DerivedSource;

type CoreFr = ark_bn254::Fr;

fn core_to_bytes(f: CoreFr) -> [u8; 32] {
    use ark_serialize::CanonicalSerialize;
    let mut buf = [0u8; 32];
    f.serialize_compressed(&mut buf[..]).expect("serialize");
    buf
}

fn zkvm_to_bytes(f: Fr) -> [u8; 32] {
    f.to_bytes()
}

fn fields_eq(core: CoreFr, zkvm: Fr) -> bool {
    core_to_bytes(core) == zkvm_to_bytes(zkvm)
}

#[test]
fn product_left_projection_equivalence() {
    // ── jolt-zkvm: extract trace + R1CS witness ─────────────────────
    let mut program = Program::new("muldiv-guest");
    let (bytecode_raw, _init_mem, _program_size, entry_address) = program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, trace, _, io_device) = program.trace(&inputs, &[], &[]);

    let bytecode = BytecodePreprocessing::preprocess(bytecode_raw, entry_address);
    let trace_length = trace.len().next_power_of_two();
    let matrices = rv64::rv64_constraints::<Fr>();
    let r1cs_key = R1csKey::new(matrices, trace_length);
    let v_pad = r1cs_key.num_vars_padded;

    let (_, r1cs_witness, _) = extract_trace::<_, Fr>(
        &trace,
        trace_length,
        &bytecode,
        &io_device.memory_layout,
        v_pad,
    );

    // ── jolt-core: trace (same guest, same inputs) ──────────────────
    let mut core_program = jolt_core::host::Program::new("muldiv-guest");
    let _ = core_program.decode();
    let core_inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, core_trace, _, _) = core_program.trace(&core_inputs, &[], &[]);

    let mut padded_core_trace = core_trace;
    padded_core_trace.resize(trace_length, tracer::instruction::Cycle::NoOp);

    eprintln!("trace_length = {trace_length}");

    // ── Step 1: Compare raw A-row witness variables ─────────────────
    let a_vars = [V_LEFT_INSTRUCTION_INPUT, V_LOOKUP_OUTPUT, V_FLAG_JUMP];
    let a_names = ["LeftInstructionInput", "LookupOutput", "JumpFlag"];

    let mut raw_mismatches = 0;
    for c in 0..trace_length {
        let row = ProductCycleInputs::from_trace::<CoreFr>(&padded_core_trace, c);
        let w = c * v_pad;

        let core_vals: [CoreFr; 3] = [
            CoreFr::from_u64(row.instruction_left_input),
            CoreFr::from_u64(row.should_branch_lookup_output),
            if row.jump_flag {
                CoreFr::one()
            } else {
                CoreFr::zero()
            },
        ];

        for (k, (&var, &name)) in a_vars.iter().zip(a_names.iter()).enumerate() {
            let zkvm_val = r1cs_witness[w + var];
            if !fields_eq(core_vals[k], zkvm_val) {
                if raw_mismatches < 5 {
                    eprintln!(
                        "RAW MISMATCH cycle={c} var={name}: core={:?} zkvm={:?}",
                        core_to_bytes(core_vals[k]),
                        zkvm_to_bytes(zkvm_val),
                    );
                }
                raw_mismatches += 1;
            }
        }
    }
    eprintln!(
        "Raw A-row variable mismatches: {raw_mismatches} / {}",
        trace_length * 3
    );

    // ── Step 2: Build domain-indexed product_left from DerivedSource ──
    let derived = DerivedSource::<Fr>::new(&r1cs_witness, trace_length, v_pad);
    let domain_left = derived.compute(jolt_compiler::PolynomialId::ProductLeft);
    let domain_left = domain_left.as_ref();
    let domain_right = derived.compute(jolt_compiler::PolynomialId::ProductRight);
    let domain_right = domain_right.as_ref();

    let stride = 4usize;
    eprintln!(
        "domain_left len = {}, domain_right len = {}",
        domain_left.len(),
        domain_right.len()
    );

    // ── Step 3: Lagrange basis at a fixed r0 ────────────────────────
    let r0 = Fr::from_u64(42);
    let r0_core = CoreFr::from_u64(42);

    let basis_zkvm = jolt_poly::lagrange::lagrange_evals(-1i64, 3, r0);
    let basis_core: [CoreFr; 3] = LagrangePolynomial::<CoreFr>::evals::<CoreFr, 3>(&r0_core);

    eprintln!("\n=== Lagrange basis comparison at r0=42 ===");
    for k in 0..3 {
        let eq = fields_eq(basis_core[k], basis_zkvm[k]);
        eprintln!("  k={k}: match={eq}");
        if !eq {
            eprintln!("    core = {:?}", core_to_bytes(basis_core[k]));
            eprintln!("    zkvm = {:?}", zkvm_to_bytes(basis_zkvm[k]));
        }
    }

    // ── Step 4: Compare projected left values ───────────────────────
    eprintln!("\n=== Projected LEFT comparison (first 16 cycles) ===");
    let mut left_mismatches = 0;
    for c in 0..trace_length {
        let row = ProductCycleInputs::from_trace::<CoreFr>(&padded_core_trace, c);
        let (core_left, _) = ProductVirtualEval::fused_left_right_at_r::<CoreFr>(&row, &basis_core);

        let mut zkvm_left = Fr::zero();
        for (k, &lk) in basis_zkvm.iter().enumerate() {
            zkvm_left += lk * domain_left[c * stride + k];
        }

        if !fields_eq(core_left, zkvm_left) {
            if left_mismatches < 16 {
                eprintln!(
                    "  c={c}: MISMATCH core={:?} zkvm={:?}",
                    core_to_bytes(core_left),
                    zkvm_to_bytes(zkvm_left),
                );
            }
            left_mismatches += 1;
        }
    }
    eprintln!("Left projection mismatches: {left_mismatches} / {trace_length}");

    // ── Step 5: Compare projected right values ──────────────────────
    eprintln!("\n=== Projected RIGHT comparison (first 16 cycles) ===");
    let mut right_mismatches = 0;
    for c in 0..trace_length {
        let row = ProductCycleInputs::from_trace::<CoreFr>(&padded_core_trace, c);
        let (_, core_right) =
            ProductVirtualEval::fused_left_right_at_r::<CoreFr>(&row, &basis_core);

        let mut zkvm_right = Fr::zero();
        for (k, &lk) in basis_zkvm.iter().enumerate() {
            zkvm_right += lk * domain_right[c * stride + k];
        }

        if !fields_eq(core_right, zkvm_right) {
            if right_mismatches < 16 {
                eprintln!(
                    "  c={c}: MISMATCH core={:?} zkvm={:?}",
                    core_to_bytes(core_right),
                    zkvm_to_bytes(zkvm_right),
                );
            }
            right_mismatches += 1;
        }
    }
    eprintln!("Right projection mismatches: {right_mismatches} / {trace_length}");

    assert_eq!(
        raw_mismatches, 0,
        "Raw A-row witness variables differ between systems"
    );
    assert_eq!(
        left_mismatches, 0,
        "Projected left values differ — Lagrange projection bug"
    );
    assert_eq!(
        right_mismatches, 0,
        "Projected right values differ — Lagrange projection bug"
    );
}
