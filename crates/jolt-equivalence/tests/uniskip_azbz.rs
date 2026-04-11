//! Diagnostic: compare per-cycle R1CS witness variables between jolt-r1cs and
//! jolt-core to find divergences that cause extended evaluation mismatches.
#![allow(non_snake_case, clippy::print_stderr)]

use jolt_field::{Field, Fr};
use jolt_host::{extract_trace, BytecodePreprocessing, Program};
use jolt_r1cs::constraints::rv64::{self, *};
use jolt_r1cs::R1csKey;

// jolt-core types
use jolt_core::field::JoltField;
use jolt_core::zkvm::bytecode::BytecodePreprocessing as CoreBytecodePP;
use jolt_core::zkvm::r1cs::inputs::R1CSCycleInputs;

type CoreFr = ark_bn254::Fr;

#[test]
fn compare_witness_variables() {
    // ── Our system: extract trace + R1CS witness ─────────────────────────
    let mut program = Program::new("muldiv-guest");
    let (bytecode_raw, _init_mem, _program_size, entry_address) = program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, trace, _, io_device) = program.trace(&inputs, &[], &[]);

    let bytecode = BytecodePreprocessing::preprocess(bytecode_raw, entry_address);
    let trace_length = trace.len().next_power_of_two();
    let matrices = rv64::rv64_constraints::<Fr>();
    let r1cs_key = R1csKey::new(matrices.clone(), trace_length);
    let v_pad = r1cs_key.num_vars_padded;

    let (_, r1cs_witness, _) = extract_trace::<_, Fr>(
        &trace,
        trace_length,
        &bytecode,
        &io_device.memory_layout,
        v_pad,
    );

    // ── jolt-core: trace + BytecodePreprocessing ─────────────────────────
    let mut core_program = jolt_core::host::Program::new("muldiv-guest");
    let (core_bytecode_raw, _core_init_mem, _core_prog_size, core_entry) = core_program.decode();
    let core_inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, core_trace, _, _core_io) = core_program.trace(&core_inputs, &[], &[]);
    let core_bytecode = CoreBytecodePP::preprocess(core_bytecode_raw, core_entry);

    let real_len = trace.len();
    eprintln!("trace len={real_len}, padded={trace_length}");

    // Pad core_trace to match the prover (jolt-core pads before calling from_trace)
    let mut padded_core_trace = core_trace;
    padded_core_trace.resize(trace_length, tracer::instruction::Cycle::NoOp);
    let core_trace = padded_core_trace;

    // Variable names for pretty printing
    let var_names: [&str; NUM_VARS_PER_CYCLE] = [
        /*  0 */ "CONST",
        /*  1 */ "LeftInstructionInput",
        /*  2 */ "RightInstructionInput",
        /*  3 */ "Product",
        /*  4 */ "ShouldBranch",
        /*  5 */ "PC",
        /*  6 */ "UnexpandedPC",
        /*  7 */ "Imm",
        /*  8 */ "RamAddress",
        /*  9 */ "Rs1Value",
        /* 10 */ "Rs2Value",
        /* 11 */ "RdWriteValue",
        /* 12 */ "RamReadValue",
        /* 13 */ "RamWriteValue",
        /* 14 */ "LeftLookupOperand",
        /* 15 */ "RightLookupOperand",
        /* 16 */ "NextUnexpandedPC",
        /* 17 */ "NextPC",
        /* 18 */ "NextIsVirtual",
        /* 19 */ "NextIsFirstInSequence",
        /* 20 */ "LookupOutput",
        /* 21 */ "ShouldJump",
        /* 22 */ "FLAG_Add",
        /* 23 */ "FLAG_Sub",
        /* 24 */ "FLAG_Mul",
        /* 25 */ "FLAG_Load",
        /* 26 */ "FLAG_Store",
        /* 27 */ "FLAG_Jump",
        /* 28 */ "FLAG_WriteLookupToRd",
        /* 29 */ "FLAG_Virtual",
        /* 30 */ "FLAG_Assert",
        /* 31 */ "FLAG_DoNotUpdatePC",
        /* 32 */ "FLAG_Advice",
        /* 33 */ "FLAG_IsCompressed",
        /* 34 */ "FLAG_IsFirstInSeq",
        /* 35 */ "FLAG_IsLastInSeq",
        /* 36 */ "Branch",
        /* 37 */ "NextIsNoop",
    ];

    // Compare witness variables for real cycles.
    // We compare variables 0..=35 (CONST + 35 inputs).
    // Variables 36 (Branch) and 37 (NextIsNoop) are product factors that
    // jolt-core doesn't expose directly — we skip them.
    let compare_range = 0..=35usize;

    let mut total_mismatches = 0usize;
    let mut mismatch_by_var: [usize; NUM_VARS_PER_CYCLE] = [0; NUM_VARS_PER_CYCLE];

    for c in 0..real_len.min(core_trace.len()) {
        let our_w = &r1cs_witness[c * v_pad..(c + 1) * v_pad];

        let core_row = R1CSCycleInputs::from_trace::<CoreFr>(&core_bytecode, &core_trace, c);

        // Build the 36 comparable witness values from jolt-core
        let core_vals = build_core_comparable(&core_row);

        for v in compare_range.clone() {
            if our_w[v] != core_vals[v] {
                if mismatch_by_var[v] < 5 {
                    let add =
                        core_row.flags[jolt_core::zkvm::instruction::CircuitFlags::AddOperands];
                    let sub = core_row.flags
                        [jolt_core::zkvm::instruction::CircuitFlags::SubtractOperands];
                    let mul = core_row.flags
                        [jolt_core::zkvm::instruction::CircuitFlags::MultiplyOperands];
                    let adv = core_row.flags[jolt_core::zkvm::instruction::CircuitFlags::Advice];
                    eprintln!(
                        "  MISMATCH cycle={c} var={v} ({}): ours={}, core={} [add={add} sub={sub} mul={mul} adv={adv}]",
                        var_names[v], our_w[v], core_vals[v]
                    );
                    if v == V_RIGHT_LOOKUP_OPERAND {
                        eprintln!(
                            "    core.right_lookup={}, core.left_input={}, core.right_input=({}, {})",
                            core_row.right_lookup,
                            core_row.left_input,
                            core_row.right_input.magnitude.0[0],
                            if core_row.right_input.is_positive { "+" } else { "-" }
                        );
                    }
                }
                mismatch_by_var[v] += 1;
                total_mismatches += 1;
            }
        }
    }

    if total_mismatches == 0 {
        eprintln!(
            "All witness variables [0..35] match for {} real cycles!",
            real_len.min(core_trace.len())
        );
    } else {
        eprintln!("\n=== Summary: {total_mismatches} total mismatches ===");
        for v in 0..NUM_VARS_PER_CYCLE {
            if mismatch_by_var[v] > 0 {
                eprintln!(
                    "  var={v} ({}): {} mismatches",
                    var_names[v], mismatch_by_var[v]
                );
            }
        }
    }

    assert_eq!(total_mismatches, 0, "witness variable mismatches detected");
}

/// Build comparable witness values [0..36] from jolt-core's R1CSCycleInputs.
///
/// Maps R1CSCycleInputs fields to the jolt-r1cs variable layout.
fn build_core_comparable(row: &R1CSCycleInputs) -> [Fr; 36] {
    use jolt_core::zkvm::instruction::CircuitFlags;

    let mut w = [Fr::from_u64(0); 36];

    w[V_CONST] = Fr::from_u64(1);
    w[V_LEFT_INSTRUCTION_INPUT] = Fr::from_u64(row.left_input);

    // right_input is ark_ff::biginteger::S64 — extract magnitude + sign
    w[V_RIGHT_INSTRUCTION_INPUT] = {
        let mag = row.right_input.magnitude.0[0];
        if mag == 0 {
            Fr::from_u64(0)
        } else if row.right_input.is_positive {
            Fr::from_u64(mag)
        } else {
            -Fr::from_u64(mag)
        }
    };

    // product is ark_ff::biginteger::S128 — magnitude can exceed i128::MAX
    w[V_PRODUCT] = {
        let lo = row.product.magnitude.0[0] as u128;
        let hi = (row.product.magnitude.0[1] as u128) << 64;
        let mag = lo | hi;
        if mag == 0 {
            Fr::from_u64(0)
        } else if row.product.is_positive {
            Fr::from_u128(mag)
        } else {
            -Fr::from_u128(mag)
        }
    };

    w[V_SHOULD_BRANCH] = Fr::from_u64(row.should_branch as u64);
    w[V_PC] = Fr::from_u64(row.pc);
    w[V_UNEXPANDED_PC] = Fr::from_u64(row.unexpanded_pc);

    // imm is S64
    w[V_IMM] = {
        let mag = row.imm.magnitude.0[0];
        if mag == 0 {
            Fr::from_u64(0)
        } else if row.imm.is_positive {
            Fr::from_u64(mag)
        } else {
            -Fr::from_u64(mag)
        }
    };

    w[V_RAM_ADDRESS] = Fr::from_u64(row.ram_addr);
    w[V_RS1_VALUE] = Fr::from_u64(row.rs1_read_value);
    w[V_RS2_VALUE] = Fr::from_u64(row.rs2_read_value);
    w[V_RD_WRITE_VALUE] = Fr::from_u64(row.rd_write_value);
    w[V_RAM_READ_VALUE] = Fr::from_u64(row.ram_read_value);
    w[V_RAM_WRITE_VALUE] = Fr::from_u64(row.ram_write_value);
    w[V_LEFT_LOOKUP_OPERAND] = Fr::from_u64(row.left_lookup);
    // right_lookup is u128 — may exceed u64::MAX for add-like instructions
    w[V_RIGHT_LOOKUP_OPERAND] = Fr::from_u128(row.right_lookup);
    w[V_NEXT_UNEXPANDED_PC] = Fr::from_u64(row.next_unexpanded_pc);
    w[V_NEXT_PC] = Fr::from_u64(row.next_pc);
    w[V_NEXT_IS_VIRTUAL] = Fr::from_u64(row.next_is_virtual as u64);
    w[V_NEXT_IS_FIRST_IN_SEQUENCE] = Fr::from_u64(row.next_is_first_in_sequence as u64);
    w[V_LOOKUP_OUTPUT] = Fr::from_u64(row.lookup_output);
    w[V_SHOULD_JUMP] = Fr::from_u64(row.should_jump as u64);

    w[V_FLAG_ADD_OPERANDS] = Fr::from_u64(row.flags[CircuitFlags::AddOperands] as u64);
    w[V_FLAG_SUBTRACT_OPERANDS] = Fr::from_u64(row.flags[CircuitFlags::SubtractOperands] as u64);
    w[V_FLAG_MULTIPLY_OPERANDS] = Fr::from_u64(row.flags[CircuitFlags::MultiplyOperands] as u64);
    w[V_FLAG_LOAD] = Fr::from_u64(row.flags[CircuitFlags::Load] as u64);
    w[V_FLAG_STORE] = Fr::from_u64(row.flags[CircuitFlags::Store] as u64);
    w[V_FLAG_JUMP] = Fr::from_u64(row.flags[CircuitFlags::Jump] as u64);
    w[V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD] =
        Fr::from_u64(row.flags[CircuitFlags::WriteLookupOutputToRD] as u64);
    w[V_FLAG_VIRTUAL_INSTRUCTION] =
        Fr::from_u64(row.flags[CircuitFlags::VirtualInstruction] as u64);
    w[V_FLAG_ASSERT] = Fr::from_u64(row.flags[CircuitFlags::Assert] as u64);
    w[V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] =
        Fr::from_u64(row.flags[CircuitFlags::DoNotUpdateUnexpandedPC] as u64);
    w[V_FLAG_ADVICE] = Fr::from_u64(row.flags[CircuitFlags::Advice] as u64);
    w[V_FLAG_IS_COMPRESSED] = Fr::from_u64(row.flags[CircuitFlags::IsCompressed] as u64);
    w[V_FLAG_IS_FIRST_IN_SEQUENCE] =
        Fr::from_u64(row.flags[CircuitFlags::IsFirstInSequence] as u64);
    w[V_FLAG_IS_LAST_IN_SEQUENCE] = Fr::from_u64(row.flags[CircuitFlags::IsLastInSequence] as u64);

    w
}
