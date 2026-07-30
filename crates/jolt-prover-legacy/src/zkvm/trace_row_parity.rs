//! Real-trace parity for `jolt_riscv::JoltTraceRow`.
//!
//! Builds the proof-facing trace via the `tracer` conversion over a genuine
//! traced program and checks every logical accessor against the reference
//! derivation used by `R1CSCycleInputs::from_trace`. This exercises final
//! `LD`/`SD` rows, expanded narrow loads/stores, and no-op padding on real data.
//!
//! Lives in `jolt-prover-legacy` only because the reference (`R1CSCycleInputs`) and the
//! host program loader still do; the row type and its conversion are in
//! `jolt-riscv` / `tracer`.

use ark_ff::biginteger::{S128, S64};
use common::constants::XLEN;
use jolt_riscv::RV64IMAC_JOLT;
use strum::IntoEnumIterator;
use tracer::instruction::{Cycle, RAMAccess};

use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::instruction::{
    CircuitFlags, Flags, InstructionFlags, InstructionLookup, JoltTraceCycle, LookupQuery,
    NUM_CIRCUIT_FLAGS,
};
use crate::zkvm::r1cs::inputs::{R1CSCycleInputs, ALL_R1CS_INPUTS};

fn legacy_r1cs_inputs(
    bytecode_preprocessing: &BytecodePreprocessing,
    trace: &[Cycle],
    t: usize,
) -> R1CSCycleInputs {
    let cycle = JoltTraceCycle::try_new(&trace[t]).unwrap();
    let next = (t + 1 < trace.len()).then(|| JoltTraceCycle::try_new(&trace[t + 1]).unwrap());
    let flags_view = Flags::circuit_flags(&cycle);
    let instruction_flags = Flags::instruction_flags(&cycle);
    let instruction = cycle.instruction();

    let (left_input, right_i128) = LookupQuery::<XLEN>::to_instruction_inputs(&cycle);
    let left_s64 = S64::from_u64(left_input);
    let right_input = S64::from_u64_with_sign(right_i128.unsigned_abs() as u64, right_i128 >= 0);
    let product = left_s64.mul_trunc::<2, 2>(&S128::from_i128(right_i128));
    let (left_lookup, right_lookup) = LookupQuery::<XLEN>::to_lookup_operands(&cycle);
    let lookup_output = LookupQuery::<XLEN>::to_lookup_output(&cycle);

    let rs1_read_value = trace[t].rs1_read().unwrap_or_default().1;
    let rs2_read_value = trace[t].rs2_read().unwrap_or_default().1;
    let rd_write_value = trace[t].rd_write().unwrap_or_default().2;
    let ram_addr = trace[t].ram_access().address() as u64;
    let (ram_read_value, ram_write_value) = match trace[t].ram_access() {
        RAMAccess::Read(read) => (read.value, read.value),
        RAMAccess::Write(write) => (write.pre_value, write.post_value),
        RAMAccess::NoOp => (0, 0),
    };

    let pc = crate::zkvm::bytecode::get_pc_for_cycle(bytecode_preprocessing, &trace[t]) as u64;
    let next_pc = next.as_ref().map_or(0, |_| {
        crate::zkvm::bytecode::get_pc_for_cycle(bytecode_preprocessing, &trace[t + 1]) as u64
    });
    let unexpanded_pc = instruction.address as u64;
    let next_unexpanded_pc = next
        .as_ref()
        .map_or(0, |next| next.instruction().address as u64);
    let imm_i128 = instruction.operands.imm;
    let imm = S64::from_u64_with_sign(imm_i128.unsigned_abs() as u64, imm_i128 >= 0);

    let mut flags = [false; NUM_CIRCUIT_FLAGS];
    for flag in CircuitFlags::iter() {
        flags[flag] = flags_view[flag];
    }
    let next_is_noop = next
        .as_ref()
        .is_some_and(|next| Flags::instruction_flags(next)[InstructionFlags::IsNoop]);
    let should_jump = flags_view[CircuitFlags::Jump] && !next_is_noop;
    let should_branch = instruction_flags[InstructionFlags::Branch] && lookup_output == 1;
    let (next_is_virtual, next_is_first_in_sequence) =
        next.as_ref().map_or((false, false), |next| {
            let next_flags = Flags::circuit_flags(next);
            (
                next_flags[CircuitFlags::VirtualInstruction],
                next_flags[CircuitFlags::IsFirstInSequence],
            )
        });

    R1CSCycleInputs {
        left_input,
        right_input,
        product,
        left_lookup,
        right_lookup,
        lookup_output,
        rs1_read_value,
        rs2_read_value,
        rd_write_value,
        ram_addr,
        ram_read_value,
        ram_write_value,
        pc,
        next_pc,
        unexpanded_pc,
        next_unexpanded_pc,
        imm,
        flags,
        next_is_noop,
        should_jump,
        should_branch,
        next_is_virtual,
        next_is_first_in_sequence,
    }
}

#[test]
fn accessors_match_reference_on_real_trace() {
    let mut program = crate::host::Program::new("fibonacci-guest");
    let inputs = postcard::to_stdvec(&10u32).unwrap();
    let (bytecode, _init, _size, entry) = program.decode();
    let (_lazy, mut trace, _memory, _io) = program.trace(&inputs, &[], &[]);
    let bytecode_preprocessing =
        BytecodePreprocessing::preprocess(bytecode, entry, RV64IMAC_JOLT).unwrap();

    let padded_len = (trace.len() + 1).next_power_of_two();
    let mut rows = tracer::build_trace_rows(&trace, &bytecode_preprocessing).unwrap();
    assert_eq!(
        jolt_riscv::JoltTraceRow::default(),
        tracer::cycle_to_trace_row(&Cycle::NoOp, &bytecode_preprocessing).unwrap()
    );
    trace.resize(padded_len, Cycle::NoOp);
    rows.resize(padded_len, jolt_riscv::JoltTraceRow::default());
    assert_eq!(rows.len(), trace.len());

    let mut saw_load = false;
    let mut saw_store = false;
    for (t, (row, cycle)) in rows.iter().zip(trace.iter()).enumerate() {
        let reference = legacy_r1cs_inputs(&bytecode_preprocessing, &trace, t);
        let candidate =
            R1CSCycleInputs::from_trace::<ark_bn254::Fr>(&bytecode_preprocessing, &rows, t);

        for input in ALL_R1CS_INPUTS {
            assert_eq!(
                candidate.get_input_value(input),
                reference.get_input_value(input),
                "{input:?} @ {t}"
            );
        }

        assert_eq!(row.rs1_value(), reference.rs1_read_value, "rs1 @ {t}");
        assert_eq!(row.rs2_value(), reference.rs2_read_value, "rs2 @ {t}");
        assert_eq!(row.rd_write_value(), reference.rd_write_value, "rd @ {t}");
        assert_eq!(row.ram_address(), reference.ram_addr, "ram_addr @ {t}");
        assert_eq!(
            row.ram_read_value(),
            reference.ram_read_value,
            "ram_read @ {t}"
        );
        assert_eq!(
            row.ram_write_value(),
            reference.ram_write_value,
            "ram_write @ {t}"
        );
        assert_eq!(row.pc(), reference.pc, "pc @ {t}");
        assert_eq!(
            row.unexpanded_pc(),
            reference.unexpanded_pc,
            "unexpanded_pc @ {t}"
        );
        assert_eq!(row.imm(), reference.imm.to_i128(), "imm @ {t}");

        // rd pre-value comes from the cycle; register indices from operands.
        let rd = cycle.rd_write();
        assert_eq!(
            row.rd_pre_value(),
            rd.map_or(0, |(_, pre, _)| pre),
            "rd_pre @ {t}"
        );
        let instruction = cycle.instruction().try_jolt_instruction_row().unwrap();
        assert_eq!(row.rs1_index(), instruction.operands.rs1, "rs1_idx @ {t}");
        assert_eq!(row.rs2_index(), instruction.operands.rs2, "rs2_idx @ {t}");
        assert_eq!(row.rd_index(), instruction.operands.rd, "rd_idx @ {t}");
        assert_eq!(
            row.rs1_index(),
            cycle.rs1_read().map(|(index, _)| index),
            "logical rs1 presence @ {t}"
        );
        assert_eq!(
            row.rs2_index(),
            cycle.rs2_read().map(|(index, _)| index),
            "logical rs2 presence @ {t}"
        );
        assert_eq!(
            row.rd_index(),
            cycle.rd_write().map(|(index, _, _)| index),
            "logical rd presence @ {t}"
        );

        let legacy_cycle = JoltTraceCycle::try_new(cycle).unwrap();
        assert_eq!(
            Flags::circuit_flags(row),
            Flags::circuit_flags(&legacy_cycle),
            "circuit flags @ {t}"
        );
        assert_eq!(
            Flags::instruction_flags(row),
            Flags::instruction_flags(&legacy_cycle),
            "instruction flags @ {t}"
        );
        assert_eq!(
            LookupQuery::<XLEN>::to_lookup_index(row),
            LookupQuery::<XLEN>::to_lookup_index(&legacy_cycle),
            "lookup index @ {t}"
        );
        let row_table = InstructionLookup::<XLEN>::lookup_table(row);
        let legacy_table = InstructionLookup::<XLEN>::lookup_table(&legacy_cycle);
        assert_eq!(
            row_table.as_ref().map(std::mem::discriminant),
            legacy_table.as_ref().map(std::mem::discriminant),
            "lookup table @ {t}"
        );

        saw_load |= row.is_load();
        saw_store |= row.is_store();
    }
    assert!(saw_load, "fibonacci trace should contain final loads");
    assert!(saw_store, "fibonacci trace should contain final stores");
}
