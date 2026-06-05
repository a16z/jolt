//! Real-trace parity for `jolt_riscv::JoltTraceRow`.
//!
//! Builds the proof-facing trace via the `tracer` conversion over a genuine
//! traced program and checks every logical accessor against the reference
//! derivation used by `R1CSCycleInputs::from_trace`. This exercises final
//! `LD`/`SD` rows, expanded narrow loads/stores, and no-op padding on real data.
//!
//! Lives in `jolt-core` only because the reference (`R1CSCycleInputs`) and the
//! host program loader still do; the row type and its conversion are in
//! `jolt-riscv` / `tracer`.

use ark_bn254::Fr;
use jolt_riscv::RV64IMAC_JOLT;

use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::r1cs::inputs::R1CSCycleInputs;

#[test]
fn accessors_match_reference_on_real_trace() {
    let mut program = crate::host::Program::new("fibonacci-guest");
    let inputs = postcard::to_stdvec(&10u32).unwrap();
    let (bytecode, _init, _size, entry) = program.decode();
    let (_lazy, trace, _memory, _io) = program.trace(&inputs, &[], &[]);
    let bytecode_preprocessing =
        BytecodePreprocessing::preprocess(bytecode, entry, RV64IMAC_JOLT).unwrap();

    let rows = tracer::build_trace_rows(&trace, &bytecode_preprocessing).unwrap();
    assert_eq!(rows.len(), trace.len());

    let mut saw_load = false;
    let mut saw_store = false;
    for (t, (row, cycle)) in rows.iter().zip(trace.iter()).enumerate() {
        let reference = R1CSCycleInputs::from_trace::<Fr>(&bytecode_preprocessing, &trace, t);

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

        // rd pre-value and register indices come straight from the cycle.
        let rd = cycle.rd_write();
        assert_eq!(
            row.rd_pre_value(),
            rd.map_or(0, |(_, pre, _)| pre),
            "rd_pre @ {t}"
        );
        assert_eq!(
            row.rs1_index(),
            cycle.rs1_read().map(|(i, _)| i),
            "rs1_idx @ {t}"
        );
        assert_eq!(
            row.rs2_index(),
            cycle.rs2_read().map(|(i, _)| i),
            "rs2_idx @ {t}"
        );
        assert_eq!(row.rd_index(), rd.map(|(i, _, _)| i), "rd_idx @ {t}");

        saw_load |= row.is_load();
        saw_store |= row.is_store();
    }
    assert!(saw_load, "fibonacci trace should contain final loads");
    assert!(saw_store, "fibonacci trace should contain final stores");
}
