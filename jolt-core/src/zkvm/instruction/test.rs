use crate::{field::JoltField, zkvm::instruction::LookupQuery};
use common::constants::XLEN;
use rand::prelude::*;
use tracer::{
    emulator::{cpu::Cpu, terminal::DummyTerminal},
    instruction::{
        self, format::InstructionRegisterState, Cycle, JoltInstructionRow, RISCVCycle,
        RISCVInstruction, RISCVTrace,
    },
};

use super::{CircuitFlags, Flags, InstructionLookup};

pub fn materialize_entry_test<F, T>()
where
    RISCVCycle<T>: LookupQuery<XLEN>,
    T: InstructionLookup<XLEN> + RISCVInstruction + Default,
    F: JoltField,
{
    let cycle: RISCVCycle<T> = Default::default();
    let table = cycle.instruction.lookup_table().unwrap();
    let mut rng = StdRng::seed_from_u64(12345);
    for _ in 0..10000 {
        let random_cycle = cycle.random(&mut rng);
        assert_eq!(
            random_cycle.to_lookup_output(),
            table.materialize_entry(random_cycle.to_lookup_index()),
            "{:?}",
            random_cycle.register_state
        );
    }
}

/// Test that certain combinations of circuit flags are exclusive.
mod flags {
    use std::panic;

    use crate::zkvm::instruction::{Flags, InstructionFlags, SupportedInstruction};

    use super::CircuitFlags;
    use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands};
    use strum::IntoEnumIterator;
    use tracer::instruction::{Cycle, Instruction};

    #[test]
    fn left_operand_exclusive() {
        for cycle in Cycle::iter() {
            if let Cycle::INLINE(_) = cycle {
                continue;
            }
            let instr = cycle.instruction();
            if let Ok(flags) = panic::catch_unwind(|| instr.instruction_flags()) {
                assert!(
                    !(flags[InstructionFlags::LeftOperandIsPC]
                        && flags[InstructionFlags::LeftOperandIsRs1Value]),
                    "Left operand flags not exclusive for {instr:?}",
                );
            }
        }
    }

    #[test]
    fn right_operand_exclusive() {
        for cycle in Cycle::iter() {
            if let Cycle::INLINE(_) = cycle {
                continue;
            }
            let instr = cycle.instruction();
            if let Ok(flags) = panic::catch_unwind(|| instr.instruction_flags()) {
                assert!(
                    !(flags[InstructionFlags::RightOperandIsRs2Value]
                        && flags[InstructionFlags::RightOperandIsImm]),
                    "Right operand flags not exclusive for {instr:?}",
                );
            }
        }
    }

    #[test]
    fn lookup_shape_exclusive() {
        for cycle in Cycle::iter() {
            if let Cycle::INLINE(_) = cycle {
                continue;
            }
            let instr = cycle.instruction();
            if let Ok(flags) = panic::catch_unwind(|| instr.circuit_flags()) {
                let num_true = [
                    flags[CircuitFlags::AddOperands],
                    flags[CircuitFlags::SubtractOperands],
                    flags[CircuitFlags::MultiplyOperands],
                    flags[CircuitFlags::Advice],
                ]
                .iter()
                .filter(|&&b| b)
                .count();
                assert!(
                    num_true <= 1,
                    "Lookup shaping flags not exclusive for {instr:?}",
                );
            }
        }
    }

    #[test]
    fn load_store_exclusive() {
        for cycle in Cycle::iter() {
            if let Cycle::INLINE(_) = cycle {
                continue;
            }
            let instr = cycle.instruction();
            if let Ok(flags) = panic::catch_unwind(|| instr.circuit_flags()) {
                assert!(
                    !(flags[CircuitFlags::Load] && flags[CircuitFlags::Store]),
                    "Load/Store flags not exclusive for {instr:?}",
                );
            }
        }
    }

    #[test]
    fn normalized_flags_match_concrete_instruction_flags() {
        for cycle in Cycle::iter() {
            if let Cycle::INLINE(_) = cycle {
                continue;
            }
            let instr = cycle.instruction();
            if !instr.is_supported_instruction() {
                continue;
            }

            let normalized = instr.jolt_instruction_row();
            assert_eq!(
                normalized.circuit_flags(),
                instr.circuit_flags(),
                "circuit flags differ for {instr:?}"
            );
            assert_eq!(
                normalized.instruction_flags(),
                instr.instruction_flags(),
                "instruction flags differ for {instr:?}"
            );
        }
    }

    #[test]
    fn concrete_terminal_virtual_flags_match_normalized_flags() {
        let normalized = JoltInstructionRow {
            instruction_kind: JoltInstructionKind::ADDI,
            address: 0x8000_0000,
            operands: NormalizedOperands {
                rd: Some(1),
                rs1: Some(2),
                rs2: None,
                imm: 3,
            },
            virtual_sequence_remaining: Some(0),
            is_first_in_sequence: false,
            is_compressed: false,
        };
        let concrete = Instruction::try_from_jolt_instruction_row(normalized)
            .expect("ADDI should convert from normalized form");

        assert_eq!(normalized.circuit_flags(), concrete.circuit_flags());
        assert!(concrete.circuit_flags()[CircuitFlags::IsLastInSequence]);
    }

    #[cfg(feature = "host")]
    #[test]
    fn normalized_flags_match_expanded_program_bytecode() {
        let mut program = crate::host::Program::new("fibonacci-guest");
        let (bytecode, _, _, _) = program.decode();

        for normalized in bytecode {
            let concrete = Instruction::try_from_jolt_instruction_row(normalized)
                .expect("expanded bytecode should convert to a concrete instruction");
            assert_eq!(
                normalized.circuit_flags(),
                concrete.circuit_flags(),
                "circuit flags differ for {normalized:?}"
            );
            assert_eq!(
                normalized.instruction_flags(),
                concrete.instruction_flags(),
                "instruction flags differ for {normalized:?}"
            );
        }
    }

    #[test]
    fn branch_lookup_output_is_boolean() {
        use crate::zkvm::instruction::LookupQuery;
        use common::constants::XLEN;

        for cycle in Cycle::iter() {
            if let Cycle::INLINE(_) = cycle {
                continue;
            }
            let instr = cycle.instruction();
            if let Ok(instr_flags) = panic::catch_unwind(|| instr.instruction_flags()) {
                if instr_flags[InstructionFlags::Branch] {
                    let out = LookupQuery::<XLEN>::to_lookup_output(&cycle);
                    assert!(
                        out == 0 || out == 1,
                        "Branch lookup output not boolean for {instr:?}: got {out}",
                    );
                }
            }
        }
    }
}

/// Fuzz-check that every instruction's `LookupQuery::to_instruction_inputs`
/// agrees with the instruction input R1CS constraints:
///
///   left_input  = LeftOperandIsRs1Value  · Rs1Value     + LeftOperandIsPC   · UnexpandedPC
///   right_input = RightOperandIsRs2Value · Rs2Value     + RightOperandIsImm · Imm
///
/// Source of truth for the constraint:
/// `jolt-core/src/zkvm/spartan/instruction_input.rs::output_claim_constraint`.
///
/// A mismatch here means the trace witness polynomials `LeftInstructionInput` /
/// `RightInstructionInput` disagree with what the constraint reconstructs from
/// `Rs1Value` / `Rs2Value` / `Imm` / `UnexpandedPC` — causing a Stage 3 sumcheck
/// verification failure whenever any high-order bits of a register value are set.
mod r1cs_consistency {
    use common::constants::XLEN;
    use rand::{rngs::StdRng, SeedableRng};
    use strum::IntoEnumIterator;
    use tracer::instruction::Cycle;

    use crate::zkvm::instruction::{Flags, InstructionFlags, LookupQuery, SupportedInstruction};

    #[test]
    fn instruction_inputs_match_constraint() {
        let mut rng = StdRng::seed_from_u64(12345);
        let mut failures: Vec<String> = Vec::new();

        for default_cycle in Cycle::iter() {
            // Skip enum variants without LookupQuery/Flags impls in jolt-core.
            // These are either the structural variants (NoOp, INLINE) or
            // architectural instructions that are always lowered to a virtual
            // sequence before appearing in a trace (DIV, LW, AMOSWAP.W, ...).
            let default_instr = default_cycle.instruction();
            if !default_instr.is_supported_instruction() {
                continue;
            }
            let variant: &'static str = (&default_instr).into();
            let mut first_failure_for_variant: Option<String> = None;

            for _ in 0..10_000 {
                let cycle = default_cycle.random(&mut rng);
                let instr = cycle.instruction();
                let flags = instr.instruction_flags();
                let norm = instr.jolt_instruction_row();

                let rs1 = cycle.rs1_read().map(|(_, v)| v).unwrap_or(0);
                let rs2 = cycle.rs2_read().map(|(_, v)| v).unwrap_or(0);
                let unexpanded_pc = norm.address as u64;
                let imm = norm.operands.imm;

                let left_expected: u64 = if flags[InstructionFlags::LeftOperandIsRs1Value] {
                    rs1
                } else if flags[InstructionFlags::LeftOperandIsPC] {
                    unexpanded_pc
                } else {
                    0
                };
                let right_expected: i128 = if flags[InstructionFlags::RightOperandIsRs2Value] {
                    rs2 as i128
                } else if flags[InstructionFlags::RightOperandIsImm] {
                    imm
                } else {
                    0
                };

                let (left_actual, right_actual) =
                    LookupQuery::<XLEN>::to_instruction_inputs(&cycle);

                if left_actual != left_expected || right_actual != right_expected {
                    first_failure_for_variant.get_or_insert_with(|| {
                        format!(
                            "{variant}: left actual={left_actual:#x} expected={left_expected:#x}; \
                             right actual={right_actual} expected={right_expected}; \
                             flags={flags:?}, rs1={rs1:#x}, rs2={rs2:#x}, \
                             unexpanded_pc={unexpanded_pc:#x}, imm={imm}"
                        )
                    });
                    break;
                }
            }
            if let Some(msg) = first_failure_for_variant {
                failures.push(msg);
            }
        }

        if !failures.is_empty() {
            panic!(
                "{} instruction(s) violate the instruction input constraints:\n\n{}",
                failures.len(),
                failures.join("\n\n")
            );
        }
    }
}

pub fn lookup_output_matches_trace_test<T>()
where
    T: InstructionLookup<XLEN> + RISCVInstruction + RISCVTrace + Default + Flags + 'static,
    RISCVCycle<T>: LookupQuery<XLEN> + Into<Cycle>,
{
    let cycle: RISCVCycle<T> = Default::default();
    let mut rng = StdRng::seed_from_u64(12345);
    for _ in 0..10000 {
        let random_cycle = cycle.random(&mut rng);
        let normalized_instr: JoltInstructionRow = random_cycle.instruction.into();
        let normalized_operands = normalized_instr.operands;

        let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
        if let Some(rs1_val) = random_cycle.register_state.rs1_value() {
            cpu.write_register(normalized_operands.rs1.unwrap() as usize, rs1_val as i64);
        }
        if let Some(rs2_val) = random_cycle.register_state.rs2_value() {
            cpu.write_register(normalized_operands.rs2.unwrap() as usize, rs2_val as i64);
        }

        random_cycle.instruction.trace(&mut cpu, None);
        let lookup_result = LookupQuery::<XLEN>::to_lookup_output(&random_cycle);

        use std::any::TypeId;
        let is_jal = TypeId::of::<T>() == TypeId::of::<instruction::jal::JAL>();
        let is_jalr = TypeId::of::<T>() == TypeId::of::<instruction::jalr::JALR>();
        if is_jal || is_jalr {
            let cpu_pc = cpu.read_pc();
            assert_eq!(cpu_pc, lookup_result, "{random_cycle:?}");
        } else if let Some(rd) = normalized_operands.rd {
            // x0 is hardwired to zero; writes are discarded so the
            // CPU result will always be 0 regardless of the lookup output.
            if rd != 0 {
                let cpu_result = cpu.x[rd as usize] as u64;
                assert_eq!(cpu_result, lookup_result, "{random_cycle:?}");
            }
        }
    }
}
