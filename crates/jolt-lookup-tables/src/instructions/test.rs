//! Per-instruction test helpers.

use std::any::TypeId;

use jolt_riscv::{Flags, InstructionFlags, JoltCycle, JoltInstructionRowData};
use rand::prelude::*;
use tracer::emulator::{cpu::Cpu, terminal::DummyTerminal};
use tracer::instruction::format::{InstructionFormat, InstructionRegisterState};
use tracer::instruction::{jal::JAL, jalr::JALR, Cycle, RISCVCycle, RISCVTrace};

use crate::{InstructionLookupTable, LookupQuery, XLEN};

pub trait RandomLookupCycle: JoltCycle {
    fn random(rng: &mut StdRng) -> Self;
}

impl<T> RandomLookupCycle for RISCVCycle<T>
where
    T: tracer::instruction::RISCVInstruction + JoltInstructionRowData,
{
    fn random(rng: &mut StdRng) -> Self {
        let instruction = T::random(rng);
        let concrete: tracer::instruction::Instruction = instruction.into();
        let source_instruction = concrete.source_instruction();
        let register_state =
            <<T::Format as InstructionFormat>::RegisterState as InstructionRegisterState>::random(
                rng,
                &source_instruction.row().operands,
            );
        Self {
            instruction,
            register_state,
            ram_access: T::RAMAccess::default(),
        }
    }
}

/// Internal helper for [`materialize_entry_test!`]. The macro picks up the
/// verbose `Foo<RISCVCycle<TracerType>>` / `RISCVCycle<TracerType>` type pair
/// from a Jolt struct ident and a tracer instruction path, and passes the
/// tuple-struct constructor as `construct`.
#[doc(hidden)]
#[expect(clippy::unwrap_used)]
pub fn materialize_entry_test_fn<T, C, I>(
    cycle_wrapper: impl Fn(C) -> T,
    instr_wrapper: impl Fn(C::Instruction) -> I,
) where
    T: LookupQuery<XLEN> + core::fmt::Debug,
    C: RandomLookupCycle,
    I: InstructionLookupTable<XLEN>,
{
    let mut rng = StdRng::seed_from_u64(12345);
    for _ in 0..10_000 {
        let raw: C = RandomLookupCycle::random(&mut rng);
        let table = instr_wrapper(raw.instruction()).lookup_table().unwrap();
        let cycle: T = cycle_wrapper(raw);
        assert_eq!(
            cycle.to_lookup_output(),
            table.materialize_entry(cycle.to_lookup_index()),
            "{cycle:?}",
        );
    }
}

/// Internal helper for [`instruction_inputs_match_constraint_test!`].
///
/// Fuzz-checks that an instruction's `LookupQuery::to_instruction_inputs`
/// agrees with the instruction-input R1CS constraint:
///
/// ```text
/// left_input  = LeftOperandIsRs1Value  · Rs1Value     + LeftOperandIsPC   · UnexpandedPC
/// right_input = RightOperandIsRs2Value · Rs2Value     + RightOperandIsImm · Imm
/// ```
///
/// A mismatch means the trace witness polynomials `LeftInstructionInput` /
/// `RightInstructionInput` disagree with what the constraint reconstructs from
/// `Rs1Value` / `Rs2Value` / `Imm` / `UnexpandedPC` — causing a Stage 3
/// sumcheck verification failure whenever any high-order bits of a register
/// value or PC are set.
#[doc(hidden)]
pub fn instruction_inputs_match_constraint_fn<C, T, I>(
    cycle_wrapper: impl Fn(C) -> T,
    instr_wrapper: impl Fn(C::Instruction) -> I,
) where
    C: RandomLookupCycle,
    T: LookupQuery<XLEN> + core::fmt::Debug,
    I: JoltInstructionRowData + Flags,
{
    let mut rng = StdRng::seed_from_u64(12345);
    for _ in 0..10_000 {
        let raw: C = RandomLookupCycle::random(&mut rng);
        let instr = raw.instruction();
        let normalized = instr.jolt_instruction_row();
        let unexpanded_pc = normalized.address as u64;
        let imm = normalized.operands.imm;
        let flags = instr_wrapper(instr).instruction_flags();
        let rs1 = raw.rs1_val().unwrap_or(0);
        let rs2 = raw.rs2_val().unwrap_or(0);

        let cycle: T = cycle_wrapper(raw);

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

        let (left_actual, right_actual) = LookupQuery::<XLEN>::to_instruction_inputs(&cycle);
        assert_eq!(
            (left_actual, right_actual),
            (left_expected, right_expected),
            "{cycle:?}: flags={flags:?}, rs1={rs1:#x}, rs2={rs2:#x}, \
             unexpanded_pc={unexpanded_pc:#x}, imm={imm}",
        );
    }
}

/// Internal helper for [`lookup_output_matches_trace_test!`].
///
/// Fuzz-checks that an instruction's `LookupQuery::to_lookup_output` agrees
/// with the value tracer's CPU emulator writes (to `rd`, or to PC for
/// `JAL`/`JALR`) after executing the instruction. Catches divergences
/// between the lookup-table semantics and the RISC-V semantics implemented
/// by `tracer`.
///
/// `C: Copy` lets us print the failing cycle in the assert message after it
/// has been moved into the wrapper.
#[doc(hidden)]
#[expect(clippy::unwrap_used)]
#[expect(
    clippy::panic,
    reason = "deliberate guard against silent passes; see body"
)]
pub fn lookup_output_matches_trace_test_fn<C, T>(cycle_wrapper: impl Fn(C) -> T)
where
    C: RandomLookupCycle + Copy + core::fmt::Debug,
    C::Instruction: RISCVTrace + 'static,
    RISCVCycle<C::Instruction>: Into<Cycle>,
    T: LookupQuery<XLEN>,
{
    let mut rng = StdRng::seed_from_u64(12345);
    for _ in 0..10_000 {
        let raw: C = RandomLookupCycle::random(&mut rng);
        let instr = raw.instruction();
        let normalized = instr.jolt_instruction_row();
        let rs1_idx = normalized.operands.rs1;
        let rs2_idx = normalized.operands.rs2;
        let rd_idx = normalized.operands.rd;

        let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
        if let Some(rs1_val) = raw.rs1_val() {
            cpu.write_register(rs1_idx.unwrap() as usize, rs1_val as i64);
        }
        if let Some(rs2_val) = raw.rs2_val() {
            cpu.write_register(rs2_idx.unwrap() as usize, rs2_val as i64);
        }

        instr.trace(&mut cpu, None);

        let wrapped: T = cycle_wrapper(raw);
        let lookup_result = LookupQuery::<XLEN>::to_lookup_output(&wrapped);

        let is_jal = TypeId::of::<C::Instruction>() == TypeId::of::<JAL>();
        let is_jalr = TypeId::of::<C::Instruction>() == TypeId::of::<JALR>();
        if is_jal || is_jalr {
            let cpu_pc = cpu.read_pc();
            assert_eq!(cpu_pc, lookup_result, "{raw:?}");
        } else if let Some(rd) = rd_idx {
            // x0 is hardwired to zero; writes are discarded so the CPU
            // result is always 0 regardless of the lookup output.
            if rd != 0 {
                let cpu_result = cpu.x[rd as usize] as u64;
                assert_eq!(cpu_result, lookup_result, "{raw:?}");
            }
        } else {
            // Instruction has no `rd` and isn't `JAL`/`JALR`: the oracle
            // here doesn't apply (e.g. asserts, branches, stores, fence,
            // ecall/ebreak). Without an explicit panic the loop would run
            // 10k iterations and silently pass, hiding a coverage gap.
            // Either restrict the macro's call sites to instructions that
            // write `rd` or jump, or extend the oracle to handle the new
            // case (asserts → 1, branches → taken-bit, etc.).
            panic!(
                "lookup_output_matches_trace_test_fn invoked for an instruction \
                 without `rd` and not `JAL`/`JALR`; extend the oracle or skip \
                 this instruction. cycle = {raw:?}"
            );
        }
    }
}

/// Fuzz-check that an instruction's `to_lookup_output` agrees with the
/// corresponding lookup table's `materialize_entry(to_lookup_index)` across a
/// batch of random cycles. Pass the Jolt instruction newtype and the tracer
/// instruction path; the macro builds the `Foo<RISCVCycle<TracerType>>` /
/// `RISCVCycle<TracerType>` type pair.
///
/// ```ignore
/// materialize_entry_test!(Add, tracer::instruction::add::ADD);
/// ```
#[macro_export]
macro_rules! materialize_entry_test {
    ($jolt:ident, $tracer:path $(,)?) => {
        $crate::instructions::test::materialize_entry_test_fn::<
            $jolt<tracer::instruction::RISCVCycle<$tracer>>,
            tracer::instruction::RISCVCycle<$tracer>,
            $jolt<$tracer>,
        >($jolt, $jolt)
    };
}

/// Fuzz-check that an instruction's `LookupQuery::to_instruction_inputs`
/// matches the instruction-input R1CS constraint (see
/// [`instruction_inputs_match_constraint_fn`] for the formula).
///
/// ```ignore
/// instruction_inputs_match_constraint_test!(Add, tracer::instruction::add::ADD);
/// ```
#[macro_export]
macro_rules! instruction_inputs_match_constraint_test {
    ($jolt:ident, $tracer:path $(,)?) => {
        $crate::instructions::test::instruction_inputs_match_constraint_fn::<
            tracer::instruction::RISCVCycle<$tracer>,
            $jolt<tracer::instruction::RISCVCycle<$tracer>>,
            $jolt<$tracer>,
        >($jolt, $jolt)
    };
}

/// Fuzz-check that an instruction's `to_lookup_output` agrees with the value
/// tracer's CPU emulator writes to `rd` (or PC, for `JAL`/`JALR`) after
/// executing the instruction. Pass the Jolt instruction newtype and the
/// tracer instruction path; the macro builds the
/// `Foo<RISCVCycle<TracerType>>` / `RISCVCycle<TracerType>` type pair.
///
/// ```ignore
/// lookup_output_matches_trace_test!(Add, tracer::instruction::add::ADD);
/// ```
#[macro_export]
macro_rules! lookup_output_matches_trace_test {
    ($jolt:ident, $tracer:path $(,)?) => {
        $crate::instructions::test::lookup_output_matches_trace_test_fn::<
            tracer::instruction::RISCVCycle<$tracer>,
            $jolt<tracer::instruction::RISCVCycle<$tracer>>,
        >($jolt)
    };
}
