use std::any::TypeId;
use std::fmt::Debug;

use jolt_riscv::{Flags, InstructionFlags, JoltCycle, JoltInstructionRowData};
use rand::prelude::*;
use tracer::emulator::{cpu::Cpu, terminal::DummyTerminal};
use tracer::instruction::{jal::JAL, jalr::JALR, Cycle, RISCVCycle, RISCVInstruction, RISCVTrace};

use crate::{InstructionLookupTable, LookupQuery, XLEN};

pub trait RandomLookupCycle: JoltCycle {
    fn random(rng: &mut StdRng) -> Self;
    fn initialize_cpu(&self, cpu: &mut Cpu);
}

impl<T> RandomLookupCycle for RISCVCycle<T>
where
    T: RISCVInstruction + JoltInstructionRowData,
{
    fn random(rng: &mut StdRng) -> Self {
        T::random_cycle(rng)
    }

    #[expect(
        clippy::unwrap_used,
        reason = "register values require corresponding operands"
    )]
    fn initialize_cpu(&self, cpu: &mut Cpu) {
        let operands = self.instruction.jolt_instruction_row().operands;
        if let Some((pre, _)) = self.rd_vals() {
            cpu.write_register(operands.rd.unwrap() as usize, pre as i64);
        }
        if let Some(value) = self.rs1_val() {
            cpu.write_register(operands.rs1.unwrap() as usize, value as i64);
        }
        if let Some(value) = self.rs2_val() {
            cpu.write_register(operands.rs2.unwrap() as usize, value as i64);
        }
        T::initialize_test_cpu(self, cpu);
    }
}

#[doc(hidden)]
#[expect(clippy::unwrap_used)]
pub fn materialize_entry_test_fn<T, C, I>(
    cycle_wrapper: impl Fn(C) -> T,
    instr_wrapper: impl Fn(C::Instruction) -> I,
) where
    T: LookupQuery<XLEN> + Debug,
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

#[doc(hidden)]
pub fn instruction_inputs_match_constraint_fn<C, T, I>(
    cycle_wrapper: impl Fn(C) -> T,
    instr_wrapper: impl Fn(C::Instruction) -> I,
) where
    C: RandomLookupCycle,
    T: LookupQuery<XLEN> + Debug,
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

#[doc(hidden)]
#[expect(
    clippy::panic,
    reason = "deliberate guard against silent passes; see body"
)]
pub fn lookup_output_matches_trace_test_fn<C, T>(cycle_wrapper: impl Fn(C) -> T)
where
    C: RandomLookupCycle + Copy + Debug,
    C::Instruction: RISCVTrace + 'static,
    RISCVCycle<C::Instruction>: Into<Cycle>,
    T: LookupQuery<XLEN>,
{
    let mut rng = StdRng::seed_from_u64(12345);
    for _ in 0..10_000 {
        let raw: C = RandomLookupCycle::random(&mut rng);
        let instr = raw.instruction();
        let normalized = instr.jolt_instruction_row();
        let rd_idx = normalized.operands.rd;

        let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
        raw.initialize_cpu(&mut cpu);

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
