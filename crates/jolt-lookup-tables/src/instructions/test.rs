//! Per-instruction test helpers.

use jolt_trace::{Flags, InstructionFlags, JoltCycle, JoltInstruction};
use rand::prelude::*;

use crate::{InstructionLookupTable, LookupQuery, XLEN};

/// Internal helper for [`materialize_entry_test!`]. The macro picks up the
/// verbose `Foo<RISCVCycle<TracerType>>` / `RISCVCycle<TracerType>` type pair
/// from a Jolt struct ident and a tracer instruction path, and passes the
/// tuple-struct constructor as `construct`.
#[doc(hidden)]
#[expect(clippy::unwrap_used)]
pub fn materialize_entry_test_fn<T, C>(construct: impl Fn(C) -> T)
where
    T: InstructionLookupTable<XLEN> + LookupQuery<XLEN> + core::fmt::Debug,
    C: JoltCycle,
{
    let mut rng = StdRng::seed_from_u64(12345);
    for _ in 0..10_000 {
        let cycle: T = construct(C::random(&mut rng));
        let table = cycle.lookup_table().unwrap();
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
pub fn instruction_inputs_match_constraint_fn<T, C>(construct: impl Fn(C) -> T)
where
    T: LookupQuery<XLEN> + Flags + core::fmt::Debug,
    C: JoltCycle,
{
    let mut rng = StdRng::seed_from_u64(12345);
    for _ in 0..10_000 {
        let raw: C = C::random(&mut rng);
        let instr = raw.instruction();
        let unexpanded_pc = instr.address();
        let imm = instr.imm();
        let rs1 = raw.rs1_val().unwrap_or(0);
        let rs2 = raw.rs2_val().unwrap_or(0);

        let cycle: T = construct(raw);
        let flags = cycle.instruction_flags();

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
        >($jolt)
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
            $jolt<tracer::instruction::RISCVCycle<$tracer>>,
            tracer::instruction::RISCVCycle<$tracer>,
        >($jolt)
    };
}
