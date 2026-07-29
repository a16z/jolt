use jolt_field::Field;
use jolt_lookup_tables::{InstructionLookupTable, LookupQuery};
use jolt_program::execution::TraceRow;
use jolt_riscv::{
    CircuitFlags, Flags, InstructionFlags as InstructionFlagKind, InterleavedBitsMarker,
    JoltInstruction,
};

use super::{
    decode_instruction, lookup_query, row_circuit_flags, row_is_noop, Extract, ExtractIndexed,
    ToField, WitnessEnv,
};
use crate::WitnessError;
use crate::RV64_XLEN;

/// Whether the successor row is a no-op. The last cycle's missing successor
/// counts as a no-op: the product/shift family requires `NextIsNoop = 1` at
/// `T - 1` (legacy forces `not_next_noop = false` there — "EqPlusOne does not
/// do overflow").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NextIsNoop(pub bool);

/// Whether the successor row is a virtual instruction; false at the last
/// cycle and for undecodable successors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NextIsVirtual(pub bool);

/// Whether the successor row starts a virtual sequence; false at the last
/// cycle and for undecodable successors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NextIsFirstInSequence(pub bool);

/// Jump instruction actually taken (jump flag set and the successor is a
/// real instruction; unlike [`NextIsNoop`], a missing successor does NOT
/// count as a no-op here). At `T - 1` constraint 21
/// (`ShouldJump = Jump · (1 − NextIsNoop)`) forces `ShouldJump = 0`, which
/// holds under either convention only because the padded trace ends in a
/// NoOp row (`Jump = 0` there) — the padding every prover config guarantees.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShouldJump(pub bool);

/// Branch instruction whose comparison output is 1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShouldBranch(pub bool);

/// Set when the instruction's lookup operands are NOT interleaved (the RAF
/// address decomposition applies).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InstructionRafFlag(pub bool);

/// One circuit flag of the instruction; which flag is bound at the use site.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpFlag(pub bool);

/// One instruction flag of the instruction; which flag is bound at the use
/// site.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InstructionFlag(pub bool);

/// Whether the instruction's lookup targets the table bound at the use site.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LookupTableFlag(pub bool);

macro_rules! bool_to_field {
    ($($name:ident),* $(,)?) => {
        $(impl ToField for $name {
            fn to_field<F: Field>(self) -> F {
                F::from_bool(self.0)
            }
        })*
    };
}

bool_to_field!(
    NextIsNoop,
    NextIsVirtual,
    NextIsFirstInSequence,
    ShouldJump,
    ShouldBranch,
    InstructionRafFlag,
    OpFlag,
    InstructionFlag,
    LookupTableFlag,
);

impl Extract for NextIsNoop {
    fn extract(
        _row: &TraceRow,
        next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(next.is_none_or(row_is_noop)))
    }
}

impl Extract for NextIsVirtual {
    fn extract(
        _row: &TraceRow,
        next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(next.is_some_and(|row| {
            row_circuit_flags(row).is_ok_and(|flags| flags[CircuitFlags::VirtualInstruction])
        })))
    }
}

impl Extract for NextIsFirstInSequence {
    fn extract(
        _row: &TraceRow,
        next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(next.is_some_and(|row| {
            row_circuit_flags(row).is_ok_and(|flags| flags[CircuitFlags::IsFirstInSequence])
        })))
    }
}

impl Extract for ShouldJump {
    fn extract(
        row: &TraceRow,
        next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let circuit_flags = row_circuit_flags(row)?;
        let next_is_noop = next.is_some_and(row_is_noop);
        Ok(Self(circuit_flags[CircuitFlags::Jump] && !next_is_noop))
    }
}

impl Extract for ShouldBranch {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let instruction_flags = decode_instruction(row)?.instruction_flags();
        let lookup_output = LookupQuery::<RV64_XLEN>::to_lookup_output(&lookup_query(row));
        Ok(Self(
            instruction_flags[InstructionFlagKind::Branch] && lookup_output == 1,
        ))
    }
}

impl Extract for InstructionRafFlag {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let circuit_flags = row_circuit_flags(row)?;
        Ok(Self(!circuit_flags.is_interleaved_operands()))
    }
}

impl ExtractIndexed<CircuitFlags> for OpFlag {
    fn extract_indexed(
        flag: CircuitFlags,
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row_circuit_flags(row)?[flag]))
    }
}

impl ExtractIndexed<InstructionFlagKind> for InstructionFlag {
    fn extract_indexed(
        flag: InstructionFlagKind,
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(decode_instruction(row)?.instruction_flags()[flag]))
    }
}

impl ExtractIndexed<usize> for LookupTableFlag {
    fn extract_indexed(
        table: usize,
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let instruction = decode_instruction(row)?;
        let table_index =
            <JoltInstruction as InstructionLookupTable<RV64_XLEN>>::lookup_table(&instruction)
                .map(|kind| kind.index());
        Ok(Self(table_index == Some(table)))
    }
}
