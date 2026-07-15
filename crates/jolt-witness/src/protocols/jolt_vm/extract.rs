//! Single-sourced per-witness derivations from trace rows.
//!
//! Each atomic witness in [`crate::witnesses`] has exactly one [`Extract`]
//! impl here; every consumer path (oracle tables, bundles, streams)
//! dispatches to these impls. Extractors recompute from row accessors — no
//! memoization. The two irreducible non-row inputs are the lookahead window
//! (the `Next*` family is a function of rows `t` and `t + 1`, with padding
//! semantics at `T - 1`) and the environment ([`WitnessEnv`]).

use jolt_claims::protocols::jolt::JoltVirtualPolynomial;
use jolt_field::{
    signed::{S128, S64},
    Field,
};
use jolt_lookup_tables::{InstructionLookupTable, JoltLookupQuery, LookupQuery, LookupTableKind};
use jolt_program::{
    execution::{RamAccess, TraceRow},
    preprocess::JoltProgramPreprocessing,
};
use jolt_riscv::{
    CircuitFlags, Flags, InstructionFlags as InstructionFlagKind, InterleavedBitsMarker,
    JoltInstruction, JoltInstructionKind,
};

use super::{ram_access_address, JOLT_VM_NAMESPACE, RV64_XLEN};
use crate::witnesses::{
    Imm, InstructionFlag, InstructionRafFlag, LeftInstructionInput, LeftLookupOperand,
    LookupOutput, LookupTableFlag, NextIsFirstInSequence, NextIsNoop, NextIsVirtual, NextPc,
    NextUnexpandedPc, OpFlag, Pc, Product, RamAddress, RamHammingWeight, RamReadValue,
    RamWriteValue, RdWriteValue, RightInstructionInput, RightLookupOperand, Rs1Value, Rs2Value,
    ShouldBranch, ShouldJump, UnexpandedPc,
};
use crate::WitnessError;

/// Non-row inputs of witness extraction: the preprocessing (bytecode PC
/// mapping, memory layout).
pub(crate) struct WitnessEnv<'a> {
    pub(crate) preprocessing: &'a JoltProgramPreprocessing,
}

/// The single-sourced derivation of one atomic witness from a trace row.
pub(crate) trait Extract: Sized {
    fn extract(
        row: &TraceRow,
        next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError>;
}

/// [`Extract`] for indexed witness families ([`OpFlag`], [`InstructionFlag`],
/// [`LookupTableFlag`]): which member is extracted is bound at the use site.
pub(crate) trait ExtractIndexed<I>: Sized {
    fn extract_indexed(
        index: I,
        row: &TraceRow,
        next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError>;
}

impl Extract for Pc {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        pc_for_row(row, env.preprocessing).map(|pc| Self(pc as u64))
    }
}

impl Extract for UnexpandedPc {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.instruction.address as u64))
    }
}

impl Extract for NextPc {
    fn extract(
        _row: &TraceRow,
        next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(
            next.map(|row| pc_for_row(row, env.preprocessing))
                .transpose()?
                .map_or(0, |pc| pc as u64),
        ))
    }
}

impl Extract for NextUnexpandedPc {
    fn extract(
        _row: &TraceRow,
        next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(next.map_or(0, |row| row.instruction.address as u64)))
    }
}

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

impl Extract for LeftLookupOperand {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let (left, _) = LookupQuery::<RV64_XLEN>::to_lookup_operands(&lookup_query(row));
        Ok(Self(left))
    }
}

impl Extract for RightLookupOperand {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let (_, right) = LookupQuery::<RV64_XLEN>::to_lookup_operands(&lookup_query(row));
        Ok(Self(right))
    }
}

impl Extract for LeftInstructionInput {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let (left, _) = LookupQuery::<RV64_XLEN>::to_instruction_inputs(&lookup_query(row));
        Ok(Self(left))
    }
}

impl Extract for RightInstructionInput {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let (_, right) = LookupQuery::<RV64_XLEN>::to_instruction_inputs(&lookup_query(row));
        Ok(Self(right))
    }
}

impl Extract for Product {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let (left, right) = LookupQuery::<RV64_XLEN>::to_instruction_inputs(&lookup_query(row));
        Ok(Self(
            S64::from_u64(left).mul_trunc::<2, 2>(&S128::from_i128(right)),
        ))
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

impl Extract for Imm {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.instruction.operands.imm))
    }
}

impl Extract for Rs1Value {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.registers.rs1.map_or(0, |read| read.value)))
    }
}

impl Extract for Rs2Value {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.registers.rs2.map_or(0, |read| read.value)))
    }
}

impl Extract for RdWriteValue {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.registers.rd.map_or(0, |write| write.post_value)))
    }
}

impl Extract for LookupOutput {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(LookupQuery::<RV64_XLEN>::to_lookup_output(
            &lookup_query(row),
        )))
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

impl Extract for RamAddress {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(ram_access_address(row.ram_access).unwrap_or(0)))
    }
}

impl Extract for RamReadValue {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(match row.ram_access {
            RamAccess::Read(read) => read.value,
            RamAccess::Write(write) => write.pre_value,
            RamAccess::NoOp => 0,
        }))
    }
}

impl Extract for RamWriteValue {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(match row.ram_access {
            RamAccess::Read(read) => read.value,
            RamAccess::Write(write) => write.post_value,
            RamAccess::NoOp => 0,
        }))
    }
}

impl Extract for RamHammingWeight {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(
            ram_access_address(row.ram_access).is_some_and(|address| address != 0),
        ))
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

pub(crate) const fn supported_trace_virtual(id: JoltVirtualPolynomial) -> bool {
    matches!(
        id,
        JoltVirtualPolynomial::PC
            | JoltVirtualPolynomial::UnexpandedPC
            | JoltVirtualPolynomial::NextPC
            | JoltVirtualPolynomial::NextUnexpandedPC
            | JoltVirtualPolynomial::NextIsNoop
            | JoltVirtualPolynomial::NextIsVirtual
            | JoltVirtualPolynomial::NextIsFirstInSequence
            | JoltVirtualPolynomial::LeftLookupOperand
            | JoltVirtualPolynomial::RightLookupOperand
            | JoltVirtualPolynomial::LeftInstructionInput
            | JoltVirtualPolynomial::RightInstructionInput
            | JoltVirtualPolynomial::Product
            | JoltVirtualPolynomial::ShouldJump
            | JoltVirtualPolynomial::ShouldBranch
            | JoltVirtualPolynomial::Imm
            | JoltVirtualPolynomial::Rs1Value
            | JoltVirtualPolynomial::Rs2Value
            | JoltVirtualPolynomial::RdWriteValue
            | JoltVirtualPolynomial::LookupOutput
            | JoltVirtualPolynomial::RamAddress
            | JoltVirtualPolynomial::RamReadValue
            | JoltVirtualPolynomial::RamWriteValue
            | JoltVirtualPolynomial::RamHammingWeight
            | JoltVirtualPolynomial::InstructionRafFlag
            | JoltVirtualPolynomial::OpFlags(_)
            | JoltVirtualPolynomial::InstructionFlags(_)
            | JoltVirtualPolynomial::LookupTableFlag(_)
    )
}

/// One cycle-domain witness value, dispatched to its atomic extractor.
pub(crate) fn cycle_witness_value<F: Field>(
    id: JoltVirtualPolynomial,
    row: &TraceRow,
    next: Option<&TraceRow>,
    env: &WitnessEnv<'_>,
) -> Result<F, WitnessError> {
    match id {
        JoltVirtualPolynomial::PC => Pc::extract(row, next, env).map(|w| w.to_field()),
        JoltVirtualPolynomial::UnexpandedPC => {
            UnexpandedPc::extract(row, next, env).map(|w| w.to_field())
        }
        JoltVirtualPolynomial::NextPC => NextPc::extract(row, next, env).map(|w| w.to_field()),
        JoltVirtualPolynomial::NextUnexpandedPC => {
            NextUnexpandedPc::extract(row, next, env).map(|w| w.to_field())
        }
        JoltVirtualPolynomial::NextIsNoop => {
            NextIsNoop::extract(row, next, env).map(|w| w.to_field())
        }
        JoltVirtualPolynomial::NextIsVirtual => {
            NextIsVirtual::extract(row, next, env).map(|w| w.to_field())
        }
        JoltVirtualPolynomial::NextIsFirstInSequence => {
            NextIsFirstInSequence::extract(row, next, env).map(|w| w.to_field())
        }
        JoltVirtualPolynomial::LeftLookupOperand => {
            LeftLookupOperand::extract(row, next, env).map(|w| w.to_field())
        }
        JoltVirtualPolynomial::RightLookupOperand => {
            RightLookupOperand::extract(row, next, env).map(|w| w.to_field())
        }
        JoltVirtualPolynomial::LeftInstructionInput => {
            LeftInstructionInput::extract(row, next, env).map(|w| w.to_field())
        }
        JoltVirtualPolynomial::RightInstructionInput => {
            RightInstructionInput::extract(row, next, env).map(|w| w.to_field())
        }
        JoltVirtualPolynomial::Product => Product::extract(row, next, env).map(|w| w.to_field()),
        JoltVirtualPolynomial::ShouldJump => {
            ShouldJump::extract(row, next, env).map(|w| w.to_field())
        }
        JoltVirtualPolynomial::ShouldBranch => {
            ShouldBranch::extract(row, next, env).map(|w| w.to_field())
        }
        JoltVirtualPolynomial::Imm => Imm::extract(row, next, env).map(|w| w.to_field()),
        JoltVirtualPolynomial::Rs1Value => Rs1Value::extract(row, next, env).map(|w| w.to_field()),
        JoltVirtualPolynomial::Rs2Value => Rs2Value::extract(row, next, env).map(|w| w.to_field()),
        JoltVirtualPolynomial::RdWriteValue => {
            RdWriteValue::extract(row, next, env).map(|w| w.to_field())
        }
        JoltVirtualPolynomial::LookupOutput => {
            LookupOutput::extract(row, next, env).map(|w| w.to_field())
        }
        JoltVirtualPolynomial::InstructionRafFlag => {
            InstructionRafFlag::extract(row, next, env).map(|w| w.to_field())
        }
        JoltVirtualPolynomial::RamAddress => {
            RamAddress::extract(row, next, env).map(|w| w.to_field())
        }
        JoltVirtualPolynomial::RamReadValue => {
            RamReadValue::extract(row, next, env).map(|w| w.to_field())
        }
        JoltVirtualPolynomial::RamWriteValue => {
            RamWriteValue::extract(row, next, env).map(|w| w.to_field())
        }
        JoltVirtualPolynomial::RamHammingWeight => {
            RamHammingWeight::extract(row, next, env).map(|w| w.to_field())
        }
        JoltVirtualPolynomial::OpFlags(flag) => {
            OpFlag::extract_indexed(flag, row, next, env).map(|w| w.to_field())
        }
        JoltVirtualPolynomial::InstructionFlags(flag) => {
            InstructionFlag::extract_indexed(flag, row, next, env).map(|w| w.to_field())
        }
        JoltVirtualPolynomial::LookupTableFlag(table) => {
            if table >= LookupTableKind::<RV64_XLEN>::COUNT {
                return Err(WitnessError::UnknownOracle {
                    namespace: JOLT_VM_NAMESPACE.name,
                });
            }
            LookupTableFlag::extract_indexed(table, row, next, env).map(|w| w.to_field())
        }
        _ => Err(WitnessError::UnknownOracle {
            namespace: JOLT_VM_NAMESPACE.name,
        }),
    }
}

fn lookup_query(row: &TraceRow) -> JoltLookupQuery<&TraceRow> {
    JoltLookupQuery::new(row.instruction.instruction_kind, row)
}

pub(crate) fn decode_instruction(row: &TraceRow) -> Result<JoltInstruction, WitnessError> {
    JoltInstruction::try_from(row.instruction).map_err(|kind| WitnessError::InvalidWitnessData {
        namespace: JOLT_VM_NAMESPACE.name,
        reason: format!("unsupported Jolt instruction kind in trace row: {kind:?}"),
    })
}

pub(crate) fn row_circuit_flags(
    row: &TraceRow,
) -> Result<jolt_riscv::CircuitFlagSet, WitnessError> {
    Ok(decode_instruction(row)?.circuit_flags())
}

pub(crate) fn row_is_noop(row: &TraceRow) -> bool {
    row.instruction.instruction_kind == JoltInstructionKind::NoOp
}

pub(crate) fn pc_for_row(
    row: &TraceRow,
    preprocessing: &JoltProgramPreprocessing,
) -> Result<usize, WitnessError> {
    preprocessing
        .bytecode
        .get_pc(&row.instruction)
        .ok_or_else(|| missing_pc_mapping(row))
}

pub(crate) fn missing_pc_mapping(row: &TraceRow) -> WitnessError {
    WitnessError::InvalidWitnessData {
        namespace: JOLT_VM_NAMESPACE.name,
        reason: format!(
            "bytecode preprocessing is missing PC mapping for address {:#x} with virtual_sequence_remaining {:?}",
            row.instruction.address, row.instruction.virtual_sequence_remaining
        ),
    }
}
