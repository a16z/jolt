//! Cycle-domain virtual polynomial materialization over the atomic
//! extractors.

use super::*;
use crate::witnesses::{
    row_is_noop, Extract, ExtractIndexed, Imm, InstructionFlag, InstructionRafFlag,
    LeftInstructionInput, LeftLookupOperand, LookupOutput, LookupTableFlag, NextIsFirstInSequence,
    NextIsNoop, NextIsVirtual, NextPc, NextUnexpandedPc, OpFlag, Pc, Product, RamAddress,
    RamHammingWeight, RamReadValue, RamWriteValue, RdWriteValue, RightInstructionInput,
    RightLookupOperand, Rs1Value, Rs2Value, ShouldBranch, ShouldJump, UnexpandedPc, WitnessEnv,
};

impl<T: TraceSource + Clone> TraceBackend<'_, T> {
    pub(crate) fn materialize_trace_virtual<F: Field>(
        &self,
        id: JoltVirtualPolynomial,
    ) -> Result<Vec<F>, WitnessError> {
        let rows = checked_pow2(self.config.log_t)?;
        let env = WitnessEnv {
            preprocessing: self.preprocessing,
        };
        let mut values = Vec::with_capacity(rows);
        let mut trace = self.trace.trace.clone();
        let mut current = trace.next_row().unwrap_or_default();
        for index in 0..rows {
            let next = (index + 1 < rows).then(|| trace.next_row().unwrap_or_default());
            values.push(cycle_witness_value::<F>(id, &current, next.as_ref(), &env)?);
            if let Some(row) = next {
                current = row;
            }
        }
        Ok(values)
    }
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
                    label: JOLT_VM_LABEL,
                });
            }
            LookupTableFlag::extract_indexed(table, row, next, env).map(|w| w.to_field())
        }
        _ => Err(WitnessError::UnknownOracle {
            label: JOLT_VM_LABEL,
        }),
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct PcLookupCache {
    values: HashMap<(usize, u16), usize>,
}

impl PcLookupCache {
    pub(crate) fn pc_for_row_optional(
        &mut self,
        row: &TraceRow,
        preprocessing: &JoltProgramPreprocessing,
    ) -> Option<usize> {
        if row_is_noop(row) {
            return Some(0);
        }
        let key = pc_lookup_key(row);
        if let Some(&pc) = self.values.get(&key) {
            return Some(pc);
        }
        let pc = preprocessing.bytecode.get_pc(&row.instruction)?;
        let _ = self.values.insert(key, pc);
        Some(pc)
    }
}

pub(crate) fn pc_lookup_key(row: &TraceRow) -> (usize, u16) {
    (
        row.instruction.address,
        row.instruction.virtual_sequence_remaining.unwrap_or(0),
    )
}
