//! Committed-column walks over the atomic extractors: the committed analog
//! of the bundle pass, driven by the same [`RowSource`] row walk.
//!
//! Padding beyond the physical trace is default (no-op) rows, whose
//! extraction coincides with the committed conventions by construction: a
//! no-op's lookup index is 0 and `get_pc` short-circuits no-ops to slot 0,
//! so instruction/bytecode one-hots pad to the address-0 chunk, RAM one-hots
//! to cold cycles, and increments to zero.

use super::*;
use crate::witnesses::{
    BytecodeRaChunk, Extract, ExtractIndexed, InstructionRaChunk, RaChunkSelector, RamInc,
    RamRaChunk, RdInc, WitnessEnv,
};
use crate::{ColumnVisitor, CommittedChunk, RowSource};

impl<T: TraceSource + Clone> TraceBackend<'_, T> {
    /// Walks the committed polynomial's coefficients in order as chunks of
    /// at most `chunk_size` values, over the same row walk as the bundle
    /// pass.
    pub fn visit_committed_column<F: Field>(
        &self,
        id: JoltCommittedPolynomial,
        chunk_size: usize,
        visitor: &mut ColumnVisitor<'_, F>,
    ) -> Result<(), WitnessError> {
        if chunk_size == 0 {
            return Err(WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: "column chunk size must be nonzero".to_owned(),
            });
        }
        let layout = self.ra_layout()?;
        match self.committed_column_kind(id, layout)? {
            JoltVmColumnKind::Increment(kind) => {
                self.visit_cycle_column(chunk_size, kind, visitor, |buffer| {
                    CommittedChunk::Increments(buffer)
                })
            }
            JoltVmColumnKind::OneHot(kind) => {
                self.visit_cycle_column(chunk_size, kind, visitor, |buffer| {
                    CommittedChunk::HotAddresses(buffer)
                })
            }
            JoltVmColumnKind::Advice(kind) => {
                let bytes = self.advice_bytes(kind);
                let rows = kind.rows(self.preprocessing);
                let mut buffer = Vec::with_capacity(chunk_size.min(rows));
                let mut emitted = 0;
                while emitted < rows {
                    let end = emitted.saturating_add(chunk_size).min(rows);
                    buffer.clear();
                    buffer.extend((emitted..end).map(|word| advice_word_le(bytes, word)));
                    visitor(CommittedChunk::Words(&buffer))?;
                    emitted = end;
                }
                Ok(())
            }
        }
    }

    /// One [`RowSource`] walk over the `2^log_t` cycle domain, each row
    /// buffer mapped through `values`'s per-row derivation.
    fn visit_cycle_column<F, S, V: ColumnValues<S>>(
        &self,
        chunk_size: usize,
        values: V,
        visitor: &mut ColumnVisitor<'_, F>,
        wrap: impl for<'b> Fn(&'b [S]) -> CommittedChunk<'b, F>,
    ) -> Result<(), WitnessError> {
        let total = checked_pow2(self.config.log_t)?;
        let mut buffer = Vec::with_capacity(chunk_size.min(total));
        self.visit_chunks(0..total, chunk_size, &mut |rows, _next_after, env| {
            buffer.clear();
            for row in rows {
                buffer.push(values.value(row, env)?);
            }
            visitor(wrap(&buffer))
        })
    }
}

/// A committed column's per-row scalar derivation, dispatching to the atomic
/// extractors.
trait ColumnValues<S> {
    fn value(&self, row: &TraceRow, env: &WitnessEnv<'_>) -> Result<S, WitnessError>;
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum JoltVmColumnKind {
    Increment(JoltVmIncrementKind),
    OneHot(JoltVmOneHotKind),
    Advice(JoltVmAdviceKind),
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum JoltVmIncrementKind {
    RdInc,
    RamInc,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum JoltVmOneHotKind {
    Instruction(RaChunkSelector),
    Bytecode(RaChunkSelector),
    Ram(RaChunkSelector),
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum JoltVmAdviceKind {
    Trusted,
    Untrusted,
}

impl ColumnValues<i128> for JoltVmIncrementKind {
    fn value(&self, row: &TraceRow, env: &WitnessEnv<'_>) -> Result<i128, WitnessError> {
        Ok(match self {
            Self::RdInc => RdInc::extract(row, None, env)?.0,
            Self::RamInc => RamInc::extract(row, None, env)?.0,
        })
    }
}

impl ColumnValues<Option<usize>> for JoltVmOneHotKind {
    fn value(&self, row: &TraceRow, env: &WitnessEnv<'_>) -> Result<Option<usize>, WitnessError> {
        Ok(match self {
            Self::Instruction(selector) => {
                Some(InstructionRaChunk::extract_indexed(*selector, row, None, env)?.0)
            }
            Self::Bytecode(selector) => {
                BytecodeRaChunk::extract_indexed(*selector, row, None, env)?.0
            }
            Self::Ram(selector) => RamRaChunk::extract_indexed(*selector, row, None, env)?.0,
        })
    }
}

impl<T: TraceSource + Clone> TraceBackend<'_, T> {
    pub(crate) fn committed_column_kind(
        &self,
        polynomial: JoltCommittedPolynomial,
        layout: JoltRaPolynomialLayout,
    ) -> Result<JoltVmColumnKind, WitnessError> {
        match polynomial {
            JoltCommittedPolynomial::RdInc => {
                Ok(JoltVmColumnKind::Increment(JoltVmIncrementKind::RdInc))
            }
            JoltCommittedPolynomial::RamInc => {
                Ok(JoltVmColumnKind::Increment(JoltVmIncrementKind::RamInc))
            }
            JoltCommittedPolynomial::InstructionRa(index) => {
                require_index(index, layout.instruction())?;
                Ok(JoltVmColumnKind::OneHot(JoltVmOneHotKind::Instruction(
                    RaChunkSelector::new(
                        index,
                        layout.instruction(),
                        self.config.one_hot.committed_chunk_bits(),
                    )?,
                )))
            }
            JoltCommittedPolynomial::BytecodeRa(index) => {
                require_index(index, layout.bytecode())?;
                Ok(JoltVmColumnKind::OneHot(JoltVmOneHotKind::Bytecode(
                    RaChunkSelector::new(
                        index,
                        layout.bytecode(),
                        self.config.one_hot.committed_chunk_bits(),
                    )?,
                )))
            }
            JoltCommittedPolynomial::RamRa(index) => {
                require_index(index, layout.ram())?;
                Ok(JoltVmColumnKind::OneHot(JoltVmOneHotKind::Ram(
                    RaChunkSelector::new(
                        index,
                        layout.ram(),
                        self.config.one_hot.committed_chunk_bits(),
                    )?,
                )))
            }
            JoltCommittedPolynomial::TrustedAdvice => {
                self.advice_column_kind(JoltVmAdviceKind::Trusted)
            }
            JoltCommittedPolynomial::UntrustedAdvice => {
                self.advice_column_kind(JoltVmAdviceKind::Untrusted)
            }
            JoltCommittedPolynomial::BytecodeChunk(_)
            | JoltCommittedPolynomial::ProgramImageInit
            | JoltCommittedPolynomial::UnsignedIncChunk(_)
            | JoltCommittedPolynomial::UnsignedIncMsb
            | JoltCommittedPolynomial::TrustedAdviceBytes
            | JoltCommittedPolynomial::UntrustedAdviceBytes
            | JoltCommittedPolynomial::BytecodeRegisterSelector { .. }
            | JoltCommittedPolynomial::BytecodeCircuitFlag { .. }
            | JoltCommittedPolynomial::BytecodeInstructionFlag { .. }
            | JoltCommittedPolynomial::BytecodeLookupSelector { .. }
            | JoltCommittedPolynomial::BytecodeRafFlag { .. }
            | JoltCommittedPolynomial::BytecodeUnexpandedPcBytes { .. }
            | JoltCommittedPolynomial::BytecodeImmBytes { .. }
            | JoltCommittedPolynomial::ProgramImageBytes => Err(WitnessError::UnknownOracle {
                label: JOLT_VM_LABEL,
            }),
        }
    }

    fn advice_column_kind(&self, kind: JoltVmAdviceKind) -> Result<JoltVmColumnKind, WitnessError> {
        if !kind.is_included(&self.config) {
            return Err(WitnessError::UnknownOracle {
                label: JOLT_VM_LABEL,
            });
        }
        kind.validate_len(
            self.advice_bytes(kind).len(),
            self.preprocessing,
            JOLT_VM_LABEL,
        )?;
        Ok(JoltVmColumnKind::Advice(kind))
    }

    fn advice_bytes(&self, kind: JoltVmAdviceKind) -> &[u8] {
        match kind {
            JoltVmAdviceKind::Trusted => &self.trace.device.trusted_advice,
            JoltVmAdviceKind::Untrusted => &self.trace.device.untrusted_advice,
        }
    }
}

impl JoltVmAdviceKind {
    const fn is_included(self, config: &JoltVmWitnessConfig) -> bool {
        match self {
            Self::Trusted => config.include_trusted_advice,
            Self::Untrusted => config.include_untrusted_advice,
        }
    }

    const fn max_bytes(self, preprocessing: &JoltProgramPreprocessing) -> usize {
        match self {
            Self::Trusted => preprocessing.memory_layout.max_trusted_advice_size as usize,
            Self::Untrusted => preprocessing.memory_layout.max_untrusted_advice_size as usize,
        }
    }

    pub(crate) fn rows(self, preprocessing: &JoltProgramPreprocessing) -> usize {
        let words = self.max_bytes(preprocessing) / 8;
        words.next_power_of_two().max(1)
    }

    fn validate_len(
        self,
        bytes_len: usize,
        preprocessing: &JoltProgramPreprocessing,
        label: &'static str,
    ) -> Result<(), WitnessError> {
        let max_bytes = self.max_bytes(preprocessing);
        if bytes_len > max_bytes {
            return Err(WitnessError::InvalidWitnessData {
                label,
                reason: format!(
                    "{self:?} advice has {bytes_len} bytes, exceeding configured max {max_bytes}",
                ),
            });
        }
        Ok(())
    }
}

pub(crate) fn advice_word_le(bytes: &[u8], word_index: usize) -> u64 {
    let Some(start) = word_index.checked_mul(8) else {
        return 0;
    };
    if start >= bytes.len() {
        return 0;
    }
    let end = start.saturating_add(8).min(bytes.len());
    let mut word = [0_u8; 8];
    word[..end - start].copy_from_slice(&bytes[start..end]);
    u64::from_le_bytes(word)
}

impl<T: TraceSource + Clone> TraceBackend<'_, T> {
    pub(crate) fn materialize_compact_committed<F: Field>(
        &self,
        id: JoltCommittedPolynomial,
    ) -> Result<Vec<F>, WitnessError> {
        if matches!(
            id,
            JoltCommittedPolynomial::InstructionRa(_)
                | JoltCommittedPolynomial::BytecodeRa(_)
                | JoltCommittedPolynomial::RamRa(_)
        ) {
            return self.materialize_one_hot_committed(id);
        }

        let shape = self.shape_of(JoltPolynomialId::Committed(id))?;
        let mut values = Vec::with_capacity(shape.rows());
        self.visit_committed_column::<F>(id, shape.rows().max(1), &mut |chunk| {
            match chunk {
                CommittedChunk::Dense(chunk) => values.extend_from_slice(chunk),
                CommittedChunk::Zeros(rows) => {
                    values.extend(std::iter::repeat_n(F::zero(), rows));
                }
                CommittedChunk::Words(chunk) => {
                    values.extend(chunk.iter().copied().map(F::from_u64));
                }
                CommittedChunk::Increments(chunk) => {
                    values.extend(chunk.iter().copied().map(F::from_i128));
                }
                CommittedChunk::HotAddresses(_) => {
                    return Err(WitnessError::UnsupportedView {
                        view: "one-hot chunk materialization as compact field values",
                    });
                }
            }
            Ok(())
        })?;
        if values.len() != shape.rows() {
            return Err(WitnessError::InvalidWitnessData {
                label: JOLT_VM_LABEL,
                reason: format!(
                    "committed oracle {id:?} materialized {} rows, expected {}",
                    values.len(),
                    shape.rows()
                ),
            });
        }
        Ok(values)
    }

    pub(crate) fn materialize_one_hot_committed<F: Field>(
        &self,
        id: JoltCommittedPolynomial,
    ) -> Result<Vec<F>, WitnessError> {
        let shape = self.shape_of(JoltPolynomialId::Committed(id))?;
        let cycles = checked_pow2(self.config.log_t)?;
        if !shape.rows().is_multiple_of(cycles) {
            return Err(WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: format!(
                    "committed oracle {id:?} has {} rows, not divisible by {cycles} cycles",
                    shape.rows()
                ),
            });
        }
        let addresses = shape.rows() / cycles;
        let mut values = vec![F::zero(); shape.rows()];
        let mut cycle = 0usize;
        self.visit_committed_column::<F>(id, cycles.max(1), &mut |chunk| {
            let CommittedChunk::HotAddresses(chunk) = chunk else {
                return Err(WitnessError::InvalidWitnessData {
                    label: JOLT_VM_LABEL,
                    reason: format!("committed oracle {id:?} did not stream one-hot chunks"),
                });
            };
            for &address in chunk {
                if let Some(address) = address {
                    if address >= addresses {
                        return Err(WitnessError::InvalidWitnessData {
                            label: JOLT_VM_LABEL,
                            reason: format!(
                                "committed oracle {id:?} streamed address {address}, beyond {addresses}"
                            ),
                        });
                    }
                    values[address * cycles + cycle] = F::one();
                }
                cycle += 1;
            }
            Ok(())
        })?;
        if cycle != cycles {
            return Err(WitnessError::InvalidWitnessData {
                label: JOLT_VM_LABEL,
                reason: format!("committed oracle {id:?} streamed {cycle} rows, expected {cycles}"),
            });
        }
        Ok(values)
    }
}
