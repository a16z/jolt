//! One-hot RA chunk selection and RA-family evaluation.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RaChunkSelector {
    shift: usize,
    mask: u128,
}

impl RaChunkSelector {
    pub(crate) fn new(
        index: usize,
        chunks: usize,
        chunk_bits: usize,
    ) -> Result<Self, WitnessError> {
        let remaining = chunks
            .checked_sub(index + 1)
            .ok_or(WitnessError::UnknownOracle {
                namespace: JOLT_VM_NAMESPACE.name,
            })?;
        let shift =
            remaining
                .checked_mul(chunk_bits)
                .ok_or_else(|| WitnessError::InvalidDimensions {
                    namespace: JOLT_VM_NAMESPACE.name,
                    reason: "RA chunk shift overflow".to_owned(),
                })?;
        let k = checked_pow2_u128(chunk_bits)?;
        Ok(Self { shift, mask: k - 1 })
    }

    pub(crate) const fn chunk_usize(self, value: usize) -> usize {
        self.chunk_u128(value as u128)
    }

    pub(crate) const fn chunk_u128(self, value: u128) -> usize {
        ((value >> self.shift) & self.mask) as usize
    }
}

pub(crate) fn ra_family_selectors(
    ids: &[JoltCommittedPolynomial],
    index: impl Fn(JoltCommittedPolynomial) -> Option<usize>,
    chunks: usize,
    chunk_bits: usize,
) -> Result<Vec<RaChunkSelector>, WitnessError> {
    ids.iter()
        .copied()
        .map(|id| {
            let index = index(id).ok_or_else(|| WitnessError::InvalidWitnessData {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!("unexpected RA-family committed polynomial {id:?}"),
            })?;
            require_index(index, chunks)?;
            RaChunkSelector::new(index, chunks, chunk_bits)
        })
        .collect()
}

pub(crate) fn ra_chunk_to_u8(value: usize) -> Result<u8, WitnessError> {
    u8::try_from(value).map_err(|_| WitnessError::InvalidWitnessData {
        namespace: JOLT_VM_NAMESPACE.name,
        reason: format!("RA chunk index {value} exceeds the u8 chunk-index range"),
    })
}

pub(crate) fn bytecode_pc_for_row(
    row: &TraceRow,
    preprocessing: &JoltProgramPreprocessing,
) -> Option<usize> {
    if row_is_noop(row) {
        Some(0)
    } else {
        preprocessing.bytecode.get_pc(&row.instruction)
    }
}

impl<T: TraceSource + Clone> TraceBackedJoltVmWitness<'_, T> {
    pub(crate) fn evaluate_committed_ra<F: Field>(
        &self,
        id: JoltCommittedPolynomial,
        point: &[F],
    ) -> Result<F, WitnessError> {
        let chunk_bits = self.config.one_hot.committed_chunk_bits();
        let expected_vars = chunk_bits.checked_add(self.config.log_t).ok_or_else(|| {
            WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: "committed RA point length overflow".to_owned(),
            }
        })?;
        if point.len() != expected_vars {
            return Err(WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!(
                    "committed RA point has {} variables, expected {expected_vars}",
                    point.len()
                ),
            });
        }

        let kind = self.committed_stream_kind(id, self.ra_layout()?)?;
        let JoltVmCommittedStreamKind::OneHot(kind) = kind else {
            return Err(WitnessError::UnknownOracle {
                namespace: JOLT_VM_NAMESPACE.name,
            });
        };

        let (address_point, cycle_point) = point.split_at(chunk_bits);
        let address_eq = eq_evals_msb(address_point)?;
        let cycle_eq = eq_evals_msb(cycle_point)?;
        let mut trace = self.trace.trace.clone();
        let mut result = F::zero();
        for &cycle_weight in &cycle_eq {
            let value = trace.next_row().map_or_else(
                || Ok(kind.padding_value()),
                |row| kind.value_from_row(&row, self.preprocessing),
            )?;
            let Some(value) = value else {
                continue;
            };
            result += address_eq[value] * cycle_weight;
        }
        Ok(result)
    }

    pub(crate) fn materialize_instruction_ra<F: Field>(
        &self,
        index: usize,
    ) -> Result<Vec<F>, WitnessError> {
        let cycles = checked_pow2(self.config.log_t)?;
        let chunk_bits = self.config.one_hot.lookup_virtual_chunk_bits();
        let chunks = self.instruction_virtual_ra_count()?;
        require_index(index, chunks)?;
        let selector = RaChunkSelector::new(index, chunks, chunk_bits)?;
        let addresses = checked_pow2(chunk_bits)?;
        let mut values = vec![F::zero(); addresses * cycles];
        let mut trace = self.trace.trace.clone();

        for cycle in 0..cycles {
            let value = trace.next_row().map_or_else(
                || Ok(selector.chunk_u128(0)),
                |row| {
                    instruction_lookup_index::<RV64_XLEN>(&row)
                        .map(|lookup_index| selector.chunk_u128(lookup_index))
                        .map_err(|error| WitnessError::InvalidWitnessData {
                            namespace: JOLT_VM_NAMESPACE.name,
                            reason: error.to_string(),
                        })
                },
            )?;
            values[value * cycles + cycle] = F::one();
        }

        Ok(values)
    }

    pub(crate) fn evaluate_instruction_ra<F: Field>(
        &self,
        index: usize,
        point: &[F],
    ) -> Result<F, WitnessError> {
        let chunk_bits = self.config.one_hot.lookup_virtual_chunk_bits();
        let chunks = self.instruction_virtual_ra_count()?;
        require_index(index, chunks)?;
        let expected_vars = chunk_bits.checked_add(self.config.log_t).ok_or_else(|| {
            WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: "instruction virtual RA point length overflow".to_owned(),
            }
        })?;
        if point.len() != expected_vars {
            return Err(WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!(
                    "instruction virtual RA point has {} variables, expected {expected_vars}",
                    point.len()
                ),
            });
        }

        let selector = RaChunkSelector::new(index, chunks, chunk_bits)?;
        let (address_point, cycle_point) = point.split_at(chunk_bits);
        let address_eq = eq_evals_msb(address_point)?;
        let cycle_eq = eq_evals_msb(cycle_point)?;
        let mut trace = self.trace.trace.clone();
        let mut result = F::zero();
        for &cycle_weight in &cycle_eq {
            let value = trace.next_row().map_or_else(
                || Ok(selector.chunk_u128(0)),
                |row| {
                    instruction_lookup_index::<RV64_XLEN>(&row)
                        .map(|lookup_index| selector.chunk_u128(lookup_index))
                        .map_err(|error| WitnessError::InvalidWitnessData {
                            namespace: JOLT_VM_NAMESPACE.name,
                            reason: error.to_string(),
                        })
                },
            )?;
            result += address_eq[value] * cycle_weight;
        }
        Ok(result)
    }
}

pub const RA_FAMILY_MAX_INSTRUCTION_CHUNKS: usize = 32;
pub const RA_FAMILY_MAX_BYTECODE_CHUNKS: usize = 6;
pub const RA_FAMILY_MAX_RAM_CHUNKS: usize = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RaFamilyCycleIndices {
    pub instruction: [u8; RA_FAMILY_MAX_INSTRUCTION_CHUNKS],
    pub bytecode: [u8; RA_FAMILY_MAX_BYTECODE_CHUNKS],
    pub ram: [Option<u8>; RA_FAMILY_MAX_RAM_CHUNKS],
}

/// Opt-in fast path collecting every RA family's chunk index per cycle in a
/// single trace pass. Jolt-VM-specific accelerator: backends probe it and
/// fall back to per-polynomial streaming on `None`, which is what the
/// default impl returns for every other provider.
pub trait RaFamilyCycleIndexSource<F, N: WitnessNamespace>: crate::WitnessProvider<F, N> {
    fn try_collect_ra_family_cycle_indices(
        &self,
        _instruction_ids: &[N::CommittedId],
        _bytecode_ids: &[N::CommittedId],
        _ram_ids: &[N::CommittedId],
        _log_k_chunk: usize,
        _log_t: usize,
    ) -> Result<Option<Vec<RaFamilyCycleIndices>>, WitnessError> {
        Ok(None)
    }
}

impl<F: Field, T: TraceSource + Clone> RaFamilyCycleIndexSource<F, JoltVmNamespace>
    for TraceBackedJoltVmWitness<'_, T>
{
    fn try_collect_ra_family_cycle_indices(
        &self,
        instruction_ids: &[JoltCommittedPolynomial],
        bytecode_ids: &[JoltCommittedPolynomial],
        ram_ids: &[JoltCommittedPolynomial],
        log_k_chunk: usize,
        log_t: usize,
    ) -> Result<Option<Vec<RaFamilyCycleIndices>>, WitnessError> {
        let Some(rows) = self.trace.trace.rows() else {
            return Ok(None);
        };
        let expected_rows = checked_pow2(log_t)?;
        let layout = self.ra_layout()?;
        let committed_chunk_bits = self.config.one_hot.committed_chunk_bits();
        if committed_chunk_bits != log_k_chunk {
            return Err(WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!(
                    "RA fast-path log_k_chunk {log_k_chunk} differs from witness committed chunk bits {committed_chunk_bits}"
                ),
            });
        }
        if instruction_ids.len() > RA_FAMILY_MAX_INSTRUCTION_CHUNKS
            || bytecode_ids.len() > RA_FAMILY_MAX_BYTECODE_CHUNKS
            || ram_ids.len() > RA_FAMILY_MAX_RAM_CHUNKS
        {
            return Ok(None);
        }

        let instruction_selectors = ra_family_selectors(
            instruction_ids,
            |id| match id {
                JoltCommittedPolynomial::InstructionRa(index) => Some(index),
                _ => None,
            },
            layout.instruction(),
            committed_chunk_bits,
        )?;
        let bytecode_selectors = ra_family_selectors(
            bytecode_ids,
            |id| match id {
                JoltCommittedPolynomial::BytecodeRa(index) => Some(index),
                _ => None,
            },
            layout.bytecode(),
            committed_chunk_bits,
        )?;
        let ram_selectors = ra_family_selectors(
            ram_ids,
            |id| match id {
                JoltCommittedPolynomial::RamRa(index) => Some(index),
                _ => None,
            },
            layout.ram(),
            committed_chunk_bits,
        )?;

        let preprocessing = self.preprocessing;
        let indices = (0..expected_rows)
            .into_par_iter()
            .map(|cycle| {
                let mut row_indices = RaFamilyCycleIndices::default();
                if let Some(row) = rows.get(cycle) {
                    if !instruction_selectors.is_empty() {
                        let lookup_index =
                            instruction_lookup_index::<RV64_XLEN>(row).map_err(|error| {
                                WitnessError::InvalidWitnessData {
                                    namespace: JOLT_VM_NAMESPACE.name,
                                    reason: error.to_string(),
                                }
                            })?;
                        for (chunk, selector) in instruction_selectors.iter().copied().enumerate() {
                            row_indices.instruction[chunk] =
                                ra_chunk_to_u8(selector.chunk_u128(lookup_index))?;
                        }
                    }
                    if !bytecode_selectors.is_empty() {
                        let pc = bytecode_pc_for_row(row, preprocessing)
                            .ok_or_else(|| missing_pc_mapping(row))?;
                        for (chunk, selector) in bytecode_selectors.iter().copied().enumerate() {
                            row_indices.bytecode[chunk] = ra_chunk_to_u8(selector.chunk_usize(pc))?;
                        }
                    }
                    if !ram_selectors.is_empty() {
                        let address = ram_access_address(row.ram_access)
                            .and_then(|address| {
                                preprocessing
                                    .memory_layout
                                    .remap_word_address(address)
                                    .ok()
                                    .flatten()
                            })
                            .map(|address| address as usize);
                        if let Some(address) = address {
                            for (chunk, selector) in ram_selectors.iter().copied().enumerate() {
                                row_indices.ram[chunk] =
                                    Some(ra_chunk_to_u8(selector.chunk_usize(address))?);
                            }
                        }
                    }
                } else {
                    for (chunk, selector) in instruction_selectors.iter().copied().enumerate() {
                        row_indices.instruction[chunk] = ra_chunk_to_u8(selector.chunk_usize(0))?;
                    }
                    for (chunk, selector) in bytecode_selectors.iter().copied().enumerate() {
                        row_indices.bytecode[chunk] = ra_chunk_to_u8(selector.chunk_usize(0))?;
                    }
                }
                Ok(row_indices)
            })
            .collect::<Result<Vec<_>, WitnessError>>()?;
        Ok(Some(indices))
    }
}
