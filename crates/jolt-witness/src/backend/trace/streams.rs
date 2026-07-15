//! Chunked streaming and committed-polynomial materialization.

use super::*;

impl<T: TraceSource + Clone> TraceBackend<'_, T> {
    pub fn committed_stream(
        &self,
        polynomial: JoltCommittedPolynomial,
        chunk_size: usize,
    ) -> Result<JoltVmCommittedStream<'_, T>, WitnessError> {
        if chunk_size == 0 {
            return Err(WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: "stream chunk size must be nonzero".to_owned(),
            });
        }
        let layout = self.ra_layout()?;
        let kind = self.committed_stream_kind(polynomial, layout)?;
        Ok(JoltVmCommittedStream {
            kind,
            trace_rows: self.trace.trace.rows(),
            trace: self.trace.trace.clone(),
            emitted: 0,
            rows: self.committed_stream_rows(kind)?,
            chunk_size,
            preprocessing: self.preprocessing,
            trusted_advice: &self.trace.device.trusted_advice,
            untrusted_advice: &self.trace.device.untrusted_advice,
        })
    }

    pub fn committed_batch_stream(
        &self,
        polynomials: &[JoltCommittedPolynomial],
        chunk_size: usize,
    ) -> Result<JoltVmCommittedBatchStream<'_, T>, WitnessError> {
        if chunk_size == 0 {
            return Err(WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: "stream chunk size must be nonzero".to_owned(),
            });
        }
        let rows = checked_pow2(self.config.log_t)?;
        let layout = self.ra_layout()?;
        let plan = polynomials
            .iter()
            .copied()
            .map(|polynomial| {
                let kind = self.committed_stream_kind(polynomial, layout)?;
                Ok((polynomial, kind))
            })
            .collect::<Result<Vec<_>, WitnessError>>()?;
        if plan
            .iter()
            .any(|(_, kind)| matches!(kind, JoltVmCommittedStreamKind::Advice(_)))
        {
            return Err(WitnessError::UnsupportedView {
                view: "batched Jolt VM advice streams",
            });
        }
        Ok(JoltVmCommittedBatchStream {
            needs: JoltVmBatchNeeds::from_plan(&plan),
            plan,
            trace_rows: self.trace.trace.rows(),
            trace: self.trace.trace.clone(),
            pc_cache: PcLookupCache::default(),
            emitted: 0,
            rows,
            chunk_size,
            preprocessing: self.preprocessing,
        })
    }

    pub(crate) fn committed_stream_kind(
        &self,
        polynomial: JoltCommittedPolynomial,
        layout: JoltRaPolynomialLayout,
    ) -> Result<JoltVmCommittedStreamKind, WitnessError> {
        match polynomial {
            JoltCommittedPolynomial::RdInc => Ok(JoltVmCommittedStreamKind::Increment(
                JoltVmIncrementStreamKind::RdInc,
            )),
            JoltCommittedPolynomial::RamInc => Ok(JoltVmCommittedStreamKind::Increment(
                JoltVmIncrementStreamKind::RamInc,
            )),
            JoltCommittedPolynomial::InstructionRa(index) => {
                require_index(index, layout.instruction())?;
                Ok(JoltVmCommittedStreamKind::OneHot(
                    JoltVmOneHotStreamKind::Instruction(RaChunkSelector::new(
                        index,
                        layout.instruction(),
                        self.config.one_hot.committed_chunk_bits(),
                    )?),
                ))
            }
            JoltCommittedPolynomial::BytecodeRa(index) => {
                require_index(index, layout.bytecode())?;
                Ok(JoltVmCommittedStreamKind::OneHot(
                    JoltVmOneHotStreamKind::Bytecode(RaChunkSelector::new(
                        index,
                        layout.bytecode(),
                        self.config.one_hot.committed_chunk_bits(),
                    )?),
                ))
            }
            JoltCommittedPolynomial::RamRa(index) => {
                require_index(index, layout.ram())?;
                Ok(JoltVmCommittedStreamKind::OneHot(
                    JoltVmOneHotStreamKind::Ram(RaChunkSelector::new(
                        index,
                        layout.ram(),
                        self.config.one_hot.committed_chunk_bits(),
                    )?),
                ))
            }
            JoltCommittedPolynomial::TrustedAdvice => {
                self.advice_stream_kind(JoltVmAdviceStreamKind::Trusted)
            }
            JoltCommittedPolynomial::UntrustedAdvice => {
                self.advice_stream_kind(JoltVmAdviceStreamKind::Untrusted)
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

    pub(crate) fn advice_stream_kind(
        &self,
        kind: JoltVmAdviceStreamKind,
    ) -> Result<JoltVmCommittedStreamKind, WitnessError> {
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
        Ok(JoltVmCommittedStreamKind::Advice(kind))
    }

    pub(crate) fn committed_stream_rows(
        &self,
        kind: JoltVmCommittedStreamKind,
    ) -> Result<usize, WitnessError> {
        match kind {
            JoltVmCommittedStreamKind::Increment(_) | JoltVmCommittedStreamKind::OneHot(_) => {
                checked_pow2(self.config.log_t)
            }
            JoltVmCommittedStreamKind::Advice(kind) => Ok(kind.rows(self.preprocessing)),
        }
    }

    pub(crate) fn advice_bytes(&self, kind: JoltVmAdviceStreamKind) -> &[u8] {
        match kind {
            JoltVmAdviceStreamKind::Trusted => &self.trace.device.trusted_advice,
            JoltVmAdviceStreamKind::Untrusted => &self.trace.device.untrusted_advice,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JoltVmCommittedStreamKind {
    Increment(JoltVmIncrementStreamKind),
    OneHot(JoltVmOneHotStreamKind),
    Advice(JoltVmAdviceStreamKind),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JoltVmIncrementStreamKind {
    RdInc,
    RamInc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JoltVmOneHotStreamKind {
    Instruction(RaChunkSelector),
    Bytecode(RaChunkSelector),
    Ram(RaChunkSelector),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JoltVmAdviceStreamKind {
    Trusted,
    Untrusted,
}

#[derive(Clone, Debug)]
pub struct JoltVmCommittedStream<'a, T: TraceSource> {
    kind: JoltVmCommittedStreamKind,
    trace_rows: Option<&'a [TraceRow]>,
    trace: T,
    emitted: usize,
    rows: usize,
    chunk_size: usize,
    preprocessing: &'a JoltProgramPreprocessing,
    trusted_advice: &'a [u8],
    untrusted_advice: &'a [u8],
}

#[derive(Clone, Debug)]
pub struct JoltVmCommittedBatchStream<'a, T: TraceSource> {
    needs: JoltVmBatchNeeds,
    plan: Vec<(JoltCommittedPolynomial, JoltVmCommittedStreamKind)>,
    trace_rows: Option<&'a [TraceRow]>,
    trace: T,
    pc_cache: PcLookupCache,
    emitted: usize,
    rows: usize,
    chunk_size: usize,
    preprocessing: &'a JoltProgramPreprocessing,
}

pub(crate) enum JoltVmBatchBuffer {
    I128(Vec<i128>),
    OneHot(Vec<Option<usize>>),
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct JoltVmBatchNeeds {
    instruction: bool,
    bytecode: bool,
    ram: bool,
}

pub(crate) struct JoltVmBatchRow {
    rd_inc: i128,
    ram_inc: i128,
    lookup_index: u128,
    bytecode_pc: Option<usize>,
    ram_address: Option<usize>,
}

impl<F, T: TraceSource> PolynomialStream<F> for JoltVmCommittedStream<'_, T> {
    fn next_chunk(&mut self) -> Result<Option<PolynomialChunk<F>>, WitnessError> {
        if self.emitted >= self.rows {
            return Ok(None);
        }
        let end = self.emitted.saturating_add(self.chunk_size).min(self.rows);

        match self.kind {
            JoltVmCommittedStreamKind::Increment(kind) => {
                let mut values = Vec::with_capacity(end - self.emitted);
                if let Some(rows) = self.trace_rows {
                    while self.emitted < end {
                        let value = rows
                            .get(self.emitted)
                            .map_or(0, |row| kind.value_from_row(row));
                        values.push(value);
                        self.emitted += 1;
                    }
                } else {
                    while self.emitted < end {
                        let value = self
                            .trace
                            .next_row()
                            .map_or(0, |row| kind.value_from_row(&row));
                        values.push(value);
                        self.emitted += 1;
                    }
                }
                Ok(Some(PolynomialChunk::I128(values)))
            }
            JoltVmCommittedStreamKind::OneHot(kind) => {
                let mut values = Vec::with_capacity(end - self.emitted);
                if let Some(rows) = self.trace_rows {
                    while self.emitted < end {
                        let value = rows.get(self.emitted).map_or_else(
                            || Ok(kind.padding_value()),
                            |row| kind.value_from_row(row, self.preprocessing),
                        )?;
                        values.push(value);
                        self.emitted += 1;
                    }
                } else {
                    while self.emitted < end {
                        let value = self.trace.next_row().map_or_else(
                            || Ok(kind.padding_value()),
                            |row| kind.value_from_row(&row, self.preprocessing),
                        )?;
                        values.push(value);
                        self.emitted += 1;
                    }
                }
                Ok(Some(PolynomialChunk::OneHot(values)))
            }
            JoltVmCommittedStreamKind::Advice(kind) => {
                let bytes = kind.bytes(self.trusted_advice, self.untrusted_advice);
                let mut values = Vec::with_capacity(end - self.emitted);
                while self.emitted < end {
                    values.push(advice_word_le(bytes, self.emitted));
                    self.emitted += 1;
                }
                Ok(Some(PolynomialChunk::U64(values)))
            }
        }
    }
}

impl<F, T: TraceSource> PolynomialBatchStream<F, JoltCommittedPolynomial>
    for JoltVmCommittedBatchStream<'_, T>
{
    fn next_batch(
        &mut self,
    ) -> Result<Option<PolynomialBatchChunk<JoltCommittedPolynomial, F>>, WitnessError> {
        if self.emitted >= self.rows {
            return Ok(None);
        }
        let end = self.emitted.saturating_add(self.chunk_size).min(self.rows);
        let mut buffers = self
            .plan
            .iter()
            .map(|(_, kind)| match kind {
                JoltVmCommittedStreamKind::Increment(_) => {
                    JoltVmBatchBuffer::I128(Vec::with_capacity(end - self.emitted))
                }
                JoltVmCommittedStreamKind::OneHot(_) => {
                    JoltVmBatchBuffer::OneHot(Vec::with_capacity(end - self.emitted))
                }
                JoltVmCommittedStreamKind::Advice(_) => {
                    unreachable!("advice streams are rejected before batch construction")
                }
            })
            .collect::<Vec<_>>();

        while self.emitted < end {
            let owned_row;
            let row = if let Some(rows) = self.trace_rows {
                rows.get(self.emitted)
            } else {
                owned_row = self.trace.next_row();
                owned_row.as_ref()
            };
            let facts = row
                .map(|row| {
                    JoltVmBatchRow::new(row, self.preprocessing, self.needs, &mut self.pc_cache)
                })
                .transpose()?;
            for ((_, kind), buffer) in self.plan.iter().zip(&mut buffers) {
                match (kind, buffer) {
                    (
                        JoltVmCommittedStreamKind::Increment(kind),
                        JoltVmBatchBuffer::I128(values),
                    ) => {
                        values.push(
                            facts
                                .as_ref()
                                .map_or(0, |facts| kind.value_from_facts(facts)),
                        );
                    }
                    (
                        JoltVmCommittedStreamKind::OneHot(kind),
                        JoltVmBatchBuffer::OneHot(values),
                    ) => {
                        let value = facts.as_ref().map_or_else(
                            || Ok(kind.padding_value()),
                            |facts| Ok(kind.value_from_facts(facts)),
                        )?;
                        values.push(value);
                    }
                    (JoltVmCommittedStreamKind::Advice(_), _) => {
                        unreachable!("advice streams are rejected before batch construction")
                    }
                    _ => unreachable!("batch buffer kind must match committed stream kind"),
                }
            }
            self.emitted += 1;
        }

        let chunks = self
            .plan
            .iter()
            .map(|(id, _)| *id)
            .zip(buffers)
            .map(|(id, buffer)| {
                let chunk = match buffer {
                    JoltVmBatchBuffer::I128(values) => PolynomialChunk::I128(values),
                    JoltVmBatchBuffer::OneHot(values) => PolynomialChunk::OneHot(values),
                };
                (id, chunk)
            })
            .collect();
        Ok(Some(PolynomialBatchChunk::new(chunks)))
    }
}

impl JoltVmBatchNeeds {
    pub(crate) fn from_plan(plan: &[(JoltCommittedPolynomial, JoltVmCommittedStreamKind)]) -> Self {
        let mut needs = Self::default();
        for (_, kind) in plan {
            match kind {
                JoltVmCommittedStreamKind::OneHot(JoltVmOneHotStreamKind::Instruction(_)) => {
                    needs.instruction = true;
                }
                JoltVmCommittedStreamKind::OneHot(JoltVmOneHotStreamKind::Bytecode(_)) => {
                    needs.bytecode = true;
                }
                JoltVmCommittedStreamKind::OneHot(JoltVmOneHotStreamKind::Ram(_)) => {
                    needs.ram = true;
                }
                JoltVmCommittedStreamKind::Increment(_) | JoltVmCommittedStreamKind::Advice(_) => {}
            }
        }
        needs
    }
}

impl JoltVmBatchRow {
    pub(crate) fn new(
        row: &TraceRow,
        preprocessing: &JoltProgramPreprocessing,
        needs: JoltVmBatchNeeds,
        pc_cache: &mut PcLookupCache,
    ) -> Result<Self, WitnessError> {
        let lookup_index = if needs.instruction {
            instruction_lookup_index::<RV64_XLEN>(row).map_err(|error| {
                WitnessError::InvalidWitnessData {
                    label: JOLT_VM_LABEL,
                    reason: error.to_string(),
                }
            })?
        } else {
            0
        };
        let bytecode_pc = needs
            .bytecode
            .then(|| pc_cache.pc_for_row_optional(row, preprocessing))
            .flatten();
        let ram_address = if needs.ram {
            ram_access_address(row.ram_access)
                .and_then(|address| preprocessing.memory_layout.remap_word_address(address).ok())
                .flatten()
                .map(|address| address as usize)
        } else {
            None
        };
        Ok(Self {
            rd_inc: JoltVmIncrementStreamKind::RdInc.value_from_row(row),
            ram_inc: JoltVmIncrementStreamKind::RamInc.value_from_row(row),
            lookup_index,
            bytecode_pc,
            ram_address,
        })
    }
}

impl JoltVmIncrementStreamKind {
    pub(crate) const fn value_from_row(self, row: &TraceRow) -> i128 {
        match self {
            Self::RdInc => match row.registers.rd {
                Some(write) => write.post_value as i128 - write.pre_value as i128,
                None => 0,
            },
            Self::RamInc => match row.ram_access {
                RamAccess::Write(write) => write.post_value as i128 - write.pre_value as i128,
                RamAccess::Read(_) | RamAccess::NoOp => 0,
            },
        }
    }

    pub(crate) const fn value_from_facts(self, facts: &JoltVmBatchRow) -> i128 {
        match self {
            Self::RdInc => facts.rd_inc,
            Self::RamInc => facts.ram_inc,
        }
    }
}

impl JoltVmOneHotStreamKind {
    pub(crate) const fn padding_value(self) -> Option<usize> {
        match self {
            Self::Instruction(selector) | Self::Bytecode(selector) => Some(selector.chunk_usize(0)),
            Self::Ram(_) => None,
        }
    }

    pub(crate) fn value_from_row(
        self,
        row: &TraceRow,
        preprocessing: &JoltProgramPreprocessing,
    ) -> Result<Option<usize>, WitnessError> {
        match self {
            Self::Instruction(selector) => instruction_lookup_index::<RV64_XLEN>(row)
                .map(|index| Some(selector.chunk_u128(index)))
                .map_err(|error| WitnessError::InvalidWitnessData {
                    label: JOLT_VM_LABEL,
                    reason: error.to_string(),
                }),
            Self::Bytecode(selector) => Ok(preprocessing
                .bytecode
                .get_pc(&row.instruction)
                .map(|pc| selector.chunk_usize(pc))),
            Self::Ram(selector) => Ok(ram_access_address(row.ram_access)
                .and_then(|address| preprocessing.memory_layout.remap_word_address(address).ok())
                .flatten()
                .map(|address| selector.chunk_usize(address as usize))),
        }
    }

    pub(crate) fn value_from_facts(self, facts: &JoltVmBatchRow) -> Option<usize> {
        match self {
            Self::Instruction(selector) => Some(selector.chunk_u128(facts.lookup_index)),
            Self::Bytecode(selector) => facts.bytecode_pc.map(|pc| selector.chunk_usize(pc)),
            Self::Ram(selector) => facts
                .ram_address
                .map(|address| selector.chunk_usize(address)),
        }
    }
}

impl JoltVmAdviceStreamKind {
    pub(crate) const fn is_included(self, config: &JoltVmWitnessConfig) -> bool {
        match self {
            Self::Trusted => config.include_trusted_advice,
            Self::Untrusted => config.include_untrusted_advice,
        }
    }

    pub(crate) const fn max_bytes(self, preprocessing: &JoltProgramPreprocessing) -> usize {
        match self {
            Self::Trusted => preprocessing.memory_layout.max_trusted_advice_size as usize,
            Self::Untrusted => preprocessing.memory_layout.max_untrusted_advice_size as usize,
        }
    }

    pub(crate) fn rows(self, preprocessing: &JoltProgramPreprocessing) -> usize {
        Self::rows_from_max_bytes(self.max_bytes(preprocessing))
    }

    pub(crate) const fn bytes<'a>(self, trusted: &'a [u8], untrusted: &'a [u8]) -> &'a [u8] {
        match self {
            Self::Trusted => trusted,
            Self::Untrusted => untrusted,
        }
    }

    pub(crate) fn validate_len(
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

    pub(crate) fn rows_from_max_bytes(max_bytes: usize) -> usize {
        let words = max_bytes / 8;
        words.next_power_of_two().max(1)
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
        let mut stream = self.committed_stream(id, shape.dimensions.rows().max(1))?;
        let mut values = Vec::with_capacity(shape.dimensions.rows());
        while let Some(chunk) = stream.next_chunk()? {
            append_compact_chunk(&mut values, chunk)?;
        }
        if values.len() != shape.dimensions.rows() {
            return Err(WitnessError::InvalidWitnessData {
                label: JOLT_VM_LABEL,
                reason: format!(
                    "committed oracle {id:?} materialized {} rows, expected {}",
                    values.len(),
                    shape.dimensions.rows()
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
        if !shape.dimensions.rows().is_multiple_of(cycles) {
            return Err(WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: format!(
                    "committed oracle {id:?} has {} rows, not divisible by {cycles} cycles",
                    shape.dimensions.rows()
                ),
            });
        }
        let addresses = shape.dimensions.rows() / cycles;
        let mut stream = self.committed_stream(id, cycles.max(1))?;
        let mut values = vec![F::zero(); shape.dimensions.rows()];
        let mut cycle = 0usize;
        loop {
            let next: Option<PolynomialChunk<F>> = stream.next_chunk()?;
            let Some(chunk) = next else {
                break;
            };
            let PolynomialChunk::OneHot(chunk) = chunk else {
                return Err(WitnessError::InvalidWitnessData {
                    label: JOLT_VM_LABEL,
                    reason: format!("committed oracle {id:?} did not stream one-hot chunks"),
                });
            };
            for address in chunk {
                if let Some(address) = address {
                    if address >= addresses {
                        return Err(WitnessError::InvalidWitnessData {
                            label: JOLT_VM_LABEL,
                            reason: format!(
                                "committed oracle {id:?} streamed address {address}, beyond {addresses}"
                            ),
                        });
                    }
                    let index = address
                        .checked_mul(cycles)
                        .and_then(|base| base.checked_add(cycle))
                        .ok_or_else(|| WitnessError::InvalidDimensions {
                            label: JOLT_VM_LABEL,
                            reason: format!("committed oracle {id:?} index overflow"),
                        })?;
                    values[index] = F::one();
                }
                cycle += 1;
            }
        }
        if cycle != cycles {
            return Err(WitnessError::InvalidWitnessData {
                label: JOLT_VM_LABEL,
                reason: format!("committed oracle {id:?} streamed {cycle} rows, expected {cycles}"),
            });
        }
        Ok(values)
    }
}

pub(crate) fn append_compact_chunk<F: Field>(
    values: &mut Vec<F>,
    chunk: PolynomialChunk<F>,
) -> Result<(), WitnessError> {
    match chunk {
        PolynomialChunk::Dense(chunk) => values.extend(chunk),
        PolynomialChunk::Zeros(rows) => values.extend(std::iter::repeat_n(F::zero(), rows)),
        PolynomialChunk::U8(chunk) => {
            values.extend(chunk.into_iter().map(|value| F::from_u64(value as u64)));
        }
        PolynomialChunk::U16(chunk) => {
            values.extend(chunk.into_iter().map(|value| F::from_u64(value as u64)));
        }
        PolynomialChunk::U32(chunk) => {
            values.extend(chunk.into_iter().map(|value| F::from_u64(value as u64)));
        }
        PolynomialChunk::U64(chunk) => values.extend(chunk.into_iter().map(F::from_u64)),
        PolynomialChunk::I64(chunk) => values.extend(chunk.into_iter().map(F::from_i64)),
        PolynomialChunk::I128(chunk) => values.extend(chunk.into_iter().map(F::from_i128)),
        PolynomialChunk::OneHot(_) => {
            return Err(WitnessError::UnsupportedView {
                view: "one-hot chunk materialization as compact field values",
            });
        }
    }
    Ok(())
}
