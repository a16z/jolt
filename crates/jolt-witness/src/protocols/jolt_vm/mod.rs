use core::marker::PhantomData;

use jolt_claims::protocols::jolt::{
    formulas::{committed_openings, ra::JoltRaPolynomialLayout},
    JoltChallengeId, JoltCommittedPolynomial, JoltFormulaDimensions, JoltOneHotConfig,
    JoltOpeningId, JoltPublicId, JoltVirtualPolynomial,
};
use jolt_program::{
    execution::{JoltProgram, RamAccess, TraceOutput, TraceRow, TraceSource},
    lookup::instruction_lookup_index,
    preprocess::JoltProgramPreprocessing,
};

use crate::{
    MaterializationPolicy, NamespaceId, OracleDescriptor, OracleKind, OracleRef, OracleViewRequest,
    PolynomialBatchChunk, PolynomialBatchStream, PolynomialChunk, PolynomialEncoding,
    PolynomialStream, PolynomialView, RetentionHint, ViewRequirement, WitnessBuilder,
    WitnessDimensions, WitnessError, WitnessNamespace,
};

pub mod rv64;

#[cfg(feature = "field-inline")]
pub mod field_inline;

pub const JOLT_VM_NAMESPACE: NamespaceId = NamespaceId::new("jolt_vm");
pub const RV64_XLEN: usize = 64;
pub const RV64_LOOKUP_ADDRESS_BITS: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JoltVmNamespace {}

impl WitnessNamespace for JoltVmNamespace {
    type CommittedId = JoltCommittedPolynomial;
    type VirtualId = JoltVirtualPolynomial;
    type OpeningId = JoltOpeningId;
    type PublicId = JoltPublicId;
    type ChallengeId = JoltChallengeId;

    const ID: NamespaceId = JOLT_VM_NAMESPACE;
}

#[derive(Clone, Debug, Default)]
pub struct JoltVmWitness;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JoltVmWitnessConfig {
    pub retain_trace_rows: bool,
    pub log_t: usize,
    pub ram_k: usize,
    pub one_hot: JoltOneHotConfig,
    pub include_trusted_advice: bool,
    pub include_untrusted_advice: bool,
}

impl Default for JoltVmWitnessConfig {
    fn default() -> Self {
        Self::new(
            0,
            1,
            JoltOneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 16,
            },
        )
    }
}

impl JoltVmWitnessConfig {
    pub fn new(log_t: usize, ram_k: usize, one_hot: JoltOneHotConfig) -> Self {
        Self {
            retain_trace_rows: false,
            log_t,
            ram_k,
            one_hot,
            include_trusted_advice: false,
            include_untrusted_advice: false,
        }
    }

    pub const fn retain_trace_rows(mut self, retain_trace_rows: bool) -> Self {
        self.retain_trace_rows = retain_trace_rows;
        self
    }

    pub const fn with_log_t(mut self, log_t: usize) -> Self {
        self.log_t = log_t;
        self
    }

    pub const fn include_trusted_advice(mut self, include_trusted_advice: bool) -> Self {
        self.include_trusted_advice = include_trusted_advice;
        self
    }

    pub const fn include_untrusted_advice(mut self, include_untrusted_advice: bool) -> Self {
        self.include_untrusted_advice = include_untrusted_advice;
        self
    }
}

pub struct JoltVmWitnessInputs<'a, T: TraceSource> {
    pub program: &'a JoltProgram,
    pub preprocessing: &'a JoltProgramPreprocessing,
    pub trace: TraceOutput<T>,
}

impl<'a, T: TraceSource> JoltVmWitnessInputs<'a, T> {
    pub const fn new(
        program: &'a JoltProgram,
        preprocessing: &'a JoltProgramPreprocessing,
        trace: TraceOutput<T>,
    ) -> Self {
        Self {
            program,
            preprocessing,
            trace,
        }
    }
}

#[derive(Clone, Debug)]
pub struct JoltVmWitnessBuilder<T> {
    _trace: PhantomData<T>,
}

impl<T> Default for JoltVmWitnessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> JoltVmWitnessBuilder<T> {
    pub const fn new() -> Self {
        Self {
            _trace: PhantomData,
        }
    }
}

pub struct TraceBackedJoltVmWitness<'a, T: TraceSource> {
    pub config: JoltVmWitnessConfig,
    pub program: &'a JoltProgram,
    pub preprocessing: &'a JoltProgramPreprocessing,
    pub trace: TraceOutput<T>,
}

impl<'a, T: TraceSource> TraceBackedJoltVmWitness<'a, T> {
    pub fn new(config: JoltVmWitnessConfig, inputs: JoltVmWitnessInputs<'a, T>) -> Self {
        Self {
            config,
            program: inputs.program,
            preprocessing: inputs.preprocessing,
            trace: inputs.trace,
        }
    }

    pub fn committed_polynomial_order(&self) -> Result<Vec<JoltCommittedPolynomial>, WitnessError> {
        let mut order = committed_openings::proof_commitment_order(self.ra_layout()?);
        if self.config.include_trusted_advice {
            order.push(JoltCommittedPolynomial::TrustedAdvice);
        }
        if self.config.include_untrusted_advice {
            order.push(JoltCommittedPolynomial::UntrustedAdvice);
        }
        Ok(order)
    }

    fn ra_layout(&self) -> Result<JoltRaPolynomialLayout, WitnessError> {
        self.formula_dimensions()
            .map(|dimensions| dimensions.ra_layout)
    }

    fn formula_dimensions(&self) -> Result<JoltFormulaDimensions, WitnessError> {
        let dimensions = self.config.one_hot.dimensions(
            self.config.log_t,
            RV64_LOOKUP_ADDRESS_BITS,
            self.preprocessing.bytecode.code_size,
            self.config.ram_k,
        );
        JoltFormulaDimensions::try_from(dimensions).map_err(|error| {
            WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: error.to_string(),
            }
        })
    }

    fn trace_dimensions(&self) -> Result<WitnessDimensions, WitnessError> {
        let rows = checked_pow2(self.config.log_t)?;
        Ok(WitnessDimensions::new(rows, self.config.log_t))
    }

    fn one_hot_dimensions(&self) -> Result<WitnessDimensions, WitnessError> {
        let log_rows = self
            .config
            .log_t
            .checked_add(self.config.one_hot.committed_chunk_bits())
            .ok_or_else(|| WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: "one-hot committed rows overflow".to_owned(),
            })?;
        let rows = checked_pow2(log_rows)?;
        Ok(WitnessDimensions::new(rows, log_rows))
    }

    fn advice_dimensions(words: usize) -> WitnessDimensions {
        let rows = words.next_power_of_two().max(1);
        WitnessDimensions::new(rows, rows.ilog2() as usize)
    }
}

impl<T: TraceSource + Clone> TraceBackedJoltVmWitness<'_, T> {
    pub fn committed_stream(
        &self,
        polynomial: JoltCommittedPolynomial,
        chunk_size: usize,
    ) -> Result<JoltVmCommittedStream<'_, T>, WitnessError> {
        if chunk_size == 0 {
            return Err(WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: "stream chunk size must be nonzero".to_owned(),
            });
        }
        let rows = checked_pow2(self.config.log_t)?;
        let layout = self.ra_layout()?;
        let kind = self.committed_stream_kind(polynomial, layout)?;
        Ok(JoltVmCommittedStream {
            kind,
            trace: self.trace.trace.clone(),
            emitted: 0,
            rows,
            chunk_size,
            preprocessing: self.preprocessing,
        })
    }

    pub fn committed_batch_stream(
        &self,
        polynomials: &[JoltCommittedPolynomial],
        chunk_size: usize,
    ) -> Result<JoltVmCommittedBatchStream<'_, T>, WitnessError> {
        if chunk_size == 0 {
            return Err(WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
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
        Ok(JoltVmCommittedBatchStream {
            needs: JoltVmBatchNeeds::from_plan(&plan),
            plan,
            trace: self.trace.trace.clone(),
            emitted: 0,
            rows,
            chunk_size,
            preprocessing: self.preprocessing,
        })
    }

    fn committed_stream_kind(
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
            JoltCommittedPolynomial::TrustedAdvice | JoltCommittedPolynomial::UntrustedAdvice => {
                Err(WitnessError::UnsupportedFrontier {
                    frontier: "Jolt VM committed stream",
                })
            }
        }
    }
}

#[cfg(feature = "field-inline")]
impl<'a, T: TraceSource + Clone> TraceBackedJoltVmWitness<'a, T> {
    pub fn field_inline_witness<'w>(
        &'w self,
    ) -> Result<field_inline::TraceBackedFieldInlineWitness<'w, 'a, T>, WitnessError> {
        field_inline::TraceBackedFieldInlineWitness::new(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JoltVmCommittedStreamKind {
    Increment(JoltVmIncrementStreamKind),
    OneHot(JoltVmOneHotStreamKind),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JoltVmIncrementStreamKind {
    RdInc,
    RamInc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JoltVmOneHotStreamKind {
    Instruction(RaChunkSelector),
    Bytecode(RaChunkSelector),
    Ram(RaChunkSelector),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RaChunkSelector {
    shift: usize,
    mask: u128,
}

#[derive(Clone, Debug)]
pub struct JoltVmCommittedStream<'a, T: TraceSource> {
    kind: JoltVmCommittedStreamKind,
    trace: T,
    emitted: usize,
    rows: usize,
    chunk_size: usize,
    preprocessing: &'a JoltProgramPreprocessing,
}

#[derive(Clone, Debug)]
pub struct JoltVmCommittedBatchStream<'a, T: TraceSource> {
    needs: JoltVmBatchNeeds,
    plan: Vec<(JoltCommittedPolynomial, JoltVmCommittedStreamKind)>,
    trace: T,
    emitted: usize,
    rows: usize,
    chunk_size: usize,
    preprocessing: &'a JoltProgramPreprocessing,
}

enum JoltVmBatchBuffer {
    I128(Vec<i128>),
    OneHot(Vec<Option<usize>>),
}

#[derive(Clone, Copy, Debug, Default)]
struct JoltVmBatchNeeds {
    instruction: bool,
    bytecode: bool,
    ram: bool,
}

struct JoltVmBatchRow {
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
                while self.emitted < end {
                    let value = self
                        .trace
                        .next_row()
                        .map_or(0, |row| kind.value_from_row(&row));
                    values.push(value);
                    self.emitted += 1;
                }
                Ok(Some(PolynomialChunk::I128(values)))
            }
            JoltVmCommittedStreamKind::OneHot(kind) => {
                let mut values = Vec::with_capacity(end - self.emitted);
                while self.emitted < end {
                    let value = self.trace.next_row().map_or_else(
                        || Ok(kind.padding_value()),
                        |row| kind.value_from_row(&row, self.preprocessing),
                    )?;
                    values.push(value);
                    self.emitted += 1;
                }
                Ok(Some(PolynomialChunk::OneHot(values)))
            }
        }
    }
}

impl<F, T: TraceSource> PolynomialBatchStream<F, JoltVmNamespace>
    for JoltVmCommittedBatchStream<'_, T>
{
    fn next_batch(
        &mut self,
    ) -> Result<Option<PolynomialBatchChunk<JoltVmNamespace, F>>, WitnessError> {
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
            })
            .collect::<Vec<_>>();

        while self.emitted < end {
            let row = self.trace.next_row();
            let facts = row
                .as_ref()
                .map(|row| JoltVmBatchRow::new(row, self.preprocessing, self.needs))
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
    fn from_plan(plan: &[(JoltCommittedPolynomial, JoltVmCommittedStreamKind)]) -> Self {
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
                JoltVmCommittedStreamKind::Increment(_) => {}
            }
        }
        needs
    }
}

impl JoltVmBatchRow {
    fn new(
        row: &TraceRow,
        preprocessing: &JoltProgramPreprocessing,
        needs: JoltVmBatchNeeds,
    ) -> Result<Self, WitnessError> {
        let lookup_index = if needs.instruction {
            instruction_lookup_index::<RV64_XLEN>(row).map_err(|error| {
                WitnessError::InvalidWitnessData {
                    namespace: JOLT_VM_NAMESPACE.name,
                    reason: error.to_string(),
                }
            })?
        } else {
            0
        };
        let bytecode_pc = needs
            .bytecode
            .then(|| preprocessing.bytecode.get_pc(&row.instruction))
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
    const fn value_from_row(self, row: &TraceRow) -> i128 {
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

    const fn value_from_facts(self, facts: &JoltVmBatchRow) -> i128 {
        match self {
            Self::RdInc => facts.rd_inc,
            Self::RamInc => facts.ram_inc,
        }
    }
}

impl JoltVmOneHotStreamKind {
    const fn padding_value(self) -> Option<usize> {
        match self {
            Self::Instruction(selector) | Self::Bytecode(selector) => Some(selector.chunk_usize(0)),
            Self::Ram(_) => None,
        }
    }

    fn value_from_row(
        self,
        row: &TraceRow,
        preprocessing: &JoltProgramPreprocessing,
    ) -> Result<Option<usize>, WitnessError> {
        match self {
            Self::Instruction(selector) => instruction_lookup_index::<RV64_XLEN>(row)
                .map(|index| Some(selector.chunk_u128(index)))
                .map_err(|error| WitnessError::InvalidWitnessData {
                    namespace: JOLT_VM_NAMESPACE.name,
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

    fn value_from_facts(self, facts: &JoltVmBatchRow) -> Option<usize> {
        match self {
            Self::Instruction(selector) => Some(selector.chunk_u128(facts.lookup_index)),
            Self::Bytecode(selector) => facts.bytecode_pc.map(|pc| selector.chunk_usize(pc)),
            Self::Ram(selector) => facts
                .ram_address
                .map(|address| selector.chunk_usize(address)),
        }
    }
}

impl RaChunkSelector {
    fn new(index: usize, chunks: usize, chunk_bits: usize) -> Result<Self, WitnessError> {
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

    const fn chunk_usize(self, value: usize) -> usize {
        self.chunk_u128(value as u128)
    }

    const fn chunk_u128(self, value: u128) -> usize {
        ((value >> self.shift) & self.mask) as usize
    }
}

const fn ram_access_address(access: RamAccess) -> Option<u64> {
    match access {
        RamAccess::Read(read) => Some(read.address),
        RamAccess::Write(write) => Some(write.address),
        RamAccess::NoOp => None,
    }
}

impl<F, T: TraceSource + Clone> WitnessBuilder<F> for JoltVmWitnessBuilder<T> {
    type Config = JoltVmWitnessConfig;
    type Inputs<'a>
        = JoltVmWitnessInputs<'a, T>
    where
        Self: 'a,
        F: 'a;
    type Namespace = JoltVmNamespace;
    type Witness<'a>
        = TraceBackedJoltVmWitness<'a, T>
    where
        Self: 'a,
        F: 'a;

    fn build<'a>(
        &mut self,
        config: &Self::Config,
        inputs: Self::Inputs<'a>,
    ) -> Result<Self::Witness<'a>, WitnessError>
    where
        Self: 'a,
        F: 'a,
    {
        Ok(TraceBackedJoltVmWitness::new(config.clone(), inputs))
    }
}

impl<F, T: TraceSource + Clone> crate::WitnessProvider<F, JoltVmNamespace>
    for TraceBackedJoltVmWitness<'_, T>
{
    fn describe_oracle(
        &self,
        oracle: OracleRef<JoltVmNamespace>,
    ) -> Result<OracleDescriptor<JoltVmNamespace>, WitnessError> {
        let dimensions = match oracle.kind {
            OracleKind::Committed(
                JoltCommittedPolynomial::RdInc | JoltCommittedPolynomial::RamInc,
            ) => self.trace_dimensions()?,
            OracleKind::Committed(JoltCommittedPolynomial::InstructionRa(index)) => {
                require_index(index, self.ra_layout()?.instruction())?;
                self.one_hot_dimensions()?
            }
            OracleKind::Committed(JoltCommittedPolynomial::BytecodeRa(index)) => {
                require_index(index, self.ra_layout()?.bytecode())?;
                self.one_hot_dimensions()?
            }
            OracleKind::Committed(JoltCommittedPolynomial::RamRa(index)) => {
                require_index(index, self.ra_layout()?.ram())?;
                self.one_hot_dimensions()?
            }
            OracleKind::Committed(JoltCommittedPolynomial::TrustedAdvice) => {
                if !self.config.include_trusted_advice {
                    return Err(WitnessError::UnknownOracle {
                        namespace: JOLT_VM_NAMESPACE.name,
                    });
                }
                Self::advice_dimensions(
                    self.preprocessing.memory_layout.max_trusted_advice_size as usize / 8,
                )
            }
            OracleKind::Committed(JoltCommittedPolynomial::UntrustedAdvice) => {
                if !self.config.include_untrusted_advice {
                    return Err(WitnessError::UnknownOracle {
                        namespace: JOLT_VM_NAMESPACE.name,
                    });
                }
                Self::advice_dimensions(
                    self.preprocessing.memory_layout.max_untrusted_advice_size as usize / 8,
                )
            }
            OracleKind::Virtual(_) => {
                return Err(WitnessError::UnsupportedFrontier {
                    frontier: "Jolt VM virtual oracle descriptors",
                });
            }
        };
        Ok(OracleDescriptor::new(
            oracle,
            dimensions,
            committed_encoding(oracle.kind),
        ))
    }

    fn view_requirements(
        &self,
        oracle: OracleRef<JoltVmNamespace>,
    ) -> Result<Vec<ViewRequirement<JoltVmNamespace>>, WitnessError> {
        let descriptor =
            <Self as crate::WitnessProvider<F, JoltVmNamespace>>::describe_oracle(self, oracle)?;
        let retention = match oracle.kind {
            OracleKind::Committed(
                JoltCommittedPolynomial::TrustedAdvice | JoltCommittedPolynomial::UntrustedAdvice,
            ) => RetentionHint::ThroughBlindFold,
            _ => RetentionHint::ThroughStage8,
        };
        Ok(vec![ViewRequirement::new(
            descriptor.reference,
            descriptor.encoding,
            MaterializationPolicy::BackendChoice,
            retention,
        )])
    }

    fn oracle_view(
        &self,
        _request: OracleViewRequest<JoltVmNamespace>,
    ) -> Result<PolynomialView<'_, F, JoltVmNamespace>, WitnessError> {
        Err(WitnessError::UnsupportedFrontier {
            frontier: "Jolt VM oracle views",
        })
    }

    fn committed_stream<'a>(
        &'a self,
        id: JoltCommittedPolynomial,
        chunk_size: usize,
    ) -> Result<Box<dyn PolynomialStream<F> + 'a>, WitnessError>
    where
        F: 'a,
        JoltVmNamespace: 'a,
    {
        Ok(Box::new(TraceBackedJoltVmWitness::committed_stream(
            self, id, chunk_size,
        )?))
    }

    fn committed_batch_stream<'a>(
        &'a self,
        ids: &'a [JoltCommittedPolynomial],
        chunk_size: usize,
    ) -> Result<Box<dyn PolynomialBatchStream<F, JoltVmNamespace> + 'a>, WitnessError>
    where
        F: 'a,
        JoltVmNamespace: 'a,
        JoltCommittedPolynomial: 'a,
    {
        Ok(Box::new(TraceBackedJoltVmWitness::committed_batch_stream(
            self, ids, chunk_size,
        )?))
    }
}

impl<F, T: TraceSource + Clone> crate::CommittedWitnessProvider<F, JoltVmNamespace>
    for TraceBackedJoltVmWitness<'_, T>
{
    fn committed_oracle_order(&self) -> Result<Vec<JoltCommittedPolynomial>, WitnessError> {
        self.committed_polynomial_order()
    }
}

fn checked_pow2(log_rows: usize) -> Result<usize, WitnessError> {
    if log_rows >= usize::BITS as usize {
        return Err(WitnessError::InvalidDimensions {
            namespace: JOLT_VM_NAMESPACE.name,
            reason: "witness row count overflow".to_owned(),
        });
    }
    1_usize
        .checked_shl(log_rows as u32)
        .ok_or_else(|| WitnessError::InvalidDimensions {
            namespace: JOLT_VM_NAMESPACE.name,
            reason: "witness row count overflow".to_owned(),
        })
}

fn checked_pow2_u128(log_rows: usize) -> Result<u128, WitnessError> {
    if log_rows >= u128::BITS as usize {
        return Err(WitnessError::InvalidDimensions {
            namespace: JOLT_VM_NAMESPACE.name,
            reason: "witness row count overflow".to_owned(),
        });
    }
    1_u128
        .checked_shl(log_rows as u32)
        .ok_or_else(|| WitnessError::InvalidDimensions {
            namespace: JOLT_VM_NAMESPACE.name,
            reason: "witness row count overflow".to_owned(),
        })
}

fn require_index(index: usize, len: usize) -> Result<(), WitnessError> {
    if index < len {
        Ok(())
    } else {
        Err(WitnessError::UnknownOracle {
            namespace: JOLT_VM_NAMESPACE.name,
        })
    }
}

const fn committed_encoding(
    kind: OracleKind<JoltCommittedPolynomial, JoltVirtualPolynomial>,
) -> PolynomialEncoding {
    match kind {
        OracleKind::Committed(
            JoltCommittedPolynomial::InstructionRa(_)
            | JoltCommittedPolynomial::BytecodeRa(_)
            | JoltCommittedPolynomial::RamRa(_),
        ) => PolynomialEncoding::OneHot,
        OracleKind::Committed(_) => PolynomialEncoding::Compact,
        OracleKind::Virtual(_) => PolynomialEncoding::Derived,
    }
}

#[cfg(test)]
mod tests {
    use common::{
        constants::RAM_START_ADDRESS,
        jolt_device::{MemoryConfig, MemoryLayout},
    };
    use jolt_program::{
        execution::{
            JoltProgram, OwnedTrace, RamAccess, RamRead, RamWrite, RegisterRead, RegisterState,
            RegisterWrite, TraceOutput, TraceRow,
        },
        preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing},
    };
    use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands, RV64IMAC_JOLT};

    use super::*;
    use crate::{PolynomialChunk, PolynomialStream, WitnessBuilder, WitnessProvider};

    fn preprocessing() -> JoltProgramPreprocessing {
        let bytecode = BytecodePreprocessing {
            code_size: 32,
            ..Default::default()
        };
        let mut preprocessing = JoltProgramPreprocessing {
            bytecode,
            ram: RAMPreprocessing::default(),
            memory_layout: Default::default(),
            max_padded_trace_length: 16,
        };
        preprocessing.memory_layout.max_trusted_advice_size = 64;
        preprocessing.memory_layout.max_untrusted_advice_size = 128;
        preprocessing
    }

    fn preprocessing_with_bytecode(bytecode: BytecodePreprocessing) -> JoltProgramPreprocessing {
        JoltProgramPreprocessing {
            bytecode,
            ..preprocessing()
        }
    }

    fn preprocessing_with_memory_layout(memory_layout: MemoryLayout) -> JoltProgramPreprocessing {
        JoltProgramPreprocessing {
            memory_layout,
            ..preprocessing()
        }
    }

    fn config() -> JoltVmWitnessConfig {
        JoltVmWitnessConfig::new(
            4,
            64,
            JoltOneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 16,
            },
        )
    }

    fn trace_output() -> TraceOutput<OwnedTrace> {
        TraceOutput::new(OwnedTrace::default(), Default::default(), None)
    }

    fn trace_output_with_rows(rows: Vec<TraceRow>) -> TraceOutput<OwnedTrace> {
        TraceOutput::new(OwnedTrace::new(rows), Default::default(), None)
    }

    fn instruction(address: usize) -> JoltInstructionRow {
        JoltInstructionRow {
            instruction_kind: JoltInstructionKind::ADDI,
            address,
            operands: NormalizedOperands {
                rd: Some(1),
                rs1: Some(2),
                rs2: None,
                imm: 3,
            },
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        }
    }

    fn compact_memory_layout() -> MemoryLayout {
        MemoryLayout::new(&MemoryConfig {
            max_input_size: 0,
            max_trusted_advice_size: 0,
            max_untrusted_advice_size: 0,
            max_output_size: 0,
            stack_size: 0,
            heap_size: 0,
            program_size: Some(64),
        })
    }

    fn describe(
        witness: &TraceBackedJoltVmWitness<'_, OwnedTrace>,
        oracle: OracleRef<JoltVmNamespace>,
    ) -> Result<OracleDescriptor<JoltVmNamespace>, WitnessError> {
        <TraceBackedJoltVmWitness<'_, OwnedTrace> as WitnessProvider<
            u64,
            JoltVmNamespace,
        >>::describe_oracle(witness, oracle)
    }

    #[test]
    fn builder_keeps_jolt_program_execution_boundary() {
        let program = JoltProgram::default();
        let preprocessing = preprocessing();
        let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output());
        let config = config().retain_trace_rows(true);
        let mut builder = JoltVmWitnessBuilder::<OwnedTrace>::new();

        let result = <JoltVmWitnessBuilder<OwnedTrace> as WitnessBuilder<u64>>::build(
            &mut builder,
            &config,
            inputs,
        );
        let witness = match result {
            Ok(witness) => witness,
            Err(error) => {
                assert_eq!(
                    error,
                    WitnessError::UnsupportedFrontier {
                        frontier: "builder should not fail",
                    }
                );
                return;
            }
        };

        assert_eq!(
            <TraceBackedJoltVmWitness<'_, OwnedTrace> as WitnessProvider<
                u64,
                JoltVmNamespace,
            >>::namespace(&witness),
            JOLT_VM_NAMESPACE
        );
        assert_eq!(witness.config, config);
        assert_eq!(witness.program.elf_bytes(), program.elf_bytes());
        assert_eq!(
            witness.preprocessing.max_padded_trace_length,
            preprocessing.max_padded_trace_length
        );
    }

    #[test]
    fn committed_polynomial_order_uses_proof_payload_order() {
        let program = JoltProgram::default();
        let preprocessing = preprocessing();
        let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output());
        let witness = TraceBackedJoltVmWitness::new(
            config()
                .include_trusted_advice(true)
                .include_untrusted_advice(true),
            inputs,
        );
        let mut expected = vec![
            JoltCommittedPolynomial::RdInc,
            JoltCommittedPolynomial::RamInc,
        ];
        expected.extend((0..32).map(JoltCommittedPolynomial::InstructionRa));
        expected.extend((0..2).map(JoltCommittedPolynomial::RamRa));
        expected.extend((0..2).map(JoltCommittedPolynomial::BytecodeRa));
        expected.push(JoltCommittedPolynomial::TrustedAdvice);
        expected.push(JoltCommittedPolynomial::UntrustedAdvice);

        assert_eq!(witness.committed_polynomial_order(), Ok(expected));
    }

    #[test]
    fn committed_oracle_descriptors_report_dimensions_and_encoding() {
        let program = JoltProgram::default();
        let preprocessing = preprocessing();
        let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output());
        let witness = TraceBackedJoltVmWitness::new(config().include_trusted_advice(true), inputs);

        assert_eq!(
            describe(
                &witness,
                OracleRef::committed(JoltCommittedPolynomial::RamInc)
            ),
            Ok(OracleDescriptor::new(
                OracleRef::committed(JoltCommittedPolynomial::RamInc),
                WitnessDimensions::new(16, 4),
                PolynomialEncoding::Compact,
            ))
        );
        assert_eq!(
            describe(
                &witness,
                OracleRef::committed(JoltCommittedPolynomial::InstructionRa(0)),
            ),
            Ok(OracleDescriptor::new(
                OracleRef::committed(JoltCommittedPolynomial::InstructionRa(0)),
                WitnessDimensions::new(256, 8),
                PolynomialEncoding::OneHot,
            ))
        );
        assert_eq!(
            describe(
                &witness,
                OracleRef::committed(JoltCommittedPolynomial::TrustedAdvice),
            ),
            Ok(OracleDescriptor::new(
                OracleRef::committed(JoltCommittedPolynomial::TrustedAdvice),
                WitnessDimensions::new(8, 3),
                PolynomialEncoding::Compact,
            ))
        );
    }

    #[test]
    fn descriptors_reject_disabled_advice_and_out_of_range_ra() {
        let program = JoltProgram::default();
        let preprocessing = preprocessing();
        let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output());
        let witness = TraceBackedJoltVmWitness::new(config(), inputs);

        assert_eq!(
            describe(
                &witness,
                OracleRef::committed(JoltCommittedPolynomial::TrustedAdvice),
            ),
            Err(WitnessError::UnknownOracle {
                namespace: JOLT_VM_NAMESPACE.name,
            })
        );
        assert_eq!(
            describe(
                &witness,
                OracleRef::committed(JoltCommittedPolynomial::BytecodeRa(2)),
            ),
            Err(WitnessError::UnknownOracle {
                namespace: JOLT_VM_NAMESPACE.name,
            })
        );
    }

    #[test]
    fn rd_inc_streams_register_write_deltas_and_padding() {
        let program = JoltProgram::default();
        let preprocessing = preprocessing();
        let rows = vec![
            TraceRow {
                registers: RegisterState {
                    rd: Some(RegisterWrite {
                        register: 1,
                        pre_value: 10,
                        post_value: 4,
                    }),
                    ..Default::default()
                },
                ..Default::default()
            },
            TraceRow {
                registers: RegisterState {
                    rd: Some(RegisterWrite {
                        register: 2,
                        pre_value: 2,
                        post_value: 11,
                    }),
                    ..Default::default()
                },
                ..Default::default()
            },
        ];
        let inputs =
            JoltVmWitnessInputs::new(&program, &preprocessing, trace_output_with_rows(rows));
        let witness = TraceBackedJoltVmWitness::new(config().with_log_t(2), inputs);
        let stream_result = witness.committed_stream(JoltCommittedPolynomial::RdInc, 3);
        let mut stream = match stream_result {
            Ok(stream) => stream,
            Err(error) => {
                assert_eq!(
                    error,
                    WitnessError::UnsupportedFrontier {
                        frontier: "stream should be supported",
                    }
                );
                return;
            }
        };

        assert_eq!(
            stream.next_chunk(),
            Ok(Some(PolynomialChunk::<i128>::I128(vec![-6, 9, 0])))
        );
        assert_eq!(
            stream.next_chunk(),
            Ok(Some(PolynomialChunk::<i128>::I128(vec![0])))
        );
        assert_eq!(stream.next_chunk(), Ok(None::<PolynomialChunk<i128>>));
    }

    #[test]
    fn ram_inc_streams_write_deltas_only() {
        let program = JoltProgram::default();
        let preprocessing = preprocessing();
        let rows = vec![
            TraceRow {
                ram_access: RamAccess::Write(RamWrite {
                    address: 10,
                    pre_value: 5,
                    post_value: 12,
                }),
                ..Default::default()
            },
            TraceRow {
                ram_access: RamAccess::Read(RamRead {
                    address: 10,
                    value: 12,
                }),
                ..Default::default()
            },
        ];
        let inputs =
            JoltVmWitnessInputs::new(&program, &preprocessing, trace_output_with_rows(rows));
        let witness = TraceBackedJoltVmWitness::new(config().with_log_t(2), inputs);
        let stream_result = witness.committed_stream(JoltCommittedPolynomial::RamInc, 2);
        let mut stream = match stream_result {
            Ok(stream) => stream,
            Err(error) => {
                assert_eq!(
                    error,
                    WitnessError::UnsupportedFrontier {
                        frontier: "stream should be supported",
                    }
                );
                return;
            }
        };

        assert_eq!(
            stream.next_chunk(),
            Ok(Some(PolynomialChunk::<i128>::I128(vec![7, 0])))
        );
        assert_eq!(
            stream.next_chunk(),
            Ok(Some(PolynomialChunk::<i128>::I128(vec![0, 0])))
        );
        assert_eq!(stream.next_chunk(), Ok(None::<PolynomialChunk<i128>>));
    }

    #[test]
    fn bytecode_ra_streams_pc_chunks_and_noop_padding() {
        let program = JoltProgram::default();
        let first = instruction(RAM_START_ADDRESS as usize);
        let second = instruction(RAM_START_ADDRESS as usize + 4);
        let bytecode_result = BytecodePreprocessing::preprocess(
            vec![first, second],
            RAM_START_ADDRESS,
            RV64IMAC_JOLT,
        );
        assert!(
            bytecode_result.is_ok(),
            "bytecode preprocessing failed: {bytecode_result:?}"
        );
        let Ok(bytecode) = bytecode_result else {
            return;
        };
        let preprocessing = preprocessing_with_bytecode(bytecode);
        let rows = vec![
            TraceRow {
                instruction: first,
                ..Default::default()
            },
            TraceRow {
                instruction: second,
                ..Default::default()
            },
        ];
        let inputs =
            JoltVmWitnessInputs::new(&program, &preprocessing, trace_output_with_rows(rows));
        let witness = TraceBackedJoltVmWitness::new(config().with_log_t(2), inputs);
        let stream_result = witness.committed_stream(JoltCommittedPolynomial::BytecodeRa(0), 3);
        let mut stream = match stream_result {
            Ok(stream) => stream,
            Err(error) => {
                assert_eq!(
                    error,
                    WitnessError::UnsupportedFrontier {
                        frontier: "stream should be supported",
                    }
                );
                return;
            }
        };

        assert_eq!(
            stream.next_chunk(),
            Ok(Some(PolynomialChunk::<i128>::OneHot(vec![
                Some(1),
                Some(2),
                Some(0)
            ])))
        );
        assert_eq!(
            stream.next_chunk(),
            Ok(Some(PolynomialChunk::<i128>::OneHot(vec![Some(0)])))
        );
        assert_eq!(stream.next_chunk(), Ok(None::<PolynomialChunk<i128>>));
    }

    #[test]
    fn ram_ra_streams_remapped_address_chunks_and_noop_padding() {
        let program = JoltProgram::default();
        let memory_layout = compact_memory_layout();
        let access_address = memory_layout.stack_end;
        let remapped = memory_layout.remap_word_address(access_address);
        assert_eq!(remapped, Ok(Some(10)));
        let preprocessing = preprocessing_with_memory_layout(memory_layout);
        let rows = vec![
            TraceRow {
                ram_access: RamAccess::Read(RamRead {
                    address: access_address,
                    value: 12,
                }),
                ..Default::default()
            },
            TraceRow {
                ram_access: RamAccess::NoOp,
                ..Default::default()
            },
        ];
        let inputs =
            JoltVmWitnessInputs::new(&program, &preprocessing, trace_output_with_rows(rows));
        let witness = TraceBackedJoltVmWitness::new(config().with_log_t(2), inputs);
        let stream_result = witness.committed_stream(JoltCommittedPolynomial::RamRa(1), 4);
        let mut stream = match stream_result {
            Ok(stream) => stream,
            Err(error) => {
                assert_eq!(
                    error,
                    WitnessError::UnsupportedFrontier {
                        frontier: "stream should be supported",
                    }
                );
                return;
            }
        };

        assert_eq!(
            stream.next_chunk(),
            Ok(Some(PolynomialChunk::<i128>::OneHot(vec![
                Some(10),
                None,
                None,
                None
            ])))
        );
        assert_eq!(stream.next_chunk(), Ok(None::<PolynomialChunk<i128>>));
    }

    #[test]
    fn instruction_ra_streams_lookup_index_chunks_and_noop_padding() {
        let program = JoltProgram::default();
        let preprocessing = preprocessing();
        let mut instruction_row = instruction(RAM_START_ADDRESS as usize);
        instruction_row.operands.imm = -1;
        let rows = vec![TraceRow {
            instruction: instruction_row,
            registers: RegisterState {
                rs1: Some(RegisterRead {
                    register: 2,
                    value: 10,
                }),
                ..Default::default()
            },
            ..Default::default()
        }];
        let inputs =
            JoltVmWitnessInputs::new(&program, &preprocessing, trace_output_with_rows(rows));
        let witness = TraceBackedJoltVmWitness::new(config().with_log_t(2), inputs);
        let stream_result = witness.committed_stream(JoltCommittedPolynomial::InstructionRa(15), 2);
        let mut stream = match stream_result {
            Ok(stream) => stream,
            Err(error) => {
                assert_eq!(
                    error,
                    WitnessError::UnsupportedFrontier {
                        frontier: "stream should be supported",
                    }
                );
                return;
            }
        };

        assert_eq!(
            stream.next_chunk(),
            Ok(Some(PolynomialChunk::<i128>::OneHot(vec![
                Some(1),
                Some(0)
            ])))
        );
        assert_eq!(
            stream.next_chunk(),
            Ok(Some(PolynomialChunk::<i128>::OneHot(vec![
                Some(0),
                Some(0)
            ])))
        );
        assert_eq!(stream.next_chunk(), Ok(None::<PolynomialChunk<i128>>));
    }

    #[test]
    fn committed_batch_stream_preserves_single_pass_core_shape() {
        let program = JoltProgram::default();
        let instruction_row = instruction(RAM_START_ADDRESS as usize);
        let bytecode_result = BytecodePreprocessing::preprocess(
            vec![instruction_row],
            RAM_START_ADDRESS,
            RV64IMAC_JOLT,
        );
        assert!(
            bytecode_result.is_ok(),
            "bytecode preprocessing failed: {bytecode_result:?}"
        );
        let Ok(bytecode) = bytecode_result else {
            return;
        };
        let memory_layout = compact_memory_layout();
        let access_address = memory_layout.stack_end;
        let mut preprocessing = preprocessing_with_bytecode(bytecode);
        preprocessing.memory_layout = memory_layout;
        let rows = vec![
            TraceRow {
                instruction: instruction_row,
                registers: RegisterState {
                    rs1: Some(RegisterRead {
                        register: 2,
                        value: 10,
                    }),
                    rd: Some(RegisterWrite {
                        register: 1,
                        pre_value: 4,
                        post_value: 9,
                    }),
                    ..Default::default()
                },
                ram_access: RamAccess::Write(RamWrite {
                    address: access_address,
                    pre_value: 7,
                    post_value: 11,
                }),
                #[cfg(feature = "field-inline")]
                field_inline: None,
            },
            TraceRow {
                instruction: instruction_row,
                ram_access: RamAccess::NoOp,
                ..Default::default()
            },
        ];
        let inputs =
            JoltVmWitnessInputs::new(&program, &preprocessing, trace_output_with_rows(rows));
        let witness = TraceBackedJoltVmWitness::new(config().with_log_t(2), inputs);
        let ids = [
            JoltCommittedPolynomial::RdInc,
            JoltCommittedPolynomial::RamInc,
            JoltCommittedPolynomial::InstructionRa(15),
            JoltCommittedPolynomial::BytecodeRa(0),
            JoltCommittedPolynomial::RamRa(1),
        ];
        let stream_result = <TraceBackedJoltVmWitness<'_, OwnedTrace> as WitnessProvider<
            i128,
            JoltVmNamespace,
        >>::committed_batch_stream(&witness, &ids, 3);
        let mut stream = match stream_result {
            Ok(stream) => stream,
            Err(error) => {
                assert_eq!(
                    error,
                    WitnessError::UnsupportedFrontier {
                        frontier: "stream should be supported",
                    }
                );
                return;
            }
        };

        let first_result = stream.next_batch();
        assert!(first_result.is_ok(), "first batch failed: {first_result:?}");
        let Ok(Some(first)) = first_result else {
            return;
        };
        assert_eq!(first.len(), 3);
        assert_eq!(
            first.chunks,
            vec![
                (
                    JoltCommittedPolynomial::RdInc,
                    PolynomialChunk::I128(vec![5, 0, 0])
                ),
                (
                    JoltCommittedPolynomial::RamInc,
                    PolynomialChunk::I128(vec![4, 0, 0])
                ),
                (
                    JoltCommittedPolynomial::InstructionRa(15),
                    PolynomialChunk::OneHot(vec![Some(0), Some(0), Some(0)])
                ),
                (
                    JoltCommittedPolynomial::BytecodeRa(0),
                    PolynomialChunk::OneHot(vec![Some(1), Some(1), Some(0)])
                ),
                (
                    JoltCommittedPolynomial::RamRa(1),
                    PolynomialChunk::OneHot(vec![Some(10), None, None])
                ),
            ]
        );

        let second_result = stream.next_batch();
        assert!(
            second_result.is_ok(),
            "second batch failed: {second_result:?}"
        );
        let Ok(Some(second)) = second_result else {
            return;
        };
        assert_eq!(
            second.chunks,
            vec![
                (
                    JoltCommittedPolynomial::RdInc,
                    PolynomialChunk::I128(vec![0])
                ),
                (
                    JoltCommittedPolynomial::RamInc,
                    PolynomialChunk::I128(vec![0])
                ),
                (
                    JoltCommittedPolynomial::InstructionRa(15),
                    PolynomialChunk::OneHot(vec![Some(0)])
                ),
                (
                    JoltCommittedPolynomial::BytecodeRa(0),
                    PolynomialChunk::OneHot(vec![Some(0)])
                ),
                (
                    JoltCommittedPolynomial::RamRa(1),
                    PolynomialChunk::OneHot(vec![None])
                ),
            ]
        );
        assert_eq!(stream.next_batch(), Ok(None));
    }

    #[test]
    fn committed_stream_rejects_unsupported_oracles_and_empty_chunks() {
        let program = JoltProgram::default();
        let preprocessing = preprocessing();
        let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output());
        let witness = TraceBackedJoltVmWitness::new(config(), inputs);

        assert!(matches!(
            witness.committed_stream(JoltCommittedPolynomial::TrustedAdvice, 1),
            Err(WitnessError::UnsupportedFrontier {
                frontier: "Jolt VM committed stream",
            })
        ));
        assert!(matches!(
            witness.committed_stream(JoltCommittedPolynomial::RdInc, 0),
            Err(WitnessError::InvalidDimensions { .. })
        ));
    }
}
