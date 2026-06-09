use core::marker::PhantomData;
use std::collections::HashMap;

use jolt_claims::protocols::jolt::{
    formulas::{committed_openings, dimensions::REGISTER_ADDRESS_BITS, ra::JoltRaPolynomialLayout},
    JoltChallengeId, JoltCommittedPolynomial, JoltFormulaDimensions, JoltOneHotConfig,
    JoltOpeningId, JoltPolynomialId, JoltPublicId, JoltVirtualPolynomial,
};
use jolt_field::{
    signed::{S128, S64},
    Field,
};
use jolt_lookup_tables::{InstructionLookupTable, JoltLookupQuery, LookupQuery, LookupTableKind};
use jolt_program::{
    execution::{JoltProgram, RamAccess, TraceOutput, TraceRow, TraceSource},
    lookup::instruction_lookup_index,
    preprocess::JoltProgramPreprocessing,
};
use jolt_riscv::{
    CircuitFlags, Flags, InstructionFlags, InterleavedBitsMarker, JoltInstruction,
    JoltInstructionKind,
};
use rayon::prelude::*;

use crate::{
    MaterializationPolicy, NamespaceId, OracleDescriptor, OracleKind, OracleRef, OracleViewRequest,
    PolynomialBatchChunk, PolynomialBatchStream, PolynomialChunk, PolynomialEncoding,
    PolynomialStream, PolynomialView, RaFamilyCycleIndices, RetentionHint, ViewRequirement,
    WitnessBuilder, WitnessDimensions, WitnessError, WitnessNamespace,
    RA_FAMILY_MAX_BYTECODE_CHUNKS, RA_FAMILY_MAX_INSTRUCTION_CHUNKS, RA_FAMILY_MAX_RAM_CHUNKS,
};

pub mod rv64;
pub mod stage5;

#[cfg(feature = "field-inline")]
pub mod field_inline;

pub use stage5::{
    JoltVmStage5InstructionReadRafRows, JoltVmStage5InstructionReadRafWitness,
    Stage5InstructionReadRafConfig, Stage5InstructionReadRafContext,
    Stage5InstructionReadRafOutputClaims, Stage5InstructionReadRafRow,
};

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

pub fn jolt_opening_oracle_ref(
    opening: JoltOpeningId,
) -> Result<OracleRef<JoltVmNamespace>, WitnessError> {
    Ok(match opening {
        JoltOpeningId::Polynomial { polynomial, .. } => match polynomial {
            JoltPolynomialId::Committed(id) => OracleRef::committed(id),
            JoltPolynomialId::Virtual(id) => OracleRef::virtual_polynomial(id),
        },
        JoltOpeningId::TrustedAdvice { .. } => {
            OracleRef::committed(JoltCommittedPolynomial::TrustedAdvice)
        }
        JoltOpeningId::UntrustedAdvice { .. } => {
            OracleRef::committed(JoltCommittedPolynomial::UntrustedAdvice)
        }
    })
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JoltVmSpartanOuterRow {
    pub left_instruction_input: u64,
    pub right_instruction_input: i128,
    pub product_magnitude: u128,
    pub product_is_positive: bool,
    pub should_branch: bool,
    pub pc: u64,
    pub unexpanded_pc: u64,
    pub imm: i128,
    pub ram_address: u64,
    pub rs1_value: u64,
    pub rs2_value: u64,
    pub rd_write_value: u64,
    pub ram_read_value: u64,
    pub ram_write_value: u64,
    pub left_lookup_operand: u64,
    pub right_lookup_operand: u128,
    pub next_unexpanded_pc: u64,
    pub next_pc: u64,
    pub next_is_virtual: bool,
    pub next_is_first_in_sequence: bool,
    pub lookup_output: u64,
    pub should_jump: bool,
    pub flag_add_operands: bool,
    pub flag_subtract_operands: bool,
    pub flag_multiply_operands: bool,
    pub flag_load: bool,
    pub flag_store: bool,
    pub flag_jump: bool,
    pub flag_write_lookup_output_to_rd: bool,
    pub flag_virtual_instruction: bool,
    pub flag_assert: bool,
    pub flag_do_not_update_unexpanded_pc: bool,
    pub flag_advice: bool,
    pub flag_is_compressed: bool,
    pub flag_is_first_in_sequence: bool,
    pub flag_is_last_in_sequence: bool,
}

pub trait JoltVmSpartanOuterRows {
    fn spartan_outer_rows(&self) -> Result<Vec<JoltVmSpartanOuterRow>, WitnessError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JoltVmStage2TraceRow {
    pub remapped_ram_address: Option<usize>,
    pub ram_read_value: u64,
    pub ram_write_value: u64,
    pub ram_increment: i128,
    pub left_instruction_input: u64,
    pub right_instruction_input: i128,
    pub lookup_output: u64,
    pub left_lookup_operand: u64,
    pub right_lookup_operand: u128,
    pub branch_flag: bool,
    pub jump_flag: bool,
    pub write_lookup_output_to_rd_flag: bool,
    pub virtual_instruction_flag: bool,
    pub next_is_noop: bool,
}

pub trait JoltVmStage2Rows {
    fn stage2_rows(&self) -> Result<Vec<JoltVmStage2TraceRow>, WitnessError>;

    fn initial_ram_state_words(&self) -> Result<Vec<u64>, WitnessError>;

    fn final_ram_state_words(&self) -> Result<Vec<u64>, WitnessError>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JoltVmStage3ShiftRow {
    pub unexpanded_pc: u64,
    pub pc: u64,
    pub is_virtual: bool,
    pub is_first_in_sequence: bool,
    pub is_noop: bool,
}

pub trait JoltVmStage3ShiftRows {
    fn stage3_shift_rows(&self, log_t: usize) -> Result<Vec<JoltVmStage3ShiftRow>, WitnessError>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JoltVmStage3InstructionRegisterRow {
    pub right_operand_is_rs2: bool,
    pub rs2_value: u64,
    pub right_operand_is_imm: bool,
    pub imm: i128,
    pub left_operand_is_rs1: bool,
    pub rs1_value: u64,
    pub left_operand_is_pc: bool,
    pub unexpanded_pc: u64,
    pub rd_write_value: u64,
}

pub trait JoltVmStage3InstructionRegisterRows {
    fn stage3_instruction_register_rows(
        &self,
        log_t: usize,
    ) -> Result<Vec<JoltVmStage3InstructionRegisterRow>, WitnessError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JoltVmRegisterRead {
    pub register: u8,
    pub value: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JoltVmRegisterWrite {
    pub register: u8,
    pub pre_value: u64,
    pub post_value: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JoltVmRegisterReadWriteRow {
    pub rs1: Option<JoltVmRegisterRead>,
    pub rs2: Option<JoltVmRegisterRead>,
    pub rd: Option<JoltVmRegisterWrite>,
    pub rd_increment: i128,
}

pub trait JoltVmRegisterReadWriteRows {
    fn register_read_write_rows(&self) -> Result<Vec<JoltVmRegisterReadWriteRow>, WitnessError>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JoltVmStage6Row {
    pub instruction_lookup_index: u128,
    pub bytecode_index: usize,
    pub remapped_ram_address: Option<usize>,
    pub ram_access_nonzero: bool,
    pub ram_increment: i128,
    pub rd_increment: i128,
}

pub trait JoltVmStage6Rows {
    fn stage6_rows(&self) -> Result<Vec<JoltVmStage6Row>, WitnessError>;
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

    fn ram_log_k(&self) -> Result<usize, WitnessError> {
        if self.config.ram_k == 0 || !self.config.ram_k.is_power_of_two() {
            return Err(WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!(
                    "ram_k must be a nonzero power of two, got {}",
                    self.config.ram_k
                ),
            });
        }
        Ok(self.config.ram_k.ilog2() as usize)
    }

    fn ram_read_write_dimensions(&self) -> Result<WitnessDimensions, WitnessError> {
        let log_rows = self
            .config
            .log_t
            .checked_add(self.ram_log_k()?)
            .ok_or_else(|| WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: "RAM read-write rows overflow".to_owned(),
            })?;
        let rows = checked_pow2(log_rows)?;
        Ok(WitnessDimensions::new(rows, log_rows))
    }

    fn register_read_write_dimensions(&self) -> Result<WitnessDimensions, WitnessError> {
        let log_rows = self
            .config
            .log_t
            .checked_add(REGISTER_ADDRESS_BITS)
            .ok_or_else(|| WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: "register read-write rows overflow".to_owned(),
            })?;
        let rows = checked_pow2(log_rows)?;
        Ok(WitnessDimensions::new(rows, log_rows))
    }

    fn ram_final_dimensions(&self) -> Result<WitnessDimensions, WitnessError> {
        let log_rows = self.ram_log_k()?;
        Ok(WitnessDimensions::new(self.config.ram_k, log_rows))
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

    fn instruction_virtual_ra_dimensions(&self) -> Result<WitnessDimensions, WitnessError> {
        let log_rows = self
            .config
            .log_t
            .checked_add(self.config.one_hot.lookup_virtual_chunk_bits())
            .ok_or_else(|| WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: "instruction virtual RA rows overflow".to_owned(),
            })?;
        let rows = checked_pow2(log_rows)?;
        Ok(WitnessDimensions::new(rows, log_rows))
    }

    fn instruction_virtual_ra_count(&self) -> Result<usize, WitnessError> {
        let chunk_bits = self.config.one_hot.lookup_virtual_chunk_bits();
        if chunk_bits == 0 || !RV64_LOOKUP_ADDRESS_BITS.is_multiple_of(chunk_bits) {
            return Err(WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!(
                    "lookup virtual chunk bits {chunk_bits} must evenly divide {RV64_LOOKUP_ADDRESS_BITS}"
                ),
            });
        }
        Ok(RV64_LOOKUP_ADDRESS_BITS / chunk_bits)
    }

    fn advice_dimensions(words: usize) -> WitnessDimensions {
        let rows = words.next_power_of_two().max(1);
        WitnessDimensions::new(rows, rows.ilog2() as usize)
    }
}

impl<T: TraceSource + Clone> JoltVmSpartanOuterRows for TraceBackedJoltVmWitness<'_, T> {
    fn spartan_outer_rows(&self) -> Result<Vec<JoltVmSpartanOuterRow>, WitnessError> {
        let rows = checked_pow2(self.config.log_t)?;
        let mut values = Vec::with_capacity(rows);
        let mut trace = self.trace.trace.clone();
        let mut current = trace.next_row().unwrap_or_default();
        let mut pc_cache = PcLookupCache::default();
        for index in 0..rows {
            let next = (index + 1 < rows).then(|| trace.next_row().unwrap_or_default());
            values.push(spartan_outer_row(
                &current,
                next.as_ref(),
                self.preprocessing,
                &mut pc_cache,
            )?);
            if let Some(row) = next {
                current = row;
            }
        }
        Ok(values)
    }
}

impl<T: TraceSource + Clone> JoltVmStage2Rows for TraceBackedJoltVmWitness<'_, T> {
    fn stage2_rows(&self) -> Result<Vec<JoltVmStage2TraceRow>, WitnessError> {
        let rows = checked_pow2(self.config.log_t)?;
        let mut values = Vec::with_capacity(rows);
        let mut trace = self.trace.trace.clone();
        let mut current = trace.next_row().unwrap_or_default();
        for index in 0..rows {
            let next = (index + 1 < rows).then(|| trace.next_row().unwrap_or_default());
            let remapped_ram_address = ram_access_address(current.ram_access)
                .map(|address| self.remapped_ram_address(address))
                .transpose()?
                .flatten();
            values.push(stage2_trace_row(
                &current,
                next.as_ref(),
                remapped_ram_address,
            )?);
            if let Some(row) = next {
                current = row;
            }
        }
        Ok(values)
    }

    fn initial_ram_state_words(&self) -> Result<Vec<u64>, WitnessError> {
        self.initial_ram_state()
    }

    fn final_ram_state_words(&self) -> Result<Vec<u64>, WitnessError> {
        self.final_ram_state()
    }
}

impl<T: TraceSource + Clone> JoltVmStage3ShiftRows for TraceBackedJoltVmWitness<'_, T> {
    fn stage3_shift_rows(&self, log_t: usize) -> Result<Vec<JoltVmStage3ShiftRow>, WitnessError> {
        let rows = checked_pow2(log_t)?;
        let mut values = Vec::with_capacity(rows);
        let mut trace = self.trace.trace.clone();
        let mut pc_cache = PcLookupCache::default();
        for _ in 0..rows {
            let row = trace.next_row().unwrap_or_default();
            values.push(JoltVmStage3ShiftRow {
                unexpanded_pc: row.instruction.address as u64,
                pc: pc_cache.pc_for_row(&row, self.preprocessing)? as u64,
                is_virtual: row.instruction.virtual_sequence_remaining.is_some(),
                is_first_in_sequence: row.instruction.is_first_in_sequence,
                is_noop: row_is_noop(&row),
            });
        }
        Ok(values)
    }
}

impl<T: TraceSource + Clone> JoltVmStage3InstructionRegisterRows
    for TraceBackedJoltVmWitness<'_, T>
{
    fn stage3_instruction_register_rows(
        &self,
        log_t: usize,
    ) -> Result<Vec<JoltVmStage3InstructionRegisterRow>, WitnessError> {
        let rows = checked_pow2(log_t)?;
        let mut values = Vec::with_capacity(rows);
        let mut trace = self.trace.trace.clone();
        for _ in 0..rows {
            let row = trace.next_row().unwrap_or_default();
            values.push(stage3_instruction_register_row(&row)?);
        }
        Ok(values)
    }
}

impl<T: TraceSource + Clone> JoltVmRegisterReadWriteRows for TraceBackedJoltVmWitness<'_, T> {
    fn register_read_write_rows(&self) -> Result<Vec<JoltVmRegisterReadWriteRow>, WitnessError> {
        let rows = checked_pow2(self.config.log_t)?;
        let register_count = checked_pow2(REGISTER_ADDRESS_BITS)?;
        let mut values = Vec::with_capacity(rows);
        let mut trace = self.trace.trace.clone();
        for _ in 0..rows {
            let Some(row) = trace.next_row() else {
                values.push(JoltVmRegisterReadWriteRow::default());
                continue;
            };
            let rs1 = row
                .registers
                .rs1
                .map(|read| register_read(read.register, read.value, register_count))
                .transpose()?;
            let rs2 = row
                .registers
                .rs2
                .map(|read| register_read(read.register, read.value, register_count))
                .transpose()?;
            let rd = row
                .registers
                .rd
                .map(|write| {
                    register_write(
                        write.register,
                        write.pre_value,
                        write.post_value,
                        register_count,
                    )
                })
                .transpose()?;
            values.push(JoltVmRegisterReadWriteRow {
                rs1,
                rs2,
                rd,
                rd_increment: JoltVmIncrementStreamKind::RdInc.value_from_row(&row),
            });
        }
        Ok(values)
    }
}

impl<T: TraceSource + Clone> JoltVmStage6Rows for TraceBackedJoltVmWitness<'_, T> {
    fn stage6_rows(&self) -> Result<Vec<JoltVmStage6Row>, WitnessError> {
        let rows = checked_pow2(self.config.log_t)?;
        let mut values = Vec::with_capacity(rows);
        let mut trace = self.trace.trace.clone();
        let mut pc_cache = PcLookupCache::default();
        for _ in 0..rows {
            let Some(row) = trace.next_row() else {
                values.push(JoltVmStage6Row::default());
                continue;
            };
            let instruction_lookup_index =
                instruction_lookup_index::<RV64_XLEN>(&row).map_err(|error| {
                    WitnessError::InvalidWitnessData {
                        namespace: JOLT_VM_NAMESPACE.name,
                        reason: error.to_string(),
                    }
                })?;
            let bytecode_index = pc_cache
                .pc_for_row_optional(&row, self.preprocessing)
                .unwrap_or(0);
            let ram_address = ram_access_address(row.ram_access);
            let remapped_ram_address = ram_address
                .and_then(|address| {
                    self.preprocessing
                        .memory_layout
                        .remap_word_address(address)
                        .ok()
                })
                .flatten()
                .map(|address| address as usize);
            values.push(JoltVmStage6Row {
                instruction_lookup_index,
                bytecode_index,
                remapped_ram_address,
                ram_access_nonzero: ram_address.is_some_and(|address| address != 0),
                ram_increment: JoltVmIncrementStreamKind::RamInc.value_from_row(&row),
                rd_increment: JoltVmIncrementStreamKind::RdInc.value_from_row(&row),
            });
        }
        Ok(values)
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
            JoltCommittedPolynomial::TrustedAdvice => {
                self.advice_stream_kind(JoltVmAdviceStreamKind::Trusted)
            }
            JoltCommittedPolynomial::UntrustedAdvice => {
                self.advice_stream_kind(JoltVmAdviceStreamKind::Untrusted)
            }
        }
    }

    fn advice_stream_kind(
        &self,
        kind: JoltVmAdviceStreamKind,
    ) -> Result<JoltVmCommittedStreamKind, WitnessError> {
        if !kind.is_included(&self.config) {
            return Err(WitnessError::UnknownOracle {
                namespace: JOLT_VM_NAMESPACE.name,
            });
        }
        kind.validate_len(
            self.advice_bytes(kind).len(),
            self.preprocessing,
            JOLT_VM_NAMESPACE.name,
        )?;
        Ok(JoltVmCommittedStreamKind::Advice(kind))
    }

    fn committed_stream_rows(
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

    fn advice_bytes(&self, kind: JoltVmAdviceStreamKind) -> &[u8] {
        match kind {
            JoltVmAdviceStreamKind::Trusted => &self.trace.device.trusted_advice,
            JoltVmAdviceStreamKind::Untrusted => &self.trace.device.untrusted_advice,
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
    Advice(JoltVmAdviceStreamKind),
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
enum JoltVmAdviceStreamKind {
    Trusted,
    Untrusted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RaChunkSelector {
    shift: usize,
    mask: u128,
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
                JoltVmCommittedStreamKind::Increment(_) | JoltVmCommittedStreamKind::Advice(_) => {}
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
        pc_cache: &mut PcLookupCache,
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

impl JoltVmAdviceStreamKind {
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

    fn rows(self, preprocessing: &JoltProgramPreprocessing) -> usize {
        Self::rows_from_max_bytes(self.max_bytes(preprocessing))
    }

    const fn bytes<'a>(self, trusted: &'a [u8], untrusted: &'a [u8]) -> &'a [u8] {
        match self {
            Self::Trusted => trusted,
            Self::Untrusted => untrusted,
        }
    }

    fn validate_len(
        self,
        bytes_len: usize,
        preprocessing: &JoltProgramPreprocessing,
        namespace: &'static str,
    ) -> Result<(), WitnessError> {
        let max_bytes = self.max_bytes(preprocessing);
        if bytes_len > max_bytes {
            return Err(WitnessError::InvalidWitnessData {
                namespace,
                reason: format!(
                    "{self:?} advice has {bytes_len} bytes, exceeding configured max {max_bytes}",
                ),
            });
        }
        Ok(())
    }

    fn rows_from_max_bytes(max_bytes: usize) -> usize {
        let words = max_bytes / 8;
        words.next_power_of_two().max(1)
    }
}

fn advice_word_le(bytes: &[u8], word_index: usize) -> u64 {
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

fn ra_family_selectors(
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

fn ra_chunk_to_u8(value: usize) -> Result<u8, WitnessError> {
    u8::try_from(value).map_err(|_| WitnessError::InvalidWitnessData {
        namespace: JOLT_VM_NAMESPACE.name,
        reason: format!("RA chunk index {value} exceeds the u8 chunk-index range"),
    })
}

fn bytecode_pc_for_row(row: &TraceRow, preprocessing: &JoltProgramPreprocessing) -> Option<usize> {
    if row_is_noop(row) {
        Some(0)
    } else {
        preprocessing.bytecode.get_pc(&row.instruction)
    }
}

const fn ram_access_address(access: RamAccess) -> Option<u64> {
    match access {
        RamAccess::Read(read) => Some(read.address),
        RamAccess::Write(write) => Some(write.address),
        RamAccess::NoOp => None,
    }
}

impl<F: Field, T: TraceSource + Clone> WitnessBuilder<F> for JoltVmWitnessBuilder<T> {
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

impl<F: Field, T: TraceSource + Clone> crate::WitnessProvider<F, JoltVmNamespace>
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
            OracleKind::Virtual(JoltVirtualPolynomial::RamVal | JoltVirtualPolynomial::RamRa) => {
                self.ram_read_write_dimensions()?
            }
            OracleKind::Virtual(
                JoltVirtualPolynomial::RegistersVal
                | JoltVirtualPolynomial::Rs1Ra
                | JoltVirtualPolynomial::Rs2Ra
                | JoltVirtualPolynomial::RdWa,
            ) => self.register_read_write_dimensions()?,
            OracleKind::Virtual(JoltVirtualPolynomial::RamValFinal) => {
                self.ram_final_dimensions()?
            }
            OracleKind::Virtual(JoltVirtualPolynomial::InstructionRa(index)) => {
                require_index(index, self.instruction_virtual_ra_count()?)?;
                self.instruction_virtual_ra_dimensions()?
            }
            OracleKind::Virtual(JoltVirtualPolynomial::LookupTableFlag(index)) => {
                require_index(index, LookupTableKind::<RV64_XLEN>::COUNT)?;
                self.trace_dimensions()?
            }
            OracleKind::Virtual(JoltVirtualPolynomial::InstructionRafFlag) => {
                self.trace_dimensions()?
            }
            OracleKind::Virtual(id) if supported_trace_virtual(id) => self.trace_dimensions()?,
            OracleKind::Virtual(_) => {
                return Err(WitnessError::UnknownOracle {
                    namespace: JOLT_VM_NAMESPACE.name,
                });
            }
        };
        Ok(OracleDescriptor::new(
            oracle,
            dimensions,
            oracle_encoding(oracle.kind),
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
        request: OracleViewRequest<JoltVmNamespace>,
    ) -> Result<PolynomialView<'_, F, JoltVmNamespace>, WitnessError> {
        let descriptor = <Self as crate::WitnessProvider<F, JoltVmNamespace>>::describe_oracle(
            self,
            request.oracle(),
        )?;
        let values = match request.oracle().kind {
            OracleKind::Virtual(id)
                if matches!(
                    id,
                    JoltVirtualPolynomial::RamVal | JoltVirtualPolynomial::RamRa
                ) =>
            {
                self.materialize_ram_read_write_virtual(id)?
            }
            OracleKind::Virtual(id)
                if matches!(
                    id,
                    JoltVirtualPolynomial::RegistersVal
                        | JoltVirtualPolynomial::Rs1Ra
                        | JoltVirtualPolynomial::Rs2Ra
                        | JoltVirtualPolynomial::RdWa
                ) =>
            {
                self.materialize_register_read_write_virtual(id)?
            }
            OracleKind::Virtual(JoltVirtualPolynomial::RamValFinal) => {
                self.materialize_ram_val_final()?
            }
            OracleKind::Virtual(JoltVirtualPolynomial::InstructionRa(index)) => {
                self.materialize_instruction_ra(index)?
            }
            OracleKind::Virtual(id) => self.materialize_trace_virtual(id)?,
            OracleKind::Committed(id) => self.materialize_compact_committed(id)?,
        };
        Ok(PolynomialView::owned(descriptor, values))
    }

    fn try_evaluate_oracle_view(
        &self,
        request: OracleViewRequest<JoltVmNamespace>,
        point: &[F],
    ) -> Result<Option<F>, WitnessError> {
        let directly_evaluates_committed = matches!(
            request.oracle().kind,
            OracleKind::Committed(
                JoltCommittedPolynomial::RdInc
                    | JoltCommittedPolynomial::RamInc
                    | JoltCommittedPolynomial::InstructionRa(_)
                    | JoltCommittedPolynomial::BytecodeRa(_)
                    | JoltCommittedPolynomial::RamRa(_)
            )
        );
        if request.requirement.encoding != PolynomialEncoding::Dense
            && !directly_evaluates_committed
        {
            return Ok(None);
        }

        match request.oracle().kind {
            OracleKind::Committed(
                id @ (JoltCommittedPolynomial::RdInc | JoltCommittedPolynomial::RamInc),
            ) => self.evaluate_committed_trace_dense(id, point).map(Some),
            OracleKind::Committed(
                id @ (JoltCommittedPolynomial::InstructionRa(_)
                | JoltCommittedPolynomial::BytecodeRa(_)
                | JoltCommittedPolynomial::RamRa(_)),
            ) => self.evaluate_committed_ra(id, point).map(Some),
            OracleKind::Virtual(JoltVirtualPolynomial::InstructionRa(index)) => {
                self.evaluate_instruction_ra(index, point).map(Some)
            }
            OracleKind::Virtual(
                id @ (JoltVirtualPolynomial::RamVal | JoltVirtualPolynomial::RamRa),
            ) => self.evaluate_ram_read_write_virtual(id, point).map(Some),
            OracleKind::Virtual(JoltVirtualPolynomial::RamValFinal) => {
                self.evaluate_ram_val_final(point).map(Some)
            }
            OracleKind::Virtual(
                id @ (JoltVirtualPolynomial::RegistersVal
                | JoltVirtualPolynomial::Rs1Ra
                | JoltVirtualPolynomial::Rs2Ra
                | JoltVirtualPolynomial::RdWa),
            ) => self
                .evaluate_register_read_write_virtual(id, point)
                .map(Some),
            OracleKind::Virtual(
                id @ (JoltVirtualPolynomial::InstructionRafFlag
                | JoltVirtualPolynomial::LookupTableFlag(_)
                | JoltVirtualPolynomial::RamHammingWeight),
            ) => self.evaluate_trace_virtual(id, point).map(Some),
            _ => Ok(None),
        }
    }

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

impl<F: Field, T: TraceSource + Clone> crate::CommittedWitnessProvider<F, JoltVmNamespace>
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

fn eq_index_msb<F: Field>(point: &[F], index: usize) -> F {
    let mut eq = F::one();
    for (position, challenge) in point.iter().enumerate() {
        let shift = point.len() - 1 - position;
        let bit = if shift < usize::BITS as usize {
            (index >> shift) & 1
        } else {
            0
        };
        eq *= if bit == 1 {
            *challenge
        } else {
            F::one() - *challenge
        };
    }
    eq
}

fn eq_evals_msb<F: Field>(point: &[F]) -> Result<Vec<F>, WitnessError> {
    let rows = checked_pow2(point.len())?;
    Ok((0..rows).map(|index| eq_index_msb(point, index)).collect())
}

impl<T: TraceSource + Clone> TraceBackedJoltVmWitness<'_, T> {
    fn materialize_trace_virtual<F: Field>(
        &self,
        id: JoltVirtualPolynomial,
    ) -> Result<Vec<F>, WitnessError> {
        if !supported_trace_virtual(id) {
            return Err(WitnessError::UnknownOracle {
                namespace: JOLT_VM_NAMESPACE.name,
            });
        }

        let rows = checked_pow2(self.config.log_t)?;
        let mut values = Vec::with_capacity(rows);
        let mut trace = self.trace.trace.clone();
        let mut current = trace.next_row().unwrap_or_default();
        for index in 0..rows {
            let next = (index + 1 < rows).then(|| trace.next_row().unwrap_or_default());
            let value = trace_virtual_value::<F>(&current, next.as_ref(), id, self.preprocessing)?;
            values.push(value);
            if let Some(row) = next {
                current = row;
            }
        }
        Ok(values)
    }

    fn materialize_ram_read_write_virtual<F: Field>(
        &self,
        id: JoltVirtualPolynomial,
    ) -> Result<Vec<F>, WitnessError> {
        match id {
            JoltVirtualPolynomial::RamVal => self.materialize_ram_val(),
            JoltVirtualPolynomial::RamRa => self.materialize_ram_ra(),
            _ => Err(WitnessError::UnknownOracle {
                namespace: JOLT_VM_NAMESPACE.name,
            }),
        }
    }

    fn materialize_register_read_write_virtual<F: Field>(
        &self,
        id: JoltVirtualPolynomial,
    ) -> Result<Vec<F>, WitnessError> {
        match id {
            JoltVirtualPolynomial::RegistersVal
            | JoltVirtualPolynomial::Rs1Ra
            | JoltVirtualPolynomial::Rs2Ra
            | JoltVirtualPolynomial::RdWa => {}
            _ => {
                return Err(WitnessError::UnknownOracle {
                    namespace: JOLT_VM_NAMESPACE.name,
                });
            }
        }

        let cycles = checked_pow2(self.config.log_t)?;
        let register_count = checked_pow2(REGISTER_ADDRESS_BITS)?;
        let mut values = vec![F::zero(); register_count * cycles];
        let mut trace = self.trace.trace.clone();

        if id == JoltVirtualPolynomial::RegistersVal {
            let mut state = vec![0u64; register_count];
            for cycle in 0..cycles {
                for (register, value) in state.iter().copied().enumerate() {
                    values[register * cycles + cycle] = F::from_u64(value);
                }

                let Some(row) = trace.next_row() else {
                    continue;
                };
                if let Some(write) = row.registers.rd {
                    let register = usize::from(write.register);
                    if register >= register_count {
                        return Err(invalid_register_address(write.register));
                    }
                    state[register] = write.post_value;
                }
            }
            return Ok(values);
        }

        for cycle in 0..cycles {
            let Some(row) = trace.next_row() else {
                break;
            };
            let register = match id {
                JoltVirtualPolynomial::Rs1Ra => row.registers.rs1.map(|read| read.register),
                JoltVirtualPolynomial::Rs2Ra => row.registers.rs2.map(|read| read.register),
                JoltVirtualPolynomial::RdWa => row.registers.rd.map(|write| write.register),
                _ => None,
            };
            if let Some(register) = register {
                let register = usize::from(register);
                if register >= register_count {
                    return Err(invalid_register_address(register as u8));
                }
                values[register * cycles + cycle] = F::one();
            }
        }

        Ok(values)
    }

    fn evaluate_committed_ra<F: Field>(
        &self,
        id: JoltCommittedPolynomial,
        point: &[F],
    ) -> Result<F, WitnessError> {
        let cycles = checked_pow2(self.config.log_t)?;
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

        let mut trace = self.trace.trace.clone();
        let mut result = F::zero();
        for cycle in 0..cycles {
            let value = trace.next_row().map_or_else(
                || Ok(kind.padding_value()),
                |row| kind.value_from_row(&row, self.preprocessing),
            )?;
            let Some(value) = value else {
                continue;
            };
            let flat_index = value
                .checked_mul(cycles)
                .and_then(|base| base.checked_add(cycle))
                .ok_or_else(|| WitnessError::InvalidDimensions {
                    namespace: JOLT_VM_NAMESPACE.name,
                    reason: "committed RA flat index overflow".to_owned(),
                })?;
            result += eq_index_msb(point, flat_index);
        }
        Ok(result)
    }

    fn evaluate_committed_trace_dense<F: Field>(
        &self,
        id: JoltCommittedPolynomial,
        point: &[F],
    ) -> Result<F, WitnessError> {
        if point.len() != self.config.log_t {
            return Err(WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!(
                    "committed dense trace point has {} variables, expected {}",
                    point.len(),
                    self.config.log_t
                ),
            });
        }

        let rows = checked_pow2(self.config.log_t)?;
        let eq = eq_evals_msb(point)?;
        let mut stream = self.committed_stream(id, rows.max(1))?;
        let mut index = 0usize;
        let mut result = F::zero();
        loop {
            let next: Option<PolynomialChunk<F>> = stream.next_chunk()?;
            let Some(chunk) = next else {
                break;
            };
            match chunk {
                PolynomialChunk::I128(values) => {
                    for value in values {
                        if index >= rows {
                            return Err(WitnessError::InvalidWitnessData {
                                namespace: JOLT_VM_NAMESPACE.name,
                                reason: format!(
                                    "committed dense stream for {id:?} exceeded {rows} rows"
                                ),
                            });
                        }
                        result += eq[index] * F::from_i128(value);
                        index += 1;
                    }
                }
                PolynomialChunk::Dense(values) => {
                    for value in values {
                        if index >= rows {
                            return Err(WitnessError::InvalidWitnessData {
                                namespace: JOLT_VM_NAMESPACE.name,
                                reason: format!(
                                    "committed dense stream for {id:?} exceeded {rows} rows"
                                ),
                            });
                        }
                        result += eq[index] * value;
                        index += 1;
                    }
                }
                PolynomialChunk::Zeros(count) => {
                    index = index.checked_add(count).ok_or_else(|| {
                        WitnessError::InvalidWitnessData {
                            namespace: JOLT_VM_NAMESPACE.name,
                            reason: format!(
                                "committed dense stream for {id:?} zero chunk overflowed row count"
                            ),
                        }
                    })?;
                    if index > rows {
                        return Err(WitnessError::InvalidWitnessData {
                            namespace: JOLT_VM_NAMESPACE.name,
                            reason: format!(
                                "committed dense stream for {id:?} exceeded {rows} rows"
                            ),
                        });
                    }
                }
                _ => {
                    return Err(WitnessError::InvalidWitnessData {
                        namespace: JOLT_VM_NAMESPACE.name,
                        reason: format!("committed dense stream for {id:?} used non-dense chunks"),
                    });
                }
            }
        }
        if index != rows {
            return Err(WitnessError::InvalidWitnessData {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!(
                    "committed dense stream for {id:?} produced {index} rows, expected {rows}"
                ),
            });
        }
        Ok(result)
    }

    fn materialize_instruction_ra<F: Field>(&self, index: usize) -> Result<Vec<F>, WitnessError> {
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

    fn evaluate_instruction_ra<F: Field>(
        &self,
        index: usize,
        point: &[F],
    ) -> Result<F, WitnessError> {
        let cycles = checked_pow2(self.config.log_t)?;
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
        let mut trace = self.trace.trace.clone();
        let mut result = F::zero();
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
            let flat_index = value
                .checked_mul(cycles)
                .and_then(|base| base.checked_add(cycle))
                .ok_or_else(|| WitnessError::InvalidDimensions {
                    namespace: JOLT_VM_NAMESPACE.name,
                    reason: "instruction virtual RA flat index overflow".to_owned(),
                })?;
            result += eq_index_msb(point, flat_index);
        }
        Ok(result)
    }

    fn evaluate_ram_read_write_virtual<F: Field>(
        &self,
        id: JoltVirtualPolynomial,
        point: &[F],
    ) -> Result<F, WitnessError> {
        match id {
            JoltVirtualPolynomial::RamVal => self.evaluate_ram_val(point),
            JoltVirtualPolynomial::RamRa => self.evaluate_ram_ra(point),
            _ => Err(WitnessError::UnknownOracle {
                namespace: JOLT_VM_NAMESPACE.name,
            }),
        }
    }

    fn evaluate_ram_val<F: Field>(&self, point: &[F]) -> Result<F, WitnessError> {
        let log_k = self.ram_log_k()?;
        let expected_vars = log_k.checked_add(self.config.log_t).ok_or_else(|| {
            WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: "RAM value point length overflow".to_owned(),
            }
        })?;
        if point.len() != expected_vars {
            return Err(WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!(
                    "RAM value point has {} variables, expected {expected_vars}",
                    point.len()
                ),
            });
        }

        let cycles = checked_pow2(self.config.log_t)?;
        let (address_point, cycle_point) = point.split_at(log_k);
        let address_eq = eq_evals_msb(address_point)?;
        let cycle_eq = eq_evals_msb(cycle_point)?;
        let mut state = self.initial_ram_state()?;
        if state.len() != self.config.ram_k {
            return Err(WitnessError::InvalidWitnessData {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!(
                    "initial RAM state has {} words, expected {}",
                    state.len(),
                    self.config.ram_k
                ),
            });
        }

        let mut state_eval = state
            .iter()
            .copied()
            .zip(address_eq.iter().copied())
            .map(|(value, eq)| eq * F::from_u64(value))
            .sum::<F>();
        let mut result = F::zero();
        let mut trace = self.trace.trace.clone();

        for cycle_weight in cycle_eq.iter().copied().take(cycles) {
            let mut cycle_eval = state_eval;
            if let Some(row) = trace.next_row() {
                match row.ram_access {
                    RamAccess::Read(read) => {
                        if let Some(address) = self.remapped_ram_address(read.address)? {
                            let observed = F::from_u64(read.value);
                            cycle_eval +=
                                address_eq[address] * (observed - F::from_u64(state[address]));
                        }
                    }
                    RamAccess::Write(write) => {
                        if let Some(address) = self.remapped_ram_address(write.address)? {
                            let previous = state[address];
                            let pre_value = F::from_u64(write.pre_value);
                            cycle_eval += address_eq[address] * (pre_value - F::from_u64(previous));
                            state_eval += address_eq[address]
                                * (F::from_u64(write.post_value) - F::from_u64(previous));
                            state[address] = write.post_value;
                        }
                    }
                    RamAccess::NoOp => {}
                }
            }
            result += cycle_weight * cycle_eval;
        }

        Ok(result)
    }

    fn evaluate_ram_ra<F: Field>(&self, point: &[F]) -> Result<F, WitnessError> {
        let log_k = self.ram_log_k()?;
        let expected_vars = log_k.checked_add(self.config.log_t).ok_or_else(|| {
            WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: "RAM RA point length overflow".to_owned(),
            }
        })?;
        if point.len() != expected_vars {
            return Err(WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!(
                    "RAM RA point has {} variables, expected {expected_vars}",
                    point.len()
                ),
            });
        }

        let cycles = checked_pow2(self.config.log_t)?;
        let (address_point, cycle_point) = point.split_at(log_k);
        let address_eq = eq_evals_msb(address_point)?;
        let cycle_eq = eq_evals_msb(cycle_point)?;
        let mut trace = self.trace.trace.clone();
        let mut result = F::zero();

        for &cycle_weight in cycle_eq.iter().take(cycles) {
            let Some(row) = trace.next_row() else {
                break;
            };
            let Some(raw_address) = ram_access_address(row.ram_access) else {
                continue;
            };
            if let Some(address) = self.remapped_ram_address(raw_address)? {
                result += cycle_weight * address_eq[address];
            }
        }

        Ok(result)
    }

    fn evaluate_ram_val_final<F: Field>(&self, point: &[F]) -> Result<F, WitnessError> {
        let log_k = self.ram_log_k()?;
        if point.len() != log_k {
            return Err(WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!(
                    "RAM final value point has {} variables, expected {log_k}",
                    point.len()
                ),
            });
        }

        let eq = eq_evals_msb(point)?;
        let state = self.final_ram_state()?;
        if state.len() != self.config.ram_k {
            return Err(WitnessError::InvalidWitnessData {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!(
                    "final RAM state has {} words, expected {}",
                    state.len(),
                    self.config.ram_k
                ),
            });
        }
        Ok(state
            .into_iter()
            .zip(eq)
            .map(|(value, eq)| eq * F::from_u64(value))
            .sum())
    }

    fn evaluate_register_read_write_virtual<F: Field>(
        &self,
        id: JoltVirtualPolynomial,
        point: &[F],
    ) -> Result<F, WitnessError> {
        match id {
            JoltVirtualPolynomial::RegistersVal
            | JoltVirtualPolynomial::Rs1Ra
            | JoltVirtualPolynomial::Rs2Ra
            | JoltVirtualPolynomial::RdWa => {}
            _ => {
                return Err(WitnessError::UnknownOracle {
                    namespace: JOLT_VM_NAMESPACE.name,
                });
            }
        }
        let expected_vars = REGISTER_ADDRESS_BITS
            .checked_add(self.config.log_t)
            .ok_or_else(|| WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: "register read-write point length overflow".to_owned(),
            })?;
        if point.len() != expected_vars {
            return Err(WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!(
                    "register read-write point has {} variables, expected {expected_vars}",
                    point.len()
                ),
            });
        }

        let cycles = checked_pow2(self.config.log_t)?;
        let (register_point, cycle_point) = point.split_at(REGISTER_ADDRESS_BITS);
        let register_eq = eq_evals_msb(register_point)?;
        let cycle_eq = eq_evals_msb(cycle_point)?;
        let register_count = checked_pow2(REGISTER_ADDRESS_BITS)?;

        if id == JoltVirtualPolynomial::RegistersVal {
            let mut state = vec![0u64; register_count];
            let mut state_eval = F::zero();
            let mut trace = self.trace.trace.clone();
            let mut result = F::zero();
            for cycle_weight in cycle_eq.iter().copied().take(cycles) {
                result += cycle_weight * state_eval;
                let Some(row) = trace.next_row() else {
                    continue;
                };
                if let Some(write) = row.registers.rd {
                    let register = usize::from(write.register);
                    if register >= register_count {
                        return Err(invalid_register_address(write.register));
                    }
                    let previous = state[register];
                    state_eval += register_eq[register]
                        * (F::from_u64(write.post_value) - F::from_u64(previous));
                    state[register] = write.post_value;
                }
            }
            return Ok(result);
        }

        let mut trace = self.trace.trace.clone();
        let mut result = F::zero();
        for &cycle_weight in cycle_eq.iter().take(cycles) {
            let Some(row) = trace.next_row() else {
                break;
            };
            let register = match id {
                JoltVirtualPolynomial::Rs1Ra => row.registers.rs1.map(|read| read.register),
                JoltVirtualPolynomial::Rs2Ra => row.registers.rs2.map(|read| read.register),
                JoltVirtualPolynomial::RdWa => row.registers.rd.map(|write| write.register),
                _ => None,
            };
            if let Some(register) = register {
                let register = usize::from(register);
                if register >= register_count {
                    return Err(invalid_register_address(register as u8));
                }
                result += cycle_weight * register_eq[register];
            }
        }
        Ok(result)
    }

    fn evaluate_trace_virtual<F: Field>(
        &self,
        id: JoltVirtualPolynomial,
        point: &[F],
    ) -> Result<F, WitnessError> {
        if point.len() != self.config.log_t {
            return Err(WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!(
                    "trace virtual point has {} variables, expected {}",
                    point.len(),
                    self.config.log_t
                ),
            });
        }

        let cycles = checked_pow2(self.config.log_t)?;
        let mut trace = self.trace.trace.clone();
        let mut current = trace.next_row().unwrap_or_default();
        let mut result = F::zero();
        for index in 0..cycles {
            let next = (index + 1 < cycles).then(|| trace.next_row().unwrap_or_default());
            let value = trace_virtual_value::<F>(&current, next.as_ref(), id, self.preprocessing)?;
            result += value * eq_index_msb(point, index);
            if let Some(row) = next {
                current = row;
            }
        }
        Ok(result)
    }

    fn materialize_ram_val<F: Field>(&self) -> Result<Vec<F>, WitnessError> {
        let cycles = checked_pow2(self.config.log_t)?;
        let addresses = self.config.ram_k;
        let mut state = self.initial_ram_state()?;
        let mut values = vec![F::zero(); addresses * cycles];
        let mut trace = self.trace.trace.clone();

        for cycle in 0..cycles {
            for (address, value) in state.iter().copied().enumerate() {
                values[address * cycles + cycle] = F::from_u64(value);
            }

            let Some(row) = trace.next_row() else {
                continue;
            };
            match row.ram_access {
                RamAccess::Read(read) => {
                    if let Some(address) = self.remapped_ram_address(read.address)? {
                        values[address * cycles + cycle] = F::from_u64(read.value);
                    }
                }
                RamAccess::Write(write) => {
                    if let Some(address) = self.remapped_ram_address(write.address)? {
                        values[address * cycles + cycle] = F::from_u64(write.pre_value);
                        state[address] = write.post_value;
                    }
                }
                RamAccess::NoOp => {}
            }
        }

        Ok(values)
    }

    fn materialize_ram_ra<F: Field>(&self) -> Result<Vec<F>, WitnessError> {
        let cycles = checked_pow2(self.config.log_t)?;
        let addresses = self.config.ram_k;
        let mut values = vec![F::zero(); addresses * cycles];
        let mut trace = self.trace.trace.clone();

        for cycle in 0..cycles {
            let Some(row) = trace.next_row() else {
                continue;
            };
            if let Some(raw_address) = ram_access_address(row.ram_access) {
                if let Some(address) = self.remapped_ram_address(raw_address)? {
                    values[address * cycles + cycle] = F::one();
                }
            }
        }

        Ok(values)
    }

    fn materialize_ram_val_final<F: Field>(&self) -> Result<Vec<F>, WitnessError> {
        self.final_ram_state()
            .map(|state| state.into_iter().map(F::from_u64).collect())
    }

    fn materialize_compact_committed<F: Field>(
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

        let descriptor = <Self as crate::WitnessProvider<F, JoltVmNamespace>>::describe_oracle(
            self,
            OracleRef::committed(id),
        )?;
        let mut stream = self.committed_stream(id, descriptor.dimensions.rows.max(1))?;
        let mut values = Vec::with_capacity(descriptor.dimensions.rows);
        while let Some(chunk) = stream.next_chunk()? {
            append_compact_chunk(&mut values, chunk)?;
        }
        if values.len() != descriptor.dimensions.rows {
            return Err(WitnessError::InvalidWitnessData {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!(
                    "committed oracle {id:?} materialized {} rows, expected {}",
                    values.len(),
                    descriptor.dimensions.rows
                ),
            });
        }
        Ok(values)
    }

    fn materialize_one_hot_committed<F: Field>(
        &self,
        id: JoltCommittedPolynomial,
    ) -> Result<Vec<F>, WitnessError> {
        let descriptor = <Self as crate::WitnessProvider<F, JoltVmNamespace>>::describe_oracle(
            self,
            OracleRef::committed(id),
        )?;
        let cycles = checked_pow2(self.config.log_t)?;
        if !descriptor.dimensions.rows.is_multiple_of(cycles) {
            return Err(WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!(
                    "committed oracle {id:?} has {} rows, not divisible by {cycles} cycles",
                    descriptor.dimensions.rows
                ),
            });
        }
        let addresses = descriptor.dimensions.rows / cycles;
        let mut stream = self.committed_stream(id, cycles.max(1))?;
        let mut values = vec![F::zero(); descriptor.dimensions.rows];
        let mut cycle = 0usize;
        loop {
            let next: Option<PolynomialChunk<F>> = stream.next_chunk()?;
            let Some(chunk) = next else {
                break;
            };
            let PolynomialChunk::OneHot(chunk) = chunk else {
                return Err(WitnessError::InvalidWitnessData {
                    namespace: JOLT_VM_NAMESPACE.name,
                    reason: format!("committed oracle {id:?} did not stream one-hot chunks"),
                });
            };
            for address in chunk {
                if let Some(address) = address {
                    if address >= addresses {
                        return Err(WitnessError::InvalidWitnessData {
                            namespace: JOLT_VM_NAMESPACE.name,
                            reason: format!(
                                "committed oracle {id:?} streamed address {address}, beyond {addresses}"
                            ),
                        });
                    }
                    let index = address
                        .checked_mul(cycles)
                        .and_then(|base| base.checked_add(cycle))
                        .ok_or_else(|| WitnessError::InvalidDimensions {
                            namespace: JOLT_VM_NAMESPACE.name,
                            reason: format!("committed oracle {id:?} index overflow"),
                        })?;
                    values[index] = F::one();
                }
                cycle += 1;
            }
        }
        if cycle != cycles {
            return Err(WitnessError::InvalidWitnessData {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!("committed oracle {id:?} streamed {cycle} rows, expected {cycles}"),
            });
        }
        Ok(values)
    }

    fn initial_ram_state(&self) -> Result<Vec<u64>, WitnessError> {
        if self.config.ram_k == 0 {
            return Err(WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: "ram_k must be nonzero".to_owned(),
            });
        }
        let mut state = vec![0; self.config.ram_k];
        if !self.preprocessing.ram.bytecode_words.is_empty() {
            let start = self.remapped_required_address(
                self.preprocessing.ram.min_bytecode_address,
                "bytecode",
            )?;
            populate_ram_words(
                &mut state,
                start,
                &self.preprocessing.ram.bytecode_words,
                "bytecode",
            )?;
        }
        if !self.trace.device.trusted_advice.is_empty() {
            let start = self.remapped_required_address(
                self.trace.device.memory_layout.trusted_advice_start,
                "trusted advice",
            )?;
            populate_ram_bytes(
                &mut state,
                start,
                &self.trace.device.trusted_advice,
                "trusted advice",
            )?;
        }
        if !self.trace.device.untrusted_advice.is_empty() {
            let start = self.remapped_required_address(
                self.trace.device.memory_layout.untrusted_advice_start,
                "untrusted advice",
            )?;
            populate_ram_bytes(
                &mut state,
                start,
                &self.trace.device.untrusted_advice,
                "untrusted advice",
            )?;
        }
        if !self.trace.device.inputs.is_empty() {
            let start = self
                .remapped_required_address(self.trace.device.memory_layout.input_start, "input")?;
            populate_ram_bytes(&mut state, start, &self.trace.device.inputs, "input")?;
        }
        Ok(state)
    }

    fn final_ram_state(&self) -> Result<Vec<u64>, WitnessError> {
        if self.config.ram_k == 0 {
            return Err(WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: "ram_k must be nonzero".to_owned(),
            });
        }
        let mut state = vec![0; self.config.ram_k];
        if let Some(final_memory) = &self.trace.final_memory {
            self.populate_final_memory_image(&mut state, &final_memory.bytes)?;
        }
        if !self.trace.device.trusted_advice.is_empty() {
            let start = self.remapped_required_address(
                self.trace.device.memory_layout.trusted_advice_start,
                "trusted advice",
            )?;
            populate_ram_bytes(
                &mut state,
                start,
                &self.trace.device.trusted_advice,
                "trusted advice",
            )?;
        }
        if !self.trace.device.untrusted_advice.is_empty() {
            let start = self.remapped_required_address(
                self.trace.device.memory_layout.untrusted_advice_start,
                "untrusted advice",
            )?;
            populate_ram_bytes(
                &mut state,
                start,
                &self.trace.device.untrusted_advice,
                "untrusted advice",
            )?;
        }
        if !self.trace.device.inputs.is_empty() {
            let start = self
                .remapped_required_address(self.trace.device.memory_layout.input_start, "input")?;
            populate_ram_bytes(&mut state, start, &self.trace.device.inputs, "input")?;
        }
        if !self.trace.device.outputs.is_empty() {
            let start = self.remapped_required_address(
                self.trace.device.memory_layout.output_start,
                "output",
            )?;
            populate_ram_bytes(&mut state, start, &self.trace.device.outputs, "output")?;
        }

        let panic_index =
            self.remapped_required_address(self.trace.device.memory_layout.panic, "panic")?;
        set_ram_word(
            &mut state,
            panic_index,
            self.trace.device.panic as u64,
            "panic",
        )?;
        if !self.trace.device.panic {
            let termination_index = self.remapped_required_address(
                self.trace.device.memory_layout.termination,
                "termination",
            )?;
            set_ram_word(&mut state, termination_index, 1, "termination")?;
        }
        Ok(state)
    }

    fn populate_final_memory_image(
        &self,
        state: &mut [u64],
        bytes: &[(u64, u8)],
    ) -> Result<(), WitnessError> {
        let dram_start = self.dram_start_address()?;
        for &(address, byte) in bytes {
            let absolute_address = if address >= dram_start {
                address
            } else {
                dram_start
                    .checked_add(address)
                    .ok_or_else(|| WitnessError::InvalidWitnessData {
                        namespace: JOLT_VM_NAMESPACE.name,
                        reason: format!("final memory address offset {address:#x} overflows"),
                    })?
            };
            let Some(word_index) = self.remapped_ram_address(absolute_address)? else {
                continue;
            };
            if word_index >= state.len() {
                return Err(WitnessError::InvalidWitnessData {
                    namespace: JOLT_VM_NAMESPACE.name,
                    reason: format!(
                        "final memory address {absolute_address:#x} remapped to {word_index}, beyond ram_k {}",
                        state.len()
                    ),
                });
            }
            let shift = ((absolute_address & 7) * 8) as usize;
            state[word_index] =
                (state[word_index] & !(0xff_u64 << shift)) | ((byte as u64) << shift);
        }
        Ok(())
    }

    fn dram_start_address(&self) -> Result<u64, WitnessError> {
        self.preprocessing
            .memory_layout
            .stack_end
            .checked_sub(self.preprocessing.memory_layout.program_size)
            .ok_or_else(|| WitnessError::InvalidWitnessData {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: "memory layout stack_end is below program_size".to_owned(),
            })
    }

    fn remapped_required_address(
        &self,
        address: u64,
        label: &'static str,
    ) -> Result<usize, WitnessError> {
        self.preprocessing
            .memory_layout
            .remapped_word_address(address)
            .map(|address| address as usize)
            .map_err(|error| WitnessError::InvalidWitnessData {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!("failed to remap {label} address {address:#x}: {error}"),
            })
    }

    fn remapped_ram_address(&self, address: u64) -> Result<Option<usize>, WitnessError> {
        let remapped = self
            .preprocessing
            .memory_layout
            .remap_word_address(address)
            .map_err(|error| WitnessError::InvalidWitnessData {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!("failed to remap RAM access address {address:#x}: {error}"),
            })?;
        let Some(address) = remapped else {
            return Ok(None);
        };
        let address = address as usize;
        if address >= self.config.ram_k {
            return Err(WitnessError::InvalidWitnessData {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!(
                    "RAM access address remapped to {address}, beyond ram_k {}",
                    self.config.ram_k
                ),
            });
        }
        Ok(Some(address))
    }
}

fn append_compact_chunk<F: Field>(
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

fn populate_ram_bytes(
    state: &mut [u64],
    start: usize,
    bytes: &[u8],
    label: &'static str,
) -> Result<(), WitnessError> {
    let words = bytes
        .chunks(8)
        .map(|chunk| {
            let mut word = [0; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            u64::from_le_bytes(word)
        })
        .collect::<Vec<_>>();
    populate_ram_words(state, start, &words, label)
}

fn set_ram_word(
    state: &mut [u64],
    index: usize,
    word: u64,
    label: &'static str,
) -> Result<(), WitnessError> {
    let Some(slot) = state.get_mut(index) else {
        return Err(WitnessError::InvalidWitnessData {
            namespace: JOLT_VM_NAMESPACE.name,
            reason: format!("{label} memory index {index} exceeds ram_k {}", state.len()),
        });
    };
    *slot = word;
    Ok(())
}

fn populate_ram_words(
    state: &mut [u64],
    start: usize,
    words: &[u64],
    label: &'static str,
) -> Result<(), WitnessError> {
    let end = start
        .checked_add(words.len())
        .ok_or_else(|| WitnessError::InvalidWitnessData {
            namespace: JOLT_VM_NAMESPACE.name,
            reason: format!("{label} memory range overflows"),
        })?;
    if end > state.len() {
        return Err(WitnessError::InvalidWitnessData {
            namespace: JOLT_VM_NAMESPACE.name,
            reason: format!(
                "{label} memory range [{start}, {end}) exceeds ram_k {}",
                state.len()
            ),
        });
    }
    state[start..end].copy_from_slice(words);
    Ok(())
}

const fn oracle_encoding(
    kind: OracleKind<JoltCommittedPolynomial, JoltVirtualPolynomial>,
) -> PolynomialEncoding {
    match kind {
        OracleKind::Committed(
            JoltCommittedPolynomial::InstructionRa(_)
            | JoltCommittedPolynomial::BytecodeRa(_)
            | JoltCommittedPolynomial::RamRa(_),
        ) => PolynomialEncoding::OneHot,
        OracleKind::Committed(_) => PolynomialEncoding::Compact,
        OracleKind::Virtual(_) => PolynomialEncoding::Dense,
    }
}

const fn supported_trace_virtual(id: JoltVirtualPolynomial) -> bool {
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

fn trace_virtual_value<F: Field>(
    row: &TraceRow,
    next: Option<&TraceRow>,
    id: JoltVirtualPolynomial,
    preprocessing: &JoltProgramPreprocessing,
) -> Result<F, WitnessError> {
    let instruction = JoltInstruction::try_from(row.instruction).map_err(|kind| {
        WitnessError::InvalidWitnessData {
            namespace: JOLT_VM_NAMESPACE.name,
            reason: format!("unsupported Jolt instruction kind in trace row: {kind:?}"),
        }
    })?;
    let circuit_flags = instruction.circuit_flags();
    let instruction_flags = instruction.instruction_flags();
    let query = JoltLookupQuery::new(row.instruction.instruction_kind, row);

    let value = match id {
        JoltVirtualPolynomial::PC => F::from_u64(pc_for_row(row, preprocessing)? as u64),
        JoltVirtualPolynomial::UnexpandedPC => F::from_u64(row.instruction.address as u64),
        JoltVirtualPolynomial::NextPC => next
            .map(|row| pc_for_row(row, preprocessing).map(|pc| F::from_u64(pc as u64)))
            .transpose()?
            .unwrap_or_else(F::zero),
        JoltVirtualPolynomial::NextUnexpandedPC => {
            next.map_or_else(F::zero, |row| F::from_u64(row.instruction.address as u64))
        }
        JoltVirtualPolynomial::NextIsNoop => F::from_bool(next.is_some_and(row_is_noop)),
        JoltVirtualPolynomial::NextIsVirtual => F::from_bool(next.is_some_and(|row| {
            row_circuit_flags(row)
                .map(|flags| flags[CircuitFlags::VirtualInstruction])
                .unwrap_or(false)
        })),
        JoltVirtualPolynomial::NextIsFirstInSequence => F::from_bool(next.is_some_and(|row| {
            row_circuit_flags(row)
                .map(|flags| flags[CircuitFlags::IsFirstInSequence])
                .unwrap_or(false)
        })),
        JoltVirtualPolynomial::LeftLookupOperand => {
            let (left, _) = LookupQuery::<RV64_XLEN>::to_lookup_operands(&query);
            F::from_u64(left)
        }
        JoltVirtualPolynomial::RightLookupOperand => {
            let (_, right) = LookupQuery::<RV64_XLEN>::to_lookup_operands(&query);
            F::from_u128(right)
        }
        JoltVirtualPolynomial::LeftInstructionInput => {
            let (left, _) = LookupQuery::<RV64_XLEN>::to_instruction_inputs(&query);
            F::from_u64(left)
        }
        JoltVirtualPolynomial::RightInstructionInput => {
            let (_, right) = LookupQuery::<RV64_XLEN>::to_instruction_inputs(&query);
            F::from_i128(right)
        }
        JoltVirtualPolynomial::Product => {
            let (left, right) = LookupQuery::<RV64_XLEN>::to_instruction_inputs(&query);
            let product = S64::from_u64(left).mul_trunc::<2, 2>(&S128::from_i128(right));
            signed_128_to_field(product)
        }
        JoltVirtualPolynomial::ShouldJump => {
            let next_is_noop = next.is_some_and(row_is_noop);
            F::from_bool(circuit_flags[CircuitFlags::Jump] && !next_is_noop)
        }
        JoltVirtualPolynomial::ShouldBranch => {
            let lookup_output = LookupQuery::<RV64_XLEN>::to_lookup_output(&query);
            F::from_bool(instruction_flags[InstructionFlags::Branch] && lookup_output == 1)
        }
        JoltVirtualPolynomial::Imm => F::from_i128(row.instruction.operands.imm),
        JoltVirtualPolynomial::Rs1Value => {
            F::from_u64(row.registers.rs1.map_or(0, |read| read.value))
        }
        JoltVirtualPolynomial::Rs2Value => {
            F::from_u64(row.registers.rs2.map_or(0, |read| read.value))
        }
        JoltVirtualPolynomial::RdWriteValue => {
            F::from_u64(row.registers.rd.map_or(0, |write| write.post_value))
        }
        JoltVirtualPolynomial::LookupOutput => {
            F::from_u64(LookupQuery::<RV64_XLEN>::to_lookup_output(&query))
        }
        JoltVirtualPolynomial::InstructionRafFlag => {
            F::from_bool(!circuit_flags.is_interleaved_operands())
        }
        JoltVirtualPolynomial::LookupTableFlag(index) => {
            if index >= LookupTableKind::<RV64_XLEN>::COUNT {
                return Err(WitnessError::UnknownOracle {
                    namespace: JOLT_VM_NAMESPACE.name,
                });
            }
            let table_index =
                <JoltInstruction as InstructionLookupTable<RV64_XLEN>>::lookup_table(&instruction)
                    .map(|table| table.index());
            F::from_bool(table_index == Some(index))
        }
        JoltVirtualPolynomial::RamAddress => F::from_u64(match row.ram_access {
            RamAccess::Read(read) => read.address,
            RamAccess::Write(write) => write.address,
            RamAccess::NoOp => 0,
        }),
        JoltVirtualPolynomial::RamReadValue => F::from_u64(match row.ram_access {
            RamAccess::Read(read) => read.value,
            RamAccess::Write(write) => write.pre_value,
            RamAccess::NoOp => 0,
        }),
        JoltVirtualPolynomial::RamWriteValue => F::from_u64(match row.ram_access {
            RamAccess::Read(read) => read.value,
            RamAccess::Write(write) => write.post_value,
            RamAccess::NoOp => 0,
        }),
        JoltVirtualPolynomial::RamHammingWeight => {
            F::from_bool(ram_access_address(row.ram_access).is_some_and(|address| address != 0))
        }
        JoltVirtualPolynomial::OpFlags(flag) => F::from_bool(circuit_flags[flag]),
        JoltVirtualPolynomial::InstructionFlags(flag) => F::from_bool(instruction_flags[flag]),
        _ => {
            return Err(WitnessError::UnknownOracle {
                namespace: JOLT_VM_NAMESPACE.name,
            });
        }
    };

    Ok(value)
}

fn stage2_trace_row(
    row: &TraceRow,
    next: Option<&TraceRow>,
    remapped_ram_address: Option<usize>,
) -> Result<JoltVmStage2TraceRow, WitnessError> {
    let instruction = JoltInstruction::try_from(row.instruction).map_err(|kind| {
        WitnessError::InvalidWitnessData {
            namespace: JOLT_VM_NAMESPACE.name,
            reason: format!("unsupported Jolt instruction kind in trace row: {kind:?}"),
        }
    })?;
    let circuit_flags = instruction.circuit_flags();
    let instruction_flags = instruction.instruction_flags();
    let query = JoltLookupQuery::new(row.instruction.instruction_kind, row);
    let (left_instruction_input, right_instruction_input) =
        LookupQuery::<RV64_XLEN>::to_instruction_inputs(&query);
    let (left_lookup_operand, right_lookup_operand) =
        LookupQuery::<RV64_XLEN>::to_lookup_operands(&query);
    let lookup_output = LookupQuery::<RV64_XLEN>::to_lookup_output(&query);
    let (ram_read_value, ram_write_value) = match row.ram_access {
        RamAccess::Read(read) => (read.value, read.value),
        RamAccess::Write(write) => (write.pre_value, write.post_value),
        RamAccess::NoOp => (0, 0),
    };
    let next_is_noop = next.is_none_or(row_is_noop);

    Ok(JoltVmStage2TraceRow {
        remapped_ram_address,
        ram_read_value,
        ram_write_value,
        ram_increment: JoltVmIncrementStreamKind::RamInc.value_from_row(row),
        left_instruction_input,
        right_instruction_input,
        lookup_output,
        left_lookup_operand,
        right_lookup_operand,
        branch_flag: instruction_flags[InstructionFlags::Branch],
        jump_flag: circuit_flags[CircuitFlags::Jump],
        write_lookup_output_to_rd_flag: circuit_flags[CircuitFlags::WriteLookupOutputToRD],
        virtual_instruction_flag: circuit_flags[CircuitFlags::VirtualInstruction],
        next_is_noop,
    })
}

fn stage3_instruction_register_row(
    row: &TraceRow,
) -> Result<JoltVmStage3InstructionRegisterRow, WitnessError> {
    let instruction_flags = row_instruction_flags(row)?;
    Ok(JoltVmStage3InstructionRegisterRow {
        right_operand_is_rs2: instruction_flags[InstructionFlags::RightOperandIsRs2Value],
        rs2_value: row.registers.rs2.map_or(0, |read| read.value),
        right_operand_is_imm: instruction_flags[InstructionFlags::RightOperandIsImm],
        imm: row.instruction.operands.imm,
        left_operand_is_rs1: instruction_flags[InstructionFlags::LeftOperandIsRs1Value],
        rs1_value: row.registers.rs1.map_or(0, |read| read.value),
        left_operand_is_pc: instruction_flags[InstructionFlags::LeftOperandIsPC],
        unexpanded_pc: row.instruction.address as u64,
        rd_write_value: row.registers.rd.map_or(0, |write| write.post_value),
    })
}

fn spartan_outer_row(
    row: &TraceRow,
    next: Option<&TraceRow>,
    preprocessing: &JoltProgramPreprocessing,
    pc_cache: &mut PcLookupCache,
) -> Result<JoltVmSpartanOuterRow, WitnessError> {
    let instruction = JoltInstruction::try_from(row.instruction).map_err(|kind| {
        WitnessError::InvalidWitnessData {
            namespace: JOLT_VM_NAMESPACE.name,
            reason: format!("unsupported Jolt instruction kind in trace row: {kind:?}"),
        }
    })?;
    let circuit_flags = instruction.circuit_flags();
    let instruction_flags = instruction.instruction_flags();
    let query = JoltLookupQuery::new(row.instruction.instruction_kind, row);

    let (left_instruction_input, right_instruction_input) =
        LookupQuery::<RV64_XLEN>::to_instruction_inputs(&query);
    let product = S64::from_u64(left_instruction_input)
        .mul_trunc::<2, 2>(&S128::from_i128(right_instruction_input));
    let (left_lookup_operand, right_lookup_operand) =
        LookupQuery::<RV64_XLEN>::to_lookup_operands(&query);
    let lookup_output = LookupQuery::<RV64_XLEN>::to_lookup_output(&query);
    let rs1_value = row.registers.rs1.map_or(0, |read| read.value);
    let rs2_value = row.registers.rs2.map_or(0, |read| read.value);
    let rd_write_value = row.registers.rd.map_or(0, |write| write.post_value);
    let (ram_address, ram_read_value, ram_write_value) = match row.ram_access {
        RamAccess::Read(read) => (read.address, read.value, read.value),
        RamAccess::Write(write) => (write.address, write.pre_value, write.post_value),
        RamAccess::NoOp => (0, 0, 0),
    };
    let pc = pc_cache.pc_for_row(row, preprocessing)? as u64;
    let next_pc = next
        .map(|row| pc_cache.pc_for_row(row, preprocessing).map(|pc| pc as u64))
        .transpose()?
        .unwrap_or(0);
    let next_unexpanded_pc = next.map_or(0, |row| row.instruction.address as u64);
    let next_is_noop = next.is_some_and(row_is_noop);
    let next_is_virtual =
        next.is_some_and(|row| row.instruction.virtual_sequence_remaining.is_some());
    let next_is_first_in_sequence = next.is_some_and(|row| row.instruction.is_first_in_sequence);

    Ok(JoltVmSpartanOuterRow {
        left_instruction_input,
        right_instruction_input,
        product_magnitude: product.magnitude_as_u128(),
        product_is_positive: product.is_positive,
        should_branch: instruction_flags[InstructionFlags::Branch] && lookup_output == 1,
        pc,
        unexpanded_pc: row.instruction.address as u64,
        imm: row.instruction.operands.imm,
        ram_address,
        rs1_value,
        rs2_value,
        rd_write_value,
        ram_read_value,
        ram_write_value,
        left_lookup_operand,
        right_lookup_operand,
        next_unexpanded_pc,
        next_pc,
        next_is_virtual,
        next_is_first_in_sequence,
        lookup_output,
        should_jump: circuit_flags[CircuitFlags::Jump] && !next_is_noop,
        flag_add_operands: circuit_flags[CircuitFlags::AddOperands],
        flag_subtract_operands: circuit_flags[CircuitFlags::SubtractOperands],
        flag_multiply_operands: circuit_flags[CircuitFlags::MultiplyOperands],
        flag_load: circuit_flags[CircuitFlags::Load],
        flag_store: circuit_flags[CircuitFlags::Store],
        flag_jump: circuit_flags[CircuitFlags::Jump],
        flag_write_lookup_output_to_rd: circuit_flags[CircuitFlags::WriteLookupOutputToRD],
        flag_virtual_instruction: circuit_flags[CircuitFlags::VirtualInstruction],
        flag_assert: circuit_flags[CircuitFlags::Assert],
        flag_do_not_update_unexpanded_pc: circuit_flags[CircuitFlags::DoNotUpdateUnexpandedPC],
        flag_advice: circuit_flags[CircuitFlags::Advice],
        flag_is_compressed: circuit_flags[CircuitFlags::IsCompressed],
        flag_is_first_in_sequence: circuit_flags[CircuitFlags::IsFirstInSequence],
        flag_is_last_in_sequence: circuit_flags[CircuitFlags::IsLastInSequence],
    })
}

fn row_circuit_flags(row: &TraceRow) -> Result<jolt_riscv::CircuitFlagSet, WitnessError> {
    Ok(JoltInstruction::try_from(row.instruction)
        .map_err(|kind| WitnessError::InvalidWitnessData {
            namespace: JOLT_VM_NAMESPACE.name,
            reason: format!("unsupported Jolt instruction kind in trace row: {kind:?}"),
        })?
        .circuit_flags())
}

fn row_instruction_flags(row: &TraceRow) -> Result<jolt_riscv::InstructionFlagSet, WitnessError> {
    Ok(JoltInstruction::try_from(row.instruction)
        .map_err(|kind| WitnessError::InvalidWitnessData {
            namespace: JOLT_VM_NAMESPACE.name,
            reason: format!("unsupported Jolt instruction kind in trace row: {kind:?}"),
        })?
        .instruction_flags())
}

fn row_is_noop(row: &TraceRow) -> bool {
    row.instruction.instruction_kind == JoltInstructionKind::NoOp
}

#[derive(Clone, Debug, Default)]
struct PcLookupCache {
    values: HashMap<(usize, u16), usize>,
}

impl PcLookupCache {
    fn pc_for_row(
        &mut self,
        row: &TraceRow,
        preprocessing: &JoltProgramPreprocessing,
    ) -> Result<usize, WitnessError> {
        self.pc_for_row_optional(row, preprocessing)
            .ok_or_else(|| missing_pc_mapping(row))
    }

    fn pc_for_row_optional(
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

fn pc_lookup_key(row: &TraceRow) -> (usize, u16) {
    (
        row.instruction.address,
        row.instruction.virtual_sequence_remaining.unwrap_or(0),
    )
}

fn pc_for_row(
    row: &TraceRow,
    preprocessing: &JoltProgramPreprocessing,
) -> Result<usize, WitnessError> {
    preprocessing
        .bytecode
        .get_pc(&row.instruction)
        .ok_or_else(|| missing_pc_mapping(row))
}

fn missing_pc_mapping(row: &TraceRow) -> WitnessError {
    WitnessError::InvalidWitnessData {
        namespace: JOLT_VM_NAMESPACE.name,
        reason: format!(
            "bytecode preprocessing is missing PC mapping for address {:#x} with virtual_sequence_remaining {:?}",
            row.instruction.address, row.instruction.virtual_sequence_remaining
        ),
    }
}

fn signed_128_to_field<F: Field>(value: S128) -> F {
    if let Some(value) = value.to_i128() {
        F::from_i128(value)
    } else {
        let magnitude = value.magnitude_as_u128();
        if value.is_positive {
            F::from_u128(magnitude)
        } else {
            -F::from_u128(magnitude)
        }
    }
}

fn invalid_register_address(register: u8) -> WitnessError {
    WitnessError::InvalidWitnessData {
        namespace: JOLT_VM_NAMESPACE.name,
        reason: format!(
            "register index {register} exceeds {}-bit register read-write domain",
            REGISTER_ADDRESS_BITS
        ),
    }
}

fn register_read(
    register: u8,
    value: u64,
    register_count: usize,
) -> Result<JoltVmRegisterRead, WitnessError> {
    if usize::from(register) >= register_count {
        return Err(invalid_register_address(register));
    }
    Ok(JoltVmRegisterRead { register, value })
}

fn register_write(
    register: u8,
    pre_value: u64,
    post_value: u64,
    register_count: usize,
) -> Result<JoltVmRegisterWrite, WitnessError> {
    if usize::from(register) >= register_count {
        return Err(invalid_register_address(register));
    }
    Ok(JoltVmRegisterWrite {
        register,
        pre_value,
        post_value,
    })
}

#[cfg(test)]
mod tests {
    use common::{
        constants::RAM_START_ADDRESS,
        jolt_device::{JoltDevice, MemoryConfig, MemoryLayout},
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_program::{
        execution::{
            JoltProgram, MemoryImage, OwnedTrace, RamAccess, RamRead, RamWrite, RegisterRead,
            RegisterState, RegisterWrite, TraceOutput, TraceRow,
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

    fn trace_output_with_device(device: JoltDevice) -> TraceOutput<OwnedTrace> {
        TraceOutput::new(OwnedTrace::default(), device, None)
    }

    fn trace_output_with_device_and_final_memory(
        device: JoltDevice,
        final_memory: MemoryImage,
    ) -> TraceOutput<OwnedTrace> {
        TraceOutput::new(OwnedTrace::default(), device, Some(final_memory))
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
            Fr,
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

        let result = <JoltVmWitnessBuilder<OwnedTrace> as WitnessBuilder<Fr>>::build(
            &mut builder,
            &config,
            inputs,
        );
        let witness = match result {
            Ok(witness) => witness,
            Err(error) => {
                assert_eq!(
                    error,
                    WitnessError::UnsupportedView {
                        view: "builder should not fail",
                    }
                );
                return;
            }
        };

        assert_eq!(
            <TraceBackedJoltVmWitness<'_, OwnedTrace> as WitnessProvider<
                Fr,
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
    fn virtual_oracle_descriptors_report_stage1_trace_columns() -> Result<(), String> {
        let program = JoltProgram::default();
        let preprocessing = preprocessing();
        let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output());
        let witness = TraceBackedJoltVmWitness::new(config().with_log_t(2), inputs);

        assert_eq!(
            describe(
                &witness,
                OracleRef::virtual_polynomial(JoltVirtualPolynomial::Product)
            )
            .map_err(|error| error.to_string())?,
            OracleDescriptor::new(
                OracleRef::virtual_polynomial(JoltVirtualPolynomial::Product),
                WitnessDimensions::new(4, 2),
                PolynomialEncoding::Dense,
            )
        );
        assert_eq!(
            describe(
                &witness,
                OracleRef::virtual_polynomial(JoltVirtualPolynomial::RamRa),
            ),
            Ok(OracleDescriptor::new(
                OracleRef::virtual_polynomial(JoltVirtualPolynomial::RamRa),
                WitnessDimensions::new(256, 8),
                PolynomialEncoding::Dense,
            ))
        );
        assert_eq!(
            describe(
                &witness,
                OracleRef::virtual_polynomial(JoltVirtualPolynomial::RamValFinal),
            ),
            Ok(OracleDescriptor::new(
                OracleRef::virtual_polynomial(JoltVirtualPolynomial::RamValFinal),
                WitnessDimensions::new(64, 6),
                PolynomialEncoding::Dense,
            ))
        );
        Ok(())
    }

    #[test]
    fn virtual_oracle_views_materialize_stage1_r1cs_inputs() -> Result<(), String> {
        let instruction_row = instruction(0x8000_0000);
        let bytecode = BytecodePreprocessing::preprocess(
            vec![instruction_row],
            instruction_row.address as u64,
            RV64IMAC_JOLT,
        )
        .map_err(|error| error.to_string())?;
        let preprocessing = preprocessing_with_bytecode(bytecode);
        let program = JoltProgram::default();
        let rows = vec![
            TraceRow {
                instruction: instruction_row,
                registers: RegisterState {
                    rs1: Some(RegisterRead {
                        register: 2,
                        value: 5,
                    }),
                    rd: Some(RegisterWrite {
                        register: 1,
                        pre_value: 0,
                        post_value: 8,
                    }),
                    ..Default::default()
                },
                ram_access: RamAccess::Read(RamRead {
                    address: RAM_START_ADDRESS,
                    value: 7,
                }),
                #[cfg(feature = "field-inline")]
                field_inline: None,
            },
            TraceRow::default(),
        ];
        let inputs =
            JoltVmWitnessInputs::new(&program, &preprocessing, trace_output_with_rows(rows));
        let witness = TraceBackedJoltVmWitness::new(config().with_log_t(2), inputs);

        assert_virtual_values(
            &witness,
            JoltVirtualPolynomial::LeftInstructionInput,
            &[5, 0, 0, 0],
        )?;
        assert_virtual_values(
            &witness,
            JoltVirtualPolynomial::RightInstructionInput,
            &[3, 0, 0, 0],
        )?;
        assert_virtual_values(&witness, JoltVirtualPolynomial::Product, &[15, 0, 0, 0])?;
        assert_virtual_values(&witness, JoltVirtualPolynomial::LookupOutput, &[8, 0, 0, 0])?;
        assert_virtual_values(&witness, JoltVirtualPolynomial::PC, &[1, 0, 0, 0])?;
        assert_virtual_values(&witness, JoltVirtualPolynomial::NextIsNoop, &[1, 1, 1, 0])?;
        assert_virtual_values(
            &witness,
            JoltVirtualPolynomial::RamAddress,
            &[RAM_START_ADDRESS, 0, 0, 0],
        )?;
        assert_virtual_values(&witness, JoltVirtualPolynomial::RamReadValue, &[7, 0, 0, 0])?;
        assert_virtual_values(
            &witness,
            JoltVirtualPolynomial::RamWriteValue,
            &[7, 0, 0, 0],
        )?;
        assert_virtual_values(
            &witness,
            JoltVirtualPolynomial::OpFlags(CircuitFlags::AddOperands),
            &[1, 0, 0, 0],
        )?;
        assert_virtual_values(
            &witness,
            JoltVirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsImm),
            &[1, 0, 0, 0],
        )?;
        let spartan_rows = witness
            .spartan_outer_rows()
            .map_err(|error| error.to_string())?;
        assert_eq!(spartan_rows.len(), 4);
        assert_eq!(spartan_rows[0].left_instruction_input, 5);
        assert_eq!(spartan_rows[0].right_instruction_input, 3);
        assert_eq!(spartan_rows[0].product_magnitude, 15);
        assert!(spartan_rows[0].product_is_positive);
        assert_eq!(spartan_rows[0].lookup_output, 8);
        assert_eq!(spartan_rows[0].pc, 1);
        assert_eq!(spartan_rows[0].next_pc, 0);
        assert_eq!(spartan_rows[0].ram_address, RAM_START_ADDRESS);
        assert_eq!(spartan_rows[0].ram_read_value, 7);
        assert_eq!(spartan_rows[0].ram_write_value, 7);
        assert!(spartan_rows[0].flag_add_operands);
        assert!(!spartan_rows[3].next_is_virtual);
        Ok(())
    }

    #[test]
    fn ram_read_write_virtual_views_materialize_address_major_state() -> Result<(), String> {
        let program = JoltProgram::default();
        let memory_layout = compact_memory_layout();
        let access_address = memory_layout.stack_end;
        let preprocessing = preprocessing_with_memory_layout(memory_layout);
        let rows = vec![
            TraceRow {
                ram_access: RamAccess::Write(RamWrite {
                    address: access_address,
                    pre_value: 3,
                    post_value: 9,
                }),
                ..Default::default()
            },
            TraceRow {
                ram_access: RamAccess::NoOp,
                ..Default::default()
            },
            TraceRow {
                ram_access: RamAccess::Read(RamRead {
                    address: access_address,
                    value: 9,
                }),
                ..Default::default()
            },
        ];
        let inputs =
            JoltVmWitnessInputs::new(&program, &preprocessing, trace_output_with_rows(rows));
        let witness = TraceBackedJoltVmWitness::new(
            JoltVmWitnessConfig::new(2, 16, config().one_hot),
            inputs,
        );

        let val = materialized_virtual_view(&witness, JoltVirtualPolynomial::RamVal)?;
        let ra = materialized_virtual_view(&witness, JoltVirtualPolynomial::RamRa)?;
        let base = 10 * 4;
        assert_eq!(val.len(), 64);
        assert_eq!(ra.len(), 64);
        assert_eq!(val[base], Fr::from_u64(3));
        assert_eq!(val[base + 1], Fr::from_u64(9));
        assert_eq!(val[base + 2], Fr::from_u64(9));
        assert_eq!(val[base + 3], Fr::from_u64(9));
        assert_eq!(ra[base], Fr::from_u64(1));
        assert_eq!(ra[base + 1], Fr::from_u64(0));
        assert_eq!(ra[base + 2], Fr::from_u64(1));
        assert_eq!(ra[base + 3], Fr::from_u64(0));
        Ok(())
    }

    #[test]
    fn register_read_write_virtual_views_materialize_address_major_state() -> Result<(), String> {
        let program = JoltProgram::default();
        let preprocessing = preprocessing();
        let rows = vec![
            TraceRow {
                registers: RegisterState {
                    rd: Some(RegisterWrite {
                        register: 1,
                        pre_value: 0,
                        post_value: 5,
                    }),
                    ..Default::default()
                },
                ..Default::default()
            },
            TraceRow {
                registers: RegisterState {
                    rs1: Some(RegisterRead {
                        register: 1,
                        value: 5,
                    }),
                    rd: Some(RegisterWrite {
                        register: 2,
                        pre_value: 0,
                        post_value: 7,
                    }),
                    ..Default::default()
                },
                ..Default::default()
            },
            TraceRow {
                registers: RegisterState {
                    rs2: Some(RegisterRead {
                        register: 2,
                        value: 7,
                    }),
                    ..Default::default()
                },
                ..Default::default()
            },
        ];
        let inputs =
            JoltVmWitnessInputs::new(&program, &preprocessing, trace_output_with_rows(rows));
        let witness = TraceBackedJoltVmWitness::new(config().with_log_t(2), inputs);

        let val = materialized_virtual_view(&witness, JoltVirtualPolynomial::RegistersVal)?;
        let rs1_ra = materialized_virtual_view(&witness, JoltVirtualPolynomial::Rs1Ra)?;
        let rs2_ra = materialized_virtual_view(&witness, JoltVirtualPolynomial::Rs2Ra)?;
        let rd_wa = materialized_virtual_view(&witness, JoltVirtualPolynomial::RdWa)?;

        assert_eq!(val.len(), 128 * 4);
        assert_eq!(&val[4..8], &[0, 5, 5, 5].map(Fr::from_u64));
        assert_eq!(&val[8..12], &[0, 0, 7, 7].map(Fr::from_u64));
        assert_eq!(rs1_ra[4 + 1], Fr::from_u64(1));
        assert_eq!(rs2_ra[8 + 2], Fr::from_u64(1));
        assert_eq!(rd_wa[4], Fr::from_u64(1));
        assert_eq!(rd_wa[8 + 1], Fr::from_u64(1));
        Ok(())
    }

    #[test]
    fn ram_val_final_virtual_view_materializes_final_memory_and_public_io() -> Result<(), String> {
        let program = JoltProgram::default();
        let memory_layout = MemoryLayout::new(&MemoryConfig {
            max_input_size: 8,
            max_trusted_advice_size: 8,
            max_untrusted_advice_size: 8,
            max_output_size: 8,
            stack_size: 0,
            heap_size: 0,
            program_size: Some(64),
        });
        let preprocessing = preprocessing_with_memory_layout(memory_layout.clone());
        let device = JoltDevice {
            memory_layout,
            trusted_advice: vec![0x11],
            untrusted_advice: vec![0x22],
            inputs: vec![0x33],
            outputs: vec![0x44, 0x55],
            ..Default::default()
        };
        let final_memory = MemoryImage {
            bytes: vec![(64, 0x66), (65, 0x77)],
        };
        let inputs = JoltVmWitnessInputs::new(
            &program,
            &preprocessing,
            trace_output_with_device_and_final_memory(device, final_memory),
        );
        let witness = TraceBackedJoltVmWitness::new(
            JoltVmWitnessConfig::new(2, 32, config().one_hot),
            inputs,
        );

        let val_final = materialized_virtual_view(&witness, JoltVirtualPolynomial::RamValFinal)?;
        assert_eq!(val_final.len(), 32);
        assert_eq!(val_final[0], Fr::from_u64(0x11));
        assert_eq!(val_final[1], Fr::from_u64(0x22));
        assert_eq!(val_final[2], Fr::from_u64(0x33));
        assert_eq!(val_final[3], Fr::from_u64(0x5544));
        assert_eq!(val_final[4], Fr::from_u64(0));
        assert_eq!(val_final[5], Fr::from_u64(1));
        assert_eq!(val_final[16], Fr::from_u64(0x7766));
        Ok(())
    }

    fn assert_virtual_values(
        witness: &TraceBackedJoltVmWitness<'_, OwnedTrace>,
        id: JoltVirtualPolynomial,
        expected: &[u64],
    ) -> Result<(), String> {
        let actual = materialized_virtual_view(witness, id)?;
        let expected = expected
            .iter()
            .copied()
            .map(Fr::from_u64)
            .collect::<Vec<_>>();
        assert_eq!(actual.as_slice(), expected.as_slice());
        Ok(())
    }

    fn materialized_virtual_view(
        witness: &TraceBackedJoltVmWitness<'_, OwnedTrace>,
        id: JoltVirtualPolynomial,
    ) -> Result<Vec<Fr>, String> {
        let oracle = OracleRef::virtual_polynomial(id);
        let mut requirements = <TraceBackedJoltVmWitness<'_, OwnedTrace> as WitnessProvider<
            Fr,
            JoltVmNamespace,
        >>::view_requirements(witness, oracle)
        .map_err(|error| error.to_string())?;
        let requirement = requirements
            .pop()
            .ok_or_else(|| format!("missing view requirement for {id:?}"))?;
        let view = <TraceBackedJoltVmWitness<'_, OwnedTrace> as WitnessProvider<
            Fr,
            JoltVmNamespace,
        >>::oracle_view(witness, OracleViewRequest::new(requirement))
        .map_err(|error| error.to_string())?;
        let actual = view
            .as_slice()
            .ok_or_else(|| format!("virtual view for {id:?} was not materialized"))?;
        Ok(actual.to_vec())
    }

    fn assert_direct_eval_matches_dense(
        witness: &TraceBackedJoltVmWitness<'_, OwnedTrace>,
        id: JoltVirtualPolynomial,
        point: &[u64],
    ) -> Result<(), String> {
        let oracle = OracleRef::virtual_polynomial(id);
        let mut requirements = <TraceBackedJoltVmWitness<'_, OwnedTrace> as WitnessProvider<
            Fr,
            JoltVmNamespace,
        >>::view_requirements(witness, oracle)
        .map_err(|error| error.to_string())?;
        let requirement = requirements
            .pop()
            .ok_or_else(|| format!("missing view requirement for {id:?}"))?;
        let point = point.iter().copied().map(Fr::from_u64).collect::<Vec<_>>();
        let direct = <TraceBackedJoltVmWitness<'_, OwnedTrace> as WitnessProvider<
            Fr,
            JoltVmNamespace,
        >>::try_evaluate_oracle_view(
            witness, OracleViewRequest::new(requirement), &point
        )
        .map_err(|error| error.to_string())?
        .ok_or_else(|| format!("no direct evaluation for {id:?}"))?;
        let dense = materialized_virtual_view(witness, id)?;
        let expected = dense
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| value * eq_index_msb(&point, index))
            .sum::<Fr>();
        assert_eq!(direct, expected);
        Ok(())
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
                    WitnessError::UnsupportedView {
                        view: "stream should be supported",
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
                    WitnessError::UnsupportedView {
                        view: "stream should be supported",
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
                    WitnessError::UnsupportedView {
                        view: "stream should be supported",
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
                    WitnessError::UnsupportedView {
                        view: "stream should be supported",
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
                    WitnessError::UnsupportedView {
                        view: "stream should be supported",
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
    fn virtual_instruction_ra_and_flags_evaluate_without_dense_materialization(
    ) -> Result<(), String> {
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
        let config = JoltVmWitnessConfig::new(
            2,
            64,
            JoltOneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 4,
            },
        );
        let witness = TraceBackedJoltVmWitness::new(config, inputs);

        assert_direct_eval_matches_dense(
            &witness,
            JoltVirtualPolynomial::InstructionRa(31),
            &[2, 3, 5, 7, 11, 13],
        )?;
        assert_direct_eval_matches_dense(
            &witness,
            JoltVirtualPolynomial::InstructionRafFlag,
            &[17, 19],
        )?;
        assert_direct_eval_matches_dense(
            &witness,
            JoltVirtualPolynomial::LookupTableFlag(0),
            &[23, 29],
        )?;
        Ok(())
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
            Fr,
            JoltVmNamespace,
        >>::committed_batch_stream(&witness, &ids, 3);
        let mut stream = match stream_result {
            Ok(stream) => stream,
            Err(error) => {
                assert_eq!(
                    error,
                    WitnessError::UnsupportedView {
                        view: "stream should be supported",
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
    fn advice_streams_pack_device_bytes_as_little_endian_words() {
        let program = JoltProgram::default();
        let preprocessing = preprocessing();
        let device = JoltDevice {
            trusted_advice: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            untrusted_advice: vec![0xaa, 0xbb],
            ..Default::default()
        };
        let inputs =
            JoltVmWitnessInputs::new(&program, &preprocessing, trace_output_with_device(device));
        let witness = TraceBackedJoltVmWitness::new(
            config()
                .include_trusted_advice(true)
                .include_untrusted_advice(true),
            inputs,
        );

        let trusted_result = witness.committed_stream(JoltCommittedPolynomial::TrustedAdvice, 3);
        assert!(
            trusted_result.is_ok(),
            "trusted advice stream failed: {trusted_result:?}"
        );
        let Ok(mut trusted) = trusted_result else {
            return;
        };
        assert_eq!(
            trusted.next_chunk(),
            Ok(Some(PolynomialChunk::<i128>::U64(vec![
                0x0807_0605_0403_0201,
                0x0a09,
                0,
            ])))
        );
        assert_eq!(
            trusted.next_chunk(),
            Ok(Some(PolynomialChunk::<i128>::U64(vec![0, 0, 0])))
        );
        assert_eq!(
            trusted.next_chunk(),
            Ok(Some(PolynomialChunk::<i128>::U64(vec![0, 0])))
        );
        assert_eq!(trusted.next_chunk(), Ok(None::<PolynomialChunk<i128>>));

        let untrusted_result =
            witness.committed_stream(JoltCommittedPolynomial::UntrustedAdvice, 5);
        assert!(
            untrusted_result.is_ok(),
            "untrusted advice stream failed: {untrusted_result:?}"
        );
        let Ok(mut untrusted) = untrusted_result else {
            return;
        };
        assert_eq!(
            untrusted.next_chunk(),
            Ok(Some(PolynomialChunk::<i128>::U64(vec![0xbbaa, 0, 0, 0, 0])))
        );
    }

    #[test]
    fn advice_streams_reject_disabled_and_oversized_advice() {
        let program = JoltProgram::default();
        let preprocessing = preprocessing();
        let device = JoltDevice {
            trusted_advice: vec![0; 65],
            ..Default::default()
        };
        let inputs =
            JoltVmWitnessInputs::new(&program, &preprocessing, trace_output_with_device(device));
        let witness = TraceBackedJoltVmWitness::new(config().include_trusted_advice(true), inputs);

        assert!(matches!(
            witness.committed_stream(JoltCommittedPolynomial::TrustedAdvice, 1),
            Err(WitnessError::InvalidWitnessData {
                namespace: "jolt_vm",
                ..
            })
        ));

        let inputs = JoltVmWitnessInputs::new(
            &program,
            &preprocessing,
            trace_output_with_device(Default::default()),
        );
        let disabled = TraceBackedJoltVmWitness::new(config(), inputs);
        assert!(matches!(
            disabled.committed_stream(JoltCommittedPolynomial::TrustedAdvice, 1),
            Err(WitnessError::UnknownOracle {
                namespace: "jolt_vm",
            })
        ));
    }

    #[test]
    fn committed_batch_stream_rejects_advice_until_variable_length_batches_land() {
        let program = JoltProgram::default();
        let preprocessing = preprocessing();
        let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output());
        let witness = TraceBackedJoltVmWitness::new(config().include_trusted_advice(true), inputs);

        assert!(matches!(
            witness.committed_batch_stream(&[JoltCommittedPolynomial::TrustedAdvice], 1),
            Err(WitnessError::UnsupportedView {
                view: "batched Jolt VM advice streams",
            })
        ));
    }

    #[test]
    fn committed_stream_rejects_unsupported_oracles_and_empty_chunks() {
        let program = JoltProgram::default();
        let preprocessing = preprocessing();
        let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output());
        let witness = TraceBackedJoltVmWitness::new(config(), inputs);

        assert!(matches!(
            witness.committed_stream(JoltCommittedPolynomial::TrustedAdvice, 1),
            Err(WitnessError::UnknownOracle {
                namespace: "jolt_vm",
            })
        ));
        assert!(matches!(
            witness.committed_stream(JoltCommittedPolynomial::RdInc, 0),
            Err(WitnessError::InvalidDimensions { .. })
        ));
    }
}
