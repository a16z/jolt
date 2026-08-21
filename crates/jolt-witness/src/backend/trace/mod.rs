//! The trace-backed witness backend: derives every served oracle from an
//! execution trace via the atomic extractors in [`crate::witnesses`].

use jolt_claims::protocols::jolt::{
    geometry::{committed_openings, dimensions::REGISTER_ADDRESS_BITS, ra::JoltRaPolynomialLayout},
    JoltCommittedPolynomial, JoltFormulaDimensions, JoltOneHotConfig, JoltVirtualPolynomial,
};
use jolt_field::Field;
use jolt_lookup_tables::LookupTableKind;
use jolt_program::{
    execution::{JoltProgram, RamAccess, TraceOutput, TraceRow, TraceSource},
    preprocess::JoltProgramPreprocessing,
};
use jolt_riscv::{
    CapturedState, CircuitFlags, Flags, JoltInstruction, JoltTraceRow, LoadState, NonMemoryState,
    StoreState,
};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::backend::ProgramSource;
use crate::witnesses::ram_access_address;
use crate::{WitnessError, JOLT_VM_LABEL, RV64_XLEN};

mod advice;
mod cycle;
mod oracle;
mod ram;
mod registers;

pub const RV64_LOOKUP_ADDRESS_BITS: usize = 128;

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

pub struct JoltVmWitnessInputs<T: TraceSource> {
    pub program: Arc<JoltProgram>,
    pub preprocessing: Arc<JoltProgramPreprocessing>,
    pub trace: TraceOutput<T>,
}

impl<T: TraceSource> JoltVmWitnessInputs<T> {
    pub fn new(
        program: &Arc<JoltProgram>,
        preprocessing: &Arc<JoltProgramPreprocessing>,
        trace: TraceOutput<T>,
    ) -> Self {
        Self {
            program: Arc::clone(program),
            preprocessing: Arc::clone(preprocessing),
            trace,
        }
    }
}

pub struct TraceBackend<T: TraceSource> {
    pub config: JoltVmWitnessConfig,
    pub program: Arc<JoltProgram>,
    pub preprocessing: Arc<JoltProgramPreprocessing>,
    pub trace: TraceOutput<Arc<Vec<JoltTraceRow>>>,
    #[cfg(feature = "field-inline")]
    pub(crate) raw_trace_rows: Arc<Vec<TraceRow>>,
    source: PhantomData<fn() -> T>,
    #[cfg(feature = "field-inline")]
    pub(crate) field_inline: Option<crate::field_inline::TraceBackedFieldInlineWitness>,
}

impl<T: TraceSource> ProgramSource for TraceBackend<T> {
    fn program_preprocessing(&self) -> &JoltProgramPreprocessing {
        &self.preprocessing
    }
}

impl<T: TraceSource> TraceBackend<T> {
    /// Constructs a backend from a trace produced against `inputs.preprocessing`.
    ///
    /// Panics when the trace violates that producer contract. Use
    /// [`Self::try_new`] when the trace is not trusted.
    #[expect(
        clippy::panic,
        reason = "compatibility constructor for trusted prover-generated traces"
    )]
    pub fn new(config: JoltVmWitnessConfig, inputs: JoltVmWitnessInputs<T>) -> Self {
        match Self::try_new(config, inputs) {
            Ok(backend) => backend,
            Err(error) => panic!("invalid proof-facing trace: {error}"),
        }
    }

    pub fn try_new(
        config: JoltVmWitnessConfig,
        inputs: JoltVmWitnessInputs<T>,
    ) -> Result<Self, WitnessError> {
        let TraceOutput {
            trace: mut source,
            device,
            final_memory,
            advice_tape,
        } = inputs.trace;
        let mut trace_rows = Vec::new();
        let mut trailing_padding = 0;
        #[cfg(feature = "field-inline")]
        let mut raw_rows = Vec::new();
        while let Some(row) = source.next_row() {
            let compact = compact_trace_row(&row, &inputs.preprocessing)?;
            if compact == JoltTraceRow::default() {
                trailing_padding += 1;
            } else {
                trace_rows.resize(trace_rows.len() + trailing_padding, JoltTraceRow::default());
                trailing_padding = 0;
                trace_rows.push(compact);
            }
            #[cfg(feature = "field-inline")]
            raw_rows.push(row);
        }
        let trace = TraceOutput::new(Arc::new(trace_rows), device, final_memory, advice_tape);
        let backend = Self {
            config,
            program: inputs.program,
            preprocessing: inputs.preprocessing,
            trace,
            #[cfg(feature = "field-inline")]
            raw_trace_rows: Arc::new(raw_rows),
            source: PhantomData,
            #[cfg(feature = "field-inline")]
            field_inline: None,
        };
        Ok(backend)
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
                label: JOLT_VM_LABEL,
                reason: error.to_string(),
            }
        })
    }

    fn trace_log_rows(&self) -> usize {
        self.config.log_t
    }

    fn ram_log_k(&self) -> Result<usize, WitnessError> {
        if self.config.ram_k == 0 || !self.config.ram_k.is_power_of_two() {
            return Err(WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: format!(
                    "ram_k must be a nonzero power of two, got {}",
                    self.config.ram_k
                ),
            });
        }
        Ok(self.config.ram_k.ilog2() as usize)
    }

    fn ram_read_write_log_rows(&self) -> Result<usize, WitnessError> {
        self.config
            .log_t
            .checked_add(self.ram_log_k()?)
            .ok_or_else(|| WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: "RAM read-write rows overflow".to_owned(),
            })
    }

    fn register_read_write_log_rows(&self) -> Result<usize, WitnessError> {
        self.config
            .log_t
            .checked_add(REGISTER_ADDRESS_BITS)
            .ok_or_else(|| WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: "register read-write rows overflow".to_owned(),
            })
    }

    fn one_hot_log_rows(&self) -> Result<usize, WitnessError> {
        self.config
            .log_t
            .checked_add(self.config.one_hot.committed_chunk_bits())
            .ok_or_else(|| WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: "one-hot committed rows overflow".to_owned(),
            })
    }

    fn instruction_virtual_ra_log_rows(&self) -> Result<usize, WitnessError> {
        self.config
            .log_t
            .checked_add(self.config.one_hot.lookup_virtual_chunk_bits())
            .ok_or_else(|| WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: "instruction virtual RA rows overflow".to_owned(),
            })
    }

    fn instruction_virtual_ra_count(&self) -> Result<usize, WitnessError> {
        let chunk_bits = self.config.one_hot.lookup_virtual_chunk_bits();
        if chunk_bits == 0 || !RV64_LOOKUP_ADDRESS_BITS.is_multiple_of(chunk_bits) {
            return Err(WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: format!(
                    "lookup virtual chunk bits {chunk_bits} must evenly divide {RV64_LOOKUP_ADDRESS_BITS}"
                ),
            });
        }
        Ok(RV64_LOOKUP_ADDRESS_BITS / chunk_bits)
    }

    fn advice_log_rows(max_bytes: usize) -> usize {
        advice::advice_words(max_bytes).ilog2() as usize
    }
}

fn compact_trace_row(
    row: &TraceRow,
    preprocessing: &JoltProgramPreprocessing,
) -> Result<JoltTraceRow, WitnessError> {
    let register = row.registers;
    let instruction = JoltInstruction::try_from(row.instruction).map_err(|kind| {
        WitnessError::InvalidWitnessData {
            label: JOLT_VM_LABEL,
            reason: format!("unsupported Jolt instruction kind in trace row: {kind:?}"),
        }
    })?;
    let circuit_flags = instruction.circuit_flags();
    let rs1_value = register.rs1.map_or(0, |value| value.value);
    let rs2_value = register.rs2.map_or(0, |value| value.value);
    let rd_pre_value = register.rd.map_or(0, |value| value.pre_value);
    let rd_write_value = register.rd.map_or(0, |value| value.post_value);
    let state = if circuit_flags[CircuitFlags::Load] {
        let RamAccess::Read(read) = row.ram_access else {
            return Err(invalid_compact_row(
                row,
                "load instruction is missing its RAM read",
            ));
        };
        if rs2_value != 0 || read.value != rd_write_value {
            return Err(invalid_compact_row(
                row,
                "load values do not satisfy RamReadValue = RamWriteValue = RdWriteValue",
            ));
        }
        CapturedState::Load(LoadState {
            rs1_value,
            ram_address: read.address,
            rd_pre_value,
            rd_write_value,
        })
    } else if circuit_flags[CircuitFlags::Store] {
        let RamAccess::Write(write) = row.ram_access else {
            return Err(invalid_compact_row(
                row,
                "store instruction is missing its RAM write",
            ));
        };
        if rd_pre_value != 0 || rd_write_value != 0 || write.post_value != rs2_value {
            return Err(invalid_compact_row(
                row,
                "store values do not satisfy RamWriteValue = Rs2Value and no rd write",
            ));
        }
        CapturedState::Store(StoreState {
            rs1_value,
            rs2_value,
            ram_read_value: write.pre_value,
            ram_address: write.address,
        })
    } else {
        if row.ram_access != RamAccess::NoOp {
            return Err(invalid_compact_row(
                row,
                "non-memory instruction carries RAM access data",
            ));
        }
        CapturedState::NonMemory(NonMemoryState {
            rs1_value,
            rs2_value,
            rd_pre_value,
            rd_write_value,
        })
    };
    let pc = preprocessing
        .bytecode
        .get_pc(&row.instruction)
        .ok_or_else(|| WitnessError::InvalidWitnessData {
            label: JOLT_VM_LABEL,
            reason: format!(
                "bytecode preprocessing is missing PC mapping for address {:#x} with virtual_sequence_remaining {:?}",
                row.instruction.address, row.instruction.virtual_sequence_remaining
            ),
        })?;
    let pc = u32::try_from(pc).map_err(|_| WitnessError::InvalidWitnessData {
        label: JOLT_VM_LABEL,
        reason: format!("bytecode PC {pc} does not fit the compact trace row"),
    })?;
    JoltTraceRow::from_components(state, &row.instruction, pc).map_err(|error| {
        WitnessError::InvalidWitnessData {
            label: JOLT_VM_LABEL,
            reason: error.to_string(),
        }
    })
}

fn invalid_compact_row(row: &TraceRow, reason: &'static str) -> WitnessError {
    WitnessError::InvalidWitnessData {
        label: JOLT_VM_LABEL,
        reason: format!("{reason} for {:?}", row.instruction.instruction_kind),
    }
}

pub(crate) fn checked_pow2(log_rows: usize) -> Result<usize, WitnessError> {
    if log_rows >= usize::BITS as usize {
        return Err(WitnessError::InvalidDimensions {
            label: JOLT_VM_LABEL,
            reason: "witness row count overflow".to_owned(),
        });
    }
    1_usize
        .checked_shl(log_rows as u32)
        .ok_or_else(|| WitnessError::InvalidDimensions {
            label: JOLT_VM_LABEL,
            reason: "witness row count overflow".to_owned(),
        })
}

fn require_index(index: usize, len: usize) -> Result<(), WitnessError> {
    if index < len {
        Ok(())
    } else {
        Err(WitnessError::UnknownOracle {
            label: JOLT_VM_LABEL,
        })
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests;
