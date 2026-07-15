//! The trace-backed witness backend: derives every served oracle from an
//! execution trace via the atomic extractors in [`crate::witnesses`].

use jolt_claims::protocols::jolt::{
    geometry::{committed_openings, dimensions::REGISTER_ADDRESS_BITS, ra::JoltRaPolynomialLayout},
    JoltCommittedPolynomial, JoltFormulaDimensions, JoltOneHotConfig, JoltPolynomialId,
    JoltVirtualPolynomial,
};
use jolt_field::Field;
use jolt_lookup_tables::LookupTableKind;
use jolt_program::{
    execution::{JoltProgram, RamAccess, TraceOutput, TraceRow, TraceSource},
    preprocess::JoltProgramPreprocessing,
};

use crate::witnesses::ram_access_address;
use crate::{WitnessError, JOLT_VM_LABEL, RV64_XLEN};

mod committed;
mod cycle;
mod oracle;
mod ra;
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

pub struct TraceBackend<'a, T: TraceSource> {
    pub config: JoltVmWitnessConfig,
    pub program: &'a JoltProgram,
    pub preprocessing: &'a JoltProgramPreprocessing,
    pub trace: TraceOutput<T>,
    #[cfg(feature = "field-inline")]
    pub(crate) field_inline: Option<crate::field_inline::TraceBackedFieldInlineWitness<'a>>,
}

impl<'a, T: TraceSource> TraceBackend<'a, T> {
    pub fn new(config: JoltVmWitnessConfig, inputs: JoltVmWitnessInputs<'a, T>) -> Self {
        Self {
            config,
            program: inputs.program,
            preprocessing: inputs.preprocessing,
            trace: inputs.trace,
            #[cfg(feature = "field-inline")]
            field_inline: None,
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

    fn advice_log_rows(words: usize) -> usize {
        words.next_power_of_two().max(1).ilog2() as usize
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
mod tests;
