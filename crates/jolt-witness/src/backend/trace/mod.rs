//! The trace-backed witness backend: derives every served oracle from an
//! execution trace via the atomic extractors in [`crate::witnesses`].

use jolt_claims::protocols::jolt::{
    geometry::{committed_openings, dimensions::REGISTER_ADDRESS_BITS, ra::JoltRaPolynomialLayout},
    JoltCommittedPolynomial, JoltFormulaDimensions, JoltOneHotConfig, JoltVirtualPolynomial,
};
use jolt_field::JoltField;
use jolt_lookup_tables::LookupTableKind;
use jolt_program::{
    execution::{JoltProgram, OwnedTrace, TraceData, TraceOutput},
    preprocess::JoltProgramPreprocessing,
};

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
            log_t,
            ram_k,
            one_hot,
            include_trusted_advice: false,
            include_untrusted_advice: false,
        }
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

pub struct JoltVmWitnessInputs<T> {
    pub program: Arc<JoltProgram>,
    pub preprocessing: Arc<JoltProgramPreprocessing>,
    pub trace: TraceOutput<T>,
}

impl<T> JoltVmWitnessInputs<T> {
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

/// Retains the producer's rows and payloads without converting or copying them.
pub struct TraceBackend {
    pub config: JoltVmWitnessConfig,
    pub program: Arc<JoltProgram>,
    pub preprocessing: Arc<JoltProgramPreprocessing>,
    pub trace: TraceOutput<Arc<TraceData>>,
    #[cfg(feature = "field-inline")]
    pub(crate) field_inline: Option<crate::field_inline::TraceBackedFieldInlineWitness>,
}

impl ProgramSource for TraceBackend {
    fn program_preprocessing(&self) -> &JoltProgramPreprocessing {
        &self.preprocessing
    }
}

impl TraceBackend {
    #[expect(
        clippy::panic,
        reason = "infallible convenience constructor for trusted fixtures"
    )]
    pub fn new(config: JoltVmWitnessConfig, inputs: JoltVmWitnessInputs<OwnedTrace>) -> Self {
        match Self::try_new(config, inputs) {
            Ok(backend) => backend,
            Err(error) => panic!("invalid trace: {error}"),
        }
    }

    /// Transfers a complete retained trace. Iterator-only sources must remain
    /// on the streaming execution interface rather than being silently drained.
    pub fn try_new(
        config: JoltVmWitnessConfig,
        inputs: JoltVmWitnessInputs<OwnedTrace>,
    ) -> Result<Self, WitnessError> {
        let cycles = checked_pow2(config.log_t)?;
        let TraceOutput {
            trace,
            device,
            final_memory,
            advice_tape,
        } = inputs.trace;
        let trace = trace
            .into_data()
            .map_err(|error| WitnessError::InvalidWitnessData {
                label: JOLT_VM_LABEL,
                reason: error.to_string(),
            })?;
        if trace.proof_len() > cycles {
            return Err(WitnessError::InvalidWitnessData {
                label: JOLT_VM_LABEL,
                reason: format!(
                    "physical trace has {} rows but the cycle domain has {cycles}",
                    trace.proof_len()
                ),
            });
        }
        Ok(Self {
            config,
            program: inputs.program,
            preprocessing: inputs.preprocessing,
            trace: TraceOutput::new(trace, device, final_memory, advice_tape),
            #[cfg(feature = "field-inline")]
            field_inline: None,
        })
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

/// Upper bound, in bytes, on a dense `(K × T)` oracle grid materialized by
/// the trace backend: the RAM and register read-write grids and the one-hot
/// RA grids. The request grows linearly with the trace length and reaches
/// hundreds of GiB at profiling scales (`ram_K = 4096`, `log_T = 22`, 32-byte
/// field: 2^39 bytes). Past this bound the global allocator aborts the
/// process with an opaque `memory allocation of N bytes failed`; refusing
/// with a `WitnessError` keeps the failure actionable. 32 GiB admits every
/// in-tree test and the fibonacci profiling default (scale 16); the larger
/// documented profiling defaults are refused by design and belong on the
/// optimized backend. On narrower targets, the allocation limit is capped
/// at `isize::MAX` bytes instead.
pub(crate) const MAX_DENSE_GRID_BYTES: usize = match 1_usize.checked_shl(35) {
    Some(bytes) => bytes,
    None => isize::MAX as usize,
};

/// The element count of a dense `addresses × cycles` grid of `F`, refused
/// with an actionable error when the byte size overflows or exceeds
/// [`MAX_DENSE_GRID_BYTES`].
pub(crate) fn checked_dense_grid_len<F>(
    addresses: usize,
    cycles: usize,
) -> Result<usize, WitnessError> {
    let len = addresses
        .checked_mul(cycles)
        .ok_or_else(|| WitnessError::InvalidDimensions {
            label: JOLT_VM_LABEL,
            reason: format!("dense grid of {addresses} addresses x {cycles} cycles overflows"),
        })?;
    let bytes = len.checked_mul(core::mem::size_of::<F>());
    match bytes {
        Some(bytes) if bytes <= MAX_DENSE_GRID_BYTES => Ok(len),
        _ => Err(WitnessError::InvalidDimensions {
            label: JOLT_VM_LABEL,
            reason: format!(
                "dense grid of {addresses} addresses x {cycles} cycles needs {} bytes \
                 (> {MAX_DENSE_GRID_BYTES} max); this naive materialization is a test \
                 oracle sized for small traces — use the optimized backend for larger \
                 shapes",
                bytes.map_or_else(|| "overflowing".to_owned(), |bytes| bytes.to_string()),
            ),
        }),
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
#[expect(clippy::unwrap_used, clippy::panic, reason = "test module")]
mod tests;
