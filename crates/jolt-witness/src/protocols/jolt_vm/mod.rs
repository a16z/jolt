use std::collections::HashMap;

use jolt_claims::protocols::jolt::{
    formulas::{committed_openings, dimensions::REGISTER_ADDRESS_BITS, ra::JoltRaPolynomialLayout},
    JoltCommittedPolynomial, JoltFormulaDimensions, JoltOneHotConfig,
    JoltOpeningId, JoltPolynomialId, JoltPublicId, JoltVirtualPolynomial,
};
use jolt_field::{
    signed::{S128, S64},
    Field,
};
use jolt_lookup_tables::{InstructionLookupTable, JoltLookupQuery, LookupQuery, LookupTableKind};
use jolt_poly::{eq_index_msb, EqPolynomial};
use jolt_program::{
    execution::{JoltProgram, RamAccess, TraceOutput, TraceRow, TraceSource},
    preprocess::JoltProgramPreprocessing,
};

use self::lookup::instruction_lookup_index;
use jolt_riscv::{
    CircuitFlags, Flags, InstructionFlags, InterleavedBitsMarker, JoltInstruction,
    JoltInstructionKind,
};
use rayon::prelude::*;

use crate::{
    MaterializationPolicy, NamespaceId, OracleDescriptor, OracleKind, OracleRef,
    PolynomialBatchChunk, PolynomialBatchStream, PolynomialChunk, PolynomialEncoding,
    PolynomialStream, PolynomialView, RetentionHint, ViewRequirement, WitnessDimensions,
    WitnessError, WitnessNamespace,
};

pub mod rv64;
pub mod stage5;

#[cfg(feature = "field-inline")]
pub mod field_inline;

mod lookup;

pub use stage5::{JoltVmStage5InstructionReadRafRows, Stage5InstructionReadRafRow};

mod provider;
mod ra;
mod ram;
mod registers;
mod spartan_outer;
mod stage2;
mod stage3;
mod stage6;
mod streams;
mod trace;

pub use ra::{
    RaFamilyCycleIndexSource, RaFamilyCycleIndices, RA_FAMILY_MAX_BYTECODE_CHUNKS,
    RA_FAMILY_MAX_INSTRUCTION_CHUNKS, RA_FAMILY_MAX_RAM_CHUNKS,
};
pub use registers::{
    JoltVmRegisterRead, JoltVmRegisterReadWriteRow, JoltVmRegisterReadWriteRows,
    JoltVmRegisterWrite,
};
pub use spartan_outer::{JoltVmSpartanOuterRow, JoltVmSpartanOuterRows};
pub use stage2::{JoltVmStage2Rows, JoltVmStage2TraceRow};
pub use stage3::{
    JoltVmStage3InstructionRegisterRow, JoltVmStage3InstructionRegisterRows, JoltVmStage3ShiftRow,
    JoltVmStage3ShiftRows,
};
pub use stage6::{JoltVmStage6Row, JoltVmStage6Rows};
pub use streams::{JoltVmCommittedBatchStream, JoltVmCommittedStream};

pub(crate) use ra::RaChunkSelector;
pub(crate) use ram::ram_access_address;
pub(crate) use streams::{JoltVmCommittedStreamKind, JoltVmIncrementStreamKind};
pub(crate) use trace::{
    missing_pc_mapping, row_instruction_flags, row_is_noop, supported_trace_virtual, PcLookupCache,
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
    type ChallengeId = JoltPublicId;

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

pub struct TraceBackedJoltVmWitness<'a, T: TraceSource> {
    pub config: JoltVmWitnessConfig,
    pub program: &'a JoltProgram,
    pub preprocessing: &'a JoltProgramPreprocessing,
    pub trace: TraceOutput<T>,
    #[cfg(feature = "field-inline")]
    field_inline: Option<field_inline::TraceBackedFieldInlineWitness<'a>>,
}

impl<'a, T: TraceSource> TraceBackedJoltVmWitness<'a, T> {
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
                namespace: JOLT_VM_NAMESPACE.name,
                reason: error.to_string(),
            }
        })
    }

    fn trace_dimensions(&self) -> Result<WitnessDimensions, WitnessError> {
        Ok(WitnessDimensions::new(self.config.log_t))
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
        Ok(WitnessDimensions::new(log_rows))
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
        Ok(WitnessDimensions::new(log_rows))
    }

    fn ram_final_dimensions(&self) -> Result<WitnessDimensions, WitnessError> {
        let log_rows = self.ram_log_k()?;
        Ok(WitnessDimensions::new(log_rows))
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
        Ok(WitnessDimensions::new(log_rows))
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
        Ok(WitnessDimensions::new(log_rows))
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
        WitnessDimensions::new(rows.ilog2() as usize)
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

fn eq_evals_msb<F: Field>(point: &[F]) -> Result<Vec<F>, WitnessError> {
    let _ = checked_pow2(point.len())?;
    Ok(EqPolynomial::evals(point, None))
}

#[cfg(test)]
mod tests;
