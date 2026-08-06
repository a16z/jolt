#[cfg(any(test, feature = "test-utils"))]
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use jolt_field::AkitaField;
use jolt_openings::CommitmentScheme;

use super::booleanity::{BooleanityAddressMetalConfig, BooleanityMetalConfig};
use super::bytecode_read_raf::BytecodeReadRafMetalConfig;
use super::hamming_weight_claim_reduction::HammingWeightMetalConfig;
use super::instruction_input::InstructionInputMetalConfig;
use super::instruction_ra_virtualization::InstructionRaVirtualizationMetalConfig;
use super::instruction_read_raf::InstructionReadRafMetalConfig;
use super::ram_raf_evaluation::RamRafEvaluationMetalConfig;
use super::registers_val_evaluation::RegistersValEvaluationMetalConfig;
#[cfg(feature = "test-utils")]
use super::solinas::OuterKernelArtifact;
use super::solinas::{MetalError, SolinasMetal};
use super::spartan_outer::{SpartanOuterRemainderMetalConfig, SpartanOuterUniskipMetalConfig};
use crate::JoltBackend;

/// Tuning values for all currently implemented Metal slots.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MetalConfig {
    /// Stage-1 Spartan outer uni-skip settings.
    pub spartan_outer_uniskip: SpartanOuterUniskipMetalConfig,
    /// Stage-1 Spartan outer remainder settings.
    pub spartan_outer_remainder: SpartanOuterRemainderMetalConfig,
    /// Stage-3 instruction-input virtualization settings.
    pub instruction_input: InstructionInputMetalConfig,
    /// Stage-5 instruction read-RAF settings.
    pub instruction_read_raf: InstructionReadRafMetalConfig,
    /// Stage-5 registers value-evaluation settings.
    pub registers_val_evaluation: RegistersValEvaluationMetalConfig,
    /// Stage-2 RAM RAF-evaluation settings.
    pub ram_raf_evaluation: RamRafEvaluationMetalConfig,
    /// Stage-6a Booleanity address settings.
    pub booleanity_address: BooleanityAddressMetalConfig,
    /// Stage-6b Booleanity cycle settings.
    pub booleanity_cycle: BooleanityMetalConfig,
    /// Stage-6b bytecode read-RAF cycle settings.
    pub bytecode_read_raf_cycle: BytecodeReadRafMetalConfig,
    /// Stage-6b instruction RA virtualization settings.
    pub instruction_ra_virtualization: InstructionRaVirtualizationMetalConfig,
    /// Stage-7 Hamming-weight claim-reduction settings.
    pub hamming_weight_claim_reduction: HammingWeightMetalConfig,
}

/// Shared Metal device state used by the installed sumcheck slots.
#[derive(Clone)]
pub struct MetalBackend {
    pub(super) context: Arc<SolinasMetal>,
    pub(super) config: MetalConfig,
    #[cfg(any(test, feature = "test-utils"))]
    pub(super) hamming_dispatches: Arc<AtomicUsize>,
    #[cfg(any(test, feature = "test-utils"))]
    pub(super) outer_remainder_sequences: Arc<AtomicUsize>,
    #[cfg(any(test, feature = "test-utils"))]
    pub(super) registers_val_sequences: Arc<AtomicUsize>,
}

impl MetalBackend {
    /// Compiles the Akita field library and validates the hybrid cutoffs.
    pub fn new(config: MetalConfig) -> Result<Self, MetalError> {
        Self::validate_config(&config)?;
        Ok(Self::with_context(&config, SolinasMetal::for_akita()?))
    }

    #[cfg(feature = "test-utils")]
    #[doc(hidden)]
    pub fn new_with_outer_artifact(
        config: MetalConfig,
        artifact: &OuterKernelArtifact,
    ) -> Result<Self, MetalError> {
        if config.spartan_outer_remainder.dispatch.binding_plan != artifact.binding_plan() {
            return Err(MetalError::OuterArtifactBindingPlanMismatch);
        }
        Self::validate_config(&config)?;
        Ok(Self::with_context(
            &config,
            SolinasMetal::for_akita_with_outer_artifact(artifact)?,
        ))
    }

    fn validate_config(config: &MetalConfig) -> Result<(), MetalError> {
        let remainder_trace_cutoff = config.spartan_outer_remainder.trace_cutoff_elements;
        if remainder_trace_cutoff < 4 || !remainder_trace_cutoff.is_power_of_two() {
            return Err(MetalError::InvalidHybridCutoff(remainder_trace_cutoff));
        }
        if config.spartan_outer_remainder.dispatch.max_threadgroups == 0 {
            return Err(MetalError::InvalidOuterRemainderConfig(
                "max_threadgroups must be nonzero",
            ));
        }
        let cutoff = config.instruction_read_raf.cutoff_elements;
        if cutoff < 2 || !cutoff.is_power_of_two() {
            return Err(MetalError::InvalidHybridCutoff(cutoff));
        }
        let address_cutoff = config.instruction_read_raf.address_cutoff_elements;
        if address_cutoff < 2 || !address_cutoff.is_power_of_two() {
            return Err(MetalError::InvalidHybridCutoff(address_cutoff));
        }
        for cutoff in [
            config.spartan_outer_uniskip.trace_cutoff_elements,
            config.spartan_outer_remainder.dispatch.cpu_tail_elements,
            config.instruction_input.trace_cutoff_elements,
            config.instruction_input.cutoff_elements,
            config.registers_val_evaluation.trace_cutoff_elements,
            config.registers_val_evaluation.cutoff_elements,
            config.ram_raf_evaluation.dispatch.trace_cutoff,
            config.booleanity_address.trace_cutoff_elements,
            config.booleanity_cycle.trace_cutoff_elements,
            config.booleanity_cycle.cutoff_elements,
            config.bytecode_read_raf_cycle.trace_cutoff_elements,
            config.bytecode_read_raf_cycle.cutoff_elements,
            config.instruction_ra_virtualization.trace_cutoff_elements,
            config.instruction_ra_virtualization.cutoff_elements,
            config.hamming_weight_claim_reduction.trace_cutoff_elements,
        ] {
            if cutoff < 2 || !cutoff.is_power_of_two() {
                return Err(MetalError::InvalidHybridCutoff(cutoff));
            }
        }
        let instruction_ra_cutoff = config.instruction_ra_virtualization.trace_cutoff_elements;
        if instruction_ra_cutoff < address_cutoff {
            return Err(MetalError::InstructionRaRequiresAddressPlane {
                instruction_ra_cutoff,
                address_cutoff,
            });
        }
        Ok(())
    }

    fn with_context(config: &MetalConfig, context: SolinasMetal) -> Self {
        Self {
            context: Arc::new(context),
            config: *config,
            #[cfg(any(test, feature = "test-utils"))]
            hamming_dispatches: Arc::new(AtomicUsize::new(0)),
            #[cfg(any(test, feature = "test-utils"))]
            outer_remainder_sequences: Arc::new(AtomicUsize::new(0)),
            #[cfg(any(test, feature = "test-utils"))]
            registers_val_sequences: Arc::new(AtomicUsize::new(0)),
        }
    }

    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
    pub fn hamming_dispatches(&self) -> usize {
        self.hamming_dispatches
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
    pub fn outer_remainder_sequences(&self) -> usize {
        self.outer_remainder_sequences
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
    pub fn registers_val_sequences(&self) -> usize {
        self.registers_val_sequences
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl<PCS> JoltBackend<AkitaField, PCS>
where
    PCS: CommitmentScheme<Field = AkitaField>,
{
    /// Replaces implemented optimized slots with their Metal counterparts.
    pub fn with_metal_compute(mut self, metal: &MetalBackend) -> Self {
        self.spartan_outer_uniskip = Box::new(metal.clone());
        self.spartan_outer_remainder = Box::new(metal.clone());
        self.instruction_input = Box::new(metal.clone());
        self.ram_raf_evaluation = Box::new(metal.clone());
        self.instruction_read_raf = Box::new(metal.clone());
        self.booleanity_address = Box::new(metal.clone());
        self.bytecode_read_raf_cycle = Box::new(metal.clone());
        self.booleanity_cycle = Box::new(metal.clone());
        self.instruction_ra_virtualization = Box::new(metal.clone());
        self.hamming_weight_claim_reduction = Box::new(metal.clone());
        self
    }
}
