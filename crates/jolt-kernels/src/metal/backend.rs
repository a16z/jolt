#[cfg(any(test, feature = "test-utils"))]
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use jolt_field::AkitaField;
use jolt_openings::CommitmentScheme;

use super::booleanity::{BooleanityAddressMetalConfig, BooleanityMetalConfig};
use super::bytecode_read_raf::{
    BytecodeReadRafAddressImplementation, BytecodeReadRafAddressMetalConfig,
    BytecodeReadRafMetalConfig,
};
use super::hamming_weight_claim_reduction::HammingWeightMetalConfig;
use super::instruction_claim_reduction::InstructionClaimReductionMetalConfig;
use super::instruction_input::{InstructionInputDenseStorageMode, InstructionInputMetalConfig};
use super::instruction_ra_virtualization::InstructionRaVirtualizationMetalConfig;
use super::instruction_read_raf::{
    InstructionReadRafAddressImplementation, InstructionReadRafMetalConfig,
};
use super::ram_hamming_booleanity::RamHammingBooleanityMetalConfig;
use super::ram_ra_claim_reduction::RamRaClaimReductionMetalConfig;
use super::ram_ra_virtualization::RamRaVirtualizationMetalConfig;
use super::ram_raf_evaluation::RamRafEvaluationMetalConfig;
use super::ram_val_check::RamValCheckMetalConfig;
use super::registers_claim_reduction::{
    RegistersClaimReductionImplementation, RegistersClaimReductionMetalConfig,
};
use super::registers_val_evaluation::{
    RegistersValEvaluationMetalConfig, RegistersValEvaluationSource,
};
#[cfg(feature = "test-utils")]
use super::solinas::OuterKernelArtifact;
use super::solinas::{
    InstructionInputStorageInitialization, MetalError, OuterRemainderStorageInitialization,
    SolinasMetal,
};
use super::spartan_outer::{SpartanOuterRemainderMetalConfig, SpartanOuterUniskipMetalConfig};
use super::spartan_product::{SpartanProductRemainderMetalConfig, SpartanProductWitnessSource};
use super::spartan_shift::SpartanShiftMetalConfig;
use crate::JoltBackend;

/// Tuning values for all currently implemented Metal slots.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MetalConfig {
    /// Stage-1 Spartan outer uni-skip settings.
    pub spartan_outer_uniskip: SpartanOuterUniskipMetalConfig,
    /// Stage-1 Spartan outer remainder settings.
    pub spartan_outer_remainder: SpartanOuterRemainderMetalConfig,
    /// Stage-2 Spartan product-remainder settings.
    pub spartan_product_remainder: SpartanProductRemainderMetalConfig,
    /// Stage-3 Spartan shift settings.
    pub spartan_shift: SpartanShiftMetalConfig,
    /// Stage-2 instruction claim-reduction settings.
    pub instruction_claim_reduction: InstructionClaimReductionMetalConfig,
    /// Stage-3 instruction-input virtualization settings.
    pub instruction_input: InstructionInputMetalConfig,
    /// Stage-3 registers claim-reduction settings.
    pub registers_claim_reduction: RegistersClaimReductionMetalConfig,
    /// Stage-5 instruction read-RAF settings.
    pub instruction_read_raf: InstructionReadRafMetalConfig,
    /// Stage-5 registers value-evaluation settings.
    pub registers_val_evaluation: RegistersValEvaluationMetalConfig,
    /// Stage-2 RAM RAF-evaluation settings.
    pub ram_raf_evaluation: RamRafEvaluationMetalConfig,
    /// Stage-4 RAM value-check settings.
    pub ram_val_check: RamValCheckMetalConfig,
    /// Stage-5 RAM RA claim-reduction settings.
    pub ram_ra_claim_reduction: RamRaClaimReductionMetalConfig,
    /// Stage-6b RAM RA virtualization settings.
    pub ram_ra_virtualization: RamRaVirtualizationMetalConfig,
    /// Stage-6b RAM Hamming-weight booleanity settings.
    pub ram_hamming_booleanity: RamHammingBooleanityMetalConfig,
    /// Stage-6a Booleanity address settings.
    pub booleanity_address: BooleanityAddressMetalConfig,
    /// Stage-6a bytecode read-RAF address settings.
    pub bytecode_read_raf_address: BytecodeReadRafAddressMetalConfig,
    /// Stage-6b Booleanity cycle settings.
    pub booleanity_cycle: BooleanityMetalConfig,
    /// Stage-6b bytecode read-RAF cycle settings.
    pub bytecode_read_raf_cycle: BytecodeReadRafMetalConfig,
    /// Stage-6b instruction RA virtualization settings.
    pub instruction_ra_virtualization: InstructionRaVirtualizationMetalConfig,
    /// Stage-7 Hamming-weight claim-reduction settings.
    pub hamming_weight_claim_reduction: HammingWeightMetalConfig,
}

impl MetalConfig {
    /// The retained Akita Metal routes used by the production prover.
    pub fn production() -> Self {
        let mut config = Self::default();
        config.bytecode_read_raf_cycle.cpu_tail_algebra =
            crate::optimized::bytecode_read_raf::BytecodeCycleAlgebra::Q10;
        config.bytecode_read_raf_address.implementation =
            BytecodeReadRafAddressImplementation::AddressMajor;
        config.bytecode_read_raf_address.trace_cutoff_elements = 1 << 26;
        config.spartan_product_remainder.witness_source =
            SpartanProductWitnessSource::SpartanStage1;
        config.spartan_product_remainder.reuse_outer_state_a = true;
        config.registers_claim_reduction.implementation =
            RegistersClaimReductionImplementation::OuterCarrierAliasHybrid;
        config
            .spartan_outer_remainder
            .dispatch
            .registers_claim_carrier = true;
        config
            .spartan_outer_remainder
            .dispatch
            .storage_initialization = OuterRemainderStorageInitialization::Full;
        config.instruction_input.dispatch.storage_initialization =
            InstructionInputStorageInitialization::Lazy;
        config.instruction_input.dense_storage_mode =
            InstructionInputDenseStorageMode::OuterResidual;
        config.registers_val_evaluation.source = RegistersValEvaluationSource::Stage1Resident;
        config.registers_val_evaluation.trace_cutoff_elements = 1 << 26;
        config
    }
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
    pub(super) product_remainder_sequences: Arc<AtomicUsize>,
    #[cfg(any(test, feature = "test-utils"))]
    pub(super) product_uniskip_dispatches: Arc<AtomicUsize>,
    #[cfg(any(test, feature = "test-utils"))]
    pub(super) product_uniskip_carrier_hits: Arc<AtomicUsize>,
    #[cfg(any(test, feature = "test-utils"))]
    pub(super) instruction_claim_sequences: Arc<AtomicUsize>,
    #[cfg(any(test, feature = "test-utils"))]
    pub(super) registers_claim_alias_sequences: Arc<AtomicUsize>,
    #[cfg(any(test, feature = "test-utils"))]
    pub(super) registers_val_sequences: Arc<AtomicUsize>,
    #[cfg(any(test, feature = "test-utils"))]
    pub(super) ram_val_sparse_sequences: Arc<AtomicUsize>,
    #[cfg(any(test, feature = "test-utils"))]
    pub(super) ram_read_write_sparse_sequences: Arc<AtomicUsize>,
    #[cfg(any(test, feature = "test-utils"))]
    pub(super) ram_ra_claim_sparse_sequences: Arc<AtomicUsize>,
    #[cfg(any(test, feature = "test-utils"))]
    pub(super) ram_ra_virtualization_sparse_sequences: Arc<AtomicUsize>,
    #[cfg(any(test, feature = "test-utils"))]
    pub(super) ram_hamming_sparse_sequences: Arc<AtomicUsize>,
}

impl MetalBackend {
    /// Creates the supported Akita Metal backend.
    pub fn production() -> Result<Self, MetalError> {
        Self::new(MetalConfig::production())
    }

    /// Compiles the Akita field library and validates the hybrid cutoffs.
    pub fn new(mut config: MetalConfig) -> Result<Self, MetalError> {
        config
            .spartan_outer_remainder
            .dispatch
            .registers_claim_carrier = config.registers_claim_reduction.implementation
            == RegistersClaimReductionImplementation::OuterCarrierAliasHybrid;
        Self::validate_config(&config)?;
        let context = if config == MetalConfig::production() {
            SolinasMetal::for_akita_production()?
        } else {
            SolinasMetal::for_akita()?
        };
        Ok(Self::with_context(&config, context))
    }

    #[cfg(feature = "test-utils")]
    #[doc(hidden)]
    pub fn new_with_outer_artifact(
        mut config: MetalConfig,
        artifact: &OuterKernelArtifact,
    ) -> Result<Self, MetalError> {
        config
            .spartan_outer_remainder
            .dispatch
            .registers_claim_carrier = config.registers_claim_reduction.implementation
            == RegistersClaimReductionImplementation::OuterCarrierAliasHybrid;
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
        let _ = config.spartan_shift.dispatch.validate()?;
        let remainder_trace_cutoff = config.spartan_outer_remainder.trace_cutoff_elements;
        if remainder_trace_cutoff < 4 || !remainder_trace_cutoff.is_power_of_two() {
            return Err(MetalError::InvalidHybridCutoff(remainder_trace_cutoff));
        }
        if config.spartan_outer_remainder.dispatch.max_threadgroups == 0 {
            return Err(MetalError::InvalidOuterRemainderConfig(
                "max_threadgroups must be nonzero",
            ));
        }
        if config.registers_claim_reduction.implementation
            == RegistersClaimReductionImplementation::OuterCarrierAliasHybrid
            && (config.spartan_outer_remainder.trace_cutoff_elements
                > config.registers_claim_reduction.trace_cutoff_elements
                || config.instruction_input.trace_cutoff_elements
                    > config.registers_claim_reduction.trace_cutoff_elements)
        {
            return Err(MetalError::InvalidOuterRemainderConfig(
                "registers-claim carrier producers must activate no later than their consumer",
            ));
        }
        if config.bytecode_read_raf_address.implementation
            == BytecodeReadRafAddressImplementation::AddressMajor
            && (config.instruction_read_raf.address_implementation
                != InstructionReadRafAddressImplementation::Stage1Grouped
                || config.instruction_read_raf.address_cutoff_elements
                    > config.bytecode_read_raf_address.trace_cutoff_elements
                || config.bytecode_read_raf_address.trace_cutoff_elements < 1 << 15)
        {
            return Err(MetalError::InvalidBytecodeReadRafAddressConfig(
                "address-major requires the Stage-1 grouped owner at every admitted trace size",
            ));
        }
        if config.registers_val_evaluation.source == RegistersValEvaluationSource::Stage1Resident
            && (config.instruction_read_raf.address_implementation
                != InstructionReadRafAddressImplementation::Stage1Grouped
                || config.instruction_read_raf.address_cutoff_elements
                    > config.registers_val_evaluation.trace_cutoff_elements
                || !(1 << 26..=1 << 28)
                    .contains(&config.registers_val_evaluation.trace_cutoff_elements)
                || config.registers_val_evaluation.cutoff_elements
                    >= config.registers_val_evaluation.trace_cutoff_elements)
        {
            return Err(MetalError::InvalidRegistersValState(
                "Stage-1 resident RegistersVal requires the grouped owner at logs 26 through 28 and a smaller tail cutoff",
            ));
        }
        if config.instruction_input.dense_storage_mode
            == InstructionInputDenseStorageMode::OuterResidual
            && config.spartan_outer_remainder.trace_cutoff_elements
                > config.instruction_input.trace_cutoff_elements
        {
            return Err(MetalError::InvalidInstructionInputState(
                "Outer-residual InstructionInput storage requires an active OuterRemainder producer",
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
            config.spartan_product_remainder.trace_cutoff_elements,
            config.spartan_product_remainder.cpu_tail_elements,
            config.spartan_shift.trace_cutoff_elements,
            config.instruction_claim_reduction.trace_cutoff_elements,
            config.instruction_input.trace_cutoff_elements,
            config.instruction_input.cutoff_elements,
            config.registers_claim_reduction.trace_cutoff_elements,
            config.registers_val_evaluation.trace_cutoff_elements,
            config.registers_val_evaluation.cutoff_elements,
            config.ram_raf_evaluation.dispatch.trace_cutoff,
            config.ram_val_check.trace_cutoff_elements,
            config.ram_ra_claim_reduction.trace_cutoff_elements,
            config.ram_ra_virtualization.trace_cutoff_elements,
            config.ram_hamming_booleanity.trace_cutoff_elements,
            config.booleanity_address.trace_cutoff_elements,
            config.bytecode_read_raf_address.trace_cutoff_elements,
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
            product_remainder_sequences: Arc::new(AtomicUsize::new(0)),
            #[cfg(any(test, feature = "test-utils"))]
            product_uniskip_dispatches: Arc::new(AtomicUsize::new(0)),
            #[cfg(any(test, feature = "test-utils"))]
            product_uniskip_carrier_hits: Arc::new(AtomicUsize::new(0)),
            #[cfg(any(test, feature = "test-utils"))]
            instruction_claim_sequences: Arc::new(AtomicUsize::new(0)),
            #[cfg(any(test, feature = "test-utils"))]
            registers_claim_alias_sequences: Arc::new(AtomicUsize::new(0)),
            #[cfg(any(test, feature = "test-utils"))]
            registers_val_sequences: Arc::new(AtomicUsize::new(0)),
            #[cfg(any(test, feature = "test-utils"))]
            ram_val_sparse_sequences: Arc::new(AtomicUsize::new(0)),
            #[cfg(any(test, feature = "test-utils"))]
            ram_read_write_sparse_sequences: Arc::new(AtomicUsize::new(0)),
            #[cfg(any(test, feature = "test-utils"))]
            ram_ra_claim_sparse_sequences: Arc::new(AtomicUsize::new(0)),
            #[cfg(any(test, feature = "test-utils"))]
            ram_ra_virtualization_sparse_sequences: Arc::new(AtomicUsize::new(0)),
            #[cfg(any(test, feature = "test-utils"))]
            ram_hamming_sparse_sequences: Arc::new(AtomicUsize::new(0)),
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
    pub fn product_remainder_sequences(&self) -> usize {
        self.product_remainder_sequences
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
    pub fn product_uniskip_dispatches(&self) -> usize {
        self.product_uniskip_dispatches
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
    pub fn product_uniskip_carrier_hits(&self) -> usize {
        self.product_uniskip_carrier_hits
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
    pub fn instruction_claim_sequences(&self) -> usize {
        self.instruction_claim_sequences
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
    pub fn registers_claim_alias_sequences(&self) -> usize {
        self.registers_claim_alias_sequences
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
    pub fn registers_val_sequences(&self) -> usize {
        self.registers_val_sequences
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
    pub fn ram_val_sparse_sequences(&self) -> usize {
        self.ram_val_sparse_sequences
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
    pub fn ram_read_write_sparse_sequences(&self) -> usize {
        self.ram_read_write_sparse_sequences
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
    pub fn ram_ra_claim_sparse_sequences(&self) -> usize {
        self.ram_ra_claim_sparse_sequences
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
    pub fn ram_ra_virtualization_sparse_sequences(&self) -> usize {
        self.ram_ra_virtualization_sparse_sequences
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
    pub fn ram_hamming_sparse_sequences(&self) -> usize {
        self.ram_hamming_sparse_sequences
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
        self.spartan_product_uniskip = Box::new(metal.clone());
        self.spartan_product_remainder = Box::new(metal.clone());
        self.spartan_shift = Box::new(metal.clone());
        self.instruction_claim_reduction = Box::new(metal.clone());
        self.instruction_input = Box::new(metal.clone());
        self.registers_claim_reduction = Box::new(metal.clone());
        self.registers_read_write = Box::new(metal.clone());
        self.registers_val_evaluation = Box::new(metal.clone());
        self.ram_read_write = Box::new(metal.clone());
        self.ram_raf_evaluation = Box::new(metal.clone());
        self.ram_val_check = Box::new(metal.clone());
        self.ram_ra_claim_reduction = Box::new(metal.clone());
        self.ram_ra_virtualization = Box::new(metal.clone());
        self.ram_hamming_booleanity = Box::new(metal.clone());
        self.instruction_read_raf = Box::new(metal.clone());
        self.booleanity_address = Box::new(metal.clone());
        self.bytecode_read_raf_address = Box::new(metal.clone());
        self.bytecode_read_raf_cycle = Box::new(metal.clone());
        self.booleanity_cycle = Box::new(metal.clone());
        self.instruction_ra_virtualization = Box::new(metal.clone());
        self.hamming_weight_claim_reduction = Box::new(metal.clone());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_profile_selects_only_retained_routes() {
        let config = MetalConfig::production();

        assert_eq!(
            config.bytecode_read_raf_address.implementation,
            BytecodeReadRafAddressImplementation::AddressMajor
        );
        assert_eq!(
            config.registers_claim_reduction.implementation,
            RegistersClaimReductionImplementation::OuterCarrierAliasHybrid
        );
        assert_eq!(
            config.spartan_product_remainder.witness_source,
            SpartanProductWitnessSource::SpartanStage1
        );
        assert_eq!(
            config.registers_val_evaluation.source,
            RegistersValEvaluationSource::Stage1Resident
        );
        assert_eq!(
            config.registers_val_evaluation.trace_cutoff_elements,
            1 << 26
        );
        assert_eq!(
            config
                .spartan_outer_remainder
                .dispatch
                .storage_initialization,
            OuterRemainderStorageInitialization::Full
        );
        assert_eq!(
            config.instruction_input.dispatch.storage_initialization,
            InstructionInputStorageInitialization::Lazy
        );
        assert_eq!(
            config.instruction_input.dense_storage_mode,
            InstructionInputDenseStorageMode::OuterResidual
        );
        assert!(MetalBackend::validate_config(&config).is_ok());
    }

    #[test]
    fn production_profile_compiles_its_minimal_metal_library() {
        let result = MetalBackend::production();
        assert!(result.is_ok(), "{:#?}", result.err());
    }

    #[test]
    fn resident_registers_val_can_use_stage1_without_metal_registers_rw() {
        let mut config = MetalConfig::default();
        config.registers_val_evaluation.source = RegistersValEvaluationSource::Stage1Resident;
        config.registers_val_evaluation.trace_cutoff_elements = 1 << 26;
        config.instruction_read_raf.address_implementation =
            InstructionReadRafAddressImplementation::Stage1Grouped;

        let validation = MetalBackend::validate_config(&config);
        assert!(validation.is_ok(), "{validation:?}");

        config.instruction_read_raf.address_cutoff_elements = 1 << 27;
        assert!(matches!(
            MetalBackend::validate_config(&config),
            Err(MetalError::InvalidRegistersValState(_))
        ));
    }

    #[test]
    fn outer_residual_instruction_input_requires_an_active_outer_producer() {
        let mut config = MetalConfig::production();
        config.spartan_outer_remainder.trace_cutoff_elements = 1 << 26;
        config.instruction_input.trace_cutoff_elements = 1 << 25;
        config.registers_claim_reduction.trace_cutoff_elements = 1 << 26;

        assert!(matches!(
            MetalBackend::validate_config(&config),
            Err(MetalError::InvalidInstructionInputState(_))
        ));
    }
}
