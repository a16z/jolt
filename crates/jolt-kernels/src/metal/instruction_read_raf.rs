use std::sync::Arc;

use jolt_field::AkitaField;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::SumcheckInputClaims;
use jolt_verifier::stages::stage5::InstructionReadRaf;
use jolt_witness::JoltWitnessPlane;

use super::booleanity::BooleanityMetalConfig;
use super::solinas::{
    AddressPhaseSequence, AddressPhaseSequenceConfig, BooleanityRows, MetalError, Product5Sequence,
    Product5SequenceConfig, SolinasMetal, PRODUCT5_FACTORS,
};
use crate::optimized::instruction_read_raf::{
    prepare_metal_instruction_read_raf, OptimizedInstructionReadRafKernel,
};
use crate::{
    JoltBackend, KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel,
    SumcheckKernelError,
};

/// Dispatch and crossover settings for the stage-5 dense cycle tail.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionReadRafMetalConfig {
    /// First trace length whose address phases run on Metal.
    pub address_cutoff_elements: usize,
    /// Dispatch geometry for the resident address sequence.
    pub address_dispatch: AddressPhaseSequenceConfig,
    /// First table length whose next round runs on the CPU.
    pub cutoff_elements: usize,
    /// Threadgroup widths for the initial message and fused transitions.
    pub dispatch: Product5SequenceConfig,
}

impl Default for InstructionReadRafMetalConfig {
    fn default() -> Self {
        Self {
            address_cutoff_elements: 1 << 18,
            address_dispatch: AddressPhaseSequenceConfig::default(),
            cutoff_elements: 1 << 16,
            dispatch: Product5SequenceConfig::default(),
        }
    }
}

/// Tuning values for all currently implemented Metal slots.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MetalConfig {
    /// Stage-5 instruction read-RAF settings.
    pub instruction_read_raf: InstructionReadRafMetalConfig,
    /// Stage-6b Booleanity cycle settings.
    pub booleanity_cycle: BooleanityMetalConfig,
}

/// Shared Metal device state used by the installed sumcheck slots.
#[derive(Clone)]
pub struct MetalBackend {
    pub(super) context: Arc<SolinasMetal>,
    pub(super) config: MetalConfig,
}

impl MetalBackend {
    /// Compiles the Akita field library and validates the hybrid cutoffs.
    pub fn new(config: MetalConfig) -> Result<Self, MetalError> {
        let cutoff = config.instruction_read_raf.cutoff_elements;
        if cutoff < 2 || !cutoff.is_power_of_two() {
            return Err(MetalError::InvalidHybridCutoff(cutoff));
        }
        let address_cutoff = config.instruction_read_raf.address_cutoff_elements;
        if address_cutoff < 2 || !address_cutoff.is_power_of_two() {
            return Err(MetalError::InvalidHybridCutoff(address_cutoff));
        }
        for cutoff in [
            config.booleanity_cycle.trace_cutoff_elements,
            config.booleanity_cycle.cutoff_elements,
        ] {
            if cutoff < 2 || !cutoff.is_power_of_two() {
                return Err(MetalError::InvalidHybridCutoff(cutoff));
            }
        }
        Ok(Self {
            context: Arc::new(SolinasMetal::for_akita()?),
            config,
        })
    }
}

impl<PCS> JoltBackend<AkitaField, PCS>
where
    PCS: CommitmentScheme<Field = AkitaField>,
{
    /// Replaces implemented optimized slots with their Metal counterparts.
    pub fn with_metal_compute(mut self, metal: &MetalBackend) -> Self {
        self.instruction_read_raf = Box::new(metal.clone());
        self.booleanity_cycle = Box::new(metal.clone());
        self
    }
}

impl PrepareKernel<AkitaField, InstructionReadRaf<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, InstructionReadRaf<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = InstructionReadRaf<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let trace_elements = 1usize << inputs.relation.dimensions().log_t();
        let use_metal_address =
            trace_elements >= self.config.instruction_read_raf.address_cutoff_elements;
        let cpu = prepare_metal_instruction_read_raf(session, witness, inputs, use_metal_address)?;
        if trace_elements >= self.config.booleanity_cycle.trace_cutoff_elements
            && session.state::<BooleanityRows>().is_none()
        {
            let rows = cpu
                .metal_prepare_booleanity_rows(&self.context)
                .map_err(KernelError::from)?;
            session.park(rows);
        }
        Ok(Box::new(MetalInstructionReadRafKernel::new(
            cpu,
            Arc::clone(&self.context),
            self.config.instruction_read_raf,
            use_metal_address,
        )?))
    }
}

pub(crate) struct MetalInstructionReadRafKernel {
    cpu: OptimizedInstructionReadRafKernel<AkitaField>,
    context: Arc<SolinasMetal>,
    config: InstructionReadRafMetalConfig,
    address_sequence: Option<AddressPhaseSequence>,
    sequence: Option<Product5Sequence>,
    host_tail: Option<[Vec<AkitaField>; PRODUCT5_FACTORS]>,
    metal_rounds: usize,
    metal_address_phases: usize,
}

impl MetalInstructionReadRafKernel {
    pub(crate) fn new(
        cpu: OptimizedInstructionReadRafKernel<AkitaField>,
        context: Arc<SolinasMetal>,
        config: InstructionReadRafMetalConfig,
        use_metal_address: bool,
    ) -> Result<Self, SumcheckError<AkitaField>> {
        let mut kernel = Self {
            cpu,
            context,
            config,
            address_sequence: None,
            sequence: None,
            host_tail: Some(std::array::from_fn(|_| {
                vec![AkitaField::zero(); config.cutoff_elements]
            })),
            metal_rounds: 0,
            metal_address_phases: 0,
        };
        if use_metal_address {
            let mut sequence = {
                let _span =
                    tracing::info_span!("MetalInstructionReadRaf::sequence_prepare").entered();
                kernel
                    .cpu
                    .metal_prepare_address_sequence(&kernel.context, config.address_dispatch)?
            };
            let (suffix_len, previous) = kernel.cpu.metal_address_phase_request()?;
            let sums = {
                let _span =
                    tracing::info_span!("MetalInstructionReadRaf::initial_address_phase").entered();
                sequence
                    .phase(suffix_len, previous.as_ref())
                    .map_err(|error| backend_error(error.to_string()))?
            };
            kernel.cpu.metal_install_address_phase(sums)?;
            kernel.metal_address_phases = 1;
            kernel.address_sequence = Some(sequence);
        }
        Ok(kernel)
    }

    #[cfg(test)]
    pub(crate) const fn metal_rounds(&self) -> usize {
        self.metal_rounds
    }

    #[cfg(test)]
    pub(crate) const fn metal_address_phases(&self) -> usize {
        self.metal_address_phases
    }

    fn install_next_address_phase(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let (suffix_len, previous) = self.cpu.metal_address_phase_request()?;
        let sequence = self
            .address_sequence
            .as_mut()
            .ok_or_else(|| backend_error("resident address sequence disappeared"))?;
        let sums = sequence
            .phase(suffix_len, previous.as_ref())
            .map_err(|error| backend_error(error.to_string()))?;
        self.cpu.metal_install_address_phase(sums)?;
        self.metal_address_phases += 1;
        Ok(())
    }

    fn restore_cpu_tail(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let _span = tracing::info_span!("MetalInstructionReadRaf::readback").entered();
        let sequence = self
            .sequence
            .take()
            .ok_or_else(|| backend_error("device sequence is absent during readback"))?;
        let mut tables = self
            .host_tail
            .take()
            .ok_or_else(|| backend_error("CPU tail buffers were already consumed"))?;
        sequence
            .read_current_factor_tables(&mut tables)
            .map_err(|error| backend_error(error.to_string()))?;
        self.cpu.metal_restore_dense(tables)
    }
}

impl ProveRounds<AkitaField> for MetalInstructionReadRafKernel {
    fn num_rounds(&self) -> usize {
        self.cpu.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<jolt_poly::UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        let mut bind = bind;
        if self.address_sequence.is_some() && self.cpu.metal_address_active() {
            let _span = tracing::info_span!("MetalInstructionReadRaf::address_round").entered();
            if let Some(challenge) = bind.take() {
                self.cpu.metal_bind_address(challenge)?;
                if self.cpu.metal_address_phase_pending() {
                    self.install_next_address_phase()?;
                }
            }
            if self.cpu.metal_address_active() {
                return self.cpu.metal_address_message(previous_claim);
            }
        }

        if self.address_sequence.is_some() && !self.cpu.metal_resident_cycle_available() {
            self.address_sequence = None;
        }

        if self.address_sequence.is_some() {
            if let Some(challenge) = bind.take() {
                let _span =
                    tracing::info_span!("MetalInstructionReadRaf::resident_handoff").entered();
                let address_sequence = self
                    .address_sequence
                    .take()
                    .ok_or_else(|| backend_error("resident address sequence disappeared"))?;
                let (sequence, q_evals) = self.cpu.metal_offload_resident_bind(
                    challenge,
                    address_sequence,
                    self.config.dispatch,
                )?;
                let poly = self.cpu.metal_cycle_message(&q_evals, previous_claim)?;
                self.sequence = Some(sequence);
                self.metal_rounds += 1;
                return Ok(poly);
            }
            let _span =
                tracing::info_span!("MetalInstructionReadRaf::resident_first_message").entered();
            let (cpu, address_sequence) = (&self.cpu, self.address_sequence.as_mut());
            let address_sequence = address_sequence
                .ok_or_else(|| backend_error("resident address sequence disappeared"))?;
            let poly = cpu.metal_resident_cycle_message(address_sequence, previous_claim)?;
            self.metal_rounds += 1;
            return Ok(poly);
        }

        if self
            .sequence
            .as_ref()
            .is_some_and(|sequence| sequence.current_elements() <= self.config.cutoff_elements)
        {
            self.restore_cpu_tail()?;
            return self.cpu.prove_round(bind, round, previous_claim);
        }

        if self.sequence.is_some() {
            let _span = tracing::info_span!("MetalInstructionReadRaf::resident_round").entered();
            let challenge = bind.ok_or_else(|| {
                backend_error("device-resident cycle round did not receive its prior challenge")
            })?;
            self.cpu.metal_bind_offloaded(challenge)?;
            let (cpu, sequence) = (&self.cpu, self.sequence.as_mut());
            let sequence = sequence
                .ok_or_else(|| backend_error("device sequence disappeared before dispatch"))?;
            let (e_in, e_out) = cpu.metal_cycle_weights()?;
            let q_evals = sequence
                .bind_and_message(challenge, e_in, e_out)
                .map_err(|error| backend_error(error.to_string()))?;
            self.metal_rounds += 1;
            return cpu.metal_cycle_message(&q_evals, previous_claim);
        }

        if let Some(challenge) = bind {
            if self
                .cpu
                .metal_handoff_available(self.config.cutoff_elements)
            {
                let _span = tracing::info_span!("MetalInstructionReadRaf::handoff").entered();
                let mut sequence = self.cpu.metal_offload_pending_bind(
                    challenge,
                    &self.context,
                    self.config.dispatch,
                )?;
                let (e_in, e_out) = self.cpu.metal_cycle_weights()?;
                let q_evals = sequence
                    .message(e_in, e_out)
                    .map_err(|error| backend_error(error.to_string()))?;
                let poly = self.cpu.metal_cycle_message(&q_evals, previous_claim)?;
                self.metal_rounds += 1;
                self.sequence = Some(sequence);
                return Ok(poly);
            }
        }

        self.cpu.prove_round(bind, round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if self.sequence.is_some() {
            self.restore_cpu_tail()?;
        }
        self.cpu.finish_rounds(bind)
    }
}

impl SumcheckKernel<AkitaField> for MetalInstructionReadRafKernel {
    type Relation = InstructionReadRaf<AkitaField>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<
        jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafOutputClaims<
            AkitaField,
        >,
        SumcheckKernelError<AkitaField>,
    > {
        self.cpu.output_claims(inputs)
    }
}

fn backend_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
    }
}
