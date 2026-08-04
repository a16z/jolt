use std::sync::Arc;

use jolt_field::AkitaField;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::SumcheckInputClaims;
use jolt_verifier::stages::stage5::InstructionReadRaf;
use jolt_witness::JoltWitnessPlane;

use super::solinas::{
    MetalError, Product5Sequence, Product5SequenceConfig, SolinasMetal, PRODUCT5_FACTORS,
};
use crate::optimized::instruction_read_raf::{
    prepare_optimized_instruction_read_raf, OptimizedInstructionReadRafKernel,
};
use crate::{
    JoltBackend, KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel,
    SumcheckKernelError,
};

/// Dispatch and crossover settings for the stage-5 dense cycle tail.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionReadRafMetalConfig {
    /// First table length whose next round runs on the CPU.
    pub cutoff_elements: usize,
    /// Threadgroup widths for the initial message and fused transitions.
    pub dispatch: Product5SequenceConfig,
}

impl Default for InstructionReadRafMetalConfig {
    fn default() -> Self {
        Self {
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
}

/// Shared Metal device state used by the installed sumcheck slots.
#[derive(Clone)]
pub struct MetalBackend {
    context: Arc<SolinasMetal>,
    config: MetalConfig,
}

impl MetalBackend {
    /// Compiles the Akita field library and validates the hybrid cutoffs.
    pub fn new(config: MetalConfig) -> Result<Self, MetalError> {
        let cutoff = config.instruction_read_raf.cutoff_elements;
        if cutoff < 2 || !cutoff.is_power_of_two() {
            return Err(MetalError::InvalidHybridCutoff(cutoff));
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
        let cpu = prepare_optimized_instruction_read_raf(session, witness, inputs)?;
        Ok(Box::new(MetalInstructionReadRafKernel::new(
            cpu,
            Arc::clone(&self.context),
            self.config.instruction_read_raf,
        )))
    }
}

pub(crate) struct MetalInstructionReadRafKernel {
    cpu: OptimizedInstructionReadRafKernel<AkitaField>,
    context: Arc<SolinasMetal>,
    config: InstructionReadRafMetalConfig,
    sequence: Option<Product5Sequence>,
    host_tail: Option<[Vec<AkitaField>; PRODUCT5_FACTORS]>,
    metal_rounds: usize,
}

impl MetalInstructionReadRafKernel {
    pub(crate) fn new(
        cpu: OptimizedInstructionReadRafKernel<AkitaField>,
        context: Arc<SolinasMetal>,
        config: InstructionReadRafMetalConfig,
    ) -> Self {
        Self {
            cpu,
            context,
            config,
            sequence: None,
            host_tail: Some(std::array::from_fn(|_| {
                vec![AkitaField::zero(); config.cutoff_elements]
            })),
            metal_rounds: 0,
        }
    }

    #[cfg(test)]
    pub(crate) const fn metal_rounds(&self) -> usize {
        self.metal_rounds
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
