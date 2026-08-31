use jolt_claims::protocols::jolt::relations::instruction::InstructionRaVirtualizationChallenges;
use jolt_field::AkitaField;
use jolt_sumcheck::{ProveRounds, RoundExecutionDomain, SumcheckError};
use jolt_verifier::stages::relations::{
    SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualization;
use jolt_witness::JoltWitnessPlane;

use super::backend::MetalBackend;
use super::solinas::{
    InstructionRaSequence, InstructionRaSequenceConfig, InstructionRaSequenceStorage,
    ResidentLookupIndexPlane,
};
use crate::optimized::instruction_ra_virtualization::{
    prepare_instruction_ra_from_initialization, prepare_instruction_ra_initialization,
    OptimizedInstructionRaVirtualizationKernel,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionRaVirtualizationMetalConfig {
    pub trace_cutoff_elements: usize,
    pub cutoff_elements: usize,
    pub dispatch: InstructionRaSequenceConfig,
}

impl Default for InstructionRaVirtualizationMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 25,
            cutoff_elements: 1 << 10,
            dispatch: InstructionRaSequenceConfig::default(),
        }
    }
}

impl PrepareKernel<AkitaField, InstructionRaVirtualization<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, InstructionRaVirtualization<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = InstructionRaVirtualization<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let trace_elements = 1usize << inputs.relation.dimensions().log_t();
        let dispatch = self.config.instruction_ra_virtualization.dispatch;
        let initialization = prepare_instruction_ra_initialization(inputs)?;
        if trace_elements
            < self
                .config
                .instruction_ra_virtualization
                .trace_cutoff_elements
            || trace_elements < 2 * dispatch.materialize_width.elements()
            || !initialization.supports_metal_sequence()
        {
            let _ = session.take::<InstructionRaSequenceStorage>();
            let _ = session.take::<ResidentLookupIndexPlane>();
            return Ok(Box::new(prepare_instruction_ra_from_initialization(
                session,
                witness,
                initialization,
            )?));
        }

        let plane =
            session
                .take::<ResidentLookupIndexPlane>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "Metal Instruction RA requires the resident stage-5 lookup plane",
                })?;
        if plane.len() != trace_elements {
            return Err(KernelError::TableSizeMismatch {
                table: "resident Metal lookup-index plane".to_owned(),
                expected: trace_elements,
                got: plane.len(),
            });
        }
        if plane.device_registry_id() != self.context.device_registry_id() {
            return Err(KernelError::InvariantViolation {
                reason: "resident Metal lookup-index plane belongs to another device",
            });
        }

        let (cpu, chunk_tables) = initialization.into_offloaded()?;
        let (e_in_capacity, e_out_capacity) = {
            let (e_in, e_out) = cpu.metal_weights()?;
            (e_in.len(), e_out.len())
        };
        let sequence = {
            let storage = session
                .take::<InstructionRaSequenceStorage>()
                .filter(|storage| {
                    storage.matches(
                        &self.context,
                        trace_elements,
                        e_in_capacity,
                        e_out_capacity,
                        dispatch,
                    )
                });
            let _span = tracing::info_span!(
                "MetalInstructionRaVirtualization::sequence_prepare",
                preallocated = storage.is_some()
            )
            .entered();
            match storage {
                Some(storage) => storage.attach(plane, &chunk_tables),
                _ => self.context.prepare_instruction_ra_sequence_with_plane(
                    plane,
                    &chunk_tables,
                    e_in_capacity,
                    e_out_capacity,
                    dispatch,
                ),
            }
            .map_err(|error| metal_error(error.to_string()))?
        };
        Ok(Box::new(MetalInstructionRaVirtualizationKernel::new(
            cpu,
            sequence,
            self.config.instruction_ra_virtualization.cutoff_elements,
        )?))
    }
}

struct MetalInstructionRaVirtualizationKernel {
    cpu: OptimizedInstructionRaVirtualizationKernel<AkitaField>,
    sequence: Option<InstructionRaSequence>,
    host_tail: Vec<AkitaField>,
    cutoff_elements: usize,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalInstructionRaVirtualizationKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_field(allocative::Key::new("cpu"), &self.cpu);
        if let Some(sequence) = &self.sequence {
            visitor.visit_field(allocative::Key::new("sequence"), sequence);
        }
        visitor.visit_simple(
            allocative::Key::new("host_tail"),
            crate::backend::vec_heap_bytes(&self.host_tail),
        );
        visitor.exit();
    }
}

impl MetalInstructionRaVirtualizationKernel {
    fn new(
        cpu: OptimizedInstructionRaVirtualizationKernel<AkitaField>,
        sequence: InstructionRaSequence,
        cutoff_elements: usize,
    ) -> Result<Self, SumcheckError<AkitaField>> {
        let host_elements = cpu
            .metal_num_polys()
            .checked_mul(cutoff_elements)
            .ok_or_else(|| metal_error("instruction RA host-tail capacity overflow"))?;
        Ok(Self {
            cpu,
            sequence: Some(sequence),
            host_tail: vec![AkitaField::zero(); host_elements],
            cutoff_elements,
        })
    }

    fn restore_cpu_tail(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let _span = tracing::info_span!("MetalInstructionRaVirtualization::readback").entered();
        let sequence = self.sequence.take().ok_or_else(|| {
            metal_error("instruction RA sequence disappeared before dense readback")
        })?;
        if !sequence.is_dense() {
            return Err(metal_error(
                "instruction RA CPU handoff requires resident dense tables",
            ));
        }
        let elements = sequence.current_elements();
        let output_len = self
            .cpu
            .metal_num_polys()
            .checked_mul(elements)
            .ok_or_else(|| metal_error("instruction RA readback length overflow"))?;
        if output_len > self.host_tail.len() {
            return Err(metal_error(
                "instruction RA readback exceeds the preallocated CPU tail",
            ));
        }
        sequence
            .read_current_tables(&mut self.host_tail[..output_len])
            .map_err(|error| metal_error(error.to_string()))?;
        self.cpu
            .metal_restore_dense(&self.host_tail[..output_len], elements)
    }
}

impl ProveRounds<AkitaField> for MetalInstructionRaVirtualizationKernel {
    fn num_rounds(&self) -> usize {
        self.cpu.num_rounds()
    }

    fn execution_domain(&self) -> RoundExecutionDomain {
        if self.sequence.as_ref().is_some_and(|sequence| {
            !sequence.is_dense() || sequence.current_elements() > self.cutoff_elements
        }) {
            RoundExecutionDomain::Accelerator
        } else {
            RoundExecutionDomain::Host
        }
    }

    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<jolt_poly::UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if self.sequence.as_ref().is_some_and(|sequence| {
            sequence.is_dense() && sequence.current_elements() <= self.cutoff_elements
        }) {
            self.restore_cpu_tail()?;
            return self.cpu.prove_round(bind, round, previous_claim);
        }

        if self.sequence.is_some() {
            let sequence = self
                .sequence
                .as_ref()
                .ok_or_else(|| metal_error("instruction RA sequence disappeared before round"))?;
            let _span = if sequence.is_dense() {
                tracing::info_span!(
                    "MetalInstructionRaVirtualization::dense_round",
                    elements = sequence.current_elements()
                )
                .entered()
            } else {
                tracing::info_span!(
                    "MetalInstructionRaVirtualization::lazy_round",
                    elements = sequence.current_elements(),
                    branch_width = sequence.branch_width()
                )
                .entered()
            };
            let message = if let Some(challenge) = bind {
                self.cpu.metal_bind_offloaded(challenge)?;
                let (e_in, e_out) = self.cpu.metal_weights()?;
                self.sequence
                    .as_mut()
                    .ok_or_else(|| metal_error("instruction RA sequence disappeared before bind"))?
                    .bind_and_message(challenge, e_in, e_out)
                    .map_err(|error| metal_error(error.to_string()))?
            } else {
                let (e_in, e_out) = self.cpu.metal_weights()?;
                self.sequence
                    .as_mut()
                    .ok_or_else(|| {
                        metal_error("instruction RA sequence disappeared before message")
                    })?
                    .message(e_in, e_out)
                    .map_err(|error| metal_error(error.to_string()))?
            };
            return self.cpu.metal_message(message, previous_claim);
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

impl SumcheckKernel<AkitaField> for MetalInstructionRaVirtualizationKernel {
    type Relation = InstructionRaVirtualization<AkitaField>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<AkitaField, Self::Relation>, SumcheckKernelError<AkitaField>>
    {
        self.cpu.output_claims(inputs)
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &InstructionRaVirtualizationChallenges<AkitaField>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        self.cpu
            .validate_derived_tables(relation, input_points, output_points, challenges)
    }
}

fn metal_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
    }
}
