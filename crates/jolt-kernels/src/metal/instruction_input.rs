use std::mem::size_of;

use jolt_field::AkitaField;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputClaims,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage3::outputs::InstructionInput;
use jolt_witness::JoltWitnessPlane;

use super::instruction_read_raf::MetalBackend;
use super::solinas::{
    instruction_input_weight_capacities, InstructionInputSequence, InstructionInputSequenceConfig,
    InstructionInputSequenceStorage, MetalError, SpartanOuterUniskipRows, INSTRUCTION_INPUT_TABLES,
};
use crate::optimized::instruction_input::{
    OptimizedInstructionInput, OptimizedInstructionInputKernel,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionInputMetalConfig {
    pub trace_cutoff_elements: usize,
    pub cutoff_elements: usize,
    pub dispatch: InstructionInputSequenceConfig,
}

impl Default for InstructionInputMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 25,
            cutoff_elements: 1 << 16,
            dispatch: InstructionInputSequenceConfig::default(),
        }
    }
}

pub(super) struct PreparedInstructionInput {
    storage: InstructionInputSequenceStorage,
    host_tail: Vec<AkitaField>,
}

fn pair_prepared_metal_state<P, R>(
    prepared: Option<P>,
    resident_rows: Option<R>,
) -> Result<Option<(P, R)>, &'static str> {
    match (prepared, resident_rows) {
        (Some(prepared), Some(resident_rows)) => Ok(Some((prepared, resident_rows))),
        (None, None) => Ok(None),
        _ => Err("InstructionInput Metal preparation lost one half of its resident state"),
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for PreparedInstructionInput {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_field(allocative::Key::new("storage"), &self.storage);
        visitor.visit_simple(
            allocative::Key::new("host_tail"),
            crate::backend::vec_heap_bytes(&self.host_tail),
        );
        visitor.exit();
    }
}

impl MetalBackend {
    pub(crate) fn prepare_instruction_input_storage(
        &self,
        session: &mut ProofSession,
        trace_elements: usize,
    ) -> Result<(), KernelError<AkitaField>> {
        let config = self.config.instruction_input;
        if trace_elements < config.trace_cutoff_elements || trace_elements <= config.cutoff_elements
        {
            return Ok(());
        }
        let Some(resident_rows) = session.state::<SpartanOuterUniskipRows>() else {
            return Ok(());
        };
        let resident_rows_storage_id = resident_rows.allocation_identity();
        let resident_row_count = resident_rows.len();
        let resident_row_bytes = size_of::<super::solinas::SpartanOuterUniskipRow>();
        let host_elements = INSTRUCTION_INPUT_TABLES
            .checked_mul(config.cutoff_elements)
            .ok_or(MetalError::InputTooLong(config.cutoff_elements))
            .map_err(metal_prepare_error)?;
        let host_tail_bytes = host_elements
            .checked_mul(size_of::<AkitaField>())
            .ok_or(MetalError::InputTooLong(host_elements))
            .map_err(metal_prepare_error)?;
        let (e_in_capacity, e_out_capacity) =
            instruction_input_weight_capacities(trace_elements).map_err(metal_prepare_error)?;
        let _span = tracing::info_span!(
            "MetalInstructionInput::storage_prepare",
            trace_elements,
            cutoff_elements = config.cutoff_elements,
            host_tail_bytes,
            resident_rows_storage_id,
            resident_rows = resident_row_count,
            resident_row_bytes
        )
        .entered();
        let storage = self.context.prepare_instruction_input_sequence_storage(
            trace_elements,
            e_in_capacity,
            e_out_capacity,
            config.dispatch,
        );
        match storage {
            Ok(storage) => {
                tracing::info!(
                    target: "jolt::metal",
                    bytes = storage.owned_bytes(),
                    "admitted InstructionInput Metal storage"
                );
                session.park(PreparedInstructionInput {
                    storage,
                    host_tail: vec![AkitaField::zero(); host_elements],
                });
                Ok(())
            }
            Err(
                error @ (MetalError::BufferTooLong { .. } | MetalError::WorkingSetTooLarge { .. }),
            ) => {
                tracing::warn!(
                    target: "jolt::metal",
                    error = %error,
                    "InstructionInput Metal storage was not admitted"
                );
                Ok(())
            }
            Err(error) => Err(metal_prepare_error(error)),
        }
    }
}

impl PrepareKernel<AkitaField, InstructionInput<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, InstructionInput<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = InstructionInput<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let r_product = inputs.relation.product_remainder_opening_point();
        let trace_elements = 1usize << r_product.len();
        let config = self.config.instruction_input;
        if trace_elements < config.trace_cutoff_elements || trace_elements <= config.cutoff_elements
        {
            drop(session.take::<PreparedInstructionInput>());
            drop(session.take::<SpartanOuterUniskipRows>());
            return <OptimizedInstructionInput as PrepareKernel<
                AkitaField,
                InstructionInput<AkitaField>,
            >>::prepare(&OptimizedInstructionInput, session, witness, inputs);
        }

        let prepared_pair = pair_prepared_metal_state(
            session.take::<PreparedInstructionInput>(),
            session.take::<SpartanOuterUniskipRows>(),
        )
        .map_err(|reason| KernelError::InvariantViolation { reason })?;
        let (prepared, resident_rows) = match prepared_pair {
            Some(pair) => pair,
            None => {
                return <OptimizedInstructionInput as PrepareKernel<
                    AkitaField,
                    InstructionInput<AkitaField>,
                >>::prepare(
                    &OptimizedInstructionInput, session, witness, inputs
                );
            }
        };
        let PreparedInstructionInput { storage, host_tail } = prepared;
        let (e_in_capacity, e_out_capacity) =
            instruction_input_weight_capacities(trace_elements).map_err(metal_prepare_error)?;
        if !storage.matches(
            &self.context,
            trace_elements,
            e_in_capacity,
            e_out_capacity,
            config.dispatch,
        ) {
            return Err(KernelError::InvariantViolation {
                reason: "InstructionInput Metal storage disagrees with the stage-3 geometry",
            });
        }
        let resident_identity = resident_rows.allocation_identity();
        let sequence = storage.attach(resident_rows).map_err(metal_prepare_error)?;
        if sequence.resident_row_identity() != resident_identity {
            return Err(KernelError::InvariantViolation {
                reason: "InstructionInput did not retain the stage-1 Metal row allocation",
            });
        }
        let round_device_buffer_allocations = sequence.round_device_buffer_allocations();
        let _prepare_span = tracing::info_span!(
            "MetalInstructionInput::prepare",
            resident_rows_reused = true,
            round_device_buffer_allocations,
            resident_rows_storage_id = resident_identity,
            resident_rows = trace_elements
        )
        .entered();
        let cpu =
            OptimizedInstructionInputKernel::new_offloaded(r_product, inputs.challenges.gamma);
        let kernel = MetalInstructionInputKernel::new(
            cpu,
            sequence,
            host_tail,
            inputs.challenges.gamma,
            config.cutoff_elements,
        )?;
        Ok(Box::new(kernel))
    }
}

struct MetalInstructionInputKernel {
    cpu: OptimizedInstructionInputKernel<AkitaField>,
    sequence: Option<InstructionInputSequence>,
    host_tail: Option<Vec<AkitaField>>,
    gamma: AkitaField,
    cutoff_elements: usize,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalInstructionInputKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_field(allocative::Key::new("cpu"), &self.cpu);
        if let Some(sequence) = &self.sequence {
            visitor.visit_field(allocative::Key::new("sequence"), sequence);
        }
        if let Some(host_tail) = &self.host_tail {
            visitor.visit_simple(
                allocative::Key::new("host_tail"),
                crate::backend::vec_heap_bytes(host_tail),
            );
        }
        visitor.exit();
    }
}

impl MetalInstructionInputKernel {
    fn new(
        cpu: OptimizedInstructionInputKernel<AkitaField>,
        sequence: InstructionInputSequence,
        host_tail: Vec<AkitaField>,
        gamma: AkitaField,
        cutoff_elements: usize,
    ) -> Result<Self, SumcheckError<AkitaField>> {
        let host_elements = INSTRUCTION_INPUT_TABLES
            .checked_mul(cutoff_elements)
            .ok_or_else(|| metal_error("InstructionInput host-tail capacity overflow"))?;
        if host_tail.len() != host_elements {
            return Err(metal_error(
                "InstructionInput prepared host-tail capacity is invalid",
            ));
        }
        Ok(Self {
            cpu,
            sequence: Some(sequence),
            host_tail: Some(host_tail),
            gamma,
            cutoff_elements,
        })
    }

    fn restore_cpu_tail(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let sequence = self.sequence.take().ok_or_else(|| {
            metal_error("InstructionInput sequence disappeared before dense readback")
        })?;
        let mut host_tail = self.host_tail.take().ok_or_else(|| {
            metal_error("InstructionInput host tail disappeared before dense readback")
        })?;
        if !sequence.is_dense() {
            return Err(metal_error(
                "InstructionInput CPU handoff requires resident dense tables",
            ));
        }
        let elements = sequence.current_elements();
        let output_len = INSTRUCTION_INPUT_TABLES
            .checked_mul(elements)
            .ok_or_else(|| metal_error("InstructionInput readback length overflow"))?;
        if output_len > host_tail.len() {
            return Err(metal_error(
                "InstructionInput readback exceeds the preallocated CPU tail",
            ));
        }
        let _span = tracing::info_span!(
            "MetalInstructionInput::readback",
            bytes = output_len * size_of::<AkitaField>()
        )
        .entered();
        sequence
            .read_current_tables(&mut host_tail[..output_len])
            .map_err(|error| metal_error(error.to_string()))?;
        self.cpu
            .metal_restore_dense(&host_tail[..output_len], elements)
    }
}

impl ProveRounds<AkitaField> for MetalInstructionInputKernel {
    fn num_rounds(&self) -> usize {
        self.cpu.num_rounds()
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
        }

        if self.sequence.is_some() {
            let coefficients = if let Some(challenge) = bind {
                self.cpu.metal_bind_offloaded(challenge)?;
                let (e_in, e_out) = self.cpu.metal_weights()?;
                let sequence = self.sequence.as_mut().ok_or_else(|| {
                    metal_error("InstructionInput sequence disappeared before bind")
                })?;
                let _span = if sequence.is_dense() {
                    tracing::info_span!(
                        "MetalInstructionInput::dense_round",
                        round,
                        source_elements = sequence.current_elements()
                    )
                    .entered()
                } else {
                    tracing::info_span!(
                        "MetalInstructionInput::first_bind",
                        source_elements = sequence.current_elements()
                    )
                    .entered()
                };
                sequence
                    .bind_and_message(challenge, self.gamma, e_in, e_out)
                    .map_err(|error| metal_error(error.to_string()))?
            } else {
                let (e_in, e_out) = self.cpu.metal_weights()?;
                let sequence = self.sequence.as_mut().ok_or_else(|| {
                    metal_error("InstructionInput sequence disappeared before first message")
                })?;
                let _span = tracing::info_span!(
                    "MetalInstructionInput::first_message",
                    source_elements = sequence.current_elements()
                )
                .entered();
                sequence
                    .message(self.gamma, e_in, e_out)
                    .map_err(|error| metal_error(error.to_string()))?
            };
            return self.cpu.metal_message(coefficients, round, previous_claim);
        }

        let _span = tracing::info_span!("MetalInstructionInput::cpu_tail", round).entered();
        self.cpu.prove_round(bind, round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if self.sequence.is_some() {
            self.restore_cpu_tail()?;
        }
        let _span = tracing::info_span!("MetalInstructionInput::cpu_tail").entered();
        self.cpu.finish_rounds(bind)
    }
}

impl SumcheckKernel<AkitaField> for MetalInstructionInputKernel {
    type Relation = InstructionInput<AkitaField>;

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
        challenges: &ConcreteSumcheckChallenges<AkitaField, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        self.cpu
            .validate_derived_tables(relation, input_points, output_points, challenges)
    }
}

fn metal_prepare_error(error: MetalError) -> KernelError<AkitaField> {
    metal_error(error.to_string()).into()
}

fn metal_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::pair_prepared_metal_state;

    #[test]
    fn prepared_metal_state_is_all_or_nothing() {
        assert_eq!(
            pair_prepared_metal_state(Some(1), Some(2)),
            Ok(Some((1, 2)))
        );
        assert_eq!(pair_prepared_metal_state::<u8, u8>(None, None), Ok(None));
        assert!(pair_prepared_metal_state(Some(1), None::<u8>).is_err());
        assert!(pair_prepared_metal_state(None::<u8>, Some(2)).is_err());
    }
}
