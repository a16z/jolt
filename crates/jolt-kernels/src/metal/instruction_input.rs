use std::mem::size_of;

use jolt_field::Prime128OffsetA7F7 as AkitaField;
use jolt_field::Zero as _;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputClaims,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage3::outputs::InstructionInput;
use jolt_witness::JoltWitnessPlane;

use super::backend::MetalBackend;
use super::registers_claim_reduction::{
    registers_claim_alias_pair, RegistersClaimAliasPublisher, RegistersClaimReductionImplementation,
};
use super::solinas::registers_claim_reduction::{
    RegistersClaimGeometry, INSTRUCTION_INPUT_RS1_TABLE, INSTRUCTION_INPUT_RS2_TABLE,
};
use super::solinas::{
    instruction_input_weight_capacities, InstructionInputRows, InstructionInputSequence,
    InstructionInputSequenceConfig, InstructionInputSequenceStorage,
    InstructionInputStorageInitialization, MetalError, OuterRemainderSequence,
    PendingInstructionInputPrimer, SpartanOuterUniskipRows, INSTRUCTION_INPUT_PRIMER_E_IN_ELEMENTS,
    INSTRUCTION_INPUT_PRIMER_E_OUT_ELEMENTS, INSTRUCTION_INPUT_PRIMER_SOURCE_ELEMENTS,
    INSTRUCTION_INPUT_TABLES,
};
use crate::optimized::instruction_input::{
    OptimizedInstructionInput, OptimizedInstructionInputKernel,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum InstructionInputDenseStorageMode {
    #[default]
    Owned,
    OuterResidual,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionInputMetalConfig {
    pub trace_cutoff_elements: usize,
    pub cutoff_elements: usize,
    pub dense_storage_mode: InstructionInputDenseStorageMode,
    pub dispatch: InstructionInputSequenceConfig,
}

impl Default for InstructionInputMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 25,
            cutoff_elements: 1 << 16,
            dense_storage_mode: InstructionInputDenseStorageMode::Owned,
            dispatch: InstructionInputSequenceConfig {
                storage_initialization: InstructionInputStorageInitialization::Minimal,
                ..InstructionInputSequenceConfig::default()
            },
        }
    }
}

enum PreparedInstructionInputDevice {
    Storage(InstructionInputSequenceStorage),
    Priming(PendingInstructionInputPrimer),
}

pub(super) struct PreparedInstructionInput {
    device: PreparedInstructionInputDevice,
    host_tail: Vec<AkitaField>,
}

fn native_primer_supported(
    trace_elements: usize,
    e_in_capacity: usize,
    e_out_capacity: usize,
) -> bool {
    trace_elements >= INSTRUCTION_INPUT_PRIMER_SOURCE_ELEMENTS
        && e_in_capacity >= INSTRUCTION_INPUT_PRIMER_E_IN_ELEMENTS
        && e_out_capacity >= INSTRUCTION_INPUT_PRIMER_E_OUT_ELEMENTS
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for PreparedInstructionInput {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        match &self.device {
            PreparedInstructionInputDevice::Storage(storage) => {
                visitor.visit_field(allocative::Key::new("storage"), storage);
            }
            PreparedInstructionInputDevice::Priming(primer) => {
                visitor.visit_field(allocative::Key::new("primer"), primer);
            }
        }
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
        let Some(resident_rows) = session.state::<InstructionInputRows>() else {
            return Ok(());
        };
        let resident_rows_storage_id = resident_rows.allocation_identity();
        let resident_row_count = resident_rows.len();
        let resident_row_bytes = size_of::<super::solinas::InstructionInputRow>();
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
            dense_storage_mode = ?config.dense_storage_mode,
            host_tail_bytes,
            resident_rows_storage_id,
            resident_rows = resident_row_count,
            resident_row_bytes
        )
        .entered();
        let storage = match config.dense_storage_mode {
            InstructionInputDenseStorageMode::Owned => {
                self.context.prepare_instruction_input_sequence_storage(
                    trace_elements,
                    e_in_capacity,
                    e_out_capacity,
                    config.dispatch,
                )
            }
            InstructionInputDenseStorageMode::OuterResidual => {
                let Some(outer_rows) = session.state::<SpartanOuterUniskipRows>() else {
                    return Ok(());
                };
                self.context
                    .prepare_instruction_input_sequence_storage_from_outer(
                        outer_rows,
                        e_in_capacity,
                        e_out_capacity,
                        config.dispatch,
                    )
            }
        };
        match storage {
            Ok(storage) => {
                tracing::info!(
                    target: "jolt::metal",
                    bytes = storage.owned_bytes(),
                    "admitted InstructionInput Metal storage"
                );
                session.park(PreparedInstructionInput {
                    device: PreparedInstructionInputDevice::Storage(storage),
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

    fn prefetch_instruction_input(
        &self,
        session: &mut ProofSession,
    ) -> Result<(), KernelError<AkitaField>> {
        let prepared = session.take::<PreparedInstructionInput>();
        let resident_rows = session.take::<InstructionInputRows>();
        let (prepared, resident_rows) = match (prepared, resident_rows) {
            (None, None) => return Ok(()),
            (
                Some(
                    prepared @ PreparedInstructionInput {
                        device: PreparedInstructionInputDevice::Storage(_),
                        ..
                    },
                ),
                Some(resident_rows),
            ) => (prepared, resident_rows),
            (
                Some(
                    prepared @ PreparedInstructionInput {
                        device: PreparedInstructionInputDevice::Priming(_),
                        ..
                    },
                ),
                None,
            ) => {
                session.park(prepared);
                return Err(KernelError::InvariantViolation {
                    reason: "InstructionInput Metal primer was submitted twice",
                });
            }
            _ => {
                return Err(KernelError::InvariantViolation {
                    reason: "InstructionInput Metal prefetch lost one half of its resident state",
                });
            }
        };
        let PreparedInstructionInput { device, host_tail } = prepared;
        let PreparedInstructionInputDevice::Storage(mut storage) = device else {
            return Err(KernelError::InvariantViolation {
                reason: "InstructionInput Metal prefetch expected unprimed storage",
            });
        };
        let trace_elements = resident_rows.len();
        let config = self.config.instruction_input;
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
                reason: "InstructionInput Metal storage disagrees with the prefetched geometry",
            });
        }
        if storage.requires_outer_residual_release() {
            let Some(outer) = session.take::<OuterRemainderSequence>() else {
                tracing::warn!(
                    target: "jolt::metal",
                    "InstructionInput Outer residual arena was not released; selecting CPU"
                );
                return Ok(());
            };
            let receipt = outer
                .instruction_input_arena_release_receipt()
                .map_err(metal_prepare_error)?;
            let outer_residual_generation = receipt.key.generation;
            let outer_storage = outer.storage_stats().map_err(metal_prepare_error)?;
            storage
                .unlock_outer_residual(receipt, &resident_rows)
                .map_err(metal_prepare_error)?;
            let _transfer = tracing::info_span!(
                "MetalInstructionInput::outer_residual_transfer",
                resident_rows = trace_elements,
                outer_residual_generation,
                compact_rows_storage_id = outer_storage.compact_row_identity,
                residual_rows_storage_id = outer_storage.residual_row_identity,
                device_registry_id = outer_storage.row_device_registry_id,
                outer_sequence_owned_bytes = outer_storage.owned_bytes,
                outer_sequence_consumed = true,
                compact_rows_transferred = true,
                residual_rows_transferred = true,
            )
            .entered();
            drop(outer);
        }
        if !native_primer_supported(trace_elements, e_in_capacity, e_out_capacity) {
            session.park(PreparedInstructionInput {
                device: PreparedInstructionInputDevice::Storage(storage),
                host_tail,
            });
            session.park(resident_rows);
            return Ok(());
        }
        let sequence = storage.attach(resident_rows).map_err(metal_prepare_error)?;
        let resident_rows_storage_id = sequence.resident_row_identity();
        let submit_span = tracing::info_span!(
            "MetalInstructionInput::native_primer_submit",
            source_elements = INSTRUCTION_INPUT_PRIMER_SOURCE_ELEMENTS,
            e_in_elements = INSTRUCTION_INPUT_PRIMER_E_IN_ELEMENTS,
            e_out_elements = INSTRUCTION_INPUT_PRIMER_E_OUT_ELEMENTS,
            resident_rows_storage_id,
        );
        let pending = {
            let _span = submit_span.enter();
            sequence
                .submit_native_pipeline_primer()
                .map_err(metal_prepare_error)?
        };
        session.park(PreparedInstructionInput {
            device: PreparedInstructionInputDevice::Priming(pending),
            host_tail,
        });
        Ok(())
    }
}

impl PrepareKernel<AkitaField, InstructionInput<AkitaField>> for MetalBackend {
    fn prefetch(&self, session: &mut ProofSession) -> Result<(), KernelError<AkitaField>> {
        self.prefetch_instruction_input(session)
    }

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
            drop(session.take::<InstructionInputRows>());
            return <OptimizedInstructionInput as PrepareKernel<
                AkitaField,
                InstructionInput<AkitaField>,
            >>::prepare(&OptimizedInstructionInput, session, witness, inputs);
        }

        let (e_in_capacity, e_out_capacity) =
            instruction_input_weight_capacities(trace_elements).map_err(metal_prepare_error)?;
        let prepared = session.take::<PreparedInstructionInput>();
        let resident_rows = session.take::<InstructionInputRows>();
        let (device, host_tail) = match (prepared, resident_rows) {
            (None, None) => {
                return <OptimizedInstructionInput as PrepareKernel<
                    AkitaField,
                    InstructionInput<AkitaField>,
                >>::prepare(
                    &OptimizedInstructionInput, session, witness, inputs
                );
            }
            (
                Some(PreparedInstructionInput {
                    device: PreparedInstructionInputDevice::Storage(storage),
                    host_tail,
                }),
                Some(resident_rows),
            ) => {
                if !storage.matches(
                    &self.context,
                    trace_elements,
                    e_in_capacity,
                    e_out_capacity,
                    config.dispatch,
                ) {
                    return Err(KernelError::InvariantViolation {
                        reason:
                            "InstructionInput Metal storage disagrees with the stage-3 geometry",
                    });
                }
                let resident_identity = resident_rows.allocation_identity();
                let sequence = storage.attach(resident_rows).map_err(metal_prepare_error)?;
                if sequence.resident_row_identity() != resident_identity {
                    return Err(KernelError::InvariantViolation {
                        reason: "InstructionInput did not retain the stage-1 Metal row allocation",
                    });
                }
                (PreparedMetalSequence::Ready(sequence), host_tail)
            }
            (
                Some(PreparedInstructionInput {
                    device: PreparedInstructionInputDevice::Priming(pending),
                    host_tail,
                }),
                None,
            ) => {
                if !pending.matches(
                    &self.context,
                    trace_elements,
                    e_in_capacity,
                    e_out_capacity,
                    config.dispatch,
                ) {
                    return Err(KernelError::InvariantViolation {
                        reason: "InstructionInput Metal primer disagrees with the stage-3 geometry",
                    });
                }
                (PreparedMetalSequence::Priming(pending), host_tail)
            }
            _ => {
                return Err(KernelError::InvariantViolation {
                    reason: "InstructionInput Metal preparation has an invalid primer/row state",
                });
            }
        };
        let resident_identity =
            device
                .resident_row_identity()
                .ok_or(KernelError::InvariantViolation {
                    reason: "InstructionInput Metal sequence lost its resident row identity",
                })?;
        let _prepare_span = tracing::info_span!(
            "MetalInstructionInput::prepare",
            resident_rows_reused = true,
            round_device_buffer_allocations = 0,
            resident_rows_storage_id = resident_identity,
            resident_rows = trace_elements,
            native_primer = %device.primer_mode(),
        )
        .entered();
        let alias_publisher = if self.config.registers_claim_reduction.implementation
            == RegistersClaimReductionImplementation::OuterCarrierAliasHybrid
        {
            let geometry = RegistersClaimGeometry::new(trace_elements).map_err(|_| {
                KernelError::InvariantViolation {
                    reason: "registers claim alias bridge geometry is invalid",
                }
            })?;
            let required_cutoff = geometry.suffix_elements().checked_mul(2).ok_or(
                KernelError::InvariantViolation {
                    reason: "registers claim alias cutoff overflows usize",
                },
            )?;
            if config.cutoff_elements < required_cutoff {
                return Err(KernelError::InvariantViolation {
                    reason: "InstructionInput CPU cutoff is too small for registers aliases",
                });
            }
            let (publisher, receiver) =
                registers_claim_alias_pair(trace_elements, resident_identity)?;
            session.park(receiver);
            Some(publisher)
        } else {
            None
        };
        let cpu =
            OptimizedInstructionInputKernel::new_offloaded(r_product, inputs.challenges.gamma);
        let kernel = MetalInstructionInputKernel::new(
            cpu,
            device,
            host_tail,
            inputs.challenges.gamma,
            config.cutoff_elements,
            alias_publisher,
        )?;
        Ok(Box::new(kernel))
    }
}

enum PreparedMetalSequence {
    Ready(InstructionInputSequence),
    Priming(PendingInstructionInputPrimer),
}

impl PreparedMetalSequence {
    fn resident_row_identity(&self) -> Option<usize> {
        match self {
            Self::Ready(sequence) => Some(sequence.resident_row_identity()),
            Self::Priming(pending) => pending.resident_row_identity(),
        }
    }

    const fn primer_mode(&self) -> &'static str {
        match self {
            Self::Ready(_) => "none",
            Self::Priming(_) => "async",
        }
    }
}

struct MetalInstructionInputKernel {
    cpu: OptimizedInstructionInputKernel<AkitaField>,
    sequence: Option<InstructionInputSequence>,
    primer: Option<PendingInstructionInputPrimer>,
    host_tail: Option<Vec<AkitaField>>,
    gamma: AkitaField,
    cutoff_elements: usize,
    alias_publisher: Option<RegistersClaimAliasPublisher>,
    alias_challenges: Vec<AkitaField>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalInstructionInputKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_field(allocative::Key::new("cpu"), &self.cpu);
        if let Some(sequence) = &self.sequence {
            visitor.visit_field(allocative::Key::new("sequence"), sequence);
        }
        if let Some(primer) = &self.primer {
            visitor.visit_field(allocative::Key::new("primer"), primer);
        }
        if let Some(host_tail) = &self.host_tail {
            visitor.visit_simple(
                allocative::Key::new("host_tail"),
                crate::backend::vec_heap_bytes(host_tail),
            );
        }
        visitor.visit_simple(
            allocative::Key::new("alias_challenges"),
            crate::backend::vec_heap_bytes(&self.alias_challenges),
        );
        visitor.exit();
    }
}

impl MetalInstructionInputKernel {
    fn new(
        cpu: OptimizedInstructionInputKernel<AkitaField>,
        device: PreparedMetalSequence,
        host_tail: Vec<AkitaField>,
        gamma: AkitaField,
        cutoff_elements: usize,
        alias_publisher: Option<RegistersClaimAliasPublisher>,
    ) -> Result<Self, SumcheckError<AkitaField>> {
        let host_elements = INSTRUCTION_INPUT_TABLES
            .checked_mul(cutoff_elements)
            .ok_or_else(|| metal_error("InstructionInput host-tail capacity overflow"))?;
        if host_tail.len() != host_elements {
            return Err(metal_error(
                "InstructionInput prepared host-tail capacity is invalid",
            ));
        }
        let (sequence, primer) = match device {
            PreparedMetalSequence::Ready(sequence) => (Some(sequence), None),
            PreparedMetalSequence::Priming(primer) => (None, Some(primer)),
        };
        Ok(Self {
            cpu,
            sequence,
            primer,
            host_tail: Some(host_tail),
            gamma,
            cutoff_elements,
            alias_publisher,
            alias_challenges: Vec::new(),
        })
    }

    fn publish_register_aliases(&mut self, round: usize) -> Result<(), SumcheckError<AkitaField>> {
        let Some(publisher) = self.alias_publisher.take() else {
            return Ok(());
        };
        let geometry = RegistersClaimGeometry::new(1usize << self.cpu.num_rounds())
            .map_err(|error| metal_error(error.to_string()))?;
        if self.alias_challenges.len() < geometry.prefix_vars() {
            self.alias_publisher = Some(publisher);
            return Ok(());
        }
        if round != geometry.prefix_vars() || self.alias_challenges.len() != geometry.prefix_vars()
        {
            return Err(metal_error(
                "InstructionInput reached the registers alias point out of order",
            ));
        }
        let [rs1_value, rs2_value] = self.cpu.metal_copy_dense_tables(
            [INSTRUCTION_INPUT_RS1_TABLE, INSTRUCTION_INPUT_RS2_TABLE],
            geometry.prefix_vars(),
            geometry.suffix_elements(),
        )?;
        let phase = tracing::info_span!(
            "MetalInstructionInput::registers_claim_alias_publish",
            rows = publisher.rows(),
            source_compact_storage_id = publisher.source_compact_storage_id() as u64,
            alias_generation = publisher.generation(),
            prefix_challenges = self.alias_challenges.len(),
            table_0 = INSTRUCTION_INPUT_RS1_TABLE,
            table_1 = INSTRUCTION_INPUT_RS2_TABLE,
            host_table_copies = 2u64,
            snapshot_host_bytes = 2 * geometry.suffix_elements() * size_of::<AkitaField>(),
            publishes = 1u64,
        );
        let _phase_guard = phase.enter();
        publisher.publish(self.alias_challenges.clone(), rs1_value, rs2_value)
    }

    fn join_primer(
        &mut self,
        bind: Option<&AkitaField>,
        round: usize,
    ) -> Result<(), SumcheckError<AkitaField>> {
        let Some(pending) = self.primer.take() else {
            return Ok(());
        };
        if round != 0 || bind.is_some() || self.sequence.is_some() {
            return Err(metal_error(
                "InstructionInput Metal primer must join before the first unbound message",
            ));
        }
        let sequence = {
            let _span = tracing::info_span!("MetalInstructionInput::native_primer_join").entered();
            pending
                .join()
                .map_err(|error| metal_error(error.to_string()))?
        };
        if sequence.is_dense()
            || sequence.current_elements() < INSTRUCTION_INPUT_PRIMER_SOURCE_ELEMENTS
        {
            return Err(metal_error(
                "InstructionInput Metal primer changed its resources or protocol state",
            ));
        }
        self.sequence = Some(sequence);
        Ok(())
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
        self.join_primer(bind.as_ref(), round)?;
        if self.sequence.as_ref().is_some_and(|sequence| {
            sequence.is_dense() && sequence.current_elements() <= self.cutoff_elements
        }) {
            self.restore_cpu_tail()?;
        }

        let polynomial = if self.sequence.is_some() {
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
            self.cpu
                .metal_message(coefficients, round, previous_claim)?
        } else {
            let _span = tracing::info_span!("MetalInstructionInput::cpu_tail", round).entered();
            self.cpu.prove_round(bind, round, previous_claim)?
        };
        if self.alias_publisher.is_some() {
            if let Some(challenge) = bind {
                self.alias_challenges.push(challenge);
            }
            self.publish_register_aliases(round)?;
        }
        Ok(polynomial)
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if self.primer.is_some() {
            return Err(metal_error(
                "InstructionInput Metal primer was not joined before round completion",
            ));
        }
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
    use super::native_primer_supported;

    #[test]
    fn fixed_primer_requires_its_full_weight_shape() {
        assert!(!native_primer_supported(512, 1, 16));
        assert!(native_primer_supported(1024, 1, 32));
    }
}
