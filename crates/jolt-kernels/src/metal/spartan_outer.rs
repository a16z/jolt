use std::{
    collections::BTreeMap,
    mem::size_of,
    time::{Duration, Instant},
};

use jolt_claims::protocols::jolt::geometry::dimensions::OUTER_UNISKIP_DOMAIN_SIZE;
use jolt_claims::protocols::jolt::geometry::spartan::{outer_opening, SpartanOuterDimensions};
use jolt_claims::protocols::jolt::{JoltDerivedId, JoltOpeningId, SpartanOuterPublic};
use jolt_claims::{InputClaims as _, OutputClaims as _};
use jolt_field::AkitaField;
use jolt_poly::lagrange::{centered_lagrange_evals, centered_lagrange_kernel};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use jolt_r1cs::constraints::jolt::{spartan_outer_constraints, spartan_outer_row_weights};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck as _, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
use jolt_verifier::VerifierError;
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::backend::{MetalBackend, MetalConfig};
use super::instruction_input::PreparedInstructionInput;
use super::solinas::{
    instruction_input_row_bytes, instruction_input_sequence_storage_bytes,
    instruction_ra_weight_capacities, outer_remainder_sequence_max_buffer_bytes_with_config,
    outer_remainder_sequence_storage_bytes_with_config, spartan_outer_uniskip_invocation_bytes,
    spartan_outer_uniskip_row_bytes, InstructionInputRows, InstructionRaSequenceStorage,
    MetalError, OuterRemainderPhase, OuterRemainderSequence, OuterRemainderSequenceConfig,
    OuterRemainderSequenceStorage, SpartanOuterUniskipConfig, SpartanOuterUniskipRows,
};
use crate::optimized::instruction_input::PreparedInstructionInputRows;
use crate::optimized::spartan_outer::{
    prepare_metal_instruction_input_witness_rows, prepare_metal_spartan_outer_uniskip,
    prepare_metal_spartan_outer_witness_rows, take_metal_spartan_outer_tau,
    OptimizedOuterRemainder, OptimizedOuterUniskip,
};
use crate::uniskip::UniskipKernel;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[cfg(feature = "test-utils")]
mod evaluation;

#[cfg(feature = "test-utils")]
pub use evaluation::{
    OuterRemainderEvalError, OuterRemainderEvalFixture, OuterRemainderEvalResult,
    OuterRemainderEvalSample,
};

const OUTER_DOMAIN: usize = OUTER_UNISKIP_DOMAIN_SIZE;
const OUTER_VARIABLES: usize = 35;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpartanOuterUniskipMetalConfig {
    pub trace_cutoff_elements: usize,
    pub dispatch: SpartanOuterUniskipConfig,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpartanOuterRemainderMetalConfig {
    pub trace_cutoff_elements: usize,
    pub dispatch: OuterRemainderSequenceConfig,
}

impl Default for SpartanOuterRemainderMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 18,
            dispatch: OuterRemainderSequenceConfig::default(),
        }
    }
}

fn use_metal_remainder(cycles: usize, config: &MetalConfig, resident: bool) -> bool {
    cycles >= 4 && cycles >= config.spartan_outer_remainder.trace_cutoff_elements && resident
}

fn resident_row_consumers(cycles: usize, config: &MetalConfig) -> (bool, bool) {
    let stage1 = cycles
        >= config
            .spartan_outer_uniskip
            .trace_cutoff_elements
            .min(config.spartan_outer_remainder.trace_cutoff_elements);
    let instruction_input = cycles >= config.instruction_input.trace_cutoff_elements
        && cycles > config.instruction_input.cutoff_elements;
    (stage1, instruction_input)
}

fn resident_row_working_set(
    cycles: usize,
    stage1: bool,
    instruction_input: bool,
    metal_uniskip: bool,
    metal_remainder: bool,
    remainder_dispatch: OuterRemainderSequenceConfig,
) -> Result<u64, MetalError> {
    let row_bytes = if stage1 {
        spartan_outer_uniskip_row_bytes(cycles)?
    } else if instruction_input {
        instruction_input_row_bytes(cycles)?
    } else {
        0
    };
    let instruction_input_bytes = if instruction_input {
        instruction_input_sequence_storage_bytes(cycles)?
    } else {
        0
    };
    let uniskip_bytes = if metal_uniskip {
        spartan_outer_uniskip_invocation_bytes(cycles)?
    } else {
        0
    };
    let remainder_bytes = if metal_remainder {
        outer_remainder_sequence_storage_bytes_with_config(cycles, remainder_dispatch)?
    } else {
        0
    };
    row_bytes
        .checked_add(instruction_input_bytes)
        .and_then(|bytes| bytes.checked_add(uniskip_bytes))
        .and_then(|bytes| bytes.checked_add(remainder_bytes))
        .ok_or(MetalError::InputTooLong(cycles))
}

fn validate_resident_row_buffer(row_bytes: u64, maximum: u64) -> Result<(), MetalError> {
    if row_bytes > maximum {
        return Err(MetalError::BufferTooLong {
            requested: row_bytes,
            maximum,
        });
    }
    Ok(())
}

fn use_metal_stage1(cycles: usize, config: &MetalConfig, resident_rows: bool) -> bool {
    cycles >= config.spartan_outer_uniskip.trace_cutoff_elements && resident_rows
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResidentRowPlan {
    stage1: bool,
    instruction_input: bool,
}

fn resident_row_admission_candidates(
    stage1_eligible: bool,
    instruction_input_eligible: bool,
) -> Vec<ResidentRowPlan> {
    match (stage1_eligible, instruction_input_eligible) {
        (true, true) => vec![
            ResidentRowPlan {
                stage1: true,
                instruction_input: true,
            },
            ResidentRowPlan {
                stage1: false,
                instruction_input: true,
            },
            ResidentRowPlan {
                stage1: true,
                instruction_input: false,
            },
        ],
        (true, false) => vec![ResidentRowPlan {
            stage1: true,
            instruction_input: false,
        }],
        (false, true) => vec![ResidentRowPlan {
            stage1: false,
            instruction_input: true,
        }],
        (false, false) => Vec::new(),
    }
}

fn prepare_cpu_instruction_input_now(
    metal_prepared: bool,
    stage1_rows_resident: bool,
    cpu_prepared: bool,
) -> bool {
    !metal_prepared && !stage1_rows_resident && !cpu_prepared
}

impl Default for SpartanOuterUniskipMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 18,
            dispatch: SpartanOuterUniskipConfig::default(),
        }
    }
}

impl MetalBackend {
    fn prepare_outer_remainder_storage(
        &self,
        session: &mut ProofSession,
        cycles: usize,
    ) -> Result<(), KernelError<AkitaField>> {
        let config = self.config.spartan_outer_remainder;
        if cycles < config.trace_cutoff_elements
            || session.state::<SpartanOuterUniskipRows>().is_none()
        {
            return Ok(());
        }
        let planned_device_bytes =
            outer_remainder_sequence_storage_bytes_with_config(cycles, config.dispatch)
                .map_err(metal_prepare_error)?;
        let maximum_buffer_bytes =
            outer_remainder_sequence_max_buffer_bytes_with_config(cycles, config.dispatch)
                .map_err(metal_prepare_error)?;
        let device = self.context.device_info();
        let span = tracing::info_span!(
            "MetalOuterRemainder::storage_prepare",
            cycles,
            planned_device_bytes,
            maximum_buffer_bytes,
            current_device_bytes = device.current_allocated_size,
            recommended_max_working_set_bytes = device.recommended_max_working_set_size,
            initialization_mode = config.dispatch.storage_initialization.as_str(),
            admitted = tracing::field::Empty,
            initialized = tracing::field::Empty,
            fallback_reason = tracing::field::Empty,
            device_buffers = tracing::field::Empty,
            initialization_bytes = tracing::field::Empty,
            initialization_wall_ns = tracing::field::Empty,
            initialization_gpu_active_ns = tracing::field::Empty,
            buffer_0 = tracing::field::Empty,
            buffer_1 = tracing::field::Empty,
            buffer_2 = tracing::field::Empty,
            buffer_3 = tracing::field::Empty,
            buffer_4 = tracing::field::Empty,
            buffer_5 = tracing::field::Empty,
            buffer_6 = tracing::field::Empty,
            buffer_7 = tracing::field::Empty,
            buffer_8 = tracing::field::Empty,
        );
        let _span = span.enter();
        match self
            .context
            .prepare_outer_remainder_sequence_storage(cycles, config.dispatch)
        {
            Ok(storage) => {
                let initialization = storage.initialization();
                let ids = initialization.buffer_identities;
                let _ = span.record("admitted", true);
                let _ = span.record("initialized", initialization.bytes != 0);
                let _ = span.record("fallback_reason", "none");
                let _ = span.record("device_buffers", initialization.device_buffers);
                let _ = span.record("initialization_bytes", initialization.bytes);
                let _ = span.record(
                    "initialization_wall_ns",
                    duration_nanos(initialization.wall),
                );
                let _ = span.record(
                    "initialization_gpu_active_ns",
                    duration_nanos(initialization.gpu_active),
                );
                let _ = span.record("buffer_0", ids[0]);
                let _ = span.record("buffer_1", ids[1]);
                let _ = span.record("buffer_2", ids[2]);
                let _ = span.record("buffer_3", ids[3]);
                let _ = span.record("buffer_4", ids[4]);
                let _ = span.record("buffer_5", ids[5]);
                let _ = span.record("buffer_6", ids[6]);
                let _ = span.record("buffer_7", ids[7]);
                let _ = span.record("buffer_8", ids[8]);
                tracing::info!(
                    target: "jolt::metal",
                    bytes = storage.owned_bytes(),
                    "admitted pre-touched outer-remainder Metal storage"
                );
                session.park(storage);
                Ok(())
            }
            Err(error) => match outer_remainder_storage_fallback_reason(&error) {
                Some(reason) => {
                    let _ = span.record("admitted", false);
                    let _ = span.record("initialized", false);
                    let _ = span.record("fallback_reason", reason);
                    tracing::warn!(
                        target: "jolt::metal",
                        error = %error,
                        fallback_reason = reason,
                        "outer-remainder Metal storage preparation failed; using optimized CPU"
                    );
                    Ok(())
                }
                None => Err(metal_prepare_error(error)),
            },
        }
    }
}

fn outer_remainder_storage_fallback_reason(error: &MetalError) -> Option<&'static str> {
    if error.is_capacity_error() {
        return Some("capacity");
    }
    match error {
        MetalError::CommandFailed(_) => Some("command_failed"),
        MetalError::GpuTimestampLookup { .. } => Some("gpu_timestamp_lookup"),
        MetalError::InvalidGpuTimestamps { .. } => Some("invalid_gpu_timestamps"),
        _ => None,
    }
}

impl UniskipKernel<AkitaField, OuterRemainder<AkitaField>> for MetalBackend {
    fn prepare_witness(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<(), KernelError<AkitaField>> {
        let cycles = 1usize << log_t;
        self.prepare_ram_raf_witness(session, log_t, witness)?;
        self.prepare_product_remainder_witness(session, log_t, witness)?;
        let (stage1_eligible, instruction_input_eligible) =
            resident_row_consumers(cycles, &self.config);
        let instruction_ra_dispatch = self.config.instruction_ra_virtualization.dispatch;
        if cycles
            >= self
                .config
                .instruction_ra_virtualization
                .trace_cutoff_elements
            && cycles >= 2 * instruction_ra_dispatch.materialize_width.elements()
        {
            let (e_in_capacity, e_out_capacity) =
                instruction_ra_weight_capacities(cycles).map_err(metal_prepare_error)?;
            let storage = {
                let _span =
                    tracing::info_span!("MetalInstructionRaVirtualization::storage_prepare")
                        .entered();
                self.context
                    .prepare_instruction_ra_sequence_storage(
                        cycles,
                        e_in_capacity,
                        e_out_capacity,
                        instruction_ra_dispatch,
                    )
                    .map_err(metal_prepare_error)?
            };
            session.park::<InstructionRaSequenceStorage>(storage);
        }
        let mut admitted_plan = None;
        if stage1_eligible || instruction_input_eligible {
            let instruction_input_bytes =
                instruction_input_row_bytes(cycles).map_err(metal_prepare_error)?;
            let device = self.context.device_info();
            for candidate in
                resident_row_admission_candidates(stage1_eligible, instruction_input_eligible)
            {
                let residual_bytes = if candidate.stage1 {
                    spartan_outer_uniskip_row_bytes(cycles)
                        .map_err(metal_prepare_error)?
                        .checked_sub(instruction_input_bytes)
                        .ok_or(KernelError::InvariantViolation {
                            reason: "Spartan row split exceeds the original row footprint",
                        })?
                } else {
                    0
                };
                let admission =
                    validate_resident_row_buffer(instruction_input_bytes, device.max_buffer_length)
                        .and_then(|()| {
                            validate_resident_row_buffer(residual_bytes, device.max_buffer_length)
                        })
                        .and_then(|()| {
                            if candidate.stage1
                                && cycles
                                    >= self.config.spartan_outer_remainder.trace_cutoff_elements
                            {
                                validate_resident_row_buffer(
                                    outer_remainder_sequence_max_buffer_bytes_with_config(
                                        cycles,
                                        self.config.spartan_outer_remainder.dispatch,
                                    )?,
                                    device.max_buffer_length,
                                )?;
                            }
                            resident_row_working_set(
                                cycles,
                                candidate.stage1,
                                candidate.instruction_input,
                                candidate.stage1
                                    && cycles
                                        >= self.config.spartan_outer_uniskip.trace_cutoff_elements,
                                candidate.stage1
                                    && cycles
                                        >= self
                                            .config
                                            .spartan_outer_remainder
                                            .trace_cutoff_elements,
                                self.config.spartan_outer_remainder.dispatch,
                            )
                        })
                        .and_then(|additional| {
                            self.context.validate_additional_working_set(additional)
                        });
                match admission {
                    Ok(()) => {
                        admitted_plan = Some(candidate);
                        break;
                    }
                    Err(error) if error.is_capacity_error() => {
                        tracing::warn!(
                            target: "jolt::metal",
                            error = %error,
                            stage1 = candidate.stage1,
                            instruction_input = candidate.instruction_input,
                            "Metal resident-row plan was not admitted"
                        );
                    }
                    Err(error) => return Err(metal_prepare_error(error)),
                }
            }
        }
        if let Some(plan) = admitted_plan {
            if plan.stage1 {
                let mut rows =
                    prepare_metal_spartan_outer_witness_rows(&self.context, witness, cycles)?;
                let compact_rows = plan
                    .instruction_input
                    .then(|| rows.share_instruction_input_rows());
                session.park(rows);
                if let Some(compact_rows) = compact_rows {
                    session.park(compact_rows);
                }
            } else {
                let rows =
                    prepare_metal_instruction_input_witness_rows(&self.context, witness, cycles)?;
                session.park(rows);
            }
        }
        self.prepare_instruction_input_storage(session, cycles)?;
        self.prepare_outer_remainder_storage(session, cycles)?;
        if session.state::<PreparedInstructionInput>().is_none() {
            drop(session.take::<InstructionInputRows>());
            if let Some(mut rows) = session.take::<SpartanOuterUniskipRows>() {
                rows.restore_instruction_input_accounting();
                session.park(rows);
            }
        }
        if prepare_cpu_instruction_input_now(
            session.state::<PreparedInstructionInput>().is_some(),
            session.state::<SpartanOuterUniskipRows>().is_some(),
            session.state::<PreparedInstructionInputRows>().is_some(),
        ) {
            <OptimizedOuterUniskip as UniskipKernel<
                AkitaField,
                OuterRemainder<AkitaField>,
            >>::prepare_witness(&OptimizedOuterUniskip, session, log_t, witness)?;
        }
        Ok(())
    }

    fn prepare(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        tau: &[AkitaField],
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<(), KernelError<AkitaField>> {
        let cycles = 1usize << log_t;
        let resident_rows = session.state::<SpartanOuterUniskipRows>().is_some();
        if !use_metal_stage1(cycles, &self.config, resident_rows) {
            let retain_for_remainder = use_metal_remainder(cycles, &self.config, resident_rows);
            if !retain_for_remainder {
                drop(session.take::<SpartanOuterUniskipRows>());
            }
            if prepare_cpu_instruction_input_now(
                session.state::<PreparedInstructionInput>().is_some(),
                false,
                session.state::<PreparedInstructionInputRows>().is_some(),
            ) {
                <OptimizedOuterUniskip as UniskipKernel<
                    AkitaField,
                    OuterRemainder<AkitaField>,
                >>::prepare_witness(&OptimizedOuterUniskip, session, log_t, witness)?;
            }
            <OptimizedOuterUniskip as UniskipKernel<
                AkitaField,
                OuterRemainder<AkitaField>,
            >>::prepare(&OptimizedOuterUniskip, session, log_t, tau, witness)?;
            return Ok(());
        }
        let stage1_compact_rows_storage_id = session
            .state::<SpartanOuterUniskipRows>()
            .map(SpartanOuterUniskipRows::instruction_input_allocation_identity)
            .ok_or(KernelError::InvariantViolation {
                reason: "Metal Spartan stage 1 lost its compact row buffer",
            })?;
        let compact_rows_storage_id = session
            .state::<InstructionInputRows>()
            .map(InstructionInputRows::allocation_identity);
        if compact_rows_storage_id.is_some_and(|id| id != stage1_compact_rows_storage_id) {
            return Err(KernelError::InvariantViolation {
                reason: "Metal stage 1 and InstructionInput disagree on the compact allocation",
            });
        }
        prepare_metal_spartan_outer_uniskip(
            &self.context,
            self.config.spartan_outer_uniskip.dispatch,
            session,
            log_t,
            tau,
            witness,
        )?;
        if let Some(compact_rows_storage_id) = compact_rows_storage_id {
            if session
                .state::<InstructionInputRows>()
                .map(InstructionInputRows::allocation_identity)
                != Some(compact_rows_storage_id)
            {
                return Err(KernelError::InvariantViolation {
                    reason: "Metal stage 1 changed the InstructionInput compact allocation",
                });
            }
        }
        if prepare_cpu_instruction_input_now(
            session.state::<PreparedInstructionInput>().is_some(),
            false,
            session.state::<PreparedInstructionInputRows>().is_some(),
        ) {
            <OptimizedOuterUniskip as UniskipKernel<
                AkitaField,
                OuterRemainder<AkitaField>,
            >>::prepare_witness(&OptimizedOuterUniskip, session, log_t, witness)?;
        }
        Ok(())
    }

    fn first_round_poly(
        &self,
        session: &mut ProofSession,
        late_tau: &[AkitaField],
        known_values: &[AkitaField],
    ) -> Result<UnivariatePoly<AkitaField>, KernelError<AkitaField>> {
        <OptimizedOuterUniskip as UniskipKernel<
            AkitaField,
            OuterRemainder<AkitaField>,
        >>::first_round_poly(&OptimizedOuterUniskip, session, late_tau, known_values)
    }
}

fn metal_prepare_error(error: MetalError) -> KernelError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
    .into()
}

fn metal_round_error(error: MetalError) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
}

fn metal_output_error(error: MetalError) -> SumcheckKernelError<AkitaField> {
    SumcheckKernelError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
}

fn invalid_outer_remainder_state(expected: &'static str, got: &'static str) -> MetalError {
    MetalError::InvalidOuterRemainderState { expected, got }
}

fn duration_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn record_gpu_phase(
    span: &tracing::Span,
    started: Instant,
    gpu_before: Duration,
    gpu_after: Duration,
) -> Duration {
    let gpu_active = gpu_after.saturating_sub(gpu_before);
    let _ = span.record("dispatch_wall_ns", duration_nanos(started.elapsed()));
    let _ = span.record("gpu_active_ns", duration_nanos(gpu_active));
    gpu_active
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OuterRemainderGpuActiveBreakdown {
    pub materialize: Duration,
    pub first_bind: Duration,
    pub dense_rounds: Duration,
    pub openings: Duration,
}

impl OuterRemainderGpuActiveBreakdown {
    fn total(self) -> Option<Duration> {
        self.materialize
            .checked_add(self.first_bind)?
            .checked_add(self.dense_rounds)?
            .checked_add(self.openings)
    }
}

impl PrepareKernel<AkitaField, OuterRemainder<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, OuterRemainder<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = OuterRemainder<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let rounds = inputs.relation.rounds();
        let log_t = rounds
            .checked_sub(1)
            .ok_or(KernelError::InvariantViolation {
                reason: "Spartan outer remainder has no stream round",
            })?;
        let cycles = 1usize
            .checked_shl(log_t as u32)
            .ok_or(KernelError::InvariantViolation {
                reason: "Spartan outer remainder trace length overflows usize",
            })?;
        let rows_present = session.state::<SpartanOuterUniskipRows>().is_some();
        let storage_present = session.state::<OuterRemainderSequenceStorage>().is_some();
        if storage_present && !rows_present {
            return Err(KernelError::InvariantViolation {
                reason: "Metal outer remainder retained storage without its resident rows",
            });
        }
        let resident = rows_present && storage_present;
        if !use_metal_remainder(cycles, &self.config, resident) {
            drop(session.take::<SpartanOuterUniskipRows>());
            drop(session.take::<OuterRemainderSequenceStorage>());
            return OptimizedOuterRemainder.prepare(session, witness, inputs);
        }
        let _metal_prepare = tracing::info_span!(
            "MetalOuterRemainder::prepare",
            resident_rows = cycles,
            rounds,
        )
        .entered();

        let rows =
            session
                .state::<SpartanOuterUniskipRows>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "Metal outer remainder lost its resident row owner",
                })?;
        if rows.len() != cycles || rows.device_registry_id() != self.context.device_registry_id() {
            return Err(KernelError::InvariantViolation {
                reason: "Metal outer remainder resident rows have the wrong shape or device",
            });
        }
        let dispatch = self.config.spartan_outer_remainder.dispatch;
        let device = self.context.device_info();
        let planned_device_bytes =
            outer_remainder_sequence_storage_bytes_with_config(cycles, dispatch)
                .map_err(metal_prepare_error)?;
        let storage = session.state::<OuterRemainderSequenceStorage>().ok_or(
            KernelError::InvariantViolation {
                reason: "Metal outer remainder lost its prepared storage",
            },
        )?;
        if !storage.matches(&self.context, cycles, dispatch)
            || storage.owned_bytes() != planned_device_bytes
        {
            return Err(KernelError::InvariantViolation {
                reason: "prepared outer-remainder storage disagrees with the relation geometry",
            });
        }
        let existing_resident_bytes =
            spartan_outer_uniskip_row_bytes(cycles).map_err(metal_prepare_error)?;
        let _allocation_span = tracing::info_span!(
            "MetalOuterRemainder::allocation_plan",
            admitted = true,
            storage_reused = true,
            existing_resident_bytes,
            preallocated_device_bytes = planned_device_bytes,
            additional_working_set_bytes = 0u64,
            current_device_bytes = device.current_allocated_size,
            recommended_max_working_set_bytes = device.recommended_max_working_set_size,
        )
        .entered();
        drop(_allocation_span);

        let compact_rows_storage_id = rows.instruction_input_allocation_identity();
        let residual_rows_storage_id = rows.allocation_identity();
        let device_registry_id = rows.device_registry_id();
        let compact_retained = session.state::<InstructionInputRows>().is_some();
        let rows = {
            let _row_handoff = tracing::info_span!(
                "MetalOuterRemainder::row_handoff",
                compact_rows_storage_id,
                residual_rows_storage_id,
                device_registry_id,
                resident_rows = cycles,
                row_upload_bytes = 0u64,
                device_allocations = 0u64,
            )
            .entered();
            session
                .take::<SpartanOuterUniskipRows>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "Metal outer remainder rows disappeared after admission",
                })?
        };
        let storage = session.take::<OuterRemainderSequenceStorage>().ok_or(
            KernelError::InvariantViolation {
                reason: "Metal outer remainder storage disappeared after validation",
            },
        )?;
        let storage_initialization = storage.initialization();
        let sequence_span = tracing::info_span!(
            "MetalOuterRemainder::sequence_prepare",
            resident_rows = cycles,
            rounds,
            cutoff_elements = dispatch.cpu_tail_elements,
            trace_cutoff_elements = self.config.spartan_outer_remainder.trace_cutoff_elements,
            planned_device_bytes,
            compact_rows_storage_id,
            residual_rows_storage_id,
            device_registry_id,
            storage_reused = true,
            storage_initialization_mode = dispatch.storage_initialization.as_str(),
            preinitialized_device_bytes = planned_device_bytes,
            initialization_bytes = storage_initialization.bytes,
            attached_owned_bytes = tracing::field::Empty,
            storage_buffer_0 = tracing::field::Empty,
            storage_buffer_1 = tracing::field::Empty,
            storage_buffer_2 = tracing::field::Empty,
            storage_buffer_3 = tracing::field::Empty,
            storage_buffer_4 = tracing::field::Empty,
            storage_buffer_5 = tracing::field::Empty,
            storage_buffer_6 = tracing::field::Empty,
            storage_buffer_7 = tracing::field::Empty,
            storage_buffer_8 = tracing::field::Empty,
            row_upload_bytes = 0u64,
            full_domain_copy_dispatches = 0u64,
            sequence_device_buffer_allocations = 0u64,
            round_device_buffer_allocations = 0u64,
        );
        let _sequence_span = sequence_span.enter();
        let sequence = storage.attach(rows).map_err(metal_prepare_error)?;
        let attached = sequence.storage_stats().map_err(metal_prepare_error)?;
        if attached.owned_bytes != planned_device_bytes
            || attached.buffer_identities != storage_initialization.buffer_identities
        {
            return Err(KernelError::InvariantViolation {
                reason: "attached outer-remainder storage changed allocation identity",
            });
        }
        let attached_ids = attached.buffer_identities;
        let _ = sequence_span.record("attached_owned_bytes", attached.owned_bytes);
        let _ = sequence_span.record("storage_buffer_0", attached_ids[0]);
        let _ = sequence_span.record("storage_buffer_1", attached_ids[1]);
        let _ = sequence_span.record("storage_buffer_2", attached_ids[2]);
        let _ = sequence_span.record("storage_buffer_3", attached_ids[3]);
        let _ = sequence_span.record("storage_buffer_4", attached_ids[4]);
        let _ = sequence_span.record("storage_buffer_5", attached_ids[5]);
        let _ = sequence_span.record("storage_buffer_6", attached_ids[6]);
        let _ = sequence_span.record("storage_buffer_7", attached_ids[7]);
        let _ = sequence_span.record("storage_buffer_8", attached_ids[8]);
        drop(_sequence_span);

        // Prepared storage and resident rows have now been consumed together.
        // Subsequent device failures are terminal because the sumcheck state
        // may already have advanced.
        let tau = take_metal_spartan_outer_tau(session, log_t)?;
        #[cfg(any(test, feature = "test-utils"))]
        let _ = self
            .outer_remainder_sequences
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let metadata = MetalOuterResidentMetadata {
            compact_rows_storage_id,
            residual_rows_storage_id,
            device_registry_id,
            resident_rows: cycles,
            compact_retained,
        };
        Ok(Box::new(MetalOuterRemainderKernel::from_attached_sequence(
            log_t,
            &tau,
            inputs.relation.uniskip_challenge(),
            sequence,
            metadata,
            self.config
                .spartan_outer_remainder
                .dispatch
                .cpu_tail_elements,
        )?))
    }
}

struct MetalOuterCpuTail {
    az: Polynomial<AkitaField>,
    bz: Polynomial<AkitaField>,
    scratch: Vec<AkitaField>,
}

impl MetalOuterCpuTail {
    fn new(az: Vec<AkitaField>, bz: Vec<AkitaField>) -> Result<Self, SumcheckError<AkitaField>> {
        if az.len() != bz.len() || az.is_empty() || !az.len().is_power_of_two() {
            return Err(metal_round_error(invalid_outer_remainder_state(
                "equal nonempty power-of-two CPU-tail tables",
                "inconsistent CPU-tail table lengths",
            )));
        }
        Ok(Self {
            az: Polynomial::new(az),
            bz: Polynomial::new(bz),
            scratch: Vec::new(),
        })
    }

    fn bind(&mut self, challenge: AkitaField) {
        self.az
            .bind_low_to_high_reusing_scratch(challenge, &mut self.scratch);
        self.bz
            .bind_low_to_high_reusing_scratch(challenge, &mut self.scratch);
    }

    fn endpoints(
        &self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<(AkitaField, AkitaField), SumcheckError<AkitaField>> {
        let az = self.az.evals();
        let bz = self.bz.evals();
        let in_len = e_in.len();
        let expected = e_out
            .len()
            .checked_mul(in_len)
            .and_then(|pairs| pairs.checked_mul(2));
        if expected != Some(az.len()) || bz.len() != az.len() {
            return Err(metal_round_error(invalid_outer_remainder_state(
                "CPU-tail Az/Bz lengths equal 2 * e_in * e_out",
                "inconsistent CPU-tail weight geometry",
            )));
        }
        let block = |x_out: usize| {
            let mut q_zero = AkitaField::zero();
            let mut q_infinity = AkitaField::zero();
            for (x_in, &e) in e_in.iter().enumerate() {
                let pair = 2 * (x_out * in_len + x_in);
                let az_low = az[pair];
                let az_high = az[pair + 1];
                let bz_low = bz[pair];
                let bz_high = bz[pair + 1];
                q_zero += e * az_low * bz_low;
                q_infinity += e * (az_high - az_low) * (bz_high - bz_low);
            }
            (e_out[x_out] * q_zero, e_out[x_out] * q_infinity)
        };
        let add = |left: (AkitaField, AkitaField), right: (AkitaField, AkitaField)| {
            (left.0 + right.0, left.1 + right.1)
        };
        #[cfg(feature = "parallel")]
        let endpoints = (0..e_out.len())
            .into_par_iter()
            .map(block)
            .reduce(|| (AkitaField::zero(), AkitaField::zero()), add);
        #[cfg(not(feature = "parallel"))]
        let endpoints = (0..e_out.len())
            .map(block)
            .fold((AkitaField::zero(), AkitaField::zero()), add);
        Ok(endpoints)
    }
}

struct MetalOuterDerived {
    az_weights: [Vec<AkitaField>; 2],
    bz_weights: [Vec<AkitaField>; 2],
    az_constant: [AkitaField; 2],
    bz_constant: [AkitaField; 2],
}

struct MetalOuterRemainderHost {
    rounds: usize,
    split_eq: GruenSplitEqPolynomial<AkitaField>,
    challenges: Vec<AkitaField>,
    opening_ids: Vec<JoltOpeningId>,
    derived: MetalOuterDerived,
    stream_lagrange: [AkitaField; OUTER_DOMAIN],
}

impl MetalOuterRemainderHost {
    fn new(
        log_t: usize,
        tau: &[AkitaField],
        uniskip_challenge: AkitaField,
    ) -> Result<Self, KernelError<AkitaField>> {
        if tau.len() != log_t + 2 {
            return Err(KernelError::InvariantViolation {
                reason: "Spartan outer tau must carry log_t + 2 challenges",
            });
        }
        let tau_high = tau[log_t + 1];
        let tau_low = &tau[..=log_t];
        let stream_lagrange: [AkitaField; OUTER_DOMAIN] =
            centered_lagrange_evals(OUTER_DOMAIN, uniskip_challenge)?
                .try_into()
                .map_err(|_| KernelError::InvariantViolation {
                    reason: "Spartan outer stream Lagrange vector has the wrong length",
                })?;
        let kernel = centered_lagrange_kernel(OUTER_DOMAIN, tau_high, uniskip_challenge)?;
        let split_eq = GruenSplitEqPolynomial::new_with_scaling(
            tau_low,
            BindingOrder::LowToHigh,
            Some(kernel),
        );
        let opening_ids = SpartanOuterDimensions::rv64(log_t)
            .variables()
            .iter()
            .map(|&variable| outer_opening(variable))
            .collect::<Vec<_>>();
        let matrices = spartan_outer_constraints::<AkitaField>();
        let columns = (1..=opening_ids.len()).collect::<Vec<_>>();
        let mut derived = MetalOuterDerived {
            az_weights: [Vec::new(), Vec::new()],
            bz_weights: [Vec::new(), Vec::new()],
            az_constant: [AkitaField::zero(); 2],
            bz_constant: [AkitaField::zero(); 2],
        };
        for (index, stream) in [AkitaField::zero(), AkitaField::one()]
            .into_iter()
            .enumerate()
        {
            let weights = spartan_outer_row_weights(uniskip_challenge, stream)?;
            let weighted = matrices.weighted_columns(&weights, &columns)?;
            let constants = matrices.public_column_contributions(&weights, 0, AkitaField::one())?;
            derived.az_weights[index] = weighted.a;
            derived.bz_weights[index] = weighted.b;
            derived.az_constant[index] = constants.a;
            derived.bz_constant[index] = constants.b;
        }
        Ok(Self {
            rounds: log_t + 1,
            split_eq,
            challenges: Vec::with_capacity(log_t + 1),
            opening_ids,
            derived,
            stream_lagrange,
        })
    }

    fn stream_lagrange(&self) -> &[AkitaField; OUTER_DOMAIN] {
        &self.stream_lagrange
    }

    fn current_weights(&self) -> (&[AkitaField], &[AkitaField]) {
        (self.split_eq.e_in_current(), self.split_eq.e_out_current())
    }

    fn bind(&mut self, challenge: AkitaField) {
        self.split_eq.bind(challenge);
        self.challenges.push(challenge);
    }

    fn polynomial(
        &self,
        endpoints: (AkitaField, AkitaField),
        previous_claim: AkitaField,
    ) -> UnivariatePoly<AkitaField> {
        self.split_eq
            .gruen_poly_deg_3(endpoints.0, endpoints.1, previous_claim)
    }

    fn opening_weights(&self) -> (Vec<AkitaField>, Vec<AkitaField>) {
        let point = self.challenges[1..]
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();
        let split = point.len() / 2;
        let (out_point, in_point) = point.split_at(split);
        (
            EqPolynomial::evals(in_point, None),
            EqPolynomial::evals(out_point, None),
        )
    }

    fn output_claims(
        &self,
        inputs: &SumcheckInputClaims<AkitaField, OuterRemainder<AkitaField>>,
        claimed: &[AkitaField; OUTER_VARIABLES],
    ) -> Result<
        SumcheckOutputClaims<AkitaField, OuterRemainder<AkitaField>>,
        SumcheckKernelError<AkitaField>,
    > {
        let claims: BTreeMap<JoltOpeningId, AkitaField> = self
            .opening_ids
            .iter()
            .copied()
            .zip(claimed.iter().copied())
            .collect();
        SumcheckOutputClaims::<AkitaField, OuterRemainder<AkitaField>>::from_opening_values(|id| {
            claims.get(id).copied().or_else(|| inputs.resolve_input(id))
        })
        .map_err(SumcheckKernelError::from)
    }

    fn validate_derived_tables(
        &self,
        relation: &OuterRemainder<AkitaField>,
        input_points: &SumcheckInputPoints<AkitaField, OuterRemainder<AkitaField>>,
        output_points: &SumcheckOutputPoints<AkitaField, OuterRemainder<AkitaField>>,
        challenges: &ConcreteSumcheckChallenges<AkitaField, OuterRemainder<AkitaField>>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        let remaining = self.rounds - self.challenges.len();
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let stream = self.challenges[0];
        let blend = |pair: [&AkitaField; 2]| *pair[0] + stream * (*pair[1] - *pair[0]);
        let ids = std::iter::once(SpartanOuterPublic::TauKernel)
            .chain((0..self.opening_ids.len()).map(SpartanOuterPublic::AzWeight))
            .chain((0..self.opening_ids.len()).map(SpartanOuterPublic::BzWeight))
            .chain([
                SpartanOuterPublic::AzConstant,
                SpartanOuterPublic::BzConstant,
            ]);
        for public_id in ids {
            let id = JoltDerivedId::from(public_id);
            let expected =
                match relation.derive_output_term(&id, input_points, output_points, challenges) {
                    Ok(value) => value,
                    Err(VerifierError::MissingStageClaimDerived { .. }) => continue,
                    Err(error) => return Err(error.into()),
                };
            let got = match public_id {
                SpartanOuterPublic::TauKernel => self.split_eq.current_scalar(),
                SpartanOuterPublic::AzWeight(index) => blend([
                    &self.derived.az_weights[0][index],
                    &self.derived.az_weights[1][index],
                ]),
                SpartanOuterPublic::BzWeight(index) => blend([
                    &self.derived.bz_weights[0][index],
                    &self.derived.bz_weights[1][index],
                ]),
                SpartanOuterPublic::AzConstant => {
                    blend([&self.derived.az_constant[0], &self.derived.az_constant[1]])
                }
                SpartanOuterPublic::BzConstant => {
                    blend([&self.derived.bz_constant[0], &self.derived.bz_constant[1]])
                }
            };
            if got != expected {
                return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct MetalOuterResidentMetadata {
    compact_rows_storage_id: usize,
    residual_rows_storage_id: usize,
    device_registry_id: u64,
    resident_rows: usize,
    compact_retained: bool,
}

struct MetalOuterRemainderKernel {
    host: MetalOuterRemainderHost,
    sequence: Option<OuterRemainderSequence>,
    cpu_tail: Option<MetalOuterCpuTail>,
    host_tail: Option<(Vec<AkitaField>, Vec<AkitaField>)>,
    pending_endpoints: Option<(AkitaField, AkitaField)>,
    compact_rows_storage_id: usize,
    residual_rows_storage_id: usize,
    device_registry_id: u64,
    resident_rows: usize,
    compact_retained: bool,
    cpu_tail_elements: usize,
    gpu_active_breakdown: OuterRemainderGpuActiveBreakdown,
    #[cfg(feature = "test-utils")]
    completed_gpu_active: Option<Duration>,
    #[cfg(feature = "test-utils")]
    completed_gpu_active_breakdown: Option<OuterRemainderGpuActiveBreakdown>,
    #[cfg(feature = "test-utils")]
    completed_dispatch_counts: Option<super::solinas::OuterRemainderDispatchCounts>,
    #[cfg(feature = "test-utils")]
    completed_tail_elements: Option<usize>,
    #[cfg(feature = "test-utils")]
    completed_round_device_buffer_allocations: Option<usize>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalOuterRemainderKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        if let Some(sequence) = &self.sequence {
            visitor.visit_field(allocative::Key::new("sequence"), sequence);
        }
        if let Some((az, bz)) = &self.host_tail {
            visitor.visit_simple(
                allocative::Key::new("host_tail"),
                crate::backend::vec_heap_bytes(az) + crate::backend::vec_heap_bytes(bz),
            );
        }
        visitor.exit();
    }
}

impl MetalOuterRemainderKernel {
    fn from_attached_sequence(
        log_t: usize,
        tau: &[AkitaField],
        uniskip_challenge: AkitaField,
        mut sequence: OuterRemainderSequence,
        metadata: MetalOuterResidentMetadata,
        cpu_tail_elements: usize,
    ) -> Result<Self, KernelError<AkitaField>> {
        if metadata
            .resident_rows
            .checked_mul(2)
            .is_none_or(|elements| sequence.current_elements() != elements)
        {
            return Err(KernelError::InvariantViolation {
                reason: "attached Outer sequence has the wrong trace length",
            });
        }
        let host = MetalOuterRemainderHost::new(log_t, tau, uniskip_challenge)?;
        let (endpoints, materialize_gpu_active) = {
            let phase = tracing::info_span!(
                "MetalOuterRemainder::first_message",
                dispatch_wall_ns = tracing::field::Empty,
                gpu_active_ns = tracing::field::Empty,
            );
            let _phase_guard = phase.enter();
            let (e_in, e_out) = host.current_weights();
            let gpu_before = sequence.gpu_active_time();
            let started = Instant::now();
            let endpoints = sequence
                .materialize_and_first_message(host.stream_lagrange(), e_in, e_out)
                .map_err(metal_prepare_error)?;
            let gpu_active =
                record_gpu_phase(&phase, started, gpu_before, sequence.gpu_active_time());
            (endpoints, gpu_active)
        };
        let tail_elements = metadata.resident_rows.min(cpu_tail_elements);
        Ok(Self {
            host,
            sequence: Some(sequence),
            cpu_tail: None,
            host_tail: Some((
                vec![AkitaField::zero(); tail_elements],
                vec![AkitaField::zero(); tail_elements],
            )),
            pending_endpoints: Some((endpoints[0], endpoints[1])),
            compact_rows_storage_id: metadata.compact_rows_storage_id,
            residual_rows_storage_id: metadata.residual_rows_storage_id,
            device_registry_id: metadata.device_registry_id,
            resident_rows: metadata.resident_rows,
            compact_retained: metadata.compact_retained,
            cpu_tail_elements: tail_elements,
            gpu_active_breakdown: OuterRemainderGpuActiveBreakdown {
                materialize: materialize_gpu_active,
                ..OuterRemainderGpuActiveBreakdown::default()
            },
            #[cfg(feature = "test-utils")]
            completed_gpu_active: None,
            #[cfg(feature = "test-utils")]
            completed_gpu_active_breakdown: None,
            #[cfg(feature = "test-utils")]
            completed_dispatch_counts: None,
            #[cfg(feature = "test-utils")]
            completed_tail_elements: None,
            #[cfg(feature = "test-utils")]
            completed_round_device_buffer_allocations: None,
        })
    }

    fn restore_cpu_tail(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let sequence = self.sequence.as_mut().ok_or_else(|| {
            metal_round_error(invalid_outer_remainder_state(
                "resident sequence during CPU-tail export",
                "absent device sequence",
            ))
        })?;
        let current = sequence.current_elements();
        let readback_elements = current
            .checked_mul(2)
            .ok_or_else(|| metal_round_error(MetalError::InputTooLong(current)))?;
        let readback_bytes = readback_elements
            .checked_mul(size_of::<AkitaField>())
            .ok_or_else(|| metal_round_error(MetalError::InputTooLong(readback_elements)))?;
        let _span = tracing::info_span!(
            "MetalOuterRemainder::readback",
            readbacks = 1u64,
            elements = readback_elements,
            bytes = readback_bytes,
        )
        .entered();
        let (mut az, mut bz) = self.host_tail.take().ok_or_else(|| {
            metal_round_error(invalid_outer_remainder_state(
                "available CPU-tail buffers",
                "already-consumed CPU-tail buffers",
            ))
        })?;
        if current > az.len() || current > bz.len() {
            return Err(metal_round_error(invalid_outer_remainder_state(
                "resident table within CPU-tail capacity",
                "oversized resident table",
            )));
        }
        sequence
            .export_cpu_tail(&mut az[..current], &mut bz[..current])
            .map_err(metal_round_error)?;
        az.truncate(current);
        bz.truncate(current);
        self.cpu_tail = Some(MetalOuterCpuTail::new(az, bz)?);
        Ok(())
    }
}

impl ProveRounds<AkitaField> for MetalOuterRemainderKernel {
    fn num_rounds(&self) -> usize {
        self.host.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        let endpoints = if let Some(challenge) = bind {
            let should_handoff = self.cpu_tail.is_none()
                && self.sequence.as_ref().is_some_and(|sequence| {
                    sequence.phase() == OuterRemainderPhase::Interleaved
                        && sequence.current_elements() <= self.cpu_tail_elements
                });
            if should_handoff {
                self.restore_cpu_tail()?;
            }
            self.host.bind(challenge);
            let (e_in, e_out) = self.host.current_weights();
            if let Some(tail) = self.cpu_tail.as_mut() {
                let _span = tracing::info_span!("MetalOuterRemainder::cpu_tail").entered();
                tail.bind(challenge);
                tail.endpoints(e_in, e_out)?
            } else {
                let sequence = self.sequence.as_mut().ok_or_else(|| {
                    metal_round_error(invalid_outer_remainder_state(
                        "resident sequence before a Metal round",
                        "absent device sequence",
                    ))
                })?;
                let first_bind = self.host.challenges.len() == 1;
                let phase = if first_bind {
                    tracing::info_span!(
                        "MetalOuterRemainder::first_bind",
                        round,
                        dispatch_wall_ns = tracing::field::Empty,
                        gpu_active_ns = tracing::field::Empty,
                    )
                } else {
                    tracing::info_span!(
                        "MetalOuterRemainder::dense_round",
                        round,
                        dispatch_wall_ns = tracing::field::Empty,
                        gpu_active_ns = tracing::field::Empty,
                    )
                };
                let _phase_guard = phase.enter();
                let gpu_before = sequence.gpu_active_time();
                let started = Instant::now();
                let output = if first_bind {
                    sequence.bind_stream_and_message(
                        challenge,
                        self.host.stream_lagrange(),
                        e_in,
                        e_out,
                    )
                } else {
                    sequence.bind_and_message(challenge, e_in, e_out)
                }
                .map_err(metal_round_error)?;
                let gpu_active =
                    record_gpu_phase(&phase, started, gpu_before, sequence.gpu_active_time());
                if first_bind {
                    self.gpu_active_breakdown.first_bind = gpu_active;
                } else {
                    self.gpu_active_breakdown.dense_rounds += gpu_active;
                }
                (output[0], output[1])
            }
        } else {
            self.pending_endpoints.take().ok_or_else(|| {
                metal_round_error(invalid_outer_remainder_state(
                    "pending first remainder message",
                    "already-consumed first remainder message",
                ))
            })?
        };
        Ok(self.host.polynomial(endpoints, previous_claim))
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if self.cpu_tail.is_none() {
            self.restore_cpu_tail()?;
        }
        self.host.bind(bind);
        let _span = tracing::info_span!("MetalOuterRemainder::cpu_tail", terminal = true).entered();
        self.cpu_tail
            .as_mut()
            .ok_or_else(|| {
                metal_round_error(invalid_outer_remainder_state(
                    "CPU tail before the terminal bind",
                    "absent CPU tail",
                ))
            })?
            .bind(bind);
        Ok(())
    }
}

impl SumcheckKernel<AkitaField> for MetalOuterRemainderKernel {
    type Relation = OuterRemainder<AkitaField>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<AkitaField, Self::Relation>, SumcheckKernelError<AkitaField>>
    {
        let remaining = self.host.rounds - self.host.challenges.len();
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let (e_in, e_out) = self.host.opening_weights();
        let mut sequence = self
            .sequence
            .take()
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "Metal outer remainder released resident rows before openings",
            })?;
        let claimed = {
            let phase = tracing::info_span!(
                "MetalOuterRemainder::output_claims",
                dispatch_wall_ns = tracing::field::Empty,
                gpu_active_ns = tracing::field::Empty,
                readbacks = 1u64,
                output_elements = OUTER_VARIABLES,
                readback_bytes = OUTER_VARIABLES * size_of::<AkitaField>(),
                row_upload_bytes = 0u64,
            );
            let _phase_guard = phase.enter();
            let gpu_before = sequence.gpu_active_time();
            let started = Instant::now();
            let claimed = sequence
                .evaluate_openings(&e_in, &e_out)
                .map_err(metal_output_error)?;
            self.gpu_active_breakdown.openings =
                record_gpu_phase(&phase, started, gpu_before, sequence.gpu_active_time());
            claimed
        };
        let completed_gpu_active = sequence.gpu_active_time();
        if self.gpu_active_breakdown.total() != Some(completed_gpu_active) {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "Outer remainder GPU phase timings do not sum to the member total",
            });
        }
        #[cfg(feature = "test-utils")]
        {
            self.completed_gpu_active = Some(completed_gpu_active);
            self.completed_gpu_active_breakdown = Some(self.gpu_active_breakdown);
            self.completed_dispatch_counts = Some(sequence.dispatch_counts());
            self.completed_tail_elements = Some(sequence.current_elements());
            self.completed_round_device_buffer_allocations =
                Some(sequence.round_device_buffer_allocations());
        }
        let remaining_sequence_storage_bytes = sequence
            .storage_stats()
            .map_err(metal_output_error)?
            .owned_bytes;
        let compact_row_bytes =
            instruction_input_row_bytes(self.resident_rows).map_err(metal_output_error)?;
        let residual_row_bytes = spartan_outer_uniskip_row_bytes(self.resident_rows)
            .map_err(metal_output_error)?
            .checked_sub(compact_row_bytes)
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "outer-remainder residual row accounting underflowed",
            })?;
        let compact_release_bytes = if self.compact_retained {
            0
        } else {
            compact_row_bytes
        };
        let released_owned_bytes = remaining_sequence_storage_bytes
            .checked_add(residual_row_bytes)
            .and_then(|bytes| bytes.checked_add(compact_release_bytes))
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "outer-remainder release byte count overflowed",
            })?;
        let release = tracing::info_span!(
            "MetalOuterRemainder::row_release",
            compact_rows_storage_id = self.compact_rows_storage_id,
            residual_rows_storage_id = self.residual_rows_storage_id,
            device_registry_id = self.device_registry_id,
            resident_rows = self.resident_rows,
            row_upload_bytes = 0u64,
            device_allocations = 0u64,
            residual_row_bytes,
            remaining_sequence_storage_bytes,
            compact_release_bytes,
            released_owned_bytes,
            release_completed = tracing::field::Empty,
            residual_released = tracing::field::Empty,
            compact_retained = self.compact_retained,
        );
        {
            let _release = release.enter();
            let compact = sequence
                .into_instruction_input_rows()
                .map_err(metal_output_error)?;
            if compact.allocation_identity() != self.compact_rows_storage_id {
                return Err(SumcheckKernelError::InvariantViolation {
                    reason: "Metal outer remainder changed the shared compact allocation",
                });
            }
            drop(compact);
            let _ = release.record("release_completed", true);
            let _ = release.record("residual_released", true);
        }
        self.host.output_claims(inputs, &claimed)
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<AkitaField, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        self.host
            .validate_derived_tables(relation, input_points, output_points, challenges)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::{NoChallenges, OutputClaims as _};
    use jolt_poly::lagrange::centered_lagrange_kernel;
    use jolt_r1cs::constraints::jolt::{spartan_outer_constraints, spartan_outer_row_weights};
    use jolt_verifier::stages::relations::ConcreteSumcheck as _;
    use jolt_verifier::stages::stage1::outer_remainder::{
        outer_remainder_input_values_from_uniskip_output, OuterRemainderInputClaims,
    };
    use jolt_witness::testing::with_sample_backend_at_log_t;
    use jolt_witness::witnesses::{SpartanOuterRow, ToField};
    use jolt_witness::BundleSource;

    use super::{
        outer_remainder_storage_fallback_reason, prepare_cpu_instruction_input_now,
        resident_row_admission_candidates, resident_row_consumers, resident_row_working_set,
        use_metal_remainder, use_metal_stage1, validate_resident_row_buffer,
        MetalOuterRemainderHost, OuterRemainder, ResidentRowPlan, SpartanOuterDimensions,
        OUTER_DOMAIN, OUTER_VARIABLES,
    };
    use crate::metal::solinas::{MetalError, OuterRemainderSequenceConfig};
    #[cfg(feature = "test-utils")]
    use crate::metal::solinas::{OuterBindingPlan, OuterKernelArtifact};
    use crate::metal::{MetalBackend, MetalConfig, SpartanOuterRemainderMetalConfig};
    use crate::optimized::harness::run_lockstep;
    use crate::optimized::spartan_outer::{
        prepare_metal_spartan_outer_witness_rows, OptimizedOuterRemainder, OptimizedOuterUniskip,
    };
    use crate::uniskip::UniskipKernel;
    use crate::{PrepareKernel, ProofSession, ProverInputs};
    use jolt_field::AkitaField;
    use jolt_poly::EqPolynomial;

    #[derive(Clone, Copy)]
    enum OuterSource {
        Embedded,
        #[cfg(feature = "test-utils")]
        RuntimeArtifact,
    }

    fn variable_field_value(row: &SpartanOuterRow, index: usize) -> AkitaField {
        match index {
            0 => row.left_instruction_input.to_field(),
            1 => row.right_instruction_input.to_field(),
            2 => row.product.to_field(),
            3 => row.should_branch.to_field(),
            4 => row.pc.to_field(),
            5 => row.unexpanded_pc.to_field(),
            6 => row.imm.to_field(),
            7 => row.ram_address.to_field(),
            8 => row.rs1_value.to_field(),
            9 => row.rs2_value.to_field(),
            10 => row.rd_write_value.to_field(),
            11 => row.ram_read_value.to_field(),
            12 => row.ram_write_value.to_field(),
            13 => row.left_lookup_operand.to_field(),
            14 => row.right_lookup_operand.to_field(),
            15 => row.next_unexpanded_pc.to_field(),
            16 => row.next_pc.to_field(),
            17 => row.next_is_virtual.to_field(),
            18 => row.next_is_first_in_sequence.to_field(),
            19 => row.lookup_output.to_field(),
            20 => row.should_jump.to_field(),
            21 => row.add_operands.to_field(),
            22 => row.subtract_operands.to_field(),
            23 => row.multiply_operands.to_field(),
            24 => row.load.to_field(),
            25 => row.store.to_field(),
            26 => row.jump.to_field(),
            27 => row.write_lookup_output_to_rd.to_field(),
            28 => row.virtual_instruction.to_field(),
            29 => row.assert_flag.to_field(),
            30 => row.do_not_update_unexpanded_pc.to_field(),
            31 => row.advice.to_field(),
            32 => row.is_compressed.to_field(),
            33 => row.is_first_in_sequence.to_field(),
            34 => row.is_last_in_sequence.to_field(),
            _ => unreachable!("35 canonical R1CS inputs"),
        }
    }

    fn true_input_claim(
        rows: &[SpartanOuterRow],
        tau: &[AkitaField],
        r0: AkitaField,
        log_t: usize,
    ) -> AkitaField {
        let tau_low = &tau[..=log_t];
        let tau_high = tau[log_t + 1];
        let eq = EqPolynomial::new(tau_low.to_vec()).evaluations();
        let kernel = centered_lagrange_kernel(OUTER_DOMAIN, tau_high, r0).unwrap();
        let matrices = spartan_outer_constraints::<AkitaField>();
        let columns = (1..=OUTER_VARIABLES).collect::<Vec<_>>();
        let mut total = AkitaField::zero();
        for (stream_index, stream) in [AkitaField::zero(), AkitaField::one()]
            .into_iter()
            .enumerate()
        {
            let weights = spartan_outer_row_weights(r0, stream).unwrap();
            let weighted = matrices.weighted_columns(&weights, &columns).unwrap();
            let constants = matrices
                .public_column_contributions(&weights, 0, AkitaField::one())
                .unwrap();
            for (cycle, row) in rows.iter().enumerate() {
                let mut az = constants.a;
                let mut bz = constants.b;
                for (index, (&a, &b)) in weighted.a.iter().zip(&weighted.b).enumerate() {
                    let value = variable_field_value(row, index);
                    az += a * value;
                    bz += b * value;
                }
                total += eq[(cycle << 1) | stream_index] * az * bz;
            }
        }
        kernel * total
    }

    #[test]
    fn resident_rows_follow_actual_consumer_thresholds() {
        let mut config = MetalConfig::default();
        config.spartan_outer_uniskip.trace_cutoff_elements = 1 << 10;
        config.instruction_input.trace_cutoff_elements = 1 << 8;
        config.instruction_input.cutoff_elements = 1 << 9;

        assert_eq!(resident_row_consumers(1 << 7, &config), (false, false));
        assert_eq!(resident_row_consumers(1 << 8, &config), (false, false));
        assert_eq!(resident_row_consumers(1 << 9, &config), (false, false));
        assert_eq!(resident_row_consumers(1 << 10, &config), (true, true));

        config.spartan_outer_uniskip.trace_cutoff_elements = 1 << 12;
        assert_eq!(resident_row_consumers(1 << 10, &config), (false, true));
        config.spartan_outer_remainder.trace_cutoff_elements = 1 << 10;
        assert_eq!(resident_row_consumers(1 << 10, &config), (true, true));
    }

    #[test]
    fn aggregate_instruction_input_working_set_matches_production_geometry() {
        assert_eq!(
            resident_row_working_set(1 << 26, true, true, true, true, Default::default(),).unwrap(),
            21_482_505_104
        );
        assert_eq!(
            resident_row_working_set(1 << 28, true, true, true, true, Default::default(),).unwrap(),
            85_909_832_592
        );
        assert_eq!(
            resident_row_working_set(1 << 26, false, true, false, false, Default::default(),)
                .unwrap(),
            9_664_659_456
        );
        assert_eq!(
            resident_row_working_set(1 << 28, false, true, false, false, Default::default(),)
                .unwrap(),
            38_656_671_744
        );
        assert_eq!(
            resident_row_working_set(1 << 28, true, false, true, true, Default::default(),)
                .unwrap(),
            60_138_062_736
        );
    }

    #[test]
    fn resident_row_buffer_admission_is_exact() {
        let bytes = 42_949_672_960;
        assert!(validate_resident_row_buffer(bytes, bytes).is_ok());
        assert!(matches!(
            validate_resident_row_buffer(bytes, bytes - 1),
            Err(MetalError::BufferTooLong {
                requested: 42_949_672_960,
                maximum: 42_949_672_959,
            })
        ));
    }

    #[test]
    fn storage_fallback_is_limited_to_recoverable_preprotocol_errors() {
        assert_eq!(
            outer_remainder_storage_fallback_reason(&MetalError::BufferTooLong {
                requested: 2,
                maximum: 1,
            }),
            Some("capacity")
        );
        assert_eq!(
            outer_remainder_storage_fallback_reason(&MetalError::GpuTimestampLookup {
                name: "test",
                message: "missing".to_owned(),
            }),
            Some("gpu_timestamp_lookup")
        );
        assert_eq!(
            outer_remainder_storage_fallback_reason(&MetalError::InvalidOuterRemainderConfig(
                "invalid test configuration",
            )),
            None
        );
    }

    #[test]
    fn stage1_requires_prepared_resident_rows() {
        let mut config = MetalConfig::default();
        config.spartan_outer_uniskip.trace_cutoff_elements = 1 << 10;
        assert!(!use_metal_stage1(1 << 10, &config, false));
        assert!(use_metal_stage1(1 << 10, &config, true));
    }

    #[test]
    fn admission_retries_instruction_input_before_stage1() {
        assert_eq!(
            resident_row_admission_candidates(true, true),
            vec![
                ResidentRowPlan {
                    stage1: true,
                    instruction_input: true,
                },
                ResidentRowPlan {
                    stage1: false,
                    instruction_input: true,
                },
                ResidentRowPlan {
                    stage1: true,
                    instruction_input: false,
                },
            ]
        );
        assert_eq!(resident_row_admission_candidates(false, false), vec![]);
    }

    #[test]
    fn cpu_rows_wait_until_stage1_releases_resident_buffers() {
        assert!(!prepare_cpu_instruction_input_now(false, true, false));
        assert!(prepare_cpu_instruction_input_now(false, false, false));
        assert!(!prepare_cpu_instruction_input_now(true, false, false));
        assert!(!prepare_cpu_instruction_input_now(false, false, true));
    }

    #[test]
    fn remainder_requires_both_cutoff_and_resident_rows() {
        let mut config = MetalConfig::default();
        config.spartan_outer_remainder.trace_cutoff_elements = 1 << 10;
        assert!(!use_metal_remainder(2, &config, true));
        assert!(!use_metal_remainder(1 << 9, &config, true));
        assert!(!use_metal_remainder(1 << 10, &config, false));
        assert!(use_metal_remainder(1 << 10, &config, true));
    }

    #[test]
    fn opening_factorization_matches_dense_low_to_high_point() {
        let log_t = 8;
        let tau = (0..log_t + 2)
            .map(|index| AkitaField::from_u64(17 + index as u64))
            .collect::<Vec<_>>();
        let mut host =
            MetalOuterRemainderHost::new(log_t, &tau, AkitaField::from_u64(101)).unwrap();
        for index in 0..=log_t {
            host.bind(AkitaField::from_u64(211 + index as u64));
        }
        let point = host.challenges[1..]
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();
        let dense = EqPolynomial::evals(&point, None);
        let (e_in, e_out) = host.opening_weights();
        let factored = e_out
            .iter()
            .flat_map(|outer| e_in.iter().map(move |inner| *outer * *inner))
            .collect::<Vec<_>>();
        assert_eq!(factored, dense);
    }

    fn adapter_parity_case(
        log_t: usize,
        cpu_tail_elements: usize,
        source: OuterSource,
        binding_plan: OuterBindingPlan,
    ) {
        let cycles = 1usize << log_t;
        with_sample_backend_at_log_t(log_t, 4, |witness| {
            let tau = (0..log_t + 2)
                .map(|index| AkitaField::from_u64(71 + 19 * index as u64))
                .collect::<Vec<_>>();
            let r0 = AkitaField::from_u64(12_289);
            let relation =
                OuterRemainder::new(SpartanOuterDimensions::rv64(log_t), tau.clone(), r0);
            let rows = witness.bundles().unwrap();
            let input_claim = true_input_claim(&rows, &tau, r0, log_t);
            assert_ne!(input_claim, AkitaField::zero());
            let claims = outer_remainder_input_values_from_uniskip_output(input_claim);
            let points = OuterRemainderInputClaims::<Vec<AkitaField>>::default();
            let no_challenges = NoChallenges::<AkitaField>::default();

            let mut cpu_session = ProofSession::default();
            <OptimizedOuterUniskip as UniskipKernel<
                AkitaField,
                OuterRemainder<AkitaField>,
            >>::prepare(
                &OptimizedOuterUniskip,
                &mut cpu_session,
                log_t,
                &tau,
                witness,
            )
            .unwrap();

            let mut metal_session = ProofSession::default();
            <OptimizedOuterUniskip as UniskipKernel<
                AkitaField,
                OuterRemainder<AkitaField>,
            >>::prepare(
                &OptimizedOuterUniskip,
                &mut metal_session,
                log_t,
                &tau,
                witness,
            )
            .unwrap();
            let config = MetalConfig {
                spartan_outer_remainder: SpartanOuterRemainderMetalConfig {
                    trace_cutoff_elements: 4,
                    dispatch: OuterRemainderSequenceConfig {
                        binding_plan,
                        max_threadgroups: 2,
                        cpu_tail_elements,
                        ..Default::default()
                    },
                },
                ..Default::default()
            };
            let metal = match source {
                OuterSource::Embedded => MetalBackend::new(config),
                #[cfg(feature = "test-utils")]
                OuterSource::RuntimeArtifact => MetalBackend::new_with_outer_artifact(
                    config,
                    &OuterKernelArtifact::embedded(binding_plan).unwrap(),
                ),
            }
            .unwrap();
            let rows =
                prepare_metal_spartan_outer_witness_rows(&metal.context, witness, cycles).unwrap();
            let storage = metal
                .context
                .prepare_outer_remainder_sequence_storage(
                    cycles,
                    metal.config.spartan_outer_remainder.dispatch,
                )
                .unwrap();
            metal_session.park(rows);
            metal_session.park(storage);

            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &no_challenges,
            };
            let mut cpu = <OptimizedOuterRemainder as PrepareKernel<
                AkitaField,
                OuterRemainder<AkitaField>,
            >>::prepare(
                &OptimizedOuterRemainder,
                &mut cpu_session,
                witness,
                inputs(),
            )
            .unwrap();
            let mut device = <MetalBackend as PrepareKernel<
                AkitaField,
                OuterRemainder<AkitaField>,
            >>::prepare(&metal, &mut metal_session, witness, inputs())
            .unwrap();
            assert_eq!(metal.outer_remainder_sequences(), 1);

            let round_challenges = (0..=log_t)
                .map(|index| AkitaField::from_u64(65_537 + 31 * index as u64))
                .collect::<Vec<_>>();
            run_lockstep(
                cpu.as_mut(),
                device.as_mut(),
                input_claim,
                &round_challenges,
            );

            let cpu_outputs = cpu.output_claims(&claims).unwrap();
            let device_outputs = device.output_claims(&claims).unwrap();
            assert_eq!(device_outputs, cpu_outputs);
            assert_eq!(
                device_outputs.canonical_order(),
                cpu_outputs.canonical_order()
            );
            assert_eq!(
                device_outputs.opening_values(),
                cpu_outputs.opening_values()
            );
            assert_eq!(device_outputs.opening_values().len(), 35);

            let output_points = relation
                .derive_opening_points(&round_challenges, &points)
                .unwrap();
            cpu.validate_derived_tables(&relation, &points, &output_points, &no_challenges)
                .unwrap();
            device
                .validate_derived_tables(&relation, &points, &output_points, &no_challenges)
                .unwrap();
        });
    }

    #[test]
    fn outer_remainder_adapter_matches_optimized_cpu_all_rounds_and_openings() {
        adapter_parity_case(5, 8, OuterSource::Embedded, OuterBindingPlan::BOnlyV1);
    }

    #[cfg(feature = "test-utils")]
    #[test]
    fn runtime_outer_artifact_matches_optimized_cpu_all_rounds_and_openings() {
        adapter_parity_case(
            5,
            8,
            OuterSource::RuntimeArtifact,
            OuterBindingPlan::BOnlyV1,
        );
    }

    #[cfg(all(feature = "test-utils", not(feature = "metal-runtime-artifact-only")))]
    #[test]
    fn padded_56_runtime_artifact_matches_cpu_across_a_full_tile_and_tail() {
        adapter_parity_case(
            12,
            8,
            OuterSource::RuntimeArtifact,
            OuterBindingPlan::BOnlyPadded56V1,
        );
    }

    #[test]
    fn outer_remainder_handoff_waits_for_the_stream_bind() {
        adapter_parity_case(3, 16, OuterSource::Embedded, OuterBindingPlan::BOnlyV1);
    }

    #[test]
    fn invalid_remainder_geometry_is_rejected_before_device_setup() {
        let mut config = MetalConfig::default();
        config.spartan_outer_remainder.trace_cutoff_elements = 2;
        assert!(matches!(
            MetalBackend::new(config),
            Err(MetalError::InvalidHybridCutoff(2))
        ));

        let mut config = MetalConfig::default();
        config.spartan_outer_remainder.dispatch.max_threadgroups = 0;
        assert!(matches!(
            MetalBackend::new(config),
            Err(MetalError::InvalidOuterRemainderConfig(
                "max_threadgroups must be nonzero"
            ))
        ));
    }
}
