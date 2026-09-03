use std::{
    collections::BTreeMap,
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
use super::instruction_input::{InstructionInputDenseStorageMode, PreparedInstructionInput};
use super::registers_claim_reduction::{
    MetalRegistersClaimAsyncStage1Carry, MetalRegistersClaimOuterSource,
    MetalRegistersClaimPendingStage1Carry, MetalRegistersClaimStage1Carry,
};
use super::registers_val_evaluation::RegistersValEvaluationSource;
use super::solinas::bytecode_read_raf_address::{
    bytecode_address_stage1_topology_max_bytes, bytecode_address_stage1_topology_max_plane_bytes,
};
use super::solinas::spartan_shift::{SpartanShiftFlagWord, SPARTAN_SHIFT_FLAG_ROWS_PER_WORD};
use super::solinas::{
    instruction_input_row_bytes, instruction_input_sequence_auxiliary_storage_bytes,
    instruction_input_sequence_storage_bytes, instruction_read_raf_stage1_claim_bytes,
    instruction_read_raf_stage1_device_bytes, instruction_read_raf_stage1_row_bytes,
    outer_remainder_sequence_max_buffer_bytes_with_config,
    outer_remainder_sequence_storage_bytes_with_config, spartan_outer_uniskip_invocation_bytes,
    spartan_outer_uniskip_row_bytes, InstructionInputRows, MetalError, OuterRemainderPhase,
    OuterRemainderSequence, OuterRemainderSequenceConfig, OuterRemainderSequenceStorage,
    PendingRegistersReadWriteStage1Pipelines, PendingSpartanStage1SourcePrimer,
    RegistersReadWriteStage1Source, RegistersValInstructionSourceLease,
    RegistersValInstructionSourceRequest, SolinasMetal, SpartanOuterUniskipConfig,
    SpartanOuterUniskipRows,
};
use super::spartan_dense::SpartanDenseResidentOwner;
use super::spartan_product::MetalProductUniskipEndpointCarrier;
use crate::optimized::instruction_input::PreparedInstructionInputRows;
use crate::optimized::spartan_outer::{
    prepare_metal_instruction_input_witness_rows,
    prepare_metal_spartan_outer_shift_stage1_owner_witness_rows,
    prepare_metal_spartan_outer_shift_witness_rows,
    prepare_metal_spartan_outer_stage1_owner_witness_rows, prepare_metal_spartan_outer_uniskip,
    prepare_metal_spartan_outer_witness_rows, take_metal_spartan_outer_tau,
    InstructionReadRafStage1Ready, MetalSpartanDenseRowsError, OptimizedOuterRemainder,
    OptimizedOuterUniskip,
};
use crate::uniskip::UniskipKernel;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[cfg(feature = "test-utils")]
mod evaluation;
#[cfg(feature = "test-utils")]
pub use evaluation::{
    OuterRemainderCpuEvalSample, OuterRemainderCpuMetalEvalFixture, OuterRemainderEvalError,
    OuterRemainderEvalResult, OuterRemainderMetalEvalSample, OuterRemainderPipelineSnapshot,
    OuterRemainderThreadSnapshot,
};

const OUTER_DOMAIN: usize = OUTER_UNISKIP_DOMAIN_SIZE;
const OUTER_VARIABLES: usize = 35;
const STAGE1_SOURCE_PRIMER_CUTOFF_ELEMENTS: usize = 1 << 28;

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

#[expect(
    clippy::too_many_arguments,
    reason = "capacity tests exercise the resident consumers independently"
)]
fn resident_row_working_set(
    cycles: usize,
    stage1: bool,
    instruction_input: bool,
    instruction_read_raf_owner: bool,
    borrow_outer_residual: bool,
    spartan_shift: bool,
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
    let instruction_input_bytes = if instruction_input && borrow_outer_residual {
        if !stage1 {
            return Err(MetalError::InvalidInstructionInputState(
                "Outer residual borrowing requires resident Stage-1 rows",
            ));
        }
        instruction_input_sequence_auxiliary_storage_bytes(cycles)?
    } else if instruction_input {
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
    let instruction_read_raf_bytes = if instruction_read_raf_owner {
        instruction_read_raf_stage1_device_bytes(cycles)?
    } else {
        0
    };
    let shift_bytes = if spartan_shift {
        let rows = u64::try_from(cycles).map_err(|_| MetalError::InputTooLong(cycles))?;
        let flag_words = u64::try_from(cycles.div_ceil(SPARTAN_SHIFT_FLAG_ROWS_PER_WORD))
            .map_err(|_| MetalError::InputTooLong(cycles))?;
        rows.checked_mul((2 * size_of::<u64>()) as u64)
            .and_then(|bytes| {
                flag_words
                    .checked_mul(size_of::<SpartanShiftFlagWord>() as u64)
                    .and_then(|flags| bytes.checked_add(flags))
            })
            .ok_or(MetalError::InputTooLong(cycles))?
    } else {
        0
    };
    row_bytes
        .checked_add(instruction_input_bytes)
        .and_then(|bytes| bytes.checked_add(uniskip_bytes))
        .and_then(|bytes| bytes.checked_add(remainder_bytes))
        .and_then(|bytes| bytes.checked_add(instruction_read_raf_bytes))
        .and_then(|bytes| bytes.checked_add(shift_bytes))
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
        let defer_state_b = self.config.spartan_product_remainder.reuse_outer_state_a
            && cycles >= self.config.spartan_product_remainder.trace_cutoff_elements
            && session
                .state::<SpartanOuterUniskipRows>()
                .is_some_and(|rows| {
                    rows.len() == cycles
                        && rows.device_registry_id() == self.context.device_registry_id()
                });
        let span = tracing::info_span!(
            "MetalOuterRemainder::storage_prepare",
            cycles,
            planned_device_bytes,
            maximum_buffer_bytes,
            defer_state_b,
            initialization_mode = config.dispatch.storage_initialization.as_str(),
            admitted = tracing::field::Empty,
            fallback_reason = tracing::field::Empty,
        );
        let _span = span.enter();
        let storage = if defer_state_b {
            self.context
                .prepare_outer_remainder_sequence_storage_deferring_state_b(cycles, config.dispatch)
        } else {
            self.context
                .prepare_outer_remainder_sequence_storage(cycles, config.dispatch)
        };
        match storage {
            Ok(storage) => {
                let _ = span.record("admitted", true);
                let _ = span.record("fallback_reason", "none");
                tracing::info!(
                    target: "jolt::metal",
                    bytes = storage.owned_bytes(),
                    "admitted outer-remainder Metal storage"
                );
                session.park(storage);
                Ok(())
            }
            Err(error) => match outer_remainder_storage_fallback_reason(&error) {
                Some(reason) => {
                    let _ = span.record("admitted", false);
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

pub(super) fn publish_instruction_read_raf_stage1(
    context: &SolinasMetal,
    session: &mut ProofSession,
    ready: InstructionReadRafStage1Ready,
) -> Result<(), KernelError<AkitaField>> {
    if ready.ram_access.is_some() && ready.ram_read_write_records.is_some() {
        return Err(KernelError::InvariantViolation {
            reason: "Stage-1 RAM publication selected two source representations",
        });
    }
    if let Some(ram_access) = &ready.ram_access {
        ram_access.validate_publish(session)?;
    }
    if let Some(records) = &ready.ram_read_write_records {
        records.validate_publish(session)?;
    }
    if session
        .state::<super::solinas::InstructionReadRafStage1Owner>()
        .is_some()
        || (ready.bytecode_topology.is_some()
            && session
                .state::<super::solinas::bytecode_read_raf_address::BytecodeAddressStage1TopologyOwner>()
                .is_some())
        || (ready.registers_val.is_some()
            && session
                .state::<RegistersValInstructionSourceRequest>()
                .is_some())
        || (ready.registers_read_write.is_some()
            && session.state::<RegistersReadWriteStage1Source>().is_some())
        || (ready.registers_read_write.is_some()
            && session
                .state::<PendingRegistersReadWriteStage1Pipelines>()
                .is_some())
    {
        return Err(KernelError::InvariantViolation {
            reason: "InstructionReadRAF Stage-1 publication would replace resident state",
        });
    }
    session.park(ready.owner);
    if let Some(topology) = ready.bytecode_topology {
        session.park(topology);
    }
    if let Some(source) = ready.registers_read_write {
        let pending = context
            .submit_registers_read_write_stage1_pipeline_warmup(&source)
            .map_err(metal_prepare_error)?;
        session.park(pending);
        session.park(source);
    }
    if let Some(request) = ready.registers_val {
        session.park(request);
    }
    if let Some(ram_access) = ready.ram_access {
        ram_access.publish(session)?;
    }
    if let Some(records) = ready.ram_read_write_records {
        records.publish(session)?;
    }
    Ok(())
}

impl UniskipKernel<AkitaField, OuterRemainder<AkitaField>> for MetalBackend {
    fn prepare_witness(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<(), KernelError<AkitaField>> {
        let cycles = 1usize << log_t;
        let (mut stage1_eligible, instruction_input_eligible) =
            resident_row_consumers(cycles, &self.config);
        let instruction_read_raf_owner_requested = cycles
            >= self.config.instruction_read_raf.address_cutoff_elements
            && witness
                .owned_rows()
                .is_some_and(|rows| cycles <= rows.cycles());
        let register_val_owner_requested = self.config.registers_val_evaluation.source
            == RegistersValEvaluationSource::Stage1Resident
            && cycles >= self.config.registers_val_evaluation.trace_cutoff_elements;
        let prepare_registers_val = register_val_owner_requested && (26..=28).contains(&log_t);
        let prepare_registers_read_write = register_val_owner_requested && log_t == 28;
        if prepare_registers_val
            && witness
                .owned_rows()
                .is_none_or(|rows| cycles > rows.cycles())
        {
            return Err(KernelError::InvariantViolation {
                reason: "RegistersVal Stage-1 source requires random-access witness rows",
            });
        }
        if prepare_registers_val
            && (session
                .state::<RegistersValInstructionSourceRequest>()
                .is_some()
                || session
                    .state::<RegistersValInstructionSourceLease>()
                    .is_some())
        {
            return Err(KernelError::InvariantViolation {
                reason: "RegistersVal Stage-1 owner was already parked",
            });
        }
        if prepare_registers_read_write
            && session.state::<RegistersReadWriteStage1Source>().is_some()
        {
            return Err(KernelError::InvariantViolation {
                reason: "registers read-write Stage-1 source was already parked",
            });
        }
        let stage1_projection_owner_requested = instruction_read_raf_owner_requested
            || prepare_registers_val
            || prepare_registers_read_write;
        let prepare_ram_access =
            stage1_projection_owner_requested && self.ram_raf_witness_requested(log_t, witness)?;
        let prepare_ram_read_write_records = stage1_projection_owner_requested
            && !prepare_ram_access
            && cycles >= super::ram_read_write::RAM_READ_WRITE_STAGE1_SOURCE_CUTOFF_ELEMENTS;
        let prepare_bytecode_carrier = self.config.bytecode_read_raf_address.implementation
            == super::bytecode_read_raf::BytecodeReadRafAddressImplementation::AddressMajor
            && cycles >= self.config.bytecode_read_raf_address.trace_cutoff_elements
            && super::bytecode_read_raf::bytecode_address_stage1_topology_supported(witness);
        if prepare_bytecode_carrier && !instruction_read_raf_owner_requested {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode address-major requires the random-access Stage-1 owner",
            });
        }
        let bytecode_carrier_physical_rows = if prepare_bytecode_carrier {
            Some(
                witness
                    .owned_rows()
                    .map(|rows| rows.physical_rows().min(cycles))
                    .ok_or(KernelError::InvariantViolation {
                        reason: "bytecode address-major requires random-access witness rows",
                    })?,
            )
        } else {
            None
        };
        stage1_eligible |= stage1_projection_owner_requested;
        let mut admitted_plan = None;
        let mut last_admission_error = None;
        if stage1_eligible || instruction_input_eligible {
            let instruction_input_bytes =
                instruction_input_row_bytes(cycles).map_err(metal_prepare_error)?;
            let device = self.context.device_info();
            for candidate in
                resident_row_admission_candidates(stage1_eligible, instruction_input_eligible)
            {
                if (prepare_bytecode_carrier
                    || prepare_registers_val
                    || prepare_registers_read_write)
                    && !candidate.stage1
                {
                    continue;
                }
                let borrow_outer_residual = self.config.instruction_input.dense_storage_mode
                    == InstructionInputDenseStorageMode::OuterResidual;
                if borrow_outer_residual && candidate.instruction_input && !candidate.stage1 {
                    continue;
                }
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
                            if stage1_projection_owner_requested && candidate.stage1 {
                                validate_resident_row_buffer(
                                    instruction_read_raf_stage1_row_bytes(cycles)?,
                                    device.max_buffer_length,
                                )?;
                                validate_resident_row_buffer(
                                    instruction_read_raf_stage1_claim_bytes(cycles)?,
                                    device.max_buffer_length,
                                )?;
                            }
                            if prepare_bytecode_carrier && candidate.stage1 {
                                let physical_rows = bytecode_carrier_physical_rows.ok_or(
                                    MetalError::InvalidInstructionReadRafGrouped(
                                        "bytecode carrier physical rows are unavailable".to_owned(),
                                    ),
                                )?;
                                for bytes in
                                    bytecode_address_stage1_topology_max_plane_bytes(physical_rows)?
                                {
                                    validate_resident_row_buffer(bytes, device.max_buffer_length)?;
                                }
                            }
                            Ok(())
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
                            let bytes = resident_row_working_set(
                                cycles,
                                candidate.stage1,
                                candidate.instruction_input,
                                stage1_projection_owner_requested && candidate.stage1,
                                borrow_outer_residual && candidate.instruction_input,
                                candidate.stage1
                                    && cycles >= self.config.spartan_shift.trace_cutoff_elements,
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
                            )?;
                            if prepare_bytecode_carrier && candidate.stage1 {
                                let physical_rows = bytecode_carrier_physical_rows.ok_or(
                                    MetalError::InvalidInstructionReadRafGrouped(
                                        "bytecode carrier physical rows are unavailable".to_owned(),
                                    ),
                                )?;
                                bytes
                                    .checked_add(bytecode_address_stage1_topology_max_bytes(
                                        physical_rows,
                                    )?)
                                    .ok_or(MetalError::InputTooLong(cycles))
                            } else {
                                Ok(bytes)
                            }
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
                        last_admission_error = Some(error);
                    }
                    Err(error) => return Err(metal_prepare_error(error)),
                }
            }
        }
        if (prepare_bytecode_carrier || prepare_registers_val || prepare_registers_read_write)
            && admitted_plan.is_none()
        {
            return Err(last_admission_error.map_or(
                KernelError::InvariantViolation {
                    reason: "required Stage-1 owner plan was not admitted",
                },
                metal_prepare_error,
            ));
        }
        if let Some(plan) = admitted_plan {
            if plan.stage1 {
                let (mut rows, instruction_read_raf_ready) = if cycles
                    >= self.config.spartan_shift.trace_cutoff_elements
                {
                    let span = tracing::info_span!(
                        "MetalSpartanDense::witness_prepare",
                        cycles,
                        source = "stage1_single_projection",
                        owner_generation = tracing::field::Empty,
                        shift_row_extractions = tracing::field::Empty,
                        shift_late_copy_dispatches = tracing::field::Empty,
                        native_register_contract_bytes = tracing::field::Empty,
                        shift_resident_bytes = tracing::field::Empty,
                        admitted = tracing::field::Empty,
                        fallback_reason = tracing::field::Empty,
                    );
                    let _entered = span.enter();
                    let prepared = if stage1_projection_owner_requested {
                        prepare_metal_spartan_outer_shift_stage1_owner_witness_rows(
                            &self.context,
                            witness,
                            cycles,
                            prepare_bytecode_carrier,
                            prepare_registers_read_write,
                            prepare_registers_val,
                            prepare_ram_access,
                            prepare_ram_read_write_records,
                        )
                        .map(|(rows, shift_rows, prepared)| (rows, shift_rows, Some(prepared)))
                    } else {
                        prepare_metal_spartan_outer_shift_witness_rows(
                            &self.context,
                            witness,
                            cycles,
                        )
                        .map(|(rows, shift_rows)| (rows, shift_rows, None))
                    };
                    match prepared {
                        Ok((rows, shift_rows, instruction_read_raf)) => {
                            let shift_resident_bytes = shift_rows.resident_bytes();
                            let owner =
                                SpartanDenseResidentOwner::from_co_produced_shift(shift_rows)
                                    .map_err(metal_prepare_error)?;
                            let _ = span.record("owner_generation", owner.generation());
                            let _ =
                                span.record("shift_row_extractions", owner.shift_row_extractions());
                            let _ = span.record(
                                "shift_late_copy_dispatches",
                                owner.shift_late_copy_dispatches(),
                            );
                            let _ = span.record("shift_resident_bytes", shift_resident_bytes);
                            let _ = span.record("admitted", true);
                            let _ = span.record("fallback_reason", "none");
                            session.park(owner);
                            (rows, instruction_read_raf)
                        }
                        Err(error)
                            if error.is_capacity_error()
                                && !prepare_bytecode_carrier
                                && !prepare_registers_val =>
                        {
                            let _ = span.record("admitted", false);
                            let _ = span.record("fallback_reason", "shift_capacity");
                            tracing::warn!(
                                target: "jolt::metal",
                                error = ?error,
                                "Spartan dense Shift co-production was not admitted; retaining Stage-1 rows only"
                            );
                            if stage1_projection_owner_requested {
                                prepare_metal_spartan_outer_stage1_owner_witness_rows(
                                    &self.context,
                                    witness,
                                    cycles,
                                    prepare_bytecode_carrier,
                                    prepare_registers_read_write,
                                    prepare_registers_val,
                                    prepare_ram_access,
                                    prepare_ram_read_write_records,
                                )
                                .map(|(rows, prepared)| (rows, Some(prepared)))
                                .map_err(MetalSpartanDenseRowsError::into_kernel_error)?
                            } else {
                                (
                                    prepare_metal_spartan_outer_witness_rows(
                                        &self.context,
                                        witness,
                                        cycles,
                                    )?,
                                    None,
                                )
                            }
                        }
                        Err(error) => return Err(error.into_kernel_error()),
                    }
                } else if stage1_projection_owner_requested {
                    prepare_metal_spartan_outer_stage1_owner_witness_rows(
                        &self.context,
                        witness,
                        cycles,
                        prepare_bytecode_carrier,
                        prepare_registers_read_write,
                        prepare_registers_val,
                        prepare_ram_access,
                        prepare_ram_read_write_records,
                    )
                    .map(|(rows, prepared)| (rows, Some(prepared)))
                    .map_err(MetalSpartanDenseRowsError::into_kernel_error)?
                } else {
                    (
                        prepare_metal_spartan_outer_witness_rows(&self.context, witness, cycles)?,
                        None,
                    )
                };
                if let Some(ready) = instruction_read_raf_ready {
                    publish_instruction_read_raf_stage1(&self.context, session, ready)?;
                }
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
        self.prepare_ram_raf_witness(session, log_t, witness)?;
        if self.config.spartan_product_remainder.reuse_outer_state_a {
            self.prepare_instruction_input_storage(session, cycles)?;
            self.prepare_outer_remainder_storage(session, cycles)?;
            self.prepare_product_remainder_witness(session, log_t, witness)?;
        } else {
            self.prepare_product_remainder_witness(session, log_t, witness)?;
            self.prepare_instruction_input_storage(session, cycles)?;
            self.prepare_outer_remainder_storage(session, cycles)?;
        }
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
        if cycles >= STAGE1_SOURCE_PRIMER_CUTOFF_ELEMENTS {
            let pending = session
                .state::<SpartanOuterUniskipRows>()
                .zip(
                    session
                        .state::<SpartanDenseResidentOwner>()
                        .and_then(SpartanDenseResidentOwner::shift_rows),
                )
                .map(|(outer, shift)| {
                    self.context
                        .submit_spartan_stage1_source_primer(outer, shift)
                })
                .transpose()
                .map_err(metal_prepare_error)?;
            if let Some(pending) = pending {
                session.park(pending);
            }
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
            drop(session.take::<PendingSpartanStage1SourcePrimer>());
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
        if let Some(primer) = session.take::<PendingSpartanStage1SourcePrimer>() {
            let _span = tracing::info_span!("MetalSpartanStage1::source_primer_join").entered();
            primer.join().map_err(metal_prepare_error)?;
        }
        self.start_ram_read_write_sequence_prefetch(session, log_t)?;
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

fn metal_prepare_error(error: impl ToString) -> KernelError<AkitaField> {
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

fn metal_output_error(error: impl ToString) -> SumcheckKernelError<AkitaField> {
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
        if self
            .config
            .spartan_outer_remainder
            .dispatch
            .registers_claim_carrier
            && (session.state::<MetalRegistersClaimStage1Carry>().is_some()
                || session
                    .state::<MetalRegistersClaimAsyncStage1Carry>()
                    .is_some())
        {
            return Err(KernelError::InvariantViolation {
                reason: "Metal outer remainder found a stale registers-claim carry",
            });
        }
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
        let storage = session.state::<OuterRemainderSequenceStorage>().ok_or(
            KernelError::InvariantViolation {
                reason: "Metal outer remainder lost its prepared storage",
            },
        )?;
        if !storage.matches(&self.context, cycles, dispatch) {
            return Err(KernelError::InvariantViolation {
                reason: "prepared outer-remainder storage disagrees with the relation geometry",
            });
        }
        let storage_owned_bytes = storage.owned_bytes();
        let compact_rows_storage_id = rows.instruction_input_allocation_identity();
        let residual_rows_storage_id = rows.allocation_identity();
        let device_registry_id = rows.device_registry_id();
        let rows =
            session
                .take::<SpartanOuterUniskipRows>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "Metal outer remainder rows disappeared after admission",
                })?;
        let storage = session.take::<OuterRemainderSequenceStorage>().ok_or(
            KernelError::InvariantViolation {
                reason: "Metal outer remainder storage disappeared after validation",
            },
        )?;
        let storage_buffer_identities = storage.buffer_identities();
        let _sequence_span =
            tracing::info_span!("MetalOuterRemainder::sequence_prepare", cycles, rounds).entered();
        let sequence = storage.attach(rows).map_err(metal_prepare_error)?;
        let attached = sequence.storage_stats().map_err(metal_prepare_error)?;
        if attached.owned_bytes != storage_owned_bytes
            || attached.buffer_identities != storage_buffer_identities
        {
            return Err(KernelError::InvariantViolation {
                reason: "attached outer-remainder storage changed allocation identity",
            });
        }
        drop(_sequence_span);

        // Prepared storage and resident rows have now been consumed together.
        // Subsequent device failures are terminal because the sumcheck state
        // may already have advanced.
        let tau = take_metal_spartan_outer_tau(session, log_t)?;
        #[cfg(any(test, feature = "test-utils"))]
        let _ = self
            .test_counters
            .outer_remainder_sequences
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let metadata = MetalOuterResidentMetadata {
            compact_rows_storage_id,
            residual_rows_storage_id,
            device_registry_id,
            resident_rows: cycles,
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
        let point = self.product_tau_low();
        let split = point.len() / 2;
        let (out_point, in_point) = point.split_at(split);
        (
            EqPolynomial::evals(in_point, None),
            EqPolynomial::evals(out_point, None),
        )
    }

    fn product_tau_low(&self) -> Vec<AkitaField> {
        self.challenges[1..].iter().rev().copied().collect()
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
    cpu_tail_elements: usize,
    gpu_active_breakdown: OuterRemainderGpuActiveBreakdown,
    product_uniskip_endpoint_carrier: Option<MetalProductUniskipEndpointCarrier>,
    registers_claim_async_stage1_carry: Option<MetalRegistersClaimAsyncStage1Carry>,
    residue_ready: bool,
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
        if let Some(carrier) = &self.product_uniskip_endpoint_carrier {
            visitor.visit_field(
                allocative::Key::new("product_uniskip_endpoint_carrier"),
                carrier,
            );
        }
        if let Some(carry) = &self.registers_claim_async_stage1_carry {
            visitor.visit_field(allocative::Key::new("registers_claim_stage1_carry"), carry);
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
            cpu_tail_elements: tail_elements,
            gpu_active_breakdown: OuterRemainderGpuActiveBreakdown {
                materialize: materialize_gpu_active,
                ..OuterRemainderGpuActiveBreakdown::default()
            },
            product_uniskip_endpoint_carrier: None,
            registers_claim_async_stage1_carry: None,
            residue_ready: false,
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
        let sequence = self
            .sequence
            .as_mut()
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "Metal outer remainder released resident rows before openings",
            })?;
        let opening_output_elements = sequence.opening_output_count();
        let claimed = {
            let phase = tracing::info_span!(
                "MetalOuterRemainder::output_claims",
                dispatch_wall_ns = tracing::field::Empty,
                gpu_active_ns = tracing::field::Empty,
                readbacks = 1u64,
                output_elements = opening_output_elements,
                readback_bytes = opening_output_elements * size_of::<AkitaField>(),
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
        let product_tau_low = self.host.product_tau_low();
        if let Some(endpoints) = sequence.take_product_uniskip_endpoints() {
            self.product_uniskip_endpoint_carrier = Some(MetalProductUniskipEndpointCarrier {
                log_t: self.host.rounds - 1,
                tau_low: product_tau_low.clone(),
                endpoints,
                source_rows: self.resident_rows,
                source_row_storage_id: self.compact_rows_storage_id,
                device_registry_id: self.device_registry_id,
            });
        }
        if self.registers_claim_async_stage1_carry.is_some() {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "Metal outer remainder rebuilt its registers-claim carry",
            });
        }
        let context = sequence.context().clone();
        let registers_claim_async_stage1_carry = sequence
            .take_pending_registers_claim_carrier()
            .map_err(metal_output_error)?
            .map(|pending| {
                MetalRegistersClaimPendingStage1Carry::from_outer(
                    pending,
                    MetalRegistersClaimOuterSource {
                        context: &context,
                        product_tau_low: &product_tau_low,
                        rows: self.resident_rows,
                        compact_storage_id: self.compact_rows_storage_id,
                        residual_storage_id: self.residual_rows_storage_id,
                        device_registry_id: self.device_registry_id,
                    },
                )
            })
            .transpose()
            .map_err(metal_output_error)?
            .map(MetalRegistersClaimPendingStage1Carry::start);
        let completed_gpu_active = sequence.gpu_active_time();
        if self.gpu_active_breakdown.total() != Some(completed_gpu_active) {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "Outer remainder GPU phase timings do not sum to the member total",
            });
        }
        let storage = sequence.storage_stats().map_err(metal_output_error)?;
        if storage.compact_row_identity != self.compact_rows_storage_id
            || storage.residual_row_identity != self.residual_rows_storage_id
            || storage.row_device_registry_id != self.device_registry_id
        {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "Metal outer remainder changed its resident row allocation",
            });
        }
        self.residue_ready = true;
        self.registers_claim_async_stage1_carry = registers_claim_async_stage1_carry;
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

    fn park_residue(mut self: Box<Self>, session: &mut ProofSession) {
        if self.residue_ready {
            if let Some(sequence) = self.sequence.take() {
                session.park(sequence);
            }
        }
        if let Some(carrier) = self.product_uniskip_endpoint_carrier.take() {
            session.park(carrier);
        }
        if let Some(carry) = self.registers_claim_async_stage1_carry.take() {
            session.park(carry);
        }
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
    use crate::metal::solinas::{
        MetalError, OuterRemainderPhase, OuterRemainderSequence, OuterRemainderSequenceConfig,
    };
    use crate::metal::{MetalBackend, MetalConfig, SpartanOuterRemainderMetalConfig};
    use crate::optimized::harness::run_lockstep;

    use crate::optimized::spartan_outer::{
        prepare_metal_spartan_outer_witness_rows, OptimizedOuterRemainder, OptimizedOuterUniskip,
    };
    use crate::uniskip::UniskipKernel;
    use crate::{PrepareKernel, ProofSession, ProverInputs};
    use jolt_field::AkitaField;
    use jolt_poly::EqPolynomial;

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
        let working_set = |cycles,
                           stage1,
                           instruction_input,
                           instruction_read_raf,
                           borrow_outer_residual,
                           shift,
                           uniskip,
                           remainder| {
            resident_row_working_set(
                cycles,
                stage1,
                instruction_input,
                instruction_read_raf,
                borrow_outer_residual,
                shift,
                uniskip,
                remainder,
                Default::default(),
            )
            .unwrap()
        };

        assert_eq!(
            working_set(1 << 26, true, true, false, false, true, true, true),
            21_507_674_448
        );
        assert_eq!(
            working_set(1 << 28, true, true, false, false, true, true, true),
            86_010_499_408
        );
        assert_eq!(
            working_set(1 << 26, true, true, true, false, true, true, true),
            23_722_266_960
        );
        assert_eq!(
            working_set(1 << 27, true, true, true, false, true, true, true),
            47_438_500_176
        );
        assert_eq!(
            working_set(1 << 26, false, true, false, false, false, false, false),
            9_664_659_456
        );
        assert_eq!(
            working_set(1 << 28, false, true, false, false, false, false, false),
            38_656_671_744
        );
        assert_eq!(
            working_set(1 << 28, true, false, false, false, true, true, true),
            60_238_729_552
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

    fn adapter_parity_case(log_t: usize, cpu_tail_elements: usize) {
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
                        max_threadgroups: 2,
                        cpu_tail_elements,
                        ..Default::default()
                    },
                },
                ..Default::default()
            };
            let metal = MetalBackend::new(config).unwrap();
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
            device.park_residue(&mut metal_session);
            let deferred = metal_session.take::<OuterRemainderSequence>().unwrap();
            assert_eq!(deferred.phase(), OuterRemainderPhase::OpeningsComplete);
        });
    }

    #[test]
    fn outer_remainder_adapter_matches_optimized_cpu_all_rounds_and_openings() {
        adapter_parity_case(5, 8);
    }

    #[test]
    fn outer_remainder_handoff_waits_for_the_stream_bind() {
        adapter_parity_case(3, 16);
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
