use jolt_field::AkitaField;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::SumcheckError;
use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
use jolt_witness::JoltWitnessPlane;

use super::instruction_input::PreparedInstructionInput;
use super::instruction_read_raf::{MetalBackend, MetalConfig};
use super::solinas::{
    instruction_input_row_bytes, instruction_input_sequence_storage_bytes,
    instruction_ra_weight_capacities, spartan_outer_uniskip_invocation_bytes,
    spartan_outer_uniskip_row_bytes, InstructionInputRows, InstructionRaSequenceStorage,
    MetalError, SpartanOuterUniskipConfig, SpartanOuterUniskipRows,
};
use crate::optimized::instruction_input::PreparedInstructionInputRows;
use crate::optimized::spartan_outer::{
    prepare_metal_instruction_input_witness_rows, prepare_metal_spartan_outer_uniskip,
    prepare_metal_spartan_outer_witness_rows, OptimizedOuterUniskip,
};
use crate::uniskip::UniskipKernel;
use crate::{KernelError, ProofSession};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpartanOuterUniskipMetalConfig {
    pub trace_cutoff_elements: usize,
    pub dispatch: SpartanOuterUniskipConfig,
}

fn resident_row_consumers(cycles: usize, config: &MetalConfig) -> (bool, bool) {
    let stage1 = cycles >= config.spartan_outer_uniskip.trace_cutoff_elements;
    let instruction_input = cycles >= config.instruction_input.trace_cutoff_elements
        && cycles > config.instruction_input.cutoff_elements;
    (stage1, instruction_input)
}

fn resident_row_working_set(
    cycles: usize,
    stage1: bool,
    instruction_input: bool,
) -> Result<u64, MetalError> {
    let mut bytes = if stage1 {
        spartan_outer_uniskip_row_bytes(cycles)?
    } else if instruction_input {
        instruction_input_row_bytes(cycles)?
    } else {
        0
    };
    if stage1 {
        bytes = bytes
            .checked_add(spartan_outer_uniskip_invocation_bytes(cycles)?)
            .ok_or(MetalError::InputTooLong(cycles))?;
    }
    if instruction_input {
        bytes = bytes
            .checked_add(instruction_input_sequence_storage_bytes(cycles)?)
            .ok_or(MetalError::InputTooLong(cycles))?;
    }
    Ok(bytes)
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

impl UniskipKernel<AkitaField, OuterRemainder<AkitaField>> for MetalBackend {
    fn prepare_witness(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<(), KernelError<AkitaField>> {
        let cycles = 1usize << log_t;
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
                            resident_row_working_set(
                                cycles,
                                candidate.stage1,
                                candidate.instruction_input,
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
                    Err(
                        error @ (MetalError::BufferTooLong { .. }
                        | MetalError::WorkingSetTooLarge { .. }),
                    ) => {
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
            drop(session.take::<SpartanOuterUniskipRows>());
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
            return <OptimizedOuterUniskip as UniskipKernel<
                AkitaField,
                OuterRemainder<AkitaField>,
            >>::prepare(&OptimizedOuterUniskip, session, log_t, tau, witness);
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
    ) -> Result<UnivariatePoly<AkitaField>, KernelError<AkitaField>> {
        <OptimizedOuterUniskip as UniskipKernel<
            AkitaField,
            OuterRemainder<AkitaField>,
        >>::first_round_poly(&OptimizedOuterUniskip, session, late_tau)
    }
}

fn metal_prepare_error(error: MetalError) -> KernelError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
    .into()
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use super::{
        prepare_cpu_instruction_input_now, resident_row_admission_candidates,
        resident_row_consumers, resident_row_working_set, use_metal_stage1,
        validate_resident_row_buffer, ResidentRowPlan,
    };
    use crate::metal::solinas::MetalError;
    use crate::metal::MetalConfig;

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
    }

    #[test]
    fn aggregate_instruction_input_working_set_matches_production_geometry() {
        assert_eq!(
            resident_row_working_set(1 << 26, true, true).unwrap(),
            17_182_425_248
        );
        assert_eq!(
            resident_row_working_set(1 << 28, true, true).unwrap(),
            68_724_588_704
        );
        assert_eq!(
            resident_row_working_set(1 << 26, false, true).unwrap(),
            9_664_659_456
        );
        assert_eq!(
            resident_row_working_set(1 << 28, false, true).unwrap(),
            38_656_671_744
        );
        assert_eq!(
            resident_row_working_set(1 << 28, true, false).unwrap(),
            42_952_818_848
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
}
