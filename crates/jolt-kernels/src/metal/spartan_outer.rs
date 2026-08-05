use jolt_field::AkitaField;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::SumcheckError;
use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
use jolt_witness::JoltWitnessPlane;

use super::instruction_input::PreparedInstructionInput;
use super::instruction_read_raf::{MetalBackend, MetalConfig};
use super::solinas::{
    instruction_input_sequence_storage_bytes, instruction_ra_weight_capacities,
    spartan_outer_uniskip_invocation_bytes, spartan_outer_uniskip_row_bytes,
    InstructionRaSequenceStorage, MetalError, SpartanOuterUniskipConfig, SpartanOuterUniskipRows,
};
use crate::optimized::spartan_outer::{
    prepare_metal_spartan_outer_uniskip, prepare_metal_spartan_outer_witness_rows,
    OptimizedOuterUniskip,
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
    let mut bytes = spartan_outer_uniskip_row_bytes(cycles)?;
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

fn retain_rows_after_input_admission(
    instruction_input_eligible: bool,
    instruction_input_prepared: bool,
) -> bool {
    !instruction_input_eligible || instruction_input_prepared
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
        if stage1_eligible || instruction_input_eligible {
            let row_bytes = spartan_outer_uniskip_row_bytes(cycles).map_err(metal_prepare_error)?;
            let device = self.context.device_info();
            let admission = validate_resident_row_buffer(row_bytes, device.max_buffer_length)
                .and_then(|()| {
                    resident_row_working_set(cycles, stage1_eligible, instruction_input_eligible)
                })
                .and_then(|additional| self.context.validate_additional_working_set(additional));
            match admission {
                Ok(()) => {
                    let rows =
                        prepare_metal_spartan_outer_witness_rows(&self.context, witness, cycles)?;
                    session.park(rows);
                }
                Err(
                    error @ (MetalError::BufferTooLong { .. }
                    | MetalError::WorkingSetTooLarge { .. }),
                ) => {
                    tracing::warn!(
                        target: "jolt::metal",
                        error = %error,
                        "shared Spartan/InstructionInput Metal working set was not admitted"
                    );
                }
                Err(error) => return Err(metal_prepare_error(error)),
            }
        }
        self.prepare_instruction_input_storage(session, cycles)?;
        if !retain_rows_after_input_admission(
            instruction_input_eligible,
            session.state::<PreparedInstructionInput>().is_some(),
        ) {
            drop(session.take::<SpartanOuterUniskipRows>());
        }
        if session.state::<PreparedInstructionInput>().is_none() {
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
            if session.state::<PreparedInstructionInput>().is_none() {
                drop(session.take::<SpartanOuterUniskipRows>());
            }
            return <OptimizedOuterUniskip as UniskipKernel<
                AkitaField,
                OuterRemainder<AkitaField>,
            >>::prepare(&OptimizedOuterUniskip, session, log_t, tau, witness);
        }
        let retain_resident_rows = session.state::<PreparedInstructionInput>().is_some();
        prepare_metal_spartan_outer_uniskip(
            &self.context,
            self.config.spartan_outer_uniskip.dispatch,
            retain_resident_rows,
            session,
            log_t,
            tau,
            witness,
        )
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
        resident_row_consumers, resident_row_working_set, retain_rows_after_input_admission,
        use_metal_stage1, validate_resident_row_buffer,
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
    fn rejected_instruction_input_admission_discards_stage1_rows() {
        assert!(!retain_rows_after_input_admission(true, false));
        assert!(retain_rows_after_input_admission(true, true));
        assert!(retain_rows_after_input_admission(false, false));
    }
}
