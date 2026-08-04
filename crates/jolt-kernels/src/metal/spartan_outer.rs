use jolt_field::AkitaField;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::SumcheckError;
use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
use jolt_witness::JoltWitnessPlane;

use super::instruction_read_raf::MetalBackend;
use super::solinas::{
    instruction_ra_weight_capacities, InstructionRaSequenceStorage, MetalError,
    SpartanOuterUniskipConfig,
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
        if cycles >= self.config.spartan_outer_uniskip.trace_cutoff_elements {
            let rows = prepare_metal_spartan_outer_witness_rows(&self.context, witness, cycles)?;
            session.park(rows);
        }
        if cycles
            >= self
                .config
                .instruction_ra_virtualization
                .trace_cutoff_elements
            && cycles >= 32
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
                        self.config.instruction_ra_virtualization.dispatch,
                    )
                    .map_err(metal_prepare_error)?
            };
            session.park::<InstructionRaSequenceStorage>(storage);
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
        if (1usize << log_t) < self.config.spartan_outer_uniskip.trace_cutoff_elements {
            return <OptimizedOuterUniskip as UniskipKernel<
                AkitaField,
                OuterRemainder<AkitaField>,
            >>::prepare(&OptimizedOuterUniskip, session, log_t, tau, witness);
        }
        prepare_metal_spartan_outer_uniskip(
            &self.context,
            self.config.spartan_outer_uniskip.dispatch,
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
