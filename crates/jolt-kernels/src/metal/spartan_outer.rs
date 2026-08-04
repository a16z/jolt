use jolt_field::AkitaField;
use jolt_poly::UnivariatePoly;
use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
use jolt_witness::JoltWitnessPlane;

use super::instruction_read_raf::MetalBackend;
use super::solinas::SpartanOuterUniskipConfig;
use crate::optimized::spartan_outer::{prepare_metal_spartan_outer_uniskip, OptimizedOuterUniskip};
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
