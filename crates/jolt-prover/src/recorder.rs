//! The compile-time proof-mode seam: one constructor per mode-divergent
//! object, with the mode carried entirely in `#[cfg(feature = "zk")]` types.
//!
//! Stage recipes stay mode-agnostic: they call [`ProofMode::recorder`] for
//! the batch recorder, [`ProofMode::prove_uniskip`] for the uni-skip arm and
//! [`ProofMode::send_private`] for private values sent outside a sumcheck,
//! and each returns the clear or masked flavor depending on how the crate was
//! compiled. There is no runtime flag to drift — the recorder type *is* the
//! mode, exactly as the stage drivers were designed around
//! (`specs/prover-stage-drivers.md`).

#[cfg(not(feature = "zk"))]
use core::marker::PhantomData;

#[cfg(feature = "zk")]
use jolt_crypto::Commitment;
use jolt_crypto::VectorCommitment;
#[cfg(feature = "zk")]
use jolt_field::CanonicalEncoding;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::ClearSumcheckRecorder;
#[cfg(feature = "zk")]
use jolt_sumcheck::{prove_uniskip, ProvedUniskip};
#[cfg(not(feature = "zk"))]
use jolt_sumcheck::{prove_uniskip_clear, ProvedUniskip};
#[cfg(not(feature = "zk"))]
use jolt_transcript::{PadSlots, PrivateChannel};
use jolt_transcript::{ProverTranscript, Sponge};
use jolt_verifier::stages::relations::ClaimRoutes;
#[cfg(not(feature = "zk"))]
use jolt_verifier::VerifierError;
#[cfg(feature = "zk")]
use jolt_zk::{
    blindfold::BlindFoldPadOpening,
    masking::{CommittedPads, MaskingRecorder},
};

use crate::ProverError;

/// The pads a ZK proof commits before its first challenge.
#[cfg(feature = "zk")]
pub type ModePads<VC> = CommittedPads<
    <VC as VectorCommitment>::Field,
    BlindFoldPadOpening<<VC as VectorCommitment>::Field, <VC as Commitment>::Output>,
>;

/// The compiled mode's batch recorder type.
#[cfg(feature = "zk")]
pub type ModeRecorder<'a, VC> = MaskingRecorder<
    'a,
    <VC as VectorCommitment>::Field,
    BlindFoldPadOpening<<VC as VectorCommitment>::Field, <VC as Commitment>::Output>,
>;
#[cfg(not(feature = "zk"))]
pub type ModeRecorder<'a, VC> = ClearSumcheckRecorder<<VC as VectorCommitment>::Field>;

/// Which output-claim routes a recorder sends after a batch's rounds.
pub trait ClaimRecorder {
    /// The routes a proof in this recorder's mode sends on.
    fn wire_routes(routes: ClaimRoutes) -> ClaimRoutes;
}

impl<F> ClaimRecorder for ClearSumcheckRecorder<F> {
    fn wire_routes(routes: ClaimRoutes) -> ClaimRoutes {
        routes
    }
}

/// A masked proof sends every runtime-aliased cell: the pad schedule is fixed
/// before the challenges that decide the alias ([`ClaimRoutes::masked`]).
#[cfg(feature = "zk")]
impl<F, O> ClaimRecorder for MaskingRecorder<'_, F, O> {
    fn wire_routes(routes: ClaimRoutes) -> ClaimRoutes {
        routes.masked().0
    }
}

/// The per-proof mode context. Clear builds carry nothing; ZK builds carry
/// the pads committed before the first challenge, which every recorder and
/// private send masks with.
pub struct ProofMode<VC: VectorCommitment> {
    #[cfg(feature = "zk")]
    pads: ModePads<VC>,
    #[cfg(not(feature = "zk"))]
    _vc: PhantomData<VC>,
}

#[cfg(not(feature = "zk"))]
impl<VC: VectorCommitment> ProofMode<VC> {
    pub fn clear() -> Self {
        Self { _vc: PhantomData }
    }
}

#[cfg(feature = "zk")]
impl<VC: VectorCommitment> ProofMode<VC>
where
    VC::Field: CanonicalEncoding,
{
    /// The masked mode over `pads`.
    pub fn masked(pads: ModePads<VC>) -> Self {
        Self { pads }
    }

    /// The pads, for the residual proof once every masked message is sent.
    pub fn into_pads(self) -> ModePads<VC> {
        self.pads
    }
}

impl<VC: VectorCommitment> ProofMode<VC> {
    /// A fresh batch recorder for one stage.
    pub fn recorder(&mut self) -> ModeRecorder<'_, VC> {
        #[cfg(feature = "zk")]
        {
            MaskingRecorder::new(&mut self.pads)
        }
        #[cfg(not(feature = "zk"))]
        {
            ClearSumcheckRecorder::new()
        }
    }

    /// Sends private values outside a sumcheck (stage 4's staged openings):
    /// clear, or masked by the next pads.
    pub fn send_private<H: Sponge>(
        &mut self,
        values: &[VC::Field],
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), ProverError<VC::Field>> {
        #[cfg(feature = "zk")]
        {
            let _ = self.pads.send(values, transcript)?;
        }
        #[cfg(not(feature = "zk"))]
        {
            let _: PadSlots = transcript
                .exchange_private_base(&mut values.to_vec())
                .map_err(VerifierError::from)?;
        }
        Ok(())
    }

    /// Prove a uni-skip first round in the compiled mode: the full round
    /// polynomial and the output claim, sent clear or masked.
    pub fn prove_uniskip<H: Sponge>(
        &mut self,
        round_poly: UnivariatePoly<VC::Field>,
        input_claim: VC::Field,
        degree: usize,
        domain_size: usize,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<ProvedUniskip<VC::Field>, ProverError<VC::Field>> {
        #[cfg(feature = "zk")]
        {
            let _span = tracing::info_span!("prove_uniskip_masked").entered();
            let (proved, ()) = prove_uniskip(
                round_poly,
                input_claim,
                degree,
                domain_size,
                MaskingRecorder::new(&mut self.pads),
                transcript,
            )?;
            Ok(proved)
        }
        #[cfg(not(feature = "zk"))]
        {
            Ok(prove_uniskip_clear::<VC::Field, H>(
                round_poly,
                input_claim,
                degree,
                domain_size,
                transcript,
            )?)
        }
    }
}
