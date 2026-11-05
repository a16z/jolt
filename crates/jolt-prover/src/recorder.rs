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

use jolt_algebra::{Field, JoltField};
#[cfg(all(feature = "zk", not(feature = "akita")))]
use jolt_crypto::Commitment;
use jolt_crypto::VectorCommitment;
use jolt_poly::UnivariatePoly;
#[cfg(not(feature = "zk"))]
use jolt_sumcheck::prove_uniskip_clear;
#[cfg(feature = "zk")]
use jolt_sumcheck::{ClearProof, ClearSumcheckProof};
use jolt_sumcheck::{ClearSumcheckRecorder, SumcheckProof};
#[cfg(feature = "zk")]
use jolt_transcript::PadSlots;
#[cfg(not(feature = "zk"))]
use jolt_transcript::PrivateChannel;
use jolt_transcript::{Encoding, FieldTranscript, ProverTranscript};
use jolt_verifier::stages::relations::ClaimRoutes;
#[cfg(all(feature = "zk", not(feature = "akita")))]
use jolt_zk::blindfold::BlindFoldPadOpening;
#[cfg(feature = "zk")]
use jolt_zk::masking::{prove_uniskip_masked, CommittedPads, MaskedMessages, MaskingRecorder};
#[cfg(all(feature = "zk", feature = "akita"))]
use jolt_zk::whiteout::WhiteoutPadOpening;

use crate::ProverError;

/// What opens the pad commitment: BlindFold's vector-commitment opening.
#[cfg(all(feature = "zk", not(feature = "akita")))]
type ModePadOpening<VC> =
    BlindFoldPadOpening<<VC as VectorCommitment>::Field, <VC as Commitment>::Output>;
/// What opens the pad commitment: Whiteout's lattice pad opening.
#[cfg(all(feature = "zk", feature = "akita"))]
type ModePadOpening<VC> = WhiteoutPadOpening<<VC as VectorCommitment>::Field>;

/// The pads a ZK proof commits before its first challenge.
#[cfg(feature = "zk")]
pub type ModePads<VC> = CommittedPads<<VC as VectorCommitment>::Field, ModePadOpening<VC>>;

/// The compiled mode's batch recorder: compressed clear proofs or committed
/// proofs with retained witnesses for BlindFold.
#[cfg(feature = "zk")]
pub type ModeRecorder<'a, VC> =
    MaskingRecorder<'a, <VC as VectorCommitment>::Field, ModePadOpening<VC>>;
#[cfg(not(feature = "zk"))]
pub type ModeRecorder<'a, VC> = ClearSumcheckRecorder<<VC as VectorCommitment>::Field>;

/// Which output-claim routes a recorder sends after a batch's rounds.
pub trait ClaimRecorder {
    /// Whether the recorder masks: a masked proof sends its output claims on
    /// its wire routes instead of the clear curated absorb order.
    const MASKED: bool;

    /// The routes a proof in this recorder's mode sends on.
    fn wire_routes(routes: ClaimRoutes) -> ClaimRoutes;
}

impl<F: Field> ClaimRecorder for ClearSumcheckRecorder<F> {
    const MASKED: bool = false;

    fn wire_routes(routes: ClaimRoutes) -> ClaimRoutes {
        routes
    }
}

/// A masked proof sends every runtime-aliased cell: the pad schedule is fixed
/// before the challenges that decide the alias ([`ClaimRoutes::masked`]).
#[cfg(feature = "zk")]
impl<F, O> ClaimRecorder for MaskingRecorder<'_, F, O> {
    const MASKED: bool = true;

    fn wire_routes(routes: ClaimRoutes) -> ClaimRoutes {
        routes.masked().0
    }
}

/// A proved uni-skip round in the compiled mode: the wire proof, the
/// reduction challenge, and the prover-internal output claim.
pub struct ProvedUniskipMode<F: JoltField, C> {
    pub proof: SumcheckProof<F, C>,
    pub challenge: F,
    pub output_claim: F,
}

/// The per-proof mode context. Clear builds carry nothing; ZK builds carry
/// the pads committed before the first challenge, which every recorder and
/// private send masks with, and the proof's masked tape they write.
pub struct ProofMode<VC: VectorCommitment> {
    #[cfg(feature = "zk")]
    pads: ModePads<VC>,
    #[cfg(feature = "zk")]
    tape: MaskedMessages<VC::Field>,
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
impl<VC: VectorCommitment> ProofMode<VC> {
    /// The masked mode over `pads`, with an empty masked tape.
    pub fn masked(pads: ModePads<VC>) -> Self {
        Self {
            pads,
            tape: MaskedMessages::default(),
        }
    }

    /// The pads, for the residual proof once every masked message is sent,
    /// and the masked tape the proof carries.
    pub fn into_parts(self) -> (ModePads<VC>, MaskedMessages<VC::Field>) {
        (self.pads, self.tape)
    }

    /// The masked tape so far.
    pub fn tape(&self) -> &MaskedMessages<VC::Field> {
        &self.tape
    }
}

impl<VC: VectorCommitment> ProofMode<VC> {
    /// A fresh batch recorder for one stage.
    pub fn recorder(&mut self) -> ModeRecorder<'_, VC> {
        #[cfg(feature = "zk")]
        {
            MaskingRecorder::new(&mut self.pads, &mut self.tape)
        }
        #[cfg(not(feature = "zk"))]
        {
            ClearSumcheckRecorder::new()
        }
    }

    /// Sends private values outside a sumcheck (stage 4's staged openings):
    /// clear, or masked by the next pads.
    pub fn send_private<T>(
        &mut self,
        values: &[VC::Field],
        transcript: &mut T,
    ) -> Result<(), ProverError<VC::Field>>
    where
        T: FieldTranscript<VC::Field> + ProverTranscript,
    {
        #[cfg(feature = "zk")]
        {
            let _: PadSlots = self.pads.send(values, &mut self.tape, transcript)?;
        }
        #[cfg(not(feature = "zk"))]
        {
            transcript.append_private(values);
        }
        Ok(())
    }

    /// Prove a uni-skip first round in the compiled mode: the full round
    /// polynomial and the output claim, sent clear or masked.
    #[expect(
        clippy::type_complexity,
        reason = "the associated-type projections spell out one small struct"
    )]
    pub fn prove_uniskip<T>(
        &mut self,
        round_poly: UnivariatePoly<VC::Field>,
        input_claim: VC::Field,
        degree: usize,
        domain_size: usize,
        transcript: &mut T,
    ) -> Result<ProvedUniskipMode<VC::Field, VC::Output>, ProverError<VC::Field>>
    where
        VC::Output: Clone + Encoding,
        T: FieldTranscript<VC::Field> + ProverTranscript,
    {
        #[cfg(feature = "zk")]
        {
            let _span = tracing::info_span!("prove_uniskip_masked").entered();
            let proved = prove_uniskip_masked(
                &mut self.pads,
                &mut self.tape,
                &round_poly,
                input_claim,
                degree,
                domain_size,
                transcript,
            )?;
            Ok(ProvedUniskipMode {
                proof: SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof::default())),
                challenge: proved.challenge,
                output_claim: proved.output_claim,
            })
        }
        #[cfg(not(feature = "zk"))]
        {
            let proved = prove_uniskip_clear::<VC::Field, T>(
                round_poly,
                input_claim,
                degree,
                domain_size,
                transcript,
            )?;
            Ok(ProvedUniskipMode {
                proof: proved.proof.into(),
                challenge: proved.challenge,
                output_claim: proved.output_claim,
            })
        }
    }
}
