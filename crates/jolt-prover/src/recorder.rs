//! The compile-time proof-mode seam: one constructor per mode-divergent
//! object, with the mode carried entirely in `#[cfg(feature = "zk")]` types.
//!
//! Stage recipes stay mode-agnostic: they call [`ProofMode::recorder`] for
//! the batch recorder and [`ProofMode::prove_uniskip`] for the uni-skip arm,
//! and both return the clear or committed flavor depending on how the crate
//! was compiled. There is no runtime flag to drift — the recorder type *is*
//! the mode, exactly as the stage drivers were designed around
//! (`specs/prover-stage-drivers.md`).

#[cfg(not(feature = "zk"))]
use core::marker::PhantomData;

use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::SumcheckProof;
#[cfg(not(feature = "zk"))]
use jolt_sumcheck::{prove_uniskip_clear, ClearSumcheckRecorder};
#[cfg(feature = "zk")]
use jolt_sumcheck::{
    prove_uniskip_committed, CommittedSumcheckRecorder, CommittedSumcheckWitness, RecordedSumcheck,
};
use jolt_transcript::{AppendToTranscript, Transcript};

use crate::ProverError;

/// The compiled mode's batch recorder type; `Commitment = VC::Output` in both
/// modes, so everything downstream of the recorder is mode-independent.
#[cfg(feature = "zk")]
pub type ModeRecorder<'a, F, VC> = CommittedSumcheckRecorder<'a, F, VC, rand_core::OsRng>;
#[cfg(not(feature = "zk"))]
pub type ModeRecorder<'a, F, VC> =
    ClearSumcheckRecorder<F, <VC as jolt_crypto::Commitment>::Output>;

/// A proved uni-skip round in the compiled mode: the wire proof, the
/// reduction challenge, and the prover-internal output claim (absorbed by the
/// clear arm, committed and retained by the ZK arm).
pub struct ProvedUniskipMode<F: Field, C> {
    pub proof: SumcheckProof<F, C>,
    pub challenge: F,
    pub output_claim: F,
    #[cfg(feature = "zk")]
    pub witness: CommittedSumcheckWitness<F>,
}

/// The per-proof mode context. Clear builds carry nothing; ZK builds carry
/// the vector-commitment setup every committed recorder and uni-skip commit
/// against (the same setup the verifier validates in `CheckedInputs`).
pub struct ProofMode<'a, VC: VectorCommitment> {
    #[cfg(feature = "zk")]
    vc_setup: &'a VC::Setup,
    #[cfg(not(feature = "zk"))]
    _vc: PhantomData<&'a VC>,
}

impl<'a, VC: VectorCommitment> ProofMode<'a, VC> {
    /// `vc_setup` is the preprocessing's BlindFold vector-commitment setup;
    /// required (and validated against) only in ZK builds.
    pub fn new(vc_setup: Option<&'a VC::Setup>) -> Result<Self, ProverError<VC::Field>> {
        #[cfg(feature = "zk")]
        {
            let vc_setup = vc_setup.ok_or(ProverError::Verifier(
                jolt_verifier::VerifierError::MissingVectorCommitmentSetup,
            ))?;
            Ok(Self { vc_setup })
        }
        #[cfg(not(feature = "zk"))]
        {
            let _ = vc_setup;
            Ok(Self { _vc: PhantomData })
        }
    }

    /// A fresh batch recorder for one stage.
    pub fn recorder(&self) -> Result<ModeRecorder<'a, VC::Field, VC>, ProverError<VC::Field>> {
        #[cfg(feature = "zk")]
        {
            Ok(CommittedSumcheckRecorder::new(
                self.vc_setup,
                rand_core::OsRng,
            )?)
        }
        #[cfg(not(feature = "zk"))]
        {
            let _ = self;
            Ok(ClearSumcheckRecorder::new())
        }
    }

    /// Prove a uni-skip first round in the compiled mode. The clear arm
    /// absorbs the full labeled polynomial and the output claim; the ZK arm
    /// commits the coefficients and the output claim and retains the witness.
    #[expect(
        clippy::type_complexity,
        reason = "the associated-type projections spell out one small struct"
    )]
    pub fn prove_uniskip<T>(
        &self,
        round_poly: UnivariatePoly<VC::Field>,
        input_claim: VC::Field,
        degree: usize,
        domain_size: usize,
        transcript: &mut T,
    ) -> Result<ProvedUniskipMode<VC::Field, VC::Output>, ProverError<VC::Field>>
    where
        VC::Output: Clone + AppendToTranscript,
        T: Transcript<Challenge = VC::Field>,
    {
        #[cfg(feature = "zk")]
        {
            let proved = prove_uniskip_committed::<VC::Field, VC, T, _>(
                round_poly,
                input_claim,
                degree,
                domain_size,
                self.vc_setup,
                rand_core::OsRng,
                transcript,
            )?;
            Ok(ProvedUniskipMode {
                proof: proved.proof,
                challenge: proved.challenge,
                output_claim: proved.output_claim,
                witness: proved.witness,
            })
        }
        #[cfg(not(feature = "zk"))]
        {
            let _ = self;
            let proved = prove_uniskip_clear::<VC::Field, VC::Output, T>(
                round_poly,
                input_claim,
                degree,
                domain_size,
                transcript,
            )?;
            Ok(ProvedUniskipMode {
                proof: proved.proof,
                challenge: proved.challenge,
                output_claim: proved.output_claim,
            })
        }
    }
}

/// Split a recorded sumcheck into its wire proof and the ZK-retained
/// witness; an absent witness is a recorder-contract violation in a ZK build.
#[cfg(feature = "zk")]
#[expect(
    clippy::type_complexity,
    reason = "the pair is the two halves of RecordedSumcheck, nothing more"
)]
pub(crate) fn split_recorded<F: Field, C>(
    recorded: RecordedSumcheck<F, C>,
) -> Result<(SumcheckProof<F, C>, CommittedSumcheckWitness<F>), ProverError<F>> {
    let witness = recorded
        .committed_witness
        .ok_or(ProverError::InvariantViolation {
            reason: "the committed recorder retained no witness",
        })?;
    Ok((recorded.proof, witness))
}
