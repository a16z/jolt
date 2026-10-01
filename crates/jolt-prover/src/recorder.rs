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
use jolt_field::JoltField;
use jolt_poly::UnivariatePoly;
#[cfg(not(feature = "zk"))]
use jolt_sumcheck::{prove_uniskip_clear, ClearSumcheckRecorder};
#[cfg(feature = "zk")]
use jolt_sumcheck::{prove_uniskip_committed, CommittedSumcheckRecorder, CommittedSumcheckWitness};
use jolt_transcript::{ProverTranscript, Sponge};

use crate::ProverError;

/// The compiled mode's batch recorder type.
#[cfg(feature = "zk")]
pub type ModeRecorder<'a, VC> =
    CommittedSumcheckRecorder<'a, <VC as VectorCommitment>::Field, VC, rand_core::OsRng>;
#[cfg(not(feature = "zk"))]
pub type ModeRecorder<'a, VC> = ClearSumcheckRecorder<<VC as VectorCommitment>::Field>;

/// A proved uni-skip round in the compiled mode: the reduction challenge and
/// the output claim (sent by the clear arm, committed and retained by the ZK
/// arm).
pub struct ProvedUniskipMode<F: JoltField> {
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
    pub fn recorder(&self) -> Result<ModeRecorder<'a, VC>, ProverError<VC::Field>> {
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

    /// Prove a uni-skip first round in the compiled mode. The clear arm sends
    /// the full polynomial and the output claim; the ZK arm commits the
    /// coefficients and the output claim and retains the witness.
    pub fn prove_uniskip<H: Sponge>(
        &self,
        round_poly: UnivariatePoly<VC::Field>,
        input_claim: VC::Field,
        degree: usize,
        domain_size: usize,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<ProvedUniskipMode<VC::Field>, ProverError<VC::Field>> {
        #[cfg(feature = "zk")]
        {
            let proved = prove_uniskip_committed::<VC::Field, VC, H, _>(
                round_poly,
                input_claim,
                degree,
                domain_size,
                self.vc_setup,
                rand_core::OsRng,
                transcript,
            )?;
            Ok(ProvedUniskipMode {
                challenge: proved.challenge,
                output_claim: proved.output_claim,
                witness: proved.witness,
            })
        }
        #[cfg(not(feature = "zk"))]
        {
            let _ = self;
            let proved = prove_uniskip_clear::<VC::Field, H>(
                round_poly,
                input_claim,
                degree,
                domain_size,
                transcript,
            )?;
            Ok(ProvedUniskipMode {
                challenge: proved.challenge,
                output_claim: proved.output_claim,
            })
        }
    }
}
