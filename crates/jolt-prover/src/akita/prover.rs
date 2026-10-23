//! The Akita top-level prover: the stage recipes run in protocol order on
//! one transcript and one backend session, and the transcript's argument
//! string is the [`JoltProof`]; Akita's opening messages are its tail.

use common::jolt_device::JoltDevice;
use jolt_algebra::JoltField;
use jolt_crypto::VectorCommitment;
use jolt_openings::{
    CommitmentScheme, GroupCommitmentMetadata, GroupSetupMetadata, TransparentObjectSetup,
};
use jolt_transcript::Sponge;
#[cfg(not(feature = "zk"))]
use jolt_verifier::config::{JoltProtocolConfig, ZkConfig};
use jolt_verifier::proof::JoltProof;

#[cfg(not(feature = "zk"))]
use super::spine::prove_stages_1_to_7;
#[cfg(not(feature = "zk"))]
use super::stage0::prove_stage0;
#[cfg(not(feature = "zk"))]
use super::stage8::prove_stage8;
use super::witness::AdviceObject;
#[cfg(not(feature = "zk"))]
use crate::boundary::finish_stage;
#[cfg(not(feature = "zk"))]
use crate::ProofMode;
use crate::{JoltProverPreprocessing, ProverConfig, ProverError};
use jolt_kernels::{JoltBackend, KernelContext};

/// See [`super::prove`].
///
/// A Whiteout build (`zk`) rejects: its pad commitment, masked opening and
/// Σ tail are not wired yet (W6d).
#[tracing::instrument(skip_all, name = "jolt_prover::prove", fields(trace_length = config.trace_length))]
pub fn prove<F, PCS, VC, H>(
    mut backend: KernelContext<'_, F, JoltBackend<F, PCS>>,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&AdviceObject<PCS>>,
    public_io: &JoltDevice,
) -> Result<JoltProof, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + TransparentObjectSetup,
    PCS::ProverSetup: GroupSetupMetadata,
    PCS::VerifierSetup: GroupSetupMetadata,
    PCS::Output: Clone + PartialEq + GroupCommitmentMetadata,
    VC: VectorCommitment<Field = F>,
    H: Sponge,
{
    #[cfg(feature = "zk")]
    {
        let _ = (&mut backend, preprocessing, trusted_advice, public_io);
        Err(ProverError::Unsupported {
            reason: "Whiteout's pad commitment, masked Akita opening and Σ tail are not wired \
                     yet (W6d)",
        })
    }
    #[cfg(not(feature = "zk"))]
    {
        let mut mode = ProofMode::<VC>::clear();
        let stage0 = prove_stage0::<F, PCS, VC, H>(
            &mut backend,
            preprocessing,
            config,
            trusted_advice,
            public_io,
        )?;
        let log_t = config.trace_length.ilog2() as usize;
        finish_stage("stage0", log_t, backend.session(), &());
        let mut transcript = stage0.transcript;
        let (stage4, stage6b, stage7) = prove_stages_1_to_7::<F, PCS, VC, H>(
            &mut backend,
            &mut mode,
            &stage0.checked,
            config,
            preprocessing,
            public_io,
            &mut transcript,
        )?;
        prove_stage8::<F, PCS, VC, H>(
            &stage0.checked,
            config,
            preprocessing,
            &stage0.commitment,
            stage0.hint,
            stage0.untrusted_advice.as_ref(),
            trusted_advice,
            #[cfg(feature = "field-inline")]
            &stage0.field_inc,
            preprocessing
                .committed_program
                .as_ref()
                .map(|data| &data.direct_program),
            &stage4,
            &stage6b,
            &stage7,
            &mut transcript,
        )?;
        finish_stage("stage8", log_t, backend.session(), &());
        backend.finish();

        Ok(JoltProof {
            protocol: JoltProtocolConfig::for_zk(ZkConfig::Transparent),
            narg: transcript.finish(),
        })
    }
}
