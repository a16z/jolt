//! Prover-side helpers that replay verifier state to assemble proofs.

use common::jolt_device::JoltDevice;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::{BatchOpeningScheme, CommitmentLayoutDigest, CommitmentScheme};
use jolt_transcript::{AppendToTranscript, Transcript};

use crate::{
    config::JoltProtocolConfig, preprocessing::JoltVerifierPreprocessing, proof::JoltProof,
    stages::stage8, VerifierError,
};

pub fn stage8_batch_statement<F, PCS, VC, T, ZkProof>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment: Option<&PCS::Output>,
    zk: bool,
) -> Result<stage8::Stage8BatchStatement<F, PCS::Output>, VerifierError>
where
    F: Field + AppendToTranscript,
    PCS: CommitmentScheme<Field = F> + BatchOpeningScheme,
    PCS::Output: Clone + AppendToTranscript + CommitmentLayoutDigest,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    crate::verifier::stage8_batch_statement::<F, PCS, VC, T, ZkProof>(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
        zk,
    )
}

pub fn stage8_batch_statement_with_config<F, PCS, VC, T, ZkProof>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment: Option<&PCS::Output>,
    config: &JoltProtocolConfig,
) -> Result<stage8::Stage8BatchStatement<F, PCS::Output>, VerifierError>
where
    F: Field + AppendToTranscript,
    PCS: CommitmentScheme<Field = F> + BatchOpeningScheme,
    PCS::Output: Clone + AppendToTranscript + CommitmentLayoutDigest,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    crate::verifier::stage8_batch_statement_with_config::<F, PCS, VC, T, ZkProof>(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
        config,
    )
}

pub fn stage8_batch_statement_with_config_and_transcript<F, PCS, VC, T, ZkProof>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment: Option<&PCS::Output>,
    config: &JoltProtocolConfig,
) -> Result<(stage8::Stage8BatchStatement<F, PCS::Output>, T), VerifierError>
where
    F: Field + AppendToTranscript,
    PCS: CommitmentScheme<Field = F> + BatchOpeningScheme,
    PCS::Output: Clone + AppendToTranscript + CommitmentLayoutDigest,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    crate::verifier::stage8_batch_statement_with_config_and_transcript::<F, PCS, VC, T, ZkProof>(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
        config,
    )
}
