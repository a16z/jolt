use jolt_crypto::PairingGroup;
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_hyperkzg::HyperKZGScheme;
#[cfg(feature = "zk")]
use jolt_hyperkzg::{
    HyperKZGCommitment, HyperKZGProof as HyperKZGOpeningProof, HyperKZGVerifierSetup,
};
use jolt_openings::CommitmentScheme;
#[cfg(feature = "zk")]
use jolt_openings::ZkOpeningScheme;
use jolt_transcript::{AppendToTranscript, Transcript};

#[cfg(feature = "zk")]
use crate::stages::hyperkzg::inputs::HyperKzgZkInputs;
use crate::{stages::hyperkzg::inputs::HyperKzgInputs, WrapperError};

use super::outputs::HyperKzgOutput;
#[cfg(feature = "zk")]
use super::outputs::HyperKzgZkOutput;

pub fn verify<P, T>(
    inputs: HyperKzgInputs<'_, '_, P>,
    transcript: &mut T,
) -> Result<HyperKzgOutput, WrapperError>
where
    P: PairingGroup,
    P::ScalarField: Field + AppendToTranscript,
    P::G1: AppendToTranscript,
    T: Transcript<Challenge = P::ScalarField>,
{
    let opening = &inputs.deps.spartan.witness_opening;
    <HyperKZGScheme<P> as CommitmentScheme>::bind_opening_inputs(
        transcript,
        opening.ry.as_slice(),
        &opening.witness_eval,
    );
    <HyperKZGScheme<P> as CommitmentScheme>::verify(
        &inputs.proof.hyperkzg.witness_commitment,
        opening.ry.as_slice(),
        opening.witness_eval,
        &inputs.proof.hyperkzg.witness_opening_proof,
        inputs.setup,
        transcript,
    )
    .map_err(|error| WrapperError::HyperKzgVerificationFailed {
        reason: error.to_string(),
    })?;
    Ok(HyperKzgOutput)
}

#[cfg(feature = "zk")]
pub fn verify_zk<P, VC, T>(
    inputs: HyperKzgZkInputs<'_, '_, P, VC>,
    transcript: &mut T,
) -> Result<HyperKzgZkOutput<VC::Output>, WrapperError>
where
    P: PairingGroup,
    P::ScalarField: Field + AppendToTranscript,
    P::G1: AppendToTranscript,
    VC: VectorCommitment<Field = P::ScalarField>,
    T: Transcript<Challenge = P::ScalarField>,
    HyperKZGScheme<P>: jolt_crypto::Commitment<Output = HyperKZGCommitment<P>>
        + CommitmentScheme<
            Field = P::ScalarField,
            Proof = HyperKZGOpeningProof<P>,
            VerifierSetup = HyperKZGVerifierSetup<P>,
        > + ZkOpeningScheme<HidingCommitment = VC::Output>,
{
    let opening_point = inputs.deps.spartan.inner_ry.as_slice();
    let hiding_evaluation_commitment = <HyperKZGScheme<P> as ZkOpeningScheme>::verify_zk(
        &inputs.proof.hyperkzg.witness_commitment,
        opening_point,
        &inputs.proof.hyperkzg.witness_opening_proof,
        inputs.setup,
        transcript,
    )
    .map_err(|error| WrapperError::HyperKzgVerificationFailed {
        reason: error.to_string(),
    })?;
    <HyperKZGScheme<P> as ZkOpeningScheme>::bind_zk_opening_inputs(
        transcript,
        opening_point,
        &hiding_evaluation_commitment,
    );

    Ok(HyperKzgZkOutput {
        hiding_evaluation_commitment,
    })
}
