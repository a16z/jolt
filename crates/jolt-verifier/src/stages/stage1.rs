//! Stage 1 verifier skeleton.

use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_transcript::Transcript;

use crate::{
    preprocessing::JoltVerifierPreprocessing, proof::JoltProof, verifier::CheckedInputs,
    VerifierError,
};

pub struct Stage1Output<F: Field> {
    pub challenges: Vec<F>,
}

pub fn verify<PCS, VC, T, OpeningClaims, ZkProof>(
    _checked: &CheckedInputs,
    _preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    _proof: &JoltProof<PCS, VC, OpeningClaims, ZkProof>,
    _transcript: &mut T,
) -> Result<Stage1Output<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    Err(VerifierError::Unimplemented)
}
