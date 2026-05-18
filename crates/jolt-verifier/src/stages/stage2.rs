//! Stage 2 verifier skeleton.

use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_transcript::Transcript;

use crate::{
    preprocessing::JoltVerifierPreprocessing, proof::JoltProof, stages::stage1::Stage1Output,
    verifier::CheckedInputs, VerifierError,
};

pub struct Deps<'a, F: Field> {
    pub stage1: &'a Stage1Output<F>,
}

pub fn deps<F: Field>(stage1: &Stage1Output<F>) -> Deps<'_, F> {
    Deps { stage1 }
}

pub struct Stage2Output<F: Field> {
    pub challenges: Vec<F>,
}

pub fn verify<PCS, VC, T, OpeningClaims, ZkProof>(
    _checked: &CheckedInputs,
    _preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    _proof: &JoltProof<PCS, VC, OpeningClaims, ZkProof>,
    _transcript: &mut T,
    _deps: Deps<'_, PCS::Field>,
) -> Result<Stage2Output<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    Err(VerifierError::Unimplemented)
}
