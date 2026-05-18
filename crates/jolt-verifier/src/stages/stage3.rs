//! Stage 3 verifier skeleton.

use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_transcript::Transcript;

use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{stage1::Stage1Output, stage2::Stage2Output},
    verifier::CheckedInputs,
    VerifierError,
};

pub struct Deps<'a, F: Field> {
    pub stage1: &'a Stage1Output<F>,
    pub stage2: &'a Stage2Output<F>,
}

pub fn deps<'a, F: Field>(stage1: &'a Stage1Output<F>, stage2: &'a Stage2Output<F>) -> Deps<'a, F> {
    Deps { stage1, stage2 }
}

pub struct Stage3Output<F: Field> {
    pub challenges: Vec<F>,
}

pub fn verify<PCS, VC, T, ZkProof>(
    _checked: &CheckedInputs,
    _preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    _proof: &JoltProof<PCS, VC, ZkProof>,
    _transcript: &mut T,
    _deps: Deps<'_, PCS::Field>,
) -> Result<Stage3Output<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    Err(VerifierError::Unimplemented)
}
