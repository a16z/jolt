//! ZK verifier skeleton.

use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_transcript::Transcript;

use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{
        stage1::Stage1Output, stage2::Stage2Output, stage3::Stage3Output, stage4::Stage4Output,
        stage5::Stage5Output, stage6::Stage6Output, stage7::Stage7Output, stage8::Stage8Output,
    },
    verifier::CheckedInputs,
    VerifierError,
};

pub struct Deps<'a, F: Field> {
    pub stage1: &'a Stage1Output<F>,
    pub stage2: &'a Stage2Output<F>,
    pub stage3: &'a Stage3Output<F>,
    pub stage4: &'a Stage4Output<F>,
    pub stage5: &'a Stage5Output<F>,
    pub stage6: &'a Stage6Output<F>,
    pub stage7: &'a Stage7Output<F>,
    pub stage8: &'a Stage8Output<F>,
}

#[expect(
    clippy::too_many_arguments,
    reason = "ZK verification explicitly consumes every verified stage output."
)]
pub fn deps<'a, F: Field>(
    stage1: &'a Stage1Output<F>,
    stage2: &'a Stage2Output<F>,
    stage3: &'a Stage3Output<F>,
    stage4: &'a Stage4Output<F>,
    stage5: &'a Stage5Output<F>,
    stage6: &'a Stage6Output<F>,
    stage7: &'a Stage7Output<F>,
    stage8: &'a Stage8Output<F>,
) -> Deps<'a, F> {
    Deps {
        stage1,
        stage2,
        stage3,
        stage4,
        stage5,
        stage6,
        stage7,
        stage8,
    }
}

pub fn verify<PCS, VC, T, OpeningClaims, ZkProof>(
    _checked: &CheckedInputs,
    _preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    _proof: &JoltProof<PCS, VC, OpeningClaims, ZkProof>,
    _transcript: &mut T,
    _deps: Deps<'_, PCS::Field>,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    Err(VerifierError::Unimplemented)
}
