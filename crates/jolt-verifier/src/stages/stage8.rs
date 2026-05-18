//! Stage 8 verifier skeleton.

use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::{CommitmentScheme, EvaluationClaim, VerifierOpeningClaim};
use jolt_poly::Point;
use jolt_transcript::Transcript;

use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{
        stage1::Stage1Output, stage2::Stage2Output, stage3::Stage3Output, stage4::Stage4Output,
        stage5::Stage5Output, stage6::Stage6Output, stage7::Stage7Output,
    },
    verifier::CheckedInputs,
    VerifierError,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageOpening<F: Field> {
    pub id: JoltOpeningId,
    pub point: Point<F>,
    pub opening_claim: F,
}

impl<F: Field> StageOpening<F> {
    #[cfg(test)]
    pub(crate) fn new(id: JoltOpeningId, point: impl Into<Point<F>>, opening_claim: F) -> Self {
        Self {
            id,
            point: point.into(),
            opening_claim,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OpeningPlan<F: Field, C> {
    pub claims: Vec<VerifierOpeningClaim<F, C>>,
}

impl<F: Field, C> OpeningPlan<F, C> {
    pub fn new() -> Self {
        Self { claims: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.claims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    pub fn push(&mut self, commitment: C, opening: StageOpening<F>) {
        self.claims.push(VerifierOpeningClaim {
            commitment,
            evaluation: EvaluationClaim::new(opening.point, opening.opening_claim),
        });
    }
}

impl<F: Field, C> Default for OpeningPlan<F, C> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Deps<'a, F: Field> {
    pub stage1: &'a Stage1Output<F>,
    pub stage2: &'a Stage2Output<F>,
    pub stage3: &'a Stage3Output<F>,
    pub stage4: &'a Stage4Output<F>,
    pub stage5: &'a Stage5Output<F>,
    pub stage6: &'a Stage6Output<F>,
    pub stage7: &'a Stage7Output<F>,
}

pub fn deps<'a, F: Field>(
    stage1: &'a Stage1Output<F>,
    stage2: &'a Stage2Output<F>,
    stage3: &'a Stage3Output<F>,
    stage4: &'a Stage4Output<F>,
    stage5: &'a Stage5Output<F>,
    stage6: &'a Stage6Output<F>,
    stage7: &'a Stage7Output<F>,
) -> Deps<'a, F> {
    Deps {
        stage1,
        stage2,
        stage3,
        stage4,
        stage5,
        stage6,
        stage7,
    }
}

pub struct Stage8Output<F: Field, C> {
    pub opening_plan: OpeningPlan<F, C>,
}

pub fn verify<PCS, VC, T, ZkProof>(
    _checked: &CheckedInputs,
    _preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    _proof: &JoltProof<PCS, VC, ZkProof>,
    _transcript: &mut T,
    _deps: Deps<'_, PCS::Field>,
) -> Result<Stage8Output<PCS::Field, PCS::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    Err(VerifierError::Unimplemented)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_claims::protocols::jolt::{
        JoltCommittedPolynomial, JoltStageId, JoltVirtualPolynomial,
    };
    use jolt_field::{Fr, FromPrimitiveInt};

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct TestCommitment(u64);

    fn ram_inc() -> JoltOpeningId {
        JoltOpeningId::committed(
            JoltCommittedPolynomial::RamInc,
            JoltStageId::RamReadWriteChecking,
        )
    }

    fn ram_val() -> JoltOpeningId {
        JoltOpeningId::virtual_polynomial(
            JoltVirtualPolynomial::RamVal,
            JoltStageId::RamReadWriteChecking,
        )
    }

    #[test]
    fn stage_opening_pairs_named_scalar_with_stage_point() {
        let point = Point::high_to_low(vec![Fr::from_u64(2), Fr::from_u64(3)]);

        let opening = StageOpening::new(ram_inc(), point.clone(), Fr::from_u64(8));

        assert_eq!(opening.id, ram_inc());
        assert_eq!(opening.point, point);
        assert_eq!(opening.opening_claim, Fr::from_u64(8));
    }

    #[test]
    fn opening_plan_assembles_verifier_opening_claims() {
        let opening = StageOpening::new(
            ram_val(),
            Point::high_to_low(vec![Fr::from_u64(21)]),
            Fr::from_u64(13),
        );
        let mut plan = OpeningPlan::new();

        plan.push(TestCommitment(4), opening);

        assert_eq!(plan.len(), 1);
        assert_eq!(plan.claims[0].commitment, TestCommitment(4));
        assert_eq!(
            plan.claims[0].evaluation.point.as_slice(),
            &[Fr::from_u64(21)]
        );
        assert_eq!(plan.claims[0].evaluation.value, Fr::from_u64(13));
    }
}
