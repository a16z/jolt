//! Typed outputs produced by stage 4 verification.

use jolt_claims::protocols::jolt::JoltAdviceKind;
use jolt_field::Field;
use jolt_poly::{Point, HIGH_TO_LOW};
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::inputs::Stage4Claims;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4PublicOutput<F: Field> {
    pub challenges: Vec<F>,
    pub batching_coefficients: Vec<F>,
    pub registers_gamma: F,
    #[cfg(feature = "field-inline")]
    pub field_registers_gamma: F,
    pub ram_val_check_gamma: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4ClearOutput<F: Field> {
    pub public: Stage4PublicOutput<F>,
    pub output_claims: Stage4Claims<F>,
    pub batch: VerifiedStage4Batch<F>,
    pub ram_val_check_init: RamValCheckInitialEvaluation<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4ZkOutput<F: Field, C> {
    pub public: Stage4PublicOutput<F>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    pub ram_val_check_public_eval: F,
    pub registers_read_write_opening_point: Vec<F>,
    #[cfg(feature = "field-inline")]
    pub field_registers_read_write_opening_point: Vec<F>,
    pub ram_val_check_opening_point: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage4Output<F: Field, C> {
    Clear(Stage4ClearOutput<F>),
    Zk(Stage4ZkOutput<F, C>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage4Batch<F: Field> {
    pub batching_coefficients: Vec<F>,
    pub sumcheck_point: Point<HIGH_TO_LOW, F>,
    pub sumcheck_final_claim: F,
    pub expected_final_claim: F,
    pub registers_read_write: VerifiedStage4Sumcheck<F>,
    #[cfg(feature = "field-inline")]
    pub field_registers_read_write: VerifiedStage4Sumcheck<F>,
    pub ram_val_check: VerifiedStage4Sumcheck<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage4Sumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamValCheckInitialEvaluation<F: Field> {
    pub public_eval: F,
    pub advice_contributions: Vec<VerifiedRamValCheckAdviceContribution<F>>,
    pub full_eval: F,
}

impl<F: Field> RamValCheckInitialEvaluation<F> {
    pub fn advice_contribution(
        &self,
        kind: JoltAdviceKind,
    ) -> Option<&VerifiedRamValCheckAdviceContribution<F>> {
        self.advice_contributions
            .iter()
            .find(|contribution| contribution.kind == kind)
    }

    pub fn advice_opening_claim(&self, kind: JoltAdviceKind) -> Option<F> {
        self.advice_contribution(kind)
            .map(|contribution| contribution.opening_claim)
    }

    pub fn advice_opening_point(&self, kind: JoltAdviceKind) -> Option<&[F]> {
        self.advice_contribution(kind)
            .map(|contribution| contribution.opening_point.as_slice())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedRamValCheckAdviceContribution<F: Field> {
    pub kind: JoltAdviceKind,
    pub selector: F,
    pub opening_claim: F,
    pub opening_point: Vec<F>,
}

#[cfg(test)]
mod tests {
    use jolt_claims::protocols::jolt::JoltAdviceKind;
    use jolt_field::{Fr, FromPrimitiveInt};

    use super::*;

    #[test]
    fn ram_val_check_initial_evaluation_finds_advice_contribution_by_kind() {
        let trusted = VerifiedRamValCheckAdviceContribution {
            kind: JoltAdviceKind::Trusted,
            selector: Fr::from_u64(2),
            opening_claim: Fr::from_u64(3),
            opening_point: vec![Fr::from_u64(5)],
        };
        let untrusted = VerifiedRamValCheckAdviceContribution {
            kind: JoltAdviceKind::Untrusted,
            selector: Fr::from_u64(7),
            opening_claim: Fr::from_u64(11),
            opening_point: vec![Fr::from_u64(13), Fr::from_u64(17)],
        };
        let initial = RamValCheckInitialEvaluation {
            public_eval: Fr::from_u64(19),
            advice_contributions: vec![trusted.clone(), untrusted.clone()],
            full_eval: Fr::from_u64(23),
        };

        assert_eq!(
            initial.advice_contribution(JoltAdviceKind::Trusted),
            Some(&trusted)
        );
        assert_eq!(
            initial.advice_opening_claim(JoltAdviceKind::Untrusted),
            Some(untrusted.opening_claim)
        );
        assert_eq!(
            initial.advice_opening_point(JoltAdviceKind::Untrusted),
            Some(untrusted.opening_point.as_slice())
        );
    }
}
