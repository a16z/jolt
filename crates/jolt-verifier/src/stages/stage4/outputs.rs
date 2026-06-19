//! Typed outputs produced by stage 4 verification.

use jolt_claims::protocols::jolt::{
    formulas::ram::{RamValCheckInit, RamValCheckInitContribution},
    JoltAdviceKind,
};
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
    pub program_image_contribution: Option<VerifiedRamValCheckProgramImageContribution<F>>,
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

    /// The formula-side init decomposition: the public initial-RAM evaluation plus
    /// the present advice / program-image contributions (with negated selectors),
    /// in the canonical order the BlindFold constraint also uses — program image
    /// first, then advice in `advice_contributions` order. Shared by the verifier
    /// and the prover when building the `RamValCheck` relation, so the
    /// decomposition cannot drift between them.
    pub fn decomposition(&self) -> RamValCheckInit<F> {
        let mut contributions = Vec::new();
        if self.program_image_contribution.is_some() {
            contributions.push(RamValCheckInitContribution::program_image(-F::one()));
        }
        for contribution in &self.advice_contributions {
            let neg_selector = -contribution.selector;
            contributions.push(match contribution.kind {
                JoltAdviceKind::Trusted => RamValCheckInitContribution::trusted(neg_selector),
                JoltAdviceKind::Untrusted => RamValCheckInitContribution::untrusted(neg_selector),
            });
        }
        RamValCheckInit::decomposed(self.public_eval, contributions)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedRamValCheckAdviceContribution<F: Field> {
    pub kind: JoltAdviceKind,
    pub selector: F,
    pub opening_claim: F,
    pub opening_point: Vec<F>,
}

/// Staged program-image contribution to `Val_init(r_address)` in committed
/// program mode: the scalar opening claim and the full RAM address point it
/// was staged at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedRamValCheckProgramImageContribution<F: Field> {
    pub opening_claim: F,
    pub opening_point: Vec<F>,
}
