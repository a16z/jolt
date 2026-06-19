//! Typed outputs produced by stage 4 verification.

use jolt_claims::protocols::jolt::{
    formulas::ram::{RamValCheckInit, RamValCheckInitContribution},
    JoltAdviceKind,
};
use jolt_field::Field;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::relations::OpeningClaim;
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::inputs::Stage4OutputClaims;

/// The Fiat-Shamir challenges the verifier draws during stage 4: the two
/// per-relation batching gammas. (The batch's own sumcheck point and batching
/// coefficients are stage-local verification artifacts and are not propagated to
/// later stages.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4Challenges<F: Field> {
    pub registers_gamma: F,
    pub ram_val_check_gamma: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4ClearOutput<F: Field> {
    pub challenges: Stage4Challenges<F>,
    /// The produced stage-4 openings paired with their points (point + value)
    /// via the `OpeningClaim` cell. The opening points are derived from the
    /// batch's sumcheck point; pairing them with the values here lets later stages
    /// consume a ready `OpeningClaim` instead of re-joining a value with a
    /// separately-tracked point.
    pub output_claims: Stage4OutputClaims<OpeningClaim<F>>,
    pub ram_val_check_init: RamValCheckInitialEvaluation<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4ZkOutput<F: Field, C> {
    pub challenges: Stage4Challenges<F>,
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
pub struct RamValCheckInitialEvaluation<F: Field> {
    pub public_eval: F,
    /// The staged program-image contribution to `Val_init(r_address)` (committed
    /// program mode only): the opening claim with the full RAM address point.
    pub program_image_contribution: Option<OpeningClaim<F>>,
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
            .map(|contribution| contribution.opening.value)
    }

    pub fn advice_opening_point(&self, kind: JoltAdviceKind) -> Option<&[F]> {
        self.advice_contribution(kind)
            .map(|contribution| contribution.opening.point.as_slice())
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
    /// The advice block opening (claim value + the address sub-point it was
    /// evaluated at) that this contribution weights by `selector`.
    pub opening: OpeningClaim<F>,
}
