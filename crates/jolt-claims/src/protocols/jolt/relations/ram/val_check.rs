//! RAM value-check symbolic sumcheck relation.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::ram::{
    ram_inc_val_check, ram_ra_val_check, ram_val, ram_val_final,
};
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId, RamValCheckChallenge,
    RamValCheckPublic, TraceDimensions,
};
use crate::SymbolicSumcheck;
use crate::{challenge, derived, opening, InputClaims, OutputClaims, SumcheckChallenges};

/// The full set of openings produced by the RAM value-check sumcheck: the staged
/// `Val_init` advice contributions (untrusted/trusted advice block evaluations,
/// each present only when its commitment is), the staged committed program-image
/// contribution (present only in committed program mode), and the main `ram_ra`
/// /`ram_inc` openings sharing one opening point. Generic over the cell. The
/// advice / program-image leaves are absent (`None`) in the ZK point-only form,
/// where BlindFold carries those openings.
///
/// WARNING: this struct's `canonical_order()` lists every leaf contiguously, but
/// the stage-4 Fiat-Shamir append order interleaves these openings around the
/// register openings (advice + program-image first, then registers, then
/// `ram_ra`/`ram_inc`). The stage-4 aggregate therefore hand-writes its
/// `opening_values` rather than concatenating each instance's openings; see
/// `Stage4OutputClaims` in `jolt-verifier`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamValCheck)]
pub struct RamValCheckOutputClaims<C> {
    #[opening(untrusted_advice)]
    pub untrusted_advice: Option<C>,
    #[opening(trusted_advice)]
    pub trusted_advice: Option<C>,
    #[opening(ProgramImageInitContributionRw)]
    pub program_image: Option<C>,
    #[opening(RamRa)]
    pub ram_ra: C,
    #[opening(committed = RamInc)]
    pub ram_inc: C,
}

/// Consumed openings of the RAM value-check claim: the read-write `val` (stage 2)
/// and output-check `val_final` (stage 2), reduced against `Val_init`, whose
/// committed pieces (advice / program image) are present only in some proof
/// configurations. Generic over the cell.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct RamValCheckInputClaims<C> {
    #[opening(RamVal, from = RamReadWriteChecking)]
    pub ram_val: C,
    #[opening(RamValFinal, from = RamOutputCheck)]
    pub ram_val_final: C,
    #[opening(untrusted_advice, from = RamValCheck)]
    pub untrusted_advice: Option<C>,
    #[opening(trusted_advice, from = RamValCheck)]
    pub trusted_advice: Option<C>,
    #[opening(ProgramImageInitContributionRw, from = RamValCheck)]
    pub program_image: Option<C>,
}

/// One committed contribution to the `Val_init(r_address)` decomposition: a
/// `Public` selector weighting a committed advice / program-image `opening`. The
/// selector *value* is supplied by the concrete side (`resolve_public`); the
/// symbolic shape carries only the `(selector_id, opening_id)` structure, keeping
/// the relation field-independent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RamValContribution {
    pub selector: RamValCheckPublic,
    pub opening: JoltOpeningId,
}

/// The RAM value-check shape: the trace dimensions plus the present `Val_init`
/// contributions, in the canonical order the BlindFold constraint also uses
/// (program image first, then advice). An empty `contributions` is the full-init
/// form (`Val_init` is wholly public).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamValCheckShape {
    pub dimensions: TraceDimensions,
    pub contributions: Vec<RamValContribution>,
}

/// Fiat-Shamir challenge drawn by the RAM value-check sumcheck.
#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
pub struct RamValCheckChallenges<F> {
    #[challenge(RamValCheckChallenge::Gamma)]
    pub gamma: F,
}

/// The RAM value-check sumcheck. The input reconstructs `Val_init(r_address)` as
/// `derived(InitEval) - Σ derived(InitSelector)·opening(advice)` and folds it
/// against the read-write `val` and output-check `val_final` by `gamma`; the
/// output is the degree-two `LtCyclePlusGamma·inc·ra`. The `Val_init` scalars are
/// `Public` symbols resolved by the verifier — value-preserving versus the prior
/// baked coefficients (BlindFold bakes `Public` factors as matrix coefficients),
/// so the relation stays field-independent. See `specs/symbolic-sumcheck.md` §4.1.
#[derive(Clone)]
pub struct RamValCheck {
    shape: RamValCheckShape,
}

impl SymbolicSumcheck for RamValCheck {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = RamValCheckShape;
    type Challenges<F> = RamValCheckChallenges<F>;
    type Inputs<C> = RamValCheckInputClaims<C>;
    type Outputs<C> = RamValCheckOutputClaims<C>;

    fn new(shape: RamValCheckShape) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RamValCheck
    }

    fn rounds(&self) -> usize {
        self.shape.dimensions.log_t()
    }

    fn degree(&self) -> usize {
        3
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(JoltChallengeId::from(RamValCheckChallenge::Gamma));
        let mut init = derived(JoltDerivedId::from(RamValCheckPublic::InitEval));
        for contribution in &self.shape.contributions {
            init = init
                - derived(JoltDerivedId::from(contribution.selector))
                    * opening(contribution.opening);
        }
        opening(ram_val()) + gamma.clone() * opening(ram_val_final())
            - (JoltExpr::one() + gamma) * init
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        derived(JoltDerivedId::from(RamValCheckPublic::LtCyclePlusGamma))
            * opening(ram_inc_val_check())
            * opening(ram_ra_val_check())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn trace_dimensions() -> TraceDimensions {
        TraceDimensions::new(5)
    }

    #[test]
    fn ram_val_check_symbolic_matches_dependencies() {
        let relation = RamValCheck::new(RamValCheckShape {
            dimensions: trace_dimensions(),
            contributions: vec![],
        });

        assert_eq!(RamValCheck::id(), JoltRelationId::RamValCheck);
        assert_eq!(relation.rounds(), trace_dimensions().log_t());
        assert_eq!(relation.degree(), 3);
    }

    /// The remodel's soundness anchor: the `Public`-symbol input expression must
    /// evaluate to the same value the pre-remodel baked-constant decomposition did
    /// (proven equal to the full-init formula in `geometry::ram`'s tests). With
    /// `InitEval = public_eval` and `InitSelector = neg_selector`, the
    /// `public·opening` term equals the old `constant·opening` term.
    #[test]
    fn ram_val_check_symbolic_evaluates_like_decomposed_init() {
        use crate::protocols::jolt::geometry::ram::val_check_advice_opening;
        use crate::protocols::jolt::JoltAdviceKind;

        let public_eval = Fr::from_u64(3);
        let untrusted_neg_selector = -Fr::from_u64(5);
        let trusted_neg_selector = -Fr::from_u64(7);

        let relation = RamValCheck::new(RamValCheckShape {
            dimensions: trace_dimensions(),
            contributions: vec![
                RamValContribution {
                    selector: RamValCheckPublic::InitSelector(JoltAdviceKind::Untrusted),
                    opening: val_check_advice_opening(JoltAdviceKind::Untrusted),
                },
                RamValContribution {
                    selector: RamValCheckPublic::InitSelector(JoltAdviceKind::Trusted),
                    opening: val_check_advice_opening(JoltAdviceKind::Trusted),
                },
            ],
        });

        let val_rw = Fr::from_u64(11);
        let val_final = Fr::from_u64(13);
        let gamma = Fr::from_u64(17);
        let untrusted_advice = Fr::from_u64(19);
        let trusted_advice = Fr::from_u64(23);
        let zero = Fr::from_u64(0);
        let init_eval = public_eval
            - untrusted_neg_selector * untrusted_advice
            - trusted_neg_selector * trusted_advice;

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_val() => val_rw,
                id if id == ram_val_final() => val_final,
                id if id == val_check_advice_opening(JoltAdviceKind::Untrusted) => untrusted_advice,
                id if id == val_check_advice_opening(JoltAdviceKind::Trusted) => trusted_advice,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamValCheck(RamValCheckChallenge::Gamma) => gamma,
                _ => zero,
            },
            |id| match *id {
                JoltDerivedId::RamValCheck(RamValCheckPublic::InitEval) => public_eval,
                JoltDerivedId::RamValCheck(RamValCheckPublic::InitSelector(
                    JoltAdviceKind::Untrusted,
                )) => untrusted_neg_selector,
                JoltDerivedId::RamValCheck(RamValCheckPublic::InitSelector(
                    JoltAdviceKind::Trusted,
                )) => trusted_neg_selector,
                _ => zero,
            },
        );

        assert_eq!(
            input,
            (val_rw - init_eval) + gamma * (val_final - init_eval)
        );
    }
}
