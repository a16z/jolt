//! RAM Hamming-booleanity symbolic sumcheck relation.

use core::marker::PhantomData;

use jolt_field::{Field, RingCore};
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::ram::ram_hamming_weight;
use crate::protocols::jolt::{
    JoltExpr, JoltOpeningId, JoltRelationId, JoltSumcheckSpec, RamHammingBooleanityPublic,
    TraceDimensions,
};
use crate::SymbolicSumcheck;
use crate::{derived, opening, InputClaims, OutputClaims};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamHammingBooleanity)]
pub struct RamHammingBooleanityOutputClaims<C> {
    #[opening(RamHammingWeight)]
    pub ram_hamming_weight: C,
}

/// `RamHammingBooleanity` consumes no openings (its input claim is the constant
/// zero), so this carries only the cell marker. Hand-implements [`InputClaims`]
/// since the derive requires at least one `#[opening]` field.
pub struct RamHammingBooleanityInputClaims<C> {
    _cell: PhantomData<C>,
}

impl<C> Default for RamHammingBooleanityInputClaims<C> {
    fn default() -> Self {
        Self { _cell: PhantomData }
    }
}

impl<F: Field> InputClaims<F> for RamHammingBooleanityInputClaims<crate::OpeningClaim<F>> {
    fn canonical_order(&self) -> Vec<JoltOpeningId> {
        Vec::new()
    }

    fn resolve_input(&self, _id: &JoltOpeningId) -> Option<F> {
        None
    }
}

/// The RAM Hamming-booleanity sumcheck: a degree-three output enforcing that the
/// Hamming-weight opening is boolean (`h^2 - h == 0`) at each cycle, weighted by
/// the cycle-`eq` public; no input claim.
pub struct HammingBooleanity {
    shape: TraceDimensions,
}

impl SymbolicSumcheck for HammingBooleanity {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = TraceDimensions;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = RamHammingBooleanityInputClaims<C>;
    type Outputs<C> = RamHammingBooleanityOutputClaims<C>;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RamHammingBooleanity
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck(3)
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        JoltExpr::zero()
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let eq_cycle = derived(RamHammingBooleanityPublic::EqCycle);
        let h = opening(ram_hamming_weight());
        eq_cycle * (h.clone() * h.clone() - h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::JoltDerivedId;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn trace_dimensions() -> TraceDimensions {
        TraceDimensions::new(5)
    }

    #[test]
    fn hamming_booleanity_evaluates_like_core_formula() {
        let relation = HammingBooleanity::new(trace_dimensions());

        let h = Fr::from_u64(7);
        let eq_cycle = Fr::from_u64(11);
        let zero = Fr::from_u64(0);

        let input = relation
            .input_expression::<Fr>()
            .evaluate(|_| zero, |_| zero, |_| zero);
        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_hamming_weight() => h,
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                JoltDerivedId::RamHammingBooleanity(RamHammingBooleanityPublic::EqCycle) => {
                    eq_cycle
                }
                _ => zero,
            },
        );

        assert_eq!(input, zero);
        assert_eq!(output, eq_cycle * (h * h - h));
    }

    #[test]
    fn hamming_booleanity_symbolic_matches_dependencies() {
        let relation = HammingBooleanity::new(trace_dimensions());

        assert_eq!(
            HammingBooleanity::id(),
            JoltRelationId::RamHammingBooleanity
        );
        assert_eq!(relation.spec(), trace_dimensions().sumcheck(3));
        assert_eq!(
            relation.required_openings::<Fr>(),
            vec![ram_hamming_weight()]
        );
        assert!(relation.required_challenges::<Fr>().is_empty());
        assert_eq!(
            relation.required_deriveds::<Fr>(),
            vec![JoltDerivedId::from(RamHammingBooleanityPublic::EqCycle)]
        );
    }
}
