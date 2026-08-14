//! RAM activation-booleanity symbolic sumcheck relation (packed path).
//!
//! Digit-zero virtualization derives the RAM activation as
//! `M_RAM = Load + Store` (`specs/digit-zero-virtualization.md`); this
//! relation binds the two flag columns at the stage-6b cycle point and proves
//! the activation sum is Boolean, producing the `OpFlags(Load)`/
//! `OpFlags(Store)` openings the stage-7 digit-zero baselines consume.
//!
//! WARNING: the check is deliberately a single booleanity on the *sum*, not a
//! γ-batch of per-flag legs. The columns bound here are virtual — never
//! committed — so a prover chooses them after every challenge is drawn, and a
//! γ-combination of independent legs has non-Boolean solutions for any fixed
//! γ (`L² − L = c`, `S² − S = −c/γ`). `B² = B` for `B := Load + Store` has
//! only Boolean roots pointwise regardless of when the columns are chosen.
//! Only the sum flows into the reconstruction; the split between the two
//! openings is deliberately unconstrained.

use core::marker::PhantomData;

use jolt_field::{Field, RingCore};
use jolt_riscv::CircuitFlags;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::ram::{ram_activation_load, ram_activation_store};
use crate::protocols::jolt::{
    JoltExpr, JoltOpeningId, JoltRelationId, RamActivationBooleanityPublic, TraceDimensions,
};
use crate::SymbolicSumcheck;
use crate::{derived, opening, InputClaims, OutputClaims};

#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamActivationBooleanity)]
pub struct RamActivationBooleanityOutputClaims<C> {
    #[opening(OpFlags(CircuitFlags::Load))]
    pub load: C,
    #[opening(OpFlags(CircuitFlags::Store))]
    pub store: C,
}

/// `RamActivationBooleanity` consumes no openings (its input claim is the
/// constant zero), so this carries only the cell marker. Hand-implements
/// [`InputClaims`] since the derive requires at least one `#[opening]` field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamActivationBooleanityInputClaims<C> {
    _cell: PhantomData<C>,
}

impl<C> Default for RamActivationBooleanityInputClaims<C> {
    fn default() -> Self {
        Self { _cell: PhantomData }
    }
}

impl<F: Field> InputClaims<F> for RamActivationBooleanityInputClaims<F> {
    fn canonical_order(&self) -> Vec<JoltOpeningId> {
        Vec::new()
    }

    fn resolve_input(&self, _id: &JoltOpeningId) -> Option<F> {
        None
    }
}

/// The RAM activation-booleanity sumcheck: a degree-three output enforcing
/// that the activation sum is Boolean (`(Load + Store)² − (Load + Store) == 0`)
/// at each cycle, weighted by the cycle-`eq` public; no input claim.
#[derive(Clone)]
pub struct ActivationBooleanity {
    shape: TraceDimensions,
}

impl SymbolicSumcheck for ActivationBooleanity {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = TraceDimensions;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = RamActivationBooleanityInputClaims<C>;
    type Outputs<C> = RamActivationBooleanityOutputClaims<C>;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RamActivationBooleanity
    }

    fn rounds(&self) -> usize {
        self.shape.log_t()
    }

    fn degree(&self) -> usize {
        3
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        JoltExpr::zero()
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let eq_cycle = derived(RamActivationBooleanityPublic::EqCycle);
        let activation = opening(ram_activation_load()) + opening(ram_activation_store());
        eq_cycle * (activation.clone() * activation.clone() - activation)
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
    fn activation_booleanity_evaluates_like_core_formula() {
        let relation = ActivationBooleanity::new(trace_dimensions());

        let load = Fr::from_u64(7);
        let store = Fr::from_u64(29);
        let eq_cycle = Fr::from_u64(11);
        let zero = Fr::from_u64(0);

        let input = relation
            .input_expression::<Fr>()
            .evaluate(|_| zero, |_| zero, |_| zero);
        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_activation_load() => load,
                id if id == ram_activation_store() => store,
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                JoltDerivedId::RamActivationBooleanity(RamActivationBooleanityPublic::EqCycle) => {
                    eq_cycle
                }
                _ => zero,
            },
        );

        let sum = load + store;
        assert_eq!(input, zero);
        assert_eq!(output, eq_cycle * (sum * sum - sum));
    }

    #[test]
    fn activation_booleanity_symbolic_matches_dependencies() {
        let relation = ActivationBooleanity::new(trace_dimensions());

        assert_eq!(
            ActivationBooleanity::id(),
            JoltRelationId::RamActivationBooleanity
        );
        assert_eq!(relation.rounds(), trace_dimensions().log_t());
        assert_eq!(relation.degree(), 3);
    }
}
