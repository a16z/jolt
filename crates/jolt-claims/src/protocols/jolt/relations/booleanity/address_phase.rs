//! The address-phase split of the booleanity symbolic sumcheck relation.

use core::marker::PhantomData;

use jolt_field::{Field, RingCore};
use serde::{Deserialize, Serialize};

use crate::opening;
use crate::protocols::jolt::geometry::booleanity::{
    booleanity_address_phase_opening, BooleanityDimensions,
};
use crate::protocols::jolt::{
    BooleanityChallenge, JoltChallengeId, JoltExpr, JoltOpeningId, JoltRelationId,
};
use crate::{ChallengeDrawError, InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck};

/// The staged `BooleanityAddrClaim` intermediate produced by the address phase
/// and consumed by the cycle phase.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(Booleanity)]
pub struct BooleanityAddressPhaseOutputClaims<C> {
    #[opening(BooleanityAddrClaim)]
    pub intermediate: C,
}

/// The address phase consumes no openings (its input claim is the constant zero).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BooleanityAddressPhaseInputClaims<C> {
    _cell: PhantomData<C>,
}

impl<C> Default for BooleanityAddressPhaseInputClaims<C> {
    fn default() -> Self {
        Self { _cell: PhantomData }
    }
}

impl<F: Field> InputClaims<F> for BooleanityAddressPhaseInputClaims<F> {
    fn canonical_order(&self) -> Vec<JoltOpeningId> {
        Vec::new()
    }

    fn resolve_input(&self, _id: &JoltOpeningId) -> Option<F> {
        None
    }
}

/// The hand pre-batch draws the booleanity subprotocol consumes: the
/// little-endian reference address (the reversed stage-5 instruction address,
/// padded with fresh draws or truncated to the committed chunk width), the
/// little-endian reference cycle (the reversed stage-5 instruction cycle, no
/// draw of its own), and the batching gamma. The stage-6a fronts perform the
/// draws between the bytecode member's `draw_challenges` and the batch (the
/// frozen wire schedule) and hand-assemble the stage challenge aggregate; the
/// address-phase kernel reads the values off `ProverInputs.challenges`.
///
/// The vector fields rule out the `SumcheckChallenges` derive, so the impl is
/// hand-written: the vectors are not challenge-id-resolvable (they never
/// appear as `Expr` leaves; the scalar gamma resolves as a derive would), and
/// the struct cannot be built from a per-field scalar stream —
/// `from_transcript_values` fails rather than fabricate reference points.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BooleanityAddressPhaseChallenges<F> {
    pub reference_address: Vec<F>,
    pub reference_cycle: Vec<F>,
    pub gamma: F,
}

impl<F: Field> SumcheckChallenges<F> for BooleanityAddressPhaseChallenges<F> {
    fn from_transcript_values<I: Iterator<Item = F>>(
        _values: I,
    ) -> Result<Self, ChallengeDrawError> {
        Err(ChallengeDrawError {
            required: 3,
            populated: 0,
        })
    }

    fn resolve_challenge(&self, id: &JoltChallengeId) -> Option<F> {
        (*id == JoltChallengeId::from(BooleanityChallenge::Gamma)).then_some(self.gamma)
    }
}

/// The address-phase split of the booleanity sumcheck: binds the address
/// variables and reduces to the intermediate `BooleanityAddrClaim` opening.
#[derive(Clone)]
pub struct BooleanityAddressPhase {
    shape: BooleanityDimensions,
}

impl SymbolicSumcheck for BooleanityAddressPhase {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = BooleanityDimensions;
    type Challenges<F> = BooleanityAddressPhaseChallenges<F>;
    type Inputs<C> = BooleanityAddressPhaseInputClaims<C>;
    type Outputs<C> = BooleanityAddressPhaseOutputClaims<C>;

    fn new(shape: BooleanityDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::Booleanity
    }

    fn rounds(&self) -> usize {
        self.shape.log_k_chunk
    }

    fn degree(&self) -> usize {
        3
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        JoltExpr::zero()
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(booleanity_address_phase_opening())
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;

    fn dimensions(instruction: usize, bytecode: usize, ram: usize) -> BooleanityDimensions {
        let layout = JoltRaPolynomialLayout::new(instruction, bytecode, ram).unwrap();
        BooleanityDimensions::new(layout, 5, 8)
    }

    #[test]
    fn booleanity_address_phase_symbolic_matches_dependencies() {
        let relation = BooleanityAddressPhase::new(dimensions(1, 1, 1));
        assert_eq!(BooleanityAddressPhase::id(), JoltRelationId::Booleanity);
        assert_eq!(relation.rounds(), 8);
        assert_eq!(relation.degree(), 3);
    }
}
