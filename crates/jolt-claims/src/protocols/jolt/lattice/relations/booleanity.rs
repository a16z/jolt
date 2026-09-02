//! Lattice-mode booleanity: the base booleanity sumcheck (same
//! `JoltRelationId::Booleanity`) extended so the packed one-hot inc
//! polynomials are covered by the same boolean check as the `Ra` families. Precedent for
//! sharing a relation id across mode variants: the full/committed bytecode
//! read-raf pair.

use jolt_field::Ring;
use serde::{Deserialize, Serialize};

use crate::opening;
use crate::protocols::jolt::geometry::booleanity::{
    booleanity_address_phase_opening, booleanity_output, booleanity_output_openings,
    BooleanityDimensions,
};
use crate::protocols::jolt::relations::booleanity::{
    BooleanityChallenges, BooleanityCyclePhaseChallenges, BooleanityInputClaims,
};
use crate::protocols::jolt::{JoltCommittedPolynomial, JoltExpr, JoltOpeningId, JoltRelationId};
use crate::{OutputClaims, SymbolicSumcheck};

use super::super::geometry::{BalancedIncChunking, LatticeGeometryError};

/// The base booleanity dimensions plus the inc chunking they imply: the chunk
/// width equals `log_k_chunk` by the shared-final-point invariant, so it is
/// derived rather than supplied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatticeBooleanityDimensions {
    pub base: BooleanityDimensions,
    chunking: BalancedIncChunking,
}

impl LatticeBooleanityDimensions {
    pub fn new(base: BooleanityDimensions) -> Result<Self, LatticeGeometryError> {
        Ok(Self {
            base,
            chunking: BalancedIncChunking::new(base.log_k_chunk)?,
        })
    }

    pub fn chunking(self) -> BalancedIncChunking {
        self.chunking
    }
}

/// Every boolean-checked opening at the booleanity point: the base `Ra`
/// families, increment digits, and the increment carry at the same full
/// `(r_address || r_cycle)` point. The carry column is a strict one-hot column
/// over the same `K` rows as the digits, decoding to the signed carry above
/// bit 63. WARNING: the honest encoder only ever uses rows `0`, `1`, and
/// `K - 1` (value `-1`), but nothing enforces that — booleanity plus the
/// column sum pin the carry only to the full alphabet `[-K/2, K/2)`. Do not
/// rely on the narrow set; see the range note in [`super::hamming_weight`].
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(Booleanity)]
pub struct LatticeBooleanityOutputClaims<C> {
    #[opening(committed = InstructionRa)]
    pub instruction_ra: Vec<C>,
    #[opening(committed = BytecodeRa)]
    pub bytecode_ra: Vec<C>,
    #[opening(committed = RamRa)]
    pub ram_ra: Vec<C>,
    #[opening(committed = BalancedIncDigit)]
    pub balanced_inc_digits: Vec<C>,
    #[opening(committed = BalancedIncCarry)]
    pub balanced_inc_carry: C,
}

/// The base booleanity fold extended past the `Ra` families with the
/// increment digit polynomials and the carry; the formula itself is the
/// shared [`booleanity_output`] helper, so the two mode variants cannot
/// diverge.
pub struct LatticeBooleanity {
    shape: LatticeBooleanityDimensions,
}

impl SymbolicSumcheck for LatticeBooleanity {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = LatticeBooleanityDimensions;
    type Challenges<F> = BooleanityChallenges<F>;
    type Inputs<C> = crate::NoInputs<C>;
    type Outputs<C> = LatticeBooleanityOutputClaims<C>;

    fn new(shape: LatticeBooleanityDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::Booleanity
    }

    fn rounds(&self) -> usize {
        self.shape.base.sumcheck_rounds()
    }

    fn degree(&self) -> usize {
        3
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        JoltExpr::zero()
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        booleanity_output(lattice_booleanity_output_openings(self.shape))
    }
}

/// The cycle-phase split of the lattice booleanity sumcheck, mirroring the
/// base `BooleanityCyclePhase`: same `BooleanityAddrClaim` intermediate input
/// (the address phase is column-agnostic, so the base `BooleanityAddressPhase`
/// serves both modes), with the output fold extended over the increment
/// digit and carry polynomials.
#[derive(Clone)]
pub struct LatticeBooleanityCyclePhase {
    shape: LatticeBooleanityDimensions,
}

impl SymbolicSumcheck for LatticeBooleanityCyclePhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = LatticeBooleanityDimensions;
    type Challenges<F> = BooleanityCyclePhaseChallenges<F>;
    type Inputs<C> = BooleanityInputClaims<C>;
    type Outputs<C> = LatticeBooleanityOutputClaims<C>;

    fn new(shape: LatticeBooleanityDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::Booleanity
    }

    fn rounds(&self) -> usize {
        self.shape.base.log_t
    }

    fn degree(&self) -> usize {
        3
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        opening(booleanity_address_phase_opening())
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        booleanity_output(lattice_booleanity_output_openings(self.shape))
    }
}

/// The boolean-checked openings in canonical order: base `Ra` families, then
/// the digit polynomials, then the carry.
pub fn lattice_booleanity_output_openings(
    dimensions: LatticeBooleanityDimensions,
) -> Vec<JoltOpeningId> {
    let mut openings = booleanity_output_openings(dimensions.base.layout);
    openings.extend(
        (0..dimensions.chunking().chunk_count()).map(booleanity_balanced_inc_digit_opening),
    );
    openings.push(booleanity_balanced_inc_carry_opening());
    openings
}

pub fn booleanity_balanced_inc_digit_opening(index: usize) -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::BalancedIncDigit(index),
        JoltRelationId::Booleanity,
    )
}

pub fn booleanity_balanced_inc_carry_opening() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::BalancedIncCarry,
        JoltRelationId::Booleanity,
    )
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
    use crate::protocols::jolt::{
        BooleanityChallenge, BooleanityPublic, JoltChallengeId, JoltDerivedId,
    };
    use jolt_field::{Fr, Ring};

    fn dimensions() -> LatticeBooleanityDimensions {
        let layout = JoltRaPolynomialLayout::new(1, 0, 0).unwrap();
        LatticeBooleanityDimensions::new(BooleanityDimensions::new(layout, 5, 32)).unwrap()
    }

    #[test]
    fn chunking_is_derived_from_the_shared_chunk_size() {
        assert_eq!(
            dimensions().chunking(),
            BalancedIncChunking::new(32).unwrap()
        );
        let layout = JoltRaPolynomialLayout::new(1, 0, 0).unwrap();
        assert_eq!(
            LatticeBooleanityDimensions::new(BooleanityDimensions::new(layout, 5, 7)),
            Err(LatticeGeometryError::ChunkWidthMisaligned { chunk_width: 7 })
        );
    }

    #[test]
    fn lattice_booleanity_extends_base_output_with_inc_polynomials() {
        let relation = LatticeBooleanity::new(dimensions());

        let instruction_ra = Fr::from_u64(3);
        let chunk_0 = Fr::from_u64(5);
        let chunk_1 = Fr::from_u64(7);
        let carry = Fr::from_u64(11);
        let gamma = Fr::from_u64(13);
        let eq_address_cycle = Fr::from_u64(17);
        let zero = Fr::from_u64(0);

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id
                    == JoltOpeningId::committed(
                        JoltCommittedPolynomial::InstructionRa(0),
                        JoltRelationId::Booleanity,
                    ) =>
                {
                    instruction_ra
                }
                id if id == booleanity_balanced_inc_digit_opening(0) => chunk_0,
                id if id == booleanity_balanced_inc_digit_opening(1) => chunk_1,
                id if id == booleanity_balanced_inc_carry_opening() => carry,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::Booleanity(BooleanityChallenge::Gamma) => gamma,
                _ => zero,
            },
            |id| match *id {
                JoltDerivedId::Booleanity(BooleanityPublic::EqAddressCycle) => eq_address_cycle,
                _ => zero,
            },
        );

        let square = |x: Fr| x * x - x;
        let gamma_2 = gamma * gamma;
        let gamma_4 = gamma_2 * gamma_2;
        let gamma_6 = gamma_4 * gamma_2;
        assert_eq!(
            output,
            eq_address_cycle
                * (square(instruction_ra)
                    + gamma_2 * square(chunk_0)
                    + gamma_4 * square(chunk_1)
                    + gamma_6 * square(carry))
        );
    }

    #[test]
    fn lattice_booleanity_exposes_expected_dependencies() {
        let relation = LatticeBooleanity::new(dimensions());

        assert_eq!(LatticeBooleanity::id(), JoltRelationId::Booleanity);
        assert_eq!(relation.rounds(), 5 + 32);
        assert_eq!(relation.degree(), 3);
        // The output expression covers exactly the pub opening-order helper's ids
        // (the helper is the wiring-side order; the set comparison guards drift).
        assert_eq!(
            relation.expected_output_openings::<Fr>(),
            lattice_booleanity_output_openings(dimensions())
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>()
        );
        assert_eq!(
            lattice_booleanity_output_openings(dimensions()),
            vec![
                JoltOpeningId::committed(
                    JoltCommittedPolynomial::InstructionRa(0),
                    JoltRelationId::Booleanity,
                ),
                booleanity_balanced_inc_digit_opening(0),
                booleanity_balanced_inc_digit_opening(1),
                booleanity_balanced_inc_carry_opening(),
            ]
        );
    }

    /// The cycle phase consumes the address-phase intermediate and produces
    /// the same extended opening set as the monolith (whose fold formula the
    /// evaluate test above pins — both variants share `booleanity_output`).
    #[test]
    fn lattice_cycle_phase_matches_monolith_dependencies() {
        let relation = LatticeBooleanityCyclePhase::new(dimensions());
        assert_eq!(
            LatticeBooleanityCyclePhase::id(),
            JoltRelationId::Booleanity
        );
        assert_eq!(relation.rounds(), 5);
        assert_eq!(relation.degree(), 3);

        let address_claim = Fr::from_u64(7);
        let zero = Fr::from_u64(0);
        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == booleanity_address_phase_opening() => address_claim,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );
        assert_eq!(input, address_claim);

        assert_eq!(
            relation.expected_output_openings::<Fr>(),
            lattice_booleanity_output_openings(dimensions())
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>()
        );
    }
}
