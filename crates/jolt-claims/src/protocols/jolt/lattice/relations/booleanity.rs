//! Lattice-mode booleanity: the base booleanity sumcheck (same
//! `JoltRelationId::Booleanity`) extended so the packed one-hot inc
//! polynomials are covered by the same boolean check as the `Ra` families. Precedent for
//! sharing a relation id across mode variants: the full/committed bytecode
//! read-raf pair.

use jolt_field::RingCore;
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

use super::super::geometry::{LatticeGeometryError, UnsignedIncChunking};

/// The base booleanity dimensions plus the inc chunking they imply: the chunk
/// width equals `log_k_chunk` by the shared-final-point invariant, so it is
/// derived rather than supplied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatticeBooleanityDimensions {
    pub base: BooleanityDimensions,
    chunking: UnsignedIncChunking,
}

impl LatticeBooleanityDimensions {
    pub fn new(base: BooleanityDimensions) -> Result<Self, LatticeGeometryError> {
        Ok(Self {
            base,
            chunking: UnsignedIncChunking::new(base.log_k_chunk)?,
        })
    }

    pub fn chunking(self) -> UnsignedIncChunking {
        self.chunking
    }
}

/// Every boolean-checked opening at the booleanity point: the base `Ra`
/// families, unsigned-inc chunks, and increment MSB at the same full
/// `(r_address || r_cycle)` point. The MSB column is a strict one-hot column
/// whose hot address is zero or one.
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
    #[opening(committed = UnsignedIncChunk)]
    pub unsigned_inc_chunks: Vec<C>,
    #[opening(committed = UnsignedIncMsb)]
    pub unsigned_inc_msb: C,
}

/// The base booleanity fold extended past the `Ra` families with the
/// unsigned-inc chunk polynomials and the msb; the formula itself is the
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

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        JoltExpr::zero()
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        booleanity_output(lattice_booleanity_output_openings(self.shape))
    }
}

/// The cycle-phase split of the lattice booleanity sumcheck, mirroring the
/// base `BooleanityCyclePhase`: same `BooleanityAddrClaim` intermediate input
/// (the address phase is column-agnostic, so the base `BooleanityAddressPhase`
/// serves both modes), with the output fold extended over the unsigned-inc
/// chunk and msb polynomials.
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

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(booleanity_address_phase_opening())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        booleanity_output(lattice_booleanity_output_openings(self.shape))
    }
}

/// The boolean-checked openings in canonical order: base `Ra` families, then
/// the chunk polynomials, then the msb.
pub fn lattice_booleanity_output_openings(
    dimensions: LatticeBooleanityDimensions,
) -> Vec<JoltOpeningId> {
    let mut openings = booleanity_output_openings(dimensions.base.layout);
    openings.extend(
        (0..dimensions.chunking().chunk_count()).map(booleanity_unsigned_inc_chunk_opening),
    );
    openings.push(booleanity_unsigned_inc_msb_opening());
    openings
}

pub fn booleanity_unsigned_inc_chunk_opening(index: usize) -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::UnsignedIncChunk(index),
        JoltRelationId::Booleanity,
    )
}

pub fn booleanity_unsigned_inc_msb_opening() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::UnsignedIncMsb,
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
    use jolt_field::{Fr, FromPrimitiveInt};

    fn dimensions() -> LatticeBooleanityDimensions {
        let layout = JoltRaPolynomialLayout::new(1, 0, 0).unwrap();
        LatticeBooleanityDimensions::new(BooleanityDimensions::new(layout, 5, 32)).unwrap()
    }

    #[test]
    fn chunking_is_derived_from_the_shared_chunk_size() {
        assert_eq!(
            dimensions().chunking(),
            UnsignedIncChunking::new(32).unwrap()
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
        let msb = Fr::from_u64(11);
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
                id if id == booleanity_unsigned_inc_chunk_opening(0) => chunk_0,
                id if id == booleanity_unsigned_inc_chunk_opening(1) => chunk_1,
                id if id == booleanity_unsigned_inc_msb_opening() => msb,
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
                    + gamma_6 * square(msb))
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
                booleanity_unsigned_inc_chunk_opening(0),
                booleanity_unsigned_inc_chunk_opening(1),
                booleanity_unsigned_inc_msb_opening(),
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
