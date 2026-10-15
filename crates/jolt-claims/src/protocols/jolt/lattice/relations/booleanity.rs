//! Lattice-mode booleanity: the base booleanity sumcheck (same
//! `JoltRelationId::Booleanity`) extended so the one-hot increment
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
/// rely on the narrow set; see the range note in [`super::digit_zero`].
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
/// shared `booleanity_output` helper, so the two mode variants cannot
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
