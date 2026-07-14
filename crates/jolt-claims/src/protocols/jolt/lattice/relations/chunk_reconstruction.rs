//! Unsigned-inc chunk reconstruction: one sumcheck over the chunk address
//! bits **and** the cycle bits, γ-batching four duties:
//!
//! - **hamming** (`γ^{2i}`, kernel `eq(r_inc, j)`): each chunk polynomial
//!   sums to exactly one hot address per cycle row (claimed sum `1`),
//! - **reduction** (`γ^{2i+1}`, kernel `eq(r_bool_addr, k)·eq(r_bool_cyc, j)`):
//!   reduces the chunk openings produced by the lattice booleanity sumcheck
//!   to this relation's bound point,
//! - **msb reduction** (`γ^{2N}`, same booleanity kernel with the
//!   address-constant msb polynomial): reduces the booleanity msb opening
//!   likewise,
//! - **shifted decode** (`γ^{2N+1}`, kernels `id(k)·eq(r_inc, j)` over the
//!   chunks and `2^64·eq(r_bool_addr, k)·eq(r_inc, j)` over the msb):
//!   `Σ_i place_i·chunk_i + 2^64·msb = 2^64 + FusedInc` — the shifted
//!   unsigned encoding, anchored at the `IncVirtualization` cycle point where
//!   the consumed `FusedInc` claim lives. Folding the msb into the *sum* side
//!   is what makes the legs consistent across the two independently bound
//!   cycle points (the booleanity batch and the pre-address-phase
//!   `IncVirtualization` sumcheck): the input side needs only the fused claim
//!   plus the constant `2^64`.
//!
//! The produced chunk **and msb** openings land at this relation's own bound
//! point — both are the packed final claims for their slots. Every leg is at
//! most quadratic per bound variable (`eq·chunk`, `id·chunk`, `eq·msb`),
//! hence `degree() == 2`.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::{
    JoltExpr, JoltOpeningId, JoltRelationId, TraceDimensions,
    UnsignedIncChunkReconstructionChallenge, UnsignedIncChunkReconstructionPublic,
};
use crate::{
    challenge, constant, derived, opening, InputClaims, OutputClaims, SumcheckChallenges,
    SymbolicSumcheck,
};

use super::super::geometry::{LatticeGeometryError, UnsignedIncChunking, UNSIGNED_INC_BITS};
use super::booleanity::{
    booleanity_unsigned_inc_chunk_opening, booleanity_unsigned_inc_msb_opening,
};
use super::inc_virtualization::fused_inc_opening;

/// The inc chunking plus the trace size: the reconstruction binds
/// `chunk_width + log_t` variables (`(address ‖ cycle)`, cycle low).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkReconstructionDimensions {
    pub chunking: UnsignedIncChunking,
    pub trace: TraceDimensions,
}

impl ChunkReconstructionDimensions {
    pub fn new(chunk_width: usize, trace: TraceDimensions) -> Result<Self, LatticeGeometryError> {
        Ok(Self {
            chunking: UnsignedIncChunking::new(chunk_width)?,
            trace,
        })
    }
}

/// The chunk and msb openings at the relation's bound point — the final
/// claims the packed opening consumes for the `UnsignedIncChunk` slots and
/// the `UnsignedIncMsb` slot (the msb, having no address variables, opens at
/// the cycle tail of the shared point).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(UnsignedIncChunkReconstruction)]
pub struct ChunkReconstructionOutputClaims<C> {
    #[opening(committed = UnsignedIncChunk)]
    pub chunks: Vec<C>,
    #[opening(committed = UnsignedIncMsb)]
    pub msb: C,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct ChunkReconstructionInputClaims<C> {
    #[opening(committed = UnsignedIncChunk, from = Booleanity)]
    pub chunks: Vec<C>,
    #[opening(committed = UnsignedIncMsb, from = Booleanity)]
    pub msb: C,
    #[opening(FusedInc, from = IncVirtualization)]
    pub fused_inc: C,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
pub struct ChunkReconstructionChallenges<F> {
    #[challenge(UnsignedIncChunkReconstructionChallenge::Gamma)]
    pub gamma: F,
}

pub struct ChunkReconstruction {
    shape: ChunkReconstructionDimensions,
}

impl ChunkReconstruction {
    fn msb_reduction_scale<F: RingCore>(&self) -> JoltExpr<F> {
        challenge(UnsignedIncChunkReconstructionChallenge::Gamma)
            .pow(2 * self.shape.chunking.chunk_count())
    }

    fn decode_scale<F: RingCore>(&self) -> JoltExpr<F> {
        challenge(UnsignedIncChunkReconstructionChallenge::Gamma)
            .pow(2 * self.shape.chunking.chunk_count() + 1)
    }
}

impl SymbolicSumcheck for ChunkReconstruction {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = ChunkReconstructionDimensions;
    type Challenges<F> = ChunkReconstructionChallenges<F>;
    type Inputs<C> = ChunkReconstructionInputClaims<C>;
    type Outputs<C> = ChunkReconstructionOutputClaims<C>;

    fn new(shape: ChunkReconstructionDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::UnsignedIncChunkReconstruction
    }

    fn rounds(&self) -> usize {
        self.shape.chunking.chunk_width() + self.shape.trace.log_t()
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(UnsignedIncChunkReconstructionChallenge::Gamma);

        let mut input = JoltExpr::zero();
        for index in 0..self.shape.chunking.chunk_count() {
            input = input
                + gamma.clone().pow(2 * index)
                + gamma.clone().pow(2 * index + 1)
                    * opening(booleanity_unsigned_inc_chunk_opening(index));
        }
        input = input + self.msb_reduction_scale() * opening(booleanity_unsigned_inc_msb_opening());
        // The shifted unsigned encoding: `2^64 + FusedInc`.
        input
            + self.decode_scale()
                * (opening(fused_inc_opening()) + constant(F::pow2(UNSIGNED_INC_BITS)))
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(UnsignedIncChunkReconstructionChallenge::Gamma);
        let decode_scale = self.decode_scale();
        let eq_booleanity = derived(UnsignedIncChunkReconstructionPublic::EqBooleanityAddress)
            * derived(UnsignedIncChunkReconstructionPublic::EqBooleanityCycle);
        let eq_inc_cycle = derived(UnsignedIncChunkReconstructionPublic::EqIncCycle);
        let identity_at_address = derived(UnsignedIncChunkReconstructionPublic::IdentityAtAddress);

        let mut output = JoltExpr::zero();
        for index in 0..self.shape.chunking.chunk_count() {
            let coefficient = gamma.clone().pow(2 * index) * eq_inc_cycle.clone()
                + gamma.clone().pow(2 * index + 1) * eq_booleanity.clone()
                + decode_scale.clone()
                    * constant(self.shape.chunking.place_value::<F>(index))
                    * identity_at_address.clone()
                    * eq_inc_cycle.clone();
            output = output + coefficient * opening(reconstructed_chunk_opening(index));
        }
        let msb_coefficient = self.msb_reduction_scale() * eq_booleanity
            + decode_scale
                * constant(F::pow2(UNSIGNED_INC_BITS))
                * derived(UnsignedIncChunkReconstructionPublic::EqBooleanityAddress)
                * eq_inc_cycle;
        output + msb_coefficient * opening(reconstructed_msb_opening())
    }
}

pub fn reconstructed_chunk_opening(index: usize) -> JoltOpeningId {
    JoltOpeningId::committed(
        crate::protocols::jolt::JoltCommittedPolynomial::UnsignedIncChunk(index),
        JoltRelationId::UnsignedIncChunkReconstruction,
    )
}

pub fn reconstructed_msb_opening() -> JoltOpeningId {
    JoltOpeningId::committed(
        crate::protocols::jolt::JoltCommittedPolynomial::UnsignedIncMsb,
        JoltRelationId::UnsignedIncChunkReconstruction,
    )
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::protocols::jolt::JoltDerivedId;
    use jolt_field::{Fr, FromPrimitiveInt, RingCore};

    fn dimensions() -> ChunkReconstructionDimensions {
        ChunkReconstructionDimensions::new(16, TraceDimensions::new(5)).unwrap()
    }

    fn pow(base: Fr, exponent: usize) -> Fr {
        (0..exponent).fold(Fr::from_u64(1), |acc, _| acc * base)
    }

    #[test]
    fn reconstruction_evaluates_like_core_formula() {
        let relation = ChunkReconstruction::new(dimensions());
        let count = dimensions().chunking.chunk_count();

        let gamma = Fr::from_u64(37);
        let fused_inc = Fr::from_u64(41);
        let msb_in = Fr::from_u64(43);
        let msb_out = Fr::from_u64(59);
        let eq_bool_addr = Fr::from_u64(47);
        let eq_bool_cycle = Fr::from_u64(61);
        let eq_inc_cycle = Fr::from_u64(67);
        let id_addr = Fr::from_u64(53);
        let zero = Fr::from_u64(0);
        let chunk_in = |index: usize| Fr::from_u64(3 + index as u64);
        let chunk_out = |index: usize| Fr::from_u64(101 + index as u64);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| {
                (0..count)
                    .find(|&index| *id == booleanity_unsigned_inc_chunk_opening(index))
                    .map(chunk_in)
                    .or((*id == fused_inc_opening()).then_some(fused_inc))
                    .or((*id == booleanity_unsigned_inc_msb_opening()).then_some(msb_in))
                    .unwrap_or(zero)
            },
            |_| gamma,
            |_| zero,
        );
        let msb_scale = pow(gamma, 2 * count);
        let decode_scale = pow(gamma, 2 * count + 1);
        let mut expected = msb_scale * msb_in + decode_scale * (fused_inc + Fr::pow2(64));
        for index in 0..count {
            expected += pow(gamma, 2 * index) + pow(gamma, 2 * index + 1) * chunk_in(index);
        }
        assert_eq!(input, expected);

        let output = relation.output_expression::<Fr>().evaluate(
            |id| {
                (0..count)
                    .find(|&index| *id == reconstructed_chunk_opening(index))
                    .map(chunk_out)
                    .or((*id == reconstructed_msb_opening()).then_some(msb_out))
                    .unwrap_or(zero)
            },
            |_| gamma,
            |id| match *id {
                JoltDerivedId::UnsignedIncChunkReconstruction(
                    UnsignedIncChunkReconstructionPublic::EqBooleanityAddress,
                ) => eq_bool_addr,
                JoltDerivedId::UnsignedIncChunkReconstruction(
                    UnsignedIncChunkReconstructionPublic::EqBooleanityCycle,
                ) => eq_bool_cycle,
                JoltDerivedId::UnsignedIncChunkReconstruction(
                    UnsignedIncChunkReconstructionPublic::EqIncCycle,
                ) => eq_inc_cycle,
                JoltDerivedId::UnsignedIncChunkReconstruction(
                    UnsignedIncChunkReconstructionPublic::IdentityAtAddress,
                ) => id_addr,
                _ => zero,
            },
        );
        let eq_bool = eq_bool_addr * eq_bool_cycle;
        let mut expected = Fr::from_u64(0);
        for index in 0..count {
            let place = dimensions().chunking.place_value::<Fr>(index);
            expected += (pow(gamma, 2 * index) * eq_inc_cycle
                + pow(gamma, 2 * index + 1) * eq_bool
                + decode_scale * place * id_addr * eq_inc_cycle)
                * chunk_out(index);
        }
        expected += (msb_scale * eq_bool
            + decode_scale * Fr::pow2(64) * eq_bool_addr * eq_inc_cycle)
            * msb_out;
        assert_eq!(output, expected);
    }

    /// The shifted decode: `Σ place·chunk + 2^64·msb` round-trips the signed
    /// value as `2^64 + v` — the msb folds into the sum side, so the input
    /// side needs only the fused claim plus the constant shift.
    #[test]
    fn shifted_decode_matches_unsigned_encoding() {
        let value: i128 = -41;
        let fused_inc = Fr::from_i128(value);
        let unsigned = (value + (1i128 << 64)) as u128;
        let msb = Fr::from_u128(unsigned >> 64);
        let lower = Fr::from_u128(unsigned & ((1u128 << 64) - 1));

        assert_eq!(
            lower + Fr::pow2(64) * msb,
            fused_inc + Fr::pow2(64),
            "shifted decode must equal the fused claim plus the constant shift"
        );
    }

    #[test]
    fn reconstruction_exposes_expected_dependencies() {
        let relation = ChunkReconstruction::new(dimensions());
        let count = dimensions().chunking.chunk_count();

        assert_eq!(
            ChunkReconstruction::id(),
            JoltRelationId::UnsignedIncChunkReconstruction
        );
        assert_eq!(relation.rounds(), 16 + 5);
        assert_eq!(relation.degree(), 2);

        // The msb final claim is produced here, alongside the chunk claims.
        let expected_outputs = (0..count)
            .map(reconstructed_chunk_opening)
            .chain(std::iter::once(reconstructed_msb_opening()))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(relation.expected_output_openings::<Fr>(), expected_outputs);
    }
}
