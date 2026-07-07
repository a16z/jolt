//! Unsigned-inc chunk reconstruction: one sumcheck over the chunk address
//! bits carrying three γ-batched legs per chunk polynomial:
//!
//! - **hamming**: `Σ_addr chunk_j(addr, r_cycle) = 1` — the exactly-one-hot
//!   row property (claimed sum `1`, no address weighting),
//! - **reduction**: reduces the chunk opening produced by the lattice
//!   booleanity sumcheck at `r_booleanity_address` to this relation's bound
//!   address point (`EqBooleanityAddress` derived),
//! - **reconstruction**: `Σ_j place_j · Σ_addr id(addr) · chunk_j(addr,
//!   r_cycle) = FusedInc + 2^64 · (1 − msb)` — the decoded chunks equal the
//!   low 64 bits of the unsigned fused increment (`IdentityAtAddress`
//!   derived). The `+2^64` unsigned shift is a constant, so it is free at any
//!   opening point and folded directly into this leg — there is no separate
//!   shift relation.
//!
//! All legs share the fixed cycle point `r_cycle` bound by the lattice
//! booleanity / inc-virtualization sumchecks, so the produced chunk openings
//! land on the shared `(r_address ‖ r_cycle)` packed point. jolt-verifier's
//! staging is responsible for arranging that point equality.
//!
//! Every leg is at most quadratic per bound variable (`eq · chunk` and
//! `id · chunk` are products of two multilinears; the hamming leg is linear),
//! hence `degree() == 2`.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::{
    JoltExpr, JoltOpeningId, JoltRelationId, UnsignedIncChunkReconstructionChallenge,
    UnsignedIncChunkReconstructionPublic,
};
use crate::{
    challenge, constant, derived, opening, InputClaims, OutputClaims, SumcheckChallenges,
    SymbolicSumcheck,
};

use super::super::geometry::{UnsignedIncChunking, UNSIGNED_INC_BITS};
use super::booleanity::{
    booleanity_unsigned_inc_chunk_opening, booleanity_unsigned_inc_msb_opening,
};
use super::inc_virtualization::fused_inc_opening;

/// The chunk openings at the final shared address point — the final claims
/// the packed opening consumes for the `UnsignedIncChunk` slots.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(UnsignedIncChunkReconstruction)]
pub struct ChunkReconstructionOutputClaims<C> {
    #[opening(committed = UnsignedIncChunk)]
    pub chunks: Vec<C>,
}

#[derive(Clone, Debug, InputClaims)]
pub struct ChunkReconstructionInputClaims<C> {
    #[opening(committed = UnsignedIncChunk, from = Booleanity)]
    pub chunks: Vec<C>,
    #[opening(committed = UnsignedIncMsb, from = Booleanity)]
    pub msb: C,
    #[opening(FusedInc, from = IncVirtualization)]
    pub fused_inc: C,
}

#[derive(Clone, Copy, Debug, SumcheckChallenges)]
pub struct ChunkReconstructionChallenges<F> {
    #[challenge(UnsignedIncChunkReconstructionChallenge::Gamma)]
    pub gamma: F,
}

pub struct ChunkReconstruction {
    shape: UnsignedIncChunking,
}

impl ChunkReconstruction {
    fn value_leg_scale<F: RingCore>(&self) -> JoltExpr<F> {
        challenge(UnsignedIncChunkReconstructionChallenge::Gamma).pow(2 * self.shape.chunk_count())
    }
}

impl SymbolicSumcheck for ChunkReconstruction {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = UnsignedIncChunking;
    type Challenges<F> = ChunkReconstructionChallenges<F>;
    type Inputs<C> = ChunkReconstructionInputClaims<C>;
    type Outputs<C> = ChunkReconstructionOutputClaims<C>;

    fn new(shape: UnsignedIncChunking) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::UnsignedIncChunkReconstruction
    }

    fn rounds(&self) -> usize {
        self.shape.chunk_width()
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(UnsignedIncChunkReconstructionChallenge::Gamma);

        let mut input = JoltExpr::zero();
        for index in 0..self.shape.chunk_count() {
            input = input
                + gamma.clone().pow(2 * index)
                + gamma.clone().pow(2 * index + 1)
                    * opening(booleanity_unsigned_inc_chunk_opening(index));
        }
        // The unsigned low bits: `FusedInc + 2^64 − 2^64·msb`.
        let lower_value = opening(fused_inc_opening())
            + constant(F::pow2(UNSIGNED_INC_BITS))
                * (JoltExpr::one() - opening(booleanity_unsigned_inc_msb_opening()));
        input + self.value_leg_scale() * lower_value
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(UnsignedIncChunkReconstructionChallenge::Gamma);
        let value_leg_scale = self.value_leg_scale();
        let eq_booleanity_address =
            derived(UnsignedIncChunkReconstructionPublic::EqBooleanityAddress);
        let identity_at_address = derived(UnsignedIncChunkReconstructionPublic::IdentityAtAddress);

        let mut output = JoltExpr::zero();
        for index in 0..self.shape.chunk_count() {
            let coefficient = gamma.clone().pow(2 * index)
                + gamma.clone().pow(2 * index + 1) * eq_booleanity_address.clone()
                + value_leg_scale.clone()
                    * constant(self.shape.place_value::<F>(index))
                    * identity_at_address.clone();
            output = output + coefficient * opening(reconstructed_chunk_opening(index));
        }
        output
    }
}

pub fn reconstructed_chunk_opening(index: usize) -> JoltOpeningId {
    JoltOpeningId::committed(
        crate::protocols::jolt::JoltCommittedPolynomial::UnsignedIncChunk(index),
        JoltRelationId::UnsignedIncChunkReconstruction,
    )
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::protocols::jolt::{JoltChallengeId, JoltDerivedId};
    use jolt_field::{Fr, FromPrimitiveInt, RingCore};

    fn chunking() -> UnsignedIncChunking {
        UnsignedIncChunking::new(16).unwrap()
    }

    fn pow(base: Fr, exponent: usize) -> Fr {
        (0..exponent).fold(Fr::from_u64(1), |acc, _| acc * base)
    }

    #[test]
    fn reconstruction_evaluates_like_core_formula() {
        let relation = ChunkReconstruction::new(chunking());
        let count = chunking().chunk_count();

        let gamma = Fr::from_u64(37);
        let fused_inc = Fr::from_u64(41);
        let msb = Fr::from_u64(43);
        let eq_addr = Fr::from_u64(47);
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
                    .or((*id == booleanity_unsigned_inc_msb_opening()).then_some(msb))
                    .unwrap_or(zero)
            },
            |_| gamma,
            |_| zero,
        );
        let delta = pow(gamma, 2 * count);
        let mut expected = delta * (fused_inc + Fr::pow2(64) * (Fr::from_u64(1) - msb));
        for index in 0..count {
            expected += pow(gamma, 2 * index) + pow(gamma, 2 * index + 1) * chunk_in(index);
        }
        assert_eq!(input, expected);

        let output = relation.output_expression::<Fr>().evaluate(
            |id| {
                (0..count)
                    .find(|&index| *id == reconstructed_chunk_opening(index))
                    .map_or(zero, chunk_out)
            },
            |_| gamma,
            |id| match *id {
                JoltDerivedId::UnsignedIncChunkReconstruction(
                    UnsignedIncChunkReconstructionPublic::EqBooleanityAddress,
                ) => eq_addr,
                JoltDerivedId::UnsignedIncChunkReconstruction(
                    UnsignedIncChunkReconstructionPublic::IdentityAtAddress,
                ) => id_addr,
                _ => zero,
            },
        );
        let mut expected = Fr::from_u64(0);
        for index in 0..count {
            let place = chunking().place_value::<Fr>(index);
            expected += (pow(gamma, 2 * index)
                + pow(gamma, 2 * index + 1) * eq_addr
                + delta * place * id_addr)
                * chunk_out(index);
        }
        assert_eq!(output, expected);
    }

    /// The shift fold: a signed increment `v` round-trips through the
    /// unsigned decomposition. For `v >= 0` the msb is hot and the chunks
    /// carry `v`; for `v < 0` the msb is cold and the chunks carry
    /// `2^64 - |v|`.
    #[test]
    fn shift_fold_matches_unsigned_decomposition() {
        let value: i128 = -41;
        let fused_inc = Fr::from_i128(value);
        let unsigned = (value + (1i128 << 64)) as u128;
        let msb = Fr::from_u128(unsigned >> 64);
        let lower = Fr::from_u128(unsigned & ((1u128 << 64) - 1));

        assert_eq!(
            fused_inc + Fr::pow2(64) * (Fr::from_u64(1) - msb),
            lower,
            "lower-bits identity must match the msb split"
        );
    }

    #[test]
    fn reconstruction_exposes_expected_dependencies() {
        let relation = ChunkReconstruction::new(chunking());
        let count = chunking().chunk_count();

        assert_eq!(
            ChunkReconstruction::id(),
            JoltRelationId::UnsignedIncChunkReconstruction
        );
        assert_eq!(relation.rounds(), 16);
        assert_eq!(relation.degree(), 2);

        let mut expected_inputs = (0..count)
            .map(booleanity_unsigned_inc_chunk_opening)
            .collect::<Vec<_>>();
        expected_inputs.push(fused_inc_opening());
        expected_inputs.push(booleanity_unsigned_inc_msb_opening());
        assert_eq!(
            relation.input_expression::<Fr>().required_openings(),
            expected_inputs
        );
        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            (0..count)
                .map(reconstructed_chunk_opening)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(
                UnsignedIncChunkReconstructionChallenge::Gamma
            )]
        );
        assert_eq!(
            relation.required_deriveds::<Fr>(),
            vec![
                JoltDerivedId::from(UnsignedIncChunkReconstructionPublic::EqBooleanityAddress),
                JoltDerivedId::from(UnsignedIncChunkReconstructionPublic::IdentityAtAddress),
            ]
        );
    }
}
