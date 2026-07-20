//! Lattice-mode Hamming-weight claim reduction, extended with the fused
//! increment's one-hot decomposition.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::hamming_weight::{
    booleanity_claim, hamming_weight_claim, reduced_claim,
};
use crate::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
use crate::protocols::jolt::relations::claim_reductions::hamming_weight::HammingWeightClaimReductionChallenges;
use crate::protocols::jolt::{
    HammingWeightClaimReductionChallenge, HammingWeightClaimReductionPublic, JoltExpr,
    JoltOpeningId, JoltRelationId,
};
use crate::{challenge, constant, derived, opening, InputClaims, OutputClaims, SymbolicSumcheck};

use crate::protocols::jolt::geometry::bytecode::fused_inc_read_raf_opening;

use super::super::geometry::{LatticeGeometryError, UnsignedIncChunking, UNSIGNED_INC_BITS};
use super::booleanity::{
    booleanity_unsigned_inc_chunk_opening, booleanity_unsigned_inc_msb_opening,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatticeHammingWeightClaimReductionDimensions {
    pub layout: JoltRaPolynomialLayout,
    pub log_k_chunk: usize,
    chunking: UnsignedIncChunking,
}

impl LatticeHammingWeightClaimReductionDimensions {
    pub fn new(
        layout: JoltRaPolynomialLayout,
        log_k_chunk: usize,
    ) -> Result<Self, LatticeGeometryError> {
        Ok(Self {
            layout,
            log_k_chunk,
            chunking: UnsignedIncChunking::new(log_k_chunk)?,
        })
    }

    pub fn chunking(self) -> UnsignedIncChunking {
        self.chunking
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct LatticeHammingWeightClaimReductionInputClaims<C> {
    #[opening(RamHammingWeight, from = RamHammingBooleanity)]
    pub ram_hamming_weight: C,
    #[opening(committed = InstructionRa, from = Booleanity)]
    pub instruction_booleanity: Vec<C>,
    #[opening(committed = BytecodeRa, from = Booleanity)]
    pub bytecode_booleanity: Vec<C>,
    #[opening(committed = RamRa, from = Booleanity)]
    pub ram_booleanity: Vec<C>,
    #[opening(committed = InstructionRa, from = InstructionRaVirtualization)]
    pub instruction_virtualization: Vec<C>,
    #[opening(committed = BytecodeRa, from = BytecodeReadRaf)]
    pub bytecode_virtualization: Vec<C>,
    #[opening(committed = RamRa, from = RamRaVirtualization)]
    pub ram_virtualization: Vec<C>,
    #[opening(committed = UnsignedIncChunk, from = Booleanity)]
    pub unsigned_inc_booleanity: Vec<C>,
    #[opening(committed = UnsignedIncMsb, from = Booleanity)]
    pub unsigned_inc_msb_booleanity: C,
    #[opening(FusedInc, from = BytecodeReadRaf)]
    pub fused_inc: C,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(HammingWeightClaimReduction)]
pub struct LatticeHammingWeightClaimReductionOutputClaims<C> {
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

pub struct LatticeHammingWeightClaimReduction {
    shape: LatticeHammingWeightClaimReductionDimensions,
}

impl LatticeHammingWeightClaimReduction {
    fn ra_terms(&self) -> usize {
        3 * self.shape.layout.total()
    }

    fn inc_column_count(&self) -> usize {
        self.shape.chunking.chunk_count() + 1
    }

    fn decode_power(&self) -> usize {
        self.ra_terms() + 2 * self.inc_column_count()
    }
}

impl SymbolicSumcheck for LatticeHammingWeightClaimReduction {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = LatticeHammingWeightClaimReductionDimensions;
    type Challenges<F> = HammingWeightClaimReductionChallenges<F>;
    type Inputs<C> = LatticeHammingWeightClaimReductionInputClaims<C>;
    type Outputs<C> = LatticeHammingWeightClaimReductionOutputClaims<C>;

    fn new(shape: Self::Shape) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::HammingWeightClaimReduction
    }

    fn rounds(&self) -> usize {
        self.shape.log_k_chunk
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(HammingWeightClaimReductionChallenge::Gamma);
        let mut input = JoltExpr::zero();
        for (i, polynomial) in self.shape.layout.polynomials().enumerate() {
            input = input
                + gamma.clone().pow(3 * i) * hamming_weight_claim(polynomial)
                + gamma.clone().pow(3 * i + 1) * opening(booleanity_claim(polynomial))
                + gamma.clone().pow(3 * i + 2)
                    * opening(crate::protocols::jolt::geometry::claim_reductions::hamming_weight::virtualization_claim(polynomial));
        }
        for index in 0..self.shape.chunking.chunk_count() {
            let offset = self.ra_terms() + 2 * index;
            input = input
                + gamma.clone().pow(offset)
                + gamma.clone().pow(offset + 1)
                    * opening(booleanity_unsigned_inc_chunk_opening(index));
        }
        let msb_offset = self.ra_terms() + 2 * self.shape.chunking.chunk_count();
        input = input
            + gamma.clone().pow(msb_offset)
            + gamma.clone().pow(msb_offset + 1) * opening(booleanity_unsigned_inc_msb_opening());
        input
            + gamma.pow(self.decode_power())
                * (opening(fused_inc_read_raf_opening()) + constant(F::pow2(UNSIGNED_INC_BITS)))
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(HammingWeightClaimReductionChallenge::Gamma);
        let eq_booleanity = derived(HammingWeightClaimReductionPublic::EqBooleanity);
        let identity = derived(HammingWeightClaimReductionPublic::IdentityAtAddress);
        let decode_scale = gamma.clone().pow(self.decode_power());
        let mut output = JoltExpr::zero();

        for (i, polynomial) in self.shape.layout.polynomials().enumerate() {
            let coefficient = gamma.clone().pow(3 * i)
                + gamma.clone().pow(3 * i + 1) * eq_booleanity.clone()
                + gamma.clone().pow(3 * i + 2)
                    * derived(HammingWeightClaimReductionPublic::EqVirtualization(i));
            output = output + coefficient * opening(reduced_claim(polynomial));
        }
        for index in 0..self.shape.chunking.chunk_count() {
            let offset = self.ra_terms() + 2 * index;
            let coefficient = gamma.clone().pow(offset)
                + gamma.clone().pow(offset + 1) * eq_booleanity.clone()
                + decode_scale.clone()
                    * constant(self.shape.chunking.place_value::<F>(index))
                    * identity.clone();
            output = output + coefficient * opening(reduced_unsigned_inc_chunk_opening(index));
        }
        let msb_offset = self.ra_terms() + 2 * self.shape.chunking.chunk_count();
        let msb_coefficient = gamma.clone().pow(msb_offset)
            + gamma.pow(msb_offset + 1) * eq_booleanity
            + decode_scale * constant(F::pow2(UNSIGNED_INC_BITS)) * identity;
        output + msb_coefficient * opening(reduced_unsigned_inc_msb_opening())
    }
}

pub fn reduced_unsigned_inc_chunk_opening(index: usize) -> JoltOpeningId {
    JoltOpeningId::committed(
        crate::protocols::jolt::JoltCommittedPolynomial::UnsignedIncChunk(index),
        JoltRelationId::HammingWeightClaimReduction,
    )
}

pub fn reduced_unsigned_inc_msb_opening() -> JoltOpeningId {
    JoltOpeningId::committed(
        crate::protocols::jolt::JoltCommittedPolynomial::UnsignedIncMsb,
        JoltRelationId::HammingWeightClaimReduction,
    )
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::ra::JoltRaPolynomial;
    use crate::protocols::jolt::geometry::ram::ram_hamming_weight;
    use crate::protocols::jolt::{
        HammingWeightClaimReductionChallenge, JoltChallengeId, JoltCommittedPolynomial,
        JoltDerivedId,
    };
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn fused_increment_terms_extend_the_ra_reduction() {
        let layout = JoltRaPolynomialLayout::new(0, 0, 1).unwrap();
        let relation = LatticeHammingWeightClaimReduction::new(
            LatticeHammingWeightClaimReductionDimensions::new(layout, 32).unwrap(),
        );
        let gamma = Fr::from_u64(3);
        let values = (2..=12).map(Fr::from_u64).collect::<Vec<_>>();
        let [hamming, bool_ra, virt_ra, bool_0, bool_1, bool_msb, fused, out_ra, out_0, out_1, out_msb] =
            values.as_slice()
        else {
            unreachable!()
        };
        let eq_bool = Fr::from_u64(13);
        let eq_virt = Fr::from_u64(17);
        let identity = Fr::from_u64(19);
        let power = |exponent: usize| {
            (0..exponent).fold(Fr::from_u64(1), |accumulator, _| accumulator * gamma)
        };
        let zero = Fr::from_u64(0);

        let opening_value = |id: &JoltOpeningId| {
            match *id {
            id if id == ram_hamming_weight() => *hamming,
            id if id == booleanity_claim(JoltRaPolynomial::Ram(0)) => *bool_ra,
            id if id
                == crate::protocols::jolt::geometry::claim_reductions::hamming_weight::virtualization_claim(
                    JoltRaPolynomial::Ram(0),
                ) => *virt_ra,
            id if id == booleanity_unsigned_inc_chunk_opening(0) => *bool_0,
            id if id == booleanity_unsigned_inc_chunk_opening(1) => *bool_1,
            id if id == booleanity_unsigned_inc_msb_opening() => *bool_msb,
            id if id == fused_inc_read_raf_opening() => *fused,
            id if id == reduced_claim(JoltRaPolynomial::Ram(0)) => *out_ra,
            id if id == reduced_unsigned_inc_chunk_opening(0) => *out_0,
            id if id == reduced_unsigned_inc_chunk_opening(1) => *out_1,
            id if id == reduced_unsigned_inc_msb_opening() => *out_msb,
            _ => zero,
        }
        };
        let challenge_value = |id: &JoltChallengeId| match *id {
            JoltChallengeId::HammingWeightClaimReduction(
                HammingWeightClaimReductionChallenge::Gamma,
            ) => gamma,
            _ => zero,
        };
        let derived_value = |id: &JoltDerivedId| match *id {
            JoltDerivedId::HammingWeightClaimReduction(
                HammingWeightClaimReductionPublic::EqBooleanity,
            ) => eq_bool,
            JoltDerivedId::HammingWeightClaimReduction(
                HammingWeightClaimReductionPublic::EqVirtualization(0),
            ) => eq_virt,
            JoltDerivedId::HammingWeightClaimReduction(
                HammingWeightClaimReductionPublic::IdentityAtAddress,
            ) => identity,
            _ => zero,
        };

        let input = relation.input_expression::<Fr>().evaluate(
            opening_value,
            challenge_value,
            derived_value,
        );
        let expected_input = *hamming
            + power(1) * *bool_ra
            + power(2) * *virt_ra
            + power(3)
            + power(4) * *bool_0
            + power(5)
            + power(6) * *bool_1
            + power(7)
            + power(8) * *bool_msb
            + power(9) * (*fused + Fr::pow2(64));
        assert_eq!(input, expected_input);

        let output = relation.output_expression::<Fr>().evaluate(
            opening_value,
            challenge_value,
            derived_value,
        );
        let expected_output = *out_ra * (power(0) + power(1) * eq_bool + power(2) * eq_virt)
            + *out_0 * (power(3) + power(4) * eq_bool + power(9) * identity)
            + *out_1 * (power(5) + power(6) * eq_bool + power(9) * Fr::pow2(32) * identity)
            + *out_msb * (power(7) + power(8) * eq_bool + power(9) * Fr::pow2(64) * identity);
        assert_eq!(output, expected_output);

        assert_eq!(
            reduced_unsigned_inc_msb_opening(),
            JoltOpeningId::committed(
                JoltCommittedPolynomial::UnsignedIncMsb,
                JoltRelationId::HammingWeightClaimReduction,
            )
        );
    }
}
