//! Lattice-mode Hamming-weight claim reduction, extended with the fused
//! increment's one-hot decomposition.
//!
//! WARNING: the decode leg is not a range check on `FusedInc`. One-hotness pins
//! each digit and the carry only to `[-K/2, K/2)`, so the reachable set is the
//! `K · 2^64` integers the balanced numeral spans (~`±2^71` at `K = 256`), not
//! the honest `|delta| < 2^64`. That is safe because the encoding is injective,
//! fp128 leaves 56 bits of headroom over `2^71`, and base mode commits `Inc`
//! with no range relation at all — see "Increment range" in
//! `specs/lattice-claims.md`. Do not treat this reduction as bounding `Inc`.

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

use super::super::geometry::{BalancedIncChunking, LatticeGeometryError, FUSED_INC_BITS};
use super::booleanity::{
    booleanity_balanced_inc_carry_opening, booleanity_balanced_inc_digit_opening,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatticeHammingWeightClaimReductionDimensions {
    pub layout: JoltRaPolynomialLayout,
    pub log_k_chunk: usize,
    chunking: BalancedIncChunking,
}

impl LatticeHammingWeightClaimReductionDimensions {
    pub fn new(
        layout: JoltRaPolynomialLayout,
        log_k_chunk: usize,
    ) -> Result<Self, LatticeGeometryError> {
        Ok(Self {
            layout,
            log_k_chunk,
            chunking: BalancedIncChunking::new(log_k_chunk)?,
        })
    }

    pub fn chunking(self) -> BalancedIncChunking {
        self.chunking
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct LatticeHammingWeightClaimReductionInputClaims<C> {
    /// The RAM activation `A` — the reconstruction coefficient of the omitted
    /// lane zero, `L(k,t) = Q(k,t) + eq(k,0)·(A(t) − S(t))`. Each `RamRa` leg's
    /// baseline `w(0)·A` is folded into the input claim, so this claim enters
    /// with a negative `eq(0, ·)`-weighted coefficient rather than through a
    /// standalone hamming term (which would be vacuous: `Σ_k L(k,t) = A(t)`
    /// holds by construction).
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
    #[opening(committed = BalancedIncDigit, from = Booleanity)]
    pub balanced_inc_digit_booleanity: Vec<C>,
    #[opening(committed = BalancedIncCarry, from = Booleanity)]
    pub balanced_inc_carry_booleanity: C,
    #[opening(FusedInc, from = BytecodeReadRaf)]
    pub fused_inc: C,
}

#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
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
    #[opening(committed = BalancedIncDigit)]
    pub balanced_inc_digits: Vec<C>,
    #[opening(committed = BalancedIncCarry)]
    pub balanced_inc_carry: C,
}

#[derive(Clone)]
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
        let eq_booleanity_default =
            derived(HammingWeightClaimReductionPublic::EqBooleanityAtDefault);
        let mut input = JoltExpr::zero();
        // The committed columns omit lane zero, reconstructed as `A − Σ_{k≠0} Q`.
        // Each leg's lane-zero baseline `w(0)·A` is folded into the input claim,
        // so the sumcheck runs over the committed `Q` alone:
        // `Σ_k (w(k) − w(0))·Q(k) = c − w(0)·A`. The hamming legs (`w ≡ 1`)
        // cancel entirely — `Σ_k L(k,t) = A(t)` holds by construction — so their
        // γ powers (`3i` and the even inc offsets) are left unused rather than
        // renumbered, keeping this layout aligned with the base relation's
        // (where the hamming leg is a real anchor) and with the shared prover's
        // `gamma_powers` indexing.
        for (i, polynomial) in self.shape.layout.polynomials().enumerate() {
            let eq_virtualization_default =
                derived(HammingWeightClaimReductionPublic::EqVirtualizationAtDefault(i));
            let activation = hamming_weight_claim(polynomial);
            input = input
                + gamma.clone().pow(3 * i + 1)
                    * (opening(booleanity_claim(polynomial))
                        - eq_booleanity_default.clone() * activation.clone())
                + gamma.clone().pow(3 * i + 2)
                    * (opening(crate::protocols::jolt::geometry::claim_reductions::hamming_weight::virtualization_claim(polynomial))
                        - eq_virtualization_default * activation);
        }
        for index in 0..self.shape.chunking.chunk_count() {
            let offset = self.ra_terms() + 2 * index;
            input = input
                + gamma.clone().pow(offset + 1)
                    * (opening(booleanity_balanced_inc_digit_opening(index))
                        - eq_booleanity_default.clone());
        }
        let msb_offset = self.ra_terms() + 2 * self.shape.chunking.chunk_count();
        input = input
            + gamma.clone().pow(msb_offset + 1)
                * (opening(booleanity_balanced_inc_carry_opening()) - eq_booleanity_default);
        input + gamma.pow(self.decode_power()) * opening(fused_inc_read_raf_opening())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(HammingWeightClaimReductionChallenge::Gamma);
        let eq_booleanity = derived(HammingWeightClaimReductionPublic::EqBooleanity);
        let eq_booleanity_default =
            derived(HammingWeightClaimReductionPublic::EqBooleanityAtDefault);
        let inc_value = derived(HammingWeightClaimReductionPublic::BalancedIncValueAtAddress);
        let decode_scale = gamma.clone().pow(self.decode_power());
        let mut output = JoltExpr::zero();

        // Purely sparse: every leg's lane-zero baseline lives in
        // `input_expression`, so each coefficient here is `w(ρ) − w(0)` (the
        // decode leg's `w(0)` is zero already: `balanced_inc_value(0) = 0`).
        for (i, polynomial) in self.shape.layout.polynomials().enumerate() {
            let eq_virtualization = derived(HammingWeightClaimReductionPublic::EqVirtualization(i));
            let eq_virtualization_default =
                derived(HammingWeightClaimReductionPublic::EqVirtualizationAtDefault(i));
            let coefficient = gamma.clone().pow(3 * i + 1)
                * (eq_booleanity.clone() - eq_booleanity_default.clone())
                + gamma.clone().pow(3 * i + 2) * (eq_virtualization - eq_virtualization_default);
            output = output + coefficient * opening(reduced_claim(polynomial));
        }
        for index in 0..self.shape.chunking.chunk_count() {
            let offset = self.ra_terms() + 2 * index;
            let coefficient = gamma.clone().pow(offset + 1)
                * (eq_booleanity.clone() - eq_booleanity_default.clone())
                + decode_scale.clone()
                    * constant(self.shape.chunking.place_value::<F>(index))
                    * inc_value.clone();
            output = output + coefficient * opening(reduced_balanced_inc_digit_opening(index));
        }
        let msb_offset = self.ra_terms() + 2 * self.shape.chunking.chunk_count();
        let coefficient = gamma.pow(msb_offset + 1) * (eq_booleanity - eq_booleanity_default)
            + decode_scale * constant(F::pow2(FUSED_INC_BITS)) * inc_value;
        output + coefficient * opening(reduced_balanced_inc_carry_opening())
    }
}

pub fn reduced_balanced_inc_digit_opening(index: usize) -> JoltOpeningId {
    JoltOpeningId::committed(
        crate::protocols::jolt::JoltCommittedPolynomial::BalancedIncDigit(index),
        JoltRelationId::HammingWeightClaimReduction,
    )
}

pub fn reduced_balanced_inc_carry_opening() -> JoltOpeningId {
    JoltOpeningId::committed(
        crate::protocols::jolt::JoltCommittedPolynomial::BalancedIncCarry,
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
        let eq_bool_default = Fr::from_u64(17);
        let eq_virt = Fr::from_u64(19);
        let eq_virt_default = Fr::from_u64(23);
        let inc_value = Fr::from_u64(31);
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
            id if id == booleanity_balanced_inc_digit_opening(0) => *bool_0,
            id if id == booleanity_balanced_inc_digit_opening(1) => *bool_1,
            id if id == booleanity_balanced_inc_carry_opening() => *bool_msb,
            id if id == fused_inc_read_raf_opening() => *fused,
            id if id == reduced_claim(JoltRaPolynomial::Ram(0)) => *out_ra,
            id if id == reduced_balanced_inc_digit_opening(0) => *out_0,
            id if id == reduced_balanced_inc_digit_opening(1) => *out_1,
            id if id == reduced_balanced_inc_carry_opening() => *out_msb,
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
                HammingWeightClaimReductionPublic::EqBooleanityAtDefault,
            ) => eq_bool_default,
            JoltDerivedId::HammingWeightClaimReduction(
                HammingWeightClaimReductionPublic::EqVirtualization(0),
            ) => eq_virt,
            JoltDerivedId::HammingWeightClaimReduction(
                HammingWeightClaimReductionPublic::EqVirtualizationAtDefault(0),
            ) => eq_virt_default,
            JoltDerivedId::HammingWeightClaimReduction(
                HammingWeightClaimReductionPublic::BalancedIncValueAtAddress,
            ) => inc_value,
            _ => zero,
        };

        let input = relation.input_expression::<Fr>().evaluate(
            opening_value,
            challenge_value,
            derived_value,
        );
        // The hamming legs (powers 0, 3, 5, 7) are gone, and each remaining
        // booleanity/virtualization claim carries its lane-zero baseline:
        // `c − w(0)·A`, with `A = hamming` for the RAM family and `1` for the
        // increment columns.
        let expected_input = power(1) * (*bool_ra - eq_bool_default * *hamming)
            + power(2) * (*virt_ra - eq_virt_default * *hamming)
            + power(4) * (*bool_0 - eq_bool_default)
            + power(6) * (*bool_1 - eq_bool_default)
            + power(8) * (*bool_msb - eq_bool_default)
            + power(9) * *fused;
        assert_eq!(input, expected_input);

        let output = relation.output_expression::<Fr>().evaluate(
            opening_value,
            challenge_value,
            derived_value,
        );
        // Purely sparse: no baseline terms, no `EqDefault`.
        let expected_output = *out_ra
            * (power(1) * (eq_bool - eq_bool_default) + power(2) * (eq_virt - eq_virt_default))
            + *out_0 * (power(4) * (eq_bool - eq_bool_default) + power(9) * inc_value)
            + *out_1
                * (power(6) * (eq_bool - eq_bool_default) + power(9) * Fr::pow2(32) * inc_value)
            + *out_msb
                * (power(8) * (eq_bool - eq_bool_default) + power(9) * Fr::pow2(64) * inc_value);
        assert_eq!(output, expected_output);

        assert_eq!(
            reduced_balanced_inc_carry_opening(),
            JoltOpeningId::committed(
                JoltCommittedPolynomial::BalancedIncCarry,
                JoltRelationId::HammingWeightClaimReduction,
            )
        );
    }
}
