//! Lattice stage-7 claim reduction for digit-zero virtualization and the fused
//! increment decode (`specs/digit-zero-virtualization.md`).
//!
//! Instruction and bytecode implement the note's public
//! `M_mu(r_cycle) = 1` specialization. The commitment omits `ra_i(0, j)`, and
//! stage 7 applies the reconstruction coefficient
//! `eq(r_address, k_i) - eq(r_address, 0)` to the committed nonzero rows. The
//! balanced-increment columns use the same algebra with one Booleanity leg per
//! column; their decode weight is zero at digit zero. RAM is deliberately not
//! virtualized in this change and retains its fully committed three-leg base
//! reduction.
//!
//! The gamma layout uses two powers per virtualized RA polynomial, three per
//! RAM polynomial, one per increment column, then the decode power.
//!
//! WARNING: the decode leg is not a range check on `FusedInc`. One-hotness pins
//! each digit and the carry only to `[-K/2, K/2)`, so the reachable set is the
//! `K · 2^64` integers the balanced numeral spans (~`±2^71` at `K = 256`), not
//! the honest `|delta| < 2^64`. That is safe because the encoding is injective,
//! fp128 leaves 56 bits of headroom over `2^71`, and base mode commits `Inc`
//! with no range relation at all — see "Increment range" in
//! `specs/lattice-claims.md`. Do not treat this reduction as bounding `Inc`.

use jolt_field::Ring;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::hamming_weight::{
    booleanity_claim, reduced_claim, virtualization_claim,
};
use crate::protocols::jolt::geometry::ra::{JoltRaPolynomial, JoltRaPolynomialLayout};
use crate::protocols::jolt::geometry::ram::ram_hamming_weight;
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
pub struct LatticeDigitZeroClaimReductionDimensions {
    pub layout: JoltRaPolynomialLayout,
    pub log_k_chunk: usize,
    chunking: BalancedIncChunking,
}

impl LatticeDigitZeroClaimReductionDimensions {
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

/// Number of γ powers a family's RA polynomial occupies: RAM takes the base
/// three legs (Hamming, Booleanity, virtualization); the digit-zero families
/// take two (Booleanity, virtualization).
fn ra_leg_count(polynomial: JoltRaPolynomial) -> usize {
    match polynomial {
        JoltRaPolynomial::Ram(_) => 3,
        JoltRaPolynomial::Instruction(_) | JoltRaPolynomial::Bytecode(_) => 2,
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct LatticeDigitZeroClaimReductionInputClaims<C> {
    /// The RAM access indicator produced by `RamHammingBooleanity`. RAM is not
    /// virtualized, so this remains the base Hamming-weight leg.
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
pub struct LatticeDigitZeroClaimReductionOutputClaims<C> {
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

/// The lattice instantiation of the stage-7 reduction slot, selected by the verifier's
/// `akita` feature (`jolt-verifier/src/stages/stage7/hamming_weight_claim_reduction.rs`,
/// the `mode` module — the single place base and lattice are swapped).
///
/// [`Self::id`] deliberately returns the base `JoltRelationId::HammingWeightClaimReduction`:
/// the relation id is protocol data (opening ids, transcript labels, tamper-manifest
/// paths), RAM still carries a genuine Hamming-weight leg here, and base mode must keep
/// producing bit-identical proofs. So the *slot* is named for the base algebra while this
/// type is named for the lattice one — the verifier aliases between the two names once,
/// at the `mode` seam, and nowhere else.
#[derive(Clone)]
pub struct LatticeDigitZeroClaimReduction {
    shape: LatticeDigitZeroClaimReductionDimensions,
}

impl LatticeDigitZeroClaimReduction {
    /// Total γ powers consumed by the RA legs (variable: 3 per RAM poly, 2
    /// per virtualized poly). The increment columns and decode power follow.
    fn ra_terms(&self) -> usize {
        self.shape.layout.polynomials().map(ra_leg_count).sum()
    }

    fn inc_column_count(&self) -> usize {
        self.shape.chunking.chunk_count() + 1
    }

    fn decode_power(&self) -> usize {
        self.ra_terms() + self.inc_column_count()
    }
}

impl SymbolicSumcheck for LatticeDigitZeroClaimReduction {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = LatticeDigitZeroClaimReductionDimensions;
    type Challenges<F> = HammingWeightClaimReductionChallenges<F>;
    type Inputs<C> = LatticeDigitZeroClaimReductionInputClaims<C>;
    type Outputs<C> = LatticeDigitZeroClaimReductionOutputClaims<C>;

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

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        let gamma = challenge(HammingWeightClaimReductionChallenge::Gamma);
        let eq_booleanity_digit_zero =
            derived(HammingWeightClaimReductionPublic::EqBooleanityAtDigitZero);
        let mut input = JoltExpr::zero();
        let mut power = 0usize;
        for (i, polynomial) in self.shape.layout.polynomials().enumerate() {
            match polynomial {
                // RAM: base three legs, no reconstruction. The committed column
                // includes the digit-zero row.
                JoltRaPolynomial::Ram(_) => {
                    input = input
                        + gamma.clone().pow(power) * opening(ram_hamming_weight())
                        + gamma.clone().pow(power + 1) * opening(booleanity_claim(polynomial))
                        + gamma.clone().pow(power + 2) * opening(virtualization_claim(polynomial));
                    power += 3;
                }
                // Public M_mu = 1: fold eq(r_address, 0) into each input claim.
                JoltRaPolynomial::Instruction(_) | JoltRaPolynomial::Bytecode(_) => {
                    let eq_virtualization_digit_zero =
                        derived(HammingWeightClaimReductionPublic::EqVirtualizationAtDigitZero(i));
                    input = input
                        + gamma.clone().pow(power)
                            * (opening(booleanity_claim(polynomial))
                                - eq_booleanity_digit_zero.clone())
                        + gamma.clone().pow(power + 1)
                            * (opening(virtualization_claim(polynomial))
                                - eq_virtualization_digit_zero);
                    power += 2;
                }
            }
        }
        for index in 0..self.shape.chunking.chunk_count() {
            input = input
                + gamma.clone().pow(power)
                    * (opening(booleanity_balanced_inc_digit_opening(index))
                        - eq_booleanity_digit_zero.clone());
            power += 1;
        }
        input = input
            + gamma.clone().pow(power)
                * (opening(booleanity_balanced_inc_carry_opening()) - eq_booleanity_digit_zero);
        power += 1;
        debug_assert_eq!(power, self.decode_power());
        input + gamma.pow(self.decode_power()) * opening(fused_inc_read_raf_opening())
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        let gamma = challenge(HammingWeightClaimReductionChallenge::Gamma);
        let eq_booleanity = derived(HammingWeightClaimReductionPublic::EqBooleanity);
        let eq_booleanity_digit_zero =
            derived(HammingWeightClaimReductionPublic::EqBooleanityAtDigitZero);
        let inc_value = derived(HammingWeightClaimReductionPublic::BalancedIncValueAtAddress);
        let decode_scale = gamma.clone().pow(self.decode_power());
        let mut output = JoltExpr::zero();
        let mut power = 0usize;

        for (i, polynomial) in self.shape.layout.polynomials().enumerate() {
            let eq_virtualization = derived(HammingWeightClaimReductionPublic::EqVirtualization(i));
            let coefficient = match polynomial {
                // RAM: base Hamming, Booleanity, and virtualization legs.
                JoltRaPolynomial::Ram(_) => {
                    let c = gamma.clone().pow(power)
                        + gamma.clone().pow(power + 1) * eq_booleanity.clone()
                        + gamma.clone().pow(power + 2) * eq_virtualization;
                    power += 3;
                    c
                }
                // The committed rows use eq(r_address, k_i) - eq(r_address, 0).
                JoltRaPolynomial::Instruction(_) | JoltRaPolynomial::Bytecode(_) => {
                    let eq_virtualization_digit_zero =
                        derived(HammingWeightClaimReductionPublic::EqVirtualizationAtDigitZero(i));
                    let c = gamma.clone().pow(power)
                        * (eq_booleanity.clone() - eq_booleanity_digit_zero.clone())
                        + gamma.clone().pow(power + 1)
                            * (eq_virtualization - eq_virtualization_digit_zero);
                    power += 2;
                    c
                }
            };
            output = output + coefficient * opening(reduced_claim(polynomial));
        }
        for index in 0..self.shape.chunking.chunk_count() {
            let coefficient = gamma.clone().pow(power)
                * (eq_booleanity.clone() - eq_booleanity_digit_zero.clone())
                + decode_scale.clone()
                    * constant(self.shape.chunking.place_value::<F>(index))
                    * inc_value.clone();
            output = output + coefficient * opening(reduced_balanced_inc_digit_opening(index));
            power += 1;
        }
        let coefficient = gamma.pow(power) * (eq_booleanity - eq_booleanity_digit_zero)
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
    use crate::protocols::jolt::{
        HammingWeightClaimReductionChallenge, JoltChallengeId, JoltDerivedId,
    };
    use jolt_field::{Fr, Ring};

    #[test]
    fn public_unit_activation_ra_is_reconstructed_but_ram_keeps_base_legs() {
        let layout = JoltRaPolynomialLayout::new(1, 1, 1).unwrap();
        let relation = LatticeDigitZeroClaimReduction::new(
            LatticeDigitZeroClaimReductionDimensions::new(layout, 32).unwrap(),
        );
        assert_eq!(relation.decode_power(), 10);

        let gamma = Fr::from_u64(3);
        let eq_bool = Fr::from_u64(13);
        let eq_bool_zero = Fr::from_u64(17);
        let eq_virt = [Fr::from_u64(19), Fr::from_u64(23), Fr::from_u64(29)];
        let eq_virt_zero = [Fr::from_u64(31), Fr::from_u64(37)];
        let inst_bool = Fr::from_u64(41);
        let inst_virt = Fr::from_u64(43);
        let bytecode_bool = Fr::from_u64(47);
        let bytecode_virt = Fr::from_u64(53);
        let ram_hamming = Fr::from_u64(59);
        let ram_bool = Fr::from_u64(61);
        let ram_virt = Fr::from_u64(67);
        let out = [Fr::from_u64(71), Fr::from_u64(73), Fr::from_u64(79)];
        let zero = Fr::from_u64(0);
        let power = |exponent: usize| {
            (0..exponent).fold(Fr::from_u64(1), |accumulator, _| accumulator * gamma)
        };

        let opening_value = |id: &JoltOpeningId| match *id {
            id if id == booleanity_claim(JoltRaPolynomial::Instruction(0)) => inst_bool,
            id if id == virtualization_claim(JoltRaPolynomial::Instruction(0)) => inst_virt,
            id if id == booleanity_claim(JoltRaPolynomial::Bytecode(0)) => bytecode_bool,
            id if id == virtualization_claim(JoltRaPolynomial::Bytecode(0)) => bytecode_virt,
            id if id == ram_hamming_weight() => ram_hamming,
            id if id == booleanity_claim(JoltRaPolynomial::Ram(0)) => ram_bool,
            id if id == virtualization_claim(JoltRaPolynomial::Ram(0)) => ram_virt,
            id if id == reduced_claim(JoltRaPolynomial::Instruction(0)) => out[0],
            id if id == reduced_claim(JoltRaPolynomial::Bytecode(0)) => out[1],
            id if id == reduced_claim(JoltRaPolynomial::Ram(0)) => out[2],
            id if id == booleanity_balanced_inc_digit_opening(0)
                || id == booleanity_balanced_inc_digit_opening(1)
                || id == booleanity_balanced_inc_carry_opening() =>
            {
                eq_bool_zero
            }
            _ => zero,
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
                HammingWeightClaimReductionPublic::EqBooleanityAtDigitZero,
            ) => eq_bool_zero,
            JoltDerivedId::HammingWeightClaimReduction(
                HammingWeightClaimReductionPublic::EqVirtualization(index),
            ) => eq_virt[index],
            JoltDerivedId::HammingWeightClaimReduction(
                HammingWeightClaimReductionPublic::EqVirtualizationAtDigitZero(index),
            ) => eq_virt_zero[index],
            JoltDerivedId::HammingWeightClaimReduction(
                HammingWeightClaimReductionPublic::BalancedIncValueAtAddress,
            ) => zero,
            _ => zero,
        };

        let input = relation.input_expression::<Fr>().evaluate(
            opening_value,
            challenge_value,
            derived_value,
        );
        let expected_input = power(0) * (inst_bool - eq_bool_zero)
            + power(1) * (inst_virt - eq_virt_zero[0])
            + power(2) * (bytecode_bool - eq_bool_zero)
            + power(3) * (bytecode_virt - eq_virt_zero[1])
            + power(4) * ram_hamming
            + power(5) * ram_bool
            + power(6) * ram_virt;
        assert_eq!(input, expected_input);

        let output = relation.output_expression::<Fr>().evaluate(
            opening_value,
            challenge_value,
            derived_value,
        );
        let expected_output = out[0]
            * (power(0) * (eq_bool - eq_bool_zero) + power(1) * (eq_virt[0] - eq_virt_zero[0]))
            + out[1]
                * (power(2) * (eq_bool - eq_bool_zero) + power(3) * (eq_virt[1] - eq_virt_zero[1]))
            + out[2] * (power(4) + power(5) * eq_bool + power(6) * eq_virt[2]);
        assert_eq!(output, expected_output);
    }

    #[test]
    fn ram_base_legs_and_fused_increment_terms() {
        // One RAM polynomial (base 3-leg) plus two increment digits and a carry.
        let layout = JoltRaPolynomialLayout::new(0, 0, 1).unwrap();
        let relation = LatticeDigitZeroClaimReduction::new(
            LatticeDigitZeroClaimReductionDimensions::new(layout, 32).unwrap(),
        );
        // RAM occupies powers 0,1,2; inc digits 3,4; carry 5; decode 6.
        assert_eq!(relation.decode_power(), 6);
        let gamma = Fr::from_u64(3);
        let values = (2..=12).map(Fr::from_u64).collect::<Vec<_>>();
        let [hamming, bool_ra, virt_ra, bool_0, bool_1, bool_msb, fused, out_ra, out_0, out_1, out_msb] =
            values.as_slice()
        else {
            unreachable!()
        };
        let eq_bool = Fr::from_u64(17);
        let eq_bool_digit_zero = Fr::from_u64(19);
        let eq_virt = Fr::from_u64(23);
        let inc_value = Fr::from_u64(31);
        let power = |exponent: usize| {
            (0..exponent).fold(Fr::from_u64(1), |accumulator, _| accumulator * gamma)
        };
        let zero = Fr::from_u64(0);

        let opening_value = |id: &JoltOpeningId| match *id {
            id if id == ram_hamming_weight() => *hamming,
            id if id == booleanity_claim(JoltRaPolynomial::Ram(0)) => *bool_ra,
            id if id == virtualization_claim(JoltRaPolynomial::Ram(0)) => *virt_ra,
            id if id == booleanity_balanced_inc_digit_opening(0) => *bool_0,
            id if id == booleanity_balanced_inc_digit_opening(1) => *bool_1,
            id if id == booleanity_balanced_inc_carry_opening() => *bool_msb,
            id if id == fused_inc_read_raf_opening() => *fused,
            id if id == reduced_claim(JoltRaPolynomial::Ram(0)) => *out_ra,
            id if id == reduced_balanced_inc_digit_opening(0) => *out_0,
            id if id == reduced_balanced_inc_digit_opening(1) => *out_1,
            id if id == reduced_balanced_inc_carry_opening() => *out_msb,
            _ => zero,
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
                HammingWeightClaimReductionPublic::EqBooleanityAtDigitZero,
            ) => eq_bool_digit_zero,
            JoltDerivedId::HammingWeightClaimReduction(
                HammingWeightClaimReductionPublic::EqVirtualization(0),
            ) => eq_virt,
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
        // RAM base 3-leg (no recentering) at powers 0,1,2; increment
        // booleanity legs (recentered) at 3,4; carry at 5; decode at 6.
        let expected_input = power(0) * *hamming
            + power(1) * *bool_ra
            + power(2) * *virt_ra
            + power(3) * (*bool_0 - eq_bool_digit_zero)
            + power(4) * (*bool_1 - eq_bool_digit_zero)
            + power(5) * (*bool_msb - eq_bool_digit_zero)
            + power(6) * *fused;
        assert_eq!(input, expected_input);

        let output = relation.output_expression::<Fr>().evaluate(
            opening_value,
            challenge_value,
            derived_value,
        );
        let expected_output = *out_ra * (power(0) + power(1) * eq_bool + power(2) * eq_virt)
            + *out_0 * (power(3) * (eq_bool - eq_bool_digit_zero) + power(6) * inc_value)
            + *out_1
                * (power(4) * (eq_bool - eq_bool_digit_zero) + power(6) * Fr::pow2(32) * inc_value)
            + *out_msb
                * (power(5) * (eq_bool - eq_bool_digit_zero) + power(6) * Fr::pow2(64) * inc_value);
        assert_eq!(output, expected_output);
    }
}
