//! Lattice-mode digit-zero claim reduction, extended with the fused
//! increment's one-hot decomposition (`specs/digit-zero-virtualization.md`).
//!
//! The committed columns omit the digit-zero row, defined as
//! `ra_i(0, t) := M_µ(t) − Σ_{k≠0} ra_i(k, t)` for the family's activation
//! `M_µ` — constant 1 everywhere except RAM, whose activation is
//! `Load + Store` (the flag openings produced by `RamActivationBooleanity`).
//! This reduction recenters each Booleanity/virtualization claim on a
//! semantic (digit-zero-inclusive) column into a claim on the committed
//! nonzero-digit rows: `Σ_k (w(k) − w(0))·ra_i(k) = c − w(0)·M̃_µ(r_cycle)`.
//! The weight identity `Σ_k ra_i(k, t) = M_µ(t)` holds by construction, so
//! there is no Hamming-weight leg of any kind.
//!
//! WARNING: the decode leg is not a range check on `FusedInc`. One-hotness pins
//! each digit and the carry only to `[-K/2, K/2)`, so the reachable set is the
//! `K · 2^64` integers the balanced numeral spans (~`±2^71` at `K = 256`), not
//! the honest `|delta| < 2^64`. That is safe because the encoding is injective,
//! fp128 leaves 56 bits of headroom over `2^71`, and base mode commits `Inc`
//! with no range relation at all — see "Increment range" in
//! `specs/lattice-claims.md`. Do not treat this reduction as bounding `Inc`.

use jolt_field::RingCore;
use jolt_riscv::CircuitFlags;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::hamming_weight::{
    booleanity_claim, reduced_claim,
};
use crate::protocols::jolt::geometry::ra::{JoltRaPolynomial, JoltRaPolynomialLayout};
use crate::protocols::jolt::geometry::ram::{ram_activation_load, ram_activation_store};
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

/// The family's activation evaluation `M̃_µ(r_cycle)`: constant 1 for the
/// always-active families, `Load + Store` (the `RamActivationBooleanity`
/// openings at the shared stage-6b cycle point) for RAM.
fn activation_claim<F: RingCore>(polynomial: JoltRaPolynomial) -> JoltExpr<F> {
    match polynomial {
        JoltRaPolynomial::Instruction(_) | JoltRaPolynomial::Bytecode(_) => JoltExpr::one(),
        JoltRaPolynomial::Ram(_) => {
            opening(ram_activation_load()) + opening(ram_activation_store())
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct LatticeDigitZeroClaimReductionInputClaims<C> {
    /// The RAM activation `M_RAM = Load + Store` — the digit-zero
    /// reconstruction coefficient. Each `RamRa` leg's digit-zero baseline
    /// `w(0)·M̃_RAM` is folded into the input claim, so these claims enter with
    /// a negative `eq(0, ·)`-weighted coefficient. Only their sum is
    /// constrained (see `RamActivationBooleanity`).
    #[opening(OpFlags(CircuitFlags::Load), from = RamActivationBooleanity)]
    pub ram_activation_load: C,
    #[opening(OpFlags(CircuitFlags::Store), from = RamActivationBooleanity)]
    pub ram_activation_store: C,
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

/// The lattice instantiation of the stage-7 reduction slot. It keeps base
/// mode's `HammingWeightClaimReduction` relation id (the shared final-opening
/// map keys on it) but performs no Hamming-weight check: under digit-zero
/// virtualization the weight identity holds by construction, so the γ layout
/// is two powers per RA family (Booleanity, virtualization), one per
/// increment column (Booleanity), and the decode power.
#[derive(Clone)]
pub struct LatticeDigitZeroClaimReduction {
    shape: LatticeDigitZeroClaimReductionDimensions,
}

impl LatticeDigitZeroClaimReduction {
    fn ra_terms(&self) -> usize {
        2 * self.shape.layout.total()
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

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(HammingWeightClaimReductionChallenge::Gamma);
        let eq_booleanity_digit_zero =
            derived(HammingWeightClaimReductionPublic::EqBooleanityAtDigitZero);
        let mut input = JoltExpr::zero();
        // Each leg's digit-zero baseline `w(0)·M̃_µ` is folded into the input
        // claim, so the sumcheck runs over the committed nonzero-digit rows
        // alone: `Σ_k (w(k) − w(0))·ra_i(k) = c − w(0)·M̃_µ`.
        for (i, polynomial) in self.shape.layout.polynomials().enumerate() {
            let eq_virtualization_digit_zero =
                derived(HammingWeightClaimReductionPublic::EqVirtualizationAtDigitZero(i));
            let activation = activation_claim(polynomial);
            input = input
                + gamma.clone().pow(2 * i)
                    * (opening(booleanity_claim(polynomial))
                        - eq_booleanity_digit_zero.clone() * activation.clone())
                + gamma.clone().pow(2 * i + 1)
                    * (opening(crate::protocols::jolt::geometry::claim_reductions::hamming_weight::virtualization_claim(polynomial))
                        - eq_virtualization_digit_zero * activation);
        }
        for index in 0..self.shape.chunking.chunk_count() {
            let offset = self.ra_terms() + index;
            input = input
                + gamma.clone().pow(offset)
                    * (opening(booleanity_balanced_inc_digit_opening(index))
                        - eq_booleanity_digit_zero.clone());
        }
        let carry_offset = self.ra_terms() + self.shape.chunking.chunk_count();
        input = input
            + gamma.clone().pow(carry_offset)
                * (opening(booleanity_balanced_inc_carry_opening()) - eq_booleanity_digit_zero);
        input + gamma.pow(self.decode_power()) * opening(fused_inc_read_raf_opening())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(HammingWeightClaimReductionChallenge::Gamma);
        let eq_booleanity = derived(HammingWeightClaimReductionPublic::EqBooleanity);
        let eq_booleanity_digit_zero =
            derived(HammingWeightClaimReductionPublic::EqBooleanityAtDigitZero);
        let inc_value = derived(HammingWeightClaimReductionPublic::BalancedIncValueAtAddress);
        let decode_scale = gamma.clone().pow(self.decode_power());
        let mut output = JoltExpr::zero();

        // Purely sparse: every leg's digit-zero baseline lives in
        // `input_expression`, so each coefficient here is `w(ρ) − w(0)` (the
        // decode leg's `w(0)` is zero already: `balanced_inc_value(0) = 0`).
        for (i, polynomial) in self.shape.layout.polynomials().enumerate() {
            let eq_virtualization = derived(HammingWeightClaimReductionPublic::EqVirtualization(i));
            let eq_virtualization_digit_zero =
                derived(HammingWeightClaimReductionPublic::EqVirtualizationAtDigitZero(i));
            let coefficient = gamma.clone().pow(2 * i)
                * (eq_booleanity.clone() - eq_booleanity_digit_zero.clone())
                + gamma.clone().pow(2 * i + 1) * (eq_virtualization - eq_virtualization_digit_zero);
            output = output + coefficient * opening(reduced_claim(polynomial));
        }
        for index in 0..self.shape.chunking.chunk_count() {
            let offset = self.ra_terms() + index;
            let coefficient = gamma.clone().pow(offset)
                * (eq_booleanity.clone() - eq_booleanity_digit_zero.clone())
                + decode_scale.clone()
                    * constant(self.shape.chunking.place_value::<F>(index))
                    * inc_value.clone();
            output = output + coefficient * opening(reduced_balanced_inc_digit_opening(index));
        }
        let carry_offset = self.ra_terms() + self.shape.chunking.chunk_count();
        let coefficient = gamma.pow(carry_offset) * (eq_booleanity - eq_booleanity_digit_zero)
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
        HammingWeightClaimReductionChallenge, JoltChallengeId, JoltCommittedPolynomial,
        JoltDerivedId,
    };
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn fused_increment_terms_extend_the_ra_reduction() {
        let layout = JoltRaPolynomialLayout::new(0, 0, 1).unwrap();
        let relation = LatticeDigitZeroClaimReduction::new(
            LatticeDigitZeroClaimReductionDimensions::new(layout, 32).unwrap(),
        );
        let gamma = Fr::from_u64(3);
        let values = (2..=13).map(Fr::from_u64).collect::<Vec<_>>();
        let [load, store, bool_ra, virt_ra, bool_0, bool_1, bool_msb, fused, out_ra, out_0, out_1, out_msb] =
            values.as_slice()
        else {
            unreachable!()
        };
        let eq_bool = Fr::from_u64(17);
        let eq_bool_digit_zero = Fr::from_u64(19);
        let eq_virt = Fr::from_u64(23);
        let eq_virt_digit_zero = Fr::from_u64(29);
        let inc_value = Fr::from_u64(31);
        let power = |exponent: usize| {
            (0..exponent).fold(Fr::from_u64(1), |accumulator, _| accumulator * gamma)
        };
        let zero = Fr::from_u64(0);

        let opening_value = |id: &JoltOpeningId| {
            match *id {
            id if id == ram_activation_load() => *load,
            id if id == ram_activation_store() => *store,
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
                HammingWeightClaimReductionPublic::EqBooleanityAtDigitZero,
            ) => eq_bool_digit_zero,
            JoltDerivedId::HammingWeightClaimReduction(
                HammingWeightClaimReductionPublic::EqVirtualization(0),
            ) => eq_virt,
            JoltDerivedId::HammingWeightClaimReduction(
                HammingWeightClaimReductionPublic::EqVirtualizationAtDigitZero(0),
            ) => eq_virt_digit_zero,
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
        // No Hamming legs and no reserved powers: the RA legs sit at
        // `γ^{2i}`/`γ^{2i+1}`, the increment columns at consecutive powers,
        // and each claim carries its digit-zero baseline `c − w(0)·M̃_µ`, with
        // `M̃_RAM = load + store` for the RAM family and 1 for the increment
        // columns.
        let activation = *load + *store;
        let expected_input = power(0) * (*bool_ra - eq_bool_digit_zero * activation)
            + power(1) * (*virt_ra - eq_virt_digit_zero * activation)
            + power(2) * (*bool_0 - eq_bool_digit_zero)
            + power(3) * (*bool_1 - eq_bool_digit_zero)
            + power(4) * (*bool_msb - eq_bool_digit_zero)
            + power(5) * *fused;
        assert_eq!(input, expected_input);

        let output = relation.output_expression::<Fr>().evaluate(
            opening_value,
            challenge_value,
            derived_value,
        );
        // Purely sparse: no baseline terms.
        let expected_output = *out_ra
            * (power(0) * (eq_bool - eq_bool_digit_zero)
                + power(1) * (eq_virt - eq_virt_digit_zero))
            + *out_0 * (power(2) * (eq_bool - eq_bool_digit_zero) + power(5) * inc_value)
            + *out_1
                * (power(3) * (eq_bool - eq_bool_digit_zero) + power(5) * Fr::pow2(32) * inc_value)
            + *out_msb
                * (power(4) * (eq_bool - eq_bool_digit_zero) + power(5) * Fr::pow2(64) * inc_value);
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
