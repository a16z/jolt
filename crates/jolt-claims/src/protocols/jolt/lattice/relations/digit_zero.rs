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
