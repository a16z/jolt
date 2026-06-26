//! The stage 3 `SpartanShift` sumcheck instance.
//!
//! A self-contained relation object driven identically by the prover (while
//! producing the stage 3 batch proof) and the verifier (after checking it). It
//! owns the shift opening-point derivation and the `EqPlusOne` public-value
//! computations (against the product uni-skip `tau_low` and the product-remainder
//! opening point), so the input/output claim algebra lives here once.

use jolt_claims::protocols::jolt::relations;
use jolt_claims::protocols::jolt::{
    geometry::dimensions::TraceDimensions, JoltChallengeId, JoltDerivedId, SpartanShiftChallenge,
    SpartanShiftPublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::EqPlusOnePolynomial;
use jolt_riscv::{CircuitFlags, InstructionFlags};
use jolt_claims_derive::{InputClaims, OutputClaims};
use serde::{Deserialize, Serialize};

use crate::stages::relations::{ConcreteSumcheck, GetPoint, OpeningClaim};
use crate::stages::stage1::Stage1ClearOutput;
use crate::stages::stage2::Stage2ClearOutput;
use crate::VerifierError;

/// Produced Spartan shift openings (the shifted unexpanded-PC / PC / virtual /
/// first-in-sequence / noop columns), all sharing the single shift opening point.
/// Generic over the cell.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(SpartanShift)]
pub struct SpartanShiftOutputClaims<C> {
    #[opening(UnexpandedPC)]
    pub unexpanded_pc: C,
    #[opening(PC)]
    pub pc: C,
    #[opening(OpFlags(CircuitFlags::VirtualInstruction))]
    pub is_virtual: C,
    #[opening(OpFlags(CircuitFlags::IsFirstInSequence))]
    pub is_first_in_sequence: C,
    #[opening(InstructionFlags(InstructionFlags::IsNoop))]
    pub is_noop: C,
}

/// Consumed shift openings: the `Next*` PC/flag columns from stage 1's outer
/// sumcheck and `next_is_noop` from stage 2's product remainder. Shift reads only
/// these values, so the input points are left empty. Generic over the cell.
#[derive(Clone, Debug, InputClaims)]
pub struct SpartanShiftInputClaims<C> {
    #[opening(NextUnexpandedPC, from = SpartanOuter)]
    pub next_unexpanded_pc: C,
    #[opening(NextPC, from = SpartanOuter)]
    pub next_pc: C,
    #[opening(NextIsVirtual, from = SpartanOuter)]
    pub next_is_virtual: C,
    #[opening(NextIsFirstInSequence, from = SpartanOuter)]
    pub next_is_first_in_sequence: C,
    #[opening(NextIsNoop, from = SpartanProductVirtualization)]
    pub next_is_noop: C,
}

impl<F: Field> SpartanShiftInputClaims<OpeningClaim<F>> {
    /// Wire shift's consumed openings from stage 1's outer sumcheck (`Next*`
    /// PC/flag values) and stage 2's product-remainder `next_is_noop`. Shift reads
    /// only these values — its output points come from its own sumcheck point and
    /// the stage-2 eq tables — so the input opening points are left empty.
    pub fn from_upstream(stage1: &Stage1ClearOutput<F>, stage2: &Stage2ClearOutput<F>) -> Self {
        let value = |value: F| OpeningClaim {
            point: Vec::new(),
            value,
        };
        Self {
            next_unexpanded_pc: value(stage1.outer.next_unexpanded_pc),
            next_pc: value(stage1.outer.next_pc),
            next_is_virtual: value(stage1.outer.next_is_virtual),
            next_is_first_in_sequence: value(stage1.outer.next_is_first_in_sequence),
            next_is_noop: value(stage2.output_claims.product_remainder.next_is_noop.value),
        }
    }
}

pub struct SpartanShift<F: Field> {
    symbolic: relations::spartan::Shift,
    gamma: F,
    product_uniskip_tau_low: Vec<F>,
    product_remainder_opening_point: Vec<F>,
}

impl<F: Field> SpartanShift<F> {
    pub fn new(
        trace_dimensions: TraceDimensions,
        gamma: F,
        product_uniskip_tau_low: Vec<F>,
        product_remainder_opening_point: Vec<F>,
    ) -> Self {
        Self {
            symbolic: relations::spartan::Shift::new(trace_dimensions),
            gamma,
            product_uniskip_tau_low,
            product_remainder_opening_point,
        }
    }
}

impl<F: Field> ConcreteSumcheck<F> for SpartanShift<F> {
    type Symbolic = relations::spartan::Shift;
    type Inputs<C> = SpartanShiftInputClaims<C>;
    type Outputs<C> = SpartanShiftOutputClaims<C>;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &SpartanShiftInputClaims<C>,
    ) -> Result<SpartanShiftOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        Ok(SpartanShiftOutputClaims {
            unexpanded_pc: opening_point.clone(),
            pc: opening_point.clone(),
            is_virtual: opening_point.clone(),
            is_first_in_sequence: opening_point.clone(),
            is_noop: opening_point,
        })
    }

    fn resolve_challenge(&self, id: &JoltChallengeId) -> Result<F, VerifierError> {
        match id {
            JoltChallengeId::SpartanShift(SpartanShiftChallenge::Gamma) => Ok(self.gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        }
    }

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltDerivedId,
        _inputs: &SpartanShiftInputClaims<C>,
        outputs: Option<&SpartanShiftOutputClaims<OpeningClaim<F>>>,
    ) -> Result<F, VerifierError> {
        let outputs = outputs.ok_or(VerifierError::MissingStageClaimDerived { id: *id })?;
        let JoltDerivedId::SpartanShift(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        // Every shift output shares the one shift opening point.
        let opening_point = outputs.unexpanded_pc.point();
        match public_id {
            SpartanShiftPublic::EqPlusOneOuter => Ok(EqPlusOnePolynomial::new(
                self.product_uniskip_tau_low.clone(),
            )
            .evaluate(opening_point)),
            SpartanShiftPublic::EqPlusOneProduct => Ok(EqPlusOnePolynomial::new(
                self.product_remainder_opening_point.clone(),
            )
            .evaluate(opening_point)),
        }
    }
}
