//! The stage 3 `SpartanShift` sumcheck instance.
//!
//! Owns the shift opening-point derivation and the `EqPlusOne` public-value
//! computations (against the product uni-skip `tau_low` and the product-remainder
//! opening point).

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::spartan::{
    SpartanShiftChallenges, SpartanShiftInputClaims, SpartanShiftOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::dimensions::TraceDimensions, JoltDerivedId, SpartanShiftPublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::EqPlusOnePolynomial;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage1::Stage1BatchOutputClaims;
use crate::stages::stage2::Stage2BatchOutputClaims;
use crate::VerifierError;

/// Wire shift's consumed opening *values* from stage 1's outer sumcheck (`Next*`
/// PC/flag values) and stage 2's product-remainder `next_is_noop`. Takes the
/// ZK-agnostic upstream output-claims aggregates.
pub fn spartan_shift_input_values_from_upstream<F: Field>(
    stage1: &Stage1BatchOutputClaims<F>,
    stage2: &Stage2BatchOutputClaims<F>,
) -> SpartanShiftInputClaims<F> {
    let outer = &stage1.outer_remainder;
    SpartanShiftInputClaims {
        next_unexpanded_pc: outer.next_unexpanded_pc,
        next_pc: outer.next_pc,
        next_is_virtual: outer.next_is_virtual,
        next_is_first_in_sequence: outer.next_is_first_in_sequence,
        next_is_noop: stage2.product_remainder.next_is_noop,
    }
}

#[derive(Clone)]
pub struct SpartanShift<F: Field> {
    symbolic: relations::spartan::Shift,
    product_uniskip_tau_low: Vec<F>,
    product_remainder_opening_point: Vec<F>,
}

impl<F: Field> SpartanShift<F> {
    pub fn new(
        trace_dimensions: TraceDimensions,
        product_uniskip_tau_low: Vec<F>,
        product_remainder_opening_point: Vec<F>,
    ) -> Self {
        Self {
            symbolic: relations::spartan::Shift::new(trace_dimensions),
            product_uniskip_tau_low,
            product_remainder_opening_point,
        }
    }
}

impl<F: Field> SpartanShift<F> {
    pub fn product_uniskip_tau_low(&self) -> &[F] {
        &self.product_uniskip_tau_low
    }

    pub fn product_remainder_opening_point(&self) -> &[F] {
        &self.product_remainder_opening_point
    }
}

impl<F: Field> ConcreteSumcheck<F> for SpartanShift<F> {
    type Symbolic = relations::spartan::Shift;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &SpartanShiftInputClaims<Vec<F>>,
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

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &SpartanShiftInputClaims<Vec<F>>,
        output_points: &SpartanShiftOutputClaims<Vec<F>>,
        _challenges: &SpartanShiftChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::SpartanShift(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        // Every shift output shares the one shift opening point.
        let opening_point = output_points.unexpanded_pc();
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
