//! The stage 1 `SpartanOuter` univariate-skip sumcheck instance.
//!
//! The companion of the [`OuterRemainder`](super::outer_remainder) relation: the
//! Spartan outer uni-skip first round, a standalone centered-integer sumcheck whose
//! reduced opening the remainder consumes. Its input claim is zero (the first round
//! has no consumed openings), so this relation carries no inputs and overrides
//! neither `derive_input_term` nor `derive_output_term`; it exists to single-source
//! the uni-skip's spec / point-derivation alongside the BlindFold constraint, which
//! evaluates the same (zero) input formula and the same bare-opening output.
//!
//! Like the product uni-skip, the centered-integer first-round verification itself
//! stays hand-coded in the stage-1 verifier (`verify_compressed_boolean` cannot
//! handle the centered-integer domain); this relation supplies only the algebra.

use jolt_claims::protocols::jolt::geometry::spartan::SpartanOuterDimensions;
use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::spartan::{
    OuterUniskipInputClaims, OuterUniskipOutputClaims,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;

use crate::stages::relations::{ConcreteSumcheck, GetPoint};
use crate::VerifierError;

pub struct OuterUniskip<F: Field> {
    symbolic: relations::spartan::OuterUniskip,
    _marker: core::marker::PhantomData<F>,
}

impl<F: Field> OuterUniskip<F> {
    pub fn new(dimensions: SpartanOuterDimensions) -> Self {
        Self {
            symbolic: relations::spartan::OuterUniskip::new(dimensions),
            _marker: core::marker::PhantomData,
        }
    }
}

impl<F: Field> ConcreteSumcheck<F> for OuterUniskip<F> {
    type Symbolic = relations::spartan::OuterUniskip;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &OuterUniskipInputClaims<C>,
    ) -> Result<OuterUniskipOutputClaims<Vec<F>>, VerifierError> {
        Ok(OuterUniskipOutputClaims {
            univariate_skip: sumcheck_point.to_vec(),
        })
    }
}
