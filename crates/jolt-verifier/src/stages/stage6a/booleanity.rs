//! The stage 6a booleanity address-phase sumcheck instance.
//!
//! Booleanity proves every one-hot `Ra` chunk (instruction, bytecode, RAM) is
//! boolean. It runs in two phases: this stage-6a address phase binds the
//! `log_k_chunk` address variables and stages the `BooleanityAddrClaim`
//! intermediate consumed by the stage-6b cycle phase.

use core::marker::PhantomData;

use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::booleanity::{
    BooleanityAddressPhaseInputClaims, BooleanityAddressPhaseOutputClaims,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;

use crate::stages::relations::ConcreteSumcheck;
use crate::VerifierError;

pub struct BooleanityAddressPhase<F: Field> {
    symbolic: relations::booleanity::BooleanityAddressPhase,
    _field: PhantomData<F>,
}

impl<F: Field> BooleanityAddressPhase<F> {
    pub fn new(dimensions: BooleanityDimensions) -> Self {
        Self {
            symbolic: relations::booleanity::BooleanityAddressPhase::new(dimensions),
            _field: PhantomData,
        }
    }
}

impl<F: Field> ConcreteSumcheck<F> for BooleanityAddressPhase<F> {
    type Symbolic = relations::booleanity::BooleanityAddressPhase;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &BooleanityAddressPhaseInputClaims<Vec<F>>,
    ) -> Result<BooleanityAddressPhaseOutputClaims<Vec<F>>, VerifierError> {
        // The address opening point (`booleanity_r_address`) is the reversed
        // address sumcheck point; the cycle phase prepends it to its cycle point.
        Ok(BooleanityAddressPhaseOutputClaims {
            intermediate: sumcheck_point.iter().rev().copied().collect(),
        })
    }
}
