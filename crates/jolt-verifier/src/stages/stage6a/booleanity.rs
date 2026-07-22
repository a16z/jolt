//! The stage 6a booleanity address-phase sumcheck instance.
//!
//! Booleanity proves every one-hot `Ra` chunk (instruction, bytecode, RAM) is
//! boolean. It runs in two phases: this stage-6a address phase binds the
//! `log_k_chunk` address variables and stages the `BooleanityAddrClaim`
//! intermediate consumed by the stage-6b cycle phase.

use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::booleanity::{
    BooleanityAddressPhaseInputClaims, BooleanityAddressPhaseOutputClaims,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;

use crate::stages::relations::ConcreteSumcheck;
use crate::VerifierError;

/// The hand pre-batch draws the booleanity subprotocol consumes: the
/// little-endian reference address (the reversed stage-5 instruction address,
/// padded with fresh draws or truncated to the committed chunk width), the
/// little-endian reference cycle, and the batching gamma. Drawn by the stage
/// front between `draw_challenges` and the batch (the frozen wire schedule),
/// then attached here so the address-phase kernel reads them off the relation
/// like any other carried vector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BooleanityReferenceDraws<F: Field> {
    pub reference_address: Vec<F>,
    pub reference_cycle: Vec<F>,
    pub gamma: F,
}

#[derive(Clone)]
pub struct BooleanityAddressPhase<F: Field> {
    symbolic: relations::booleanity::BooleanityAddressPhase,
    dimensions: BooleanityDimensions,
    /// `None` until the stage front's post-`draw_challenges` hand draws run;
    /// the verifier never evaluates these (only the prover's kernel does).
    reference_draws: Option<BooleanityReferenceDraws<F>>,
}

impl<F: Field> BooleanityAddressPhase<F> {
    pub fn new(dimensions: BooleanityDimensions) -> Self {
        Self {
            symbolic: relations::booleanity::BooleanityAddressPhase::new(dimensions),
            dimensions,
            reference_draws: None,
        }
    }

    pub fn dimensions(&self) -> BooleanityDimensions {
        self.dimensions
    }

    /// Attach the hand pre-batch draws (the stage-2 `set_output_address_challenges`
    /// idiom: the batch is constructed before these transcript positions).
    pub fn set_reference_draws(&mut self, reference_draws: BooleanityReferenceDraws<F>) {
        self.reference_draws = Some(reference_draws);
    }

    pub fn reference_draws(&self) -> Option<&BooleanityReferenceDraws<F>> {
        self.reference_draws.as_ref()
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
