//! The stage 6a booleanity address-phase sumcheck instance.
//!
//! Booleanity proves every one-hot `Ra` chunk (instruction, bytecode, RAM) is
//! boolean. It runs in two phases: this stage-6a address phase binds the
//! `log_k_chunk` address variables and stages the `BooleanityAddrClaim`
//! intermediate consumed by the stage-6b cycle phase.

use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::booleanity::{
    BooleanityAddressPhaseChallenges, BooleanityAddressPhaseInputClaims,
    BooleanityAddressPhaseOutputClaims,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_transcript::Transcript;

use crate::stages::relations::ConcreteSumcheck;
use crate::VerifierError;

#[derive(Clone)]
pub struct BooleanityAddressPhase<F: Field> {
    symbolic: relations::booleanity::BooleanityAddressPhase,
    dimensions: BooleanityDimensions,
    /// The stage-5 instruction read-RAF opening points (big-endian) the
    /// reference draws derive from: `draw_challenges` reverses them into the
    /// little-endian reference address/cycle (the same construction-geometry
    /// idiom as `BytecodeReadRafAddressPhase`'s `BytecodeStagePoints`).
    instruction_r_address: Vec<F>,
    instruction_r_cycle: Vec<F>,
}

impl<F: Field> BooleanityAddressPhase<F> {
    pub fn new(
        dimensions: BooleanityDimensions,
        instruction_r_address: Vec<F>,
        instruction_r_cycle: Vec<F>,
    ) -> Self {
        Self {
            symbolic: relations::booleanity::BooleanityAddressPhase::new(dimensions),
            dimensions,
            instruction_r_address,
            instruction_r_cycle,
        }
    }

    pub fn dimensions(&self) -> BooleanityDimensions {
        self.dimensions
    }
}

impl<F: Field> ConcreteSumcheck<F> for BooleanityAddressPhase<F> {
    type Symbolic = relations::booleanity::BooleanityAddressPhase;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    /// Draws the booleanity pre-batch challenges at the frozen wire positions:
    /// the reference address is the reversed stage-5 instruction address,
    /// padded with a fresh `challenge_vector` draw or truncated to the
    /// committed chunk width (`log_k_chunk`); the reference cycle is the
    /// reversed stage-5 instruction cycle (no draw of its own); then the
    /// batching gamma. This member is declared after the bytecode read-RAF one
    /// in `Stage6aSumchecks`, so the generated aggregate draw lands these
    /// squeezes exactly where the hand pre-batch block drew them — after the
    /// bytecode member's six gammas.
    ///
    /// MUST stay `challenge_vector` + `challenge()` (not `challenge_scalar`):
    /// both decode the same 16-byte squeeze, but differently, so switching
    /// would silently change the reference/gamma values without changing the
    /// transcript bytes.
    fn draw_challenges<T: Transcript<Challenge = F>>(
        &self,
        transcript: &mut T,
    ) -> Result<BooleanityAddressPhaseChallenges<F>, VerifierError> {
        let chunk_bits = self.dimensions.log_k_chunk;
        let mut reference_address: Vec<F> =
            self.instruction_r_address.iter().rev().copied().collect();
        if reference_address.len() < chunk_bits {
            let missing = chunk_bits - reference_address.len();
            reference_address.extend(transcript.challenge_vector(missing));
        } else {
            reference_address = reference_address[reference_address.len() - chunk_bits..].to_vec();
        }
        Ok(BooleanityAddressPhaseChallenges {
            reference_address,
            reference_cycle: self.instruction_r_cycle.iter().rev().copied().collect(),
            gamma: transcript.challenge(),
        })
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
