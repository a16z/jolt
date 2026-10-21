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
use jolt_field::JoltField;
use jolt_transcript::FieldTranscript;

use crate::stages::relations::ConcreteSumcheck;
use crate::VerifierError;

#[derive(Clone)]
pub struct BooleanityAddressPhase<F: JoltField> {
    symbolic: relations::booleanity::BooleanityAddressPhase,
    dimensions: BooleanityDimensions,
    /// The stage-5 instruction read-RAF opening points (big-endian) the
    /// reference points derive from: `draw_challenges` reverses the address
    /// into the little-endian reference address,
    /// [`reference_cycle`](Self::reference_cycle) the cycle (the same
    /// construction-geometry idiom as `BytecodeReadRafAddressPhase`'s
    /// `BytecodeStagePoints`).
    instruction_r_address: Vec<F>,
    instruction_r_cycle: Vec<F>,
}

impl<F: JoltField> BooleanityAddressPhase<F> {
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

    /// The little-endian reference cycle the booleanity legs compare against:
    /// the reversed stage-5 instruction cycle. Pure construction geometry (no
    /// draw of its own), so it lives on the instance rather than in the
    /// challenges struct; the address-phase kernel and the stage-6b monolith
    /// consume it.
    pub fn reference_cycle(&self) -> Vec<F> {
        self.instruction_r_cycle.iter().rev().copied().collect()
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for BooleanityAddressPhase<F> {
    type Symbolic = relations::booleanity::BooleanityAddressPhase;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    /// The fresh reference-address coordinates enter a multilinear zero-check,
    /// and the gamma folds the booleanity openings with even powers up to
    /// `2 (N - 1)`.
    fn challenge_loss(&self) -> u64 {
        let missing = self
            .dimensions
            .log_k_chunk
            .saturating_sub(self.instruction_r_address.len());
        #[cfg(not(feature = "akita"))]
        let openings =
            jolt_claims::protocols::jolt::geometry::booleanity::booleanity_output_openings(
                self.dimensions.layout,
            )
            .len();
        #[cfg(feature = "akita")]
        let openings = {
            use jolt_claims::protocols::jolt::lattice::relations::booleanity::{
                lattice_booleanity_output_openings, LatticeBooleanityDimensions,
            };
            // An unpackable shape fails closed: no difficulty fits this loss.
            let Ok(dimensions) = LatticeBooleanityDimensions::new(self.dimensions) else {
                return u64::MAX;
            };
            lattice_booleanity_output_openings(dimensions).len()
        };
        crate::num::u64_from_usize(missing).saturating_add(
            crate::num::u64_from_usize(openings.saturating_sub(1)).saturating_mul(2),
        )
    }

    /// Draws the booleanity pre-batch challenges at the frozen wire positions:
    /// the reference address is the reversed stage-5 instruction address,
    /// padded with a fresh `challenge_vector` draw or truncated to the
    /// committed chunk width (`log_k_chunk`); then the batching gamma. This
    /// member is declared after the bytecode read-RAF one in
    /// `Stage6aSumchecks`, so the generated aggregate draw lands these
    /// squeezes exactly where the hand pre-batch block drew them — after the
    /// bytecode member's six gammas.
    ///
    /// MUST stay `challenge_vector` + `challenge()` (not `challenge_scalar`):
    /// both decode the same 16-byte squeeze, but differently, so switching
    /// would silently change the reference/gamma values without changing the
    /// transcript bytes.
    fn draw_challenges<T: FieldTranscript<F>>(
        &self,
        transcript: &mut T,
    ) -> Result<BooleanityAddressPhaseChallenges<F>, VerifierError> {
        let chunk_bits = self.dimensions.log_k_chunk;
        let mut reference_address: Vec<F> =
            self.instruction_r_address.iter().rev().copied().collect();
        // Both subtractions are exact under the branch guard; `saturating_sub`
        // only settles the (unreachable) underflow for the arithmetic lint.
        if reference_address.len() < chunk_bits {
            let missing = chunk_bits.saturating_sub(reference_address.len());
            reference_address.extend(transcript.challenge_vector(missing)?);
        } else {
            let excess = reference_address.len().saturating_sub(chunk_bits);
            reference_address = reference_address.split_off(excess);
        }
        Ok(BooleanityAddressPhaseChallenges {
            reference_address,
            gamma: transcript.challenge()?,
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
