//! Typed inputs consumed and outputs produced by stage 1 verification.

use jolt_field::Field;
use jolt_sumcheck::{BatchedCommittedSumcheckConsistency, CommittedSumcheckConsistency};
use serde::{Deserialize, Serialize};

use super::outer_remainder::OuterRemainder;
use crate::stages::relations::SumcheckBatch;
use crate::stages::zk::outputs::CommittedOutputClaimOutput;
use crate::VerifierError;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: for<'a> Deserialize<'a>"))]
pub struct Stage1OutputClaims<F: Field> {
    pub uniskip_output_claim: F,
    pub outer: Stage1BatchOutputClaims<F>,
}

/// Source-of-truth for stage 1's singleton sumcheck batch: the Spartan outer
/// *remainder* sumcheck (the companion uni-skip first round is a separate
/// univariate-skip sub-sumcheck, not a batch member — verified inline in
/// `stage1::verify` against a literal-zero input claim).
/// `#[derive(SumcheckBatch)]` generates the `Stage1BatchInputClaims<F>` /
/// `Stage1BatchInputPoints<F>`, `Stage1BatchOutputClaims<F>` /
/// `Stage1BatchOutputPoints<F>`, and `Stage1BatchChallenges<F>` aggregates — one
/// field per instance, in this declaration order. With a single instance and no
/// cross-relation aliasing there is no `custom_opening_values` opt-out: the
/// generated `opening_values` / `append_to_transcript` delegates to
/// `OuterRemainderOutputClaims` in `dimensions.variables()` order (the canonical 35
/// R1CS-input order), byte-identical to the previous explicit append loop.
///
/// The full driver-flag set applies. `expected_final_claim` additionally requires
/// the member's late [`bind_coefficients`](OuterRemainder::bind_coefficients)
/// (its coefficient table depends on the batch's own bound point).
#[derive(SumcheckBatch)]
#[sumcheck_batch(
    verify_clear,
    verify_zk,
    derive_opening_points,
    expected_final_claim,
    output_shape
)]
pub struct Stage1BatchSumchecks<F: Field> {
    pub outer_remainder: OuterRemainder<F>,
}

/// The Fiat-Shamir values the verifier draws during stage 1: the irreducible
/// Spartan outer `tau` point and the uni-skip reduction challenge. Drawn
/// path-agnostically before the ZK/clear branch; carried in [`Stage1ZkOutput`]
/// so BlindFold can source `tau`/`uniskip` from `challenges.<field>` (matching
/// the `input.stageN.challenges.<field>` idiom used by the sibling stages). The
/// remainder sumcheck point is opening-derived, so it lives on the produced
/// reduction (clear: `output_points.outer_remainder`; ZK:
/// `remainder_consistency`) rather than here; the singleton remainder batching
/// coefficient is likewise
/// read from `remainder_consistency` on the ZK path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1Challenges<F: Field> {
    pub tau: Vec<F>,
    pub uniskip_challenge: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1ClearOutput<F: Field> {
    /// The produced remainder opening *values* (wire form). The opening point is
    /// derived from the remainder's sumcheck point; later stages read values through
    /// `.outer_remainder.<field>`.
    pub output_values: Stage1BatchOutputClaims<F>,
    /// The produced remainder opening *points*, paired field-for-field with
    /// `output_values`. All 35 openings share the single remainder point; the raw
    /// reduction point is exposed by [`Stage1Output::remainder_point`].
    pub output_points: Stage1BatchOutputPoints<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1ZkOutput<F: Field, C> {
    pub challenges: Stage1Challenges<F>,
    pub uniskip_consistency: CommittedSumcheckConsistency<F, C>,
    pub uniskip_output_claims: CommittedOutputClaimOutput<C>,
    pub remainder_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub remainder_output_claims: CommittedOutputClaimOutput<C>,
    /// The produced opening points (the *reversed* bound point on every cell), the
    /// ZK counterpart of the clear path's `output_points`. The raw un-reversed
    /// reduction point stays exposed by [`Stage1Output::remainder_point`].
    pub output_points: Stage1BatchOutputPoints<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage1Output<F: Field, C> {
    Clear(Stage1ClearOutput<F>),
    Zk(Stage1ZkOutput<F, C>),
}

impl<F: Field, C> Stage1Output<F, C> {
    /// The raw (un-reversed) Spartan outer remainder sumcheck reduction point,
    /// available regardless of proving mode. The remainder is a singleton batch, so
    /// the clear-path bound point and the ZK committed round challenges are the same
    /// vector. Downstream consumers (stage 2's `tau_low`, BlindFold's stage-1 cycle
    /// bindings) slice and reverse this point themselves, so it must NOT be the
    /// already-reversed opening point: the clear path stores the openings at the
    /// reversed point (`derive_opening_points`), so we reverse it back here to
    /// recover the raw reduction point the ZK `challenges()` returns directly. All 35
    /// stage-1 openings share this single reversed opening point, so `left_instruction_input`
    /// is a representative field accessor for it.
    pub fn remainder_point(&self) -> Vec<F> {
        match self {
            Self::Clear(output) => output
                .output_points
                .outer_remainder
                .left_instruction_input()
                .iter()
                .rev()
                .copied()
                .collect(),
            Self::Zk(output) => output.remainder_consistency.challenges(),
        }
    }

    pub fn clear(&self) -> Result<&Stage1ClearOutput<F>, VerifierError> {
        match self {
            Self::Clear(output) => Ok(output),
            Self::Zk(_) => Err(VerifierError::ExpectedClearProof { field: "stage1" }),
        }
    }

    pub fn zk(&self) -> Result<&Stage1ZkOutput<F, C>, VerifierError> {
        match self {
            Self::Zk(output) => Ok(output),
            Self::Clear(_) => Err(VerifierError::ExpectedCommittedProof { field: "stage1" }),
        }
    }
}
