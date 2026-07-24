//! Typed inputs consumed and outputs produced by stage 1 verification.

use jolt_claims::protocols::jolt::JoltRelationId;
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
/// cross-relation aliasing there is no `no_opening_values` opt-out: the
/// generated absorb (`opening_values` / `append_output_claims` on this struct)
/// delegates to `OuterRemainderOutputClaims` in `dimensions.variables()` order
/// (the canonical 35 R1CS-input order), byte-identical to the previous explicit
/// append loop.
///
/// The member's `SpartanOuterPublic` coefficient table depends on the batch's own
/// bound point, so it completes itself lazily: `derive_opening_points` captures
/// the point and the first `derive_output_term` call builds the table.
#[derive(SumcheckBatch)]
#[sumcheck_batch(crate = "crate")]
pub struct Stage1BatchSumchecks<F: Field> {
    /// On the prove side the remainder kernel is minted from the state the
    /// uni-skip slot parked in the proof session, through its regular
    /// universal backend slot.
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

impl<F: Field> Stage1ClearOutput<F> {
    /// The raw (un-reversed) Spartan outer remainder reduction point: the
    /// clear path stores the openings at the REVERSED point
    /// (`derive_opening_points`), so this reverses it back. All 35 stage-1
    /// openings share the point; `left_instruction_input` is a representative
    /// accessor. Promoted so the prove-side recipes stop hand-copying the
    /// derivation (see [`Stage1Output::remainder_point`]).
    pub fn remainder_point(&self) -> Vec<F> {
        self.output_points
            .outer_remainder
            .left_instruction_input()
            .iter()
            .rev()
            .copied()
            .collect()
    }

    /// The stage-1 Spartan-outer cycle binding: the tail (`[1..]`) of the raw
    /// [`remainder_point`](Self::remainder_point), or `None` when that point
    /// is empty.
    pub fn cycle_binding(&self) -> Option<Vec<F>> {
        let raw_point = self.remainder_point();
        let (_, cycle) = raw_point.split_first()?;
        Some(cycle.to_vec())
    }

    /// [`cycle_binding`](Self::cycle_binding), attributing an empty remainder
    /// point to the consuming `stage`.
    pub fn cycle_binding_checked(&self, stage: JoltRelationId) -> Result<Vec<F>, VerifierError> {
        self.cycle_binding()
            .ok_or_else(|| empty_remainder_point(stage))
    }
}

fn empty_remainder_point(stage: JoltRelationId) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage,
        reason: "Stage 1 remainder point is empty".to_string(),
    }
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
            Self::Clear(output) => output.remainder_point(),
            Self::Zk(output) => output.remainder_consistency.challenges(),
        }
    }

    /// The stage-1 Spartan-outer cycle binding: the tail (`[1..]`) of the raw
    /// (un-reversed) [`remainder_point`](Self::remainder_point), or `None` when
    /// that point is empty. This matches the ZK/BlindFold path, which slices
    /// `remainder_consistency.challenges()[1..]` off the same raw point.
    pub fn cycle_binding(&self) -> Option<Vec<F>> {
        let raw_point = self.remainder_point();
        let (_, cycle) = raw_point.split_first()?;
        Some(cycle.to_vec())
    }

    /// [`cycle_binding`](Self::cycle_binding), attributing an empty remainder
    /// point to the consuming `stage`.
    pub fn cycle_binding_checked(&self, stage: JoltRelationId) -> Result<Vec<F>, VerifierError> {
        self.cycle_binding()
            .ok_or_else(|| empty_remainder_point(stage))
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
