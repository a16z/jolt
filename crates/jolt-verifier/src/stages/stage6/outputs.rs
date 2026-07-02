//! Typed inputs consumed and outputs produced by stage 6 verification.

use jolt_field::Field;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;
use serde::{Deserialize, Serialize};

use crate::stages::relations::SumcheckBatch;
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

// The per-relation produced-claim structs live in their relation modules
// (cell-generic, `#[derive(OutputClaims)]`); re-export them so consumers and the
// generated stage-6 aggregates keep resolving them through `stage6::outputs`.
pub use super::booleanity::{BooleanityAddressPhaseOutputClaims, BooleanityOutputClaims};
pub use super::bytecode_read_raf::{
    BytecodeReadRafAddressPhaseOutputClaims, BytecodeReadRafOutputClaims,
};
pub use super::committed_reduction_cycle_phase::{
    AdviceCyclePhaseOutputClaims, BytecodeReductionCyclePhaseOutputClaims,
    ProgramImageReductionCyclePhaseOutputClaims,
};
pub use super::inc_claim_reduction::IncClaimReductionOutputClaims;
pub use super::instruction_ra_virtualization::InstructionRaVirtualizationOutputClaims;
pub use super::ram_hamming_booleanity::RamHammingBooleanityOutputClaims;
pub use super::ram_ra_virtualization::RamRaVirtualizationOutputClaims;

use super::booleanity::{Booleanity, BooleanityAddressPhase};
use super::bytecode_read_raf::{BytecodeReadRafAddressPhase, BytecodeReadRafCycle};
use super::committed_reduction_cycle_phase::{
    AdviceCyclePhase, BytecodeReductionCyclePhase, ProgramImageReductionCyclePhase,
};
use super::inc_claim_reduction::IncClaimReduction;
use super::instruction_ra_virtualization::InstructionRaVirtualization;
use super::ram_hamming_booleanity::RamHammingBooleanity;
use super::ram_ra_virtualization::RamRaVirtualization;

/// Source-of-truth for stage 6a's two-instance address-phase sumcheck batch
/// (bytecode read-RAF, booleanity). `#[derive(SumcheckBatch)]` generates the
/// `Stage6AddressPhase{Input,Output}{Claims,Points}<F>` and
/// `Stage6AddressPhaseChallenges<F>` aggregates — one field per instance, in
/// this declaration order — plus the batched-verify drivers. No alias dedup in
/// the address phase, so the generated `opening_values` / `append_to_transcript`
/// (member order: bytecode read-RAF's `intermediate` then `val_stages`, then
/// booleanity's `intermediate`) is the canonical Fiat-Shamir order.
///
/// `output_shape` is intentionally NOT enabled: the address-phase output `Expr`
/// carries only the staged intermediate, but committed-program mode additionally
/// commits the `BytecodeValStage` openings, so the ZK commitment count
/// (`2 + NUM_BYTECODE_VAL_STAGES`) and the val-stage presence check stay
/// hand-written in `verify`.
#[derive(SumcheckBatch)]
#[sumcheck_batch(verify_clear, verify_zk, derive_opening_points, expected_final_claim)]
pub struct Stage6AddressPhaseSumchecks<F: Field> {
    pub bytecode_read_raf: BytecodeReadRafAddressPhase<F>,
    pub booleanity: BooleanityAddressPhase<F>,
}

/// The stage-6a produced opening points: both intermediates open at the
/// (reversed) address sumcheck point of their instance.
impl<F: Field> Stage6AddressPhaseOutputPoints<F> {
    /// The bytecode read-RAF address opening (`bytecode_r_address`).
    pub fn bytecode_r_address(&self) -> &[F] {
        &self.bytecode_read_raf.intermediate
    }

    /// The booleanity address opening (`booleanity_r_address`).
    pub fn booleanity_r_address(&self) -> &[F] {
        &self.booleanity.intermediate
    }
}

/// Source-of-truth for stage 6b's cycle-phase sumcheck batch, in canonical
/// Fiat-Shamir batch order. `#[derive(SumcheckBatch)]` generates the
/// `Stage6CyclePhase{Input,Output}{Claims,Points}<F>` and
/// `Stage6CyclePhaseChallenges<F>` aggregates — one field per instance, in this
/// declaration order — plus the batched-verify drivers. The four `Option`
/// members are present exactly when their precommitted layout is committed, in
/// BOTH proving modes, so the coefficient count matches the prover's instance
/// count.
///
/// `bytecode_read_raf` is the runtime dispatch [`BytecodeReadRafCycle`], whose
/// `ConcreteSumcheck` impl is anchored on the committed cycle symbolic (see the
/// invariant on that impl); the aggregates project through the anchor, which both
/// variants share cell-for-cell.
///
/// The generated `draw_challenges` is suppressed (`no_draw_challenges`): the
/// members' challenges have stage-level provenance (the bytecode gamma shares
/// stage 6a's squeeze, the booleanity gamma is drawn pre-6a with a
/// prover-matched zero-replacement, and the instruction-RA gamma keeps
/// `powers(n)[1].unwrap_or(one)`), so `verify` hand-assembles
/// `Stage6CyclePhaseChallenges` from the stage-level draws — a generated
/// per-member draw would squeeze at the wrong transcript position if it
/// existed to be called.
///
/// The opt-out `#[sumcheck_batch(custom_opening_values)]` suppresses the generated
/// `opening_values` / `append_to_transcript`: booleanity's `bytecode_ra` openings
/// alias the bytecode-read-RAF points and must NOT be re-absorbed, so the canonical
/// order is curated by [`append_opening_claims`](super::verify::append_opening_claims)
/// which threads the dedup points. `output_shape` is NOT applicable: the committed
/// bytecode output `Expr` consumes the 6a-produced `BytecodeValStage` openings
/// (not 6b outputs), and the ZK commitment count dedups runtime point aliases.
#[derive(SumcheckBatch)]
#[sumcheck_batch(
    custom_opening_values,
    no_draw_challenges,
    verify_clear,
    verify_zk,
    derive_opening_points,
    expected_final_claim
)]
pub struct Stage6CyclePhaseSumchecks<F: Field> {
    pub bytecode_read_raf: BytecodeReadRafCycle<F>,
    pub booleanity: Booleanity<F>,
    pub ram_hamming_booleanity: RamHammingBooleanity<F>,
    pub ram_ra_virtualization: RamRaVirtualization<F>,
    pub instruction_ra_virtualization: InstructionRaVirtualization<F>,
    pub inc_claim_reduction: IncClaimReduction<F>,
    pub trusted_advice: Option<AdviceCyclePhase<F>>,
    pub untrusted_advice: Option<AdviceCyclePhase<F>>,
    pub bytecode_reduction: Option<BytecodeReductionCyclePhase<F>>,
    pub program_image_reduction: Option<ProgramImageReductionCyclePhase<F>>,
}

/// The stage 6 produced opening *values* (wire form). Combines the stage-6a
/// address-phase aggregate and the stage-6b cycle-phase aggregate so the single
/// serialized stage-6 proof field stays byte-identical: address-phase openings
/// absorbed in 6a, cycle-phase openings absorbed in 6b.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct Stage6OutputClaims<F: Field> {
    pub address_phase: Stage6AddressPhaseOutputClaims<F>,
    pub cycle_phase: Stage6CyclePhaseOutputClaims<F>,
}

/// The stage 6 produced opening *points* (point-only form), paired field-for-field
/// with [`Stage6OutputClaims`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6OutputPoints<F: Field> {
    pub address_phase: Stage6AddressPhaseOutputPoints<F>,
    pub cycle_phase: Stage6CyclePhaseOutputPoints<F>,
}

/// Opening-point accessors over the point-only form of the stage-6 produced claims.
/// Stages 7 and 8 read each relation's produced opening point off these cells,
/// single-sourcing the points that the retired `VerifiedStage6Batch` used to carry.
/// The per-cycle-phase reduction's `cycle_phase_variables` are recovered as
/// `reverse(opening_point)` (see `cycle_phase_opening_point` in `jolt-claims`
/// `claim_reductions::precommitted`).
impl<F: Field> Stage6OutputPoints<F> {
    /// The shared booleanity opening point (`r_address ++ r_cycle`); every
    /// produced booleanity RA opening uses it. `None` only if booleanity produced
    /// no openings (never in practice — at least one RA family is always present).
    pub fn booleanity_opening_point(&self) -> Option<&[F]> {
        self.cycle_phase
            .booleanity
            .instruction_ra
            .first()
            .or_else(|| self.cycle_phase.booleanity.bytecode_ra.first())
            .or_else(|| self.cycle_phase.booleanity.ram_ra.first())
            .map(Vec::as_slice)
    }

    /// The increment claim-reduction opening point (the reversed cycle point shared
    /// by the `RamInc`/`RdInc` reduced openings).
    pub fn inc_opening_point(&self) -> &[F] {
        &self.cycle_phase.inc_claim_reduction.ram_inc
    }

    /// The advice cycle-phase opening point for `kind`, present only when that
    /// advice reduction ran a cycle phase.
    pub fn advice_cycle_phase_opening_point(
        &self,
        kind: jolt_claims::protocols::jolt::JoltAdviceKind,
    ) -> Option<&[F]> {
        use jolt_claims::protocols::jolt::JoltAdviceKind;
        let member = match kind {
            JoltAdviceKind::Trusted => self.cycle_phase.trusted_advice.as_ref()?,
            JoltAdviceKind::Untrusted => self.cycle_phase.untrusted_advice.as_ref()?,
        };
        let opening = match kind {
            JoltAdviceKind::Trusted => member.trusted.as_ref(),
            JoltAdviceKind::Untrusted => member.untrusted.as_ref(),
        }?;
        Some(opening.as_slice())
    }

    /// The program-image claim-reduction cycle-phase opening point, present only in
    /// committed-program mode when the reduction ran a cycle phase.
    pub fn program_image_opening_point(&self) -> Option<&[F]> {
        self.cycle_phase
            .program_image_reduction
            .as_ref()
            .map(|claim| claim.program_image.as_slice())
    }

    /// The bytecode claim-reduction cycle-phase opening point, present only in
    /// committed-program mode. Every produced chunk (or the intermediate) shares
    /// the single cycle opening point, so the first cell is canonical.
    pub fn bytecode_reduction_opening_point(&self) -> Option<&[F]> {
        let reduction = self.cycle_phase.bytecode_reduction.as_ref()?;
        match &reduction.intermediate {
            Some(point) => Some(point.as_slice()),
            None => reduction.chunks.first().map(Vec::as_slice),
        }
    }

    /// The advice cycle-phase `cycle_phase_variables` for `kind`: the raw active
    /// cycle challenges, recovered as `reverse(opening_point)` (the cycle opening
    /// point is the reverse of the variable challenges). Stage 7's address phase
    /// reconstructs its opening point from these.
    pub fn advice_cycle_phase_variables(
        &self,
        kind: jolt_claims::protocols::jolt::JoltAdviceKind,
    ) -> Option<Vec<F>> {
        Some(reversed(self.advice_cycle_phase_opening_point(kind)?))
    }

    /// The program-image cycle-phase `cycle_phase_variables` (`reverse(opening_point)`).
    pub fn program_image_cycle_phase_variables(&self) -> Option<Vec<F>> {
        Some(reversed(self.program_image_opening_point()?))
    }

    /// The bytecode-reduction cycle-phase `cycle_phase_variables` (`reverse(opening_point)`).
    pub fn bytecode_cycle_phase_variables(&self) -> Option<Vec<F>> {
        Some(reversed(self.bytecode_reduction_opening_point()?))
    }
}

fn reversed<F: Field>(point: &[F]) -> Vec<F> {
    point.iter().rev().copied().collect()
}

/// The Fiat-Shamir challenges the verifier draws during stage 6: the batching
/// gammas for the bytecode read-RAF address fold and each upstream stage, the
/// booleanity reference point and gamma, the instruction-RA and increment
/// gammas, and (committed-program only) the bytecode claim-reduction `eta`. The
/// two `booleanity_reference_*` points ride here rather than in an Outputs cell
/// because the address half is padded with a fresh `transcript.challenge_vector`
/// draw that lives nowhere else, so it is not recoverable from any opening cell.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6Challenges<F: Field> {
    pub bytecode_gamma_powers: Vec<F>,
    pub stage1_gammas: Vec<F>,
    pub stage2_gammas: Vec<F>,
    pub stage3_gammas: Vec<F>,
    pub stage4_gammas: Vec<F>,
    pub stage5_gammas: Vec<F>,
    pub booleanity_reference_address: Vec<F>,
    pub booleanity_reference_cycle: Vec<F>,
    pub booleanity_gamma: F,
    pub instruction_ra_gamma_powers: Vec<F>,
    pub inc_gamma: F,
    /// Committed program mode only: bytecode claim-reduction batching
    /// challenge (the prover's `eta`).
    pub bytecode_reduction_eta: Option<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6ClearOutput<F: Field> {
    /// The produced opening *values* (wire form); read by later stages and the
    /// Fiat-Shamir opening-claim encoder.
    pub output_values: Stage6OutputClaims<F>,
    /// The produced opening *points*, paired field-for-field with `output_values`.
    /// Stages 7 and 8 read each relation's opening point off these cells (via the
    /// `Stage6OutputPoints<F>` accessors).
    pub output_points: Stage6OutputPoints<F>,
    /// Committed-program mode only: the bytecode claim-reduction's per-chunk
    /// weights (`r_bc`, chunk weights, gamma-folded lane weights). These are
    /// public derived data (not openings), so stage 7's bytecode address phase
    /// reads them here rather than recomputing them.
    pub bytecode_reduction_weights: Option<BytecodeReductionWeights<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6ZkOutput<F: Field, C> {
    pub challenges: Stage6Challenges<F>,
    pub address_phase_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub address_phase_output_claims: CommittedOutputClaimOutput<C>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    /// The produced opening *points*, the ZK counterpart of the clear path's
    /// `Stage6ClearOutput::output_points`. Stages 7/8 and BlindFold read each
    /// relation's opening point off these cells through the same `Stage6OutputPoints<F>`
    /// accessors. (BlindFold recomputes the bytecode reduction weights locally, so
    /// the ZK output carries no weights aux.)
    pub output_points: Stage6OutputPoints<F>,
}

// The clear variant carries the located opening claims read on the hot path; the
// ZK variant carries committed consistency plus the point-only `output_points`.
// Boxing the common clear variant to shrink the rarer ZK one would add indirection
// to every clear-path access.
#[expect(
    clippy::large_enum_variant,
    reason = "clear variant holds the located opening claims read on the hot path; boxing it would penalize the common case"
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage6Output<F: Field, C> {
    Clear(Stage6ClearOutput<F>),
    Zk(Stage6ZkOutput<F, C>),
}

impl<F: Field, C> Stage6Output<F, C> {
    /// The produced opening *points*, available regardless of proving mode.
    pub fn output_points(&self) -> &Stage6OutputPoints<F> {
        match self {
            Self::Clear(output) => &output.output_points,
            Self::Zk(output) => &output.output_points,
        }
    }

    pub fn clear(&self) -> Result<&Stage6ClearOutput<F>, crate::VerifierError> {
        match self {
            Self::Clear(output) => Ok(output),
            Self::Zk(_) => Err(crate::VerifierError::ExpectedClearProof { field: "stage6" }),
        }
    }

    pub fn zk(&self) -> Result<&Stage6ZkOutput<F, C>, crate::VerifierError> {
        match self {
            Self::Zk(output) => Ok(output),
            Self::Clear(_) => Err(crate::VerifierError::ExpectedCommittedProof { field: "stage6" }),
        }
    }
}

/// Public bytecode claim-reduction state shared by the cycle and address
/// phases: the per-chunk weights over dropped address bits, the chunk-local
/// cycle point, and the gamma-folded lane weights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReductionWeights<F: Field> {
    pub r_bc: Vec<F>,
    pub chunk_rbc_weights: Vec<F>,
    pub lane_weights: Vec<F>,
}
