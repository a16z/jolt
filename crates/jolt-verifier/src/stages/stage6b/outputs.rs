//! Typed inputs consumed and outputs produced by stage 6b (cycle-phase)
//! verification.

use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::BytecodeOutputWeightInputs;
use jolt_field::Field;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::relations::SumcheckBatch;
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

// The per-relation produced-claim structs live in their relation modules
// (cell-generic, `#[derive(OutputClaims)]`); re-export them so consumers and the
// generated stage-6b aggregates keep resolving them through `stage6b::outputs`.
pub use super::booleanity::BooleanityOutputClaims;
pub use super::bytecode_read_raf::BytecodeReadRafOutputClaims;
pub use super::committed_reduction_cycle_phase::{
    BytecodeReductionCyclePhaseOutputClaims, ProgramImageReductionCyclePhaseOutputClaims,
    TrustedAdviceCyclePhaseOutputClaims, UntrustedAdviceCyclePhaseOutputClaims,
};
pub use super::inc_claim_reduction::IncClaimReductionOutputClaims;
pub use super::instruction_ra_virtualization::InstructionRaVirtualizationOutputClaims;
pub use super::ram_hamming_booleanity::RamHammingBooleanityOutputClaims;
pub use super::ram_ra_virtualization::RamRaVirtualizationOutputClaims;

use super::booleanity::Booleanity;
use super::bytecode_read_raf::BytecodeReadRafCycle;
use super::committed_reduction_cycle_phase::{
    BytecodeReductionCyclePhase, ProgramImageReductionCyclePhase, TrustedAdviceCyclePhase,
    UntrustedAdviceCyclePhase,
};
use super::inc_claim_reduction::IncClaimReduction;
use super::instruction_ra_virtualization::InstructionRaVirtualization;
use super::ram_hamming_booleanity::RamHammingBooleanity;
use super::ram_ra_virtualization::RamRaVirtualization;

/// Source-of-truth for stage 6b's cycle-phase sumcheck batch, in canonical
/// Fiat-Shamir batch order. `#[derive(SumcheckBatch)]` generates the
/// `Stage6b{Input,Output}{Claims,Points}<F>` and `Stage6bChallenges<F>`
/// aggregates — one field per instance, in this declaration order — plus the
/// batched-verify drivers. The four `Option` members are present exactly when
/// their precommitted layout is committed, in BOTH proving modes, so the
/// coefficient count matches the prover's instance count.
///
/// `bytecode_read_raf` is the runtime dispatch [`BytecodeReadRafCycle`], whose
/// `ConcreteSumcheck` impl is anchored on the committed cycle symbolic (see the
/// invariant on that impl); the aggregates project through the anchor, which both
/// variants share cell-for-cell.
///
/// The generated `draw_challenges` is suppressed (`no_draw_challenges`): the
/// members' challenges have stage-level provenance (the bytecode gamma shares
/// stage 6a's squeeze and the booleanity gamma is drawn pre-6a where the
/// prover's booleanity subprotocol samples it), so `verify` hand-assembles
/// `Stage6bChallenges` from the stage-level draws — a generated per-member draw
/// would squeeze at the wrong transcript position if it existed to be called.
///
/// The opt-out `#[sumcheck_batch(custom_opening_values)]` suppresses the generated
/// absorb methods: booleanity's `bytecode_ra` openings
/// alias the bytecode-read-RAF points and must NOT be re-absorbed, so the canonical
/// order is curated by [`append_opening_claims`](super::verify::append_opening_claims)
/// which threads the dedup points. `output_shape` is NOT applicable: the committed
/// bytecode output `Expr` consumes the 6a-produced `BytecodeValStage` openings
/// (not 6b outputs), and the ZK commitment count dedups runtime point aliases.
#[derive(SumcheckBatch)]
#[sumcheck_batch(custom_opening_values, no_draw_challenges)]
pub struct Stage6bSumchecks<F: Field> {
    pub bytecode_read_raf: BytecodeReadRafCycle<F>,
    pub booleanity: Booleanity<F>,
    pub ram_hamming_booleanity: RamHammingBooleanity<F>,
    pub ram_ra_virtualization: RamRaVirtualization<F>,
    pub instruction_ra_virtualization: InstructionRaVirtualization<F>,
    pub inc_claim_reduction: IncClaimReduction<F>,
    pub trusted_advice: Option<TrustedAdviceCyclePhase<F>>,
    pub untrusted_advice: Option<UntrustedAdviceCyclePhase<F>>,
    pub bytecode_reduction: Option<BytecodeReductionCyclePhase<F>>,
    pub program_image_reduction: Option<ProgramImageReductionCyclePhase<F>>,
}

/// Opening-point accessors over the point-only form of the stage-6b produced
/// claims. Stages 7 and 8 read each relation's produced opening point off these
/// cells. The per-reduction `cycle_phase_variables` are recovered as
/// `reverse(opening_point)` (see `cycle_phase_opening_point` in `jolt-claims`
/// `claim_reductions::precommitted`).
impl<F: Field> Stage6bOutputPoints<F> {
    /// The shared booleanity opening point (`r_address ++ r_cycle`); every
    /// produced booleanity RA opening uses it. `None` only if booleanity produced
    /// no openings (never in practice — at least one RA family is always present).
    pub fn booleanity_opening_point(&self) -> Option<&[F]> {
        self.booleanity
            .instruction_ra
            .first()
            .or_else(|| self.booleanity.bytecode_ra.first())
            .or_else(|| self.booleanity.ram_ra.first())
            .map(Vec::as_slice)
    }

    /// The increment claim-reduction opening point (the reversed cycle point shared
    /// by the `RamInc`/`RdInc` reduced openings).
    pub fn inc_opening_point(&self) -> &[F] {
        &self.inc_claim_reduction.ram_inc
    }

    /// The advice cycle-phase opening point for `kind`, present only when that
    /// advice reduction ran a cycle phase.
    pub fn advice_cycle_phase_opening_point(
        &self,
        kind: jolt_claims::protocols::jolt::JoltAdviceKind,
    ) -> Option<&[F]> {
        use jolt_claims::protocols::jolt::JoltAdviceKind;
        match kind {
            JoltAdviceKind::Trusted => self.trusted_advice.as_ref().map(|claims| claims.trusted()),
            JoltAdviceKind::Untrusted => self
                .untrusted_advice
                .as_ref()
                .map(|claims| claims.untrusted()),
        }
    }

    /// The program-image claim-reduction cycle-phase opening point, present only in
    /// committed-program mode when the reduction ran a cycle phase.
    pub fn program_image_opening_point(&self) -> Option<&[F]> {
        self.program_image_reduction
            .as_ref()
            .map(|claim| claim.program_image.as_slice())
    }

    /// The bytecode claim-reduction cycle-phase opening point, present only in
    /// committed-program mode. Every produced chunk (or the intermediate) shares
    /// the single cycle opening point, so the first cell is canonical.
    pub fn bytecode_reduction_opening_point(&self) -> Option<&[F]> {
        let reduction = self.bytecode_reduction.as_ref()?;
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

    /// The total number of produced opening-point cells across every member. This
    /// is the derived, layout-independent claim count; the ZK path subtracts its
    /// runtime bytecode/booleanity point-alias dedup from it to size the committed
    /// output claims.
    pub fn point_count(&self) -> usize {
        self.bytecode_read_raf.bytecode_ra.len()
            + self.booleanity.instruction_ra.len()
            + self.booleanity.bytecode_ra.len()
            + self.booleanity.ram_ra.len()
            + 1
            + self.ram_ra_virtualization.ram_ra.len()
            + self
                .instruction_ra_virtualization
                .committed_instruction_ra
                .len()
            + 2
            + usize::from(self.trusted_advice.is_some())
            + usize::from(self.untrusted_advice.is_some())
            + self.bytecode_reduction.as_ref().map_or(0, |reduction| {
                usize::from(reduction.intermediate.is_some()) + reduction.chunks.len()
            })
            + usize::from(self.program_image_reduction.is_some())
    }
}

impl<F: Field> Stage6bOutputClaims<F> {
    /// The consumed cycle-phase advice opening *value* for `kind` (the trusted /
    /// untrusted slot of that advice member), present only when the advice
    /// reduction ran a cycle phase. Read by stage 7's advice input wiring and stage
    /// 8's precommitted finals resolution.
    pub fn advice_cycle_phase_claim(
        &self,
        kind: jolt_claims::protocols::jolt::JoltAdviceKind,
    ) -> Option<F> {
        use jolt_claims::protocols::jolt::JoltAdviceKind;
        match kind {
            JoltAdviceKind::Trusted => self.trusted_advice.as_ref().map(|claim| claim.trusted),
            JoltAdviceKind::Untrusted => {
                self.untrusted_advice.as_ref().map(|claim| claim.untrusted)
            }
        }
    }
}

fn reversed<F: Field>(point: &[F]) -> Vec<F> {
    point.iter().rev().copied().collect()
}

/// The stage-6b Fiat-Shamir challenges drawn after the stage-6a batch: the
/// instruction-RA and increment gammas, and (committed-program only) the bytecode
/// claim-reduction `eta`. Kept as field names greppable from BlindFold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6bCarriedChallenges<F: Field> {
    pub instruction_ra_gamma: F,
    pub inc_gamma: F,
    /// Committed program mode only: bytecode claim-reduction batching
    /// challenge (the prover's `eta`).
    pub bytecode_reduction_eta: Option<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6bClearOutput<F: Field> {
    /// The produced opening *values* (wire form); read by later stages and the
    /// Fiat-Shamir opening-claim encoder.
    pub output_values: Stage6bOutputClaims<F>,
    /// The produced opening *points*, paired field-for-field with `output_values`.
    /// Stages 7 and 8 read each relation's opening point off these cells (via the
    /// `Stage6bOutputPoints<F>` accessors).
    pub output_points: Stage6bOutputPoints<F>,
    /// Committed-program mode only: the bytecode claim-reduction's per-chunk
    /// weights (`r_bc`, chunk weights, gamma-folded lane weights). These are
    /// public derived data (not openings), so stage 7's bytecode address phase
    /// reads them here rather than recomputing them.
    pub bytecode_reduction_weights: Option<BytecodeReductionWeights<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6bZkOutput<F: Field, C> {
    pub challenges: Stage6bCarriedChallenges<F>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    /// The produced opening *points*, the ZK counterpart of the clear path's
    /// `Stage6bClearOutput::output_points`. Stages 7/8 and BlindFold read each
    /// relation's opening point off these cells through the same
    /// `Stage6bOutputPoints<F>` accessors. (BlindFold recomputes the bytecode
    /// reduction weights locally, so the ZK output carries no weights aux.)
    pub output_points: Stage6bOutputPoints<F>,
}

// The clear variant carries the located opening claims read on the hot path; the
// ZK variant carries committed consistency plus the point-only `output_points`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage6bOutput<F: Field, C> {
    Clear(Stage6bClearOutput<F>),
    Zk(Stage6bZkOutput<F, C>),
}

impl<F: Field, C> Stage6bOutput<F, C> {
    /// The produced opening *points*, available regardless of proving mode.
    pub fn output_points(&self) -> &Stage6bOutputPoints<F> {
        match self {
            Self::Clear(output) => &output.output_points,
            Self::Zk(output) => &output.output_points,
        }
    }

    pub fn clear(&self) -> Result<&Stage6bClearOutput<F>, crate::VerifierError> {
        match self {
            Self::Clear(output) => Ok(output),
            Self::Zk(_) => Err(crate::VerifierError::ExpectedClearProof { field: "stage6b" }),
        }
    }

    pub fn zk(&self) -> Result<&Stage6bZkOutput<F, C>, crate::VerifierError> {
        match self {
            Self::Zk(output) => Ok(output),
            Self::Clear(_) => {
                Err(crate::VerifierError::ExpectedCommittedProof { field: "stage6b" })
            }
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

impl<F: Field> BytecodeReductionWeights<F> {
    /// Borrow the weights as the jolt-claims `BytecodeOutputWeightInputs` the
    /// bytecode reduction's output-weight publics resolve against.
    pub(crate) fn as_inputs(&self) -> BytecodeOutputWeightInputs<'_, F> {
        BytecodeOutputWeightInputs {
            r_bc: &self.r_bc,
            chunk_rbc_weights: &self.chunk_rbc_weights,
            lane_weights: &self.lane_weights,
        }
    }
}
