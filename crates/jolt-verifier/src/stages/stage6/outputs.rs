//! Typed inputs consumed and outputs produced by stage 6 verification.

use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;
use jolt_claims::protocols::jolt::JoltAdviceKind;
use jolt_field::Field;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;
use serde::{Deserialize, Serialize};

use crate::stages::zk::outputs::CommittedOutputClaimOutput;

// The per-relation produced-claim structs live in their relation modules
// (cell-generic, `#[derive(OutputClaims)]`); re-export them so consumers and the
// `Stage6OutputClaims` aggregate keep resolving them through `stage6::outputs`.
pub use super::booleanity::BooleanityOutputClaims;
pub use super::bytecode_read_raf::BytecodeReadRafOutputClaims;
pub use super::inc_claim_reduction::IncClaimReductionOutputClaims;
pub use super::instruction_ra_virtualization::InstructionRaVirtualizationOutputClaims;
pub use super::ram_hamming_booleanity::RamHammingBooleanityOutputClaims;
pub use super::ram_ra_virtualization::RamRaVirtualizationOutputClaims;

/// The stage 6 produced opening claims, generic over the cell (`F` on the wire,
/// `Vec<F>` for derived points, `OpeningClaim<F>` (point + value) on the clear
/// path). The per-relation members are each `#[derive(OutputClaims)]` structs;
/// the address-phase and committed-reduction members are hand-written but follow
/// the same cell convention.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
pub struct Stage6OutputClaims<C> {
    pub address_phase: Stage6AddressPhaseClaims<C>,
    pub bytecode_read_raf: BytecodeReadRafOutputClaims<C>,
    pub booleanity: BooleanityOutputClaims<C>,
    pub ram_hamming_booleanity: RamHammingBooleanityOutputClaims<C>,
    pub ram_ra_virtualization: RamRaVirtualizationOutputClaims<C>,
    pub instruction_ra_virtualization: InstructionRaVirtualizationOutputClaims<C>,
    pub inc_claim_reduction: IncClaimReductionOutputClaims<C>,
    pub advice_cycle_phase: Stage6AdviceCyclePhaseClaims<C>,
    /// Committed program mode only.
    pub bytecode_claim_reduction: Option<BytecodeCyclePhaseOutputClaims<C>>,
    /// Committed program mode only.
    pub program_image_claim_reduction: Option<ProgramImageCyclePhaseOutputClaim<C>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
pub struct Stage6AddressPhaseClaims<C> {
    pub bytecode_read_raf: C,
    pub booleanity: C,
    /// `BytecodeValStage(s)` openings staged at the address-phase point;
    /// present only in committed program mode.
    pub bytecode_val_stages: Option<[C; NUM_BYTECODE_VAL_STAGES]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
pub struct Stage6AdviceCyclePhaseClaims<C> {
    pub trusted: Option<AdviceCyclePhaseOutputClaim<C>>,
    pub untrusted: Option<AdviceCyclePhaseOutputClaim<C>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
pub struct AdviceCyclePhaseOutputClaim<C> {
    pub opening_claim: C,
}

/// Openings cached when the committed-bytecode claim reduction's cycle phase
/// completes: the intermediate claim when address-phase rounds remain, or the
/// per-chunk `BytecodeChunk(i)` claims when the reduction finishes in the
/// cycle phase.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
pub enum BytecodeCyclePhaseOutputClaims<C> {
    Intermediate(C),
    Chunks(Vec<C>),
}

/// Opening cached when the program-image claim reduction's cycle phase
/// completes (the intermediate or final `ProgramImageInit` claim).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
pub struct ProgramImageCyclePhaseOutputClaim<C> {
    pub opening_claim: C,
}

/// Opening-point accessors over the point-only (`Vec<F>`) cell form of the stage-6
/// produced claims. Stages 7 and 8 read each relation's produced opening point off
/// these cells, single-sourcing the points that the retired `VerifiedStage6Batch`
/// used to carry. The per-cycle-phase reduction's `cycle_phase_variables` are
/// recovered as `reverse(opening_point)` (see `cycle_phase_opening_point` in
/// `jolt-claims` `claim_reductions::precommitted`).
impl<F: Field> Stage6OutputClaims<Vec<F>> {
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
    pub fn advice_cycle_phase_opening_point(&self, kind: JoltAdviceKind) -> Option<&[F]> {
        let claim = match kind {
            JoltAdviceKind::Trusted => self.advice_cycle_phase.trusted.as_ref(),
            JoltAdviceKind::Untrusted => self.advice_cycle_phase.untrusted.as_ref(),
        }?;
        Some(claim.opening_claim.as_slice())
    }

    /// The program-image claim-reduction cycle-phase opening point, present only in
    /// committed-program mode when the reduction ran a cycle phase.
    pub fn program_image_opening_point(&self) -> Option<&[F]> {
        self.program_image_claim_reduction
            .as_ref()
            .map(|claim| claim.opening_claim.as_slice())
    }

    /// The bytecode claim-reduction cycle-phase opening point, present only in
    /// committed-program mode. Every produced chunk (or the intermediate) shares
    /// the single cycle opening point, so the first cell is canonical.
    pub fn bytecode_reduction_opening_point(&self) -> Option<&[F]> {
        match self.bytecode_claim_reduction.as_ref()? {
            BytecodeCyclePhaseOutputClaims::Intermediate(point) => Some(point.as_slice()),
            BytecodeCyclePhaseOutputClaims::Chunks(points) => points.first().map(Vec::as_slice),
        }
    }

    /// The advice cycle-phase `cycle_phase_variables` for `kind`: the raw active
    /// cycle challenges, recovered as `reverse(opening_point)` (the cycle opening
    /// point is the reverse of the variable challenges). Stage 7's address phase
    /// reconstructs its opening point from these.
    pub fn advice_cycle_phase_variables(&self, kind: JoltAdviceKind) -> Option<Vec<F>> {
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
    pub output_claims: Stage6OutputClaims<F>,
    /// The produced opening *points* (point-only cell), paired field-for-field with
    /// `output_claims`. Stages 7 and 8 read each relation's opening point off these
    /// cells (via the `Stage6OutputClaims<Vec<F>>` accessors).
    pub output_points: Stage6OutputClaims<Vec<F>>,
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
    /// The produced opening *points* (point-only cell), the ZK counterpart of the
    /// clear path's `Stage6ClearOutput::output_points`. Stages 7/8 and BlindFold
    /// read each relation's opening point off these cells through the same
    /// `Stage6OutputClaims<Vec<F>>` accessors. (BlindFold recomputes the bytecode
    /// reduction weights locally, so the ZK output carries no weights aux.)
    pub output_points: Stage6OutputClaims<Vec<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage6Output<F: Field, C> {
    Clear(Stage6ClearOutput<F>),
    Zk(Stage6ZkOutput<F, C>),
}

impl<F: Field, C> Stage6Output<F, C> {
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
