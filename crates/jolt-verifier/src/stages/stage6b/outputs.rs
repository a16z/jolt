use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::BytecodeOutputWeightInputs;
use jolt_field::JoltField;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::relations::SumcheckBatch;
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

pub use super::booleanity::BooleanityOutputClaims;
pub use super::bytecode_read_raf::BytecodeReadRafOutputClaims;
pub use super::committed_reduction_cycle_phase::{
    BytecodeReductionCyclePhaseOutputClaims, ProgramImageReductionCyclePhaseOutputClaims,
};
#[cfg(not(feature = "akita"))]
pub use super::committed_reduction_cycle_phase::{
    TrustedAdviceCyclePhaseOutputClaims, UntrustedAdviceCyclePhaseOutputClaims,
};
#[cfg(feature = "field-inline")]
pub use super::field_registers_inc_claim_reduction::{
    FieldRegistersIncClaimReduction, FieldRegistersIncClaimReductionOutputClaims,
};
pub use super::inc_claim_reduction::IncClaimReductionOutputClaims;
pub use super::instruction_ra_virtualization::InstructionRaVirtualizationOutputClaims;
pub use super::ram_hamming_booleanity::RamHammingBooleanityOutputClaims;
pub use super::ram_ra_virtualization::RamRaVirtualizationOutputClaims;

use super::booleanity::Booleanity;
use super::bytecode_read_raf::BytecodeReadRafCycle;
#[cfg(feature = "akita")]
use super::bytecode_read_raf::LatticeBytecodeReadRafOutputClaims;
use super::committed_reduction_cycle_phase::{
    BytecodeReductionCyclePhase, ProgramImageReductionCyclePhase,
};
#[cfg(not(feature = "akita"))]
use super::committed_reduction_cycle_phase::{TrustedAdviceCyclePhase, UntrustedAdviceCyclePhase};
#[cfg(not(feature = "akita"))]
use super::inc_claim_reduction::IncClaimReduction;
use super::instruction_ra_virtualization::InstructionRaVirtualization;
use super::ram_hamming_booleanity::RamHammingBooleanity;
use super::ram_ra_virtualization::RamRaVirtualization;
#[cfg(feature = "akita")]
use jolt_claims::protocols::jolt::lattice::relations::booleanity::LatticeBooleanityOutputClaims;

#[derive(SumcheckBatch)]
#[sumcheck_batch(
    no_opening_values,
    no_draw_challenges,
    no_output_shape,
    crate = "crate"
)]
pub struct Stage6bSumchecks<F: JoltField> {
    pub bytecode_read_raf: BytecodeReadRafCycle<F>,
    pub booleanity: Booleanity<F>,
    pub ram_hamming_booleanity: RamHammingBooleanity<F>,
    pub ram_ra_virtualization: RamRaVirtualization<F>,
    pub instruction_ra_virtualization: InstructionRaVirtualization<F>,
    #[cfg(not(feature = "akita"))]
    pub inc_claim_reduction: IncClaimReduction<F>,
    #[cfg(feature = "field-inline")]
    pub field_registers_inc_claim_reduction: FieldRegistersIncClaimReduction<F>,
    #[cfg(not(feature = "akita"))]
    pub trusted_advice: Option<TrustedAdviceCyclePhase<F>>,
    #[cfg(not(feature = "akita"))]
    pub untrusted_advice: Option<UntrustedAdviceCyclePhase<F>>,
    pub bytecode_reduction: Option<BytecodeReductionCyclePhase<F>>,
    pub program_image_reduction: Option<ProgramImageReductionCyclePhase<F>>,
}

impl<F: JoltField> Stage6bOutputPoints<F> {
    /// The shared booleanity opening point (`r_address ++ r_cycle`); every
    /// produced booleanity RA opening uses it. `None` only if booleanity produced
    /// no openings (never in practice — at least one RA family is always present).
    pub fn booleanity_opening_point(&self) -> Option<&[F]> {
        #[cfg(not(feature = "akita"))]
        let chunk_fallback = None;
        #[cfg(feature = "akita")]
        let chunk_fallback = self.booleanity.balanced_inc_digits.first();
        self.booleanity
            .instruction_ra
            .first()
            .or_else(|| self.booleanity.bytecode_ra.first())
            .or_else(|| self.booleanity.ram_ra.first())
            .or(chunk_fallback)
            .map(Vec::as_slice)
    }

    /// The increment claim-reduction opening point (the reversed cycle point shared
    /// by the `RamInc`/`RdInc` reduced openings).
    #[cfg(not(feature = "akita"))]
    pub fn inc_opening_point(&self) -> &[F] {
        &self.inc_claim_reduction.ram_inc
    }

    /// The field-register increment claim-reduction opening point (the reversed cycle point of
    /// the reduced `FieldRdInc` opening), consumed by the stage-8 joint opening.
    #[cfg(feature = "field-inline")]
    pub fn field_registers_inc_opening_point(&self) -> &[F] {
        &self.field_registers_inc_claim_reduction.rd_inc
    }

    /// The packed fused-inc opening point: the read-raf cycle suffix (the
    /// stage-6b cycle point).
    #[cfg(feature = "akita")]
    pub fn fused_inc_opening_point(&self) -> &[F] {
        &self.bytecode_read_raf.fused_inc
    }

    /// The advice cycle-phase opening point for `kind`, present only when that
    /// advice reduction ran a cycle phase.
    #[cfg(not(feature = "akita"))]
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
    #[cfg(not(feature = "akita"))]
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
    /// output claims. ZK-only, hence base-only (no zk protocol exists over the
    /// packed axis).
    #[cfg(not(feature = "akita"))]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "a sum of in-memory vector lengths and small constants cannot overflow usize"
    )]
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
            + usize::from(cfg!(feature = "field-inline"))
            + usize::from(self.trusted_advice.is_some())
            + usize::from(self.untrusted_advice.is_some())
            + self.bytecode_reduction.as_ref().map_or(0, |reduction| {
                usize::from(reduction.intermediate.is_some()) + reduction.chunks.len()
            })
            + usize::from(self.program_image_reduction.is_some())
    }
}

impl<F: JoltField> Stage6bOutputClaims<F> {
    /// Construct the ordinary stage-6b claims. Producers without field-inline semantics use
    /// this regardless of the build's feature set — the field-inline increment-reduction slot
    /// defaults to an all-zero claim, inert because such producers' proofs never declare the
    /// field-inline axis. Base wire shape only (the akita converter assembles its
    /// lattice-shaped claims itself).
    #[cfg(not(feature = "akita"))]
    #[expect(
        clippy::too_many_arguments,
        reason = "one argument per batch member, mirroring the generated aggregate's shape"
    )]
    pub fn new(
        bytecode_read_raf: BytecodeReadRafOutputClaims<F>,
        booleanity: BooleanityOutputClaims<F>,
        ram_hamming_booleanity: RamHammingBooleanityOutputClaims<F>,
        ram_ra_virtualization: RamRaVirtualizationOutputClaims<F>,
        instruction_ra_virtualization: InstructionRaVirtualizationOutputClaims<F>,
        inc_claim_reduction: IncClaimReductionOutputClaims<F>,
        trusted_advice: Option<TrustedAdviceCyclePhaseOutputClaims<F>>,
        untrusted_advice: Option<UntrustedAdviceCyclePhaseOutputClaims<F>>,
        bytecode_reduction: Option<BytecodeReductionCyclePhaseOutputClaims<F>>,
        program_image_reduction: Option<ProgramImageReductionCyclePhaseOutputClaims<F>>,
    ) -> Self {
        Self {
            bytecode_read_raf,
            booleanity,
            ram_hamming_booleanity,
            ram_ra_virtualization,
            instruction_ra_virtualization,
            inc_claim_reduction,
            #[cfg(feature = "field-inline")]
            field_registers_inc_claim_reduction: Default::default(),
            trusted_advice,
            untrusted_advice,
            bytecode_reduction,
            program_image_reduction,
        }
    }

    /// The packed-shape twin of [`Self::new`]: the lattice batch has no increment or advice
    /// members, and the field-register increment-reduction slot again defaults to the inert
    /// all-zero claim.
    #[cfg(feature = "akita")]
    pub fn new(
        bytecode_read_raf: LatticeBytecodeReadRafOutputClaims<F>,
        booleanity: LatticeBooleanityOutputClaims<F>,
        ram_hamming_booleanity: RamHammingBooleanityOutputClaims<F>,
        ram_ra_virtualization: RamRaVirtualizationOutputClaims<F>,
        instruction_ra_virtualization: InstructionRaVirtualizationOutputClaims<F>,
        bytecode_reduction: Option<BytecodeReductionCyclePhaseOutputClaims<F>>,
        program_image_reduction: Option<ProgramImageReductionCyclePhaseOutputClaims<F>>,
    ) -> Self {
        Self {
            bytecode_read_raf,
            booleanity,
            ram_hamming_booleanity,
            ram_ra_virtualization,
            instruction_ra_virtualization,
            #[cfg(feature = "field-inline")]
            field_registers_inc_claim_reduction: Default::default(),
            bytecode_reduction,
            program_image_reduction,
        }
    }

    /// The consumed cycle-phase advice opening *value* for `kind` (the trusted /
    /// untrusted slot of that advice member), present only when the advice
    /// reduction ran a cycle phase. Read by stage 7's advice input wiring and stage
    /// 8's precommitted finals resolution.
    #[cfg(not(feature = "akita"))]
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

fn reversed<F: JoltField>(point: &[F]) -> Vec<F> {
    point.iter().rev().copied().collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6bCarriedChallenges<F: JoltField> {
    pub instruction_ra_gamma: F,
    #[cfg(not(feature = "akita"))]
    pub inc_gamma: F,
    #[cfg(feature = "field-inline")]
    pub field_registers_inc_gamma: F,
    pub bytecode_reduction_eta: Option<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct Stage6bClearOutput<F: JoltField> {
    pub output_values: Stage6bOutputClaims<F>,
    pub output_points: Stage6bOutputPoints<F>,
    pub bytecode_reduction_weights: Option<BytecodeReductionWeights<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6bZkOutput<F: JoltField, C> {
    pub challenges: Stage6bCarriedChallenges<F>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    pub output_points: Stage6bOutputPoints<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage6bOutput<F: JoltField, C> {
    Clear(Stage6bClearOutput<F>),
    Zk(Stage6bZkOutput<F, C>),
}

impl<F: JoltField, C> Stage6bOutput<F, C> {
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
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct BytecodeReductionWeights<F: JoltField> {
    pub r_bc: Vec<F>,
    pub chunk_rbc_weights: Vec<F>,
    pub lane_weights: Vec<F>,
}

impl<F: JoltField> BytecodeReductionWeights<F> {
    pub(crate) fn as_inputs(&self) -> BytecodeOutputWeightInputs<'_, F> {
        BytecodeOutputWeightInputs {
            r_bc: &self.r_bc,
            chunk_rbc_weights: &self.chunk_rbc_weights,
            lane_weights: &self.lane_weights,
        }
    }
}
