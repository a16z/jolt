//! The stage 6b committed-program claim-reduction cycle phases.
//!
//! In committed-program mode the advice, program-image, and per-chunk bytecode
//! polynomials are reduced over the shared precommitted schedule. Each reduction's
//! cycle phase runs here (stage 6b); when the polynomial still has active
//! address-phase rounds it stages an intermediate opening consumed by stage 7,
//! otherwise it completes here, opening the committed polynomial under a
//! `FinalScale` / `ChunkOutputWeight` public.
//!
//! Because the produced opening point is the (reverse-ordered) cycle opening
//! point while the scale is evaluated at the Dory-permuted point, each relation
//! OVERRIDES [`ConcreteSumcheck::expected_output`] to recover the scale from the
//! produced opening point via the layout's `cycle_phase_*_at_opening_point`
//! helpers — see [`PrecommittedClaimReduction::cycle_phase_permuted_from_opening_point`].
//! The output expression is bypassed (not the `resolve_public` path) because the
//! produced opening id is dynamic in `has_address_phase`; the override computes
//! exactly the formula value, so the clear path and BlindFold stay in sync.

use jolt_claims::protocols::jolt::{
    formulas::claim_reductions::{
        advice, bytecode as bytecode_reduction, bytecode::BytecodeOutputWeightInputs, program_image,
    },
    AdviceClaimReductionLayout, BytecodeClaimReductionChallenge, BytecodeClaimReductionLayout,
    JoltAdviceKind, JoltChallengeId, JoltRelationClaims, JoltRelationId,
    PrecommittedReductionLayout, ProgramImageClaimReductionLayout,
};
use jolt_field::Field;
use jolt_verifier_derive::{InputClaims, OutputClaims};
use serde::{Deserialize, Serialize};

use super::outputs::BytecodeReductionWeights;
use crate::stages::relations::{GetPoint, OpeningClaim, ConcreteSumcheck};
use crate::stages::stage4::Stage4ClearOutput;
use crate::VerifierError;

// ---------------------------------------------------------------------------
// Advice cycle phase (per-kind)
// ---------------------------------------------------------------------------

/// The produced advice opening (the intermediate when an address phase follows,
/// else the final advice opening), keyed by kind.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(AdviceClaimReductionCyclePhase)]
pub struct AdviceCyclePhaseOutputClaims<C> {
    #[opening(trusted_advice)]
    pub trusted: Option<C>,
    #[opening(untrusted_advice)]
    pub untrusted: Option<C>,
}

/// The consumed RAM value-check advice opening, keyed by kind.
#[derive(Clone, Debug, InputClaims)]
pub struct AdviceCyclePhaseInputClaims<C> {
    #[opening(trusted_advice, from = RamValCheck)]
    pub trusted: Option<C>,
    #[opening(untrusted_advice, from = RamValCheck)]
    pub untrusted: Option<C>,
}

impl<F: Field> AdviceCyclePhaseInputClaims<OpeningClaim<F>> {
    pub fn from_upstream(stage4: &Stage4ClearOutput<F>) -> Self {
        let opening = |kind| {
            stage4
                .ram_val_check_init
                .advice_contributions
                .iter()
                .find(|contribution| contribution.kind == kind)
                .map(|contribution| OpeningClaim {
                    point: contribution.opening.point.clone(),
                    value: contribution.opening.value,
                })
        };
        Self {
            trusted: opening(JoltAdviceKind::Trusted),
            untrusted: opening(JoltAdviceKind::Untrusted),
        }
    }
}

pub struct AdviceCyclePhase<F: Field> {
    claims: JoltRelationClaims<F>,
    kind: JoltAdviceKind,
    layout: AdviceClaimReductionLayout,
    /// The RAM address point of the staged advice opening from RAM value-check;
    /// the `FinalScale` public compares the produced opening point against it.
    reference_opening_point: Vec<F>,
}

impl<F: Field> AdviceCyclePhase<F> {
    pub fn new(
        kind: JoltAdviceKind,
        layout: &AdviceClaimReductionLayout,
        reference_opening_point: Vec<F>,
    ) -> Self {
        Self {
            claims: advice::cycle_phase(kind, layout.dimensions()),
            kind,
            layout: layout.clone(),
            reference_opening_point,
        }
    }

    pub fn kind(&self) -> JoltAdviceKind {
        self.kind
    }

    fn output<'a>(
        &self,
        outputs: &'a AdviceCyclePhaseOutputClaims<OpeningClaim<F>>,
    ) -> Result<&'a OpeningClaim<F>, VerifierError> {
        match self.kind {
            JoltAdviceKind::Trusted => outputs.trusted.as_ref(),
            JoltAdviceKind::Untrusted => outputs.untrusted.as_ref(),
        }
        .ok_or_else(|| advice_public_failed("advice cycle phase produced no opening"))
    }
}

fn advice_public_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::AdviceClaimReductionCyclePhase,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for AdviceCyclePhase<F> {
    type Inputs<C> = AdviceCyclePhaseInputClaims<C>;
    type Outputs<C> = AdviceCyclePhaseOutputClaims<C>;

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F> {
        &self.claims
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &AdviceCyclePhaseInputClaims<C>,
    ) -> Result<AdviceCyclePhaseOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .layout
            .cycle_phase_opening_point(sumcheck_point)
            .map_err(advice_public_failed)?;
        Ok(match self.kind {
            JoltAdviceKind::Trusted => AdviceCyclePhaseOutputClaims {
                trusted: Some(opening_point),
                untrusted: None,
            },
            JoltAdviceKind::Untrusted => AdviceCyclePhaseOutputClaims {
                trusted: None,
                untrusted: Some(opening_point),
            },
        })
    }

    fn expected_output<C: GetPoint<F>>(
        &self,
        _inputs: &AdviceCyclePhaseInputClaims<C>,
        outputs: &AdviceCyclePhaseOutputClaims<OpeningClaim<F>>,
    ) -> Result<F, VerifierError> {
        let opening = self.output(outputs)?;
        if self.layout.dimensions().has_address_phase() {
            Ok(opening.value)
        } else {
            let scale = self
                .layout
                .cycle_phase_scale_at_opening_point(&self.reference_opening_point, &opening.point)
                .map_err(advice_public_failed)?;
            Ok(scale * opening.value)
        }
    }
}

// ---------------------------------------------------------------------------
// Program-image cycle phase
// ---------------------------------------------------------------------------

/// The produced `ProgramImageInit` opening (intermediate or final).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(ProgramImageClaimReductionCyclePhase)]
pub struct ProgramImageReductionCyclePhaseOutputClaims<C> {
    #[opening(committed = ProgramImageInit)]
    pub program_image: C,
}

/// The consumed RAM value-check program-image contribution.
#[derive(Clone, Debug, InputClaims)]
pub struct ProgramImageReductionCyclePhaseInputClaims<C> {
    #[opening(ProgramImageInitContributionRw, from = RamValCheck)]
    pub contribution: C,
}

impl<F: Field> ProgramImageReductionCyclePhaseInputClaims<OpeningClaim<F>> {
    pub fn from_upstream(stage4: &Stage4ClearOutput<F>) -> Result<Self, VerifierError> {
        let contribution = stage4
            .ram_val_check_init
            .program_image_contribution
            .as_ref()
            .ok_or_else(|| {
                program_image_public_failed("missing RAM value-check program-image contribution")
            })?;
        Ok(Self {
            contribution: OpeningClaim {
                point: contribution.point.clone(),
                value: contribution.value,
            },
        })
    }
}

pub struct ProgramImageReductionCyclePhase<F: Field> {
    claims: JoltRelationClaims<F>,
    layout: ProgramImageClaimReductionLayout,
    /// The RAM address component of the `RamVal` opening from RAM read-write
    /// checking; the `FinalScale` public compares the produced opening point
    /// against it.
    r_addr_rw: Vec<F>,
}

impl<F: Field> ProgramImageReductionCyclePhase<F> {
    pub fn new(layout: &ProgramImageClaimReductionLayout, r_addr_rw: Vec<F>) -> Self {
        Self {
            claims: program_image::cycle_phase(layout.dimensions()),
            layout: layout.clone(),
            r_addr_rw,
        }
    }
}

fn program_image_public_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::ProgramImageClaimReductionCyclePhase,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for ProgramImageReductionCyclePhase<F> {
    type Inputs<C> = ProgramImageReductionCyclePhaseInputClaims<C>;
    type Outputs<C> = ProgramImageReductionCyclePhaseOutputClaims<C>;

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F> {
        &self.claims
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &ProgramImageReductionCyclePhaseInputClaims<C>,
    ) -> Result<ProgramImageReductionCyclePhaseOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .layout
            .cycle_phase_opening_point(sumcheck_point)
            .map_err(program_image_public_failed)?;
        Ok(ProgramImageReductionCyclePhaseOutputClaims {
            program_image: opening_point,
        })
    }

    fn expected_output<C: GetPoint<F>>(
        &self,
        _inputs: &ProgramImageReductionCyclePhaseInputClaims<C>,
        outputs: &ProgramImageReductionCyclePhaseOutputClaims<OpeningClaim<F>>,
    ) -> Result<F, VerifierError> {
        let opening = &outputs.program_image;
        if self.layout.dimensions().has_address_phase() {
            Ok(opening.value)
        } else {
            let scale = self
                .layout
                .cycle_phase_scale_at_opening_point(&self.r_addr_rw, &opening.point)
                .map_err(program_image_public_failed)?;
            Ok(scale * opening.value)
        }
    }
}

// ---------------------------------------------------------------------------
// Bytecode reduction cycle phase
// ---------------------------------------------------------------------------

/// The produced bytecode-reduction openings: the intermediate when an address
/// phase follows, else the per-chunk final `BytecodeChunk` openings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(BytecodeClaimReductionCyclePhase)]
pub struct BytecodeReductionCyclePhaseOutputClaims<C> {
    #[opening(BytecodeClaimReductionIntermediate)]
    pub intermediate: Option<C>,
    #[opening(committed = BytecodeChunk)]
    pub chunks: Vec<C>,
}

/// The consumed staged `BytecodeValStage` openings from the bytecode read-RAF
/// address phase.
#[derive(Clone, Debug, InputClaims)]
pub struct BytecodeReductionCyclePhaseInputClaims<C> {
    #[opening(BytecodeValStage, from = BytecodeReadRaf)]
    pub val_stages: Vec<C>,
}

impl<F: Field> BytecodeReductionCyclePhaseInputClaims<OpeningClaim<F>> {
    pub fn from_values(val_stages: Vec<OpeningClaim<F>>) -> Self {
        Self { val_stages }
    }
}

pub struct BytecodeReductionCyclePhase<F: Field> {
    claims: JoltRelationClaims<F>,
    layout: BytecodeClaimReductionLayout,
    eta: F,
    weights: BytecodeReductionWeights<F>,
    chunk_count: usize,
}

impl<F: Field> BytecodeReductionCyclePhase<F> {
    pub fn new(
        layout: &BytecodeClaimReductionLayout,
        eta: F,
        weights: BytecodeReductionWeights<F>,
    ) -> Self {
        Self {
            claims: bytecode_reduction::cycle_phase(layout.dimensions(), layout.chunk_count()),
            layout: layout.clone(),
            eta,
            weights,
            chunk_count: layout.chunk_count(),
        }
    }

    fn output_weight_inputs(&self) -> BytecodeOutputWeightInputs<'_, F> {
        BytecodeOutputWeightInputs {
            r_bc: &self.weights.r_bc,
            chunk_rbc_weights: &self.weights.chunk_rbc_weights,
            lane_weights: &self.weights.lane_weights,
        }
    }
}

fn bytecode_public_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::BytecodeClaimReductionCyclePhase,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for BytecodeReductionCyclePhase<F> {
    type Inputs<C> = BytecodeReductionCyclePhaseInputClaims<C>;
    type Outputs<C> = BytecodeReductionCyclePhaseOutputClaims<C>;

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F> {
        &self.claims
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &BytecodeReductionCyclePhaseInputClaims<C>,
    ) -> Result<BytecodeReductionCyclePhaseOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .layout
            .cycle_phase_opening_point(sumcheck_point)
            .map_err(bytecode_public_failed)?;
        Ok(if self.layout.dimensions().has_address_phase() {
            BytecodeReductionCyclePhaseOutputClaims {
                intermediate: Some(opening_point),
                chunks: Vec::new(),
            }
        } else {
            BytecodeReductionCyclePhaseOutputClaims {
                intermediate: None,
                chunks: vec![opening_point; self.chunk_count],
            }
        })
    }

    fn resolve_challenge(&self, id: &JoltChallengeId) -> Result<F, VerifierError> {
        match id {
            JoltChallengeId::BytecodeClaimReduction(BytecodeClaimReductionChallenge::Eta) => {
                Ok(self.eta)
            }
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        }
    }

    fn expected_output<C: GetPoint<F>>(
        &self,
        _inputs: &BytecodeReductionCyclePhaseInputClaims<C>,
        outputs: &BytecodeReductionCyclePhaseOutputClaims<OpeningClaim<F>>,
    ) -> Result<F, VerifierError> {
        if self.layout.dimensions().has_address_phase() {
            let intermediate = outputs.intermediate.as_ref().ok_or_else(|| {
                bytecode_public_failed("bytecode reduction produced no intermediate")
            })?;
            return Ok(intermediate.value);
        }
        if outputs.chunks.len() != self.chunk_count {
            return Err(bytecode_public_failed(format!(
                "bytecode chunk claim count mismatch: expected {}, got {}",
                self.chunk_count,
                outputs.chunks.len()
            )));
        }
        let opening_point = outputs
            .chunks
            .first()
            .map(|chunk| chunk.point.as_slice())
            .ok_or_else(|| {
                bytecode_public_failed("bytecode reduction produced no chunk openings")
            })?;
        let weights = self
            .layout
            .cycle_phase_final_output_weights_at_opening_point(
                self.output_weight_inputs(),
                opening_point,
            )
            .map_err(bytecode_public_failed)?;
        Ok(outputs
            .chunks
            .iter()
            .zip(weights)
            .map(|(chunk, weight)| chunk.value * weight)
            .sum())
    }
}
