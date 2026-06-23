//! The stage 7 `HammingWeightClaimReduction` sumcheck instance.
//!
//! A self-contained relation object driven identically by the prover (while
//! producing the stage 7 batch proof) and the verifier (after checking it). It
//! reduces the per-family RA booleanity, virtualization, and hamming-weight
//! claims (instruction, bytecode, RAM) into the one-hot `Ra` opening claims that
//! anchor the stage 8 final batched opening. It owns the shared opening-point
//! derivation and the `EqBooleanity` / `EqVirtualization` public-value
//! computation, so the input/output claim algebra lives here once instead of
//! being hand-coded on each side.

use jolt_claims::protocols::jolt::{
    formulas::claim_reductions::hamming_weight::{self, HammingWeightClaimReductionDimensions},
    HammingWeightClaimReductionChallenge, HammingWeightClaimReductionPublic, JoltChallengeId,
    JoltPublicId, JoltRelationClaims, JoltRelationId,
};
use jolt_field::Field;
use jolt_poly::try_eq_mle;
use jolt_verifier_derive::{InputClaims, OutputClaims};
use serde::{Deserialize, Serialize};

use crate::stages::relations::{GetPoint, OpeningClaim, SumcheckInstance};
use crate::VerifierError;

/// Produced one-hot `Ra` opening claims, grouped by family (instruction,
/// bytecode, RAM) in canonical layout order. Every produced opening shares the
/// single hamming-weight opening point. Generic over the cell (`F` on the wire,
/// `Vec<F>` for ZK points, `OpeningClaim<F>` on the clear path).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(HammingWeightClaimReduction)]
pub struct HammingWeightClaimReductionOutputClaims<C> {
    #[opening(committed = InstructionRa)]
    pub instruction_ra: Vec<C>,
    #[opening(committed = BytecodeRa)]
    pub bytecode_ra: Vec<C>,
    #[opening(committed = RamRa)]
    pub ram_ra: Vec<C>,
}

/// Consumed claims reduced by the hamming-weight sumcheck: the RAM hamming-weight
/// claim (from RAM hamming booleanity) plus the per-family booleanity and
/// virtualization claims (each wired from its producing stage-6 relation).
/// Generic over the cell.
#[derive(Clone, Debug, InputClaims)]
pub struct HammingWeightClaimReductionInputClaims<C> {
    #[opening(RamHammingWeight, from = RamHammingBooleanity)]
    pub ram_hamming_weight: C,
    #[opening(committed = InstructionRa, from = Booleanity)]
    pub instruction_booleanity: Vec<C>,
    #[opening(committed = BytecodeRa, from = Booleanity)]
    pub bytecode_booleanity: Vec<C>,
    #[opening(committed = RamRa, from = Booleanity)]
    pub ram_booleanity: Vec<C>,
    #[opening(committed = InstructionRa, from = InstructionRaVirtualization)]
    pub instruction_virtualization: Vec<C>,
    #[opening(committed = BytecodeRa, from = BytecodeReadRaf)]
    pub bytecode_virtualization: Vec<C>,
    #[opening(committed = RamRa, from = RamRaVirtualization)]
    pub ram_virtualization: Vec<C>,
}

pub struct HammingWeightClaimReduction<F: Field> {
    claims: JoltRelationClaims<F>,
    dimensions: HammingWeightClaimReductionDimensions,
    gamma: F,
    /// The shared cycle suffix appended to every produced opening point (the
    /// stage-6 booleanity cycle point).
    r_cycle: Vec<F>,
    /// The stage-6 booleanity address point that `EqBooleanity` compares against.
    r_address: Vec<F>,
    /// The per-RA virtualization address chunks (one per layout polynomial, in
    /// canonical order) that `EqVirtualization(i)` compares against.
    virtualization_points: Vec<Vec<F>>,
}

impl<F: Field> HammingWeightClaimReduction<F> {
    pub fn new(
        dimensions: HammingWeightClaimReductionDimensions,
        gamma: F,
        r_cycle: Vec<F>,
        r_address: Vec<F>,
        virtualization_points: Vec<Vec<F>>,
    ) -> Self {
        Self {
            claims: hamming_weight::claim_reduction(dimensions),
            dimensions,
            gamma,
            r_cycle,
            r_address,
            virtualization_points,
        }
    }

    /// The reduction's address chunk point `rho` in reversed order: the leading
    /// `log_k_chunk` coordinates of the (shared) produced opening point. Equal to
    /// the hamming sumcheck point reversed — `opening_point` prepends the reversed
    /// challenges — so the EQ publics evaluate against it directly.
    fn rho_reversed<'a>(
        &self,
        outputs: &'a HammingWeightClaimReductionOutputClaims<OpeningClaim<F>>,
    ) -> Result<&'a [F], VerifierError> {
        let opening_point = outputs
            .instruction_ra
            .first()
            .or_else(|| outputs.bytecode_ra.first())
            .or_else(|| outputs.ram_ra.first())
            .map(|claim| claim.point.as_slice())
            .ok_or_else(|| {
                public_input_failed("HammingWeight reduction produced no openings".to_string())
            })?;
        opening_point
            .get(..self.dimensions.log_k_chunk)
            .ok_or_else(|| {
                public_input_failed(format!(
                    "HammingWeight opening point has {} variables, fewer than log_k_chunk {}",
                    opening_point.len(),
                    self.dimensions.log_k_chunk
                ))
            })
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::HammingWeightClaimReduction,
        reason: reason.to_string(),
    }
}

impl<F: Field> SumcheckInstance<F> for HammingWeightClaimReduction<F> {
    type Inputs<C> = HammingWeightClaimReductionInputClaims<C>;
    type Outputs<C> = HammingWeightClaimReductionOutputClaims<C>;

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F> {
        &self.claims
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &HammingWeightClaimReductionInputClaims<C>,
    ) -> Result<HammingWeightClaimReductionOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .dimensions
            .opening_point(sumcheck_point, &self.r_cycle)
            .map_err(public_input_failed)?;
        let layout = self.dimensions.layout;
        Ok(HammingWeightClaimReductionOutputClaims {
            instruction_ra: vec![opening_point.clone(); layout.instruction()],
            bytecode_ra: vec![opening_point.clone(); layout.bytecode()],
            ram_ra: vec![opening_point; layout.ram()],
        })
    }

    fn resolve_challenge(&self, id: &JoltChallengeId) -> Result<F, VerifierError> {
        match id {
            JoltChallengeId::HammingWeightClaimReduction(
                HammingWeightClaimReductionChallenge::Gamma,
            ) => Ok(self.gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        }
    }

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltPublicId,
        _inputs: &HammingWeightClaimReductionInputClaims<C>,
        outputs: &HammingWeightClaimReductionOutputClaims<OpeningClaim<F>>,
    ) -> Result<F, VerifierError> {
        let JoltPublicId::HammingWeightClaimReduction(public_id) = id else {
            return Err(VerifierError::MissingStageClaimPublic { id: *id });
        };
        let rho_rev = self.rho_reversed(outputs)?;
        match public_id {
            HammingWeightClaimReductionPublic::EqBooleanity => {
                try_eq_mle(rho_rev, &self.r_address).map_err(public_input_failed)
            }
            HammingWeightClaimReductionPublic::EqVirtualization(index) => {
                let point = self.virtualization_points.get(*index).ok_or_else(|| {
                    public_input_failed(format!(
                        "missing HammingWeight virtualization point for index {index}"
                    ))
                })?;
                try_eq_mle(rho_rev, point).map_err(public_input_failed)
            }
        }
    }
}
