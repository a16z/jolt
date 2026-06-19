//! The stage 4 `RamValCheck` sumcheck instance.
//!
//! A self-contained relation object driven identically by the prover and the
//! verifier. It owns the RAM value-check point derivation and the
//! `LtCyclePlusGamma` public-value computation; the decomposition of
//! `Val_init(r_address)` into a public evaluation plus committed
//! advice/program-image contributions lives in its `jolt-claims` formula
//! (`ram::val_check`), so the clear path, the prover, and the BlindFold
//! constraint all consume the same decomposition.
//!
//! WARNING: the advice/program-image openings are dual-role — they are *consumed*
//! by the input claim (init reconstruction) and *also* appended/serialized as
//! stage-4 openings. They therefore appear both as [`RamValCheckInputClaims`]
//! fields and in the serialized `Stage4Claims` aggregate. Only their values feed
//! the input claim; their staged points are carried for completeness.

use jolt_claims::protocols::jolt::{
    formulas::{
        dimensions::TraceDimensions,
        ram::{self, RamValCheckInit},
    },
    JoltAdviceKind, JoltChallengeId, JoltPublicId, JoltRelationClaims, JoltRelationId,
    RamValCheckChallenge, RamValCheckPublic,
};
use jolt_field::Field;
use jolt_poly::LtPolynomial;
use jolt_verifier_derive::{InputClaims, OutputClaims};
use serde::{Deserialize, Serialize};

use crate::stages::relations::{GetPoint, OpeningClaim, SumcheckInstance};
use crate::stages::stage2::outputs::Stage2ClearOutput;
use crate::VerifierError;

use super::outputs::RamValCheckInitialEvaluation;

/// Produced RAM value-check openings (`ram_ra`, `ram_inc`) sharing one opening
/// point. Generic over the cell.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamValCheck)]
pub struct RamValCheckOutputClaims<C> {
    #[opening(RamRa)]
    pub ram_ra: C,
    #[opening(committed = RamInc)]
    pub ram_inc: C,
}

/// The staged advice openings contributing to `Val_init`: untrusted/trusted
/// advice block evaluations, each present only when its commitment is. Appended
/// before the register openings (see the `Stage4Claims` field order).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamValCheck)]
pub struct RamValCheckAdviceClaims<C> {
    #[opening(untrusted_advice)]
    pub untrusted: Option<C>,
    #[opening(trusted_advice)]
    pub trusted: Option<C>,
}

/// Consumed openings of the RAM value-check claim: the read-write `val` (stage 2)
/// and output-check `val_final` (stage 2), reduced against `Val_init`, whose
/// committed pieces (advice / program image) are present only in some proof
/// configurations. Generic over the cell.
#[derive(Clone, Debug, InputClaims)]
pub struct RamValCheckInputClaims<C> {
    #[opening(RamVal, from = RamReadWriteChecking)]
    pub ram_val: C,
    #[opening(RamValFinal, from = RamOutputCheck)]
    pub ram_val_final: C,
    #[opening(untrusted_advice, from = RamValCheck)]
    pub untrusted_advice: Option<C>,
    #[opening(trusted_advice, from = RamValCheck)]
    pub trusted_advice: Option<C>,
    #[opening(ProgramImageInitContributionRw, from = RamValCheck)]
    pub program_image: Option<C>,
}

impl<F: Field> RamValCheckInputClaims<OpeningClaim<F>> {
    /// Wire the consumed openings from stage 2's RAM read-write `val` and
    /// output-check `val_final`, plus the reconstructed init contributions (the
    /// same advice / program-image openings the init evaluation is decomposed
    /// into). The init pieces carry their staged opening points for completeness,
    /// though only their values feed the input claim.
    pub fn from_upstream(
        stage2: &Stage2ClearOutput<F>,
        init: &RamValCheckInitialEvaluation<F>,
    ) -> Self {
        let advice =
            |kind: JoltAdviceKind| init.advice_contribution(kind).map(|c| c.opening.clone());
        Self {
            ram_val: OpeningClaim {
                point: stage2.batch.ram_read_write.opening_point.clone(),
                value: stage2.output_claims.ram_read_write.val,
            },
            ram_val_final: OpeningClaim {
                point: stage2.batch.ram_output_check.opening_point.clone(),
                value: stage2.output_claims.ram_output_check,
            },
            untrusted_advice: advice(JoltAdviceKind::Untrusted),
            trusted_advice: advice(JoltAdviceKind::Trusted),
            program_image: init.program_image_contribution.clone(),
        }
    }
}

pub struct RamValCheck<F: Field> {
    claims: JoltRelationClaims<F>,
    trace_dimensions: TraceDimensions,
    ram_log_k: usize,
    gamma: F,
}

impl<F: Field> RamValCheck<F> {
    /// Build the relation from its per-proof init decomposition. `init` carries
    /// the public initial-RAM evaluation plus the present advice/program-image
    /// contributions, baked into the formula's input `Expr`.
    pub fn new(
        trace_dimensions: TraceDimensions,
        ram_log_k: usize,
        gamma: F,
        init: RamValCheckInit<F>,
    ) -> Self {
        Self {
            claims: ram::val_check(trace_dimensions, init),
            trace_dimensions,
            ram_log_k,
            gamma,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::RamValCheck,
        reason: reason.to_string(),
    }
}

impl<F: Field> SumcheckInstance<F> for RamValCheck<F> {
    type Inputs<C> = RamValCheckInputClaims<C>;
    type Outputs<C> = RamValCheckOutputClaims<C>;

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F> {
        &self.claims
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        inputs: &RamValCheckInputClaims<C>,
    ) -> Result<RamValCheckOutputClaims<Vec<F>>, VerifierError> {
        let log_t = self.trace_dimensions.log_t();
        let expected_len = self.ram_log_k + log_t;
        let ram_read_write_point = inputs.ram_val.point();
        if ram_read_write_point.len() != expected_len {
            return Err(public_input_failed(format!(
                "RAM read-write opening point has {} variables, expected {expected_len}",
                ram_read_write_point.len()
            )));
        }
        let (r_address, r_cycle) = ram_read_write_point.split_at(self.ram_log_k);
        let cycle = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        if cycle.len() != r_cycle.len() {
            return Err(public_input_failed(format!(
                "RAM value cycle point length mismatch: expected {}, got {}",
                r_cycle.len(),
                cycle.len()
            )));
        }
        let opening_point = [r_address, cycle.as_slice()].concat();
        Ok(RamValCheckOutputClaims {
            ram_ra: opening_point.clone(),
            ram_inc: opening_point,
        })
    }

    fn resolve_challenge(&self, id: &JoltChallengeId) -> Result<F, VerifierError> {
        match id {
            JoltChallengeId::RamValCheck(RamValCheckChallenge::Gamma) => Ok(self.gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        }
    }

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltPublicId,
        inputs: &RamValCheckInputClaims<C>,
        outputs: &RamValCheckOutputClaims<OpeningClaim<F>>,
    ) -> Result<F, VerifierError> {
        let JoltPublicId::RamValCheck(public_id) = id else {
            return Err(VerifierError::MissingStageClaimPublic { id: *id });
        };
        match public_id {
            // LtCyclePlusGamma folds the batching gamma into the `Lt` evaluation
            // of the produced cycle point against the fixed read-write cycle.
            RamValCheckPublic::LtCyclePlusGamma => {
                let output_cycle = &outputs.ram_ra.point()[self.ram_log_k..];
                let fixed_cycle = &inputs.ram_val.point()[self.ram_log_k..];
                Ok(LtPolynomial::evaluate(output_cycle, fixed_cycle) + self.gamma)
            }
        }
    }
}
