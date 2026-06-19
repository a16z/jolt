//! The stage 6 bytecode read-RAF sumcheck instances.
//!
//! This module currently provides the stage-6a **address phase**. The cycle phase
//! (stage 6b) is deferred to the wiring step: its expected output depends on the
//! bytecode-table public values (`read_raf_public_values`, which needs the
//! preprocessing bytecode rows) and, in committed-program mode, consumes the
//! staged `BytecodeValStage` openings inside its *output* expression — both of
//! which are cleanest to finalize against the live `verify()`/prover wiring.
//!
//! The address phase binds the `log_k` address variables. Its input claim is the
//! gamma-folded bind of the entire prior proof (every stage-1..5 opening plus the
//! two PC claims); that 25-opening formula already lives in the single-sourced
//! [`stage6_bytecode_read_raf_address_input`] helper, so this relation takes the
//! precomputed value and overrides [`SumcheckInstance::input_claim`] rather than
//! restating the bind as a 25-field `InputClaims`. Its output is the staged
//! `BytecodeReadRafAddrClaim` intermediate (consumed by the cycle phase) followed,
//! in committed mode, by the `BytecodeValStage` openings.
//!
//! [`stage6_bytecode_read_raf_address_input`]: super::verify::stage6_bytecode_read_raf_address_input

use core::marker::PhantomData;

use jolt_claims::protocols::jolt::{
    formulas::{
        bytecode::{
            self, BytecodeReadRafCommittedEvaluationInputs, BytecodeReadRafDimensions,
            BytecodeReadRafEvaluationInputs,
        },
        claim_reductions::bytecode::bytecode_val_stage_opening,
        dimensions::committed_address_chunks,
    },
    BytecodeReadRafChallenge, JoltChallengeId, JoltOpeningId, JoltPublicId, JoltRelationClaims,
    JoltRelationId,
};
use jolt_field::Field;
use jolt_riscv::JoltInstructionRow;
use jolt_verifier_derive::{InputClaims, OutputClaims};
use serde::{Deserialize, Serialize};

use super::verify::{
    stage6_bytecode_read_raf_expected_output, Stage6BytecodeReadRafExpectedOutputInputs,
};
use crate::stages::relations::{GetPoint, GetValue, OpeningClaim, SumcheckInstance};
use crate::VerifierError;

/// The address-phase produced openings: the `BytecodeReadRafAddrClaim`
/// intermediate, plus (committed-program mode only) the staged `BytecodeValStage`
/// openings. In full-program mode `val_stages` is empty.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(BytecodeReadRaf)]
pub struct BytecodeReadRafAddressPhaseOutputClaims<C> {
    #[opening(BytecodeReadRafAddrClaim)]
    pub intermediate: C,
    #[opening(BytecodeValStage)]
    pub val_stages: Vec<C>,
}

/// The address phase's input claim is the gamma-folded prior-proof bind, supplied
/// precomputed (see the module docs), so it consumes no openings through the
/// generic path.
pub struct BytecodeReadRafAddressPhaseInputClaims<C> {
    _cell: PhantomData<C>,
}

impl<C> Default for BytecodeReadRafAddressPhaseInputClaims<C> {
    fn default() -> Self {
        Self { _cell: PhantomData }
    }
}

impl<F: Field> BytecodeReadRafAddressPhaseInputClaims<OpeningClaim<F>> {
    pub fn from_upstream() -> Self {
        Self::default()
    }
}

impl<F: Field> crate::stages::relations::InputClaims<F>
    for BytecodeReadRafAddressPhaseInputClaims<OpeningClaim<F>>
{
    fn resolve_input(&self, _id: &JoltOpeningId) -> Option<F> {
        None
    }
}

pub struct BytecodeReadRafAddressPhase<F: Field> {
    claims: JoltRelationClaims<F>,
    /// The gamma-folded prior-proof bind, precomputed by
    /// `stage6_bytecode_read_raf_address_input`.
    input_claim: F,
    /// `NUM_BYTECODE_VAL_STAGES` in committed-program mode, else 0.
    num_val_stages: usize,
}

impl<F: Field> BytecodeReadRafAddressPhase<F> {
    pub fn new(
        dimensions: BytecodeReadRafDimensions,
        input_claim: F,
        num_val_stages: usize,
    ) -> Self {
        Self {
            claims: bytecode::read_raf_address_phase(dimensions),
            input_claim,
            num_val_stages,
        }
    }
}

impl<F: Field> SumcheckInstance<F> for BytecodeReadRafAddressPhase<F> {
    type Inputs<C> = BytecodeReadRafAddressPhaseInputClaims<C>;
    type Outputs<C> = BytecodeReadRafAddressPhaseOutputClaims<C>;

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F> {
        &self.claims
    }

    fn input_claim(
        &self,
        _inputs: &BytecodeReadRafAddressPhaseInputClaims<OpeningClaim<F>>,
    ) -> Result<F, VerifierError> {
        Ok(self.input_claim)
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &BytecodeReadRafAddressPhaseInputClaims<C>,
    ) -> Result<BytecodeReadRafAddressPhaseOutputClaims<Vec<F>>, VerifierError> {
        // `bytecode_r_address` is the reversed address sumcheck point; the
        // intermediate and every staged Val column open there.
        let r_address = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        Ok(BytecodeReadRafAddressPhaseOutputClaims {
            intermediate: r_address.clone(),
            val_stages: vec![r_address; self.num_val_stages],
        })
    }
}

// ---------------------------------------------------------------------------
// Cycle phase (stage 6b) — full-program mode
// ---------------------------------------------------------------------------

/// The cycle-phase produced openings: the per-chunk committed `BytecodeRa` claims,
/// all sharing the `r_address ++ r_cycle` opening point.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(BytecodeReadRaf)]
pub struct BytecodeReadRafOutputClaims<C> {
    #[opening(committed = BytecodeRa)]
    pub bytecode_ra: Vec<C>,
}

/// The `BytecodeReadRafAddrClaim` intermediate consumed from the address phase.
#[derive(Clone, Debug, InputClaims)]
pub struct BytecodeReadRafInputClaims<C> {
    #[opening(BytecodeReadRafAddrClaim, from = BytecodeReadRaf)]
    pub address_phase: C,
}

impl<F: Field> BytecodeReadRafInputClaims<OpeningClaim<F>> {
    pub fn from_upstream(address_phase: OpeningClaim<F>) -> Self {
        Self { address_phase }
    }
}

/// Construction inputs for the full-program bytecode cycle relation. The bytecode
/// rows are borrowed from preprocessing; the points/gammas are the verifier's
/// per-stage cycle bindings and Fiat-Shamir gammas. `stage_cycle_points` /
/// `stage_gammas` are indexed by stage (1..=5) in order.
pub struct BytecodeReadRafCycleInputs<'a, F: Field> {
    pub dimensions: BytecodeReadRafDimensions,
    pub gamma: F,
    pub bytecode: &'a [JoltInstructionRow],
    pub r_address: Vec<F>,
    pub stage_cycle_points: [Vec<F>; 5],
    pub register_read_write_point: Vec<F>,
    pub register_val_evaluation_point: Vec<F>,
    pub entry_bytecode_index: usize,
    pub stage_gammas: [Vec<F>; 5],
    pub committed_chunk_bits: usize,
}

/// The stage-6b bytecode read-RAF cycle phase, full-program mode.
///
/// Its expected output is the bytecode-table public values evaluated at
/// `(r_address, r_cycle)` folded against the committed `BytecodeRa` product — the
/// same quantity `read_raf`'s output expression computes. Rather than resolve each
/// public id individually (which would recompute the `O(2^log_k)` table fold once
/// per public), it OVERRIDES [`SumcheckInstance::expected_output`] to evaluate the
/// public values once and reuse the shared
/// [`stage6_bytecode_read_raf_expected_output`] helper.
///
/// Committed-program mode — which folds the staged `BytecodeValStage` openings into
/// the output expression and draws its publics from a committed evaluation — stays
/// on the verifier's existing committed helper for now.
pub struct BytecodeReadRaf<'a, F: Field> {
    claims: JoltRelationClaims<F>,
    inputs: BytecodeReadRafCycleInputs<'a, F>,
}

impl<'a, F: Field> BytecodeReadRaf<'a, F> {
    pub fn new(inputs: BytecodeReadRafCycleInputs<'a, F>) -> Self {
        Self {
            claims: bytecode::read_raf_cycle_phase(inputs.dimensions),
            inputs,
        }
    }

    /// The `log_t`-variable cycle suffix of a produced `BytecodeRa` opening point
    /// (`chunk ++ r_cycle`).
    fn r_cycle<'p>(&self, opening_point: &'p [F]) -> Result<&'p [F], VerifierError> {
        let log_t = self.inputs.dimensions.log_t();
        opening_point
            .get(opening_point.len() - log_t..)
            .ok_or_else(|| public_input_failed("bytecode cycle opening point shorter than log_t"))
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::BytecodeReadRaf,
        reason: reason.to_string(),
    }
}

impl<F: Field> SumcheckInstance<F> for BytecodeReadRaf<'_, F> {
    type Inputs<C> = BytecodeReadRafInputClaims<C>;
    type Outputs<C> = BytecodeReadRafOutputClaims<C>;

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F> {
        &self.claims
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &BytecodeReadRafInputClaims<C>,
    ) -> Result<BytecodeReadRafOutputClaims<Vec<F>>, VerifierError> {
        let r_cycle = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        let bytecode_ra =
            committed_address_chunks(&self.inputs.r_address, self.inputs.committed_chunk_bits)
                .into_iter()
                .map(|chunk| [chunk.as_slice(), r_cycle.as_slice()].concat())
                .collect();
        Ok(BytecodeReadRafOutputClaims { bytecode_ra })
    }

    fn resolve_challenge(&self, id: &JoltChallengeId) -> Result<F, VerifierError> {
        match id {
            JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Gamma) => {
                Ok(self.inputs.gamma)
            }
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        }
    }

    fn expected_output<C: GetPoint<F>>(
        &self,
        _inputs: &BytecodeReadRafInputClaims<C>,
        outputs: &BytecodeReadRafOutputClaims<OpeningClaim<F>>,
    ) -> Result<F, VerifierError> {
        let opening_point = outputs
            .bytecode_ra
            .first()
            .map(GetPoint::point)
            .ok_or_else(|| public_input_failed("bytecode cycle produced no openings"))?;
        let r_cycle = self.r_cycle(opening_point)?;
        let public_values =
            bytecode::read_raf_public_values::<F>(BytecodeReadRafEvaluationInputs {
                bytecode: self.inputs.bytecode,
                r_address: &self.inputs.r_address,
                r_cycle,
                stage_cycle_points: [
                    &self.inputs.stage_cycle_points[0],
                    &self.inputs.stage_cycle_points[1],
                    &self.inputs.stage_cycle_points[2],
                    &self.inputs.stage_cycle_points[3],
                    &self.inputs.stage_cycle_points[4],
                ],
                register_read_write_point: &self.inputs.register_read_write_point,
                register_val_evaluation_point: &self.inputs.register_val_evaluation_point,
                entry_bytecode_index: self.inputs.entry_bytecode_index,
                stage1_gammas: &self.inputs.stage_gammas[0],
                stage2_gammas: &self.inputs.stage_gammas[1],
                stage3_gammas: &self.inputs.stage_gammas[2],
                stage4_gammas: &self.inputs.stage_gammas[3],
                stage5_gammas: &self.inputs.stage_gammas[4],
            })
            .map_err(public_input_failed)?;
        let bytecode_ra = outputs
            .bytecode_ra
            .iter()
            .map(GetValue::value)
            .collect::<Vec<_>>();
        stage6_bytecode_read_raf_expected_output(Stage6BytecodeReadRafExpectedOutputInputs {
            dimensions: self.inputs.dimensions,
            public_values: &public_values,
            bytecode_ra: &bytecode_ra,
            gamma: self.inputs.gamma,
        })
    }
}

/// Construction inputs for the committed-program bytecode cycle relation.
pub struct BytecodeReadRafCommittedCycleInputs<F: Field> {
    pub dimensions: BytecodeReadRafDimensions,
    pub gamma: F,
    pub r_address: Vec<F>,
    pub stage_cycle_points: [Vec<F>; 5],
    pub entry_bytecode_index: usize,
    pub committed_chunk_bits: usize,
    /// The staged `BytecodeValStage` opening values from the address phase.
    pub val_stages: Vec<F>,
}

/// The stage-6b bytecode read-RAF cycle phase, committed-program mode.
///
/// Mirrors [`BytecodeReadRaf`] but folds the staged `BytecodeValStage` openings
/// into the output expression and draws its publics from a committed bytecode
/// evaluation (`read_raf_committed_public_values`) rather than the full bytecode
/// table. Like the full-mode relation it OVERRIDES
/// [`SumcheckInstance::expected_output`]: the staged Val openings are inputs mixed
/// into the output, and the committed public values are evaluated once.
pub struct BytecodeReadRafCommitted<F: Field> {
    claims: JoltRelationClaims<F>,
    dimensions: BytecodeReadRafDimensions,
    gamma: F,
    r_address: Vec<F>,
    stage_cycle_points: [Vec<F>; 5],
    entry_bytecode_index: usize,
    committed_chunk_bits: usize,
    val_stages: Vec<F>,
}

impl<F: Field> BytecodeReadRafCommitted<F> {
    pub fn new(inputs: BytecodeReadRafCommittedCycleInputs<F>) -> Self {
        Self {
            claims: bytecode::read_raf_cycle_phase_committed(inputs.dimensions),
            dimensions: inputs.dimensions,
            gamma: inputs.gamma,
            r_address: inputs.r_address,
            stage_cycle_points: inputs.stage_cycle_points,
            entry_bytecode_index: inputs.entry_bytecode_index,
            committed_chunk_bits: inputs.committed_chunk_bits,
            val_stages: inputs.val_stages,
        }
    }

    fn r_cycle<'p>(&self, opening_point: &'p [F]) -> Result<&'p [F], VerifierError> {
        let log_t = self.dimensions.log_t();
        opening_point
            .get(opening_point.len() - log_t..)
            .ok_or_else(|| public_input_failed("bytecode cycle opening point shorter than log_t"))
    }
}

impl<F: Field> SumcheckInstance<F> for BytecodeReadRafCommitted<F> {
    type Inputs<C> = BytecodeReadRafInputClaims<C>;
    type Outputs<C> = BytecodeReadRafOutputClaims<C>;

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F> {
        &self.claims
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &BytecodeReadRafInputClaims<C>,
    ) -> Result<BytecodeReadRafOutputClaims<Vec<F>>, VerifierError> {
        let r_cycle = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        let bytecode_ra = committed_address_chunks(&self.r_address, self.committed_chunk_bits)
            .into_iter()
            .map(|chunk| [chunk.as_slice(), r_cycle.as_slice()].concat())
            .collect();
        Ok(BytecodeReadRafOutputClaims { bytecode_ra })
    }

    fn resolve_challenge(&self, id: &JoltChallengeId) -> Result<F, VerifierError> {
        match id {
            JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Gamma) => Ok(self.gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        }
    }

    fn expected_output<C: GetPoint<F>>(
        &self,
        _inputs: &BytecodeReadRafInputClaims<C>,
        outputs: &BytecodeReadRafOutputClaims<OpeningClaim<F>>,
    ) -> Result<F, VerifierError> {
        let opening_point = outputs
            .bytecode_ra
            .first()
            .map(GetPoint::point)
            .ok_or_else(|| public_input_failed("bytecode cycle produced no openings"))?;
        let r_cycle = self.r_cycle(opening_point)?;
        let public_values = bytecode::read_raf_committed_public_values::<F>(
            BytecodeReadRafCommittedEvaluationInputs {
                r_address: &self.r_address,
                r_cycle,
                stage_cycle_points: [
                    &self.stage_cycle_points[0],
                    &self.stage_cycle_points[1],
                    &self.stage_cycle_points[2],
                    &self.stage_cycle_points[3],
                    &self.stage_cycle_points[4],
                ],
                entry_bytecode_index: self.entry_bytecode_index,
            },
        );
        let output_openings = bytecode::read_raf_output_openings(self.dimensions);
        self.claims.output.expression().try_evaluate(
            |id| {
                for (stage, value) in self.val_stages.iter().enumerate() {
                    if *id == bytecode_val_stage_opening(stage) {
                        return Ok(*value);
                    }
                }
                for (index, opening_id) in output_openings.bytecode_ra.iter().enumerate() {
                    if *id == *opening_id {
                        return outputs
                            .bytecode_ra
                            .get(index)
                            .map(|claim| claim.value)
                            .ok_or(VerifierError::MissingOpeningClaim { id: *id });
                    }
                }
                Err(VerifierError::MissingOpeningClaim { id: *id })
            },
            |id| self.resolve_challenge(id),
            |id| match id {
                JoltPublicId::BytecodeReadRaf(public_id) => public_values
                    .value(*public_id)
                    .ok_or(VerifierError::MissingStageClaimPublic { id: *id }),
                _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
            },
        )
    }
}
