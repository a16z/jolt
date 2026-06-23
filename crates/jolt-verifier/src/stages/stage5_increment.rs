//! Lattice-only increment virtualization verifier stage.

use jolt_claims::protocols::jolt::{
    formulas::{
        dimensions::{TraceDimensions, REGISTER_ADDRESS_BITS},
        lattice, ram,
    },
    IncVirtualizationChallenge, IncVirtualizationPublic, JoltChallengeId, JoltPublicId,
    JoltRelationClaims, JoltRelationId, RamRaClaimReductionChallenge,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_poly::try_eq_mle;
use jolt_sumcheck::{BatchedSumcheckVerifier, SumcheckClaim};
use jolt_transcript::Transcript;
use serde::{Deserialize, Serialize};

use crate::{
    config::{validate_protocol_config, PcsFamily},
    proof::JoltProof,
    stages::{
        stage2::{Stage2ClearOutput, Stage2Output},
        stage4::{Stage4ClearOutput, Stage4Output},
        stage5::{self, Stage5ClearOutput, Stage5Output},
    },
    verifier::CheckedInputs,
    VerifierError,
};

#[derive(Clone, Copy)]
pub enum Deps<'a, F: Field> {
    Clear {
        stage2: &'a Stage2ClearOutput<F>,
        stage4: &'a Stage4ClearOutput<F>,
        stage5: &'a Stage5ClearOutput<F>,
    },
    Zk,
    Mixed,
}

pub fn deps<'a, F: Field, C>(
    stage2: &'a Stage2Output<F, C>,
    stage4: &'a Stage4Output<F, C>,
    stage5: &'a Stage5Output<F, C>,
) -> Deps<'a, F> {
    match (stage2, stage4, stage5) {
        (Stage2Output::Clear(stage2), Stage4Output::Clear(stage4), Stage5Output::Clear(stage5)) => {
            Deps::Clear {
                stage2,
                stage4,
                stage5,
            }
        }
        (Stage2Output::Zk(_), Stage4Output::Zk(_), Stage5Output::Zk(_)) => Deps::Zk,
        _ => Deps::Mixed,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct Stage5IncrementClaims<F: Field> {
    pub ram_ra_claim_reduction: stage5::inputs::RamRaClaimReductionOutputOpeningClaims<F>,
    pub inc_virtualization: IncVirtualizationOutputClaims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct IncVirtualizationOutputClaims<F: Field> {
    pub inc: F,
    pub store: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5IncrementClearOutput<F: Field> {
    pub ram_gamma: F,
    pub inc_gamma: F,
    pub output_claims: Stage5IncrementClaims<F>,
    pub ram_ra_claim_reduction: VerifiedRamRaClaimReductionSumcheck<F>,
    pub batch: VerifiedIncVirtualizationSumcheck<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedRamRaClaimReductionSumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedIncVirtualizationSumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub expected_output_claim: F,
}

pub type Stage5IncrementOutput<F> = Option<Stage5IncrementClearOutput<F>>;

pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    deps: Deps<'_, PCS::Field>,
) -> Result<Stage5IncrementOutput<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let lattice = validate_protocol_config(&proof.protocol)? == PcsFamily::Lattice;
    let claims = proof
        .clear_claims()
        .ok()
        .and_then(|claims| claims.stage5_increment.as_ref());
    let proof_payload = proof.stages.stage5_increment_sumcheck_proof.as_ref();

    validate_stage5_increment_shape(
        lattice,
        checked.zk,
        claims.is_some(),
        proof_payload.is_some(),
    )?;
    if !lattice {
        return Ok(None);
    }

    let Deps::Clear {
        stage2,
        stage4,
        stage5,
    } = deps
    else {
        return Err(VerifierError::ExpectedClearProof {
            field: "stage5_increment",
        });
    };
    let claims = claims.ok_or(VerifierError::MissingOpeningClaim {
        id: lattice::inc_virtualization_inc_opening(),
    })?;
    let proof_payload = proof_payload.ok_or(VerifierError::MissingStageProof {
        field: "stage5_increment_sumcheck_proof",
    })?;

    let log_t = checked.trace_length.ilog2() as usize;
    let log_k = checked.ram_K.ilog2() as usize;
    let trace_dimensions = TraceDimensions::new(log_t);
    let ram_relation_claim = ram::ra_claim_reduction::<PCS::Field>(trace_dimensions);
    let relation_claim = lattice::inc_virtualization_claim::<PCS::Field>(trace_dimensions);
    validate_compressed_stage_claim(&ram_relation_claim)?;
    validate_compressed_stage_claim(&relation_claim)?;

    let ram_gamma = transcript.challenge_scalar();
    let inc_gamma = transcript.challenge_scalar();
    let [ram_ra_raf, ram_ra_read_write, ram_ra_val_check] =
        ram::ra_claim_reduction_input_openings();
    let ram_input_claim = ram_relation_claim.input.expression().try_evaluate(
        |id| match *id {
            id if id == ram_ra_raf => Ok(stage2.output_claims.ram_raf_evaluation),
            id if id == ram_ra_read_write => Ok(stage2.output_claims.ram_read_write.ra),
            id if id == ram_ra_val_check => Ok(stage4.output_claims.ram_val_check.ram_ra),
            id => Err(VerifierError::MissingOpeningClaim { id }),
        },
        |id| match id {
            JoltChallengeId::RamRaClaimReduction(RamRaClaimReductionChallenge::Gamma) => {
                Ok(ram_gamma)
            }
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
    )?;
    let [ram_read_write, ram_val_check, rd_read_write, rd_val_evaluation] =
        lattice::inc_virtualization_input_openings();
    let input_claim = relation_claim.input.expression().try_evaluate(
        |id| match *id {
            id if id == ram_read_write => Ok(stage2.output_claims.ram_read_write.inc),
            id if id == ram_val_check => Ok(stage4.output_claims.ram_val_check.ram_inc),
            id if id == rd_read_write => Ok(stage4.output_claims.registers_read_write.rd_inc),
            id if id == rd_val_evaluation => {
                Ok(stage5.output_claims.registers_val_evaluation.rd_inc)
            }
            id => Err(VerifierError::MissingOpeningClaim { id }),
        },
        |id| match id {
            JoltChallengeId::IncVirtualization(IncVirtualizationChallenge::Gamma) => Ok(inc_gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
    )?;

    let batch = BatchedSumcheckVerifier::verify_compressed_boolean(
        &[
            SumcheckClaim::new(
                ram_relation_claim.sumcheck.rounds,
                ram_relation_claim.sumcheck.degree,
                ram_input_claim,
            ),
            SumcheckClaim::new(
                relation_claim.sumcheck.rounds,
                relation_claim.sumcheck.degree,
                input_claim,
            ),
        ],
        proof_payload,
        transcript,
    )
    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
        stage: JoltRelationId::IncVirtualization,
        reason: error.to_string(),
    })?;
    let ram_point = batch
        .try_instance_point(ram_relation_claim.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RamRaClaimReduction,
            reason: error.to_string(),
        })?;
    let ram_cycle = trace_dimensions
        .cycle_opening_point(ram_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamRaClaimReduction,
            reason: error.to_string(),
        })?;
    let ram_raf_opening_point = &stage2.batch.ram_raf_evaluation.opening_point;
    let ram_read_write_opening_point = &stage2.batch.ram_read_write.opening_point;
    let ram_val_check_opening_point = &stage4.batch.ram_val_check.opening_point;
    for (label, opening_point) in [
        ("RAM RAF evaluation", ram_raf_opening_point),
        ("RAM read-write", ram_read_write_opening_point),
        ("RAM value check", ram_val_check_opening_point),
    ] {
        if opening_point.len() != log_k + log_t {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRaClaimReduction,
                reason: format!(
                    "{label} opening point length mismatch: expected {}, got {}",
                    log_k + log_t,
                    opening_point.len()
                ),
            });
        }
    }
    let (ram_raf_address, ram_raf_cycle) = ram_raf_opening_point.split_at(log_k);
    let (ram_read_write_address, ram_read_write_cycle) =
        ram_read_write_opening_point.split_at(log_k);
    let (ram_val_check_address, ram_val_check_cycle) = ram_val_check_opening_point.split_at(log_k);
    if ram_raf_address != ram_read_write_address {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::RamRaClaimReduction,
            left: ram_ra_raf,
            right: ram_ra_read_write,
        });
    }
    if ram_val_check_address != ram_read_write_address {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::RamRaClaimReduction,
            left: ram_ra_val_check,
            right: ram_ra_read_write,
        });
    }
    let ram_public_values = ram::RamRaClaimReductionPublicValues {
        eq_cycle_raf: eq_cycle(
            JoltRelationId::RamRaClaimReduction,
            &ram_cycle,
            ram_raf_cycle,
        )?,
        eq_cycle_read_write: eq_cycle(
            JoltRelationId::RamRaClaimReduction,
            &ram_cycle,
            ram_read_write_cycle,
        )?,
        eq_cycle_val_check: eq_cycle(
            JoltRelationId::RamRaClaimReduction,
            &ram_cycle,
            ram_val_check_cycle,
        )?,
    };
    let [ram_ra_reduced] = ram::ra_claim_reduction_output_openings();
    let ram_expected_output_claim = ram_relation_claim.output.expression().try_evaluate(
        |id| match *id {
            id if id == ram_ra_reduced => Ok(claims.ram_ra_claim_reduction.ram_ra),
            id => Err(VerifierError::MissingOpeningClaim { id }),
        },
        |id| match id {
            JoltChallengeId::RamRaClaimReduction(RamRaClaimReductionChallenge::Gamma) => {
                Ok(ram_gamma)
            }
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| match id {
            JoltPublicId::RamRaClaimReduction(public_id) => Ok(ram_public_values.value(*public_id)),
            _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
        },
    )?;
    let ram_opening_point = [ram_read_write_address, ram_cycle.as_slice()].concat();

    let point = batch
        .try_instance_point(relation_claim.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::IncVirtualization,
            reason: error.to_string(),
        })?;
    let opening_point = trace_dimensions
        .cycle_opening_point(point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::IncVirtualization,
            reason: error.to_string(),
        })?;

    let (_, ram_read_write_cycle) = stage2.batch.ram_read_write.opening_point.split_at(log_k);
    let (_, ram_val_check_cycle) = stage4.batch.ram_val_check.opening_point.split_at(log_k);
    let (_, registers_read_write_cycle) = stage4
        .batch
        .registers_read_write
        .opening_point
        .split_at(REGISTER_ADDRESS_BITS);
    let (_, registers_val_evaluation_cycle) = stage5
        .batch
        .registers_val_evaluation
        .opening_point
        .split_at(REGISTER_ADDRESS_BITS);

    let eq_ram_read_write = eq_cycle(
        JoltRelationId::IncVirtualization,
        &opening_point,
        ram_read_write_cycle,
    )?;
    let eq_ram_val_check = eq_cycle(
        JoltRelationId::IncVirtualization,
        &opening_point,
        ram_val_check_cycle,
    )?;
    let eq_registers_read_write = eq_cycle(
        JoltRelationId::IncVirtualization,
        &opening_point,
        registers_read_write_cycle,
    )?;
    let eq_registers_val_evaluation = eq_cycle(
        JoltRelationId::IncVirtualization,
        &opening_point,
        registers_val_evaluation_cycle,
    )?;

    let expected_output_claim = relation_claim.output.expression().try_evaluate(
        |id| match *id {
            id if id == lattice::inc_virtualization_inc_opening() => {
                Ok(claims.inc_virtualization.inc)
            }
            id if id == lattice::inc_virtualization_store_opening() => {
                Ok(claims.inc_virtualization.store)
            }
            id => Err(VerifierError::MissingOpeningClaim { id }),
        },
        |id| match id {
            JoltChallengeId::IncVirtualization(IncVirtualizationChallenge::Gamma) => Ok(inc_gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| match id {
            JoltPublicId::IncVirtualization(IncVirtualizationPublic::EqRamReadWrite) => {
                Ok(eq_ram_read_write)
            }
            JoltPublicId::IncVirtualization(IncVirtualizationPublic::EqRamValCheck) => {
                Ok(eq_ram_val_check)
            }
            JoltPublicId::IncVirtualization(IncVirtualizationPublic::EqRegistersReadWrite) => {
                Ok(eq_registers_read_write)
            }
            JoltPublicId::IncVirtualization(IncVirtualizationPublic::EqRegistersValEvaluation) => {
                Ok(eq_registers_val_evaluation)
            }
            _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
        },
    )?;

    let [ram_coefficient, inc_coefficient] = batch.batching_coefficients.as_slice() else {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::IncVirtualization,
            reason: format!(
                "Stage 5 increment batch verifier returned {} coefficients for two instances",
                batch.batching_coefficients.len()
            ),
        });
    };
    let expected_final_claim =
        *ram_coefficient * ram_expected_output_claim + *inc_coefficient * expected_output_claim;
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch {
            stage: JoltRelationId::IncVirtualization,
        });
    }

    transcript.append_labeled(b"opening_claim", &claims.ram_ra_claim_reduction.ram_ra);
    transcript.append_labeled(b"opening_claim", &claims.inc_virtualization.inc);
    transcript.append_labeled(b"opening_claim", &claims.inc_virtualization.store);

    Ok(Some(Stage5IncrementClearOutput {
        ram_gamma,
        inc_gamma,
        output_claims: claims.clone(),
        ram_ra_claim_reduction: VerifiedRamRaClaimReductionSumcheck {
            input_claim: ram_input_claim,
            sumcheck_point: ram_point.to_vec(),
            opening_point: ram_opening_point,
            expected_output_claim: ram_expected_output_claim,
        },
        batch: VerifiedIncVirtualizationSumcheck {
            input_claim,
            sumcheck_point: point.to_vec(),
            opening_point,
            expected_output_claim,
        },
    }))
}

fn validate_stage5_increment_shape(
    lattice: bool,
    zk: bool,
    has_claims: bool,
    has_proof_payload: bool,
) -> Result<(), VerifierError> {
    if !lattice {
        if has_claims {
            return Err(VerifierError::UnexpectedOpeningClaim {
                id: lattice::inc_virtualization_inc_opening(),
            });
        }
        if has_proof_payload {
            return Err(VerifierError::UnexpectedStageProof {
                field: "stage5_increment_sumcheck_proof",
            });
        }
        return Ok(());
    }
    if zk {
        return Err(VerifierError::ExpectedClearProof {
            field: "stage5_increment",
        });
    }
    if !has_claims {
        return Err(VerifierError::MissingOpeningClaim {
            id: lattice::inc_virtualization_inc_opening(),
        });
    }
    if !has_proof_payload {
        return Err(VerifierError::MissingStageProof {
            field: "stage5_increment_sumcheck_proof",
        });
    }
    Ok(())
}

fn eq_cycle<F: Field>(stage: JoltRelationId, left: &[F], right: &[F]) -> Result<F, VerifierError> {
    try_eq_mle(left, right).map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage,
        reason: error.to_string(),
    })
}

fn validate_compressed_stage_claim<F: Field>(
    claim: &JoltRelationClaims<F>,
) -> Result<(), VerifierError> {
    if claim.sumcheck.degree == 0 {
        return Err(VerifierError::InvalidStageSumcheckDegree {
            stage: claim.id,
            degree: claim.sumcheck.degree,
        });
    }
    if !matches!(
        claim.sumcheck.domain,
        jolt_claims::protocols::jolt::JoltSumcheckDomain::BooleanHypercube
    ) {
        return Err(VerifierError::CompressedStageClaimRequiresBooleanDomain { stage: claim.id });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "test setup should fail loudly when helper contracts change"
    )]

    use super::*;

    #[test]
    fn curve_stage5_increment_rejects_lattice_claims() {
        let error = validate_stage5_increment_shape(false, false, true, false)
            .expect_err("curve mode must reject lattice increment claims");
        assert!(matches!(
            error,
            VerifierError::UnexpectedOpeningClaim { id }
                if id == lattice::inc_virtualization_inc_opening()
        ));
    }

    #[test]
    fn curve_stage5_increment_rejects_lattice_proof_payload() {
        let error = validate_stage5_increment_shape(false, false, false, true)
            .expect_err("curve mode must reject lattice increment proof payload");
        assert!(matches!(
            error,
            VerifierError::UnexpectedStageProof {
                field: "stage5_increment_sumcheck_proof"
            }
        ));
    }

    #[test]
    fn lattice_stage5_increment_requires_claims_and_proof_payload() {
        let error = validate_stage5_increment_shape(true, false, false, true)
            .expect_err("lattice mode must require increment claims");
        assert!(matches!(
            error,
            VerifierError::MissingOpeningClaim { id }
                if id == lattice::inc_virtualization_inc_opening()
        ));

        let error = validate_stage5_increment_shape(true, false, true, false)
            .expect_err("lattice mode must require increment proof payload");
        assert!(matches!(
            error,
            VerifierError::MissingStageProof {
                field: "stage5_increment_sumcheck_proof"
            }
        ));

        validate_stage5_increment_shape(true, false, true, true)
            .expect("lattice clear mode accepts complete increment payload");
    }

    #[test]
    fn lattice_stage5_increment_rejects_zk_mode() {
        let error = validate_stage5_increment_shape(true, true, true, true)
            .expect_err("lattice increment virtualization is clear-only");
        assert!(matches!(
            error,
            VerifierError::ExpectedClearProof {
                field: "stage5_increment"
            }
        ));
    }
}
