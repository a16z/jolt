use jolt_claims::protocols::jolt::{
    geometry::{
        dimensions::TraceDimensions, ram::RamRafEvaluationDimensions,
        spartan::SpartanProductDimensions,
    },
    JoltRelationId,
};
use jolt_claims::NoChallenges;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_program::preprocess::PublicIoMemory;
use jolt_transcript::Transcript;

use super::{
    instruction_claim_reduction::{
        instruction_claim_reduction_input_values_from_upstream, InstructionClaimReduction,
    },
    outputs::{
        Stage2BatchInputClaims, Stage2BatchSumchecks, Stage2ClearOutput, Stage2Output,
        Stage2ZkOutput,
    },
    product_remainder::{product_remainder_input_values_from_uniskip_output, ProductRemainder},
    product_uniskip::{product_uniskip_input_values_from_stage1, ProductUniskip},
    ram_output_check::{RamOutputCheck, RamOutputCheckInputClaims},
    ram_raf_evaluation::{ram_raf_evaluation_input_values_from_upstream, RamRafEvaluation},
    ram_read_write_checking::{ram_read_write_input_values_from_upstream, RamReadWriteChecking},
};
use crate::{
    proof::JoltProof,
    stages::{
        relations::ConcreteSumcheck,
        stage1::{Stage1ClearOutput, Stage1Output},
        uniskip,
        zk::committed,
    },
    verifier::CheckedInputs,
    VerifierError,
};

/// The product uni-skip step's outputs: the tau bindings and the uni-skip
/// reduction challenge are extracted mode-agnostically (clear: the single-entry
/// reduction point; ZK: the committed round challenge) so the batch relations —
/// `ProductRemainder::new` in particular — can be built before the mode branch.
struct ProductUniskipStep<F: Field, C> {
    tau_low: Vec<F>,
    tau_high: F,
    challenge: F,
    verified: ProductUniskipVerified<F, C>,
}

enum ProductUniskipVerified<F: Field, C> {
    Clear,
    Zk(uniskip::UniskipZk<F, C>),
}

/// Assemble the stage-2 batch consumed opening *values* from the upstream clear
/// outputs into the generated `Stage2BatchInputClaims` aggregate. Each per-relation
/// `*_from_upstream` helper wires which upstream opening feeds which downstream
/// input. The product-remainder input is the product uni-skip's output claim (a
/// separate stage-2 sub-sumcheck), not an upstream stage's opening. Errors only
/// under `field-inline`, where the FR claim-reduction inputs are required
/// fail-closed from the stage-1 FR carrier.
pub fn stage2_batch_input_values_from_upstream<F: Field>(
    stage1: &Stage1ClearOutput<F>,
    product_uniskip_output_claim: F,
) -> Result<Stage2BatchInputClaims<F>, VerifierError> {
    Ok(Stage2BatchInputClaims {
        ram_read_write: ram_read_write_input_values_from_upstream(stage1),
        product_remainder: product_remainder_input_values_from_uniskip_output(
            product_uniskip_output_claim,
        ),
        instruction_claim_reduction: instruction_claim_reduction_input_values_from_upstream(stage1),
        #[cfg(feature = "field-inline")]
        field_registers_claim_reduction:
            super::field_registers_claim_reduction::field_registers_claim_reduction_input_values_from_upstream(
                stage1,
            )?,
        ram_raf_evaluation: ram_raf_evaluation_input_values_from_upstream(stage1),
        ram_output_check: RamOutputCheckInputClaims::default(),
    })
}

#[jolt_verifier_derive::fs_scope(Stage2)]
pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    stage1: &Stage1Output<PCS::Field, VC::Output>,
) -> Result<Stage2Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    match (checked.zk, stage1) {
        (true, Stage1Output::Clear(_)) => {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage1" });
        }
        (false, Stage1Output::Zk(_)) => {
            return Err(VerifierError::ExpectedClearProof { field: "stage1" });
        }
        _ => {}
    }

    let log_t = crate::num::ilog2(checked.trace_length);
    let log_k = crate::num::ilog2(checked.ram_K);
    let trace_dimensions = TraceDimensions::new(log_t);
    let read_write_dimensions = proof.rw_config.ram_dimensions(log_t, log_k);
    let product_dimensions = SpartanProductDimensions::new(log_t);
    let raf_dimensions =
        RamRafEvaluationDimensions::try_from(read_write_dimensions).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRafEvaluation,
                reason: error.to_string(),
            }
        })?;

    let uniskip =
        verify_product_uniskip::<PCS, VC, T, ZkProof>(checked, proof, transcript, stage1)?;

    // Build the batch relations once, pre-branch; each owns its input/output
    // claim algebra (single-sourced with its jolt-claims formula and the BlindFold
    // constraint). The product uni-skip stays hand-coded above.
    let lowest_address = checked.public_io.memory_layout.get_lowest_address();
    let public_memory = PublicIoMemory::new(&checked.public_io).map_err(|error| {
        VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamOutputCheck,
            reason: error.to_string(),
        }
    })?;
    let sumchecks = Stage2BatchSumchecks {
        ram_read_write: RamReadWriteChecking::new(
            read_write_dimensions,
            log_k,
            uniskip.tau_low.clone(),
        ),
        product_remainder: ProductRemainder::new(
            product_dimensions,
            uniskip.challenge,
            uniskip.tau_high,
            uniskip.tau_low.clone(),
        ),
        instruction_claim_reduction: InstructionClaimReduction::new(
            trace_dimensions,
            uniskip.tau_low.clone(),
        ),
        #[cfg(feature = "field-inline")]
        field_registers_claim_reduction:
            super::field_registers_claim_reduction::FieldRegistersClaimReduction::new(
                jolt_claims::protocols::field_inline::FieldRegistersTraceDimensions::new(log_t),
                uniskip.tau_low.clone(),
            ),
        ram_raf_evaluation: RamRafEvaluation::new(
            read_write_dimensions,
            raf_dimensions,
            log_k,
            lowest_address,
            uniskip.tau_low.clone(),
        ),
        ram_output_check: RamOutputCheck::new(read_write_dimensions, public_memory),
    };

    // Draw each relation's challenges in declaration order: the RAM read-write
    // gamma, the instruction claim-reduction gamma, under `field-inline` the FR
    // claim-reduction gamma (each a single `challenge_scalar`), then the RAM
    // output-check address reference point (the last member's `draw_challenges`
    // override — one raw `challenge()` per RAM address variable, landing after
    // the gammas as the inline draw did). The drawn challenges feed the
    // input/output claims and populate the stage aggregate carried downstream.
    let challenges = sumchecks.draw_challenges(transcript)?;

    // Every member's input points are empty (each derives its output points from its
    // own sumcheck point).
    let input_points = sumchecks.empty_input_points();

    if checked.zk {
        let ProductUniskipVerified::Zk(product_uniskip) = uniskip.verified else {
            return Err(VerifierError::ExpectedCommittedProof {
                field: "stage2_uni_skip_first_round_proof",
            });
        };
        let consistency = sumchecks.verify_zk(&proof.stages.stage2_sumcheck_proof, transcript)?;
        // The committed shell carries the curated row order: the member
        // openings plus (under `field-inline`) the three FR product-appendage
        // rows spliced after the product-remainder outputs — the clear absorb
        // order exactly.
        let output_claim_count = sumchecks.output_claim_count();
        #[cfg(feature = "field-inline")]
        let output_claim_count = output_claim_count
            .checked_add(
                jolt_claims::protocols::field_inline::geometry::product::selected_product_remainder_output_openings()
                    .len(),
            )
            .ok_or_else(|| VerifierError::StageClaimSumcheckFailed {
                stage: format!("{:?}", JoltRelationId::SpartanProductVirtualization),
                reason: "composed stage-2 output-claim count overflows usize".to_string(),
            })?;
        let batch_output_claims = committed::verify_output_claim_commitments(
            checked,
            &proof.stages.stage2_sumcheck_proof,
            "stage2_sumcheck_proof",
            output_claim_count,
            JoltRelationId::RamReadWriteChecking,
        )?;
        let output_points =
            sumchecks.derive_opening_points(&consistency.challenges(), &input_points)?;

        return Ok(Stage2Output::Zk(Stage2ZkOutput {
            challenges,
            product_uniskip_challenge: uniskip.challenge,
            product_tau_low: uniskip.tau_low,
            product_tau_high: uniskip.tau_high,
            product_uniskip_consistency: product_uniskip.consistency,
            product_uniskip_output_claims: product_uniskip.output_claims,
            batch_consistency: consistency,
            batch_output_claims,
            output_points,
        }));
    }

    let ProductUniskipVerified::Clear = uniskip.verified else {
        return Err(VerifierError::ExpectedClearProof {
            field: "stage2_uni_skip_first_round_proof",
        });
    };
    let stage1 = stage1.clear()?;
    let claims = &proof.clear_claims()?.stage2;
    sumchecks.validate_output_claims(&claims.batch_outputs)?;

    // The three FR product-row openings ride the proof beside the batch
    // outputs; the composed remainder's FR lanes factor over them. Required
    // fail-closed on FR-on builds.
    #[cfg(feature = "field-inline")]
    let field_inline_product = {
        let field_inline_product =
            claims
                .field_inline_product
                .clone()
                .ok_or(VerifierError::MissingProofPayload {
                    field: "claims.stage2.field_inline_product",
                })?;
        sumchecks
            .product_remainder
            .set_field_inline_outputs(field_inline_product.clone())?;
        // WHY: the spec's alias table (field-inline-protocol.md, "Stage 2
        // Composition") aliases the FR claim-reduction outputs into the FR
        // product-remainder rows. The generated alias machinery resolves
        // canonical sources across batch members' typed claims only, and the FR
        // product openings ride the proof as an appendage — so the alias is
        // enforced as this explicit equality instead. It is a
        // same-polynomial-at-the-same-point statement for the same structural
        // reason as the jolt aliases: both relations bind the same batch-point
        // suffix and derive the same reversed opening point (pinned by
        // `field_registers_claim_reduction_shares_the_product_remainder_point`).
        validate_field_inline_product_aliases(&claims.batch_outputs, &field_inline_product)?;
        field_inline_product
    };

    let input_values =
        stage2_batch_input_values_from_upstream(stage1, claims.product_uniskip_output_claim)?;

    let output_points = sumchecks.verify_clear(
        &input_values,
        &input_points,
        &challenges,
        &claims.batch_outputs,
        &proof.stages.stage2_sumcheck_proof,
        transcript,
        2,
    )?;

    #[cfg(not(feature = "field-inline"))]
    sumchecks.append_output_claims(transcript, &claims.batch_outputs);
    // The curated FR-on absorb: the FR product appendage splices in after the
    // product-remainder outputs, per the spec's committed output row order.
    #[cfg(feature = "field-inline")]
    sumchecks.append_output_claims(transcript, &claims.batch_outputs, &field_inline_product);

    Ok(Stage2Output::Clear(Stage2ClearOutput {
        output_values: claims.batch_outputs.clone(),
        output_points,
        product_tau_low: uniskip.tau_low,
    }))
}

/// The spec's stage-2 alias table (`field-inline-protocol.md`, "Stage 2
/// Composition") as its polynomial list: each FR claim-reduction output aliases
/// the FR product-remainder opening of the same polynomial. Shared by the clear
/// equality check below and the BlindFold lowering's `OpeningEquality` rows, so
/// the two enforcement paths cannot drift.
#[cfg(feature = "field-inline")]
pub(crate) fn field_inline_product_alias_polynomials(
) -> [jolt_claims::protocols::field_inline::FieldInlineVirtualPolynomial; 3] {
    use jolt_claims::protocols::field_inline::FieldInlineVirtualPolynomial;
    [
        FieldInlineVirtualPolynomial::FieldRs1Value,
        FieldInlineVirtualPolynomial::FieldRs2Value,
        FieldInlineVirtualPolynomial::FieldRdValue,
    ]
}

/// Enforce the spec's stage-2 alias table: each FR claim-reduction output
/// equals the FR product-remainder opening of the same polynomial (see the WHY
/// at the call site). Value-only, like the generated `validate_aliases`.
#[cfg(feature = "field-inline")]
fn validate_field_inline_product_aliases<F: Field>(
    batch_outputs: &super::outputs::Stage2BatchOutputClaims<F>,
    field_inline_product: &super::outputs::FieldRegistersProductOutputClaims<F>,
) -> Result<(), VerifierError> {
    use jolt_claims::protocols::field_inline::{FieldInlineOpeningId, FieldInlineRelationId};
    use jolt_claims::OutputClaims as _;

    let reduction = &batch_outputs.field_registers_claim_reduction;
    for polynomial in field_inline_product_alias_polynomials() {
        let aliased_id = FieldInlineOpeningId::virtual_polynomial(
            polynomial,
            FieldInlineRelationId::FieldRegistersClaimReduction,
        );
        let source_id = FieldInlineOpeningId::virtual_polynomial(
            polynomial,
            FieldInlineRelationId::FieldRegistersProduct,
        );
        let aliased =
            reduction
                .resolve_output(&aliased_id)
                .ok_or(VerifierError::MissingOpeningClaim {
                    id: aliased_id.into(),
                })?;
        let source = field_inline_product.resolve_output(&source_id).ok_or(
            VerifierError::MissingOpeningClaim {
                id: source_id.into(),
            },
        )?;
        if aliased != source {
            return Err(VerifierError::StageClaimOpeningMismatch {
                stage: format!("{:?}", FieldInlineRelationId::FieldRegistersClaimReduction),
                left: aliased_id.into(),
                right: source_id.into(),
            });
        }
    }
    Ok(())
}

/// The product uni-skip's low binding tau_low: the tail (`[1..]`) of stage
/// 1's raw remainder point, reversed. Shared by `verify_product_uniskip` and
/// the prove-side stage-2 recipe, so the derivation cannot drift.
pub fn product_tau_low<F: Field>(
    stage1_remainder: &[F],
    log_t: usize,
) -> Result<Vec<F>, VerifierError> {
    let mut tau_low = stage1_remainder
        .get(1..)
        .ok_or_else(|| VerifierError::StageClaimSumcheckFailed {
            stage: format!("{:?}", JoltRelationId::SpartanProductVirtualization),
            reason: "Stage 1 remainder challenge vector is empty".to_string(),
        })?
        .to_vec();
    if tau_low.len() != log_t {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: format!("{:?}", JoltRelationId::SpartanProductVirtualization),
            reason: format!(
                "Stage 1 remainder challenge tail length mismatch: expected {log_t}, got {}",
                tau_low.len()
            ),
        });
    }
    tau_low.reverse();
    Ok(tau_low)
}

fn verify_product_uniskip<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    stage1: &Stage1Output<PCS::Field, VC::Output>,
) -> Result<ProductUniskipStep<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let log_t = crate::num::ilog2(checked.trace_length);
    let product_dimensions = SpartanProductDimensions::new(log_t);
    let tau_low = product_tau_low(&stage1.remainder_point(), log_t)?;

    let tau_high = uniskip::draw_spartan_product_tau_high(transcript);
    let uniskip_params = uniskip::UniskipParams::spartan_product();
    match stage1 {
        Stage1Output::Clear(stage1) => {
            let claims = &proof.clear_claims()?.stage2;
            let uniskip_relation = ProductUniskip::new(product_dimensions, tau_high);
            // The FR lanes' input claims (FieldProduct/FieldInvProduct at the
            // FR Spartan-outer segment) enter the composed uni-skip input
            // exactly as the ordinary lanes do — Lagrange-weighted at the lane
            // indices following them. Required fail-closed on FR-on builds.
            #[cfg(feature = "field-inline")]
            {
                let field_inline = stage1.field_inline_output_values.as_ref().ok_or(
                    VerifierError::MissingProofPayload {
                        field: "stage1.field_inline_output_values",
                    },
                )?;
                uniskip_relation
                    .set_field_inline_inputs(field_inline.product, field_inline.inv_product)?;
            }
            let uniskip_input_values = product_uniskip_input_values_from_stage1(stage1);
            let uniskip_input_claim =
                uniskip_relation.input_claim(&uniskip_input_values, &NoChallenges::default())?;

            let challenge = uniskip::verify_clear(
                &proof.stages.stage2_uni_skip_first_round_proof,
                &uniskip_params,
                uniskip_input_claim,
                claims.product_uniskip_output_claim,
                transcript,
            )?;
            Ok(ProductUniskipStep {
                tau_low,
                tau_high,
                challenge,
                verified: ProductUniskipVerified::Clear,
            })
        }
        Stage1Output::Zk(_) => {
            let verified = uniskip::verify_zk(
                checked,
                &proof.stages.stage2_uni_skip_first_round_proof,
                &uniskip_params,
                transcript,
            )?;
            Ok(ProductUniskipStep {
                tau_low,
                tau_high,
                challenge: verified.challenge,
                verified: ProductUniskipVerified::Zk(verified),
            })
        }
    }
}

#[cfg(all(test, feature = "field-inline"))]
mod field_inline_tests {
    use super::super::outputs::{
        FieldRegistersClaimReductionOutputClaims, FieldRegistersProductOutputClaims,
        InstructionClaimReductionOutputClaims, ProductRemainderOutputClaims,
        RamOutputCheckOutputClaims, RamRafEvaluationOutputClaims, RamReadWriteOutputClaims,
        Stage2BatchOutputClaims,
    };
    use super::validate_field_inline_product_aliases;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn batch_outputs_with_reduction(rd: u64, rs1: u64, rs2: u64) -> Stage2BatchOutputClaims<Fr> {
        Stage2BatchOutputClaims {
            ram_read_write: RamReadWriteOutputClaims {
                val: fr(1),
                ra: fr(2),
                inc: fr(3),
            },
            product_remainder: ProductRemainderOutputClaims {
                left_instruction_input: fr(4),
                right_instruction_input: fr(5),
                jump_flag: fr(6),
                write_lookup_output_to_rd: fr(7),
                lookup_output: fr(8),
                branch_flag: fr(9),
                next_is_noop: fr(10),
                virtual_instruction: fr(11),
            },
            instruction_claim_reduction: InstructionClaimReductionOutputClaims {
                lookup_output: fr(8),
                left_lookup_operand: fr(12),
                right_lookup_operand: fr(13),
                left_instruction_input: fr(4),
                right_instruction_input: fr(5),
            },
            field_registers_claim_reduction: FieldRegistersClaimReductionOutputClaims {
                rd_value: fr(rd),
                rs1_value: fr(rs1),
                rs2_value: fr(rs2),
            },
            ram_raf_evaluation: RamRafEvaluationOutputClaims { ram_ra: fr(14) },
            ram_output_check: RamOutputCheckOutputClaims { val_final: fr(15) },
        }
    }

    fn appendage() -> FieldRegistersProductOutputClaims<Fr> {
        FieldRegistersProductOutputClaims {
            rs1_value: fr(21),
            rs2_value: fr(22),
            rd_value: fr(23),
        }
    }

    #[test]
    fn field_inline_product_aliases_accept_matching_values() {
        let batch = batch_outputs_with_reduction(23, 21, 22);
        assert!(validate_field_inline_product_aliases(&batch, &appendage()).is_ok());
    }

    #[test]
    fn field_inline_product_aliases_reject_rs1_mismatch() {
        let batch = batch_outputs_with_reduction(23, 99, 22);
        assert!(validate_field_inline_product_aliases(&batch, &appendage()).is_err());
    }

    #[test]
    fn field_inline_product_aliases_reject_rs2_mismatch() {
        let batch = batch_outputs_with_reduction(23, 21, 99);
        assert!(validate_field_inline_product_aliases(&batch, &appendage()).is_err());
    }

    #[test]
    fn field_inline_product_aliases_reject_rd_mismatch() {
        let batch = batch_outputs_with_reduction(99, 21, 22);
        assert!(validate_field_inline_product_aliases(&batch, &appendage()).is_err());
    }
}
