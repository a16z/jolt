//! Stage 2's field-inline seam: every FR-specific divergence of the stage-2
//! verifier in one place — the FR claim-reduction member and its input wiring,
//! the product uni-skip/remainder appendage attachment, the spec's alias
//! table, the composed committed row count, and the curated absorb.
//! `verify.rs` interacts with the FR protocol only through the functions here
//! (plus the FR carrier fields on the outputs, which are proof shape).

use jolt_claims::protocols::field_inline::relations::product::FieldRegistersProductOutputClaims;
use jolt_claims::protocols::field_inline::{
    FieldInlineOpeningId, FieldInlineRelationId, FieldInlineVirtualPolynomial,
    FieldRegistersTraceDimensions,
};
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_claims::OutputClaims as _;
use jolt_field::JoltField;
use jolt_transcript::Transcript;

use super::field_registers_claim_reduction::{
    FieldRegistersClaimReduction, FieldRegistersClaimReductionInputClaims,
};
use super::instruction_claim_reduction::InstructionClaimReduction;
use super::outputs::{
    ProductRemainder, RamOutputCheck, RamRafEvaluation, RamReadWriteChecking,
    Stage2BatchOutputClaims, Stage2BatchSumchecks, Stage2OutputClaims,
};
use super::product_uniskip::ProductUniskip;
use crate::stages::relations::absorbed_opening_values;
use crate::stages::stage1::Stage1ClearOutput;
use crate::VerifierError;

/// The stage-2 FR batch member. The FR claim reduction shares the trace domain
/// (`log_T` rounds) with the product remainder, so both bind the same batch
/// suffix — the spec's `r_prod` sharing.
pub fn claim_reduction_member<F: JoltField>(
    log_t: usize,
    tau_low: Vec<F>,
) -> FieldRegistersClaimReduction<F> {
    FieldRegistersClaimReduction::new(FieldRegistersTraceDimensions::new(log_t), tau_low)
}

/// Wire the consumed FR value opening *values* from stage 1's composed outer
/// sumcheck (the FR-local appended segment). Fail-closed: an FR-on proof whose
/// stage-1 carrier lacks the FR payload cannot feed this reduction.
pub fn claim_reduction_inputs<F: JoltField>(
    stage1: &Stage1ClearOutput<F>,
) -> Result<FieldRegistersClaimReductionInputClaims<F>, VerifierError> {
    let outer =
        stage1
            .field_inline_output_values
            .as_ref()
            .ok_or(VerifierError::MissingProofPayload {
                field: "stage1.field_inline_output_values",
            })?;
    Ok(FieldRegistersClaimReductionInputClaims {
        rd_value: outer.rd_value,
        rs1_value: outer.rs1_value,
        rs2_value: outer.rs2_value,
    })
}

/// Supply the FR lane input claims (`FieldProduct`/`FieldInvProduct` at the FR
/// Spartan-outer segment) to the composed product uni-skip. They enter the
/// composed input exactly as the ordinary lanes do — Lagrange-weighted at the
/// lane indices following them. Fail-closed on a missing stage-1 FR carrier.
pub fn attach_uniskip_inputs<F: JoltField>(
    uniskip: &ProductUniskip<F>,
    stage1: &Stage1ClearOutput<F>,
) -> Result<(), VerifierError> {
    let field_inline =
        stage1
            .field_inline_output_values
            .as_ref()
            .ok_or(VerifierError::MissingProofPayload {
                field: "stage1.field_inline_output_values",
            })?;
    uniskip.set_field_inline_inputs(field_inline.product, field_inline.inv_product)
}

/// Extract the FR product appendage from the stage-2 claims (fail-closed),
/// supply it to the composed product remainder, and enforce the spec's alias
/// table against the FR claim-reduction member outputs.
///
/// WHY the explicit equality: the spec's alias table (field-inline-protocol.md,
/// "Stage 2 Composition") aliases the FR claim-reduction outputs into the FR
/// product-remainder rows. The generated alias machinery resolves canonical
/// sources across batch members' typed claims only, and the FR product
/// openings ride the proof as an appendage — so the alias is enforced as this
/// explicit equality instead. It is a same-polynomial-at-the-same-point
/// statement for the same structural reason as the jolt aliases: both
/// relations bind the same batch-point suffix and derive the same reversed
/// opening point (pinned by
/// `field_registers_claim_reduction_shares_the_product_remainder_point`).
pub fn attach_product_outputs<F: JoltField>(
    sumchecks: &Stage2BatchSumchecks<F>,
    claims: &Stage2OutputClaims<F>,
) -> Result<FieldRegistersProductOutputClaims<F>, VerifierError> {
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
    validate_product_aliases(&claims.batch_outputs, &field_inline_product)?;
    Ok(field_inline_product)
}

/// The spec's stage-2 alias table (`field-inline-protocol.md`, "Stage 2
/// Composition") as its polynomial list: each FR claim-reduction output aliases
/// the FR product-remainder opening of the same polynomial. Shared by the clear
/// equality check below and the BlindFold lowering's `OpeningEquality` rows, so
/// the two enforcement paths cannot drift.
pub(crate) fn product_alias_polynomials() -> [FieldInlineVirtualPolynomial; 3] {
    [
        FieldInlineVirtualPolynomial::FieldRs1Value,
        FieldInlineVirtualPolynomial::FieldRs2Value,
        FieldInlineVirtualPolynomial::FieldRdValue,
    ]
}

/// Enforce the spec's stage-2 alias table: each FR claim-reduction output
/// equals the FR product-remainder opening of the same polynomial (see the WHY
/// on [`attach_product_outputs`]). Value-only, like the generated
/// `validate_aliases`.
fn validate_product_aliases<F: JoltField>(
    batch_outputs: &Stage2BatchOutputClaims<F>,
    field_inline_product: &FieldRegistersProductOutputClaims<F>,
) -> Result<(), VerifierError> {
    let reduction = &batch_outputs.field_registers_claim_reduction;
    for polynomial in product_alias_polynomials() {
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

/// The composed stage-2 committed row count: the member openings plus the
/// three FR product-appendage rows spliced after the product-remainder outputs
/// — the clear absorb order exactly.
pub fn composed_output_claim_count(base: usize) -> Result<usize, VerifierError> {
    base.checked_add(
        jolt_claims::protocols::field_inline::geometry::product::selected_product_remainder_output_openings()
            .len(),
    )
    .ok_or_else(|| VerifierError::StageClaimSumcheckFailed {
        stage: format!("{:?}", JoltRelationId::SpartanProductVirtualization),
        reason: "composed stage-2 output-claim count overflows usize".to_string(),
    })
}

/// The curated field-inline absorb: the spec's committed output-claim row order
/// (`specs/field-inline-protocol.md`, "Stage 2 Composition") appends the three
/// FR product-appendage openings after the ordinary product-remainder outputs
/// and before the instruction claim-reduction non-aliased outputs; the FR
/// claim-reduction member's three openings absorb at its member position (they
/// are equality-checked against the appendage by [`attach_product_outputs`]
/// rather than alias-elided, so they stay Fiat-Shamir-bound like any member
/// output).
impl<F: JoltField> Stage2BatchSumchecks<F> {
    /// Absorbed opening scalars in the curated order above. This is the
    /// Fiat-Shamir order and MUST match the prover's commitment order.
    pub fn opening_values(
        &self,
        claims: &Stage2BatchOutputClaims<F>,
        field_inline_product: &FieldRegistersProductOutputClaims<F>,
    ) -> Vec<F> {
        let mut values =
            absorbed_opening_values::<F, RamReadWriteChecking<F>>(&claims.ram_read_write);
        values.extend(absorbed_opening_values::<F, ProductRemainder<F>>(
            &claims.product_remainder,
        ));
        values.extend(field_inline_product.opening_values());
        values.extend(absorbed_opening_values::<F, InstructionClaimReduction<F>>(
            &claims.instruction_claim_reduction,
        ));
        values.extend(
            absorbed_opening_values::<F, FieldRegistersClaimReduction<F>>(
                &claims.field_registers_claim_reduction,
            ),
        );
        values.extend(absorbed_opening_values::<F, RamRafEvaluation<F>>(
            &claims.ram_raf_evaluation,
        ));
        values.extend(absorbed_opening_values::<F, RamOutputCheck<F>>(
            &claims.ram_output_check,
        ));
        values
    }

    /// Append every absorbed opening to the transcript in the curated order,
    /// each under the `b"opening_claim"` label, matching the prover's
    /// commitment order.
    pub fn append_output_claims<T: Transcript<Challenge = F>>(
        &self,
        transcript: &mut T,
        claims: &Stage2BatchOutputClaims<F>,
        field_inline_product: &FieldRegistersProductOutputClaims<F>,
    ) {
        for value in self.opening_values(claims, field_inline_product) {
            transcript.append_labeled(b"opening_claim", &value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::outputs::{
        FieldRegistersClaimReductionOutputClaims, InstructionClaimReductionOutputClaims,
        ProductRemainderOutputClaims, RamOutputCheckOutputClaims, RamRafEvaluationOutputClaims,
        RamReadWriteOutputClaims,
    };
    use super::*;
    use jolt_field::{Fr, Ring};

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
    fn product_aliases_accept_matching_values() {
        let batch = batch_outputs_with_reduction(23, 21, 22);
        assert!(validate_product_aliases(&batch, &appendage()).is_ok());
    }

    #[test]
    fn product_aliases_reject_rs1_mismatch() {
        let batch = batch_outputs_with_reduction(23, 99, 22);
        assert!(validate_product_aliases(&batch, &appendage()).is_err());
    }

    #[test]
    fn product_aliases_reject_rs2_mismatch() {
        let batch = batch_outputs_with_reduction(23, 21, 99);
        assert!(validate_product_aliases(&batch, &appendage()).is_err());
    }

    #[test]
    fn product_aliases_reject_rd_mismatch() {
        let batch = batch_outputs_with_reduction(99, 21, 22);
        assert!(validate_product_aliases(&batch, &appendage()).is_err());
    }
}
