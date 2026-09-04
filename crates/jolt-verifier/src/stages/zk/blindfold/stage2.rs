use super::*;

use jolt_claims::protocols::jolt::relations::claim_reductions::instruction::InstructionClaimReductionOutputClaims;
use jolt_claims::protocols::jolt::relations::ram::{
    RamOutputCheckOutputClaims, RamRafEvaluationOutputClaims, RamReadWriteOutputClaims,
};
use jolt_claims::protocols::jolt::relations::spartan::ProductRemainderOutputClaims;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage2::outputs::InstructionClaimReduction;

// Binding the scalar field to a bare `F` parameter (rather than spelling
// `PCS::Field`) lets clippy.toml's `arithmetic-side-effects-allowed = ["F"]`
// recognize the side-effect-free field arithmetic in the body.
pub(super) fn add_stage2<F, PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    mut builder: Builder<F, VC::Output>,
    values: &mut SourceValues<F>,
) -> Result<Builder<F, VC::Output>, VerifierError>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: Clone,
{
    let log_t = crate::num::ilog2(input.checked.trace_length);
    let log_k = crate::num::ilog2(input.checked.ram_K);
    let trace_dimensions = jolt_claims::protocols::jolt::TraceDimensions::new(log_t);
    let read_write_dimensions = input.proof.rw_config.ram_dimensions(log_t, log_k);
    let product_dimensions = SpartanProductDimensions::new(log_t);
    let raf_dimensions =
        ram::RamRafEvaluationDimensions::try_from(read_write_dimensions).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRafEvaluation,
                reason: error.to_string(),
            }
        })?;

    // `product_tau_low` is stage 1's remainder cycle point (low half), computed once
    // by the stage-2 verifier and carried on the ZK output.
    let product_tau_low = input.stage2.product_tau_low.clone();

    let product_uniskip_rounds = 1;
    let product_uniskip_degree = SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE;
    let product_uniskip_domain =
        SumcheckDomain::centered_integer(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE);
    let product_weights = centered_lagrange_evals(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        input.stage2.product_tau_high,
    )
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::SpartanProductVirtualization,
        reason: error.to_string(),
    })?;
    let product_uniskip_input =
        selected_product_uniskip_input_expr::<PCS::Field>(&product_weights)?;
    builder = add_stage(
        builder,
        "stage2.product_uniskip",
        SumcheckStatement::new(product_uniskip_rounds, product_uniskip_degree),
        domain_spec(product_uniskip_domain),
        input.stage2.product_uniskip_consistency.clone(),
        &input.stage2.product_uniskip_output_claims,
        values,
        vec![product_uniskip_opening().into()],
        Vec::new(),
        Vec::new(),
        product_uniskip_input,
        opening(product_uniskip_opening()),
    )?;

    let ram_read_write = relations::ram::ReadWriteChecking::new(read_write_dimensions);
    let product_remainder = relations::spartan::ProductRemainder::new(product_dimensions);
    let instruction_reduction =
        relations::claim_reductions::instruction::ClaimReduction::new(trace_dimensions);
    let ram_raf = relations::ram::RafEvaluation::new(raf_dimensions);
    let ram_output = relations::ram::OutputCheck::new(read_write_dimensions);

    let ram_read_write_point = input
        .stage2
        .batch_consistency
        .try_instance_point(ram_read_write.rounds())
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamReadWriteChecking, error))?;
    let ram_read_write_opening = read_write_dimensions
        .read_write_opening_point(&ram_read_write_point)
        .map_err(|error| public_error(JoltRelationId::RamReadWriteChecking, error))?;
    let eq_cycle = try_eq_mle(&product_tau_low, &ram_read_write_opening.r_cycle)
        .map_err(|error| public_error(JoltRelationId::RamReadWriteChecking, error))?;
    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(RamReadWriteChallenge::Gamma)),
        input.stage2.challenges.ram_read_write.gamma,
    )?;
    values.public(JoltDerivedId::from(RamReadWritePublic::EqCycle), eq_cycle)?;

    let product_point = input
        .stage2
        .batch_consistency
        .try_instance_point(product_remainder.rounds())
        .map_err(|error| {
            stage_sumcheck_error(JoltRelationId::SpartanProductVirtualization, error)
        })?;
    let product_opening_point = product_point.iter().rev().copied().collect::<Vec<_>>();
    let product_tau_high_bound = centered_lagrange_kernel(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        input.stage2.product_tau_high,
        input.stage2.product_uniskip_challenge,
    )
    .map_err(|error| public_error(JoltRelationId::SpartanProductVirtualization, error))?;
    let product_tau_low_eq = try_eq_mle(&product_tau_low, &product_opening_point)
        .map_err(|error| public_error(JoltRelationId::SpartanProductVirtualization, error))?;
    let product_lagrange_weights = centered_lagrange_evals(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        input.stage2.product_uniskip_challenge,
    )
    .map_err(|error| public_error(JoltRelationId::SpartanProductVirtualization, error))?;
    let product_tau_kernel = product_tau_high_bound * product_tau_low_eq;
    let product_remainder_output = selected_product_remainder_output_expr::<PCS::Field>(
        &product_lagrange_weights,
        product_tau_kernel,
    )?;

    let instruction_point = input
        .stage2
        .batch_consistency
        .try_instance_point(instruction_reduction.rounds())
        .map_err(|error| stage_sumcheck_error(JoltRelationId::InstructionClaimReduction, error))?;
    let instruction_opening_point = instruction_point.iter().rev().copied().collect::<Vec<_>>();
    let eq_spartan = try_eq_mle(&instruction_opening_point, &product_tau_low)
        .map_err(|error| public_error(JoltRelationId::InstructionClaimReduction, error))?;
    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(
            InstructionClaimReductionChallenge::Gamma,
        )),
        input.stage2.challenges.instruction_claim_reduction.gamma,
    )?;
    values.public(
        JoltDerivedId::from(InstructionClaimReductionPublic::EqSpartan),
        eq_spartan,
    )?;

    // The FR claim reduction member and its baked publics (relation + gamma +
    // EqSpartan), at the same source-values position as before.
    #[cfg(feature = "field-inline")]
    let field_registers_reduction = super::field_inline::stage2_claim_reduction(
        values,
        log_t,
        &input.stage2.batch_consistency,
        input
            .stage2
            .challenges
            .field_registers_claim_reduction
            .gamma,
        &product_tau_low,
    )?;

    #[expect(
        clippy::arithmetic_side_effects,
        reason = "log_t and log_k are ilog2 results (< 64); the sum cannot overflow usize"
    )]
    let active_stage2_rounds = log_t + log_k;
    let phase1_offset = input
        .stage2
        .batch_consistency
        .try_round_offset(active_stage2_rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamRafEvaluation, error))?
        .checked_add(read_write_dimensions.phase1_num_rounds())
        .ok_or_else(|| VerifierError::BlindFoldConstructionFailed {
            reason: "stage2: RAM RAF phase-1 round offset overflows usize".to_string(),
        })?;
    let ram_raf_point = input
        .stage2
        .batch_consistency
        .try_instance_point_at(phase1_offset, ram_raf.rounds())
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamRafEvaluation, error))?;
    let ram_raf_address = read_write_dimensions
        .address_opening_point(&ram_raf_point)
        .map_err(|error| public_error(JoltRelationId::RamRafEvaluation, error))?;
    let ram_raf_unmap_address = IdentityPolynomial::new(log_k).evaluate(&ram_raf_address)
        * PCS::Field::from_u64(8)
        + PCS::Field::from_u64(input.checked.public_io.memory_layout.get_lowest_address());
    values.public(
        JoltDerivedId::from(RamRafEvaluationPublic::UnmapAddress),
        ram_raf_unmap_address,
    )?;

    let ram_output_point = input
        .stage2
        .batch_consistency
        .try_instance_point_at(phase1_offset, ram_output.rounds())
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamOutputCheck, error))?;
    let ram_output_address = read_write_dimensions
        .address_opening_point(&ram_output_point)
        .map_err(|error| public_error(JoltRelationId::RamOutputCheck, error))?;
    let output_publics = ram_output_publics(
        input,
        &input.stage2.challenges.ram_output_check.output_address,
        &ram_output_address,
    )?;
    values.public(
        JoltDerivedId::from(RamOutputCheckPublic::EqAddress),
        output_publics.0,
    )?;
    values.public(
        JoltDerivedId::from(RamOutputCheckPublic::IoMask),
        output_publics.1,
    )?;
    values.public(
        JoltDerivedId::from(RamOutputCheckPublic::ValIo),
        output_publics.2,
    )?;

    let (output_ids, aliases) = stage2_output_ids_and_aliases::<PCS::Field>();

    // Member declaration order (= batching-coefficient draw order): the FR
    // claim reduction sits between the instruction reduction and RAM RAF
    // evaluation, exactly as in `Stage2BatchSumchecks`.
    let mut batch_claims = vec![
        relation_claim(&ram_read_write),
        (
            product_remainder.rounds(),
            map_expr(product_remainder.input_expression::<PCS::Field>()),
            product_remainder_output,
        ),
        relation_claim(&instruction_reduction),
    ];
    #[cfg(feature = "field-inline")]
    batch_claims.push(relation_claim(&field_registers_reduction));
    batch_claims.push(relation_claim(&ram_raf));
    batch_claims.push(relation_claim(&ram_output));

    add_batched_stage(
        builder,
        "stage2.batch",
        ram_read_write.domain(),
        &batch_claims,
        &input.stage2.batch_consistency,
        &input.stage2.batch_output_claims,
        values,
        output_ids,
        aliases,
        stage2_opening_equalities(),
    )
}

/// The stage-2 committed output row order and alias rows.
///
/// FR-off: the jolt members' canonical orders with the instruction reduction's
/// aliased ids elided (absorbed once via their product-remainder sources).
/// FR-on: the clear path's curated splice — the three FR product-appendage
/// rows after the product-remainder outputs and before the instruction
/// claim-reduction non-aliased outputs, with the FR claim-reduction member's
/// rows at its member position (equality-constrained against the appendage by
/// [`stage2_opening_equalities`], not alias-elided).
fn stage2_output_ids_and_aliases<F: JoltField>(
) -> (Vec<VerifierOpeningId>, Vec<OpeningAlias<VerifierOpeningId>>) {
    let product_order = ProductRemainderOutputClaims::<F> {
        left_instruction_input: F::zero(),
        right_instruction_input: F::zero(),
        jump_flag: F::zero(),
        write_lookup_output_to_rd: F::zero(),
        lookup_output: F::zero(),
        branch_flag: F::zero(),
        next_is_noop: F::zero(),
        virtual_instruction: F::zero(),
    }
    .canonical_order();
    let instruction_outputs = InstructionClaimReductionOutputClaims::<F> {
        lookup_output: F::zero(),
        left_lookup_operand: F::zero(),
        right_lookup_operand: F::zero(),
        left_instruction_input: F::zero(),
        right_instruction_input: F::zero(),
    }
    .canonical_order();

    // Single-sourced from the reduction's declared alias pairs
    // (`ConcreteSumcheck::aliased_output_openings`): the committed output rows
    // absorb the reduction's canonical openings minus its aliased ids, and the
    // `OpeningAlias` rows mirror the same `(aliased, source)` pairs — so
    // BlindFold's row layout cannot drift from the clear path's generated absorb
    // and `validate_aliases`.
    let alias_pairs =
        <InstructionClaimReduction<F> as ConcreteSumcheck<F>>::aliased_output_openings();
    let aliased_targets: std::collections::BTreeSet<_> =
        alias_pairs.iter().map(|(aliased, _)| *aliased).collect();

    let mut output_ids: Vec<VerifierOpeningId> = composite_ids(
        RamReadWriteOutputClaims::<F> {
            val: F::zero(),
            ra: F::zero(),
            inc: F::zero(),
        }
        .canonical_order(),
    );
    output_ids.extend(composite_ids(product_order));
    // The FR product appendage rows, spliced at the clear absorb position:
    // after the product-remainder outputs, before the instruction
    // claim-reduction non-aliased outputs.
    #[cfg(feature = "field-inline")]
    output_ids.extend(super::field_inline::stage2_product_appendage_ids());
    output_ids.extend(
        instruction_outputs
            .into_iter()
            .filter(|id| !aliased_targets.contains(id))
            .map(VerifierOpeningId::from),
    );
    // The FR claim-reduction member's rows at its member position (after the
    // instruction reduction, before RAM RAF evaluation).
    #[cfg(feature = "field-inline")]
    output_ids.extend(super::field_inline::stage2_claim_reduction_output_ids());
    output_ids.extend(composite_ids(
        RamRafEvaluationOutputClaims::<F> { ram_ra: F::zero() }.canonical_order(),
    ));
    output_ids.extend(composite_ids(
        RamOutputCheckOutputClaims::<F> {
            val_final: F::zero(),
        }
        .canonical_order(),
    ));
    (output_ids, composite_aliases(alias_pairs))
}

/// The spec's stage-2 alias table over hidden rows: each FR claim-reduction
/// output row must equal the FR product-appendage row of the same polynomial —
/// the same equality the clear path enforces via the stage-2 seam's
/// `validate_product_aliases` (`stages/stage2/field_inline.rs`),
/// single-sourced from the promoted polynomial table. Both sides are
/// committed rows, so the binding is an [`OpeningEquality`] (an
/// [`OpeningAlias`] would leave one row unconstrained).
#[cfg(feature = "field-inline")]
use super::field_inline::stage2_opening_equalities;

#[cfg(not(feature = "field-inline"))]
fn stage2_opening_equalities() -> Vec<OpeningEquality<VerifierOpeningId>> {
    Vec::new()
}

/// The composed product uni-skip input claim over the feature-aware lane
/// domain: the three ordinary Spartan-outer lanes, then (under `field-inline`)
/// the two FR lanes — `FieldProduct`/`FieldInvProduct` from the STAGE-1 FR
/// Spartan-outer rows — at the following Lagrange-weight indices, exactly the
/// clear `ProductUniskip::input_claim` composition.
fn selected_product_uniskip_input_expr<F: JoltField>(
    weights: &[F],
) -> Result<VerifierExpr<F>, VerifierError> {
    let [product_weight, should_branch_weight, should_jump_weight, rest @ ..] = weights else {
        return Err(VerifierError::BlindFoldConstructionFailed {
            reason: format!(
                "stage2.product_uniskip: expected {} weights, got {}",
                SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
                weights.len()
            ),
        });
    };
    let expr = scale_expr(opening(product_outer_opening()), *product_weight)
        + scale_expr(
            opening(product_should_branch_outer_opening()),
            *should_branch_weight,
        )
        + scale_expr(
            opening(product_should_jump_outer_opening()),
            *should_jump_weight,
        );

    #[cfg(not(feature = "field-inline"))]
    {
        if !rest.is_empty() {
            return Err(VerifierError::BlindFoldConstructionFailed {
                reason: format!(
                    "stage2.product_uniskip: expected no field weights, got {}",
                    rest.len()
                ),
            });
        }
        Ok(expr)
    }
    #[cfg(feature = "field-inline")]
    {
        Ok(expr + super::field_inline::uniskip_lane_terms(rest)?)
    }
}

/// The composed product-remainder output claim over the feature-aware lane
/// domain: `tau_kernel · (ordinary_left + fr_left) · (ordinary_right +
/// fr_right)`, where the FR factor terms reference the FR product-appendage
/// rows at their composed Lagrange weights — exactly the clear
/// `ProductRemainder::expected_output` composition.
fn selected_product_remainder_output_expr<F: JoltField>(
    weights: &[F],
    tau_kernel: F,
) -> Result<VerifierExpr<F>, VerifierError> {
    let [instruction_product_weight, should_branch_weight, should_jump_weight, rest @ ..] = weights
    else {
        return Err(VerifierError::BlindFoldConstructionFailed {
            reason: format!(
                "stage2.batch: expected {} product weights, got {}",
                SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
                weights.len()
            ),
        });
    };
    let left_base = scale_expr(
        opening(left_instruction_input_product()),
        *instruction_product_weight,
    ) + scale_expr(opening(lookup_output_product()), *should_branch_weight)
        + scale_expr(opening(jump_flag_product()), *should_jump_weight);
    let right_base = scale_expr(
        opening(right_instruction_input_product()),
        *instruction_product_weight,
    ) + scale_expr(opening(branch_flag_product()), *should_branch_weight)
        + scale_expr(VerifierExpr::one(), *should_jump_weight)
        + scale_expr(opening(next_is_noop_product()), -*should_jump_weight);

    let (left, right) = {
        #[cfg(not(feature = "field-inline"))]
        {
            if !rest.is_empty() {
                return Err(VerifierError::BlindFoldConstructionFailed {
                    reason: format!(
                        "stage2.batch: expected no field product weights, got {}",
                        rest.len()
                    ),
                });
            }
            (left_base, right_base)
        }
        #[cfg(feature = "field-inline")]
        {
            let (fr_left, fr_right) = super::field_inline::remainder_factor_terms(rest)?;
            (left_base + fr_left, right_base + fr_right)
        }
    };

    Ok(scale_expr(left * right, tau_kernel))
}

#[cfg(test)]
#[cfg_attr(feature = "field-inline", expect(clippy::unwrap_used))]
mod tests {
    use super::*;
    #[cfg(feature = "field-inline")]
    use crate::stages::stage2::outputs::{
        FieldRegistersClaimReductionOutputClaims, FieldRegistersProductOutputClaims,
    };
    use jolt_field::{Fr, Ring};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    /// The stage-2 committed row order is the clear curated absorb order,
    /// locked entry-for-entry: build sentinel-valued claim structs, run the
    /// batch's `opening_values` curation, and check every lowered id resolves
    /// to the value at its row position.
    #[test]
    fn stage2_output_ids_match_the_clear_absorb_order() {
        use crate::stages::stage2::outputs::{
            InstructionClaimReductionOutputClaims, ProductRemainderOutputClaims,
            RamOutputCheckOutputClaims, RamRafEvaluationOutputClaims, RamReadWriteOutputClaims,
            Stage2BatchOutputClaims,
        };
        use jolt_claims::OutputClaims as _;

        let claims = Stage2BatchOutputClaims::<Fr> {
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
            #[cfg(feature = "field-inline")]
            field_registers_claim_reduction: FieldRegistersClaimReductionOutputClaims {
                rd_value: fr(16),
                rs1_value: fr(17),
                rs2_value: fr(18),
            },
            ram_raf_evaluation: RamRafEvaluationOutputClaims { ram_ra: fr(14) },
            ram_output_check: RamOutputCheckOutputClaims { val_final: fr(15) },
        };

        // Resolve a lowered composite id against the claim structs (the FR
        // appendage rides beside the batch, so it has its own resolver).
        #[cfg(feature = "field-inline")]
        let appendage = FieldRegistersProductOutputClaims::<Fr> {
            rs1_value: fr(201),
            rs2_value: fr(202),
            rd_value: fr(203),
        };
        let resolve = |id: &VerifierOpeningId| -> Option<Fr> {
            match id {
                VerifierOpeningId::Jolt(id) => claims
                    .ram_read_write
                    .resolve_output(id)
                    .or_else(|| claims.product_remainder.resolve_output(id))
                    .or_else(|| claims.instruction_claim_reduction.resolve_output(id))
                    .or_else(|| claims.ram_raf_evaluation.resolve_output(id))
                    .or_else(|| claims.ram_output_check.resolve_output(id)),
                #[cfg(feature = "field-inline")]
                VerifierOpeningId::FieldInline(id) => claims
                    .field_registers_claim_reduction
                    .resolve_output(id)
                    .or_else(|| appendage.resolve_output(id)),
                #[cfg(not(feature = "field-inline"))]
                VerifierOpeningId::FieldInline(_) => None,
            }
        };

        // The clear curated order over the same sentinels.
        #[cfg(not(feature = "field-inline"))]
        let expected_values: Vec<Fr> = (1..=15).map(fr).collect();
        #[cfg(feature = "field-inline")]
        let expected_values: Vec<Fr> = (1..=11)
            .map(fr)
            .chain([fr(201), fr(202), fr(203)])
            .chain([fr(12), fr(13)])
            .chain([fr(16), fr(17), fr(18)])
            .chain([fr(14), fr(15)])
            .collect();

        let (output_ids, aliases) = stage2_output_ids_and_aliases::<Fr>();
        assert_eq!(output_ids.len(), expected_values.len());
        #[cfg(not(feature = "field-inline"))]
        assert_eq!(output_ids.len(), 15);
        #[cfg(feature = "field-inline")]
        assert_eq!(output_ids.len(), 21);
        for (id, expected) in output_ids.iter().zip(expected_values) {
            assert_eq!(
                resolve(id),
                Some(expected),
                "row {id:?} must sit at the clear absorb position of value {expected:?}",
            );
        }
        assert_eq!(aliases.len(), 3);
    }

    /// The lowered alias-equality rows are the clear path's alias table: one
    /// equality per polynomial, CR row on the left, product-appendage row on
    /// the right, and both sides are committed rows of the lowered order.
    #[cfg(feature = "field-inline")]
    #[test]
    fn stage2_opening_equalities_bind_the_cr_rows_to_the_appendage() {
        use jolt_claims::protocols::field_inline::{FieldInlineOpeningId, FieldInlineRelationId};

        let equalities = stage2_opening_equalities();
        let polynomials = crate::stages::stage2::field_inline::product_alias_polynomials();
        assert_eq!(equalities.len(), polynomials.len());

        let (output_ids, _) = stage2_output_ids_and_aliases::<Fr>();
        for (equality, polynomial) in equalities.iter().zip(polynomials) {
            assert_eq!(
                equality.left,
                VerifierOpeningId::from(FieldInlineOpeningId::virtual_polynomial(
                    polynomial,
                    FieldInlineRelationId::FieldRegistersClaimReduction,
                )),
            );
            assert_eq!(
                equality.right,
                VerifierOpeningId::from(FieldInlineOpeningId::virtual_polynomial(
                    polynomial,
                    FieldInlineRelationId::FieldRegistersProduct,
                )),
            );
            assert!(output_ids.contains(&equality.left));
            assert!(output_ids.contains(&equality.right));
        }
    }

    /// The lowered composed uni-skip input expression evaluates identically to
    /// the clear `ProductUniskip::input_claim` composition on synthetic values
    /// (FR-on: all five lanes; the FR inputs read from the stage-1 FR carrier
    /// rows).
    #[cfg(feature = "field-inline")]
    #[test]
    fn lowered_uniskip_input_matches_the_clear_composed_claim() {
        use crate::stages::relations::ConcreteSumcheck as _;
        use crate::stages::stage2::product_uniskip::{ProductUniskip, ProductUniskipInputClaims};
        use jolt_claims::protocols::field_inline::geometry::spartan::outer_opening;
        use jolt_claims::protocols::field_inline::FieldInlineVirtualPolynomial;
        use jolt_claims::NoChallenges;

        let tau_high = fr(23);
        let relation = ProductUniskip::new(SpartanProductDimensions::new(4), tau_high);
        let inputs = ProductUniskipInputClaims::<Fr> {
            product: fr(3),
            should_branch: fr(5),
            should_jump: fr(7),
        };
        let field_product = fr(11);
        let field_inv_product = fr(13);
        relation
            .set_field_inline_inputs(field_product, field_inv_product)
            .unwrap();
        let clear = relation
            .input_claim(&inputs, &NoChallenges::default())
            .unwrap();

        let weights =
            centered_lagrange_evals(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, tau_high).unwrap();
        let lowered_expr = selected_product_uniskip_input_expr::<Fr>(&weights).unwrap();
        let lowered = lowered_expr.evaluate(
            |id| match id {
                VerifierOpeningId::Jolt(id) => {
                    if *id == product_outer_opening() {
                        inputs.product
                    } else if *id == product_should_branch_outer_opening() {
                        inputs.should_branch
                    } else if *id == product_should_jump_outer_opening() {
                        inputs.should_jump
                    } else {
                        fr(0)
                    }
                }
                VerifierOpeningId::FieldInline(id) => {
                    if *id == outer_opening(FieldInlineVirtualPolynomial::FieldProduct) {
                        field_product
                    } else if *id == outer_opening(FieldInlineVirtualPolynomial::FieldInvProduct) {
                        field_inv_product
                    } else {
                        fr(0)
                    }
                }
            },
            |_| fr(0),
            |_| fr(0),
        );

        assert_eq!(lowered, clear);
    }

    /// The lowered composed remainder output expression evaluates identically
    /// to the clear `ProductRemainder::expected_output` composition on
    /// synthetic values (FR-on: `tau_kernel · (ord_left + fr_left) ·
    /// (ord_right + fr_right)` with the FR factors read from the appendage
    /// rows).
    #[cfg(feature = "field-inline")]
    #[test]
    fn lowered_remainder_output_matches_the_clear_composed_claim() {
        use crate::stages::relations::ConcreteSumcheck as _;
        use crate::stages::stage2::product_remainder::{
            ProductRemainder, ProductRemainderInputClaims, ProductRemainderOutputClaims,
        };
        use jolt_claims::{NoChallenges, OutputClaims as _};

        let log_t = 4usize;
        let uniskip_challenge = fr(37);
        let tau_high = fr(41);
        let tau_low: Vec<Fr> = (50..54).map(fr).collect();
        let relation = ProductRemainder::new(
            SpartanProductDimensions::new(log_t),
            uniskip_challenge,
            tau_high,
            tau_low.clone(),
        );
        let outputs = ProductRemainderOutputClaims::<Fr> {
            left_instruction_input: fr(2),
            right_instruction_input: fr(3),
            jump_flag: fr(5),
            write_lookup_output_to_rd: fr(7),
            lookup_output: fr(11),
            branch_flag: fr(13),
            next_is_noop: fr(17),
            virtual_instruction: fr(19),
        };
        let appendage = FieldRegistersProductOutputClaims::<Fr> {
            rs1_value: fr(23),
            rs2_value: fr(29),
            rd_value: fr(31),
        };
        relation
            .set_field_inline_outputs(appendage.clone())
            .unwrap();

        let sumcheck_point: Vec<Fr> = (60..64).map(fr).collect();
        let input_points = ProductRemainderInputClaims::<Vec<Fr>>::default();
        let output_points = relation
            .derive_opening_points(&sumcheck_point, &input_points)
            .unwrap();
        let clear = relation
            .expected_output(
                &input_points,
                &outputs,
                &output_points,
                &NoChallenges::default(),
            )
            .unwrap();

        let weights =
            centered_lagrange_evals(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, uniskip_challenge)
                .unwrap();
        let opening_point = output_points.left_instruction_input();
        let tau_kernel = centered_lagrange_kernel(
            SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
            tau_high,
            uniskip_challenge,
        )
        .unwrap()
            * try_eq_mle(&tau_low, opening_point).unwrap();
        let lowered_expr =
            selected_product_remainder_output_expr::<Fr>(&weights, tau_kernel).unwrap();
        let lowered = lowered_expr.evaluate(
            |id| match id {
                VerifierOpeningId::Jolt(id) => outputs.resolve_output(id).unwrap_or_else(|| fr(0)),
                VerifierOpeningId::FieldInline(id) => {
                    appendage.resolve_output(id).unwrap_or_else(|| fr(0))
                }
            },
            |_| fr(0),
            |_| fr(0),
        );

        assert_eq!(lowered, clear);
    }
}
