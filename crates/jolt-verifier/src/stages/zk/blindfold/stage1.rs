use super::*;

// Binding the scalar field to a bare `F` parameter (rather than spelling
// `PCS::Field`) lets clippy.toml's `arithmetic-side-effects-allowed = ["F"]`
// recognize the side-effect-free field arithmetic in the body.
pub(super) fn add_stage1<F, PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    mut builder: Builder<F, VC::Output>,
    values: &mut SourceValues<F>,
) -> Result<Builder<F, VC::Output>, VerifierError>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: Clone,
{
    let log_t = crate::num::ilog2(input.checked.trace_length);
    let dimensions = SpartanOuterDimensions::rv64(log_t);
    let uniskip_rounds = 1;
    let uniskip_degree = SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE;
    let uniskip_domain = SumcheckDomain::centered_integer(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE);
    builder = add_stage(
        builder,
        "stage1.outer_uniskip",
        SumcheckStatement::new(uniskip_rounds, uniskip_degree),
        domain_spec(uniskip_domain),
        input.stage1.uniskip_consistency.clone(),
        &input.stage1.uniskip_output_claims,
        values,
        vec![outer_uniskip_opening()],
        Vec::new(),
        VerifierExpr::zero(),
        opening(outer_uniskip_opening()),
    )?;

    let opening_order = dimensions.variables().to_vec();
    // The remainder sumcheck point is opening-derived: for the singleton remainder
    // batch the committed round challenges are the raw (un-reversed) point that the
    // clear path obtains from the bound remainder reduction.
    let remainder_challenges = input.stage1.remainder_consistency.challenges();
    let remainder_formula = JoltSpartanOuterRemainder::new(JoltSpartanOuterRemainderChallenges {
        tau: &input.stage1.challenges.tau,
        uniskip: input.stage1.challenges.uniskip_challenge,
        remainder: &remainder_challenges,
    })
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::SpartanOuter,
        reason: error.to_string(),
    })?;
    for (id, value) in remainder_formula.public_coefficients() {
        values.public(VerifierPublicId::SpartanOuter(id), value)?;
    }

    #[expect(
        clippy::arithmetic_side_effects,
        reason = "log_t is an ilog2 result (< 64); 1 + log_t cannot overflow usize"
    )]
    let remainder_rounds = 1 + log_t;
    let remainder_domain = SumcheckDomain::BooleanHypercube;
    let [remainder_batching_coefficient] = input
        .stage1
        .remainder_consistency
        .batching_coefficients
        .as_slice()
    else {
        return Err(VerifierError::BlindFoldConstructionFailed {
            reason: "stage1.outer_remainder: expected one batching coefficient".to_string(),
        });
    };
    let remainder_extra_vars = input
        .stage1
        .remainder_consistency
        .max_num_vars
        .checked_sub(remainder_rounds)
        .ok_or_else(|| VerifierError::BlindFoldConstructionFailed {
            reason: format!(
                "stage1.outer_remainder: {remainder_rounds} rounds exceed the batch's {} variables",
                input.stage1.remainder_consistency.max_num_vars
            ),
        })?;
    let input_claim = scale_expr(
        opening(outer_uniskip_opening()),
        *remainder_batching_coefficient * F::pow2(remainder_extra_vars),
    );
    let output_claim = scale_expr(
        stage1_spartan_outer_output_expr(&opening_order),
        *remainder_batching_coefficient,
    );
    add_stage(
        builder,
        "stage1.outer_remainder",
        SumcheckStatement::new(
            input.stage1.remainder_consistency.max_num_vars,
            input.stage1.remainder_consistency.max_degree,
        ),
        domain_spec(remainder_domain),
        input.stage1.remainder_consistency.consistency.clone(),
        &input.stage1.remainder_output_claims,
        values,
        opening_order.iter().copied().map(outer_opening).collect(),
        Vec::new(),
        input_claim,
        output_claim,
    )
}

fn stage1_spartan_outer_output_expr<F: Field>(
    openings: &[JoltVirtualPolynomial],
) -> VerifierExpr<F> {
    // The factored quadratic form, mirroring the jolt-claims relation: each
    // derived leaf one constituent multilinear.
    let mut az = VerifierExpr::zero();
    let mut bz = VerifierExpr::zero();
    for (index, variable) in openings.iter().copied().enumerate() {
        az = az
            + derived(VerifierPublicId::SpartanOuter(
                JoltSpartanOuterPublic::AzWeight(index),
            )) * opening(outer_opening(variable));
        bz = bz
            + derived(VerifierPublicId::SpartanOuter(
                JoltSpartanOuterPublic::BzWeight(index),
            )) * opening(outer_opening(variable));
    }
    az = az
        + derived(VerifierPublicId::SpartanOuter(
            JoltSpartanOuterPublic::AzConstant,
        ));
    bz = bz
        + derived(VerifierPublicId::SpartanOuter(
            JoltSpartanOuterPublic::BzConstant,
        ));
    derived(VerifierPublicId::SpartanOuter(
        JoltSpartanOuterPublic::TauKernel,
    )) * az
        * bz
}
