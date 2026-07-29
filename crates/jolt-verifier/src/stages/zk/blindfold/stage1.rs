use super::*;

pub(super) fn add_stage1<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    mut builder: Builder<PCS::Field, VC::Output>,
    values: &mut SourceValues<PCS::Field>,
) -> Result<Builder<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: Clone,
{
    let log_t = input.checked.trace_length.ilog2() as usize;
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
    let input_claim = scale_expr(
        opening(outer_uniskip_opening()),
        *remainder_batching_coefficient
            * PCS::Field::pow2(input.stage1.remainder_consistency.max_num_vars - remainder_rounds),
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
