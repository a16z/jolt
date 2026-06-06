use super::*;

pub(super) fn add_stage1<PCS, VC>(
    input: &BlindFoldInputs<'_, PCS, VC>,
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
    let uniskip_sumcheck = JoltSumcheckSpec::centered_integer(
        SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
        1,
        SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
    );
    builder = add_stage(
        builder,
        "stage1.outer_uniskip",
        SumcheckStatement::new(uniskip_sumcheck.rounds, uniskip_sumcheck.degree),
        domain_spec(uniskip_sumcheck),
        input.stage1.uniskip_consistency.clone(),
        &input.stage1.uniskip_output_claims,
        values,
        vec![VerifierOpeningId::Jolt(outer_uniskip_opening())],
        Vec::new(),
        VerifierExpr::zero(),
        opening(VerifierOpeningId::Jolt(outer_uniskip_opening())),
    )?;

    let opening_order = spartan_outer_opening_order(&dimensions);
    let remainder_formula = JoltSpartanOuterRemainder::new(JoltSpartanOuterRemainderChallenges {
        tau: &input.stage1.public.tau,
        uniskip: input.stage1.public.uniskip_challenge,
        remainder: &input.stage1.public.remainder_challenges,
    })
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::SpartanOuter,
        reason: error.to_string(),
    })?;
    for (id, value) in remainder_formula.public_coefficients() {
        values.public(VerifierPublicId::SpartanOuter(id), value)?;
    }

    let remainder_spec = JoltSumcheckSpec::boolean(1 + log_t, SPARTAN_OUTER_REMAINDER_DEGREE);
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
        opening(VerifierOpeningId::Jolt(outer_uniskip_opening())),
        *remainder_batching_coefficient
            * PCS::Field::pow2(
                input.stage1.remainder_consistency.max_num_vars - remainder_spec.rounds,
            ),
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
        domain_spec(remainder_spec),
        input.stage1.remainder_consistency.consistency.clone(),
        &input.stage1.remainder_output_claims,
        values,
        opening_order
            .into_iter()
            .map(stage1_spartan_outer_opening_id)
            .collect(),
        Vec::new(),
        input_claim,
        output_claim,
    )
}

fn stage1_spartan_outer_output_expr<F: Field>(
    openings: &[Stage1SpartanOuterOpening],
) -> VerifierExpr<F> {
    let mut output = VerifierExpr::zero();
    for left in 0..openings.len() {
        for right in 0..openings.len() {
            output = output
                + public(VerifierPublicId::SpartanOuter(
                    JoltSpartanOuterPublic::QuadraticCoefficient { left, right },
                )) * opening(stage1_spartan_outer_opening_id(openings[left]))
                    * opening(stage1_spartan_outer_opening_id(openings[right]));
        }
    }
    for (index, opening_id) in openings.iter().copied().enumerate() {
        output = output
            + public(VerifierPublicId::SpartanOuter(
                JoltSpartanOuterPublic::LinearCoefficient(index),
            )) * opening(stage1_spartan_outer_opening_id(opening_id));
    }
    output
        + public(VerifierPublicId::SpartanOuter(
            JoltSpartanOuterPublic::ConstantCoefficient,
        ))
}

fn stage1_spartan_outer_opening_id(opening_id: Stage1SpartanOuterOpening) -> VerifierOpeningId {
    match opening_id {
        Stage1SpartanOuterOpening::Jolt(variable) => {
            VerifierOpeningId::Jolt(outer_opening(variable))
        }
        #[cfg(feature = "field-inline")]
        Stage1SpartanOuterOpening::FieldInline(variable) => {
            VerifierOpeningId::FieldInline(field_spartan::outer_opening(variable))
        }
    }
}
