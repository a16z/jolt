use super::*;

pub(super) fn add_stage7<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    builder: Builder<PCS::Field, VC::Output>,
    values: &mut SourceValues<PCS::Field>,
) -> Result<Builder<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: Clone,
{
    let formula_dimensions = formula_dimensions(input)?;
    let hamming_dimensions = hamming_weight::HammingWeightClaimReductionDimensions::new(
        formula_dimensions.ra_layout,
        input.proof.one_hot_config.committed_chunk_bits(),
    );
    let hamming_claims = hamming_weight::claim_reduction::<PCS::Field>(hamming_dimensions);
    let (trusted_layout, trusted_claims) = advice_address_claim(input, JoltAdviceKind::Trusted);
    let (untrusted_layout, untrusted_claims) =
        advice_address_claim(input, JoltAdviceKind::Untrusted);
    let bytecode_reduction_layout = input.checked.precommitted.bytecode.clone();
    let program_image_reduction_layout = input.checked.precommitted.program_image.clone();
    let bytecode_reduction_claims = bytecode_reduction_layout.as_ref().and_then(|layout| {
        layout.dimensions().has_address_phase().then(|| {
            bytecode_reduction::address_phase::<PCS::Field>(
                layout.dimensions(),
                layout.chunk_count(),
            )
        })
    });
    let program_image_reduction_claims =
        program_image_reduction_layout.as_ref().and_then(|layout| {
            layout
                .dimensions()
                .has_address_phase()
                .then(|| program_image::address_phase::<PCS::Field>(layout.dimensions()))
        });

    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(
            HammingWeightClaimReductionChallenge::Gamma,
        )),
        input.stage7.public.hamming_gamma,
    )?;
    let hamming_point = input
        .stage7
        .batch_consistency
        .try_instance_point(hamming_claims.sumcheck.rounds)
        .map_err(|error| {
            stage_sumcheck_error(JoltRelationId::HammingWeightClaimReduction, error)
        })?;
    let rho_rev = hamming_point.iter().rev().copied().collect::<Vec<_>>();
    values.public(
        JoltPublicId::from(HammingWeightClaimReductionPublic::EqBooleanity),
        try_eq_mle(&rho_rev, &input.stage6.booleanity.r_address)
            .map_err(|error| public_error(JoltRelationId::HammingWeightClaimReduction, error))?,
    )?;
    let virtualization_points = stage6_virtualization_points(input, hamming_dimensions)?;
    for (index, point) in virtualization_points.iter().enumerate() {
        values.public(
            JoltPublicId::from(HammingWeightClaimReductionPublic::EqVirtualization(index)),
            try_eq_mle(&rho_rev, point).map_err(|error| {
                public_error(JoltRelationId::HammingWeightClaimReduction, error)
            })?,
        )?;
    }
    if let (Some(layout), Some(_claim), Some(public)) = (
        trusted_layout.as_ref(),
        trusted_claims.as_ref(),
        input.stage7.trusted_advice_address_phase.as_ref(),
    ) {
        add_advice_address_publics(input, values, layout, JoltAdviceKind::Trusted, public)?;
    }
    if let (Some(layout), Some(_claim), Some(public)) = (
        untrusted_layout.as_ref(),
        untrusted_claims.as_ref(),
        input.stage7.untrusted_advice_address_phase.as_ref(),
    ) {
        add_advice_address_publics(input, values, layout, JoltAdviceKind::Untrusted, public)?;
    }
    if let (Some(layout), Some(_claim), Some(public)) = (
        bytecode_reduction_layout.as_ref(),
        bytecode_reduction_claims.as_ref(),
        input.stage7.bytecode_address_phase.as_ref(),
    ) {
        add_bytecode_reduction_address_publics(input, values, layout, public)?;
    }
    if let (Some(layout), Some(_claim), Some(public)) = (
        program_image_reduction_layout.as_ref(),
        program_image_reduction_claims.as_ref(),
        input.stage7.program_image_address_phase.as_ref(),
    ) {
        add_program_image_reduction_address_publics(input, values, layout, public)?;
    }

    let mut claims = vec![hamming_claims];
    if let Some(claim) = trusted_claims {
        claims.push(claim);
    }
    if let Some(claim) = untrusted_claims {
        claims.push(claim);
    }
    if let Some(claim) = bytecode_reduction_claims {
        claims.push(claim);
    }
    if let Some(claim) = program_image_reduction_claims {
        claims.push(claim);
    }
    let output_openings = hamming_weight::claim_reduction_output_openings(hamming_dimensions);
    let mut output_ids = output_openings.all();
    if let Some(layout) = trusted_layout {
        if layout.dimensions().has_address_phase() {
            output_ids.push(advice::final_advice_opening(JoltAdviceKind::Trusted));
        }
    }
    if let Some(layout) = untrusted_layout {
        if layout.dimensions().has_address_phase() {
            output_ids.push(advice::final_advice_opening(JoltAdviceKind::Untrusted));
        }
    }
    if let Some(layout) = bytecode_reduction_layout.as_ref() {
        if layout.dimensions().has_address_phase() {
            output_ids.extend(
                (0..layout.chunk_count()).map(bytecode_reduction::final_bytecode_chunk_opening),
            );
        }
    }
    if let Some(layout) = program_image_reduction_layout.as_ref() {
        if layout.dimensions().has_address_phase() {
            output_ids.push(program_image::final_program_image_opening());
        }
    }
    add_batched_stage(
        builder,
        "stage7.batch",
        &claims,
        &input.stage7.batch_consistency,
        &input.stage7.batch_output_claims,
        values,
        output_ids,
        Vec::new(),
    )
}
