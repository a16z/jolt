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
    // The stage-7 ZK output no longer carries each address phase's sumcheck point;
    // recompute the prefix-aligned point from the committed consistency, matching
    // `try_instance_point_at(0, rounds)` in the verifier's ZK arm.
    let address_phase_point = |claim: &JoltRelationClaims<PCS::Field>, stage| {
        input
            .stage7
            .batch_consistency
            .try_instance_point_at(0, claim.sumcheck.rounds)
            .map_err(|error| stage_sumcheck_error(stage, error))
    };
    if let (Some(layout), Some(claim)) = (trusted_layout.as_ref(), trusted_claims.as_ref()) {
        let point = address_phase_point(claim, JoltRelationId::AdviceClaimReduction)?;
        add_advice_address_publics(input, values, layout, JoltAdviceKind::Trusted, &point)?;
    }
    if let (Some(layout), Some(claim)) = (untrusted_layout.as_ref(), untrusted_claims.as_ref()) {
        let point = address_phase_point(claim, JoltRelationId::AdviceClaimReduction)?;
        add_advice_address_publics(input, values, layout, JoltAdviceKind::Untrusted, &point)?;
    }
    if let (Some(layout), Some(claim)) = (
        bytecode_reduction_layout.as_ref(),
        bytecode_reduction_claims.as_ref(),
    ) {
        let point = address_phase_point(claim, JoltRelationId::BytecodeClaimReduction)?;
        add_bytecode_reduction_address_publics(input, values, layout, &point)?;
    }
    if let (Some(layout), Some(claim)) = (
        program_image_reduction_layout.as_ref(),
        program_image_reduction_claims.as_ref(),
    ) {
        let point = address_phase_point(claim, JoltRelationId::ProgramImageClaimReduction)?;
        add_program_image_reduction_address_publics(input, values, layout, &point)?;
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
