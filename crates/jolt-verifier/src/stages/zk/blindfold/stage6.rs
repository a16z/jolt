use super::*;

pub(super) fn add_stage6<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    builder: Builder<PCS::Field, VC::Output>,
    values: &mut SourceValues<PCS::Field>,
) -> Result<Builder<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: Clone,
{
    let log_t = input.checked.trace_length.ilog2() as usize;
    let trace_dimensions = jolt_claims::protocols::jolt::TraceDimensions::new(log_t);
    let formula_dimensions = formula_dimensions(input)?;
    let bytecode_reduction_layout = input.checked.precommitted.bytecode.clone();
    let program_image_reduction_layout = input.checked.precommitted.program_image.clone();
    let bytecode_address_claims =
        relations::bytecode::ReadRafAddressPhase::new(formula_dimensions.bytecode_read_raf);
    let booleanity_dimensions = BooleanityDimensions::new(
        formula_dimensions.ra_layout,
        log_t,
        input.proof.one_hot_config.committed_chunk_bits(),
    );
    let booleanity_address_claims =
        relations::booleanity::BooleanityAddressPhase::new(booleanity_dimensions);
    let booleanity_claims = relations::booleanity::BooleanityCyclePhase::new(booleanity_dimensions);
    let ram_hamming_claims = relations::ram::HammingBooleanity::new(trace_dimensions);
    let ram_ra_claims =
        relations::ram::RaVirtualization::new(formula_dimensions.ram_ra_virtualization);
    let instruction_ra_claims = relations::instruction::RaVirtualization::new(
        formula_dimensions.instruction_ra_virtualization,
    );
    let inc_claims = relations::claim_reductions::increments::ClaimReduction::new(trace_dimensions);
    let (trusted_layout, trusted_claims) = advice_cycle_claim(input, JoltAdviceKind::Trusted);
    let (untrusted_layout, untrusted_claims) = advice_cycle_claim(input, JoltAdviceKind::Untrusted);
    let bytecode_reduction_claims = bytecode_reduction_layout.as_ref().map(|layout| {
        relations::claim_reductions::bytecode::CyclePhase::new((
            layout.dimensions(),
            layout.chunk_count(),
        ))
    });
    let program_image_reduction_claims = program_image_reduction_layout.as_ref().map(|layout| {
        relations::claim_reductions::program_image::CyclePhase::new(layout.dimensions())
    });

    // The committed and uncommitted cycle-phase relations are distinct types, so
    // collapse the active one into its spec and input/output expressions here.
    let (bytecode_spec, bytecode_input, bytecode_output) = if bytecode_reduction_layout.is_some() {
        let claims = relations::bytecode::ReadRafCyclePhaseCommitted::new(
            formula_dimensions.bytecode_read_raf,
        );
        (
            claims.spec(),
            claims.input_expression::<PCS::Field>(),
            claims.output_expression::<PCS::Field>(),
        )
    } else {
        let claims =
            relations::bytecode::ReadRafCyclePhase::new(formula_dimensions.bytecode_read_raf);
        (
            claims.spec(),
            claims.input_expression::<PCS::Field>(),
            claims.output_expression::<PCS::Field>(),
        )
    };

    add_stage6_publics_and_challenges(
        input,
        values,
        bytecode_address_claims.spec(),
        bytecode_spec,
        booleanity_address_claims.spec(),
        booleanity_claims.spec(),
        ram_hamming_claims.spec(),
        ram_ra_claims.spec(),
        instruction_ra_claims.spec(),
        inc_claims.spec(),
    )?;
    if let (Some(layout), Some(_claim)) = (trusted_layout.as_ref(), trusted_claims.as_ref()) {
        add_advice_cycle_publics(input, values, layout, JoltAdviceKind::Trusted)?;
    }
    if let (Some(layout), Some(_claim)) = (untrusted_layout.as_ref(), untrusted_claims.as_ref()) {
        add_advice_cycle_publics(input, values, layout, JoltAdviceKind::Untrusted)?;
    }
    if let Some(layout) = bytecode_reduction_layout.as_ref() {
        let eta = input.stage6.public.bytecode_reduction_eta.ok_or_else(|| {
            VerifierError::MissingStageClaimChallenge {
                id: JoltChallengeId::from(BytecodeClaimReductionChallenge::Eta),
            }
        })?;
        values.public(
            VerifierPublicId::Challenge(JoltChallengeId::from(
                BytecodeClaimReductionChallenge::Eta,
            )),
            eta,
        )?;
        add_bytecode_reduction_cycle_publics(input, values, layout)?;
    }
    if let Some(layout) = program_image_reduction_layout.as_ref() {
        add_program_image_reduction_cycle_publics(input, values, layout)?;
    }

    let mut address_phase_output_ids = vec![bytecode::bytecode_read_raf_address_phase_opening()];
    if bytecode_reduction_layout.is_some() {
        address_phase_output_ids.extend(
            (0..bytecode_reduction::NUM_BYTECODE_VAL_STAGES)
                .map(bytecode_reduction::bytecode_val_stage_opening),
        );
    }
    address_phase_output_ids.push(booleanity::booleanity_address_phase_opening());
    let builder = add_batched_stage(
        builder,
        "stage6.address_phase",
        &[
            bytecode_address_claims.spec(),
            booleanity_address_claims.spec(),
        ],
        &[
            bytecode_address_claims.input_expression::<PCS::Field>(),
            booleanity_address_claims.input_expression::<PCS::Field>(),
        ],
        &[
            bytecode_address_claims.output_expression::<PCS::Field>(),
            booleanity_address_claims.output_expression::<PCS::Field>(),
        ],
        &input.stage6.address_phase_consistency,
        &input.stage6.address_phase_output_claims,
        values,
        address_phase_output_ids,
        Vec::new(),
    )?;

    let bytecode_input_expr = map_jolt_expr(bytecode_input);

    let mut batch_claims = vec![
        (
            bytecode_spec.rounds,
            bytecode_input_expr,
            map_jolt_expr(bytecode_output),
        ),
        (
            booleanity_claims.spec().rounds,
            map_jolt_expr(booleanity_claims.input_expression::<PCS::Field>()),
            map_jolt_expr(booleanity_claims.output_expression::<PCS::Field>()),
        ),
        (
            ram_hamming_claims.spec().rounds,
            map_jolt_expr(ram_hamming_claims.input_expression::<PCS::Field>()),
            map_jolt_expr(ram_hamming_claims.output_expression::<PCS::Field>()),
        ),
        (
            ram_ra_claims.spec().rounds,
            map_jolt_expr(ram_ra_claims.input_expression::<PCS::Field>()),
            map_jolt_expr(ram_ra_claims.output_expression::<PCS::Field>()),
        ),
        (
            instruction_ra_claims.spec().rounds,
            map_jolt_expr(instruction_ra_claims.input_expression::<PCS::Field>()),
            map_jolt_expr(instruction_ra_claims.output_expression::<PCS::Field>()),
        ),
        (
            inc_claims.spec().rounds,
            map_jolt_expr(inc_claims.input_expression::<PCS::Field>()),
            map_jolt_expr(inc_claims.output_expression::<PCS::Field>()),
        ),
    ];
    if let Some(claim) = trusted_claims {
        batch_claims.push((
            claim.spec().rounds,
            map_jolt_expr(claim.input_expression::<PCS::Field>()),
            map_jolt_expr(claim.output_expression::<PCS::Field>()),
        ));
    }
    if let Some(claim) = untrusted_claims {
        batch_claims.push((
            claim.spec().rounds,
            map_jolt_expr(claim.input_expression::<PCS::Field>()),
            map_jolt_expr(claim.output_expression::<PCS::Field>()),
        ));
    }
    if let Some(claim) = &bytecode_reduction_claims {
        batch_claims.push((
            claim.spec().rounds,
            map_jolt_expr(claim.input_expression::<PCS::Field>()),
            map_jolt_expr(claim.output_expression::<PCS::Field>()),
        ));
    }
    if let Some(claim) = &program_image_reduction_claims {
        batch_claims.push((
            claim.spec().rounds,
            map_jolt_expr(claim.input_expression::<PCS::Field>()),
            map_jolt_expr(claim.output_expression::<PCS::Field>()),
        ));
    }

    let booleanity_opening_point = input
        .stage6
        .output_points
        .booleanity_opening_point()
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::Booleanity,
            reason: "Stage 6 booleanity produced no opening point".to_string(),
        })?;
    let (mut output_ids, aliases) = stage6_cycle_output_openings_and_aliases(
        formula_dimensions,
        &input.stage6.output_points.bytecode_read_raf.bytecode_ra,
        booleanity_opening_point,
    );
    output_ids.extend(map_jolt_opening_ids(
        ram::hamming_booleanity_output_openings().to_vec(),
    ));
    output_ids.extend(map_jolt_opening_ids(
        ram::ra_virtualization_output_openings(formula_dimensions.ram_ra_virtualization),
    ));
    output_ids.extend(map_jolt_opening_ids(
        instruction::ra_virtualization_output_openings(
            formula_dimensions.instruction_ra_virtualization,
        )
        .all(),
    ));
    output_ids.extend(map_jolt_opening_ids(
        increments::claim_reduction_output_openings().to_vec(),
    ));
    if let Some(layout) = trusted_layout {
        output_ids.extend(map_jolt_opening_ids(advice::cycle_phase_output_openings(
            JoltAdviceKind::Trusted,
            layout.dimensions(),
        )));
    }
    if let Some(layout) = untrusted_layout {
        output_ids.extend(map_jolt_opening_ids(advice::cycle_phase_output_openings(
            JoltAdviceKind::Untrusted,
            layout.dimensions(),
        )));
    }
    if let Some(layout) = bytecode_reduction_layout.as_ref() {
        output_ids.extend(map_jolt_opening_ids(
            bytecode_reduction::cycle_phase_output_openings(
                layout.dimensions(),
                layout.chunk_count(),
            ),
        ));
    }
    if let Some(layout) = program_image_reduction_layout.as_ref() {
        output_ids.extend(map_jolt_opening_ids(
            program_image::cycle_phase_output_openings(layout.dimensions()),
        ));
    }

    let coefficients = &input.stage6.batch_consistency.batching_coefficients;
    if batch_claims.len() != coefficients.len() {
        return Err(VerifierError::BlindFoldConstructionFailed {
            reason: format!(
                "stage6.batch: expected {} batching coefficients, got {}",
                batch_claims.len(),
                coefficients.len()
            ),
        });
    }
    let input_claim = batch_claims.iter().zip(coefficients).fold(
        VerifierExpr::zero(),
        |acc, ((rounds, input_expr, _), coefficient)| {
            let scale = *coefficient
                * PCS::Field::pow2(input.stage6.batch_consistency.max_num_vars - *rounds);
            acc + scale_expr(input_expr.clone(), scale)
        },
    );
    let output_claim = batch_claims.iter().zip(coefficients).fold(
        VerifierExpr::zero(),
        |acc, ((_, _, output_expr), coefficient)| {
            acc + scale_expr(output_expr.clone(), *coefficient)
        },
    );

    add_stage(
        builder,
        "stage6.cycle_phase",
        SumcheckStatement::new(
            input.stage6.batch_consistency.max_num_vars,
            input.stage6.batch_consistency.max_degree,
        ),
        domain_spec(bytecode_spec),
        input.stage6.batch_consistency.consistency.clone(),
        &input.stage6.batch_output_claims,
        values,
        output_ids,
        aliases,
        input_claim,
        output_claim,
    )
}

fn stage6_cycle_output_openings_and_aliases<F: Field>(
    formula_dimensions: JoltFormulaDimensions,
    bytecode_ra_opening_points: &[Vec<F>],
    booleanity_opening_point: &[F],
) -> (Vec<VerifierOpeningId>, Vec<OpeningAlias<VerifierOpeningId>>) {
    let bytecode_output_openings =
        bytecode::read_raf_output_openings(formula_dimensions.bytecode_read_raf);
    let booleanity_output_openings =
        booleanity::booleanity_output_openings(formula_dimensions.ra_layout);

    let mut output_ids = map_jolt_opening_ids(bytecode_output_openings.bytecode_ra.clone());
    let mut aliases = Vec::new();
    for id in booleanity_output_openings {
        let source = match id {
            JoltOpeningId::Polynomial {
                polynomial: JoltPolynomialId::Committed(JoltCommittedPolynomial::BytecodeRa(index)),
                relation: JoltRelationId::Booleanity,
            } if bytecode_ra_opening_points
                .get(index)
                .is_some_and(|point| point.as_slice() == booleanity_opening_point) =>
            {
                bytecode_output_openings.bytecode_ra.get(index).copied()
            }
            _ => None,
        };
        if let Some(source) = source {
            aliases.push(OpeningAlias::new(id.into(), source.into()));
        } else {
            output_ids.push(id.into());
        }
    }

    (output_ids, aliases)
}
