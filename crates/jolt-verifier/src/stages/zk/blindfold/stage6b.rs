use super::*;

pub(super) fn add_stage6b<PCS, VC, ZkProof>(
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
    let booleanity_dimensions = BooleanityDimensions::new(
        formula_dimensions.ra_layout,
        log_t,
        input.proof.one_hot_config.committed_chunk_bits(),
    );
    let booleanity_claims = relations::booleanity::BooleanityCyclePhase::new(booleanity_dimensions);
    let ram_hamming_claims = relations::ram::HammingBooleanity::new(trace_dimensions);
    let ram_ra_claims =
        relations::ram::RaVirtualization::new(formula_dimensions.ram_ra_virtualization);
    let instruction_ra_claims = relations::instruction::RaVirtualization::new(
        formula_dimensions.instruction_ra_virtualization,
    );
    let inc_claims = relations::claim_reductions::increments::ClaimReduction::new(trace_dimensions);
    let trusted_layout = advice_layout(input, JoltAdviceKind::Trusted);
    let trusted_claims = trusted_layout.as_ref().map(|layout| {
        relations::claim_reductions::advice::TrustedCyclePhase::new(layout.dimensions())
    });
    let untrusted_layout = advice_layout(input, JoltAdviceKind::Untrusted);
    let untrusted_claims = untrusted_layout.as_ref().map(|layout| {
        relations::claim_reductions::advice::UntrustedCyclePhase::new(layout.dimensions())
    });
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
    // collapse the active one into its domain and batch tuple here.
    let (bytecode_domain, bytecode_claim) = if bytecode_reduction_layout.is_some() {
        let claims = relations::bytecode::ReadRafCyclePhaseCommitted::new(
            formula_dimensions.bytecode_read_raf,
        );
        (claims.domain(), relation_claim(&claims))
    } else {
        let claims =
            relations::bytecode::ReadRafCyclePhase::new(formula_dimensions.bytecode_read_raf);
        (claims.domain(), relation_claim(&claims))
    };

    let mut batch_claims = vec![
        bytecode_claim,
        relation_claim(&booleanity_claims),
        relation_claim(&ram_hamming_claims),
        relation_claim(&ram_ra_claims),
        relation_claim(&instruction_ra_claims),
        relation_claim(&inc_claims),
    ];
    if let Some(claim) = trusted_claims {
        batch_claims.push(relation_claim(&claim));
    }
    if let Some(claim) = untrusted_claims {
        batch_claims.push(relation_claim(&claim));
    }
    if let Some(claim) = &bytecode_reduction_claims {
        batch_claims.push(relation_claim(claim));
    }
    if let Some(claim) = &program_image_reduction_claims {
        batch_claims.push(relation_claim(claim));
    }

    let booleanity_opening_point = input
        .stage6b
        .output_points
        .booleanity_opening_point()
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::Booleanity,
            reason: "Stage 6 booleanity produced no opening point".to_string(),
        })?;
    let (mut output_ids, aliases) = stage6_cycle_output_openings_and_aliases(
        formula_dimensions,
        &input.stage6b.output_points.bytecode_read_raf.bytecode_ra,
        booleanity_opening_point,
    );
    output_ids.extend(
        relations::ram::RamHammingBooleanityOutputClaims::<PCS::Field> {
            ram_hamming_weight: PCS::Field::zero(),
        }
        .canonical_order(),
    );
    output_ids.extend(
        relations::ram::RamRaVirtualizationOutputClaims::<PCS::Field> {
            ram_ra: vec![
                PCS::Field::zero();
                formula_dimensions
                    .ram_ra_virtualization
                    .num_committed_ra_polys()
            ],
        }
        .canonical_order(),
    );
    output_ids.extend(
        instruction::ra_virtualization_output_openings(
            formula_dimensions.instruction_ra_virtualization,
        )
        .all(),
    );
    output_ids.extend(
        relations::claim_reductions::increments::IncClaimReductionOutputClaims::<PCS::Field> {
            ram_inc: PCS::Field::zero(),
            rd_inc: PCS::Field::zero(),
        }
        .canonical_order(),
    );
    if let Some(layout) = trusted_layout {
        output_ids.extend(advice::cycle_phase_output_openings(
            JoltAdviceKind::Trusted,
            layout.dimensions(),
        ));
    }
    if let Some(layout) = untrusted_layout {
        output_ids.extend(advice::cycle_phase_output_openings(
            JoltAdviceKind::Untrusted,
            layout.dimensions(),
        ));
    }
    if let Some(layout) = bytecode_reduction_layout.as_ref() {
        output_ids.extend(bytecode_reduction::cycle_phase_output_openings(
            layout.dimensions(),
            layout.chunk_count(),
        ));
    }
    if let Some(layout) = program_image_reduction_layout.as_ref() {
        output_ids.extend(program_image::cycle_phase_output_openings(
            layout.dimensions(),
        ));
    }

    add_batched_stage(
        builder,
        "stage6.cycle_phase",
        bytecode_domain,
        &batch_claims,
        &input.stage6b.batch_consistency,
        &input.stage6b.batch_output_claims,
        values,
        output_ids,
        aliases,
    )
}

fn stage6_cycle_output_openings_and_aliases<F: Field>(
    formula_dimensions: JoltFormulaDimensions,
    bytecode_ra_opening_points: &[Vec<F>],
    booleanity_opening_point: &[F],
) -> (Vec<JoltOpeningId>, Vec<OpeningAlias<JoltOpeningId>>) {
    let bytecode_output_openings =
        bytecode::read_raf_output_openings(formula_dimensions.bytecode_read_raf);
    let booleanity_output_openings =
        booleanity::booleanity_output_openings(formula_dimensions.ra_layout);

    let mut output_ids = bytecode_output_openings.bytecode_ra.clone();
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
            aliases.push(OpeningAlias::new(id, source));
        } else {
            output_ids.push(id);
        }
    }

    (output_ids, aliases)
}
