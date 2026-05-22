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
    let bytecode_claims = bytecode::read_raf::<PCS::Field>(formula_dimensions.bytecode_read_raf);
    let booleanity_dimensions = BooleanityDimensions::new(
        formula_dimensions.ra_layout,
        log_t,
        input.proof.one_hot_config.committed_chunk_bits(),
    );
    let booleanity_claims = booleanity::booleanity::<PCS::Field>(booleanity_dimensions);
    let ram_hamming_claims = ram::hamming_booleanity::<PCS::Field>(trace_dimensions);
    let ram_ra_claims =
        ram::ra_virtualization::<PCS::Field>(formula_dimensions.ram_ra_virtualization);
    let instruction_ra_claims = instruction::ra_virtualization::<PCS::Field>(
        formula_dimensions.instruction_ra_virtualization,
    );
    let inc_claims = increments::claim_reduction::<PCS::Field>(trace_dimensions);
    let (trusted_layout, trusted_claims) = advice_cycle_claim(input, JoltAdviceKind::Trusted);
    let (untrusted_layout, untrusted_claims) = advice_cycle_claim(input, JoltAdviceKind::Untrusted);

    add_stage6_publics_and_challenges(
        input,
        values,
        &bytecode_claims,
        &booleanity_claims,
        &ram_hamming_claims,
        &ram_ra_claims,
        &instruction_ra_claims,
        &inc_claims,
    )?;
    if let (Some(layout), Some(_claim), Some(public)) = (
        trusted_layout.as_ref(),
        trusted_claims.as_ref(),
        input.stage6.trusted_advice_cycle_phase.as_ref(),
    ) {
        add_advice_cycle_publics(input, values, layout, JoltAdviceKind::Trusted, public)?;
    }
    if let (Some(layout), Some(_claim), Some(public)) = (
        untrusted_layout.as_ref(),
        untrusted_claims.as_ref(),
        input.stage6.untrusted_advice_cycle_phase.as_ref(),
    ) {
        add_advice_cycle_publics(input, values, layout, JoltAdviceKind::Untrusted, public)?;
    }

    let mut claims = vec![
        bytecode_claims,
        booleanity_claims,
        ram_hamming_claims,
        ram_ra_claims,
        instruction_ra_claims,
        inc_claims,
    ];
    if let Some(claim) = trusted_claims {
        claims.push(claim);
    }
    if let Some(claim) = untrusted_claims {
        claims.push(claim);
    }

    let mut output_ids = Vec::new();
    output_ids.extend(
        bytecode::read_raf_output_openings(formula_dimensions.bytecode_read_raf).bytecode_ra,
    );
    output_ids.extend(booleanity::booleanity_output_openings(
        formula_dimensions.ra_layout,
    ));
    output_ids.extend(ram::hamming_booleanity_output_openings());
    output_ids.extend(ram::ra_virtualization_output_openings(
        formula_dimensions.ram_ra_virtualization,
    ));
    output_ids.extend(
        instruction::ra_virtualization_output_openings(
            formula_dimensions.instruction_ra_virtualization,
        )
        .all(),
    );
    output_ids.extend(increments::claim_reduction_output_openings());
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
    add_batched_stage(
        builder,
        "stage6.batch",
        &claims,
        &input.stage6.batch_consistency,
        &input.stage6.batch_output_claims,
        values,
        output_ids,
        Vec::new(),
    )
}
