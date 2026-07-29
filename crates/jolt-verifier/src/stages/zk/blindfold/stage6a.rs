use super::*;

pub(super) fn add_stage6a<PCS, VC, ZkProof>(
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
    let trusted_layout = input.checked.precommitted.advice(JoltAdviceKind::Trusted);
    let untrusted_layout = input.checked.precommitted.advice(JoltAdviceKind::Untrusted);

    // The cycle bytecode round count is needed by the shared publics helper; the
    // committed and uncommitted cycle-phase relations are distinct types, so pick
    // the active one's rounds here.
    let bytecode_rounds = if bytecode_reduction_layout.is_some() {
        relations::bytecode::ReadRafCyclePhaseCommitted::new((
            formula_dimensions.bytecode_read_raf,
            bytecode_reduction::NUM_BYTECODE_VAL_STAGES,
        ))
        .rounds()
    } else {
        relations::bytecode::ReadRafCyclePhase::new((
            formula_dimensions.bytecode_read_raf,
            bytecode_reduction::NUM_BYTECODE_VAL_STAGES,
        ))
        .rounds()
    };

    add_stage6_publics_and_challenges(
        input,
        values,
        bytecode_address_claims.rounds(),
        bytecode_rounds,
        booleanity_address_claims.rounds(),
        booleanity_claims.rounds(),
        ram_hamming_claims.rounds(),
        ram_ra_claims.rounds(),
        instruction_ra_claims.rounds(),
        inc_claims.rounds(),
    )?;
    if let Some(layout) = trusted_layout {
        add_advice_cycle_publics(input, values, layout, JoltAdviceKind::Trusted)?;
    }
    if let Some(layout) = untrusted_layout {
        add_advice_cycle_publics(input, values, layout, JoltAdviceKind::Untrusted)?;
    }
    if let Some(layout) = bytecode_reduction_layout.as_ref() {
        let eta = input
            .stage6b
            .challenges
            .bytecode_reduction_eta
            .ok_or_else(|| VerifierError::MissingStageClaimChallenge {
                id: JoltChallengeId::from(BytecodeClaimReductionChallenge::Eta),
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
    add_batched_stage(
        builder,
        "stage6.address_phase",
        bytecode_address_claims.domain(),
        &[
            relation_claim(&bytecode_address_claims),
            relation_claim(&booleanity_address_claims),
        ],
        &input.stage6a.consistency,
        &input.stage6a.output_claims,
        values,
        address_phase_output_ids,
        Vec::new(),
    )
}
