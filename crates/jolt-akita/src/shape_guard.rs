//! Pre-deserialization validation of proof-controlled Akita payload shapes.
//!
//! Commitment lengths arrive inside the prover-controlled Jolt proof. The
//! backend proof shape is derived from Akita's trusted schedule before a
//! backend deserializer can reserve payload-sized buffers.

use akita_config::{derive_transcript_grinding_plan, effective_batched_schedule, CommitmentConfig};
use akita_pcs::AkitaError;
use akita_schedules::{ResolvedScheduleRow, TrustedScheduleCatalog};
use akita_types::{
    canonical_proof_shape, CompressionChainPlan, FoldSchedule, GroupCommitPhaseParams,
    OpeningClaimsLayout, OpeningScheduleSelection, PolynomialGroupLayout,
};
use jolt_field::Zero;
use jolt_openings::OpeningsError;

use crate::adapters::{
    deserialize_akita, invalid_batch, AkitaBackendCommitment, AkitaBackendCommitmentPayload,
    AkitaBackendFlavor, AkitaBackendProof, AkitaBackendProofShape, AkitaBatchProof,
    AkitaCommitment, AkitaConfig, AkitaField, AkitaOneHotK16Config, AkitaOneHotK256Config,
    AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256,
};

/// Deserializes the backend commitment and proof only after their declared
/// shapes have been derived from the trusted resolved schedule.
pub(crate) fn deserialize_checked_backend_payload(
    schedules: &TrustedScheduleCatalog,
    commitment: &AkitaCommitment,
    proof: &AkitaBatchProof,
    statement_len: usize,
    backend_point: &[AkitaField],
) -> Result<
    (
        OpeningScheduleSelection,
        AkitaBackendCommitment,
        AkitaBackendProof,
    ),
    OpeningsError,
> {
    let layout = OpeningClaimsLayout::new(backend_point.len(), statement_len)
        .map_err(|err| invalid_batch(format!("Akita opening layout is invalid: {err}")))?;
    let selection = proof.selection();
    match commitment.backend_flavor {
        AkitaBackendFlavor::Dense => deserialize_checked_single_payload::<AkitaConfig>(
            schedules,
            commitment,
            proof,
            selection,
            &layout,
            backend_point,
        ),
        AkitaBackendFlavor::OneHot => match commitment.one_hot_k {
            AKITA_ONE_HOT_K16 => deserialize_checked_single_payload::<AkitaOneHotK16Config>(
                schedules,
                commitment,
                proof,
                selection,
                &layout,
                backend_point,
            ),
            AKITA_ONE_HOT_K256 => deserialize_checked_single_payload::<AkitaOneHotK256Config>(
                schedules,
                commitment,
                proof,
                selection,
                &layout,
                backend_point,
            ),
            one_hot_k => Err(invalid_batch(format!(
                "unsupported Akita one-hot K={one_hot_k}"
            ))),
        },
    }
}

/// Guard and decode the ordered grouped root in public order
/// `[dense precommits.., final streamed one-hot]`.
pub(crate) fn deserialize_checked_grouped_backend_payload(
    schedules: &TrustedScheduleCatalog,
    precommitted: &[&AkitaCommitment],
    main: &AkitaCommitment,
    proof: &AkitaBatchProof,
    main_backend_point: &[AkitaField],
    one_hot_k: usize,
) -> Result<
    (
        OpeningScheduleSelection,
        Vec<AkitaBackendCommitment>,
        AkitaBackendCommitment,
        AkitaBackendProof,
    ),
    OpeningsError,
> {
    let selection = proof.selection();
    let mut group_layouts = precommitted
        .iter()
        .map(|commitment| PolynomialGroupLayout::new(commitment.num_vars, commitment.poly_count))
        .collect::<Vec<_>>();
    group_layouts.push(PolynomialGroupLayout::new(main.num_vars, main.poly_count));
    let layout = OpeningClaimsLayout::from_groups(group_layouts)
        .map_err(|err| invalid_batch(format!("Akita grouped opening layout is invalid: {err}")))?;
    match one_hot_k {
        AKITA_ONE_HOT_K256 => deserialize_checked_grouped_payload::<AkitaOneHotK256Config>(
            schedules,
            precommitted,
            main,
            proof,
            selection,
            &layout,
            main_backend_point,
        ),
        AKITA_ONE_HOT_K16 => deserialize_checked_grouped_payload::<AkitaOneHotK16Config>(
            schedules,
            precommitted,
            main,
            proof,
            selection,
            &layout,
            main_backend_point,
        ),
        _ => Err(invalid_batch("unsupported grouped one-hot configuration")),
    }
}

fn deserialize_checked_single_payload<Cfg>(
    schedules: &TrustedScheduleCatalog,
    commitment: &AkitaCommitment,
    proof: &AkitaBatchProof,
    selection: OpeningScheduleSelection,
    layout: &OpeningClaimsLayout,
    backend_point: &[AkitaField],
) -> Result<
    (
        OpeningScheduleSelection,
        AkitaBackendCommitment,
        AkitaBackendProof,
    ),
    OpeningsError,
>
where
    Cfg: CommitmentConfig<Field = AkitaField, ExtField = AkitaField>,
{
    let resolved = resolve_schedule_row::<Cfg>(schedules, selection, layout, backend_point)
        .map_err(|err| invalid_batch(format!("Akita schedule resolution failed: {err}")))?;
    validate_commitment_profile_len(commitment, &resolved.profiles().final_group)?;
    let backend_payload = deserialize_akita::<AkitaBackendCommitmentPayload>(
        &commitment.serialized_backend_bytes,
        &commitment.backend_coeff_len,
    )?;
    let backend_commitment =
        AkitaBackendCommitment::new(resolved.profiles().final_group, backend_payload);
    let backend_proof = deserialize_checked_proof::<Cfg>(&resolved, layout, proof)?;
    Ok((resolved.selection(), backend_commitment, backend_proof))
}

fn deserialize_checked_grouped_payload<Cfg>(
    schedules: &TrustedScheduleCatalog,
    precommitted: &[&AkitaCommitment],
    main: &AkitaCommitment,
    proof: &AkitaBatchProof,
    selection: OpeningScheduleSelection,
    layout: &OpeningClaimsLayout,
    main_backend_point: &[AkitaField],
) -> Result<
    (
        OpeningScheduleSelection,
        Vec<AkitaBackendCommitment>,
        AkitaBackendCommitment,
        AkitaBackendProof,
    ),
    OpeningsError,
>
where
    Cfg: CommitmentConfig<Field = AkitaField, ExtField = AkitaField>,
{
    let resolved = resolve_schedule_row::<Cfg>(schedules, selection, layout, main_backend_point)
        .map_err(|err| invalid_batch(format!("Akita grouped schedule resolution failed: {err}")))?;
    let profiles = resolved.profiles();

    let mut precommitted_backend = Vec::with_capacity(precommitted.len());
    for (commitment, profile) in precommitted.iter().zip(profiles.precommitteds.iter()) {
        validate_commitment_profile_len(commitment, profile)?;
        let payload = deserialize_akita::<AkitaBackendCommitmentPayload>(
            &commitment.serialized_backend_bytes,
            &commitment.backend_coeff_len,
        )?;
        precommitted_backend.push(AkitaBackendCommitment::new(*profile, payload));
    }
    validate_commitment_profile_len(main, &profiles.final_group)?;
    let main_payload = deserialize_akita::<AkitaBackendCommitmentPayload>(
        &main.serialized_backend_bytes,
        &main.backend_coeff_len,
    )?;
    let main_backend = AkitaBackendCommitment::new(profiles.final_group, main_payload);
    let backend_proof = deserialize_checked_proof::<Cfg>(&resolved, layout, proof)?;

    Ok((
        resolved.selection(),
        precommitted_backend,
        main_backend,
        backend_proof,
    ))
}

fn deserialize_checked_proof<Cfg>(
    resolved: &ResolvedScheduleRow,
    layout: &OpeningClaimsLayout,
    proof: &AkitaBatchProof,
) -> Result<AkitaBackendProof, OpeningsError>
where
    Cfg: CommitmentConfig<Field = AkitaField, ExtField = AkitaField>,
{
    let proof_shape = derive_proof_shape::<Cfg>(resolved.schedule(), layout)?;
    proof_shape
        .validate_decode_budget(
            proof.backend_proof.len(),
            field_elem_bytes(),
            field_elem_bytes(),
        )
        .map_err(|err| {
            invalid_batch(format!("Akita proof shape exceeds its byte budget: {err}"))
        })?;
    let backend_proof = deserialize_akita::<AkitaBackendProof>(&proof.backend_proof, &proof_shape)?;
    Ok(backend_proof)
}

fn resolve_schedule_row<Cfg>(
    schedules: &TrustedScheduleCatalog,
    selection: OpeningScheduleSelection,
    layout: &OpeningClaimsLayout,
    backend_point: &[AkitaField],
) -> Result<ResolvedScheduleRow, AkitaError>
where
    Cfg: CommitmentConfig<Field = AkitaField, ExtField = AkitaField>,
{
    akita_config::validate_trusted_schedule_catalog::<Cfg>(schedules)?;
    let resolved = schedules.resolve_selection(selection)?;
    effective_batched_schedule::<Cfg>(resolved, layout, backend_point)
}

fn validate_commitment_profile_len(
    commitment: &AkitaCommitment,
    profile: &GroupCommitPhaseParams,
) -> Result<(), OpeningsError> {
    let expected_coeff_len = expected_commitment_coeff_len_for_profile(profile)?;
    if commitment.backend_coeff_len != expected_coeff_len {
        return Err(invalid_batch(format!(
            "Akita commitment declares {} backend coefficients but its frozen profile requires {expected_coeff_len}",
            commitment.backend_coeff_len
        )));
    }
    let expected_bytes = expected_coeff_len
        .checked_mul(field_elem_bytes())
        .ok_or_else(|| invalid_batch("Akita commitment byte size overflows"))?;
    if commitment.serialized_backend_bytes.len() != expected_bytes {
        return Err(invalid_batch(format!(
            "Akita commitment has {} serialized bytes but its frozen profile requires {expected_bytes}",
            commitment.serialized_backend_bytes.len()
        )));
    }
    Ok(())
}

fn expected_commitment_coeff_len_for_profile(
    profile: &GroupCommitPhaseParams,
) -> Result<usize, OpeningsError> {
    let source_coefficients = profile
        .outer_slice_count
        .complete_source_coefficients(
            profile.outer.matrix.output_rank(),
            profile.outer.matrix.ring_dimension(),
        )
        .map_err(|err| invalid_batch(format!("Akita commitment profile is invalid: {err}")))?;
    CompressionChainPlan::for_complete_source(
        profile.outer.matrix.sis_modulus_profile(),
        source_coefficients,
    )
    .map(|plan| plan.terminal_coefficients())
    .map_err(|err| invalid_batch(format!("Akita commitment profile is invalid: {err}")))
}

fn field_elem_bytes() -> usize {
    use akita_pcs::AkitaSerialize;
    AkitaField::zero().compressed_size()
}

fn derive_proof_shape<Cfg>(
    schedule: &FoldSchedule,
    layout: &OpeningClaimsLayout,
) -> Result<AkitaBackendProofShape, OpeningsError>
where
    Cfg: CommitmentConfig<Field = AkitaField, ExtField = AkitaField>,
{
    let grinding_plan = derive_transcript_grinding_plan::<Cfg>(schedule, layout)
        .map_err(|err| invalid_batch(format!("Akita grinding plan is invalid: {err}")))?;
    canonical_proof_shape(schedule, layout, Cfg::EXT_DEGREE, &grinding_plan)
        .map_err(|err| invalid_batch(format!("Akita schedule proof shape is invalid: {err}")))
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "tests assert successful schedule resolution"
    )]

    use super::*;
    use akita_types::AkitaScheduleLookupKey;
    use jolt_field::Ring;

    use crate::AkitaScheduleArtifacts;

    fn dense_commitment(num_vars: usize, poly_count: usize) -> AkitaCommitment {
        AkitaCommitment {
            backend_flavor: AkitaBackendFlavor::Dense,
            layout_digest: [7; 32],
            num_vars,
            poly_count,
            one_hot_k: 0,
            backend_coeff_len: 0,
            serialized_backend_bytes: Vec::new(),
        }
    }

    fn point(num_vars: usize) -> Vec<AkitaField> {
        (0..num_vars as u64).map(AkitaField::from_u64).collect()
    }

    fn dense_schedules() -> TrustedScheduleCatalog {
        let artifacts =
            AkitaScheduleArtifacts::from_directory(AkitaScheduleArtifacts::packaged_directory())
                .expect("workspace schedule artifacts");
        artifacts.dense_catalog().expect("dense schedule catalog")
    }

    fn test_selection(
        schedules: &TrustedScheduleCatalog,
        num_vars: usize,
        poly_count: usize,
    ) -> OpeningScheduleSelection {
        schedules
            .resolve_key(&AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(
                num_vars, poly_count,
            )))
            .expect("test schedule")
            .selection()
    }

    fn resolved_dense(
        num_vars: usize,
        poly_count: usize,
    ) -> (
        AkitaCommitment,
        Vec<AkitaField>,
        OpeningClaimsLayout,
        ResolvedScheduleRow,
        TrustedScheduleCatalog,
    ) {
        let schedules = dense_schedules();
        let point = point(num_vars);
        let layout = OpeningClaimsLayout::new(num_vars, poly_count).expect("layout");
        let commitment = dense_commitment(num_vars, poly_count);
        let resolved = resolve_schedule_row::<AkitaConfig>(
            &schedules,
            test_selection(&schedules, num_vars, poly_count),
            &layout,
            &point,
        )
        .expect("schedule");
        (commitment, point, layout, resolved, schedules)
    }

    #[test]
    fn forged_commitment_coeff_len_rejects_before_deserialization() {
        let (mut commitment, point, _, resolved, schedules) = resolved_dense(16, 2);
        commitment.backend_coeff_len = 1 << 25;
        let proof = AkitaBatchProof::new(resolved.selection(), Vec::new());
        let err = deserialize_checked_backend_payload(&schedules, &commitment, &proof, 2, &point)
            .expect_err("forged coefficient count must be rejected");
        assert_ne!(
            commitment.backend_coeff_len,
            expected_commitment_coeff_len_for_profile(&resolved.profiles().final_group)
                .expect("expected coefficients")
        );
        assert!(err.to_string().contains("coefficients"));
    }

    #[test]
    fn commitment_byte_length_must_match_coeff_len() {
        let (mut commitment, point, _, resolved, schedules) = resolved_dense(16, 2);
        commitment.backend_coeff_len =
            expected_commitment_coeff_len_for_profile(&resolved.profiles().final_group)
                .expect("coefficients");
        commitment.serialized_backend_bytes = vec![0u8; field_elem_bytes()];
        let proof = AkitaBatchProof::new(resolved.selection(), Vec::new());
        let err = deserialize_checked_backend_payload(&schedules, &commitment, &proof, 2, &point)
            .expect_err("truncated commitment bytes must be rejected");
        assert!(err.to_string().contains("bytes"));
    }
}
