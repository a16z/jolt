//! Pre-deserialization validation of proof-controlled Akita payload shapes.
//!
//! The commitment length and serialized proof shape arrive inside the
//! prover-controlled Jolt proof. Validate both against Akita's trusted
//! schedule before a backend deserializer can reserve payload-sized buffers.

use akita_config::{effective_batched_schedule, CommitmentConfig};
use akita_pcs::AkitaError;
use akita_schedules::ResolvedScheduleRow;
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

/// Honest shapes are a few hundred bytes. This cap keeps shape-descriptor
/// parsing itself small before the descriptor is compared with the schedule.
const MAX_PROOF_SHAPE_BYTES: usize = 16 * 1024;
const SCHEDULE_SELECTION_BYTES: usize = 32;

fn deserialize_selection(
    proof: &AkitaBatchProof,
) -> Result<OpeningScheduleSelection, OpeningsError> {
    if proof.serialized_schedule_selection.len() != SCHEDULE_SELECTION_BYTES {
        return Err(invalid_batch(format!(
            "Akita schedule selection is {} bytes but the protocol requires {SCHEDULE_SELECTION_BYTES}",
            proof.serialized_schedule_selection.len()
        )));
    }
    deserialize_akita::<OpeningScheduleSelection>(&proof.serialized_schedule_selection, &())
}

/// Deserializes the backend commitment and proof only after their declared
/// shapes have been derived from the trusted resolved schedule.
pub(crate) fn deserialize_checked_backend_payload(
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
    let selection = deserialize_selection(proof)?;
    let resolved = resolve_schedule(commitment, selection, &layout, backend_point)?;

    validate_commitment_len(commitment, &resolved)?;
    let backend_payload = deserialize_akita::<AkitaBackendCommitmentPayload>(
        &commitment.serialized_backend_bytes,
        &commitment.backend_coeff_len,
    )?;
    let backend_commitment =
        AkitaBackendCommitment::new(resolved.profiles().final_group, backend_payload);

    if proof.serialized_akita_proof_shape.len() > MAX_PROOF_SHAPE_BYTES {
        return Err(invalid_batch(format!(
            "Akita proof shape blob is {} bytes but the protocol cap is {MAX_PROOF_SHAPE_BYTES}",
            proof.serialized_akita_proof_shape.len()
        )));
    }
    let proof_shape =
        deserialize_akita::<AkitaBackendProofShape>(&proof.serialized_akita_proof_shape, &())?;
    validate_proof_shape(&proof_shape, resolved.schedule(), &layout)?;
    proof_shape
        .validate_decode_budget(
            proof.serialized_akita_proof.len(),
            field_elem_bytes(),
            field_elem_bytes(),
        )
        .map_err(|err| {
            invalid_batch(format!("Akita proof shape exceeds its byte budget: {err}"))
        })?;
    let backend_proof =
        deserialize_akita::<AkitaBackendProof>(&proof.serialized_akita_proof, &proof_shape)?;
    Ok((resolved.selection(), backend_commitment, backend_proof))
}

/// Guard and decode the ordered grouped root in public order
/// `[dense precommits.., final streamed one-hot]`.
pub(crate) fn deserialize_checked_grouped_backend_payload(
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
    let selection = deserialize_selection(proof)?;
    let mut group_layouts = precommitted
        .iter()
        .map(|commitment| PolynomialGroupLayout::new(commitment.num_vars, commitment.poly_count))
        .collect::<Vec<_>>();
    group_layouts.push(PolynomialGroupLayout::new(main.num_vars, main.poly_count));
    let layout = OpeningClaimsLayout::from_groups(group_layouts)
        .map_err(|err| invalid_batch(format!("Akita grouped opening layout is invalid: {err}")))?;
    let resolved = match one_hot_k {
        AKITA_ONE_HOT_K256 => resolve_grouped_schedule::<AkitaOneHotK256Config>(
            selection,
            &layout,
            main_backend_point,
        ),
        AKITA_ONE_HOT_K16 => {
            resolve_grouped_schedule::<AkitaOneHotK16Config>(selection, &layout, main_backend_point)
        }
        _ => return Err(invalid_batch("unsupported grouped one-hot configuration")),
    }
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

    if proof.serialized_akita_proof_shape.len() > MAX_PROOF_SHAPE_BYTES {
        return Err(invalid_batch(format!(
            "Akita proof shape blob is {} bytes but the protocol cap is {MAX_PROOF_SHAPE_BYTES}",
            proof.serialized_akita_proof_shape.len()
        )));
    }
    let proof_shape =
        deserialize_akita::<AkitaBackendProofShape>(&proof.serialized_akita_proof_shape, &())?;
    validate_proof_shape(&proof_shape, resolved.schedule(), &layout)?;
    proof_shape
        .validate_decode_budget(
            proof.serialized_akita_proof.len(),
            field_elem_bytes(),
            field_elem_bytes(),
        )
        .map_err(|err| {
            invalid_batch(format!("Akita proof shape exceeds its byte budget: {err}"))
        })?;
    let backend_proof =
        deserialize_akita::<AkitaBackendProof>(&proof.serialized_akita_proof, &proof_shape)?;
    Ok((
        resolved.selection(),
        precommitted_backend,
        main_backend,
        backend_proof,
    ))
}

fn resolve_grouped_schedule<Cfg>(
    selection: OpeningScheduleSelection,
    layout: &OpeningClaimsLayout,
    main_backend_point: &[AkitaField],
) -> Result<ResolvedScheduleRow, AkitaError>
where
    Cfg: CommitmentConfig<Field = AkitaField, ExtField = AkitaField>,
{
    Cfg::resolve_schedule_selection(selection)
        .and_then(|row| effective_batched_schedule::<Cfg>(row, layout, main_backend_point))
}

fn resolve_schedule_row<Cfg>(
    selection: OpeningScheduleSelection,
    layout: &OpeningClaimsLayout,
    backend_point: &[AkitaField],
) -> Result<ResolvedScheduleRow, AkitaError>
where
    Cfg: CommitmentConfig<Field = AkitaField, ExtField = AkitaField>,
{
    let resolved = Cfg::resolve_schedule_selection(selection)?;
    effective_batched_schedule::<Cfg>(resolved, layout, backend_point)
}

fn resolve_schedule(
    commitment: &AkitaCommitment,
    selection: OpeningScheduleSelection,
    layout: &OpeningClaimsLayout,
    backend_point: &[AkitaField],
) -> Result<ResolvedScheduleRow, OpeningsError> {
    let schedule = match commitment.backend_flavor {
        AkitaBackendFlavor::Dense => {
            resolve_schedule_row::<AkitaConfig>(selection, layout, backend_point)
        }
        AkitaBackendFlavor::OneHot => match commitment.one_hot_k {
            AKITA_ONE_HOT_K16 => {
                resolve_schedule_row::<AkitaOneHotK16Config>(selection, layout, backend_point)
            }
            AKITA_ONE_HOT_K256 => {
                resolve_schedule_row::<AkitaOneHotK256Config>(selection, layout, backend_point)
            }
            one_hot_k => {
                return Err(invalid_batch(format!(
                    "unsupported Akita one-hot K={one_hot_k}"
                )))
            }
        },
    };
    schedule.map_err(|err| invalid_batch(format!("Akita schedule resolution failed: {err}")))
}

fn validate_commitment_len(
    commitment: &AkitaCommitment,
    resolved: &ResolvedScheduleRow,
) -> Result<(), OpeningsError> {
    let expected_coeff_len = expected_commitment_coeff_len(resolved)?;
    if commitment.backend_coeff_len != expected_coeff_len {
        return Err(invalid_batch(format!(
            "Akita commitment declares {} backend coefficients but the schedule requires {expected_coeff_len}",
            commitment.backend_coeff_len
        )));
    }
    let expected_bytes = expected_coeff_len
        .checked_mul(field_elem_bytes())
        .ok_or_else(|| invalid_batch("Akita commitment byte size overflows"))?;
    if commitment.serialized_backend_bytes.len() != expected_bytes {
        return Err(invalid_batch(format!(
            "Akita commitment has {} serialized bytes but {expected_coeff_len} coefficients require {expected_bytes}",
            commitment.serialized_backend_bytes.len()
        )));
    }
    Ok(())
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

/// Mirrors the backend verifier's frozen-profile commitment check.
fn expected_commitment_coeff_len(resolved: &ResolvedScheduleRow) -> Result<usize, OpeningsError> {
    expected_commitment_coeff_len_for_profile(&resolved.profiles().final_group)
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

fn validate_proof_shape(
    shape: &AkitaBackendProofShape,
    schedule: &FoldSchedule,
    layout: &OpeningClaimsLayout,
) -> Result<(), OpeningsError> {
    let expected = canonical_proof_shape(schedule, layout, AkitaConfig::EXT_DEGREE)
        .map_err(|err| invalid_batch(format!("Akita schedule proof shape is invalid: {err}")))?;
    if *shape != expected {
        return Err(invalid_batch(
            "Akita proof shape does not match the resolved schedule",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "tests assert successful schedule resolution"
    )]

    use super::*;
    use crate::adapters::serialize_akita;
    use akita_types::AkitaScheduleLookupKey;
    use jolt_field::Ring;

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

    fn test_selection(num_vars: usize, poly_count: usize) -> OpeningScheduleSelection {
        AkitaConfig::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::new(num_vars, poly_count),
        ))
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
    ) {
        let point = point(num_vars);
        let layout = OpeningClaimsLayout::new(num_vars, poly_count).expect("layout");
        let commitment = dense_commitment(num_vars, poly_count);
        let resolved = resolve_schedule(
            &commitment,
            test_selection(num_vars, poly_count),
            &layout,
            &point,
        )
        .expect("schedule");
        (commitment, point, layout, resolved)
    }

    #[test]
    fn forged_commitment_coeff_len_rejects_before_deserialization() {
        let (mut commitment, point, _, resolved) = resolved_dense(16, 2);
        commitment.backend_coeff_len = 1 << 25;
        let proof = AkitaBatchProof {
            statement_bridge: Vec::new(),
            serialized_schedule_selection: serialize_akita(&resolved.selection())
                .expect("serialize selection"),
            serialized_akita_proof_shape: Vec::new(),
            serialized_akita_proof: Vec::new(),
        };
        let err = deserialize_checked_backend_payload(&commitment, &proof, 2, &point)
            .expect_err("forged coefficient count must be rejected");
        assert_ne!(
            commitment.backend_coeff_len,
            expected_commitment_coeff_len(&resolved).expect("expected coefficients")
        );
        assert!(err.to_string().contains("coefficients"));
    }

    #[test]
    fn commitment_byte_length_must_match_coeff_len() {
        let (mut commitment, point, _, resolved) = resolved_dense(16, 2);
        commitment.backend_coeff_len =
            expected_commitment_coeff_len(&resolved).expect("coefficients");
        commitment.serialized_backend_bytes = vec![0u8; field_elem_bytes()];
        let proof = AkitaBatchProof {
            statement_bridge: Vec::new(),
            serialized_schedule_selection: serialize_akita(&resolved.selection())
                .expect("serialize selection"),
            serialized_akita_proof_shape: Vec::new(),
            serialized_akita_proof: Vec::new(),
        };
        let err = deserialize_checked_backend_payload(&commitment, &proof, 2, &point)
            .expect_err("truncated commitment bytes must be rejected");
        assert!(err.to_string().contains("bytes"));
    }

    #[test]
    fn oversized_proof_shape_blob_rejects() {
        let (mut commitment, point, _, resolved) = resolved_dense(16, 2);
        let coeff_len = expected_commitment_coeff_len(&resolved).expect("coefficients");
        commitment.backend_coeff_len = coeff_len;
        commitment.serialized_backend_bytes = vec![0u8; coeff_len * field_elem_bytes()];
        let proof = AkitaBatchProof {
            statement_bridge: Vec::new(),
            serialized_schedule_selection: serialize_akita(&resolved.selection())
                .expect("serialize selection"),
            serialized_akita_proof_shape: vec![0u8; MAX_PROOF_SHAPE_BYTES + 1],
            serialized_akita_proof: Vec::new(),
        };
        let err = deserialize_checked_backend_payload(&commitment, &proof, 2, &point)
            .expect_err("oversized shape must be rejected");
        assert!(err.to_string().contains("protocol cap"));
    }

    #[test]
    fn canonical_shape_passes_and_forged_count_rejects() {
        let (_, _, layout, resolved) = resolved_dense(16, 2);
        let mut shape =
            canonical_proof_shape(resolved.schedule(), &layout, AkitaConfig::EXT_DEGREE)
                .expect("canonical shape");
        validate_proof_shape(&shape, resolved.schedule(), &layout).expect("valid shape");
        shape.root.opening_payload_coeffs = 1 << 25;
        let encoded = serialize_akita(&shape).expect("serialize shape");
        let decoded = deserialize_akita::<AkitaBackendProofShape>(&encoded, &()).expect("shape");
        let err = validate_proof_shape(&decoded, resolved.schedule(), &layout)
            .expect_err("forged shape must reject");
        assert!(err.to_string().contains("resolved schedule"));
    }
}
