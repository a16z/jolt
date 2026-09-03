//! Pre-deserialization validation of proof-controlled Akita payload shapes.
//!
//! The commitment length and serialized proof shape arrive inside the
//! prover-controlled Jolt proof. Validate both against Akita's trusted
//! schedule before a backend deserializer can reserve payload-sized buffers.

use akita_config::{derive_transcript_grinding_plan, effective_batched_schedule, CommitmentConfig};
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
    match commitment.backend_flavor {
        AkitaBackendFlavor::Dense => deserialize_checked_single_payload::<AkitaConfig>(
            commitment,
            proof,
            selection,
            &layout,
            backend_point,
        ),
        AkitaBackendFlavor::OneHot => match commitment.one_hot_k {
            AKITA_ONE_HOT_K16 => deserialize_checked_single_payload::<AkitaOneHotK16Config>(
                commitment,
                proof,
                selection,
                &layout,
                backend_point,
            ),
            AKITA_ONE_HOT_K256 => deserialize_checked_single_payload::<AkitaOneHotK256Config>(
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
    match one_hot_k {
        AKITA_ONE_HOT_K256 => deserialize_checked_grouped_payload::<AkitaOneHotK256Config>(
            precommitted,
            main,
            proof,
            selection,
            &layout,
            main_backend_point,
        ),
        AKITA_ONE_HOT_K16 => deserialize_checked_grouped_payload::<AkitaOneHotK16Config>(
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
    let resolved = resolve_schedule_row::<Cfg>(selection, layout, backend_point)
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
    let resolved = resolve_schedule_row::<Cfg>(selection, layout, main_backend_point)
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
    if proof.serialized_akita_proof_shape.len() > MAX_PROOF_SHAPE_BYTES {
        return Err(invalid_batch(format!(
            "Akita proof shape blob is {} bytes but the protocol cap is {MAX_PROOF_SHAPE_BYTES}",
            proof.serialized_akita_proof_shape.len()
        )));
    }
    let proof_shape =
        deserialize_akita::<AkitaBackendProofShape>(&proof.serialized_akita_proof_shape, &())?;
    validate_proof_shape::<Cfg>(&proof_shape, resolved.schedule(), layout)?;
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
    Ok(backend_proof)
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

fn validate_proof_shape<Cfg>(
    shape: &AkitaBackendProofShape,
    schedule: &FoldSchedule,
    layout: &OpeningClaimsLayout,
) -> Result<(), OpeningsError>
where
    Cfg: CommitmentConfig<Field = AkitaField, ExtField = AkitaField>,
{
    let grinding_plan = derive_transcript_grinding_plan::<Cfg>(schedule, layout)
        .map_err(|err| invalid_batch(format!("Akita grinding plan is invalid: {err}")))?;
    let expected = canonical_proof_shape(schedule, layout, Cfg::EXT_DEGREE, &grinding_plan)
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
    #![expect(
        clippy::panic,
        reason = "tests destructure honest proof shapes and fail loudly on fixture drift"
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
        let resolved = resolve_schedule_row::<AkitaConfig>(
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
            expected_commitment_coeff_len_for_profile(&resolved.profiles().final_group)
                .expect("expected coefficients")
        );
        assert!(err.to_string().contains("coefficients"));
    }

    #[test]
    fn commitment_byte_length_must_match_coeff_len() {
        let (mut commitment, point, _, resolved) = resolved_dense(16, 2);
        commitment.backend_coeff_len =
            expected_commitment_coeff_len_for_profile(&resolved.profiles().final_group)
                .expect("coefficients");
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
        let coeff_len = expected_commitment_coeff_len_for_profile(&resolved.profiles().final_group)
            .expect("coefficients");
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
    fn canonical_shape_passes_and_forged_fields_reject() {
        let (_, _, layout, resolved) = resolved_dense(16, 2);
        let grinding_plan =
            derive_transcript_grinding_plan::<AkitaConfig>(resolved.schedule(), &layout)
                .expect("grinding plan");
        let mut shape = canonical_proof_shape(
            resolved.schedule(),
            &layout,
            AkitaConfig::EXT_DEGREE,
            &grinding_plan,
        )
        .expect("canonical shape");
        validate_proof_shape::<AkitaConfig>(&shape, resolved.schedule(), &layout)
            .expect("valid shape");

        shape.nonce_stream_bits ^= 1;
        let err = validate_proof_shape::<AkitaConfig>(&shape, resolved.schedule(), &layout)
            .expect_err("forged grinding stream width must reject");
        assert!(err.to_string().contains("resolved schedule"));

        shape.nonce_stream_bits ^= 1;
        shape.root.opening_payload_coeffs = 1 << 25;
        let encoded = serialize_akita(&shape).expect("serialize shape");
        let decoded = deserialize_akita::<AkitaBackendProofShape>(&encoded, &()).expect("shape");
        let err = validate_proof_shape::<AkitaConfig>(&decoded, resolved.schedule(), &layout)
            .expect_err("forged shape must reject");
        assert!(err.to_string().contains("resolved schedule"));
    }

    #[test]
    fn resolve_schedule_rejects_unknown_one_hot_chunk_size() {
        let num_vars = 13;
        let point = point(num_vars);
        let layout = OpeningClaimsLayout::new(num_vars, 1).expect("layout");
        let mut commitment = dense_commitment(num_vars, 1);
        commitment.backend_flavor = AkitaBackendFlavor::OneHot;
        commitment.one_hot_k = 32;
        let err = resolve_schedule(&commitment, &layout, &point)
            .expect_err("unknown one-hot chunk size must be rejected");
        assert!(
            err.to_string().contains("unsupported Akita one-hot K"),
            "unexpected error: {err}"
        );
    }

    /// A real prover run must realize exactly the fold structure the
    /// schedule prescribes — `scheduled_proof_shape` is derived from the
    /// schedule, so this ties the validator's model to actual backend
    /// prover output.
    #[test]
    fn real_proof_shape_validates_against_the_resolved_schedule() {
        use crate::{AkitaScheme, AkitaSetupParams};
        use jolt_openings::CommitmentScheme;
        use jolt_poly::Polynomial;
        use jolt_transcript::{Blake2bTranscript, Transcript};

        let num_vars = 14;
        let (prover_setup, _) = AkitaScheme::setup(AkitaSetupParams::new(num_vars, 1, [7; 32]))
            .expect("dense setup should build");
        let poly = Polynomial::new(
            (0..1u64 << num_vars)
                .map(|index| AkitaField::from_u64(index + 1))
                .collect(),
        );
        let (commitment, hint) =
            AkitaScheme::commit(&poly, &prover_setup).expect("dense commit should succeed");
        let point = point(num_vars);
        let eval = poly.evaluate(&point);
        let mut transcript = Blake2bTranscript::<AkitaField>::new(b"shape-guard-fixture");
        let proof = AkitaScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut transcript,
        )
        .expect("open should succeed");

        let layout = OpeningClaimsLayout::new(num_vars, 1).expect("layout");
        let resolved = resolve_schedule(&commitment, &layout, &point).expect("schedule");
        let schedule = resolved.schedule();
        let realized =
            deserialize_akita::<AkitaBackendProofShape>(&proof.serialized_akita_proof_shape, &())
                .expect("honest proof shape should deserialize");
        validate_proof_shape(&realized, schedule).expect("realized shape must validate");
        assert_eq!(
            realized.recursive_folds.len(),
            schedule.recursive_folds.len(),
            "prover must realize the scheduled fold depth"
        );
    }

    fn expect_shape_rejection(
        shape: &AkitaBackendProofShape,
        schedule: &FoldSchedule,
        expected_fragment: &str,
    ) {
        let err =
            validate_proof_shape(shape, schedule).expect_err("forged proof shape must be rejected");
        assert!(
            err.to_string().contains(expected_fragment),
            "expected error containing {expected_fragment:?}, got: {err}"
        );
    }

    /// Every count, stage, and witness-binding field of a level shape is
    /// schedule-determined; each single-field forgery of the scheduled shape
    /// must reject with its own diagnostic.
    #[test]
    fn forged_level_and_terminal_shapes_reject_against_the_schedule() {
        let num_vars = 20;
        let point = point(num_vars);
        let layout = OpeningClaimsLayout::new(num_vars, 2).expect("layout");
        let commitment = dense_commitment(num_vars, 2);
        let resolved = resolve_schedule(&commitment, &layout, &point).expect("schedule");
        let schedule = resolved.schedule();
        assert!(
            !schedule.recursive_folds.is_empty(),
            "fixture needs at least one recursive fold"
        );
        let honest = scheduled_proof_shape(schedule);
        validate_proof_shape(&honest, schedule).expect("scheduled shape must validate");

        let mut extra_level = honest.clone();
        extra_level.recursive_folds.push(extra_level.root.clone());
        expect_shape_rejection(&extra_level, schedule, "recursive fold levels");

        let mut truncated = honest.clone();
        let _ = truncated.recursive_folds.pop();
        expect_shape_rejection(&truncated, schedule, "recursive fold levels");

        let mut forged_stage2 = honest.clone();
        forged_stage2
            .root
            .stage2_sumcheck_proof
            .push(STAGE2_SUMCHECK_DEGREE);
        expect_shape_rejection(&forged_stage2, schedule, "stage-2");

        let mut forged_stage1 = honest.clone();
        let _ = forged_stage1.root.stage1_stages.pop();
        expect_shape_rejection(&forged_stage1, schedule, "stage-1");

        let mut forged_stage3 = honest.clone();
        forged_stage3.root.stage3_sumcheck = Some(akita_types::SetupProductSumcheckShape {
            sumcheck: vec![STAGE2_SUMCHECK_DEGREE],
        });
        expect_shape_rejection(&forged_stage3, schedule, "stage-3");

        // The root has a recursive successor, so its outgoing binding must be
        // an outer commitment with the successor's exact coefficient count.
        let NextWitnessBindingShape::OuterPayload { coeffs } = honest.root.next_witness_binding
        else {
            panic!("a root with a successor must bind an outer commitment");
        };
        let mut forged_commit = honest.clone();
        forged_commit.root.next_witness_binding =
            NextWitnessBindingShape::OuterPayload { coeffs: coeffs + 1 };
        expect_shape_rejection(&forged_commit, schedule, "next-commitment");

        let mut forged_binding = honest.clone();
        forged_binding.root.next_witness_binding = NextWitnessBindingShape::TerminalInnerState;
        expect_shape_rejection(&forged_binding, schedule, "witness binding");

        // The last recursive fold precedes the terminal, so an outer
        // commitment there contradicts the schedule position.
        let mut forged_tail = honest.clone();
        forged_tail
            .recursive_folds
            .last_mut()
            .expect("fixture has a recursive fold")
            .next_witness_binding = NextWitnessBindingShape::OuterPayload { coeffs: 1 };
        expect_shape_rejection(&forged_tail, schedule, "witness binding");

        let mut forged_partials = honest.clone();
        forged_partials.root.extension_opening_reduction = Some(ExtensionOpeningReductionShape {
            partials: MAX_EXT_REDUCTION_PARTIALS + 1,
            sumcheck: Vec::new(),
        });
        expect_shape_rejection(&forged_partials, schedule, "partials");

        let mut forged_reduction_rounds = honest.clone();
        forged_reduction_rounds.terminal.extension_opening_reduction =
            Some(ExtensionOpeningReductionShape {
                partials: 1,
                sumcheck: vec![2; MAX_SUMCHECK_ROUNDS + 1],
            });
        expect_shape_rejection(&forged_reduction_rounds, schedule, "protocol cap");
    }
}
