//! Pre-deserialization validation of proof-controlled Akita payload shapes.
//!
//! `AkitaCommitment::backend_coeff_len` and the serialized backend proof shape
//! arrive inside the (prover-controlled) proof, and the upstream Akita
//! deserializers reserve memory from those counts before reading the first
//! payload byte, bounded only by a generic `2^25`-element cap — a single
//! forged length can request ~512 MiB for the 128-bit field even when the
//! byte buffer is empty. This module re-derives the expected shapes from the
//! trusted verifier setup and the resolved schedule (the same
//! `effective_batched_schedule` the backend verifier replays) and rejects
//! mismatches *before* any shape-backed allocation happens, so verifier
//! memory stays proportional to the bytes the prover actually supplied.

use akita_config::{effective_batched_schedule, CommitmentConfig};
use akita_pcs::AkitaError;
use akita_schedules::ResolvedScheduleRow;
use akita_types::{
    relation_rhs_layout_for, sumcheck_rounds, CommittedGroupParams, CommittedGroupProfile,
    CompressionChainPlan, DigitRangePlan, ExtensionOpeningReductionShape, FoldSchedule,
    LevelProofShape, NextWitnessBindingShape, OpeningClaimsLayout, OpeningScheduleSelection,
    PolynomialGroupLayout, RecursiveFoldParams, TerminalLevelProofShape,
};

use crate::adapters::{
    deserialize_akita, invalid_batch, AkitaBackendCommitment, AkitaBackendCommitmentPayload,
    AkitaBackendFlavor, AkitaBackendProof, AkitaBackendProofShape, AkitaBatchProof,
    AkitaCommitment, AkitaConfig, AkitaField, AkitaOneHotK16Config, AkitaOneHotK256Config,
    AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256,
};
use jolt_openings::OpeningsError;

/// Serialized proof-shape blob cap. Honest shapes are a few hundred bytes (a
/// handful of fold levels, each a few dozen words); this leaves two orders of
/// magnitude of margin while keeping worst-case shape-blob deserialization
/// allocations trivial.
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

/// Fold sumcheck round counts are `log2(ring_dim) + log2(witness columns)`,
/// far below 64 for any representable witness.
const MAX_SUMCHECK_ROUNDS: usize = 64;

/// Per-round compact coefficient counts are `degree`-sized; every sumcheck in
/// the batched protocol has degree <= 4 (stage-1 tree arities, degree-3
/// stage-2, degree-2 reductions).
const MAX_ROUND_DEGREE: usize = 8;

/// Extension-opening partials are one short vector of basis-conversion
/// evaluations; bound them so a forged shape cannot drive a large reserve.
const MAX_EXT_REDUCTION_PARTIALS: usize = 1 << 12;

/// Stage-2 fold sumchecks are degree 3 (see akita's `proof_size.rs`, the
/// planner's single source of truth for per-level proof accounting).
const STAGE2_SUMCHECK_DEGREE: usize = 3;

/// Deserializes the backend commitment and batched proof after validating
/// every prover-controlled shape against the trusted schedule.
///
/// `backend_point` must be the point in the backend's coordinate order (the
/// same order `verify_batch` hands to the backend verifier), and the
/// statement must already have passed `validate_statement`.
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
    let schedule = resolved.schedule();

    validate_commitment_len(commitment, schedule, &layout)?;
    let backend_payload = deserialize_akita::<AkitaBackendCommitmentPayload>(
        &commitment.serialized_backend_bytes,
        &commitment.backend_coeff_len,
    )?;
    // The commitment profile comes from the trusted resolved row; the proof
    // supplies only the payload coefficients.
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
    validate_proof_shape(&proof_shape, schedule)?;
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
        AKITA_ONE_HOT_K16 => resolve_grouped_schedule::<crate::adapters::AkitaOneHotK16Config>(
            selection,
            &layout,
            main_backend_point,
        ),
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
    validate_proof_shape(&proof_shape, resolved.schedule())?;
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

fn validate_commitment_profile_len(
    commitment: &AkitaCommitment,
    profile: &CommittedGroupProfile,
) -> Result<(), OpeningsError> {
    let source_coefficients = profile
        .outer_commit_matrix
        .output_rank()
        .checked_mul(profile.outer_commit_matrix.ring_dimension())
        .ok_or_else(|| invalid_batch("Akita commitment coefficient count overflows"))?;
    let expected_coeff_len = CompressionChainPlan::for_complete_source(
        profile.outer_commit_matrix.sis_table_key().modulus_profile,
        source_coefficients,
    )
    .map_err(|err| {
        invalid_batch(format!(
            "Akita commitment compression plan is invalid: {err}"
        ))
    })?
    .terminal_coefficients();
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

/// Selects the generated row for one single-group statement key and validates
/// its opening geometry, mirroring the backend verifier's own resolution.
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

/// Resolves the same schedule row the backend verifier will replay for this
/// statement, dispatching on the commitment's (already-validated) flavor.
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

/// The commitment is `u in R_q^{n_B}` for the schedule's root commit layout,
/// so its exact field-coefficient count is schedule-determined. Field
/// elements serialize at a fixed width, so the byte-buffer length is checked
/// too — after this, commitment deserialization reads exactly the supplied
/// bytes.
fn validate_commitment_len(
    commitment: &AkitaCommitment,
    schedule: &FoldSchedule,
    layout: &OpeningClaimsLayout,
) -> Result<(), OpeningsError> {
    let expected_coeff_len = expected_commitment_coeff_len(schedule, layout)?;
    if commitment.backend_coeff_len != expected_coeff_len {
        return Err(invalid_batch(format!(
            "Akita commitment declares {} backend coefficients but the schedule requires {expected_coeff_len}",
            commitment.backend_coeff_len
        )));
    }
    let elem_bytes = field_elem_bytes();
    let expected_bytes = expected_coeff_len
        .checked_mul(elem_bytes)
        .ok_or_else(|| invalid_batch("Akita commitment byte size overflows"))?;
    if commitment.serialized_backend_bytes.len() != expected_bytes {
        return Err(invalid_batch(format!(
            "Akita commitment has {} serialized bytes but {expected_coeff_len} coefficients require {expected_bytes}",
            commitment.serialized_backend_bytes.len()
        )));
    }
    Ok(())
}

/// Mirrors the backend verifier's root replay: the commitment payload is the
/// final group's compressed B image, whose exact transmitted coefficient
/// count comes from the relation layout's compression plan for the group.
fn expected_commitment_coeff_len(
    schedule: &FoldSchedule,
    layout: &OpeningClaimsLayout,
) -> Result<usize, OpeningsError> {
    let root_params = &schedule.root.params.final_group.commitment;
    let relation = relation_rhs_layout_for(root_params, layout)
        .map_err(|err| invalid_batch(format!("Akita schedule layout error: {err}")))?;
    let plan = relation
        .compression_plan_for_group(0)
        .map_err(|err| invalid_batch(format!("Akita schedule layout error: {err}")))?;
    Ok(plan.terminal_coefficients())
}

fn field_elem_bytes() -> usize {
    use akita_pcs::AkitaSerialize;
    AkitaField::zero().compressed_size()
}

/// Validates a deserialized proof shape against the resolved schedule before
/// the proof body is deserialized: the schedule-determined counts (fold-level
/// structure, `v`/next-commitment coefficient counts, sumcheck stage shapes)
/// must match exactly, the terminal response must be admitted by the
/// schedule's response shape (which bounds its Golomb `z` payload budgets),
/// and the remaining log-scale quantities are held to protocol bounds.
fn validate_proof_shape(
    shape: &AkitaBackendProofShape,
    schedule: &FoldSchedule,
) -> Result<(), OpeningsError> {
    if shape.recursive_folds.len() != schedule.recursive_folds.len() {
        return Err(invalid_batch(format!(
            "Akita proof shape has {} recursive fold levels but the schedule requires {}",
            shape.recursive_folds.len(),
            schedule.recursive_folds.len()
        )));
    }
    validate_level_shape(
        &shape.root,
        &schedule.root.params.final_group.commitment,
        schedule.root.output_witness_len,
        schedule.recursive_folds.first().map(|step| &step.params),
    )?;
    for (index, (level_shape, step)) in shape
        .recursive_folds
        .iter()
        .zip(schedule.recursive_folds.iter())
        .enumerate()
    {
        validate_level_shape(
            level_shape,
            &step.params.witness,
            step.output_witness_len,
            schedule
                .recursive_folds
                .get(index + 1)
                .map(|next| &next.params),
        )?;
    }
    validate_terminal_level_shape(&shape.terminal, schedule)
}

/// Validates one non-terminal fold level against the schedule step that
/// produced it. `successor` is the next recursive fold's params, or `None`
/// when the next level is the terminal (which owns the canonical `t` state,
/// so the edge ships no outer commitment).
fn validate_level_shape(
    shape: &LevelProofShape,
    params: &CommittedGroupParams,
    output_witness_len: usize,
    successor: Option<&RecursiveFoldParams>,
) -> Result<(), OpeningsError> {
    validate_ext_reduction_shape(shape.extension_opening_reduction.as_ref())?;

    let expected_opening_payload_coeffs = params
        .opening_payload_geometry()
        .map_err(|err| invalid_batch(format!("Akita schedule error: {err}")))?
        .transmitted_coefficients();
    if shape.opening_payload_coeffs != expected_opening_payload_coeffs {
        return Err(invalid_batch(format!(
            "Akita level shape declares {} opening payload coefficients but the schedule requires {expected_opening_payload_coeffs}",
            shape.opening_payload_coeffs
        )));
    }

    let rounds = sumcheck_rounds(params.d_a(), output_witness_len);
    if shape.stage2_sumcheck_proof != vec![STAGE2_SUMCHECK_DEGREE; rounds] {
        return Err(invalid_batch(
            "Akita level shape stage-2 sumcheck does not match the scheduled rounds",
        ));
    }
    let expected_stage1 = DigitRangePlan::new(1usize << params.log_basis_open)
        .map_err(|err| invalid_batch(format!("Akita schedule error: {err}")))?
        .proof_shapes_for_route(rounds, params.inner_commit_matrix.security_route())
        .map_err(|err| invalid_batch(format!("Akita schedule error: {err}")))?;
    if shape.stage1_stages != expected_stage1.0 {
        return Err(invalid_batch(
            "Akita level shape stage-1 tree does not match the scheduled stages",
        ));
    }
    if shape.stage1_norm != expected_stage1.1 {
        return Err(invalid_batch(
            "Akita level shape stage-1 norm proof does not match the scheduled security route",
        ));
    }

    match (successor, shape.next_witness_binding) {
        (Some(next), NextWitnessBindingShape::OuterPayload { coeffs }) => {
            let expected_next_commit = next
                .witness
                .outer_payload_geometry()
                .map_err(|err| invalid_batch(format!("Akita schedule error: {err}")))?
                .transmitted_coefficients();
            if coeffs != expected_next_commit {
                return Err(invalid_batch(format!(
                    "Akita level shape declares {coeffs} next-commitment coefficients but the schedule requires {expected_next_commit}",
                )));
            }
        }
        (None, NextWitnessBindingShape::TerminalInnerState) => {}
        _ => {
            return Err(invalid_batch(
                "Akita level shape witness binding does not match the schedule position",
            ));
        }
    }

    if successor.is_some_and(|next| next.incoming_setup_prefix.is_some()) {
        // Jolt's presets plan direct-only schedules; a recursive
        // setup-contribution edge would require validating its stage-3
        // payload, which this guard does not model.
        return Err(invalid_batch(
            "Akita recursive setup-contribution schedules are not supported",
        ));
    }
    if shape.stage3_sumcheck.is_some() {
        // A stage-3 setup-product payload only exists on an edge whose
        // successor consumes an incoming setup prefix, which direct-only
        // schedules never produce.
        return Err(invalid_batch(
            "Akita level shape carries a stage-3 payload in direct setup-contribution mode",
        ));
    }
    Ok(())
}

fn validate_terminal_level_shape(
    shape: &TerminalLevelProofShape,
    schedule: &FoldSchedule,
) -> Result<(), OpeningsError> {
    validate_ext_reduction_shape(shape.extension_opening_reduction.as_ref())?;
    let scheduled = &schedule.terminal.params.response_shape;
    if !scheduled.admits_realized(&shape.terminal_response) {
        return Err(invalid_batch(
            "Akita terminal response shape is not admitted by the scheduled response shape",
        ));
    }
    Ok(())
}

fn validate_ext_reduction_shape(
    shape: Option<&ExtensionOpeningReductionShape>,
) -> Result<(), OpeningsError> {
    let Some(shape) = shape else {
        return Ok(());
    };
    if shape.partials > MAX_EXT_REDUCTION_PARTIALS {
        return Err(invalid_batch(format!(
            "Akita extension-opening reduction declares {} partials but the protocol cap is {MAX_EXT_REDUCTION_PARTIALS}",
            shape.partials
        )));
    }
    validate_bounded_sumcheck_shape("extension-opening reduction", &shape.sumcheck)
}

fn validate_bounded_sumcheck_shape(context: &str, shape: &[usize]) -> Result<(), OpeningsError> {
    if shape.len() > MAX_SUMCHECK_ROUNDS {
        return Err(invalid_batch(format!(
            "Akita {context} sumcheck declares {} rounds but the protocol cap is {MAX_SUMCHECK_ROUNDS}",
            shape.len()
        )));
    }
    if let Some(&degree) = shape.iter().find(|&&degree| degree > MAX_ROUND_DEGREE) {
        return Err(invalid_batch(format!(
            "Akita {context} sumcheck declares a degree-{degree} round but the protocol cap is {MAX_ROUND_DEGREE}"
        )));
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

    fn test_resolve_schedule(
        commitment: &AkitaCommitment,
        layout: &OpeningClaimsLayout,
        point: &[AkitaField],
    ) -> Result<ResolvedScheduleRow, OpeningsError> {
        resolve_schedule(
            commitment,
            test_selection(commitment.num_vars, commitment.poly_count),
            layout,
            point,
        )
    }

    /// The honest proof shape the schedule prescribes, built from the same
    /// step params the validator checks against.
    fn scheduled_proof_shape(schedule: &FoldSchedule) -> AkitaBackendProofShape {
        let level = |params: &CommittedGroupParams,
                     output_witness_len: usize,
                     successor: Option<&RecursiveFoldParams>| {
            let rounds = sumcheck_rounds(params.d_a(), output_witness_len);
            let stage1 = DigitRangePlan::new(1usize << params.log_basis_open)
                .expect("scheduled range basis")
                .proof_shapes_for_route(rounds, params.inner_commit_matrix.security_route())
                .expect("scheduled stage-1 shape");
            LevelProofShape {
                extension_opening_reduction: None,
                opening_payload_coeffs: params
                    .opening_payload_geometry()
                    .expect("scheduled opening payload geometry")
                    .transmitted_coefficients(),
                stage1_stages: stage1.0,
                stage1_norm: stage1.1,
                stage2_sumcheck_proof: vec![STAGE2_SUMCHECK_DEGREE; rounds],
                stage3_sumcheck: None,
                next_witness_binding: match successor {
                    Some(next) => NextWitnessBindingShape::OuterPayload {
                        coeffs: next
                            .witness
                            .outer_payload_geometry()
                            .expect("scheduled outer payload geometry")
                            .transmitted_coefficients(),
                    },
                    None => NextWitnessBindingShape::TerminalInnerState,
                },
            }
        };
        AkitaBackendProofShape {
            root: level(
                &schedule.root.params.final_group.commitment,
                schedule.root.output_witness_len,
                schedule.recursive_folds.first().map(|step| &step.params),
            ),
            recursive_folds: schedule
                .recursive_folds
                .iter()
                .enumerate()
                .map(|(index, step)| {
                    level(
                        &step.params.witness,
                        step.output_witness_len,
                        schedule
                            .recursive_folds
                            .get(index + 1)
                            .map(|next| &next.params),
                    )
                })
                .collect(),
            terminal: TerminalLevelProofShape {
                extension_opening_reduction: None,
                terminal_response: schedule.terminal.params.response_shape.clone(),
            },
        }
    }

    #[test]
    fn forged_commitment_coeff_len_rejects_before_deserialization() {
        let num_vars = 16;
        let point = point(num_vars);
        let mut commitment = dense_commitment(num_vars, 2);
        // A honest-shape claim would be a few thousand coefficients; forge the
        // upstream 2^25 cap with an empty byte buffer.
        commitment.backend_coeff_len = 1 << 25;
        let proof = AkitaBatchProof {
            statement_bridge: Vec::new(),
            serialized_schedule_selection: serialize_akita(&test_selection(num_vars, 2))
                .expect("serialize selection"),
            serialized_akita_proof_shape: Vec::new(),
            serialized_akita_proof: Vec::new(),
        };
        let err = deserialize_checked_backend_payload(&commitment, &proof, 2, &point)
            .expect_err("forged coefficient count must be rejected");
        assert!(
            err.to_string().contains("coefficients"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn commitment_byte_length_must_match_coeff_len() {
        let num_vars = 16;
        let point = point(num_vars);
        let layout = OpeningClaimsLayout::new(num_vars, 2).expect("layout");
        let mut commitment = dense_commitment(num_vars, 2);
        let schedule = test_resolve_schedule(&commitment, &layout, &point)
            .expect("schedule")
            .into_schedule();
        let expected = expected_commitment_coeff_len(&schedule, &layout).expect("coeff len");
        commitment.backend_coeff_len = expected;
        // Correct declared count, truncated byte buffer: the deserializer
        // would reserve `expected` coefficients before hitting EOF.
        commitment.serialized_backend_bytes = vec![0u8; field_elem_bytes()];
        let proof = AkitaBatchProof {
            statement_bridge: Vec::new(),
            serialized_schedule_selection: serialize_akita(&test_selection(num_vars, 2))
                .expect("serialize selection"),
            serialized_akita_proof_shape: Vec::new(),
            serialized_akita_proof: Vec::new(),
        };
        let err = deserialize_checked_backend_payload(&commitment, &proof, 2, &point)
            .expect_err("truncated commitment bytes must be rejected");
        assert!(err.to_string().contains("bytes"), "unexpected error: {err}");
    }

    #[test]
    fn oversized_proof_shape_blob_rejects() {
        let num_vars = 16;
        let point = point(num_vars);
        let layout = OpeningClaimsLayout::new(num_vars, 2).expect("layout");
        let mut commitment = dense_commitment(num_vars, 2);
        let schedule = test_resolve_schedule(&commitment, &layout, &point)
            .expect("schedule")
            .into_schedule();
        let coeff_len = expected_commitment_coeff_len(&schedule, &layout).expect("coeff len");
        commitment.backend_coeff_len = coeff_len;
        commitment.serialized_backend_bytes = vec![0u8; coeff_len * field_elem_bytes()];
        let proof = AkitaBatchProof {
            statement_bridge: Vec::new(),
            serialized_schedule_selection: serialize_akita(&test_selection(num_vars, 2))
                .expect("serialize selection"),
            serialized_akita_proof_shape: vec![0u8; MAX_PROOF_SHAPE_BYTES + 1],
            serialized_akita_proof: Vec::new(),
        };
        let err = deserialize_checked_backend_payload(&commitment, &proof, 2, &point)
            .expect_err("oversized shape blob must be rejected");
        assert!(
            err.to_string().contains("protocol cap"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn scheduled_shape_passes_validation() {
        let num_vars = 16;
        let point = point(num_vars);
        let layout = OpeningClaimsLayout::new(num_vars, 2).expect("layout");
        let commitment = dense_commitment(num_vars, 2);
        let schedule = test_resolve_schedule(&commitment, &layout, &point)
            .expect("schedule")
            .into_schedule();
        let shape = scheduled_proof_shape(&schedule);
        validate_proof_shape(&shape, &schedule).expect("scheduled shape must validate");
    }

    #[test]
    fn forged_shape_counts_reject_against_schedule() {
        let num_vars = 16;
        let point = point(num_vars);
        let layout = OpeningClaimsLayout::new(num_vars, 2).expect("layout");
        let mut commitment = dense_commitment(num_vars, 2);
        let schedule = test_resolve_schedule(&commitment, &layout, &point)
            .expect("schedule")
            .into_schedule();
        let coeff_len = expected_commitment_coeff_len(&schedule, &layout).expect("coeff len");
        commitment.backend_coeff_len = coeff_len;
        commitment.serialized_backend_bytes = vec![0u8; coeff_len * field_elem_bytes()];

        // A structurally plausible shape whose big counts are forged to the
        // upstream cap; the schedule comparison must reject it without the
        // proof body ever being deserialized.
        let mut forged = scheduled_proof_shape(&schedule);
        forged.root.opening_payload_coeffs = 1 << 25;
        let proof = AkitaBatchProof {
            statement_bridge: Vec::new(),
            serialized_schedule_selection: serialize_akita(&test_selection(num_vars, 2))
                .expect("serialize selection"),
            serialized_akita_proof_shape: serialize_akita(&forged).expect("serialize shape"),
            serialized_akita_proof: Vec::new(),
        };
        let err = deserialize_checked_backend_payload(&commitment, &proof, 2, &point)
            .expect_err("forged shape counts must be rejected");
        assert!(
            err.to_string().contains("schedule"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn forged_terminal_payload_budget_rejects_against_schedule() {
        let num_vars = 16;
        let point = point(num_vars);
        let layout = OpeningClaimsLayout::new(num_vars, 2).expect("layout");
        let commitment = dense_commitment(num_vars, 2);
        let schedule = test_resolve_schedule(&commitment, &layout, &point)
            .expect("schedule")
            .into_schedule();

        // Forge the terminal Golomb `z` payload budget past the scheduled
        // upper bound; the admits check must reject before any payload-sized
        // reserve.
        let mut forged = scheduled_proof_shape(&schedule);
        for group in &mut forged.terminal.terminal_response.layout.groups {
            group.z_payload_bytes = 1 << 25;
        }
        let err = validate_proof_shape(&forged, &schedule)
            .expect_err("forged terminal payload budget must be rejected");
        assert!(
            err.to_string().contains("admitted"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn bounded_sumcheck_shape_enforces_protocol_caps() {
        assert!(validate_bounded_sumcheck_shape("test", &vec![3; MAX_SUMCHECK_ROUNDS]).is_ok());
        assert!(
            validate_bounded_sumcheck_shape("test", &vec![3; MAX_SUMCHECK_ROUNDS + 1]).is_err()
        );
        assert!(validate_bounded_sumcheck_shape("test", &[MAX_ROUND_DEGREE + 1]).is_err());
    }
}
