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

use akita_config::effective_batched_schedule;
use akita_types::{
    schedule_is_root_direct, schedule_root_fold_step, schedule_terminal_direct_witness_shape,
    stage1_tree_stage_shapes, sumcheck_rounds, AkitaProofStepShape, CleartextWitnessShape,
    ExtensionOpeningReductionShape, FoldStep, LevelProofShape, OpeningClaimsLayout,
    RelationMatrixRowLayout, Schedule, Step, TerminalLevelProofShape,
};

use crate::adapters::{
    deserialize_akita, invalid_batch, AkitaBackendCommitment, AkitaBackendFlavor,
    AkitaBackendProof, AkitaBackendProofShape, AkitaBatchProof, AkitaCommitment, AkitaConfig,
    AkitaField, AkitaOneHotK16Config, AkitaOneHotK256Config, AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256,
};
use jolt_openings::OpeningsError;

/// Serialized proof-shape blob cap. Honest shapes are a few hundred bytes (a
/// handful of fold levels, each a few dozen words); this leaves two orders of
/// magnitude of margin while keeping worst-case shape-blob deserialization
/// allocations trivial.
const MAX_PROOF_SHAPE_BYTES: usize = 16 * 1024;

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
) -> Result<(AkitaBackendCommitment, AkitaBackendProof), OpeningsError> {
    let layout = OpeningClaimsLayout::new(backend_point.len(), statement_len)
        .map_err(|err| invalid_batch(format!("Akita opening layout is invalid: {err}")))?;
    let schedule = resolve_schedule(commitment, &layout, backend_point)?;

    validate_commitment_len(commitment, &schedule, &layout)?;
    let backend_commitment = deserialize_akita::<AkitaBackendCommitment>(
        &commitment.serialized_backend_bytes,
        &commitment.backend_coeff_len,
    )?;

    if proof.serialized_akita_proof_shape.len() > MAX_PROOF_SHAPE_BYTES {
        return Err(invalid_batch(format!(
            "Akita proof shape blob is {} bytes but the protocol cap is {MAX_PROOF_SHAPE_BYTES}",
            proof.serialized_akita_proof_shape.len()
        )));
    }
    let proof_shape =
        deserialize_akita::<AkitaBackendProofShape>(&proof.serialized_akita_proof_shape, &())?;
    validate_proof_shape(&proof_shape, &schedule, &layout)?;
    let backend_proof =
        deserialize_akita::<AkitaBackendProof>(&proof.serialized_akita_proof, &proof_shape)?;
    Ok((backend_commitment, backend_proof))
}

/// Resolves the same schedule the backend verifier will replay for this
/// statement, dispatching on the commitment's (already-validated) flavor.
fn resolve_schedule(
    commitment: &AkitaCommitment,
    layout: &OpeningClaimsLayout,
    backend_point: &[AkitaField],
) -> Result<Schedule, OpeningsError> {
    let schedule = match commitment.backend_flavor {
        AkitaBackendFlavor::Dense => {
            effective_batched_schedule::<AkitaConfig>(layout, backend_point)
        }
        AkitaBackendFlavor::OneHot => match commitment.one_hot_k {
            AKITA_ONE_HOT_K16 => {
                effective_batched_schedule::<AkitaOneHotK16Config>(layout, backend_point)
            }
            AKITA_ONE_HOT_K256 => {
                effective_batched_schedule::<AkitaOneHotK256Config>(layout, backend_point)
            }
            other => {
                return Err(invalid_batch(format!(
                    "Akita one-hot chunk size must be 16 or 256, got {other}"
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
    schedule: &Schedule,
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

fn expected_commitment_coeff_len(
    schedule: &Schedule,
    layout: &OpeningClaimsLayout,
) -> Result<usize, OpeningsError> {
    let schedule_error =
        |err: &dyn std::fmt::Display| invalid_batch(format!("Akita schedule layout error: {err}"));
    if let Some(root_step) = schedule_root_fold_step(schedule) {
        // Mirrors the backend verifier's suffix replay: the commitment rows
        // are the root relation matrix's B block, decoded at the A-role ring
        // dimension.
        let rows = root_step
            .params
            .commitment_row_range(layout, 0, RelationMatrixRowLayout::WithDBlock)
            .map_err(|err| schedule_error(&err))?
            .len();
        return rows
            .checked_mul(root_step.params.role_dims().d_a())
            .ok_or_else(|| invalid_batch("Akita commitment coefficient count overflows"));
    }
    // Root-direct schedule: the commitment is replayed against the direct
    // step's root commit layout at the B-role ring dimension.
    let Some(Step::Direct(direct)) = schedule.steps.first() else {
        return Err(invalid_batch("Akita schedule has no steps"));
    };
    let params = direct.params.as_ref().ok_or_else(|| {
        invalid_batch("Akita root-direct schedule has no committable root layout")
    })?;
    let rows = params
        .group_commitment_rows(layout, 0)
        .map_err(|err| schedule_error(&err))?;
    rows.checked_mul(params.role_dims().d_b())
        .ok_or_else(|| invalid_batch("Akita commitment coefficient count overflows"))
}

fn field_elem_bytes() -> usize {
    use akita_pcs::AkitaSerialize;
    AkitaField::zero().compressed_size()
}

/// Validates a deserialized proof shape against the resolved schedule before
/// the proof body is deserialized: the schedule-determined counts (fold-level
/// structure, `v`/next-commitment coefficient counts, sumcheck stage shapes)
/// must match exactly, the terminal witness must be admitted by the scheduled
/// witness shape (which bounds its Golomb `z` payload budgets), and the
/// remaining log-scale quantities are held to protocol bounds.
fn validate_proof_shape(
    shape: &AkitaBackendProofShape,
    schedule: &Schedule,
    layout: &OpeningClaimsLayout,
) -> Result<(), OpeningsError> {
    let fold_steps: Vec<&FoldStep> = schedule.fold_steps().collect();
    match shape {
        AkitaBackendProofShape::ZeroFold { witness_shapes } => {
            if !schedule_is_root_direct(schedule) {
                return Err(invalid_batch(
                    "Akita proof shape is zero-fold but the schedule is not root-direct",
                ));
            }
            let expected_witnesses = layout.num_total_polynomials();
            if witness_shapes.len() != expected_witnesses {
                return Err(invalid_batch(format!(
                    "Akita zero-fold proof shape has {} witnesses but the statement has {expected_witnesses}",
                    witness_shapes.len()
                )));
            }
            let scheduled = schedule_terminal_direct_witness_shape(schedule)
                .map_err(|err| invalid_batch(format!("Akita schedule error: {err}")))?;
            for realized in witness_shapes {
                validate_witness_shape(scheduled, realized)?;
            }
            Ok(())
        }
        AkitaBackendProofShape::Terminal(terminal) => {
            if fold_steps.len() != 1 {
                return Err(invalid_batch(format!(
                    "Akita proof shape is terminal-rooted but the schedule has {} fold levels",
                    fold_steps.len()
                )));
            }
            validate_terminal_level_shape(terminal, schedule)
        }
        AkitaBackendProofShape::Fold {
            root_shape,
            step_shapes,
        } => {
            if fold_steps.len() < 2 {
                return Err(invalid_batch(format!(
                    "Akita proof shape is fold-rooted but the schedule has {} fold levels",
                    fold_steps.len()
                )));
            }
            if step_shapes.len() != fold_steps.len() - 1 {
                return Err(invalid_batch(format!(
                    "Akita proof shape has {} recursive steps but the schedule requires {}",
                    step_shapes.len(),
                    fold_steps.len() - 1
                )));
            }
            validate_intermediate_level_shape(root_shape, fold_steps[0], fold_steps[1], layout)?;
            for (index, step_shape) in step_shapes.iter().enumerate() {
                let level = index + 1;
                let is_last = level == fold_steps.len() - 1;
                match (step_shape, is_last) {
                    (AkitaProofStepShape::Intermediate(level_shape), false) => {
                        validate_intermediate_level_shape(
                            level_shape,
                            fold_steps[level],
                            fold_steps[level + 1],
                            layout,
                        )?;
                    }
                    (AkitaProofStepShape::Terminal(terminal), true) => {
                        validate_terminal_level_shape(terminal, schedule)?;
                    }
                    (AkitaProofStepShape::Intermediate(_), true)
                    | (AkitaProofStepShape::Terminal(_), false) => {
                        return Err(invalid_batch(format!(
                            "Akita proof step {level} kind does not match the schedule position"
                        )));
                    }
                }
            }
            Ok(())
        }
    }
}

fn validate_intermediate_level_shape(
    shape: &LevelProofShape,
    step: &FoldStep,
    next_step: &FoldStep,
    layout: &OpeningClaimsLayout,
) -> Result<(), OpeningsError> {
    validate_ext_reduction_shape(shape.extension_opening_reduction.as_ref())?;

    let expected_v_coeffs = step
        .params
        .d_key
        .row_len()
        .checked_mul(step.params.ring_dimension)
        .ok_or_else(|| invalid_batch("Akita v coefficient count overflows"))?;
    if shape.v_coeffs != expected_v_coeffs {
        return Err(invalid_batch(format!(
            "Akita level shape declares {} v coefficients but the schedule requires {expected_v_coeffs}",
            shape.v_coeffs
        )));
    }

    let expected_next_commit = next_step
        .params
        .b_key
        .row_len()
        .checked_mul(next_step.params.ring_dimension)
        .ok_or_else(|| invalid_batch("Akita next-commitment coefficient count overflows"))?;
    if shape.next_commit_coeffs != expected_next_commit {
        return Err(invalid_batch(format!(
            "Akita level shape declares {} next-commitment coefficients but the schedule requires {expected_next_commit}",
            shape.next_commit_coeffs
        )));
    }

    let rounds = sumcheck_rounds(step.params.ring_dimension, step.next_w_len);
    if shape.stage2_sumcheck_proof != vec![STAGE2_SUMCHECK_DEGREE; rounds] {
        return Err(invalid_batch(
            "Akita level shape stage-2 sumcheck does not match the scheduled rounds",
        ));
    }
    let expected_stage1 = stage1_tree_stage_shapes(rounds, 1usize << step.params.log_basis);
    if shape.stage1_stages != expected_stage1 {
        return Err(invalid_batch(
            "Akita level shape stage-1 tree does not match the scheduled stages",
        ));
    }
    if shape.stage3_sumcheck.is_some() {
        // Jolt always verifies with `SetupContributionMode::Direct`; a stage-3
        // setup-product payload only exists in recursive mode.
        return Err(invalid_batch(
            "Akita level shape carries a stage-3 payload in direct setup-contribution mode",
        ));
    }
    let _ = layout;
    Ok(())
}

fn validate_terminal_level_shape(
    shape: &TerminalLevelProofShape,
    schedule: &Schedule,
) -> Result<(), OpeningsError> {
    validate_ext_reduction_shape(shape.extension_opening_reduction.as_ref())?;
    validate_bounded_sumcheck_shape("terminal stage-2", &shape.stage2_sumcheck)?;
    let scheduled = schedule_terminal_direct_witness_shape(schedule)
        .map_err(|err| invalid_batch(format!("Akita schedule error: {err}")))?;
    validate_witness_shape(scheduled, &shape.final_witness)
}

fn validate_witness_shape(
    scheduled: &CleartextWitnessShape,
    realized: &CleartextWitnessShape,
) -> Result<(), OpeningsError> {
    if !scheduled.admits_realized(realized) {
        return Err(invalid_batch(
            "Akita terminal witness shape is not admitted by the scheduled witness shape",
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

    #[test]
    fn forged_commitment_coeff_len_rejects_before_deserialization() {
        let num_vars = 4;
        let point = point(num_vars);
        let mut commitment = dense_commitment(num_vars, 2);
        // A honest-shape claim would be a few thousand coefficients; forge the
        // upstream 2^25 cap with an empty byte buffer.
        commitment.backend_coeff_len = 1 << 25;
        let proof = AkitaBatchProof {
            statement_bridge: Vec::new(),
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
        let num_vars = 4;
        let point = point(num_vars);
        let layout = OpeningClaimsLayout::new(num_vars, 2).expect("layout");
        let mut commitment = dense_commitment(num_vars, 2);
        let schedule = resolve_schedule(&commitment, &layout, &point).expect("schedule");
        let expected = expected_commitment_coeff_len(&schedule, &layout).expect("coeff len");
        commitment.backend_coeff_len = expected;
        // Correct declared count, truncated byte buffer: the deserializer
        // would reserve `expected` coefficients before hitting EOF.
        commitment.serialized_backend_bytes = vec![0u8; field_elem_bytes()];
        let proof = AkitaBatchProof {
            statement_bridge: Vec::new(),
            serialized_akita_proof_shape: Vec::new(),
            serialized_akita_proof: Vec::new(),
        };
        let err = deserialize_checked_backend_payload(&commitment, &proof, 2, &point)
            .expect_err("truncated commitment bytes must be rejected");
        assert!(err.to_string().contains("bytes"), "unexpected error: {err}");
    }

    #[test]
    fn oversized_proof_shape_blob_rejects() {
        let num_vars = 4;
        let point = point(num_vars);
        let layout = OpeningClaimsLayout::new(num_vars, 2).expect("layout");
        let mut commitment = dense_commitment(num_vars, 2);
        let schedule = resolve_schedule(&commitment, &layout, &point).expect("schedule");
        let coeff_len = expected_commitment_coeff_len(&schedule, &layout).expect("coeff len");
        commitment.backend_coeff_len = coeff_len;
        commitment.serialized_backend_bytes = vec![0u8; coeff_len * field_elem_bytes()];
        let proof = AkitaBatchProof {
            statement_bridge: Vec::new(),
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
    fn forged_shape_counts_reject_against_schedule() {
        let num_vars = 4;
        let point = point(num_vars);
        let layout = OpeningClaimsLayout::new(num_vars, 2).expect("layout");
        let mut commitment = dense_commitment(num_vars, 2);
        let schedule = resolve_schedule(&commitment, &layout, &point).expect("schedule");
        let coeff_len = expected_commitment_coeff_len(&schedule, &layout).expect("coeff len");
        commitment.backend_coeff_len = coeff_len;
        commitment.serialized_backend_bytes = vec![0u8; coeff_len * field_elem_bytes()];

        // A structurally plausible shape whose big counts are forged to the
        // upstream cap; the schedule comparison must reject it without the
        // proof body ever being deserialized.
        let forged = AkitaBackendProofShape::ZeroFold {
            witness_shapes: vec![CleartextWitnessShape::FieldElements(1 << 25); 2],
        };
        let proof = AkitaBatchProof {
            statement_bridge: Vec::new(),
            serialized_akita_proof_shape: serialize_akita(&forged).expect("serialize shape"),
            serialized_akita_proof: Vec::new(),
        };
        let err = deserialize_checked_backend_payload(&commitment, &proof, 2, &point)
            .expect_err("forged shape counts must be rejected");
        assert!(
            err.to_string().contains("schedule") || err.to_string().contains("witness"),
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
