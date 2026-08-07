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
    sumcheck_rounds, CommittedGroupParams, DigitRangePlan, ExtensionOpeningReductionShape,
    FoldSchedule, LevelProofShape, NextWitnessBindingShape, OpeningClaimsLayout,
    RecursiveFoldParams, TerminalLevelProofShape,
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
    validate_proof_shape(&proof_shape, &schedule)?;
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
) -> Result<FoldSchedule, OpeningsError> {
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

/// Mirrors the backend verifier's suffix replay: the commitment rows are the
/// final group's B block of the root relation matrix, decoded at the B-role
/// ring dimension.
fn expected_commitment_coeff_len(
    schedule: &FoldSchedule,
    layout: &OpeningClaimsLayout,
) -> Result<usize, OpeningsError> {
    let root_params = &schedule.root.params.final_group.commitment;
    let rows = root_params
        .commitment_row_range(layout, 0)
        .map_err(|err| invalid_batch(format!("Akita schedule layout error: {err}")))?
        .len();
    rows.checked_mul(root_params.role_dims().d_b())
        .ok_or_else(|| invalid_batch("Akita commitment coefficient count overflows"))
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
    for (index, level_shape) in shape.recursive_folds.iter().enumerate() {
        let step = &schedule.recursive_folds[index];
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

    let expected_v_coeffs = params
        .open_commit_matrix
        .output_rank()
        .checked_mul(params.role_dims().d_d())
        .ok_or_else(|| invalid_batch("Akita v coefficient count overflows"))?;
    if shape.v_coeffs != expected_v_coeffs {
        return Err(invalid_batch(format!(
            "Akita level shape declares {} v coefficients but the schedule requires {expected_v_coeffs}",
            shape.v_coeffs
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
        .stage_shapes(rounds);
    if shape.stage1_stages != expected_stage1 {
        return Err(invalid_batch(
            "Akita level shape stage-1 tree does not match the scheduled stages",
        ));
    }

    match (successor, shape.next_witness_binding) {
        (Some(next), NextWitnessBindingShape::OuterCommitment { coeffs }) => {
            let expected_next_commit = next
                .witness
                .outer_commit_matrix
                .output_rank()
                .checked_mul(next.witness.role_dims().d_b())
                .ok_or_else(|| {
                    invalid_batch("Akita next-commitment coefficient count overflows")
                })?;
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
    #![expect(
        clippy::panic,
        reason = "tests destructure honest proof shapes and fail loudly on fixture drift"
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

    /// The honest proof shape the schedule prescribes, built from the same
    /// step params the validator checks against.
    fn scheduled_proof_shape(schedule: &FoldSchedule) -> AkitaBackendProofShape {
        let level = |params: &CommittedGroupParams,
                     output_witness_len: usize,
                     successor: Option<&RecursiveFoldParams>| {
            let rounds = sumcheck_rounds(params.d_a(), output_witness_len);
            LevelProofShape {
                extension_opening_reduction: None,
                v_coeffs: params.open_commit_matrix.output_rank() * params.role_dims().d_d(),
                stage1_stages: DigitRangePlan::new(1usize << params.log_basis_open)
                    .expect("scheduled range basis")
                    .stage_shapes(rounds),
                stage2_sumcheck_proof: vec![STAGE2_SUMCHECK_DEGREE; rounds],
                stage3_sumcheck: None,
                next_witness_binding: match successor {
                    Some(next) => NextWitnessBindingShape::OuterCommitment {
                        coeffs: next.witness.outer_commit_matrix.output_rank()
                            * next.witness.role_dims().d_b(),
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
        let num_vars = 13;
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
        let num_vars = 13;
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
        let num_vars = 13;
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
    fn scheduled_shape_passes_validation() {
        let num_vars = 13;
        let point = point(num_vars);
        let layout = OpeningClaimsLayout::new(num_vars, 2).expect("layout");
        let commitment = dense_commitment(num_vars, 2);
        let schedule = resolve_schedule(&commitment, &layout, &point).expect("schedule");
        let shape = scheduled_proof_shape(&schedule);
        validate_proof_shape(&shape, &schedule).expect("scheduled shape must validate");
    }

    #[test]
    fn forged_shape_counts_reject_against_schedule() {
        let num_vars = 13;
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
        let mut forged = scheduled_proof_shape(&schedule);
        forged.root.v_coeffs = 1 << 25;
        let proof = AkitaBatchProof {
            statement_bridge: Vec::new(),
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
        let num_vars = 13;
        let point = point(num_vars);
        let layout = OpeningClaimsLayout::new(num_vars, 2).expect("layout");
        let commitment = dense_commitment(num_vars, 2);
        let schedule = resolve_schedule(&commitment, &layout, &point).expect("schedule");

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
            err.to_string().contains("must be 16 or 256"),
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

        let num_vars = 13;
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
        let schedule = resolve_schedule(&commitment, &layout, &point).expect("schedule");
        let realized =
            deserialize_akita::<AkitaBackendProofShape>(&proof.serialized_akita_proof_shape, &())
                .expect("honest proof shape should deserialize");
        validate_proof_shape(&realized, &schedule).expect("realized shape must validate");
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
        let num_vars = 13;
        let point = point(num_vars);
        let layout = OpeningClaimsLayout::new(num_vars, 2).expect("layout");
        let commitment = dense_commitment(num_vars, 2);
        let schedule = resolve_schedule(&commitment, &layout, &point).expect("schedule");
        assert!(
            !schedule.recursive_folds.is_empty(),
            "fixture needs at least one recursive fold"
        );
        let honest = scheduled_proof_shape(&schedule);
        validate_proof_shape(&honest, &schedule).expect("scheduled shape must validate");

        let mut extra_level = honest.clone();
        extra_level.recursive_folds.push(extra_level.root.clone());
        expect_shape_rejection(&extra_level, &schedule, "recursive fold levels");

        let mut truncated = honest.clone();
        let _ = truncated.recursive_folds.pop();
        expect_shape_rejection(&truncated, &schedule, "recursive fold levels");

        let mut forged_stage2 = honest.clone();
        forged_stage2
            .root
            .stage2_sumcheck_proof
            .push(STAGE2_SUMCHECK_DEGREE);
        expect_shape_rejection(&forged_stage2, &schedule, "stage-2");

        let mut forged_stage1 = honest.clone();
        let _ = forged_stage1.root.stage1_stages.pop();
        expect_shape_rejection(&forged_stage1, &schedule, "stage-1");

        let mut forged_stage3 = honest.clone();
        forged_stage3.root.stage3_sumcheck = Some(akita_types::SetupProductSumcheckShape {
            sumcheck: vec![STAGE2_SUMCHECK_DEGREE],
        });
        expect_shape_rejection(&forged_stage3, &schedule, "stage-3");

        // The root has a recursive successor, so its outgoing binding must be
        // an outer commitment with the successor's exact coefficient count.
        let NextWitnessBindingShape::OuterCommitment { coeffs } = honest.root.next_witness_binding
        else {
            panic!("a root with a successor must bind an outer commitment");
        };
        let mut forged_commit = honest.clone();
        forged_commit.root.next_witness_binding =
            NextWitnessBindingShape::OuterCommitment { coeffs: coeffs + 1 };
        expect_shape_rejection(&forged_commit, &schedule, "next-commitment");

        let mut forged_binding = honest.clone();
        forged_binding.root.next_witness_binding = NextWitnessBindingShape::TerminalInnerState;
        expect_shape_rejection(&forged_binding, &schedule, "witness binding");

        // The last recursive fold precedes the terminal, so an outer
        // commitment there contradicts the schedule position.
        let mut forged_tail = honest.clone();
        forged_tail
            .recursive_folds
            .last_mut()
            .expect("fixture has a recursive fold")
            .next_witness_binding = NextWitnessBindingShape::OuterCommitment { coeffs: 1 };
        expect_shape_rejection(&forged_tail, &schedule, "witness binding");

        let mut forged_partials = honest.clone();
        forged_partials.root.extension_opening_reduction = Some(ExtensionOpeningReductionShape {
            partials: MAX_EXT_REDUCTION_PARTIALS + 1,
            sumcheck: Vec::new(),
        });
        expect_shape_rejection(&forged_partials, &schedule, "partials");

        let mut forged_reduction_rounds = honest.clone();
        forged_reduction_rounds.terminal.extension_opening_reduction =
            Some(ExtensionOpeningReductionShape {
                partials: 1,
                sumcheck: vec![2; MAX_SUMCHECK_ROUNDS + 1],
            });
        expect_shape_rejection(&forged_reduction_rounds, &schedule, "protocol cap");
    }
}
