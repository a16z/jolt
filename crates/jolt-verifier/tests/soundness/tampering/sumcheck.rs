#![cfg_attr(
    all(feature = "core-fixtures", not(feature = "zk")),
    expect(
        clippy::panic,
        reason = "test fixtures should fail loudly when their assumed proof shape changes"
    )
)]

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use crate::support::{
    core_fixtures::{CorePrecompatVerifierCase, CoreVerifierCase, LegacyProofStageTarget},
    tamper_manifest,
};

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use jolt_claims::protocols::jolt::{
    formulas::{
        booleanity, bytecode,
        claim_reductions::{
            advice, hamming_weight, increments, instruction as instruction_claim_reduction,
            registers as registers_claim_reduction,
        },
        dimensions::JoltFormulaDimensions,
        instruction, ram, registers,
        spartan::{
            outer_opening, outer_uniskip_opening, product_outer_opening,
            product_remainder_output_openings, product_should_branch_outer_opening,
            product_should_jump_outer_opening, product_uniskip_opening, shift_output_openings,
            SpartanOuterDimensions,
        },
    },
    JoltAdviceKind, JoltCommittedPolynomial, JoltOpeningId, JoltPolynomialId, JoltRelationId,
    JoltVirtualPolynomial, PrecommittedReductionLayout,
};
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use jolt_field::{Fr, FromPrimitiveInt};
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use jolt_poly::{CompressedPoly, UnivariatePoly};
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use jolt_sumcheck::{ClearProof, SumcheckProof};
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use jolt_verifier::compat::claims::{offset_opening_claim, opening_claim, upsert_opening_claim};
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use jolt_verifier::stages::PrecommittedSchedule;

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage1_sumcheck_payload_reject() {
    let base = real_core_case();
    tamper_each_stage1_uniskip_round(&base);
    tamper_each_stage1_remainder_round(&base);
    tamper_stage1_round_counts(&base);
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage1_opening_claims_reject() {
    let base = real_core_case();

    for (target_name, id) in stage1_required_openings(&base) {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage2_uniskip_payload_reject() {
    let base = real_core_case();
    tamper_each_stage2_uniskip_round(&base);
    tamper_stage2_uniskip_round_counts(&base);
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage2_sumcheck_payload_reject() {
    let base = real_core_case();
    tamper_each_stage2_batch_round(&base);
    tamper_stage2_batch_round_counts(&base);
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage2_input_claims_reject() {
    let base = real_core_case();

    for id in stage2_uniskip_openings()
        .into_iter()
        .chain(stage2_batch_input_openings())
    {
        offset_claim_rejects(&base, id.0, id.1);
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage2_output_claims_reject() {
    let base = real_core_case();

    for (target_name, id) in stage2_formula_output_openings() {
        let replacement_claim = stage2_effective_output_claim(&base, id) + Fr::from_u64(1);
        tamper_manifest::assert_core_tamper_rejects(manifest_target(target_name), &base, |case| {
            upsert_opening_claim(&mut case.proof, id, replacement_claim);
        });
    }

    let [_, _, _, product_write_lookup_output_to_rd, _, _, _, product_virtual_instruction] =
        product_remainder_output_openings();
    for (target_name, id) in [
        (
            "stage2.claims.batch_outputs.product_remainder.write_lookup_output_to_rd",
            product_write_lookup_output_to_rd,
        ),
        (
            "stage2.claims.batch_outputs.product_remainder.virtual_instruction",
            product_virtual_instruction,
        ),
    ] {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage2_ram_phase_config_reject() {
    let base = real_core_case();

    tamper_manifest::assert_core_tamper_rejects(
        manifest_target("proof.rw_config"),
        &base,
        |case| {
            case.proof.rw_config.ram_rw_phase1_num_rounds =
                case.proof.trace_length.ilog2() as u8 + 1;
        },
    );

    tamper_manifest::assert_core_tamper_rejects(
        manifest_target("proof.rw_config"),
        &base,
        |case| {
            case.proof.rw_config.ram_rw_phase2_num_rounds = case.proof.ram_K.ilog2() as u8 + 1;
        },
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage3_sumcheck_payload_reject() {
    let base = real_core_case();
    tamper_each_stage3_batch_round(&base);
    tamper_stage3_batch_round_counts(&base);
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage3_output_claims_reject() {
    let base = real_core_case();

    for (target_name, id) in stage3_formula_output_openings() {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage4_sumcheck_payload_reject() {
    let base = real_core_case();
    tamper_each_stage4_batch_round(&base);
    tamper_stage4_batch_round_counts(&base);
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage4_output_claims_reject() {
    let base = real_core_case();

    for (target_name, id) in stage4_formula_output_openings() {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage4_advice_claims_reject() {
    let base = real_advice_case();

    for (target_name, id) in stage4_advice_openings() {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage5_sumcheck_payload_reject() {
    let base = real_core_case();
    tamper_each_stage5_batch_round(&base);
    tamper_stage5_batch_round_counts(&base);
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage5_output_claims_reject() {
    let base = real_core_case();

    for (target_name, id) in stage5_formula_output_openings(&base) {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage6_sumcheck_payload_reject() {
    let base = real_core_case();
    tamper_each_stage6_address_phase_round(&base);
    tamper_stage6_address_phase_round_counts(&base);
    tamper_each_stage6_cycle_phase_round(&base);
    tamper_stage6_cycle_phase_round_counts(&base);
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage6_output_claims_reject() {
    let base = real_core_case();

    for (target_name, id) in stage6_formula_output_openings(&base) {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage6_advice_claims_reject() {
    let base = real_advice_case();

    for (target_name, id) in stage6_advice_output_openings(&base) {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage7_sumcheck_payload_reject() {
    let base = real_core_case();
    tamper_each_stage7_batch_round(&base);
    tamper_stage7_batch_round_counts(&base);
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage7_output_claims_reject() {
    let base = real_core_case();

    for (target_name, id) in stage7_formula_output_openings(&base) {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage7_advice_claims_reject() {
    let base = real_advice_case();

    for (target_name, id) in stage7_advice_output_openings(&base) {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn precompat_tampered_stage1_sumcheck_payload_reject() {
    let base = real_precompat_core_case();
    precompat_tamper_each_round(
        &base,
        LegacyProofStageTarget::Stage1UniSkip,
        "stage1.uni_skip.round_polynomial",
    );
    precompat_tamper_round_counts(
        &base,
        LegacyProofStageTarget::Stage1UniSkip,
        "stage1.uni_skip.round_count.missing",
        "stage1.uni_skip.round_count.extra",
    );
    precompat_tamper_each_round(
        &base,
        LegacyProofStageTarget::Stage1Batch,
        "stage1.remainder.round_polynomial",
    );
    precompat_tamper_round_counts(
        &base,
        LegacyProofStageTarget::Stage1Batch,
        "stage1.remainder.round_count.missing",
        "stage1.remainder.round_count.extra",
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn precompat_tampered_stage1_opening_claims_reject() {
    let converted = real_core_case();
    let base = real_precompat_core_case();

    for (target_name, id) in stage1_required_openings(&converted) {
        precompat_offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn precompat_tampered_stage2_uniskip_payload_reject() {
    let base = real_precompat_core_case();
    precompat_tamper_each_round(
        &base,
        LegacyProofStageTarget::Stage2UniSkip,
        "stage2.product_uniskip.round_polynomial",
    );
    precompat_tamper_round_counts(
        &base,
        LegacyProofStageTarget::Stage2UniSkip,
        "stage2.product_uniskip.round_count.missing",
        "stage2.product_uniskip.round_count.extra",
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn precompat_tampered_stage2_sumcheck_payload_reject() {
    let base = real_precompat_core_case();
    precompat_tamper_each_round(
        &base,
        LegacyProofStageTarget::Stage2Batch,
        "stage2.batch.round_polynomial",
    );
    precompat_tamper_round_counts(
        &base,
        LegacyProofStageTarget::Stage2Batch,
        "stage2.batch.round_count.missing",
        "stage2.batch.round_count.extra",
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn precompat_tampered_stage2_input_claims_reject() {
    let base = real_precompat_core_case();

    for id in stage2_uniskip_openings()
        .into_iter()
        .chain(stage2_batch_input_openings())
    {
        precompat_offset_claim_rejects(&base, id.0, id.1);
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn precompat_tampered_stage2_output_claims_reject() {
    let converted = real_core_case();
    let base = real_precompat_core_case();

    for (target_name, id) in stage2_formula_output_openings() {
        precompat_offset_claim_rejects(&base, target_name, id);
    }

    let [_, _, _, product_write_lookup_output_to_rd, _, _, _, product_virtual_instruction] =
        product_remainder_output_openings();
    for (target_name, id) in [
        (
            "stage2.claims.batch_outputs.product_remainder.write_lookup_output_to_rd",
            product_write_lookup_output_to_rd,
        ),
        (
            "stage2.claims.batch_outputs.product_remainder.virtual_instruction",
            product_virtual_instruction,
        ),
    ] {
        if opening_claim(&converted.proof, id).is_some() {
            precompat_offset_claim_rejects(&base, target_name, id);
        }
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn precompat_tampered_stage3_sumcheck_payload_reject() {
    let base = real_precompat_core_case();
    precompat_tamper_each_round(
        &base,
        LegacyProofStageTarget::Stage3Batch,
        "stage3.batch.round_polynomial",
    );
    precompat_tamper_round_counts(
        &base,
        LegacyProofStageTarget::Stage3Batch,
        "stage3.batch.round_count.missing",
        "stage3.batch.round_count.extra",
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn precompat_tampered_stage3_output_claims_reject() {
    let base = real_precompat_core_case();

    for (target_name, id) in stage3_formula_output_openings() {
        precompat_offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn precompat_tampered_stage4_sumcheck_payload_reject() {
    let base = real_precompat_core_case();
    precompat_tamper_each_round(
        &base,
        LegacyProofStageTarget::Stage4Batch,
        "stage4.batch.round_polynomial",
    );
    precompat_tamper_round_counts(
        &base,
        LegacyProofStageTarget::Stage4Batch,
        "stage4.batch.round_count.missing",
        "stage4.batch.round_count.extra",
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn precompat_tampered_stage4_output_claims_reject() {
    let base = real_precompat_core_case();

    for (target_name, id) in stage4_formula_output_openings() {
        precompat_offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn precompat_tampered_stage4_advice_claims_reject() {
    let base = real_precompat_advice_case();

    for (target_name, id) in stage4_advice_openings() {
        precompat_offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn precompat_tampered_stage5_sumcheck_payload_reject() {
    let base = real_precompat_core_case();
    precompat_tamper_each_round(
        &base,
        LegacyProofStageTarget::Stage5Batch,
        "stage5.batch.round_polynomial",
    );
    precompat_tamper_round_counts(
        &base,
        LegacyProofStageTarget::Stage5Batch,
        "stage5.batch.round_count.missing",
        "stage5.batch.round_count.extra",
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn precompat_tampered_stage5_output_claims_reject() {
    let converted = real_core_case();
    let base = real_precompat_core_case();

    for (target_name, id) in stage5_formula_output_openings(&converted) {
        precompat_offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn precompat_tampered_stage6_sumcheck_payload_reject() {
    let base = real_precompat_core_case();
    precompat_tamper_each_round(
        &base,
        LegacyProofStageTarget::Stage6AddressPhase,
        "stage6.address_phase.round_polynomial",
    );
    precompat_tamper_round_counts(
        &base,
        LegacyProofStageTarget::Stage6AddressPhase,
        "stage6.address_phase.round_count.missing",
        "stage6.address_phase.round_count.extra",
    );
    precompat_tamper_each_round(
        &base,
        LegacyProofStageTarget::Stage6CyclePhase,
        "stage6.cycle_phase.round_polynomial",
    );
    precompat_tamper_round_counts(
        &base,
        LegacyProofStageTarget::Stage6CyclePhase,
        "stage6.cycle_phase.round_count.missing",
        "stage6.cycle_phase.round_count.extra",
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn precompat_tampered_stage6_output_claims_reject() {
    let converted = real_core_case();
    let base = real_precompat_core_case();

    for (target_name, id) in stage6_formula_output_openings(&converted) {
        precompat_offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn precompat_tampered_stage6_advice_claims_reject() {
    let converted = real_advice_case();
    let base = real_precompat_advice_case();

    for (target_name, id) in stage6_advice_output_openings(&converted) {
        precompat_offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn precompat_tampered_stage7_sumcheck_payload_reject() {
    let base = real_precompat_core_case();
    precompat_tamper_each_round(
        &base,
        LegacyProofStageTarget::Stage7Batch,
        "stage7.batch.round_polynomial",
    );
    precompat_tamper_round_counts(
        &base,
        LegacyProofStageTarget::Stage7Batch,
        "stage7.batch.round_count.missing",
        "stage7.batch.round_count.extra",
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn precompat_tampered_stage7_output_claims_reject() {
    let converted = real_core_case();
    let base = real_precompat_core_case();

    for (target_name, id) in stage7_formula_output_openings(&converted) {
        precompat_offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn precompat_tampered_stage7_advice_claims_reject() {
    let converted = real_advice_case();
    let base = real_precompat_advice_case();

    for (target_name, id) in stage7_advice_output_openings(&converted) {
        precompat_offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features core-fixtures in a non-ZK build to live-generate, cast, and tamper real core proofs"]
fn tampered_stage1_sumcheck_payload_reject() {}

#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features core-fixtures in a non-ZK build to live-generate, cast, and tamper real core proofs"]
fn tampered_stage2_uniskip_payload_reject() {}

#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features core-fixtures in a non-ZK build to live-generate, cast, and tamper real core proofs"]
fn tampered_stage2_sumcheck_payload_reject() {}

#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features core-fixtures in a non-ZK build to live-generate, cast, and tamper real core proofs"]
fn tampered_stage3_sumcheck_payload_reject() {}

#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features core-fixtures in a non-ZK build to live-generate, cast, and tamper real core proofs"]
fn tampered_stage4_sumcheck_payload_reject() {}

#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features core-fixtures in a non-ZK build to live-generate, cast, and tamper real core proofs"]
fn tampered_stage5_sumcheck_payload_reject() {}

#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features core-fixtures in a non-ZK build to live-generate, cast, and tamper real core proofs"]
fn tampered_stage6_sumcheck_payload_reject() {}

#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features core-fixtures in a non-ZK build to live-generate, cast, and tamper real core proofs"]
fn tampered_stage7_sumcheck_payload_reject() {}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn real_core_case() -> CoreVerifierCase {
    crate::support::core_fixtures::standard_muldiv_case()
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn real_advice_case() -> CoreVerifierCase {
    crate::support::core_fixtures::standard_advice_consumer_case()
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn real_precompat_core_case() -> CorePrecompatVerifierCase {
    crate::support::core_fixtures::standard_muldiv_precompat_case()
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn real_precompat_advice_case() -> CorePrecompatVerifierCase {
    crate::support::core_fixtures::standard_advice_consumer_precompat_case()
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tamper_each_stage1_uniskip_round(base: &CoreVerifierCase) {
    let round_count = clear_full_round_count(&base.proof.stages.stage1_uni_skip_first_round_proof);
    for round_index in 0..round_count {
        tamper_manifest::assert_core_tamper_rejects(
            manifest_target("stage1.uni_skip.round_polynomial"),
            base,
            |case| {
                mutate_full_round(
                    &mut case.proof.stages.stage1_uni_skip_first_round_proof,
                    round_index,
                );
            },
        );
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tamper_each_stage1_remainder_round(base: &CoreVerifierCase) {
    let round_count = compressed_round_count(&base.proof.stages.stage1_sumcheck_proof);
    for round_index in 0..round_count {
        tamper_manifest::assert_core_tamper_rejects(
            manifest_target("stage1.remainder.round_polynomial"),
            base,
            |case| {
                mutate_compressed_round(&mut case.proof.stages.stage1_sumcheck_proof, round_index);
            },
        );
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tamper_stage1_round_counts(base: &CoreVerifierCase) {
    tamper_manifest::assert_core_tamper_rejects(
        manifest_target("stage1.uni_skip.round_count.missing"),
        base,
        |case| {
            pop_full_round(&mut case.proof.stages.stage1_uni_skip_first_round_proof);
        },
    );

    tamper_manifest::assert_core_tamper_rejects(
        manifest_target("stage1.uni_skip.round_count.extra"),
        base,
        |case| {
            push_full_round(&mut case.proof.stages.stage1_uni_skip_first_round_proof);
        },
    );

    tamper_manifest::assert_core_tamper_rejects(
        manifest_target("stage1.remainder.round_count.missing"),
        base,
        |case| {
            pop_compressed_round(&mut case.proof.stages.stage1_sumcheck_proof);
        },
    );

    tamper_manifest::assert_core_tamper_rejects(
        manifest_target("stage1.remainder.round_count.extra"),
        base,
        |case| {
            push_compressed_round(&mut case.proof.stages.stage1_sumcheck_proof);
        },
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tamper_each_stage2_uniskip_round(base: &CoreVerifierCase) {
    let round_count = clear_full_round_count(&base.proof.stages.stage2_uni_skip_first_round_proof);
    for round_index in 0..round_count {
        tamper_manifest::assert_core_tamper_rejects(
            manifest_target("stage2.product_uniskip.round_polynomial"),
            base,
            |case| {
                mutate_full_round(
                    &mut case.proof.stages.stage2_uni_skip_first_round_proof,
                    round_index,
                );
            },
        );
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tamper_stage2_uniskip_round_counts(base: &CoreVerifierCase) {
    tamper_manifest::assert_core_tamper_rejects(
        manifest_target("stage2.product_uniskip.round_count.missing"),
        base,
        |case| {
            pop_full_round(&mut case.proof.stages.stage2_uni_skip_first_round_proof);
        },
    );

    tamper_manifest::assert_core_tamper_rejects(
        manifest_target("stage2.product_uniskip.round_count.extra"),
        base,
        |case| {
            push_full_round(&mut case.proof.stages.stage2_uni_skip_first_round_proof);
        },
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tamper_each_stage2_batch_round(base: &CoreVerifierCase) {
    let round_count = compressed_round_count(&base.proof.stages.stage2_sumcheck_proof);
    for round_index in 0..round_count {
        tamper_manifest::assert_core_tamper_rejects(
            manifest_target("stage2.batch.round_polynomial"),
            base,
            |case| {
                mutate_compressed_round(&mut case.proof.stages.stage2_sumcheck_proof, round_index);
            },
        );
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tamper_stage2_batch_round_counts(base: &CoreVerifierCase) {
    tamper_manifest::assert_core_tamper_rejects(
        manifest_target("stage2.batch.round_count.missing"),
        base,
        |case| {
            pop_compressed_round(&mut case.proof.stages.stage2_sumcheck_proof);
        },
    );

    tamper_manifest::assert_core_tamper_rejects(
        manifest_target("stage2.batch.round_count.extra"),
        base,
        |case| {
            push_compressed_round(&mut case.proof.stages.stage2_sumcheck_proof);
        },
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tamper_each_stage3_batch_round(base: &CoreVerifierCase) {
    let round_count = compressed_round_count(&base.proof.stages.stage3_sumcheck_proof);
    for round_index in 0..round_count {
        tamper_manifest::assert_core_tamper_rejects(
            manifest_target("stage3.batch.round_polynomial"),
            base,
            |case| {
                mutate_compressed_round(&mut case.proof.stages.stage3_sumcheck_proof, round_index);
            },
        );
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tamper_stage3_batch_round_counts(base: &CoreVerifierCase) {
    tamper_manifest::assert_core_tamper_rejects(
        manifest_target("stage3.batch.round_count.missing"),
        base,
        |case| {
            pop_compressed_round(&mut case.proof.stages.stage3_sumcheck_proof);
        },
    );

    tamper_manifest::assert_core_tamper_rejects(
        manifest_target("stage3.batch.round_count.extra"),
        base,
        |case| {
            push_compressed_round(&mut case.proof.stages.stage3_sumcheck_proof);
        },
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tamper_each_stage4_batch_round(base: &CoreVerifierCase) {
    let round_count = compressed_round_count(&base.proof.stages.stage4_sumcheck_proof);
    for round_index in 0..round_count {
        tamper_manifest::assert_core_tamper_rejects(
            manifest_target("stage4.batch.round_polynomial"),
            base,
            |case| {
                mutate_compressed_round(&mut case.proof.stages.stage4_sumcheck_proof, round_index);
            },
        );
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tamper_stage4_batch_round_counts(base: &CoreVerifierCase) {
    tamper_manifest::assert_core_tamper_rejects(
        manifest_target("stage4.batch.round_count.missing"),
        base,
        |case| {
            pop_compressed_round(&mut case.proof.stages.stage4_sumcheck_proof);
        },
    );

    tamper_manifest::assert_core_tamper_rejects(
        manifest_target("stage4.batch.round_count.extra"),
        base,
        |case| {
            push_compressed_round(&mut case.proof.stages.stage4_sumcheck_proof);
        },
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tamper_each_stage5_batch_round(base: &CoreVerifierCase) {
    let round_count = compressed_round_count(&base.proof.stages.stage5_sumcheck_proof);
    for round_index in 0..round_count {
        tamper_manifest::assert_core_tamper_rejects(
            manifest_target("stage5.batch.round_polynomial"),
            base,
            |case| {
                mutate_compressed_round(&mut case.proof.stages.stage5_sumcheck_proof, round_index);
            },
        );
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tamper_stage5_batch_round_counts(base: &CoreVerifierCase) {
    tamper_manifest::assert_core_tamper_rejects(
        manifest_target("stage5.batch.round_count.missing"),
        base,
        |case| {
            pop_compressed_round(&mut case.proof.stages.stage5_sumcheck_proof);
        },
    );

    tamper_manifest::assert_core_tamper_rejects(
        manifest_target("stage5.batch.round_count.extra"),
        base,
        |case| {
            push_compressed_round(&mut case.proof.stages.stage5_sumcheck_proof);
        },
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tamper_each_stage6_address_phase_round(base: &CoreVerifierCase) {
    let round_count = compressed_round_count(&base.proof.stages.stage6a_sumcheck_proof);
    for round_index in 0..round_count {
        tamper_manifest::assert_core_tamper_rejects(
            manifest_target("stage6.address_phase.round_polynomial"),
            base,
            |case| {
                mutate_compressed_round(&mut case.proof.stages.stage6a_sumcheck_proof, round_index);
            },
        );
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tamper_stage6_address_phase_round_counts(base: &CoreVerifierCase) {
    tamper_manifest::assert_core_tamper_rejects(
        manifest_target("stage6.address_phase.round_count.missing"),
        base,
        |case| {
            pop_compressed_round(&mut case.proof.stages.stage6a_sumcheck_proof);
        },
    );

    tamper_manifest::assert_core_tamper_rejects(
        manifest_target("stage6.address_phase.round_count.extra"),
        base,
        |case| {
            push_compressed_round(&mut case.proof.stages.stage6a_sumcheck_proof);
        },
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tamper_each_stage6_cycle_phase_round(base: &CoreVerifierCase) {
    let round_count = compressed_round_count(&base.proof.stages.stage6b_sumcheck_proof);
    for round_index in 0..round_count {
        tamper_manifest::assert_core_tamper_rejects(
            manifest_target("stage6.cycle_phase.round_polynomial"),
            base,
            |case| {
                mutate_compressed_round(&mut case.proof.stages.stage6b_sumcheck_proof, round_index);
            },
        );
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tamper_stage6_cycle_phase_round_counts(base: &CoreVerifierCase) {
    tamper_manifest::assert_core_tamper_rejects(
        manifest_target("stage6.cycle_phase.round_count.missing"),
        base,
        |case| {
            pop_compressed_round(&mut case.proof.stages.stage6b_sumcheck_proof);
        },
    );

    tamper_manifest::assert_core_tamper_rejects(
        manifest_target("stage6.cycle_phase.round_count.extra"),
        base,
        |case| {
            push_compressed_round(&mut case.proof.stages.stage6b_sumcheck_proof);
        },
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tamper_each_stage7_batch_round(base: &CoreVerifierCase) {
    let round_count = compressed_round_count(&base.proof.stages.stage7_sumcheck_proof);
    for round_index in 0..round_count {
        tamper_manifest::assert_core_tamper_rejects(
            manifest_target("stage7.batch.round_polynomial"),
            base,
            |case| {
                mutate_compressed_round(&mut case.proof.stages.stage7_sumcheck_proof, round_index);
            },
        );
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tamper_stage7_batch_round_counts(base: &CoreVerifierCase) {
    tamper_manifest::assert_core_tamper_rejects(
        manifest_target("stage7.batch.round_count.missing"),
        base,
        |case| {
            pop_compressed_round(&mut case.proof.stages.stage7_sumcheck_proof);
        },
    );

    tamper_manifest::assert_core_tamper_rejects(
        manifest_target("stage7.batch.round_count.extra"),
        base,
        |case| {
            push_compressed_round(&mut case.proof.stages.stage7_sumcheck_proof);
        },
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn clear_full_round_count(proof: &SumcheckProof<Fr, jolt_crypto::Bn254G1>) -> usize {
    let SumcheckProof::Clear(ClearProof::Full(proof)) = proof else {
        panic!("converted core fixture must use a clear full uni-skip proof");
    };
    proof.round_polynomials.len()
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn compressed_round_count(proof: &SumcheckProof<Fr, jolt_crypto::Bn254G1>) -> usize {
    let SumcheckProof::Clear(ClearProof::Compressed(proof)) = proof else {
        panic!("converted core fixture must use a clear compressed sumcheck proof");
    };
    proof.round_polynomials.len()
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn mutate_full_round(proof: &mut SumcheckProof<Fr, jolt_crypto::Bn254G1>, round_index: usize) {
    let SumcheckProof::Clear(ClearProof::Full(proof)) = proof else {
        panic!("converted core fixture must use a clear full uni-skip proof");
    };
    let Some(round) = proof.round_polynomials.get_mut(round_index) else {
        panic!("converted core fixture is missing expected uni-skip round {round_index}");
    };
    *round = UnivariatePoly::new(vec![Fr::from_u64(round_index as u64 + 1)]);
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn mutate_compressed_round(
    proof: &mut SumcheckProof<Fr, jolt_crypto::Bn254G1>,
    round_index: usize,
) {
    let SumcheckProof::Clear(ClearProof::Compressed(proof)) = proof else {
        panic!("converted core fixture must use a clear compressed sumcheck proof");
    };
    let Some(round) = proof.round_polynomials.get_mut(round_index) else {
        panic!("converted core fixture is missing expected compressed round {round_index}");
    };
    *round = CompressedPoly::new(vec![Fr::from_u64(round_index as u64 + 1)]);
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn pop_full_round(proof: &mut SumcheckProof<Fr, jolt_crypto::Bn254G1>) {
    let SumcheckProof::Clear(ClearProof::Full(proof)) = proof else {
        panic!("converted core fixture must use a clear full uni-skip proof");
    };
    let removed = proof.round_polynomials.pop();
    assert!(
        removed.is_some(),
        "converted core fixture has no full round to remove"
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn push_full_round(proof: &mut SumcheckProof<Fr, jolt_crypto::Bn254G1>) {
    let SumcheckProof::Clear(ClearProof::Full(proof)) = proof else {
        panic!("converted core fixture must use a clear full uni-skip proof");
    };
    proof
        .round_polynomials
        .push(UnivariatePoly::new(vec![Fr::from_u64(1)]));
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn pop_compressed_round(proof: &mut SumcheckProof<Fr, jolt_crypto::Bn254G1>) {
    let SumcheckProof::Clear(ClearProof::Compressed(proof)) = proof else {
        panic!("converted core fixture must use a clear compressed sumcheck proof");
    };
    let removed = proof.round_polynomials.pop();
    assert!(
        removed.is_some(),
        "converted core fixture has no compressed round to remove"
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn push_compressed_round(proof: &mut SumcheckProof<Fr, jolt_crypto::Bn254G1>) {
    let SumcheckProof::Clear(ClearProof::Compressed(proof)) = proof else {
        panic!("converted core fixture must use a clear compressed sumcheck proof");
    };
    proof
        .round_polynomials
        .push(CompressedPoly::new(vec![Fr::from_u64(1)]));
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn offset_claim_rejects(base: &CoreVerifierCase, target_name: &str, id: JoltOpeningId) {
    tamper_manifest::assert_core_tamper_rejects(manifest_target(target_name), base, |case| {
        assert!(
            offset_opening_claim(&mut case.proof, id, Fr::from_u64(1)),
            "converted core fixture is missing opening claim {id:?}"
        );
    });
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn precompat_tamper_each_round(
    base: &CorePrecompatVerifierCase,
    stage: LegacyProofStageTarget,
    target_name: &str,
) {
    let round_count = base.round_count(stage);
    for round_index in 0..round_count {
        tamper_manifest::assert_precompat_core_tamper_rejects(
            manifest_target(target_name),
            base,
            |case| case.replace_round(stage, round_index),
        );
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn precompat_tamper_round_counts(
    base: &CorePrecompatVerifierCase,
    stage: LegacyProofStageTarget,
    missing_target_name: &str,
    extra_target_name: &str,
) {
    tamper_manifest::assert_precompat_core_tamper_rejects(
        manifest_target(missing_target_name),
        base,
        |case| case.pop_round(stage),
    );

    tamper_manifest::assert_precompat_core_tamper_rejects(
        manifest_target(extra_target_name),
        base,
        |case| case.push_round(stage),
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn precompat_offset_claim_rejects(
    base: &CorePrecompatVerifierCase,
    target_name: &str,
    id: JoltOpeningId,
) {
    tamper_manifest::assert_precompat_core_tamper_rejects(
        manifest_target(target_name),
        base,
        |case| {
            if case.offset_opening_claim(id, 1) {
                return;
            }
            if let Some(alias) = precompat_opening_alias(id) {
                if case.offset_opening_claim(alias, 1) {
                    return;
                }
            }
            panic!("legacy core fixture is missing opening claim {id:?}");
        },
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn stage1_required_openings(base: &CoreVerifierCase) -> Vec<(&'static str, JoltOpeningId)> {
    let log_t = base.proof.trace_length.ilog2() as usize;
    let dimensions = SpartanOuterDimensions::rv64(log_t);
    let mut openings = Vec::with_capacity(dimensions.variables().len() + 1);
    openings.push((
        "stage1.claims.uniskip_output_claim",
        outer_uniskip_opening(),
    ));
    openings.extend(
        dimensions
            .variables()
            .iter()
            .copied()
            .map(outer_opening)
            .map(|id| ("stage1.claims.outer", id)),
    );
    openings
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn stage2_uniskip_openings() -> Vec<(&'static str, JoltOpeningId)> {
    vec![
        ("stage1.claims.outer", product_outer_opening()),
        ("stage1.claims.outer", product_should_branch_outer_opening()),
        ("stage1.claims.outer", product_should_jump_outer_opening()),
        (
            "stage2.claims.product_uniskip_output_claim",
            product_uniskip_opening(),
        ),
    ]
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn stage2_batch_input_openings() -> Vec<(&'static str, JoltOpeningId)> {
    vec![
        (
            "stage1.claims.outer",
            outer_virtual(JoltVirtualPolynomial::RamReadValue),
        ),
        (
            "stage1.claims.outer",
            outer_virtual(JoltVirtualPolynomial::RamWriteValue),
        ),
        (
            "stage2.claims.product_uniskip_output_claim",
            product_uniskip_opening(),
        ),
        (
            "stage1.claims.outer",
            outer_virtual(JoltVirtualPolynomial::LookupOutput),
        ),
        (
            "stage1.claims.outer",
            outer_virtual(JoltVirtualPolynomial::LeftLookupOperand),
        ),
        (
            "stage1.claims.outer",
            outer_virtual(JoltVirtualPolynomial::RightLookupOperand),
        ),
        (
            "stage1.claims.outer",
            outer_virtual(JoltVirtualPolynomial::LeftInstructionInput),
        ),
        (
            "stage1.claims.outer",
            outer_virtual(JoltVirtualPolynomial::RightInstructionInput),
        ),
        (
            "stage1.claims.outer",
            outer_virtual(JoltVirtualPolynomial::RamAddress),
        ),
    ]
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn stage2_formula_output_openings() -> Vec<(&'static str, JoltOpeningId)> {
    let [product_left_instruction_input, product_right_instruction_input, product_jump, _product_write_lookup_output_to_rd, product_lookup_output, product_branch, product_next_is_noop, _product_virtual_instruction] =
        product_remainder_output_openings();

    let mut openings = Vec::new();
    openings.extend(
        ram::read_write_checking_output_openings()
            .into_iter()
            .map(|id| ("stage2.claims.batch_outputs.ram_read_write", id)),
    );
    openings.extend([
        (
            "stage2.claims.batch_outputs.product_remainder.checked",
            product_left_instruction_input,
        ),
        (
            "stage2.claims.batch_outputs.product_remainder.checked",
            product_right_instruction_input,
        ),
        (
            "stage2.claims.batch_outputs.product_remainder.checked",
            product_jump,
        ),
        (
            "stage2.claims.batch_outputs.product_remainder.checked",
            product_lookup_output,
        ),
        (
            "stage2.claims.batch_outputs.product_remainder.checked",
            product_branch,
        ),
        (
            "stage2.claims.batch_outputs.product_remainder.checked",
            product_next_is_noop,
        ),
    ]);
    openings.extend(
        instruction_claim_reduction::claim_reduction_output_openings()
            .into_iter()
            .map(|id| {
                (
                    "stage2.claims.batch_outputs.instruction_claim_reduction",
                    id,
                )
            }),
    );
    openings.extend(
        ram::raf_evaluation_output_openings()
            .into_iter()
            .map(|id| ("stage2.claims.batch_outputs.ram_raf_evaluation", id)),
    );
    openings.extend(
        ram::output_check_output_openings()
            .into_iter()
            .map(|id| ("stage2.claims.batch_outputs.ram_output_check", id)),
    );
    openings
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn stage3_formula_output_openings() -> Vec<(&'static str, JoltOpeningId)> {
    let mut openings = Vec::new();
    openings.extend(
        shift_output_openings()
            .into_iter()
            .map(|id| ("stage3.claims.shift", id)),
    );
    openings.extend(
        instruction::input_virtualization_output_openings()
            .into_iter()
            .map(|id| ("stage3.claims.instruction_input", id)),
    );
    openings.extend(
        registers_claim_reduction::claim_reduction_output_openings()
            .into_iter()
            .map(|id| ("stage3.claims.registers_claim_reduction", id)),
    );
    openings
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn stage4_formula_output_openings() -> Vec<(&'static str, JoltOpeningId)> {
    let mut openings = Vec::new();
    openings.extend(
        registers::read_write_checking_output_openings()
            .into_iter()
            .map(|id| ("stage4.claims.registers_read_write", id)),
    );
    openings.extend(
        ram::val_check_output_openings()
            .into_iter()
            .map(|id| ("stage4.claims.ram_val_check", id)),
    );
    openings
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn stage4_advice_openings() -> Vec<(&'static str, JoltOpeningId)> {
    vec![
        (
            "stage4.claims.advice.untrusted",
            ram::val_check_advice_opening(JoltAdviceKind::Untrusted),
        ),
        (
            "stage4.claims.advice.trusted",
            ram::val_check_advice_opening(JoltAdviceKind::Trusted),
        ),
    ]
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn stage5_formula_output_openings(base: &CoreVerifierCase) -> Vec<(&'static str, JoltOpeningId)> {
    let mut openings = Vec::new();
    openings.extend(LookupTableKind::<RISCV_XLEN>::iter().map(|table| {
        (
            "stage5.claims.instruction_read_raf.lookup_table_flags",
            instruction::read_raf_lookup_table_flag_opening(table),
        )
    }));
    for index in 0.. {
        let id = instruction::read_raf_instruction_ra_opening(index);
        if opening_claim(&base.proof, id).is_none() {
            break;
        }
        openings.push(("stage5.claims.instruction_read_raf.instruction_ra", id));
    }
    openings.push((
        "stage5.claims.instruction_read_raf.instruction_raf_flag",
        instruction::read_raf_instruction_raf_flag_opening(),
    ));
    openings.extend(
        ram::ra_claim_reduction_output_openings()
            .into_iter()
            .map(|id| ("stage5.claims.ram_ra_claim_reduction", id)),
    );
    openings.extend(
        registers::val_evaluation_output_openings()
            .into_iter()
            .map(|id| ("stage5.claims.registers_val_evaluation", id)),
    );
    openings
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn stage6_formula_output_openings(base: &CoreVerifierCase) -> Vec<(&'static str, JoltOpeningId)> {
    let dimensions = stage6_dimensions(base);
    let mut openings = Vec::new();

    openings.extend([
        (
            "stage6.claims.address_phase.bytecode_read_raf",
            bytecode::bytecode_read_raf_address_phase_opening(),
        ),
        (
            "stage6.claims.address_phase.booleanity",
            booleanity::booleanity_address_phase_opening(),
        ),
    ]);
    openings.extend(
        bytecode::read_raf_output_openings(dimensions.bytecode_read_raf)
            .bytecode_ra
            .into_iter()
            .map(|id| ("stage6.claims.bytecode_read_raf.bytecode_ra", id)),
    );
    for index in 0..dimensions.ra_layout.instruction() {
        openings.push((
            "stage6.claims.booleanity.instruction_ra",
            JoltOpeningId::committed(
                JoltCommittedPolynomial::InstructionRa(index),
                JoltRelationId::Booleanity,
            ),
        ));
    }
    for index in 0..dimensions.ra_layout.bytecode() {
        openings.push((
            "stage6.claims.booleanity.bytecode_ra",
            JoltOpeningId::committed(
                JoltCommittedPolynomial::BytecodeRa(index),
                JoltRelationId::Booleanity,
            ),
        ));
    }
    for index in 0..dimensions.ra_layout.ram() {
        openings.push((
            "stage6.claims.booleanity.ram_ra",
            JoltOpeningId::committed(
                JoltCommittedPolynomial::RamRa(index),
                JoltRelationId::Booleanity,
            ),
        ));
    }
    openings.extend(
        ram::hamming_booleanity_output_openings()
            .into_iter()
            .map(|id| {
                (
                    "stage6.claims.ram_hamming_booleanity.ram_hamming_weight",
                    id,
                )
            }),
    );
    openings.extend(
        ram::ra_virtualization_output_openings(dimensions.ram_ra_virtualization)
            .into_iter()
            .map(|id| ("stage6.claims.ram_ra_virtualization.ram_ra", id)),
    );
    openings.extend(
        instruction::ra_virtualization_output_openings(dimensions.instruction_ra_virtualization)
            .all()
            .into_iter()
            .map(|id| {
                (
                    "stage6.claims.instruction_ra_virtualization.committed_instruction_ra",
                    id,
                )
            }),
    );
    let [ram_inc, rd_inc] = increments::claim_reduction_output_openings();
    openings.extend([
        ("stage6.claims.inc_claim_reduction.ram_inc", ram_inc),
        ("stage6.claims.inc_claim_reduction.rd_inc", rd_inc),
    ]);

    openings
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn case_advice_layouts(base: &CoreVerifierCase) -> PrecommittedSchedule {
    PrecommittedSchedule::new(
        base.proof.trace_polynomial_order,
        base.proof.trace_length.ilog2() as usize,
        base.proof.one_hot_config.committed_chunk_bits(),
        base.trusted_advice_commitment
            .is_some()
            .then_some(base.public_io.memory_layout.max_trusted_advice_size as usize),
        base.proof
            .untrusted_advice_commitment
            .is_some()
            .then_some(base.public_io.memory_layout.max_untrusted_advice_size as usize),
        None,
    )
    .unwrap_or_else(|error| panic!("precommitted schedule should build: {error}"))
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn stage6_advice_output_openings(base: &CoreVerifierCase) -> Vec<(&'static str, JoltOpeningId)> {
    let schedule = case_advice_layouts(base);
    let (trusted_layout, untrusted_layout) = (schedule.trusted_advice, schedule.untrusted_advice);
    let mut openings = Vec::new();

    if let Some(layout) = trusted_layout {
        openings.extend(
            advice::cycle_phase_output_openings(JoltAdviceKind::Trusted, layout.dimensions())
                .into_iter()
                .map(|id| ("stage6.claims.advice_cycle_phase.trusted.opening_claim", id)),
        );
    }
    if let Some(layout) = untrusted_layout {
        openings.extend(
            advice::cycle_phase_output_openings(JoltAdviceKind::Untrusted, layout.dimensions())
                .into_iter()
                .map(|id| {
                    (
                        "stage6.claims.advice_cycle_phase.untrusted.opening_claim",
                        id,
                    )
                }),
        );
    }

    openings
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn stage7_formula_output_openings(base: &CoreVerifierCase) -> Vec<(&'static str, JoltOpeningId)> {
    let dimensions = stage6_dimensions(base);
    let output_openings = hamming_weight::claim_reduction_output_openings(
        hamming_weight::HammingWeightClaimReductionDimensions::new(
            dimensions.ra_layout,
            base.proof.one_hot_config.committed_chunk_bits(),
        ),
    );
    let mut openings = Vec::new();
    openings.extend(output_openings.instruction_ra.into_iter().map(|id| {
        (
            "stage7.claims.hamming_weight_claim_reduction.instruction_ra",
            id,
        )
    }));
    openings.extend(output_openings.bytecode_ra.into_iter().map(|id| {
        (
            "stage7.claims.hamming_weight_claim_reduction.bytecode_ra",
            id,
        )
    }));
    openings.extend(
        output_openings
            .ram_ra
            .into_iter()
            .map(|id| ("stage7.claims.hamming_weight_claim_reduction.ram_ra", id)),
    );
    openings
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn stage7_advice_output_openings(base: &CoreVerifierCase) -> Vec<(&'static str, JoltOpeningId)> {
    let schedule = case_advice_layouts(base);
    let (trusted_layout, untrusted_layout) = (schedule.trusted_advice, schedule.untrusted_advice);
    let mut openings = Vec::new();

    if trusted_layout.is_some_and(|layout| layout.dimensions().has_address_phase()) {
        openings.push((
            "stage7.claims.advice_address_phase.trusted.opening_claim",
            advice::final_advice_opening(JoltAdviceKind::Trusted),
        ));
    }
    if untrusted_layout.is_some_and(|layout| layout.dimensions().has_address_phase()) {
        openings.push((
            "stage7.claims.advice_address_phase.untrusted.opening_claim",
            advice::final_advice_opening(JoltAdviceKind::Untrusted),
        ));
    }

    openings
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn stage6_dimensions(base: &CoreVerifierCase) -> JoltFormulaDimensions {
    let log_t = base.proof.trace_length.ilog2() as usize;
    JoltFormulaDimensions::try_from(base.proof.one_hot_config.dimensions(
        log_t,
        2 * RISCV_XLEN,
        base.preprocessing.program.bytecode_len(),
        base.proof.ram_K,
    ))
    .unwrap_or_else(|error| panic!("core fixture has invalid Stage 6 dimensions: {error}"))
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn manifest_target(name: &str) -> tamper_manifest::TamperTarget {
    tamper_manifest::required_target(name)
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn outer_virtual(polynomial: JoltVirtualPolynomial) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(polynomial, JoltRelationId::SpartanOuter)
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn stage2_effective_output_claim(base: &CoreVerifierCase, id: JoltOpeningId) -> Fr {
    opening_claim(&base.proof, id)
        .or_else(|| stage2_output_alias_claim(base, id))
        .unwrap_or_else(|| Fr::from_u64(0))
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn stage2_output_alias_claim(base: &CoreVerifierCase, id: JoltOpeningId) -> Option<Fr> {
    let alias = stage2_output_alias(id)?;
    opening_claim(&base.proof, alias)
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn precompat_opening_alias(id: JoltOpeningId) -> Option<JoltOpeningId> {
    stage2_output_alias(id)
        .or_else(|| stage3_output_alias(id))
        .or_else(|| stage6_output_alias(id))
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn stage2_output_alias(id: JoltOpeningId) -> Option<JoltOpeningId> {
    let [product_left_instruction_input, product_right_instruction_input, _product_jump, _product_write_lookup_output_to_rd, product_lookup_output, _product_branch, _product_next_is_noop, _product_virtual_instruction] =
        product_remainder_output_openings();
    let [instruction_lookup_output, _instruction_left_lookup_operand, _instruction_right_lookup_operand, instruction_left_instruction_input, instruction_right_instruction_input] =
        instruction_claim_reduction::claim_reduction_output_openings();

    if id == instruction_lookup_output {
        Some(product_lookup_output)
    } else if id == instruction_left_instruction_input {
        Some(product_left_instruction_input)
    } else if id == instruction_right_instruction_input {
        Some(product_right_instruction_input)
    } else {
        None
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn stage3_output_alias(id: JoltOpeningId) -> Option<JoltOpeningId> {
    let [unexpanded_pc_shift, _pc_shift, _is_virtual_shift, _is_first_in_sequence_shift, _is_noop_shift] =
        shift_output_openings();
    let [_right_operand_is_rs2, rs2_value_input, _right_operand_is_imm, _imm_input, _left_operand_is_rs1, rs1_value_input, _left_operand_is_pc, unexpanded_pc_input] =
        instruction::input_virtualization_output_openings();
    let [_rd_write_value_reduced, rs1_value_reduced, rs2_value_reduced] =
        registers_claim_reduction::claim_reduction_output_openings();

    if id == unexpanded_pc_input {
        Some(unexpanded_pc_shift)
    } else if id == rs1_value_reduced {
        Some(rs1_value_input)
    } else if id == rs2_value_reduced {
        Some(rs2_value_input)
    } else {
        None
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn stage6_output_alias(id: JoltOpeningId) -> Option<JoltOpeningId> {
    match id {
        JoltOpeningId::Polynomial {
            polynomial: JoltPolynomialId::Committed(JoltCommittedPolynomial::BytecodeRa(index)),
            relation: JoltRelationId::Booleanity,
        } => Some(JoltOpeningId::committed(
            JoltCommittedPolynomial::BytecodeRa(index),
            JoltRelationId::BytecodeReadRaf,
        )),
        _ => None,
    }
}
