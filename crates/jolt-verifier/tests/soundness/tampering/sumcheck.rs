#![cfg_attr(
    all(feature = "prover-fixtures", not(feature = "zk")),
    expect(
        clippy::panic,
        reason = "test fixtures should fail loudly when their assumed proof shape changes"
    )
)]

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use crate::support::{
    tamper_manifest,
    verifier_fixtures::{standard_muldiv_case, VerifierFixtureCase},
};

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use crate::support::proof_claims::{offset_opening_claim, opening_claim, upsert_opening_claim};
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use jolt_claims::protocols::jolt::{
    geometry::{
        booleanity, bytecode,
        claim_reductions::{
            advice, hamming_weight, increments, instruction as instruction_claim_reduction,
            registers as registers_claim_reduction,
        },
        dimensions::JoltFormulaDimensions,
        instruction, ram, registers,
        spartan::{
            outer_opening, outer_uniskip_opening, product_uniskip_opening, SpartanOuterDimensions,
        },
    },
    JoltAdviceKind, JoltCommittedPolynomial, JoltOpeningId, JoltRelationId,
    PrecommittedReductionLayout,
};
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use jolt_claims::{protocols::jolt::geometry::spartan, protocols::jolt::relations, OutputClaims};
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use jolt_field::{Fr, FromPrimitiveInt};
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use jolt_poly::{CompressedPoly, UnivariatePoly};
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use jolt_sumcheck::{ClearProof, SumcheckProof};
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use jolt_verifier::stages::PrecommittedSchedule;
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use num_traits::Zero;

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage1_sumcheck_payload_reject() {
    let base = standard_muldiv_case();
    tamper_each_stage1_uniskip_round(&base);
    tamper_each_stage1_remainder_round(&base);
    tamper_stage1_round_counts(&base);
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage1_opening_claims_reject() {
    let base = standard_muldiv_case();

    for (target_name, id) in stage1_required_openings(&base) {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage2_uniskip_payload_reject() {
    let base = standard_muldiv_case();
    tamper_each_stage2_uniskip_round(&base);
    tamper_stage2_uniskip_round_counts(&base);
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage2_sumcheck_payload_reject() {
    let base = standard_muldiv_case();
    tamper_each_stage2_batch_round(&base);
    tamper_stage2_batch_round_counts(&base);
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage2_input_claims_reject() {
    let base = standard_muldiv_case();

    // The Spartan-outer input openings this stage consumes are already swept by
    // tampered_stage1_opening_claims_reject (every SPARTAN_OUTER_R1CS_INPUTS
    // variable). Only the product uni-skip output claim, which lives under the
    // SpartanProductVirtualization relation, is unique to this stage.
    offset_claim_rejects(
        &base,
        "stage2.claims.product_uniskip_output_claim",
        product_uniskip_opening(),
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage2_output_claims_reject() {
    let base = standard_muldiv_case();

    for (target_name, id) in stage2_formula_output_openings() {
        let replacement_claim = stage2_effective_output_claim(&base, id) + Fr::from_u64(1);
        tamper_manifest::assert_verifier_fixture_tamper_rejects(
            manifest_target(target_name),
            &base,
            |case| {
                upsert_opening_claim(&mut case.proof, id, replacement_claim);
            },
        );
    }

    for (target_name, id) in [
        (
            "stage2.claims.batch_outputs.product_remainder.write_lookup_output_to_rd",
            spartan::write_lookup_output_to_rd_product(),
        ),
        (
            "stage2.claims.batch_outputs.product_remainder.virtual_instruction",
            spartan::virtual_instruction_product(),
        ),
    ] {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage2_ram_phase_config_reject() {
    let base = standard_muldiv_case();

    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target("proof.rw_config"),
        &base,
        |case| {
            case.proof.rw_config.ram_rw_phase1_num_rounds =
                case.proof.trace_length.ilog2() as u8 + 1;
        },
    );

    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target("proof.rw_config"),
        &base,
        |case| {
            case.proof.rw_config.ram_rw_phase2_num_rounds = case.proof.ram_K.ilog2() as u8 + 1;
        },
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage3_sumcheck_payload_reject() {
    let base = standard_muldiv_case();
    tamper_each_stage3_batch_round(&base);
    tamper_stage3_batch_round_counts(&base);
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage3_output_claims_reject() {
    let base = standard_muldiv_case();

    for (target_name, id) in stage3_formula_output_openings() {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage4_sumcheck_payload_reject() {
    let base = standard_muldiv_case();
    tamper_each_stage4_batch_round(&base);
    tamper_stage4_batch_round_counts(&base);
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage4_output_claims_reject() {
    let base = standard_muldiv_case();

    for (target_name, id) in stage4_formula_output_openings() {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage4_advice_claims_reject() {
    let base = real_advice_case();

    for (target_name, id) in stage4_advice_openings() {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage5_sumcheck_payload_reject() {
    let base = standard_muldiv_case();
    tamper_each_stage5_batch_round(&base);
    tamper_stage5_batch_round_counts(&base);
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage5_output_claims_reject() {
    let base = standard_muldiv_case();

    for (target_name, id) in stage5_formula_output_openings(&base) {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage6_sumcheck_payload_reject() {
    let base = standard_muldiv_case();
    tamper_each_stage6_address_phase_round(&base);
    tamper_stage6_address_phase_round_counts(&base);
    tamper_each_stage6_cycle_phase_round(&base);
    tamper_stage6_cycle_phase_round_counts(&base);
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage6_output_claims_reject() {
    let base = standard_muldiv_case();

    for (target_name, id) in stage6_formula_output_openings(&base) {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage6_advice_claims_reject() {
    let base = real_advice_case();

    for (target_name, id) in stage6_advice_output_openings(&base) {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage7_sumcheck_payload_reject() {
    let base = standard_muldiv_case();
    tamper_each_stage7_batch_round(&base);
    tamper_stage7_batch_round_counts(&base);
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage7_output_claims_reject() {
    let base = standard_muldiv_case();

    for (target_name, id) in stage7_formula_output_openings(&base) {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage7_advice_claims_reject() {
    let base = real_advice_case();

    for (target_name, id) in stage7_advice_output_openings(&base) {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features prover-fixtures in a non-ZK build to live-generate and tamper verifier-native proofs"]
fn tampered_stage1_sumcheck_payload_reject() {}

#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features prover-fixtures in a non-ZK build to live-generate and tamper verifier-native proofs"]
fn tampered_stage2_uniskip_payload_reject() {}

#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features prover-fixtures in a non-ZK build to live-generate and tamper verifier-native proofs"]
fn tampered_stage2_sumcheck_payload_reject() {}

#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features prover-fixtures in a non-ZK build to live-generate and tamper verifier-native proofs"]
fn tampered_stage3_sumcheck_payload_reject() {}

#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features prover-fixtures in a non-ZK build to live-generate and tamper verifier-native proofs"]
fn tampered_stage4_sumcheck_payload_reject() {}

#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features prover-fixtures in a non-ZK build to live-generate and tamper verifier-native proofs"]
fn tampered_stage5_sumcheck_payload_reject() {}

#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features prover-fixtures in a non-ZK build to live-generate and tamper verifier-native proofs"]
fn tampered_stage6_sumcheck_payload_reject() {}

#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features prover-fixtures in a non-ZK build to live-generate and tamper verifier-native proofs"]
fn tampered_stage7_sumcheck_payload_reject() {}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn real_advice_case() -> VerifierFixtureCase {
    crate::support::verifier_fixtures::standard_advice_consumer_case()
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_each_stage1_uniskip_round(base: &VerifierFixtureCase) {
    let round_count = clear_full_round_count(&base.proof.stages.stage1_uni_skip_first_round_proof);
    for round_index in 0..round_count {
        tamper_manifest::assert_verifier_fixture_tamper_rejects(
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

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_each_stage1_remainder_round(base: &VerifierFixtureCase) {
    let round_count = compressed_round_count(&base.proof.stages.stage1_sumcheck_proof);
    for round_index in 0..round_count {
        tamper_manifest::assert_verifier_fixture_tamper_rejects(
            manifest_target("stage1.remainder.round_polynomial"),
            base,
            |case| {
                mutate_compressed_round(&mut case.proof.stages.stage1_sumcheck_proof, round_index);
            },
        );
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_stage1_round_counts(base: &VerifierFixtureCase) {
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target("stage1.uni_skip.round_count.missing"),
        base,
        |case| {
            pop_full_round(&mut case.proof.stages.stage1_uni_skip_first_round_proof);
        },
    );

    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target("stage1.uni_skip.round_count.extra"),
        base,
        |case| {
            push_full_round(&mut case.proof.stages.stage1_uni_skip_first_round_proof);
        },
    );

    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target("stage1.remainder.round_count.missing"),
        base,
        |case| {
            pop_compressed_round(&mut case.proof.stages.stage1_sumcheck_proof);
        },
    );

    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target("stage1.remainder.round_count.extra"),
        base,
        |case| {
            push_compressed_round(&mut case.proof.stages.stage1_sumcheck_proof);
        },
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_each_stage2_uniskip_round(base: &VerifierFixtureCase) {
    let round_count = clear_full_round_count(&base.proof.stages.stage2_uni_skip_first_round_proof);
    for round_index in 0..round_count {
        tamper_manifest::assert_verifier_fixture_tamper_rejects(
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

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_stage2_uniskip_round_counts(base: &VerifierFixtureCase) {
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target("stage2.product_uniskip.round_count.missing"),
        base,
        |case| {
            pop_full_round(&mut case.proof.stages.stage2_uni_skip_first_round_proof);
        },
    );

    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target("stage2.product_uniskip.round_count.extra"),
        base,
        |case| {
            push_full_round(&mut case.proof.stages.stage2_uni_skip_first_round_proof);
        },
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_each_stage2_batch_round(base: &VerifierFixtureCase) {
    let round_count = compressed_round_count(&base.proof.stages.stage2_sumcheck_proof);
    for round_index in 0..round_count {
        tamper_manifest::assert_verifier_fixture_tamper_rejects(
            manifest_target("stage2.batch.round_polynomial"),
            base,
            |case| {
                mutate_compressed_round(&mut case.proof.stages.stage2_sumcheck_proof, round_index);
            },
        );
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_stage2_batch_round_counts(base: &VerifierFixtureCase) {
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target("stage2.batch.round_count.missing"),
        base,
        |case| {
            pop_compressed_round(&mut case.proof.stages.stage2_sumcheck_proof);
        },
    );

    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target("stage2.batch.round_count.extra"),
        base,
        |case| {
            push_compressed_round(&mut case.proof.stages.stage2_sumcheck_proof);
        },
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_each_stage3_batch_round(base: &VerifierFixtureCase) {
    let round_count = compressed_round_count(&base.proof.stages.stage3_sumcheck_proof);
    for round_index in 0..round_count {
        tamper_manifest::assert_verifier_fixture_tamper_rejects(
            manifest_target("stage3.batch.round_polynomial"),
            base,
            |case| {
                mutate_compressed_round(&mut case.proof.stages.stage3_sumcheck_proof, round_index);
            },
        );
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_stage3_batch_round_counts(base: &VerifierFixtureCase) {
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target("stage3.batch.round_count.missing"),
        base,
        |case| {
            pop_compressed_round(&mut case.proof.stages.stage3_sumcheck_proof);
        },
    );

    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target("stage3.batch.round_count.extra"),
        base,
        |case| {
            push_compressed_round(&mut case.proof.stages.stage3_sumcheck_proof);
        },
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_each_stage4_batch_round(base: &VerifierFixtureCase) {
    let round_count = compressed_round_count(&base.proof.stages.stage4_sumcheck_proof);
    for round_index in 0..round_count {
        tamper_manifest::assert_verifier_fixture_tamper_rejects(
            manifest_target("stage4.batch.round_polynomial"),
            base,
            |case| {
                mutate_compressed_round(&mut case.proof.stages.stage4_sumcheck_proof, round_index);
            },
        );
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_stage4_batch_round_counts(base: &VerifierFixtureCase) {
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target("stage4.batch.round_count.missing"),
        base,
        |case| {
            pop_compressed_round(&mut case.proof.stages.stage4_sumcheck_proof);
        },
    );

    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target("stage4.batch.round_count.extra"),
        base,
        |case| {
            push_compressed_round(&mut case.proof.stages.stage4_sumcheck_proof);
        },
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_each_stage5_batch_round(base: &VerifierFixtureCase) {
    let round_count = compressed_round_count(&base.proof.stages.stage5_sumcheck_proof);
    for round_index in 0..round_count {
        tamper_manifest::assert_verifier_fixture_tamper_rejects(
            manifest_target("stage5.batch.round_polynomial"),
            base,
            |case| {
                mutate_compressed_round(&mut case.proof.stages.stage5_sumcheck_proof, round_index);
            },
        );
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_stage5_batch_round_counts(base: &VerifierFixtureCase) {
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target("stage5.batch.round_count.missing"),
        base,
        |case| {
            pop_compressed_round(&mut case.proof.stages.stage5_sumcheck_proof);
        },
    );

    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target("stage5.batch.round_count.extra"),
        base,
        |case| {
            push_compressed_round(&mut case.proof.stages.stage5_sumcheck_proof);
        },
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_each_stage6_address_phase_round(base: &VerifierFixtureCase) {
    let round_count = compressed_round_count(&base.proof.stages.stage6a_sumcheck_proof);
    for round_index in 0..round_count {
        tamper_manifest::assert_verifier_fixture_tamper_rejects(
            manifest_target("stage6.address_phase.round_polynomial"),
            base,
            |case| {
                mutate_compressed_round(&mut case.proof.stages.stage6a_sumcheck_proof, round_index);
            },
        );
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_stage6_address_phase_round_counts(base: &VerifierFixtureCase) {
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target("stage6.address_phase.round_count.missing"),
        base,
        |case| {
            pop_compressed_round(&mut case.proof.stages.stage6a_sumcheck_proof);
        },
    );

    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target("stage6.address_phase.round_count.extra"),
        base,
        |case| {
            push_compressed_round(&mut case.proof.stages.stage6a_sumcheck_proof);
        },
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_each_stage6_cycle_phase_round(base: &VerifierFixtureCase) {
    let round_count = compressed_round_count(&base.proof.stages.stage6b_sumcheck_proof);
    for round_index in 0..round_count {
        tamper_manifest::assert_verifier_fixture_tamper_rejects(
            manifest_target("stage6.cycle_phase.round_polynomial"),
            base,
            |case| {
                mutate_compressed_round(&mut case.proof.stages.stage6b_sumcheck_proof, round_index);
            },
        );
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_stage6_cycle_phase_round_counts(base: &VerifierFixtureCase) {
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target("stage6.cycle_phase.round_count.missing"),
        base,
        |case| {
            pop_compressed_round(&mut case.proof.stages.stage6b_sumcheck_proof);
        },
    );

    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target("stage6.cycle_phase.round_count.extra"),
        base,
        |case| {
            push_compressed_round(&mut case.proof.stages.stage6b_sumcheck_proof);
        },
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_each_stage7_batch_round(base: &VerifierFixtureCase) {
    let round_count = compressed_round_count(&base.proof.stages.stage7_sumcheck_proof);
    for round_index in 0..round_count {
        tamper_manifest::assert_verifier_fixture_tamper_rejects(
            manifest_target("stage7.batch.round_polynomial"),
            base,
            |case| {
                mutate_compressed_round(&mut case.proof.stages.stage7_sumcheck_proof, round_index);
            },
        );
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_stage7_batch_round_counts(base: &VerifierFixtureCase) {
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target("stage7.batch.round_count.missing"),
        base,
        |case| {
            pop_compressed_round(&mut case.proof.stages.stage7_sumcheck_proof);
        },
    );

    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target("stage7.batch.round_count.extra"),
        base,
        |case| {
            push_compressed_round(&mut case.proof.stages.stage7_sumcheck_proof);
        },
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn clear_full_round_count(proof: &SumcheckProof<Fr, jolt_crypto::Bn254G1>) -> usize {
    let SumcheckProof::Clear(ClearProof::Full(proof)) = proof else {
        panic!("converted verifier fixture must use a clear full uni-skip proof");
    };
    proof.round_polynomials.len()
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn compressed_round_count(proof: &SumcheckProof<Fr, jolt_crypto::Bn254G1>) -> usize {
    let SumcheckProof::Clear(ClearProof::Compressed(proof)) = proof else {
        panic!("converted verifier fixture must use a clear compressed sumcheck proof");
    };
    proof.round_polynomials.len()
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn mutate_full_round(proof: &mut SumcheckProof<Fr, jolt_crypto::Bn254G1>, round_index: usize) {
    let SumcheckProof::Clear(ClearProof::Full(proof)) = proof else {
        panic!("converted verifier fixture must use a clear full uni-skip proof");
    };
    let Some(round) = proof.round_polynomials.get_mut(round_index) else {
        panic!("converted verifier fixture is missing expected uni-skip round {round_index}");
    };
    *round = UnivariatePoly::new(vec![Fr::from_u64(round_index as u64 + 1)]);
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn mutate_compressed_round(
    proof: &mut SumcheckProof<Fr, jolt_crypto::Bn254G1>,
    round_index: usize,
) {
    let SumcheckProof::Clear(ClearProof::Compressed(proof)) = proof else {
        panic!("converted verifier fixture must use a clear compressed sumcheck proof");
    };
    let Some(round) = proof.round_polynomials.get_mut(round_index) else {
        panic!("converted verifier fixture is missing expected compressed round {round_index}");
    };
    *round = CompressedPoly::new(vec![Fr::from_u64(round_index as u64 + 1)]);
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn pop_full_round(proof: &mut SumcheckProof<Fr, jolt_crypto::Bn254G1>) {
    let SumcheckProof::Clear(ClearProof::Full(proof)) = proof else {
        panic!("converted verifier fixture must use a clear full uni-skip proof");
    };
    let removed = proof.round_polynomials.pop();
    assert!(
        removed.is_some(),
        "converted verifier fixture has no full round to remove"
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn push_full_round(proof: &mut SumcheckProof<Fr, jolt_crypto::Bn254G1>) {
    let SumcheckProof::Clear(ClearProof::Full(proof)) = proof else {
        panic!("converted verifier fixture must use a clear full uni-skip proof");
    };
    proof
        .round_polynomials
        .push(UnivariatePoly::new(vec![Fr::from_u64(1)]));
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn pop_compressed_round(proof: &mut SumcheckProof<Fr, jolt_crypto::Bn254G1>) {
    let SumcheckProof::Clear(ClearProof::Compressed(proof)) = proof else {
        panic!("converted verifier fixture must use a clear compressed sumcheck proof");
    };
    let removed = proof.round_polynomials.pop();
    assert!(
        removed.is_some(),
        "converted verifier fixture has no compressed round to remove"
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn push_compressed_round(proof: &mut SumcheckProof<Fr, jolt_crypto::Bn254G1>) {
    let SumcheckProof::Clear(ClearProof::Compressed(proof)) = proof else {
        panic!("converted verifier fixture must use a clear compressed sumcheck proof");
    };
    proof
        .round_polynomials
        .push(CompressedPoly::new(vec![Fr::from_u64(1)]));
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn offset_claim_rejects(base: &VerifierFixtureCase, target_name: &str, id: JoltOpeningId) {
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target(target_name),
        base,
        |case| {
            assert!(
                offset_opening_claim(&mut case.proof, id, Fr::from_u64(1)),
                "converted verifier fixture is missing opening claim {id:?}"
            );
        },
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn stage1_required_openings(base: &VerifierFixtureCase) -> Vec<(&'static str, JoltOpeningId)> {
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

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn stage2_formula_output_openings() -> Vec<(&'static str, JoltOpeningId)> {
    let mut openings = Vec::new();
    openings.extend(
        [ram::ram_val(), ram::ram_ra(), ram::ram_inc()]
            .into_iter()
            .map(|id| ("stage2.claims.batch_outputs.ram_read_write", id)),
    );
    openings.extend(
        [
            spartan::left_instruction_input_product(),
            spartan::right_instruction_input_product(),
            spartan::jump_flag_product(),
            spartan::lookup_output_product(),
            spartan::branch_flag_product(),
            spartan::next_is_noop_product(),
        ]
        .into_iter()
        .map(|id| ("stage2.claims.batch_outputs.product_remainder.checked", id)),
    );
    openings.extend(
        [
            instruction_claim_reduction::lookup_output_reduced(),
            instruction_claim_reduction::left_lookup_operand_reduced(),
            instruction_claim_reduction::right_lookup_operand_reduced(),
            instruction_claim_reduction::left_instruction_input_reduced(),
            instruction_claim_reduction::right_instruction_input_reduced(),
        ]
        .into_iter()
        .map(|id| {
            (
                "stage2.claims.batch_outputs.instruction_claim_reduction",
                id,
            )
        }),
    );
    openings.extend(
        [ram::ram_ra_raf_evaluation()]
            .into_iter()
            .map(|id| ("stage2.claims.batch_outputs.ram_raf_evaluation", id)),
    );
    openings.extend(
        [ram::ram_val_final()]
            .into_iter()
            .map(|id| ("stage2.claims.batch_outputs.ram_output_check", id)),
    );
    openings
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn stage3_formula_output_openings() -> Vec<(&'static str, JoltOpeningId)> {
    let mut openings = Vec::new();
    openings.extend(
        [
            spartan::unexpanded_pc_shift(),
            spartan::pc_shift(),
            spartan::is_virtual_shift(),
            spartan::is_first_in_sequence_shift(),
            spartan::is_noop_shift(),
        ]
        .into_iter()
        .map(|id| ("stage3.claims.shift", id)),
    );
    openings.extend(
        [
            instruction::right_operand_is_rs2(),
            instruction::rs2_value(),
            instruction::right_operand_is_imm(),
            instruction::imm(),
            instruction::left_operand_is_rs1(),
            instruction::rs1_value(),
            instruction::left_operand_is_pc(),
            instruction::unexpanded_pc(),
        ]
        .into_iter()
        .map(|id| ("stage3.claims.instruction_input", id)),
    );
    openings.extend(
        [
            registers_claim_reduction::rd_write_value_reduced(),
            registers_claim_reduction::rs1_value_reduced(),
            registers_claim_reduction::rs2_value_reduced(),
        ]
        .into_iter()
        .map(|id| ("stage3.claims.registers_claim_reduction", id)),
    );
    openings
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn stage4_formula_output_openings() -> Vec<(&'static str, JoltOpeningId)> {
    let mut openings = Vec::new();
    openings.extend(
        [
            registers::registers_val_read_write(),
            registers::rs1_ra_read_write(),
            registers::rs2_ra_read_write(),
            registers::rd_wa_read_write(),
            registers::rd_inc_read_write(),
        ]
        .into_iter()
        .map(|id| ("stage4.claims.registers_read_write", id)),
    );
    openings.extend(
        [ram::ram_ra_val_check(), ram::ram_inc_val_check()]
            .into_iter()
            .map(|id| ("stage4.claims.ram_val_check", id)),
    );
    openings
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
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

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn stage5_formula_output_openings(
    base: &VerifierFixtureCase,
) -> Vec<(&'static str, JoltOpeningId)> {
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
        [ram::ram_ra_claim_reduction()]
            .into_iter()
            .map(|id| ("stage5.claims.ram_ra_claim_reduction", id)),
    );
    openings.extend(
        [
            registers::rd_inc_val_evaluation(),
            registers::rd_wa_val_evaluation(),
        ]
        .into_iter()
        .map(|id| ("stage5.claims.registers_val_evaluation", id)),
    );
    openings
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn stage6_formula_output_openings(
    base: &VerifierFixtureCase,
) -> Vec<(&'static str, JoltOpeningId)> {
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
    openings.extend([ram::ram_hamming_weight()].into_iter().map(|id| {
        (
            "stage6.claims.ram_hamming_booleanity.ram_hamming_weight",
            id,
        )
    }));
    openings.extend(
        relations::ram::RamRaVirtualizationOutputClaims::<Fr> {
            ram_ra: vec![Fr::zero(); dimensions.ram_ra_virtualization.num_committed_ra_polys()],
        }
        .canonical_order()
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
    let [ram_inc, rd_inc] = [increments::ram_inc_reduced(), increments::rd_inc_reduced()];
    openings.extend([
        ("stage6.claims.inc_claim_reduction.ram_inc", ram_inc),
        ("stage6.claims.inc_claim_reduction.rd_inc", rd_inc),
    ]);

    openings
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn case_advice_layouts(base: &VerifierFixtureCase) -> PrecommittedSchedule {
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

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn stage6_advice_output_openings(base: &VerifierFixtureCase) -> Vec<(&'static str, JoltOpeningId)> {
    let schedule = case_advice_layouts(base);
    let (trusted_layout, untrusted_layout) = (schedule.trusted_advice, schedule.untrusted_advice);
    let mut openings = Vec::new();

    if let Some(layout) = trusted_layout {
        openings.extend(
            advice::cycle_phase_output_openings(JoltAdviceKind::Trusted, layout.dimensions())
                .into_iter()
                .map(|id| ("stage6.claims.trusted_advice.trusted", id)),
        );
    }
    if let Some(layout) = untrusted_layout {
        openings.extend(
            advice::cycle_phase_output_openings(JoltAdviceKind::Untrusted, layout.dimensions())
                .into_iter()
                .map(|id| ("stage6.claims.untrusted_advice.untrusted", id)),
        );
    }

    openings
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn stage7_formula_output_openings(
    base: &VerifierFixtureCase,
) -> Vec<(&'static str, JoltOpeningId)> {
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

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn stage7_advice_output_openings(base: &VerifierFixtureCase) -> Vec<(&'static str, JoltOpeningId)> {
    let schedule = case_advice_layouts(base);
    let (trusted_layout, untrusted_layout) = (schedule.trusted_advice, schedule.untrusted_advice);
    let mut openings = Vec::new();

    if trusted_layout.is_some_and(|layout| layout.dimensions().has_address_phase()) {
        openings.push((
            "stage7.claims.trusted_advice.trusted",
            advice::final_advice_opening(JoltAdviceKind::Trusted),
        ));
    }
    if untrusted_layout.is_some_and(|layout| layout.dimensions().has_address_phase()) {
        openings.push((
            "stage7.claims.untrusted_advice.untrusted",
            advice::final_advice_opening(JoltAdviceKind::Untrusted),
        ));
    }

    openings
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn stage6_dimensions(base: &VerifierFixtureCase) -> JoltFormulaDimensions {
    let log_t = base.proof.trace_length.ilog2() as usize;
    JoltFormulaDimensions::try_from(base.proof.one_hot_config.dimensions(
        log_t,
        2 * RISCV_XLEN,
        base.preprocessing.program.bytecode_len(),
        base.proof.ram_K,
    ))
    .unwrap_or_else(|error| panic!("verifier fixture has invalid Stage 6 dimensions: {error}"))
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn manifest_target(name: &str) -> tamper_manifest::TamperTarget {
    tamper_manifest::required_target(name)
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn stage2_effective_output_claim(base: &VerifierFixtureCase, id: JoltOpeningId) -> Fr {
    opening_claim(&base.proof, id)
        .or_else(|| stage2_output_alias_claim(base, id))
        .unwrap_or_else(|| Fr::from_u64(0))
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn stage2_output_alias_claim(base: &VerifierFixtureCase, id: JoltOpeningId) -> Option<Fr> {
    let alias = stage2_output_alias(id)?;
    opening_claim(&base.proof, alias)
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn stage2_output_alias(id: JoltOpeningId) -> Option<JoltOpeningId> {
    if id == instruction_claim_reduction::lookup_output_reduced() {
        Some(spartan::lookup_output_product())
    } else if id == instruction_claim_reduction::left_instruction_input_reduced() {
        Some(spartan::left_instruction_input_product())
    } else if id == instruction_claim_reduction::right_instruction_input_reduced() {
        Some(spartan::right_instruction_input_product())
    } else {
        None
    }
}
