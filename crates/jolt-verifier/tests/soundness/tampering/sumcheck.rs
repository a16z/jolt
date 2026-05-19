#![cfg_attr(
    all(feature = "core-fixtures", not(feature = "zk")),
    expect(
        clippy::panic,
        reason = "test fixtures should fail loudly when their assumed proof shape changes"
    )
)]

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use crate::support::{core_fixtures::CoreVerifierCase, tamper_manifest};
#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
use crate::{
    soundness::tampering,
    support::{soundness_expectation, HarnessExpectation},
};

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use jolt_claims::protocols::jolt::{
    formulas::{
        claim_reductions::instruction as instruction_claim_reduction,
        ram,
        spartan::{
            outer_opening, outer_uniskip_opening, product_outer_opening,
            product_remainder_output_openings, product_should_branch_outer_opening,
            product_should_jump_outer_opening, product_uniskip_opening, SpartanOuterDimensions,
        },
    },
    JoltOpeningId, JoltStageId, JoltVirtualPolynomial,
};
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use jolt_field::{Fr, FromPrimitiveInt};
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use jolt_poly::{CompressedPoly, UnivariatePoly};
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use jolt_sumcheck::{ClearProof, SumcheckProof};
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use jolt_verifier::compat::claims::{offset_opening_claim, opening_claim, upsert_opening_claim};

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

#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features core-fixtures in a non-ZK build to live-generate, cast, and tamper real core proofs"]
fn tampered_stage1_sumcheck_payload_reject() {
    assert_eq!(
        soundness_expectation(tampering::STAGE1_SUMCHECK_PAYLOAD),
        HarnessExpectation::RejectsAtOrBeforeFrontier,
    );
}

#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features core-fixtures in a non-ZK build to live-generate, cast, and tamper real core proofs"]
fn tampered_stage2_uniskip_payload_reject() {
    assert_eq!(
        soundness_expectation(tampering::STAGE2_UNISKIP_PAYLOAD),
        HarnessExpectation::RejectsAtOrBeforeFrontier,
    );
}

#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features core-fixtures in a non-ZK build to live-generate, cast, and tamper real core proofs"]
fn tampered_stage2_sumcheck_payload_reject() {
    assert_eq!(
        soundness_expectation(tampering::STAGE2_SUMCHECK_PAYLOAD),
        HarnessExpectation::RejectsAtOrBeforeFrontier,
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn real_core_case() -> CoreVerifierCase {
    crate::support::core_fixtures::standard_muldiv_case()
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
fn manifest_target(name: &str) -> tamper_manifest::TamperTarget {
    tamper_manifest::required_target(name)
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn outer_virtual(polynomial: JoltVirtualPolynomial) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(polynomial, JoltStageId::SpartanOuter)
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn stage2_effective_output_claim(base: &CoreVerifierCase, id: JoltOpeningId) -> Fr {
    opening_claim(&base.proof, id)
        .or_else(|| stage2_output_alias_claim(base, id))
        .unwrap_or_else(|| Fr::from_u64(0))
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn stage2_output_alias_claim(base: &CoreVerifierCase, id: JoltOpeningId) -> Option<Fr> {
    let [product_left_instruction_input, product_right_instruction_input, _product_jump, _product_write_lookup_output_to_rd, product_lookup_output, _product_branch, _product_next_is_noop, _product_virtual_instruction] =
        product_remainder_output_openings();
    let [instruction_lookup_output, _instruction_left_lookup_operand, _instruction_right_lookup_operand, instruction_left_instruction_input, instruction_right_instruction_input] =
        instruction_claim_reduction::claim_reduction_output_openings();

    let alias = if id == instruction_lookup_output {
        product_lookup_output
    } else if id == instruction_left_instruction_input {
        product_left_instruction_input
    } else if id == instruction_right_instruction_input {
        product_right_instruction_input
    } else {
        return None;
    };

    opening_claim(&base.proof, alias)
}
