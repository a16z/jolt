#![cfg_attr(
    all(feature = "prover-fixtures", not(feature = "zk")),
    expect(
        clippy::panic,
        reason = "test fixtures should fail loudly when their assumed proof shape changes"
    )
)]

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use crate::support::{
    narg_frame_has_body, narg_frame_ranges, tamper_manifest, verifier_fixtures::VerifierFixtureCase,
};

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use crate::support::proof_claims::{offset_opening_claim, opening_claim, upsert_opening_claim};
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use jolt_claims::protocols::jolt::formulas::ram::RamRafEvaluationDimensions;
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use jolt_claims::protocols::jolt::{
    formulas::{
        booleanity, bytecode,
        claim_reductions::{
            advice, hamming_weight, increments, instruction as instruction_claim_reduction,
            registers as registers_claim_reduction,
        },
        dimensions::{JoltFormulaDimensions, TraceDimensions, REGISTER_ADDRESS_BITS},
        instruction, ram, registers,
        spartan::{
            self, outer_opening, outer_uniskip_opening, product_outer_opening,
            product_remainder_output_openings, product_should_branch_outer_opening,
            product_should_jump_outer_opening, product_uniskip_opening, shift_output_openings,
            SpartanOuterDimensions, SpartanProductDimensions,
        },
    },
    JoltAdviceKind, JoltCommittedPolynomial, JoltOpeningId, JoltPolynomialId, JoltRelationId,
    JoltVirtualPolynomial, PrecommittedReductionLayout,
};
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use jolt_field::{Fr, FromPrimitiveInt};
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use jolt_verifier::stages::PrecommittedSchedule;

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage1_sumcheck_payload_reject() {
    let base = verifier_fixture_case();
    tamper_each_stage1_uniskip_round(&base);
    tamper_each_stage1_remainder_round(&base);
    tamper_stage1_round_counts(&base);
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage1_opening_claims_reject() {
    let base = verifier_fixture_case();

    for (target_name, id) in stage1_required_openings(&base) {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage2_uniskip_payload_reject() {
    let base = verifier_fixture_case();
    tamper_each_stage2_uniskip_round(&base);
    tamper_stage2_uniskip_round_counts(&base);
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage2_sumcheck_payload_reject() {
    let base = verifier_fixture_case();
    tamper_each_stage2_batch_round(&base);
    tamper_stage2_batch_round_counts(&base);
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage2_input_claims_reject() {
    let base = verifier_fixture_case();

    for id in stage2_uniskip_openings()
        .into_iter()
        .chain(stage2_batch_input_openings())
    {
        offset_claim_rejects(&base, id.0, id.1);
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage2_output_claims_reject() {
    let base = verifier_fixture_case();

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

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage2_ram_phase_config_reject() {
    let base = verifier_fixture_case();

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
    let base = verifier_fixture_case();
    tamper_each_stage3_batch_round(&base);
    tamper_stage3_batch_round_counts(&base);
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage3_output_claims_reject() {
    let base = verifier_fixture_case();

    for (target_name, id) in stage3_formula_output_openings() {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage4_sumcheck_payload_reject() {
    let base = verifier_fixture_case();
    tamper_each_stage4_batch_round(&base);
    tamper_stage4_batch_round_counts(&base);
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage4_output_claims_reject() {
    let base = verifier_fixture_case();

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
    let base = verifier_fixture_case();
    tamper_each_stage5_batch_round(&base);
    tamper_stage5_batch_round_counts(&base);
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage5_output_claims_reject() {
    let base = verifier_fixture_case();

    for (target_name, id) in stage5_formula_output_openings(&base) {
        offset_claim_rejects(&base, target_name, id);
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage6_sumcheck_payload_reject() {
    let base = verifier_fixture_case();
    tamper_each_stage6_address_phase_round(&base);
    tamper_stage6_address_phase_round_counts(&base);
    tamper_each_stage6_cycle_phase_round(&base);
    tamper_stage6_cycle_phase_round_counts(&base);
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage6_output_claims_reject() {
    let base = verifier_fixture_case();

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
    let base = verifier_fixture_case();
    tamper_each_stage7_batch_round(&base);
    tamper_stage7_batch_round_counts(&base);
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_stage7_output_claims_reject() {
    let base = verifier_fixture_case();

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
fn verifier_fixture_case() -> VerifierFixtureCase {
    crate::support::verifier_fixtures::standard_muldiv_case()
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn real_advice_case() -> VerifierFixtureCase {
    crate::support::verifier_fixtures::standard_advice_consumer_case()
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_each_stage1_uniskip_round(base: &VerifierFixtureCase) {
    tamper_each_narg_round(
        base,
        "stage1.uni_skip.round_polynomial",
        narg_payload_spans(base).stage1_uniskip,
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_each_stage1_remainder_round(base: &VerifierFixtureCase) {
    tamper_each_narg_round(
        base,
        "stage1.remainder.round_polynomial",
        narg_payload_spans(base).stage1_remainder,
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_stage1_round_counts(base: &VerifierFixtureCase) {
    tamper_narg_round_count(
        base,
        "stage1.uni_skip.round_count.missing",
        "stage1.uni_skip.round_count.extra",
        narg_payload_spans(base).stage1_uniskip,
    );
    tamper_narg_round_count(
        base,
        "stage1.remainder.round_count.missing",
        "stage1.remainder.round_count.extra",
        narg_payload_spans(base).stage1_remainder,
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_each_stage2_uniskip_round(base: &VerifierFixtureCase) {
    tamper_each_narg_round(
        base,
        "stage2.product_uniskip.round_polynomial",
        narg_payload_spans(base).stage2_uniskip,
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_stage2_uniskip_round_counts(base: &VerifierFixtureCase) {
    tamper_narg_round_count(
        base,
        "stage2.product_uniskip.round_count.missing",
        "stage2.product_uniskip.round_count.extra",
        narg_payload_spans(base).stage2_uniskip,
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_each_stage2_batch_round(base: &VerifierFixtureCase) {
    tamper_each_narg_round(
        base,
        "stage2.batch.round_polynomial",
        narg_payload_spans(base).stage2_batch,
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_stage2_batch_round_counts(base: &VerifierFixtureCase) {
    tamper_narg_round_count(
        base,
        "stage2.batch.round_count.missing",
        "stage2.batch.round_count.extra",
        narg_payload_spans(base).stage2_batch,
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_each_stage3_batch_round(base: &VerifierFixtureCase) {
    tamper_each_narg_round(
        base,
        "stage3.batch.round_polynomial",
        narg_payload_spans(base).stage3_batch,
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_stage3_batch_round_counts(base: &VerifierFixtureCase) {
    tamper_narg_round_count(
        base,
        "stage3.batch.round_count.missing",
        "stage3.batch.round_count.extra",
        narg_payload_spans(base).stage3_batch,
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_each_stage4_batch_round(base: &VerifierFixtureCase) {
    tamper_each_narg_round(
        base,
        "stage4.batch.round_polynomial",
        narg_payload_spans(base).stage4_batch,
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_stage4_batch_round_counts(base: &VerifierFixtureCase) {
    tamper_narg_round_count(
        base,
        "stage4.batch.round_count.missing",
        "stage4.batch.round_count.extra",
        narg_payload_spans(base).stage4_batch,
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_each_stage5_batch_round(base: &VerifierFixtureCase) {
    tamper_each_narg_round(
        base,
        "stage5.batch.round_polynomial",
        narg_payload_spans(base).stage5_batch,
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_stage5_batch_round_counts(base: &VerifierFixtureCase) {
    tamper_narg_round_count(
        base,
        "stage5.batch.round_count.missing",
        "stage5.batch.round_count.extra",
        narg_payload_spans(base).stage5_batch,
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_each_stage6_address_phase_round(base: &VerifierFixtureCase) {
    tamper_each_narg_round(
        base,
        "stage6.address_phase.round_polynomial",
        narg_payload_spans(base).stage6_address,
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_stage6_address_phase_round_counts(base: &VerifierFixtureCase) {
    tamper_narg_round_count(
        base,
        "stage6.address_phase.round_count.missing",
        "stage6.address_phase.round_count.extra",
        narg_payload_spans(base).stage6_address,
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_each_stage6_cycle_phase_round(base: &VerifierFixtureCase) {
    tamper_each_narg_round(
        base,
        "stage6.cycle_phase.round_polynomial",
        narg_payload_spans(base).stage6_cycle,
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_stage6_cycle_phase_round_counts(base: &VerifierFixtureCase) {
    tamper_narg_round_count(
        base,
        "stage6.cycle_phase.round_count.missing",
        "stage6.cycle_phase.round_count.extra",
        narg_payload_spans(base).stage6_cycle,
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_each_stage7_batch_round(base: &VerifierFixtureCase) {
    tamper_each_narg_round(
        base,
        "stage7.batch.round_polynomial",
        narg_payload_spans(base).stage7_batch,
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_stage7_batch_round_counts(base: &VerifierFixtureCase) {
    tamper_narg_round_count(
        base,
        "stage7.batch.round_count.missing",
        "stage7.batch.round_count.extra",
        narg_payload_spans(base).stage7_batch,
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[derive(Clone, Copy)]
struct NargFrameSpan {
    start: usize,
    len: usize,
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
impl NargFrameSpan {
    const fn last(self) -> usize {
        self.start + self.len - 1
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
struct NargPayloadSpans {
    stage1_uniskip: NargFrameSpan,
    stage1_remainder: NargFrameSpan,
    stage2_uniskip: NargFrameSpan,
    stage2_batch: NargFrameSpan,
    stage3_batch: NargFrameSpan,
    stage4_batch: NargFrameSpan,
    stage5_batch: NargFrameSpan,
    stage6_address: NargFrameSpan,
    stage6_cycle: NargFrameSpan,
    stage7_batch: NargFrameSpan,
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn narg_payload_spans(base: &VerifierFixtureCase) -> NargPayloadSpans {
    let log_t = base.proof.trace_length.ilog2() as usize;
    let mut cursor = 2;

    let spans = NargPayloadSpans {
        stage1_uniskip: take_narg_span(&mut cursor, 1),
        stage1_remainder: take_narg_span(&mut cursor, 1 + log_t),
        stage2_uniskip: take_narg_span(&mut cursor, 1),
        stage2_batch: take_narg_span(&mut cursor, stage2_batch_round_count(base)),
        stage3_batch: take_narg_span(&mut cursor, stage3_batch_round_count(base)),
        stage4_batch: take_narg_span(&mut cursor, stage4_batch_round_count(base)),
        stage5_batch: take_narg_span(&mut cursor, stage5_batch_round_count(base)),
        stage6_address: take_narg_span(&mut cursor, stage6_address_round_count(base)),
        stage6_cycle: take_narg_span(&mut cursor, stage6_cycle_round_count(base)),
        stage7_batch: take_narg_span(&mut cursor, stage7_batch_round_count(base)),
    };
    let frame_count = narg_frame_ranges(&base.proof.narg).len();
    assert_eq!(
        cursor, frame_count,
        "NARG frame plan must cover every verifier payload frame"
    );
    spans
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
const fn take_narg_span(cursor: &mut usize, len: usize) -> NargFrameSpan {
    let span = NargFrameSpan {
        start: *cursor,
        len,
    };
    *cursor += len;
    span
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_each_narg_round(base: &VerifierFixtureCase, target_name: &str, span: NargFrameSpan) {
    assert!(
        span.len > 0,
        "NARG span for {target_name} must not be empty"
    );
    for offset in 0..span.len {
        let frame_index = span.start + offset;
        tamper_manifest::assert_verifier_fixture_tamper_rejects(
            manifest_target(target_name),
            base,
            |case| mutate_narg_frame(case, frame_index),
        );
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_narg_round_count(
    base: &VerifierFixtureCase,
    missing_target: &str,
    extra_target: &str,
    span: NargFrameSpan,
) {
    assert!(
        span.len > 0,
        "NARG span for {missing_target}/{extra_target} must not be empty"
    );
    let last_frame = span.last();
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target(missing_target),
        base,
        |case| remove_narg_frame(case, last_frame),
    );
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        manifest_target(extra_target),
        base,
        |case| duplicate_narg_frame(case, last_frame),
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn mutate_narg_frame(case: &mut VerifierFixtureCase, frame_index: usize) {
    let ranges = narg_frame_ranges(&case.proof.narg);
    let range = ranges
        .get(frame_index)
        .unwrap_or_else(|| panic!("NARG is missing expected frame {frame_index}"));
    assert!(
        !range.body.is_empty(),
        "NARG frame {frame_index} should carry a round payload"
    );
    case.proof.narg[range.body.start] ^= 1;
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn remove_narg_frame(case: &mut VerifierFixtureCase, frame_index: usize) {
    let ranges = narg_frame_ranges(&case.proof.narg);
    let full = ranges
        .get(frame_index)
        .unwrap_or_else(|| panic!("NARG is missing expected frame {frame_index}"))
        .full
        .clone();
    drop(case.proof.narg.drain(full));
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn duplicate_narg_frame(case: &mut VerifierFixtureCase, frame_index: usize) {
    let ranges = narg_frame_ranges(&case.proof.narg);
    let full = ranges
        .get(frame_index)
        .unwrap_or_else(|| panic!("NARG is missing expected frame {frame_index}"))
        .full
        .clone();
    let frame = case.proof.narg[full.clone()].to_vec();
    drop(case.proof.narg.splice(full.end..full.end, frame));
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn stage2_batch_round_count(base: &VerifierFixtureCase) -> usize {
    let log_t = base.proof.trace_length.ilog2() as usize;
    let log_k = base.proof.ram_K.ilog2() as usize;
    let trace = TraceDimensions::new(log_t);
    let read_write = base.proof.rw_config.ram_dimensions(log_t, log_k);
    let product = SpartanProductDimensions::new(log_t);
    let raf = RamRafEvaluationDimensions::try_from(read_write)
        .unwrap_or_else(|error| panic!("fixture has invalid RAM RAF dimensions: {error}"));
    max_round_count([
        ram::read_write_checking::<Fr>(read_write).sumcheck.rounds,
        spartan::product_remainder::<Fr>(product).sumcheck.rounds,
        instruction_claim_reduction::claim_reduction::<Fr>(trace)
            .sumcheck
            .rounds,
        ram::raf_evaluation::<Fr>(raf).sumcheck.rounds,
        ram::output_check::<Fr>(read_write).sumcheck.rounds,
    ])
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn stage3_batch_round_count(base: &VerifierFixtureCase) -> usize {
    let trace = TraceDimensions::new(base.proof.trace_length.ilog2() as usize);
    max_round_count([
        spartan::shift::<Fr>(trace).sumcheck.rounds,
        instruction::input_virtualization::<Fr>(trace)
            .sumcheck
            .rounds,
        registers_claim_reduction::claim_reduction::<Fr>(trace)
            .sumcheck
            .rounds,
    ])
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn stage4_batch_round_count(base: &VerifierFixtureCase) -> usize {
    let log_t = base.proof.trace_length.ilog2() as usize;
    let trace = TraceDimensions::new(log_t);
    let registers_dimensions = base
        .proof
        .rw_config
        .register_dimensions(log_t, REGISTER_ADDRESS_BITS);
    max_round_count([
        registers::read_write_checking::<Fr>(registers_dimensions)
            .sumcheck
            .rounds,
        ram::val_check::<Fr>(trace, ram::RamValCheckInit::full(Fr::from_u64(0)))
            .sumcheck
            .rounds,
    ])
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn stage5_batch_round_count(base: &VerifierFixtureCase) -> usize {
    let dimensions = stage6_dimensions(base);
    max_round_count([
        instruction::read_raf::<Fr>(dimensions.instruction_read_raf)
            .sumcheck
            .rounds,
        ram::ra_claim_reduction::<Fr>(dimensions.trace)
            .sumcheck
            .rounds,
        registers::val_evaluation::<Fr>(dimensions.trace)
            .sumcheck
            .rounds,
    ])
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn stage6_address_round_count(base: &VerifierFixtureCase) -> usize {
    let dimensions = stage6_dimensions(base);
    let booleanity = booleanity::BooleanityDimensions::new(
        dimensions.ra_layout,
        base.proof.trace_length.ilog2() as usize,
        base.proof.one_hot_config.committed_chunk_bits(),
    );
    max_round_count([
        bytecode::read_raf_address_phase::<Fr>(dimensions.bytecode_read_raf)
            .sumcheck
            .rounds,
        booleanity::booleanity_address_phase::<Fr>(booleanity)
            .sumcheck
            .rounds,
    ])
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn stage6_cycle_round_count(base: &VerifierFixtureCase) -> usize {
    let dimensions = stage6_dimensions(base);
    let booleanity = booleanity::BooleanityDimensions::new(
        dimensions.ra_layout,
        dimensions.trace.log_t(),
        base.proof.one_hot_config.committed_chunk_bits(),
    );
    max_round_count([
        bytecode::read_raf_cycle_phase::<Fr>(dimensions.bytecode_read_raf)
            .sumcheck
            .rounds,
        booleanity::booleanity_cycle_phase::<Fr>(booleanity)
            .sumcheck
            .rounds,
        ram::hamming_booleanity::<Fr>(dimensions.trace)
            .sumcheck
            .rounds,
        ram::ra_virtualization::<Fr>(dimensions.ram_ra_virtualization)
            .sumcheck
            .rounds,
        instruction::ra_virtualization::<Fr>(dimensions.instruction_ra_virtualization)
            .sumcheck
            .rounds,
        increments::claim_reduction::<Fr>(dimensions.trace)
            .sumcheck
            .rounds,
    ])
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn stage7_batch_round_count(base: &VerifierFixtureCase) -> usize {
    let dimensions = stage6_dimensions(base);
    let hamming = hamming_weight::HammingWeightClaimReductionDimensions::new(
        dimensions.ra_layout,
        base.proof.one_hot_config.committed_chunk_bits(),
    );
    hamming_weight::claim_reduction::<Fr>(hamming)
        .sumcheck
        .rounds
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn max_round_count<const N: usize>(rounds: [usize; N]) -> usize {
    rounds.into_iter().fold(0, usize::max)
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

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
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

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
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

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
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

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
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

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn case_advice_layouts(base: &VerifierFixtureCase) -> PrecommittedSchedule {
    PrecommittedSchedule::new(
        base.proof.trace_polynomial_order,
        base.proof.trace_length.ilog2() as usize,
        base.proof.one_hot_config.committed_chunk_bits(),
        base.trusted_advice_commitment
            .is_some()
            .then_some(base.public_io.memory_layout.max_trusted_advice_size as usize),
        narg_frame_has_body(&base.proof.narg, 1)
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
fn outer_virtual(polynomial: JoltVirtualPolynomial) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(polynomial, JoltRelationId::SpartanOuter)
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

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
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

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
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
