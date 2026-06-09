#![expect(
    dead_code,
    reason = "The shared fixture support exposes helpers for the broader verifier harness."
)]

mod support;

use support::{
    assert_accepts, assert_rejects, clear_base_case, clear_base_case_with_transcript,
    clear_multiround_case, clear_multiround_case_with_transcript, tamper_clear_eval,
    tamper_public_output, zk_base_case, zk_base_case_with_transcript, zk_multiround_case,
    zk_multiround_case_with_transcript,
};

use jolt_field::Fr;
use jolt_transcript::PoseidonTranscript;

type PoseidonFrTranscript = PoseidonTranscript<Fr>;

#[test]
fn clear_valid_fixture_accepts_through_pcs_assist_trait() {
    assert_accepts(clear_base_case().verify_clear_via_pcs_assist());
}

#[test]
fn clear_multiround_fixture_accepts_through_pcs_assist_trait() {
    assert_accepts(clear_multiround_case().verify_clear_via_pcs_assist());
}

#[test]
fn zk_valid_fixture_accepts_through_pcs_assist_trait() {
    let case = zk_base_case();
    let direct = case.verify_zk();
    let via_trait = case.verify_zk_via_pcs_assist();

    assert!(
        matches!((&direct, &via_trait), (Ok(direct), Ok(via_trait)) if direct == via_trait),
        "direct ZK verify and PCS-assist trait verify diverged: direct={direct:?}, via_trait={via_trait:?}",
    );
}

#[test]
fn zk_multiround_fixture_accepts_through_pcs_assist_trait() {
    let case = zk_multiround_case();
    let direct = case.verify_zk();
    let via_trait = case.verify_zk_via_pcs_assist();

    assert!(
        matches!((&direct, &via_trait), (Ok(direct), Ok(via_trait)) if direct == via_trait),
        "direct ZK multiround verify and PCS-assist trait verify diverged: direct={direct:?}, via_trait={via_trait:?}",
    );
}

#[test]
fn clear_tampered_eval_rejects_through_pcs_assist_trait() {
    let mut case = clear_base_case();
    tamper_clear_eval(&mut case);

    assert_rejects(case.verify_clear_via_pcs_assist());
}

#[test]
fn zk_tampered_public_output_rejects_through_pcs_assist_trait() {
    let mut case = zk_base_case();
    tamper_public_output(&mut case);

    assert_rejects(case.verify_zk_via_pcs_assist());
}

#[test]
fn clear_poseidon_fixture_accepts_through_pcs_assist_trait() {
    assert_accepts(
        clear_base_case_with_transcript::<PoseidonFrTranscript>()
            .verify_clear_via_pcs_assist_with_transcript::<PoseidonFrTranscript>(),
    );
}

#[test]
fn clear_multiround_poseidon_fixture_accepts_through_pcs_assist_trait() {
    assert_accepts(
        clear_multiround_case_with_transcript::<PoseidonFrTranscript>()
            .verify_clear_via_pcs_assist_with_transcript::<PoseidonFrTranscript>(),
    );
}

#[test]
fn zk_poseidon_fixture_accepts_through_pcs_assist_trait() {
    assert_accepts(
        zk_base_case_with_transcript::<PoseidonFrTranscript>()
            .verify_zk_via_pcs_assist_with_transcript::<PoseidonFrTranscript>(),
    );
}

#[test]
fn zk_multiround_poseidon_fixture_accepts_through_pcs_assist_trait() {
    assert_accepts(
        zk_multiround_case_with_transcript::<PoseidonFrTranscript>()
            .verify_zk_via_pcs_assist_with_transcript::<PoseidonFrTranscript>(),
    );
}

#[test]
fn clear_blake_fixture_rejects_under_poseidon_transcript() {
    assert_rejects(
        clear_base_case().verify_clear_via_pcs_assist_with_transcript::<PoseidonFrTranscript>(),
    );
}

#[test]
fn clear_multiround_blake_fixture_rejects_under_poseidon_transcript() {
    assert_rejects(
        clear_multiround_case()
            .verify_clear_via_pcs_assist_with_transcript::<PoseidonFrTranscript>(),
    );
}

#[test]
fn zk_blake_fixture_rejects_under_poseidon_transcript() {
    assert_rejects(
        zk_base_case().verify_zk_via_pcs_assist_with_transcript::<PoseidonFrTranscript>(),
    );
}

#[test]
fn zk_multiround_blake_fixture_rejects_under_poseidon_transcript() {
    assert_rejects(
        zk_multiround_case().verify_zk_via_pcs_assist_with_transcript::<PoseidonFrTranscript>(),
    );
}
