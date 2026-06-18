#![cfg_attr(
    all(feature = "prover-fixtures", not(feature = "zk")),
    expect(
        clippy::expect_used,
        reason = "fixture dimension helpers should fail loudly when stored verifier objects are malformed"
    )
)]

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use crate::support::proof_claims::offset_opening_claim;
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use crate::support::{tamper_manifest, verifier_fixtures};
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use jolt_claims::protocols::jolt::{
    formulas::{committed_openings::final_opening_ids, dimensions::JoltFormulaDimensions},
    JoltOpeningId,
};
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use jolt_field::{Fr, FromPrimitiveInt};
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use jolt_lookup_tables::XLEN as RISCV_XLEN;

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tampered_opening_value_reject() {
    tamper_final_opening_claims(&verifier_fixtures::standard_muldiv_case());
    tamper_final_opening_claims(&verifier_fixtures::standard_advice_consumer_case());
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tamper_final_opening_claims(base: &verifier_fixtures::VerifierFixtureCase) {
    for id in stage8_final_opening_ids(base) {
        tamper_manifest::assert_verifier_fixture_tamper_rejects(
            tamper_manifest::required_target("stage8.opening_claim_values"),
            base,
            |case| {
                assert!(
                    offset_opening_claim(&mut case.proof, id, Fr::from_u64(1)),
                    "converted verifier fixture is missing final opening claim {id:?}"
                );
            },
        );
    }
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn stage8_final_opening_ids(base: &verifier_fixtures::VerifierFixtureCase) -> Vec<JoltOpeningId> {
    let log_t = base.proof.trace_length.ilog2() as usize;
    let dimensions = JoltFormulaDimensions::try_from(base.proof.one_hot_config.dimensions(
        log_t,
        2 * RISCV_XLEN,
        base.preprocessing.program.bytecode_len(),
        base.proof.ram_K,
    ))
    .expect("verifier fixture has invalid final opening dimensions");
    final_opening_ids(
        dimensions.ra_layout,
        base.trusted_advice_commitment.is_some(),
        base.proof.untrusted_advice_commitment.is_some(),
        base.preprocessing
            .program
            .committed()
            .map(|committed| committed.bytecode_chunk_count()),
    )
}

#[test]
#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[ignore = "opening verification is not wired yet"]
fn tampered_opening_value_reject() {}
