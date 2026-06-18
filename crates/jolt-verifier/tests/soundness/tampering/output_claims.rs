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
use jolt_claims::protocols::jolt::formulas::{
    committed_openings::final_opening_ids, dimensions::JoltFormulaDimensions,
};
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use jolt_field::{Fr, FromPrimitiveInt};
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use jolt_lookup_tables::XLEN as RISCV_XLEN;

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tampered_output_claim_reject() {
    let base = verifier_fixtures::standard_muldiv_case();
    let log_t = base.proof.trace_length.ilog2() as usize;
    let dimensions = JoltFormulaDimensions::try_from(base.proof.one_hot_config.dimensions(
        log_t,
        2 * RISCV_XLEN,
        base.preprocessing.program.bytecode_len(),
        base.proof.ram_K,
    ))
    .expect("verifier fixture has invalid dimensions");
    let id = final_opening_ids(dimensions.ra_layout, false, false, None)[0];

    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("stage8.opening_claim_values"),
        &base,
        |case| {
            assert!(
                offset_opening_claim(&mut case.proof, id, Fr::from_u64(1)),
                "converted verifier fixture is missing final output claim {id:?}"
            );
        },
    );
}

#[test]
#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[ignore = "enable --features prover-fixtures in a non-ZK build to tamper real final output claims"]
fn tampered_output_claim_reject() {}
