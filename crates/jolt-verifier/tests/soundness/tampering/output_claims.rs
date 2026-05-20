#![cfg_attr(
    all(feature = "core-fixtures", not(feature = "zk")),
    expect(
        clippy::expect_used,
        reason = "fixture dimension helpers should fail loudly when stored core artifacts are malformed"
    )
)]

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use crate::support::{core_fixtures, tamper_manifest};
#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
use crate::{
    soundness::tampering,
    support::{soundness_expectation, HarnessExpectation},
};
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use jolt_claims::protocols::jolt::formulas::{
    committed_openings::final_opening_ids, dimensions::JoltFormulaDimensions,
};
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use jolt_field::{Fr, FromPrimitiveInt};
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use jolt_lookup_tables::XLEN as RISCV_XLEN;
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use jolt_verifier::compat::claims::offset_opening_claim;

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tampered_output_claim_reject() {
    let base = core_fixtures::standard_muldiv_case();
    let log_t = base.proof.trace_length.ilog2() as usize;
    let dimensions = JoltFormulaDimensions::try_from(base.proof.one_hot_config.dimensions(
        log_t,
        2 * RISCV_XLEN,
        base.preprocessing.program.bytecode.code_size,
        base.proof.ram_K,
    ))
    .expect("core fixture has invalid dimensions");
    let id = final_opening_ids(dimensions.ra_layout, false, false)[0];

    tamper_manifest::assert_core_tamper_rejects(
        tamper_manifest::required_target("stage8.opening_claim_values"),
        &base,
        |case| {
            assert!(
                offset_opening_claim(&mut case.proof, id, Fr::from_u64(1)),
                "converted core fixture is missing final output claim {id:?}"
            );
        },
    );
}

#[test]
#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[ignore = "enable --features core-fixtures in a non-ZK build to tamper real final output claims"]
fn tampered_output_claim_reject() {
    assert_eq!(
        soundness_expectation(tampering::OUTPUT_CLAIM),
        HarnessExpectation::RejectsAtOrBeforeFrontier,
    );
}
