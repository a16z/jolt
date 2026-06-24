#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use crate::support;

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn verifier_rejects_config_mismatch() {
    support::assert_rejects(
        crate::support::verifier_fixtures::invalid_trace_length_case().verify(),
    );
}

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn verifier_rejects_ram_domain_mismatch() {
    support::assert_rejects(crate::support::verifier_fixtures::invalid_ram_k_case().verify());
}

#[test]
#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[ignore = "enable --features prover-fixtures in a non-ZK build to run this verifier-transitivity fixture"]
fn verifier_rejects_config_mismatch() {}
