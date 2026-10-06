#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use crate::support;

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn standard_advice_consumer_verifier_proof_is_accepted() {
    support::assert_accepts(
        crate::support::verifier_fixtures::standard_advice_consumer_case().verify(),
    );
}

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn standard_committed_advice_verifier_proof_is_accepted() {
    support::assert_accepts(
        crate::support::verifier_fixtures::fresh_standard_committed_advice_case().verify(),
    );
}
