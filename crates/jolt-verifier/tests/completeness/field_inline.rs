#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use crate::support;

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn field_inline_eqpoly_verifier_proof_is_accepted() {
    support::assert_accepts(
        crate::support::verifier_fixtures::standard_field_inline_eqpoly_case().verify(),
    );
}
