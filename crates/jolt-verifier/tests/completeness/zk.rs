#![cfg_attr(
    all(feature = "prover-fixtures", feature = "zk"),
    expect(
        clippy::expect_used,
        reason = "fixture audit tests should fail loudly when verifier object shape assumptions break"
    )
)]

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use crate::support;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_crypto::{Bn254G1, Pedersen};
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_dory::DoryScheme;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_field::Fr;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_transcript::Blake2b512;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_verifier::JoltProofClaims;

#[test]
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn zk_muldiv_verifier_proof_is_accepted() {
    support::assert_zk_accepts(crate::support::verifier_fixtures::zk_muldiv_case().verify());
}

#[test]
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn zk_committed_muldiv_verifier_proof_is_accepted() {
    support::assert_zk_accepts(
        crate::support::verifier_fixtures::zk_committed_muldiv_case().verify(),
    );
}

#[test]
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn zk_committed_muldiv_blindfold_shape_audit_matches_modular_protocol() {
    let case = crate::support::verifier_fixtures::zk_committed_muldiv_case();
    let modular = support::zk_audit::audit_zk_blindfold_protocol_shape::<
        Fr,
        DoryScheme,
        Pedersen<Bn254G1>,
        Blake2b512,
    >(&case.preprocessing, &case.public_io, &case.proof, None)
    .expect("build modular BlindFold protocol shape");

    assert!(matches!(case.proof.claims, JoltProofClaims::Zk));
    assert!(!case.proof.narg.is_empty());
    assert!(modular.coefficient_rows > 0);
    assert!(modular.output_claim_rows > 0);
    assert!(modular.auxiliary_rows > 0);
    assert!(modular.error_row_count > 0);
    assert!(modular.eval_commitments > 0);
}

#[test]
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn zk_muldiv_blindfold_shape_audit_matches_modular_protocol() {
    let case = crate::support::verifier_fixtures::zk_muldiv_case();
    let modular = support::zk_audit::audit_zk_blindfold_protocol_shape::<
        Fr,
        DoryScheme,
        Pedersen<Bn254G1>,
        Blake2b512,
    >(&case.preprocessing, &case.public_io, &case.proof, None)
    .expect("build modular BlindFold protocol shape");

    assert!(matches!(case.proof.claims, JoltProofClaims::Zk));
    assert!(!case.proof.narg.is_empty());

    assert_eq!(modular.coefficient_rows, 221);
    assert_eq!(modular.output_claim_rows, 15);
    assert_eq!(modular.eval_commitments, 1);
    assert_eq!(modular.auxiliary_rows, 33);
    assert_eq!(modular.error_row_count, 64);
}

#[test]
#[cfg(any(not(feature = "prover-fixtures"), not(feature = "zk")))]
#[ignore = "enable --features prover-fixtures,zk to live-generate this verifier ZK fixture"]
fn zk_muldiv_verifier_proof_is_accepted() {}

#[test]
#[ignore = "prefix BlindFold fixture generation is not wired yet"]
fn zk_stage1_prefix_is_accepted() {}
