#![cfg_attr(
    all(feature = "prover-fixtures", feature = "zk"),
    expect(
        clippy::expect_used,
        clippy::panic,
        reason = "fixture audit tests should fail loudly when verifier object shape assumptions break"
    )
)]

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use crate::support::{self, verifier_fixtures::ZkVerifierFixtureCase};
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_blindfold::BlindFoldProtocol;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_crypto::{Bn254G1, Pedersen};
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_dory::DoryScheme;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_field::Fr;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_transcript::VerifierTranscript;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_verifier::{
    jolt_protocol_id, verify_stages, JoltSponge, ProofHeader, VerifiedStages, JOLT_SESSION,
};

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

/// Golden BlindFold shape of the muldiv fixture at trace_length 1024 /
/// ram_K 8192. The header is pinned alongside so a drift here reads as a
/// protocol change rather than a fixture change.
#[test]
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn zk_muldiv_blindfold_shape_matches_golden() {
    let case = crate::support::verifier_fixtures::zk_muldiv_case();
    let mut transcript = VerifierTranscript::<JoltSponge>::new(
        &jolt_protocol_id::<JoltSponge>(),
        JOLT_SESSION,
        &case.proof.narg,
    );
    let header = ProofHeader::receive(&mut transcript).expect("decode proof header");
    assert_eq!(header.trace_length, 1024);
    assert_eq!(header.ram_K, 8192);

    let protocol = blindfold_protocol(&case);
    assert_eq!(protocol.dimensions.coefficient_rows, 227);
    assert_eq!(protocol.dimensions.output_claim_rows, 16);
    assert_eq!(protocol.dimensions.auxiliary_rows, 37);
    assert_eq!(protocol.dimensions.error.row_count, 64);
    assert_eq!(protocol.eval_commitments.len(), 1);
}

/// Replays the stage spine over the fixture's argument string and returns
/// the BlindFold protocol its committed outputs lower to.
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn blindfold_protocol(case: &ZkVerifierFixtureCase) -> BlindFoldProtocol<Fr, Bn254G1> {
    let mut transcript = VerifierTranscript::<JoltSponge>::new(
        &jolt_protocol_id::<JoltSponge>(),
        JOLT_SESSION,
        &case.proof.narg,
    );
    let stages = verify_stages::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
        &case.preprocessing,
        &case.public_io,
        case.trusted_advice_commitment.as_ref(),
        &mut transcript,
    )
    .expect("stage spine accepts the honest ZK fixture");
    let VerifiedStages::Zk { protocol, .. } = stages else {
        panic!("ZK verifier fixture must lower to a BlindFold protocol");
    };
    *protocol
}

#[test]
#[cfg(any(not(feature = "prover-fixtures"), not(feature = "zk")))]
#[ignore = "enable --features prover-fixtures,zk to live-generate this verifier ZK fixture"]
fn zk_muldiv_verifier_proof_is_accepted() {}

#[test]
#[ignore = "prefix BlindFold fixture generation is not wired yet"]
fn zk_stage1_prefix_is_accepted() {}
