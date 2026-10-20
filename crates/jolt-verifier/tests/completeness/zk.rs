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
use jolt_algebra::Fr;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_crypto::{Bn254G1, Pedersen};
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_dory::DoryScheme;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_transcript::VerifierTranscript;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_verifier::{
    jolt_protocol_id, verify_stages, JoltSponge, MaskedStages, ProofHeader, VerifiedStages,
    JOLT_SESSION,
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

/// Golden masked shape of the muldiv fixture at trace_length 1024 /
/// ram_K 8192: the pad layout and the residual relation's row counts. The
/// header is pinned alongside so a drift here reads as a protocol change
/// rather than a fixture change.
#[test]
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn zk_muldiv_masked_shape_matches_golden() {
    let case = crate::support::verifier_fixtures::zk_muldiv_case();
    let mut transcript = VerifierTranscript::<JoltSponge>::new(
        &jolt_protocol_id::<JoltSponge>(),
        JOLT_SESSION,
        &case.proof.narg,
    );
    let header = ProofHeader::receive(&mut transcript).expect("decode proof header");
    assert_eq!(header.trace_length, 1024);
    assert_eq!(header.ram_K, 8192);

    let mut transcript = VerifierTranscript::<JoltSponge>::new(
        &jolt_protocol_id::<JoltSponge>(),
        JOLT_SESSION,
        &case.proof.narg,
    );
    let mut masked = masked_stages(&case, &mut transcript);
    let auxiliary = masked
        .receive_auxiliary(&mut transcript)
        .expect("auxiliary block");
    let relation = masked.relation(&auxiliary).expect("residual relation");
    let layout = masked.pads.layout();
    assert_eq!(layout.num_pads, 2073);
    assert_eq!(layout.auxiliary.len(), 108);
    assert_eq!(relation.affine.len(), 16);
    assert_eq!(relation.products.len(), 106);
}

/// Every pad slot of the muldiv fixture's layout, base and auxiliary, has a
/// nonzero coefficient in some row of its residual relation: no masked value
/// is left out of the relation. This is a structural check. It does not show
/// that changing a value breaks a row, since a product row with a zero
/// cofactor or two compensating rows would pass it; the prover-side probe
/// `zk_muldiv_shifting_any_decoded_value_breaks_a_row` (jolt-prover
/// `zk_e2e`) checks that.
#[test]
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn zk_muldiv_every_masked_value_is_constrained() {
    use jolt_algebra::Zero;
    use jolt_zk::masking::{AffineRow, ResidualVar};
    use std::collections::BTreeSet;

    let case = crate::support::verifier_fixtures::zk_muldiv_case();
    let mut transcript = VerifierTranscript::<JoltSponge>::new(
        &jolt_protocol_id::<JoltSponge>(),
        JOLT_SESSION,
        &case.proof.narg,
    );
    let mut masked = masked_stages(&case, &mut transcript);
    let auxiliary = masked
        .receive_auxiliary(&mut transcript)
        .expect("auxiliary block");
    let relation = masked.relation(&auxiliary).expect("residual relation");

    let mut constrained = BTreeSet::new();
    let mut add_row = |row: &AffineRow<Fr>| {
        for (variable, coefficient) in &row.terms {
            if let ResidualVar::Pad(pad) = variable {
                if !coefficient.is_zero() {
                    let _ = constrained.insert(pad.0);
                }
            }
        }
    };
    for row in &relation.affine {
        add_row(row);
    }
    for row in &relation.products {
        add_row(&row.x.form());
        add_row(&row.y);
        add_row(&row.z);
    }
    let unconstrained: Vec<usize> = (0..relation.layout.num_pads)
        .filter(|pad| !constrained.contains(pad))
        .collect();
    assert!(
        unconstrained.is_empty(),
        "masked values no row constrains: {unconstrained:?}"
    );
}

/// Replays the stage spine over the fixture's argument string and returns
/// the masked stage outputs.
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn masked_stages(
    case: &ZkVerifierFixtureCase,
    transcript: &mut VerifierTranscript<'_, JoltSponge>,
) -> MaskedStages<Fr, Bn254G1> {
    let stages = verify_stages::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
        &case.preprocessing,
        &case.public_io,
        case.trusted_advice_commitment.as_ref(),
        transcript,
    )
    .expect("stage spine accepts the honest ZK fixture");
    let VerifiedStages::Zk(masked) = stages else {
        panic!("ZK verifier fixture must produce masked stage outputs");
    };
    *masked
}
