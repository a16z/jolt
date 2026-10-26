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
    // Auxiliary slots: one per distinct intermediate product that is not a
    // term's last (26; this layout repeats none). BlindFold takes affine operands, so Spartan's outer and
    // product remainders need no defining slot. Each term's last product is
    // a pair of its stage's check row: stage 1's single pair, then the
    // multi-pair rows of stages 2–6, after the 26 single-pair rows of the
    // slots.
    assert_eq!(layout.num_pads, 967);
    assert_eq!(layout.auxiliary.len(), 26);
    assert_eq!(relation.affine.len(), 8);
    assert_eq!(relation.products.len(), 32);
    let pairs: Vec<usize> = relation
        .products
        .iter()
        .map(|row| row.pairs().count())
        .collect();
    assert_eq!(pairs.iter().sum::<usize>(), 92);
    // Stages 2–6 land their output products on one row each.
    assert_eq!(pairs[..27], [1; 27]);
    assert_eq!(pairs[27..], [4, 4, 5, 2, 50]);
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
        for pair in row.pairs() {
            add_row(&pair.x.form());
            add_row(&pair.y);
        }
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

/// The committed-program fixture's auxiliary slots. Its bytecode read-RAF
/// cycle output (`ReadRafCyclePhaseCommitted`) repeats the bytecode RA
/// product in every staged-Val term; the lowering takes one slot per
/// distinct product (D34): 27, where one slot per repetition took 38.
#[test]
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn zk_committed_muldiv_auxiliary_slots_match_golden() {
    let case = crate::support::verifier_fixtures::zk_committed_muldiv_case();
    let mut transcript = VerifierTranscript::<JoltSponge>::new(
        &jolt_protocol_id::<JoltSponge>(),
        JOLT_SESSION,
        &case.proof.narg,
    );
    let masked = masked_stages(&case, &mut transcript);
    assert_eq!(masked.pads.layout().auxiliary.len(), 27);
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
