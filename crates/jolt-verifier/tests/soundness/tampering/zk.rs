#![cfg_attr(
    all(feature = "prover-fixtures", feature = "zk"),
    expect(
        clippy::expect_used,
        clippy::panic,
        reason = "ZK tampering helpers assert fixture shape before mutating real verifier objects"
    )
)]
//! Tampers of a masked (BlindFold over committed pads) proof: every
//! proof-carried ZK payload — the masked tape, the pad commitment, the
//! BlindFold proof, the stage-8 hiding opening (`proof_shape.rs` covers the
//! masked proof's empty stage proofs). Per-value binding beyond the sampled sweep here is the
//! prover suite's `zk_muldiv_shifting_any_decoded_value_breaks_a_row`.

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use crate::support;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use crate::support::tamper_manifest;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_crypto::Bn254G1;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_field::{Fr, Ring as _};
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_verifier::{BlindFoldZkProof, JoltProofClaims};
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_zk::masking::MaskedMessages;

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn missing_zk_vector_commitment_setup_rejects_now() {
    with_zk_verifier_stack(|| {
        tamper_manifest::assert_zk_target_active("zk.vector_commitment_setup");
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        case.preprocessing.vc_setup = None;

        support::assert_zk_rejects(case.verify());
    });
}

/// An offset at sampled positions across the masked tape (its first and last
/// values and evenly spaced ones between, so every stage's region is hit)
/// rejects.
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_masked_values_reject_now() {
    with_zk_verifier_stack(|| {
        tamper_manifest::assert_zk_target_active("zk.masked_values");
        let case = crate::support::verifier_fixtures::zk_muldiv_case();
        let len = masked(&case.proof.claims).values.len();
        assert!(len > 1, "a ZK muldiv proof masks its stage messages");
        let samples = 48;
        let mut positions: Vec<usize> = (0..samples)
            .map(|k| k * (len - 1) / (samples - 1))
            .collect();
        positions.dedup();
        for position in positions {
            let mut tampered = case.clone();
            masked_mut(&mut tampered.proof.claims).values[position] += Fr::from_u64(1);
            support::assert_zk_rejects(tampered.verify());
        }
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn truncated_or_extended_zk_masked_tape_rejects_now() {
    with_zk_verifier_stack(|| {
        tamper_manifest::assert_zk_target_active("zk.masked_tape_length");
        let case = crate::support::verifier_fixtures::zk_muldiv_case();

        let mut truncated = case.clone();
        let _ = masked_mut(&mut truncated.proof.claims).values.pop();
        support::assert_zk_rejects(truncated.verify());

        let mut extended = case.clone();
        masked_mut(&mut extended.proof.claims)
            .values
            .push(Fr::from_u64(1));
        support::assert_zk_rejects(extended.verify());

        let mut nonce = case;
        masked_mut(&mut nonce.proof.claims).nonces.push(0);
        support::assert_zk_rejects(nonce.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_pad_commitment_rejects_now() {
    with_zk_verifier_stack(|| {
        tamper_manifest::assert_zk_target_active("zk.pad_commitment");
        let case = crate::support::verifier_fixtures::zk_muldiv_case();

        let mut missing = case.clone();
        let _ = zk_payload_mut(&mut missing.proof.claims)
            .pad_commitment
            .pop();
        support::assert_zk_rejects(missing.verify());

        let mut swapped = case;
        let rows = &mut zk_payload_mut(&mut swapped.proof.claims).pad_commitment;
        assert!(rows.len() > 1, "muldiv's pads take several rows");
        rows.swap(0, 1);
        support::assert_zk_rejects(swapped.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_joint_opening_eval_commitment_rejects_now() {
    with_zk_verifier_stack(|| {
        tamper_manifest::assert_zk_target_active("zk.joint_opening_proof.eval_commitment");
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        case.proof.joint_opening_proof.0.y_com = None;

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_blindfold_proof_rejects_now() {
    with_zk_verifier_stack(|| {
        tamper_manifest::assert_zk_target_active("zk.blindfold_proof");
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        zk_payload_mut(&mut case.proof.claims)
            .blindfold
            .folding
            .random_u += Fr::from_u64(1);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn with_zk_verifier_stack(test: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("zk-verifier-tamper".to_string())
        .stack_size(128 * 1024 * 1024)
        .spawn(test)
        .expect("spawn ZK verifier tamper test")
        .join()
        .expect("ZK verifier tamper test panicked");
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
type ZkClaims = JoltProofClaims<Fr, BlindFoldZkProof<Fr, Bn254G1>>;

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn masked(claims: &ZkClaims) -> &MaskedMessages<Fr> {
    let JoltProofClaims::Zk { masked, .. } = claims else {
        panic!("ZK fixture must carry a masked payload");
    };
    masked
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn masked_mut(claims: &mut ZkClaims) -> &mut MaskedMessages<Fr> {
    let JoltProofClaims::Zk { masked, .. } = claims else {
        panic!("ZK fixture must carry a masked payload");
    };
    masked
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn zk_payload_mut(claims: &mut ZkClaims) -> &mut BlindFoldZkProof<Fr, Bn254G1> {
    let JoltProofClaims::Zk { zk_proof, .. } = claims else {
        panic!("ZK fixture must carry a masked payload");
    };
    zk_proof
}
