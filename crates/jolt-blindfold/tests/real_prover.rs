//! Drives the shipped prover (`jolt_blindfold::prove`) end-to-end against the
//! real verifier, over the committed-stage messages that precede BlindFold in
//! the same proof.

#![expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "integration tests should fail loudly"
)]

mod support;

use jolt_blindfold::{ProverError, VerificationError};
use jolt_transcript::TranscriptError;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use support::*;

fn honest_proof(seed: u8) -> (ProtocolBackedInstance, Vec<u8>) {
    let mut rng = ChaCha20Rng::from_seed([seed; 32]);
    let instance = build_protocol_backed_instance(&mut rng);
    let narg = instance
        .prove_real(&mut rng)
        .expect("real prover succeeds on a valid witness");
    (instance, narg)
}

#[test]
fn real_prover_roundtrip_verifies() {
    let (instance, narg) = honest_proof(101);
    instance
        .fixture
        .verify(&narg)
        .expect("real prover's proof verifies");
}

/// Two final-opening bindings: the prover must send each binding's (output,
/// blinding) openings in the order the verifier reads them.
#[test]
fn real_prover_roundtrip_verifies_with_two_final_openings() {
    let mut rng = ChaCha20Rng::from_seed([107; 32]);
    let instance = build_protocol_backed_instance_with_bindings(&mut rng, 2);
    assert_eq!(instance.fixture.protocol.eval_commitments.len(), 2);
    let narg = instance
        .prove_real(&mut rng)
        .expect("real prover succeeds on a valid witness");
    instance
        .fixture
        .verify(&narg)
        .expect("real prover's proof verifies with two bindings");
}

#[test]
fn real_prover_rejects_missing_witness_row() {
    let mut rng = ChaCha20Rng::from_seed([103; 32]);
    let mut instance = build_protocol_backed_instance(&mut rng);
    let _ = instance.rows.pop();

    let err = instance
        .prove_real(&mut rng)
        .expect_err("row count mismatch must be rejected");
    assert!(
        matches!(
            err,
            ProverError::LengthMismatch {
                name: "witness rows",
                ..
            }
        ),
        "expected witness-row length mismatch, got: {err}"
    );
}

#[test]
fn real_prover_rejects_truncated_witness_row() {
    let mut rng = ChaCha20Rng::from_seed([104; 32]);
    let mut instance = build_protocol_backed_instance(&mut rng);
    let _ = instance.rows[0].pop();

    let err = instance
        .prove_real(&mut rng)
        .expect_err("row length mismatch must be rejected");
    assert!(
        matches!(err, ProverError::WitnessRowLengthMismatch { row: 0, .. }),
        "expected witness-row-0 length mismatch, got: {err}"
    );
}

#[test]
fn real_prover_rejects_eval_output_not_matching_commitment() {
    let mut rng = ChaCha20Rng::from_seed([105; 32]);
    let mut instance = build_protocol_backed_instance(&mut rng);
    instance.fixture.eval_outputs[0] += f(1);

    let err = instance
        .prove_real(&mut rng)
        .expect_err("evaluation output inconsistent with its commitment must be rejected");
    assert!(
        matches!(err, ProverError::EvalCommitmentMismatch { index: 0 }),
        "expected eval-commitment mismatch at index 0, got: {err}"
    );
}

/// Flips one bit in bytes spread across the whole proof, committed-stage
/// prefix included. The odd stride walks the flip through every position
/// within the fixed-width field and commitment atoms.
#[test]
fn every_sampled_byte_flip_is_rejected() {
    const FLIPS: usize = 512;
    let (instance, narg) = honest_proof(106);
    let stride = (narg.len() / FLIPS) | 1;

    let mut offsets = (0..narg.len()).step_by(stride).collect::<Vec<_>>();
    offsets.push(narg.len() - 1);
    for offset in offsets {
        let mut tampered = narg.clone();
        tampered[offset] ^= 1;
        assert!(
            instance.fixture.verify(&tampered).is_err(),
            "flipping byte {offset} of {} was accepted",
            narg.len()
        );
    }
}

#[test]
fn rejects_truncated_proof() {
    let (instance, narg) = honest_proof(107);

    let error = instance
        .fixture
        .verify(&narg[..narg.len() - 1])
        .expect_err("truncated proof is rejected");

    assert!(matches!(
        error,
        VerificationError::Transcript(TranscriptError::Truncated)
    ));
}

#[test]
fn rejects_trailing_bytes() {
    let (instance, mut narg) = honest_proof(108);
    narg.push(0);

    let error = instance
        .fixture
        .verify(&narg)
        .expect_err("trailing bytes are rejected");

    assert!(matches!(
        error,
        VerificationError::Transcript(TranscriptError::TrailingBytes)
    ));
}

#[test]
fn rejects_proof_under_another_session() {
    let (instance, narg) = honest_proof(109);
    let fixture = &instance.fixture;

    assert!(fixture
        .template
        .verify_with_session(&fixture.setup, b"another-session", &narg)
        .is_err());
}
