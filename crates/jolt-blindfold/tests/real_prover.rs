//! Drives the shipped prover (`jolt_blindfold::prove`) end-to-end against the
//! real verifier. The rest of the suite exercises the verifier via the test
//! harness's reference prover; these tests are what tie the production
//! folding/Spartan prover itself to the verifier's acceptance criteria.

#![expect(clippy::expect_used, reason = "integration tests should fail loudly")]

mod support;

use jolt_blindfold::{prove, BlindFoldProof, BlindFoldWitness, ProverError, VerificationError};
use jolt_crypto::Bn254G1;
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use support::*;

fn prove_real(
    instance: &ProtocolBackedInstance,
    rng: &mut ChaCha20Rng,
) -> Result<BlindFoldProof<F, Bn254G1>, ProverError<F>> {
    let mut transcript = Blake2bTranscript::<F>::new(PROTOCOL_BACKED_TRANSCRIPT_LABEL);
    append_protocol_transcript_prefix(&instance.protocol, &mut transcript);
    prove::<F, VC, _, _>(
        &instance.setup,
        &instance.protocol,
        &mut transcript,
        BlindFoldWitness {
            rows: &instance.rows,
            blindings: &instance.blindings,
            eval_outputs: &instance.eval_outputs,
            eval_blindings: &instance.eval_blindings,
        },
        rng,
    )
}

fn verify_real(
    instance: &ProtocolBackedInstance,
    proof: &BlindFoldProof<F, Bn254G1>,
) -> Result<(), VerificationError<F>> {
    let mut transcript = Blake2bTranscript::<F>::new(PROTOCOL_BACKED_TRANSCRIPT_LABEL);
    append_protocol_transcript_prefix(&instance.protocol, &mut transcript);
    instance
        .protocol
        .verify::<VC, _>(proof, &instance.setup, &mut transcript)
}

#[test]
fn real_prover_roundtrip_verifies() {
    let mut rng = ChaCha20Rng::from_seed([101; 32]);
    let instance = build_protocol_backed_instance(&mut rng);
    let proof = prove_real(&instance, &mut rng).expect("real prover succeeds on a valid witness");
    verify_real(&instance, &proof).expect("real prover's proof verifies");
}

#[test]
fn real_prover_roundtrip_verifies_with_two_final_openings() {
    let mut rng = ChaCha20Rng::from_seed([107; 32]);
    let instance = build_protocol_backed_instance_with_bindings(&mut rng, 2);
    assert_eq!(instance.protocol.eval_commitments.len(), 2);
    let proof = prove_real(&instance, &mut rng).expect("real prover succeeds on a valid witness");
    verify_real(&instance, &proof).expect("real prover's proof verifies with two bindings");
}

#[test]
fn real_prover_rejects_missing_witness_row() {
    let mut rng = ChaCha20Rng::from_seed([103; 32]);
    let mut instance = build_protocol_backed_instance(&mut rng);
    let _ = instance.rows.pop();

    let err = prove_real(&instance, &mut rng).expect_err("row count mismatch must be rejected");
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

    let err = prove_real(&instance, &mut rng).expect_err("row length mismatch must be rejected");
    assert!(
        matches!(err, ProverError::WitnessRowLengthMismatch { row: 0, .. }),
        "expected witness-row-0 length mismatch, got: {err}"
    );
}

#[test]
fn real_prover_rejects_eval_output_not_matching_commitment() {
    let mut rng = ChaCha20Rng::from_seed([105; 32]);
    let mut instance = build_protocol_backed_instance(&mut rng);
    instance.eval_outputs[0] += f(1);

    let err = prove_real(&instance, &mut rng)
        .expect_err("evaluation output inconsistent with its commitment must be rejected");
    assert!(
        matches!(err, ProverError::EvalCommitmentMismatch { index: 0 }),
        "expected eval-commitment mismatch at index 0, got: {err}"
    );
}
