#![expect(
    clippy::expect_used,
    reason = "tests assert successful proof and serialization setup"
)]
#![expect(
    clippy::unwrap_used,
    reason = "benchmarks and tests unwrap successful PCS operations"
)]

pub mod support;

use jolt_akita::{
    AkitaBackendFlavor, AkitaCommitment, AkitaNativeBatchStatement, AkitaNativeBatching,
    AkitaScheme,
};
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, OpeningsError, ZkBatchOpeningScheme, ZkOpeningScheme,
};
use jolt_poly::{MultilinearPoly, OneHotPolynomial, Point, Polynomial, HIGH_TO_LOW};
use jolt_transcript::TranscriptError;
use serde_json::json;
use support::{
    assert_transcripts_agree, batch_polynomials, f, layout, native_setup, new_prover_transcript,
    new_verifier_transcript, polynomial, setup_for,
};

type VerifierSetup = <AkitaScheme as CommitmentScheme>::VerifierSetup;

#[test]
fn akita_public_commit_rejects_unsupported_one_hot_shape() {
    let num_vars = 16;
    let (prover_setup, _) = setup_for(num_vars, 1, layout(7));
    let k = 4;
    let indices = (0..(1usize << num_vars) / k)
        .map(|row| {
            if row % 5 == 4 {
                None
            } else {
                Some((row % k) as u8)
            }
        })
        .collect::<Vec<_>>();
    let one_hot = OneHotPolynomial::new(k, indices);
    let error = AkitaScheme::commit(&one_hot, &prover_setup)
        .expect_err("unsupported one-hot K must reject before dense materialization");
    assert!(error.to_string().contains("row-major K=256"));
}

#[test]
fn akita_public_commit_open_uses_upstream_one_hot_path_for_k256() {
    let num_vars = 16;
    let (prover_setup, verifier_setup) = setup_for(num_vars, 1, layout(9));
    let k = 256;
    let indices = (0..(1usize << num_vars) / k)
        .map(|row| {
            if row % 7 == 3 {
                None
            } else {
                Some(((row * 11) % k) as u8)
            }
        })
        .collect::<Vec<_>>();
    let one_hot = OneHotPolynomial::new(k, indices.clone());
    let mut dense = vec![f(0); 1 << num_vars];
    for (row, col) in indices.iter().enumerate() {
        if let Some(col) = col {
            dense[row * k + *col as usize] = f(1);
        }
    }
    let dense = Polynomial::new(dense);
    let (one_hot_commitment, one_hot_hint) = AkitaScheme::commit(&one_hot, &prover_setup).unwrap();
    let (dense_commitment, _) = AkitaScheme::commit(&dense, &prover_setup).unwrap();
    assert_eq!(
        one_hot_commitment.backend_flavor(),
        AkitaBackendFlavor::OneHot
    );
    assert_eq!(dense_commitment.backend_flavor(), AkitaBackendFlavor::Dense);
    assert_ne!(
        one_hot_commitment, dense_commitment,
        "native Akita one-hot uses a separate backend setup from dense commitments"
    );

    let point = (0..num_vars)
        .map(|index| f(index as u64 + 3))
        .collect::<Vec<_>>();
    let eval = one_hot.evaluate(&point);
    assert_eq!(eval, dense.evaluate(&point));

    let mut prover_transcript = new_prover_transcript(b"akita-native-one-hot");
    AkitaScheme::open(
        &one_hot,
        &point,
        eval,
        &prover_setup,
        Some(one_hot_hint),
        &mut prover_transcript,
    )
    .unwrap();
    let proof = prover_transcript.narg().to_vec();

    let mut verifier_transcript = new_verifier_transcript(b"akita-native-one-hot", &proof);
    AkitaScheme::verify(
        &one_hot_commitment,
        &point,
        eval,
        &verifier_setup,
        &mut verifier_transcript,
    )
    .expect("native Akita one-hot proof should verify");
    assert_transcripts_agree(prover_transcript, verifier_transcript);
}

#[test]
fn akita_commitments_reject_unknown_serialized_fields() {
    let (_, statement, _) = native_proof_fixture(b"akita-payload-unknown-fields");

    let commitment = &statement[0].commitment;
    let mut tampered = serde_json::to_value(commitment).expect("commitment should serialize");
    let _ = tampered
        .as_object_mut()
        .expect("commitment should serialize as object")
        .insert("unexpected".to_owned(), json!(true));
    assert!(serde_json::from_value::<AkitaCommitment>(tampered).is_err());
}

#[test]
fn akita_forged_commitment_metadata_rejects_before_shape_backed_allocation() {
    let (verifier_setup, statement, proof) = native_proof_fixture(b"akita-forged-metadata");

    // Forge the commitment's declared coefficient count to the upstream
    // deserializer's 2^25 cap: without the shape guard this would reserve
    // ~512 MiB before hitting EOF. The statement must be internally
    // consistent, so every claim carries the forged commitment.
    let mut forged =
        serde_json::to_value(&statement[0].commitment).expect("commitment should serialize");
    *forged
        .get_mut("backend_coeff_len")
        .expect("commitment should expose backend_coeff_len") = json!(1u64 << 25);
    let forged: AkitaCommitment =
        serde_json::from_value(forged).expect("forged commitment should deserialize");
    let forged_statement: AkitaNativeBatchStatement = statement
        .iter()
        .map(|claim| jolt_openings::VerifierOpeningClaim {
            commitment: forged.clone(),
            evaluation: claim.evaluation.clone(),
        })
        .collect();
    let mut transcript = new_verifier_transcript(b"akita-forged-metadata", &proof);
    let err = <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        &forged_statement,
        &mut transcript,
    )
    .expect_err("forged backend_coeff_len should reject");
    assert!(
        matches!(&err, OpeningsError::InvalidBatch(message) if message.contains("coefficients")),
        "expected a shape-guard rejection, got: {err}"
    );
}

#[test]
fn akita_native_batching_rejects_corrupted_proof_payloads() {
    let (verifier_setup, statement, proof) = native_proof_fixture(b"akita-corrupt-proof");

    // The selected schedule row leads the opening; Akita's messages follow.
    for position in [0, 32, proof.len() / 2, proof.len() - 1] {
        let mut tampered = proof.clone();
        tampered[position] ^= 1;
        let mut transcript = new_verifier_transcript(b"akita-corrupt-proof", &tampered);
        assert!(
            <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
                &verifier_setup,
                &statement,
                &mut transcript,
            )
            .is_err(),
            "a flipped bit at proof byte {position} should reject"
        );
    }
}

/// A commitment's wire form decodes only from the exact bytes
/// `send_commitment` writes: a known flavor, the flavor's chunk size,
/// canonical coefficients, and nothing after them.
#[test]
fn akita_commitment_wire_form_rejects_noncanonical_encodings() {
    let (verifier_setup, statement, _) = native_proof_fixture(b"akita-commitment-wire");
    let commitment = &statement[0].commitment;
    let mut prover_transcript = new_prover_transcript(b"akita-commitment-wire");
    AkitaScheme::send_commitment(commitment, &mut prover_transcript);
    let wire = prover_transcript.finish();
    let receive = |bytes: &[u8]| {
        let mut transcript = new_verifier_transcript(b"akita-commitment-wire", bytes);
        let received = AkitaScheme::receive_commitment(&verifier_setup, &mut transcript)?;
        transcript.finish()?;
        Ok::<_, OpeningsError>(received)
    };
    assert_eq!(receive(&wire).as_ref(), Ok(commitment));

    // Header: flavor tag, 32-byte layout digest, then u64 num_vars,
    // poly_count, one_hot_k, and coefficient count.
    let mut unknown_flavor = wire.clone();
    unknown_flavor[0] = 2;
    assert!(receive(&unknown_flavor).is_err(), "unknown flavor tag");

    let mut dense_chunk_size = wire.clone();
    dense_chunk_size[1 + 32 + 16] = 16;
    assert!(
        receive(&dense_chunk_size).is_err(),
        "a dense commitment must carry no one-hot chunk size"
    );

    let mut noncanonical = wire.clone();
    let coefficient_start = noncanonical.len() - 16;
    noncanonical[coefficient_start..].fill(0xff);
    assert!(
        receive(&noncanonical).is_err(),
        "a coefficient at or above the modulus"
    );

    let mut trailing = wire.clone();
    trailing.push(0);
    assert_eq!(
        receive(&trailing),
        Err(OpeningsError::Transcript(TranscriptError::TrailingBytes))
    );
    assert!(
        receive(&wire[..wire.len() - 1]).is_err(),
        "truncated payload"
    );
}

#[test]
fn akita_zk_interfaces_are_explicitly_unsupported() {
    let (prover_setup, verifier_setup) = native_setup();
    let poly = polynomial(16, 1);
    let point: Vec<_> = (0..16).map(|i| f(2 + 3 * i)).collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) = AkitaScheme::commit_zk(&poly, &prover_setup).unwrap();

    let mut prover_transcript = new_prover_transcript(b"akita-zk-unsupported");
    let _ = AkitaScheme::open_zk(
        &poly,
        &point,
        eval,
        &prover_setup,
        hint.clone(),
        &mut prover_transcript,
    )
    .unwrap();
    let proof = prover_transcript.finish();

    let mut verifier_transcript = new_verifier_transcript(b"akita-zk-unsupported", &proof);
    assert_transparent_zk_error(AkitaScheme::verify_zk(
        &commitment,
        &point,
        &verifier_setup,
        &mut verifier_transcript,
    ));

    let zk_point = Point::<HIGH_TO_LOW, _>::high_to_low(point);
    let mut transcript = new_prover_transcript(b"akita-zk-batch-prove-unsup");
    assert_transparent_zk_error(
        <AkitaNativeBatching as ZkBatchOpeningScheme>::prove_batch_zk(
            &prover_setup,
            zk_point.clone(),
            vec![commitment.clone()],
            batch_polynomials([&poly]),
            hint,
            vec![eval],
            &mut transcript,
        ),
    );

    let mut transcript = new_verifier_transcript(b"akita-zk-batch-verify-unsup", &proof);
    assert_transparent_zk_error(
        <AkitaNativeBatching as ZkBatchOpeningScheme>::verify_batch_zk(
            &verifier_setup,
            zk_point,
            vec![commitment],
            &mut transcript,
        ),
    );
}

fn native_proof_fixture(
    label: &'static [u8],
) -> (VerifierSetup, AkitaNativeBatchStatement, Vec<u8>) {
    let (prover_setup, verifier_setup) = native_setup();
    let poly_a = polynomial(16, 1);
    let poly_b = polynomial(16, 20);
    let point: Vec<_> = (0..16).map(|i| f(2 + 3 * i)).collect();
    let eval_a = poly_a.evaluate(&point);
    let eval_b = poly_b.evaluate(&point);
    let (commitment, hint) =
        AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), poly_b.clone()])
            .expect("grouped commit should succeed");
    let statement = support::native_statement(commitment, &point, [eval_a, eval_b]);

    let mut transcript = new_prover_transcript(label);
    <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement.clone(),
        batch_polynomials([&poly_a, &poly_b]),
        hint,
        &mut transcript,
    )
    .expect("black-box proof should be produced");
    (verifier_setup, statement, transcript.finish())
}

fn assert_transparent_zk_error<T>(result: Result<T, OpeningsError>) {
    assert!(
        matches!(result, Err(OpeningsError::InvalidBatch(message)) if message.contains("transparent-only")),
        "Akita ZK APIs should fail with the explicit transparent-only error"
    );
}
