#![expect(
    clippy::expect_used,
    reason = "tests assert successful proof and serialization setup"
)]
#![expect(
    clippy::unwrap_used,
    reason = "benchmarks and tests unwrap successful PCS operations"
)]

mod support;

use jolt_akita::{
    AkitaBackendFlavor, AkitaBatchProof, AkitaCommitment, AkitaField, AkitaNativeBatchStatement,
    AkitaNativeBatching, AkitaScheme,
};
use jolt_field::Field;
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, OpeningsError, ZkBatchOpeningScheme, ZkOpeningScheme,
};
use jolt_poly::{MultilinearPoly, OneHotPolynomial, Point, Polynomial, HIGH_TO_LOW};
use jolt_transcript::{Blake2bTranscript, Transcript};
use serde_json::{json, Value};
use support::{batch_polynomials, f, layout, native_setup, polynomial, setup_for};

type VerifierSetup = <AkitaScheme as CommitmentScheme>::VerifierSetup;

fn require_jolt_field<F: Field>() {}

#[test]
fn akita_field_satisfies_jolt_field_bundle() {
    require_jolt_field::<AkitaField>();
    assert_eq!(f(3) + f(4), f(7));
    assert_eq!(f(3) * f(4), f(12));
}

#[test]
fn akita_public_commit_open_uses_sparse_one_hot_path() {
    assert!(!AkitaScheme::supports_unit_sparse_dimension(5));
    assert!(AkitaScheme::supports_unit_sparse_dimension(6));

    let num_vars = 13;
    let (prover_setup, verifier_setup) = setup_for(num_vars, 1, layout(7));
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
        AkitaBackendFlavor::Dense
    );
    assert_eq!(
        one_hot_commitment, dense_commitment,
        "public one-hot commitment must match the equivalent dense polynomial"
    );

    let point = (0..num_vars)
        .map(|index| f(index as u64 + 2))
        .collect::<Vec<_>>();
    let eval = one_hot.evaluate(&point);
    assert_eq!(eval, dense.evaluate(&point));

    let mut prover_transcript = Blake2bTranscript::new(b"akita-sparse-unit");
    let proof = AkitaScheme::open(
        &one_hot,
        &point,
        eval,
        &prover_setup,
        Some(one_hot_hint),
        &mut prover_transcript,
    )
    .unwrap();

    let mut verifier_transcript = Blake2bTranscript::new(b"akita-sparse-unit");
    AkitaScheme::verify(
        &one_hot_commitment,
        &point,
        eval,
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
    )
    .expect("sparse unit proof should verify");
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn akita_public_commit_open_uses_upstream_one_hot_path_for_k256() {
    let num_vars = 13;
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

    let mut prover_transcript = Blake2bTranscript::new(b"akita-native-one-hot");
    let proof = AkitaScheme::open(
        &one_hot,
        &point,
        eval,
        &prover_setup,
        Some(one_hot_hint),
        &mut prover_transcript,
    )
    .unwrap();

    let mut verifier_transcript = Blake2bTranscript::new(b"akita-native-one-hot");
    AkitaScheme::verify(
        &one_hot_commitment,
        &point,
        eval,
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
    )
    .expect("native Akita one-hot proof should verify");
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn akita_proof_payloads_reject_unknown_serialized_fields() {
    let (_, statement, proof) = native_proof_fixture(b"akita-payload-unknown-fields");

    let mut top_level = serde_json::to_value(&proof).expect("proof should serialize");
    let _ = top_level
        .as_object_mut()
        .expect("proof should serialize as object")
        .insert("unexpected".to_owned(), json!(true));
    assert!(serde_json::from_value::<AkitaBatchProof>(top_level).is_err());

    let commitment = &statement[0].commitment;
    let mut tampered = serde_json::to_value(commitment).expect("commitment should serialize");
    let _ = tampered
        .as_object_mut()
        .expect("commitment should serialize as object")
        .insert("unexpected".to_owned(), json!(true));
    assert!(serde_json::from_value::<AkitaCommitment>(tampered).is_err());
}

#[test]
fn akita_forged_shape_metadata_rejects_before_shape_backed_allocation() {
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
    let mut transcript = Blake2bTranscript::new(b"akita-forged-metadata");
    let err = <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        &forged_statement,
        &proof,
        &mut transcript,
    )
    .expect_err("forged backend_coeff_len should reject");
    assert!(
        matches!(&err, OpeningsError::InvalidBatch(message) if message.contains("coefficients")),
        "expected a shape-guard rejection, got: {err}"
    );

    // An oversized proof-shape blob must be rejected by the protocol cap
    // before shape deserialization ever runs.
    let mut oversized = proof.clone();
    let mut value = serde_json::to_value(&oversized).expect("proof should serialize");
    *value
        .get_mut("serialized_akita_proof_shape")
        .expect("proof should expose the shape blob") =
        serde_json::to_value(vec![0u8; 64 * 1024]).expect("blob should serialize");
    oversized = serde_json::from_value(value).expect("oversized proof should deserialize");
    let mut transcript = Blake2bTranscript::new(b"akita-forged-metadata");
    let err = <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        &statement,
        &oversized,
        &mut transcript,
    )
    .expect_err("oversized shape blob should reject");
    assert!(
        matches!(&err, OpeningsError::InvalidBatch(message) if message.contains("protocol cap")),
        "expected the shape-blob cap rejection, got: {err}"
    );
}

#[test]
fn akita_native_batching_rejects_corrupted_proof_payloads() {
    let (verifier_setup, statement, proof) = native_proof_fixture(b"akita-corrupt-proof");

    for field in [
        "statement_bridge",
        "serialized_akita_proof_shape",
        "serialized_akita_proof",
    ] {
        let tampered = mutate_byte_array_field(&proof, field);
        let mut transcript = Blake2bTranscript::new(b"akita-corrupt-proof");
        assert!(
            <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
                &verifier_setup,
                &statement.clone(),
                &tampered,
                &mut transcript,
            )
            .is_err(),
            "tampered {field} should reject"
        );
    }
}

#[test]
fn akita_zk_interfaces_are_explicitly_unsupported() {
    let (prover_setup, verifier_setup) = native_setup();
    let poly = polynomial(13, 1);
    let point: Vec<_> = (0..13).map(|i| f(2 + 3 * i)).collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) = AkitaScheme::commit_zk(&poly, &prover_setup).unwrap();

    let mut prover_transcript = Blake2bTranscript::new(b"akita-zk-unsupported");
    let (proof, _, ()) = AkitaScheme::open_zk(
        &poly,
        &point,
        eval,
        &prover_setup,
        hint.clone(),
        &mut prover_transcript,
    )
    .unwrap();

    let mut verifier_transcript = Blake2bTranscript::new(b"akita-zk-unsupported");
    assert_transparent_zk_error(AkitaScheme::verify_zk(
        &commitment,
        &point,
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
    ));

    let zk_point = Point::<HIGH_TO_LOW, _>::high_to_low(point);
    let mut transcript = Blake2bTranscript::new(b"akita-zk-batch-prove-unsup");
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

    let mut transcript = Blake2bTranscript::new(b"akita-zk-batch-verify-unsup");
    assert_transparent_zk_error(
        <AkitaNativeBatching as ZkBatchOpeningScheme>::verify_batch_zk(
            &verifier_setup,
            zk_point,
            vec![commitment],
            &proof,
            &mut transcript,
        ),
    );
}

fn native_proof_fixture(
    label: &'static [u8],
) -> (VerifierSetup, AkitaNativeBatchStatement, AkitaBatchProof) {
    let (prover_setup, verifier_setup) = native_setup();
    let poly_a = polynomial(13, 1);
    let poly_b = polynomial(13, 20);
    let point: Vec<_> = (0..13).map(|i| f(2 + 3 * i)).collect();
    let eval_a = poly_a.evaluate(&point);
    let eval_b = poly_b.evaluate(&point);
    let (commitment, hint) =
        AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), poly_b.clone()])
            .expect("grouped commit should succeed");
    let statement = support::native_statement(commitment, &point, [eval_a, eval_b]);

    let mut transcript = Blake2bTranscript::new(label);
    let proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement.clone(),
        batch_polynomials([&poly_a, &poly_b]),
        hint,
        &mut transcript,
    )
    .expect("black-box proof should be produced");
    (verifier_setup, statement, proof)
}

fn mutate_byte_array_field(proof: &AkitaBatchProof, field: &str) -> AkitaBatchProof {
    let mut value = serde_json::to_value(proof).expect("proof should serialize");
    let bytes = value
        .get_mut(field)
        .expect("proof should contain field")
        .as_array_mut()
        .expect("proof field should serialize as byte array");
    let first = bytes
        .first_mut()
        .expect("proof byte array should not be empty");
    let byte = u8::try_from(
        first
            .as_u64()
            .expect("proof byte array element should be a number"),
    )
    .expect("proof byte array element should fit in u8");
    *first = Value::from(byte ^ 1);
    serde_json::from_value(value).expect("mutated proof should still deserialize")
}

fn assert_transparent_zk_error<T>(result: Result<T, OpeningsError>) {
    assert!(
        matches!(result, Err(OpeningsError::InvalidBatch(message)) if message.contains("transparent-only")),
        "Akita ZK APIs should fail with the explicit transparent-only error"
    );
}
