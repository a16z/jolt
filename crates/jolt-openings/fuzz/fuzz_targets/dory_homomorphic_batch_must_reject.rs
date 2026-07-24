#![no_main]

//! Structured coverage for homomorphic Dory batch openings.
//!
//! The target builds honest same-point batch openings over byte-controlled
//! polynomials, then mutates public statements and prover-side source lists.
//! This reaches the batching adapter plus Dory's transparent and hiding
//! opening APIs with inputs that stay well-formed up to the mutation boundary.

use std::sync::OnceLock;

use jolt_dory::{DoryCommitment, DoryHint, DoryProverSetup, DoryScheme, DoryVerifierSetup};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, EvaluationClaim, HomomorphicBatch, VerifierOpeningClaim,
    ZkBatchOpeningScheme, ZkOpeningScheme,
};
use jolt_poly::{MultilinearPoly, Point, Polynomial, HIGH_TO_LOW};
use jolt_transcript::{Blake2bTranscript, Transcript};
use libfuzzer_sys::fuzz_target;

const MAX_NUM_VARS: usize = 4;
const CLEAR_LABEL: &[u8] = b"fuzz-openings-clear";
const ZK_LABEL: &[u8] = b"fuzz-openings-zk";

type DoryBatch = HomomorphicBatch<DoryScheme>;
type DoryClaim = VerifierOpeningClaim<Fr, DoryCommitment>;

fn word(data: &[u8], cursor: &mut usize) -> u64 {
    let mut bytes = [0u8; 8];
    for byte in &mut bytes {
        *byte = data[*cursor % data.len()];
        *cursor += 1;
    }
    u64::from_le_bytes(bytes)
}

fn field(data: &[u8], cursor: &mut usize) -> Fr {
    Fr::from_u64(word(data, cursor))
}

fn setups(num_vars: usize) -> &'static (DoryProverSetup, DoryVerifierSetup) {
    static SETUPS: OnceLock<Vec<(DoryProverSetup, DoryVerifierSetup)>> = OnceLock::new();
    &SETUPS.get_or_init(|| {
        (1..=MAX_NUM_VARS)
            .map(|num_vars| {
                (
                    DoryScheme::setup_prover(num_vars),
                    DoryScheme::setup_verifier(num_vars),
                )
            })
            .collect()
    })[num_vars - 1]
}

fn sources(polynomials: &[Polynomial<Fr>]) -> Vec<&dyn MultilinearPoly<Fr>> {
    polynomials
        .iter()
        .map(|polynomial| polynomial as &dyn MultilinearPoly<Fr>)
        .collect()
}

fn commit_clear(
    polynomials: &[Polynomial<Fr>],
    point: &Point<HIGH_TO_LOW, Fr>,
    setup: &DoryProverSetup,
) -> (Vec<DoryClaim>, Vec<DoryHint>) {
    let mut claims = Vec::with_capacity(polynomials.len());
    let mut hints = Vec::with_capacity(polynomials.len());
    for polynomial in polynomials {
        let (commitment, hint) = DoryScheme::commit(polynomial.evaluations(), setup)
            .unwrap_or_else(|error| panic!("Dory clear commit failed: {error}"));
        claims.push(DoryClaim {
            commitment,
            evaluation: EvaluationClaim::new(point.clone(), polynomial.evaluate(point)),
        });
        hints.push(hint);
    }
    (claims, hints)
}

fn commit_zk(
    polynomials: &[Polynomial<Fr>],
    point: &Point<HIGH_TO_LOW, Fr>,
    setup: &DoryProverSetup,
) -> (Vec<DoryCommitment>, Vec<DoryHint>, Vec<Fr>) {
    let mut commitments = Vec::with_capacity(polynomials.len());
    let mut hints = Vec::with_capacity(polynomials.len());
    let mut evaluations = Vec::with_capacity(polynomials.len());
    for polynomial in polynomials {
        let (commitment, hint) = DoryScheme::commit_zk(polynomial.evaluations(), setup)
            .unwrap_or_else(|error| panic!("Dory ZK commit failed: {error}"));
        commitments.push(commitment);
        hints.push(hint);
        evaluations.push(polynomial.evaluate(point));
    }
    (commitments, hints, evaluations)
}

fn prove_clear(
    setup: &DoryProverSetup,
    claims: Vec<DoryClaim>,
    polynomials: &[Polynomial<Fr>],
    hints: Vec<DoryHint>,
) -> <DoryBatch as BatchOpeningScheme>::Proof {
    let mut transcript = Blake2bTranscript::new(CLEAR_LABEL);
    <DoryBatch as BatchOpeningScheme>::prove_batch(
        setup,
        claims,
        sources(polynomials),
        hints,
        &mut transcript,
    )
    .unwrap_or_else(|error| panic!("Dory clear batch proof failed: {error}"))
}

fn shifted_point(point: &Point<HIGH_TO_LOW, Fr>) -> Point<HIGH_TO_LOW, Fr> {
    let mut shifted = point.as_slice().to_vec();
    shifted[0] += Fr::from_u64(1);
    Point::new(shifted)
}

fn alternate_polynomial(polynomial: &Polynomial<Fr>) -> Polynomial<Fr> {
    let mut evals = polynomial.evaluations().to_vec();
    evals[0] += Fr::from_u64(1);
    Polynomial::new(evals)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    let num_vars = 1 + data[0] as usize % MAX_NUM_VARS;
    let polynomial_count = 2 + data[1] as usize % 4;
    let mut cursor = 3;
    let len = 1usize << num_vars;
    let polynomials: Vec<_> = (0..polynomial_count)
        .map(|_| Polynomial::new((0..len).map(|_| field(data, &mut cursor)).collect()))
        .collect();
    let point = Point::new(
        (0..num_vars)
            .map(|_| field(data, &mut cursor))
            .collect::<Vec<_>>(),
    );
    let (prover_setup, verifier_setup) = setups(num_vars);

    match data[2] % 10 {
        0 => {
            let (claims, hints) = commit_clear(&polynomials, &point, prover_setup);
            let mut prover_transcript = Blake2bTranscript::new(CLEAR_LABEL);
            let proof = <DoryBatch as BatchOpeningScheme>::prove_batch(
                prover_setup,
                claims.clone(),
                sources(&polynomials),
                hints,
                &mut prover_transcript,
            )
            .unwrap_or_else(|error| panic!("Dory clear batch proof failed: {error}"));

            let mut verifier_transcript = Blake2bTranscript::new(CLEAR_LABEL);
            <DoryBatch as BatchOpeningScheme>::verify_batch(
                verifier_setup,
                &claims,
                &proof,
                &mut verifier_transcript,
            )
            .unwrap_or_else(|error| panic!("honest clear batch verification failed: {error}"));
            assert_eq!(prover_transcript.state(), verifier_transcript.state());
        }
        1 => {
            let (mut claims, hints) = commit_clear(&polynomials, &point, prover_setup);
            let proof = prove_clear(prover_setup, claims.clone(), &polynomials, hints);
            claims[0].evaluation.value += Fr::from_u64(1);

            let mut transcript = Blake2bTranscript::new(CLEAR_LABEL);
            assert!(
                <DoryBatch as BatchOpeningScheme>::verify_batch(
                    verifier_setup,
                    &claims,
                    &proof,
                    &mut transcript,
                )
                .is_err(),
                "clear batch accepted a tampered evaluation"
            );
        }
        2 => {
            let (claims, hints) = commit_clear(&polynomials, &point, prover_setup);
            let proof = prove_clear(prover_setup, claims.clone(), &polynomials, hints);
            let alternate = alternate_polynomial(&polynomials[0]);
            let (wrong_commitment, _) = DoryScheme::commit(alternate.evaluations(), prover_setup)
                .unwrap_or_else(|error| panic!("alternate clear commit failed: {error}"));
            if wrong_commitment == claims[0].commitment {
                return;
            }

            let mut tampered = claims;
            tampered[0].commitment = wrong_commitment;
            let mut transcript = Blake2bTranscript::new(CLEAR_LABEL);
            assert!(
                <DoryBatch as BatchOpeningScheme>::verify_batch(
                    verifier_setup,
                    &tampered,
                    &proof,
                    &mut transcript,
                )
                .is_err(),
                "clear batch accepted a substituted commitment"
            );
        }
        3 => {
            let (mut claims, hints) = commit_clear(&polynomials, &point, prover_setup);
            claims[1].evaluation.point = shifted_point(&point);
            let mut transcript = Blake2bTranscript::new(CLEAR_LABEL);
            assert!(
                <DoryBatch as BatchOpeningScheme>::prove_batch(
                    prover_setup,
                    claims,
                    sources(&polynomials),
                    hints,
                    &mut transcript,
                )
                .is_err(),
                "clear batch accepted claims at different points"
            );
        }
        4 => {
            let (claims, hints) = commit_clear(&polynomials, &point, prover_setup);
            let wrong = Polynomial::new(vec![Fr::from_u64(1); 1usize << (num_vars + 1)]);
            let mut polynomial_sources = sources(&polynomials);
            polynomial_sources[0] = &wrong;
            let mut transcript = Blake2bTranscript::new(CLEAR_LABEL);
            assert!(
                <DoryBatch as BatchOpeningScheme>::prove_batch(
                    prover_setup,
                    claims,
                    polynomial_sources,
                    hints,
                    &mut transcript,
                )
                .is_err(),
                "clear batch accepted a wrong-dimension witness"
            );
        }
        5 => {
            let (claims, mut hints) = commit_clear(&polynomials, &point, prover_setup);
            let mut polynomial_sources = sources(&polynomials);
            polynomial_sources.pop();
            hints.pop();
            let mut transcript = Blake2bTranscript::new(CLEAR_LABEL);
            assert!(
                <DoryBatch as BatchOpeningScheme>::prove_batch(
                    prover_setup,
                    claims,
                    polynomial_sources,
                    hints,
                    &mut transcript,
                )
                .is_err(),
                "clear batch accepted missing prover sources"
            );
        }
        6 => {
            let (claims, hints) = commit_clear(&polynomials, &point, prover_setup);
            let proof = prove_clear(prover_setup, claims.clone(), &polynomials, hints);
            let mut transcript = Blake2bTranscript::new(b"fuzz-openings-clear-alt");
            assert!(
                <DoryBatch as BatchOpeningScheme>::verify_batch(
                    verifier_setup,
                    &claims,
                    &proof,
                    &mut transcript,
                )
                .is_err(),
                "clear batch accepted a proof under a different transcript"
            );
        }
        7 => {
            let (commitments, hints, evaluations) = commit_zk(&polynomials, &point, prover_setup);
            let mut prover_transcript = Blake2bTranscript::new(ZK_LABEL);
            let (proof, hiding_commitment, _blind) =
                <DoryBatch as ZkBatchOpeningScheme>::prove_batch_zk(
                    prover_setup,
                    point.clone(),
                    commitments.clone(),
                    sources(&polynomials),
                    hints,
                    evaluations,
                    &mut prover_transcript,
                )
                .unwrap_or_else(|error| panic!("Dory ZK batch proof failed: {error}"));

            let mut verifier_transcript = Blake2bTranscript::new(ZK_LABEL);
            let verifier_hiding = <DoryBatch as ZkBatchOpeningScheme>::verify_batch_zk(
                verifier_setup,
                point,
                commitments,
                &proof,
                &mut verifier_transcript,
            )
            .unwrap_or_else(|error| panic!("honest ZK batch verification failed: {error}"));
            assert_eq!(hiding_commitment, verifier_hiding);
            assert_eq!(prover_transcript.state(), verifier_transcript.state());
        }
        8 => {
            let (commitments, mut hints, mut evaluations) =
                commit_zk(&polynomials, &point, prover_setup);
            let mut polynomial_sources = sources(&polynomials);
            polynomial_sources.pop();
            hints.pop();
            evaluations.pop();
            let mut transcript = Blake2bTranscript::new(ZK_LABEL);
            assert!(
                <DoryBatch as ZkBatchOpeningScheme>::prove_batch_zk(
                    prover_setup,
                    point,
                    commitments,
                    polynomial_sources,
                    hints,
                    evaluations,
                    &mut transcript,
                )
                .is_err(),
                "ZK batch accepted missing prover sources"
            );
        }
        _ => {
            let (mut commitments, hints, evaluations) =
                commit_zk(&polynomials, &point, prover_setup);
            let mut prover_transcript = Blake2bTranscript::new(ZK_LABEL);
            let (proof, _hiding_commitment, _blind) =
                <DoryBatch as ZkBatchOpeningScheme>::prove_batch_zk(
                    prover_setup,
                    point.clone(),
                    commitments.clone(),
                    sources(&polynomials),
                    hints,
                    evaluations,
                    &mut prover_transcript,
                )
                .unwrap_or_else(|error| panic!("Dory ZK batch proof failed: {error}"));

            let alternate = alternate_polynomial(&polynomials[0]);
            let (wrong_commitment, _) = DoryScheme::commit_zk(alternate.evaluations(), prover_setup)
                .unwrap_or_else(|error| panic!("alternate ZK commit failed: {error}"));
            if wrong_commitment == commitments[0] {
                return;
            }
            commitments[0] = wrong_commitment;

            let mut verifier_transcript = Blake2bTranscript::new(ZK_LABEL);
            assert!(
                <DoryBatch as ZkBatchOpeningScheme>::verify_batch_zk(
                    verifier_setup,
                    point,
                    commitments,
                    &proof,
                    &mut verifier_transcript,
                )
                .is_err(),
                "ZK batch accepted a substituted commitment"
            );
        }
    }
});
