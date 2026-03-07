//! Cross-crate integration tests: RlcReduction + DoryScheme.
//!
//! Exercises the full Stage-8 pipeline — accumulate claims, RLC reduce, combine
//! hints, open, verify — using real Dory proofs instead of MockCommitmentScheme.

#![allow(unused_results)]

use jolt_dory::DoryScheme;
use jolt_field::{Field, Fr};
use jolt_openings::{
    AdditivelyHomomorphic, CommitmentScheme, OpeningReduction, ProverClaim, RlcReduction,
    VerifierClaim,
};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, KeccakTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn challenge_fn(c: u128) -> Fr {
    Fr::from_u128(c)
}

fn rho_powers(rho: Fr, n: usize) -> Vec<Fr> {
    std::iter::successors(Some(Fr::from_u64(1)), |prev| Some(*prev * rho))
        .take(n)
        .collect()
}

/// Groups items by their point (first element), preserving insertion order.
/// Mirrors the grouping logic inside `RlcReduction::reduce_prover`.
fn group_by_point<T>(items: Vec<(Vec<Fr>, T)>) -> Vec<(Vec<Fr>, Vec<T>)> {
    let mut groups: Vec<(Vec<Fr>, Vec<T>)> = Vec::new();
    for (point, item) in items {
        if let Some((_, group)) = groups.iter_mut().find(|(p, _)| *p == point) {
            group.push(item);
        } else {
            groups.push((point, vec![item]));
        }
    }
    groups
}

type DoryHintType = <DoryScheme as CommitmentScheme>::OpeningHint;
type DoryProofType = <DoryScheme as CommitmentScheme>::Proof;
type DoryProverSetupType = <DoryScheme as CommitmentScheme>::ProverSetup;
type DoryVerifierSetupType = <DoryScheme as CommitmentScheme>::VerifierSetup;

/// Full RLC reduction → Dory open → Dory verify pipeline.
///
/// When `use_hints` is true, hints are combined via `DoryScheme::combine_hints`
/// using rho powers replayed from a cloned transcript. When false, hints are
/// discarded and `None` is passed to `DoryScheme::open`.
fn pipeline_round_trip<T: Transcript<Challenge = u128>>(
    polys: &[Polynomial<Fr>],
    points: &[Vec<Fr>],
    prover_setup: &DoryProverSetupType,
    verifier_setup: &DoryVerifierSetupType,
    use_hints: bool,
    label: &'static [u8],
) -> Result<(), jolt_openings::OpeningsError> {
    assert_eq!(polys.len(), points.len());

    let mut prover_claims = Vec::new();
    let mut verifier_claims = Vec::new();
    // Collect (point, hint) pairs for hint threading
    let mut hint_pairs: Vec<(Vec<Fr>, DoryHintType)> = Vec::new();

    for (poly, point) in polys.iter().zip(points.iter()) {
        let eval = poly.evaluate(point);
        let (commitment, hint) = DoryScheme::commit(poly.evaluations(), prover_setup);

        prover_claims.push(ProverClaim {
            evaluations: poly.evaluations().to_vec(),
            point: point.clone(),
            eval,
        });
        verifier_claims.push(VerifierClaim {
            commitment,
            point: point.clone(),
            eval,
        });
        hint_pairs.push((point.clone(), hint));
    }

    // Clone transcript before reduce_prover so we can replay challenges for hints
    let mut transcript_p = T::new(label);
    let transcript_hint = transcript_p.clone();

    let (reduced_p, ()) = <RlcReduction as OpeningReduction<DoryScheme>>::reduce_prover(
        prover_claims,
        &mut transcript_p,
        challenge_fn,
    );

    // Replay challenges on the cloned transcript to recover per-group rho values,
    // then combine hints per group using the same rho powers the reduction used.
    let combined_hints: Vec<Option<DoryHintType>> = if use_hints {
        let hint_groups = group_by_point(hint_pairs);
        let mut replay = transcript_hint;
        hint_groups
            .into_iter()
            .map(|(_point, group_hints)| {
                let rho = challenge_fn(replay.challenge());
                let powers = rho_powers(rho, group_hints.len());
                Some(DoryScheme::combine_hints(group_hints, &powers))
            })
            .collect()
    } else {
        vec![None; reduced_p.len()]
    };

    // Open each reduced claim
    let proofs: Vec<DoryProofType> = reduced_p
        .iter()
        .zip(combined_hints)
        .map(|(claim, hint)| {
            let poly: Polynomial<Fr> = claim.evaluations.clone().into();
            DoryScheme::open(
                &poly,
                &claim.point,
                claim.eval,
                prover_setup,
                hint,
                &mut transcript_p,
            )
        })
        .collect();

    // Verifier side
    let mut transcript_v = T::new(label);
    let reduced_v = <RlcReduction as OpeningReduction<DoryScheme>>::reduce_verifier(
        verifier_claims,
        &(),
        &mut transcript_v,
        challenge_fn,
    )?;

    assert_eq!(reduced_v.len(), proofs.len());
    for (claim, proof) in reduced_v.iter().zip(proofs.iter()) {
        DoryScheme::verify(
            &claim.commitment,
            &claim.point,
            claim.eval,
            proof,
            verifier_setup,
            &mut transcript_v,
        )?;
    }

    Ok(())
}

#[test]
fn single_claim_through_pipeline() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(100);
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);

    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

    pipeline_round_trip::<Blake2bTranscript>(
        &[poly],
        &[point],
        &prover_setup,
        &verifier_setup,
        true,
        b"single-claim",
    )
    .expect("single claim pipeline must verify");
}

#[test]
fn multiple_claims_same_point() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(200);
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);

    let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let polys: Vec<_> = (0..4)
        .map(|_| Polynomial::<Fr>::random(num_vars, &mut rng))
        .collect();
    let points: Vec<_> = (0..4).map(|_| point.clone()).collect();

    pipeline_round_trip::<Blake2bTranscript>(
        &polys,
        &points,
        &prover_setup,
        &verifier_setup,
        true,
        b"same-point",
    )
    .expect("same-point pipeline must verify");
}

#[test]
fn multiple_claims_distinct_points() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(300);
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);

    let polys: Vec<_> = (0..3)
        .map(|_| Polynomial::<Fr>::random(num_vars, &mut rng))
        .collect();
    let points: Vec<Vec<Fr>> = (0..3)
        .map(|_| (0..num_vars).map(|_| Fr::random(&mut rng)).collect())
        .collect();

    pipeline_round_trip::<Blake2bTranscript>(
        &polys,
        &points,
        &prover_setup,
        &verifier_setup,
        true,
        b"distinct-points",
    )
    .expect("distinct-points pipeline must verify");
}

#[test]
fn mixed_shared_and_distinct_points() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(400);
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);

    let point_a: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let point_b: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

    let polys: Vec<_> = (0..5)
        .map(|_| Polynomial::<Fr>::random(num_vars, &mut rng))
        .collect();
    let points = vec![
        point_a.clone(),
        point_a.clone(),
        point_a,
        point_b.clone(),
        point_b,
    ];

    pipeline_round_trip::<Blake2bTranscript>(
        &polys,
        &points,
        &prover_setup,
        &verifier_setup,
        true,
        b"mixed-points",
    )
    .expect("mixed-points pipeline must verify");
}

#[test]
fn hint_threading_matches_hintless() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(500);
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);

    let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let polys: Vec<_> = (0..3)
        .map(|_| Polynomial::<Fr>::random(num_vars, &mut rng))
        .collect();
    let points: Vec<_> = (0..3).map(|_| point.clone()).collect();

    pipeline_round_trip::<Blake2bTranscript>(
        &polys,
        &points,
        &prover_setup,
        &verifier_setup,
        true,
        b"hint-with",
    )
    .expect("with-hint path must verify");

    pipeline_round_trip::<Blake2bTranscript>(
        &polys,
        &points,
        &prover_setup,
        &verifier_setup,
        false,
        b"hint-without",
    )
    .expect("without-hint path must verify");
}

#[test]
fn combined_hint_consistency() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(600);
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);

    let poly_a = Polynomial::<Fr>::random(num_vars, &mut rng);
    let poly_b = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

    let (commit_a, hint_a) = DoryScheme::commit(poly_a.evaluations(), &prover_setup);
    let (commit_b, hint_b) = DoryScheme::commit(poly_b.evaluations(), &prover_setup);

    let rho = Fr::from_u64(42);
    let powers = rho_powers(rho, 2);

    // Combine commitments via homomorphism
    let combined_commit = DoryScheme::combine(&[commit_a, commit_b], &powers);
    let combined_hint = DoryScheme::combine_hints(vec![hint_a, hint_b], &powers);

    // Build the combined polynomial manually
    let combined_evals: Vec<Fr> = poly_a
        .evaluations()
        .iter()
        .zip(poly_b.evaluations().iter())
        .map(|(a, b)| powers[0] * *a + powers[1] * *b)
        .collect();
    let combined_poly = Polynomial::new(combined_evals);
    let combined_eval = combined_poly.evaluate(&point);

    // Open with combined hint and verify against combined commitment
    let mut pt = Blake2bTranscript::new(b"hint-consistency");
    let proof = DoryScheme::open(
        &combined_poly,
        &point,
        combined_eval,
        &prover_setup,
        Some(combined_hint),
        &mut pt,
    );

    let mut vt = Blake2bTranscript::new(b"hint-consistency");
    DoryScheme::verify(
        &combined_commit,
        &point,
        combined_eval,
        &proof,
        &verifier_setup,
        &mut vt,
    )
    .expect("RLC-combined commitment+hint must verify");
}

#[test]
fn both_hint_paths_verify() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(700);
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);

    let point_a: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let point_b: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

    let polys: Vec<_> = (0..4)
        .map(|_| Polynomial::<Fr>::random(num_vars, &mut rng))
        .collect();
    let points = vec![point_a.clone(), point_a, point_b.clone(), point_b];

    for use_hints in [true, false] {
        pipeline_round_trip::<Blake2bTranscript>(
            &polys,
            &points,
            &prover_setup,
            &verifier_setup,
            use_hints,
            b"both-paths",
        )
        .unwrap_or_else(|e| panic!("use_hints={use_hints} must verify: {e}"));
    }
}

#[test]
fn both_transcript_backends() {
    let num_vars = 2;
    let mut rng = ChaCha20Rng::seed_from_u64(800);
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);

    let polys: Vec<_> = (0..2)
        .map(|_| Polynomial::<Fr>::random(num_vars, &mut rng))
        .collect();
    let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let points = vec![point.clone(), point];

    pipeline_round_trip::<Blake2bTranscript>(
        &polys,
        &points,
        &prover_setup,
        &verifier_setup,
        true,
        b"blake2b-pipeline",
    )
    .expect("Blake2b pipeline must verify");

    pipeline_round_trip::<KeccakTranscript>(
        &polys,
        &points,
        &prover_setup,
        &verifier_setup,
        true,
        b"keccak-pipeline",
    )
    .expect("Keccak pipeline must verify");
}

#[test]
fn transcript_mismatch_causes_failure() {
    let num_vars = 2;
    let mut rng = ChaCha20Rng::seed_from_u64(850);
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);

    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let eval = poly.evaluate(&point);

    let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);

    let prover_claims = vec![ProverClaim {
        evaluations: poly.evaluations().to_vec(),
        point: point.clone(),
        eval,
    }];
    let verifier_claims = vec![VerifierClaim {
        commitment,
        point: point.clone(),
        eval,
    }];

    // Prover uses label "aaa"
    let mut tp = Blake2bTranscript::new(b"aaa");
    let (reduced_p, ()) = <RlcReduction as OpeningReduction<DoryScheme>>::reduce_prover(
        prover_claims,
        &mut tp,
        challenge_fn,
    );
    let proof = {
        let claim = &reduced_p[0];
        let p: Polynomial<Fr> = claim.evaluations.clone().into();
        DoryScheme::open(
            &p,
            &claim.point,
            claim.eval,
            &prover_setup,
            Some(hint),
            &mut tp,
        )
    };

    // Verifier uses label "bbb" — Fiat-Shamir mismatch
    let mut tv = Blake2bTranscript::new(b"bbb");
    let reduced_v = <RlcReduction as OpeningReduction<DoryScheme>>::reduce_verifier(
        verifier_claims,
        &(),
        &mut tv,
        challenge_fn,
    )
    .expect("reduction itself succeeds");

    let result = DoryScheme::verify(
        &reduced_v[0].commitment,
        &reduced_v[0].point,
        reduced_v[0].eval,
        &proof,
        &verifier_setup,
        &mut tv,
    );
    assert!(
        result.is_err(),
        "mismatched transcript labels must cause verification failure"
    );
}

#[test]
fn tampered_eval_after_reduction() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(900);
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);

    let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let poly_a = Polynomial::<Fr>::random(num_vars, &mut rng);
    let poly_b = Polynomial::<Fr>::random(num_vars, &mut rng);
    let eval_a = poly_a.evaluate(&point);
    let eval_b = poly_b.evaluate(&point);

    let (com_a, _) = DoryScheme::commit(poly_a.evaluations(), &prover_setup);
    let (com_b, _) = DoryScheme::commit(poly_b.evaluations(), &prover_setup);

    // Prover: honest claims
    let prover_claims = vec![
        ProverClaim {
            evaluations: poly_a.evaluations().to_vec(),
            point: point.clone(),
            eval: eval_a,
        },
        ProverClaim {
            evaluations: poly_b.evaluations().to_vec(),
            point: point.clone(),
            eval: eval_b,
        },
    ];

    let mut tp = Blake2bTranscript::new(b"tampered-eval");
    let (reduced_p, ()) = <RlcReduction as OpeningReduction<DoryScheme>>::reduce_prover(
        prover_claims,
        &mut tp,
        challenge_fn,
    );
    let proof = {
        let claim = &reduced_p[0];
        let p: Polynomial<Fr> = claim.evaluations.clone().into();
        DoryScheme::open(&p, &claim.point, claim.eval, &prover_setup, None, &mut tp)
    };

    // Verifier: tampered eval on poly_b
    let verifier_claims = vec![
        VerifierClaim {
            commitment: com_a,
            point: point.clone(),
            eval: eval_a,
        },
        VerifierClaim {
            commitment: com_b,
            point: point.clone(),
            eval: eval_b + Fr::from_u64(1),
        },
    ];

    let mut tv = Blake2bTranscript::new(b"tampered-eval");
    let reduced_v = <RlcReduction as OpeningReduction<DoryScheme>>::reduce_verifier(
        verifier_claims,
        &(),
        &mut tv,
        challenge_fn,
    )
    .expect("reduction itself succeeds");

    let result = DoryScheme::verify(
        &reduced_v[0].commitment,
        &reduced_v[0].point,
        reduced_v[0].eval,
        &proof,
        &verifier_setup,
        &mut tv,
    );
    assert!(
        result.is_err(),
        "tampered evaluation must cause verification failure"
    );
}

#[test]
fn tampered_commitment_in_verifier() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(950);
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);

    let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let wrong_poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let eval = poly.evaluate(&point);

    let (commitment, _) = DoryScheme::commit(poly.evaluations(), &prover_setup);
    let (wrong_commitment, _) = DoryScheme::commit(wrong_poly.evaluations(), &prover_setup);
    assert_ne!(commitment, wrong_commitment);

    let prover_claims = vec![ProverClaim {
        evaluations: poly.evaluations().to_vec(),
        point: point.clone(),
        eval,
    }];

    let mut tp = Blake2bTranscript::new(b"tampered-commit");
    let (reduced_p, ()) = <RlcReduction as OpeningReduction<DoryScheme>>::reduce_prover(
        prover_claims,
        &mut tp,
        challenge_fn,
    );
    let proof = {
        let claim = &reduced_p[0];
        let p: Polynomial<Fr> = claim.evaluations.clone().into();
        DoryScheme::open(&p, &claim.point, claim.eval, &prover_setup, None, &mut tp)
    };

    // Verifier uses the wrong commitment
    let verifier_claims = vec![VerifierClaim {
        commitment: wrong_commitment,
        point: point.clone(),
        eval,
    }];

    let mut tv = Blake2bTranscript::new(b"tampered-commit");
    let reduced_v = <RlcReduction as OpeningReduction<DoryScheme>>::reduce_verifier(
        verifier_claims,
        &(),
        &mut tv,
        challenge_fn,
    )
    .expect("reduction itself succeeds");

    let result = DoryScheme::verify(
        &reduced_v[0].commitment,
        &reduced_v[0].point,
        reduced_v[0].eval,
        &proof,
        &verifier_setup,
        &mut tv,
    );
    assert!(
        result.is_err(),
        "wrong commitment must cause verification failure"
    );
}

#[test]
fn extra_claim_causes_fiat_shamir_mismatch() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(980);
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);

    let point_a: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let point_b: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

    let poly_a = Polynomial::<Fr>::random(num_vars, &mut rng);
    let poly_b = Polynomial::<Fr>::random(num_vars, &mut rng);
    let poly_extra = Polynomial::<Fr>::random(num_vars, &mut rng);

    let eval_a = poly_a.evaluate(&point_a);
    let eval_b = poly_b.evaluate(&point_b);
    let eval_extra = poly_extra.evaluate(&point_b);

    let (com_a, _) = DoryScheme::commit(poly_a.evaluations(), &prover_setup);
    let (com_b, _) = DoryScheme::commit(poly_b.evaluations(), &prover_setup);
    let (com_extra, _) = DoryScheme::commit(poly_extra.evaluations(), &prover_setup);

    // Prover has 2 claims
    let prover_claims = vec![
        ProverClaim {
            evaluations: poly_a.evaluations().to_vec(),
            point: point_a.clone(),
            eval: eval_a,
        },
        ProverClaim {
            evaluations: poly_b.evaluations().to_vec(),
            point: point_b.clone(),
            eval: eval_b,
        },
    ];

    let mut tp = Blake2bTranscript::new(b"extra-claim");
    let (reduced_p, ()) = <RlcReduction as OpeningReduction<DoryScheme>>::reduce_prover(
        prover_claims,
        &mut tp,
        challenge_fn,
    );
    let proofs: Vec<DoryProofType> = reduced_p
        .iter()
        .map(|claim| {
            let p: Polynomial<Fr> = claim.evaluations.clone().into();
            DoryScheme::open(&p, &claim.point, claim.eval, &prover_setup, None, &mut tp)
        })
        .collect();

    // Verifier has 3 claims — extra claim at point_b creates a different group,
    // causing the RLC challenge for the point_b group to differ
    let verifier_claims = vec![
        VerifierClaim {
            commitment: com_a,
            point: point_a,
            eval: eval_a,
        },
        VerifierClaim {
            commitment: com_b,
            point: point_b.clone(),
            eval: eval_b,
        },
        VerifierClaim {
            commitment: com_extra,
            point: point_b,
            eval: eval_extra,
        },
    ];

    let mut tv = Blake2bTranscript::new(b"extra-claim");
    let reduced_v = <RlcReduction as OpeningReduction<DoryScheme>>::reduce_verifier(
        verifier_claims,
        &(),
        &mut tv,
        challenge_fn,
    )
    .expect("reduction itself succeeds");

    // Verifier now has 2 reduced claims, prover had 2 proofs — counts match,
    // but the RLC challenge for the point_b group differs → verify fails
    let mut any_failed = false;
    for (claim, proof) in reduced_v.iter().zip(proofs.iter()) {
        if DoryScheme::verify(
            &claim.commitment,
            &claim.point,
            claim.eval,
            proof,
            &verifier_setup,
            &mut tv,
        )
        .is_err()
        {
            any_failed = true;
        }
    }
    assert!(
        any_failed,
        "extra verifier claim must cause Fiat-Shamir mismatch"
    );
}

#[test]
fn property_random_claims_always_verify() {
    for seed in 0..10u64 {
        let mut rng = ChaCha20Rng::seed_from_u64(1000 + seed);
        let num_vars = 2 + (seed as usize % 3); // 2..4
        let num_polys = 2 + (seed as usize % 5); // 2..6
        let num_distinct_points = 1 + (seed as usize % 3); // 1..3

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryScheme::setup_verifier(num_vars);

        let distinct_points: Vec<Vec<Fr>> = (0..num_distinct_points)
            .map(|_| (0..num_vars).map(|_| Fr::random(&mut rng)).collect())
            .collect();

        let polys: Vec<_> = (0..num_polys)
            .map(|_| Polynomial::<Fr>::random(num_vars, &mut rng))
            .collect();
        let points: Vec<_> = (0..num_polys)
            .map(|i| distinct_points[i % num_distinct_points].clone())
            .collect();

        for use_hints in [true, false] {
            pipeline_round_trip::<Blake2bTranscript>(
                &polys,
                &points,
                &prover_setup,
                &verifier_setup,
                use_hints,
                b"property",
            )
            .unwrap_or_else(|e| {
                panic!(
                    "seed={seed} num_vars={num_vars} num_polys={num_polys} hints={use_hints}: {e}"
                )
            });
        }
    }
}
