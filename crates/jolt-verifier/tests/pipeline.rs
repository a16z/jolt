//! End-to-end pipeline tests for jolt-verifier.
//!
//! Tests the full S1→S2→S8 verification pipeline using a simple R1CS
//! circuit and a single claim reduction sumcheck stage.

use jolt_field::{Field, Fr};
use jolt_openings::mock::{MockCommitment, MockCommitmentScheme};
use jolt_openings::{CommitmentScheme, ProverClaim, VerifierClaim};
use jolt_poly::{EqPolynomial, Polynomial};
use jolt_spartan::{FirstRoundStrategy, SimpleR1CS, SpartanKey, SpartanProver};
use jolt_sumcheck::{BatchedSumcheckProver, SumcheckClaim};
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::proof::{BatchOpeningProofs, SumcheckStageProof};
use jolt_verifier::{
    verify, verify_openings, verify_spartan, JoltError, JoltProof, JoltVerifyingKey, VerifierStage,
};
use num_traits::Zero;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type MockPCS = MockCommitmentScheme<Fr>;
type C = MockCommitment<Fr>;

fn challenge_fn(c: u128) -> Fr {
    Fr::from_u128(c)
}


/// A trivial verifier stage for testing: `Σ_x eq(r, x) · Σ_i c_i · p_i(x)`.
struct TestClaimReductionVerifier {
    eq_point: Vec<Fr>,
    coefficients: Vec<Fr>,
    eval_to_commitment: Vec<usize>,
    claimed_sum: Fr,
    num_vars: usize,
}

impl<Cm: Clone, T: Transcript> VerifierStage<Fr, Cm, T> for TestClaimReductionVerifier {
    fn build_claims(
        &mut self,
        _prior_claims: &[VerifierClaim<Fr, Cm>],
        _transcript: &mut T,
    ) -> Vec<SumcheckClaim<Fr>> {
        vec![SumcheckClaim {
            num_vars: self.num_vars,
            degree: 2,
            claimed_sum: self.claimed_sum,
        }]
    }

    fn check_and_extract(
        &mut self,
        final_eval: Fr,
        challenges: &[Fr],
        evaluations: &[Fr],
        commitments: &[Cm],
    ) -> Result<Vec<VerifierClaim<Fr, Cm>>, JoltError> {
        let eq_eval = EqPolynomial::new(self.eq_point.clone()).evaluate(challenges);
        let weighted_sum: Fr = self
            .coefficients
            .iter()
            .zip(evaluations.iter())
            .map(|(&c, &e)| c * e)
            .sum();
        let expected = eq_eval * weighted_sum;

        if expected != final_eval {
            return Err(JoltError::EvaluationMismatch {
                stage: 2,
                reason: format!("expected {expected:?}, got {final_eval:?}"),
            });
        }

        Ok(evaluations
            .iter()
            .enumerate()
            .map(|(i, &eval)| VerifierClaim {
                commitment: commitments[self.eval_to_commitment[i]].clone(),
                point: challenges.to_vec(),
                eval,
            })
            .collect())
    }
}


use jolt_sumcheck::prover::SumcheckCompute;

struct EqGWitness {
    eq: Vec<Fr>,
    g: Vec<Fr>,
}

impl SumcheckCompute<Fr> for EqGWitness {
    fn round_polynomial(&self) -> jolt_poly::UnivariatePoly<Fr> {
        let half = self.eq.len() / 2;
        let mut evals = [Fr::zero(); 3];
        // High-to-low binding: pair (j, j+half) — matches Polynomial::bind
        for j in 0..half {
            let e_lo = self.eq[j];
            let e_hi = self.eq[j + half];
            let g_lo = self.g[j];
            let g_hi = self.g[j + half];
            evals[0] += e_lo * g_lo;
            evals[1] += e_hi * g_hi;
            evals[2] += (e_hi + e_hi - e_lo) * (g_hi + g_hi - g_lo);
        }
        jolt_poly::UnivariatePoly::interpolate_over_integers(&evals)
    }

    fn bind(&mut self, c: Fr) {
        let half = self.eq.len() / 2;
        // High-to-low: bind MSB first, pairing (j, j+half)
        for j in 0..half {
            self.eq[j] = self.eq[j] + c * (self.eq[j + half] - self.eq[j]);
            self.g[j] = self.g[j] + c * (self.g[j + half] - self.g[j]);
        }
        self.eq.truncate(half);
        self.g.truncate(half);
    }
}


#[test]
fn verify_spartan_round_trip() {
    let r1cs = SimpleR1CS::new(
        1,
        3,
        vec![(0, 1, Fr::from_u64(1))],
        vec![(0, 1, Fr::from_u64(1))],
        vec![(0, 2, Fr::from_u64(1))],
    );
    let key = SpartanKey::from_r1cs(&r1cs);
    let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];

    let mut pt = Blake2bTranscript::new(b"verify-spartan-rt");
    let (proof, prover_r_x, prover_r_y) =
        SpartanProver::prove_with_challenges::<MockPCS, _>(
            &r1cs,
            &key,
            &witness,
            &(),
            &mut pt,
            FirstRoundStrategy::Standard,
        )
        .expect("proving should succeed");

    let mut vt = Blake2bTranscript::new(b"verify-spartan-rt");
    let (verifier_r_x, verifier_r_y) =
        verify_spartan::<MockPCS, _>(&key, &proof, &(), &mut vt)
            .expect("verification should succeed");

    assert_eq!(prover_r_x, verifier_r_x);
    assert_eq!(prover_r_y, verifier_r_y);
}

#[test]
fn verify_openings_round_trip() {
    let num_vars = 4;
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

    let mut prover_claims = Vec::new();
    let mut verifier_claims = Vec::new();

    for _ in 0..3 {
        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let eval = poly.evaluate(&point);
        let commitment = MockPCS::commit(poly.evaluations(), &()).0;

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
    }

    // Prove opening
    let mut pt = Blake2bTranscript::new(b"verify-openings-rt");
    use jolt_openings::{OpeningReduction, RlcReduction};
    let (reduced_prover, ()) = <RlcReduction as OpeningReduction<MockPCS>>::reduce_prover(
        prover_claims,
        &mut pt,
        &challenge_fn,
    );
    let proofs: Vec<_> = reduced_prover
        .into_iter()
        .map(|claim| {
            let poly: <MockPCS as CommitmentScheme>::Polynomial = claim.evaluations.into();
            MockPCS::open(&poly, &claim.point, claim.eval, &(), None, &mut pt)
        })
        .collect();
    let opening_proofs = BatchOpeningProofs { proofs };

    // Verify
    let mut vt = Blake2bTranscript::new(b"verify-openings-rt");
    verify_openings::<MockPCS, _>(verifier_claims, &opening_proofs, &(), &mut vt, challenge_fn)
        .expect("opening verification should succeed");
}

#[test]
fn verify_rejects_wrong_stage_count() {
    let r1cs = SimpleR1CS::new(
        1,
        3,
        vec![(0, 1, Fr::from_u64(1))],
        vec![(0, 1, Fr::from_u64(1))],
        vec![(0, 2, Fr::from_u64(1))],
    );
    let key = SpartanKey::from_r1cs(&r1cs);
    let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];

    let mut pt = Blake2bTranscript::new(b"stage-count");
    let spartan_proof = SpartanProver::prove::<MockPCS, _>(
        &r1cs,
        &key,
        &witness,
        &(),
        &mut pt,
        FirstRoundStrategy::Standard,
    )
    .expect("proving should succeed");

    let proof = JoltProof::<Fr, MockPCS> {
        spartan_proof,
        stage_proofs: vec![], // 0 stages
        opening_proofs: BatchOpeningProofs { proofs: vec![] },
        commitments: vec![],
        trace_length: 1,
    };

    let vk = JoltVerifyingKey {
        spartan_key: key,
        pcs_setup: (),
    };

    let dummy_stage = TestClaimReductionVerifier {
        eq_point: vec![],
        coefficients: vec![],
        eval_to_commitment: vec![],
        claimed_sum: Fr::zero(),
        num_vars: 0,
    };

    let mut stages: Vec<Box<dyn VerifierStage<Fr, C, Blake2bTranscript>>> =
        vec![Box::new(dummy_stage)]; // 1 stage, but proof has 0

    let mut vt = Blake2bTranscript::new(b"stage-count");
    let result = verify::<MockPCS, _>(&proof, &vk, &mut stages, &mut vt, challenge_fn);
    assert!(
        matches!(result, Err(JoltError::InvalidProof(_))),
        "should reject stage count mismatch, got: {result:?}"
    );
}

#[test]
fn verify_rejects_bad_evaluation() {
    let mut rng = ChaCha20Rng::seed_from_u64(77);

    let r1cs = SimpleR1CS::new(
        1,
        3,
        vec![(0, 1, Fr::from_u64(1))],
        vec![(0, 1, Fr::from_u64(1))],
        vec![(0, 2, Fr::from_u64(1))],
    );
    let key = SpartanKey::from_r1cs(&r1cs);
    let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];

    let num_vars = 3;
    let n = 1usize << num_vars;
    let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let coefficients = vec![Fr::from_u64(1)];
    let poly_table: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
    let commitment = MockPCS::commit(&poly_table, &()).0;

    let eq_table = EqPolynomial::new(eq_point.clone()).evaluations();
    let claimed_sum: Fr = eq_table
        .iter()
        .zip(poly_table.iter())
        .map(|(&e, &p)| e * p)
        .sum();

    // Prove Spartan
    let mut pt = Blake2bTranscript::new(b"bad-eval");
    let spartan_proof = SpartanProver::prove::<MockPCS, _>(
        &r1cs,
        &key,
        &witness,
        &(),
        &mut pt,
        FirstRoundStrategy::Standard,
    )
    .expect("proving should succeed");

    // Prove sumcheck
    let claim = SumcheckClaim {
        num_vars,
        degree: 2,
        claimed_sum,
    };

    let mut witnesses: Vec<Box<dyn SumcheckCompute<Fr>>> = vec![Box::new(EqGWitness {
        eq: eq_table,
        g: poly_table.clone(),
    })];
    let sumcheck_proof =
        BatchedSumcheckProver::prove(&[claim], &mut witnesses, &mut pt, challenge_fn);

    // Tamper with evaluation
    let stage_proof = SumcheckStageProof {
        sumcheck_proof,
        evaluations: vec![Fr::from_u64(999)], // WRONG
    };

    let proof = JoltProof::<Fr, MockPCS> {
        spartan_proof,
        stage_proofs: vec![stage_proof],
        opening_proofs: BatchOpeningProofs { proofs: vec![] },
        commitments: vec![commitment],
        trace_length: 1,
    };

    let vk = JoltVerifyingKey {
        spartan_key: key,
        pcs_setup: (),
    };

    let verifier_stage = TestClaimReductionVerifier {
        eq_point,
        coefficients,
        eval_to_commitment: vec![0],
        claimed_sum,
        num_vars,
    };

    let mut stages: Vec<Box<dyn VerifierStage<Fr, C, Blake2bTranscript>>> =
        vec![Box::new(verifier_stage)];

    let mut vt = Blake2bTranscript::new(b"bad-eval");
    let result = verify::<MockPCS, _>(&proof, &vk, &mut stages, &mut vt, challenge_fn);
    assert!(
        result.is_err(),
        "tampered evaluation should be rejected, got: {result:?}"
    );
}

/// Standalone test: prove + verify a single eq*g sumcheck, check final_eval matches formula.
#[test]
fn sumcheck_eq_g_final_eval_matches() {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let num_vars = 3;
    let n = 1usize << num_vars;

    let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let poly_table: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

    let eq_table = EqPolynomial::new(eq_point.clone()).evaluations();
    let claimed_sum: Fr = eq_table.iter().zip(poly_table.iter()).map(|(&e, &p)| e * p).sum();

    let claim = SumcheckClaim {
        num_vars,
        degree: 2,
        claimed_sum,
    };

    // Prove
    let mut pt = Blake2bTranscript::new(b"eq-g-test");
    let mut witnesses: Vec<Box<dyn SumcheckCompute<Fr>>> = vec![Box::new(EqGWitness {
        eq: eq_table,
        g: poly_table.clone(),
    })];
    let proof = BatchedSumcheckProver::prove(&[claim.clone()], &mut witnesses, &mut pt, challenge_fn);

    // Verify
    let mut vt = Blake2bTranscript::new(b"eq-g-test");
    let (final_eval, challenges) =
        jolt_sumcheck::BatchedSumcheckVerifier::verify(&[claim], &proof, &mut vt, challenge_fn)
            .expect("verification should succeed");

    // Check: final_eval should equal eq(r, challenges) * p(challenges)
    let eq_eval = EqPolynomial::new(eq_point).evaluate(&challenges);
    let poly_eval = Polynomial::new(poly_table).evaluate(&challenges);
    let expected = eq_eval * poly_eval;

    assert_eq!(
        final_eval, expected,
        "final_eval from sumcheck should equal eq(r, challenges) * p(challenges)"
    );
}

#[test]
fn full_pipeline_spartan_plus_one_stage() {
    let mut rng = ChaCha20Rng::seed_from_u64(99);

    // S1: Simple x*x=y Spartan circuit
    let r1cs = SimpleR1CS::new(
        1,
        3,
        vec![(0, 1, Fr::from_u64(1))],
        vec![(0, 1, Fr::from_u64(1))],
        vec![(0, 2, Fr::from_u64(1))],
    );
    let key = SpartanKey::from_r1cs(&r1cs);
    let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];

    // S2: Claim reduction — 2 random polynomials
    let num_vars = 3;
    let num_polys = 2;
    let n = 1usize << num_vars;

    let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let coefficients: Vec<Fr> = (0..num_polys).map(|_| Fr::random(&mut rng)).collect();
    let poly_tables: Vec<Vec<Fr>> = (0..num_polys)
        .map(|_| (0..n).map(|_| Fr::random(&mut rng)).collect())
        .collect();

    let commitments: Vec<C> = poly_tables
        .iter()
        .map(|table| MockPCS::commit(table, &()).0)
        .collect();

    // Compute claimed_sum = Σ_x eq(r, x) * Σ c_i * p_i(x)
    let eq_table = EqPolynomial::new(eq_point.clone()).evaluations();
    let mut g_table = vec![Fr::zero(); n];
    for (i, table) in poly_tables.iter().enumerate() {
        for (j, g) in g_table.iter_mut().enumerate() {
            *g += coefficients[i] * table[j];
        }
    }
    let claimed_sum: Fr = eq_table.iter().zip(g_table.iter()).map(|(&e, &g)| e * g).sum();

    // ── Prover side ──

    let mut pt = Blake2bTranscript::new(b"full-pipeline");

    // S1: Spartan
    let (spartan_proof, _prover_r_x, _prover_r_y) =
        SpartanProver::prove_with_challenges::<MockPCS, _>(
            &r1cs,
            &key,
            &witness,
            &(),
            &mut pt,
            FirstRoundStrategy::Standard,
        )
        .expect("spartan proving should succeed");

    // S2: Sumcheck
    let claim = SumcheckClaim {
        num_vars,
        degree: 2,
        claimed_sum,
    };
    let mut witnesses: Vec<Box<dyn SumcheckCompute<Fr>>> = vec![Box::new(EqGWitness {
        eq: eq_table.clone(),
        g: g_table,
    })];
    let sumcheck_proof =
        BatchedSumcheckProver::prove(&[claim.clone()], &mut witnesses, &mut pt, challenge_fn);

    // Extract challenges by replaying verifier transcript up to this point
    let mut vt_replay = Blake2bTranscript::new(b"full-pipeline");
    let _ = verify_spartan::<MockPCS, _>(&key, &spartan_proof, &(), &mut vt_replay).unwrap();
    let (_, s2_challenges) = jolt_sumcheck::BatchedSumcheckVerifier::verify(
        &[claim],
        &sumcheck_proof,
        &mut vt_replay,
        challenge_fn,
    )
    .expect("sumcheck replay should succeed");

    // Extract evaluations at challenge point
    let evaluations: Vec<Fr> = poly_tables
        .iter()
        .map(|table| Polynomial::new(table.clone()).evaluate(&s2_challenges))
        .collect();

    let stage_proof = SumcheckStageProof {
        sumcheck_proof,
        evaluations: evaluations.clone(),
    };

    // S8: Opening proofs using the VERIFIER transcript (vt_replay) for RLC
    let prover_claims: Vec<ProverClaim<Fr>> = poly_tables
        .iter()
        .zip(evaluations.iter())
        .map(|(table, &eval)| ProverClaim {
            evaluations: table.clone(),
            point: s2_challenges.clone(),
            eval,
        })
        .collect();

    use jolt_openings::{OpeningReduction, RlcReduction};
    // Continue using pt (prover transcript) for opening proofs
    let (reduced, ()) = <RlcReduction as OpeningReduction<MockPCS>>::reduce_prover(
        prover_claims,
        &mut pt,
        &challenge_fn,
    );
    let pcs_proofs: Vec<_> = reduced
        .into_iter()
        .map(|claim| {
            let poly: <MockPCS as CommitmentScheme>::Polynomial = claim.evaluations.into();
            MockPCS::open(&poly, &claim.point, claim.eval, &(), None, &mut pt)
        })
        .collect();
    let opening_proofs = BatchOpeningProofs { proofs: pcs_proofs };

    // Assemble proof
    let proof = JoltProof {
        spartan_proof,
        stage_proofs: vec![stage_proof],
        opening_proofs,
        commitments: commitments.clone(),
        trace_length: 1,
    };

    let vk = JoltVerifyingKey {
        spartan_key: key,
        pcs_setup: (),
    };

    let verifier_stage = TestClaimReductionVerifier {
        eq_point,
        coefficients,
        eval_to_commitment: (0..num_polys).collect(),
        claimed_sum,
        num_vars,
    };

    let mut stages: Vec<Box<dyn VerifierStage<Fr, C, Blake2bTranscript>>> =
        vec![Box::new(verifier_stage)];

    // ── Verifier side ──

    let mut vt = Blake2bTranscript::new(b"full-pipeline");
    let (_r_x, r_y) = verify::<MockPCS, _>(&proof, &vk, &mut stages, &mut vt, challenge_fn)
        .expect("full pipeline verification should succeed");

    // r_x may be empty for a 1-constraint circuit (log2(1) = 0 outer vars)
    assert!(!r_y.is_empty());
}


mod dory {
    use super::*;
    use jolt_dory::{DoryCommitment, DoryScheme};
    use jolt_openings::{OpeningReduction, RlcReduction};

    type DC = DoryCommitment;

    #[test]
    fn dory_verify_openings_round_trip() {
        let num_vars = 4;
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryScheme::setup_verifier(num_vars);

        let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        let mut prover_claims = Vec::new();
        let mut verifier_claims = Vec::new();

        for _ in 0..3 {
            let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
            let eval = poly.evaluate(&point);
            let (commitment, _hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);

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
        }

        let mut pt = Blake2bTranscript::new(b"dory-openings-rt");
        let (reduced_prover, ()) =
            <RlcReduction as OpeningReduction<DoryScheme>>::reduce_prover(
                prover_claims,
                &mut pt,
                &challenge_fn,
            );
        let proofs: Vec<_> = reduced_prover
            .into_iter()
            .map(|claim| {
                let poly: <DoryScheme as CommitmentScheme>::Polynomial =
                    claim.evaluations.into();
                DoryScheme::open(&poly, &claim.point, claim.eval, &prover_setup, None, &mut pt)
            })
            .collect();
        let opening_proofs = BatchOpeningProofs { proofs };

        let mut vt = Blake2bTranscript::new(b"dory-openings-rt");
        verify_openings::<DoryScheme, _>(
            verifier_claims,
            &opening_proofs,
            &verifier_setup,
            &mut vt,
            challenge_fn,
        )
        .expect("Dory opening verification should succeed");
    }

    #[test]
    fn dory_full_pipeline_spartan_plus_one_stage() {
        let mut rng = ChaCha20Rng::seed_from_u64(99);

        let num_vars = 3;
        let num_polys = 2;
        let n = 1usize << num_vars;

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryScheme::setup_verifier(num_vars);

        // S1: Simple x*x=y Spartan circuit
        let r1cs = SimpleR1CS::new(
            1,
            3,
            vec![(0, 1, Fr::from_u64(1))],
            vec![(0, 1, Fr::from_u64(1))],
            vec![(0, 2, Fr::from_u64(1))],
        );
        let key = SpartanKey::from_r1cs(&r1cs);
        let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];

        // S2: Claim reduction — 2 random polynomials
        let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let coefficients: Vec<Fr> = (0..num_polys).map(|_| Fr::random(&mut rng)).collect();
        let poly_tables: Vec<Vec<Fr>> = (0..num_polys)
            .map(|_| (0..n).map(|_| Fr::random(&mut rng)).collect())
            .collect();

        let commitments: Vec<DC> = poly_tables
            .iter()
            .map(|table| DoryScheme::commit(table, &prover_setup).0)
            .collect();

        let eq_table = EqPolynomial::new(eq_point.clone()).evaluations();
        let mut g_table = vec![Fr::zero(); n];
        for (i, table) in poly_tables.iter().enumerate() {
            for (j, g) in g_table.iter_mut().enumerate() {
                *g += coefficients[i] * table[j];
            }
        }
        let claimed_sum: Fr =
            eq_table.iter().zip(g_table.iter()).map(|(&e, &g)| e * g).sum();

        // ── Prover side ──

        let mut pt = Blake2bTranscript::new(b"dory-full-pipeline");

        // S1: Spartan
        let (spartan_proof, _prover_r_x, _prover_r_y) =
            SpartanProver::prove_with_challenges::<DoryScheme, _>(
                &r1cs,
                &key,
                &witness,
                &prover_setup,
                &mut pt,
                FirstRoundStrategy::Standard,
            )
            .expect("spartan proving should succeed");

        // S2: Sumcheck
        let claim = SumcheckClaim {
            num_vars,
            degree: 2,
            claimed_sum,
        };
        let mut witnesses: Vec<Box<dyn SumcheckCompute<Fr>>> = vec![Box::new(EqGWitness {
            eq: eq_table.clone(),
            g: g_table,
        })];
        let sumcheck_proof =
            BatchedSumcheckProver::prove(&[claim.clone()], &mut witnesses, &mut pt, challenge_fn);

        // Extract challenges by replaying verifier
        let mut vt_replay = Blake2bTranscript::new(b"dory-full-pipeline");
        let _ = verify_spartan::<DoryScheme, _>(
            &key,
            &spartan_proof,
            &verifier_setup,
            &mut vt_replay,
        )
        .unwrap();
        let (_, s2_challenges) = jolt_sumcheck::BatchedSumcheckVerifier::verify(
            &[claim],
            &sumcheck_proof,
            &mut vt_replay,
            challenge_fn,
        )
        .expect("sumcheck replay should succeed");

        let evaluations: Vec<Fr> = poly_tables
            .iter()
            .map(|table| Polynomial::new(table.clone()).evaluate(&s2_challenges))
            .collect();

        let stage_proof = SumcheckStageProof {
            sumcheck_proof,
            evaluations: evaluations.clone(),
        };

        // S8: Opening proofs
        let prover_claims: Vec<ProverClaim<Fr>> = poly_tables
            .iter()
            .zip(evaluations.iter())
            .map(|(table, &eval)| ProverClaim {
                evaluations: table.clone(),
                point: s2_challenges.clone(),
                eval,
            })
            .collect();

        let (reduced, ()) = <RlcReduction as OpeningReduction<DoryScheme>>::reduce_prover(
            prover_claims,
            &mut pt,
            &challenge_fn,
        );
        let pcs_proofs: Vec<_> = reduced
            .into_iter()
            .map(|claim| {
                let poly: <DoryScheme as CommitmentScheme>::Polynomial =
                    claim.evaluations.into();
                DoryScheme::open(
                    &poly,
                    &claim.point,
                    claim.eval,
                    &prover_setup,
                    None,
                    &mut pt,
                )
            })
            .collect();
        let opening_proofs = BatchOpeningProofs { proofs: pcs_proofs };

        let proof = JoltProof {
            spartan_proof,
            stage_proofs: vec![stage_proof],
            opening_proofs,
            commitments: commitments.clone(),
            trace_length: 1,
        };

        let vk = JoltVerifyingKey {
            spartan_key: key,
            pcs_setup: verifier_setup,
        };

        let verifier_stage = TestClaimReductionVerifier {
            eq_point,
            coefficients,
            eval_to_commitment: (0..num_polys).collect(),
            claimed_sum,
            num_vars,
        };

        let mut stages: Vec<Box<dyn VerifierStage<Fr, DC, Blake2bTranscript>>> =
            vec![Box::new(verifier_stage)];

        // ── Verifier side ──

        let mut vt = Blake2bTranscript::new(b"dory-full-pipeline");
        let (_r_x, r_y) =
            verify::<DoryScheme, _>(&proof, &vk, &mut stages, &mut vt, challenge_fn)
                .expect("Dory full pipeline verification should succeed");

        assert!(!r_y.is_empty());
    }
}
