//! End-to-end pipeline tests for jolt-verifier.
//!
//! Tests the full S1→S2→S8 verification pipeline using uniform Spartan
//! with per-cycle constraints and config-driven stage descriptors.

use jolt_field::{Field, Fr};
use jolt_openings::mock::{MockCommitment, MockCommitmentScheme};
use jolt_openings::{CommitmentScheme, ProverClaim, VerifierClaim};
use jolt_poly::{EqPolynomial, Polynomial};
use jolt_spartan::{UniformSpartanKey, UniformSpartanProver};
use jolt_sumcheck::{BatchedSumcheckProver, SumcheckClaim};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Transcript};
use jolt_verifier::config::ProverConfig;
use jolt_verifier::proof::SumcheckStageProof;
use jolt_verifier::stage::StageDescriptor;
use jolt_verifier::{
    verify, verify_openings, verify_spartan, JoltError, JoltProof, JoltVerifyingKey,
};
use num_traits::Zero;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type MockPCS = MockCommitmentScheme<Fr>;

fn test_prover_config() -> ProverConfig {
    use jolt_verifier::config::{OneHotConfig, ReadWriteConfig};
    ProverConfig {
        trace_length: 2,
        ram_k: 16,
        one_hot_config: OneHotConfig::new(1),
        rw_config: ReadWriteConfig::new(1, 4),
    }
}

fn test_uniform_key(num_cycles: usize) -> UniformSpartanKey<Fr> {
    let one = Fr::from_u64(1);
    UniformSpartanKey::new(
        num_cycles,
        1,
        3,
        vec![vec![(1, one)]],
        vec![vec![(1, one)]],
        vec![vec![(2, one)]],
    )
}

fn make_cycle_witness(x: u64) -> Vec<Fr> {
    vec![Fr::from_u64(1), Fr::from_u64(x), Fr::from_u64(x * x)]
}

fn flatten_witnesses(key: &UniformSpartanKey<Fr>, cycle_witnesses: &[Vec<Fr>]) -> Vec<Fr> {
    let total_cols_padded = key.total_cols().next_power_of_two();
    let mut flat = vec![Fr::from_u64(0); total_cols_padded];
    for (c, w) in cycle_witnesses.iter().enumerate() {
        let base = c * key.num_vars_padded;
        for (v, &val) in w.iter().enumerate().take(key.num_vars) {
            flat[base + v] = val;
        }
    }
    flat
}

/// Commit witness and append to transcript. Returns the commitment.
fn commit_and_append<PCS: CommitmentScheme<Field = Fr>>(
    flat: &[Fr],
    setup: &PCS::ProverSetup,
    transcript: &mut Blake2bTranscript,
) -> PCS::Output {
    let (commitment, _) = PCS::commit(flat, setup);
    transcript.append_bytes(format!("{commitment:?}").as_bytes());
    commitment
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
    let key = test_uniform_key(2);
    let witnesses = vec![make_cycle_witness(3), make_cycle_witness(5)];
    let flat = flatten_witnesses(&key, &witnesses);

    let mut pt = Blake2bTranscript::new(b"verify-spartan-rt");
    let _ = commit_and_append::<MockPCS>(&flat, &(), &mut pt);
    let (proof, prover_r_x, prover_r_y) =
        UniformSpartanProver::prove_dense_with_challenges(&key, &flat, &mut pt)
            .expect("proving should succeed");

    let mut vt = Blake2bTranscript::new(b"verify-spartan-rt");
    let _ = commit_and_append::<MockPCS>(&flat, &(), &mut vt);
    let (verifier_r_x, verifier_r_y) =
        verify_spartan(&key, &proof, &mut vt).expect("verification should succeed");

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

    let mut pt = Blake2bTranscript::new(b"verify-openings-rt");
    use jolt_openings::{OpeningReduction, RlcReduction};
    let (reduced_prover, ()) =
        <RlcReduction as OpeningReduction<MockPCS>>::reduce_prover(prover_claims, &mut pt);
    let proofs: Vec<_> = reduced_prover
        .into_iter()
        .map(|claim| {
            let poly: <MockPCS as CommitmentScheme>::Polynomial = claim.evaluations.into();
            MockPCS::open(&poly, &claim.point, claim.eval, &(), None, &mut pt)
        })
        .collect();
    let mut vt = Blake2bTranscript::new(b"verify-openings-rt");
    verify_openings::<MockPCS, _>(verifier_claims, &proofs, &(), &mut vt)
        .expect("opening verification should succeed");
}

#[test]
fn verify_rejects_wrong_stage_count() {
    let key = test_uniform_key(2);
    let witnesses = vec![make_cycle_witness(3), make_cycle_witness(5)];
    let flat = flatten_witnesses(&key, &witnesses);

    let mut pt = Blake2bTranscript::new(b"stage-count");
    let witness_commitment = commit_and_append::<MockPCS>(&flat, &(), &mut pt);
    let (spartan_proof, _, _) =
        UniformSpartanProver::prove_dense_with_challenges(&key, &flat, &mut pt)
            .expect("proving should succeed");

    let proof = JoltProof::<Fr, MockPCS> {
        config: test_prover_config(),
        spartan_proof,
        stage_proofs: vec![],
        opening_proofs: vec![],
        witness_commitment,
        commitments: vec![],
    };

    let vk = JoltVerifyingKey {
        spartan_key: key,
        pcs_setup: (),
    };

    // Verify builds one descriptor but proof has 0 stage proofs → mismatch.
    let mut vt = Blake2bTranscript::new(b"stage-count");
    let result = verify::<MockPCS, _>(
        &proof,
        &vk,
        |_r_x, _r_y, _t| {
            vec![StageDescriptor::claim_reduction(
                vec![],
                vec![],
                Fr::zero(),
                vec![],
            )]
        },
        &mut vt,
    );
    assert!(
        matches!(result, Err(JoltError::InvalidProof(_))),
        "should reject stage count mismatch, got: {result:?}"
    );
}

#[test]
fn verify_rejects_bad_evaluation() {
    let mut rng = ChaCha20Rng::seed_from_u64(77);

    let key = test_uniform_key(2);
    let witnesses = vec![make_cycle_witness(3), make_cycle_witness(5)];
    let flat = flatten_witnesses(&key, &witnesses);

    let num_vars = 3;
    let n = 1usize << num_vars;
    let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let poly_table: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
    let commitment = MockPCS::commit(&poly_table, &()).0;

    let eq_table = EqPolynomial::new(eq_point.clone()).evaluations();
    let claimed_sum: Fr = eq_table
        .iter()
        .zip(poly_table.iter())
        .map(|(&e, &p)| e * p)
        .sum();

    let mut pt = Blake2bTranscript::new(b"bad-eval");
    let witness_commitment = commit_and_append::<MockPCS>(&flat, &(), &mut pt);
    let (spartan_proof, _, _) =
        UniformSpartanProver::prove_dense_with_challenges(&key, &flat, &mut pt)
            .expect("proving should succeed");

    let claim = SumcheckClaim {
        num_vars,
        degree: 2,
        claimed_sum,
    };

    let mut sc_witnesses: Vec<Box<dyn SumcheckCompute<Fr>>> = vec![Box::new(EqGWitness {
        eq: eq_table,
        g: poly_table.clone(),
    })];
    let sumcheck_proof = BatchedSumcheckProver::prove(&[claim], &mut sc_witnesses, &mut pt);

    let stage_proof = SumcheckStageProof {
        sumcheck_proof,
        evaluations: vec![Fr::from_u64(999)], // WRONG
    };

    let proof = JoltProof::<Fr, MockPCS> {
        config: test_prover_config(),
        spartan_proof,
        stage_proofs: vec![stage_proof],
        opening_proofs: vec![],
        witness_commitment,
        commitments: vec![commitment],
    };

    let vk = JoltVerifyingKey {
        spartan_key: key,
        pcs_setup: (),
    };

    let mut vt = Blake2bTranscript::new(b"bad-eval");
    let result = verify::<MockPCS, _>(
        &proof,
        &vk,
        |_r_x, _r_y, _t| {
            vec![StageDescriptor::claim_reduction(
                eq_point,
                vec![Fr::from_u64(1)],
                claimed_sum,
                vec![0],
            )]
        },
        &mut vt,
    );
    assert!(
        result.is_err(),
        "tampered evaluation should be rejected, got: {result:?}"
    );
}

#[test]
fn sumcheck_eq_g_final_eval_matches() {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let num_vars = 3;
    let n = 1usize << num_vars;

    let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let poly_table: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

    let eq_table = EqPolynomial::new(eq_point.clone()).evaluations();
    let claimed_sum: Fr = eq_table
        .iter()
        .zip(poly_table.iter())
        .map(|(&e, &p)| e * p)
        .sum();

    let claim = SumcheckClaim {
        num_vars,
        degree: 2,
        claimed_sum,
    };

    let mut pt = Blake2bTranscript::new(b"eq-g-test");
    let mut sc_witnesses: Vec<Box<dyn SumcheckCompute<Fr>>> = vec![Box::new(EqGWitness {
        eq: eq_table,
        g: poly_table.clone(),
    })];
    let proof = BatchedSumcheckProver::prove(&[claim.clone()], &mut sc_witnesses, &mut pt);

    let mut vt = Blake2bTranscript::new(b"eq-g-test");
    let (final_eval, challenges) =
        jolt_sumcheck::BatchedSumcheckVerifier::verify(&[claim], &proof, &mut vt)
            .expect("verification should succeed");

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

    let key = test_uniform_key(2);
    let witnesses = vec![make_cycle_witness(3), make_cycle_witness(5)];
    let flat = flatten_witnesses(&key, &witnesses);

    let num_vars = 3;
    let num_polys = 2;
    let n = 1usize << num_vars;

    let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let coefficients: Vec<Fr> = (0..num_polys).map(|_| Fr::random(&mut rng)).collect();
    let poly_tables: Vec<Vec<Fr>> = (0..num_polys)
        .map(|_| (0..n).map(|_| Fr::random(&mut rng)).collect())
        .collect();

    let commitments: Vec<MockCommitment<Fr>> = poly_tables
        .iter()
        .map(|table| MockPCS::commit(table, &()).0)
        .collect();

    let eq_table = EqPolynomial::new(eq_point.clone()).evaluations();
    let mut g_table = vec![Fr::zero(); n];
    for (i, table) in poly_tables.iter().enumerate() {
        for (j, g) in g_table.iter_mut().enumerate() {
            *g += coefficients[i] * table[j];
        }
    }
    let claimed_sum: Fr = eq_table
        .iter()
        .zip(g_table.iter())
        .map(|(&e, &g)| e * g)
        .sum();

    let mut pt = Blake2bTranscript::new(b"full-pipeline");

    let witness_commitment = commit_and_append::<MockPCS>(&flat, &(), &mut pt);
    let (spartan_proof, _prover_r_x, _prover_r_y) =
        UniformSpartanProver::prove_dense_with_challenges(&key, &flat, &mut pt)
            .expect("spartan proving should succeed");

    let claim = SumcheckClaim {
        num_vars,
        degree: 2,
        claimed_sum,
    };
    let mut sc_witnesses: Vec<Box<dyn SumcheckCompute<Fr>>> = vec![Box::new(EqGWitness {
        eq: eq_table.clone(),
        g: g_table,
    })];
    let sumcheck_proof = BatchedSumcheckProver::prove(&[claim.clone()], &mut sc_witnesses, &mut pt);

    // Replay verifier transcript to extract challenges
    let mut vt_replay = Blake2bTranscript::new(b"full-pipeline");
    let _ = commit_and_append::<MockPCS>(&flat, &(), &mut vt_replay);
    let _ = verify_spartan(&key, &spartan_proof, &mut vt_replay).unwrap();
    let (_, s2_challenges) =
        jolt_sumcheck::BatchedSumcheckVerifier::verify(&[claim], &sumcheck_proof, &mut vt_replay)
            .expect("sumcheck replay should succeed");

    let evaluations: Vec<Fr> = poly_tables
        .iter()
        .map(|table| Polynomial::new(table.clone()).evaluate(&s2_challenges))
        .collect();

    // Fiat-Shamir: flush opening claim evals to transcript (matches verifier).
    for &eval in &evaluations {
        eval.append_to_transcript(&mut pt);
    }

    let stage_proof = SumcheckStageProof {
        sumcheck_proof,
        evaluations: evaluations.clone(),
    };

    // Opening claims: stage polys + witness
    let mut prover_claims: Vec<ProverClaim<Fr>> = poly_tables
        .iter()
        .zip(evaluations.iter())
        .map(|(table, &eval)| ProverClaim {
            evaluations: table.clone(),
            point: s2_challenges.clone(),
            eval,
        })
        .collect();

    // Witness opening claim — must be last to match verifier ordering.
    let witness_eval = spartan_proof.witness_eval;
    let spartan_r_y = _prover_r_y;
    prover_claims.push(ProverClaim {
        evaluations: flat.clone(),
        point: spartan_r_y.clone(),
        eval: witness_eval,
    });

    use jolt_openings::{OpeningReduction, RlcReduction};
    let (reduced, ()) =
        <RlcReduction as OpeningReduction<MockPCS>>::reduce_prover(prover_claims, &mut pt);
    let pcs_proofs: Vec<_> = reduced
        .into_iter()
        .map(|claim| {
            let poly: <MockPCS as CommitmentScheme>::Polynomial = claim.evaluations.into();
            MockPCS::open(&poly, &claim.point, claim.eval, &(), None, &mut pt)
        })
        .collect();
    let proof = JoltProof {
        config: test_prover_config(),
        spartan_proof,
        stage_proofs: vec![stage_proof],
        opening_proofs: pcs_proofs,
        witness_commitment,
        commitments: commitments.clone(),
    };

    let vk = JoltVerifyingKey {
        spartan_key: key,
        pcs_setup: (),
    };

    let mut vt = Blake2bTranscript::new(b"full-pipeline");
    let (_r_x, r_y) = verify::<MockPCS, _>(
        &proof,
        &vk,
        |_r_x, _r_y, _t| {
            vec![StageDescriptor::claim_reduction(
                eq_point,
                coefficients,
                claimed_sum,
                (0..num_polys).collect(),
            )]
        },
        &mut vt,
    )
    .expect("full pipeline verification should succeed");

    assert!(!r_y.is_empty());
}

mod dory {
    use super::*;
    use jolt_dory::{DoryCommitment, DoryScheme};
    use jolt_openings::{OpeningReduction, RlcReduction};

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
            <RlcReduction as OpeningReduction<DoryScheme>>::reduce_prover(prover_claims, &mut pt);
        let proofs: Vec<_> = reduced_prover
            .into_iter()
            .map(|claim| {
                let poly: <DoryScheme as CommitmentScheme>::Polynomial = claim.evaluations.into();
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
        let mut vt = Blake2bTranscript::new(b"dory-openings-rt");
        verify_openings::<DoryScheme, _>(verifier_claims, &proofs, &verifier_setup, &mut vt)
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

        let key = test_uniform_key(2);
        let witnesses = vec![make_cycle_witness(3), make_cycle_witness(5)];
        let flat = flatten_witnesses(&key, &witnesses);

        let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let coefficients: Vec<Fr> = (0..num_polys).map(|_| Fr::random(&mut rng)).collect();
        let poly_tables: Vec<Vec<Fr>> = (0..num_polys)
            .map(|_| (0..n).map(|_| Fr::random(&mut rng)).collect())
            .collect();

        let commitments: Vec<DoryCommitment> = poly_tables
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
        let claimed_sum: Fr = eq_table
            .iter()
            .zip(g_table.iter())
            .map(|(&e, &g)| e * g)
            .sum();

        let mut pt = Blake2bTranscript::new(b"dory-full-pipeline");

        let witness_commitment = commit_and_append::<DoryScheme>(&flat, &prover_setup, &mut pt);
        let (spartan_proof, _prover_r_x, prover_r_y) =
            UniformSpartanProver::prove_dense_with_challenges(&key, &flat, &mut pt)
                .expect("spartan proving should succeed");

        let claim = SumcheckClaim {
            num_vars,
            degree: 2,
            claimed_sum,
        };
        let mut sc_witnesses: Vec<Box<dyn SumcheckCompute<Fr>>> = vec![Box::new(EqGWitness {
            eq: eq_table.clone(),
            g: g_table,
        })];
        let sumcheck_proof =
            BatchedSumcheckProver::prove(&[claim.clone()], &mut sc_witnesses, &mut pt);

        let mut vt_replay = Blake2bTranscript::new(b"dory-full-pipeline");
        let _ = commit_and_append::<DoryScheme>(&flat, &prover_setup, &mut vt_replay);
        let _ = verify_spartan(&key, &spartan_proof, &mut vt_replay).unwrap();
        let (_, s2_challenges) = jolt_sumcheck::BatchedSumcheckVerifier::verify(
            &[claim],
            &sumcheck_proof,
            &mut vt_replay,
        )
        .expect("sumcheck replay should succeed");

        let evaluations: Vec<Fr> = poly_tables
            .iter()
            .map(|table| Polynomial::new(table.clone()).evaluate(&s2_challenges))
            .collect();

        // Fiat-Shamir: flush opening claim evals to transcript (matches verifier).
        for &eval in &evaluations {
            eval.append_to_transcript(&mut pt);
        }

        let stage_proof = SumcheckStageProof {
            sumcheck_proof,
            evaluations: evaluations.clone(),
        };

        let mut prover_claims: Vec<ProverClaim<Fr>> = poly_tables
            .iter()
            .zip(evaluations.iter())
            .map(|(table, &eval)| ProverClaim {
                evaluations: table.clone(),
                point: s2_challenges.clone(),
                eval,
            })
            .collect();

        // Witness opening claim — last.
        prover_claims.push(ProverClaim {
            evaluations: flat.clone(),
            point: prover_r_y,
            eval: spartan_proof.witness_eval,
        });

        let (reduced, ()) =
            <RlcReduction as OpeningReduction<DoryScheme>>::reduce_prover(prover_claims, &mut pt);
        let pcs_proofs: Vec<_> = reduced
            .into_iter()
            .map(|claim| {
                let poly: <DoryScheme as CommitmentScheme>::Polynomial = claim.evaluations.into();
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
        let proof = JoltProof {
            config: test_prover_config(),
            spartan_proof,
            stage_proofs: vec![stage_proof],
            opening_proofs: pcs_proofs,
            witness_commitment,
            commitments: commitments.clone(),
        };

        let vk = JoltVerifyingKey {
            spartan_key: key,
            pcs_setup: verifier_setup,
        };

        let mut vt = Blake2bTranscript::new(b"dory-full-pipeline");
        let (_r_x, r_y) = verify::<DoryScheme, _>(
            &proof,
            &vk,
            |_r_x, _r_y, _t| {
                vec![StageDescriptor::claim_reduction(
                    eq_point,
                    coefficients,
                    claimed_sum,
                    (0..num_polys).collect(),
                )]
            },
            &mut vt,
        )
        .expect("Dory full pipeline verification should succeed");

        assert!(!r_y.is_empty());
    }
}
