//! Integration tests for stage regrouping via `CompositeStage`.
//!
//! Validates that heterogeneous stages (different `num_vars`, binding
//! orders, and evaluator types) compose correctly into a single batched
//! sumcheck matching the old jolt-core pipeline grouping.

use std::sync::Arc;

use jolt_cpu::CpuBackend;
use jolt_field::{Field, Fr};
use jolt_poly::{EqPolynomial, LtPolynomial, Polynomial};
use jolt_sumcheck::{BatchedSumcheckProver, BatchedSumcheckVerifier};
use jolt_transcript::{Blake2bTranscript, Transcript};
use num_traits::{One, Zero};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

use jolt_zkvm::stage::{CompositeStage, ProverStage};
use jolt_zkvm::stages::s3_claim_reductions::ClaimReductionStage;
use jolt_zkvm::stages::s5_registers_val_eval::RegistersValEvalStage;
use jolt_zkvm::stages::s6_booleanity::HammingBooleanityStage;
use jolt_zkvm::stages::s6_ra_booleanity::RaBooleanityStage;
use jolt_zkvm::stages::s_instruction_read_raf::InstructionReadRafStage;

fn cpu() -> Arc<CpuBackend> {
    Arc::new(CpuBackend)
}

fn random_table(n: usize, rng: &mut ChaCha20Rng) -> Vec<Fr> {
    (0..n).map(|_| Fr::random(rng)).collect()
}

fn random_bool_table(n: usize, rng: &mut ChaCha20Rng) -> Vec<Fr> {
    (0..n)
        .map(|_| Fr::from_u64(rng.next_u64() % 2))
        .collect()
}

/// Brute-force instruction read-RAF claimed sum.
fn brute_force_instruction_raf(
    f_table: &[Fr],
    val_table: &[Fr],
) -> Fr {
    f_table
        .iter()
        .zip(val_table.iter())
        .map(|(&f, &v)| f * v)
        .sum()
}

/// Brute-force registers val eval sum: Σ inc(j) · wa(j) · LT(r_cycle, j).
fn brute_force_val_eval(inc: &[Fr], wa: &[Fr], r_cycle: &[Fr]) -> Fr {
    let lt_table = LtPolynomial::evaluations(r_cycle);
    inc.iter()
        .zip(wa.iter())
        .zip(lt_table.iter())
        .map(|((&i, &w), &lt)| i * w * lt)
        .sum()
}

/// Group 5 test: InstructionReadRaf (log_K+log_T vars, SegmentedEvaluator) +
/// ClaimReduction (log_T vars, LowToHigh) + RegistersValEval (log_T vars, HighToLow).
///
/// This is the most critical grouping test:
/// - Heterogeneous num_vars (front-loaded batching)
/// - Heterogeneous binding orders
/// - SegmentedEvaluator (multi-phase address→cycle) within a composite
#[test]
fn group5_instruction_raf_plus_reduction_plus_val_eval() {
    let log_k = 3;
    let log_t = 4;
    let k = 1usize << log_k;
    let t = 1usize << log_t;
    let n_virtual = 1;
    let mut rng = ChaCha20Rng::seed_from_u64(42);

    // Shared eq point for cycle-domain instances.
    let r_reduction: Vec<Fr> = (0..log_t).map(|_| Fr::random(&mut rng)).collect();

    // --- InstructionReadRaf data ---
    let val_table = random_table(k, &mut rng);
    let pc_per_cycle: Vec<usize> = (0..t).map(|_| rng.next_u32() as usize % k).collect();
    let eq_table = EqPolynomial::new(r_reduction.clone()).evaluations();

    let mut f_table = vec![Fr::zero(); k];
    for j in 0..t {
        f_table[pc_per_cycle[j]] += eq_table[j];
    }
    let raf_claimed_sum = brute_force_instruction_raf(&f_table, &val_table);

    let pc_chunks = pc_per_cycle.clone();
    let ra_materializer: Box<dyn FnOnce(&[Fr], usize) -> Vec<Vec<Fr>> + Send + Sync> =
        Box::new(move |r_addr: &[Fr], _n_virtual: usize| {
            let eq_addr = EqPolynomial::new(r_addr.to_vec()).evaluations();
            let ra: Vec<Fr> = pc_chunks.iter().map(|&idx| eq_addr[idx]).collect();
            vec![ra]
        });

    let raf_stage: Box<dyn ProverStage<Fr, Blake2bTranscript>> = Box::new(
        InstructionReadRafStage::new(
            f_table,
            val_table,
            r_reduction.clone(),
            n_virtual,
            ra_materializer,
            raf_claimed_sum,
            cpu(),
        ),
    );

    // --- ClaimReduction data ---
    let ram_inc = random_table(t, &mut rng);
    let rd_inc = random_table(t, &mut rng);
    let c0 = Fr::random(&mut rng);
    let c1 = Fr::random(&mut rng);

    let reduction_stage: Box<dyn ProverStage<Fr, Blake2bTranscript>> = Box::new(
        ClaimReductionStage::increment(
            ram_inc.clone(),
            rd_inc.clone(),
            r_reduction.clone(),
            c0,
            c1,
            cpu(),
        ),
    );

    // --- RegistersValEval data ---
    let inc = random_table(t, &mut rng);
    let wa = random_table(t, &mut rng);
    let val_eval_sum = brute_force_val_eval(&inc, &wa, &r_reduction);

    let val_eval_stage: Box<dyn ProverStage<Fr, Blake2bTranscript>> =
        Box::new(RegistersValEvalStage::new(
            inc.clone(),
            wa.clone(),
            r_reduction.clone(),
            val_eval_sum,
        ));

    // --- Compose into Group 5 ---
    let mut composite = CompositeStage::new(
        "S5_composite",
        vec![raf_stage, reduction_stage, val_eval_stage],
    );

    let mut pt = Blake2bTranscript::new(b"group5_test");
    let mut batch = composite.build(&[], &mut pt);

    // InstructionReadRaf: 1 claim with log_K+log_T vars
    // ClaimReduction: 1 claim with log_T vars
    // RegistersValEval: 1 claim with log_T vars
    assert_eq!(batch.claims.len(), 3);
    assert_eq!(batch.claims[0].num_vars, log_k + log_t); // RAF
    assert_eq!(batch.claims[1].num_vars, log_t); // Reduction
    assert_eq!(batch.claims[2].num_vars, log_t); // ValEval

    let claims_snapshot: Vec<_> = batch.claims.clone();
    let proof = BatchedSumcheckProver::prove(&batch.claims, &mut batch.witnesses, &mut pt);

    // Verify
    let mut vt = Blake2bTranscript::new(b"group5_test");
    let (final_eval, challenges) =
        BatchedSumcheckVerifier::verify(&claims_snapshot, &proof, &mut vt)
            .expect("group5 verification should succeed");

    assert_eq!(challenges.len(), log_k + log_t);

    // Extract claims
    let all_claims = composite.extract_claims(&challenges, final_eval);

    // RAF: n_virtual=1 → 1 RA claim
    // Reduction: 2 claims (ram_inc, rd_inc)
    // ValEval: 2 claims (inc, wa)
    assert_eq!(all_claims.len(), 5, "expected 5 opening claims");

    // Verify each claim's eval matches poly.evaluate(point)
    for (i, claim) in all_claims.iter().enumerate() {
        let poly = Polynomial::new(claim.evaluations.clone());
        assert_eq!(
            poly.evaluate(&claim.point),
            claim.eval,
            "claim {i}: eval mismatch"
        );
    }

    // Verify specific claim values:

    // Claims 0: RAF RA poly (cycle point, reversed from LowToHigh)
    // Point should have length log_T (cycle portion only)
    assert_eq!(all_claims[0].point.len(), log_t, "RA claim point length");

    // Claims 1-2: ClaimReduction (ram_inc, rd_inc at cycle point, reversed from LowToHigh)
    // Their offset = log_K, so they get challenges[log_K..log_K+log_T], reversed
    let reduction_challenges = &challenges[log_k..];
    let reduction_eval_point: Vec<Fr> = reduction_challenges.iter().rev().copied().collect();
    let expected_ram_inc = Polynomial::new(ram_inc).evaluate(&reduction_eval_point);
    assert_eq!(
        all_claims[1].eval, expected_ram_inc,
        "ram_inc eval mismatch"
    );
    let expected_rd_inc = Polynomial::new(rd_inc).evaluate(&reduction_eval_point);
    assert_eq!(
        all_claims[2].eval, expected_rd_inc,
        "rd_inc eval mismatch"
    );

    // Claims 3-4: RegistersValEval (inc, wa at cycle point, HighToLow = no reversal)
    let val_eval_challenges = &challenges[log_k..];
    let val_eval_point: Vec<Fr> = val_eval_challenges.to_vec();
    let expected_inc = Polynomial::new(inc).evaluate(&val_eval_point);
    assert_eq!(
        all_claims[3].eval, expected_inc,
        "inc eval mismatch"
    );
    let expected_wa = Polynomial::new(wa).evaluate(&val_eval_point);
    assert_eq!(
        all_claims[4].eval, expected_wa,
        "wa eval mismatch"
    );
}

/// Group 6 test: RaBooleanity (log_K+log_T, HighToLow) +
/// HammingBooleanity (log_T, LowToHigh) +
/// ClaimReduction/inc (log_T, LowToHigh).
///
/// Validates front-loaded batching across both zero-check (claimed_sum=0)
/// and non-zero claim instances with mixed binding orders.
#[test]
fn group6_ra_booleanity_plus_hamming_plus_reduction() {
    let log_k = 2;
    let log_t = 3;
    let log_kt = log_k + log_t;
    let kt = 1usize << log_kt;
    let t = 1usize << log_t;
    let d = 3; // number of RA polynomials
    let mut rng = ChaCha20Rng::seed_from_u64(999);

    // RA booleanity data: boolean RA polys over (address, cycle) domain
    let ra_polys: Vec<Vec<Fr>> = (0..d).map(|_| random_bool_table(kt, &mut rng)).collect();
    let eq_point_kt: Vec<Fr> = (0..log_kt).map(|_| Fr::random(&mut rng)).collect();
    let gamma = Fr::random(&mut rng);
    let gamma_powers: Vec<Fr> = {
        let mut g = Fr::one(); // requires num_traits::One
        (0..d)
            .map(|_| {
                let v = g;
                g *= gamma;
                v
            })
            .collect()
    };

    let ra_bool_stage: Box<dyn ProverStage<Fr, Blake2bTranscript>> = Box::new(
        RaBooleanityStage::new(ra_polys.clone(), eq_point_kt.clone(), gamma_powers),
    );

    // Hamming booleanity: boolean h poly over cycle domain
    let h_evals: Vec<Fr> = random_bool_table(t, &mut rng);
    let eq_point_t: Vec<Fr> = (0..log_t).map(|_| Fr::random(&mut rng)).collect();

    let hamming_stage: Box<dyn ProverStage<Fr, Blake2bTranscript>> =
        Box::new(HammingBooleanityStage::new(h_evals.clone(), eq_point_t, cpu()));

    // ClaimReduction: 2 polys over cycle domain
    let p0 = random_table(t, &mut rng);
    let p1 = random_table(t, &mut rng);
    let r_cycle: Vec<Fr> = (0..log_t).map(|_| Fr::random(&mut rng)).collect();
    let c0 = Fr::random(&mut rng);
    let c1 = Fr::random(&mut rng);

    let reduction_stage: Box<dyn ProverStage<Fr, Blake2bTranscript>> = Box::new(
        ClaimReductionStage::increment(p0.clone(), p1.clone(), r_cycle, c0, c1, cpu()),
    );

    // Compose into Group 6
    let mut composite = CompositeStage::new(
        "S6_composite",
        vec![ra_bool_stage, hamming_stage, reduction_stage],
    );

    let mut pt = Blake2bTranscript::new(b"group6_test");
    let mut batch = composite.build(&[], &mut pt);

    assert_eq!(batch.claims.len(), 3);
    assert_eq!(batch.claims[0].num_vars, log_kt); // RA booleanity
    assert_eq!(batch.claims[1].num_vars, log_t); // Hamming booleanity
    assert_eq!(batch.claims[2].num_vars, log_t); // Reduction

    // Both booleanity claims should be zero-checks
    assert!(batch.claims[0].claimed_sum.is_zero());
    assert!(batch.claims[1].claimed_sum.is_zero());

    let claims_snapshot: Vec<_> = batch.claims.clone();
    let proof = BatchedSumcheckProver::prove(&batch.claims, &mut batch.witnesses, &mut pt);

    let mut vt = Blake2bTranscript::new(b"group6_test");
    let (final_eval, challenges) =
        BatchedSumcheckVerifier::verify(&claims_snapshot, &proof, &mut vt)
            .expect("group6 verification should succeed");

    assert_eq!(challenges.len(), log_kt);

    let all_claims = composite.extract_claims(&challenges, final_eval);

    // RA booleanity: d=3 RA polys
    // Hamming: 1 h poly
    // Reduction: 2 polys
    assert_eq!(all_claims.len(), d + 1 + 2);

    for (i, claim) in all_claims.iter().enumerate() {
        let poly = Polynomial::new(claim.evaluations.clone());
        assert_eq!(
            poly.evaluate(&claim.point),
            claim.eval,
            "claim {i}: eval mismatch"
        );
    }

    // RA booleanity claims: HighToLow, full challenge vector (offset=0)
    let ra_eval_point: Vec<Fr> = challenges.to_vec();
    for i in 0..d {
        let expected = Polynomial::new(ra_polys[i].clone()).evaluate(&ra_eval_point);
        assert_eq!(all_claims[i].eval, expected, "RA poly {i}: eval mismatch");
        assert_eq!(all_claims[i].point.len(), log_kt);
    }

    // Hamming booleanity: LowToHigh, offset=log_K, reversed
    let hamming_challenges = &challenges[log_k..];
    let hamming_eval_point: Vec<Fr> = hamming_challenges.iter().rev().copied().collect();
    let expected_h = Polynomial::new(h_evals).evaluate(&hamming_eval_point);
    assert_eq!(all_claims[d].eval, expected_h, "hamming h eval mismatch");
    assert_eq!(all_claims[d].point.len(), log_t);

    // Reduction: LowToHigh, offset=log_K, reversed
    let reduction_challenges = &challenges[log_k..];
    let reduction_eval_point: Vec<Fr> = reduction_challenges.iter().rev().copied().collect();
    let expected_p0 = Polynomial::new(p0).evaluate(&reduction_eval_point);
    assert_eq!(
        all_claims[d + 1].eval, expected_p0,
        "reduction p0 eval mismatch"
    );
    let expected_p1 = Polynomial::new(p1).evaluate(&reduction_eval_point);
    assert_eq!(
        all_claims[d + 2].eval, expected_p1,
        "reduction p1 eval mismatch"
    );
}

/// Verifies that the pipeline's sequential stage execution with
/// `CompositeStage`s produces correct claim chaining (Group 5 → Group 6).
///
/// This simulates two sequential grouped sumchecks where Group 6
/// receives prior claims from Group 5.
#[test]
fn sequential_groups_claim_chaining() {
    let log_t = 4;
    let t = 1usize << log_t;
    let mut rng = ChaCha20Rng::seed_from_u64(777);

    // Group A: single ClaimReduction (log_T vars)
    let r_a: Vec<Fr> = (0..log_t).map(|_| Fr::random(&mut rng)).collect();
    let p0 = random_table(t, &mut rng);
    let p1 = random_table(t, &mut rng);
    let c0 = Fr::random(&mut rng);
    let c1 = Fr::random(&mut rng);

    let stage_a: Box<dyn ProverStage<Fr, Blake2bTranscript>> = Box::new(
        ClaimReductionStage::increment(p0.clone(), p1.clone(), r_a, c0, c1, cpu()),
    );
    let mut group_a = CompositeStage::new("group_a", vec![stage_a]);

    // Group B: Hamming booleanity (log_T vars)
    let h_evals: Vec<Fr> = random_bool_table(t, &mut rng);
    let eq_b: Vec<Fr> = (0..log_t).map(|_| Fr::random(&mut rng)).collect();

    let stage_b: Box<dyn ProverStage<Fr, Blake2bTranscript>> =
        Box::new(HammingBooleanityStage::new(h_evals.clone(), eq_b, cpu()));
    let mut group_b = CompositeStage::new("group_b", vec![stage_b]);

    // Run Group A
    let mut transcript = Blake2bTranscript::new(b"chain_test");
    let mut batch_a = group_a.build(&[], &mut transcript);
    let claims_a = batch_a.claims.clone();
    let proof_a =
        BatchedSumcheckProver::prove(&batch_a.claims, &mut batch_a.witnesses, &mut transcript);

    let mut vt = Blake2bTranscript::new(b"chain_test");
    let (fe_a, challenges_a) = BatchedSumcheckVerifier::verify(&claims_a, &proof_a, &mut vt)
        .expect("group_a verification failed");

    let prior_claims = group_a.extract_claims(&challenges_a, fe_a);
    assert_eq!(prior_claims.len(), 2); // ram_inc, rd_inc

    // Absorb prior claim evals into prover transcript (matching pipeline behavior)
    for claim in &prior_claims {
        jolt_transcript::AppendToTranscript::append_to_transcript(&claim.eval, &mut transcript);
    }

    // Run Group B with prior claims from Group A
    let mut batch_b = group_b.build(&prior_claims, &mut transcript);
    let claims_b = batch_b.claims.clone();
    let proof_b =
        BatchedSumcheckProver::prove(&batch_b.claims, &mut batch_b.witnesses, &mut transcript);

    // Absorb prior claim evals into verifier transcript
    for claim in &prior_claims {
        jolt_transcript::AppendToTranscript::append_to_transcript(&claim.eval, &mut vt);
    }

    let (fe_b, challenges_b) = BatchedSumcheckVerifier::verify(&claims_b, &proof_b, &mut vt)
        .expect("group_b verification failed");

    let claims_b_out = group_b.extract_claims(&challenges_b, fe_b);
    assert_eq!(claims_b_out.len(), 1); // h poly

    // Verify h claim
    let poly = Polynomial::new(h_evals);
    assert_eq!(
        poly.evaluate(&claims_b_out[0].point),
        claims_b_out[0].eval,
        "chained h eval mismatch"
    );
}
