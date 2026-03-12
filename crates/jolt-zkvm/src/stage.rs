//! Prover stage trait and batch output.
//!
//! Each sumcheck stage in the Jolt pipeline implements [`ProverStage`],
//! producing claims and witnesses for [`BatchedSumcheckProver`](jolt_sumcheck::BatchedSumcheckProver).

use jolt_field::WithChallenge;
use jolt_ir::ClaimDefinition;
use jolt_openings::ProverClaim;
use jolt_sumcheck::claim::SumcheckClaim;
use jolt_sumcheck::prover::SumcheckCompute;
use jolt_transcript::Transcript;

/// Output of a stage's [`build()`](ProverStage::build) method.
///
/// Contains paired sumcheck claims and witnesses, ready to feed into
/// [`BatchedSumcheckProver::prove`](jolt_sumcheck::BatchedSumcheckProver::prove).
pub struct StageBatch<F: WithChallenge> {
    /// Sumcheck claims (one per instance in this stage).
    pub claims: Vec<SumcheckClaim<F>>,
    /// Sumcheck witnesses (element-wise paired with `claims`).
    pub witnesses: Vec<Box<dyn SumcheckCompute<F>>>,
}

/// A proving stage that contributes batched sumcheck instances.
///
/// Each of the 7 sumcheck stages in the Jolt pipeline implements this trait.
/// The prover pipeline calls [`build()`](Self::build) to construct claims and
/// witnesses, runs the batched sumcheck prover, then calls
/// [`extract_claims()`](Self::extract_claims) to produce opening claims for
/// downstream stages and stage 8.
///
/// Generic over `T: Transcript` for dyn-compatibility — the pipeline fixes
/// a concrete transcript type and uses `dyn ProverStage<F, T>`.
pub trait ProverStage<F: WithChallenge, T: Transcript> {
    /// Human-readable stage name for tracing and profiling.
    ///
    /// Used by the pipeline to create named tracing spans (e.g. in Perfetto).
    /// Must be one of the known stage names so the pipeline can create
    /// properly-named spans visible in Perfetto/Chrome traces.
    fn name(&self) -> &'static str;

    /// Constructs sumcheck claims and witnesses for this stage.
    ///
    /// `prior_claims` contains opening claims from all previous stages,
    /// used to derive input claims for this stage's sumcheck instances.
    fn build(&mut self, prior_claims: &[ProverClaim<F>], transcript: &mut T) -> StageBatch<F>;

    /// Extracts opening claims after sumcheck completes.
    ///
    /// Given the sumcheck challenge vector and final evaluation, produces
    /// `ProverClaim`s that:
    /// - Feed into subsequent stages as input claims
    /// - Feed into stage 8 for RLC reduction and PCS opening proofs
    ///
    /// Implementations typically evaluate the original polynomials at the
    /// challenge point and pair evaluations with the evaluation tables
    /// (moved from the [`WitnessStore`](crate::witness::WitnessStore)).
    fn extract_claims(&mut self, challenges: &[F], final_eval: F) -> Vec<ProverClaim<F>>;

    /// IR-based claim definitions for this stage's sumcheck instances.
    ///
    /// Returns the same formulas used by [`build()`](Self::build) to
    /// construct claims, in symbolic form. Used by BlindFold (ZK mode)
    /// to build verifier R1CS constraints, and by tests to verify that
    /// claim formulas produce correct evaluations.
    fn claim_definitions(&self) -> Vec<ClaimDefinition>;
}

/// Composes multiple [`ProverStage`]s into a single batched stage.
///
/// All sub-stages' claims and witnesses are combined into one [`StageBatch`],
/// which the pipeline processes as a single batched sumcheck. Claims with
/// different `num_vars` are handled by
/// [`BatchedSumcheckProver`](jolt_sumcheck::BatchedSumcheckProver)'s
/// front-loaded batching.
///
/// In [`extract_claims`](ProverStage::extract_claims), each sub-stage
/// receives the challenge slice corresponding to its active rounds
/// (accounting for the front-loaded offset).
pub struct CompositeStage<F: WithChallenge, T: Transcript> {
    sub_stages: Vec<Box<dyn ProverStage<F, T>>>,
    /// `(offset, num_vars)` per sub-stage, computed during `build()`.
    /// `offset = max_num_vars - sub_stage_max_num_vars`.
    sub_stage_layout: Vec<(usize, usize)>,
    name: &'static str,
}

impl<F: WithChallenge, T: Transcript> CompositeStage<F, T> {
    pub fn new(name: &'static str, sub_stages: Vec<Box<dyn ProverStage<F, T>>>) -> Self {
        Self {
            sub_stages,
            sub_stage_layout: Vec::new(),
            name,
        }
    }
}

impl<F: WithChallenge, T: Transcript> ProverStage<F, T> for CompositeStage<F, T> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn build(&mut self, prior_claims: &[ProverClaim<F>], transcript: &mut T) -> StageBatch<F> {
        let mut all_claims = Vec::new();
        let mut all_witnesses = Vec::new();

        let mut sub_max_vars = Vec::with_capacity(self.sub_stages.len());
        for stage in &mut self.sub_stages {
            let batch = stage.build(prior_claims, transcript);
            let nv = batch.claims.iter().map(|c| c.num_vars).max().unwrap_or(0);
            sub_max_vars.push(nv);
            all_claims.extend(batch.claims);
            all_witnesses.extend(batch.witnesses);
        }

        let max_nv = all_claims.iter().map(|c| c.num_vars).max().unwrap_or(0);
        self.sub_stage_layout = sub_max_vars.iter().map(|&nv| (max_nv - nv, nv)).collect();

        StageBatch {
            claims: all_claims,
            witnesses: all_witnesses,
        }
    }

    fn extract_claims(&mut self, challenges: &[F], final_eval: F) -> Vec<ProverClaim<F>> {
        self.sub_stages
            .iter_mut()
            .zip(&self.sub_stage_layout)
            .flat_map(|(stage, &(offset, num_vars))| {
                let sub_challenges = &challenges[offset..offset + num_vars];
                stage.extract_claims(sub_challenges, final_eval)
            })
            .collect()
    }

    fn claim_definitions(&self) -> Vec<ClaimDefinition> {
        self.sub_stages
            .iter()
            .flat_map(|stage| stage.claim_definitions())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};
    use jolt_poly::Polynomial;
    use jolt_sumcheck::{BatchedSumcheckProver, BatchedSumcheckVerifier};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use crate::stages::s2_product_virtual::{
        brute_force_product_virtual_sum, ProductVirtualStage, NUM_CONSTRAINTS,
    };
    use crate::stages::s3_claim_reductions::ClaimReductionStage;
    use crate::stages::s4_ram_rw::RamRwCheckingStage;
    use crate::stages::s5_ram_checking::RamCheckingStage;
    use crate::stages::s6_booleanity::HammingBooleanityStage;
    use jolt_cpu::CpuBackend;
    use num_traits::One;
    use rand_core::RngCore;
    use std::sync::Arc;

    fn cpu() -> Arc<CpuBackend> {
        Arc::new(CpuBackend)
    }

    #[test]
    fn composite_same_num_vars() {
        let num_vars = 4;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let gamma = Fr::random(&mut rng);

        // Sub-stage 1: booleanity check (degree 3)
        let h_evals: Vec<Fr> = (0..n).map(|i| Fr::from_u64(i as u64 % 2)).collect();
        let booleanity: Box<dyn ProverStage<Fr, Blake2bTranscript>> = Box::new(
            HammingBooleanityStage::new(h_evals.clone(), eq_point.clone(), cpu()),
        );

        // Sub-stage 2: claim reduction (degree 2)
        let p0: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let p1: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let reduction: Box<dyn ProverStage<Fr, Blake2bTranscript>> =
            Box::new(ClaimReductionStage::registers(
                p0.clone(),
                p1.clone(),
                (0..n).map(|_| Fr::random(&mut rng)).collect(),
                eq_point.clone(),
                gamma,
                cpu(),
            ));

        let mut composite = CompositeStage::new("S2_composite_test", vec![booleanity, reduction]);

        let mut pt = Blake2bTranscript::new(b"composite_test");
        let mut batch = composite.build(&[], &mut pt);

        // 1 claim from booleanity + 1 from reduction
        assert_eq!(batch.claims.len(), 2);
        assert_eq!(batch.claims[0].degree, 3);
        assert_eq!(batch.claims[1].degree, 2);

        let claims_snapshot: Vec<_> = batch.claims.clone();
        let proof = BatchedSumcheckProver::prove(
            &batch.claims,
            &mut batch.witnesses,
            &mut pt,
        );

        let mut vt = Blake2bTranscript::new(b"composite_test");
        let (_final_eval, challenges) = BatchedSumcheckVerifier::verify(
            &claims_snapshot,
            &proof,
            &mut vt,
        )
        .expect("verification should succeed");

        let all_claims = composite.extract_claims(&challenges, _final_eval);

        // Booleanity: 1 poly (h). Reduction: 3 polys (rd_wv, rs1_v, rs2_v).
        assert_eq!(all_claims.len(), 4);

        let eval_point: Vec<Fr> = challenges.iter().rev().copied().collect();

        // First claim from booleanity
        let expected = Polynomial::new(h_evals).evaluate(&eval_point);
        assert_eq!(all_claims[0].eval, expected, "booleanity eval mismatch");

        // Reduction claims
        let expected_p0 = Polynomial::new(p0).evaluate(&eval_point);
        assert_eq!(all_claims[1].eval, expected_p0, "reduction poly 0 mismatch");
        let expected_p1 = Polynomial::new(p1).evaluate(&eval_point);
        assert_eq!(all_claims[2].eval, expected_p1, "reduction poly 1 mismatch");
    }

    #[test]
    fn composite_different_num_vars() {
        let mut rng = ChaCha20Rng::seed_from_u64(77);

        // Sub-stage 1: booleanity with 3 vars (degree 3)
        let nv1 = 3;
        let n1 = 1usize << nv1;
        let eq1: Vec<Fr> = (0..nv1).map(|_| Fr::random(&mut rng)).collect();
        let h1: Vec<Fr> = (0..n1).map(|i| Fr::from_u64(i as u64 % 2)).collect();
        let stage1: Box<dyn ProverStage<Fr, Blake2bTranscript>> =
            Box::new(HammingBooleanityStage::new(h1.clone(), eq1, cpu()));

        // Sub-stage 2: booleanity with 5 vars (degree 3) — longer
        let nv2 = 5;
        let n2 = 1usize << nv2;
        let eq2: Vec<Fr> = (0..nv2).map(|_| Fr::random(&mut rng)).collect();
        let h2: Vec<Fr> = (0..n2).map(|i| Fr::from_u64(i as u64 % 2)).collect();
        let stage2: Box<dyn ProverStage<Fr, Blake2bTranscript>> =
            Box::new(HammingBooleanityStage::new(h2.clone(), eq2, cpu()));

        let mut composite = CompositeStage::new("S2_mixed_vars_test", vec![stage1, stage2]);

        let mut pt = Blake2bTranscript::new(b"mixed_vars");
        let mut batch = composite.build(&[], &mut pt);

        assert_eq!(batch.claims.len(), 2);
        assert_eq!(batch.claims[0].num_vars, nv1);
        assert_eq!(batch.claims[1].num_vars, nv2);

        // Layout: stage1 offset = 5-3 = 2, stage2 offset = 0
        assert_eq!(composite.sub_stage_layout, vec![(2, 3), (0, 5)]);

        let claims_snapshot: Vec<_> = batch.claims.clone();
        let proof = BatchedSumcheckProver::prove(
            &batch.claims,
            &mut batch.witnesses,
            &mut pt,
        );

        let mut vt = Blake2bTranscript::new(b"mixed_vars");
        let (_final_eval, challenges) = BatchedSumcheckVerifier::verify(
            &claims_snapshot,
            &proof,
            &mut vt,
        )
        .expect("verification should succeed");

        assert_eq!(challenges.len(), nv2); // max_num_vars

        let all_claims = composite.extract_claims(&challenges, _final_eval);
        assert_eq!(all_claims.len(), 2);

        // Stage 1 (offset=2, nv=3): uses challenges[2..5], reversed for eval
        let sub1_challenges = &challenges[2..5];
        let eval_point1: Vec<Fr> = sub1_challenges.iter().rev().copied().collect();
        let expected1 = Polynomial::new(h1).evaluate(&eval_point1);
        assert_eq!(all_claims[0].eval, expected1, "stage1 eval mismatch");

        // Stage 2 (offset=0, nv=5): uses all challenges, reversed
        let eval_point2: Vec<Fr> = challenges.iter().rev().copied().collect();
        let expected2 = Polynomial::new(h2).evaluate(&eval_point2);
        assert_eq!(all_claims[1].eval, expected2, "stage2 eval mismatch");
    }

    /// S2-like composition: ProductVirtual (deg 3) + RamRW (deg 3) +
    /// RamChecking (2× deg 2). Four sub-stages, three degree types.
    #[test]
    fn composite_s2_like_batch() {
        let num_vars = 4;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(999);

        let random_poly =
            |rng: &mut ChaCha20Rng| -> Vec<Fr> { (0..n).map(|_| Fr::random(rng)).collect() };
        let random_bool_poly = |rng: &mut ChaCha20Rng| -> Vec<Fr> {
            (0..n).map(|_| Fr::from_u64(rng.next_u64() % 2)).collect()
        };

        let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        let factor_polys: Vec<Vec<Fr>> = vec![
            random_poly(&mut rng),      // left_inst
            random_poly(&mut rng),      // right_inst
            random_bool_poly(&mut rng), // is_rd_nz
            random_bool_poly(&mut rng), // wl_flag
            random_bool_poly(&mut rng), // jump_flag
            random_poly(&mut rng),      // lookup_out
            random_bool_poly(&mut rng), // branch_flag
            random_bool_poly(&mut rng), // next_noop
        ];
        let gamma_pv = Fr::from_u64(13);
        let g_pv: Vec<Fr> = {
            let mut g = Fr::one();
            (0..NUM_CONSTRAINTS)
                .map(|_| {
                    let v = g;
                    g *= gamma_pv;
                    v
                })
                .collect()
        };
        let pv_sum = brute_force_product_virtual_sum(&factor_polys, &eq_point, &g_pv);
        let pv_stage: Box<dyn ProverStage<Fr, Blake2bTranscript>> = Box::new(
            ProductVirtualStage::new(factor_polys, eq_point.clone(), g_pv, pv_sum, cpu()),
        );

        let ra = random_poly(&mut rng);
        let val = random_poly(&mut rng);
        let inc = random_poly(&mut rng);
        let c0 = Fr::random(&mut rng);
        let c1 = Fr::random(&mut rng);
        let rw_stage: Box<dyn ProverStage<Fr, Blake2bTranscript>> = Box::new(
            RamRwCheckingStage::new(ra, val, inc, eq_point.clone(), [c0, c1], cpu()),
        );

        let val_final = random_poly(&mut rng);
        let ram_ra = random_poly(&mut rng);
        let oc0 = Fr::random(&mut rng);
        let oc1 = Fr::random(&mut rng);
        let raf_c0 = Fr::random(&mut rng);
        let rc_stage: Box<dyn ProverStage<Fr, Blake2bTranscript>> =
            Box::new(RamCheckingStage::new(
                val_final,
                eq_point.clone(),
                [oc0, oc1],
                ram_ra,
                eq_point.clone(),
                raf_c0,
                cpu(),
            ));

        let mut composite = CompositeStage::new("S2_composite", vec![pv_stage, rw_stage, rc_stage]);

        let mut pt = Blake2bTranscript::new(b"s2_batch");
        let mut batch = composite.build(&[], &mut pt);

        // PV: 1 claim (deg 3), RW: 1 claim (deg 3), RC: 2 claims (deg 2)
        assert_eq!(batch.claims.len(), 4);
        assert_eq!(batch.claims[0].degree, 3);
        assert_eq!(batch.claims[1].degree, 3);
        assert_eq!(batch.claims[2].degree, 2);
        assert_eq!(batch.claims[3].degree, 2);

        let claims_snapshot: Vec<_> = batch.claims.clone();
        let proof = BatchedSumcheckProver::prove(
            &batch.claims,
            &mut batch.witnesses,
            &mut pt,
        );

        let mut vt = Blake2bTranscript::new(b"s2_batch");
        let (_final_eval, challenges) = BatchedSumcheckVerifier::verify(
            &claims_snapshot,
            &proof,
            &mut vt,
        )
        .expect("verification should succeed");

        let all_claims = composite.extract_claims(&challenges, _final_eval);
        // PV: 8 polys, RW: 3 polys, RC: 2 polys → 13 total
        assert_eq!(all_claims.len(), 13);

        // All eval_points should be correct (same num_vars → same challenges)
        let eval_point: Vec<Fr> = challenges.iter().rev().copied().collect();
        for claim in &all_claims {
            let poly = Polynomial::new(claim.evaluations.clone());
            assert_eq!(poly.evaluate(&eval_point), claim.eval, "eval mismatch");
        }
    }
}
