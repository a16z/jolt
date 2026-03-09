//! Stage 3: Claim reduction sumchecks.
//!
//! Reduces multiple opening claims from the Spartan outer sumcheck into
//! fewer opening points via eq-weighted sumchecks. Each reduction batches
//! related polynomial claims using γ-power coefficients.
//!
//! All claim reductions share the same structure:
//! $\sum_{x} \widetilde{eq}(r, x) \cdot \sum_i c_i \cdot p_i(x) = v$
//! which is degree 2 (1 from eq + 1 from p_i).

use jolt_field::Field;
use jolt_ir::ClaimDefinition;
use jolt_openings::ProverClaim;
use jolt_poly::{EqPolynomial, Polynomial};
use jolt_sumcheck::claim::SumcheckClaim;
use jolt_transcript::Transcript;

use crate::claims::reductions;
use crate::stage::{ProverStage, StageBatch};
use crate::witnesses::eq_product::EqProductCompute;

/// A single claim reduction instance within the stage.
///
/// Each instance reduces a set of related polynomial openings into
/// a single opening point. The coefficients combine eq evaluations
/// and γ-powers from the batching challenge.
struct ReductionInstance<F: Field> {
    /// Evaluation tables for the polynomials being reduced.
    poly_tables: Vec<Vec<F>>,
    /// Pre-computed scalar coefficients for the linear combination.
    coefficients: Vec<F>,
}

/// Claim reduction prover stage.
///
/// Contains multiple reduction instances, each handling a group of
/// related polynomial claims. Each instance produces one sumcheck
/// claim and witness pair.
pub struct ClaimReductionStage<F: Field> {
    instances: Option<Vec<ReductionInstance<F>>>,
    /// Eq evaluation point (shared across all instances).
    eq_point: Vec<F>,
    /// Claim definition generators, one per instance.
    claim_def_fns: Vec<fn() -> ClaimDefinition>,
    num_vars: usize,
}

impl<F: Field> ClaimReductionStage<F> {
    /// Creates a new claim reduction stage.
    ///
    /// Each element of `instances` is a `(poly_tables, coefficients)` pair
    /// describing one reduction instance. The coefficients are typically
    /// derived from eq evaluations and γ-powers.
    ///
    /// `claim_def_fns` must have the same length as `instances`.
    ///
    /// # Panics
    ///
    /// Panics if table lengths don't match 2^eq_point.len(), or if
    /// `instances` and `claim_def_fns` differ in length.
    pub fn new(
        instances: Vec<(Vec<Vec<F>>, Vec<F>)>,
        eq_point: Vec<F>,
        claim_def_fns: Vec<fn() -> ClaimDefinition>,
    ) -> Self {
        let num_vars = eq_point.len();
        let expected = 1usize << num_vars;
        assert_eq!(
            instances.len(),
            claim_def_fns.len(),
            "instances and claim_def_fns length mismatch"
        );

        let instances: Vec<_> = instances
            .into_iter()
            .enumerate()
            .map(|(inst_idx, (poly_tables, coefficients))| {
                assert_eq!(
                    poly_tables.len(),
                    coefficients.len(),
                    "instance {inst_idx}: poly_tables and coefficients length mismatch"
                );
                for (i, table) in poly_tables.iter().enumerate() {
                    assert_eq!(
                        table.len(),
                        expected,
                        "instance {inst_idx}, poly_tables[{i}] length {} != {expected}",
                        table.len(),
                    );
                }
                ReductionInstance {
                    poly_tables,
                    coefficients,
                }
            })
            .collect();

        Self {
            instances: Some(instances),
            eq_point,
            claim_def_fns,
            num_vars,
        }
    }

    /// Creates a single-instance registers claim reduction stage.
    ///
    /// Convenience constructor for the registers reduction pattern:
    /// `eq · (rd_wv + γ·rs1_v + γ²·rs2_v)`.
    pub fn registers(
        rd_wv: Vec<F>,
        rs1_v: Vec<F>,
        rs2_v: Vec<F>,
        eq_point: Vec<F>,
        gamma: F,
    ) -> Self {
        let gamma_sq = gamma * gamma;
        let eq_eval = F::one(); // Will be derived from prior claims in production
        let coefficients = vec![eq_eval, eq_eval * gamma, eq_eval * gamma_sq];
        Self::new(
            vec![(vec![rd_wv, rs1_v, rs2_v], coefficients)],
            eq_point,
            vec![reductions::registers_claim_reduction as fn() -> ClaimDefinition],
        )
    }

    /// Creates a single-instance increment claim reduction stage.
    ///
    /// Convenience constructor for `c0·ram_inc + c1·rd_inc`.
    pub fn increment(ram_inc: Vec<F>, rd_inc: Vec<F>, eq_point: Vec<F>, c0: F, c1: F) -> Self {
        Self::new(
            vec![(vec![ram_inc, rd_inc], vec![c0, c1])],
            eq_point,
            vec![reductions::increment_claim_reduction as fn() -> ClaimDefinition],
        )
    }
}

impl<F: Field, T: Transcript> ProverStage<F, T> for ClaimReductionStage<F> {
    fn build(&mut self, _prior_claims: &[ProverClaim<F>], _transcript: &mut T) -> StageBatch<F> {
        let instances = self
            .instances
            .as_ref()
            .expect("build() called after extract_claims()");
        let n = 1usize << self.num_vars;
        let eq_table = EqPolynomial::new(self.eq_point.clone()).evaluations();

        let mut claims = Vec::with_capacity(instances.len());
        let mut witnesses: Vec<Box<dyn SumcheckCompute<F>>> = Vec::with_capacity(instances.len());

        for inst in instances {
            let mut g_table = vec![F::zero(); n];
            for (i, table) in inst.poly_tables.iter().enumerate() {
                let c = inst.coefficients[i];
                for (j, g) in g_table.iter_mut().enumerate() {
                    *g += c * table[j];
                }
            }

            let claimed_sum: F = eq_table
                .iter()
                .zip(g_table.iter())
                .map(|(&e, &g)| e * g)
                .sum();

            claims.push(SumcheckClaim {
                num_vars: self.num_vars,
                degree: 2,
                claimed_sum,
            });

            witnesses.push(Box::new(EqProductCompute::new(
                g_table,
                eq_table.clone(),
                self.num_vars,
            )));
        }

        StageBatch { claims, witnesses }
    }

    fn extract_claims(&mut self, challenges: &[F], _final_eval: F) -> Vec<ProverClaim<F>> {
        let instances = self
            .instances
            .take()
            .expect("extract_claims() called twice");

        instances
            .into_iter()
            .flat_map(|inst| {
                inst.poly_tables.into_iter().map(|evals| {
                    let poly = Polynomial::new(evals.clone());
                    let eval = poly.evaluate(challenges);
                    ProverClaim {
                        evaluations: evals,
                        point: challenges.to_vec(),
                        eval,
                    }
                })
            })
            .collect()
    }

    fn claim_definitions(&self) -> Vec<ClaimDefinition> {
        self.claim_def_fns.iter().map(|f| f()).collect()
    }
}

use jolt_sumcheck::prover::SumcheckCompute;

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};
    use jolt_sumcheck::{BatchedSumcheckProver, BatchedSumcheckVerifier};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn registers_reduction_prove_verify() {
        let num_vars = 3;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let gamma = Fr::random(&mut rng);

        let rd_wv: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let rs1_v: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let rs2_v: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let mut stage: ClaimReductionStage<Fr> =
            ClaimReductionStage::registers(rd_wv, rs1_v, rs2_v, eq_point, gamma);

        let mut pt = Blake2bTranscript::new(b"s3_registers");
        let mut batch = stage.build(&[], &mut pt);

        assert_eq!(batch.claims.len(), 1);
        assert_eq!(batch.claims[0].degree, 2);

        let claim = batch.claims[0].clone();
        let proof = BatchedSumcheckProver::prove(
            &batch.claims,
            &mut batch.witnesses,
            &mut pt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        let mut vt = Blake2bTranscript::new(b"s3_registers");
        let (final_eval, challenges) = BatchedSumcheckVerifier::verify(
            &[claim],
            &proof,
            &mut vt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        )
        .expect("verification should succeed");

        let prover_claims =
            <ClaimReductionStage<Fr> as ProverStage<Fr, Blake2bTranscript>>::extract_claims(
                &mut stage,
                &challenges,
                final_eval,
            );
        // 3 polynomials → 3 opening claims
        assert_eq!(prover_claims.len(), 3);

        for pc in &prover_claims {
            let poly = Polynomial::new(pc.evaluations.clone());
            assert_eq!(poly.evaluate(&challenges), pc.eval);
        }
    }

    #[test]
    fn increment_reduction_prove_verify() {
        let num_vars = 4;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(77);
        let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        let ram_inc: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let rd_inc: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let c0 = Fr::random(&mut rng);
        let c1 = Fr::random(&mut rng);

        let mut stage: ClaimReductionStage<Fr> =
            ClaimReductionStage::increment(ram_inc, rd_inc, eq_point, c0, c1);

        let mut pt = Blake2bTranscript::new(b"s3_increment");
        let mut batch = stage.build(&[], &mut pt);
        let claim = batch.claims[0].clone();

        let proof = BatchedSumcheckProver::prove(
            &batch.claims,
            &mut batch.witnesses,
            &mut pt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        let mut vt = Blake2bTranscript::new(b"s3_increment");
        let (final_eval, challenges) = BatchedSumcheckVerifier::verify(
            &[claim],
            &proof,
            &mut vt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        )
        .expect("verification should succeed");

        let prover_claims =
            <ClaimReductionStage<Fr> as ProverStage<Fr, Blake2bTranscript>>::extract_claims(
                &mut stage,
                &challenges,
                final_eval,
            );
        assert_eq!(prover_claims.len(), 2);
    }

    #[test]
    fn multi_instance_stage() {
        let num_vars = 3;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        // Instance 1: two polynomials
        let p0: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let p1: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let c0 = Fr::random(&mut rng);
        let c1 = Fr::random(&mut rng);

        // Instance 2: one polynomial
        let p2: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let c2 = Fr::random(&mut rng);

        let mut stage: ClaimReductionStage<Fr> = ClaimReductionStage::new(
            vec![(vec![p0, p1], vec![c0, c1]), (vec![p2], vec![c2])],
            eq_point,
            vec![
                reductions::increment_claim_reduction as fn() -> ClaimDefinition,
                reductions::ram_ra_claim_reduction as fn() -> ClaimDefinition,
            ],
        );

        let mut pt = Blake2bTranscript::new(b"multi_instance");
        let mut batch = stage.build(&[], &mut pt);

        assert_eq!(batch.claims.len(), 2);

        let proof = BatchedSumcheckProver::prove(
            &batch.claims,
            &mut batch.witnesses,
            &mut pt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        let mut vt = Blake2bTranscript::new(b"multi_instance");
        let (final_eval, challenges) = BatchedSumcheckVerifier::verify(
            &batch.claims,
            &proof,
            &mut vt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        )
        .expect("verification should succeed");

        let prover_claims =
            <ClaimReductionStage<Fr> as ProverStage<Fr, Blake2bTranscript>>::extract_claims(
                &mut stage,
                &challenges,
                final_eval,
            );
        // 2 + 1 = 3 opening claims total
        assert_eq!(prover_claims.len(), 3);
    }
}
