//! Stage 7: Hamming weight claim reduction.
//!
//! Reduces multiple Hamming weight polynomial opening claims from stage 6
//! into a single opening point via a γ-weighted eq sumcheck.
//!
//! Proves: $\sum_{x \in \{0,1\}^n} \widetilde{eq}(r, x) \cdot
//!   \sum_i c_i \cdot p_i(x) = v$
//!
//! where $c_i = \text{eq\_eval}_i \cdot \gamma^i$ and $p_i$ are the
//! Hamming weight polynomials for each chunk type.

use jolt_field::Field;
use jolt_ir::ClaimDefinition;
use jolt_openings::ProverClaim;
use jolt_poly::{EqPolynomial, Polynomial};
use jolt_sumcheck::claim::SumcheckClaim;
use jolt_transcript::Transcript;

use crate::claims::reductions;
use crate::stage::{ProverStage, StageBatch};
use crate::witnesses::eq_product::EqProductCompute;

/// Hamming weight claim reduction prover stage.
///
/// Receives Hamming weight polynomial evaluation tables for each chunk type
/// and reduces them to a single opening point. The sumcheck formula is
/// `eq · (Σ c_i · poly_i)`, which is degree 2.
///
/// The γ-weighted coefficients c_i combine the eq evaluation at the prior
/// sumcheck challenge point with γ-powers from the batching challenge.
pub struct HammingReductionStage<F: Field> {
    /// Evaluation tables for each Hamming weight polynomial type.
    poly_tables: Option<Vec<Vec<F>>>,
    /// Eq evaluation point (from the prior sumcheck challenge vector).
    eq_point: Vec<F>,
    /// Pre-computed coefficients combining eq_eval and γ-powers.
    /// Populated during `build()`.
    coefficients: Vec<F>,
    num_vars: usize,
}

impl<F: Field> HammingReductionStage<F> {
    /// Creates a new stage from Hamming weight polynomial tables.
    ///
    /// `poly_tables[i]` contains the evaluation table for the i-th
    /// Hamming weight polynomial type. `eq_point` is the challenge
    /// vector from the prior sumcheck.
    ///
    /// # Panics
    ///
    /// Panics if any table has length != 2^eq_point.len().
    pub fn new(poly_tables: Vec<Vec<F>>, eq_point: Vec<F>) -> Self {
        let num_vars = eq_point.len();
        let expected = 1usize << num_vars;
        for (i, table) in poly_tables.iter().enumerate() {
            assert_eq!(
                table.len(),
                expected,
                "poly_tables[{i}] length {} != {expected}",
                table.len(),
            );
        }
        Self {
            poly_tables: Some(poly_tables),
            eq_point,
            coefficients: Vec::new(),
            num_vars,
        }
    }
}

impl<F: Field, T: Transcript> ProverStage<F, T> for HammingReductionStage<F> {
    fn build(
        &mut self,
        prior_claims: &[ProverClaim<F>],
        transcript: &mut T,
    ) -> StageBatch<F> {
        let poly_tables = self
            .poly_tables
            .as_ref()
            .expect("build() called after extract_claims()");
        let n_polys = poly_tables.len();
        let n = 1usize << self.num_vars;

        // Derive γ from transcript
        let gamma: F = transcript.challenge().into();

        // Compute coefficients: c_i = eq_eval_i · γ^i
        // where eq_eval_i comes from the prior claim's eval at the
        // reduction point. For standalone use, if no prior claims,
        // coefficients are just γ-powers.
        let mut gamma_power = F::one();
        self.coefficients = Vec::with_capacity(n_polys);
        for i in 0..n_polys {
            let eq_weight = if i < prior_claims.len() {
                prior_claims[i].eval
            } else {
                F::one()
            };
            self.coefficients.push(eq_weight * gamma_power);
            gamma_power *= gamma;
        }

        // Pre-compute g(x) = Σ c_i · p_i(x)
        let mut g_table = vec![F::zero(); n];
        for (i, table) in poly_tables.iter().enumerate() {
            let c = self.coefficients[i];
            for (j, g) in g_table.iter_mut().enumerate() {
                *g += c * table[j];
            }
        }

        let eq_table = EqPolynomial::new(self.eq_point.clone()).evaluations();

        // Claimed sum = Σ_x eq(r, x) · g(x)
        let claimed_sum: F = eq_table
            .iter()
            .zip(g_table.iter())
            .map(|(&e, &g)| e * g)
            .sum();

        let claim = SumcheckClaim {
            num_vars: self.num_vars,
            degree: 2,
            claimed_sum,
        };

        let witness = EqProductCompute::new(g_table, eq_table, self.num_vars);

        StageBatch {
            claims: vec![claim],
            witnesses: vec![Box::new(witness)],
        }
    }

    fn extract_claims(
        &mut self,
        challenges: &[F],
        _final_eval: F,
    ) -> Vec<ProverClaim<F>> {
        let poly_tables = self
            .poly_tables
            .take()
            .expect("extract_claims() called twice");

        poly_tables
            .into_iter()
            .map(|evals| {
                let poly = Polynomial::new(evals.clone());
                let eval = poly.evaluate(challenges);
                ProverClaim {
                    evaluations: evals,
                    point: challenges.to_vec(),
                    eval,
                }
            })
            .collect()
    }

    fn claim_definitions(&self) -> Vec<ClaimDefinition> {
        let n_polys = self
            .poly_tables
            .as_ref()
            .map_or(0, |t| t.len());
        vec![reductions::hamming_weight_claim_reduction(n_polys)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};
    use jolt_sumcheck::{BatchedSumcheckProver, BatchedSumcheckVerifier};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn stage_build_produces_degree_2_claim() {
        let num_vars = 3;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        let polys: Vec<Vec<Fr>> = (0..3)
            .map(|_| (0..n).map(|_| Fr::random(&mut rng)).collect())
            .collect();

        let mut stage = HammingReductionStage::new(polys, eq_point);
        let mut transcript = Blake2bTranscript::new(b"test_s7");
        let batch = stage.build(&[], &mut transcript);

        assert_eq!(batch.claims.len(), 1);
        assert_eq!(batch.claims[0].degree, 2);
        assert_eq!(batch.claims[0].num_vars, num_vars);
    }

    #[test]
    fn stage_full_prove_verify_round_trip() {
        let num_vars = 4;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(123);
        let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        let polys: Vec<Vec<Fr>> = (0..3)
            .map(|_| (0..n).map(|_| Fr::random(&mut rng)).collect())
            .collect();

        let mut stage: HammingReductionStage<Fr> =
            HammingReductionStage::new(polys, eq_point);

        let mut prover_transcript = Blake2bTranscript::new(b"s7_roundtrip");
        let mut batch = stage.build(&[], &mut prover_transcript);

        let claim = batch.claims[0].clone();

        let proof = BatchedSumcheckProver::prove(
            &batch.claims,
            &mut batch.witnesses,
            &mut prover_transcript,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        let mut verifier_transcript = Blake2bTranscript::new(b"s7_roundtrip");
        // Derive the same γ challenge the prover consumed
        let _gamma: Fr = verifier_transcript.challenge().into();

        let (final_eval, challenges) = BatchedSumcheckVerifier::verify(
            &[claim],
            &proof,
            &mut verifier_transcript,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        )
        .expect("verification should succeed");

        let prover_claims =
            <HammingReductionStage<Fr> as ProverStage<Fr, Blake2bTranscript>>::extract_claims(
                &mut stage,
                &challenges,
                final_eval,
            );
        assert_eq!(prover_claims.len(), 3);

        // Each claim's eval matches polynomial evaluation at challenge point
        for pc in &prover_claims {
            let poly = Polynomial::new(pc.evaluations.clone());
            assert_eq!(poly.evaluate(&challenges), pc.eval);
            assert_eq!(pc.point, challenges);
        }
    }

    #[test]
    fn extract_claims_consumes_tables() {
        let num_vars = 2;
        let polys = vec![vec![Fr::from_u64(1); 4], vec![Fr::from_u64(2); 4]];
        let eq_point = vec![Fr::from_u64(3), Fr::from_u64(5)];

        let mut stage = HammingReductionStage::new(polys, eq_point);
        let mut t = Blake2bTranscript::new(b"test");
        let _ = stage.build(&[], &mut t);

        let challenges = vec![Fr::from_u64(7), Fr::from_u64(11)];
        let claims = <HammingReductionStage<Fr> as ProverStage<Fr, Blake2bTranscript>>::extract_claims(
            &mut stage,
            &challenges,
            Fr::zero(),
        );
        assert_eq!(claims.len(), 2);

        // Second call should panic
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            <HammingReductionStage<Fr> as ProverStage<Fr, Blake2bTranscript>>::extract_claims(
                &mut stage,
                &challenges,
                Fr::zero(),
            );
        }));
        assert!(result.is_err());
    }

    #[test]
    fn claim_definitions_returns_correct_count() {
        let polys = vec![
            vec![Fr::from_u64(1); 8],
            vec![Fr::from_u64(2); 8],
            vec![Fr::from_u64(3); 8],
        ];
        let eq_point: Vec<Fr> = (0..3).map(|i| Fr::from_u64(i + 1)).collect();
        let stage = HammingReductionStage::new(polys, eq_point);

        let defs = <HammingReductionStage<Fr> as ProverStage<Fr, Blake2bTranscript>>::claim_definitions(&stage);
        assert_eq!(defs.len(), 1);
    }
}
