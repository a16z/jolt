//! Stage 6: Hamming booleanity check.
//!
//! Proves that the Hamming weight polynomial `h` is Boolean-valued on
//! the Boolean hypercube using a sumcheck over
//! $\widetilde{eq}(r, x) \cdot h(x) \cdot (h(x) - 1) = 0$.

use jolt_field::Field;
use jolt_ir::ClaimDefinition;
use jolt_openings::ProverClaim;
use jolt_poly::{EqPolynomial, Polynomial};
use jolt_sumcheck::claim::SumcheckClaim;
use jolt_transcript::Transcript;

use crate::claims::ram;
use crate::stage::{ProverStage, StageBatch};
use crate::witnesses::hamming::HammingBooleanityCompute;

/// Hamming booleanity prover stage.
///
/// Constructed with the Hamming weight polynomial evaluations (moved from
/// [`WitnessStore`](crate::witness::WitnessStore)) and the random point `r`
/// for the equality polynomial.
pub struct HammingBooleanityStage<F: Field> {
    /// Original evaluation table — preserved for `extract_claims()`.
    h_evals: Option<Vec<F>>,
    /// Random evaluation point for the eq polynomial.
    eq_point: Vec<F>,
    num_vars: usize,
}

impl<F: Field> HammingBooleanityStage<F> {
    /// Creates a new stage from the Hamming weight evaluations and eq point.
    ///
    /// `h_evals` should be moved out of the [`WitnessStore`](crate::witness::WitnessStore)
    /// via [`take()`](crate::witness::WitnessStore::take).
    ///
    /// # Panics
    ///
    /// Panics if `h_evals.len() != 2^eq_point.len()`.
    pub fn new(h_evals: Vec<F>, eq_point: Vec<F>) -> Self {
        let num_vars = eq_point.len();
        assert_eq!(
            h_evals.len(),
            1 << num_vars,
            "h_evals length {} != 2^{num_vars} = {}",
            h_evals.len(),
            1 << num_vars,
        );
        Self {
            h_evals: Some(h_evals),
            eq_point,
            num_vars,
        }
    }
}

impl<F: Field, T: Transcript> ProverStage<F, T> for HammingBooleanityStage<F> {
    fn build(&mut self, _prior_claims: &[ProverClaim<F>], _transcript: &mut T) -> StageBatch<F> {
        let h_evals = self
            .h_evals
            .as_ref()
            .expect("build() called after extract_claims() consumed h_evals");

        let eq_table = EqPolynomial::new(self.eq_point.clone()).evaluations();

        let claim = SumcheckClaim {
            num_vars: self.num_vars,
            degree: 3,
            claimed_sum: F::zero(),
        };

        let witness = HammingBooleanityCompute::new(h_evals.clone(), eq_table, self.num_vars);

        StageBatch {
            claims: vec![claim],
            witnesses: vec![Box::new(witness)],
        }
    }

    fn extract_claims(&mut self, challenges: &[F], _final_eval: F) -> Vec<ProverClaim<F>> {
        let h_evals = self.h_evals.take().expect("extract_claims() called twice");

        let poly = Polynomial::new(h_evals.clone());
        let eval = poly.evaluate(challenges);

        vec![ProverClaim {
            evaluations: h_evals,
            point: challenges.to_vec(),
            eval,
        }]
    }

    fn claim_definitions(&self) -> Vec<ClaimDefinition> {
        vec![ram::hamming_booleanity()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};
    use jolt_sumcheck::{BatchedSumcheckProver, BatchedSumcheckVerifier};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use num_traits::Zero;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn stage_build_produces_valid_sumcheck() {
        let num_vars = 4;
        let mut rng = ChaCha20Rng::seed_from_u64(555);
        let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        // Boolean h
        let h_evals: Vec<Fr> = (0..(1 << num_vars)).map(|i| Fr::from_u64(i % 2)).collect();

        let mut stage = HammingBooleanityStage::new(h_evals, eq_point);

        let mut transcript = Blake2bTranscript::new(b"test_stage");
        let batch = stage.build(&[], &mut transcript);

        assert_eq!(batch.claims.len(), 1);
        assert_eq!(batch.witnesses.len(), 1);
        assert!(batch.claims[0].claimed_sum.is_zero());
        assert_eq!(batch.claims[0].degree, 3);
        assert_eq!(batch.claims[0].num_vars, num_vars);
    }

    #[test]
    fn stage_full_prove_verify_round_trip() {
        let num_vars = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(777);
        let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        let h_evals: Vec<Fr> = (0..(1 << num_vars)).map(|i| Fr::from_u64(i % 2)).collect();

        let mut stage: HammingBooleanityStage<Fr> = HammingBooleanityStage::new(h_evals, eq_point);

        let mut prover_transcript = Blake2bTranscript::new(b"stage_roundtrip");
        let mut batch = stage.build(&[], &mut prover_transcript);

        assert_eq!(batch.claims.len(), 1);
        let claim = batch.claims[0].clone();

        // Prove using batched prover (accepts Box<dyn SumcheckCompute>)
        let proof = BatchedSumcheckProver::prove(
            &batch.claims,
            &mut batch.witnesses,
            &mut prover_transcript,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        // Verify
        let mut verifier_transcript = Blake2bTranscript::new(b"stage_roundtrip");
        let (final_eval, challenges) = BatchedSumcheckVerifier::verify(
            &[claim],
            &proof,
            &mut verifier_transcript,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        )
        .expect("verification should succeed");

        // Extract claims
        let prover_claims =
            <HammingBooleanityStage<Fr> as ProverStage<Fr, Blake2bTranscript>>::extract_claims(
                &mut stage,
                &challenges,
                final_eval,
            );
        assert_eq!(prover_claims.len(), 1);

        // The opening claim's eval should match evaluating h at the challenge point
        let h_poly = Polynomial::new(prover_claims[0].evaluations.clone());
        let expected_eval = h_poly.evaluate(&challenges);
        assert_eq!(prover_claims[0].eval, expected_eval);
        assert_eq!(prover_claims[0].point, challenges);
    }

    #[test]
    fn claim_definitions_returns_hamming_booleanity() {
        let stage = HammingBooleanityStage::<Fr>::new(
            vec![Fr::zero(); 4],
            vec![Fr::from_u64(1), Fr::from_u64(2)],
        );
        let defs =
            <HammingBooleanityStage<Fr> as ProverStage<Fr, Blake2bTranscript>>::claim_definitions(
                &stage,
            );
        assert_eq!(defs.len(), 1);
    }
}
