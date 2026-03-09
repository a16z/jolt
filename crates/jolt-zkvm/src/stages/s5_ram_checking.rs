//! Stage 5: RAM output check and RAF evaluation sumchecks.
//!
//! Contains two sumcheck instances:
//!
//! 1. **RAM output check** (degree 2):
//!    `eq · (c0·val_final + c1)` — verifies final RAM values match
//!    expected I/O values.
//!
//! 2. **RAM RAF evaluation** (degree 2):
//!    `eq · (c0·ra)` — relates the read-address polynomial to the
//!    address unmapping polynomial.

use jolt_field::Field;
use jolt_ir::ClaimDefinition;
use jolt_openings::ProverClaim;
use jolt_poly::{EqPolynomial, Polynomial};
use jolt_sumcheck::claim::SumcheckClaim;
use jolt_transcript::Transcript;

use crate::claims::ram;
use crate::stage::{ProverStage, StageBatch};
use crate::witnesses::eq_product::EqProductCompute;

/// RAM output check + RAF evaluation prover stage.
///
/// Both instances are degree 2 and use the pre-compute-g + EqProductCompute
/// pattern.
pub struct RamCheckingStage<F: Field> {
    /// RAM val_final polynomial evaluations.
    val_final: Option<Vec<F>>,
    /// RAM read-address polynomial evaluations.
    ram_ra: Option<Vec<F>>,
    /// Eq point for RAM output check.
    output_eq_point: Vec<F>,
    /// Challenges for output check: [c0, c1] where c0=eq·io_mask, c1=-eq·io_mask·val_io.
    output_challenges: [F; 2],
    /// Eq point for RAF evaluation.
    raf_eq_point: Vec<F>,
    /// Challenge for RAF evaluation: c0 = unmap_eval.
    raf_c0: F,
    num_vars: usize,
}

impl<F: Field> RamCheckingStage<F> {
    /// Creates a new stage.
    ///
    /// # Arguments
    ///
    /// * `val_final` — RAM val_final evaluation table
    /// * `output_eq_point` — Eq point for output check
    /// * `output_challenges` — `[c0, c1]` for output check formula
    /// * `ram_ra` — RAM read-address evaluation table
    /// * `raf_eq_point` — Eq point for RAF evaluation
    /// * `raf_c0` — `unmap_eval` challenge for RAF formula
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        val_final: Vec<F>,
        output_eq_point: Vec<F>,
        output_challenges: [F; 2],
        ram_ra: Vec<F>,
        raf_eq_point: Vec<F>,
        raf_c0: F,
    ) -> Self {
        let num_vars = output_eq_point.len();
        let expected = 1usize << num_vars;
        assert_eq!(val_final.len(), expected);
        assert_eq!(ram_ra.len(), expected);
        assert_eq!(raf_eq_point.len(), num_vars);
        Self {
            val_final: Some(val_final),
            ram_ra: Some(ram_ra),
            output_eq_point,
            output_challenges,
            raf_eq_point,
            raf_c0,
            num_vars,
        }
    }
}

impl<F: Field, T: Transcript> ProverStage<F, T> for RamCheckingStage<F> {
    fn build(&mut self, _prior_claims: &[ProverClaim<F>], _transcript: &mut T) -> StageBatch<F> {
        let val_final = self.val_final.as_ref().unwrap();
        let ram_ra = self.ram_ra.as_ref().unwrap();
        let n = 1usize << self.num_vars;

        let [c0, c1] = self.output_challenges;
        let mut output_g = vec![F::zero(); n];
        for (j, g) in output_g.iter_mut().enumerate() {
            *g = c0 * val_final[j] + c1;
        }
        let output_eq = EqPolynomial::new(self.output_eq_point.clone()).evaluations();
        let output_sum: F = output_eq
            .iter()
            .zip(output_g.iter())
            .map(|(&e, &g)| e * g)
            .sum();

        let mut raf_g = vec![F::zero(); n];
        for (j, g) in raf_g.iter_mut().enumerate() {
            *g = self.raf_c0 * ram_ra[j];
        }
        let raf_eq = EqPolynomial::new(self.raf_eq_point.clone()).evaluations();
        let raf_sum: F = raf_eq.iter().zip(raf_g.iter()).map(|(&e, &g)| e * g).sum();

        StageBatch {
            claims: vec![
                SumcheckClaim {
                    num_vars: self.num_vars,
                    degree: 2,
                    claimed_sum: output_sum,
                },
                SumcheckClaim {
                    num_vars: self.num_vars,
                    degree: 2,
                    claimed_sum: raf_sum,
                },
            ],
            witnesses: vec![
                Box::new(EqProductCompute::new(output_g, output_eq, self.num_vars)),
                Box::new(EqProductCompute::new(raf_g, raf_eq, self.num_vars)),
            ],
        }
    }

    fn extract_claims(&mut self, challenges: &[F], _final_eval: F) -> Vec<ProverClaim<F>> {
        let val_final = self.val_final.take().unwrap();
        let ram_ra = self.ram_ra.take().unwrap();

        vec![val_final, ram_ra]
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
        vec![ram::ram_output_check(), ram::ram_raf_evaluation()]
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
    fn stage_produces_two_degree_2_claims() {
        let num_vars = 3;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let output_eq: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let raf_eq: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let val_final: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let ram_ra: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let c0 = Fr::random(&mut rng);
        let c1 = Fr::random(&mut rng);
        let raf_c0 = Fr::random(&mut rng);

        let mut stage =
            RamCheckingStage::new(val_final, output_eq, [c0, c1], ram_ra, raf_eq, raf_c0);

        let mut t = Blake2bTranscript::new(b"test_s5");
        let batch = stage.build(&[], &mut t);

        assert_eq!(batch.claims.len(), 2);
        assert_eq!(batch.claims[0].degree, 2);
        assert_eq!(batch.claims[1].degree, 2);
    }

    #[test]
    fn full_prove_verify_round_trip() {
        let num_vars = 4;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(99);

        let output_eq: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let raf_eq: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let val_final: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let ram_ra: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let c0 = Fr::random(&mut rng);
        let c1 = Fr::random(&mut rng);
        let raf_c0 = Fr::random(&mut rng);

        let mut stage: RamCheckingStage<Fr> =
            RamCheckingStage::new(val_final, output_eq, [c0, c1], ram_ra, raf_eq, raf_c0);

        let mut pt = Blake2bTranscript::new(b"s5_roundtrip");
        let mut batch = stage.build(&[], &mut pt);

        let claims_snapshot: Vec<_> = batch.claims.clone();
        let proof = BatchedSumcheckProver::prove(
            &batch.claims,
            &mut batch.witnesses,
            &mut pt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        let mut vt = Blake2bTranscript::new(b"s5_roundtrip");
        let (final_eval, challenges) = BatchedSumcheckVerifier::verify(
            &claims_snapshot,
            &proof,
            &mut vt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        )
        .expect("verification should succeed");

        let prover_claims =
            <RamCheckingStage<Fr> as ProverStage<Fr, Blake2bTranscript>>::extract_claims(
                &mut stage,
                &challenges,
                final_eval,
            );
        assert_eq!(prover_claims.len(), 2);

        for pc in &prover_claims {
            let poly = Polynomial::new(pc.evaluations.clone());
            assert_eq!(poly.evaluate(&challenges), pc.eval);
        }
    }
}
