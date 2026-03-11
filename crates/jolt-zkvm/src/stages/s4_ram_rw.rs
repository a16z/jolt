//! Stage 4b: RAM read-write checking sumcheck.
//!
//! Proves: $\sum_{x} \widetilde{eq}(r, x) \cdot (c_0 \cdot ra(x) \cdot val(x) + c_1 \cdot ra(x) \cdot inc(x)) = v$
//!
//! This is degree 3 (1 from eq + 2 from ra·val or ra·inc).

use std::sync::Arc;

use jolt_compute::ComputeBackend;
use jolt_field::Field;
use jolt_ir::ClaimDefinition;
use jolt_openings::ProverClaim;
use jolt_poly::{EqPolynomial, Polynomial};
use jolt_sumcheck::claim::SumcheckClaim;
use jolt_transcript::Transcript;

use crate::evaluators::catalog::{formula_descriptor, Term};
use crate::evaluators::kernel::KernelEvaluator;
use crate::stage::{ProverStage, StageBatch};
use jolt_ir::zkvm::claims::ram;

/// RAM read-write checking prover stage.
///
/// Formula: `eq · (c0·ra·val + c1·ra·inc)` where:
/// - `c0 = eq_eval · (1+γ)`
/// - `c1 = eq_eval · γ`
pub struct RamRwCheckingStage<F: Field, B: ComputeBackend> {
    ra: Option<Vec<F>>,
    val: Option<Vec<F>>,
    inc: Option<Vec<F>>,
    eq_point: Vec<F>,
    /// Challenge coefficients: [c0, c1].
    challenges: [F; 2],
    num_vars: usize,
    backend: Arc<B>,
}

impl<F: Field, B: ComputeBackend> RamRwCheckingStage<F, B> {
    pub fn new(
        ra: Vec<F>,
        val: Vec<F>,
        inc: Vec<F>,
        eq_point: Vec<F>,
        challenges: [F; 2],
        backend: Arc<B>,
    ) -> Self {
        let num_vars = eq_point.len();
        let expected = 1usize << num_vars;
        assert_eq!(ra.len(), expected);
        assert_eq!(val.len(), expected);
        assert_eq!(inc.len(), expected);
        Self {
            ra: Some(ra),
            val: Some(val),
            inc: Some(inc),
            eq_point,
            challenges,
            num_vars,
            backend,
        }
    }
}

impl<F: Field, B: ComputeBackend, T: Transcript> ProverStage<F, T> for RamRwCheckingStage<F, B> {
    fn name(&self) -> &'static str {
        "S4_ram_rw_checking"
    }

    fn build(&mut self, _prior_claims: &[ProverClaim<F>], _transcript: &mut T) -> StageBatch<F> {
        let ra = self.ra.as_ref().unwrap();
        let val = self.val.as_ref().unwrap();
        let inc = self.inc.as_ref().unwrap();
        let n = 1usize << self.num_vars;

        let [c0, c1] = self.challenges;

        let poly_tables = [ra.clone(), val.clone(), inc.clone()];

        let terms = vec![
            Term {
                coeff: c0,
                factors: vec![0, 1], // ra · val
            },
            Term {
                coeff: c1,
                factors: vec![0, 2], // ra · inc
            },
        ];

        let eq_table = EqPolynomial::new(self.eq_point.clone()).evaluations();

        let claimed_sum: F = (0..n)
            .map(|x| {
                eq_table[x]
                    * (c0 * poly_tables[0][x] * poly_tables[1][x]
                        + c1 * poly_tables[0][x] * poly_tables[2][x])
            })
            .sum();

        let degree = 3;

        let (desc, challenges) = formula_descriptor(&terms, poly_tables.len(), degree);
        let kernel = self
            .backend
            .compile_kernel_with_challenges::<F>(&desc, &challenges);
        let backend = Arc::clone(&self.backend);
        let mut inputs = Vec::with_capacity(1 + poly_tables.len());
        inputs.push(backend.upload(&eq_table));
        inputs.extend(poly_tables.iter().map(|t| backend.upload(t)));
        let witness = KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), backend);

        StageBatch {
            claims: vec![SumcheckClaim {
                num_vars: self.num_vars,
                degree,
                claimed_sum,
            }],
            witnesses: vec![Box::new(witness)],
        }
    }

    fn extract_claims(&mut self, challenges: &[F], _final_eval: F) -> Vec<ProverClaim<F>> {
        let tables = vec![
            self.ra.take().unwrap(),
            self.val.take().unwrap(),
            self.inc.take().unwrap(),
        ];

        // LowToHigh binding → reverse for MSB-first evaluation.
        let eval_point: Vec<F> = challenges.iter().rev().copied().collect();

        tables
            .into_iter()
            .map(|evals| {
                let poly = Polynomial::new(evals.clone());
                let eval = poly.evaluate(&eval_point);
                ProverClaim {
                    evaluations: evals,
                    point: eval_point.clone(),
                    eval,
                }
            })
            .collect()
    }

    fn claim_definitions(&self) -> Vec<ClaimDefinition> {
        vec![ram::ram_read_write_checking()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_cpu::CpuBackend;
    use jolt_field::{Field, Fr};
    use jolt_sumcheck::{BatchedSumcheckProver, BatchedSumcheckVerifier};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use num_traits::One;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn cpu() -> Arc<CpuBackend> {
        Arc::new(CpuBackend)
    }

    #[test]
    fn prove_verify_round_trip() {
        let num_vars = 4;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let ra: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let val: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let inc: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let gamma = Fr::random(&mut rng);
        let eq_eval = Fr::random(&mut rng);
        let c0 = eq_eval * (Fr::one() + gamma);
        let c1 = eq_eval * gamma;

        let mut stage: RamRwCheckingStage<Fr, CpuBackend> =
            RamRwCheckingStage::new(ra, val, inc, eq_point, [c0, c1], cpu());

        let mut pt = Blake2bTranscript::new(b"ram_rw");
        let mut batch = stage.build(&[], &mut pt);

        assert_eq!(batch.claims.len(), 1);
        assert_eq!(batch.claims[0].degree, 3);

        let claim = batch.claims[0].clone();
        let proof = BatchedSumcheckProver::prove(
            &batch.claims,
            &mut batch.witnesses,
            &mut pt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        let mut vt = Blake2bTranscript::new(b"ram_rw");
        let (final_eval, challenges) = BatchedSumcheckVerifier::verify(
            &[claim],
            &proof,
            &mut vt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        )
        .expect("verification should succeed");

        let prover_claims = <RamRwCheckingStage<Fr, CpuBackend> as ProverStage<
            Fr,
            Blake2bTranscript,
        >>::extract_claims(&mut stage, &challenges, final_eval);
        assert_eq!(prover_claims.len(), 3); // ra, val, inc

        let eval_point: Vec<Fr> = challenges.iter().rev().copied().collect();
        for pc in &prover_claims {
            let poly = Polynomial::new(pc.evaluations.clone());
            assert_eq!(poly.evaluate(&eval_point), pc.eval);
            assert_eq!(pc.point, eval_point);
        }
    }
}
