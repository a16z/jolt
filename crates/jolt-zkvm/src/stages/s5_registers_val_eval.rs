//! Stage 5c: Registers value evaluation sumcheck.
//!
//! Proves that the register value at `(r_address, r_cycle)` equals the sum of
//! all increments weighted by write-address matches occurring before `r_cycle`:
//!
//! $$\sum_j \text{inc}(j) \cdot \text{wa}(r_{\text{addr}}, j) \cdot \text{LT}(r_{\text{cycle}}, j) = \text{Val}(r_{\text{addr}}, r_{\text{cycle}})$$
//!
//! Degree 3 (product of three multilinear polynomials), `log_T` rounds,
//! HighToLow binding.

use jolt_field::{Field, WithChallenge};
use jolt_ir::ClaimDefinition;
use jolt_openings::ProverClaim;
use jolt_poly::{LtPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::claim::SumcheckClaim;
use jolt_sumcheck::prover::SumcheckCompute;
use jolt_transcript::Transcript;

use crate::stage::{ProverStage, StageBatch};

/// Number of witness polynomials (inc, wa).
pub const NUM_VAL_EVAL_POLYS: usize = 2;

const INC: usize = 0;
const WA: usize = 1;

/// Registers value evaluation prover stage.
///
/// Proves the accumulation identity: the register value at a point equals
/// the sum of all prior writes, using the less-than polynomial to select
/// cycles before the evaluation point.
pub struct RegistersValEvalStage<F: Field> {
    /// Evaluation tables `[inc, wa]`, preserved for [`extract_claims`].
    witness_polys: Option<[Vec<F>; NUM_VAL_EVAL_POLYS]>,
    /// Challenge point for the LT polynomial (big-endian, from prior stage).
    r_cycle: Vec<F>,
    /// Expected sum.
    claimed_sum: F,
    num_vars: usize,
}

impl<F: Field> RegistersValEvalStage<F> {
    /// Creates a new registers value evaluation stage.
    ///
    /// # Arguments
    ///
    /// * `inc` — Increment polynomial evaluations, length `2^num_vars`.
    /// * `wa` — Write-address polynomial evaluations (pre-evaluated at `r_address`),
    ///   length `2^num_vars`.
    /// * `r_cycle` — LT challenge point (big-endian).
    /// * `claimed_sum` — Expected sum from prior claims (`Val(r_addr, r_cycle)`).
    pub fn new(inc: Vec<F>, wa: Vec<F>, r_cycle: Vec<F>, claimed_sum: F) -> Self {
        let num_vars = r_cycle.len();
        let expected_len = 1usize << num_vars;
        assert_eq!(inc.len(), expected_len, "inc length mismatch");
        assert_eq!(wa.len(), expected_len, "wa length mismatch");

        Self {
            witness_polys: Some([inc, wa]),
            r_cycle,
            claimed_sum,
            num_vars,
        }
    }
}

impl<F: WithChallenge, T: Transcript> ProverStage<F, T> for RegistersValEvalStage<F> {
    fn name(&self) -> &'static str {
        "S5_registers_val_eval"
    }

    fn build(&mut self, _prior_claims: &[ProverClaim<F>], _transcript: &mut T) -> StageBatch<F> {
        let polys = self
            .witness_polys
            .as_ref()
            .expect("build() called after extract_claims()");

        let inc = Polynomial::new(polys[INC].clone());
        let wa = Polynomial::new(polys[WA].clone());
        let lt = LtPolynomial::new(&self.r_cycle);

        let evaluator = RegistersValEvalWitness {
            inc,
            wa,
            lt,
            claim: F::zero(),
        };

        StageBatch {
            claims: vec![SumcheckClaim {
                num_vars: self.num_vars,
                degree: 3,
                claimed_sum: self.claimed_sum,
            }],
            witnesses: vec![Box::new(evaluator)],
        }
    }

    fn extract_claims(&mut self, challenges: &[F], _final_eval: F) -> Vec<ProverClaim<F>> {
        let polys = self
            .witness_polys
            .take()
            .expect("extract_claims() called twice");

        // HighToLow binding: challenges are MSB-first, no reversal needed.
        let eval_point = challenges.to_vec();

        polys
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
        vec![]
    }
}

/// Sumcheck witness for the registers value evaluation identity (degree 3).
///
/// Formula: `inc(j) · wa(j) · LT(r_cycle, j)`.
///
/// Uses HighToLow binding. The LT polynomial uses its split representation
/// for √N memory efficiency.
struct RegistersValEvalWitness<F: WithChallenge> {
    inc: Polynomial<F>,
    wa: Polynomial<F>,
    lt: LtPolynomial<F>,
    claim: F,
}

impl<F: WithChallenge> SumcheckCompute<F> for RegistersValEvalWitness<F> {
    fn set_claim(&mut self, claim: F) {
        self.claim = claim;
    }

    fn round_polynomial(&self) -> UnivariatePoly<F> {
        let half = self.inc.len() / 2;

        let mut eval_0 = F::zero();
        let mut eval_2 = F::zero();
        let mut eval_3 = F::zero();

        // HighToLow: pair [j] (MSB=0) with [j + half] (MSB=1).
        for j in 0..half {
            let (inc_lo, inc_hi) = self.inc.sumcheck_eval_pair(j, jolt_poly::BindingOrder::HighToLow);
            let inc_d = inc_hi - inc_lo;

            let (wa_lo, wa_hi) = self.wa.sumcheck_eval_pair(j, jolt_poly::BindingOrder::HighToLow);
            let wa_d = wa_hi - wa_lo;

            let (lt_lo, lt_hi) = self.lt.sumcheck_eval_pair(j);
            let lt_d = lt_hi - lt_lo;

            // t=0: all lo values.
            eval_0 += inc_lo * wa_lo * lt_lo;

            // t=2: p(2) = lo + 2·delta.
            let inc_2 = inc_lo + inc_d + inc_d;
            let wa_2 = wa_lo + wa_d + wa_d;
            let lt_2 = lt_lo + lt_d + lt_d;
            eval_2 += inc_2 * wa_2 * lt_2;

            // t=3: p(3) = p(2) + delta.
            let inc_3 = inc_2 + inc_d;
            let wa_3 = wa_2 + wa_d;
            let lt_3 = lt_2 + lt_d;
            eval_3 += inc_3 * wa_3 * lt_3;
        }

        // Degree 3: hint = s(0)+s(1), evals at {0, 2, 3}.
        UnivariatePoly::from_evals_and_hint(self.claim, &[eval_0, eval_2, eval_3])
    }

    fn bind(&mut self, challenge: F::Challenge) {
        let c: F = challenge.into();
        self.inc.bind(c);
        self.wa.bind(c);
        self.lt.bind(c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};
    use jolt_sumcheck::{BatchedSumcheckProver, BatchedSumcheckVerifier};
    use jolt_transcript::Blake2bTranscript;
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    /// Brute-force computation of the value evaluation sum.
    fn brute_force_val_eval_sum(inc: &[Fr], wa: &[Fr], r_cycle: &[Fr]) -> Fr {
        let lt_table = LtPolynomial::evaluations(r_cycle);
        inc.iter()
            .zip(wa.iter())
            .zip(lt_table.iter())
            .map(|((&i, &w), &lt)| i * w * lt)
            .sum()
    }

    fn run_val_eval_test(num_vars: usize, seed: u64) {
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        let r_cycle: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        let inc: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let wa: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let claimed_sum = brute_force_val_eval_sum(&inc, &wa, &r_cycle);

        let inc_copy = inc.clone();
        let wa_copy = wa.clone();
        let mut stage = RegistersValEvalStage::new(inc, wa, r_cycle.clone(), claimed_sum);

        let mut pt = Blake2bTranscript::new(b"val_eval");
        let mut batch = stage.build(&[], &mut pt);

        assert_eq!(batch.claims.len(), 1);
        assert_eq!(batch.claims[0].degree, 3);
        assert_eq!(batch.claims[0].num_vars, num_vars);

        let claims_snapshot = batch.claims.clone();
        let proof = BatchedSumcheckProver::prove(
            &batch.claims,
            &mut batch.witnesses,
            &mut pt,
        );

        let mut vt = Blake2bTranscript::new(b"val_eval");
        let (final_eval, challenges) = BatchedSumcheckVerifier::verify(
            &claims_snapshot,
            &proof,
            &mut vt,
        )
        .expect("verification should succeed");

        // Oracle check: final_eval = inc(r) * wa(r) * LT(r, r_cycle).
        let inc_at_r = Polynomial::new(inc_copy.clone()).evaluate(&challenges);
        let wa_at_r = Polynomial::new(wa_copy.clone()).evaluate(&challenges);
        let lt_at_r = LtPolynomial::evaluate(&challenges, &r_cycle);
        let expected = inc_at_r * wa_at_r * lt_at_r;
        assert_eq!(final_eval, expected, "oracle check failed");

        // Extract claims and verify evaluations.
        let claims =
            <RegistersValEvalStage<Fr> as ProverStage<Fr, Blake2bTranscript>>::extract_claims(
                &mut stage,
                &challenges,
                final_eval,
            );
        assert_eq!(claims.len(), NUM_VAL_EVAL_POLYS);

        let expected_inc = Polynomial::new(inc_copy).evaluate(&challenges);
        let expected_wa = Polynomial::new(wa_copy).evaluate(&challenges);
        assert_eq!(claims[INC].eval, expected_inc, "inc eval mismatch");
        assert_eq!(claims[WA].eval, expected_wa, "wa eval mismatch");
        assert_eq!(claims[INC].point, challenges);
        assert_eq!(claims[WA].point, challenges);
    }

    #[test]
    fn val_eval_even_vars() {
        run_val_eval_test(6, 42);
    }

    #[test]
    fn val_eval_odd_vars() {
        run_val_eval_test(7, 123);
    }

    #[test]
    fn val_eval_small() {
        run_val_eval_test(4, 777);
    }

    #[test]
    fn val_eval_minimum() {
        run_val_eval_test(2, 999);
    }

    #[test]
    fn val_eval_boolean_witness() {
        // inc and wa are boolean (0/1) as in real traces.
        let num_vars = 5;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(555);

        let r_cycle: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        use rand_core::RngCore;
        let inc: Vec<Fr> = (0..n)
            .map(|_| {
                if rng.next_u32() & 1 == 1 {
                    Fr::one()
                } else {
                    Fr::zero()
                }
            })
            .collect();
        let wa: Vec<Fr> = (0..n)
            .map(|_| {
                if rng.next_u32() & 1 == 1 {
                    Fr::one()
                } else {
                    Fr::zero()
                }
            })
            .collect();

        let claimed_sum = brute_force_val_eval_sum(&inc, &wa, &r_cycle);

        let mut stage =
            RegistersValEvalStage::new(inc, wa, r_cycle.clone(), claimed_sum);

        let mut pt = Blake2bTranscript::new(b"bool_val");
        let mut batch = stage.build(&[], &mut pt);

        let claims_snapshot = batch.claims.clone();
        let proof = BatchedSumcheckProver::prove(
            &batch.claims,
            &mut batch.witnesses,
            &mut pt,
        );

        let mut vt = Blake2bTranscript::new(b"bool_val");
        let result = BatchedSumcheckVerifier::verify(
            &claims_snapshot,
            &proof,
            &mut vt,
        );
        assert!(result.is_ok(), "boolean-witness verification failed");
    }

    #[test]
    fn val_eval_zero_sum() {
        // When inc is all zeros, the sum should be zero.
        let num_vars = 4;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(888);

        let r_cycle: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let inc = vec![Fr::zero(); n];
        let wa: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let claimed_sum = Fr::zero();

        let mut stage =
            RegistersValEvalStage::new(inc, wa, r_cycle, claimed_sum);

        let mut pt = Blake2bTranscript::new(b"zero_val");
        let mut batch = stage.build(&[], &mut pt);

        let claims_snapshot = batch.claims.clone();
        let proof = BatchedSumcheckProver::prove(
            &batch.claims,
            &mut batch.witnesses,
            &mut pt,
        );

        let mut vt = Blake2bTranscript::new(b"zero_val");
        let result = BatchedSumcheckVerifier::verify(
            &claims_snapshot,
            &proof,
            &mut vt,
        );
        assert!(result.is_ok(), "zero-sum verification failed");
    }
}
