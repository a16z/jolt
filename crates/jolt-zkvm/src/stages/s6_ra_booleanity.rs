//! Stage 6b: RA polynomial booleanity check.
//!
//! Proves that ALL RA polynomials (instruction, bytecode, RAM) are
//! Boolean-valued on the Boolean hypercube using a γ-batched sumcheck:
//!
//! $$0 = \sum_x \widetilde{eq}(r, x) \cdot \sum_i \gamma^i \cdot (ra_i(x)^2 - ra_i(x))$$
//!
//! where $r = (r_{\text{address}}, r_{\text{cycle}})$ and $i$ ranges over all
//! RA polynomials from the instruction, bytecode, and RAM families.
//!
//! Degree 3 (`eq · ra · (ra - 1)`), `log_k_chunk + log_t` rounds,
//! HighToLow binding. All RA polynomials share the same evaluation point,
//! which is required by the HammingWeightClaimReduction in Stage 7.

use jolt_field::Field;
use jolt_ir::ClaimDefinition;
use jolt_openings::ProverClaim;
use jolt_poly::{EqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::claim::SumcheckClaim;
use jolt_sumcheck::prover::SumcheckCompute;
use jolt_transcript::Transcript;

use crate::stage::{ProverStage, StageBatch};

/// RA booleanity prover stage.
///
/// Batches all RA polynomials (instruction + bytecode + RAM) into a single
/// zero-check sumcheck using γ-powers as batching coefficients.
pub struct RaBooleanityStage<F: Field> {
    /// Evaluation tables for all RA polynomials, preserved for [`extract_claims`].
    /// Each table has length `2^num_vars`.
    ra_polys: Option<Vec<Vec<F>>>,
    /// Combined evaluation point `(r_address || r_cycle)` (big-endian).
    eq_point: Vec<F>,
    /// Per-polynomial batching coefficients `[γ^0, γ^1, ..., γ^(total_d-1)]`.
    gamma_powers: Vec<F>,
    num_vars: usize,
}

impl<F: Field> RaBooleanityStage<F> {
    /// Creates a new RA booleanity stage.
    ///
    /// # Arguments
    ///
    /// * `ra_polys` — Evaluation tables for all RA polynomials (instruction, bytecode,
    ///   RAM), each of length `2^num_vars`.
    /// * `eq_point` — Combined `(r_address || r_cycle)` point (big-endian).
    /// * `gamma_powers` — `[γ^0, γ^1, ..., γ^(total_d-1)]` batching coefficients.
    pub fn new(ra_polys: Vec<Vec<F>>, eq_point: Vec<F>, gamma_powers: Vec<F>) -> Self {
        let num_vars = eq_point.len();
        let expected_len = 1usize << num_vars;
        assert!(
            !ra_polys.is_empty(),
            "must have at least one RA polynomial"
        );
        assert_eq!(
            ra_polys.len(),
            gamma_powers.len(),
            "ra_polys.len() = {} != gamma_powers.len() = {}",
            ra_polys.len(),
            gamma_powers.len(),
        );
        for (i, poly) in ra_polys.iter().enumerate() {
            assert_eq!(
                poly.len(),
                expected_len,
                "ra_polys[{i}].len() = {} != {expected_len}",
                poly.len(),
            );
        }

        Self {
            ra_polys: Some(ra_polys),
            eq_point,
            gamma_powers,
            num_vars,
        }
    }
}

impl<F: Field, T: Transcript> ProverStage<F, T> for RaBooleanityStage<F> {
    fn name(&self) -> &'static str {
        "S6_ra_booleanity"
    }

    fn build(&mut self, _prior_claims: &[ProverClaim<F>], _transcript: &mut T) -> StageBatch<F> {
        let ra_evals = self
            .ra_polys
            .as_ref()
            .expect("build() called after extract_claims()");

        let eq_table = EqPolynomial::new(self.eq_point.clone()).evaluations();
        let eq = Polynomial::new(eq_table);

        let ra_witness: Vec<Polynomial<F>> = ra_evals
            .iter()
            .map(|evals| Polynomial::new(evals.clone()))
            .collect();

        let evaluator = RaBooleanityWitness {
            eq,
            ra_polys: ra_witness,
            gamma_powers: self.gamma_powers.clone(),
            claim: F::zero(),
            uniskip_tau_1: self.eq_point.first().copied(),
        };

        StageBatch {
            claims: vec![SumcheckClaim {
                num_vars: self.num_vars,
                degree: 3,
                claimed_sum: F::zero(),
            }],
            witnesses: vec![Box::new(evaluator)],
        }
    }

    fn extract_claims(&mut self, challenges: &[F], _final_eval: F) -> Vec<ProverClaim<F>> {
        let ra_polys = self
            .ra_polys
            .take()
            .expect("extract_claims() called twice");

        // HighToLow binding: challenges are MSB-first, no reversal needed.
        let eval_point = challenges.to_vec();

        ra_polys
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

/// Sumcheck witness for the RA booleanity zero-check (degree 3).
///
/// Formula: `eq(r, x) · Σ_i γ^i · (ra_i(x)² - ra_i(x))`.
///
/// Uses HighToLow binding. The eq polynomial contributes degree 1,
/// and `ra²-ra` contributes degree 2, for degree 3 total.
struct RaBooleanityWitness<F: Field> {
    eq: Polynomial<F>,
    ra_polys: Vec<Polynomial<F>>,
    gamma_powers: Vec<F>,
    claim: F,
    /// First component of the eq challenge point, for uni-skip.
    uniskip_tau_1: Option<F>,
}

impl<F: Field> SumcheckCompute<F> for RaBooleanityWitness<F> {
    fn set_claim(&mut self, claim: F) {
        self.claim = claim;
    }

    fn first_round_polynomial(&self) -> Option<UnivariatePoly<F>> {
        let tau_1 = self.uniskip_tau_1?;
        // Uni-skip exploits t₁(0)=t₁(1)=0, valid only for zero-checks.
        if !self.claim.is_zero() {
            return None;
        }
        let half = self.eq.len() / 2;
        let two = F::from_u64(2);
        let one = F::one();

        // t₁(2) = Σ_j eq_rest[j] · Σ_i γ^i · (ra_i(2,j)² - ra_i(2,j))
        let mut t1_at_2 = F::zero();
        for j in 0..half {
            let (eq_lo, eq_hi) =
                self.eq
                    .sumcheck_eval_pair(j, jolt_poly::BindingOrder::HighToLow);
            let eq_rest = eq_lo + eq_hi;

            let mut inner_2 = F::zero();
            for (i, ra) in self.ra_polys.iter().enumerate() {
                let (ra_lo, ra_hi) =
                    ra.sumcheck_eval_pair(j, jolt_poly::BindingOrder::HighToLow);
                let ra_at_2 = two * ra_hi - ra_lo;
                inner_2 += self.gamma_powers[i] * ra_at_2 * (ra_at_2 - one);
            }

            t1_at_2 += eq_rest * inner_2;
        }

        Some(jolt_spartan::uniskip_round_poly(t1_at_2, tau_1))
    }

    fn round_polynomial(&self) -> UnivariatePoly<F> {
        let half = self.eq.len() / 2;
        let one = F::one();

        let mut eval_0 = F::zero();
        let mut eval_2 = F::zero();
        let mut eval_3 = F::zero();

        // HighToLow: pair [j] (MSB=0) with [j + half] (MSB=1).
        for j in 0..half {
            let (eq_lo, eq_hi) =
                self.eq
                    .sumcheck_eval_pair(j, jolt_poly::BindingOrder::HighToLow);
            let eq_d = eq_hi - eq_lo;

            // Accumulate γ-batched booleanity sum at t=0, t=2, t=3.
            let mut inner_0 = F::zero();
            let mut inner_2 = F::zero();
            let mut inner_3 = F::zero();

            for (i, ra) in self.ra_polys.iter().enumerate() {
                let (ra_lo, ra_hi) =
                    ra.sumcheck_eval_pair(j, jolt_poly::BindingOrder::HighToLow);
                let ra_d = ra_hi - ra_lo;
                let gamma_i = self.gamma_powers[i];

                // t=0: ra_lo² - ra_lo = ra_lo · (ra_lo - 1)
                inner_0 += gamma_i * ra_lo * (ra_lo - one);

                // t=2: ra_2 = ra_lo + 2·ra_d
                let ra_2 = ra_lo + ra_d + ra_d;
                inner_2 += gamma_i * ra_2 * (ra_2 - one);

                // t=3: ra_3 = ra_2 + ra_d
                let ra_3 = ra_2 + ra_d;
                inner_3 += gamma_i * ra_3 * (ra_3 - one);
            }

            // Multiply by eq at each evaluation point.
            eval_0 += eq_lo * inner_0;

            let eq_2 = eq_lo + eq_d + eq_d;
            eval_2 += eq_2 * inner_2;

            let eq_3 = eq_2 + eq_d;
            eval_3 += eq_3 * inner_3;
        }

        // Degree 3: hint = s(0)+s(1), evals at {0, 2, 3}.
        UnivariatePoly::from_evals_and_hint(self.claim, &[eval_0, eval_2, eval_3])
    }

    fn bind(&mut self, challenge: F) {
        self.eq.bind(challenge);
        for ra in &mut self.ra_polys {
            ra.bind(challenge);
        }
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
    use rand_core::{RngCore, SeedableRng};

    /// Brute-force computation of the RA booleanity sum.
    fn brute_force_ra_booleanity(
        ra_polys: &[Vec<Fr>],
        eq_point: &[Fr],
        gamma_powers: &[Fr],
    ) -> Fr {
        let n = ra_polys[0].len();
        let eq_table = EqPolynomial::new(eq_point.to_vec()).evaluations();
        let one = Fr::one();

        (0..n)
            .map(|j| {
                let inner: Fr = ra_polys
                    .iter()
                    .zip(gamma_powers.iter())
                    .map(|(ra, &g)| g * ra[j] * (ra[j] - one))
                    .sum();
                eq_table[j] * inner
            })
            .sum()
    }

    fn run_ra_booleanity_test(num_vars: usize, total_d: usize, seed: u64) {
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let gamma = Fr::random(&mut rng);

        // Build gamma powers [γ^0, γ^1, ..., γ^(total_d-1)].
        let gamma_powers: Vec<Fr> = {
            let mut powers = Vec::with_capacity(total_d);
            let mut g = Fr::one();
            for _ in 0..total_d {
                powers.push(g);
                g *= gamma;
            }
            powers
        };

        // Boolean RA polynomials (0 or 1 — booleanity sum should be 0).
        let ra_polys: Vec<Vec<Fr>> = (0..total_d)
            .map(|_| {
                (0..n)
                    .map(|_| {
                        if rng.next_u32() & 1 == 1 {
                            Fr::one()
                        } else {
                            Fr::zero()
                        }
                    })
                    .collect()
            })
            .collect();

        // Verify brute-force sum is zero for boolean inputs.
        let brute_sum = brute_force_ra_booleanity(&ra_polys, &eq_point, &gamma_powers);
        assert!(brute_sum.is_zero(), "brute-force sum should be zero for boolean RA polys");

        let ra_copy: Vec<Vec<Fr>> = ra_polys.clone();
        let mut stage = RaBooleanityStage::new(ra_polys, eq_point, gamma_powers.clone());

        let mut pt = Blake2bTranscript::new(b"ra_bool");
        let mut batch = stage.build(&[], &mut pt);

        assert_eq!(batch.claims.len(), 1);
        assert_eq!(batch.claims[0].degree, 3);
        assert_eq!(batch.claims[0].num_vars, num_vars);
        assert!(batch.claims[0].claimed_sum.is_zero());

        let claims_snapshot = batch.claims.clone();
        let proof = BatchedSumcheckProver::prove(
            &batch.claims,
            &mut batch.witnesses,
            &mut pt,
        );

        let mut vt = Blake2bTranscript::new(b"ra_bool");
        let (final_eval, challenges) = BatchedSumcheckVerifier::verify(
            &claims_snapshot,
            &proof,
            &mut vt,
        )
        .expect("verification should succeed");

        // Oracle check: final_eval = eq(r, challenges) · Σ_i γ^i · (ra_i(challenges)² - ra_i(challenges)).
        let eq_at_r =
            Polynomial::new(EqPolynomial::new(stage.eq_point.clone()).evaluations())
                .evaluate(&challenges);
        let one = Fr::one();
        let inner: Fr = ra_copy
            .iter()
            .zip(gamma_powers.iter())
            .map(|(ra, &g)| {
                let ra_at_r = Polynomial::new(ra.clone()).evaluate(&challenges);
                g * ra_at_r * (ra_at_r - one)
            })
            .sum();
        let expected = eq_at_r * inner;
        assert_eq!(final_eval, expected, "oracle check failed");

        // Extract claims and verify evaluations.
        let claims =
            <RaBooleanityStage<Fr> as ProverStage<Fr, Blake2bTranscript>>::extract_claims(
                &mut stage,
                &challenges,
                final_eval,
            );
        assert_eq!(claims.len(), total_d);
        for (i, claim) in claims.iter().enumerate() {
            let expected_eval = Polynomial::new(ra_copy[i].clone()).evaluate(&challenges);
            assert_eq!(claim.eval, expected_eval, "ra[{i}] eval mismatch");
            assert_eq!(claim.point, challenges, "ra[{i}] point mismatch");
        }
    }

    #[test]
    fn ra_booleanity_small_3_polys() {
        run_ra_booleanity_test(4, 3, 42);
    }

    #[test]
    fn ra_booleanity_medium_6_polys() {
        run_ra_booleanity_test(5, 6, 123);
    }

    #[test]
    fn ra_booleanity_single_poly() {
        run_ra_booleanity_test(4, 1, 777);
    }

    #[test]
    fn ra_booleanity_large_batch() {
        // Simulates a realistic-ish batch: 4 instruction + 5 bytecode + 3 ram = 12 polys.
        run_ra_booleanity_test(6, 12, 999);
    }

    #[test]
    fn ra_booleanity_non_boolean_witness() {
        // Non-boolean RA values → nonzero sum. Prove/verify should still work
        // (the sumcheck proves a correct claim, just not zero).
        let num_vars = 4;
        let n = 1usize << num_vars;
        let total_d = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(555);

        let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let gamma = Fr::random(&mut rng);

        let gamma_powers: Vec<Fr> = {
            let mut powers = Vec::with_capacity(total_d);
            let mut g = Fr::one();
            for _ in 0..total_d {
                powers.push(g);
                g *= gamma;
            }
            powers
        };

        // Random (non-boolean) RA polynomials.
        let ra_polys: Vec<Vec<Fr>> = (0..total_d)
            .map(|_| (0..n).map(|_| Fr::random(&mut rng)).collect())
            .collect();

        let claimed_sum = brute_force_ra_booleanity(&ra_polys, &eq_point, &gamma_powers);
        assert!(!claimed_sum.is_zero(), "random witness should have nonzero sum");

        // Build stage with the actual (nonzero) claimed sum.
        let mut stage = RaBooleanityStage {
            ra_polys: Some(ra_polys),
            eq_point,
            gamma_powers,
            num_vars,
        };

        // Override claimed_sum in the batch.
        let mut pt = Blake2bTranscript::new(b"nonbool");
        let mut batch = stage.build(&[], &mut pt);
        batch.claims[0].claimed_sum = claimed_sum;

        let claims_snapshot = batch.claims.clone();
        let proof = BatchedSumcheckProver::prove(
            &batch.claims,
            &mut batch.witnesses,
            &mut pt,
        );

        let mut vt = Blake2bTranscript::new(b"nonbool");
        let result = BatchedSumcheckVerifier::verify(
            &claims_snapshot,
            &proof,
            &mut vt,
        );
        assert!(result.is_ok(), "non-boolean verification failed");
    }

    #[test]
    fn ra_booleanity_all_zeros() {
        // All-zero RA polys → zero sum (trivially boolean).
        let num_vars = 3;
        let n = 1usize << num_vars;
        let total_d = 4;
        let mut rng = ChaCha20Rng::seed_from_u64(888);

        let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let gamma_powers = vec![Fr::one(); total_d];
        let ra_polys = vec![vec![Fr::zero(); n]; total_d];

        let mut stage = RaBooleanityStage::new(ra_polys, eq_point, gamma_powers);

        let mut pt = Blake2bTranscript::new(b"zeros");
        let mut batch = stage.build(&[], &mut pt);

        let claims_snapshot = batch.claims.clone();
        let proof = BatchedSumcheckProver::prove(
            &batch.claims,
            &mut batch.witnesses,
            &mut pt,
        );

        let mut vt = Blake2bTranscript::new(b"zeros");
        let result = BatchedSumcheckVerifier::verify(
            &claims_snapshot,
            &proof,
            &mut vt,
        );
        assert!(result.is_ok(), "all-zeros verification failed");
    }
}
