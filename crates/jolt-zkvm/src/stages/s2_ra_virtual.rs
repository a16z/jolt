//! Stage 2: RA virtual sumcheck.
//!
//! Materializes the RA (read-address) virtual polynomial as a product of
//! committed RA chunks and reduces via eq-weighted sumcheck.
//!
//! The sumcheck proves:
//! ```text
//! Σ_x eq(w, x) · Σ_i γ^i · Π_{j=0}^{m-1} ra_{i·m+j}(x) = claimed_sum
//! ```
//!
//! where `m = n_committed_per_virtual` and the sum is over all virtual
//! RA polynomials.

use std::sync::Arc;

use jolt_field::WithChallenge;
use jolt_ir::ClaimDefinition;
use jolt_openings::ProverClaim;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_sumcheck::claim::SumcheckClaim;
use jolt_sumcheck::prover::SumcheckCompute;
use jolt_sumcheck::SplitEqEvaluator;
use jolt_transcript::Transcript;

use crate::claims::instruction;
use crate::stage::{ProverStage, StageBatch};
use crate::evaluators::ra_poly::RaPolynomial;
use crate::evaluators::ra_virtual::RaVirtualCompute;

/// RA virtual sumcheck prover stage.
///
/// Constructs a single sumcheck instance from committed RA chunk polynomials
/// and a split-eq evaluator. After sumcheck completes, extracts opening claims
/// for each committed RA chunk polynomial at the challenge point.
pub struct RaVirtualStage<F: WithChallenge> {
    /// RA chunk polynomial evaluation tables (consumed by build).
    ra_tables: Option<Vec<Vec<F>>>,
    /// Lookup indices shared across RA polynomials.
    lookup_indices: Vec<Arc<Vec<Option<u8>>>>,
    /// Eq polynomial evaluation point (from transcript challenges).
    eq_point: Vec<F::Challenge>,
    /// γ-power coefficients for combining virtual polynomials.
    #[allow(dead_code)]
    gamma_powers: Vec<F>,
    /// Number of virtual RA polynomials.
    n_virtual: usize,
    /// Number of committed chunks per virtual polynomial.
    n_committed_per_virtual: usize,
    /// Number of sumcheck variables.
    num_vars: usize,
    /// Claimed sum for the sumcheck instance.
    claimed_sum: F,
}

impl<F: WithChallenge> RaVirtualStage<F> {
    /// Creates a new RA virtual sumcheck stage.
    ///
    /// # Arguments
    ///
    /// * `ra_tables` — evaluation tables for each committed RA chunk,
    ///   ordered `[ra_{0,0}, ra_{0,1}, ..., ra_{n_v-1, m-1}]`
    /// * `lookup_indices` — one index vector per committed chunk
    /// * `eq_point` — challenge vector for the split-eq evaluator
    /// * `gamma_powers` — `[γ^0, γ^1, ..., γ^{n_virtual-1}]` scaled by eq
    /// * `n_virtual` — number of virtual RA polynomials
    /// * `n_committed_per_virtual` — chunks per virtual polynomial
    /// * `claimed_sum` — expected sum `g(0) + g(1) + ... = claimed_sum`
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ra_tables: Vec<Vec<F>>,
        lookup_indices: Vec<Arc<Vec<Option<u8>>>>,
        eq_point: Vec<F::Challenge>,
        gamma_powers: Vec<F>,
        n_virtual: usize,
        n_committed_per_virtual: usize,
        claimed_sum: F,
    ) -> Self {
        let num_vars = eq_point.len();
        let total_chunks = n_virtual * n_committed_per_virtual;
        assert_eq!(ra_tables.len(), total_chunks);
        assert_eq!(lookup_indices.len(), total_chunks);
        assert_eq!(gamma_powers.len(), n_virtual);

        Self {
            ra_tables: Some(ra_tables),
            lookup_indices,
            eq_point,
            gamma_powers,
            n_virtual,
            n_committed_per_virtual,
            num_vars,
            claimed_sum,
        }
    }
}

impl<F: WithChallenge, T: Transcript> ProverStage<F, T> for RaVirtualStage<F> {
    fn build(&mut self, _prior_claims: &[ProverClaim<F>], _transcript: &mut T) -> StageBatch<F> {
        let ra_tables = self
            .ra_tables
            .as_ref()
            .expect("build() called after extract_claims()");

        let mut ra_polys: Vec<RaPolynomial<u8, F>> =
            Vec::with_capacity(self.n_virtual * self.n_committed_per_virtual);

        for (chunk_idx, table) in ra_tables.iter().enumerate() {
            let indices = Arc::clone(&self.lookup_indices[chunk_idx]);
            let ra = RaPolynomial::new(indices, table.clone());
            ra_polys.push(ra);
        }

        let eq_poly =
            SplitEqEvaluator::new_with_scaling(&self.eq_point, BindingOrder::LowToHigh, None);

        let degree = self.n_committed_per_virtual + 1; // eq contributes 1

        let claim = SumcheckClaim {
            num_vars: self.num_vars,
            degree,
            claimed_sum: self.claimed_sum,
        };

        let witness: Box<dyn SumcheckCompute<F>> = Box::new(RaVirtualCompute {
            mles: ra_polys,
            eq_poly,
            claim: self.claimed_sum,
            binding_order: BindingOrder::LowToHigh,
            gamma_powers: self.gamma_powers.clone(),
            n_products: self.n_virtual,
        });

        StageBatch {
            claims: vec![claim],
            witnesses: vec![witness],
        }
    }

    fn extract_claims(&mut self, challenges: &[F], _final_eval: F) -> Vec<ProverClaim<F>> {
        let ra_tables = self
            .ra_tables
            .take()
            .expect("extract_claims() called twice");

        ra_tables
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
        vec![instruction::instruction_ra_virtual(
            self.n_virtual,
            self.n_committed_per_virtual,
        )]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr, WithChallenge};
    use jolt_poly::{EqPolynomial, UnivariatePoly};
    use jolt_sumcheck::{BatchedSumcheckProver, BatchedSumcheckVerifier, ClearRoundHandler};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use num_traits::Zero;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    type Challenge = <Fr as WithChallenge>::Challenge;

    fn random_eq_point(num_vars: usize, rng: &mut ChaCha20Rng) -> Vec<Challenge> {
        (0..num_vars)
            .map(|_| Challenge::from(rng.next_u64() as u128))
            .collect()
    }

    /// Brute-force computation of the RA virtual sumcheck sum.
    fn brute_force_sum(
        polys: &[Vec<Fr>],
        eq_point: &[Challenge],
        gamma_powers: &[Fr],
        n_virtual: usize,
        m: usize,
    ) -> Fr {
        let eq_evals =
            EqPolynomial::new(eq_point.iter().map(|&c| Fr::from(c)).collect()).evaluations();

        let mut sum = Fr::zero();
        for (x, &eq_val) in eq_evals.iter().enumerate() {
            let mut inner = Fr::zero();
            for i in 0..n_virtual {
                let mut product = gamma_powers[i];
                for j in 0..m {
                    product *= polys[i * m + j][x];
                }
                inner += product;
            }
            sum += eq_val * inner;
        }
        sum
    }

    type Indices = Vec<Arc<Vec<Option<u8>>>>;

    #[allow(clippy::type_complexity)]
    fn random_ra_data(
        num_vars: usize,
        num_polys: usize,
        rng: &mut ChaCha20Rng,
    ) -> (Vec<Vec<Fr>>, Indices) {
        let n = 1usize << num_vars;
        let tables: Vec<Vec<Fr>> = (0..num_polys)
            .map(|_| (0..n).map(|_| Fr::from_u64(rng.next_u64() % 100)).collect())
            .collect();

        let indices: Vec<Arc<Vec<Option<u8>>>> = (0..num_polys)
            .map(|_| Arc::new((0..n).map(|j| Some((j % 256) as u8)).collect()))
            .collect();

        (tables, indices)
    }

    #[test]
    fn small_2_virtual_2_committed() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let num_vars = 3;
        let n_virtual = 2;
        let m = 2;
        let total = n_virtual * m;

        let (tables, indices) = random_ra_data(num_vars, total, &mut rng);

        let eq_point = random_eq_point(num_vars, &mut rng);
        let gamma = Fr::from_u64(rng.next_u64() % 1000 + 1);
        let gamma_powers = vec![Fr::from_u64(1), gamma];

        let claimed_sum = brute_force_sum(&tables, &eq_point, &gamma_powers, n_virtual, m);

        let mut stage = RaVirtualStage::new(
            tables.clone(),
            indices,
            eq_point,
            gamma_powers,
            n_virtual,
            m,
            claimed_sum,
        );

        let mut transcript = Blake2bTranscript::new(b"s2-test");
        let mut batch = stage.build(&[], &mut transcript);

        assert_eq!(batch.claims.len(), 1);
        assert_eq!(batch.claims[0].num_vars, num_vars);
        assert_eq!(batch.claims[0].degree, m + 1);
        assert_eq!(batch.claims[0].claimed_sum, claimed_sum);

        // Run sumcheck
        let handler = ClearRoundHandler::with_capacity(num_vars);
        let proof = BatchedSumcheckProver::prove_with_handler(
            &batch.claims,
            &mut batch.witnesses,
            &mut transcript,
            |c: u128| Fr::from_u128(c),
            handler,
        );

        // Verify
        let mut vt = Blake2bTranscript::new(b"s2-test");
        let verify_claim = SumcheckClaim {
            num_vars,
            degree: m + 1,
            claimed_sum,
        };
        let _result =
            BatchedSumcheckVerifier::verify(&[verify_claim], &proof, &mut vt, |c: u128| {
                Fr::from_u128(c)
            })
            .expect("verification should succeed");
    }

    #[test]
    fn extract_claims_produces_correct_evaluations() {
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let num_vars = 2;
        let n_virtual = 1;
        let m = 2;
        let total = n_virtual * m;

        let (tables, indices) = random_ra_data(num_vars, total, &mut rng);

        let eq_point = random_eq_point(num_vars, &mut rng);
        let gamma_powers = vec![Fr::from_u64(1)];

        let claimed_sum = brute_force_sum(&tables, &eq_point, &gamma_powers, n_virtual, m);

        let mut stage = RaVirtualStage::new(
            tables.clone(),
            indices,
            eq_point,
            gamma_powers,
            n_virtual,
            m,
            claimed_sum,
        );

        let mut transcript = Blake2bTranscript::new(b"s2-extract");
        let mut batch = stage.build(&[], &mut transcript);

        let (_proof, challenges) = {
            struct TrackingHandler<F: Field> {
                inner: ClearRoundHandler<F>,
                challenges: Vec<F>,
            }

            impl<F: Field> jolt_sumcheck::handler::RoundHandler<F> for TrackingHandler<F> {
                type Proof = (jolt_sumcheck::SumcheckProof<F>, Vec<F>);

                fn absorb_round_poly(
                    &mut self,
                    poly: &UnivariatePoly<F>,
                    transcript: &mut impl Transcript,
                ) {
                    self.inner.absorb_round_poly(poly, transcript);
                }

                fn on_challenge(&mut self, challenge: F) {
                    self.challenges.push(challenge);
                }

                fn finalize(self) -> (jolt_sumcheck::SumcheckProof<F>, Vec<F>) {
                    (self.inner.finalize(), self.challenges)
                }
            }

            let handler = TrackingHandler {
                inner: ClearRoundHandler::with_capacity(num_vars),
                challenges: Vec::with_capacity(num_vars),
            };
            BatchedSumcheckProver::prove_with_handler(
                &batch.claims,
                &mut batch.witnesses,
                &mut transcript,
                |c: u128| Fr::from_u128(c),
                handler,
            )
        };

        let claims = <RaVirtualStage<Fr> as ProverStage<Fr, Blake2bTranscript>>::extract_claims(
            &mut stage,
            &challenges,
            Fr::zero(),
        );
        assert_eq!(claims.len(), total);

        for (i, claim) in claims.iter().enumerate() {
            let poly = Polynomial::new(tables[i].clone());
            let expected_eval = poly.evaluate(&challenges);
            assert_eq!(claim.eval, expected_eval, "claim {i} eval mismatch");
            assert_eq!(claim.point, challenges);
        }
    }

    #[test]
    fn claim_definitions_match_parameters() {
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let num_vars = 2;
        let n_virtual = 3;
        let m = 2;
        let total = n_virtual * m;

        let (tables, indices) = random_ra_data(num_vars, total, &mut rng);
        let eq_point = random_eq_point(num_vars, &mut rng);
        let gamma_powers: Vec<Fr> = (0..n_virtual)
            .map(|i| Fr::from_u64((i + 1) as u64))
            .collect();
        let claimed_sum = brute_force_sum(&tables, &eq_point, &gamma_powers, n_virtual, m);

        let stage = RaVirtualStage::new(
            tables,
            indices,
            eq_point,
            gamma_powers.clone(),
            n_virtual,
            m,
            claimed_sum,
        );

        let defs =
            <RaVirtualStage<Fr> as ProverStage<Fr, Blake2bTranscript>>::claim_definitions(&stage);
        assert_eq!(defs.len(), 1);

        let def = &defs[0];
        assert_eq!(def.opening_bindings.len(), total);
        assert_eq!(def.challenge_bindings.len(), n_virtual);

        let openings: Vec<Fr> = (1..=total as u64).map(Fr::from_u64).collect();
        let challenges_eval: Vec<Fr> = gamma_powers.clone();

        let result = def.evaluate::<Fr>(&openings, &challenges_eval);

        let mut expected = Fr::zero();
        for i in 0..n_virtual {
            let mut product = challenges_eval[i];
            for j in 0..m {
                product *= openings[i * m + j];
            }
            expected += product;
        }
        assert_eq!(result, expected);
    }
}
