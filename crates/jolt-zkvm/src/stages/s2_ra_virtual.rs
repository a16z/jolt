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

use jolt_compute::ComputeBackend;
use jolt_field::WithChallenge;
use jolt_ir::ClaimDefinition;
use jolt_openings::ProverClaim;
use jolt_poly::Polynomial;
use jolt_sumcheck::claim::SumcheckClaim;
use jolt_transcript::Transcript;

use crate::evaluators::catalog;
use crate::evaluators::kernel::KernelEvaluator;
use crate::stage::{ProverStage, StageBatch};

/// RA virtual sumcheck prover stage.
///
/// Constructs a single sumcheck instance from committed RA chunk polynomials
/// and a split-eq evaluator. After sumcheck completes, extracts opening claims
/// for each committed RA chunk polynomial at the challenge point.
pub struct RaVirtualStage<F: WithChallenge, B: ComputeBackend> {
    /// RA chunk polynomial evaluation tables (consumed by build).
    ra_tables: Option<Vec<Vec<F>>>,
    /// Lookup indices shared across RA polynomials.
    lookup_indices: Vec<Arc<Vec<Option<u8>>>>,
    /// Eq polynomial evaluation point (from transcript challenges).
    eq_point: Vec<F::Challenge>,
    /// γ-power coefficients for combining virtual polynomials.
    gamma_powers: Vec<F>,
    /// Number of virtual RA polynomials.
    n_virtual: usize,
    /// Number of committed chunks per virtual polynomial.
    n_committed_per_virtual: usize,
    /// Number of sumcheck variables.
    num_vars: usize,
    /// Claimed sum for the sumcheck instance.
    claimed_sum: F,
    /// Compute backend for kernel compilation and buffer management.
    backend: Arc<B>,
    /// Claim definition for this RA virtual instance.
    claim_definition: ClaimDefinition,
}

impl<F: WithChallenge, B: ComputeBackend> RaVirtualStage<F, B> {
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
    /// * `backend` — compute backend for kernel compilation and buffer uploads
    /// * `claim_definition` — claim definition for this RA virtual instance
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ra_tables: Vec<Vec<F>>,
        lookup_indices: Vec<Arc<Vec<Option<u8>>>>,
        eq_point: Vec<F::Challenge>,
        gamma_powers: Vec<F>,
        n_virtual: usize,
        n_committed_per_virtual: usize,
        claimed_sum: F,
        backend: Arc<B>,
        claim_definition: ClaimDefinition,
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
            backend,
            claim_definition,
        }
    }
}

impl<F: WithChallenge, B: ComputeBackend, T: Transcript> ProverStage<F, T>
    for RaVirtualStage<F, B>
{
    fn name(&self) -> &'static str {
        "S2_ra_virtual"
    }

    fn build(&mut self, _prior_claims: &[ProverClaim<F>], _transcript: &mut T) -> StageBatch<F> {
        let ra_tables = self
            .ra_tables
            .as_ref()
            .expect("build() called after extract_claims()");

        let n = 1usize << self.num_vars;
        let total_chunks = self.n_virtual * self.n_committed_per_virtual;
        let d = self.n_committed_per_virtual;

        // Materialize RA polynomials to dense evaluation tables by gathering
        // through lookup indices. Pre-scale the first polynomial of each
        // product group by γ^t so the plain ProductSum kernel computes
        // Σ_t γ^t · Π_k p_{t·D+k} without needing per-term coefficients.
        let dense_polys: Vec<Vec<F>> = (0..total_chunks)
            .map(|chunk_idx| {
                let table = &ra_tables[chunk_idx];
                let indices = &self.lookup_indices[chunk_idx];
                let group = chunk_idx / d;
                let is_first_in_group = chunk_idx % d == 0;
                let gamma = if is_first_in_group {
                    self.gamma_powers[group]
                } else {
                    F::one()
                };
                (0..n)
                    .map(|j| indices[j].map_or(F::zero(), |i| gamma * table[i as usize]))
                    .collect()
            })
            .collect();

        let eq_f: Vec<F> = self.eq_point.iter().map(|&c| c.into()).collect();
        let degree = d + 1;

        let desc = catalog::product_sum(d, self.n_virtual);
        let kernel = self.backend.compile_kernel::<F>(&desc);

        let backend = Arc::clone(&self.backend);
        let inputs: Vec<_> = dense_polys.iter().map(|p| backend.upload(p)).collect();
        let witness = KernelEvaluator::with_toom_cook_eq(
            inputs,
            kernel,
            desc.num_evals(),
            eq_f,
            self.claimed_sum,
            backend,
        );

        StageBatch {
            claims: vec![SumcheckClaim {
                num_vars: self.num_vars,
                degree,
                claimed_sum: self.claimed_sum,
            }],
            witnesses: vec![Box::new(witness)],
        }
    }

    fn extract_claims(&mut self, challenges: &[F], _final_eval: F) -> Vec<ProverClaim<F>> {
        let ra_tables = self
            .ra_tables
            .take()
            .expect("extract_claims() called twice");

        // LowToHigh binding → reverse for MSB-first evaluation.
        let eval_point: Vec<F> = challenges.iter().rev().copied().collect();

        ra_tables
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
        vec![self.claim_definition.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_cpu::CpuBackend;
    use jolt_field::{Field, Fr, WithChallenge};
    use jolt_ir::zkvm::claims::{instruction, ram};
    use jolt_poly::{EqPolynomial, UnivariatePoly};
    use jolt_sumcheck::{BatchedSumcheckProver, BatchedSumcheckVerifier, ClearRoundHandler};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use num_traits::Zero;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    type Challenge = <Fr as WithChallenge>::Challenge;

    fn cpu() -> Arc<CpuBackend> {
        Arc::new(CpuBackend)
    }

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
            cpu(),
            instruction::instruction_ra_virtual(n_virtual, m),
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
            BatchedSumcheckVerifier::verify(&[verify_claim], &proof, &mut vt)
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
            cpu(),
            instruction::instruction_ra_virtual(n_virtual, m),
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
                handler,
            )
        };

        let claims =
            <RaVirtualStage<Fr, CpuBackend> as ProverStage<Fr, Blake2bTranscript>>::extract_claims(
                &mut stage,
                &challenges,
                Fr::zero(),
            );
        assert_eq!(claims.len(), total);

        let eval_point: Vec<Fr> = challenges.iter().rev().copied().collect();
        for (i, claim) in claims.iter().enumerate() {
            let poly = Polynomial::new(tables[i].clone());
            let expected_eval = poly.evaluate(&eval_point);
            assert_eq!(claim.eval, expected_eval, "claim {i} eval mismatch");
            assert_eq!(claim.point, eval_point);
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
            cpu(),
            instruction::instruction_ra_virtual(n_virtual, m),
        );

        let defs =
            <RaVirtualStage<Fr, CpuBackend> as ProverStage<Fr, Blake2bTranscript>>::claim_definitions(&stage);
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

    /// RAM RA virtual: n_virtual=1, d chunks, gamma_powers=[1].
    /// Same RaVirtualStage but with RAM claim definition.
    #[test]
    fn ram_ra_virtual_single_product() {
        let mut rng = ChaCha20Rng::seed_from_u64(55);
        let num_vars = 4;
        let n_virtual = 1;
        let d = 4; // typical ram_d
        let total = n_virtual * d;

        let (tables, indices) = random_ra_data(num_vars, total, &mut rng);
        let eq_point = random_eq_point(num_vars, &mut rng);
        let gamma_powers = vec![Fr::from_u64(1)];

        let claimed_sum = brute_force_sum(&tables, &eq_point, &gamma_powers, n_virtual, d);

        let mut stage = RaVirtualStage::new(
            tables.clone(),
            indices,
            eq_point,
            gamma_powers,
            n_virtual,
            d,
            claimed_sum,
            cpu(),
            ram::ram_ra_virtual(d),
        );

        let mut transcript = Blake2bTranscript::new(b"ram-ra-virtual");
        let mut batch = stage.build(&[], &mut transcript);

        assert_eq!(batch.claims.len(), 1);
        assert_eq!(batch.claims[0].degree, d + 1);
        assert_eq!(batch.claims[0].num_vars, num_vars);

        let claims_snapshot = batch.claims.clone();
        let handler = ClearRoundHandler::with_capacity(num_vars);
        let proof = BatchedSumcheckProver::prove_with_handler(
            &batch.claims,
            &mut batch.witnesses,
            &mut transcript,
            handler,
        );

        let mut vt = Blake2bTranscript::new(b"ram-ra-virtual");
        let _result = BatchedSumcheckVerifier::verify(
            &claims_snapshot,
            &proof,
            &mut vt,
        )
        .expect("RAM RA virtual verification should succeed");
    }

    #[test]
    fn ram_ra_virtual_claim_definition() {
        let mut rng = ChaCha20Rng::seed_from_u64(77);
        let num_vars = 3;
        let d = 3;

        let (tables, indices) = random_ra_data(num_vars, d, &mut rng);
        let eq_point = random_eq_point(num_vars, &mut rng);
        let gamma_powers = vec![Fr::from_u64(1)];
        let claimed_sum = brute_force_sum(&tables, &eq_point, &gamma_powers, 1, d);

        let stage = RaVirtualStage::new(
            tables,
            indices,
            eq_point,
            gamma_powers,
            1,
            d,
            claimed_sum,
            cpu(),
            ram::ram_ra_virtual(d),
        );

        let defs =
            <RaVirtualStage<Fr, CpuBackend> as ProverStage<Fr, Blake2bTranscript>>::claim_definitions(&stage);
        assert_eq!(defs.len(), 1);

        let def = &defs[0];
        assert_eq!(def.opening_bindings.len(), d);
        // RAM: single challenge (eq_eval)
        assert_eq!(def.challenge_bindings.len(), 1);

        // Verify formula: c0 * ra0 * ra1 * ra2
        let openings: Vec<Fr> = (2..=4).map(Fr::from_u64).collect();
        let eq_eval = Fr::from_u64(7);

        let result = def.evaluate::<Fr>(&openings, &[eq_eval]);
        let expected = eq_eval * Fr::from_u64(2) * Fr::from_u64(3) * Fr::from_u64(4);
        assert_eq!(result, expected);
    }
}
