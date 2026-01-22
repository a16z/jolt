//! Shift sumcheck for verifying rho_next claims in packed GT exponentiation
//! Proves: Σ_i γ^i * v_i = Σ_{s,x} EqPlusOne(r_s*_i, s) × Eq(r_x*_i, x) × rho_i(s,x)
//!
//! This sumcheck runs after packed GT constraint sumcheck (Stage 1b) to verify
//! that claimed rho_next values equal rho at shifted positions.

use crate::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    virtual_claims,
    zkvm::{recursion::utils::virtual_polynomial_utils::*, witness::VirtualPolynomial},
};
use rayon::prelude::*;

/// Parameters for shift rho sumcheck
#[derive(Clone)]
pub struct ShiftRhoParams {
    /// Number of variables (11 = 7 step + 4 element)
    pub num_vars: usize,
    /// Number of claims to verify
    pub num_claims: usize,
    /// Sumcheck instance identifier
    pub sumcheck_id: SumcheckId,
}

impl ShiftRhoParams {
    pub fn new(num_claims: usize) -> Self {
        Self {
            num_vars: 11, // 7 step vars + 4 element vars
            num_claims,
            sumcheck_id: SumcheckId::ShiftRho,
        }
    }
}

/// Prover for shift rho sumcheck
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct ShiftRhoProver<F: JoltField, T: Transcript> {
    /// Parameters
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub params: ShiftRhoParams,

    /// Rho polynomials (11-var, one per claim)
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub rho_polys: Vec<MultilinearPolynomial<F>>,

    /// EqPlusOne polynomial for step variables (7-var, shared by all claims)
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub eq_plus_one_poly: MultilinearPolynomial<F>,

    /// Eq polynomial for element variables (4-var, shared by all claims)
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub eq_x_poly: MultilinearPolynomial<F>,

    /// Gamma for batching
    pub gamma: F,

    /// Current round
    pub round: usize,

    /// Claimed values for each rho_next
    pub claimed_values: Vec<F>,

    pub _marker: std::marker::PhantomData<T>,
}

impl<F: JoltField, T: Transcript> ShiftRhoProver<F, T> {
    pub fn new(
        params: ShiftRhoParams,
        rho_polys: Vec<Vec<F>>,
        claim_indices: Vec<usize>,
        accumulator: &ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Self {
        assert_eq!(params.num_claims, claim_indices.len());
        assert_eq!(params.num_claims, rho_polys.len());

        // Sample batching coefficient
        let gamma: F = transcript.challenge_scalar_optimized::<F>().into();

        struct ShiftEntry<F: JoltField> {
            constraint_idx: usize,
            rho_poly: Vec<F>,
            point: OpeningPoint<BIG_ENDIAN, F>,
            claimed_value: F,
        }

        let mut entries: Vec<ShiftEntry<F>> = claim_indices
            .into_iter()
            .zip(rho_polys.into_iter())
            .map(|(claim, rho_poly)| {
                let (point, claimed_value) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::PackedGtExpRhoNext(claim),
                    SumcheckId::PackedGtExp,
                );
                ShiftEntry {
                    constraint_idx: claim,
                    rho_poly,
                    point,
                    claimed_value,
                }
            })
            .collect();

        entries.sort_by_key(|entry| entry.constraint_idx);

        // All claims should share the same evaluation point from PackedGtExp sumcheck.
        let point = entries
            .first()
            .expect("shift rho requires at least one claim")
            .point
            .clone();
        for entry in &entries[1..] {
            debug_assert_eq!(
                entry.point.r, point.r,
                "ShiftRho claims must share the same opening point"
            );
        }

        // Convert rho polynomials
        let rho_polys = entries
            .iter()
            .map(|entry| MultilinearPolynomial::from(entry.rho_poly.clone()))
            .collect::<Vec<_>>();

        // Eq eval tables expect MSB-first ordering; sumcheck challenges are LSB-first.
        let r_s: Vec<_> = point.r[..7].to_vec();
        let r_x: Vec<_> = point.r[7..].to_vec();

        // Create single EqPlusOne polynomial for step variables (7-var).
        // point.r is in sumcheck round order (LSB first). EqPlusOnePolynomial::evals expects
        // big-endian and will interpret this as MSB-first, effectively matching LSB variable order.
        let eq_plus_one_evals = eq_plus_one_lsb_evals::<F>(&r_s);
        let eq_plus_one_poly = MultilinearPolynomial::from(eq_plus_one_evals);

        // Create single Eq polynomial for element variables (4-var).
        // Same endianness convention as above.
        let eq_x_evals = eq_lsb_evals::<F>(&r_x);
        let eq_x_poly = MultilinearPolynomial::from(eq_x_evals);

        // Collect all claimed values in the same order as rho polynomials.
        let claimed_values = entries
            .into_iter()
            .map(|entry| entry.claimed_value)
            .collect();

        Self {
            params,
            rho_polys,
            eq_plus_one_poly,
            eq_x_poly,
            gamma,
            round: 0,
            claimed_values,
            _marker: std::marker::PhantomData,
        }
    }

    /// Check if we're in Phase 1 (step variable rounds 0-6)
    /// Data layout: index = x * 128 + s (s in low 7 bits)
    /// With LowToHigh binding: rounds 0-6 bind s (step), rounds 7-10 bind x (element)
    fn in_step_phase(&self) -> bool {
        self.round < 7 // First 7 rounds are step phase
    }
}

pub(crate) fn eq_lsb_evals<F: JoltField>(r: &[F::Challenge]) -> Vec<F> {
    let n = r.len();
    let mut evals = vec![F::zero(); 1 << n];
    for idx in 0..(1 << n) {
        let mut prod = F::one();
        for i in 0..n {
            let bit = ((idx >> i) & 1) == 1;
            let r_i: F = r[i].into();
            let y_i = if bit { F::one() } else { F::zero() };
            prod *= r_i * y_i + (F::one() - r_i) * (F::one() - y_i);
        }
        evals[idx] = prod;
    }
    evals
}

pub(crate) fn eq_lsb_mle<F: JoltField>(r: &[F::Challenge], y: &[F::Challenge]) -> F {
    let mut prod = F::one();
    for i in 0..r.len() {
        let r_i: F = r[i].into();
        let y_i: F = y[i].into();
        prod *= r_i * y_i + (F::one() - r_i) * (F::one() - y_i);
    }
    prod
}

pub(crate) fn eq_plus_one_lsb_mle<F: JoltField>(r: &[F::Challenge], y: &[F::Challenge]) -> F {
    let n = r.len();
    let one = F::one();
    let mut sum = F::zero();
    for k in 0..n {
        let mut lower = F::one();
        for i in 0..k {
            let r_i: F = r[i].into();
            let y_i: F = y[i].into();
            lower *= r_i * (one - y_i);
        }

        let r_k: F = r[k].into();
        let y_k: F = y[k].into();
        let kth = (one - r_k) * y_k;

        let mut higher = F::one();
        for i in (k + 1)..n {
            let r_i: F = r[i].into();
            let y_i: F = y[i].into();
            higher *= r_i * y_i + (one - r_i) * (one - y_i);
        }

        sum += lower * kth * higher;
    }
    sum
}

pub(crate) fn eq_plus_one_lsb_evals<F: JoltField>(r: &[F::Challenge]) -> Vec<F> {
    let n = r.len();
    let mut evals = vec![F::zero(); 1 << n];
    for idx in 0..(1 << n) {
        let mut y = vec![F::Challenge::from(0u128); n];
        for i in 0..n {
            if ((idx >> i) & 1) == 1 {
                y[i] = F::Challenge::from(1u128);
            }
        }
        evals[idx] = eq_plus_one_lsb_mle::<F>(r, &y);
    }
    evals
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ShiftRhoProver<F, T> {
    fn degree(&self) -> usize {
        3 // EqPlusOne * Eq * rho (each degree 1)
    }

    fn num_rounds(&self) -> usize {
        self.params.num_vars
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        // The sum should equal the batched claimed values
        let mut sum = F::zero();
        let mut gamma_power = F::one();

        for claimed_value in &self.claimed_values {
            sum += gamma_power * claimed_value;
            gamma_power *= self.gamma;
        }

        sum
    }

    #[tracing::instrument(skip_all, name = "ShiftRho::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        const DEGREE: usize = 3;

        let half = if !self.rho_polys.is_empty() {
            self.rho_polys[0].len() / 2
        } else {
            return UniPoly::from_evals_and_hint(previous_claim, &[F::zero(); DEGREE]);
        };

        let gamma = self.gamma;
        let in_step_phase = self.in_step_phase();

        // Phase-aware sizes
        let eq_plus_one_half = if in_step_phase {
            self.eq_plus_one_poly.len() / 2
        } else {
            1 // Fully bound in element phase
        };
        let eq_x_len = self.eq_x_poly.len();

        // Compute evaluations in parallel
        let evals = (0..half)
            .into_par_iter()
            .map(|i| {
                // Compute eq contributions based on phase (shared across all claims)
                let (eq_plus_one_evals, eq_x_evals) = if in_step_phase {
                    // Phase 1 (rounds 0-6): sumcheck over s, eq_x is constant per x-block
                    // Index i maps to: s_pair_idx = i % eq_plus_one_half, x_idx = i / eq_plus_one_half
                    let s_pair_idx = i % eq_plus_one_half;
                    let x_idx = i / eq_plus_one_half;

                    let eq_plus_one_arr = self
                        .eq_plus_one_poly
                        .sumcheck_evals_array::<DEGREE>(s_pair_idx, BindingOrder::LowToHigh);

                    // eq_x[x_idx] is constant for this s-block
                    let eq_x_val = if x_idx < eq_x_len {
                        self.eq_x_poly.get_bound_coeff(x_idx)
                    } else {
                        F::zero()
                    };
                    let eq_x_arr = [eq_x_val; DEGREE];

                    (eq_plus_one_arr, eq_x_arr)
                } else {
                    // Phase 2 (rounds 7-10): eq_plus_one is fully bound (constant), sumcheck over x
                    let eq_plus_one_val = self.eq_plus_one_poly.get_bound_coeff(0);
                    let eq_plus_one_arr = [eq_plus_one_val; DEGREE];

                    let eq_x_arr = self
                        .eq_x_poly
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                    (eq_plus_one_arr, eq_x_arr)
                };

                let mut term_evals = [F::zero(); DEGREE];
                let mut gamma_power = F::one();

                for claim_idx in 0..self.params.num_claims {
                    let rho_evals = self.rho_polys[claim_idx]
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                    for t in 0..DEGREE {
                        term_evals[t] +=
                            eq_plus_one_evals[t] * eq_x_evals[t] * gamma_power * rho_evals[t];
                    }
                    gamma_power *= gamma;
                }

                term_evals
            })
            .reduce(
                || [F::zero(); DEGREE],
                |mut acc, evals| {
                    for (a, e) in acc.iter_mut().zip(evals.iter()) {
                        *a += *e;
                    }
                    acc
                },
            );

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        // Always bind rho polynomials (11-var)
        for rho in &mut self.rho_polys {
            rho.bind_parallel(r_j, BindingOrder::LowToHigh);
        }

        if self.in_step_phase() {
            // Phase 1: Bind eq_plus_one polynomial (7-var)
            self.eq_plus_one_poly
                .bind_parallel(r_j, BindingOrder::LowToHigh);
            // eq_x polynomial remains unbound in this phase
        } else {
            // Phase 2: Bind eq_x polynomial (4-var)
            // eq_plus_one polynomial is already fully bound
            self.eq_x_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }

        self.round = round + 1;
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = OpeningPoint::<{ BIG_ENDIAN }, F>::new(sumcheck_challenges.to_vec());

        for idx in 0..self.params.num_claims {
            // Get the final rho evaluation after all rounds
            let rho_eval = self.rho_polys[idx].get_bound_coeff(0);

            // Cache the rho evaluation at the shift sumcheck challenge point
            let claims = virtual_claims![
                VirtualPolynomial::PackedGtExpRho(idx) => rho_eval,
            ];
            append_virtual_claims(
                accumulator,
                transcript,
                SumcheckId::ShiftRho,
                &opening_point,
                &claims,
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Verifier for shift rho sumcheck
pub struct ShiftRhoVerifier<F: JoltField> {
    pub params: ShiftRhoParams,
    pub claim_indices: Vec<usize>,
    pub gamma: F,
}

impl<F: JoltField> ShiftRhoVerifier<F> {
    pub fn new<T: Transcript>(
        params: ShiftRhoParams,
        claim_indices: Vec<usize>,
        transcript: &mut T,
    ) -> Self {
        assert_eq!(params.num_claims, claim_indices.len());

        // Sample same batching coefficient
        let gamma: F = transcript.challenge_scalar_optimized::<F>().into();

        Self {
            params,
            claim_indices,
            gamma,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ShiftRhoVerifier<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.params.num_vars
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        // The sum should equal the batched claimed values fetched from accumulator
        let mut sum = F::zero();
        let mut gamma_power = F::one();

        for claim in &self.claim_indices {
            let (_, value) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::PackedGtExpRhoNext(*claim),
                SumcheckId::PackedGtExp,
            );

            sum += gamma_power * value;
            gamma_power *= self.gamma;
        }

        sum
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // Get the shared point from the first claim (all claims share the same point)
        let (rho_next_point, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::PackedGtExpRhoNext(self.claim_indices[0]),
            SumcheckId::PackedGtExp,
        );

        // Eq eval tables expect MSB-first ordering; sumcheck challenges are LSB-first.
        let r_s: Vec<_> = rho_next_point.r[..7].to_vec();
        let r_x: Vec<_> = rho_next_point.r[7..].to_vec();

        let s_challenges: Vec<_> = sumcheck_challenges[..7].to_vec();
        let x_challenges: Vec<_> = sumcheck_challenges[7..].to_vec();

        // Compute eq polynomials once (shared across all claims).
        // Use LSB-first ordering for both the opening point and sumcheck challenges.
        let eq_plus_one = eq_plus_one_lsb_mle::<F>(&r_s, &s_challenges);
        let eq_x = eq_lsb_mle::<F>(&r_x, &x_challenges);
        let eq_product = eq_plus_one * eq_x;

        // Compute batched rho sum
        let mut rho_sum = F::zero();
        let mut gamma_power = F::one();

        for claim in &self.claim_indices {
            // Get rho evaluation at the shift sumcheck challenge point
            let (_, rho_eval) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::PackedGtExpRho(*claim),
                SumcheckId::ShiftRho, // ShiftRho sumcheck ID!
            );

            rho_sum += gamma_power * rho_eval;
            gamma_power *= self.gamma;
        }

        let sum = eq_product * rho_sum;

        sum
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = OpeningPoint::<{ BIG_ENDIAN }, F>::new(sumcheck_challenges.to_vec());

        for claim in &self.claim_indices {
            let polynomials = vec![VirtualPolynomial::PackedGtExpRho(*claim)];
            append_virtual_openings(
                accumulator,
                transcript,
                SumcheckId::ShiftRho,
                &opening_point,
                &polynomials,
            );
        }
    }
}
