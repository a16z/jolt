//! Shift sumcheck for verifying rho_next claims in packed GT exponentiation
//! Proves: Σ_i γ^i * v_i = Σ_{s,x} EqPlusOne(r_s*_i, s) × Eq(r_x*_i, x) × rho_i(s,x)
//!
//! This sumcheck runs after packed GT constraint sumcheck (Stage 1b) to verify
//! that claimed rho_next values equal rho at shifted positions.

use crate::{
    field::JoltField,
    poly::{
        eq_plus_one_poly::EqPlusOnePolynomial,
        eq_poly::EqPolynomial,
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

/// Shift claim to be verified
#[derive(Clone, Debug)]
pub struct ShiftClaim {
    /// Constraint index
    pub constraint_idx: usize,
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
        claims: Vec<ShiftClaim>,
        accumulator: &ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Self {
        assert_eq!(params.num_claims, claims.len());
        assert_eq!(params.num_claims, rho_polys.len());

        // Sample batching coefficient
        let gamma: F = transcript.challenge_scalar_optimized::<F>().into();

        // Convert rho polynomials
        let rho_polys = rho_polys
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect();

        // All claims share the same evaluation point from PackedGtExp sumcheck
        // Fetch the point from the first claim
        let (point, first_value) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::PackedGtExpRhoNext(claims[0].constraint_idx),
            SumcheckId::PackedGtExp,
        );

        // Split point into step and element parts
        let r_s = &point.r[..7];
        let r_x = &point.r[7..];

        // Create single EqPlusOne polynomial for step variables (7-var).
        // point.r is in sumcheck round order (LSB first). EqPlusOnePolynomial::evals expects
        // big-endian and will interpret this as MSB-first, effectively matching LSB variable order.
        let (_, eq_plus_one_evals) = EqPlusOnePolynomial::<F>::evals(r_s, None);
        let eq_plus_one_poly = MultilinearPolynomial::from(eq_plus_one_evals);

        // Create single Eq polynomial for element variables (4-var).
        // Same endianness convention as above.
        let eq_x_evals = EqPolynomial::<F>::evals(r_x);
        let eq_x_poly = MultilinearPolynomial::from(eq_x_evals);

        // Collect all claimed values
        let mut claimed_values = vec![first_value];
        for claim in &claims[1..] {
            let (_, value) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::PackedGtExpRhoNext(claim.constraint_idx),
                SumcheckId::PackedGtExp,
            );
            claimed_values.push(value);
        }

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

        #[cfg(debug_assertions)]
        {
            eprintln!(
                "Prover input_claim: num_claims = {}",
                self.claimed_values.len()
            );
            eprintln!("Prover input_claim: gamma = {:?}", self.gamma);
            eprintln!("Prover input_claim total = {:?}", sum);
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
                let mut term_evals = [F::zero(); DEGREE];

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

                // First compute the batched rho sum: Σ_i γ^i × rho_i
                let mut rho_sum = [F::zero(); DEGREE];
                let mut gamma_power = F::one();
                for claim_idx in 0..self.params.num_claims {
                    let rho_evals = self.rho_polys[claim_idx]
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                    for t in 0..DEGREE {
                        rho_sum[t] += gamma_power * rho_evals[t];
                    }
                    gamma_power *= gamma;
                }

                // Then multiply by eq polynomials: (EqPlusOne * Eq) * [Σ_i γ^i × rho_i]
                for t in 0..DEGREE {
                    term_evals[t] = eq_plus_one_evals[t] * eq_x_evals[t] * rho_sum[t];
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

        // Debug: verify s(0) + s(1) = previous_claim
        #[cfg(test)]
        {
            // Directly compute s(0) and s(1) by summing over all indices
            let mut s_0 = F::zero();
            let mut s_1 = F::zero();

            let full_len = half * 2;
            let in_step_phase = self.in_step_phase();

            // Phase-aware sizes for debug calculation
            let eq_plus_one_len = if in_step_phase {
                self.eq_plus_one_poly.len()
            } else {
                1 // Fully bound in element phase
            };
            let eq_x_len = self.eq_x_poly.len();

            for i in 0..full_len {
                // Compute eq contributions based on phase (shared across all claims)
                let eq_combined = if in_step_phase {
                    // Phase 1: index i maps to s and x components
                    let s_idx = i % eq_plus_one_len;
                    let x_idx = i / eq_plus_one_len;

                    let eq_plus_one_val = self.eq_plus_one_poly.get_bound_coeff(s_idx);
                    let eq_x_val = if x_idx < eq_x_len {
                        self.eq_x_poly.get_bound_coeff(x_idx)
                    } else {
                        F::zero()
                    };
                    eq_plus_one_val * eq_x_val
                } else {
                    // Phase 2: eq_plus_one is fully bound
                    let eq_plus_one_val = self.eq_plus_one_poly.get_bound_coeff(0);
                    let eq_x_val = self.eq_x_poly.get_bound_coeff(i);
                    eq_plus_one_val * eq_x_val
                };

                // First compute the batched rho sum at index i
                let mut rho_sum = F::zero();
                let mut gamma_power = F::one();
                for claim_idx in 0..self.params.num_claims {
                    let rho_val = self.rho_polys[claim_idx].get_bound_coeff(i);
                    rho_sum += gamma_power * rho_val;
                    gamma_power *= self.gamma;
                }

                // Then multiply by eq polynomials
                let term = eq_combined * rho_sum;

                // Even indices contribute to s(0), odd indices to s(1)
                if i % 2 == 0 {
                    s_0 += term;
                } else {
                    s_1 += term;
                }
            }

            let sum = s_0 + s_1;
            if sum != previous_claim {
                eprintln!("ShiftRho round {}: s(0) + s(1) != previous_claim!", _round);
                eprintln!("  rho_len = {}", self.rho_polys[0].len());
                eprintln!("  eq_plus_one_len = {}", eq_plus_one_len);
                eprintln!("  eq_x_len = {}", eq_x_len);
                eprintln!("  half = {}", half);
                eprintln!("  full_len = {}", full_len);
                eprintln!("  in_step_phase = {}", in_step_phase);
                eprintln!("  round = {}", _round);
                eprintln!("  s(0) = {:?}", s_0);
                eprintln!("  s(1) = {:?}", s_1);
                eprintln!("  s(0) + s(1) = {:?}", sum);
                eprintln!("  previous_claim = {:?}", previous_claim);
                panic!("Sumcheck relation violated!");
            }
        }

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

        // Debug: compute expected output claim to verify consistency with verifier
        #[cfg(test)]
        {
            // After all rounds, compute what the expected output should be
            // eq_plus_one and eq_x should be fully bound now (shared across all claims)
            let eq_plus_one_eval = self.eq_plus_one_poly.get_bound_coeff(0);
            let eq_x_eval = self.eq_x_poly.get_bound_coeff(0);
            let eq_product = eq_plus_one_eval * eq_x_eval;

            // Compute batched rho sum
            let mut rho_sum = F::zero();
            let mut gamma_power = F::one();
            for idx in 0..self.params.num_claims {
                let rho_eval = self.rho_polys[idx].get_bound_coeff(0);
                rho_sum += gamma_power * rho_eval;

                eprintln!(
                    "Prover final claim[{}]: rho={:?}, gamma_power={:?}",
                    idx, rho_eval, gamma_power
                );

                gamma_power *= self.gamma;
            }

            let expected_output = eq_product * rho_sum;
            eprintln!(
                "Prover: eq_plus_one={:?}, eq_x={:?}, eq_product={:?}",
                eq_plus_one_eval, eq_x_eval, eq_product
            );
            eprintln!("Prover: rho_sum={:?}", rho_sum);
            eprintln!(
                "Prover computed expected output claim = {:?}",
                expected_output
            );
            eprintln!("Prover cached rho evaluations at shift point");
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
    pub claims: Vec<ShiftClaim>,
    pub gamma: F,
}

impl<F: JoltField> ShiftRhoVerifier<F> {
    pub fn new<T: Transcript>(
        params: ShiftRhoParams,
        claims: Vec<ShiftClaim>,
        transcript: &mut T,
    ) -> Self {
        assert_eq!(params.num_claims, claims.len());

        // Sample same batching coefficient
        let gamma: F = transcript.challenge_scalar_optimized::<F>().into();

        Self {
            params,
            claims,
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

        for claim in &self.claims {
            let (_, value) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::PackedGtExpRhoNext(claim.constraint_idx),
                SumcheckId::PackedGtExp,
            );

            sum += gamma_power * value;
            gamma_power *= self.gamma;
        }

        #[cfg(debug_assertions)]
        eprintln!("Shift verifier: input_claim sum = {:?}", sum);

        sum
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        #[cfg(debug_assertions)]
        {
            eprintln!(
                "Shift verifier: computing expected output with {} claims",
                self.claims.len()
            );
        }

        // Get the shared point from the first claim (all claims share the same point)
        let (rho_next_point, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::PackedGtExpRhoNext(self.claims[0].constraint_idx),
            SumcheckId::PackedGtExp,
        );

        // Split the original point into step and element parts
        let r_s = &rho_next_point.r[..7];
        let r_x = &rho_next_point.r[7..];

        // The sumcheck challenges are the shift point
        let s_challenges = &sumcheck_challenges[..7];
        let x_challenges = &sumcheck_challenges[7..];

        // Compute eq polynomials once (shared across all claims).
        // r_s/r_x and sumcheck challenges are both in LSB-first order.
        let eq_plus_one = EqPlusOnePolynomial::<F>::mle(
            &r_s.to_vec(),
            &s_challenges.to_vec(),
        );
        let eq_x = EqPolynomial::<F>::mle(&r_x.to_vec(), &x_challenges.to_vec());
        let eq_product = eq_plus_one * eq_x;

        #[cfg(debug_assertions)]
        {
            eprintln!("Verifier: eq_plus_one = {:?}", eq_plus_one);
            eprintln!("Verifier: eq_x = {:?}", eq_x);
            eprintln!("Verifier: eq_product = {:?}", eq_product);
        }

        // Compute batched rho sum
        let mut rho_sum = F::zero();
        let mut gamma_power = F::one();

        for claim in &self.claims {
            // Get rho evaluation at the shift sumcheck challenge point
            let (_, rho_eval) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::PackedGtExpRho(claim.constraint_idx),
                SumcheckId::ShiftRho, // ShiftRho sumcheck ID!
            );

            #[cfg(debug_assertions)]
            eprintln!(
                "Shift verifier fetching rho[{}] = {:?} from accumulator",
                claim.constraint_idx, rho_eval
            );

            rho_sum += gamma_power * rho_eval;
            gamma_power *= self.gamma;
        }

        #[cfg(debug_assertions)]
        eprintln!("Verifier: rho_sum = {:?}", rho_sum);

        let sum = eq_product * rho_sum;

        #[cfg(debug_assertions)]
        eprintln!("Shift verifier: expected output sum = {:?}", sum);

        sum
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        // Imports already at the top of the file

        // The verifier adds expected openings for rho polynomials
        // at the shift sumcheck challenge point
        let opening_point = OpeningPoint::<{ BIG_ENDIAN }, F>::new(sumcheck_challenges.to_vec());

        for claim in &self.claims {
            let polynomials = vec![VirtualPolynomial::PackedGtExpRho(claim.constraint_idx)];
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
