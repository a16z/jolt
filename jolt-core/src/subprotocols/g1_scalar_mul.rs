//! G1 scalar multiplication sumcheck for proving G1 scalar multiplication constraints
//! Proves: 0 = Σ_x eq(r_x, x) * Σ_i γ^i * (Σ_j δ^j * C_{i,j}(x))
//! Where C_{i,j} are the 4 constraints (C1-C4) for each scalar multiplication instance
//!
//! This follows the same pattern as square_and_multiply.rs but with two-level batching:
//! - Delta (δ) batches the 4 constraints within each scalar multiplication
//! - Gamma (γ) batches multiple scalar multiplication instances

use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
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
    zkvm::witness::VirtualPolynomial,
};
use ark_bn254::Fq;
use ark_ff::{One, Zero};
use rayon::prelude::*;

/// Helper to append all virtual claims for a G1 scalar mul constraint
fn append_g1_scalar_mul_virtual_claims<T: Transcript>(
    accumulator: &mut ProverOpeningAccumulator<Fq>,
    transcript: &mut T,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
    opening_point: &OpeningPoint<BIG_ENDIAN, Fq>,
    x_a_claim: Fq,
    y_a_claim: Fq,
    x_t_claim: Fq,
    y_t_claim: Fq,
    x_a_next_claim: Fq,
    y_a_next_claim: Fq,
    t_is_infinity_claim: Fq,
) {
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionG1ScalarMulXA(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
        x_a_claim,
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionG1ScalarMulYA(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
        y_a_claim,
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionG1ScalarMulXT(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
        x_t_claim,
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionG1ScalarMulYT(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
        y_t_claim,
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionG1ScalarMulXANext(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
        x_a_next_claim,
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionG1ScalarMulYANext(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
        y_a_next_claim,
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionG1ScalarMulIndicator(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
        t_is_infinity_claim,
    );
}

/// Helper to retrieve all virtual claims for a G1 scalar mul constraint
fn get_g1_scalar_mul_virtual_claims(
    accumulator: &VerifierOpeningAccumulator<Fq>,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
) -> (Fq, Fq, Fq, Fq, Fq, Fq, Fq) {
    let (_, x_a_claim) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RecursionG1ScalarMulXA(constraint_idx),
        sumcheck_id,
    );
    let (_, y_a_claim) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RecursionG1ScalarMulYA(constraint_idx),
        sumcheck_id,
    );
    let (_, x_t_claim) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RecursionG1ScalarMulXT(constraint_idx),
        sumcheck_id,
    );
    let (_, y_t_claim) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RecursionG1ScalarMulYT(constraint_idx),
        sumcheck_id,
    );
    let (_, x_a_next_claim) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RecursionG1ScalarMulXANext(constraint_idx),
        sumcheck_id,
    );
    let (_, y_a_next_claim) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RecursionG1ScalarMulYANext(constraint_idx),
        sumcheck_id,
    );
    let (_, t_is_infinity_claim) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RecursionG1ScalarMulIndicator(constraint_idx),
        sumcheck_id,
    );

    (
        x_a_claim,
        y_a_claim,
        x_t_claim,
        y_t_claim,
        x_a_next_claim,
        y_a_next_claim,
        t_is_infinity_claim,
    )
}

/// Helper to append virtual opening points for a G1 scalar mul constraint (verifier side)
fn append_g1_scalar_mul_virtual_openings<T: Transcript>(
    accumulator: &mut VerifierOpeningAccumulator<Fq>,
    transcript: &mut T,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
    opening_point: &OpeningPoint<BIG_ENDIAN, Fq>,
) {
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionG1ScalarMulXA(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionG1ScalarMulYA(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionG1ScalarMulXT(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionG1ScalarMulYT(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionG1ScalarMulXANext(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionG1ScalarMulYANext(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
    );
    accumulator.append_virtual(
        transcript,
        VirtualPolynomial::RecursionG1ScalarMulIndicator(constraint_idx),
        sumcheck_id,
        opening_point.clone(),
    );
}

// Helper functions for computing constraints
fn compute_c1(x_a: Fq, y_a: Fq, x_t: Fq) -> Fq {
    let four = Fq::from(4u64);
    let two = Fq::from(2u64);
    let nine = Fq::from(9u64);

    let y_a_sq = y_a * y_a;
    let x_a_sq = x_a * x_a;
    let x_a_fourth = x_a_sq * x_a_sq;

    four * y_a_sq * (x_t + two * x_a) - nine * x_a_fourth
}

fn compute_c2(x_a: Fq, y_a: Fq, x_t: Fq, y_t: Fq) -> Fq {
    let three = Fq::from(3u64);
    let two = Fq::from(2u64);

    let x_a_sq = x_a * x_a;
    three * x_a_sq * (x_t - x_a) + two * y_a * (y_t + y_a)
}

/// C3: Unified constraint handling both finite and infinity cases
/// - When T ≠ O (ind = 0): checks chord formula for addition
/// - When T = O (ind = 1): checks x_a_next ∈ {0, x_p}
fn compute_c3(ind: Fq, x_a_next: Fq, x_t: Fq, y_t: Fq, x_p: Fq, y_p: Fq) -> Fq {
    // Infinity case: x_a_next must be 0 (stayed at O) or x_p (added P)
    // Constraint: x_a_next * (x_a_next - x_p) = 0
    let c3_infinity = x_a_next * (x_a_next - x_p);

    // Finite case: chord addition formula
    let x_diff = x_p - x_t;
    let y_diff = y_p - y_t;
    let x_a_diff = x_a_next - x_t;
    let c3_finite = x_a_diff * ((x_a_next + x_t + x_p) * x_diff * x_diff - y_diff * y_diff);

    // Combined: ind * infinity_case + (1 - ind) * finite_case
    ind * c3_infinity + (Fq::one() - ind) * c3_finite
}

/// C4: Unified constraint handling both finite and infinity cases
fn compute_c4(ind: Fq, x_a_next: Fq, y_a_next: Fq, x_t: Fq, y_t: Fq, x_p: Fq, y_p: Fq) -> Fq {
    // Infinity case: y_a_next must be 0 (stayed at O) or y_p (added P)
    let c4_infinity = y_a_next * (y_a_next - y_p);

    // Finite case: chord addition formula
    let y_a_diff = y_a_next - y_t;
    let c4_finite =
        y_a_diff * (x_t * (y_p + y_a_next) - x_p * (y_t + y_a_next) + x_a_next * (y_t - y_p));

    // Combined
    ind * c4_infinity + (Fq::one() - ind) * c4_finite
}

/// Individual polynomial data for a single G1 scalar multiplication constraint
#[derive(Clone)]
pub struct G1ScalarMulConstraintPolynomials {
    pub x_a: Vec<Fq>,            // x-coords of accumulator (all 256 steps)
    pub y_a: Vec<Fq>,            // y-coords of accumulator (all 256 steps)
    pub x_t: Vec<Fq>,            // x-coords of doubled point (all 256 steps)
    pub y_t: Vec<Fq>,            // y-coords of doubled point (all 256 steps)
    pub x_a_next: Vec<Fq>,       // x-coords of A_{i+1} (shifted by 1)
    pub y_a_next: Vec<Fq>,       // y-coords of A_{i+1} (shifted by 1)
    pub t_is_infinity: Vec<Fq>,  // Indicator polynomial (1 if T = O, 0 otherwise)
    pub base_point: (Fq, Fq),    // Base point coordinates
    pub constraint_index: usize, // Global constraint index
}

/// Parameters for G1 scalar multiplication sumcheck
#[derive(Clone)]
pub struct G1ScalarMulParams {
    /// Number of constraint variables (x) - 8 for 256-bit scalars
    pub num_constraint_vars: usize,

    /// Number of G1 scalar multiplication instances
    pub num_constraints: usize,

    /// Sumcheck instance identifier
    pub sumcheck_id: SumcheckId,
}

impl G1ScalarMulParams {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            num_constraint_vars: 8, // Fixed for 256-bit scalars
            num_constraints,
            sumcheck_id: SumcheckId::G1ScalarMul,
        }
    }
}

/// Prover for G1 scalar multiplication sumcheck
pub struct G1ScalarMulProver {
    /// Parameters
    pub params: G1ScalarMulParams,

    /// Base points for each scalar multiplication instance
    pub base_points: Vec<(Fq, Fq)>,

    /// Global constraint indices for each constraint
    pub constraint_indices: Vec<usize>,

    /// Equality polynomial for constraint variables x
    pub eq_x: MultilinearPolynomial<Fq>,

    /// Random challenge for eq(r_x, x)
    pub r_x: Vec<<Fq as JoltField>::Challenge>,

    /// Gamma coefficient for batching scalar multiplication instances
    pub gamma: Fq,

    /// Delta coefficient for batching 4 constraints within each instance
    pub delta: Fq,

    /// x_a polynomials as multilinear (one per instance, contains all steps)
    pub x_a_mlpoly: Vec<MultilinearPolynomial<Fq>>,

    /// y_a polynomials as multilinear (one per instance, contains all steps)
    pub y_a_mlpoly: Vec<MultilinearPolynomial<Fq>>,

    /// x_t polynomials as multilinear (one per instance, contains all steps)
    pub x_t_mlpoly: Vec<MultilinearPolynomial<Fq>>,

    /// y_t polynomials as multilinear (one per instance, contains all steps)
    pub y_t_mlpoly: Vec<MultilinearPolynomial<Fq>>,

    /// x_a_next polynomials as multilinear (shifted A_{i+1} values)
    pub x_a_next_mlpoly: Vec<MultilinearPolynomial<Fq>>,

    /// y_a_next polynomials as multilinear (shifted A_{i+1} values)
    pub y_a_next_mlpoly: Vec<MultilinearPolynomial<Fq>>,

    /// Infinity indicator polynomials (1 if T = O, 0 otherwise)
    pub t_is_infinity_mlpoly: Vec<MultilinearPolynomial<Fq>>,

    /// Individual claims for each constraint (not batched)
    pub x_a_claims: Vec<Fq>,
    pub y_a_claims: Vec<Fq>,
    pub x_t_claims: Vec<Fq>,
    pub y_t_claims: Vec<Fq>,
    pub x_a_next_claims: Vec<Fq>,
    pub y_a_next_claims: Vec<Fq>,
    pub t_is_infinity_claims: Vec<Fq>,

    /// Current round
    pub round: usize,
}

impl G1ScalarMulProver {
    pub fn new<T: Transcript>(
        params: G1ScalarMulParams,
        constraint_polys: Vec<G1ScalarMulConstraintPolynomials>,
        transcript: &mut T,
    ) -> Self {
        let r_x: Vec<<Fq as JoltField>::Challenge> = (0..params.num_constraint_vars)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();

        let gamma = transcript.challenge_scalar_optimized::<Fq>();
        let delta = transcript.challenge_scalar_optimized::<Fq>();

        let eq_x = MultilinearPolynomial::from(EqPolynomial::<Fq>::evals(&r_x));
        let mut base_points = Vec::new();
        let mut constraint_indices = Vec::new();
        let mut x_a_mlpoly = Vec::new();
        let mut y_a_mlpoly = Vec::new();
        let mut x_t_mlpoly = Vec::new();
        let mut y_t_mlpoly = Vec::new();
        let mut x_a_next_mlpoly = Vec::new();
        let mut y_a_next_mlpoly = Vec::new();
        let mut t_is_infinity_mlpoly = Vec::new();

        for poly in constraint_polys {
            base_points.push(poly.base_point);
            constraint_indices.push(poly.constraint_index);
            x_a_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                poly.x_a,
            )));
            y_a_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                poly.y_a,
            )));
            x_t_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                poly.x_t,
            )));
            y_t_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                poly.y_t,
            )));
            x_a_next_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                poly.x_a_next,
            )));
            y_a_next_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                poly.y_a_next,
            )));
            t_is_infinity_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                poly.t_is_infinity,
            )));
        }

        let prover = Self {
            params,
            base_points,
            constraint_indices,
            eq_x,
            r_x,
            gamma: gamma.into(),
            delta: delta.into(),
            x_a_mlpoly,
            y_a_mlpoly,
            x_t_mlpoly,
            y_t_mlpoly,
            x_a_next_mlpoly,
            y_a_next_mlpoly,
            t_is_infinity_mlpoly,
            x_a_claims: vec![],
            y_a_claims: vec![],
            x_t_claims: vec![],
            y_t_claims: vec![],
            x_a_next_claims: vec![],
            y_a_next_claims: vec![],
            t_is_infinity_claims: vec![],
            round: 0,
        };

        prover
    }
}

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for G1ScalarMulProver {
    fn degree(&self) -> usize {
        6 // Was 5, now 6 due to indicator multiplication in C3/C4
    }

    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }

    #[tracing::instrument(skip_all, name = "G1ScalarMul::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        const DEGREE: usize = 6; // INCREASED from 5 due to ind multiplication
        let num_x_remaining = self.eq_x.get_num_vars();
        let x_half = 1 << (num_x_remaining - 1);

        let total_evals = (0..x_half)
            .into_par_iter()
            .map(|x_idx| {
                let eq_x_evals = self
                    .eq_x
                    .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);

                let mut x_evals = [Fq::zero(); DEGREE];
                let mut gamma_power = self.gamma;

                // For each G1 scalar multiplication instance
                for i in 0..self.params.num_constraints {
                    let x_a_evals_hint = self.x_a_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let y_a_evals_hint = self.y_a_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let x_t_evals_hint = self.x_t_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let y_t_evals_hint = self.y_t_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);

                    // For A_{i+1}, use the pre-computed shifted MLEs
                    let x_a_next_evals_hint = self.x_a_next_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let y_a_next_evals_hint = self.y_a_next_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);

                    // NEW: get indicator evals
                    let ind_evals = self.t_is_infinity_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);

                    let (x_p, y_p) = self.base_points[i];

                    for t in 0..DEGREE {
                        let ind = ind_evals[t];
                        let x_a = x_a_evals_hint[t];
                        let y_a = y_a_evals_hint[t];
                        let x_t = x_t_evals_hint[t];
                        let y_t = y_t_evals_hint[t];
                        let x_a_next = x_a_next_evals_hint[t];
                        let y_a_next = y_a_next_evals_hint[t];

                        // C1 and C2 unchanged
                        let c1 = compute_c1(x_a, y_a, x_t);
                        let c2 = compute_c2(x_a, y_a, x_t, y_t);

                        // C3 and C4 now take indicator
                        let c3 = compute_c3(ind, x_a_next, x_t, y_t, x_p, y_p);
                        let c4 = compute_c4(ind, x_a_next, y_a_next, x_t, y_t, x_p, y_p);

                        // Two-level batching:
                        // 1. Use delta to batch the 4 constraints within each step
                        let delta_sq = self.delta * self.delta;
                        let delta_cube = delta_sq * self.delta;
                        let constraint_val = c1 + self.delta * c2 + delta_sq * c3 + delta_cube * c4;

                        x_evals[t] += eq_x_evals[t] * gamma_power * constraint_val;
                    }

                    gamma_power *= self.gamma;
                }
                x_evals
            })
            .reduce(
                || [Fq::zero(); DEGREE],
                |mut acc, evals| {
                    for (a, e) in acc.iter_mut().zip(evals.iter()) {
                        *a += *e;
                    }
                    acc
                },
            );

        let uni_poly = UniPoly::from_evals_and_hint(previous_claim, &total_evals);

        uni_poly
    }

    #[tracing::instrument(skip_all, name = "G1ScalarMul::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: <Fq as JoltField>::Challenge, round: usize) {
        self.eq_x.bind_parallel(r_j, BindingOrder::LowToHigh);

        for poly in &mut self.x_a_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.y_a_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.x_t_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.y_t_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.x_a_next_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.y_a_next_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.t_is_infinity_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }

        self.round = round + 1;

        if self.round == self.params.num_constraint_vars {
            self.x_a_claims.clear();
            self.y_a_claims.clear();
            self.x_t_claims.clear();
            self.y_t_claims.clear();
            self.x_a_next_claims.clear();
            self.y_a_next_claims.clear();
            self.t_is_infinity_claims.clear();

            for i in 0..self.params.num_constraints {
                self.x_a_claims.push(self.x_a_mlpoly[i].get_bound_coeff(0));
                self.y_a_claims.push(self.y_a_mlpoly[i].get_bound_coeff(0));
                self.x_t_claims.push(self.x_t_mlpoly[i].get_bound_coeff(0));
                self.y_t_claims.push(self.y_t_mlpoly[i].get_bound_coeff(0));
                self.x_a_next_claims
                    .push(self.x_a_next_mlpoly[i].get_bound_coeff(0));
                self.y_a_next_claims
                    .push(self.y_a_next_mlpoly[i].get_bound_coeff(0));
                self.t_is_infinity_claims
                    .push(self.t_is_infinity_mlpoly[i].get_bound_coeff(0));
            }
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(sumcheck_challenges.to_vec());

        #[cfg(test)]
        {
            // Compute expected output claim on prover side
            use crate::poly::eq_poly::EqPolynomial;

            let r_x_fq: Vec<Fq> = self.r_x.iter().map(|c| (*c).into()).collect();
            let r_star_fq: Vec<Fq> = sumcheck_challenges
                .iter()
                .rev()
                .map(|c| (*c).into())
                .collect();
            let eq_eval = EqPolynomial::mle(&r_x_fq, &r_star_fq);

            let mut total = Fq::zero();
            let mut gamma_power = self.gamma;

            for i in 0..self.params.num_constraints {
                let (x_p, y_p) = self.base_points[i];

                // Compute all 4 constraints using final claims
                let ind = self.t_is_infinity_claims[i];
                let c1 = compute_c1(self.x_a_claims[i], self.y_a_claims[i], self.x_t_claims[i]);
                let c2 = compute_c2(
                    self.x_a_claims[i],
                    self.y_a_claims[i],
                    self.x_t_claims[i],
                    self.y_t_claims[i],
                );
                let c3 = compute_c3(
                    ind,
                    self.x_a_next_claims[i],
                    self.x_t_claims[i],
                    self.y_t_claims[i],
                    x_p,
                    y_p,
                );
                let c4 = compute_c4(
                    ind,
                    self.x_a_next_claims[i],
                    self.y_a_next_claims[i],
                    self.x_t_claims[i],
                    self.y_t_claims[i],
                    x_p,
                    y_p,
                );

                let delta_sq = self.delta * self.delta;
                let delta_cube = delta_sq * self.delta;
                let constraint_value = c1 + self.delta * c2 + delta_sq * c3 + delta_cube * c4;

                total += gamma_power * constraint_value;
                gamma_power *= self.gamma;
            }

            let prover_expected = eq_eval * total;
            eprintln!("=== G1 PROVER EXPECTED OUTPUT CLAIM ===");
            eprintln!("eq_eval: {:?}", eq_eval);
            eprintln!("total: {:?}", total);
            eprintln!("Expected claim: {:?}", prover_expected);
        }

        for i in 0..self.params.num_constraints {
            #[cfg(test)]
            {
                eprintln!("=== G1 Prover cache_openings ===");
                eprintln!("Constraint {}: storing claims", i);
                eprintln!("  x_a_claim: {:?}", self.x_a_claims[i]);
                eprintln!("  y_a_claim: {:?}", self.y_a_claims[i]);
                eprintln!("  x_t_claim: {:?}", self.x_t_claims[i]);
                eprintln!("  y_t_claim: {:?}", self.y_t_claims[i]);
                eprintln!("  x_a_next_claim: {:?}", self.x_a_next_claims[i]);
                eprintln!("  y_a_next_claim: {:?}", self.y_a_next_claims[i]);
                eprintln!("  t_is_infinity_claim: {:?}", self.t_is_infinity_claims[i]);
            }

            append_g1_scalar_mul_virtual_claims(
                accumulator,
                transcript,
                self.constraint_indices[i], // Use global constraint index
                self.params.sumcheck_id,
                &opening_point,
                self.x_a_claims[i],
                self.y_a_claims[i],
                self.x_t_claims[i],
                self.y_t_claims[i],
                self.x_a_next_claims[i],
                self.y_a_next_claims[i],
                self.t_is_infinity_claims[i],
            );
        }
    }
}

/// Verifier for G1 scalar multiplication sumcheck
pub struct G1ScalarMulVerifier {
    pub params: G1ScalarMulParams,
    pub r_x: Vec<<Fq as JoltField>::Challenge>,
    pub gamma: Fq,
    pub delta: Fq,
    pub num_constraints: usize,
    pub base_points: Vec<(Fq, Fq)>,
    pub constraint_indices: Vec<usize>,
}

impl G1ScalarMulVerifier {
    pub fn new<T: Transcript>(
        params: G1ScalarMulParams,
        base_points: Vec<(Fq, Fq)>,
        constraint_indices: Vec<usize>,
        transcript: &mut T,
    ) -> Self {
        let r_x: Vec<<Fq as JoltField>::Challenge> = (0..params.num_constraint_vars)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();

        let gamma = transcript.challenge_scalar_optimized::<Fq>();
        let delta = transcript.challenge_scalar_optimized::<Fq>();
        let num_constraints = params.num_constraints;

        Self {
            params,
            r_x,
            gamma: gamma.into(),
            delta: delta.into(),
            num_constraints,
            base_points,
            constraint_indices,
        }
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for G1ScalarMulVerifier {
    fn degree(&self) -> usize {
        6
    }

    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars
    }

    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<Fq>,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        use crate::poly::eq_poly::EqPolynomial;

        let r_x_fq: Vec<Fq> = self.r_x.iter().map(|c| (*c).into()).collect();
        let r_star_fq: Vec<Fq> = sumcheck_challenges
            .iter()
            .rev()
            .map(|c| (*c).into())
            .collect();
        let eq_eval = EqPolynomial::mle(&r_x_fq, &r_star_fq);

        let mut total = Fq::zero();
        let mut gamma_power = self.gamma;

        for i in 0..self.num_constraints {
            let (
                x_a_claim,
                y_a_claim,
                x_t_claim,
                y_t_claim,
                x_a_next_claim,
                y_a_next_claim,
                t_is_infinity_claim,
            ) = get_g1_scalar_mul_virtual_claims(
                accumulator,
                self.constraint_indices[i],
                self.params.sumcheck_id,
            );

            let (x_p, y_p) = self.base_points[i];

            // Compute all 4 constraints
            let c1 = compute_c1(x_a_claim, y_a_claim, x_t_claim);
            let c2 = compute_c2(x_a_claim, y_a_claim, x_t_claim, y_t_claim);
            let c3 = compute_c3(
                t_is_infinity_claim,
                x_a_next_claim,
                x_t_claim,
                y_t_claim,
                x_p,
                y_p,
            );
            let c4 = compute_c4(
                t_is_infinity_claim,
                x_a_next_claim,
                y_a_next_claim,
                x_t_claim,
                y_t_claim,
                x_p,
                y_p,
            );

            // Two-level batching
            let delta_sq = self.delta * self.delta;
            let delta_cube = delta_sq * self.delta;
            let constraint_value = c1 + self.delta * c2 + delta_sq * c3 + delta_cube * c4;

            total += gamma_power * constraint_value;
            gamma_power *= self.gamma;
        }

        let verifier_expected = eq_eval * total;

        verifier_expected
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(sumcheck_challenges.to_vec());

        for i in 0..self.num_constraints {
            append_g1_scalar_mul_virtual_openings(
                accumulator,
                transcript,
                self.constraint_indices[i],
                self.params.sumcheck_id,
                &opening_point,
            );
        }
    }
}
