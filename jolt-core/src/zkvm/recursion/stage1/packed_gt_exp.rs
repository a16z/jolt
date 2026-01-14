//! Packed GT exponentiation sumcheck with 2-phase protocol
//!
//! This implements the optimized GT exponentiation verification that packs all 254 steps
//! into larger MLEs, reducing claims from ~1,016 to 5 per GT exp.
//!
//! Polynomial structure:
//! - rho(s, x): 12-var (8 step + 4 element) - intermediate results
//! - rho_next(s, x): 12-var - shifted: rho_next(i, x) = rho(i+1, x)
//! - quotient(s, x): 12-var - quotient polynomials
//! - bit(s): 8-var padded to 12-var - scalar bits
//! - base(x): 4-var padded to 12-var - base element
//!
//! Constraint: C(s, x) = rho_next(s, x) - rho(s, x)² × base(x)^{bit(s)} - quotient(s, x) × g(x)
//!
//! 2-phase sumcheck:
//! - Phase 1: 8 rounds over step variables (s)
//! - Phase 2: 4 rounds over element variables (x)

use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::{recursion::utils::virtual_polynomial_utils::*, witness::VirtualPolynomial},
    virtual_claims,
};
use ark_bn254::Fq;
use ark_ff::Zero;
use rayon::prelude::*;

/// Number of step variables (8 for 256 steps, sufficient for 254 scalar bits)
pub const NUM_STEP_VARS: usize = 8;

/// Number of element variables (4 for Fq12 = 12 field elements, but we use 16 for power of 2)
pub const NUM_ELEMENT_VARS: usize = 4;

/// Total variables = step + element
pub const NUM_TOTAL_VARS: usize = NUM_STEP_VARS + NUM_ELEMENT_VARS;

/// Packed witness for GT exponentiation
#[derive(Clone)]
pub struct PackedGtExpWitness {
    /// rho(s, x) - all intermediate results packed into 12-var MLE
    /// Layout: rho[s * 16 + x] = ρ_s[x]
    pub rho_packed: Vec<Fq>,

    /// rho_next(s, x) = rho(s+1, x) - shifted intermediate results
    /// Layout: rho_next[s * 16 + x] = ρ_{s+1}[x]
    pub rho_next_packed: Vec<Fq>,

    /// quotient(s, x) - all quotients packed into 12-var MLE
    /// Layout: quotient[s * 16 + x] = Q_s[x]
    pub quotient_packed: Vec<Fq>,

    /// bit(s) - scalar bits as 8-var MLE (padded to 12-var)
    /// Layout: bit[s] = b_s (0 or 1)
    pub bit_packed: Vec<Fq>,

    /// base(x) - base element as 4-var MLE (padded to 12-var)
    pub base_packed: Vec<Fq>,

    /// Number of actual steps (254 for BN254 scalar)
    pub num_steps: usize,
}

impl PackedGtExpWitness {
    /// Create packed witness from individual step data
    pub fn from_steps(
        rho_mles: &[Vec<Fq>],        // rho_mles[step][x] for step in 0..=num_steps
        quotient_mles: &[Vec<Fq>],   // quotient_mles[step][x] for step in 0..num_steps
        bits: &[bool],               // bits[step] for step in 0..num_steps
        base_mle: &[Fq],             // base[x] - 16 values
    ) -> Self {
        let num_steps = bits.len();
        assert_eq!(rho_mles.len(), num_steps + 1, "Need num_steps + 1 rho MLEs");
        assert_eq!(quotient_mles.len(), num_steps, "Need num_steps quotient MLEs");
        assert_eq!(base_mle.len(), 16, "Base must be 4-var MLE (16 values)");

        let step_size = 1 << NUM_STEP_VARS;   // 256
        let elem_size = 1 << NUM_ELEMENT_VARS; // 16
        let total_size = 1 << NUM_TOTAL_VARS;  // 4096

        // Pack rho: rho_packed[s * 16 + x] = rho_mles[s][x]
        let mut rho_packed = vec![Fq::zero(); total_size];
        for s in 0..=num_steps.min(step_size - 1) {
            for x in 0..elem_size {
                if s < rho_mles.len() && x < rho_mles[s].len() {
                    rho_packed[s * elem_size + x] = rho_mles[s][x];
                }
            }
        }

        // Pack rho_next: rho_next_packed[s * 16 + x] = rho_mles[s+1][x]
        let mut rho_next_packed = vec![Fq::zero(); total_size];
        for s in 0..num_steps.min(step_size) {
            for x in 0..elem_size {
                if s + 1 < rho_mles.len() && x < rho_mles[s + 1].len() {
                    rho_next_packed[s * elem_size + x] = rho_mles[s + 1][x];
                }
            }
        }

        // Pack quotient: quotient_packed[s * 16 + x] = quotient_mles[s][x]
        let mut quotient_packed = vec![Fq::zero(); total_size];
        for s in 0..num_steps.min(step_size) {
            for x in 0..elem_size {
                if s < quotient_mles.len() && x < quotient_mles[s].len() {
                    quotient_packed[s * elem_size + x] = quotient_mles[s][x];
                }
            }
        }

        // Pack bits: bit_packed[s * 16 + x] = bits[s] (replicated across x)
        // Note: We replicate so that bit_packed(r_s, r_x) = bit(r_s) for any r_x
        let mut bit_packed = vec![Fq::zero(); total_size];
        for s in 0..num_steps.min(step_size) {
            let bit_val = if bits[s] { Fq::from(1u64) } else { Fq::zero() };
            for x in 0..elem_size {
                bit_packed[s * elem_size + x] = bit_val;
            }
        }

        // Pack base: base_packed[s * 16 + x] = base_mle[x] (replicated across s)
        let mut base_packed = vec![Fq::zero(); total_size];
        for s in 0..step_size {
            for x in 0..elem_size {
                base_packed[s * elem_size + x] = base_mle[x];
            }
        }

        Self {
            rho_packed,
            rho_next_packed,
            quotient_packed,
            bit_packed,
            base_packed,
            num_steps,
        }
    }
}

/// Parameters for packed GT exp sumcheck
#[derive(Clone)]
pub struct PackedGtExpParams {
    /// Total number of constraint variables (12 = 8 step + 4 element)
    pub num_constraint_vars: usize,

    /// Number of step variables
    pub num_step_vars: usize,

    /// Number of element variables
    pub num_element_vars: usize,

    /// Sumcheck instance identifier
    pub sumcheck_id: SumcheckId,
}

impl PackedGtExpParams {
    pub fn new() -> Self {
        Self {
            num_constraint_vars: NUM_TOTAL_VARS,
            num_step_vars: NUM_STEP_VARS,
            num_element_vars: NUM_ELEMENT_VARS,
            sumcheck_id: SumcheckId::PackedGtExp,
        }
    }
}

impl Default for PackedGtExpParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Prover for packed GT exponentiation sumcheck
pub struct PackedGtExpProver<F: JoltField> {
    /// Parameters
    pub params: PackedGtExpParams,

    /// Packed rho polynomial
    pub rho_poly: MultilinearPolynomial<F>,

    /// Packed rho_next polynomial
    pub rho_next_poly: MultilinearPolynomial<F>,

    /// Packed quotient polynomial
    pub quotient_poly: MultilinearPolynomial<F>,

    /// Packed bit polynomial
    pub bit_poly: MultilinearPolynomial<F>,

    /// Packed base polynomial
    pub base_poly: MultilinearPolynomial<F>,

    /// g(x) polynomial (padded to 12-var)
    pub g_poly: MultilinearPolynomial<F>,

    /// eq(r_s, s) polynomial for step batching
    pub eq_s: MultilinearPolynomial<F>,

    /// eq(r_x, x) polynomial for element batching
    pub eq_x: MultilinearPolynomial<F>,

    /// Random challenges for eq(r_s, s)
    pub r_s: Vec<F::Challenge>,

    /// Random challenges for eq(r_x, x)
    pub r_x: Vec<F::Challenge>,

    /// Current round (0 to 11)
    pub round: usize,

    /// Final claims after all rounds
    pub rho_claim: F,
    pub rho_next_claim: F,
    pub quotient_claim: F,
    pub bit_claim: F,
    pub base_claim: F,
}

impl<F: JoltField> PackedGtExpProver<F> {
    /// Create a new packed GT exp prover
    pub fn new<T: Transcript>(
        params: PackedGtExpParams,
        witness: &PackedGtExpWitness,
        g_poly: DensePolynomial<F>,
        transcript: &mut T,
    ) -> Self {
        // Sample random challenges for step and element variables
        let r_s: Vec<F::Challenge> = (0..params.num_step_vars)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();

        let r_x: Vec<F::Challenge> = (0..params.num_element_vars)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();

        // For Phase 1, we need separate eq_s (will be bound during Phase 1)
        let eq_s = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&r_s));

        // For Phase 2, eq_x stays constant during Phase 1, then gets bound in Phase 2
        let eq_x = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&r_x));

        // Convert witness to field F (assuming F = Fq)
        let convert_vec = |v: &[Fq]| -> Vec<F> {
            v.iter()
                .map(|fq| unsafe { std::mem::transmute_copy(fq) })
                .collect()
        };

        Self {
            params,
            rho_poly: MultilinearPolynomial::from(convert_vec(&witness.rho_packed)),
            rho_next_poly: MultilinearPolynomial::from(convert_vec(&witness.rho_next_packed)),
            quotient_poly: MultilinearPolynomial::from(convert_vec(&witness.quotient_packed)),
            bit_poly: MultilinearPolynomial::from(convert_vec(&witness.bit_packed)),
            base_poly: MultilinearPolynomial::from(convert_vec(&witness.base_packed)),
            g_poly: MultilinearPolynomial::LargeScalars(g_poly),
            eq_s,
            eq_x,
            r_s,
            r_x,
            round: 0,
            rho_claim: F::zero(),
            rho_next_claim: F::zero(),
            quotient_claim: F::zero(),
            bit_claim: F::zero(),
            base_claim: F::zero(),
        }
    }

    /// Check if we're in Phase 1 (step variable rounds)
    fn in_phase1(&self) -> bool {
        self.round < self.params.num_step_vars
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for PackedGtExpProver<F> {
    fn degree(&self) -> usize {
        // Degree from constraint: rho² × base (degree 3) or rho² × bit × base (degree 4)
        // With eq terms: eq_s × eq_x × constraint = up to degree 4
        4
    }

    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars // 12 rounds total
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        F::zero() // The constraint should sum to zero
    }

    #[tracing::instrument(skip_all, name = "PackedGtExp::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        const DEGREE: usize = 4;

        // The polynomial sizes halve each round
        let half = self.rho_poly.len() / 2;

        let evals = (0..half)
            .into_par_iter()
            .map(|i| {
                // Get evaluations at t=0, t=1, t=2, t=3 for each polynomial
                let rho = self
                    .rho_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let rho_next = self
                    .rho_next_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let quotient = self
                    .quotient_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let bit = self
                    .bit_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let base = self
                    .base_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let g = self
                    .g_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                // Compute eq term based on phase
                // In the unified approach, we bind eq_s in Phase 1 and eq_x in Phase 2
                // But for simplicity, we use a combined eq that covers the full 12-var space
                // Here we just compute the constraint contribution

                // During Phase 1, eq_s gets bound but eq_x is summed over
                // During Phase 2, eq_s is fully bound (scalar) and eq_x gets bound
                let eq_s_evals = if self.in_phase1() {
                    self.eq_s
                        .sumcheck_evals_array::<DEGREE>(i >> self.params.num_element_vars, BindingOrder::LowToHigh)
                } else {
                    // eq_s is fully bound, just use scalar value
                    [self.eq_s.get_bound_coeff(0); DEGREE]
                };

                let eq_x_evals = if self.in_phase1() {
                    // eq_x is constant during Phase 1 - look up the value for this x index
                    let x_idx = i & ((1 << self.params.num_element_vars) - 1);
                    let eq_x_val = if x_idx < self.eq_x.len() {
                        self.eq_x.get_bound_coeff(x_idx)
                    } else {
                        F::zero()
                    };
                    [eq_x_val; DEGREE]
                } else {
                    self.eq_x
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh)
                };

                let mut term_evals = [F::zero(); DEGREE];
                for t in 0..DEGREE {
                    // base^{bit}: linear interpolation between 1 and base
                    // base^bit = 1 + bit * (base - 1) = 1 - bit + bit * base
                    let base_power = F::one() - bit[t] + bit[t] * base[t];

                    // C(s, x) = rho_next - rho² × base_power - quotient × g
                    let constraint = rho_next[t]
                        - rho[t] * rho[t] * base_power
                        - quotient[t] * g[t];

                    term_evals[t] = eq_s_evals[t] * eq_x_evals[t] * constraint;
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

    #[tracing::instrument(skip_all, name = "PackedGtExp::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        // Bind all polynomials
        self.rho_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.rho_next_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.quotient_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.bit_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.base_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.g_poly.bind_parallel(r_j, BindingOrder::LowToHigh);

        // Bind eq polynomials based on phase
        if self.in_phase1() {
            self.eq_s.bind_parallel(r_j, BindingOrder::LowToHigh);
        } else {
            self.eq_x.bind_parallel(r_j, BindingOrder::LowToHigh);
        }

        self.round = round + 1;

        // After all 12 rounds, extract final claims
        if self.round == self.params.num_constraint_vars {
            self.rho_claim = self.rho_poly.get_bound_coeff(0);
            self.rho_next_claim = self.rho_next_poly.get_bound_coeff(0);
            self.quotient_claim = self.quotient_poly.get_bound_coeff(0);
            self.bit_claim = self.bit_poly.get_bound_coeff(0);
            self.base_claim = self.base_poly.get_bound_coeff(0);
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(sumcheck_challenges.to_vec());

        // Cache the 5 polynomial opening claims
        let claims = virtual_claims![
            VirtualPolynomial::PackedGtExpRho => self.rho_claim,
            VirtualPolynomial::PackedGtExpRhoNext => self.rho_next_claim,
            VirtualPolynomial::PackedGtExpQuotient => self.quotient_claim,
            VirtualPolynomial::PackedGtExpBit => self.bit_claim,
            VirtualPolynomial::PackedGtExpBase => self.base_claim,
        ];
        append_virtual_claims(
            accumulator,
            transcript,
            self.params.sumcheck_id,
            &opening_point,
            &claims,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Verifier for packed GT exponentiation sumcheck
pub struct PackedGtExpVerifier<F: JoltField> {
    pub params: PackedGtExpParams,
    pub r_s: Vec<F::Challenge>,
    pub r_x: Vec<F::Challenge>,
}

impl<F: JoltField> PackedGtExpVerifier<F> {
    pub fn new<T: Transcript>(params: PackedGtExpParams, transcript: &mut T) -> Self {
        let r_s: Vec<F::Challenge> = (0..params.num_step_vars)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();

        let r_x: Vec<F::Challenge> = (0..params.num_element_vars)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();

        Self { params, r_s, r_x }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for PackedGtExpVerifier<F> {
    fn degree(&self) -> usize {
        4
    }

    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars
    }

    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        use crate::poly::dense_mlpoly::DensePolynomial;
        use jolt_optimizations::get_g_mle;
        use std::any::TypeId;

        // Runtime check that F = Fq
        if TypeId::of::<F>() != TypeId::of::<Fq>() {
            panic!("PackedGtExp requires F = Fq");
        }

        // Get polynomial claims from accumulator
        let polynomials = vec![
            VirtualPolynomial::PackedGtExpRho,
            VirtualPolynomial::PackedGtExpRhoNext,
            VirtualPolynomial::PackedGtExpQuotient,
            VirtualPolynomial::PackedGtExpBit,
            VirtualPolynomial::PackedGtExpBase,
        ];
        let claims = get_virtual_claims(accumulator, self.params.sumcheck_id, &polynomials);
        let rho_claim = claims[0];
        let rho_next_claim = claims[1];
        let quotient_claim = claims[2];
        let bit_claim = claims[3];
        let base_claim = claims[4];

        // Compute g(r_x*) - need to evaluate g at the element part of challenges
        let r_x_star: Vec<F> = sumcheck_challenges
            .iter()
            .skip(self.params.num_step_vars)
            .map(|c| (*c).into())
            .collect();

        let g_eval: F = {
            let g_mle_4var = get_g_mle();

            // Create 4-var polynomial and evaluate at r_x_star (4 vars)
            let g_poly_fq =
                MultilinearPolynomial::<Fq>::LargeScalars(DensePolynomial::new(g_mle_4var));
            let r_x_star_fq: &Vec<Fq> = unsafe { std::mem::transmute(&r_x_star) };
            let g_eval_fq = g_poly_fq.evaluate_dot_product(r_x_star_fq);
            unsafe { std::mem::transmute_copy(&g_eval_fq) }
        };

        // Compute eq evaluations
        let r_s_f: Vec<F> = self.r_s.iter().map(|c| (*c).into()).collect();
        let r_x_f: Vec<F> = self.r_x.iter().map(|c| (*c).into()).collect();

        let r_s_star: Vec<F> = sumcheck_challenges
            .iter()
            .take(self.params.num_step_vars)
            .rev()
            .map(|c| (*c).into())
            .collect();

        let r_x_star_rev: Vec<F> = sumcheck_challenges
            .iter()
            .skip(self.params.num_step_vars)
            .rev()
            .map(|c| (*c).into())
            .collect();

        let eq_s_eval = EqPolynomial::mle(&r_s_f, &r_s_star);
        let eq_x_eval = EqPolynomial::mle(&r_x_f, &r_x_star_rev);

        // base^{bit} using linear interpolation
        let base_power = F::one() - bit_claim + bit_claim * base_claim;

        // Constraint at challenge point:
        // C(r_s*, r_x*) = rho_next - rho² × base_power - quotient × g
        let constraint_eval =
            rho_next_claim - rho_claim * rho_claim * base_power - quotient_claim * g_eval;

        // Expected output = eq_s(r_s, r_s*) × eq_x(r_x, r_x*) × C(r_s*, r_x*)
        eq_s_eval * eq_x_eval * constraint_eval
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(sumcheck_challenges.to_vec());

        let polynomials = vec![
            VirtualPolynomial::PackedGtExpRho,
            VirtualPolynomial::PackedGtExpRhoNext,
            VirtualPolynomial::PackedGtExpQuotient,
            VirtualPolynomial::PackedGtExpBit,
            VirtualPolynomial::PackedGtExpBase,
        ];
        append_virtual_openings(
            accumulator,
            transcript,
            self.params.sumcheck_id,
            &opening_point,
            &polynomials,
        );
    }
}
