//! Two-phase sumcheck for proving that Dory hints are well-formed
//! Proves: 0 = Σ_{i,x} eq(r_i, i) * eq(r_x, x) * C_i(x)
//! Where C_i(x) = ρ_curr(x) - ρ_prev(x)² × a(x)^{b_i} - Q_i(x) × g(x)

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
            VerifierOpeningAccumulator,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        recursion_constraints::{
            compute_constraint_formula, index_to_binary, MatrixConstraint, RowOffset,
        },
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::witness::CommittedPolynomial,
};
use ark_bn254::Fq;
use ark_ff::{One, Zero};
use rayon::prelude::*;

/// Parameters shared between prover and verifier
///
/// M has structure: M(offset_bits, constraint_index_bits, x_bits)
/// - Phase 1 binds x_bits (4 rounds)
/// - Phase 2 binds constraint_index_bits (num_constraint_index_vars rounds)
/// - offset_bits (2 bits) remain unbound → 4 final openings
#[derive(Clone)]
pub struct RecursionSumcheckParams {
    /// Number of constraint variables (x) - fixed at 4 for Fq12
    pub num_constraint_vars: usize,

    /// Number of constraint index variables - ceil(log2(num_constraints))
    pub num_constraint_index_vars: usize,

    /// Number of constraints (actual, no padding)
    pub num_constraints: usize,

    /// Sumcheck instance identifier
    pub sumcheck_id: SumcheckId,

    /// Reference to recursion polynomial
    pub polynomial: CommittedPolynomial,
}

impl RecursionSumcheckParams {
    pub fn new(
        num_constraint_index_vars: usize,
        num_constraints: usize,
        polynomial: CommittedPolynomial,
    ) -> Self {
        Self {
            num_constraint_vars: 4, // Fixed for Fq12
            num_constraint_index_vars,
            num_constraints,
            sumcheck_id: SumcheckId::RecursionZeroCheck,
            polynomial,
        }
    }

    /// Total sumcheck rounds: x_vars + constraint_index_vars
    /// Note: offset bits (2) are NOT bound during sumcheck
    pub fn num_rounds(&self) -> usize {
        self.num_constraint_vars + self.num_constraint_index_vars
    }

    pub fn get_opening_point<const E: crate::poly::opening_proof::Endianness>(
        &self,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> OpeningPoint<E, Fq> {
        OpeningPoint::new(sumcheck_challenges.to_vec())
    }
}

/// Prover for recursion zero-check sumcheck
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct RecursionSumcheckProver {
    /// Materialized M(s, x) as a multilinear polynomial
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub m_poly: MultilinearPolynomial<Fq>,

    /// g(x) polynomial for constraint evaluation
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub g_poly: MultilinearPolynomial<Fq>,

    /// Dense bit array: constraint_bits[i] = bit for constraint i (false for padding)
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub constraint_bits: Vec<bool>,

    /// Equality polynomial for constraint variables x (phase 1)
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub eq_x: MultilinearPolynomial<Fq>,

    /// Equality polynomial for constraint index i (phase 2)
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub eq_i: MultilinearPolynomial<Fq>,

    /// Random challenge for eq(r, x)
    pub r_x: Vec<<Fq as JoltField>::Challenge>,

    /// Random challenge for eq(r', i)
    pub r_i: Vec<<Fq as JoltField>::Challenge>,

    /// Scalar from phase 1 completion: eq(r', x_bound)
    pub eq_r_x: Fq,

    /// Current round number
    pub round: usize,

    /// Parameters
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub params: RecursionSumcheckParams,

    /// Values of eq_r_x * eq(r_i, i) * C_i(r_x) over all padded constraint indices.
    /// This is the polynomial we sumcheck over in Phase 2.
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub phase2_values: Vec<Fq>,
}

/// Helper struct to hold row evaluations for all offset types
struct RowEvaluations<const DEGREE: usize> {
    base: [Fq; DEGREE],
    rho_prev: [Fq; DEGREE],
    rho_curr: [Fq; DEGREE],
    quotient: [Fq; DEGREE],
}

impl RecursionSumcheckProver {
    pub fn gen<T: Transcript>(
        params: RecursionSumcheckParams,
        constraint_system: &super::recursion_constraints::ConstraintSystem,
        transcript: &mut T,
    ) -> Self {
        // Extract random challenges for eq polynomials
        // r_x for constraint variables (Phase 1)
        let r_x: Vec<<Fq as JoltField>::Challenge> = (0..params.num_constraint_vars)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();

        // r_i for constraint indices (Phase 2)
        let r_i: Vec<<Fq as JoltField>::Challenge> = (0..params.num_constraint_index_vars)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();

        // Materialize M polynomial directly
        let m_poly = MultilinearPolynomial::from(constraint_system.matrix.evaluations.clone());

        let eq_x = MultilinearPolynomial::from(EqPolynomial::<Fq>::evals(&r_x));
        let eq_i = MultilinearPolynomial::from(EqPolynomial::<Fq>::evals(&r_i));

        let mut constraint_bits = vec![false; params.num_constraints];

        for constraint in &constraint_system.constraints {
            constraint_bits[constraint.constraint_index] = constraint.bit;

            #[cfg(test)]
            if constraint.constraint_index < 5 {
                eprintln!(
                    "  Setting constraint_bits[{}] = {}",
                    constraint.constraint_index, constraint.bit
                );
            }
        }
        Self {
            m_poly,
            g_poly: MultilinearPolynomial::LargeScalars(constraint_system.g_poly.clone()),
            constraint_bits,
            eq_x,
            eq_i,
            r_x,
            r_i,
            eq_r_x: Fq::one(),
            round: 0,
            params,
            phase2_values: Vec::new(),
        }
    }
}

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for RecursionSumcheckProver {
    fn degree(&self) -> usize {
        4
    }

    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars + self.params.num_constraint_index_vars
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<Fq>) -> Fq {
        // Zero-check: claim the sum is zero
        Fq::zero()
    }

    fn compute_message(&mut self, round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        debug_assert!(
            round < self.params.num_rounds(),
            "Round {} exceeds total rounds {}",
            round,
            self.params.num_rounds()
        );

        #[cfg(test)]
        eprintln!(
            "  compute_message: round={}, previous_claim={:?}, phase={}",
            round,
            previous_claim,
            if round < self.params.num_constraint_vars {
                "Phase1"
            } else {
                "Phase2"
            }
        );

        if round < self.params.num_constraint_vars {
            // Phase 1: Sum over constraint variables (x)
            self.compute_phase1_message(round, previous_claim)
        } else {
            // Phase 2: Sum over constraint index variables (i)
            self.compute_phase2_message(round - self.params.num_constraint_vars, previous_claim)
        }
    }

    fn ingest_challenge(&mut self, r_j: <Fq as JoltField>::Challenge, round: usize) {
        debug_assert_eq!(
            round, self.round,
            "Expected round {}, got {}",
            self.round, round
        );

        if round < self.params.num_constraint_vars {
            // Phase 1: Bind constraint variable x (low-order bits in M)
            self.eq_x.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.m_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.g_poly.bind_parallel(r_j, BindingOrder::LowToHigh);

            // At phase transition, capture eq(r, x_bound)
            if round == self.params.num_constraint_vars - 1 {
                self.eq_r_x = self.eq_x.get_bound_coeff(0);
                debug_assert_eq!(
                    self.eq_x.len(),
                    1,
                    "eq_x should be fully bound after Phase 1"
                );

                #[cfg(test)]
                {
                    eprintln!(
                        "  After Phase 1: m_poly.len() = {} (expected = {})",
                        self.m_poly.len(),
                        1 << (self.params.num_constraint_index_vars + 2)
                    );
                    eprintln!("  Prover eq_r_x after Phase 1 = {:?}", self.eq_r_x);
                    eprintln!("  (This should be eq(r_x, x_challenges) where x_challenges are the Phase 1 sumcheck challenges)");
                }

                // Build phase2_values from m_poly, eq_i, and constraint_bits
                self.init_phase2_values();
            }
        } else {
            // Phase 2: Bind constraint index variable i
            // First fold the phase2_values table with the challenge
            let r_scalar: Fq = r_j.into();
            self.fold_phase2_values(r_scalar);

            // M layout: M(offset_bits, constraint_index_bits, x_bits)
            // After Phase 1, x_bits are bound. Now we bind constraint_index_bits.
            // offset_bits (2 bits, high-order) remain unbound → 4 final openings
            self.eq_i.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.m_poly.bind_parallel(r_j, BindingOrder::LowToHigh);

            #[cfg(debug_assertions)]
            {
                let phase2_round = round - self.params.num_constraint_vars;
                eprintln!("After binding Phase 2 round {}:", phase2_round);
                eprintln!(
                    "  m_poly.len() = {} (expected = {})",
                    self.m_poly.len(),
                    4 << (self.params.num_constraint_index_vars - phase2_round - 1)
                );
                eprintln!("  m_poly.num_vars = {}", self.m_poly.get_num_vars());
            }
        }

        self.round = round + 1;
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // After Phase 1 (x) and Phase 2 (constraint index), M has 2 unbound offset bits.
        // M layout: M(offset_bits, constraint_index_bits, x_bits)
        // After binding: m_poly has 4 coefficients, one per offset:
        //   - m_poly.get_bound_coeff(0) = M(00, r_i, r_x) = base
        //   - m_poly.get_bound_coeff(1) = M(01, r_i, r_x) = rho_prev
        //   - m_poly.get_bound_coeff(2) = M(10, r_i, r_x) = rho_curr
        //   - m_poly.get_bound_coeff(3) = M(11, r_i, r_x) = quotient

        let base_val = self.m_poly.get_bound_coeff(RowOffset::Base as usize);
        let rho_prev_val = self.m_poly.get_bound_coeff(RowOffset::RhoPrev as usize);
        let rho_curr_val = self.m_poly.get_bound_coeff(RowOffset::RhoCurr as usize);
        let quotient_val = self.m_poly.get_bound_coeff(RowOffset::Quotient as usize);

        #[cfg(debug_assertions)]
        {
            eprintln!("\n=== PROVER cache_openings ===");
            eprintln!("Final m_poly values after all sumcheck rounds:");
            eprintln!("  base_val     = {:?}", base_val);
            eprintln!("  rho_prev_val = {:?}", rho_prev_val);
            eprintln!("  rho_curr_val = {:?}", rho_curr_val);
            eprintln!("  quotient_val = {:?}", quotient_val);
            eprintln!("  m_poly.len() = {} (should be 4)", self.m_poly.len());

            // Compute the same values as verifier for comparison
            let (x_challenges_for_calc, i_challenges_for_calc) =
                sumcheck_challenges.split_at(self.params.num_constraint_vars);

            // g_val computation
            // g_poly is already fully bound after Phase 1, so it's just a scalar
            let g_val_computed = if self.g_poly.len() == 1 {
                self.g_poly.get_bound_coeff(0)
            } else {
                let mut x_challenges_reversed = x_challenges_for_calc.to_vec();
                x_challenges_reversed.reverse();
                self.g_poly.evaluate(&x_challenges_reversed)
            };

            // bit_eval computation
            let r_i_fq: Vec<Fq> = i_challenges_for_calc.iter().map(|c| (*c).into()).collect();
            let bit_eval_computed = self.compute_bit_mle_at_point(&r_i_fq);

            eprintln!("Additional computed values (prover side):");
            eprintln!("  g_val        = {:?}", g_val_computed);
            eprintln!("  bit_eval     = {:?}", bit_eval_computed);

            // Compute constraint eval
            let base_power = Fq::one() + (base_val - Fq::one()) * bit_eval_computed;
            let constraint_eval =
                rho_curr_val - rho_prev_val.square() * base_power - quotient_val * g_val_computed;

            // Compute eq polynomials
            let eq_r_x_computed = EqPolynomial::<Fq>::mle(&self.r_x, x_challenges_for_calc);
            let eq_r_i_computed = EqPolynomial::<Fq>::mle(&self.r_i, i_challenges_for_calc);

            let expected_claim_prover = eq_r_x_computed * eq_r_i_computed * constraint_eval;

            eprintln!("Constraint evaluation (prover side):");
            eprintln!("  base_power     = {:?}", base_power);
            eprintln!("  constraint_eval = {:?}", constraint_eval);
            eprintln!("  eq_r_x         = {:?}", eq_r_x_computed);
            eprintln!("  eq_r_i         = {:?}", eq_r_i_computed);
            eprintln!("  expected_claim = {:?}", expected_claim_prover);

            // Add more debug info about the challenges
            eprintln!("Challenge info:");
            eprintln!(
                "  x_challenges_for_calc.len() = {}",
                x_challenges_for_calc.len()
            );
            eprintln!(
                "  i_challenges_for_calc.len() = {}",
                i_challenges_for_calc.len()
            );
            eprintln!("  r_x.len() = {}", self.r_x.len());
            eprintln!("  r_i.len() = {}", self.r_i.len());
            eprintln!("===========================\n");
        }

        let (x_challenges, i_challenges) =
            sumcheck_challenges.split_at(self.params.num_constraint_vars);

        // Append 4 dense claims for the 4 offset evaluations
        // Each uses opening point: [offset_bits || i_challenges || x_challenges]
        for (offset, value) in [
            (RowOffset::Base, base_val),
            (RowOffset::RhoPrev, rho_prev_val),
            (RowOffset::RhoCurr, rho_curr_val),
            (RowOffset::Quotient, quotient_val),
        ] {
            let mut offset_bits = offset_bits_to_challenges(offset);
            // The matrix layout is M(offset_bits, constraint_index_bits, x_bits)
            // The offset bits need to be reversed from little-endian to big-endian
            offset_bits.reverse();

            #[cfg(debug_assertions)]
            {
                eprintln!("PROVER appending opening for offset {:?}:", offset);
                eprintln!("  offset_bits (after reverse) = {:?}", offset_bits);
                eprintln!("  value = {:?}", value);
            }

            // The polynomial has variables in storage order: [x_bits, i_bits, offset_bits] (little-endian)
            // MLE evaluate expects big-endian order: highest variable first
            // So for storage order [v0, v1, ..., v18], MLE expects point for [v18, v17, ..., v0]
            //
            // Storage: [x0, x1, x2, x3, i0, i1, ..., i12, offset0, offset1]
            // MLE expects: [offset1, offset0, i12, ..., i0, x3, x2, x1, x0]
            let mut reversed_x = x_challenges.to_vec();
            reversed_x.reverse();
            let mut reversed_i = i_challenges.to_vec();
            reversed_i.reverse();
            // Construct opening point in big-endian order to match MLE expectations
            let opening_point = [&offset_bits[..], &reversed_i[..], &reversed_x[..]].concat();

            #[cfg(debug_assertions)]
            {
                eprintln!("  Full opening_point = {:?}", opening_point);
                eprintln!("  Length = {}", opening_point.len());
            }

            accumulator.append_dense(
                transcript,
                self.params.polynomial,
                offset_to_sumcheck_id(offset),
                opening_point,
                value,
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

fn offset_to_sumcheck_id(offset: RowOffset) -> SumcheckId {
    match offset {
        RowOffset::Base => SumcheckId::RecursionBase,
        RowOffset::RhoPrev => SumcheckId::RecursionRhoPrev,
        RowOffset::RhoCurr => SumcheckId::RecursionRhoCurr,
        RowOffset::Quotient => SumcheckId::RecursionQuotient,
    }
}

fn offset_bits_to_challenges(offset: RowOffset) -> Vec<<Fq as JoltField>::Challenge> {
    let bits = offset.to_bits(); // Returns [Fq; 2]
    bits.into_iter()
        .map(|bit| <Fq as JoltField>::Challenge::from(bit))
        .collect()
}

// Helper methods for phase computation
impl RecursionSumcheckProver {
    /// Initialize the Phase 2 table with precomputed values.
    /// Called once at the transition from Phase 1 to Phase 2.
    fn init_phase2_values(&mut self) {
        // Only do it once
        if !self.phase2_values.is_empty() {
            return;
        }

        let num_i_bits = self.params.num_constraint_index_vars;
        let padded_size = 1 << num_i_bits;

        // After Phase 1, m_poly has vars [i_bits][offset_bits] and len = 2^(num_i_bits + 2)
        debug_assert_eq!(self.m_poly.len(), 1 << (num_i_bits + 2));

        // g(x) is fully bound after Phase 1
        debug_assert_eq!(self.g_poly.len(), 1);
        let g_val = self.g_poly.get_bound_coeff(0);

        self.phase2_values = Vec::with_capacity(padded_size);

        for i in 0..padded_size {
            // Index in m_poly: i_bits are low-order, offset_bits are high-order
            let row = |offset: RowOffset| -> Fq {
                let offset_as_usize = offset as usize;
                let base_idx = (offset_as_usize << num_i_bits) | i;
                self.m_poly.get_bound_coeff(base_idx)
            };

            let base = row(RowOffset::Base);
            let rho_prev = row(RowOffset::RhoPrev);
            let rho_curr = row(RowOffset::RhoCurr);
            let quotient = row(RowOffset::Quotient);

            let bit = if i < self.params.num_constraints {
                self.constraint_bits[i]
            } else {
                false
            };

            // Use the same interpolation formula as the verifier
            let bit_as_field = if bit { Fq::one() } else { Fq::zero() };
            let base_power = Fq::one() + (base - Fq::one()) * bit_as_field;

            let c_i = rho_curr - rho_prev * rho_prev * base_power - quotient * g_val;

            // eq_i still has all i-bits unbound at this moment
            let eq_i_val = self.eq_i.get_bound_coeff(i);

            // Multiply by eq_r_x at Phase-2 entry
            let value = self.eq_r_x * eq_i_val * c_i;

            self.phase2_values.push(value);
        }
    }

    /// Fold the Phase 2 table after receiving a challenge.
    fn fold_phase2_values(&mut self, r: Fq) {
        let len = self.phase2_values.len();
        debug_assert!(len.is_power_of_two());
        debug_assert!(len >= 2);

        let half = len / 2;
        let one_minus_r = Fq::one() - r;

        let mut next = vec![Fq::zero(); half];

        for i in 0..half {
            let g0 = self.phase2_values[2 * i];
            let g1 = self.phase2_values[2 * i + 1];
            next[i] = one_minus_r * g0 + r * g1;
        }

        self.phase2_values = next;
    }

    /// Compute the MLE of the bit vector evaluated at a specific point
    /// bit_eval = Σ_i eq(eval_point, i) * b_i
    pub fn compute_bit_mle_at_point(&self, eval_point: &[Fq]) -> Fq {
        // Use EqPolynomial::evals to ensure consistent bit ordering
        let eq_vals = EqPolynomial::<Fq>::evals(eval_point);
        let mut result = Fq::zero();

        for (idx, &coeff) in eq_vals.iter().enumerate() {
            if idx < self.constraint_bits.len() && self.constraint_bits[idx] {
                result += coeff;
            }
        }
        result
    }
    /// Computes the index into the partially-bound M polynomial for sumcheck evaluation.
    ///
    /// # Polynomial Structure
    /// M has layout: [offset_bits][constraint_bits][x_bits] (big-endian)
    /// This matches the construction in recursion_constraints.rs
    /// After binding k x variables, remaining structure is:
    /// - x_remaining = original_x_bits - k
    /// - i_bits unchanged
    /// - offset_bits unchanged (always 2)
    ///
    /// # Index Formula Derivation
    /// For sumcheck_evals_array to work, we need index < size/2 where:
    /// - size = 2^(x_remaining + i_bits + offset_bits)
    /// - We're accessing the "left half" for interpolation
    ///
    /// The formula matches the actual construction layout: M(offset_bits, constraint_bits, x_bits)
    /// represents the position in the flattened array where:
    /// - offset: highest order bits (selects which row: Base/RhoPrev/RhoCurr/Quotient)
    /// - constraint_idx: middle bits (selects which constraint)
    /// - x_idx: lowest order bits (position within x variables)
    fn compute_m_sumcheck_index(
        &self,
        offset: RowOffset,
        constraint_idx: usize,
        x_idx: usize,
        num_i_bits: usize,
        num_x_remaining: usize,
    ) -> usize {
        // Fixed formula to match actual construction layout:
        // M(offset_bits, constraint_bits, x_bits)
        ((offset as usize) << (num_i_bits + num_x_remaining - 1))
            | (constraint_idx << (num_x_remaining - 1))
            | x_idx
    }

    /// Phase 1: Compute sumcheck message while binding constraint variables x
    /// p(t) = Σ_i eq(r', i) * Σ_{x_remaining} eq(r, x) * C_i(x)
    ///
    /// M layout: M(offset_bits, constraint_index_bits, x_bits)
    /// During Phase 1, we're binding x_bits (low-order).
    fn compute_phase1_message(&self, _round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        const DEGREE: usize = 4;
        const NUM_EVALS: usize = DEGREE + 1; // Need degree + 1 points
        let num_x_remaining = self.eq_x.get_num_vars();
        let x_half = 1 << (num_x_remaining - 1);

        debug_assert!(
            num_x_remaining > 0,
            "eq_x should have unbound variables in Phase 1"
        );

        // Calculate bit positions for index computation
        let num_i_bits = self.params.num_constraint_index_vars;

        // Compute evaluations at t = 0, 1, 2, 3, 4 explicitly (need 5 points for degree 4)
        let evals: [Fq; NUM_EVALS] = std::array::from_fn(|t| {
            let t_scalar = Fq::from(t as u64);

            // For each evaluation point t, sum over all x indices
            (0..x_half)
                .into_par_iter()
                .map(|x_idx| {
                    // Compute eq_x evaluation at point t
                    // For multilinear: p(t) = (1-t)*p(0) + t*p(1)
                    let eq_x_0 = self.eq_x.get_bound_coeff(2 * x_idx);
                    let eq_x_1 = self.eq_x.get_bound_coeff(2 * x_idx + 1);
                    let eq_x_t = (Fq::one() - t_scalar) * eq_x_0 + t_scalar * eq_x_1;

                    // Similarly for g_poly
                    let g_0 = self.g_poly.get_bound_coeff(2 * x_idx);
                    let g_1 = self.g_poly.get_bound_coeff(2 * x_idx + 1);
                    let g_t = (Fq::one() - t_scalar) * g_0 + t_scalar * g_1;

                    let mut constraint_sum = Fq::zero();

                    for constraint_idx in 0..self.params.num_constraints {
                        let eq_i_val = self.eq_i.get_bound_coeff(constraint_idx);

                        // Compute constraint evaluation at point t
                        let constraint_val = self.evaluate_constraint_at_point(
                            constraint_idx,
                            x_idx,
                            t,
                            g_t,
                            num_i_bits,
                            num_x_remaining,
                        );

                        constraint_sum += eq_i_val * constraint_val;
                    }

                    eq_x_t * constraint_sum
                })
                .reduce(|| Fq::zero(), |a, b| a + b)
        });

        // Use standard polynomial interpolation without hints
        let poly = UniPoly::from_evals(&evals);

        // Debug verification
        #[cfg(debug_assertions)]
        {
            let sum = poly.evaluate(&Fq::zero()) + poly.evaluate(&Fq::one());
            eprintln!(
                "Clean approach Phase1 - H(0)+H(1) = {:?}, expected = {:?}",
                sum, previous_claim
            );
            if sum != previous_claim {
                eprintln!("WARNING: Phase 1 sumcheck relation not satisfied!");
                eprintln!("  Evaluations at t=0,1,2,3,4: {:?}", evals);
            }
        }

        poly
    }

    /// Phase 2: Compute sumcheck message while binding constraint index variables i
    /// At this point, x is fully bound, so C_i(x_bound) is a scalar for each constraint.
    /// We bind both eq_i AND m_poly on the constraint index variables.
    ///
    /// The polynomial being summed is: f(i) = eq_r_x * eq(r_i, i) * C_i(r_x)
    /// While eq is multilinear, C_i contains squares and exponentiation, so the
    fn compute_phase2_message(&self, phase2_round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        // Phase 2 has degree 4 (same as Phase 1)
        const DEGREE: usize = 4;
        const NUM_EVALS: usize = DEGREE + 1; // 5 points: t = 0,1,2,3,4

        let len = self.phase2_values.len();
        debug_assert!(len.is_power_of_two());
        debug_assert!(len >= 2);

        let half = len / 2;

        // Compute evaluations at t = 0, 1, 2, 3, 4
        let evals: [Fq; NUM_EVALS] = std::array::from_fn(|t| {
            let t_scalar = Fq::from(t as u64);
            let one_minus_t = Fq::one() - t_scalar;

            (0..half)
                .into_par_iter()
                .map(|i| {
                    let g0 = self.phase2_values[2 * i];
                    let g1 = self.phase2_values[2 * i + 1];
                    one_minus_t * g0 + t_scalar * g1
                })
                .reduce(Fq::zero, |a, b| a + b)
        });

        let poly = UniPoly::from_evals(&evals);

        #[cfg(debug_assertions)]
        {
            let h0 = poly.evaluate(&Fq::zero());
            let h1 = poly.evaluate(&Fq::one());
            let sum = h0 + h1;
            eprintln!(
                "Phase2 round {} - H(0)+H(1) = {:?}, expected = {:?}",
                phase2_round, sum, previous_claim
            );
            if sum != previous_claim {
                eprintln!("WARNING: Phase 2 sumcheck relation not satisfied!");
            }
        }

        poly
    }

    /// Helper function to evaluate constraint at a specific point t
    fn evaluate_constraint_at_point(
        &self,
        constraint_idx: usize,
        x_idx: usize,
        t: usize,
        g_t: Fq,
        num_i_bits: usize,
        num_x_remaining: usize,
    ) -> Fq {
        // Get the constraint bit
        let bit = self.constraint_bits[constraint_idx];

        // Compute row evaluations at point t
        let base_t = self.evaluate_row_at_t(
            RowOffset::Base,
            constraint_idx,
            x_idx,
            t,
            num_i_bits,
            num_x_remaining,
        );
        let rho_prev_t = self.evaluate_row_at_t(
            RowOffset::RhoPrev,
            constraint_idx,
            x_idx,
            t,
            num_i_bits,
            num_x_remaining,
        );
        let rho_curr_t = self.evaluate_row_at_t(
            RowOffset::RhoCurr,
            constraint_idx,
            x_idx,
            t,
            num_i_bits,
            num_x_remaining,
        );
        let quotient_t = self.evaluate_row_at_t(
            RowOffset::Quotient,
            constraint_idx,
            x_idx,
            t,
            num_i_bits,
            num_x_remaining,
        );

        // Compute C_i(x) = ρ_curr(x) - ρ_prev(x)² × base(x)^{b_i} - quotient(x) × g(x)
        // Use the same interpolation formula as in phase2 and verifier
        let bit_as_field = if bit { Fq::one() } else { Fq::zero() };
        let base_power = Fq::one() + (base_t - Fq::one()) * bit_as_field;
        rho_curr_t - rho_prev_t * rho_prev_t * base_power - quotient_t * g_t
    }

    /// Helper function to evaluate a row of the matrix at point t
    fn evaluate_row_at_t(
        &self,
        offset: RowOffset,
        constraint_idx: usize,
        x_idx: usize,
        t: usize,
        num_i_bits: usize,
        num_x_remaining: usize,
    ) -> Fq {
        let t_scalar = Fq::from(t as u64);
        let idx = self.compute_m_sumcheck_index(
            offset,
            constraint_idx,
            x_idx,
            num_i_bits,
            num_x_remaining,
        );

        // Get evaluations at 0 and 1
        let val_0 = self.m_poly.get_bound_coeff(2 * idx);
        let val_1 = self.m_poly.get_bound_coeff(2 * idx + 1);

        // Linear interpolation for multilinear polynomial
        (Fq::one() - t_scalar) * val_0 + t_scalar * val_1
    }
}

/// Verifier for recursion zero-check sumcheck
///
/// After sumcheck completes, M has 2 unbound offset bits, giving 4 openings:
/// - M(00, r_i, r_x) = base
/// - M(01, r_i, r_x) = rho_prev
/// - M(10, r_i, r_x) = rho_curr
/// - M(11, r_i, r_x) = quotient
///
/// The verifier computes the constraint:
/// C(r_i, r_x) = rho_curr - rho_prev² × [1 + (base - 1) × bit_eval] - quotient × g(r_x)
/// where bit_eval = Σ_i eq(r_i, i) * b_i (MLE of bits at r_i)
pub struct RecursionSumcheckVerifier {
    pub params: RecursionSumcheckParams,
    /// Random challenge for eq(r, x) - constraint variables
    pub r_x: Vec<<Fq as JoltField>::Challenge>,
    /// Random challenge for eq(r', i) - constraint indices
    pub r_i: Vec<<Fq as JoltField>::Challenge>,
    /// Dense bit array: constraint_bits[i] = bit for constraint i (false for padding)
    pub constraint_bits: Vec<bool>,
    /// Precomputed g(x) polynomial for constraint evaluation
    pub g_poly: DensePolynomial<Fq>,
}

impl RecursionSumcheckVerifier {
    pub fn new<T: Transcript>(
        params: RecursionSumcheckParams,
        constraints: Vec<MatrixConstraint>,
        g_poly: DensePolynomial<Fq>,
        transcript: &mut T,
    ) -> Self {
        let r_x: Vec<<Fq as JoltField>::Challenge> = (0..params.num_constraint_vars)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();

        let r_i: Vec<<Fq as JoltField>::Challenge> = (0..params.num_constraint_index_vars)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();

        // Build dense bit array from constraints
        let mut constraint_bits = vec![false; params.num_constraints];
        for constraint in &constraints {
            constraint_bits[constraint.constraint_index] = constraint.bit;
        }
        Self {
            params,
            r_x,
            r_i,
            constraint_bits,
            g_poly,
        }
    }

    /// Compute the MLE of the bit vector evaluated at a specific point
    /// bit_eval = Σ_i eq(eval_point, i) * b_i
    pub fn compute_bit_mle_at_point(&self, eval_point: &[Fq]) -> Fq {
        // Use EqPolynomial::evals to ensure consistent bit ordering
        let eq_vals = EqPolynomial::<Fq>::evals(eval_point);
        let mut result = Fq::zero();

        for (idx, &coeff) in eq_vals.iter().enumerate() {
            if idx < self.constraint_bits.len() && self.constraint_bits[idx] {
                result += coeff;
            }
        }
        result
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for RecursionSumcheckVerifier {
    fn degree(&self) -> usize {
        // C_i(x) = ρ_curr - ρ_prev² × a^b - Q × g has degree 3 due to ρ_prev² × a
        4
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<Fq>,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        let (x_challenges, i_challenges) =
            sumcheck_challenges.split_at(self.params.num_constraint_vars);

        let mut values = vec![];
        for offset in [
            RowOffset::Base,
            RowOffset::RhoPrev,
            RowOffset::RhoCurr,
            RowOffset::Quotient,
        ] {
            let (_point, value) = accumulator.get_committed_polynomial_opening(
                self.params.polynomial,
                offset_to_sumcheck_id(offset),
            );
            values.push(value);
        }

        let [base_val, rho_prev_val, rho_curr_val, quotient_val] =
            values.try_into().expect("Should have exactly 4 values");

        let mut x_challenges_reversed = x_challenges.to_vec();
        x_challenges_reversed.reverse();
        let g_val = self.g_poly.evaluate(&x_challenges_reversed);

        let r_i_fq: Vec<Fq> = i_challenges.iter().map(|c| (*c).into()).collect();

        let bit_eval = self.compute_bit_mle_at_point(&r_i_fq);

        #[cfg(debug_assertions)]
        {
            eprintln!("\n=== VERIFIER expected_output_claim ===");
            eprintln!("Values from accumulator:");
            eprintln!("  base_val     = {:?}", base_val);
            eprintln!("  rho_prev_val = {:?}", rho_prev_val);
            eprintln!("  rho_curr_val = {:?}", rho_curr_val);
            eprintln!("  quotient_val = {:?}", quotient_val);
            eprintln!("Additional computed values:");
            eprintln!("  g_val        = {:?}", g_val);
            eprintln!("  bit_eval     = {:?}", bit_eval);
        }

        // Compute constraint: C = ρ_curr - ρ_prev² × [1 + (base - 1) × bit_eval] - quotient × g
        let base_power = Fq::one() + (base_val - Fq::one()) * bit_eval;
        let constraint_eval =
            rho_curr_val - rho_prev_val.square() * base_power - quotient_val * g_val;

        let eq_r_x = EqPolynomial::<Fq>::mle(&self.r_x, x_challenges);
        let eq_r_i = EqPolynomial::<Fq>::mle(&self.r_i, i_challenges);

        let expected_claim = eq_r_x * eq_r_i * constraint_eval;

        #[cfg(debug_assertions)]
        {
            eprintln!("Constraint evaluation:");
            eprintln!("  base_power     = {:?}", base_power);
            eprintln!("  constraint_eval = {:?}", constraint_eval);
            eprintln!("  eq_r_x         = {:?}", eq_r_x);
            eprintln!("  eq_r_i         = {:?}", eq_r_i);
            eprintln!("  expected_claim = {:?}", expected_claim);
            eprintln!("===========================\n");
        }

        expected_claim
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // The prover now appends 4 dense openings for the 4 M evaluations
        // at points M(offset, r_i, r_x) for offset ∈ {0,1,2,3}

        let (x_challenges, i_challenges) =
            sumcheck_challenges.split_at(self.params.num_constraint_vars);

        // Each uses opening point: [offset_bits || i_challenges || x_challenges]
        for offset in [
            RowOffset::Base,
            RowOffset::RhoPrev,
            RowOffset::RhoCurr,
            RowOffset::Quotient,
        ] {
            let mut offset_bits = offset_bits_to_challenges(offset);
            offset_bits.reverse();

            #[cfg(debug_assertions)]
            {
                eprintln!("VERIFIER appending opening for offset {:?}:", offset);
                eprintln!("  offset_bits (after reverse) = {:?}", offset_bits);
            }
            // The polynomial has variables in storage order: [x_bits, i_bits, offset_bits] (little-endian)
            // MLE evaluate expects big-endian order: highest variable first
            let mut reversed_x = x_challenges.to_vec();
            reversed_x.reverse();
            let mut reversed_i = i_challenges.to_vec();
            reversed_i.reverse();
            let opening_point = [&offset_bits[..], &reversed_i[..], &reversed_x[..]].concat();

            #[cfg(debug_assertions)]
            {
                eprintln!("  Full opening_point = {:?}", opening_point);
                eprintln!("  Length = {}", opening_point.len());
            }

            accumulator.append_dense(
                transcript,
                self.params.polynomial,
                offset_to_sumcheck_id(offset),
                opening_point,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        poly::{
            commitment::{
                commitment_scheme::CommitmentScheme,
                dory::{DoryCommitmentScheme, DoryGlobals},
                hyrax::{Hyrax, HyraxOpeningProof},
            },
            dense_mlpoly::DensePolynomial,
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        },
        subprotocols::{recursion_constraints::ConstraintSystem, sumcheck::BatchedSumcheck},
        transcripts::Blake2bTranscript,
    };
    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use ark_grumpkin::Projective as GrumpkinProjective;
    use rand::thread_rng;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_dory_witness_recursion_sumcheck_hyrax_reduce_and_prove() {
        use crate::poly::commitment::commitment_scheme::RecursionExt;
        use crate::poly::commitment::hyrax::Hyrax;
        use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
        use std::collections::HashMap;

        const RATIO: usize = 1;

        // Setup using Dory commitment
        DoryGlobals::reset();
        DoryGlobals::initialize(1 << 2, 1 << 2);
        let num_vars = 4;
        let mut rng = thread_rng();

        // 1. Create Dory proof and extract witnesses
        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
        let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);

        let coefficients: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
        let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coefficients));
        let (commitment, hint) = DoryCommitmentScheme::commit(&poly, &prover_setup);

        let point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::rand(&mut rng))
            .collect();

        let mut prover_transcript = Blake2bTranscript::new(b"test");
        let proof = DoryCommitmentScheme::prove(
            &prover_setup,
            &poly,
            &point,
            Some(hint.clone()),
            &mut prover_transcript,
        );

        // Extract witnesses using witness_gen
        let evaluation = PolynomialEvaluation::evaluate(&poly, &point);
        let mut witness_transcript = Blake2bTranscript::new(b"test");
        let (_witnesses, _hints) = DoryCommitmentScheme::witness_gen(
            &proof,
            &verifier_setup,
            &mut witness_transcript,
            &point,
            &evaluation,
            &commitment,
        )
        .expect("Witness generation should succeed");

        // 2. Build constraint system from witnesses
        let mut constraint_transcript = Blake2bTranscript::new(b"test");
        let (constraint_system, _constraint_hints) = ConstraintSystem::new(
            &proof,
            &verifier_setup,
            &mut constraint_transcript,
            &point,
            &evaluation,
            &commitment,
        )
        .expect("Constraint system creation should succeed");

        // 3. Setup Hyrax and commit to constraint matrix M
        let m_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(
            constraint_system.matrix.evaluations.clone(),
        ));

        let hyrax_prover_setup =
            Hyrax::<RATIO, GrumpkinProjective>::setup_prover(m_poly.get_num_vars());
        let hyrax_verifier_setup =
            Hyrax::<RATIO, GrumpkinProjective>::setup_verifier(&hyrax_prover_setup);

        let (m_commitment, _) =
            Hyrax::<RATIO, GrumpkinProjective>::commit(&m_poly, &hyrax_prover_setup);

        // 4. Run recursion sumcheck (two-phase)
        let params = RecursionSumcheckParams::new(
            constraint_system.matrix.num_constraint_index_vars,
            constraint_system.matrix.num_constraints,
            CommittedPolynomial::DoryConstraintMatrix,
        );

        let mut sumcheck_transcript = Blake2bTranscript::new(b"recursion_sumcheck");

        // Clone transcript for verifier to read from same state
        let mut verifier_transcript = sumcheck_transcript.clone();

        // Create opening accumulator for prover
        let log_T = m_poly.get_num_vars();
        let mut prover_accumulator = ProverOpeningAccumulator::<Fq>::new(log_T);

        // Create prover instance
        let mut prover = RecursionSumcheckProver::gen(
            params.clone(),
            &constraint_system,
            &mut sumcheck_transcript,
        );

        // Create verifier instance from same transcript state
        let verifier = RecursionSumcheckVerifier::new(
            params.clone(),
            constraint_system.constraints.clone(),
            constraint_system.g_poly.clone(),
            &mut verifier_transcript,
        );

        // Use batched sumcheck to prove
        let (sumcheck_proof, sumcheck_challenges) = BatchedSumcheck::prove(
            vec![&mut prover],
            &mut prover_accumulator,
            &mut sumcheck_transcript,
        );

        // 5. Reduce 4 offset claims to single claim using Hyrax
        // Create HashMaps for reduce_and_prove
        let mut committed_polynomials = HashMap::new();
        let mut committed_hints = HashMap::new();

        // Add the constraint matrix M to the committed polynomial map
        committed_polynomials.insert(CommittedPolynomial::DoryConstraintMatrix, m_poly.clone());

        // For Hyrax, the opening hint is just () (empty tuple)
        committed_hints.insert(CommittedPolynomial::DoryConstraintMatrix, ());

        // Use reduce_and_prove to reduce the 4 offset claims to a single claim
        let reduced_proof = prover_accumulator
            .reduce_and_prove::<Blake2bTranscript, Hyrax<RATIO, GrumpkinProjective>>(
                committed_polynomials,
                committed_hints,
                &hyrax_prover_setup,
                &mut sumcheck_transcript,
                None, // No streaming context for this test
            );

        // 6. Verify the sumcheck and reduced openings
        // Create verifier accumulator
        let mut verifier_accumulator = VerifierOpeningAccumulator::<Fq>::new(log_T);

        // Populate claims in the verifier accumulator (but not the points)
        for (key, (_, claim)) in &prover_accumulator.openings {
            verifier_accumulator
                .openings
                .insert(*key, (OpeningPoint::default(), *claim));
        }

        let verified_challenges = BatchedSumcheck::verify(
            &sumcheck_proof,
            vec![&verifier],
            &mut verifier_accumulator,
            &mut verifier_transcript,
        )
        .expect("Sumcheck verification should succeed");

        // Verify challenges match
        assert_eq!(
            verified_challenges.len(),
            sumcheck_challenges.len(),
            "Challenge count mismatch"
        );
        for (i, (prover_challenge, verifier_challenge)) in sumcheck_challenges
            .iter()
            .zip(verified_challenges.iter())
            .enumerate()
        {
            assert_eq!(
                prover_challenge, verifier_challenge,
                "Challenge mismatch at round {}",
                i
            );
        }

        // Create commitment maps for verifier
        let mut commitment_map = HashMap::new();
        commitment_map.insert(
            CommittedPolynomial::DoryConstraintMatrix,
            m_commitment.clone(),
        );

        // Reduce and verify with Hyrax
        let result = verifier_accumulator
            .reduce_and_verify::<Blake2bTranscript, Hyrax<RATIO, GrumpkinProjective>>(
                &hyrax_verifier_setup,
                &mut commitment_map,
                &reduced_proof,
                &mut verifier_transcript,
            );

        result.expect("Sumcheck verification should succeed");
    }
}
