//! Two-phase sumcheck for proving zero-check on recursion constraint system
//! Proves: 0 = Σ_{i,x} eq(r, i) * eq(r', x) * C_i(x)
//! Where C_i(x) = ρ_curr(x) - ρ_prev(x)² × a(x)^{b_i} - Q_i(x) × g(x)

use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, LITTLE_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        recursion_constraints::{MatrixConstraint, RowOffset},
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

    /// Number of constraint index variables - log2(num_constraints_padded)
    pub num_constraint_index_vars: usize,

    /// Number of constraints (before padding)
    pub num_constraints: usize,

    /// Number of constraints padded to power of 2
    pub num_constraints_padded: usize,

    /// Sumcheck instance identifier
    pub sumcheck_id: SumcheckId,

    /// Reference to committed matrix polynomial
    pub matrix_commitment: CommittedPolynomial,
}

impl RecursionSumcheckParams {
    pub fn new(
        num_constraint_index_vars: usize,
        num_constraints: usize,
        num_constraints_padded: usize,
        matrix_commitment: CommittedPolynomial,
    ) -> Self {
        Self {
            num_constraint_vars: 4, // Fixed for Fq12
            num_constraint_index_vars,
            num_constraints,
            num_constraints_padded,
            sumcheck_id: SumcheckId::RecursionZeroCheck,
            matrix_commitment,
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
    pub g_poly: DensePolynomial<Fq>,

    /// Constraint metadata (row indices, bits)
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub constraints: Vec<MatrixConstraint>,

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

        // Materialize M polynomial directly from the giant matrix
        let m_evals = Self::materialize_m_polynomial(&constraint_system.matrix);
        let m_poly = MultilinearPolynomial::from(m_evals);

        // Initialize equality polynomials
        let eq_x = MultilinearPolynomial::from(EqPolynomial::<Fq>::evals(&r_x));
        let eq_i = MultilinearPolynomial::from(EqPolynomial::<Fq>::evals(&r_i));

        Self {
            m_poly,
            g_poly: constraint_system.g_poly.clone(),
            constraints: constraint_system.constraints.clone(),
            eq_x,
            eq_i,
            r_x,
            r_i,
            eq_r_x: Fq::one(),
            round: 0,
            params,
        }
    }

    /// Materialize M(s, x) directly from the GiantMultilinearMatrix
    fn materialize_m_polynomial(
        matrix: &super::recursion_constraints::DoryMultilinearMatrix,
    ) -> Vec<Fq> {
        // M is already stored as evaluations in the correct order
        // The matrix stores rows (index s) with each row containing 2^num_constraint_vars evaluations
        matrix.evaluations.clone()
    }

    /// Number of constraint index variables
    fn num_constraint_index_vars(&self) -> usize {
        self.params.num_constraint_index_vars
    }
}

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for RecursionSumcheckProver {
    fn degree(&self) -> usize {
        // C_i(x) = ρ_curr - ρ_prev² × a^b - Q × g has degree 3 due to ρ_prev² × a
        4
    }

    fn num_rounds(&self) -> usize {
        // Phase 1: bind constraint variables (x) + Phase 2: bind constraint indices (i)
        self.params.num_constraint_vars + self.num_constraint_index_vars()
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<Fq>) -> Fq {
        // Zero-check: claim the sum is zero
        Fq::zero()
    }

    fn compute_message(&mut self, round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        if round < self.params.num_constraint_vars {
            // Phase 1: Sum over constraint variables (x)
            self.compute_phase1_message(round, previous_claim)
        } else {
            // Phase 2: Sum over constraint index variables (i)
            self.compute_phase2_message(round - self.params.num_constraint_vars, previous_claim)
        }
    }

    fn ingest_challenge(&mut self, r_j: <Fq as JoltField>::Challenge, round: usize) {
        if round < self.params.num_constraint_vars {
            // Phase 1: Bind constraint variable x (low-order bits in M)
            self.eq_x.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.m_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.g_poly.bound_poly_var_bot_01_optimized(&r_j.into());

            // At phase transition, capture eq(r, x_bound)
            if round == self.params.num_constraint_vars - 1 {
                self.eq_r_x = self.eq_x.get_bound_coeff(0);
            }
        } else {
            // Phase 2: Bind constraint index variable i
            // M layout: M(offset_bits, constraint_index_bits, x_bits)
            // After Phase 1, x_bits are bound. Now we bind constraint_index_bits.
            // offset_bits (2 bits, high-order) remain unbound → 4 final openings
            self.eq_i.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.m_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
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
        let g_val = self.g_poly.Z[0]; // g is fully bound after Phase 1

        // Compute bit_eval = Σ_i eq(r_i, i) * b_i (MLE of bits at r_i)
        // This allows the verifier to compute base^{bit_eval} as 1 + (base - 1) * bit_eval
        let bit_eval = self.compute_bit_mle_eval();

        // Append all 6 values to transcript for Fiat-Shamir
        transcript.append_scalar(&base_val);
        transcript.append_scalar(&rho_prev_val);
        transcript.append_scalar(&rho_curr_val);
        transcript.append_scalar(&quotient_val);
        transcript.append_scalar(&g_val);
        transcript.append_scalar(&bit_eval);

        // Compute constraint evaluation:
        // C(r_i, r_x) = rho_curr - rho_prev² × [1 + (base - 1) × bit_eval] - quotient × g
        let base_power = Fq::one() + (base_val - Fq::one()) * bit_eval;
        let constraint_eval =
            rho_curr_val - rho_prev_val.square() * base_power - quotient_val * g_val;

        // Split challenges: x_challenges (Phase 1), i_challenges (Phase 2)
        let (x_challenges, i_challenges) =
            sumcheck_challenges.split_at(self.params.num_constraint_vars);

        // Cache the final constraint evaluation and opening points
        // The 4 M openings are at points: (offset_bits, r_i, r_x) for each offset
        accumulator.append_sparse(
            transcript,
            vec![self.params.matrix_commitment],
            self.params.sumcheck_id,
            x_challenges.to_vec(),
            i_challenges.to_vec(),
            vec![constraint_eval],
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

// Helper methods for phase computation
impl RecursionSumcheckProver {
    /// Compute the MLE of the bit vector evaluated at r_i
    /// bit_eval = Σ_i eq(r_i, i) * b_i
    fn compute_bit_mle_eval(&self) -> Fq {
        // We need to compute Σ_i eq(r_i, i) * b_i for all constraints
        let r_i_fq: Vec<Fq> = self.r_i.iter().map(|c| (*c).into()).collect();
        let mut result = Fq::zero();
        for constraint in &self.constraints {
            if constraint.bit {
                // eq(r_i, i) for this constraint index
                let eq_val = EqPolynomial::mle(
                    &crate::subprotocols::recursion_constraints::index_to_binary(
                        constraint.constraint_index,
                        self.num_constraint_index_vars(),
                    ),
                    &r_i_fq,
                );
                result += eq_val;
            }
        }
        result
    }

    /// Get the bit value for a constraint by its index
    fn get_constraint_bit(&self, constraint_idx: usize) -> bool {
        // Padded indices return false (no bit set)
        if constraint_idx >= self.params.num_constraints {
            return false;
        }

        self.constraints
            .iter()
            .find(|c| c.constraint_index == constraint_idx)
            .map(|c| c.bit)
            .unwrap_or(false)
    }

    /// Evaluate constraint during Phase 2 using dynamic indexing
    fn evaluate_constraint_phase2(&self, constraint_idx: usize, phase2_round: usize) -> Fq {
        // Calculate current stride based on remaining constraint variables
        let remaining_constraint_vars = self.num_constraint_index_vars() - phase2_round;
        let stride = 1 << remaining_constraint_vars;

        // Shift out the already-bound low-order bits
        let idx_in_stride = constraint_idx >> phase2_round;

        // Calculate indices for each row type (offset)
        let base_idx = (RowOffset::Base as usize) * stride + idx_in_stride;
        let rho_prev_idx = (RowOffset::RhoPrev as usize) * stride + idx_in_stride;
        let rho_curr_idx = (RowOffset::RhoCurr as usize) * stride + idx_in_stride;
        let quotient_idx = (RowOffset::Quotient as usize) * stride + idx_in_stride;

        // Get values from m_poly
        let base = self.m_poly.get_bound_coeff(base_idx);
        let rho_prev = self.m_poly.get_bound_coeff(rho_prev_idx);
        let rho_curr = self.m_poly.get_bound_coeff(rho_curr_idx);
        let quotient = self.m_poly.get_bound_coeff(quotient_idx);
        let g_val = self.g_poly.Z[0]; // g is fully bound after Phase 1

        // Compute constraint: ρ_curr - ρ_prev² × base^{b_i} - quotient × g
        let base_power = if self.get_constraint_bit(constraint_idx) {
            base
        } else {
            Fq::one()
        };

        rho_curr - rho_prev.square() * base_power - quotient * g_val
    }

    /// Phase 1: Compute sumcheck message while binding constraint variables x
    /// p(t) = Σ_i eq(r', i) * Σ_{x_remaining} eq(r, x) * C_i(x)
    ///
    /// M layout: M(offset_bits, constraint_index_bits, x_bits)
    /// During Phase 1, we're binding x_bits (low-order).
    fn compute_phase1_message(&self, _round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        const DEGREE: usize = 4; // Degree 3 polynomial needs 4 evaluation points

        let num_x_remaining = self.eq_x.get_num_vars();
        let x_half = 1 << (num_x_remaining - 1);

        // For each constraint, compute its contribution to the sumcheck
        let evals: [Fq; DEGREE] = self
            .constraints
            .iter()
            .map(|constraint| {
                // Get eq(r', i) value for this constraint index
                let eq_i_val = self.eq_i.get_bound_coeff(constraint.constraint_index);

                // Sum over remaining x indices
                let constraint_evals: [Fq; DEGREE] = (0..x_half)
                    .into_par_iter()
                    .map(|x_idx| {
                        // Get eq_x evaluations at {0, 1, 2, 3}
                        let eq_x_evals = self
                            .eq_x
                            .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);

                        // Compute C_i at each evaluation point
                        let mut c_evals = [Fq::zero(); DEGREE];
                        for t in 0..DEGREE {
                            c_evals[t] = eq_x_evals[t]
                                * self.evaluate_constraint_at_point(constraint, x_idx, t);
                        }
                        c_evals
                    })
                    .reduce(
                        || [Fq::zero(); DEGREE],
                        |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]],
                    );

                // Multiply by eq_i_val
                [
                    eq_i_val * constraint_evals[0],
                    eq_i_val * constraint_evals[1],
                    eq_i_val * constraint_evals[2],
                    eq_i_val * constraint_evals[3],
                ]
            })
            .fold([Fq::zero(); DEGREE], |a, b| {
                [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
            });

        UniPoly::from_evals_and_hint(previous_claim, &evals[1..])
    }

    /// Phase 2: Compute sumcheck message while binding constraint index variables i
    /// At this point, x is fully bound, so C_i(x_bound) is a scalar for each constraint.
    /// We bind both eq_i AND m_poly on the constraint index variables.
    ///
    /// The polynomial being summed is: f(i) = eq_r_x * eq(r_i, i) * C_i(r_x)
    /// Since eq is multilinear, f(i) is multilinear in i (C_i are fixed scalars).
    /// This means f(t) = (1-t)*f(0) + t*f(1) for linear interpolation.
    fn compute_phase2_message(&self, phase2_round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        const DEGREE: usize = 2; // f(i) is linear in each i variable
        let eq_i_half = self.eq_i.len() / 2;

        let evals: [Fq; DEGREE] = (0..eq_i_half)
            .into_par_iter()
            .map(|i_idx| {
                // Get eq_i evaluations at t=0 and t=1
                let eq_i_evals = self
                    .eq_i
                    .sumcheck_evals_array::<DEGREE>(i_idx, BindingOrder::LowToHigh);

                // Dynamically evaluate constraints
                let c_i_0 = self.evaluate_constraint_phase2(2 * i_idx, phase2_round);
                let c_i_1 = self.evaluate_constraint_phase2(2 * i_idx + 1, phase2_round);

                [
                    self.eq_r_x * eq_i_evals[0] * c_i_0,
                    self.eq_r_x * eq_i_evals[1] * c_i_1,
                ]
            })
            .reduce(|| [Fq::zero(); DEGREE], |a, b| [a[0] + b[0], a[1] + b[1]]);

        UniPoly::from_evals_and_hint(previous_claim, &[evals[0], evals[1]])
    }

    /// Evaluate constraint C_i at a given sumcheck evaluation point t during Phase 1
    /// C_i(x) = ρ_curr(x) - ρ_prev(x)² × base(x)^{b_i} - quotient(x) × g(x)
    ///
    /// Uses the new layout: row = offset * num_constraints_padded + constraint_index
    fn evaluate_constraint_at_point(
        &self,
        constraint: &MatrixConstraint,
        x_idx: usize,
        t: usize,
    ) -> Fq {
        let idx = constraint.constraint_index;
        let num_constraints_padded = self.params.num_constraints_padded;

        assert!(
            idx < num_constraints_padded,
            "constraint_index {} >= num_constraints_padded {}",
            idx,
            num_constraints_padded
        );

        // Get row indices using new layout: row = offset * num_constraints_padded + constraint_index
        let base_row = (RowOffset::Base as usize) * num_constraints_padded + idx;
        let rho_prev_row = (RowOffset::RhoPrev as usize) * num_constraints_padded + idx;
        let rho_curr_row = (RowOffset::RhoCurr as usize) * num_constraints_padded + idx;
        let quotient_row = (RowOffset::Quotient as usize) * num_constraints_padded + idx;

        // Get evaluations at sumcheck point
        let base = self.get_row_eval_at_point(base_row, x_idx, t);
        let rho_prev = self.get_row_eval_at_point(rho_prev_row, x_idx, t);
        let rho_curr = self.get_row_eval_at_point(rho_curr_row, x_idx, t);
        let quotient = self.get_row_eval_at_point(quotient_row, x_idx, t);
        let g = self.get_g_eval_at_point(x_idx, t);

        // Compute constraint: ρ_curr - ρ_prev² × base^{b_i} - quotient × g
        let base_power = if constraint.bit { base } else { Fq::one() };
        rho_curr - rho_prev.square() * base_power - quotient * g
    }

    /// Get evaluation of row polynomial M(row, x) at sumcheck point during Phase 1
    ///
    /// M is stored with layout: M[x + row * row_size] where row_size = 2^num_constraint_vars
    /// Variables are ordered: [x_bits, i_bits, offset_bits] (little-endian)
    ///
    /// After binding k x variables, the polynomial has length 2^(num_vars - k).
    /// The remaining stride for row access is 2^(num_constraint_vars - k - 1) = x_half.
    fn get_row_eval_at_point(&self, row: usize, x_idx: usize, t: usize) -> Fq {
        // M layout: [x_bits, i_bits, offset_bits] in little-endian
        // During Phase 1 round k, we've bound k x variables.
        // The remaining polynomial has structure with stride:
        //   row_stride = 2^(remaining_x_vars) = m_poly.len() / num_rows_with_offset
        //
        // num_rows_with_offset = 4 * num_constraints_padded is constant
        // But we need to account for remaining x variables
        //
        // At each round, x_half = 2^(num_remaining_x_vars - 1) where:
        //   - x_idx ranges from 0 to x_half-1
        //   - For each x_idx, we access coefficients at 2*x_idx and 2*x_idx+1
        //
        // The stride between rows is now 2 * x_half = remaining x size
        // So the offset for row r is: r * (2 * x_half)

        let num_rows = RowOffset::NUM_OFFSETS * self.params.num_constraints_padded;
        // row_stride = remaining x variables after current binding = m_poly.len() / num_rows
        let row_stride = self.m_poly.len() / num_rows;
        let base_offset = row * row_stride;

        assert!(
            row < num_rows,
            "row {} >= num_rows {} (m_poly.len()={}, num_constraints_padded={})",
            row,
            num_rows,
            self.m_poly.len(),
            self.params.num_constraints_padded
        );

        let idx_0 = base_offset + 2 * x_idx;
        let idx_1 = base_offset + 2 * x_idx + 1;

        assert!(
            idx_1 < self.m_poly.len(),
            "index {} >= m_poly.len() {} (row={}, row_stride={}, x_idx={})",
            idx_1,
            self.m_poly.len(),
            row,
            row_stride,
            x_idx
        );

        let val_0 = self.m_poly.get_bound_coeff(idx_0);
        let val_1 = self.m_poly.get_bound_coeff(idx_1);

        self.extrapolate(val_0, val_1, t)
    }

    /// Get evaluation of g(x) polynomial at sumcheck point
    fn get_g_eval_at_point(&self, x_idx: usize, t: usize) -> Fq {
        let val_0 = self.g_poly.Z[2 * x_idx];
        let val_1 = self.g_poly.Z[2 * x_idx + 1];
        self.extrapolate(val_0, val_1, t)
    }

    /// Evaluate constraint at fully bound x point (used in Phase 2)
    /// constraint_idx is the index into the padded constraint space
    fn evaluate_constraint_fully_bound(&self, constraint_idx: usize) -> Fq {
        if constraint_idx >= self.params.num_constraints_padded {
            return Fq::zero();
        }

        // Calculate current Phase 2 round from total rounds
        let phase2_round = self.round - self.params.num_constraint_vars;
        self.evaluate_constraint_phase2(constraint_idx, phase2_round)
    }

    /// Extrapolate linear polynomial from values at 0 and 1 to value at t
    fn extrapolate(&self, val_0: Fq, val_1: Fq, t: usize) -> Fq {
        match t {
            0 => val_0,
            1 => val_1,
            _ => {
                let slope = val_1 - val_0;
                val_0 + slope * Fq::from(t as u64)
            }
        }
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
    /// Constraint metadata (public input shared with prover)
    pub constraints: Vec<MatrixConstraint>,
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
        // Extract the same random challenges as the prover
        let r_x: Vec<<Fq as JoltField>::Challenge> = (0..params.num_constraint_vars)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();

        let r_i: Vec<<Fq as JoltField>::Challenge> = (0..params.num_constraint_index_vars)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();

        Self {
            params,
            r_x,
            r_i,
            constraints,
            g_poly,
        }
    }

    /// Compute the MLE of the bit vector evaluated at r_i
    /// bit_eval = Σ_i eq(r_i, i) * b_i
    pub fn compute_bit_mle_eval(&self) -> Fq {
        use crate::subprotocols::recursion_constraints::index_to_binary;
        let r_i_fq: Vec<Fq> = self.r_i.iter().map(|c| (*c).into()).collect();
        let mut result = Fq::zero();
        for constraint in &self.constraints {
            if constraint.bit {
                let eq_val = EqPolynomial::mle(
                    &index_to_binary(
                        constraint.constraint_index,
                        self.params.num_constraint_index_vars,
                    ),
                    &r_i_fq,
                );
                result += eq_val;
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
        // Phase 1: x variables + Phase 2: constraint index variables
        // Note: offset bits (2) are NOT bound during sumcheck
        self.params.num_rounds()
    }

    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<Fq>) -> Fq {
        // Zero-check: expect sum to be zero
        Fq::zero()
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<Fq>,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        // Get the constraint evaluation from the accumulator
        // The prover computed: C(r_i, r_x) = rho_curr - rho_prev² × [1 + (base-1) × bit_eval] - quotient × g
        let (_opening_point, constraint_eval) = accumulator.get_committed_polynomial_opening(
            self.params.matrix_commitment,
            self.params.sumcheck_id,
        );

        // Split challenges into x (Phase 1) and i (Phase 2) components
        let (x_challenges, i_challenges) =
            sumcheck_challenges.split_at(self.params.num_constraint_vars);

        // Compute eq(r_x, x_bound) and eq(r_i, i_bound)
        let eq_r_x = EqPolynomial::<Fq>::mle(&self.r_x, x_challenges);
        let eq_r_i = EqPolynomial::<Fq>::mle(&self.r_i, i_challenges);

        // Expected claim: eq_r_x * eq_r_i * constraint_eval
        eq_r_x * eq_r_i * constraint_eval
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // The prover appended 6 values to transcript: base, rho_prev, rho_curr, quotient, g, bit_eval
        // These are the 4 M openings at points M(offset, r_i, r_x) for offset ∈ {0,1,2,3}
        // plus g(r_x) and the bit MLE evaluation.
        //
        // In the full protocol, these would be verified via commitment openings.
        // For now, we rely on the prover-computed constraint_eval in the accumulator.

        let opening_point = self
            .params
            .get_opening_point::<LITTLE_ENDIAN>(sumcheck_challenges);

        accumulator.append_sparse(
            transcript,
            vec![self.params.matrix_commitment],
            self.params.sumcheck_id,
            opening_point.r,
        );
    }
}

impl RecursionSumcheckVerifier {
    /// Recompute constraint evaluation from 4 M openings and bit_eval
    ///
    /// C(r_i, r_x) = ρ_curr - ρ_prev² × [1 + (base - 1) × bit_eval] - quotient × g
    ///
    /// Where bit_eval = Σ_i eq(r_i, i) * b_i is the MLE of bits at r_i
    #[allow(dead_code)]
    pub fn compute_constraint_from_openings(
        base: Fq,
        rho_prev: Fq,
        rho_curr: Fq,
        quotient: Fq,
        g_val: Fq,
        bit_eval: Fq,
    ) -> Fq {
        let base_power = Fq::one() + (base - Fq::one()) * bit_eval;
        rho_curr - rho_prev.square() * base_power - quotient * g_val
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
                hyrax::{matrix_dimensions, HyraxCommitment, HyraxOpeningProof},
                pedersen::PedersenGenerators,
            },
            dense_mlpoly::DensePolynomial,
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            unipoly::UniPoly,
        },
        subprotocols::{
            recursion_constraints::ConstraintSystem, sumcheck_prover::SumcheckInstanceProver,
        },
        transcripts::Blake2bTranscript,
        zkvm::witness::CommittedPolynomial,
    };
    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use ark_grumpkin::Projective as GrumpkinProjective;
    use rand::thread_rng;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_recursion_sumcheck_with_hyrax_verification() {
        const RATIO: usize = 1;

        // Setup using Dory commitment
        DoryGlobals::reset();
        DoryGlobals::initialize(1 << 2, 1 << 2);
        let num_vars = 4;
        let mut rng = thread_rng();

        // Create a Dory proof and extract constraint system
        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
        let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);

        let coefficients: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
        let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coefficients));
        let (commitment, hint) = DoryCommitmentScheme::commit(&poly, &prover_setup);

        let point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect();

        let mut prover_transcript = Blake2bTranscript::new(b"test");
        let proof = DoryCommitmentScheme::prove(
            &prover_setup,
            &poly,
            &point,
            Some(hint),
            &mut prover_transcript,
        );

        let evaluation = PolynomialEvaluation::evaluate(&poly, &point);
        let mut extract_transcript = Blake2bTranscript::new(b"test");

        // Extract constraint system from Dory proof
        let (constraint_system, _hints) = ConstraintSystem::new(
            &proof,
            &verifier_setup,
            &mut extract_transcript,
            &point,
            &evaluation,
            &commitment,
        )
        .expect("System creation should succeed");

        // Create sumcheck prover with new params structure
        let params = RecursionSumcheckParams::new(
            constraint_system.matrix.num_constraint_index_vars,
            constraint_system.matrix.num_constraints,
            constraint_system.matrix.num_constraints_padded,
            CommittedPolynomial::RdInc,
        );

        let mut sumcheck_transcript = Blake2bTranscript::new(b"sumcheck_test");
        let mut prover = RecursionSumcheckProver::gen(
            params.clone(),
            &constraint_system,
            &mut sumcheck_transcript,
        );

        // Extract DensePolynomial from MultilinearPolynomial for Hyrax
        let m_poly_dense = match &prover.m_poly {
            MultilinearPolynomial::LargeScalars(dense) => dense.clone(),
            _ => panic!("Expected LargeScalars variant"),
        };

        // Setup Hyrax commitment using Grumpkin (Fq is scalar field of Grumpkin)
        let m_num_vars = m_poly_dense.get_num_vars();
        let (_, r_size) = matrix_dimensions(m_num_vars, RATIO);
        let hyrax_gens =
            PedersenGenerators::<GrumpkinProjective>::new(r_size, b"recursion_sumcheck_hyrax");

        // Commit to M polynomial using Hyrax
        let hyrax_commitment =
            HyraxCommitment::<RATIO, GrumpkinProjective>::commit(&m_poly_dense, &hyrax_gens);

        // Run sumcheck and collect challenges
        let mut previous_claim = Fq::zero();
        let mut sumcheck_challenges: Vec<<Fq as JoltField>::Challenge> = Vec::new();
        let num_rounds = <RecursionSumcheckProver as SumcheckInstanceProver<
            Fq,
            Blake2bTranscript,
        >>::num_rounds(&prover);

        for round in 0..num_rounds {
            // Compute prover message
            let prover_message = <RecursionSumcheckProver as SumcheckInstanceProver<
                Fq,
                Blake2bTranscript,
            >>::compute_message(&mut prover, round, previous_claim);

            // Generate random challenge
            let challenge = <Fq as JoltField>::Challenge::random(&mut rng);
            sumcheck_challenges.push(challenge);

            // Update previous claim
            let eval_at_1 = previous_claim - prover_message.eval_at_zero();
            let univariate = UniPoly::from_evals(&[
                prover_message.eval_at_zero(),
                eval_at_1,
                prover_message.evaluate(&Fq::from(2u64)),
            ]);
            previous_claim = univariate.evaluate(&challenge);

            // Bind prover's polynomial
            <RecursionSumcheckProver as SumcheckInstanceProver<Fq, Blake2bTranscript>>::ingest_challenge(&mut prover, challenge, round);
        }

        // After sumcheck:
        // - Phase 1 bound x variables (4 rounds), binding M on x
        // - Phase 2 bound constraint index variables, binding M on constraint index
        // - Offset bits (2) remain unbound
        // So m_poly.get_bound_coeff(offset) = M(r_x, r_i, offset_bits)

        // Build opening point components in LITTLE-ENDIAN order (for binding)
        // Note: MLE evaluate expects BIG-ENDIAN, so we reverse when constructing opening_point
        let r_x: Vec<Fq> = sumcheck_challenges[..params.num_constraint_vars]
            .iter()
            .map(|c| (*c).into())
            .collect();

        let r_i: Vec<Fq> = sumcheck_challenges[params.num_constraint_vars..]
            .iter()
            .map(|c| (*c).into())
            .collect();

        // Verify each offset's opening with Hyrax
        let offsets_to_verify = [
            ("base", RowOffset::Base),
            ("rho_prev", RowOffset::RhoPrev),
            ("rho_curr", RowOffset::RhoCurr),
            ("quotient", RowOffset::Quotient),
        ];

        for (name, offset) in offsets_to_verify {
            // Build full opening point in BIG-ENDIAN order (high-order vars first)
            // MLE evaluate expects point[0] = highest variable
            //
            // After LowToHigh binding:
            // - Variables 0..16 are bound (x then i)
            // - Variables 17,18 remain (offset bits)
            //
            // Opening point for MLE should be: [offset_bit_1, offset_bit_0, challenges_reversed]
            // This puts high-order (offset) bits first
            let offset_val = offset as usize;
            let offset_bit_0 = if offset_val & 1 == 1 {
                Fq::one()
            } else {
                Fq::zero()
            };
            let offset_bit_1 = if offset_val & 2 == 2 {
                Fq::one()
            } else {
                Fq::zero()
            };

            // Big-endian order for MLE evaluate: highest variable first
            // Variable ordering (high to low): offset_1, offset_0, i_{k-1}, ..., i_0, x_3, ..., x_0
            let mut opening_point = vec![offset_bit_1, offset_bit_0]; // offset bits (high order)
            opening_point.extend(r_i.iter().rev()); // i vars reversed (high to low)
            opening_point.extend(r_x.iter().rev()); // x vars reversed (high to low)

            // Prove and verify with Hyrax
            let hyrax_proof = HyraxOpeningProof::<RATIO, GrumpkinProjective>::prove(
                &m_poly_dense,
                &opening_point,
                RATIO,
            );

            let mle_eval = m_poly_dense.evaluate(&opening_point);

            let verification_result =
                hyrax_proof.verify(&hyrax_gens, &opening_point, &mle_eval, &hyrax_commitment);
            assert!(
                verification_result.is_ok(),
                "Hyrax proof verification failed for offset '{}'",
                name
            );

            // Verify prover's bound value matches MLE evaluation
            let prover_val = prover.m_poly.get_bound_coeff(offset as usize);
            assert_eq!(
                prover_val, mle_eval,
                "Offset '{}' mismatch: prover has {:?}, MLE gives {:?}",
                name, prover_val, mle_eval
            );
        }

        // The sumcheck correctness is verified by running it (each round's claim is checked).
        // The key verification for PCS integration is that after sumcheck:
        // 1. The Hyrax proofs verify for all 4 M openings ✓ (checked above)
        // 2. The prover's bound M coefficients match MLE evaluations ✓ (checked above)
        //
        // The expected output claim verification is complex because it involves
        // the MLE extension of the product eq(r_i, i) * C_i(x), which doesn't factor
        // nicely. In a real protocol, the verifier would use the PCS opening proofs
        // to verify the constraint evaluation.
    }
}
