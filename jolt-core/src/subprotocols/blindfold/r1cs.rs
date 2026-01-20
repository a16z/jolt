//! BlindFold Verifier R1CS
//!
//! Builds sparse R1CS matrices for verifying sumcheck rounds.
//! The circuit is O(log n) in size - only encoding the verifier's algebraic checks.

use super::{Constraint, LinearCombination, OutputClaimConstraint, StageConfig, ValueSource, Variable};
use crate::field::JoltField;
use crate::poly::opening_proof::OpeningId;
use std::collections::HashMap;

/// Sparse R1CS matrix stored as (row, col, value) triplets
#[derive(Clone, Debug, Default)]
pub struct SparseR1CSMatrix<F> {
    pub entries: Vec<(usize, usize, F)>,
    pub num_rows: usize,
    pub num_cols: usize,
}

impl<F: JoltField> SparseR1CSMatrix<F> {
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        Self {
            entries: Vec::new(),
            num_rows,
            num_cols,
        }
    }

    pub fn push(&mut self, row: usize, col: usize, value: F) {
        debug_assert!(
            row < self.num_rows,
            "Row {} >= num_rows {}",
            row,
            self.num_rows
        );
        debug_assert!(
            col < self.num_cols,
            "Col {} >= num_cols {}",
            col,
            self.num_cols
        );
        if !value.is_zero() {
            self.entries.push((row, col, value));
        }
    }

    /// Compute matrix-vector product: result = M * z
    pub fn mul_vector(&self, z: &[F]) -> Vec<F> {
        debug_assert_eq!(z.len(), self.num_cols);
        let mut result = vec![F::zero(); self.num_rows];
        for &(row, col, ref value) in &self.entries {
            result[row] += *value * z[col];
        }
        result
    }
}

/// Verifier R1CS for BlindFold sumcheck verification
///
/// The witness vector Z is laid out as:
/// ```text
/// Z = [1, challenges..., initial_claim, witness...]
///      │  └────────────────────────────┘  └───────┘
///      │         public inputs            private witness
///      └─ constant 1 (index 0)
/// ```
///
/// Per sumcheck round, the witness contains:
/// - c0, c1, c2, c3: polynomial coefficients
/// - t1, t2: Horner intermediates
/// - next_claim: output (becomes next round's claimed_sum)
#[derive(Clone, Debug)]
pub struct VerifierR1CS<F: JoltField> {
    /// Sparse matrix A
    pub a: SparseR1CSMatrix<F>,
    /// Sparse matrix B
    pub b: SparseR1CSMatrix<F>,
    /// Sparse matrix C
    pub c: SparseR1CSMatrix<F>,
    /// Total number of variables in Z (including constant 1)
    pub num_vars: usize,
    /// Number of public input variables (challenges + initial claim)
    pub num_public_inputs: usize,
    /// Number of constraints
    pub num_constraints: usize,
    /// Stage configurations used to build this R1CS
    pub stage_configs: Vec<StageConfig>,
}

impl<F: JoltField> VerifierR1CS<F> {
    /// Check if the R1CS is satisfied: Az ∘ Bz = Cz
    pub fn is_satisfied(&self, z: &[F]) -> bool {
        assert_eq!(z.len(), self.num_vars);
        assert_eq!(z[0], F::one(), "Z[0] must be 1");

        let az = self.a.mul_vector(z);
        let bz = self.b.mul_vector(z);
        let cz = self.c.mul_vector(z);

        for row in 0..self.num_constraints {
            if az[row] * bz[row] != cz[row] {
                return false;
            }
        }
        true
    }

    /// Check satisfaction and return the first failing constraint index if any
    pub fn check_satisfaction(&self, z: &[F]) -> Result<(), usize> {
        assert_eq!(z.len(), self.num_vars);
        assert_eq!(z[0], F::one(), "Z[0] must be 1");

        let az = self.a.mul_vector(z);
        let bz = self.b.mul_vector(z);
        let cz = self.c.mul_vector(z);

        for row in 0..self.num_constraints {
            if az[row] * bz[row] != cz[row] {
                return Err(row);
            }
        }
        Ok(())
    }
}

/// Variables for final output binding constraint
#[derive(Clone, Debug)]
pub struct FinalOutputVariables {
    /// Batching coefficient variables (public inputs) - α_j (for simple linear constraints)
    pub batching_coeff_vars: Vec<Variable>,
    /// Expected evaluation variables (witness) - y_j (for simple linear constraints)
    pub evaluation_vars: Vec<Variable>,
    /// Mapping from OpeningId to witness variable (for general constraints)
    pub opening_vars: HashMap<OpeningId, Variable>,
    /// Challenge variables for the constraint (public inputs)
    pub constraint_challenge_vars: Vec<Variable>,
    /// Auxiliary variables for intermediate products
    pub aux_vars: Vec<Variable>,
}

/// Builder for constructing the verifier R1CS
pub struct VerifierR1CSBuilder<F: JoltField> {
    /// Constraints accumulated so far
    constraints: Vec<Constraint<F>>,
    /// Next variable index to allocate
    next_var: usize,
    /// Challenge variable indices (public inputs)
    challenge_vars: Vec<Variable>,
    /// Initial claim variables (public inputs) - one per independent chain
    initial_claim_vars: Vec<Variable>,
    /// Batching coefficient variables (public inputs) for final output constraints
    batching_coeff_vars: Vec<Variable>,
    /// Stage configurations
    stage_configs: Vec<StageConfig>,
    /// Mapping from (stage, round) to the round's variables
    round_vars: Vec<Vec<RoundVariables>>,
    /// Final output variables for each stage with final_output config
    final_output_vars: Vec<Option<FinalOutputVariables>>,
}

/// Variables allocated for a single sumcheck round
#[derive(Clone, Debug)]
pub struct RoundVariables {
    /// Polynomial coefficients c0, c1, c2, c3
    pub coeffs: Vec<Variable>,
    /// Horner intermediates (degree - 1 variables)
    pub intermediates: Vec<Variable>,
    /// Output claim (becomes next round's input)
    pub next_claim: Variable,
    /// Challenge for this round (public input)
    pub challenge: Variable,
    /// Input claim (from previous round or initial)
    pub claimed_sum: Variable,
}

impl<F: JoltField> VerifierR1CSBuilder<F> {
    /// Create a new builder for the given stage configurations
    pub fn new(stage_configs: &[StageConfig]) -> Self {
        let total_rounds: usize = stage_configs.iter().map(|s| s.num_rounds).sum();

        // Count independent chains (first stage + stages with starts_new_chain)
        let num_chains = 1 + stage_configs
            .iter()
            .skip(1)
            .filter(|s| s.starts_new_chain)
            .count();

        // Count total batching coefficients for simple final output constraints.
        // General constraints (with fo.constraint.is_some()) handle their own variables.
        let total_batching_coeffs: usize = stage_configs
            .iter()
            .filter_map(|s| s.final_output.as_ref())
            .filter(|fo| fo.constraint.is_none()) // Only simple constraints
            .map(|fo| fo.num_evaluations)
            .sum();

        // Allocate public inputs: challenges + initial claims + batching coefficients
        // Index 0 is the constant 1 (u scalar)
        // Indices 1..=total_rounds are challenges
        // Indices total_rounds+1..=total_rounds+num_chains are initial claims
        // Indices total_rounds+num_chains+1..=... are batching coefficients
        let challenge_vars: Vec<Variable> = (1..=total_rounds).map(Variable::new).collect();
        let initial_claim_vars: Vec<Variable> = (0..num_chains)
            .map(|i| Variable::new(total_rounds + 1 + i))
            .collect();
        let batching_coeff_vars: Vec<Variable> = (0..total_batching_coeffs)
            .map(|i| Variable::new(total_rounds + 1 + num_chains + i))
            .collect();
        let next_var = total_rounds + 1 + num_chains + total_batching_coeffs;

        Self {
            constraints: Vec::new(),
            next_var,
            challenge_vars,
            initial_claim_vars,
            batching_coeff_vars,
            stage_configs: stage_configs.to_vec(),
            round_vars: Vec::new(),
            final_output_vars: Vec::new(),
        }
    }

    /// Add a constraint: a * b = c
    fn add_constraint(
        &mut self,
        a: LinearCombination<F>,
        b: LinearCombination<F>,
        c: LinearCombination<F>,
    ) {
        self.constraints.push(Constraint::new(a, b, c));
    }

    /// Build constraints for a single sumcheck round (generic degree)
    ///
    /// For g(X) = c0 + c1*X + c2*X^2 + ... + cd*X^d:
    ///
    /// 1. Sum check: g(0) + g(1) = claimed_sum
    ///    g(0) = c0
    ///    g(1) = c0 + c1 + c2 + ... + cd
    ///    => 2*c0 + c1 + c2 + ... + cd = claimed_sum
    ///
    /// 2. Horner evaluation: next_claim = g(r) where r is the challenge
    ///    Using Horner's method: g(r) = c0 + r*(c1 + r*(c2 + ... + r*cd))
    ///
    ///    For degree d, we need d-1 intermediate variables.
    ///    Let's denote intermediates as t[0], t[1], ..., t[d-2]
    ///
    ///    Constraints (for degree d >= 2):
    ///    - cd * r = t[d-2] - c_{d-1}       (first: t[d-2] = c_{d-1} + r*cd)
    ///    - t[i] * r = t[i-1] - c_i         (middle: for i from d-2 down to 1)
    ///    - t[0] * r = next_claim - c0      (final)
    ///
    ///    For degree 1 (linear), no intermediates needed:
    ///    - c1 * r = next_claim - c0
    fn add_round_constraints(
        &mut self,
        vars: &RoundVariables,
        uniskip_power_sums: Option<&[i128]>,
    ) {
        let RoundVariables {
            coeffs,
            intermediates,
            next_claim,
            challenge,
            claimed_sum,
        } = vars;

        let degree = coeffs.len() - 1; // d coefficients means degree d-1

        // Constraint 1: Sum check
        let a = if let Some(power_sums) = uniskip_power_sums {
            // Uni-skip: Σ_j coeff[j] * PowerSum[j] = claimed_sum
            let mut lc = LinearCombination::new();
            for (j, &coeff_var) in coeffs.iter().enumerate() {
                if power_sums[j] != 0 {
                    lc = lc.add_term(coeff_var, F::from_i128(power_sums[j]));
                }
            }
            lc
        } else {
            // Standard sumcheck: 2*c0 + c1 + c2 + ... + cd = claimed_sum
            let mut lc = LinearCombination::new().add_term(coeffs[0], F::from_u64(2));
            for i in 1..coeffs.len() {
                lc = lc.add_var(coeffs[i]);
            }
            lc
        };
        let b = LinearCombination::constant(F::one());
        let c_lc = LinearCombination::variable(*claimed_sum);
        self.add_constraint(a, b, c_lc);

        // Horner evaluation constraints
        if degree == 0 {
            // Constant polynomial: next_claim = c0 (no multiplication needed)
            // This is just 1 * 1 = next_claim - c0 + c0, but we need a proper constraint
            // Actually for degree 0, next_claim = c0 always, so:
            // 1 * c0 = next_claim
            let a = LinearCombination::constant(F::one());
            let b = LinearCombination::variable(coeffs[0]);
            let c_lc = LinearCombination::variable(*next_claim);
            self.add_constraint(a, b, c_lc);
        } else if degree == 1 {
            // Linear: g(r) = c0 + c1*r
            // c1 * r = next_claim - c0
            let a = LinearCombination::variable(coeffs[1]);
            let b = LinearCombination::variable(*challenge);
            let c_lc = LinearCombination::variable(*next_claim).sub_var(coeffs[0]);
            self.add_constraint(a, b, c_lc);
        } else {
            // Degree >= 2: use Horner's method with intermediates
            // intermediates[i] corresponds to the accumulated value at step i
            // We build from highest degree down:
            // t[d-2] = c_{d-1} + r * c_d
            // t[i-1] = c_i + r * t[i] for i from d-2 down to 1
            // next_claim = c0 + r * t[0]

            // First constraint: cd * r = t[d-2] - c_{d-1}
            let cd = coeffs[degree];
            let cd_minus_1 = coeffs[degree - 1];
            let t_last = intermediates[degree - 2];

            let a = LinearCombination::variable(cd);
            let b = LinearCombination::variable(*challenge);
            let c_lc = LinearCombination::variable(t_last).sub_var(cd_minus_1);
            self.add_constraint(a, b, c_lc);

            // Middle constraints: t[i] * r = t[i-1] - c_i for i from d-2 down to 1
            // t[i] represents the accumulated Horner value up to coefficient i+1
            // So the constraint encodes: t[i-1] = c_i + r * t[i]
            for i in (1..degree - 1).rev() {
                let t_curr = intermediates[i];
                let t_prev = intermediates[i - 1];
                let ci = coeffs[i]; // c_i is the coefficient we're adding at this step

                let a = LinearCombination::variable(t_curr);
                let b = LinearCombination::variable(*challenge);
                let c_lc = LinearCombination::variable(t_prev).sub_var(ci);
                self.add_constraint(a, b, c_lc);
            }

            // Final constraint: t[0] * r = next_claim - c0
            let t0 = intermediates[0];
            let c0 = coeffs[0];

            let a = LinearCombination::variable(t0);
            let b = LinearCombination::variable(*challenge);
            let c_lc = LinearCombination::variable(*next_claim).sub_var(c0);
            self.add_constraint(a, b, c_lc);
        }
    }

    /// Build the complete verifier R1CS
    pub fn build(mut self) -> VerifierR1CS<F> {
        let mut challenge_idx = 0;
        let mut chain_idx = 0;
        let mut batching_coeff_idx = 0;
        let mut current_claim = self.initial_claim_vars[chain_idx];

        // Clone data to avoid borrow conflict
        let stage_configs = self.stage_configs.clone();
        let batching_coeff_vars = self.batching_coeff_vars.clone();

        // Allocate variables and build constraints for each stage/round
        for (stage_idx, config) in stage_configs.iter().enumerate() {
            // Check if this stage starts a new chain
            if stage_idx > 0 && config.starts_new_chain {
                chain_idx += 1;
                current_claim = self.initial_claim_vars[chain_idx];
            }

            let mut stage_rounds = Vec::with_capacity(config.num_rounds);

            for _round in 0..config.num_rounds {
                // Allocate witness variables for this round
                let num_coeffs = config.poly_degree + 1;
                let num_intermediates = config.poly_degree - 1;

                let coeffs: Vec<Variable> = (0..num_coeffs)
                    .map(|_| {
                        let var = Variable::new(self.next_var);
                        self.next_var += 1;
                        var
                    })
                    .collect();

                let intermediates: Vec<Variable> = (0..num_intermediates)
                    .map(|_| {
                        let var = Variable::new(self.next_var);
                        self.next_var += 1;
                        var
                    })
                    .collect();

                let next_claim = Variable::new(self.next_var);
                self.next_var += 1;

                let vars = RoundVariables {
                    coeffs,
                    intermediates,
                    next_claim,
                    challenge: self.challenge_vars[challenge_idx],
                    claimed_sum: current_claim,
                };

                // Add constraints for this round
                let power_sums = config.uniskip_power_sums.as_deref();
                self.add_round_constraints(&vars, power_sums);

                // Chain: this round's output becomes next round's input
                current_claim = next_claim;
                challenge_idx += 1;

                stage_rounds.push(vars);
            }

            // Add final output constraint if configured
            let final_output_vars = if let Some(ref fo_config) = config.final_output {
                let last_round = stage_rounds
                    .last()
                    .expect("Stage must have at least one round");

                if let Some(ref constraint) = fo_config.constraint {
                    // General sum-of-products constraint

                    // Allocate witness variables for required openings
                    let mut opening_vars_map: HashMap<OpeningId, Variable> = HashMap::new();
                    for opening_id in &constraint.required_openings {
                        if !opening_vars_map.contains_key(opening_id) {
                            let var = Variable::new(self.next_var);
                            self.next_var += 1;
                            opening_vars_map.insert(*opening_id, var);
                        }
                    }

                    // Allocate public input variables for challenges needed by the constraint
                    let constraint_challenge_vars: Vec<Variable> = (0..constraint.num_challenges)
                        .map(|_| {
                            let var = Variable::new(self.next_var);
                            self.next_var += 1;
                            var
                        })
                        .collect();

                    // Add the sum-of-products constraint
                    let aux_vars = self.add_sum_of_products_constraint(
                        last_round.next_claim,
                        constraint,
                        &opening_vars_map,
                        &constraint_challenge_vars,
                    );

                    Some(FinalOutputVariables {
                        batching_coeff_vars: Vec::new(),
                        evaluation_vars: Vec::new(),
                        opening_vars: opening_vars_map,
                        constraint_challenge_vars,
                        aux_vars,
                    })
                } else {
                    // Simple linear constraint: final_claim = Σⱼ αⱼ · yⱼ
                    let num_evals = fo_config.num_evaluations;

                    // Get batching coefficient variables (public inputs)
                    let coeff_vars: Vec<Variable> = batching_coeff_vars
                        [batching_coeff_idx..batching_coeff_idx + num_evals]
                        .to_vec();
                    batching_coeff_idx += num_evals;

                    // Allocate witness variables for expected evaluations
                    let eval_vars: Vec<Variable> = (0..num_evals)
                        .map(|_| {
                            let var = Variable::new(self.next_var);
                            self.next_var += 1;
                            var
                        })
                        .collect();

                    // Add final output constraint: final_claim = Σⱼ αⱼ · yⱼ
                    self.add_final_output_constraint(last_round.next_claim, &coeff_vars, &eval_vars);

                    Some(FinalOutputVariables {
                        batching_coeff_vars: coeff_vars,
                        evaluation_vars: eval_vars,
                        opening_vars: HashMap::new(),
                        constraint_challenge_vars: Vec::new(),
                        aux_vars: Vec::new(),
                    })
                }
            } else {
                None
            };

            self.round_vars.push(stage_rounds);
            self.final_output_vars.push(final_output_vars);
        }

        let num_constraints = self.constraints.len();
        let num_vars = self.next_var;
        // Public inputs: challenges + initial_claims + batching_coeffs
        let num_public_inputs = self.challenge_vars.len()
            + self.initial_claim_vars.len()
            + self.batching_coeff_vars.len();

        // Build sparse matrices
        let mut a = SparseR1CSMatrix::new(num_constraints, num_vars);
        let mut b = SparseR1CSMatrix::new(num_constraints, num_vars);
        let mut c = SparseR1CSMatrix::new(num_constraints, num_vars);

        for (row, constraint) in self.constraints.iter().enumerate() {
            for term in &constraint.a.terms {
                a.push(row, term.var.index(), term.coeff);
            }
            for term in &constraint.b.terms {
                b.push(row, term.var.index(), term.coeff);
            }
            for term in &constraint.c.terms {
                c.push(row, term.var.index(), term.coeff);
            }
        }

        VerifierR1CS {
            a,
            b,
            c,
            num_vars,
            num_public_inputs,
            num_constraints,
            stage_configs: self.stage_configs,
        }
    }

    /// Add final output constraint: final_claim = Σⱼ αⱼ · yⱼ
    ///
    /// This constraint binds the sumcheck's final output to the expected polynomial
    /// evaluations. The evaluations yⱼ are witness variables proven correct via ZK-Dory.
    fn add_final_output_constraint(
        &mut self,
        final_claim: Variable,
        batching_coeffs: &[Variable],
        evaluations: &[Variable],
    ) {
        debug_assert_eq!(
            batching_coeffs.len(),
            evaluations.len(),
            "Batching coefficients and evaluations must have same length"
        );

        // We want: final_claim = Σⱼ αⱼ · yⱼ
        // But R1CS requires A * B = C form.
        //
        // For a single evaluation (n=1): α₀ * y₀ = final_claim
        // For multiple evaluations: we need auxiliary variables.
        //
        // Strategy: Add one constraint per evaluation using accumulator pattern.
        // Let acc_0 = α₀ · y₀
        // Let acc_j = acc_{j-1} + αⱼ · yⱼ for j > 0
        // Final: acc_{n-1} = final_claim

        let n = batching_coeffs.len();
        if n == 0 {
            return;
        }

        if n == 1 {
            // Single evaluation: α₀ * y₀ = final_claim
            let a = LinearCombination::variable(batching_coeffs[0]);
            let b = LinearCombination::variable(evaluations[0]);
            let c = LinearCombination::variable(final_claim);
            self.add_constraint(a, b, c);
            return;
        }

        // Multiple evaluations: use accumulator variables
        // We need n-1 accumulator variables for n evaluations
        let mut accumulators: Vec<Variable> = Vec::with_capacity(n - 1);
        for _ in 0..n - 1 {
            let var = Variable::new(self.next_var);
            self.next_var += 1;
            accumulators.push(var);
        }

        // First constraint: α₀ * y₀ = acc₀
        let a = LinearCombination::variable(batching_coeffs[0]);
        let b = LinearCombination::variable(evaluations[0]);
        let c = LinearCombination::variable(accumulators[0]);
        self.add_constraint(a, b, c);

        // Middle constraints: αⱼ * yⱼ = accⱼ - acc_{j-1}
        for j in 1..n - 1 {
            let a = LinearCombination::variable(batching_coeffs[j]);
            let b = LinearCombination::variable(evaluations[j]);
            let c = LinearCombination::variable(accumulators[j]).sub_var(accumulators[j - 1]);
            self.add_constraint(a, b, c);
        }

        // Final constraint: α_{n-1} * y_{n-1} = final_claim - acc_{n-2}
        let a = LinearCombination::variable(batching_coeffs[n - 1]);
        let b = LinearCombination::variable(evaluations[n - 1]);
        let c = LinearCombination::variable(final_claim).sub_var(accumulators[n - 2]);
        self.add_constraint(a, b, c);
    }

    /// Add general sum-of-products constraint: final_claim = Σᵢ coeffᵢ * ∏ⱼ factorᵢⱼ
    ///
    /// This method handles the general constraint form described by `OutputClaimConstraint`.
    /// It allocates auxiliary variables for intermediate products and generates R1CS constraints.
    ///
    /// Returns the auxiliary variables allocated for intermediate products.
    fn add_sum_of_products_constraint(
        &mut self,
        final_claim: Variable,
        constraint: &OutputClaimConstraint,
        opening_vars: &HashMap<OpeningId, Variable>,
        challenge_vars: &[Variable],
    ) -> Vec<Variable> {
        let mut aux_vars = Vec::new();

        if constraint.terms.is_empty() {
            return aux_vars;
        }

        // Helper to resolve a ValueSource to a LinearCombination
        let resolve_value = |vs: &ValueSource| -> LinearCombination<F> {
            match vs {
                ValueSource::Opening(id) => {
                    LinearCombination::variable(*opening_vars.get(id).unwrap_or_else(|| {
                        panic!("Opening {id:?} not found in variable map")
                    }))
                }
                ValueSource::Challenge(idx) => {
                    LinearCombination::variable(challenge_vars[*idx])
                }
                ValueSource::Constant(val) => {
                    LinearCombination::constant(F::from_i128(*val))
                }
            }
        };

        // For each term, compute the product and collect the result variable
        let mut term_results: Vec<(LinearCombination<F>, Variable)> = Vec::with_capacity(constraint.terms.len());

        for term in &constraint.terms {
            let coeff_lc = resolve_value(&term.coeff);

            if term.factors.is_empty() {
                // No factors: the term is just the coefficient
                // Allocate aux var for the coefficient value
                let aux = Variable::new(self.next_var);
                self.next_var += 1;
                aux_vars.push(aux);

                // Constraint: coeff * u = aux (where u is the scalar, 1 for non-relaxed)
                let a = coeff_lc;
                let b = LinearCombination::constant(F::one());
                let c = LinearCombination::variable(aux);
                self.add_constraint(a, b, c);

                term_results.push((LinearCombination::constant(F::one()), aux));
            } else if term.factors.len() == 1 {
                // Single factor: product = coeff * factor
                let factor_lc = resolve_value(&term.factors[0]);

                let aux = Variable::new(self.next_var);
                self.next_var += 1;
                aux_vars.push(aux);

                // Constraint: coeff * factor = aux
                self.add_constraint(coeff_lc, factor_lc, LinearCombination::variable(aux));

                term_results.push((LinearCombination::constant(F::one()), aux));
            } else {
                // Multiple factors: use chain of multiplications
                // First, compute the product of all factors
                // Then multiply by the coefficient

                // Product of factors: f0 * f1 * ... * fn
                let factor0_lc = resolve_value(&term.factors[0]);
                let factor1_lc = resolve_value(&term.factors[1]);

                // First multiplication: aux0 = f0 * f1
                let mut current_product = Variable::new(self.next_var);
                self.next_var += 1;
                aux_vars.push(current_product);

                self.add_constraint(
                    factor0_lc,
                    factor1_lc,
                    LinearCombination::variable(current_product),
                );

                // Chain remaining factors: aux_{i} = aux_{i-1} * f_{i+1}
                for factor in &term.factors[2..] {
                    let factor_lc = resolve_value(factor);
                    let next_product = Variable::new(self.next_var);
                    self.next_var += 1;
                    aux_vars.push(next_product);

                    self.add_constraint(
                        LinearCombination::variable(current_product),
                        factor_lc,
                        LinearCombination::variable(next_product),
                    );

                    current_product = next_product;
                }

                // Now multiply by coefficient: final_term = coeff * product
                let final_term = Variable::new(self.next_var);
                self.next_var += 1;
                aux_vars.push(final_term);

                self.add_constraint(
                    coeff_lc,
                    LinearCombination::variable(current_product),
                    LinearCombination::variable(final_term),
                );

                term_results.push((LinearCombination::constant(F::one()), final_term));
            }
        }

        // Sum all term results: final_claim = Σᵢ term_results[i]
        // We use the same accumulator pattern as add_final_output_constraint
        let n = term_results.len();

        if n == 1 {
            // Single term: final_claim = term_result
            // Constraint: 1 * term = final_claim
            let (_, term_var) = &term_results[0];
            let a = LinearCombination::constant(F::one());
            let b = LinearCombination::variable(*term_var);
            let c = LinearCombination::variable(final_claim);
            self.add_constraint(a, b, c);
        } else {
            // Multiple terms: use accumulator pattern
            // But since all coefficients in term_results are 1, we just need to sum the variables
            // acc_0 = term_0
            // acc_1 = acc_0 + term_1
            // ...
            // final_claim = acc_{n-1}

            // Build linear combination for the sum
            let mut sum_lc = LinearCombination::new();
            for (_, term_var) in &term_results {
                sum_lc = sum_lc.add_var(*term_var);
            }

            // Constraint: 1 * sum = final_claim
            let a = LinearCombination::constant(F::one());
            let b = sum_lc;
            let c = LinearCombination::variable(final_claim);
            self.add_constraint(a, b, c);
        }

        aux_vars
    }

    /// Get the round variables for witness assignment
    pub fn get_round_vars(&self) -> &Vec<Vec<RoundVariables>> {
        &self.round_vars
    }

    /// Get the initial claim variable indices (one per chain)
    pub fn initial_claim_vars(&self) -> &[Variable] {
        &self.initial_claim_vars
    }

    /// Get the challenge variable indices
    pub fn challenge_vars(&self) -> &[Variable] {
        &self.challenge_vars
    }

    /// Get the batching coefficient variable indices (public inputs)
    pub fn batching_coeff_vars(&self) -> &[Variable] {
        &self.batching_coeff_vars
    }

    /// Get the final output variables for each stage
    pub fn final_output_vars(&self) -> &[Option<FinalOutputVariables>] {
        &self.final_output_vars
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subprotocols::blindfold::witness::{BlindFoldWitness, RoundWitness, StageWitness};
    use ark_bn254::Fr;

    #[test]
    fn test_single_round_constraint_satisfaction() {
        type F = Fr;

        let configs = [super::super::StageConfig::new(1, 3)];
        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();

        // Set up a valid sumcheck round
        // claimed_sum = g(0) + g(1) = 2*c0 + c1 + c2 + c3 = 110
        let c0 = F::from_u64(40); // 2*40 = 80
        let c1 = F::from_u64(10);
        let c2 = F::from_u64(15);
        let c3 = F::from_u64(5); // 80 + 10 + 15 + 5 = 110
        let initial_claim = F::from_u64(110);
        let r = F::from_u64(7);

        let round = RoundWitness::new(vec![c0, c1, c2, c3], r);
        let witness = BlindFoldWitness::new(initial_claim, vec![StageWitness::new(vec![round])]);

        let z = witness.assign(&r1cs);
        assert!(r1cs.is_satisfied(&z), "R1CS should be satisfied");
    }

    #[test]
    fn test_multi_round_chaining() {
        type F = Fr;

        let configs = [super::super::StageConfig::new(2, 3)];
        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();

        assert_eq!(r1cs.num_constraints, 8); // 4 per round

        // Round 1: 2*c0 + c1 + c2 + c3 = 55
        let c0_1 = F::from_u64(20); // 2*20 = 40
        let c1_1 = F::from_u64(5);
        let c2_1 = F::from_u64(7);
        let c3_1 = F::from_u64(3); // 40 + 5 + 7 + 3 = 55
        let initial_claim = F::from_u64(55);
        let r1 = F::from_u64(3);

        let round1 = RoundWitness::new(vec![c0_1, c1_1, c2_1, c3_1], r1);
        let next1 = round1.evaluate(r1);

        // Round 2: 2*c0 + c1 + c2 + c3 = next1
        // next1 = 20 + 3*5 + 9*7 + 27*3 = 20 + 15 + 63 + 81 = 179
        // We need 2*c0 + c1 + c2 + c3 = 179
        // Let's use c0 = 85, c1 = 4, c2 = 3, c3 = 2 => 170 + 4 + 3 + 2 = 179
        let c0_2 = F::from_u64(85);
        let c1_2 = F::from_u64(4);
        let c2_2 = F::from_u64(3);
        let c3_2 = F::from_u64(2);
        let r2 = F::from_u64(5);

        let round2 = RoundWitness::new(vec![c0_2, c1_2, c2_2, c3_2], r2);

        // Verify the chain
        let claim2 = F::from_u64(2) * c0_2 + c1_2 + c2_2 + c3_2;
        assert_eq!(claim2, next1, "Round 2 claim should equal round 1 output");

        let witness =
            BlindFoldWitness::new(initial_claim, vec![StageWitness::new(vec![round1, round2])]);

        let z = witness.assign(&r1cs);
        match r1cs.check_satisfaction(&z) {
            Ok(()) => {}
            Err(row) => panic!("Constraint {row} failed"),
        }
    }

    #[test]
    fn test_constraint_count() {
        type F = Fr;

        // 6 stages, 20 rounds each
        let configs: Vec<_> = (0..6)
            .map(|_| super::super::StageConfig::new(20, 3))
            .collect();
        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();

        // 4 constraints per round, 6 stages * 20 rounds = 120 rounds
        assert_eq!(r1cs.num_constraints, 4 * 6 * 20);
        assert_eq!(r1cs.num_constraints, 480);

        // Public inputs: 120 challenges + 1 initial claim = 121
        assert_eq!(r1cs.num_public_inputs, 121);

        // Variables: 1 (constant) + 121 (public) + 120 * 7 (witness per round)
        // Per round: 4 coeffs + 2 intermediates + 1 next_claim = 7
        assert_eq!(r1cs.num_vars, 1 + 121 + 120 * 7);
    }

    #[test]
    fn test_invalid_witness_fails() {
        type F = Fr;

        let configs = [super::super::StageConfig::new(1, 3)];
        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();

        // Create invalid witness where sum check fails
        // 2*c0 + c1 + c2 + c3 = 100 (computed claimed_sum)
        // But we'll use initial_claim = 200, which doesn't match
        let c0 = F::from_u64(40);
        let c1 = F::from_u64(10);
        let c2 = F::from_u64(5);
        let c3 = F::from_u64(5); // 80 + 10 + 5 + 5 = 100
        let initial_claim = F::from_u64(200); // Doesn't match the coefficients!
        let r = F::from_u64(3);

        // Use with_claimed_sum to explicitly set a claimed_sum that won't match
        let round = RoundWitness::with_claimed_sum(vec![c0, c1, c2, c3], r, F::from_u64(200));

        // The assignment will succeed, but R1CS should NOT be satisfied
        // because the sum check constraint (2*c0 + c1 + c2 + c3 = claimed_sum)
        // will evaluate to 100 = 200, which is false
        let witness = BlindFoldWitness::new(initial_claim, vec![StageWitness::new(vec![round])]);
        let z = witness.assign(&r1cs);

        // The R1CS should fail on the sum check constraint (constraint 0)
        assert!(
            r1cs.check_satisfaction(&z).is_err(),
            "R1CS should NOT be satisfied with invalid witness"
        );
    }

    #[test]
    fn test_uniskip_constraint_satisfaction() {
        type F = Fr;

        // Symmetric domain {-2, -1, 1, 2} has power sums:
        // PowerSum[0] = 4, PowerSum[1] = 0, PowerSum[2] = 10, PowerSum[3] = 0
        let power_sums = vec![4, 0, 10, 0];

        let configs = [super::super::StageConfig::new_uniskip(3, power_sums)];
        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();

        // Uni-skip sum constraint: c0*4 + c1*0 + c2*10 + c3*0 = claimed_sum
        // => 4*c0 + 10*c2 = claimed_sum
        // With c0=5, c2=3: 4*5 + 10*3 = 20 + 30 = 50
        let c0 = F::from_u64(5);
        let c1 = F::from_u64(7); // ignored in sum
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(9); // ignored in sum
        let initial_claim = F::from_u64(50);
        let r = F::from_u64(2);

        let round = RoundWitness::new(vec![c0, c1, c2, c3], r);
        let witness = BlindFoldWitness::new(initial_claim, vec![StageWitness::new(vec![round])]);

        let z = witness.assign(&r1cs);
        match r1cs.check_satisfaction(&z) {
            Ok(()) => {}
            Err(row) => panic!("Constraint {row} failed"),
        }
    }

    #[test]
    fn test_uniskip_invalid_witness_fails() {
        type F = Fr;

        // Same power sums as above
        let power_sums = vec![4, 0, 10, 0];

        let configs = [super::super::StageConfig::new_uniskip(3, power_sums)];
        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();

        // Correct sum would be 4*5 + 10*3 = 50, but we use initial_claim = 100
        let c0 = F::from_u64(5);
        let c1 = F::from_u64(7);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(9);
        let r = F::from_u64(2);

        let round = RoundWitness::with_claimed_sum(vec![c0, c1, c2, c3], r, F::from_u64(100));
        let witness = BlindFoldWitness::new(F::from_u64(100), vec![StageWitness::new(vec![round])]);

        let z = witness.assign(&r1cs);
        assert!(
            r1cs.check_satisfaction(&z).is_err(),
            "R1CS should NOT be satisfied with invalid uni-skip witness"
        );
    }

    #[test]
    fn test_final_output_constraint_single_eval() {
        use crate::subprotocols::blindfold::witness::FinalOutputWitness;

        type F = Fr;

        // Stage with final output constraint (1 evaluation)
        let config = super::super::StageConfig::new(1, 3).with_final_output(1);
        let builder = VerifierR1CSBuilder::<F>::new(&[config]);
        let r1cs = builder.build();

        // 4 constraints per round + 1 final output constraint = 5
        assert_eq!(r1cs.num_constraints, 5);

        // Set up valid round: 2*c0 + c1 + c2 + c3 = 100
        let c0 = F::from_u64(40);
        let c1 = F::from_u64(5);
        let c2 = F::from_u64(10);
        let c3 = F::from_u64(5);
        let initial_claim = F::from_u64(100);
        let r = F::from_u64(3);

        let round = RoundWitness::new(vec![c0, c1, c2, c3], r);
        let final_claim = round.evaluate(r);

        // Final output constraint: α * y = final_claim
        let alpha = F::from_u64(1);
        let y = final_claim; // Single eval means y = final_claim / alpha

        let fo_witness = FinalOutputWitness::new(vec![alpha], vec![y]);
        let stage = StageWitness::with_final_output(vec![round], fo_witness);
        let witness = BlindFoldWitness::new(initial_claim, vec![stage]);

        let z = witness.assign(&r1cs);
        match r1cs.check_satisfaction(&z) {
            Ok(()) => {}
            Err(row) => panic!("Constraint {row} failed"),
        }
    }

    #[test]
    fn test_final_output_constraint_multiple_evals() {
        use crate::subprotocols::blindfold::witness::FinalOutputWitness;

        type F = Fr;

        // Stage with final output constraint (3 evaluations)
        let config = super::super::StageConfig::new(1, 3).with_final_output(3);
        let builder = VerifierR1CSBuilder::<F>::new(&[config]);
        let r1cs = builder.build();

        // 4 constraints per round + 3 final output constraints = 7
        assert_eq!(r1cs.num_constraints, 7);

        // Set up valid round
        let c0 = F::from_u64(40);
        let c1 = F::from_u64(5);
        let c2 = F::from_u64(10);
        let c3 = F::from_u64(5);
        let initial_claim = F::from_u64(100);
        let r = F::from_u64(3);

        let round = RoundWitness::new(vec![c0, c1, c2, c3], r);
        let final_claim = round.evaluate(r);

        // Final output constraint: α₀*y₀ + α₁*y₁ + α₂*y₂ = final_claim
        let alpha0 = F::from_u64(2);
        let alpha1 = F::from_u64(3);
        let alpha2 = F::from_u64(5);
        // Choose y values such that α₀*y₀ + α₁*y₁ + α₂*y₂ = final_claim
        let y0 = F::from_u64(10);
        let y1 = F::from_u64(20);
        // y2 = (final_claim - α₀*y₀ - α₁*y₁) / α₂
        let partial_sum = alpha0 * y0 + alpha1 * y1;
        let y2 = (final_claim - partial_sum) * alpha2.inverse().unwrap();

        let fo_witness = FinalOutputWitness::new(vec![alpha0, alpha1, alpha2], vec![y0, y1, y2]);
        let stage = StageWitness::with_final_output(vec![round], fo_witness);
        let witness = BlindFoldWitness::new(initial_claim, vec![stage]);

        let z = witness.assign(&r1cs);
        match r1cs.check_satisfaction(&z) {
            Ok(()) => {}
            Err(row) => panic!("Constraint {row} failed"),
        }
    }

    #[test]
    fn test_final_output_constraint_invalid_fails() {
        use crate::subprotocols::blindfold::witness::FinalOutputWitness;

        type F = Fr;

        let config = super::super::StageConfig::new(1, 3).with_final_output(1);
        let builder = VerifierR1CSBuilder::<F>::new(&[config]);
        let r1cs = builder.build();

        // Set up valid round
        let c0 = F::from_u64(40);
        let c1 = F::from_u64(5);
        let c2 = F::from_u64(10);
        let c3 = F::from_u64(5);
        let initial_claim = F::from_u64(100);
        let r = F::from_u64(3);

        let round = RoundWitness::new(vec![c0, c1, c2, c3], r);
        let final_claim = round.evaluate(r);

        // Invalid: y doesn't satisfy α * y = final_claim
        let alpha = F::from_u64(1);
        let y = final_claim + F::from_u64(1); // Wrong value!

        let fo_witness = FinalOutputWitness::new(vec![alpha], vec![y]);
        let stage = StageWitness::with_final_output(vec![round], fo_witness);
        let witness = BlindFoldWitness::new(initial_claim, vec![stage]);

        let z = witness.assign(&r1cs);
        assert!(
            r1cs.check_satisfaction(&z).is_err(),
            "R1CS should NOT be satisfied with invalid final output"
        );
    }
}
