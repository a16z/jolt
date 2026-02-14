use super::{
    compute_hyrax_params, BakedPublicInputs, Constraint, HyraxParams, LinearCombination,
    OutputClaimConstraint, StageConfig, ValueSource, Variable,
};
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
/// Public inputs are baked into matrix coefficients, so the witness vector Z is:
/// ```text
/// Z = [u, witness_grid...]
/// ```
///
/// The witness grid has R' × C layout:
/// - Rows 0..total_rounds: coefficient rows (one per sumcheck round, zero-padded to C)
/// - Rows R_coeff..R_coeff+noncoeff_rows: non-coefficient values (next_claims, etc.)
/// - Remaining rows: zero padding
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
    /// Number of constraints
    pub num_constraints: usize,
    /// Stage configurations used to build this R1CS
    pub stage_configs: Vec<StageConfig>,
    /// Extra constraints (e.g., PCS binding) appended after all stages
    pub extra_constraints: Vec<OutputClaimConstraint>,
    /// Output variables for extra constraints (one per constraint)
    pub extra_output_vars: Vec<Variable>,
    /// Blinding variables for extra constraints (one per constraint)
    pub extra_blinding_vars: Vec<Variable>,
    /// Hyrax grid layout parameters
    pub hyrax: HyraxParams,
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
    /// Expected evaluation variables (witness) - y_j (for simple linear constraints)
    pub evaluation_vars: Vec<Variable>,
    /// Mapping from OpeningId to witness variable (for general constraints)
    pub opening_vars: HashMap<OpeningId, Variable>,
    /// Auxiliary variables for intermediate products
    pub aux_vars: Vec<Variable>,
}

/// Builder for constructing the verifier R1CS
pub struct VerifierR1CSBuilder<F: JoltField> {
    /// Constraints accumulated so far
    constraints: Vec<Constraint<F>>,
    /// Next variable index to allocate
    next_var: usize,
    /// Stage configurations
    stage_configs: Vec<StageConfig>,
    /// Extra constraints appended after all stages
    extra_constraints: Vec<OutputClaimConstraint>,
    /// Mapping from (stage, round) to the round's variables
    round_vars: Vec<Vec<RoundVariables>>,
    /// Final output variables for each stage with final_output config
    final_output_vars: Vec<Option<FinalOutputVariables>>,
    /// Initial input variables for each stage with initial_input config
    initial_input_vars: Vec<Option<FinalOutputVariables>>,
    /// Output variables for extra constraints
    extra_output_vars: Vec<Variable>,
    /// Blinding variables for extra constraints
    extra_blinding_vars: Vec<Variable>,
    /// Baked public input values (challenges, initial claims, etc.)
    baked: BakedPublicInputs<F>,
}

/// Variables allocated for a single sumcheck round
#[derive(Clone, Debug)]
pub struct RoundVariables {
    /// Polynomial coefficients c0, c1, c2, ...
    pub coeffs: Vec<Variable>,
    /// Output claim (becomes next round's input)
    pub next_claim: Variable,
}

impl<F: JoltField> VerifierR1CSBuilder<F> {
    /// Create a new builder for the given stage configurations with baked public inputs
    pub fn new(stage_configs: &[StageConfig], baked: &BakedPublicInputs<F>) -> Self {
        Self::new_with_extra(stage_configs, &[], baked)
    }

    /// Create a new builder with extra constraints and baked public inputs.
    pub fn new_with_extra(
        stage_configs: &[StageConfig],
        extra_constraints: &[OutputClaimConstraint],
        baked: &BakedPublicInputs<F>,
    ) -> Self {
        // No public input variables — everything is baked into matrix coefficients.
        // Index 0 is the u scalar. Witness variables start at index 1.
        let next_var = 1;

        Self {
            constraints: Vec::new(),
            next_var,
            stage_configs: stage_configs.to_vec(),
            extra_constraints: extra_constraints.to_vec(),
            round_vars: Vec::new(),
            final_output_vars: Vec::new(),
            initial_input_vars: Vec::new(),
            extra_output_vars: Vec::new(),
            extra_blinding_vars: Vec::new(),
            baked: baked.clone(),
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

    /// Build constraints for a single sumcheck round with baked challenge.
    ///
    /// For g(X) = c0 + c1*X + c2*X^2 + ... + cd*X^d:
    ///
    /// 1. Sum check: g(0) + g(1) = claimed_sum
    ///    => (2*c0 + c1 + c2 + ... + cd) * u = claimed_sum
    ///
    /// 2. Evaluation: g(γ) = next_claim with baked γ
    ///    => (c0 + γ·c1 + γ²·c2 + ... + γ^d·cd) * u = next_claim
    ///    Single linear constraint — no Horner intermediates needed.
    fn add_round_constraints(
        &mut self,
        vars: &RoundVariables,
        claimed_sum: LinearCombination<F>,
        challenge_value: F,
        uniskip_power_sums: Option<&[i128]>,
    ) {
        let RoundVariables { coeffs, next_claim } = vars;
        let degree = coeffs.len() - 1;

        // Constraint 1: Sum check
        let a = if let Some(power_sums) = uniskip_power_sums {
            let mut lc = LinearCombination::new();
            for (j, &coeff_var) in coeffs.iter().enumerate() {
                if power_sums[j] != 0 {
                    lc = lc.add_term(coeff_var, F::from_i128(power_sums[j]));
                }
            }
            lc
        } else {
            let mut lc = LinearCombination::new().add_term(coeffs[0], F::from_u64(2));
            for i in 1..coeffs.len() {
                lc = lc.add_var(coeffs[i]);
            }
            lc
        };
        let b = LinearCombination::constant(F::one());
        self.add_constraint(a, b, claimed_sum);

        // Constraint 2: Evaluation g(γ) = next_claim with baked powers of γ
        if degree == 0 {
            // g(γ) = c0
            let a = LinearCombination::variable(coeffs[0]);
            let b = LinearCombination::constant(F::one());
            let c_lc = LinearCombination::variable(*next_claim);
            self.add_constraint(a, b, c_lc);
        } else {
            // g(γ) = Σ_k γ^k · c_k
            let mut a = LinearCombination::new();
            let mut gamma_power = F::one();
            for (k, &coeff_var) in coeffs.iter().enumerate() {
                a = a.add_term(coeff_var, gamma_power);
                if k < degree {
                    gamma_power *= challenge_value;
                }
            }
            let b = LinearCombination::constant(F::one());
            let c_lc = LinearCombination::variable(*next_claim);
            self.add_constraint(a, b, c_lc);
        }
    }

    /// Build the complete verifier R1CS
    pub fn build(mut self) -> VerifierR1CS<F> {
        let total_rounds: usize = self.stage_configs.iter().map(|s| s.num_rounds).sum();
        let max_coeffs = self
            .stage_configs
            .iter()
            .map(|c| c.poly_degree + 1)
            .max()
            .unwrap_or(1);
        let hyrax_C = max_coeffs.next_power_of_two();
        let hyrax_R_coeff = if total_rounds == 0 {
            1
        } else {
            total_rounds.next_power_of_two()
        };
        // witness_start = 1 (right after u, no public inputs)
        let witness_start = self.next_var;

        // Non-coefficient variables start after the coefficient grid
        self.next_var = witness_start + hyrax_R_coeff * hyrax_C;

        let mut challenge_idx = 0usize;
        let mut chain_idx = 0usize;
        let mut batching_coeff_idx = 0usize;
        let mut output_challenge_idx = 0usize;
        let mut input_challenge_idx = 0usize;

        // First round of first chain uses baked initial claim
        let mut current_claim: LinearCombination<F> =
            LinearCombination::constant(self.baked.initial_claims[chain_idx]);
        let mut round_idx = 0usize;

        let mut global_opening_vars: HashMap<OpeningId, Variable> = HashMap::new();

        let stage_configs = self.stage_configs.clone();
        let baked = self.baked.clone();

        for (stage_idx, config) in stage_configs.iter().enumerate() {
            // Check if this stage starts a new chain
            if stage_idx > 0 && config.starts_new_chain {
                chain_idx += 1;
                current_claim = LinearCombination::constant(baked.initial_claims[chain_idx]);
            }

            // Handle initial input constraint
            let initial_input_vars = if let Some(ref ii_config) = config.initial_input {
                if let Some(ref constraint) = ii_config.constraint {
                    for opening_id in &constraint.required_openings {
                        if !global_opening_vars.contains_key(opening_id) {
                            let var = Variable::new(self.next_var);
                            self.next_var += 1;
                            global_opening_vars.insert(*opening_id, var);
                        }
                    }

                    let num_challenges = constraint.num_challenges;
                    let baked_challenge_values: Vec<F> = baked.input_constraint_challenges
                        [input_challenge_idx..input_challenge_idx + num_challenges]
                        .to_vec();
                    input_challenge_idx += num_challenges;

                    // current_claim is the "result" of the input constraint
                    let aux_vars = self.add_sum_of_products_constraint_baked(
                        current_claim.clone(),
                        constraint,
                        &global_opening_vars,
                        &baked_challenge_values,
                    );

                    Some(FinalOutputVariables {
                        evaluation_vars: Vec::new(),
                        opening_vars: global_opening_vars.clone(),
                        aux_vars,
                    })
                } else {
                    None
                }
            } else {
                None
            };
            self.initial_input_vars.push(initial_input_vars);

            let mut stage_rounds = Vec::with_capacity(config.num_rounds);

            for _round in 0..config.num_rounds {
                let num_coeffs = config.poly_degree + 1;

                // Coefficients at grid position: witness_start + round_idx * C + k
                let coeffs: Vec<Variable> = (0..num_coeffs)
                    .map(|k| Variable::new(witness_start + round_idx * hyrax_C + k))
                    .collect();

                // next_claim in non-coeff section
                let next_claim = Variable::new(self.next_var);
                self.next_var += 1;

                let vars = RoundVariables { coeffs, next_claim };

                let challenge_value = baked.challenges[challenge_idx];
                let power_sums = config.uniskip_power_sums.as_deref();
                self.add_round_constraints(&vars, current_claim, challenge_value, power_sums);

                current_claim = LinearCombination::variable(next_claim);
                challenge_idx += 1;
                round_idx += 1;

                stage_rounds.push(vars);
            }

            // Add final output constraint if configured
            let final_output_vars = if let Some(ref fout) = config.final_output {
                let last_round = stage_rounds
                    .last()
                    .expect("Stage must have at least one round");

                if let Some(exact_vars) = fout.exact_num_witness_vars {
                    for _ in 0..exact_vars {
                        self.next_var += 1;
                    }

                    Some(FinalOutputVariables {
                        evaluation_vars: Vec::new(),
                        opening_vars: HashMap::new(),
                        aux_vars: Vec::new(),
                    })
                } else if let Some(ref constraint) = fout.constraint {
                    for opening_id in &constraint.required_openings {
                        if !global_opening_vars.contains_key(opening_id) {
                            let var = Variable::new(self.next_var);
                            self.next_var += 1;
                            global_opening_vars.insert(*opening_id, var);
                        }
                    }

                    let num_challenges = constraint.num_challenges;
                    let baked_challenge_values: Vec<F> = baked.output_constraint_challenges
                        [output_challenge_idx..output_challenge_idx + num_challenges]
                        .to_vec();
                    output_challenge_idx += num_challenges;

                    let aux_vars = self.add_sum_of_products_constraint_baked(
                        LinearCombination::variable(last_round.next_claim),
                        constraint,
                        &global_opening_vars,
                        &baked_challenge_values,
                    );

                    Some(FinalOutputVariables {
                        evaluation_vars: Vec::new(),
                        opening_vars: global_opening_vars.clone(),
                        aux_vars,
                    })
                } else {
                    // Simple linear constraint: final_claim = Σⱼ αⱼ · yⱼ
                    // Batching coefficients are baked — single constraint for all evaluations
                    let num_evals = fout.num_evaluations;
                    let baked_coeffs: Vec<F> = baked.batching_coefficients
                        [batching_coeff_idx..batching_coeff_idx + num_evals]
                        .to_vec();
                    batching_coeff_idx += num_evals;

                    let eval_vars: Vec<Variable> = (0..num_evals)
                        .map(|_| {
                            let var = Variable::new(self.next_var);
                            self.next_var += 1;
                            var
                        })
                        .collect();

                    self.add_final_output_constraint_baked(
                        last_round.next_claim,
                        &baked_coeffs,
                        &eval_vars,
                    );

                    Some(FinalOutputVariables {
                        evaluation_vars: eval_vars,
                        opening_vars: HashMap::new(),
                        aux_vars: Vec::new(),
                    })
                }
            } else {
                None
            };

            self.round_vars.push(stage_rounds);
            self.final_output_vars.push(final_output_vars);
        }

        let extra_constraints = self.extra_constraints.clone();
        let mut extra_challenge_idx = 0usize;

        for constraint in &extra_constraints {
            for opening_id in &constraint.required_openings {
                if !global_opening_vars.contains_key(opening_id) {
                    let var = Variable::new(self.next_var);
                    self.next_var += 1;
                    global_opening_vars.insert(*opening_id, var);
                }
            }

            let output_var = Variable::new(self.next_var);
            self.next_var += 1;

            let num_challenges = constraint.num_challenges;
            let baked_challenge_values: Vec<F> = baked.extra_constraint_challenges
                [extra_challenge_idx..extra_challenge_idx + num_challenges]
                .to_vec();
            extra_challenge_idx += num_challenges;

            let _aux_vars = self.add_sum_of_products_constraint_baked(
                LinearCombination::variable(output_var),
                constraint,
                &global_opening_vars,
                &baked_challenge_values,
            );

            let blinding_var = Variable::new(self.next_var);
            self.next_var += 1;

            self.extra_output_vars.push(output_var);
            self.extra_blinding_vars.push(blinding_var);
        }

        let num_constraints = self.constraints.len();

        let noncoeff_count = self.next_var - (witness_start + hyrax_R_coeff * hyrax_C);
        let hyrax = compute_hyrax_params(&self.stage_configs, noncoeff_count);
        let num_vars = witness_start + hyrax.R_prime * hyrax.C;

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
            num_constraints,
            stage_configs: self.stage_configs,
            extra_constraints: self.extra_constraints,
            extra_output_vars: self.extra_output_vars,
            extra_blinding_vars: self.extra_blinding_vars,
            hyrax,
        }
    }

    /// Add final output constraint with baked batching coefficients:
    /// final_claim = Σⱼ α_val_j · yⱼ
    ///
    /// Single constraint: (Σⱼ α_val_j · yⱼ) · u = final_claim
    fn add_final_output_constraint_baked(
        &mut self,
        final_claim: Variable,
        baked_coeffs: &[F],
        evaluations: &[Variable],
    ) {
        debug_assert_eq!(baked_coeffs.len(), evaluations.len());
        let n = baked_coeffs.len();
        if n == 0 {
            return;
        }

        // Single constraint for any number of evaluations:
        // A·Z = Σⱼ αⱼ · yⱼ (baked coefficients on witness variables)
        // B·Z = u
        // C·Z = final_claim
        let mut a = LinearCombination::new();
        for (coeff, &eval_var) in baked_coeffs.iter().zip(evaluations.iter()) {
            a = a.add_term(eval_var, *coeff);
        }
        let b = LinearCombination::constant(F::one());
        let c = LinearCombination::variable(final_claim);
        self.add_constraint(a, b, c);
    }

    /// Add general sum-of-products constraint with baked challenge values.
    ///
    /// Challenges resolve to constants (baked into matrix coefficients) instead of variables.
    /// When both operands of a multiplication are constants, their product is folded
    /// into a single coefficient to avoid unnecessary aux vars.
    fn add_sum_of_products_constraint_baked(
        &mut self,
        final_claim: LinearCombination<F>,
        constraint: &OutputClaimConstraint,
        opening_vars: &HashMap<OpeningId, Variable>,
        baked_challenge_values: &[F],
    ) -> Vec<Variable> {
        let mut aux_vars = Vec::new();

        if constraint.terms.is_empty() {
            return aux_vars;
        }

        // Resolve a ValueSource. Challenges → baked constant; Openings → variable.
        enum Resolved<F> {
            Constant(F),
            Variable(LinearCombination<F>),
        }

        let resolve = |vs: &ValueSource| -> Resolved<F> {
            match vs {
                ValueSource::Opening(id) => Resolved::Variable(LinearCombination::variable(
                    *opening_vars
                        .get(id)
                        .unwrap_or_else(|| panic!("Opening {id:?} not found")),
                )),
                ValueSource::Challenge(idx) => Resolved::Constant(baked_challenge_values[*idx]),
                ValueSource::Constant(val) => Resolved::Constant(F::from_i128(*val)),
            }
        };

        // For each term, compute the product and collect result
        let mut term_results: Vec<(LinearCombination<F>, Variable)> =
            Vec::with_capacity(constraint.terms.len());

        for term in &constraint.terms {
            let coeff_resolved = resolve(&term.coeff);

            if term.factors.is_empty() {
                let aux = Variable::new(self.next_var);
                self.next_var += 1;
                aux_vars.push(aux);

                match coeff_resolved {
                    Resolved::Constant(val) => {
                        // coeff is constant: constant(val) * u = aux
                        let a = LinearCombination::constant(val);
                        let b = LinearCombination::constant(F::one());
                        self.add_constraint(a, b, LinearCombination::variable(aux));
                    }
                    Resolved::Variable(lc) => {
                        let a = lc;
                        let b = LinearCombination::constant(F::one());
                        self.add_constraint(a, b, LinearCombination::variable(aux));
                    }
                }

                term_results.push((LinearCombination::constant(F::one()), aux));
            } else if term.factors.len() == 1 {
                let factor_resolved = resolve(&term.factors[0]);

                let aux = Variable::new(self.next_var);
                self.next_var += 1;
                aux_vars.push(aux);

                match (coeff_resolved, factor_resolved) {
                    (Resolved::Constant(cv), Resolved::Constant(fv)) => {
                        // Both constant: product is known, bake it
                        let a = LinearCombination::constant(cv * fv);
                        let b = LinearCombination::constant(F::one());
                        self.add_constraint(a, b, LinearCombination::variable(aux));
                    }
                    (Resolved::Constant(cv), Resolved::Variable(fv_lc)) => {
                        // constant * variable
                        let a = LinearCombination::constant(cv);
                        self.add_constraint(a, fv_lc, LinearCombination::variable(aux));
                    }
                    (Resolved::Variable(cv_lc), Resolved::Constant(fv)) => {
                        // variable * constant
                        let b = LinearCombination::constant(fv);
                        self.add_constraint(cv_lc, b, LinearCombination::variable(aux));
                    }
                    (Resolved::Variable(cv_lc), Resolved::Variable(fv_lc)) => {
                        self.add_constraint(cv_lc, fv_lc, LinearCombination::variable(aux));
                    }
                }

                term_results.push((LinearCombination::constant(F::one()), aux));
            } else {
                // Multiple factors: chain multiplication
                let factor0_resolved = resolve(&term.factors[0]);
                let factor1_resolved = resolve(&term.factors[1]);

                let mut current_product = Variable::new(self.next_var);
                self.next_var += 1;
                aux_vars.push(current_product);

                // Helper to get linear combination from Resolved
                let to_lc = |r: Resolved<F>| -> LinearCombination<F> {
                    match r {
                        Resolved::Constant(v) => LinearCombination::constant(v),
                        Resolved::Variable(lc) => lc,
                    }
                };

                self.add_constraint(
                    to_lc(factor0_resolved),
                    to_lc(factor1_resolved),
                    LinearCombination::variable(current_product),
                );

                for factor in &term.factors[2..] {
                    let factor_lc = to_lc(resolve(factor));
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

                // Multiply by coefficient
                let final_term = Variable::new(self.next_var);
                self.next_var += 1;
                aux_vars.push(final_term);

                let coeff_lc = match coeff_resolved {
                    Resolved::Constant(v) => LinearCombination::constant(v),
                    Resolved::Variable(lc) => lc,
                };

                self.add_constraint(
                    coeff_lc,
                    LinearCombination::variable(current_product),
                    LinearCombination::variable(final_term),
                );

                term_results.push((LinearCombination::constant(F::one()), final_term));
            }
        }

        // Sum all term results: final_claim = Σᵢ term_results[i]
        let n = term_results.len();

        if n == 1 {
            let (_, term_var) = &term_results[0];
            let a = LinearCombination::constant(F::one());
            let b = LinearCombination::variable(*term_var);
            self.add_constraint(a, b, final_claim);
        } else {
            let mut sum_lc = LinearCombination::new();
            for (_, term_var) in &term_results {
                sum_lc = sum_lc.add_var(*term_var);
            }

            let a = LinearCombination::constant(F::one());
            let b = sum_lc;
            self.add_constraint(a, b, final_claim);
        }

        aux_vars
    }

    /// Get the round variables for witness assignment
    pub fn get_round_vars(&self) -> &Vec<Vec<RoundVariables>> {
        &self.round_vars
    }

    /// Get the final output variables for each stage
    pub fn final_output_vars(&self) -> &[Option<FinalOutputVariables>] {
        &self.final_output_vars
    }

    /// Get the initial input variables for each stage
    pub fn initial_input_vars(&self) -> &[Option<FinalOutputVariables>] {
        &self.initial_input_vars
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subprotocols::blindfold::witness::{BlindFoldWitness, RoundWitness, StageWitness};
    use crate::subprotocols::blindfold::BakedPublicInputs;
    use ark_bn254::Fr;

    #[test]
    fn test_single_round_constraint_satisfaction() {
        type F = Fr;

        let c0 = F::from_u64(40);
        let c1 = F::from_u64(10);
        let c2 = F::from_u64(15);
        let c3 = F::from_u64(5);
        let initial_claim = F::from_u64(110);
        let r = F::from_u64(7);

        let round = RoundWitness::new(vec![c0, c1, c2, c3], r);
        let configs = [super::super::StageConfig::new(1, 3)];
        let witness = BlindFoldWitness::new(initial_claim, vec![StageWitness::new(vec![round])]);

        let baked = BakedPublicInputs::from_witness(&witness, &configs);
        let builder = VerifierR1CSBuilder::<F>::new(&configs, &baked);
        let r1cs = builder.build();

        let z = witness.assign(&r1cs);
        assert!(r1cs.is_satisfied(&z), "R1CS should be satisfied");
    }

    #[test]
    fn test_multi_round_chaining() {
        type F = Fr;

        let c0_1 = F::from_u64(20);
        let c1_1 = F::from_u64(5);
        let c2_1 = F::from_u64(7);
        let c3_1 = F::from_u64(3);
        let initial_claim = F::from_u64(55);
        let r1 = F::from_u64(3);

        let round1 = RoundWitness::new(vec![c0_1, c1_1, c2_1, c3_1], r1);
        let next1 = round1.evaluate(r1);

        let c0_2 = F::from_u64(85);
        let c1_2 = F::from_u64(4);
        let c2_2 = F::from_u64(3);
        let c3_2 = F::from_u64(2);
        let r2 = F::from_u64(5);

        let round2 = RoundWitness::new(vec![c0_2, c1_2, c2_2, c3_2], r2);

        let claim2 = F::from_u64(2) * c0_2 + c1_2 + c2_2 + c3_2;
        assert_eq!(claim2, next1, "Round 2 claim should equal round 1 output");

        let configs = [super::super::StageConfig::new(2, 3)];
        let witness =
            BlindFoldWitness::new(initial_claim, vec![StageWitness::new(vec![round1, round2])]);

        let baked = BakedPublicInputs::from_witness(&witness, &configs);
        let builder = VerifierR1CSBuilder::<F>::new(&configs, &baked);
        let r1cs = builder.build();

        // 2 constraints per round (sum + eval), 2 rounds = 4
        assert_eq!(r1cs.num_constraints, 4);

        let z = witness.assign(&r1cs);
        match r1cs.check_satisfaction(&z) {
            Ok(()) => {}
            Err(row) => panic!("Constraint {row} failed"),
        }
    }

    #[test]
    fn test_constraint_count() {
        type F = Fr;

        // Build dummy baked values for 120 rounds, 1 chain
        let configs: Vec<_> = (0..6)
            .map(|_| super::super::StageConfig::new(20, 3))
            .collect();

        let baked = BakedPublicInputs {
            challenges: vec![F::from_u64(1); 120],
            initial_claims: vec![F::from_u64(0)],
            ..Default::default()
        };

        let builder = VerifierR1CSBuilder::<F>::new(&configs, &baked);
        let r1cs = builder.build();

        // 2 constraints per round, 120 rounds = 240
        assert_eq!(r1cs.num_constraints, 240);
        assert_eq!(r1cs.hyrax.C, 4);
        assert_eq!(r1cs.hyrax.R_coeff, 128);
    }

    #[test]
    fn test_invalid_witness_fails() {
        type F = Fr;

        let c0 = F::from_u64(40);
        let c1 = F::from_u64(10);
        let c2 = F::from_u64(5);
        let c3 = F::from_u64(5);
        let initial_claim = F::from_u64(200); // Doesn't match coefficients
        let r = F::from_u64(3);

        let round = RoundWitness::with_claimed_sum(vec![c0, c1, c2, c3], r, F::from_u64(200));
        let configs = [super::super::StageConfig::new(1, 3)];
        let witness = BlindFoldWitness::new(initial_claim, vec![StageWitness::new(vec![round])]);

        let baked = BakedPublicInputs::from_witness(&witness, &configs);
        let builder = VerifierR1CSBuilder::<F>::new(&configs, &baked);
        let r1cs = builder.build();

        let z = witness.assign(&r1cs);
        assert!(
            r1cs.check_satisfaction(&z).is_err(),
            "R1CS should NOT be satisfied with invalid witness"
        );
    }

    #[test]
    fn test_uniskip_constraint_satisfaction() {
        type F = Fr;

        let power_sums = vec![4, 0, 10, 0];
        let configs = [super::super::StageConfig::new_uniskip(3, power_sums)];

        let c0 = F::from_u64(5);
        let c1 = F::from_u64(7);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(9);
        let initial_claim = F::from_u64(50); // 4*5 + 10*3 = 50
        let r = F::from_u64(2);

        let round = RoundWitness::new(vec![c0, c1, c2, c3], r);
        let witness = BlindFoldWitness::new(initial_claim, vec![StageWitness::new(vec![round])]);

        let baked = BakedPublicInputs::from_witness(&witness, &configs);
        let builder = VerifierR1CSBuilder::<F>::new(&configs, &baked);
        let r1cs = builder.build();

        let z = witness.assign(&r1cs);
        match r1cs.check_satisfaction(&z) {
            Ok(()) => {}
            Err(row) => panic!("Constraint {row} failed"),
        }
    }

    #[test]
    fn test_uniskip_invalid_witness_fails() {
        type F = Fr;

        let power_sums = vec![4, 0, 10, 0];
        let configs = [super::super::StageConfig::new_uniskip(3, power_sums)];

        let c0 = F::from_u64(5);
        let c1 = F::from_u64(7);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(9);
        let r = F::from_u64(2);

        let round = RoundWitness::with_claimed_sum(vec![c0, c1, c2, c3], r, F::from_u64(100));

        let baked = BakedPublicInputs {
            challenges: vec![r],
            initial_claims: vec![F::from_u64(100)], // Correct initial_claim = 100 but coeffs give 50
            ..Default::default()
        };

        let builder = VerifierR1CSBuilder::<F>::new(&configs, &baked);
        let r1cs = builder.build();

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

        let config = super::super::StageConfig::new(1, 3).with_final_output(1);
        let c0 = F::from_u64(40);
        let c1 = F::from_u64(5);
        let c2 = F::from_u64(10);
        let c3 = F::from_u64(5);
        let initial_claim = F::from_u64(100);
        let r = F::from_u64(3);

        let round = RoundWitness::new(vec![c0, c1, c2, c3], r);
        let final_claim = round.evaluate(r);

        let alpha = F::from_u64(1);
        let y = final_claim;

        let fout_witness = FinalOutputWitness::new(vec![alpha], vec![y]);
        let stage = StageWitness::with_final_output(vec![round], fout_witness);
        let witness = BlindFoldWitness::new(initial_claim, vec![stage]);

        let baked = BakedPublicInputs::from_witness(&witness, &[config.clone()]);
        let builder = VerifierR1CSBuilder::<F>::new(&[config], &baked);
        let r1cs = builder.build();

        // 2 round constraints + 1 final output = 3
        assert_eq!(r1cs.num_constraints, 3);

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

        let config = super::super::StageConfig::new(1, 3).with_final_output(3);
        let c0 = F::from_u64(40);
        let c1 = F::from_u64(5);
        let c2 = F::from_u64(10);
        let c3 = F::from_u64(5);
        let initial_claim = F::from_u64(100);
        let r = F::from_u64(3);

        let round = RoundWitness::new(vec![c0, c1, c2, c3], r);
        let final_claim = round.evaluate(r);

        let alpha0 = F::from_u64(2);
        let alpha1 = F::from_u64(3);
        let alpha2 = F::from_u64(5);
        let y0 = F::from_u64(10);
        let y1 = F::from_u64(20);
        let partial_sum = alpha0 * y0 + alpha1 * y1;
        let y2 = (final_claim - partial_sum) * alpha2.inverse().unwrap();

        let fout_witness = FinalOutputWitness::new(vec![alpha0, alpha1, alpha2], vec![y0, y1, y2]);
        let stage = StageWitness::with_final_output(vec![round], fout_witness);
        let witness = BlindFoldWitness::new(initial_claim, vec![stage]);

        let baked = BakedPublicInputs::from_witness(&witness, &[config.clone()]);
        let builder = VerifierR1CSBuilder::<F>::new(&[config], &baked);
        let r1cs = builder.build();

        // 2 round constraints + 1 final output = 3 (single constraint for all evals!)
        assert_eq!(r1cs.num_constraints, 3);

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
        let c0 = F::from_u64(40);
        let c1 = F::from_u64(5);
        let c2 = F::from_u64(10);
        let c3 = F::from_u64(5);
        let initial_claim = F::from_u64(100);
        let r = F::from_u64(3);

        let round = RoundWitness::new(vec![c0, c1, c2, c3], r);
        let final_claim = round.evaluate(r);

        let alpha = F::from_u64(1);
        let y = final_claim + F::from_u64(1); // Wrong value!

        let fout_witness = FinalOutputWitness::new(vec![alpha], vec![y]);
        let stage = StageWitness::with_final_output(vec![round], fout_witness);
        let witness = BlindFoldWitness::new(initial_claim, vec![stage]);

        let baked = BakedPublicInputs::from_witness(&witness, &[config.clone()]);
        let builder = VerifierR1CSBuilder::<F>::new(&[config], &baked);
        let r1cs = builder.build();

        let z = witness.assign(&r1cs);
        assert!(
            r1cs.check_satisfaction(&z).is_err(),
            "R1CS should NOT be satisfied with invalid final output"
        );
    }
}
