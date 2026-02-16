use super::output_constraint::SumOfProductsVisitor;
use super::{
    compute_hyrax_params, BakedPublicInputs, Constraint, HyraxParams, LinearCombination,
    OutputClaimConstraint, StageConfig, ValueSource, Variable,
};
use crate::field::JoltField;
use crate::poly::opening_proof::OpeningId;
use std::collections::HashMap;

enum ResolvedValue<F> {
    Constant(F),
    Variable(LinearCombination<F>),
}

impl<F: JoltField> ResolvedValue<F> {
    fn into_lc(self) -> LinearCombination<F> {
        match self {
            Self::Constant(v) => LinearCombination::constant(v),
            Self::Variable(lc) => lc,
        }
    }
}

struct R1csConstraintVisitor<'a, F> {
    next_var: usize,
    opening_vars: &'a HashMap<OpeningId, Variable>,
    baked_challenge_values: &'a [F],
    current_product: Variable,
}

struct R1csConstraintAcc<F> {
    aux_vars: Vec<Variable>,
    term_results: Vec<Variable>,
    constraints: Vec<(
        LinearCombination<F>,
        LinearCombination<F>,
        LinearCombination<F>,
    )>,
}

impl<F> R1csConstraintAcc<F> {
    fn new() -> Self {
        Self {
            aux_vars: Vec::new(),
            term_results: Vec::new(),
            constraints: Vec::new(),
        }
    }
}

impl<'a, F: JoltField> R1csConstraintVisitor<'a, F> {
    fn new(
        next_var: usize,
        opening_vars: &'a HashMap<OpeningId, Variable>,
        baked_challenge_values: &'a [F],
    ) -> Self {
        Self {
            next_var,
            opening_vars,
            baked_challenge_values,
            current_product: Variable::new(0),
        }
    }

    fn alloc_var(&mut self) -> Variable {
        let var = Variable::new(self.next_var);
        self.next_var += 1;
        var
    }
}

impl<F: JoltField> SumOfProductsVisitor for R1csConstraintVisitor<'_, F> {
    type Resolved = ResolvedValue<F>;
    type Acc = R1csConstraintAcc<F>;

    fn resolve(&self, vs: &ValueSource) -> ResolvedValue<F> {
        match vs {
            ValueSource::Opening(id) => ResolvedValue::Variable(LinearCombination::variable(
                *self
                    .opening_vars
                    .get(id)
                    .unwrap_or_else(|| panic!("Opening {id:?} not found")),
            )),
            ValueSource::Challenge(idx) => {
                ResolvedValue::Constant(self.baked_challenge_values[*idx])
            }
            ValueSource::Constant(val) => ResolvedValue::Constant(F::from_i128(*val)),
        }
    }

    fn on_no_factors(&mut self, acc: &mut R1csConstraintAcc<F>, coeff: ResolvedValue<F>) {
        let aux = self.alloc_var();
        acc.aux_vars.push(aux);
        let a = coeff.into_lc();
        let b = LinearCombination::constant(F::one());
        acc.constraints
            .push((a, b, LinearCombination::variable(aux)));
        acc.term_results.push(aux);
    }

    fn on_single_factor(
        &mut self,
        acc: &mut R1csConstraintAcc<F>,
        coeff: ResolvedValue<F>,
        factor: ResolvedValue<F>,
    ) {
        let aux = self.alloc_var();
        acc.aux_vars.push(aux);
        match (coeff, factor) {
            (ResolvedValue::Constant(cv), ResolvedValue::Constant(fv)) => {
                let a = LinearCombination::constant(cv * fv);
                let b = LinearCombination::constant(F::one());
                acc.constraints
                    .push((a, b, LinearCombination::variable(aux)));
            }
            (ResolvedValue::Constant(cv), ResolvedValue::Variable(fv)) => {
                acc.constraints.push((
                    LinearCombination::constant(cv),
                    fv,
                    LinearCombination::variable(aux),
                ));
            }
            (ResolvedValue::Variable(cv), ResolvedValue::Constant(fv)) => {
                acc.constraints.push((
                    cv,
                    LinearCombination::constant(fv),
                    LinearCombination::variable(aux),
                ));
            }
            (ResolvedValue::Variable(cv), ResolvedValue::Variable(fv)) => {
                acc.constraints
                    .push((cv, fv, LinearCombination::variable(aux)));
            }
        }
        acc.term_results.push(aux);
    }

    fn on_chain_start(
        &mut self,
        acc: &mut R1csConstraintAcc<F>,
        f0: ResolvedValue<F>,
        f1: ResolvedValue<F>,
    ) {
        let product = self.alloc_var();
        acc.aux_vars.push(product);
        acc.constraints.push((
            f0.into_lc(),
            f1.into_lc(),
            LinearCombination::variable(product),
        ));
        self.current_product = product;
    }

    fn on_chain_step(&mut self, acc: &mut R1csConstraintAcc<F>, factor: ResolvedValue<F>) {
        let next = self.alloc_var();
        acc.aux_vars.push(next);
        acc.constraints.push((
            LinearCombination::variable(self.current_product),
            factor.into_lc(),
            LinearCombination::variable(next),
        ));
        self.current_product = next;
    }

    fn on_chain_finalize(&mut self, acc: &mut R1csConstraintAcc<F>, coeff: ResolvedValue<F>) {
        let final_term = self.alloc_var();
        acc.aux_vars.push(final_term);
        acc.constraints.push((
            coeff.into_lc(),
            LinearCombination::variable(self.current_product),
            LinearCombination::variable(final_term),
        ));
        acc.term_results.push(final_term);
    }
}

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

    /// M'[j - col_start] = Σ_row M[row,j] · eq_row[row]
    ///
    /// Projects out rows via `eq_row` weighting, keeping only columns
    /// in `[col_start .. col_start + out_len)`.
    pub fn project_columns(
        &self,
        eq_row: &[F],
        col_start: usize,
        out_len: usize,
        row_bound: usize,
    ) -> Vec<F> {
        let mut out = vec![F::zero(); out_len];
        let col_end = col_start + out_len;
        for &(row, col, ref val) in &self.entries {
            if col >= col_start && col < col_end && row < row_bound {
                out[col - col_start] += *val * eq_row[row];
            }
        }
        out
    }

    /// Σ_{row,col} M[row,col] · eq_row[row] · eq_col[col - col_start]
    pub fn bilinear_eval(
        &self,
        eq_row: &[F],
        eq_col: &[F],
        col_start: usize,
        col_len: usize,
        row_bound: usize,
    ) -> F {
        let col_end = col_start + col_len;
        let mut result = F::zero();
        for &(row, col, ref val) in &self.entries {
            if col >= col_start && col < col_end && row < row_bound {
                result += *val * eq_row[row] * eq_col[col - col_start];
            }
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

/// Builder for constructing the verifier R1CS
pub struct VerifierR1CSBuilder<F: JoltField> {
    constraints: Vec<Constraint<F>>,
    next_var: usize,
    stage_configs: Vec<StageConfig>,
    extra_constraints: Vec<OutputClaimConstraint>,
    extra_output_vars: Vec<Variable>,
    extra_blinding_vars: Vec<Variable>,
    baked: BakedPublicInputs<F>,
}

struct RoundVariables {
    coeffs: Vec<Variable>,
    next_claim: Variable,
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
        Self {
            constraints: Vec::new(),
            next_var: 1,
            stage_configs: stage_configs.to_vec(),
            extra_constraints: extra_constraints.to_vec(),
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
        use super::layout::{compute_witness_layout, ConstraintKind, LayoutStep};

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
        let witness_start = self.next_var;
        self.next_var = witness_start + hyrax_R_coeff * hyrax_C;

        let mut challenge_idx = 0usize;
        let mut batching_coeff_idx = 0usize;
        let mut output_challenge_idx = 0usize;
        let mut input_challenge_idx = 0usize;
        let mut extra_challenge_idx = 0usize;

        let mut global_opening_vars: HashMap<OpeningId, Variable> = HashMap::new();
        let mut current_claim = LinearCombination::<F>::new();
        let mut pending_coeffs: Option<Vec<Variable>> = None;

        let stage_configs = self.stage_configs.clone();
        let extra_constraints = self.extra_constraints.clone();
        let baked = self.baked.clone();

        let layout = compute_witness_layout(&stage_configs, &extra_constraints);

        for step in &layout {
            match step {
                LayoutStep::ConstantInitialClaim { chain_idx } => {
                    current_claim = LinearCombination::constant(baked.initial_claims[*chain_idx]);
                }
                LayoutStep::InitialClaimVar { .. } => {
                    let var = Variable::new(self.next_var);
                    self.next_var += 1;
                    current_claim = LinearCombination::variable(var);
                }
                LayoutStep::ConstraintVars {
                    constraint, kind, ..
                } => {
                    for opening_id in &constraint.required_openings {
                        if !global_opening_vars.contains_key(opening_id) {
                            let var = Variable::new(self.next_var);
                            self.next_var += 1;
                            global_opening_vars.insert(*opening_id, var);
                        }
                    }

                    let baked_challenge_values = match kind {
                        ConstraintKind::InitialInput => {
                            let n = constraint.num_challenges;
                            let vals = &baked.input_constraint_challenges
                                [input_challenge_idx..input_challenge_idx + n];
                            input_challenge_idx += n;
                            vals
                        }
                        ConstraintKind::FinalOutput => {
                            let n = constraint.num_challenges;
                            let vals = &baked.output_constraint_challenges
                                [output_challenge_idx..output_challenge_idx + n];
                            output_challenge_idx += n;
                            vals
                        }
                    };

                    self.add_sum_of_products_constraint_baked(
                        current_claim.clone(),
                        constraint,
                        &global_opening_vars,
                        baked_challenge_values,
                    );
                }
                LayoutStep::CoeffRow {
                    round_idx,
                    num_coeffs,
                    ..
                } => {
                    let coeffs: Vec<Variable> = (0..*num_coeffs)
                        .map(|k| Variable::new(witness_start + round_idx * hyrax_C + k))
                        .collect();
                    pending_coeffs = Some(coeffs);
                }
                LayoutStep::NextClaim { stage_idx, .. } => {
                    let next_claim = Variable::new(self.next_var);
                    self.next_var += 1;

                    let coeffs = pending_coeffs
                        .take()
                        .expect("CoeffRow must precede NextClaim");
                    let vars = RoundVariables { coeffs, next_claim };

                    let config = &stage_configs[*stage_idx];
                    let challenge_value = baked.challenges[challenge_idx];
                    let power_sums = config.uniskip_power_sums.as_deref();

                    let claimed_sum = std::mem::replace(
                        &mut current_claim,
                        LinearCombination::variable(next_claim),
                    );
                    self.add_round_constraints(&vars, claimed_sum, challenge_value, power_sums);
                    challenge_idx += 1;
                }
                LayoutStep::LinearFinalOutput {
                    num_evaluations, ..
                } => {
                    let baked_coeffs = &baked.batching_coefficients
                        [batching_coeff_idx..batching_coeff_idx + num_evaluations];
                    batching_coeff_idx += num_evaluations;

                    let eval_vars: Vec<Variable> = (0..*num_evaluations)
                        .map(|_| {
                            let var = Variable::new(self.next_var);
                            self.next_var += 1;
                            var
                        })
                        .collect();

                    self.add_final_output_constraint_baked(
                        current_claim.clone(),
                        baked_coeffs,
                        &eval_vars,
                    );
                }
                LayoutStep::PlaceholderVars { num_vars } => {
                    self.next_var += num_vars;
                }
                LayoutStep::ExtraConstraintVars { constraint, .. } => {
                    for opening_id in &constraint.required_openings {
                        if !global_opening_vars.contains_key(opening_id) {
                            let var = Variable::new(self.next_var);
                            self.next_var += 1;
                            global_opening_vars.insert(*opening_id, var);
                        }
                    }

                    let output_var = Variable::new(self.next_var);
                    self.next_var += 1;

                    let n = constraint.num_challenges;
                    let baked_challenge_values = &baked.extra_constraint_challenges
                        [extra_challenge_idx..extra_challenge_idx + n];
                    extra_challenge_idx += n;

                    self.add_sum_of_products_constraint_baked(
                        LinearCombination::variable(output_var),
                        constraint,
                        &global_opening_vars,
                        baked_challenge_values,
                    );

                    let blinding_var = Variable::new(self.next_var);
                    self.next_var += 1;

                    self.extra_output_vars.push(output_var);
                    self.extra_blinding_vars.push(blinding_var);
                }
            }
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
        final_claim: LinearCombination<F>,
        baked_coeffs: &[F],
        evaluations: &[Variable],
    ) {
        debug_assert_eq!(baked_coeffs.len(), evaluations.len());
        if baked_coeffs.is_empty() {
            return;
        }

        let mut a = LinearCombination::new();
        for (coeff, &eval_var) in baked_coeffs.iter().zip(evaluations.iter()) {
            a = a.add_term(eval_var, *coeff);
        }
        let b = LinearCombination::constant(F::one());
        self.add_constraint(a, b, final_claim);
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
        if constraint.terms.is_empty() {
            return Vec::new();
        }

        let mut visitor =
            R1csConstraintVisitor::new(self.next_var, opening_vars, baked_challenge_values);
        let mut r1cs_acc = R1csConstraintAcc::new();
        constraint.visit(&mut visitor, &mut r1cs_acc);

        self.next_var = visitor.next_var;
        for (a, b, c) in r1cs_acc.constraints {
            self.add_constraint(a, b, c);
        }

        let n = r1cs_acc.term_results.len();
        if n == 1 {
            let a = LinearCombination::constant(F::one());
            let b = LinearCombination::variable(r1cs_acc.term_results[0]);
            self.add_constraint(a, b, final_claim);
        } else {
            let mut sum_lc = LinearCombination::new();
            for var in &r1cs_acc.term_results {
                sum_lc = sum_lc.add_var(*var);
            }
            let a = LinearCombination::constant(F::one());
            self.add_constraint(a, sum_lc, final_claim);
        }

        r1cs_acc.aux_vars
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

        let fout_witness = FinalOutputWitness::linear(vec![alpha], vec![y]);
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

        let fout_witness =
            FinalOutputWitness::linear(vec![alpha0, alpha1, alpha2], vec![y0, y1, y2]);
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

        let fout_witness = FinalOutputWitness::linear(vec![alpha], vec![y]);
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
