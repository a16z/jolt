//! BlindFold Verifier R1CS
//!
//! Builds sparse R1CS matrices for verifying sumcheck rounds.
//! The circuit is O(log n) in size - only encoding the verifier's algebraic checks.

use super::{Constraint, LinearCombination, StageConfig, Variable};
use crate::field::JoltField;

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

/// Builder for constructing the verifier R1CS
pub struct VerifierR1CSBuilder<F: JoltField> {
    /// Constraints accumulated so far
    constraints: Vec<Constraint<F>>,
    /// Next variable index to allocate
    next_var: usize,
    /// Challenge variable indices (public inputs)
    challenge_vars: Vec<Variable>,
    /// Initial claim variable (public input)
    initial_claim_var: Variable,
    /// Stage configurations
    stage_configs: Vec<StageConfig>,
    /// Mapping from (stage, round) to the round's variables
    round_vars: Vec<Vec<RoundVariables>>,
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

        // Allocate public inputs: challenges + initial claim
        // Index 0 is the constant 1
        // Indices 1..=total_rounds are challenges
        // Index total_rounds+1 is the initial claim
        let challenge_vars: Vec<Variable> = (1..=total_rounds).map(Variable::new).collect();
        let initial_claim_var = Variable::new(total_rounds + 1);
        let next_var = total_rounds + 2; // Start witness allocation after public inputs

        Self {
            constraints: Vec::new(),
            next_var,
            challenge_vars,
            initial_claim_var,
            stage_configs: stage_configs.to_vec(),
            round_vars: Vec::new(),
        }
    }

    /// Allocate a new witness variable
    #[allow(dead_code)]
    fn alloc_var(&mut self) -> Variable {
        let var = Variable::new(self.next_var);
        self.next_var += 1;
        var
    }

    /// Allocate multiple witness variables
    #[allow(dead_code)]
    fn alloc_vars(&mut self, count: usize) -> Vec<Variable> {
        (0..count).map(|_| self.alloc_var()).collect()
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
    fn add_round_constraints(&mut self, vars: &RoundVariables) {
        let RoundVariables {
            coeffs,
            intermediates,
            next_claim,
            challenge,
            claimed_sum,
        } = vars;

        let degree = coeffs.len() - 1; // d coefficients means degree d-1

        // Constraint 1: Sum check (2*c0 + c1 + c2 + ... + cd) * 1 = claimed_sum
        let mut a = LinearCombination::new().add_term(coeffs[0], F::from_u64(2));
        for i in 1..coeffs.len() {
            a = a.add_var(coeffs[i]);
        }
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
        let mut current_claim = self.initial_claim_var;

        // Clone stage configs to avoid borrow conflict
        let stage_configs = self.stage_configs.clone();

        // Allocate variables and build constraints for each stage/round
        for config in &stage_configs {
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
                self.add_round_constraints(&vars);

                // Chain: this round's output becomes next round's input
                current_claim = next_claim;
                challenge_idx += 1;

                stage_rounds.push(vars);
            }

            self.round_vars.push(stage_rounds);
        }

        let num_constraints = self.constraints.len();
        let num_vars = self.next_var;
        let num_public_inputs = self.challenge_vars.len() + 1; // challenges + initial_claim

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

    /// Get the round variables for witness assignment
    pub fn get_round_vars(&self) -> &Vec<Vec<RoundVariables>> {
        &self.round_vars
    }

    /// Get the initial claim variable index
    pub fn initial_claim_var(&self) -> Variable {
        self.initial_claim_var
    }

    /// Get the challenge variable indices
    pub fn challenge_vars(&self) -> &[Variable] {
        &self.challenge_vars
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
}
