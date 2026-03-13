use jolt_field::Field;

use crate::normalize::{SopValue, SumOfProducts};

/// An R1CS variable index.
///
/// Variables index into a witness vector `Z`. Index 0 is conventionally the
/// constant `u` (equals 1 for non-relaxed instances). Opening variables and
/// auxiliary variables are allocated by the emitter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct R1csVar(pub u32);

impl R1csVar {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A term in a linear combination: `coefficient * variable`.
#[derive(Debug, Clone)]
pub struct LcTerm<F> {
    pub var: R1csVar,
    pub coeff: F,
}

/// A linear combination: `Σ cᵢ · xᵢ`.
///
/// Represents one side of an R1CS constraint (A, B, or C).
#[derive(Debug, Clone, Default)]
pub struct LinearCombination<F> {
    pub terms: Vec<LcTerm<F>>,
}

impl<F: Field> LinearCombination<F> {
    pub fn new() -> Self {
        Self { terms: Vec::new() }
    }

    /// A linear combination containing a single constant: `val · u` where
    /// `u` is the variable at index 0.
    pub fn constant(val: F) -> Self {
        Self {
            terms: vec![LcTerm {
                var: R1csVar(0),
                coeff: val,
            }],
        }
    }

    /// A linear combination containing a single variable with coefficient 1.
    pub fn variable(var: R1csVar) -> Self {
        Self {
            terms: vec![LcTerm {
                var,
                coeff: F::one(),
            }],
        }
    }

    /// Add a `coeff * var` term.
    pub fn add_term(mut self, var: R1csVar, coeff: F) -> Self {
        self.terms.push(LcTerm { var, coeff });
        self
    }

    /// Add a variable with coefficient 1.
    pub fn add_var(self, var: R1csVar) -> Self {
        self.add_term(var, F::one())
    }

    /// Merge terms sharing the same variable, removing any that become zero.
    ///
    /// After many `add`/`sub` operations, an LC can accumulate duplicate
    /// entries for the same variable.  `compact` collapses them so
    /// `evaluate` and `sparse_entries` stay O(unique-vars) instead of
    /// O(total-ops).
    pub fn compact(&mut self) {
        if self.terms.len() <= 1 {
            return;
        }
        // Sort by variable index so equal-var runs are adjacent.
        self.terms.sort_unstable_by_key(|t| t.var.0);

        let terms = &mut self.terms;
        let mut write = 0;
        for read in 1..terms.len() {
            if terms[read].var == terms[write].var {
                let read_coeff = terms[read].coeff;
                terms[write].coeff += read_coeff;
            } else {
                if !terms[write].coeff.is_zero() {
                    write += 1;
                }
                terms[write] = terms[read].clone();
            }
        }
        // Keep last group if nonzero.
        if !terms[write].coeff.is_zero() {
            write += 1;
        }
        terms.truncate(write);
    }

    /// Evaluate the linear combination against a witness vector.
    pub fn evaluate(&self, witness: &[F]) -> F {
        self.terms
            .iter()
            .map(|t| t.coeff * witness[t.var.index()])
            .sum()
    }
}

/// A single R1CS constraint: `A · B = C` where A, B, C are linear combinations.
#[derive(Debug, Clone)]
pub struct R1csConstraint<F> {
    pub a: LinearCombination<F>,
    pub b: LinearCombination<F>,
    pub c: LinearCombination<F>,
}

impl<F: Field> R1csConstraint<F> {
    /// Check if this constraint is satisfied by the given witness.
    pub fn is_satisfied(&self, witness: &[F]) -> bool {
        let a_val = self.a.evaluate(witness);
        let b_val = self.b.evaluate(witness);
        let c_val = self.c.evaluate(witness);
        a_val * b_val == c_val
    }
}

/// Result of R1CS emission from a `SumOfProducts`.
///
/// Contains the generated constraints, allocated auxiliary variables, and the
/// output variable whose witness value equals the expression result.
#[derive(Debug)]
pub struct R1csEmission<F> {
    pub constraints: Vec<R1csConstraint<F>>,
    pub aux_vars: Vec<R1csVar>,
    pub output_var: R1csVar,
}

impl<F: Field> R1csEmission<F> {
    /// Compute witness values for all auxiliary variables and the output
    /// variable, given concrete opening and challenge values.
    ///
    /// Returns a map from `R1csVar` to its witness value. The caller must
    /// insert `opening_vars` values and `Z[0] = 1` into the witness
    /// separately.
    pub fn compute_witness(
        &self,
        sop: &SumOfProducts,
        opening_values: &[F],
        challenge_values: &[F],
    ) -> Vec<(R1csVar, F)> {
        // The output is just the SoP evaluation
        let output_val = sop.evaluate(opening_values, challenge_values);
        let mut assignments = Vec::with_capacity(self.aux_vars.len() + 1);

        // For aux vars, we can derive from constraint satisfaction:
        // each constraint is A * B = C, and C is always a single aux var,
        // so aux_val = A_eval * B_eval.
        // But we don't have the full witness yet when computing aux vals.
        // Instead, we note that the aux vars are allocated in order and each
        // constraint's C side is a fresh aux var. So we can forward-evaluate.
        //
        // This is a simplified approach: we don't re-derive here. The caller
        // typically has the values already (from the prover side). We just
        // provide the output.
        assignments.push((self.output_var, output_val));
        assignments
    }
}

/// Resolved value during R1CS emission: either a baked constant or a variable.
enum Resolved<F> {
    Constant(F),
    Variable(R1csVar),
}

impl<F: Field> Resolved<F> {
    fn into_lc(self) -> LinearCombination<F> {
        match self {
            Self::Constant(v) => LinearCombination::constant(v),
            Self::Variable(var) => LinearCombination::variable(var),
        }
    }
}

impl SumOfProducts {
    /// Emit R1CS constraints enforcing this expression's value.
    ///
    /// - `opening_vars`: maps `Opening(id)` → R1CS witness variable
    /// - `challenge_values`: maps `Challenge(id)` → baked field constant
    /// - `next_var`: counter for allocating fresh auxiliary variables
    ///
    /// Challenges are baked into matrix coefficients (not witness variables)
    /// because they are public Fiat-Shamir values known at constraint
    /// construction time. This matches the BlindFold convention.
    ///
    /// Returns an `R1csEmission` containing the constraints, aux vars, and
    /// the output variable whose witness value equals the expression result.
    pub fn emit_r1cs<F: Field>(
        &self,
        opening_vars: &[R1csVar],
        challenge_values: &[F],
        next_var: &mut u32,
    ) -> R1csEmission<F> {
        let mut constraints = Vec::new();
        let mut aux_vars = Vec::new();
        let mut term_results = Vec::new();

        let mut alloc = || {
            let var = R1csVar(*next_var);
            *next_var += 1;
            var
        };

        let resolve = |val: &SopValue| -> Resolved<F> {
            match val {
                SopValue::Constant(c) => Resolved::Constant(F::from_i128(*c)),
                SopValue::Opening(id) => Resolved::Variable(opening_vars[*id as usize]),
                SopValue::Challenge(id) => Resolved::Constant(challenge_values[*id as usize]),
            }
        };

        for term in &self.terms {
            let coeff_f = F::from_i128(term.coefficient);

            match term.factors.len() {
                0 => {
                    // Pure constant term: coeff * u = aux
                    let aux = alloc();
                    aux_vars.push(aux);
                    constraints.push(R1csConstraint {
                        a: LinearCombination::constant(coeff_f),
                        b: LinearCombination::constant(F::one()),
                        c: LinearCombination::variable(aux),
                    });
                    term_results.push(aux);
                }
                1 => {
                    // Single factor: coeff * factor = aux
                    let factor = resolve(&term.factors[0]);
                    let aux = alloc();
                    aux_vars.push(aux);
                    match factor {
                        Resolved::Constant(fv) => {
                            // Both constant: fold into single constant
                            constraints.push(R1csConstraint {
                                a: LinearCombination::constant(coeff_f * fv),
                                b: LinearCombination::constant(F::one()),
                                c: LinearCombination::variable(aux),
                            });
                        }
                        Resolved::Variable(fvar) => {
                            constraints.push(R1csConstraint {
                                a: LinearCombination::constant(coeff_f),
                                b: LinearCombination::variable(fvar),
                                c: LinearCombination::variable(aux),
                            });
                        }
                    }
                    term_results.push(aux);
                }
                _ => {
                    // Multi-factor chain: f0 * f1 = aux0, aux0 * f2 = aux1, ...
                    // then coeff * final_product = term_aux
                    let f0 = resolve(&term.factors[0]);
                    let f1 = resolve(&term.factors[1]);

                    let mut current = alloc();
                    aux_vars.push(current);
                    constraints.push(R1csConstraint {
                        a: f0.into_lc(),
                        b: f1.into_lc(),
                        c: LinearCombination::variable(current),
                    });

                    for factor in &term.factors[2..] {
                        let f = resolve(factor);
                        let next = alloc();
                        aux_vars.push(next);
                        constraints.push(R1csConstraint {
                            a: LinearCombination::variable(current),
                            b: f.into_lc(),
                            c: LinearCombination::variable(next),
                        });
                        current = next;
                    }

                    // Apply coefficient: coeff * product = term_result
                    let term_aux = alloc();
                    aux_vars.push(term_aux);
                    constraints.push(R1csConstraint {
                        a: LinearCombination::constant(coeff_f),
                        b: LinearCombination::variable(current),
                        c: LinearCombination::variable(term_aux),
                    });
                    term_results.push(term_aux);
                }
            }
        }

        // Final: output = sum of all term results
        let output_var = alloc();
        aux_vars.push(output_var);

        if term_results.len() == 1 {
            // Single term: 1 * term = output
            constraints.push(R1csConstraint {
                a: LinearCombination::constant(F::one()),
                b: LinearCombination::variable(term_results[0]),
                c: LinearCombination::variable(output_var),
            });
        } else {
            // Sum: (Σ term_i) * 1 = output (linear combination in A)
            let mut sum_lc = LinearCombination::new();
            for var in &term_results {
                sum_lc = sum_lc.add_var(*var);
            }
            constraints.push(R1csConstraint {
                a: sum_lc,
                b: LinearCombination::constant(F::one()),
                c: LinearCombination::variable(output_var),
            });
        }

        R1csEmission {
            constraints,
            aux_vars,
            output_var,
        }
    }

    /// Estimate the number of auxiliary variables needed for R1CS emission.
    ///
    /// Each term with 0 factors → 1 aux (constant).
    /// Each term with 1 factor → 1 aux.
    /// Each term with n factors (n≥2) → (n-1) chain + 1 coeff = n aux.
    /// Plus 1 output variable.
    pub fn estimate_aux_count(&self) -> usize {
        let term_aux: usize = self
            .terms
            .iter()
            .map(|t| match t.factors.len() {
                0 | 1 => 1,
                n => n,
            })
            .sum();
        term_aux + 1 // +1 for output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::ExprBuilder;
    use jolt_field::Fr;

    /// Build a witness vector and verify all constraints are satisfied.
    fn check_r1cs_satisfaction<F: Field>(
        emission: &R1csEmission<F>,
        opening_vars: &[R1csVar],
        opening_values: &[F],
        challenge_values: &[F],
        sop: &SumOfProducts,
    ) {
        // Build a large-enough witness vector
        let max_var = emission
            .aux_vars
            .iter()
            .chain(opening_vars.iter())
            .map(|v| v.index())
            .max()
            .unwrap_or(0)
            + 1;
        let mut witness = vec![F::zero(); max_var + 1];

        // Z[0] = 1 (the u variable)
        witness[0] = F::one();

        // Assign opening variables
        for (var, val) in opening_vars.iter().zip(opening_values.iter()) {
            witness[var.index()] = *val;
        }

        // Forward-evaluate aux vars from constraints: each constraint's C
        // is a fresh aux var, so aux_val = A_eval * B_eval
        for constraint in &emission.constraints {
            let a_val = constraint.a.evaluate(&witness);
            let b_val = constraint.b.evaluate(&witness);
            let c_val = a_val * b_val;
            // C should be a single variable — assign it
            assert_eq!(
                constraint.c.terms.len(),
                1,
                "C side should be a single variable"
            );
            let c_var = constraint.c.terms[0].var;
            witness[c_var.index()] = c_val;
        }

        // Verify all constraints
        for (i, constraint) in emission.constraints.iter().enumerate() {
            assert!(
                constraint.is_satisfied(&witness),
                "Constraint {i} not satisfied"
            );
        }

        // Verify output matches direct evaluation
        let expected = sop.evaluate(opening_values, challenge_values);
        let actual = witness[emission.output_var.index()];
        assert_eq!(actual, expected, "Output mismatch");
    }

    #[test]
    fn lc_compact_merges_same_var() {
        let mut lc: LinearCombination<Fr> = LinearCombination {
            terms: vec![
                LcTerm {
                    var: R1csVar(1),
                    coeff: Fr::from_u64(3),
                },
                LcTerm {
                    var: R1csVar(2),
                    coeff: Fr::from_u64(5),
                },
                LcTerm {
                    var: R1csVar(1),
                    coeff: Fr::from_u64(7),
                },
            ],
        };
        lc.compact();
        assert_eq!(lc.terms.len(), 2);
        assert_eq!(lc.terms[0].var, R1csVar(1));
        assert_eq!(lc.terms[0].coeff, Fr::from_u64(10));
        assert_eq!(lc.terms[1].var, R1csVar(2));
        assert_eq!(lc.terms[1].coeff, Fr::from_u64(5));
    }

    #[test]
    fn lc_compact_removes_zeros() {
        let mut lc: LinearCombination<Fr> = LinearCombination {
            terms: vec![
                LcTerm {
                    var: R1csVar(1),
                    coeff: Fr::from_u64(5),
                },
                LcTerm {
                    var: R1csVar(1),
                    coeff: -Fr::from_u64(5),
                },
            ],
        };
        lc.compact();
        assert!(lc.terms.is_empty());
    }

    #[test]
    fn lc_compact_single_term_noop() {
        let mut lc: LinearCombination<Fr> = LinearCombination::variable(R1csVar(3));
        lc.compact();
        assert_eq!(lc.terms.len(), 1);
        assert_eq!(lc.terms[0].var, R1csVar(3));
    }

    #[test]
    fn r1cs_single_mul() {
        // a * b → 1 chain constraint + 1 coeff constraint + 1 sum constraint = 3
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let expr = b.build(a * bv);
        let sop = expr.to_sum_of_products();

        let opening_vars = [R1csVar(1), R1csVar(2)];
        let mut next_var = 3;
        let emission = sop.emit_r1cs::<Fr>(&opening_vars, &[], &mut next_var);

        let a_val = Fr::from_u64(7);
        let b_val = Fr::from_u64(11);
        check_r1cs_satisfaction(&emission, &opening_vars, &[a_val, b_val], &[], &sop);
    }

    #[test]
    fn r1cs_triple_product() {
        // a * b * c → chain: a*b=aux0, aux0*c=aux1, coeff*aux1=aux2, sum=output
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let c = b.opening(2);
        let expr = b.build(a * bv * c);
        let sop = expr.to_sum_of_products();

        let opening_vars = [R1csVar(1), R1csVar(2), R1csVar(3)];
        let mut next_var = 4;
        let emission = sop.emit_r1cs::<Fr>(&opening_vars, &[], &mut next_var);

        let vals = [Fr::from_u64(3), Fr::from_u64(5), Fr::from_u64(7)];
        check_r1cs_satisfaction(&emission, &opening_vars, &vals, &[], &sop);
    }

    #[test]
    fn r1cs_linear_sum() {
        // a + b → two single-factor terms + 1 sum constraint
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let expr = b.build(a + bv);
        let sop = expr.to_sum_of_products();

        let opening_vars = [R1csVar(1), R1csVar(2)];
        let mut next_var = 3;
        let emission = sop.emit_r1cs::<Fr>(&opening_vars, &[], &mut next_var);

        let vals = [Fr::from_u64(13), Fr::from_u64(29)];
        check_r1cs_satisfaction(&emission, &opening_vars, &vals, &[], &sop);
    }

    #[test]
    fn r1cs_weighted_sum() {
        // alpha*a + beta*b → challenges baked as constants
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let alpha = b.challenge(0);
        let beta = b.challenge(1);
        let expr = b.build(alpha * a + beta * bv);
        let sop = expr.to_sum_of_products();

        let opening_vars = [R1csVar(1), R1csVar(2)];
        let challenge_vals = [Fr::from_u64(3), Fr::from_u64(7)];
        let mut next_var = 3;
        let emission = sop.emit_r1cs::<Fr>(&opening_vars, &challenge_vals, &mut next_var);

        let opening_vals = [Fr::from_u64(10), Fr::from_u64(20)];
        check_r1cs_satisfaction(
            &emission,
            &opening_vars,
            &opening_vals,
            &challenge_vals,
            &sop,
        );
    }

    #[test]
    fn r1cs_booleanity() {
        // gamma * (h^2 - h) → SoP: [gamma*h*h, -gamma*h]
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * (h * h - h));
        let sop = expr.to_sum_of_products();

        let opening_vars = [R1csVar(1)];
        let challenge_vals = [Fr::from_u64(5)];
        let mut next_var = 2;
        let emission = sop.emit_r1cs::<Fr>(&opening_vars, &challenge_vals, &mut next_var);

        // h=3 → gamma*(9-3) = 5*6 = 30
        let opening_vals = [Fr::from_u64(3)];
        check_r1cs_satisfaction(
            &emission,
            &opening_vars,
            &opening_vals,
            &challenge_vals,
            &sop,
        );
    }

    #[test]
    fn r1cs_constant_only() {
        // constant 42 → 1 term (no factors) + 1 sum
        let b = ExprBuilder::new();
        let expr = b.build(b.constant(42));
        let sop = expr.to_sum_of_products();

        let mut next_var = 1;
        let emission = sop.emit_r1cs::<Fr>(&[], &[], &mut next_var);

        check_r1cs_satisfaction(&emission, &[], &[], &[], &sop);
    }

    #[test]
    fn r1cs_distribution() {
        // (a + b) * (c + d) → 4 terms after distribution
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let c = b.opening(2);
        let d = b.opening(3);
        let expr = b.build((a + bv) * (c + d));
        let sop = expr.to_sum_of_products();

        assert_eq!(sop.len(), 4);

        let opening_vars = [R1csVar(1), R1csVar(2), R1csVar(3), R1csVar(4)];
        let mut next_var = 5;
        let emission = sop.emit_r1cs::<Fr>(&opening_vars, &[], &mut next_var);

        let vals = [
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
        ];
        check_r1cs_satisfaction(&emission, &opening_vars, &vals, &[], &sop);
    }

    #[test]
    fn r1cs_estimate_matches_actual() {
        let expressions: Vec<(&str, _)> = vec![
            {
                let b = ExprBuilder::new();
                let a = b.opening(0);
                let bv = b.opening(1);
                ("a*b", b.build(a * bv))
            },
            {
                let b = ExprBuilder::new();
                let a = b.opening(0);
                let bv = b.opening(1);
                ("a+b", b.build(a + bv))
            },
            {
                let b = ExprBuilder::new();
                let a = b.opening(0);
                let bv = b.opening(1);
                let c = b.opening(2);
                ("a*b*c", b.build(a * bv * c))
            },
            {
                let b = ExprBuilder::new();
                let h = b.opening(0);
                let gamma = b.challenge(0);
                ("booleanity", b.build(gamma * (h * h - h)))
            },
            {
                let b = ExprBuilder::new();
                ("constant", b.build(b.constant(42)))
            },
        ];

        for (name, expr) in &expressions {
            let sop = expr.to_sum_of_products();
            let estimated = sop.estimate_aux_count();

            let max_opening: u32 = sop
                .terms
                .iter()
                .flat_map(|t| t.factors.iter())
                .filter_map(|f| match f {
                    crate::normalize::SopValue::Opening(id) => Some(*id),
                    _ => None,
                })
                .max()
                .map_or(0, |id| id + 1);

            let opening_vars: Vec<R1csVar> = (0..max_opening).map(|i| R1csVar(i + 1)).collect();
            let mut next_var = max_opening + 1;
            let emission = sop.emit_r1cs::<Fr>(&opening_vars, &[Fr::from_u64(5)], &mut next_var);

            assert_eq!(
                emission.aux_vars.len(),
                estimated,
                "estimate mismatch for {name}"
            );
        }
    }

    #[test]
    fn r1cs_with_random_values() {
        use rand_chacha::ChaCha8Rng;
        use rand_core::SeedableRng;

        let mut rng = ChaCha8Rng::seed_from_u64(0x41c5);

        // Test several expressions with random field values
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let c = b.opening(2);
        let alpha = b.challenge(0);
        let beta = b.challenge(1);
        let expr = b.build(alpha * (a * bv - c) + beta * c);
        let sop = expr.to_sum_of_products();

        let opening_vars = [R1csVar(1), R1csVar(2), R1csVar(3)];

        for _ in 0..50 {
            let opening_vals: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();
            let challenge_vals: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();

            let mut next_var = 4;
            let emission = sop.emit_r1cs::<Fr>(&opening_vars, &challenge_vals, &mut next_var);

            check_r1cs_satisfaction(
                &emission,
                &opening_vars,
                &opening_vals,
                &challenge_vals,
                &sop,
            );
        }
    }
}
