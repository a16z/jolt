use jolt_field::Field;

use crate::composition::{CompositionFormula, Factor};

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
    pub fn compact(&mut self) {
        if self.terms.len() <= 1 {
            return;
        }
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
    pub fn is_satisfied(&self, witness: &[F]) -> bool {
        let a_val = self.a.evaluate(witness);
        let b_val = self.b.evaluate(witness);
        let c_val = self.c.evaluate(witness);
        a_val * b_val == c_val
    }
}

/// Result of R1CS emission from a [`CompositionFormula`].
#[derive(Debug)]
pub struct R1csEmission<F> {
    pub constraints: Vec<R1csConstraint<F>>,
    pub aux_vars: Vec<R1csVar>,
    pub output_var: R1csVar,
}

impl<F: Field> R1csEmission<F> {
    /// Compute the output variable's witness value from the formula evaluation.
    pub fn compute_witness(
        &self,
        formula: &CompositionFormula,
        opening_values: &[F],
        challenge_values: &[F],
    ) -> Vec<(R1csVar, F)> {
        let output_val = formula.evaluate(opening_values, challenge_values);
        vec![(self.output_var, output_val)]
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

impl CompositionFormula {
    /// Emit R1CS constraints enforcing this formula's value.
    ///
    /// - `opening_vars`: maps `Factor::Input(id)` → R1CS witness variable
    /// - `challenge_values`: maps `Factor::Challenge(id)` → baked field constant
    /// - `next_var`: counter for allocating fresh auxiliary variables
    ///
    /// Challenges are baked into matrix coefficients (not witness variables)
    /// because they are public Fiat-Shamir values known at constraint
    /// construction time.
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

        let resolve = |val: &Factor| -> Resolved<F> {
            match val {
                Factor::Input(id) => Resolved::Variable(opening_vars[*id as usize]),
                Factor::Challenge(id) => Resolved::Constant(challenge_values[*id as usize]),
            }
        };

        for term in &self.terms {
            let coeff_f = F::from_i128(term.coefficient);

            match term.factors.len() {
                0 => {
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
                    let factor = resolve(&term.factors[0]);
                    let aux = alloc();
                    aux_vars.push(aux);
                    match factor {
                        Resolved::Constant(fv) => {
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

        let output_var = alloc();
        aux_vars.push(output_var);

        if term_results.len() == 1 {
            constraints.push(R1csConstraint {
                a: LinearCombination::constant(F::one()),
                b: LinearCombination::variable(term_results[0]),
                c: LinearCombination::variable(output_var),
            });
        } else {
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
    pub fn estimate_aux_count(&self) -> usize {
        let term_aux: usize = self
            .terms
            .iter()
            .map(|t| match t.factors.len() {
                0 | 1 => 1,
                n => n,
            })
            .sum();
        term_aux + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::ExprBuilder;
    use jolt_field::Fr;

    fn check_r1cs_satisfaction<F: Field>(
        emission: &R1csEmission<F>,
        opening_vars: &[R1csVar],
        opening_values: &[F],
        challenge_values: &[F],
        formula: &CompositionFormula,
    ) {
        let max_var = emission
            .aux_vars
            .iter()
            .chain(opening_vars.iter())
            .map(|v| v.index())
            .max()
            .unwrap_or(0)
            + 1;
        let mut witness = vec![F::zero(); max_var + 1];
        witness[0] = F::one();

        for (var, val) in opening_vars.iter().zip(opening_values.iter()) {
            witness[var.index()] = *val;
        }

        for constraint in &emission.constraints {
            let a_val = constraint.a.evaluate(&witness);
            let b_val = constraint.b.evaluate(&witness);
            let c_val = a_val * b_val;
            assert_eq!(
                constraint.c.terms.len(),
                1,
                "C side should be a single variable"
            );
            let c_var = constraint.c.terms[0].var;
            witness[c_var.index()] = c_val;
        }

        for (i, constraint) in emission.constraints.iter().enumerate() {
            assert!(
                constraint.is_satisfied(&witness),
                "Constraint {i} not satisfied"
            );
        }

        let expected = formula.evaluate(opening_values, challenge_values);
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
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let expr = b.build(a * bv);
        let f = expr.to_composition_formula();

        let opening_vars = [R1csVar(1), R1csVar(2)];
        let mut next_var = 3;
        let emission = f.emit_r1cs::<Fr>(&opening_vars, &[], &mut next_var);

        let a_val = Fr::from_u64(7);
        let b_val = Fr::from_u64(11);
        check_r1cs_satisfaction(&emission, &opening_vars, &[a_val, b_val], &[], &f);
    }

    #[test]
    fn r1cs_triple_product() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let c = b.opening(2);
        let expr = b.build(a * bv * c);
        let f = expr.to_composition_formula();

        let opening_vars = [R1csVar(1), R1csVar(2), R1csVar(3)];
        let mut next_var = 4;
        let emission = f.emit_r1cs::<Fr>(&opening_vars, &[], &mut next_var);

        let vals = [Fr::from_u64(3), Fr::from_u64(5), Fr::from_u64(7)];
        check_r1cs_satisfaction(&emission, &opening_vars, &vals, &[], &f);
    }

    #[test]
    fn r1cs_linear_sum() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let expr = b.build(a + bv);
        let f = expr.to_composition_formula();

        let opening_vars = [R1csVar(1), R1csVar(2)];
        let mut next_var = 3;
        let emission = f.emit_r1cs::<Fr>(&opening_vars, &[], &mut next_var);

        let vals = [Fr::from_u64(13), Fr::from_u64(29)];
        check_r1cs_satisfaction(&emission, &opening_vars, &vals, &[], &f);
    }

    #[test]
    fn r1cs_weighted_sum() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let alpha = b.challenge(0);
        let beta = b.challenge(1);
        let expr = b.build(alpha * a + beta * bv);
        let f = expr.to_composition_formula();

        let opening_vars = [R1csVar(1), R1csVar(2)];
        let challenge_vals = [Fr::from_u64(3), Fr::from_u64(7)];
        let mut next_var = 3;
        let emission = f.emit_r1cs::<Fr>(&opening_vars, &challenge_vals, &mut next_var);

        let opening_vals = [Fr::from_u64(10), Fr::from_u64(20)];
        check_r1cs_satisfaction(&emission, &opening_vars, &opening_vals, &challenge_vals, &f);
    }

    #[test]
    fn r1cs_booleanity() {
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * (h * h - h));
        let f = expr.to_composition_formula();

        let opening_vars = [R1csVar(1)];
        let challenge_vals = [Fr::from_u64(5)];
        let mut next_var = 2;
        let emission = f.emit_r1cs::<Fr>(&opening_vars, &challenge_vals, &mut next_var);

        let opening_vals = [Fr::from_u64(3)];
        check_r1cs_satisfaction(&emission, &opening_vars, &opening_vals, &challenge_vals, &f);
    }

    #[test]
    fn r1cs_constant_only() {
        let b = ExprBuilder::new();
        let expr = b.build(b.constant(42));
        let f = expr.to_composition_formula();

        let mut next_var = 1;
        let emission = f.emit_r1cs::<Fr>(&[], &[], &mut next_var);

        check_r1cs_satisfaction(&emission, &[], &[], &[], &f);
    }

    #[test]
    fn r1cs_distribution() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let c = b.opening(2);
        let d = b.opening(3);
        let expr = b.build((a + bv) * (c + d));
        let f = expr.to_composition_formula();

        assert_eq!(f.len(), 4);

        let opening_vars = [R1csVar(1), R1csVar(2), R1csVar(3), R1csVar(4)];
        let mut next_var = 5;
        let emission = f.emit_r1cs::<Fr>(&opening_vars, &[], &mut next_var);

        let vals = [
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
        ];
        check_r1cs_satisfaction(&emission, &opening_vars, &vals, &[], &f);
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
            let f = expr.to_composition_formula();
            let estimated = f.estimate_aux_count();

            let opening_vars: Vec<R1csVar> =
                (0..f.num_inputs as u32).map(|i| R1csVar(i + 1)).collect();
            let mut next_var = f.num_inputs as u32 + 1;
            let emission = f.emit_r1cs::<Fr>(&opening_vars, &[Fr::from_u64(5)], &mut next_var);

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

        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let c = b.opening(2);
        let alpha = b.challenge(0);
        let beta = b.challenge(1);
        let expr = b.build(alpha * (a * bv - c) + beta * c);
        let f = expr.to_composition_formula();

        let opening_vars = [R1csVar(1), R1csVar(2), R1csVar(3)];

        for _ in 0..50 {
            let opening_vals: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();
            let challenge_vals: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();

            let mut next_var = 4;
            let emission = f.emit_r1cs::<Fr>(&opening_vars, &challenge_vals, &mut next_var);

            check_r1cs_satisfaction(&emission, &opening_vars, &opening_vals, &challenge_vals, &f);
        }
    }
}
