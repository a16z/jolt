//! [`R1CS`] implementation for `jolt-ir`'s [`R1csEmission`].
//!
//! Implements [`R1CS`] for [`R1csEmission`] so that expressions built with
//! [`ExprBuilder`](jolt_ir::ExprBuilder) can be proved and verified directly
//! via [`SpartanProver`](crate::SpartanProver) / [`SpartanVerifier`](crate::SpartanVerifier).

use jolt_field::Field;
use jolt_ir::{R1csEmission, R1csVar};

use crate::r1cs::R1CS;

impl<F: Field> R1CS<F> for R1csEmission<F> {
    fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    fn num_variables(&self) -> usize {
        self.constraints
            .iter()
            .flat_map(|c| {
                c.a.terms
                    .iter()
                    .chain(c.b.terms.iter())
                    .chain(c.c.terms.iter())
            })
            .map(|t| t.var.index())
            .max()
            .map_or(0, |m| m + 1)
    }

    fn multiply_witness(&self, witness: &[F]) -> (Vec<F>, Vec<F>, Vec<F>) {
        let n = self.constraints.len();
        let mut az = Vec::with_capacity(n);
        let mut bz = Vec::with_capacity(n);
        let mut cz = Vec::with_capacity(n);

        for c in &self.constraints {
            az.push(c.a.evaluate(witness));
            bz.push(c.b.evaluate(witness));
            cz.push(c.c.evaluate(witness));
        }

        (az, bz, cz)
    }

    fn sparse_entries_a(&self) -> Vec<(usize, usize, F)> {
        lc_to_entries(&self.constraints, |c| &c.a)
    }

    fn sparse_entries_b(&self) -> Vec<(usize, usize, F)> {
        lc_to_entries(&self.constraints, |c| &c.b)
    }

    fn sparse_entries_c(&self) -> Vec<(usize, usize, F)> {
        lc_to_entries(&self.constraints, |c| &c.c)
    }
}

/// Extracts sparse `(row, col, value)` entries from linear combinations.
fn lc_to_entries<F: Field>(
    constraints: &[jolt_ir::R1csConstraint<F>],
    select: impl Fn(&jolt_ir::R1csConstraint<F>) -> &jolt_ir::LinearCombination<F>,
) -> Vec<(usize, usize, F)> {
    let mut entries = Vec::new();
    for (row, c) in constraints.iter().enumerate() {
        for term in &select(c).terms {
            if !term.coeff.is_zero() {
                entries.push((row, term.var.index(), term.coeff));
            }
        }
    }
    entries
}

/// Assembles a complete witness vector from an [`R1csEmission`] and concrete
/// opening values.
///
/// The witness layout is:
/// - `Z[0] = 1` (the unit variable for constants)
/// - `Z[opening_vars[i]] = opening_values[i]`
/// - Auxiliary variables are forward-evaluated from constraints
///
/// # Invariant
///
/// This relies on `emit_r1cs` always placing a single variable on the C side
/// of each constraint (`c = LinearCombination::variable(aux)`). Under this
/// invariant, each aux variable's value is `A_eval * B_eval` and constraints
/// are ordered so that all dependencies are resolved before use.
///
/// # Panics
///
/// Panics if `opening_vars` and `opening_values` have different lengths, or
/// if a constraint references a variable beyond the witness bounds.
pub fn build_witness<F: Field>(
    emission: &R1csEmission<F>,
    opening_vars: &[R1csVar],
    opening_values: &[F],
) -> Vec<F> {
    assert_eq!(opening_vars.len(), opening_values.len());

    // Determine witness size from max variable index across all constraints
    let max_var = emission
        .constraints
        .iter()
        .flat_map(|c| {
            c.a.terms
                .iter()
                .chain(c.b.terms.iter())
                .chain(c.c.terms.iter())
        })
        .map(|t| t.var.index())
        .max()
        .unwrap_or(0);

    let mut witness = vec![F::zero(); max_var + 1];

    // Z[0] = 1 (unit variable)
    witness[0] = F::one();

    // Assign opening values
    for (var, val) in opening_vars.iter().zip(opening_values) {
        witness[var.index()] = *val;
    }

    // Forward-evaluate: each constraint's C side is a fresh aux variable
    // whose value equals A_eval * B_eval.
    for c in &emission.constraints {
        let a_val = c.a.evaluate(&witness);
        let b_val = c.b.evaluate(&witness);
        debug_assert_eq!(
            c.c.terms.len(),
            1,
            "emit_r1cs invariant: C side must be a single variable"
        );
        let c_var = c.c.terms[0].var;
        witness[c_var.index()] = a_val * b_val;
    }

    witness
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use jolt_ir::ExprBuilder;

    #[test]
    fn r1cs_trait_dimensions() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let expr = b.build(a * bv);
        let sop = expr.to_composition_formula();

        let opening_vars = [R1csVar(1), R1csVar(2)];
        let mut next_var = 3;
        let emission = sop.emit_r1cs::<Fr>(&opening_vars, &[], &mut next_var);

        assert!(emission.num_constraints() > 0);
        assert!(emission.num_variables() >= 3); // unit + 2 openings
    }

    #[test]
    fn r1cs_trait_satisfaction() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let expr = b.build(a * bv);
        let sop = expr.to_composition_formula();

        let opening_vars = [R1csVar(1), R1csVar(2)];
        let mut next_var = 3;
        let emission = sop.emit_r1cs::<Fr>(&opening_vars, &[], &mut next_var);

        let witness = build_witness(
            &emission,
            &opening_vars,
            &[Fr::from_u64(7), Fr::from_u64(11)],
        );
        let (az, bz, cz) = emission.multiply_witness(&witness);

        for i in 0..az.len() {
            assert_eq!(az[i] * bz[i], cz[i], "constraint {i} not satisfied");
        }
    }

    #[test]
    fn build_witness_output_matches_evaluation() {
        let b = ExprBuilder::new();
        let x = b.opening(0);
        let y = b.opening(1);
        let expr = b.build(x * y + x);
        let sop = expr.to_composition_formula();

        let opening_vars = [R1csVar(1), R1csVar(2)];
        let opening_vals = [Fr::from_u64(5), Fr::from_u64(3)];
        let mut next_var = 3;
        let emission = sop.emit_r1cs::<Fr>(&opening_vars, &[], &mut next_var);

        let witness = build_witness(&emission, &opening_vars, &opening_vals);
        let output = witness[emission.output_var.index()];
        let expected: Fr = sop.evaluate(&opening_vals, &[]);
        assert_eq!(output, expected);
    }
}
