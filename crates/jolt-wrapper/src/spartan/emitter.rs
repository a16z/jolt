//! [`SpartanAstEmitter`]: converts AST nodes into R1CS constraints.

use std::collections::BTreeMap;

use jolt_field::Field;
use jolt_ir::{LcTerm, LinearCombination, R1csConstraint, R1csVar};

use crate::ast_emitter::AstEmitter;
use crate::scalar_ops;

/// How to compute an auxiliary variable's witness value.
#[derive(Debug, Clone)]
enum WitnessStep<F> {
    /// `result = eval(a) * eval(b)`
    Product {
        result: R1csVar,
        a: LinearCombination<F>,
        b: LinearCombination<F>,
    },
    /// `result = 1 / eval(of)`
    Inverse {
        result: R1csVar,
        of: LinearCombination<F>,
    },
    /// `result = eval(num) / eval(den)`
    Quotient {
        result: R1csVar,
        num: LinearCombination<F>,
        den: LinearCombination<F>,
    },
}

/// Maps a bundle variable to an R1CS witness variable.
#[derive(Debug, Clone)]
pub struct InputMapping {
    /// Variable index from the [`AstBundle`](crate::bundle::AstBundle).
    pub bundle_index: u32,
    /// Human-readable name.
    pub name: String,
    /// Corresponding R1CS witness variable.
    pub var: R1csVar,
}

/// R1CS constraint system produced by [`SpartanAstEmitter`].
///
/// Contains everything needed to build a witness and feed into a Spartan prover:
/// constraints, variable count, input mapping, and witness computation recipe.
#[derive(Debug)]
pub struct SpartanCircuit<F: Field> {
    /// R1CS constraints: `A · B = C`.
    pub constraints: Vec<R1csConstraint<F>>,
    /// Total number of witness variables (including `z[0] = 1`).
    pub num_variables: u32,
    /// Maps bundle variable indices to R1CS witness variables.
    pub inputs: Vec<InputMapping>,
    /// Ordered steps for computing auxiliary witness values.
    witness_steps: Vec<WitnessStep<F>>,
}

impl<F: Field> SpartanCircuit<F> {
    /// Builds a complete witness vector from input values.
    ///
    /// Each entry in `input_values` is `(bundle_variable_index, value)`.
    /// The unit variable `z[0] = 1` is set automatically.
    /// Auxiliary variables are forward-evaluated from the constraint structure.
    pub fn build_witness(&self, input_values: &[(u32, F)]) -> Vec<F> {
        let mut witness = vec![F::zero(); self.num_variables as usize];
        witness[0] = F::one();

        let index_map: BTreeMap<u32, usize> = self
            .inputs
            .iter()
            .map(|m| (m.bundle_index, m.var.index()))
            .collect();

        for &(bundle_idx, val) in input_values {
            if let Some(&wit_idx) = index_map.get(&bundle_idx) {
                witness[wit_idx] = val;
            }
        }

        for step in &self.witness_steps {
            match step {
                WitnessStep::Product { result, a, b } => {
                    witness[result.index()] = a.evaluate(&witness) * b.evaluate(&witness);
                }
                WitnessStep::Inverse { result, of } => {
                    let val = of.evaluate(&witness);
                    witness[result.index()] = val
                        .inverse()
                        .expect("inverse of zero in witness computation");
                }
                WitnessStep::Quotient { result, num, den } => {
                    let n = num.evaluate(&witness);
                    let d = den.evaluate(&witness);
                    witness[result.index()] = n * d
                        .inverse()
                        .expect("division by zero in witness computation");
                }
            }
        }

        witness
    }

    /// Checks whether a witness satisfies all constraints.
    pub fn is_satisfied(&self, witness: &[F]) -> bool {
        self.constraints.iter().all(|c| c.is_satisfied(witness))
    }

    /// Returns sparse matrix entries `(row, col, value)` for A, B, C matrices.
    ///
    /// Suitable for constructing a `SimpleR1CS` or `SpartanKey`.
    #[allow(clippy::type_complexity)]
    pub fn sparse_entries(
        &self,
    ) -> (
        Vec<(usize, usize, F)>,
        Vec<(usize, usize, F)>,
        Vec<(usize, usize, F)>,
    ) {
        let extract = |select: fn(&R1csConstraint<F>) -> &LinearCombination<F>| {
            let mut entries = Vec::new();
            for (row, c) in self.constraints.iter().enumerate() {
                for term in &select(c).terms {
                    if !term.coeff.is_zero() {
                        entries.push((row, term.var.index(), term.coeff));
                    }
                }
            }
            entries
        };

        (extract(|c| &c.a), extract(|c| &c.b), extract(|c| &c.c))
    }

    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }
}

/// R1CS-emitting backend for the [`AstEmitter`] trait.
///
/// Converts wrapper AST nodes into R1CS constraints. Call [`finish`](Self::finish)
/// to obtain the [`SpartanCircuit`].
pub struct SpartanAstEmitter<F: Field> {
    next_var: u32,
    constraints: Vec<R1csConstraint<F>>,
    inputs: Vec<InputMapping>,
    witness_steps: Vec<WitnessStep<F>>,
}

impl<F: Field> SpartanAstEmitter<F> {
    /// Creates a new emitter. Variable index 0 is reserved for the unit
    /// variable (`z[0] = 1`).
    pub fn new() -> Self {
        Self {
            next_var: 1,
            constraints: Vec::new(),
            inputs: Vec::new(),
            witness_steps: Vec::new(),
        }
    }

    /// Consumes the emitter and returns the completed constraint system.
    pub fn finish(self) -> SpartanCircuit<F> {
        SpartanCircuit {
            constraints: self.constraints,
            num_variables: self.next_var,
            inputs: self.inputs,
            witness_steps: self.witness_steps,
        }
    }

    fn alloc_var(&mut self) -> R1csVar {
        let var = R1csVar(self.next_var);
        self.next_var += 1;
        var
    }

    fn limbs_to_field(val: [u64; 4]) -> F {
        F::from_bytes(&scalar_ops::to_bytes_le(val))
    }

    fn unit_lc() -> LinearCombination<F> {
        LinearCombination::constant(F::one())
    }

    /// Computes `2^192` as a field element.
    fn two_pow_192() -> F {
        // 2^192 = (2^64)^3
        let two_64 = F::from_u128(1u128 << 64);
        two_64 * two_64 * two_64
    }
}

impl<F: Field> Default for SpartanAstEmitter<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field> AstEmitter for SpartanAstEmitter<F> {
    type Wire = LinearCombination<F>;

    fn constant(&mut self, val: [u64; 4]) -> Self::Wire {
        LinearCombination::constant(Self::limbs_to_field(val))
    }

    fn variable(&mut self, index: u32, name: &str) -> Self::Wire {
        let var = self.alloc_var();
        self.inputs.push(InputMapping {
            bundle_index: index,
            name: name.to_owned(),
            var,
        });
        LinearCombination::variable(var)
    }

    fn neg(&mut self, inner: Self::Wire) -> Self::Wire {
        LinearCombination {
            terms: inner
                .terms
                .into_iter()
                .map(|t| LcTerm {
                    var: t.var,
                    coeff: -t.coeff,
                })
                .collect(),
        }
    }

    fn inv(&mut self, mut inner: Self::Wire) -> Self::Wire {
        // Constraint: inner * result = 1
        inner.compact();
        let result = self.alloc_var();
        let result_lc = LinearCombination::variable(result);
        self.constraints.push(R1csConstraint {
            a: inner.clone(),
            b: result_lc.clone(),
            c: Self::unit_lc(),
        });
        self.witness_steps
            .push(WitnessStep::Inverse { result, of: inner });
        result_lc
    }

    fn add(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire {
        let mut terms = lhs.terms;
        terms.extend(rhs.terms);
        let mut lc = LinearCombination { terms };
        // Compact when term count grows large to prevent O(n²) blowup
        // in deeply chained additions.
        if lc.terms.len() > 16 {
            lc.compact();
        }
        lc
    }

    fn sub(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire {
        let neg_rhs = self.neg(rhs);
        self.add(lhs, neg_rhs)
    }

    fn mul(&mut self, mut lhs: Self::Wire, mut rhs: Self::Wire) -> Self::Wire {
        // Constraint: lhs * rhs = result
        lhs.compact();
        rhs.compact();
        let result = self.alloc_var();
        let result_lc = LinearCombination::variable(result);
        self.constraints.push(R1csConstraint {
            a: lhs.clone(),
            b: rhs.clone(),
            c: result_lc.clone(),
        });
        self.witness_steps.push(WitnessStep::Product {
            result,
            a: lhs,
            b: rhs,
        });
        result_lc
    }

    fn div(&mut self, mut lhs: Self::Wire, mut rhs: Self::Wire) -> Self::Wire {
        // Constraint: rhs * result = lhs  (i.e., result = lhs / rhs)
        lhs.compact();
        rhs.compact();
        let result = self.alloc_var();
        let result_lc = LinearCombination::variable(result);
        self.constraints.push(R1csConstraint {
            a: rhs.clone(),
            b: result_lc.clone(),
            c: lhs.clone(),
        });
        self.witness_steps.push(WitnessStep::Quotient {
            result,
            num: lhs,
            den: rhs,
        });
        result_lc
    }

    fn poseidon(
        &mut self,
        _state: Self::Wire,
        _n_rounds: Self::Wire,
        _data: Self::Wire,
    ) -> Self::Wire {
        panic!(
            "Poseidon R1CS gadget not available. \
             Poseidon nodes are for SNARK-in-SNARK pipelines (gnark backend). \
             For Spartan, use a native transcript — the AST should not contain Poseidon nodes."
        );
    }

    fn byte_reverse(&mut self, _inner: Self::Wire) -> Self::Wire {
        panic!(
            "ByteReverse R1CS gadget not available. \
             ByteReverse nodes are for SNARK-in-SNARK pipelines (gnark backend). \
             For Spartan, the AST should not contain ByteReverse nodes."
        );
    }

    fn truncate_128(&mut self, _inner: Self::Wire) -> Self::Wire {
        panic!(
            "Truncate128 R1CS gadget not available. \
             Truncate128 nodes are for SNARK-in-SNARK pipelines (gnark backend). \
             For Spartan, the AST should not contain Truncate128 nodes."
        );
    }

    fn mul_two_pow_192(&mut self, inner: Self::Wire) -> Self::Wire {
        let scale = Self::two_pow_192();
        LinearCombination {
            terms: inner
                .terms
                .into_iter()
                .map(|t| LcTerm {
                    var: t.var,
                    coeff: t.coeff * scale,
                })
                .collect(),
        }
    }

    fn assert_zero(&mut self, mut expr: Self::Wire) {
        // Constraint: expr * 1 = 0
        expr.compact();
        self.constraints.push(R1csConstraint {
            a: expr,
            b: Self::unit_lc(),
            c: LinearCombination::new(),
        });
    }

    fn assert_equal(&mut self, lhs: Self::Wire, rhs: Self::Wire) {
        let diff = self.sub(lhs, rhs);
        self.assert_zero(diff);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::ArenaSession;
    use crate::bundle::VarAllocator;
    use crate::symbolic::SymbolicField;
    use jolt_field::{Field, Fr};
    use num_traits::{One, Zero};

    #[test]
    fn constant_is_free() {
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let wire = emitter.constant(scalar_ops::from_u64(42));
        assert_eq!(wire.terms.len(), 1);
        assert_eq!(wire.terms[0].var, R1csVar(0)); // unit variable
        assert_eq!(wire.terms[0].coeff, Fr::from_u64(42));
        assert_eq!(emitter.constraints.len(), 0);
    }

    #[test]
    fn variable_allocates_fresh_var() {
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let w1 = emitter.variable(0, "x");
        let w2 = emitter.variable(1, "y");
        assert_eq!(w1.terms[0].var, R1csVar(1));
        assert_eq!(w2.terms[0].var, R1csVar(2));
        assert_eq!(emitter.inputs.len(), 2);
        assert_eq!(emitter.constraints.len(), 0);
    }

    #[test]
    fn add_sub_neg_are_free() {
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let x = emitter.variable(0, "x");
        let y = emitter.variable(1, "y");

        let sum = emitter.add(x.clone(), y.clone());
        assert_eq!(sum.terms.len(), 2);
        assert_eq!(emitter.constraints.len(), 0);

        let diff = emitter.sub(x.clone(), y.clone());
        assert_eq!(diff.terms.len(), 2);
        assert_eq!(emitter.constraints.len(), 0);

        let neg_x = emitter.neg(x);
        assert_eq!(neg_x.terms.len(), 1);
        assert_eq!(neg_x.terms[0].coeff, -Fr::one());
        assert_eq!(emitter.constraints.len(), 0);
    }

    #[test]
    fn mul_emits_constraint() {
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let x = emitter.variable(0, "x");
        let y = emitter.variable(1, "y");
        let product = emitter.mul(x, y);

        assert_eq!(emitter.constraints.len(), 1);
        // Result is a fresh variable
        assert_eq!(product.terms.len(), 1);
        assert_eq!(product.terms[0].var, R1csVar(3)); // after x=1, y=2
    }

    #[test]
    fn inv_emits_constraint() {
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let x = emitter.variable(0, "x");
        let inv_x = emitter.inv(x);

        assert_eq!(emitter.constraints.len(), 1);
        // Constraint: x * inv_x = 1
        let c = &emitter.constraints[0];
        assert_eq!(c.c.terms[0].var, R1csVar(0)); // C = unit (constant 1)
        assert_eq!(inv_x.terms[0].var, R1csVar(2));
    }

    #[test]
    fn div_emits_constraint() {
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let x = emitter.variable(0, "x");
        let y = emitter.variable(1, "y");
        let quot = emitter.div(x, y);

        assert_eq!(emitter.constraints.len(), 1);
        // Constraint: y * quot = x
        let c = &emitter.constraints[0];
        assert_eq!(c.a.terms[0].var, R1csVar(2)); // A = y
        assert_eq!(c.b.terms[0].var, R1csVar(3)); // B = quot (fresh)
        assert_eq!(c.c.terms[0].var, R1csVar(1)); // C = x
        assert_eq!(quot.terms[0].var, R1csVar(3));
    }

    #[test]
    fn assert_zero_emits_constraint() {
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let x = emitter.variable(0, "x");
        emitter.assert_zero(x);

        assert_eq!(emitter.constraints.len(), 1);
        let c = &emitter.constraints[0];
        // A = x, B = 1, C = 0
        assert_eq!(c.a.terms[0].var, R1csVar(1));
        assert_eq!(c.b.terms[0].var, R1csVar(0));
        assert!(c.c.terms.is_empty());
    }

    #[test]
    fn assert_equal_emits_constraint() {
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let x = emitter.variable(0, "x");
        let y = emitter.variable(1, "y");
        emitter.assert_equal(x, y);

        assert_eq!(emitter.constraints.len(), 1);
        let c = &emitter.constraints[0];
        // A = (x - y), B = 1, C = 0
        assert_eq!(c.a.terms.len(), 2);
    }

    #[test]
    fn mul_two_pow_192_is_free() {
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let x = emitter.variable(0, "x");
        let scaled = emitter.mul_two_pow_192(x);

        assert_eq!(emitter.constraints.len(), 0);
        assert_eq!(scaled.terms.len(), 1);
        assert_eq!(scaled.terms[0].var, R1csVar(1)); // same var, scaled coeff
        let expected = SpartanAstEmitter::<Fr>::two_pow_192();
        assert_eq!(scaled.terms[0].coeff, expected);
    }

    #[test]
    fn witness_mul_satisfaction() {
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let x = emitter.variable(0, "x");
        let y = emitter.variable(1, "y");
        let product = emitter.mul(x, y);
        emitter.assert_zero(product);

        let circuit = emitter.finish();

        // x=0 * y=7 = 0 → satisfies assert_zero
        let witness = circuit.build_witness(&[(0, Fr::from_u64(0)), (1, Fr::from_u64(7))]);
        assert!(circuit.is_satisfied(&witness));

        // x=3 * y=5 = 15 ≠ 0 → violates assert_zero
        let witness = circuit.build_witness(&[(0, Fr::from_u64(3)), (1, Fr::from_u64(5))]);
        assert!(!circuit.is_satisfied(&witness));
    }

    #[test]
    fn witness_inv_satisfaction() {
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let x = emitter.variable(0, "x");
        let inv_x = emitter.inv(x.clone());
        // x * inv(x) - 1 = 0
        let product = emitter.mul(x, inv_x);
        let one = emitter.constant(scalar_ops::from_u64(1));
        let diff = emitter.sub(product, one);
        emitter.assert_zero(diff);

        let circuit = emitter.finish();
        let witness = circuit.build_witness(&[(0, Fr::from_u64(7))]);
        assert!(circuit.is_satisfied(&witness));
    }

    #[test]
    fn witness_div_satisfaction() {
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let x = emitter.variable(0, "x");
        let y = emitter.variable(1, "y");
        let quot = emitter.div(x, y.clone());
        // y * (x / y) should equal x — test via: y * quot - x = 0
        let reconstructed = emitter.mul(y, quot);
        let x2 = emitter.variable(2, "x_copy");
        let diff = emitter.sub(reconstructed, x2);
        emitter.assert_zero(diff);

        let circuit = emitter.finish();
        let witness = circuit.build_witness(&[
            (0, Fr::from_u64(21)),
            (1, Fr::from_u64(7)),
            (2, Fr::from_u64(21)),
        ]);
        assert!(circuit.is_satisfied(&witness));
    }

    #[test]
    fn witness_complex_expression() {
        // 3*x + 7*y - 42 = 0  with x=7, y=3 → 21 + 21 - 42 = 0
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let x = emitter.variable(0, "x");
        let y = emitter.variable(1, "y");
        let three = emitter.constant(scalar_ops::from_u64(3));
        let seven = emitter.constant(scalar_ops::from_u64(7));
        let forty_two = emitter.constant(scalar_ops::from_u64(42));

        let three_x = emitter.mul(three, x);
        let seven_y = emitter.mul(seven, y);
        let sum = emitter.add(three_x, seven_y);
        let constraint = emitter.sub(sum, forty_two);
        emitter.assert_zero(constraint);

        let circuit = emitter.finish();
        let witness = circuit.build_witness(&[(0, Fr::from_u64(7)), (1, Fr::from_u64(3))]);
        assert!(circuit.is_satisfied(&witness));
    }

    #[test]
    fn additions_minimize_constraints() {
        // x + y + z = 0 should need only 1 constraint (the assertion)
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let x = emitter.variable(0, "x");
        let y = emitter.variable(1, "y");
        let z = emitter.variable(2, "z");

        let sum = emitter.add(x, y);
        let sum = emitter.add(sum, z);
        emitter.assert_zero(sum);

        let circuit = emitter.finish();
        assert_eq!(circuit.num_constraints(), 1);
    }

    #[test]
    fn full_pipeline_from_symbolic() {
        let _session = ArenaSession::new();

        let x = SymbolicField::variable(0, "x");
        let y = SymbolicField::variable(1, "y");
        let constraint = x * y;

        let mut alloc = VarAllocator::new();
        let _ = alloc.input("x");
        let _ = alloc.input("y");
        alloc.assert_zero(constraint.into_edge());
        let bundle = alloc.finish();

        let mut emitter = SpartanAstEmitter::<Fr>::new();
        bundle.emit(&mut emitter);
        let circuit = emitter.finish();

        // x=0 satisfies x*y=0
        let witness = circuit.build_witness(&[(0, Fr::from_u64(0)), (1, Fr::from_u64(42))]);
        assert!(circuit.is_satisfied(&witness));

        // x=3, y=5 → 15 ≠ 0
        let witness = circuit.build_witness(&[(0, Fr::from_u64(3)), (1, Fr::from_u64(5))]);
        assert!(!circuit.is_satisfied(&witness));
    }

    #[test]
    fn full_pipeline_booleanity() {
        let _session = ArenaSession::new();

        // γ · (H² − H) == 0  (booleanity check)
        let h = SymbolicField::variable(0, "H");
        let gamma = SymbolicField::variable(1, "gamma");
        let constraint = gamma * (h * h - h);

        let mut alloc = VarAllocator::new();
        let _ = alloc.input("H");
        let _ = alloc.input("gamma");
        alloc.assert_zero(constraint.into_edge());
        let bundle = alloc.finish();

        let mut emitter = SpartanAstEmitter::<Fr>::new();
        bundle.emit(&mut emitter);
        let circuit = emitter.finish();

        // H=1 is boolean → constraint satisfied
        let witness = circuit.build_witness(&[(0, Fr::from_u64(1)), (1, Fr::from_u64(99))]);
        assert!(circuit.is_satisfied(&witness));

        // H=0 is boolean → constraint satisfied
        let witness = circuit.build_witness(&[(0, Fr::from_u64(0)), (1, Fr::from_u64(99))]);
        assert!(circuit.is_satisfied(&witness));

        // H=2 is not boolean → constraint violated
        let witness = circuit.build_witness(&[(0, Fr::from_u64(2)), (1, Fr::from_u64(1))]);
        assert!(!circuit.is_satisfied(&witness));
    }

    #[test]
    fn full_pipeline_assert_equal() {
        let _session = ArenaSession::new();

        let x = SymbolicField::variable(0, "x");
        let y = SymbolicField::variable(1, "y");

        let mut alloc = VarAllocator::new();
        let _ = alloc.input("x");
        let _ = alloc.input("y");
        alloc.assert_equal(x.into_edge(), y.into_edge());
        let bundle = alloc.finish();

        let mut emitter = SpartanAstEmitter::<Fr>::new();
        bundle.emit(&mut emitter);
        let circuit = emitter.finish();

        // x == y → satisfied
        let witness = circuit.build_witness(&[(0, Fr::from_u64(42)), (1, Fr::from_u64(42))]);
        assert!(circuit.is_satisfied(&witness));

        // x != y → violated
        let witness = circuit.build_witness(&[(0, Fr::from_u64(42)), (1, Fr::from_u64(43))]);
        assert!(!circuit.is_satisfied(&witness));
    }

    #[test]
    fn sparse_entries_match_constraints() {
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let x = emitter.variable(0, "x");
        let y = emitter.variable(1, "y");
        let product = emitter.mul(x, y);
        emitter.assert_zero(product);

        let circuit = emitter.finish();
        let (a_entries, b_entries, c_entries) = circuit.sparse_entries();

        // 2 constraints: mul + assert_zero
        assert_eq!(circuit.num_constraints(), 2);
        assert!(!a_entries.is_empty());
        assert!(!b_entries.is_empty());
        // C has entry for mul (aux var), but assert_zero has empty C
        assert!(!c_entries.is_empty());
    }

    #[test]
    #[should_panic(expected = "Poseidon R1CS gadget not available")]
    fn poseidon_panics() {
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let x = emitter.variable(0, "x");
        let _ = emitter.poseidon(x.clone(), x.clone(), x);
    }

    #[test]
    #[should_panic(expected = "ByteReverse R1CS gadget not available")]
    fn byte_reverse_panics() {
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let x = emitter.variable(0, "x");
        let _ = emitter.byte_reverse(x);
    }

    #[test]
    #[should_panic(expected = "Truncate128 R1CS gadget not available")]
    fn truncate_128_panics() {
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let x = emitter.variable(0, "x");
        let _ = emitter.truncate_128(x);
    }

    #[test]
    fn mul_two_pow_192_witness() {
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let x = emitter.variable(0, "x");
        let scaled = emitter.mul_two_pow_192(x);
        let expected_val = emitter.constant(scalar_ops::from_u64(0));
        // scaled - expected = 0 when x * 2^192 = expected
        let diff = emitter.sub(scaled, expected_val);
        emitter.assert_zero(diff);

        let circuit = emitter.finish();
        // x=0 → 0 * 2^192 = 0
        let witness = circuit.build_witness(&[(0, Fr::from_u64(0))]);
        assert!(circuit.is_satisfied(&witness));
    }

    #[test]
    fn empty_circuit() {
        let emitter = SpartanAstEmitter::<Fr>::new();
        let circuit = emitter.finish();
        assert_eq!(circuit.num_constraints(), 0);
        assert_eq!(circuit.num_variables, 1); // just the unit var
        let witness = circuit.build_witness(&[]);
        assert!(circuit.is_satisfied(&witness));
    }

    #[test]
    fn lc_compaction_merges_duplicates() {
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let x = emitter.variable(0, "x");
        // x + x + x should compact to 3*x
        let sum = emitter.add(x.clone(), x.clone());
        let sum = emitter.add(sum, x);
        emitter.assert_zero(sum);
        let circuit = emitter.finish();

        // The assertion constraint's A side should have compacted terms
        let a_terms = &circuit.constraints[0].a.terms;
        assert_eq!(a_terms.len(), 1, "should compact to single term");
        assert_eq!(a_terms[0].coeff, Fr::from_u64(3));
    }

    #[test]
    fn lc_compaction_cancels_zeros() {
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let x = emitter.variable(0, "x");
        // x - x should compact to empty LC
        let diff = emitter.sub(x.clone(), x);
        emitter.assert_zero(diff);
        let circuit = emitter.finish();

        // Should be trivially satisfied for any witness
        let witness = circuit.build_witness(&[(0, Fr::from_u64(999))]);
        assert!(circuit.is_satisfied(&witness));
    }

    #[test]
    fn many_variables() {
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let n = 64;
        let vars: Vec<_> = (0..n)
            .map(|i| emitter.variable(i, &format!("v{i}")))
            .collect();

        // sum of all variables = 0
        let mut acc = vars[0].clone();
        for v in &vars[1..] {
            acc = emitter.add(acc, v.clone());
        }
        emitter.assert_zero(acc);

        let circuit = emitter.finish();
        assert_eq!(circuit.num_constraints(), 1);

        // All zeros → satisfied
        let inputs: Vec<_> = (0..n).map(|i| (i, Fr::zero())).collect();
        let witness = circuit.build_witness(&inputs);
        assert!(circuit.is_satisfied(&witness));
    }

    #[test]
    fn randomized_quadratic_constraint() {
        use rand_chacha::ChaCha8Rng;
        use rand_core::SeedableRng;

        let mut rng = ChaCha8Rng::seed_from_u64(0xdead_beef);

        for _ in 0..50 {
            let _session = ArenaSession::new();

            // a*b - c = 0, with c = a*b
            let a_val = Fr::random(&mut rng);
            let b_val = Fr::random(&mut rng);
            let c_val = a_val * b_val;

            let a = SymbolicField::variable(0, "a");
            let b = SymbolicField::variable(1, "b");
            let c = SymbolicField::variable(2, "c");

            let mut alloc = VarAllocator::new();
            let _ = alloc.input("a");
            let _ = alloc.input("b");
            let _ = alloc.input("c");
            alloc.assert_zero((a * b - c).into_edge());
            let bundle = alloc.finish();

            let mut emitter = SpartanAstEmitter::<Fr>::new();
            bundle.emit(&mut emitter);
            let circuit = emitter.finish();

            let witness = circuit.build_witness(&[(0, a_val), (1, b_val), (2, c_val)]);
            assert!(circuit.is_satisfied(&witness));

            // Wrong c value should fail
            let bad_witness =
                circuit.build_witness(&[(0, a_val), (1, b_val), (2, c_val + Fr::one())]);
            assert!(!circuit.is_satisfied(&bad_witness));
        }
    }

    #[test]
    fn computed_assert_equal() {
        // assert_equal(x*y, z*w) — both sides require computation
        let _session = ArenaSession::new();

        let x = SymbolicField::variable(0, "x");
        let y = SymbolicField::variable(1, "y");
        let z = SymbolicField::variable(2, "z");
        let w = SymbolicField::variable(3, "w");

        let mut alloc = VarAllocator::new();
        let _ = alloc.input("x");
        let _ = alloc.input("y");
        let _ = alloc.input("z");
        let _ = alloc.input("w");
        alloc.assert_equal((x * y).into_edge(), (z * w).into_edge());
        let bundle = alloc.finish();

        let mut emitter = SpartanAstEmitter::<Fr>::new();
        bundle.emit(&mut emitter);
        let circuit = emitter.finish();

        // 3*7 = 21 = 21*1
        let witness = circuit.build_witness(&[
            (0, Fr::from_u64(3)),
            (1, Fr::from_u64(7)),
            (2, Fr::from_u64(21)),
            (3, Fr::from_u64(1)),
        ]);
        assert!(circuit.is_satisfied(&witness));

        // 3*7 = 21 ≠ 20 = 4*5
        let witness = circuit.build_witness(&[
            (0, Fr::from_u64(3)),
            (1, Fr::from_u64(7)),
            (2, Fr::from_u64(4)),
            (3, Fr::from_u64(5)),
        ]);
        assert!(!circuit.is_satisfied(&witness));
    }

    #[test]
    fn chained_multiplications() {
        // x * y * z = result, assert result == expected
        let mut emitter = SpartanAstEmitter::<Fr>::new();
        let x = emitter.variable(0, "x");
        let y = emitter.variable(1, "y");
        let z = emitter.variable(2, "z");
        let expected = emitter.variable(3, "expected");

        let xy = emitter.mul(x, y);
        let xyz = emitter.mul(xy, z);
        emitter.assert_equal(xyz, expected);

        let circuit = emitter.finish();

        // 2*3*5 = 30
        let witness = circuit.build_witness(&[
            (0, Fr::from_u64(2)),
            (1, Fr::from_u64(3)),
            (2, Fr::from_u64(5)),
            (3, Fr::from_u64(30)),
        ]);
        assert!(circuit.is_satisfied(&witness));

        // Wrong expected value
        let witness = circuit.build_witness(&[
            (0, Fr::from_u64(2)),
            (1, Fr::from_u64(3)),
            (2, Fr::from_u64(5)),
            (3, Fr::from_u64(31)),
        ]);
        assert!(!circuit.is_satisfied(&witness));
    }
}
