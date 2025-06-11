use super::{
    key::{CrossStepR1CS, CrossStepR1CSConstraint, SparseEqualityItem},
    ops::{Term, Variable, LC},
};
use crate::{
    field::JoltField,
    r1cs::key::{SparseConstraints, UniformR1CS},
};
use crate::{poly::multilinear_polynomial::MultilinearPolynomial, r1cs::inputs::JoltR1CSInputs};
use std::marker::PhantomData;

/// Constraints over a single row. Each variable points to a single item in Z and the corresponding coefficient.
#[derive(Clone)]
pub struct Constraint {
    pub a: LC,
    pub b: LC,
    pub c: LC,
}

impl Constraint {
    #[cfg(test)]
    pub(crate) fn _pretty_fmt<F: JoltField>(
        &self,
        f: &mut String,
        flattened_polynomials: &[MultilinearPolynomial<F>],
        step_index: usize,
    ) -> std::fmt::Result {
        use std::fmt::Write as _;

        use crate::r1cs::inputs::JoltR1CSInputs;

        self.a.pretty_fmt(f)?;
        write!(f, " ⋅ ")?;
        self.b.pretty_fmt(f)?;
        write!(f, " == ")?;
        self.c.pretty_fmt(f)?;
        writeln!(f)?;

        let mut terms = Vec::new();
        for term in self
            .a
            .terms()
            .iter()
            .chain(self.b.terms().iter())
            .chain(self.c.terms().iter())
        {
            if !terms.contains(term) {
                terms.push(*term);
            }
        }

        for term in terms {
            match term.0 {
                Variable::Input(var_index) => {
                    writeln!(
                        f,
                        "    {:?} = {}",
                        JoltR1CSInputs::from_index(var_index),
                        flattened_polynomials[var_index].get_coeff(step_index)
                    )?;
                }
                Variable::Constant => {}
            }
        }

        Ok(())
    }
}

#[derive(Default)]
pub struct R1CSBuilder {
    pub(crate) constraints: Vec<Constraint>,
}

impl R1CSBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn constrain_eq(&mut self, left: impl Into<LC>, right: impl Into<LC>) {
        // left - right == 0
        let left: LC = left.into();
        let right: LC = right.into();

        let a = left - right.clone();
        let b = Variable::Constant.into();
        let constraint = Constraint {
            a,
            b,
            c: LC::zero(),
        };
        self.constraints.push(constraint);
    }

    pub fn constrain_eq_conditional(
        &mut self,
        condition: impl Into<LC>,
        left: impl Into<LC>,
        right: impl Into<LC>,
    ) {
        // condition  * (left - right) == 0
        let condition: LC = condition.into();
        let left: LC = left.into();
        let right: LC = right.into();

        let a = condition;
        let b = left - right;
        let c = LC::zero();
        let constraint = Constraint { a, b, c }; // TODO(sragss): Can do better on middle term.
        self.constraints.push(constraint);
    }

    pub fn constrain_binary(&mut self, value: impl Into<LC>) {
        let one: LC = Variable::Constant.into();
        let a: LC = value.into();
        let b = one - a.clone();
        // value * (1 - value) == 0
        let constraint = Constraint {
            a,
            b,
            c: LC::zero(),
        };
        self.constraints.push(constraint);
    }

    pub fn constrain_if_else(
        &mut self,
        condition: impl Into<LC>,
        result_true: impl Into<LC>,
        result_false: impl Into<LC>,
        alleged_result: impl Into<LC>,
    ) {
        let condition: LC = condition.into();
        let result_true: LC = result_true.into();
        let result_false: LC = result_false.into();
        let alleged_result: LC = alleged_result.into();

        // result == condition * true_coutcome + (1 - condition) * false_outcome
        // simplify to single mul, single constraint => condition * (true_outcome - false_outcome) == (result - false_outcome)

        let constraint = Constraint {
            a: condition.clone(),
            b: result_true - result_false.clone(),
            c: alleged_result - result_false,
        };
        self.constraints.push(constraint);
    }

    pub fn pack_le(unpacked: Vec<Variable>, operand_bits: usize) -> LC {
        let packed: Vec<Term> = unpacked
            .into_iter()
            .enumerate()
            .map(|(idx, unpacked)| Term(unpacked, 1 << (idx * operand_bits)))
            .collect();
        packed.into()
    }

    pub fn pack_be(unpacked: Vec<Variable>, operand_bits: usize) -> LC {
        let packed: Vec<Term> = unpacked
            .into_iter()
            .rev()
            .enumerate()
            .map(|(idx, unpacked)| Term(unpacked, 1 << (idx * operand_bits)))
            .collect();
        packed.into()
    }

    pub fn constrain_pack_le(
        &mut self,
        unpacked: Vec<Variable>,
        result: impl Into<LC>,
        operand_bits: usize,
    ) {
        // Pack unpacked via a simple weighted linear combination
        // A + 2 * B + 4 * C + 8 * D, ...
        let packed: Vec<Term> = unpacked
            .into_iter()
            .enumerate()
            .map(|(idx, unpacked)| Term(unpacked, 1 << (idx * operand_bits)))
            .collect();
        self.constrain_eq(packed, result);
    }

    pub fn constrain_pack_be(
        &mut self,
        unpacked: Vec<Variable>,
        result: impl Into<LC>,
        operand_bits: usize,
    ) {
        // Pack unpacked via a simple weighted linear combination
        // A + 2 * B + 4 * C + 8 * D, ...
        // Note: Packing order is reversed from constrain_pack_le
        let packed: Vec<Term> = unpacked
            .into_iter()
            .rev()
            .enumerate()
            .map(|(idx, unpacked)| Term(unpacked, 1 << (idx * operand_bits)))
            .collect();
        self.constrain_eq(packed, result);
    }

    /// Constrain x * y == z
    pub fn constrain_prod(&mut self, x: impl Into<LC>, y: impl Into<LC>, z: impl Into<LC>) {
        let constraint = Constraint {
            a: x.into(),
            b: y.into(),
            c: z.into(),
        };
        self.constraints.push(constraint);
    }

    fn materialize<F: JoltField>(&self) -> UniformR1CS<F> {
        let a_len: usize = self.constraints.iter().map(|c| c.a.num_vars()).sum();
        let b_len: usize = self.constraints.iter().map(|c| c.b.num_vars()).sum();
        let c_len: usize = self.constraints.iter().map(|c| c.c.num_vars()).sum();
        let mut a_sparse = SparseConstraints::empty_with_capacity(a_len, self.constraints.len());
        let mut b_sparse = SparseConstraints::empty_with_capacity(b_len, self.constraints.len());
        let mut c_sparse = SparseConstraints::empty_with_capacity(c_len, self.constraints.len());

        let update_sparse = |row_index: usize, lc: &LC, sparse: &mut SparseConstraints<F>| {
            lc.terms().iter().for_each(|term| {
                match term.0 {
                    Variable::Input(inner) => {
                        sparse.vars.push((row_index, inner, F::from_i64(term.1)))
                    }
                    Variable::Constant => {}
                };
            });
            if let Some(term) = lc.constant_term() {
                sparse.consts.push((row_index, F::from_i64(term.1)));
            }
        };

        for (row_index, constraint) in self.constraints.iter().enumerate() {
            update_sparse(row_index, &constraint.a, &mut a_sparse);
            update_sparse(row_index, &constraint.b, &mut b_sparse);
            update_sparse(row_index, &constraint.c, &mut c_sparse);
        }

        assert_eq!(a_sparse.vars.len(), a_len);
        assert_eq!(b_sparse.vars.len(), b_len);
        assert_eq!(c_sparse.vars.len(), c_len);

        UniformR1CS::<F> {
            a: a_sparse,
            b: b_sparse,
            c: c_sparse,
            num_vars: JoltR1CSInputs::num_inputs(),
            num_rows: self.constraints.len(),
        }
    }

    pub fn get_constraints(&self) -> Vec<Constraint> {
        self.constraints.clone()
    }
}

/// An Offset Linear Combination. If OffsetLC.0 is true, then the OffsetLC.1 refers to the next step in a uniform
/// constraint system.
pub type OffsetLC = (bool, LC);

/// A conditional constraint that Linear Combinations a, b are equal where a and b need not be in the same step an a
/// uniform constraint system.
#[derive(Debug)]
pub struct OffsetEqConstraint {
    pub cond: OffsetLC,
    pub a: OffsetLC,
    pub b: OffsetLC,
}

impl OffsetEqConstraint {
    pub fn new(
        condition: (impl Into<LC>, bool),
        a: (impl Into<LC>, bool),
        b: (impl Into<LC>, bool),
    ) -> Self {
        Self {
            cond: (condition.1, condition.0.into()),
            a: (a.1, a.0.into()),
            b: (b.1, b.0.into()),
        }
    }

    #[cfg(test)]
    pub fn empty() -> Self {
        Self::new(
            (LC::new(vec![]), false),
            (LC::new(vec![]), false),
            (LC::new(vec![]), false),
        )
    }
}

pub(crate) fn eval_offset_lc<F: JoltField>(
    offset: &OffsetLC,
    flattened_polynomials: &[MultilinearPolynomial<F>],
    step: usize,
    next_step_m: Option<usize>,
) -> i128 {
    if !offset.0 {
        offset.1.evaluate_row(flattened_polynomials, step)
    } else if let Some(next_step) = next_step_m {
        offset.1.evaluate_row(flattened_polynomials, next_step)
    } else {
        offset.1.constant_term_field()
    }
}

pub(crate) fn shard_last_step_eval_offset_lc<F: JoltField>(
    offset: &OffsetLC,
    flattened_polynomials: &[MultilinearPolynomial<F>],
    flattened_eval: &[MultilinearPolynomial<F>],
    step: usize,
    next_step_m: Option<usize>,
) -> i128 {
    if !offset.0 {
        offset.1.evaluate_row(flattened_polynomials, step)
    } else if next_step_m.is_some() {
        offset.1.evaluate_row(flattened_eval, 0)
    } else {
        offset.1.constant_term_field()
    }
}

// TODO(sragss): Detailed documentation with wiki.
pub struct CombinedUniformBuilder<F: JoltField> {
    _field: PhantomData<F>,
    pub(crate) uniform_builder: R1CSBuilder,

    /// Padded to the nearest power of 2
    pub(crate) uniform_repeat: usize, // TODO(JP): Remove padding of steps

    pub(crate) offset_equality_constraints: Vec<OffsetEqConstraint>,
}

impl<F: JoltField> CombinedUniformBuilder<F> {
    pub fn construct(
        uniform_builder: R1CSBuilder,
        uniform_repeat: usize,
        offset_equality_constraints: Vec<OffsetEqConstraint>,
    ) -> Self {
        assert!(uniform_repeat.is_power_of_two());
        Self {
            _field: PhantomData,
            uniform_builder,
            uniform_repeat,
            offset_equality_constraints,
        }
    }

    /// Number of constraint rows per step, padded to the next power of two.
    pub(super) fn padded_rows_per_step(&self) -> usize {
        let num_constraints =
            self.uniform_builder.constraints.len() + self.offset_equality_constraints.len();
        num_constraints.next_power_of_two()
    }

    /// Total number of rows used across all repeated constraints. Not padded to nearest power of two.
    pub(super) fn constraint_rows(&self) -> usize {
        self.uniform_repeat * self.padded_rows_per_step()
    }

    pub(super) fn uniform_repeat(&self) -> usize {
        self.uniform_repeat
    }

    /// Materializes the uniform constraints into sparse (value != 0) A, B, C matrices represented in (row, col, value) format.
    pub fn materialize_uniform(&self) -> UniformR1CS<F> {
        self.uniform_builder.materialize()
    }

    /// Converts builder::OffsetEqConstraints into key::CrossStepR1CSConstraint
    pub fn materialize_offset_eq(&self) -> CrossStepR1CS<F> {
        // (a - b) * condition == 0
        // A: a - b
        // B: condition
        // C: 0

        let mut constraints = Vec::with_capacity(self.offset_equality_constraints.len());
        for constraint in &self.offset_equality_constraints {
            let mut eq = SparseEqualityItem::<F>::empty();
            let mut condition = SparseEqualityItem::<F>::empty();

            constraint
                .cond
                .1
                .terms()
                .iter()
                .for_each(|term| match term.0 {
                    Variable::Input(inner) => {
                        condition
                            .offset_vars
                            .push((inner, constraint.cond.0, F::from_i64(term.1)))
                    }
                    Variable::Constant => {}
                });
            if let Some(term) = constraint.cond.1.constant_term() {
                condition.constant = F::from_i64(term.1);
            }

            // Can't simply combine like terms because of the offset
            let lhs = constraint.a.1.clone();
            let rhs = -constraint.b.1.clone();

            lhs.terms().iter().for_each(|term| match term.0 {
                Variable::Input(inner) => {
                    eq.offset_vars
                        .push((inner, constraint.a.0, F::from_i64(term.1)))
                }
                Variable::Constant => {}
            });
            rhs.terms().iter().for_each(|term| match term.0 {
                Variable::Input(inner) => {
                    eq.offset_vars
                        .push((inner, constraint.b.0, F::from_i64(term.1)))
                }
                Variable::Constant => {}
            });

            // Handle constants
            lhs.terms().iter().for_each(|term| {
                assert!(
                    !matches!(term.0, Variable::Constant),
                    "Constants only supported in RHS"
                )
            });
            if let Some(term) = rhs.constant_term() {
                eq.constant = F::from_i64(term.1);
            }

            constraints.push(CrossStepR1CSConstraint::new(eq, condition));
        }

        CrossStepR1CS { constraints }
    }
}
