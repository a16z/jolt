use super::ops::{AzType, AzValue, BzType, BzValue, CzType, CzValue, Term, Variable, LC};
use crate::zkvm::r1cs::inputs::JoltR1CSInputs;
use crate::{
    field::JoltField,
    zkvm::r1cs::{key::SparseConstraints, key::UniformR1CS},
};
use std::marker::PhantomData;

/// Constraints over a single row. Each variable points to a single item in Z and the corresponding coefficient.
#[derive(Clone, Debug)]
pub struct Constraint {
    pub a: LC,
    pub b: LC,
    pub c: LC,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstraintType {
    pub a_type: AzType,
    pub b_type: BzType,
    pub c_type: CzType,
}

impl Constraint {
    pub fn evaluate_typed<F: JoltField>(
        &self,
        flattened_polynomials: &[crate::poly::multilinear_polynomial::MultilinearPolynomial<F>],
        step_index: usize,
    ) -> (AzValue, BzValue, CzValue) {
        (
            self.a.evaluate_as_az(flattened_polynomials, step_index),
            self.b.evaluate_as_bz(flattened_polynomials, step_index),
            self.c.evaluate_as_cz(flattened_polynomials, step_index),
        )
    }

    #[cfg(test)]
    pub(crate) fn pretty_fmt<F: JoltField>(
        &self,
        f: &mut String,
        flattened_polynomials: &[crate::poly::multilinear_polynomial::MultilinearPolynomial<F>],
        step_index: usize,
    ) -> std::fmt::Result {
        use std::fmt::Write as _;

        use crate::zkvm::r1cs::inputs::JoltR1CSInputs;

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

#[derive(Clone, Debug)]
pub struct TypedConstraint {
    pub lc: Constraint,
    pub type_info: ConstraintType,
}

#[derive(Default)]
pub struct R1CSBuilder {
    pub(crate) constraints: Vec<TypedConstraint>,
}

impl R1CSBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn constrain_eq(
        &mut self,
        left: impl Into<LC>,
        right: impl Into<LC>,
        type_annotation: ConstraintType,
    ) {
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
        self.constraints.push(TypedConstraint {
            lc: constraint,
            type_info: type_annotation,
        });
    }

    pub fn constrain_eq_conditional(
        &mut self,
        condition: impl Into<LC>,
        left: impl Into<LC>,
        right: impl Into<LC>,
        type_annotation: ConstraintType,
    ) {
        // condition  * (left - right) == 0
        let condition: LC = condition.into();
        let left: LC = left.into();
        let right: LC = right.into();

        let a = condition;
        let b = left - right;
        let c = LC::zero();
        let constraint = Constraint { a, b, c }; // TODO(sragss): Can do better on middle term.
        self.constraints.push(TypedConstraint {
            lc: constraint,
            type_info: type_annotation,
        });
    }

    pub fn constrain_binary(&mut self, value: impl Into<LC>, type_annotation: ConstraintType) {
        let one: LC = Variable::Constant.into();
        let a: LC = value.into();
        let b = one - a.clone();
        // value * (1 - value) == 0
        let constraint = Constraint {
            a,
            b,
            c: LC::zero(),
        };
        self.constraints.push(TypedConstraint {
            lc: constraint,
            type_info: type_annotation,
        });
    }

    pub fn constrain_if_else(
        &mut self,
        condition: impl Into<LC>,
        result_true: impl Into<LC>,
        result_false: impl Into<LC>,
        alleged_result: impl Into<LC>,
        type_annotation: ConstraintType,
    ) {
        let condition: LC = condition.into();
        let result_true: LC = result_true.into();
        let result_false: LC = result_false.into();
        let alleged_result: LC = alleged_result.into();

        // result == condition * true_coutcome + (1 - condition) * false_outcome
        // simplify to single mul, single constraint => condition * (true_outcome - false_outcome) == (result - false_outcome)

        let constraint = Constraint {
            a: condition.clone(),
            b: (result_true - result_false.clone()),
            c: (alleged_result - result_false),
        };
        self.constraints.push(TypedConstraint {
            lc: constraint,
            type_info: type_annotation,
        });
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
        type_annotation: ConstraintType,
    ) {
        // Pack unpacked via a simple weighted linear combination
        // A + 2 * B + 4 * C + 8 * D, ...
        let packed: Vec<Term> = unpacked
            .into_iter()
            .enumerate()
            .map(|(idx, unpacked)| Term(unpacked, 1 << (idx * operand_bits)))
            .collect();
        self.constrain_eq(packed, result, type_annotation);
    }

    pub fn constrain_pack_be(
        &mut self,
        unpacked: Vec<Variable>,
        result: impl Into<LC>,
        operand_bits: usize,
        type_annotation: ConstraintType,
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
        self.constrain_eq(packed, result, type_annotation);
    }

    /// Constrain x * y == z
    pub fn constrain_prod(
        &mut self,
        x: impl Into<LC>,
        y: impl Into<LC>,
        z: impl Into<LC>,
        type_annotation: ConstraintType,
    ) {
        let constraint = Constraint {
            a: x.into(),
            b: y.into(),
            c: z.into(),
        };
        self.constraints.push(TypedConstraint {
            lc: constraint,
            type_info: type_annotation,
        });
    }

    fn materialize<F: JoltField>(&self) -> UniformR1CS<F> {
        let a_len: usize = self.constraints.iter().map(|c| c.lc.a.num_vars()).sum();
        let b_len: usize = self.constraints.iter().map(|c| c.lc.b.num_vars()).sum();
        let c_len: usize = self.constraints.iter().map(|c| c.lc.c.num_vars()).sum();
        let mut a_sparse = SparseConstraints::empty_with_capacity(a_len, self.constraints.len());
        let mut b_sparse = SparseConstraints::empty_with_capacity(b_len, self.constraints.len());
        let mut c_sparse = SparseConstraints::empty_with_capacity(c_len, self.constraints.len());

        let update_sparse = |row_index: usize, lc: &LC, sparse: &mut SparseConstraints<F>| {
            lc.terms().iter().for_each(|term| {
                match term.0 {
                    Variable::Input(inner) => {
                        sparse.vars.push((row_index, inner, F::from_i128(term.1)))
                    }
                    Variable::Constant => {}
                };
            });
            if let Some(term) = lc.constant_term() {
                sparse.consts.push((row_index, F::from_i128(term.1)));
            }
        };

        for (row_index, constraint) in self.constraints.iter().enumerate() {
            update_sparse(row_index, &constraint.lc.a, &mut a_sparse);
            update_sparse(row_index, &constraint.lc.b, &mut b_sparse);
            update_sparse(row_index, &constraint.lc.c, &mut c_sparse);
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

    pub fn get_constraints(&self) -> Vec<TypedConstraint> {
        self.constraints.clone()
    }
}

// TODO(sragss): Detailed documentation with wiki.
pub struct CombinedUniformBuilder<F: JoltField> {
    _field: PhantomData<F>,
    pub(crate) uniform_builder: R1CSBuilder,

    /// Padded to the nearest power of 2
    pub(crate) uniform_repeat: usize, // TODO(JP): Remove padding of steps
}

impl<F: JoltField> CombinedUniformBuilder<F> {
    pub fn construct(uniform_builder: R1CSBuilder, uniform_repeat: usize) -> Self {
        assert!(uniform_repeat.is_power_of_two());
        Self {
            _field: PhantomData,
            uniform_builder,
            uniform_repeat,
        }
    }

    /// Number of constraint rows per step, padded to the next power of two.
    pub(super) fn padded_rows_per_step(&self) -> usize {
        self.uniform_builder.constraints.len().next_power_of_two()
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
}
