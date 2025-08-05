use crate::tensor_jolt::execution_trace::{JoltONNXR1CSInputs, WitnessGenerator};
use crate::tensor_jolt::r1cs::key::UniformR1CS;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::r1cs::builder::Constraint;
use jolt_core::r1cs::ops::{LC, Term, Variable};
use jolt_core::{field::JoltField, r1cs::key::SparseConstraints};
use std::fmt::Write as _;
use std::marker::PhantomData;

pub trait R1CSConstraintFormatter {
    fn format_constraint<F: JoltField>(
        &self,
        f: &mut String,
        flattened_polynomials: &[MultilinearPolynomial<F>],
        step_index: usize,
    ) -> std::fmt::Result;
}

impl R1CSConstraintFormatter for Constraint {
    fn format_constraint<F: JoltField>(
        &self,
        f: &mut String,
        flattened_polynomials: &[MultilinearPolynomial<F>],
        step_index: usize,
    ) -> std::fmt::Result {
        use std::fmt::Write as _;

        self.a.format_lc(f)?;
        write!(f, " ⋅ ")?;
        self.b.format_lc(f)?;
        write!(f, " == ")?;
        self.c.format_lc(f)?;
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
                        JoltONNXR1CSInputs::from_index(var_index),
                        flattened_polynomials[var_index].get_coeff(step_index)
                    )?;
                }
                Variable::Constant => {}
            }
        }

        Ok(())
    }
}

pub trait FormatLC {
    fn format_lc(&self, f: &mut String) -> std::fmt::Result;
}

impl FormatLC for LC {
    fn format_lc(&self, f: &mut String) -> std::fmt::Result {
        if self.0.is_empty() {
            write!(f, "0")
        } else {
            if self.0.len() > 1 {
                write!(f, "(")?;
            }
            for (index, term) in self.0.iter().enumerate() {
                if term.1 == 0 {
                    continue;
                }
                if index > 0 {
                    if term.1 < 0 {
                        write!(f, " - ")?;
                    } else {
                        write!(f, " + ")?;
                    }
                }
                term.format_term(f)?;
            }
            if self.0.len() > 1 {
                write!(f, ")")?;
            }
            Ok(())
        }
    }
}

pub trait FormatTerm {
    fn format_term(&self, f: &mut String) -> std::fmt::Result;
}

impl FormatTerm for Term {
    fn format_term(&self, f: &mut String) -> std::fmt::Result {
        match self.0 {
            Variable::Input(var_index) => match self.1.abs() {
                1 => write!(f, "{:?}", JoltONNXR1CSInputs::from_index(var_index)),
                _ => write!(
                    f,
                    "{}⋅{:?}",
                    self.1,
                    JoltONNXR1CSInputs::from_index(var_index)
                ),
            },
            Variable::Constant => write!(f, "{}", self.1),
        }
    }
}

#[derive(Default)]
pub struct R1CSBuilder {
    pub constraints: Vec<Constraint>,
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
            b: (result_true - result_false.clone()),
            c: (alleged_result - result_false),
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
            num_vars: JoltONNXR1CSInputs::len(),
            num_rows: self.constraints.len(),
        }
    }

    pub fn get_constraints(&self) -> Vec<Constraint> {
        self.constraints.clone()
    }
}

// TODO(sragss): Detailed documentation with wiki.
pub struct CombinedUniformBuilder<F: JoltField> {
    _field: PhantomData<F>,
    pub uniform_builder: R1CSBuilder,

    /// Padded to the nearest power of 2
    pub uniform_repeat: usize, // TODO(JP): Remove padding of steps
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
    pub fn padded_rows_per_step(&self) -> usize {
        self.uniform_builder.constraints.len().next_power_of_two()
    }

    /// Total number of rows used across all repeated constraints. Not padded to nearest power of two.
    pub fn constraint_rows(&self) -> usize {
        self.uniform_repeat * self.padded_rows_per_step()
    }

    pub fn uniform_repeat(&self) -> usize {
        self.uniform_repeat
    }

    /// Materializes the uniform constraints into sparse (value != 0) A, B, C matrices represented in (row, col, value) format.
    pub fn materialize_uniform(&self) -> UniformR1CS<F> {
        self.uniform_builder.materialize()
    }
}
