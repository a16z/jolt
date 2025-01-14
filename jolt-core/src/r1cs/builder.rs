use super::{
    inputs::ConstraintInput,
    key::{NonUniformR1CS, NonUniformR1CSConstraint, SparseEqualityItem},
    ops::{Term, Variable, LC},
};
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::{
    field::JoltField,
    jolt::vm::JoltPolynomials,
    poly::spartan_interleaved_poly::SpartanInterleavedPolynomial,
    r1cs::key::{SparseConstraints, UniformR1CS},
};
use ark_ff::One;
use rayon::prelude::*;
use std::{collections::BTreeMap, marker::PhantomData};

/// Constraints over a single row. Each variable points to a single item in Z and the corresponding coefficient.
#[derive(Clone)]
pub struct Constraint {
    pub(crate) a: LC,
    pub(crate) b: LC,
    pub(crate) c: LC,
}

impl Constraint {
    #[cfg(test)]
    fn pretty_fmt<const C: usize, I: ConstraintInput, F: JoltField>(
        &self,
        f: &mut String,
        flattened_polynomials: &[&MultilinearPolynomial<F>],
        step_index: usize,
    ) -> std::fmt::Result {
        use std::fmt::Write as _;

        self.a.pretty_fmt::<C, I>(f)?;
        write!(f, " ⋅ ")?;
        self.b.pretty_fmt::<C, I>(f)?;
        write!(f, " == ")?;
        self.c.pretty_fmt::<C, I>(f)?;
        writeln!(f, "")?;

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
                Variable::Input(var_index) | Variable::Auxiliary(var_index) => {
                    writeln!(
                        f,
                        "    {:?} = {}",
                        I::from_index::<C>(var_index),
                        flattened_polynomials[var_index].get_coeff(step_index)
                    )?;
                }
                Variable::Constant => {}
            }
        }

        Ok(())
    }
}

type AuxComputationFunction = dyn Fn(&[i64]) -> i128 + Send + Sync;

struct AuxComputation<F: JoltField> {
    symbolic_inputs: Vec<LC>,
    compute: Box<AuxComputationFunction>,
    _field: PhantomData<F>,
}

impl<F: JoltField> AuxComputation<F> {
    fn new(
        _output: Variable,
        symbolic_inputs: Vec<LC>,
        compute: Box<AuxComputationFunction>,
    ) -> Self {
        #[cfg(test)]
        {
            let flat_vars: Vec<_> = symbolic_inputs
                .iter()
                .flat_map(|input| {
                    input.terms().iter().filter_map(|term| {
                        if let Variable::Constant = term.0 {
                            None
                        } else {
                            Some(term.0)
                        }
                    })
                })
                .collect();

            let output_index = match _output {
                Variable::Auxiliary(output_index) => output_index,
                _ => panic!("Output must be of the Variable::Aux variant"),
            };
            for var in &flat_vars {
                if let Variable::Auxiliary(aux_index) = var {
                    // Currently do not support aux computations dependent on those allocated after. Could support with dependency graph, instead
                    // dev should write their constraints sequentially. Simplifies aux computation parallelism.
                    if output_index <= *aux_index {
                        panic!("Aux computation depends on future aux computation: {_output:?} = f({var:?})");
                    }
                }
            }
        }

        Self {
            symbolic_inputs,
            compute,
            _field: PhantomData,
        }
    }

    fn compute_aux_poly<const C: usize, I: ConstraintInput>(
        &self,
        jolt_polynomials: &JoltPolynomials<F>,
        poly_len: usize,
    ) -> MultilinearPolynomial<F> {
        let flattened_polys: Vec<&MultilinearPolynomial<F>> = I::flatten::<C>()
            .iter()
            .map(|var| var.get_ref(jolt_polynomials))
            .collect();

        let mut aux_poly: Vec<u64> = vec![0; poly_len];
        let num_threads = rayon::current_num_threads();
        let chunk_size = poly_len.div_ceil(num_threads);

        aux_poly
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                chunk.iter_mut().enumerate().for_each(|(offset, result)| {
                    let global_index = chunk_index * chunk_size + offset;
                    let compute_inputs: Vec<_> = self
                        .symbolic_inputs
                        .iter()
                        .map(|lc| {
                            let mut input = 0;
                            for term in lc.terms().iter() {
                                match term.0 {
                                    Variable::Input(index) | Variable::Auxiliary(index) => {
                                        input += flattened_polys[index].get_coeff_i64(global_index)
                                            * term.1;
                                    }
                                    Variable::Constant => input += term.1,
                                }
                            }
                            input
                        })
                        .collect();
                    *result = u64::try_from((self.compute)(&compute_inputs)).unwrap();
                });
            });

        MultilinearPolynomial::from(aux_poly)
    }
}

pub struct R1CSBuilder<const C: usize, F: JoltField, I: ConstraintInput> {
    _inputs: PhantomData<I>,
    constraints: Vec<Constraint>,
    aux_computations: BTreeMap<usize, AuxComputation<F>>,
}

impl<const C: usize, F: JoltField, I: ConstraintInput> Default for R1CSBuilder<C, F, I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const C: usize, F: JoltField, I: ConstraintInput> R1CSBuilder<C, F, I> {
    pub fn new() -> Self {
        Self {
            _inputs: PhantomData,
            constraints: vec![],
            aux_computations: BTreeMap::new(),
        }
    }

    fn allocate_aux(
        &mut self,
        aux_symbol: I,
        symbolic_inputs: Vec<LC>,
        compute: Box<AuxComputationFunction>,
    ) -> Variable {
        let aux_index = aux_symbol.to_index::<C>();
        let new_aux = Variable::Auxiliary(aux_index);
        let computation = AuxComputation::new(new_aux, symbolic_inputs, compute);
        self.aux_computations.insert(aux_index, computation);

        new_aux
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

    #[must_use]
    pub fn allocate_if_else(
        &mut self,
        aux_symbol: I,
        condition: impl Into<LC>,
        result_true: impl Into<LC>,
        result_false: impl Into<LC>,
    ) -> Variable {
        let (condition, result_true, result_false) =
            (condition.into(), result_true.into(), result_false.into());

        let aux_var = self.aux_if_else(aux_symbol, &condition, &result_true, &result_false);

        self.constrain_if_else(condition, result_true, result_false, aux_var);
        aux_var
    }

    fn aux_if_else(
        &mut self,
        aux_symbol: I,
        condition: &LC,
        result_true: &LC,
        result_false: &LC,
    ) -> Variable {
        // aux = (condition == 1) ? result_true : result_false;
        let if_else = |values: &[i64]| -> i128 {
            assert_eq!(values.len(), 3);
            let condition = values[0];
            let result_true = values[1];
            let result_false = values[2];

            if condition.is_one() {
                result_true as i128
            } else {
                result_false as i128
            }
        };

        let symbolic_inputs = vec![condition.clone(), result_true.clone(), result_false.clone()];
        let compute = Box::new(if_else);
        self.allocate_aux(aux_symbol, symbolic_inputs, compute)
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

    #[must_use]
    pub fn allocate_prod(&mut self, aux_symbol: I, x: impl Into<LC>, y: impl Into<LC>) -> Variable {
        let (x, y) = (x.into(), y.into());
        let z = self.aux_prod(aux_symbol, &x, &y);

        self.constrain_prod(x, y, z);
        z
    }

    fn aux_prod(&mut self, aux_symbol: I, x: &LC, y: &LC) -> Variable {
        let prod = |values: &[i64]| {
            assert_eq!(values.len(), 2);
            (values[0] as i128) * (values[1] as i128)
        };

        let symbolic_inputs = vec![x.clone(), y.clone()];
        let compute = Box::new(prod);
        self.allocate_aux(aux_symbol, symbolic_inputs, compute)
    }

    fn materialize(&self) -> UniformR1CS<F> {
        let a_len: usize = self.constraints.iter().map(|c| c.a.num_vars()).sum();
        let b_len: usize = self.constraints.iter().map(|c| c.b.num_vars()).sum();
        let c_len: usize = self.constraints.iter().map(|c| c.c.num_vars()).sum();
        let mut a_sparse = SparseConstraints::empty_with_capacity(a_len, self.constraints.len());
        let mut b_sparse = SparseConstraints::empty_with_capacity(b_len, self.constraints.len());
        let mut c_sparse = SparseConstraints::empty_with_capacity(c_len, self.constraints.len());

        let update_sparse = |row_index: usize, lc: &LC, sparse: &mut SparseConstraints<F>| {
            lc.terms().iter().for_each(|term| {
                match term.0 {
                    Variable::Input(inner) | Variable::Auxiliary(inner) => {
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
            num_vars: I::num_inputs::<C>(),
            num_rows: self.constraints.len(),
        }
    }
}

/// An Offset Linear Combination. If OffsetLC.0 is true, then the OffsetLC.1 refers to the next step in a uniform
/// constraint system.
pub type OffsetLC = (bool, LC);

/// A conditional constraint that Linear Combinations a, b are equal where a and b need not be in the same step an a
/// uniform constraint system.
#[derive(Debug)]
pub struct OffsetEqConstraint {
    pub(crate) cond: OffsetLC,
    pub(crate) a: OffsetLC,
    pub(crate) b: OffsetLC,
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
    flattened_polynomials: &[&MultilinearPolynomial<F>],
    step: usize,
    next_step_m: Option<usize>,
) -> i128 {
    if !offset.0 {
        offset.1.evaluate_row_i128(flattened_polynomials, step)
    } else if let Some(next_step) = next_step_m {
        offset.1.evaluate_row_i128(flattened_polynomials, next_step)
    } else {
        offset.1.constant_term_field()
    }
}

// TODO(sragss): Detailed documentation with wiki.
pub struct CombinedUniformBuilder<const C: usize, F: JoltField, I: ConstraintInput> {
    uniform_builder: R1CSBuilder<C, F, I>,

    /// Padded to the nearest power of 2
    uniform_repeat: usize, // TODO(JP): Remove padding of steps

    offset_equality_constraints: Vec<OffsetEqConstraint>,
}

impl<const C: usize, F: JoltField, I: ConstraintInput> CombinedUniformBuilder<C, F, I> {
    pub fn construct(
        uniform_builder: R1CSBuilder<C, F, I>,
        uniform_repeat: usize,
        offset_equality_constraints: Vec<OffsetEqConstraint>,
    ) -> Self {
        assert!(uniform_repeat.is_power_of_two());
        Self {
            uniform_builder,
            uniform_repeat,
            offset_equality_constraints,
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn compute_aux(&self, jolt_polynomials: &mut JoltPolynomials<F>) {
        let flattened_vars = I::flatten::<C>();
        for (aux_index, aux_compute) in self.uniform_builder.aux_computations.iter() {
            *flattened_vars[*aux_index].get_ref_mut(jolt_polynomials) =
                aux_compute.compute_aux_poly::<C, I>(jolt_polynomials, self.uniform_repeat);
        }
    }

    /// Total number of rows used across all uniform constraints across all repeats. Repeat padded to 2, but repeat * num_constraints not, num_constraints not.
    pub(super) fn uniform_repeat_constraint_rows(&self) -> usize {
        self.uniform_repeat * self.uniform_builder.constraints.len()
    }

    // TODO: #[not(cfg(feature = "reorder"))]
    pub(super) fn offset_eq_constraint_rows(&self) -> usize {
        self.uniform_repeat * self.offset_equality_constraints.len()
    }

    /// Number of constraint rows per step, padded to the next power of two.
    // #[cfg(feature = "reorder")]
    pub(super) fn padded_rows_per_step(&self) -> usize {
        let num_constraints =
            self.uniform_builder.constraints.len() + self.offset_equality_constraints.len();
        num_constraints.next_power_of_two()
    }

    /// Total number of rows used across all repeated constraints. Not padded to nearest power of two.
    pub(super) fn constraint_rows(&self) -> usize {
        if cfg!(feature = "reorder") {
            return self.uniform_repeat * self.padded_rows_per_step();
        }

        self.offset_eq_constraint_rows() + self.uniform_repeat_constraint_rows()
    }

    pub(super) fn uniform_repeat(&self) -> usize {
        self.uniform_repeat
    }

    /// Materializes the uniform constraints into sparse (value != 0) A, B, C matrices represented in (row, col, value) format.
    pub fn materialize_uniform(&self) -> UniformR1CS<F> {
        self.uniform_builder.materialize()
    }

    /// Converts builder::OffsetEqConstraints into key::NonUniformR1CSConstraint
    pub fn materialize_offset_eq(&self) -> NonUniformR1CS<F> {
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
                    Variable::Input(inner) | Variable::Auxiliary(inner) => condition
                        .offset_vars
                        .push((inner, constraint.cond.0, F::from_i64(term.1))),
                    Variable::Constant => {}
                });
            if let Some(term) = constraint.cond.1.constant_term() {
                condition.constant = F::from_i64(term.1);
            }

            // Can't simply combine like terms because of the offset
            let lhs = constraint.a.1.clone();
            let rhs = -constraint.b.1.clone();

            lhs.terms().iter().for_each(|term| match term.0 {
                Variable::Input(inner) | Variable::Auxiliary(inner) => {
                    eq.offset_vars
                        .push((inner, constraint.a.0, F::from_i64(term.1)))
                }
                Variable::Constant => {}
            });
            rhs.terms().iter().for_each(|term| match term.0 {
                Variable::Input(inner) | Variable::Auxiliary(inner) => {
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

            constraints.push(NonUniformR1CSConstraint::new(eq, condition));
        }

        NonUniformR1CS { constraints }
    }

    #[tracing::instrument(skip_all)]
    fn compute_spartan_Xz<
        U: for<'a> Fn(&'a Constraint) -> &'a LC + Send + Sync + Copy + 'static,
        O: Fn(&[&MultilinearPolynomial<F>], &OffsetEqConstraint, usize, Option<usize>) -> F
            + Send
            + Sync
            + Copy
            + 'static,
    >(
        &self,
        flattened_polynomials: &[&MultilinearPolynomial<F>], // N variables of (S steps)
        uniform_constraint: U,
        offset_constraint: O,
    ) -> Vec<(F, usize)> {
        let num_steps = flattened_polynomials[0].len();
        let padded_num_constraints = self.padded_rows_per_step();

        // Filter out constraints that won't contribute ahead of time.
        let filtered_uniform_constraints = self
            .uniform_builder
            .constraints
            .iter()
            .enumerate()
            .filter_map(|(index, constraint)| {
                let lc = uniform_constraint(constraint);
                if lc.terms().is_empty() {
                    None
                } else {
                    Some((index, lc))
                }
            })
            .collect::<Vec<_>>();

        (0..num_steps)
            .into_par_iter()
            .flat_map_iter(|step_index| {
                // uniform_constraints
                let uniform_constraints = filtered_uniform_constraints.par_iter().flat_map(
                    move |(constraint_index, lc)| {
                        // Evaluate a constraint on a given step.
                        let item = lc.evaluate_row(flattened_polynomials, step_index);
                        if !item.is_zero() {
                            let global_index =
                                step_index * padded_num_constraints + constraint_index;
                            Some((item, global_index))
                        } else {
                            None
                        }
                    },
                );

                let next_step_index_m = if step_index + 1 < num_steps {
                    Some(step_index + 1)
                } else {
                    None
                };

                // offset_equality_constraints
                // (a - b) * condition == 0
                // For the final step we will not compute the offset terms, and will assume the condition to be set to 0
                let non_uniform_constraints = self
                    .offset_equality_constraints
                    .par_iter()
                    .enumerate()
                    .flat_map(move |(constr_i, constr)| {
                        let xz = offset_constraint(
                            flattened_polynomials,
                            constr,
                            step_index,
                            next_step_index_m,
                        );
                        if !xz.is_zero() {
                            let global_index = step_index * padded_num_constraints
                                + self.uniform_builder.constraints.len()
                                + constr_i;
                            Some((xz, global_index))
                        } else {
                            None
                        }
                    });

                uniform_constraints
                    .chain(non_uniform_constraints)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[tracing::instrument(skip_all)]
    pub fn compute_spartan_Az_Bz_Cz(
        &self,
        flattened_polynomials: &[&MultilinearPolynomial<F>], // N variables of (S steps)
    ) -> SpartanInterleavedPolynomial<F> {
        SpartanInterleavedPolynomial::new(
            &self.uniform_builder.constraints,
            &self.offset_equality_constraints,
            flattened_polynomials,
            self.padded_rows_per_step(),
        )

        // let dense_len = self.padded_rows_per_step() * self.uniform_repeat();
        // let az_sparse = self.compute_spartan_Xz(
        //     flattened_polynomials,
        //     |constraint: &Constraint| &constraint.a,
        //     |flattened_polynomials, constr, step_index, next_step_index_m| {
        //         let eq_a_eval = eval_offset_lc(
        //             &constr.a,
        //             flattened_polynomials,
        //             step_index,
        //             next_step_index_m,
        //         );
        //         let eq_b_eval = eval_offset_lc(
        //             &constr.b,
        //             flattened_polynomials,
        //             step_index,
        //             next_step_index_m,
        //         );
        //         let az = eq_a_eval - eq_b_eval;
        //         F::from_i128(az)
        //     },
        // );
        // println!(
        //     "Az is {}% sparse",
        //     100.0 * (dense_len - az_sparse.len()) as f64 / dense_len as f64
        // );

        // let bz_sparse = self.compute_spartan_Xz(
        //     flattened_polynomials,
        //     |constraint: &Constraint| &constraint.b,
        //     |flattened_polynomials, constr, step_index, next_step_index_m| {
        //         let condition_eval = eval_offset_lc(
        //             &constr.cond,
        //             flattened_polynomials,
        //             step_index,
        //             next_step_index_m,
        //         );
        //         let bz = condition_eval;
        //         F::from_i128(bz)
        //     },
        // );
        // println!(
        //     "Bz is {}% sparse",
        //     100.0 * (dense_len - bz_sparse.len()) as f64 / dense_len as f64
        // );

        // let cz_sparse = self.compute_spartan_Xz(
        //     flattened_polynomials,
        //     |constraint: &Constraint| &constraint.c,
        //     |_, _, _, _| F::zero(),
        // );
        // println!(
        //     "Cz is {}% sparse",
        //     100.0 * (dense_len - cz_sparse.len()) as f64 / dense_len as f64
        // );

        // let num_vars = self.constraint_rows().next_power_of_two().log_2();
        // let az_poly = SparsePolynomial::new(num_vars, az_sparse);
        // let bz_poly = SparsePolynomial::new(num_vars, bz_sparse);
        // let cz_poly = SparsePolynomial::new(num_vars, cz_sparse);

        // #[cfg(test)]
        // self.assert_valid(flattened_polynomials, &az_poly, &bz_poly, &cz_poly);

        // (az_poly, bz_poly, cz_poly)
    }

    #[cfg(test)]
    pub fn assert_valid(
        &self,
        flattened_polynomials: &[&MultilinearPolynomial<F>],
        az: &SparsePolynomial<F>,
        bz: &SparsePolynomial<F>,
        cz: &SparsePolynomial<F>,
    ) {
        let az = az.clone().to_dense();
        let bz = bz.clone().to_dense();
        let cz = cz.clone().to_dense();

        let rows = az.len();
        assert_eq!(bz.len(), rows);
        assert_eq!(cz.len(), rows);

        for constraint_index in 0..rows {
            let uniform_constraint_index = constraint_index / self.uniform_repeat;
            if az[constraint_index] * bz[constraint_index] != cz[constraint_index] {
                let step_index = constraint_index % self.uniform_repeat;
                if uniform_constraint_index >= self.uniform_builder.constraints.len() {
                    panic!(
                        "Non-uniform constraint {} violated at step {step_index}",
                        uniform_constraint_index - self.uniform_builder.constraints.len()
                    )
                } else {
                    let mut constraint_string = String::new();
                    let _ = self.uniform_builder.constraints[uniform_constraint_index]
                        .pretty_fmt::<C, I, F>(
                            &mut constraint_string,
                            flattened_polynomials,
                            step_index,
                        );
                    println!("{constraint_string}");
                    panic!(
                        "Uniform constraint {uniform_constraint_index} violated at step {step_index}",
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    // use ark_bn254::Fr;

    // fn aux_compute_single<F: JoltField>(
    //     aux_compute: &AuxComputation<F>,
    //     single_step_inputs: &[F],
    // ) -> F {
    //     let multi_step_inputs: Vec<Vec<F>> = single_step_inputs
    //         .iter()
    //         .map(|input| vec![*input])
    //         .collect();
    //     let multi_step_inputs_ref: Vec<&[F]> =
    //         multi_step_inputs.iter().map(|v| v.as_slice()).collect();
    //     aux_compute.compute_batch(multi_step_inputs_ref, 1)[0]
    // }

    // #[test]
    // fn aux_compute_simple() {
    //     let a: LC<TestInputs> = 12i64.into();
    //     let b: LC<TestInputs> = 20i64.into();
    //     let lc = vec![a + b];
    //     let lambda = |input: &[Fr]| {
    //         assert_eq!(input.len(), 1);
    //         input[0]
    //     };
    //     let aux =
    //         AuxComputation::<Fr, TestInputs>::new(Variable::Auxiliary(0), lc, Box::new(lambda));
    //     let result = aux_compute_single(&aux, &[Fr::from(32)]);
    //     assert_eq!(result, Fr::from(32));
    // }

    // #[test]
    // #[should_panic]
    // fn aux_compute_depends_on_aux() {
    //     let a: LC<TestInputs> = 12i64.into();
    //     let b: LC<TestInputs> = Variable::Auxiliary(1).into();
    //     let lc = vec![a + b];
    //     let lambda = |_input: &[Fr]| unimplemented!();
    //     let _aux =
    //         AuxComputation::<Fr, TestInputs>::new(Variable::Auxiliary(0), lc, Box::new(lambda));
    // }

    // #[test]
    // fn eq_builder() {
    //     let mut builder = R1CSBuilder::<Fr, TestInputs>::new();

    //     // PcIn + PcOut == BytecodeA + 2 BytecodeVOpcode
    //     struct TestConstraints();
    //     impl<const C: usize, F: JoltField> R1CSConstraints<C, F> for TestConstraints {
    //         type Inputs = TestInputs;
    //         fn uniform_constraints(
    //             builder: &mut R1CSBuilder<C, F, Self::Inputs>,
    //             memory_start: u64,
    //         ) {
    //             let left = Self::Inputs::PcIn + Self::Inputs::PcOut;
    //             let right = Self::Inputs::BytecodeA + 2i64 * Self::Inputs::BytecodeVOpcode;
    //             builder.constrain_eq(left, right);
    //         }

    //         fn non_uniform_constraints() -> Vec<OffsetEqConstraint> {
    //             vec![]
    //         }
    //     }

    //     let concrete_constraints = TestConstraints();
    //     concrete_constraints.build_constraints(&mut builder);
    //     assert!(builder.constraints.len() == 1);
    //     let constraint = &builder.constraints[0];
    //     let mut z = vec![0i64; TestInputs::COUNT];

    //     // 2 + 6 == 6 + 2*1
    //     z[TestInputs::PcIn as usize] = 2;
    //     z[TestInputs::PcOut as usize] = 6;
    //     z[TestInputs::BytecodeA as usize] = 6;
    //     z[TestInputs::BytecodeVOpcode as usize] = 1;
    //     assert!(constraint.is_sat(&z));

    //     // 2 + 6 != 6 + 2*2
    //     z[TestInputs::BytecodeVOpcode as usize] = 2;
    //     assert!(!constraint.is_sat(&z));
    // }

    // #[test]
    // fn if_else_builder() {
    //     let mut builder = R1CSBuilder::<Fr, TestInputs>::new();

    //     // condition * (true_outcome - false_outcome) = (result - false_outcome)
    //     // PcIn * (BytecodeVRS1 - BytecodeVRS2) == BytecodeA - BytecodeVRS2
    //     // If PcIn == 1: BytecodeA = BytecodeVRS1
    //     // If PcIn == 0: BytecodeA = BytecodeVRS2
    //     struct TestConstraints();
    //     impl<const C: usize, F: JoltField> R1CSConstraints<C, F> for TestConstraints {
    //         type Inputs = TestInputs;
    //         fn uniform_constraints(
    //             builder: &mut R1CSBuilder<C, F, Self::Inputs>,
    //             memory_start: u64,
    //         ) {
    //             let condition = Self::Inputs::PcIn;
    //             let true_outcome = Self::Inputs::BytecodeVRS1;
    //             let false_outcome = Self::Inputs::BytecodeVRS2;
    //             let alleged_result = Self::Inputs::BytecodeA;
    //             builder.constrain_if_else(condition, true_outcome, false_outcome, alleged_result);
    //         }
    //         fn non_uniform_constraints() -> Vec<OffsetEqConstraint> {
    //             vec![]
    //         }
    //     }

    //     let concrete_constraints = TestConstraints();
    //     concrete_constraints.build_constraints(&mut builder);
    //     assert!(builder.constraints.len() == 1);
    //     let constraint = &builder.constraints[0];

    //     let mut z = vec![0i64; TestInputs::COUNT];
    //     z[TestInputs::PcIn as usize] = 1;
    //     z[TestInputs::BytecodeA as usize] = 6;
    //     z[TestInputs::BytecodeVRS1 as usize] = 6;
    //     z[TestInputs::BytecodeVRS2 as usize] = 10;
    //     assert!(constraint.is_sat(&z));
    //     z[TestInputs::PcIn as usize] = 0;
    //     assert!(!constraint.is_sat(&z));
    //     z[TestInputs::BytecodeA as usize] = 10;
    //     assert!(constraint.is_sat(&z));
    // }

    // #[test]
    // fn alloc_if_else_builder() {
    //     let mut builder = R1CSBuilder::<Fr, TestInputs>::new();

    //     // condition * (true_outcome - false_outcome) = (result - false_outcome)
    //     // PcIn * (BytecodeVRS1 - BytecodeVRS2) == AUX_RESULT - BytecodeVRS2
    //     // If PcIn == 1: AUX_RESULT = BytecodeVRS1
    //     // If PcIn == 0: AUX_RESULT = BytecodeVRS2
    //     // AUX_RESULT == BytecodeVImm
    //     struct TestConstraints();
    //     impl<const C: usize, F: JoltField> R1CSConstraints<C, F> for TestConstraints {
    //         type Inputs = TestInputs;
    //         fn uniform_constraints(
    //             builder: &mut R1CSBuilder<C, F, Self::Inputs>,
    //             memory_start: u64,
    //         ) {
    //             let condition = Self::Inputs::PcIn + Self::Inputs::PcOut;
    //             let true_outcome = Self::Inputs::BytecodeVRS1;
    //             let false_outcome = Self::Inputs::BytecodeVRS2;
    //             let branch_result =
    //                 builder.allocate_if_else(condition, true_outcome, false_outcome);
    //             builder.constrain_eq(branch_result, Self::Inputs::BytecodeVImm);
    //         }
    //         fn non_uniform_constraints() -> Vec<OffsetEqConstraint> {
    //             vec![]
    //         }
    //     }

    //     let concrete_constraints = TestConstraints();
    //     concrete_constraints.build_constraints(&mut builder);
    //     assert_eq!(builder.constraints.len(), 2);
    //     let (branch_constraint, eq_constraint) = (&builder.constraints[0], &builder.constraints[1]);

    //     let mut z = vec![0i64; TestInputs::COUNT + 1]; // 1 aux
    //     let true_branch_result: i64 = 12;
    //     let false_branch_result: i64 = 10;
    //     let aux_index = builder.witness_index(Variable::Auxiliary(0));
    //     z[TestInputs::PcIn as usize] = 1;
    //     z[TestInputs::BytecodeVRS1 as usize] = true_branch_result;
    //     z[TestInputs::BytecodeVRS2 as usize] = false_branch_result;
    //     z[TestInputs::BytecodeVImm as usize] = true_branch_result;
    //     z[aux_index] = true_branch_result;
    //     assert!(branch_constraint.is_sat(&z));
    //     assert!(eq_constraint.is_sat(&z));

    //     z[aux_index] = false_branch_result;
    //     assert!(!branch_constraint.is_sat(&z));
    //     assert!(!eq_constraint.is_sat(&z));

    //     z[TestInputs::BytecodeVImm as usize] = false_branch_result;
    //     assert!(!branch_constraint.is_sat(&z));
    //     assert!(eq_constraint.is_sat(&z));

    //     z[TestInputs::PcIn as usize] = 0;
    //     assert!(branch_constraint.is_sat(&z));
    //     assert!(eq_constraint.is_sat(&z));

    //     assert_eq!(builder.aux_computations.len(), 1);
    //     let compute_2 = aux_compute_single(
    //         &builder.aux_computations[0],
    //         &[Fr::one(), Fr::from(2), Fr::from(3)],
    //     );
    //     assert_eq!(compute_2, Fr::from(2));
    //     let compute_2 = aux_compute_single(
    //         &builder.aux_computations[0],
    //         &[Fr::zero(), Fr::from(2), Fr::from(3)],
    //     );
    //     assert_eq!(compute_2, Fr::from(3));
    // }

    // #[test]
    // fn packing_le_builder() {
    //     let mut builder = R1CSBuilder::<Fr, TestInputs>::new();

    //     // pack_le(OpFlags0, OpFlags1, OpFlags2, OpFlags3) == BytecodeA
    //     struct TestConstraints();
    //     impl<const C: usize, F: JoltField> R1CSConstraints<C, F> for TestConstraints {
    //         type Inputs = TestInputs;
    //         fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
    //             let result = Variable::Input(TestInputs::BytecodeA);
    //             let unpacked: Vec<Variable<TestInputs>> = vec![
    //                 TestInputs::OpFlags0.into(),
    //                 TestInputs::OpFlags1.into(),
    //                 TestInputs::OpFlags2.into(),
    //                 TestInputs::OpFlags3.into(),
    //             ];
    //             builder.constrain_pack_le(unpacked, result, 1);
    //         }
    //     }

    //     let concrete_constraints = TestConstraints();
    //     concrete_constraints.build_constraints(&mut builder);
    //     assert_eq!(builder.constraints.len(), 1);
    //     let constraint = &builder.constraints[0];

    //     // 1101 == 13
    //     let mut z = vec![0i64; TestInputs::COUNT];
    //     // (little endian)
    //     z[TestInputs::OpFlags0 as usize] = 1;
    //     z[TestInputs::OpFlags1 as usize] = 0;
    //     z[TestInputs::OpFlags2 as usize] = 1;
    //     z[TestInputs::OpFlags3 as usize] = 1;
    //     z[TestInputs::BytecodeA as usize] = 13;

    //     assert!(constraint.is_sat(&z));
    // }

    // #[test]
    // fn packing_be_builder() {
    //     let mut builder = R1CSBuilder::<Fr, TestInputs>::new();

    //     // pack_be(OpFlags0, OpFlags1, OpFlags2, OpFlags3) == BytecodeA
    //     struct TestConstraints();
    //     impl<const C: usize, F: JoltField> R1CSConstraints<C, F> for TestConstraints {
    //         type Inputs = TestInputs;
    //         fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
    //             let result = Variable::Input(TestInputs::BytecodeA);
    //             let unpacked: Vec<Variable<TestInputs>> = vec![
    //                 TestInputs::OpFlags0.into(),
    //                 TestInputs::OpFlags1.into(),
    //                 TestInputs::OpFlags2.into(),
    //                 TestInputs::OpFlags3.into(),
    //             ];
    //             builder.constrain_pack_be(unpacked, result, 1);
    //         }
    //     }

    //     let concrete_constraints = TestConstraints();
    //     concrete_constraints.build_constraints(&mut builder);
    //     assert_eq!(builder.constraints.len(), 1);
    //     let constraint = &builder.constraints[0];

    //     // 1101 == 13
    //     let mut z = vec![0i64; TestInputs::COUNT];
    //     // (big endian)
    //     z[TestInputs::OpFlags0 as usize] = 1;
    //     z[TestInputs::OpFlags1 as usize] = 1;
    //     z[TestInputs::OpFlags2 as usize] = 0;
    //     z[TestInputs::OpFlags3 as usize] = 1;
    //     z[TestInputs::BytecodeA as usize] = 13;

    //     assert!(constraint.is_sat(&z));
    // }

    // #[test]
    // fn prod() {
    //     let mut builder = R1CSBuilder::<Fr, TestInputs>::new();

    //     // OpFlags0 * OpFlags1 == BytecodeA
    //     // OpFlags2 * OpFlags3 == Aux
    //     struct TestConstraints();
    //     impl<const C: usize, F: JoltField> R1CSConstraints<C, F> for TestConstraints {
    //         type Inputs = TestInputs;
    //         fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
    //             builder.constrain_prod(
    //                 TestInputs::OpFlags0,
    //                 TestInputs::OpFlags1,
    //                 TestInputs::BytecodeA,
    //             );
    //             let _aux = builder.allocate_prod(TestInputs::OpFlags2, TestInputs::OpFlags3);
    //         }
    //     }

    //     let concrete_constraints = TestConstraints();
    //     concrete_constraints.build_constraints(&mut builder);
    //     assert_eq!(builder.constraints.len(), 2);
    //     assert_eq!(builder.next_aux, 1);

    //     let mut z = vec![0i64; TestInputs::COUNT];
    //     // x * y == z
    //     z[TestInputs::OpFlags0 as usize] = 7;
    //     z[TestInputs::OpFlags1 as usize] = 10;
    //     z[TestInputs::BytecodeA as usize] = 70;
    //     assert!(builder.constraints[0].is_sat(&z));
    //     z[TestInputs::BytecodeA as usize] = 71;
    //     assert!(!builder.constraints[0].is_sat(&z));

    //     // x * y == aux
    //     z[TestInputs::OpFlags2 as usize] = 5;
    //     z[TestInputs::OpFlags3 as usize] = 7;
    //     z.push(35);
    //     assert!(builder.constraints[1].is_sat(&z));
    //     z[builder.witness_index(Variable::Auxiliary(0))] = 36;
    //     assert!(!builder.constraints[1].is_sat(&z));
    // }

    // #[test]
    // fn alloc_prod() {
    //     let mut builder = R1CSBuilder::<Fr, TestInputs>::new();

    //     // OpFlags0 * OpFlags1 == Aux(0)
    //     struct TestConstraints();
    //     impl<const C: usize, F: JoltField> R1CSConstraints<C, F> for TestConstraints {
    //         type Inputs = TestInputs;
    //         fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
    //             let _aux = builder.allocate_prod(TestInputs::OpFlags0, TestInputs::OpFlags1);
    //         }
    //     }

    //     let constraints = TestConstraints();
    //     constraints.build_constraints(&mut builder);
    //     assert_eq!(builder.constraints.len(), 1);
    //     assert_eq!(builder.next_aux, 1);

    //     let mut z = vec![0i64; TestInputs::COUNT + 1];
    //     z[builder.witness_index(TestInputs::OpFlags0)] = 7;
    //     z[builder.witness_index(TestInputs::OpFlags1)] = 5;
    //     z[builder.witness_index(Variable::Auxiliary(0))] = 35;

    //     assert!(builder.constraints[0].is_sat(&z));
    //     z[builder.witness_index(Variable::Auxiliary(0))] = 36;
    //     assert!(!builder.constraints[0].is_sat(&z));
    // }

    // #[test]
    // fn alloc_compute_simple_uniform_only() {
    //     let mut uniform_builder = R1CSBuilder::<Fr, TestInputs>::new();

    //     // OpFlags0 * OpFlags1 == Aux(0)
    //     struct TestConstraints();
    //     impl<const C: usize, F: JoltField> R1CSConstraints<C, F> for TestConstraints {
    //         type Inputs = TestInputs;
    //         fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
    //             let _aux = builder.allocate_prod(TestInputs::OpFlags0, TestInputs::OpFlags1);
    //         }
    //     }

    //     let constraints = TestConstraints();
    //     constraints.build_constraints(&mut uniform_builder);
    //     assert_eq!(uniform_builder.constraints.len(), 1);
    //     assert_eq!(uniform_builder.next_aux, 1);
    //     let num_steps = 2;
    //     let combined_builder =
    //         CombinedUniformBuilder::construct(uniform_builder, num_steps, vec![]);

    //     let mut inputs = vec![vec![Fr::zero(); num_steps]; TestInputs::COUNT];
    //     inputs[TestInputs::OpFlags0 as usize][0] = Fr::from(5);
    //     inputs[TestInputs::OpFlags1 as usize][0] = Fr::from(7);
    //     inputs[TestInputs::OpFlags0 as usize][1] = Fr::from(11);
    //     inputs[TestInputs::OpFlags1 as usize][1] = Fr::from(13);
    //     let aux = combined_builder.compute_aux(&inputs);
    //     assert_eq!(aux, vec![vec![Fr::from(5 * 7), Fr::from(11 * 13)]]);

    //     let (az, bz, cz) = combined_builder.compute_spartan_Az_Bz_Cz(&inputs, &aux);
    //     combined_builder.assert_valid(&az, &bz, &cz);
    // }

    // #[test]
    // fn alloc_compute_complex_uniform_only() {
    //     let mut uniform_builder = R1CSBuilder::<Fr, TestInputs>::new();

    //     // OpFlags0 * OpFlags1 == Aux(0)
    //     // OpFlags2 + OpFlags3 == Aux(0)
    //     // (4 * RAMByte0 + 2) * OpFlags0 == Aux(1)
    //     // Aux(1) == RAMByte1
    //     struct TestConstraints();
    //     impl<const C: usize, F: JoltField> R1CSConstraints<C, F> for TestConstraints {
    //         type Inputs = TestInputs;
    //         fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
    //             let aux_0 = builder.allocate_prod(TestInputs::OpFlags0, TestInputs::OpFlags1);
    //             builder.constrain_eq(TestInputs::OpFlags2 + TestInputs::OpFlags3, aux_0);
    //             let aux_1 =
    //                 builder.allocate_prod(4 * TestInputs::RAMByte0 + 2i64, TestInputs::OpFlags0);
    //             builder.constrain_eq(aux_1, TestInputs::RAMByte1);
    //         }
    //     }

    //     let constraints = TestConstraints();
    //     constraints.build_constraints(&mut uniform_builder);
    //     assert_eq!(uniform_builder.constraints.len(), 4);
    //     assert_eq!(uniform_builder.next_aux, 2);

    //     let num_steps = 2;
    //     let combined_builder =
    //         CombinedUniformBuilder::construct(uniform_builder, num_steps, vec![]);

    //     let mut inputs = vec![vec![Fr::zero(); num_steps]; TestInputs::COUNT];
    //     inputs[TestInputs::OpFlags0 as usize][0] = Fr::from(5);
    //     inputs[TestInputs::OpFlags1 as usize][0] = Fr::from(7);
    //     inputs[TestInputs::OpFlags2 as usize][0] = Fr::from(30);
    //     inputs[TestInputs::OpFlags3 as usize][0] = Fr::from(5);
    //     inputs[TestInputs::RAMByte0 as usize][0] = Fr::from(10);
    //     inputs[TestInputs::RAMByte1 as usize][0] = Fr::from((4 * 10 + 2) * 5);

    //     inputs[TestInputs::OpFlags0 as usize][1] = Fr::from(7);
    //     inputs[TestInputs::OpFlags1 as usize][1] = Fr::from(7);
    //     inputs[TestInputs::OpFlags2 as usize][1] = Fr::from(40);
    //     inputs[TestInputs::OpFlags3 as usize][1] = Fr::from(9);
    //     inputs[TestInputs::RAMByte0 as usize][1] = Fr::from(10);
    //     inputs[TestInputs::RAMByte1 as usize][1] = Fr::from((4 * 10 + 2) * 7);

    //     let aux = combined_builder.compute_aux(&inputs);
    //     assert_eq!(
    //         aux,
    //         vec![
    //             vec![Fr::from(35), Fr::from(49)],
    //             vec![Fr::from((4 * 10 + 2) * 5), Fr::from((4 * 10 + 2) * 7)]
    //         ]
    //     );

    //     let (az, bz, cz) = combined_builder.compute_spartan_Az_Bz_Cz(&inputs, &aux);
    //     combined_builder.assert_valid(&az, &bz, &cz);
    // }

    // #[test]
    // fn alloc_compute_simple_combined() {
    //     let mut uniform_builder = R1CSBuilder::<Fr, TestInputs>::new();

    //     // OpFlags0 * OpFlags1 == Aux(0)
    //     struct TestConstraints();
    //     impl<const C: usize, F: JoltField> R1CSConstraints<C, F> for TestConstraints {
    //         type Inputs = TestInputs;
    //         fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
    //             let _aux = builder.allocate_prod(TestInputs::OpFlags0, TestInputs::OpFlags1);
    //         }
    //     }

    //     let constraints = TestConstraints();
    //     constraints.build_constraints(&mut uniform_builder);
    //     assert_eq!(uniform_builder.constraints.len(), 1);
    //     assert_eq!(uniform_builder.next_aux, 1);

    //     let num_steps = 2;

    //     // OpFlags0[n] = OpFlags0[n + 1];
    //     // PcIn[n] + 4 = PcIn[n + 1]
    //     let non_uniform_constraint: OffsetEqConstraint<TestInputs> = OffsetEqConstraint::new(
    //         (TestInputs::OpFlags0, true),
    //         (TestInputs::OpFlags0, false),
    //         (TestInputs::OpFlags0, true),
    //     );
    //     let combined_builder = CombinedUniformBuilder::construct(
    //         uniform_builder,
    //         num_steps,
    //         vec![non_uniform_constraint],
    //     );

    //     let mut inputs = vec![vec![Fr::zero(); num_steps]; TestInputs::COUNT];
    //     inputs[TestInputs::OpFlags0 as usize][0] = Fr::from(5);
    //     inputs[TestInputs::OpFlags1 as usize][0] = Fr::from(7);
    //     inputs[TestInputs::PcIn as usize][0] = Fr::from(100);
    //     inputs[TestInputs::OpFlags0 as usize][1] = Fr::from(5);
    //     inputs[TestInputs::OpFlags1 as usize][1] = Fr::from(13);
    //     inputs[TestInputs::PcIn as usize][1] = Fr::from(104);
    //     let aux = combined_builder.compute_aux(&inputs);
    //     assert_eq!(aux, vec![vec![Fr::from(5 * 7), Fr::from(5 * 13)]]);

    //     let (az, bz, cz) = combined_builder.compute_spartan_Az_Bz_Cz(&inputs, &aux);
    //     combined_builder.assert_valid(&az, &bz, &cz);
    // }

    // #[test]
    // fn materialize_offset_eq() {
    //     let mut uniform_builder = R1CSBuilder::<Fr, TestInputs>::new();

    //     // OpFlags0 * OpFlags1 == Aux(0)
    //     struct TestConstraints();
    //     impl<const C: usize, F: JoltField> R1CSConstraints<C, F> for TestConstraints {
    //         type Inputs = TestInputs;
    //         fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
    //             let _aux = builder.allocate_prod(TestInputs::OpFlags0, TestInputs::OpFlags1);
    //         }
    //     }

    //     let constraints = TestConstraints();
    //     constraints.build_constraints(&mut uniform_builder);
    //     assert_eq!(uniform_builder.constraints.len(), 1);
    //     assert_eq!(uniform_builder.next_aux, 1);

    //     let num_steps = 2;

    //     // OpFlags0[n] = OpFlags0[n + 1];
    //     // PcIn[n] + 4 = PcIn[n + 1]
    //     let non_uniform_constraint: OffsetEqConstraint<TestInputs> = OffsetEqConstraint::new(
    //         (Variable::Constant, false),
    //         (TestInputs::OpFlags0, false),
    //         (TestInputs::OpFlags0, true),
    //     );
    //     let combined_builder = CombinedUniformBuilder::construct(
    //         uniform_builder,
    //         num_steps,
    //         vec![non_uniform_constraint],
    //     );

    //     let offset_eq = combined_builder.materialize_offset_eq();
    //     let mut expected_condition = SparseEqualityItem::<Fr>::empty();
    //     expected_condition.constant = Fr::one();

    //     let mut expected_eq = SparseEqualityItem::<Fr>::empty();
    //     expected_eq.offset_vars = vec![
    //         (TestInputs::OpFlags0 as usize, false, Fr::one()),
    //         (TestInputs::OpFlags0 as usize, true, Fr::from_i64(-1)),
    //     ];

    //     assert_eq!(offset_eq.constraints[0].condition, expected_condition);
    //     assert_eq!(offset_eq.constraints[0].eq, expected_eq);
    // }

    // #[test]
    // fn compute_spartan() {
    //     // Tests that CombinedBuilder.compute_spartan matches that naively computed from the big matrices A,B,C, z
    //     let (builder, key) = simp_test_builder_key();
    //     let (big_a, big_b, big_c) = simp_test_big_matrices::<Fr>();
    //     let witness_segments: Vec<Vec<Fr>> = vec![
    //         vec![Fr::one(), Fr::from(5), Fr::from(9), Fr::from(13)], /* Q */
    //         vec![Fr::one(), Fr::from(5), Fr::from(9), Fr::from(13)], /* R */
    //         vec![Fr::one(), Fr::from(5), Fr::from(9), Fr::from(13)], /* S */
    //     ];

    //     let pad_witness: Vec<Vec<Fr>> = witness_segments
    //         .iter()
    //         .map(|segment| {
    //             let mut segment = segment.clone();
    //             segment.resize(segment.len().next_power_of_two(), Fr::zero());
    //             segment
    //         })
    //         .collect();
    //     let mut flat_witness = pad_witness.concat();
    //     flat_witness.resize(flat_witness.len().next_power_of_two(), Fr::zero());
    //     flat_witness.push(Fr::one());
    //     flat_witness.resize(flat_witness.len().next_power_of_two(), Fr::zero());

    //     let (builder_az, builder_bz, builder_cz) =
    //         builder.compute_spartan_Az_Bz_Cz(&witness_segments, &[]);
    //     let mut dense_az = builder_az.to_dense().evals();
    //     let mut dense_bz = builder_bz.to_dense().evals();
    //     let mut dense_cz = builder_cz.to_dense().evals();
    //     dense_az.resize(key.num_rows_total(), Fr::zero());
    //     dense_bz.resize(key.num_rows_total(), Fr::zero());
    //     dense_cz.resize(key.num_rows_total(), Fr::zero());

    //     for row in 0..key.num_rows_total() {
    //         let mut az_eval = Fr::zero();
    //         let mut bz_eval = Fr::zero();
    //         let mut cz_eval = Fr::zero();
    //         for col in 0..key.num_cols_total() {
    //             az_eval += big_a[row * key.num_cols_total() + col] * flat_witness[col];
    //             bz_eval += big_b[row * key.num_cols_total() + col] * flat_witness[col];
    //             cz_eval += big_c[row * key.num_cols_total() + col] * flat_witness[col];
    //         }

    //         // Row 11 is the problem! Builder thinks this row should be 0. big_a thinks this row should be 17 (13 + 4)
    //         assert_eq!(dense_az[row], az_eval, "Row {row} failed in az_eval.");
    //         assert_eq!(dense_bz[row], bz_eval, "Row {row} failed in bz_eval.");
    //         assert_eq!(dense_cz[row], cz_eval, "Row {row} failed in cz_eval.");
    //     }
    // }
}
