use crate::{
    field::JoltField,
    r1cs::key::{SparseConstraints, UniformR1CS},
    utils::{
        mul_0_1_optimized,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
    },
};
use rayon::prelude::*;
use std::{collections::HashMap, fmt::Debug};

use super::{
    key::{NonUniformR1CS, SparseEqualityItem},
    ops::{ConstraintInput, Term, Variable, LC},
};

pub trait R1CSConstraintBuilder<F: JoltField> {
    type Inputs: ConstraintInput;

    fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>);
}

/// Constraints over a single row. Each variable points to a single item in Z and the corresponding coefficient.
#[derive(Clone, Debug)]
struct Constraint<I: ConstraintInput> {
    a: LC<I>,
    b: LC<I>,
    c: LC<I>,
}

impl<I: ConstraintInput> Constraint<I> {
    #[cfg(test)]
    fn is_sat(&self, inputs: &Vec<i64>) -> bool {
        // Find the number of variables and the number of aux. Inputs should be equal to this combined length
        let num_inputs = I::COUNT;

        let mut aux_set = std::collections::HashSet::new();
        for constraint in [&self.a, &self.b, &self.c] {
            for Term(var, _value) in constraint.terms() {
                if let Variable::Auxiliary(aux) = var {
                    aux_set.insert(aux);
                }
            }
        }
        let num_aux = aux_set.len();
        if !aux_set.is_empty() {
            assert_eq!(num_aux, *aux_set.iter().max().unwrap() + 1); // Ensure there are no gaps
        }
        let aux_index = |aux_index: usize| num_inputs + aux_index;

        let num_vars = num_inputs + num_aux;
        assert_eq!(num_vars, inputs.len());

        let mut a = 0;
        let mut b = 0;
        let mut c = 0;
        let mut buckets = [&mut a, &mut b, &mut c];
        let constraints = [&self.a, &self.b, &self.c];
        for (bucket, constraint) in buckets.iter_mut().zip(constraints.iter()) {
            for Term(var, coefficient) in constraint.terms() {
                match var {
                    Variable::Input(input) => {
                        let in_u: usize = (*input).into();
                        **bucket += inputs[in_u] * *coefficient;
                    }
                    Variable::Auxiliary(aux) => {
                        **bucket += inputs[aux_index(*aux)] * *coefficient;
                    }
                    Variable::Constant => {
                        **bucket += *coefficient;
                    }
                }
            }
        }

        println!("a * b == c      {a} * {b} == {c}");

        a * b == c
    }
}

type AuxComputationFunction<F> = dyn Fn(&[F]) -> F + Send + Sync;

struct AuxComputation<F: JoltField, I: ConstraintInput> {
    symbolic_inputs: Vec<LC<I>>,
    compute: Box<AuxComputationFunction<F>>,
}

impl<F: JoltField, I: ConstraintInput> AuxComputation<F, I> {
    fn new(
        _output: Variable<I>,
        symbolic_inputs: Vec<LC<I>>,
        compute: Box<AuxComputationFunction<F>>,
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
            for aux_var in &flat_vars {
                if let Variable::Auxiliary(aux_calc_index) = aux_var {
                    // Currently do not support aux computations dependent on those allocated after. Could support with dependency graph, instead
                    // dev should write their constraints sequentially. Simplifies aux computation parallelism.
                    if output_index <= *aux_calc_index {
                        panic!("Aux computation depends on future aux computation: {_output:?} = f({aux_var:?})");
                    }
                }
            }
        }

        Self {
            symbolic_inputs,
            compute,
        }
    }

    /// Computes auxiliary variable for batch_size steps using the evaluations of each
    /// linear combination (represented by self.symbolic_inputs).
    /// inputs: self.symbolic_inputs.len() inputs each of size batch_size
    #[tracing::instrument(skip_all, name = "AuxComputation::compute_batch")]
    fn compute_batch(&self, inputs: Vec<&[F]>, batch_size: usize) -> Vec<F> {
        assert_eq!(inputs.len(), self.symbolic_inputs.len());
        assert!(inputs.iter().all(|input| input.len() == batch_size));

        // Split into num_threads chunks and copy the corresponding inputs from each step
        // in the batch to a buffer owend by each thread's chunk to minimize allocs.

        let num_threads = rayon::current_num_threads();
        let chunk_size = (batch_size + num_threads - 1) / num_threads;
        let mut results: Vec<F> = unsafe_allocate_zero_vec(batch_size);

        results
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                let span = tracing::span!(tracing::Level::DEBUG, "chunk");
                let _enter = span.enter();
                let mut input_buffer: Vec<F> = unsafe_allocate_zero_vec(inputs.len());

                chunk
                    .iter_mut()
                    .enumerate()
                    .for_each(|(batch_index, result)| {
                        let global_index = chunk_index * chunk_size + batch_index;
                        inputs.iter().enumerate().for_each(|(i, input)| {
                            input_buffer[i] = input[global_index];
                        });
                        *result = (self.compute)(&input_buffer);
                    });
            });

        results
    }
}

pub struct R1CSBuilder<F: JoltField, I: ConstraintInput> {
    constraints: Vec<Constraint<I>>,
    pub next_aux: usize,
    aux_computations: Vec<AuxComputation<F, I>>,
}

impl<F: JoltField, I: ConstraintInput> Default for R1CSBuilder<F, I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: JoltField, I: ConstraintInput> R1CSBuilder<F, I> {
    pub fn new() -> Self {
        Self {
            constraints: vec![],
            next_aux: 0,
            aux_computations: vec![],
        }
    }

    fn allocate_aux(
        &mut self,
        symbolic_inputs: Vec<LC<I>>,
        compute: Box<AuxComputationFunction<F>>,
    ) -> Variable<I> {
        let new_aux = Variable::Auxiliary(self.next_aux);
        self.next_aux += 1;

        let computation = AuxComputation::new(new_aux, symbolic_inputs, compute);
        self.aux_computations.push(computation);

        new_aux
    }

    /// Index of variable within z.
    pub fn witness_index(&self, var: impl Into<Variable<I>>) -> usize {
        let var: Variable<I> = var.into();
        match var {
            Variable::Input(inner) => inner.into(),
            Variable::Auxiliary(aux_index) => I::COUNT + aux_index,
            Variable::Constant => I::COUNT + self.next_aux,
        }
    }

    pub fn constrain_eq(&mut self, left: impl Into<LC<I>>, right: impl Into<LC<I>>) {
        // left - right == 0
        let left: LC<I> = left.into();
        let right: LC<I> = right.into();

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
        condition: impl Into<LC<I>>,
        left: impl Into<LC<I>>,
        right: impl Into<LC<I>>,
    ) {
        // condition  * (left - right) == 0
        let condition: LC<I> = condition.into();
        let left: LC<I> = left.into();
        let right: LC<I> = right.into();

        let a = condition;
        let b = left - right;
        let c = LC::zero();
        let constraint = Constraint { a, b, c };
        self.constraints.push(constraint);
    }

    pub fn constrain_binary(&mut self, value: impl Into<LC<I>>) {
        let one: LC<I> = Variable::Constant.into();
        let a: LC<I> = value.into();
        let b = one - a.clone();
        // value * (1 - value)
        let constraint = Constraint {
            a,
            b,
            c: LC::zero(),
        };
        self.constraints.push(constraint);
    }

    pub fn constrain_if_else(
        &mut self,
        condition: impl Into<LC<I>>,
        result_true: impl Into<LC<I>>,
        result_false: impl Into<LC<I>>,
        alleged_result: impl Into<LC<I>>,
    ) {
        let condition: LC<I> = condition.into();
        let result_true: LC<I> = result_true.into();
        let result_false: LC<I> = result_false.into();
        let alleged_result: LC<I> = alleged_result.into();

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
        condition: impl Into<LC<I>>,
        result_true: impl Into<LC<I>>,
        result_false: impl Into<LC<I>>,
    ) -> Variable<I> {
        let (condition, result_true, result_false) =
            (condition.into(), result_true.into(), result_false.into());

        let aux_var = self.aux_if_else(&condition, &result_true, &result_false);

        self.constrain_if_else(condition, result_true, result_false, aux_var);
        aux_var
    }

    fn aux_if_else(
        &mut self,
        condition: &LC<I>,
        result_true: &LC<I>,
        result_false: &LC<I>,
    ) -> Variable<I> {
        // aux = (condition == 1) ? result_true : result_false;
        let if_else = |values: &[F]| -> F {
            assert_eq!(values.len(), 3);
            let condition = values[0];
            let result_true = values[1];
            let result_false = values[2];

            if condition.is_one() {
                result_true
            } else {
                result_false
            }
        };

        let symbolic_inputs = vec![condition.clone(), result_true.clone(), result_false.clone()];
        let compute = Box::new(if_else);
        self.allocate_aux(symbolic_inputs, compute)
    }

    pub fn constrain_pack_le(
        &mut self,
        unpacked: Vec<Variable<I>>,
        result: impl Into<LC<I>>,
        operand_bits: usize,
    ) {
        // Pack unpacked via a simple weighted linear combination
        // A + 2 * B + 4 * C + 8 * D, ...
        let packed: Vec<Term<I>> = unpacked
            .into_iter()
            .enumerate()
            .map(|(idx, unpacked)| Term(unpacked, 1 << (idx * operand_bits)))
            .collect();
        self.constrain_eq(packed, result);
    }

    #[must_use]
    pub fn allocate_pack_le(
        &mut self,
        unpacked: Vec<Variable<I>>,
        operand_bits: usize,
    ) -> Variable<I> {
        let packed = self.aux_pack_le(&unpacked, operand_bits);

        self.constrain_pack_le(unpacked, packed, operand_bits);
        packed
    }

    fn aux_pack_le(&mut self, to_pack: &[Variable<I>], operand_bits: usize) -> Variable<I> {
        let pack = move |values: &[F]| -> F {
            values
                .iter()
                .enumerate()
                .fold(F::zero(), |acc, (idx, &value)| {
                    acc + value.mul_01_optimized(F::from_u64(1 << (idx * operand_bits)).unwrap())
                })
        };

        let symbolic_inputs = to_pack.iter().cloned().map(|sym| sym.into()).collect();
        let compute = Box::new(pack);
        self.allocate_aux(symbolic_inputs, compute)
    }

    pub fn constrain_pack_be(
        &mut self,
        unpacked: Vec<Variable<I>>,
        result: impl Into<LC<I>>,
        operand_bits: usize,
    ) {
        // Pack unpacked via a simple weighted linear combination
        // A + 2 * B + 4 * C + 8 * D, ...
        // Note: Packing order is reversed from constrain_pack_le
        let packed: Vec<Term<I>> = unpacked
            .into_iter()
            .rev()
            .enumerate()
            .map(|(idx, unpacked)| Term(unpacked, 1 << (idx * operand_bits)))
            .collect();
        self.constrain_eq(packed, result);
    }

    #[must_use]
    pub fn allocate_pack_be(
        &mut self,
        unpacked: Vec<Variable<I>>,
        operand_bits: usize,
    ) -> Variable<I> {
        let packed = self.aux_pack_be(&unpacked, operand_bits);

        self.constrain_pack_be(unpacked, packed, operand_bits);
        packed
    }

    fn aux_pack_be(&mut self, to_pack: &[Variable<I>], operand_bits: usize) -> Variable<I> {
        let pack = move |values: &[F]| -> F {
            values
                .iter()
                .rev()
                .enumerate()
                .fold(F::zero(), |acc, (idx, &value)| {
                    acc + value.mul_01_optimized(F::from_u64(1 << (idx * operand_bits)).unwrap())
                })
        };

        let symbolic_inputs = to_pack.iter().cloned().map(|sym| sym.into()).collect();
        let compute = Box::new(pack);
        self.allocate_aux(symbolic_inputs, compute)
    }

    /// Constrain x * y == z
    pub fn constrain_prod(
        &mut self,
        x: impl Into<LC<I>>,
        y: impl Into<LC<I>>,
        z: impl Into<LC<I>>,
    ) {
        let constraint = Constraint {
            a: x.into(),
            b: y.into(),
            c: z.into(),
        };
        self.constraints.push(constraint);
    }

    #[must_use]
    pub fn allocate_prod(&mut self, x: impl Into<LC<I>>, y: impl Into<LC<I>>) -> Variable<I> {
        let (x, y) = (x.into(), y.into());
        let z = self.aux_prod(&x, &y);

        self.constrain_prod(x, y, z);
        z
    }

    fn aux_prod(&mut self, x: &LC<I>, y: &LC<I>) -> Variable<I> {
        let prod = |values: &[F]| {
            assert_eq!(values.len(), 2);

            mul_0_1_optimized(&values[0], &values[1])
        };

        let symbolic_inputs = vec![x.clone(), y.clone()];
        let compute = Box::new(prod);
        self.allocate_aux(symbolic_inputs, compute)
    }

    fn num_aux(&self) -> usize {
        self.next_aux
    }

    fn variable_to_column(&self, var: Variable<I>) -> usize {
        match var {
            Variable::Input(inner) => inner.into(),
            Variable::Auxiliary(aux) => I::COUNT + aux,
            Variable::Constant => (I::COUNT + self.num_aux()).next_power_of_two(),
        }
    }

    fn materialize(&self) -> UniformR1CS<F> {
        let a_len: usize = self.constraints.iter().map(|c| c.a.num_vars()).sum();
        let b_len: usize = self.constraints.iter().map(|c| c.b.num_vars()).sum();
        let c_len: usize = self.constraints.iter().map(|c| c.c.num_vars()).sum();
        let mut a_sparse = SparseConstraints::empty_with_capacity(a_len, self.constraints.len());
        let mut b_sparse = SparseConstraints::empty_with_capacity(b_len, self.constraints.len());
        let mut c_sparse = SparseConstraints::empty_with_capacity(c_len, self.constraints.len());

        let update_sparse = |row_index: usize, lc: &LC<I>, sparse: &mut SparseConstraints<F>| {
            lc.terms()
                .iter()
                .filter(|term| matches!(term.0, Variable::Input(_) | Variable::Auxiliary(_)))
                .for_each(|term| {
                    sparse.vars.push((
                        row_index,
                        self.variable_to_column(term.0),
                        F::from_i64(term.1),
                    ))
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
            num_vars: I::COUNT + self.num_aux(),
            num_rows: self.constraints.len(),
        }
    }
}

/// An Offset Linear Combination. If OffsetLC.0 is true, then the OffsetLC.1 refers to the next step in a uniform
/// constraint system.
pub type OffsetLC<I> = (bool, LC<I>);

/// A conditional constraint that Linear Combinations a, b are equal where a and b need not be in the same step an a
/// uniform constraint system.
pub struct OffsetEqConstraint<I: ConstraintInput> {
    cond: OffsetLC<I>,
    a: OffsetLC<I>,
    b: OffsetLC<I>,
}

impl<I: ConstraintInput> OffsetEqConstraint<I> {
    pub fn new(
        condition: (impl Into<LC<I>>, bool),
        a: (impl Into<LC<I>>, bool),
        b: (impl Into<LC<I>>, bool),
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

// TODO(sragss): Detailed documentation with wiki.
pub struct CombinedUniformBuilder<F: JoltField, I: ConstraintInput> {
    uniform_builder: R1CSBuilder<F, I>,

    /// Padded to the nearest power of 2
    uniform_repeat: usize,

    offset_equality_constraint: OffsetEqConstraint<I>,
}

#[tracing::instrument(skip_all, name = "batch_inputs")]
fn batch_inputs<'a, I: ConstraintInput, F: JoltField>(
    lc: &LC<I>,
    inputs: &'a [Vec<F>],
    aux: &'a [Vec<F>],
) -> Vec<&'a [F]> {
    let mut batch: Vec<&'a [F]> = Vec::with_capacity(lc.terms().len());
    lc.terms().iter().for_each(|term| match term.0 {
        Variable::Input(input) => batch.push(&inputs[input.into()]),
        Variable::Auxiliary(aux_index) => batch.push(&aux[aux_index]),
        _ => {}
    });
    batch
}

impl<F: JoltField, I: ConstraintInput> CombinedUniformBuilder<F, I> {
    pub fn construct(
        uniform_builder: R1CSBuilder<F, I>,
        uniform_repeat: usize,
        offset_equality_constraint: OffsetEqConstraint<I>,
    ) -> Self {
        assert!(uniform_repeat.is_power_of_two());
        Self {
            uniform_builder,
            uniform_repeat,
            offset_equality_constraint,
        }
    }

    /// Computes all auxiliary variables from inputs.
    /// inputs should be of the format [[I::0, I::0, ...], [I::1, I::1, ...], ... [I::N, I::N]].
    #[tracing::instrument(skip_all, name = "CombinedUniformBuilder::compute_aux")]
    pub fn compute_aux(&self, inputs: &[Vec<F>]) -> Vec<Vec<F>> {
        assert_eq!(inputs.len(), I::COUNT);
        inputs
            .iter()
            .for_each(|inner_input| assert_eq!(inner_input.len(), self.uniform_repeat));

        let mut lc_evals: HashMap<LC<I>, Vec<F>> = HashMap::new();
        for aux_computation in &self.uniform_builder.aux_computations {
            for input in &aux_computation.symbolic_inputs {
                lc_evals
                    .entry(input.clone())
                    .or_insert_with(|| unsafe_allocate_zero_vec(self.uniform_repeat));
            }
        }

        let span_allocate = tracing::span!(tracing::Level::DEBUG, "eval_lcs");
        let _enter_allocate = span_allocate.enter();

        // Find aux vars with dependencies
        // AuxCompute.output -> Variable::Auxiliary
        let mut aux_dep_map = HashMap::new();
        for lc in lc_evals.keys() {
            let aux_term = lc
                .terms()
                .iter()
                .find(|term| matches!(term.0, Variable::Auxiliary(_)));
            if let Some(term) = aux_term {
                #[cfg(test)]
                assert_eq!(
                    lc.terms()
                        .iter()
                        .filter(|term| matches!(term.0, Variable::Auxiliary(_)))
                        .count(),
                    1
                );

                if let Variable::Auxiliary(aux_index) = term.0 {
                    aux_dep_map.insert(aux_index, lc.clone());
                }
            }
        }

        let mut aux_evals = vec![vec![]; self.uniform_builder.num_aux()];

        // Evaluate the LCs with no dependencies
        lc_evals.par_iter_mut().for_each(|(lc, result)| {
            if !aux_dep_map.values().any(|v| v == lc) {
                let inputs = batch_inputs(lc, inputs, &aux_evals);
                lc.evaluate_batch_mut(&inputs, result);
            }
        });
        drop(_enter_allocate);

        for (aux_index, aux_compute) in self.uniform_builder.aux_computations.iter().enumerate() {
            let inputs_by_step: Vec<&[F]> = aux_compute
                .symbolic_inputs
                .iter()
                .map(|lc| lc_evals.get(lc).unwrap().as_ref())
                .collect();

            aux_evals[aux_index] = aux_compute.compute_batch(inputs_by_step, self.uniform_repeat);

            if let Some(lc) = aux_dep_map.get(&aux_index) {
                let result = lc_evals.get_mut(lc).unwrap();
                let inputs = batch_inputs(lc, inputs, &aux_evals);
                lc.evaluate_batch_mut(&inputs, result);
            }
        }

        drop_in_background_thread(lc_evals);

        aux_evals
    }

    /// Total number of rows used across all uniform constraints across all repeats. Repeat padded to 2, but repeat * num_constraints not, num_constraints not.
    pub(super) fn uniform_repeat_constraint_rows(&self) -> usize {
        self.uniform_repeat * self.uniform_builder.constraints.len()
    }

    pub(super) fn offset_eq_constraint_rows(&self) -> usize {
        self.uniform_repeat
    }

    /// Total number of rows used across all repeated constraints. Not padded to nearest power of two.
    pub(super) fn constraint_rows(&self) -> usize {
        self.offset_eq_constraint_rows() + self.uniform_repeat_constraint_rows()
    }

    pub(super) fn uniform_repeat(&self) -> usize {
        self.uniform_repeat
    }

    /// Materializes the uniform constraints into a single sparse (value != 0) A, B, C matrix represented in (row, col, value) format.
    pub fn materialize_uniform(&self) -> UniformR1CS<F> {
        self.uniform_builder.materialize()
    }

    pub fn materialize_offset_eq(&self) -> NonUniformR1CS<F> {
        // (a - b) * condition == 0
        // A: a - b
        // B: condition
        // C: 0

        let mut eq = SparseEqualityItem::<F>::empty();
        let mut condition = SparseEqualityItem::<F>::empty();

        let constraint = &self.offset_equality_constraint;

        constraint
            .cond
            .1
            .terms()
            .iter()
            .filter(|term| matches!(term.0, Variable::Input(_) | Variable::Auxiliary(_)))
            .for_each(|term| {
                condition.offset_vars.push((
                    self.uniform_builder.variable_to_column(term.0),
                    constraint.cond.0,
                    F::from_i64(term.1),
                ))
            });
        if let Some(term) = constraint.cond.1.constant_term() {
            condition.constant = F::from_i64(term.1);
        }

        // Can't simply combine like terms because of the offset
        let lhs = constraint.a.1.clone();
        let rhs = -constraint.b.1.clone();

        lhs.terms()
            .iter()
            .filter(|term| matches!(term.0, Variable::Input(_) | Variable::Auxiliary(_)))
            .for_each(|term| {
                eq.offset_vars.push((
                    self.uniform_builder.variable_to_column(term.0),
                    constraint.a.0,
                    F::from_i64(term.1),
                ))
            });
        rhs.terms()
            .iter()
            .filter(|term| matches!(term.0, Variable::Input(_) | Variable::Auxiliary(_)))
            .for_each(|term| {
                eq.offset_vars.push((
                    self.uniform_builder.variable_to_column(term.0),
                    constraint.b.0,
                    F::from_i64(term.1),
                ))
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

        NonUniformR1CS::new(eq, condition)
    }

    /// inputs should be of the format [[I::0, I::0, ...], [I::1, I::1, ...], ... [I::N, I::N]]
    /// aux should be of the format [[Aux(0), Aux(0), ...], ... [Aux(self.next_aux - 1), ...]]
    #[tracing::instrument(skip_all, name = "CombinedUniformBuilder::compute_spartan")]
    pub fn compute_spartan_Az_Bz_Cz(
        &self,
        inputs: &[Vec<F>],
        aux: &[Vec<F>],
    ) -> (Vec<F>, Vec<F>, Vec<F>) {
        assert_eq!(inputs.len(), I::COUNT);
        let num_aux = self.uniform_builder.num_aux();
        assert_eq!(aux.len(), num_aux);
        assert!(inputs
            .iter()
            .chain(aux.iter())
            .all(|inner_input| inner_input.len() == self.uniform_repeat));

        let uniform_constraint_rows = self.uniform_repeat_constraint_rows();
        // TODO(sragss): Allocation can overshoot by up to a factor of 2, Spartan could handle non-pow-2 Az,Bz,Cz
        let constraint_rows = self.constraint_rows().next_power_of_two();
        let (mut Az, mut Bz, mut Cz) = (
            unsafe_allocate_zero_vec(constraint_rows),
            unsafe_allocate_zero_vec(constraint_rows),
            unsafe_allocate_zero_vec(constraint_rows),
        );

        let batch_inputs = |lc: &LC<I>| batch_inputs(lc, inputs, aux);

        // uniform_constraints: Xz[0..uniform_constraint_rows]
        // TODO(sragss): Attempt moving onto key and computing from materialized rows rather than linear combos
        let span = tracing::span!(tracing::Level::DEBUG, "compute_constraints");
        let enter = span.enter();
        let az_chunks = Az.par_chunks_mut(self.uniform_repeat);
        let bz_chunks = Bz.par_chunks_mut(self.uniform_repeat);
        let cz_chunks = Cz.par_chunks_mut(self.uniform_repeat);

        self.uniform_builder
            .constraints
            .par_iter()
            .zip(az_chunks.zip(bz_chunks.zip(cz_chunks)))
            .for_each(|(constraint, (az_chunk, (bz_chunk, cz_chunk)))| {
                let a_inputs = batch_inputs(&constraint.a);
                let b_inputs = batch_inputs(&constraint.b);
                let c_inputs = batch_inputs(&constraint.c);

                constraint.a.evaluate_batch_mut(&a_inputs, az_chunk);
                constraint.b.evaluate_batch_mut(&b_inputs, bz_chunk);
                constraint.c.evaluate_batch_mut(&c_inputs, cz_chunk);
            });
        drop(enter);

        // offset_equality_constraints: Xz[uniform_constraint_rows..uniform_constraint_rows + 1]
        // (a - b) * condition == 0
        // For the final step we will not compute the offset terms, and will assume the condition to be set to 0
        let span = tracing::span!(tracing::Level::DEBUG, "offset_eq");
        let _enter = span.enter();

        let constr = &self.offset_equality_constraint;
        let condition_evals = constr
            .cond
            .1
            .evaluate_batch(&batch_inputs(&constr.cond.1), self.uniform_repeat);
        let eq_a_evals = constr
            .a
            .1
            .evaluate_batch(&batch_inputs(&constr.a.1), self.uniform_repeat);
        let eq_b_evals = constr
            .b
            .1
            .evaluate_batch(&batch_inputs(&constr.b.1), self.uniform_repeat);

        let Az_off = Az[uniform_constraint_rows..uniform_constraint_rows + self.uniform_repeat]
            .par_iter_mut();
        let Bz_off = Bz[uniform_constraint_rows..uniform_constraint_rows + self.uniform_repeat]
            .par_iter_mut();

        (0..self.uniform_repeat)
            .into_par_iter()
            .zip(Az_off.zip(Bz_off))
            .for_each(|(step_index, (az, bz))| {
                // Write corresponding values, if outside the step range, only include the constant.
                let a_step = step_index + if constr.a.0 { 1 } else { 0 };
                let b_step = step_index + if constr.b.0 { 1 } else { 0 };
                let a = eq_a_evals
                    .get(a_step)
                    .cloned()
                    .unwrap_or(constr.a.1.constant_term_field());
                let b = eq_b_evals
                    .get(b_step)
                    .cloned()
                    .unwrap_or(constr.b.1.constant_term_field());
                *az = a - b;

                let condition_step = step_index + if constr.cond.0 { 1 } else { 0 };
                *bz = condition_evals
                    .get(condition_step)
                    .cloned()
                    .unwrap_or(constr.cond.1.constant_term_field());
            });

        (Az, Bz, Cz)
    }

    #[cfg(test)]
    pub fn assert_valid(&self, az: &[F], bz: &[F], cz: &[F]) {
        let rows = az.len();
        let expected_rows = self.constraint_rows().next_power_of_two();
        assert_eq!(az.len(), expected_rows);
        assert_eq!(bz.len(), expected_rows);
        assert_eq!(cz.len(), expected_rows);
        for constraint_index in 0..rows {
            if az[constraint_index] * bz[constraint_index] != cz[constraint_index] {
                let uniform_constraint_index = constraint_index / self.uniform_repeat;
                let step_index = constraint_index % self.uniform_repeat;
                panic!(
                    "Mismatch at global constraint {constraint_index} => {:?}\n\
                    uniform constraint: {uniform_constraint_index}\n\
                    step: {step_index}",
                    self.uniform_builder.constraints[uniform_constraint_index]
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r1cs::test::{simp_test_big_matrices, simp_test_builder_key, TestInputs};
    use ark_bn254::Fr;
    use strum::EnumCount;

    fn aux_compute_single<F: JoltField, I: ConstraintInput>(
        aux_compute: &AuxComputation<F, I>,
        single_step_inputs: &[F],
    ) -> F {
        let multi_step_inputs: Vec<Vec<F>> = single_step_inputs
            .iter()
            .map(|input| vec![input.clone()])
            .collect();
        let multi_step_inputs_ref: Vec<&[F]> =
            multi_step_inputs.iter().map(|v| v.as_slice()).collect();
        aux_compute.compute_batch(multi_step_inputs_ref, 1)[0]
    }

    #[test]
    fn aux_compute_simple() {
        let a: LC<TestInputs> = 12i64.into();
        let b: LC<TestInputs> = 20i64.into();
        let lc = vec![a + b];
        let lambda = |input: &[Fr]| {
            assert_eq!(input.len(), 1);
            input[0]
        };
        let aux =
            AuxComputation::<Fr, TestInputs>::new(Variable::Auxiliary(0), lc, Box::new(lambda));
        let result = aux_compute_single(&aux, &[Fr::from(32)]);
        assert_eq!(result, Fr::from(32));
    }

    #[test]
    #[should_panic]
    fn aux_compute_depends_on_aux() {
        let a: LC<TestInputs> = 12i64.into();
        let b: LC<TestInputs> = Variable::Auxiliary(1).into();
        let lc = vec![a + b];
        let lambda = |_input: &[Fr]| unimplemented!();
        let _aux =
            AuxComputation::<Fr, TestInputs>::new(Variable::Auxiliary(0), lc, Box::new(lambda));
    }

    #[test]
    fn eq_builder() {
        let mut builder = R1CSBuilder::<Fr, TestInputs>::new();

        // PcIn + PcOut == BytecodeA + 2 BytecodeVOpcode
        struct TestConstraints();
        impl<F: JoltField> R1CSConstraintBuilder<F> for TestConstraints {
            type Inputs = TestInputs;
            fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
                let left = Self::Inputs::PcIn + Self::Inputs::PcOut;
                let right = Self::Inputs::BytecodeA + 2i64 * Self::Inputs::BytecodeVOpcode;
                builder.constrain_eq(left, right);
            }
        }

        let concrete_constraints = TestConstraints();
        concrete_constraints.build_constraints(&mut builder);
        assert!(builder.constraints.len() == 1);
        let constraint = &builder.constraints[0];
        let mut z = vec![0i64; TestInputs::COUNT];

        // 2 + 6 == 6 + 2*1
        z[TestInputs::PcIn as usize] = 2;
        z[TestInputs::PcOut as usize] = 6;
        z[TestInputs::BytecodeA as usize] = 6;
        z[TestInputs::BytecodeVOpcode as usize] = 1;
        assert!(constraint.is_sat(&z));

        // 2 + 6 != 6 + 2*2
        z[TestInputs::BytecodeVOpcode as usize] = 2;
        assert!(!constraint.is_sat(&z));
    }

    #[test]
    fn if_else_builder() {
        let mut builder = R1CSBuilder::<Fr, TestInputs>::new();

        // condition * (true_outcome - false_outcome) = (result - false_outcome)
        // PcIn * (BytecodeVRS1 - BytecodeVRS2) == BytecodeA - BytecodeVRS2
        // If PcIn == 1: BytecodeA = BytecodeVRS1
        // If PcIn == 0: BytecodeA = BytecodeVRS2
        struct TestConstraints();
        impl<F: JoltField> R1CSConstraintBuilder<F> for TestConstraints {
            type Inputs = TestInputs;
            fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
                let condition = Self::Inputs::PcIn;
                let true_outcome = Self::Inputs::BytecodeVRS1;
                let false_outcome = Self::Inputs::BytecodeVRS2;
                let alleged_result = Self::Inputs::BytecodeA;
                builder.constrain_if_else(condition, true_outcome, false_outcome, alleged_result);
            }
        }

        let concrete_constraints = TestConstraints();
        concrete_constraints.build_constraints(&mut builder);
        assert!(builder.constraints.len() == 1);
        let constraint = &builder.constraints[0];

        let mut z = vec![0i64; TestInputs::COUNT];
        z[TestInputs::PcIn as usize] = 1;
        z[TestInputs::BytecodeA as usize] = 6;
        z[TestInputs::BytecodeVRS1 as usize] = 6;
        z[TestInputs::BytecodeVRS2 as usize] = 10;
        assert!(constraint.is_sat(&z));
        z[TestInputs::PcIn as usize] = 0;
        assert!(!constraint.is_sat(&z));
        z[TestInputs::BytecodeA as usize] = 10;
        assert!(constraint.is_sat(&z));
    }

    #[test]
    fn alloc_if_else_builder() {
        let mut builder = R1CSBuilder::<Fr, TestInputs>::new();

        // condition * (true_outcome - false_outcome) = (result - false_outcome)
        // PcIn * (BytecodeVRS1 - BytecodeVRS2) == AUX_RESULT - BytecodeVRS2
        // If PcIn == 1: AUX_RESULT = BytecodeVRS1
        // If PcIn == 0: AUX_RESULT = BytecodeVRS2
        // AUX_RESULT == BytecodeVImm
        struct TestConstraints();
        impl<F: JoltField> R1CSConstraintBuilder<F> for TestConstraints {
            type Inputs = TestInputs;
            fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
                let condition = Self::Inputs::PcIn + Self::Inputs::PcOut;
                let true_outcome = Self::Inputs::BytecodeVRS1;
                let false_outcome = Self::Inputs::BytecodeVRS2;
                let branch_result =
                    builder.allocate_if_else(condition, true_outcome, false_outcome);
                builder.constrain_eq(branch_result, Self::Inputs::BytecodeVImm);
            }
        }

        let concrete_constraints = TestConstraints();
        concrete_constraints.build_constraints(&mut builder);
        assert_eq!(builder.constraints.len(), 2);
        let (branch_constraint, eq_constraint) = (&builder.constraints[0], &builder.constraints[1]);

        let mut z = vec![0i64; TestInputs::COUNT + 1]; // 1 aux
        let true_branch_result: i64 = 12;
        let false_branch_result: i64 = 10;
        let aux_index = builder.witness_index(Variable::Auxiliary(0));
        z[TestInputs::PcIn as usize] = 1;
        z[TestInputs::BytecodeVRS1 as usize] = true_branch_result;
        z[TestInputs::BytecodeVRS2 as usize] = false_branch_result;
        z[TestInputs::BytecodeVImm as usize] = true_branch_result;
        z[aux_index] = true_branch_result;
        assert!(branch_constraint.is_sat(&z));
        assert!(eq_constraint.is_sat(&z));

        z[aux_index] = false_branch_result;
        assert!(!branch_constraint.is_sat(&z));
        assert!(!eq_constraint.is_sat(&z));

        z[TestInputs::BytecodeVImm as usize] = false_branch_result;
        assert!(!branch_constraint.is_sat(&z));
        assert!(eq_constraint.is_sat(&z));

        z[TestInputs::PcIn as usize] = 0;
        assert!(branch_constraint.is_sat(&z));
        assert!(eq_constraint.is_sat(&z));

        assert_eq!(builder.aux_computations.len(), 1);
        let compute_2 = aux_compute_single(
            &builder.aux_computations[0],
            &[Fr::one(), Fr::from(2), Fr::from(3)],
        );
        assert_eq!(compute_2, Fr::from(2));
        let compute_2 = aux_compute_single(
            &builder.aux_computations[0],
            &[Fr::zero(), Fr::from(2), Fr::from(3)],
        );
        assert_eq!(compute_2, Fr::from(3));
    }

    #[test]
    fn packing_le_builder() {
        let mut builder = R1CSBuilder::<Fr, TestInputs>::new();

        // pack_le(OpFlags0, OpFlags1, OpFlags2, OpFlags3) == BytecodeA
        struct TestConstraints();
        impl<F: JoltField> R1CSConstraintBuilder<F> for TestConstraints {
            type Inputs = TestInputs;
            fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
                let result = Variable::Input(TestInputs::BytecodeA);
                let unpacked: Vec<Variable<TestInputs>> = vec![
                    TestInputs::OpFlags0.into(),
                    TestInputs::OpFlags1.into(),
                    TestInputs::OpFlags2.into(),
                    TestInputs::OpFlags3.into(),
                ];
                builder.constrain_pack_le(unpacked, result, 1);
            }
        }

        let concrete_constraints = TestConstraints();
        concrete_constraints.build_constraints(&mut builder);
        assert_eq!(builder.constraints.len(), 1);
        let constraint = &builder.constraints[0];

        // 1101 == 13
        let mut z = vec![0i64; TestInputs::COUNT];
        // (little endian)
        z[TestInputs::OpFlags0 as usize] = 1;
        z[TestInputs::OpFlags1 as usize] = 0;
        z[TestInputs::OpFlags2 as usize] = 1;
        z[TestInputs::OpFlags3 as usize] = 1;
        z[TestInputs::BytecodeA as usize] = 13;

        assert!(constraint.is_sat(&z));
    }

    #[test]
    fn alloc_packing_le_builder() {
        let mut builder = R1CSBuilder::<Fr, TestInputs>::new();

        // pack_le(OpFlags0, OpFlags1, OpFlags2, OpFlags3) == Aux(0)
        struct TestConstraints();
        impl<F: JoltField> R1CSConstraintBuilder<F> for TestConstraints {
            type Inputs = TestInputs;
            fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
                let unpacked: Vec<Variable<TestInputs>> = vec![
                    TestInputs::OpFlags0.into(),
                    TestInputs::OpFlags1.into(),
                    TestInputs::OpFlags2.into(),
                    TestInputs::OpFlags3.into(),
                ];
                let _result = builder.allocate_pack_le(unpacked, 1);
            }
        }

        let concrete_constraints = TestConstraints();
        concrete_constraints.build_constraints(&mut builder);
        assert_eq!(builder.constraints.len(), 1);
        let constraint = &builder.constraints[0];

        // 1101 == 13
        let mut z = vec![0i64; TestInputs::COUNT + 1];
        // (little endian)
        z[TestInputs::OpFlags0 as usize] = 1;
        z[TestInputs::OpFlags1 as usize] = 0;
        z[TestInputs::OpFlags2 as usize] = 1;
        z[TestInputs::OpFlags3 as usize] = 1;

        assert_eq!(builder.aux_computations.len(), 1);
        let computed_aux = aux_compute_single(
            &builder.aux_computations[0],
            &[Fr::one(), Fr::zero(), Fr::one(), Fr::one()],
        );
        assert_eq!(computed_aux, Fr::from(13));
        z[builder.witness_index(Variable::Auxiliary(0))] = 13;
        assert!(constraint.is_sat(&z));
    }

    #[test]
    fn packing_be_builder() {
        let mut builder = R1CSBuilder::<Fr, TestInputs>::new();

        // pack_be(OpFlags0, OpFlags1, OpFlags2, OpFlags3) == BytecodeA
        struct TestConstraints();
        impl<F: JoltField> R1CSConstraintBuilder<F> for TestConstraints {
            type Inputs = TestInputs;
            fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
                let result = Variable::Input(TestInputs::BytecodeA);
                let unpacked: Vec<Variable<TestInputs>> = vec![
                    TestInputs::OpFlags0.into(),
                    TestInputs::OpFlags1.into(),
                    TestInputs::OpFlags2.into(),
                    TestInputs::OpFlags3.into(),
                ];
                builder.constrain_pack_be(unpacked, result, 1);
            }
        }

        let concrete_constraints = TestConstraints();
        concrete_constraints.build_constraints(&mut builder);
        assert_eq!(builder.constraints.len(), 1);
        let constraint = &builder.constraints[0];

        // 1101 == 13
        let mut z = vec![0i64; TestInputs::COUNT];
        // (big endian)
        z[TestInputs::OpFlags0 as usize] = 1;
        z[TestInputs::OpFlags1 as usize] = 1;
        z[TestInputs::OpFlags2 as usize] = 0;
        z[TestInputs::OpFlags3 as usize] = 1;
        z[TestInputs::BytecodeA as usize] = 13;

        assert!(constraint.is_sat(&z));
    }

    #[test]
    fn prod() {
        let mut builder = R1CSBuilder::<Fr, TestInputs>::new();

        // OpFlags0 * OpFlags1 == BytecodeA
        // OpFlags2 * OpFlags3 == Aux
        struct TestConstraints();
        impl<F: JoltField> R1CSConstraintBuilder<F> for TestConstraints {
            type Inputs = TestInputs;
            fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
                builder.constrain_prod(
                    TestInputs::OpFlags0,
                    TestInputs::OpFlags1,
                    TestInputs::BytecodeA,
                );
                let _aux = builder.allocate_prod(TestInputs::OpFlags2, TestInputs::OpFlags3);
            }
        }

        let concrete_constraints = TestConstraints();
        concrete_constraints.build_constraints(&mut builder);
        assert_eq!(builder.constraints.len(), 2);
        assert_eq!(builder.next_aux, 1);

        let mut z = vec![0i64; TestInputs::COUNT];
        // x * y == z
        z[TestInputs::OpFlags0 as usize] = 7;
        z[TestInputs::OpFlags1 as usize] = 10;
        z[TestInputs::BytecodeA as usize] = 70;
        assert!(builder.constraints[0].is_sat(&z));
        z[TestInputs::BytecodeA as usize] = 71;
        assert!(!builder.constraints[0].is_sat(&z));

        // x * y == aux
        z[TestInputs::OpFlags2 as usize] = 5;
        z[TestInputs::OpFlags3 as usize] = 7;
        z.push(35);
        assert!(builder.constraints[1].is_sat(&z));
        z[builder.witness_index(Variable::Auxiliary(0))] = 36;
        assert!(!builder.constraints[1].is_sat(&z));
    }

    #[test]
    fn alloc_prod() {
        let mut builder = R1CSBuilder::<Fr, TestInputs>::new();

        // OpFlags0 * OpFlags1 == Aux(0)
        struct TestConstraints();
        impl<F: JoltField> R1CSConstraintBuilder<F> for TestConstraints {
            type Inputs = TestInputs;
            fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
                let _aux = builder.allocate_prod(TestInputs::OpFlags0, TestInputs::OpFlags1);
            }
        }

        let constraints = TestConstraints();
        constraints.build_constraints(&mut builder);
        assert_eq!(builder.constraints.len(), 1);
        assert_eq!(builder.next_aux, 1);

        let mut z = vec![0i64; TestInputs::COUNT + 1];
        z[builder.witness_index(TestInputs::OpFlags0)] = 7;
        z[builder.witness_index(TestInputs::OpFlags1)] = 5;
        z[builder.witness_index(Variable::Auxiliary(0))] = 35;

        assert!(builder.constraints[0].is_sat(&z));
        z[builder.witness_index(Variable::Auxiliary(0))] = 36;
        assert!(!builder.constraints[0].is_sat(&z));
    }

    #[test]
    fn alloc_compute_simple_uniform_only() {
        let mut uniform_builder = R1CSBuilder::<Fr, TestInputs>::new();

        // OpFlags0 * OpFlags1 == Aux(0)
        struct TestConstraints();
        impl<F: JoltField> R1CSConstraintBuilder<F> for TestConstraints {
            type Inputs = TestInputs;
            fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
                let _aux = builder.allocate_prod(TestInputs::OpFlags0, TestInputs::OpFlags1);
            }
        }

        let constraints = TestConstraints();
        constraints.build_constraints(&mut uniform_builder);
        assert_eq!(uniform_builder.constraints.len(), 1);
        assert_eq!(uniform_builder.next_aux, 1);
        let num_steps = 2;
        let combined_builder = CombinedUniformBuilder::construct(
            uniform_builder,
            num_steps,
            OffsetEqConstraint::empty(),
        );

        let mut inputs = vec![vec![Fr::zero(); num_steps]; TestInputs::COUNT];
        inputs[TestInputs::OpFlags0 as usize][0] = Fr::from(5);
        inputs[TestInputs::OpFlags1 as usize][0] = Fr::from(7);
        inputs[TestInputs::OpFlags0 as usize][1] = Fr::from(11);
        inputs[TestInputs::OpFlags1 as usize][1] = Fr::from(13);
        let aux = combined_builder.compute_aux(&inputs);
        assert_eq!(aux, vec![vec![Fr::from(5 * 7), Fr::from(11 * 13)]]);

        let (az, bz, cz) = combined_builder.compute_spartan_Az_Bz_Cz(&inputs, &aux);
        assert_eq!(az.len(), 4);
        assert_eq!(bz.len(), 4);
        assert_eq!(cz.len(), 4);

        combined_builder.assert_valid(&az, &bz, &cz);
    }

    #[test]
    fn alloc_compute_complex_uniform_only() {
        let mut uniform_builder = R1CSBuilder::<Fr, TestInputs>::new();

        // OpFlags0 * OpFlags1 == Aux(0)
        // OpFlags2 + OpFlags3 == Aux(0)
        // (4 * RAMByte0 + 2) * OpFlags0 == Aux(1)
        // Aux(1) == RAMByte1
        struct TestConstraints();
        impl<F: JoltField> R1CSConstraintBuilder<F> for TestConstraints {
            type Inputs = TestInputs;
            fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
                let aux_0 = builder.allocate_prod(TestInputs::OpFlags0, TestInputs::OpFlags1);
                builder.constrain_eq(TestInputs::OpFlags2 + TestInputs::OpFlags3, aux_0);
                let aux_1 =
                    builder.allocate_prod(4 * TestInputs::RAMByte0 + 2i64, TestInputs::OpFlags0);
                builder.constrain_eq(aux_1, TestInputs::RAMByte1);
            }
        }

        let constraints = TestConstraints();
        constraints.build_constraints(&mut uniform_builder);
        assert_eq!(uniform_builder.constraints.len(), 4);
        assert_eq!(uniform_builder.next_aux, 2);

        let num_steps = 2;
        let combined_builder = CombinedUniformBuilder::construct(
            uniform_builder,
            num_steps,
            OffsetEqConstraint::empty(),
        );

        let mut inputs = vec![vec![Fr::zero(); num_steps]; TestInputs::COUNT];
        inputs[TestInputs::OpFlags0 as usize][0] = Fr::from(5);
        inputs[TestInputs::OpFlags1 as usize][0] = Fr::from(7);
        inputs[TestInputs::OpFlags2 as usize][0] = Fr::from(30);
        inputs[TestInputs::OpFlags3 as usize][0] = Fr::from(5);
        inputs[TestInputs::RAMByte0 as usize][0] = Fr::from(10);
        inputs[TestInputs::RAMByte1 as usize][0] = Fr::from((4 * 10 + 2) * 5);

        inputs[TestInputs::OpFlags0 as usize][1] = Fr::from(7);
        inputs[TestInputs::OpFlags1 as usize][1] = Fr::from(7);
        inputs[TestInputs::OpFlags2 as usize][1] = Fr::from(40);
        inputs[TestInputs::OpFlags3 as usize][1] = Fr::from(9);
        inputs[TestInputs::RAMByte0 as usize][1] = Fr::from(10);
        inputs[TestInputs::RAMByte1 as usize][1] = Fr::from((4 * 10 + 2) * 7);

        let aux = combined_builder.compute_aux(&inputs);
        assert_eq!(
            aux,
            vec![
                vec![Fr::from(35), Fr::from(49)],
                vec![Fr::from((4 * 10 + 2) * 5), Fr::from((4 * 10 + 2) * 7)]
            ]
        );

        let (az, bz, cz) = combined_builder.compute_spartan_Az_Bz_Cz(&inputs, &aux);
        assert_eq!(az.len(), 16);
        assert_eq!(bz.len(), 16);
        assert_eq!(cz.len(), 16);

        combined_builder.assert_valid(&az, &bz, &cz);
    }

    #[test]
    fn alloc_compute_simple_combined() {
        let mut uniform_builder = R1CSBuilder::<Fr, TestInputs>::new();

        // OpFlags0 * OpFlags1 == Aux(0)
        struct TestConstraints();
        impl<F: JoltField> R1CSConstraintBuilder<F> for TestConstraints {
            type Inputs = TestInputs;
            fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
                let _aux = builder.allocate_prod(TestInputs::OpFlags0, TestInputs::OpFlags1);
            }
        }

        let constraints = TestConstraints();
        constraints.build_constraints(&mut uniform_builder);
        assert_eq!(uniform_builder.constraints.len(), 1);
        assert_eq!(uniform_builder.next_aux, 1);

        let num_steps = 2;

        // OpFlags0[n] = OpFlags0[n + 1];
        // PcIn[n] + 4 = PcIn[n + 1]
        let non_uniform_constraint: OffsetEqConstraint<TestInputs> = OffsetEqConstraint::new(
            (TestInputs::OpFlags0, true),
            (TestInputs::OpFlags0, false),
            (TestInputs::OpFlags0, true),
        );
        let combined_builder =
            CombinedUniformBuilder::construct(uniform_builder, num_steps, non_uniform_constraint);

        let mut inputs = vec![vec![Fr::zero(); num_steps]; TestInputs::COUNT];
        inputs[TestInputs::OpFlags0 as usize][0] = Fr::from(5);
        inputs[TestInputs::OpFlags1 as usize][0] = Fr::from(7);
        inputs[TestInputs::PcIn as usize][0] = Fr::from(100);
        inputs[TestInputs::OpFlags0 as usize][1] = Fr::from(5);
        inputs[TestInputs::OpFlags1 as usize][1] = Fr::from(13);
        inputs[TestInputs::PcIn as usize][1] = Fr::from(104);
        let aux = combined_builder.compute_aux(&inputs);
        assert_eq!(aux, vec![vec![Fr::from(5 * 7), Fr::from(5 * 13)]]);

        let (az, bz, cz) = combined_builder.compute_spartan_Az_Bz_Cz(&inputs, &aux);
        assert_eq!(az.len(), 4);
        assert_eq!(bz.len(), 4);
        assert_eq!(cz.len(), 4);

        combined_builder.assert_valid(&az, &bz, &cz);
    }

    #[test]
    fn materialize_offset_eq() {
        let mut uniform_builder = R1CSBuilder::<Fr, TestInputs>::new();

        // OpFlags0 * OpFlags1 == Aux(0)
        struct TestConstraints();
        impl<F: JoltField> R1CSConstraintBuilder<F> for TestConstraints {
            type Inputs = TestInputs;
            fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
                let _aux = builder.allocate_prod(TestInputs::OpFlags0, TestInputs::OpFlags1);
            }
        }

        let constraints = TestConstraints();
        constraints.build_constraints(&mut uniform_builder);
        assert_eq!(uniform_builder.constraints.len(), 1);
        assert_eq!(uniform_builder.next_aux, 1);

        let num_steps = 2;

        // OpFlags0[n] = OpFlags0[n + 1];
        // PcIn[n] + 4 = PcIn[n + 1]
        let non_uniform_constraint: OffsetEqConstraint<TestInputs> = OffsetEqConstraint::new(
            (Variable::Constant, false),
            (TestInputs::OpFlags0, false),
            (TestInputs::OpFlags0, true),
        );
        let combined_builder =
            CombinedUniformBuilder::construct(uniform_builder, num_steps, non_uniform_constraint);

        let offset_eq = combined_builder.materialize_offset_eq();
        let mut expected_condition = SparseEqualityItem::<Fr>::empty();
        expected_condition.constant = Fr::one();

        let mut expected_eq = SparseEqualityItem::<Fr>::empty();
        expected_eq.offset_vars = vec![
            (TestInputs::OpFlags0 as usize, false, Fr::one()),
            (TestInputs::OpFlags0 as usize, true, Fr::from_i64(-1)),
        ];

        assert_eq!(offset_eq.condition, expected_condition);
        assert_eq!(offset_eq.eq, expected_eq);
    }

    #[test]
    fn compute_spartan() {
        // Tests that CombinedBuilder.compute_spartan matches that naively computed from the big matrices A,B,C, z
        let (builder, key) = simp_test_builder_key();
        let (big_a, big_b, big_c) = simp_test_big_matrices::<Fr>();
        let witness_segments: Vec<Vec<Fr>> = vec![
            vec![Fr::one(), Fr::from(5), Fr::from(9), Fr::from(13)], /* Q */
            vec![Fr::one(), Fr::from(5), Fr::from(9), Fr::from(13)], /* R */
            vec![Fr::one(), Fr::from(5), Fr::from(9), Fr::from(13)], /* S */
        ];

        let pad_witness: Vec<Vec<Fr>> = witness_segments
            .iter()
            .map(|segment| {
                let mut segment = segment.clone();
                segment.resize(segment.len().next_power_of_two(), Fr::zero());
                segment
            })
            .collect();
        let mut flat_witness = pad_witness.concat();
        flat_witness.resize(flat_witness.len().next_power_of_two(), Fr::zero());
        flat_witness.push(Fr::one());
        flat_witness.resize(flat_witness.len().next_power_of_two(), Fr::zero());
        let (mut builder_az, mut builder_bz, mut builder_cz) =
            builder.compute_spartan_Az_Bz_Cz(&witness_segments, &[]);
        builder_az.resize(key.num_rows_total(), Fr::zero());
        builder_bz.resize(key.num_rows_total(), Fr::zero());
        builder_cz.resize(key.num_rows_total(), Fr::zero());
        for row in 0..key.num_rows_total() {
            let mut az_eval = Fr::zero();
            let mut bz_eval = Fr::zero();
            let mut cz_eval = Fr::zero();
            for col in 0..key.num_cols_total() {
                az_eval += big_a[row * key.num_cols_total() + col] * flat_witness[col];
                bz_eval += big_b[row * key.num_cols_total() + col] * flat_witness[col];
                cz_eval += big_c[row * key.num_cols_total() + col] * flat_witness[col];
            }

            // Row 11 is the problem! Builder thinks this row should be 0. big_a thinks this row should be 17 (13 + 4)
            assert_eq!(builder_az[row], az_eval, "Row {row} failed in az_eval.");
            assert_eq!(builder_bz[row], bz_eval, "Row {row} failed in bz_eval.");
            assert_eq!(builder_cz[row], cz_eval, "Row {row} failed in cz_eval.");
        }
    }
}
