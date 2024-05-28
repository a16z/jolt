use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::{poly::field::JoltField, r1cs::new_r1cs::jolt_constraints::JoltInputs};
use std::fmt::Debug;
use std::ops::Range;

use super::ops::{from_i64, ConstraintInput, Term, Variable, LC};

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
        let num_input = I::COUNT;

        let mut aux_set = std::collections::HashSet::new();
        for constraint in [&self.a, &self.b, &self.c] {
            for Term(var, _value) in constraint.terms() {
                if let Variable::Auxiliary(aux) = var {
                    aux_set.insert(aux);
                }
            }
        }
        let num_aux = aux_set.len();
        if aux_set.len() > 0 {
            assert_eq!(num_aux, *aux_set.iter().max().unwrap() + 1); // Ensure there are no gaps
        }
        let aux_index = |aux_index: usize| num_input + aux_index;

        let num_vars = num_input + num_aux;
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

struct AuxComputation<F: JoltField, I: ConstraintInput> {
    output: Variable<I>,
    symbolic_inputs: Vec<LC<I>>,
    flat_vars: Vec<Variable<I>>,
    input_to_flat: Vec<Option<Range<usize>>>,
    compute: Box<dyn Fn(&[F]) -> F>,
}

impl<F: JoltField, I: ConstraintInput> AuxComputation<F, I> {
    fn new(
        output: Variable<I>,
        symbolic_inputs: Vec<LC<I>>,
        compute: Box<dyn Fn(&[F]) -> F>,
    ) -> Self {
        let flat_var_count: usize = symbolic_inputs.iter().map(|input| input.num_vars()).sum();
        let mut flat_vars = Vec::with_capacity(flat_var_count);
        let mut input_to_flat = Vec::with_capacity(symbolic_inputs.len());

        let mut range_start_index = 0;
        for input in &symbolic_inputs {
            let terms = input.terms();
            let num_vars = input.num_vars();
            for term in terms {
                if let Variable::Constant = term.0 {
                    continue;
                }
                flat_vars.push(term.0.clone());
            }
            if num_vars > 0 {
                input_to_flat.push(Some(range_start_index..(range_start_index + num_vars)));
                range_start_index += num_vars;
            } else {
                input_to_flat.push(None);
            }
        }
        assert_eq!(flat_vars.len(), flat_var_count);

        #[cfg(test)]
        for aux_var in &flat_vars {
            if let Variable::Auxiliary(_aux) = aux_var {
                // Currently do not support aux computations dependent on another. Should work if executed sequentially, but complicates
                // parallel aux computation. If needed, add via aux dependency graph.
                panic!("Aux computation depends on another aux computation: {output:?} = f({aux_var:?})");
            }
        }

        Self {
            output,
            symbolic_inputs,
            flat_vars,
            input_to_flat,
            compute,
        }
    }

    /// Takes one value per value in flat_vars.
    fn compute(&self, values: &[F]) -> F {
        assert_eq!(values.len(), self.flat_vars.len());
        assert_eq!(self.input_to_flat.len(), self.symbolic_inputs.len());
        let computed_inputs: Vec<_> = self
            .symbolic_inputs
            .iter()
            .enumerate()
            .map(|(input_index, input_lc)| {
                let values = if let Some(range) = self.input_to_flat[input_index].clone() {
                    &values[range]
                } else {
                    &[]
                };
                input_lc.evaluate(values)
            })
            .collect();
        (self.compute)(&computed_inputs)
    }
}

pub struct R1CSBuilder<F: JoltField, I: ConstraintInput> {
    pub constraints: Vec<Constraint<I>>,
    pub next_aux: usize,
    pub aux_computations: Vec<AuxComputation<F, I>>,
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
        compute: Box<dyn Fn(&[F]) -> F>,
    ) -> Variable<I> {
        let new_aux = Variable::Auxiliary(self.next_aux);
        self.next_aux += 1;

        let computation = AuxComputation::new(new_aux.clone(), symbolic_inputs, compute);
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
        let value: LC<I> = value.into();
        // value * (1 - value)
        let constraint = Constraint {
            a: value.clone(),
            b: one - value,
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
                .into_iter()
                .enumerate()
                .fold(F::zero(), |acc, (idx, &value)| {
                    acc + value * F::from_u64(1 << (idx * operand_bits)).unwrap()
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
                .into_iter()
                .rev()
                .enumerate()
                .fold(F::zero(), |acc, (idx, &value)| {
                    acc + value * F::from_u64(1 << (idx * operand_bits)).unwrap()
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

            values[0] * values[1]
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
            Variable::Auxiliary(aux) => I::COUNT + aux ,
            Variable::Constant => (I::COUNT + self.num_aux()).next_power_of_two()
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
            let sorted = lc.sorted_terms();
            sorted.iter()
                .filter(|term| matches!(term.0, Variable::Input(_) | Variable::Auxiliary(_)))
                .for_each(|term| sparse.vars.push((row_index, self.variable_to_column(term.0), from_i64::<F>(term.1))));
            if let Some(term) = sorted.last() {
                if matches!(term.0, Variable::Constant) {
                    sparse.consts.push((row_index, from_i64::<F>(term.1)));
                }
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
            num_rows: self.constraints.len()
        }
    }
}

pub type Coeff<F> = (usize, usize, F);

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct SparseConstraints<F: JoltField> {
    pub vars: Vec<Coeff<F>>,
    pub consts: Vec<(usize, F)>
}

impl<F: JoltField> SparseConstraints<F> {
    fn empty_with_capacity(vars: usize, consts: usize) -> Self {
        Self {
            vars: Vec::with_capacity(vars),
            consts: Vec::with_capacity(consts)
        }
    }

    fn empty() -> Self {
        Self {
            vars: vec![],
            consts: vec![]
        }
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct UniformR1CS<F: JoltField> {
    pub a: SparseConstraints<F>,
    pub b: SparseConstraints<F>,
    pub c: SparseConstraints<F>,

    /// Unpadded number of variables in uniform instance.
    pub num_vars: usize,

    /// Unpadded number of rows in uniform instance.
    pub num_rows: usize,
}

// TODO(sragss): More detailed documentation
/// Represents full matrices (A, B, C) with `uniform_builder` being repeated `uniform_repeat` times and `uniform_repeat - 1` "cross-uniform" constraints.
pub struct CombinedUniformBuilder<F: JoltField, I: ConstraintInput> {
    uniform_builder: R1CSBuilder<F, I>,
    uniform_repeat: usize,

    offset_equality_constraints: Vec<(LC<I>, LC<I>)>,
}

impl<F: JoltField, I: ConstraintInput> CombinedUniformBuilder<F, I> {
    pub fn construct(uniform_builder: R1CSBuilder<F, I>, uniform_repeat: usize, offset_equality_constraints: Vec<(LC<I>, LC<I>)>) -> Self {
        for (lc1, lc2) in &offset_equality_constraints {
            [lc1, lc2].iter().for_each(|lc| {
                lc.terms().iter().for_each(|term| {
                    if let Variable::Auxiliary(_) = term.0 {
                        // Actually may work fine but it's untested as it's unused.
                        panic!("Unsupported: offset_equality_constraints LC.terms contain aux");
                    }
                });
            });
        }
        Self { uniform_builder, uniform_repeat, offset_equality_constraints }
    }
    /// inputs should be of the format [[I::0, I::0, ...], [I::1, I::1, ...], ... [I::N, I::N]]
    pub fn compute_aux(&self, inputs: &[Vec<F>]) -> Vec<Vec<F>> {
        assert_eq!(inputs.len(), I::COUNT);
        inputs
            .iter()
            .for_each(|inner_input| assert_eq!(inner_input.len(), self.uniform_repeat));

        // let aux_len = self.next_aux * padded_trace_len;
        let mut aux = vec![vec![F::zero(); self.uniform_repeat]; self.uniform_builder.num_aux()];

        for (aux_index, aux_compute) in self.uniform_builder.aux_computations.iter().enumerate() {
            match aux_compute.output {
                Variable::Input(_) => panic!(),
                Variable::Constant => panic!(),
                Variable::Auxiliary(index) => assert_eq!(aux_index, index),
            }
            for step_index in 0..self.uniform_repeat {
                let required_z_values: Vec<F> = aux_compute
                    .flat_vars
                    .iter()
                    .map(|var| inputs[self.uniform_builder.witness_index(var.clone())][step_index])
                    .collect();
                aux[aux_index][step_index] = aux_compute.compute(&required_z_values);
            }
        }

        aux
    }

    // TODO(sragss): Prefix num_ for all below -- these generally need a rename
    pub(super) fn offset_eq_constraint_rows(&self) -> usize {
        self.offset_equality_constraints.len() * (self.uniform_repeat - 1) // First step is unconstrained
    }

    pub(super) fn uniform_repeat_constraint_rows(&self) -> usize {
        self.uniform_repeat * self.uniform_builder.constraints.len()
    }

    pub(super) fn constraint_rows(&self) -> usize {
        self.offset_eq_constraint_rows() + self.uniform_repeat_constraint_rows()
    }

    pub(super) fn uniform_rows(&self) -> usize {
        self.uniform_builder.constraints.len()
    }

    /// Number of columns / variables in the uniform instance. Does not include constant column.
    pub(super) fn uniform_columns(&self) -> usize {
        I::COUNT + self.uniform_builder.num_aux()
    }

    /// Total number of columns / variables across all uniform steps + 1 constant.
    pub(super) fn total_columns(&self) -> usize {
        // Uniform variables + constant variable "1"
        self.uniform_columns() * self.uniform_repeat + 1
    }

    pub(super) fn uniform_repeat(&self) -> usize {
        self.uniform_repeat
    }

    /// Materializes the uniform constraints into a single sparse (value != 0) A, B, C matrix represented in (row, col, value) format.
    pub fn materialize_uniform(&self) -> UniformR1CS<F> {
        self.uniform_builder.materialize()
    }

    /// inputs should be of the format [[I::0, I::0, ...], [I::1, I::1, ...], ... [I::N, I::N]]
    /// aux should be of the format [[Aux(0), Aux(0)], ... [Aux(self.next_aux - 1), ...]]
    pub fn compute_spartan(
        &self,
        inputs: &[Vec<F>],
        aux: &[Vec<F>],
    ) -> (Vec<F>, Vec<F>, Vec<F>) {
        assert_eq!(inputs.len(), I::COUNT);
        // let padded_trace_len = inputs[0].len();
        // TODO(sragss): Maybe this should be packed.
        inputs
            .iter()
            .for_each(|inner_input| assert_eq!(inner_input.len(), self.uniform_repeat));

        let num_aux = self.uniform_builder.num_aux();
        assert_eq!(aux.len(), num_aux);
        aux.iter()
            .for_each(|aux_segment| assert_eq!(aux_segment.len(), self.uniform_repeat));

        let offset_eq_constraint_rows = self.offset_eq_constraint_rows();
        let uniform_constraint_rows = self.uniform_repeat_constraint_rows();
        let constraint_rows = self.constraint_rows();

        // TODO(sragss): unsafe_alloc_zero_vec
        // TODO(sragss): btw ~unpadded~ should be handeled upstream
        let (mut Az, mut Bz, mut Cz) = (
            vec![F::zero(); constraint_rows],
            vec![F::zero(); constraint_rows],
            vec![F::zero(); constraint_rows],
        );

        let compute_lc = |lc: &LC<I>, inputs: &[Vec<F>], aux: &[Vec<F>], step_index: usize| {
            let mut eval = F::zero();
            lc.terms().iter().for_each(|term| match term.0 {
                Variable::Input(input) => {
                    eval += from_i64::<F>(term.1) * inputs[input.into()][step_index];
                }
                Variable::Auxiliary(aux_index) => {
                    assert!(aux_index < self.uniform_builder.num_aux());
                    eval += from_i64::<F>(term.1) * aux[aux_index][step_index];
                }
                Variable::Constant => {
                    eval += from_i64::<F>(term.1);
                }
            });
            eval
        };

        // 1. offset_equality_constraints: Xz[0..offset_eq_constraint_rows]
        // Az * 1 == Cz
        for (constraint_index, constraint) in self.offset_equality_constraints.iter().enumerate() {
            // For offset equality constraints we only constrain 1..N steps, the first does not have recursive definition.
            for step_index in 0..(self.uniform_repeat - 1) {
                let index = constraint_index * self.uniform_repeat + step_index;
                Az[index] = compute_lc(&constraint.0, &inputs, &aux, step_index);
                Bz[index] = F::one();
                Cz[index] = compute_lc(&constraint.1, &inputs, &aux, step_index);
            }
        }

        // 2. uniform_constraints: Xz[offset_eq_constraint_rows..uniform_constraint_rows]
        // TODO(sragss): Could either materialize then multiply or do it directly from the constraints. Should compare performance
        for (constraint_index, constraint) in self.uniform_builder.constraints.iter().enumerate() {
            for step_index in 0..self.uniform_repeat {
                let z_index =
                    offset_eq_constraint_rows + constraint_index * self.uniform_repeat + step_index;
                Az[z_index] = compute_lc(&constraint.a, &inputs, &aux, step_index);
                Bz[z_index] = compute_lc(&constraint.b, &inputs, &aux, step_index);
                Cz[z_index] = compute_lc(&constraint.c, &inputs, &aux, step_index);

                #[cfg(test)]
                {
                    if Az[z_index] * Bz[z_index] != Cz[z_index] {
                        println!("Constraint mismatch at step {step_index}!");
                        println!("Constraint Index: {constraint_index}");
                        println!("Constraint: {constraint:?}");
                        println!("Az: {:?}", Az[z_index]);
                        println!("Bz: {:?}", Bz[z_index]);
                        println!("Cz: {:?}", Cz[z_index]);
                        println!("Bytecode_Opcode: {:?}", inputs[JoltInputs::Bytecode_Opcode as usize][step_index]);
                        println!();
                        panic!();
                    }
                }
            }
        }

        (Az, Bz, Cz)
    }

    #[cfg(test)]
    pub fn assert_valid(
        &self,
        az: &[F],
        bz: &[F],
        cz: &[F],
    ) {
        let rows = az.len();
        let expected_rows = self.constraint_rows();
        assert_eq!(az.len(), expected_rows);
        assert_eq!(bz.len(), expected_rows);
        assert_eq!(cz.len(), expected_rows);
        for constraint_index in 0..rows {
            if az[constraint_index] * bz[constraint_index] != cz[constraint_index] {
                let (uniform_constraint_index, step_index) = if constraint_index != 0 {
                    (
                        self.uniform_repeat / constraint_index,
                        self.uniform_repeat % constraint_index,
                    )
                } else {
                    (0, 0)
                };
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
    use crate::{impl_r1cs_input_lc_conversions, input_range};

    use super::*;
    use crate::r1cs::new_r1cs::test::TestInputs;
    use ark_bn254::Fr;
    use strum::EnumCount;

    #[test]
    fn aux_compute_simple() {
        let lc = vec![LC::sum2(12i64, 20i64)];
        let lambda = |input: &[Fr]| {
            assert_eq!(input.len(), 1);
            input[0]
        };
        let aux =
            AuxComputation::<Fr, TestInputs>::new(Variable::Auxiliary(0), lc, Box::new(lambda));
        let result = aux.compute(&[]);
        assert_eq!(result, Fr::from(32));
    }

    #[test]
    fn aux_compute_advanced() {
        // (12 + 20) * (BytecodeA + PcIn) - 3 * PcOut
        let symbolic_inputs = vec![
            LC::sum2(12i64, 20i64),
            LC::sum2(TestInputs::BytecodeA, TestInputs::PcIn),
            (3 * TestInputs::PcOut).into(),
        ];
        let lambda = |input: &[Fr]| {
            assert_eq!(input.len(), 3);
            input[0] * input[1] - input[2]
        };
        let aux = AuxComputation::<Fr, TestInputs>::new(
            Variable::Auxiliary(0),
            symbolic_inputs,
            Box::new(lambda),
        );
        let result = aux.compute(&[Fr::from(5), Fr::from(10), Fr::from(7)]);
        assert_eq!(result, Fr::from((12 + 20) * (5 + 10) - (3 * 7)));
    }

    #[test]
    #[should_panic]
    fn aux_compute_depends_on_aux() {
        let lc = vec![LC::sum2(12i64, Variable::Auxiliary(1))];
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
                let left = LC::sum2(Self::Inputs::PcIn, Self::Inputs::PcOut);
                let right = LC::sum2(
                    Self::Inputs::BytecodeA,
                    (Self::Inputs::BytecodeVOpcode, 2i64),
                );
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
                let condition = LC::sum2(Self::Inputs::PcIn, Self::Inputs::PcOut);
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
        let compute_2 =
            builder.aux_computations[0].compute(&[Fr::one(), Fr::zero(), Fr::from(2), Fr::from(3)]);
        assert_eq!(compute_2, Fr::from(2));
        let compute_2 =
            builder.aux_computations[0].compute(&[Fr::zero(), Fr::one(), Fr::from(2), Fr::from(3)]);
        assert_eq!(compute_2, Fr::from(2));
        let compute_3 = builder.aux_computations[0].compute(&[
            Fr::zero(),
            Fr::zero(),
            Fr::from(2),
            Fr::from(3),
        ]);
        assert_eq!(compute_3, Fr::from(3));
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
        let computed_aux =
            builder.aux_computations[0].compute(&vec![Fr::one(), Fr::zero(), Fr::one(), Fr::one()]);
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
        let combined_builder = CombinedUniformBuilder::construct(uniform_builder, num_steps, vec![]);

        let mut inputs = vec![vec![Fr::zero(); num_steps]; TestInputs::COUNT];
        inputs[TestInputs::OpFlags0 as usize][0] = Fr::from(5);
        inputs[TestInputs::OpFlags1 as usize][0] = Fr::from(7);
        inputs[TestInputs::OpFlags0 as usize][1] = Fr::from(11);
        inputs[TestInputs::OpFlags1 as usize][1] = Fr::from(13);
        let aux = combined_builder.compute_aux(&inputs);
        assert_eq!(aux, vec![vec![Fr::from(5 * 7), Fr::from(11 * 13)]]);

        let (az, bz, cz) = combined_builder.compute_spartan(&inputs, &aux);
        assert_eq!(az.len(), 2);
        assert_eq!(bz.len(), 2);
        assert_eq!(cz.len(), 2);

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
                builder.constrain_eq(
                    LC::sum2(TestInputs::OpFlags2, TestInputs::OpFlags3),
                    aux_0.clone(),
                );
                let aux_1 = builder
                    .allocate_prod(4 * TestInputs::RAMByte0 + 2i64, TestInputs::OpFlags0);
                builder.constrain_eq(aux_1, TestInputs::RAMByte1);
            }
        }

        let constraints = TestConstraints();
        constraints.build_constraints(&mut uniform_builder);
        assert_eq!(uniform_builder.constraints.len(), 4);
        assert_eq!(uniform_builder.next_aux, 2);

        let num_steps = 2;
        let combined_builder = CombinedUniformBuilder::construct(uniform_builder, num_steps, vec![]);

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

        let (az, bz, cz) = combined_builder.compute_spartan(&inputs, &aux);
        assert_eq!(az.len(), 4 * 2);
        assert_eq!(bz.len(), 4 * 2);
        assert_eq!(cz.len(), 4 * 2);

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
        let non_uniform_constraints: Vec<(LC<TestInputs>, LC<TestInputs>)> = vec![
            (TestInputs::OpFlags0.into(), TestInputs::OpFlags0.into()),
            (LC::sum2(TestInputs::PcIn, 4), TestInputs::PcIn.into()),
        ];
        let combined_builder = CombinedUniformBuilder::construct(uniform_builder, num_steps, non_uniform_constraints);


        let mut inputs = vec![vec![Fr::zero(); num_steps]; TestInputs::COUNT];
        inputs[TestInputs::OpFlags0 as usize][0] = Fr::from(5);
        inputs[TestInputs::OpFlags1 as usize][0] = Fr::from(7);
        inputs[TestInputs::PcIn as usize][0] = Fr::from(100);
        inputs[TestInputs::OpFlags0 as usize][1] = Fr::from(5);
        inputs[TestInputs::OpFlags1 as usize][1] = Fr::from(13);
        inputs[TestInputs::PcIn as usize][1] = Fr::from(104);
        let aux = combined_builder.compute_aux(&inputs);
        assert_eq!(aux, vec![vec![Fr::from(5 * 7), Fr::from(5 * 13)]]);

        let (az, bz, cz) = combined_builder.compute_spartan(&inputs, &aux);
        assert_eq!(az.len(), 4);
        assert_eq!(bz.len(), 4);
        assert_eq!(cz.len(), 4);

        combined_builder.assert_valid(&az, &bz, &cz);
    }
}
