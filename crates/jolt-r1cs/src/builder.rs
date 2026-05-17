use std::ops::{Add, Neg, Sub};

use jolt_field::Field;
use thiserror::Error;

use crate::constraint::SparseRow;
use crate::ConstraintMatrices;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Variable(pub usize);

impl Variable {
    pub const ONE: Self = Self(0);
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum R1csBuilderError {
    #[error("missing witness value for variable {variable:?}")]
    MissingWitnessValue { variable: Variable },
    #[error("variable {variable:?} is out of bounds for witness with {num_vars} variables")]
    VariableOutOfBounds { variable: Variable, num_vars: usize },
    #[error("cannot assign the constant-one variable")]
    CannotAssignOne,
    #[error("variable {variable:?} already has a witness value")]
    WitnessAlreadyAssigned { variable: Variable },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearCombination<F> {
    pub terms: Vec<(Variable, F)>,
}

impl<F> LinearCombination<F> {
    pub fn zero() -> Self {
        Self { terms: Vec::new() }
    }
}

impl<F: Field> LinearCombination<F> {
    pub fn one() -> Self {
        Self::constant(F::one())
    }

    pub fn constant(value: F) -> Self {
        if value.is_zero() {
            Self::zero()
        } else {
            Self {
                terms: vec![(Variable::ONE, value)],
            }
        }
    }

    pub fn variable(variable: Variable) -> Self {
        Self {
            terms: vec![(variable, F::one())],
        }
    }

    pub fn scale(mut self, scale: F) -> Self {
        if scale.is_zero() {
            return Self::zero();
        }
        for (_, coefficient) in &mut self.terms {
            *coefficient *= scale;
        }
        self
    }

    pub fn evaluate(&self, witness: &[Option<F>]) -> Result<F, R1csBuilderError> {
        let mut result = F::zero();
        for &(variable, coefficient) in &self.terms {
            let value = witness
                .get(variable.0)
                .copied()
                .flatten()
                .ok_or(R1csBuilderError::MissingWitnessValue { variable })?;
            result += coefficient * value;
        }
        Ok(result)
    }

    pub fn into_sparse_row(self) -> SparseRow<F> {
        self.terms
            .into_iter()
            .filter_map(|(variable, coefficient)| {
                (!coefficient.is_zero()).then_some((variable.0, coefficient))
            })
            .collect()
    }
}

impl<F: Field> From<Variable> for LinearCombination<F> {
    fn from(variable: Variable) -> Self {
        Self::variable(variable)
    }
}

impl<F> Add for LinearCombination<F> {
    type Output = Self;

    fn add(mut self, mut rhs: Self) -> Self::Output {
        self.terms.append(&mut rhs.terms);
        self
    }
}

impl<F: Field> Sub for LinearCombination<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

impl<F: Field> Neg for LinearCombination<F> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        for (_, coefficient) in &mut self.terms {
            *coefficient = -*coefficient;
        }
        self
    }
}

#[derive(Clone, Debug)]
pub struct R1csBuilder<F: Field> {
    witness: Vec<Option<F>>,
    a: Vec<SparseRow<F>>,
    b: Vec<SparseRow<F>>,
    c: Vec<SparseRow<F>>,
}

impl<F: Field> Default for R1csBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field> R1csBuilder<F> {
    pub fn new() -> Self {
        Self {
            witness: vec![Some(F::one())],
            a: Vec::new(),
            b: Vec::new(),
            c: Vec::new(),
        }
    }

    pub fn alloc(&mut self, value: F) -> Variable {
        self.alloc_witness(Some(value))
    }

    pub fn alloc_unknown(&mut self) -> Variable {
        self.alloc_witness(None)
    }

    pub fn alloc_witness(&mut self, value: Option<F>) -> Variable {
        let variable = Variable(self.witness.len());
        self.witness.push(value);
        variable
    }

    pub fn num_vars(&self) -> usize {
        self.witness.len()
    }

    pub fn assign(&mut self, variable: Variable, value: F) -> Result<(), R1csBuilderError> {
        if variable == Variable::ONE {
            return Err(R1csBuilderError::CannotAssignOne);
        }

        let num_vars = self.witness.len();
        let slot = self
            .witness
            .get_mut(variable.0)
            .ok_or(R1csBuilderError::VariableOutOfBounds { variable, num_vars })?;
        if slot.is_some() {
            return Err(R1csBuilderError::WitnessAlreadyAssigned { variable });
        }

        *slot = Some(value);
        Ok(())
    }

    pub fn witness(&self) -> Result<Vec<F>, R1csBuilderError> {
        self.witness
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value.ok_or(R1csBuilderError::MissingWitnessValue {
                    variable: Variable(index),
                })
            })
            .collect()
    }

    pub fn assert_product<Lhs, Rhs, Output>(&mut self, lhs: Lhs, rhs: Rhs, output: Output)
    where
        Lhs: Into<LinearCombination<F>>,
        Rhs: Into<LinearCombination<F>>,
        Output: Into<LinearCombination<F>>,
    {
        let lhs = lhs.into();
        let rhs = rhs.into();
        let output = output.into();
        self.a.push(lhs.into_sparse_row());
        self.b.push(rhs.into_sparse_row());
        self.c.push(output.into_sparse_row());
    }

    pub fn assert_zero<Value>(&mut self, value: Value)
    where
        Value: Into<LinearCombination<F>>,
    {
        self.assert_product(value, LinearCombination::one(), LinearCombination::zero());
    }

    pub fn assert_equal<Lhs, Rhs>(&mut self, lhs: Lhs, rhs: Rhs)
    where
        Lhs: Into<LinearCombination<F>>,
        Rhs: Into<LinearCombination<F>>,
    {
        let lhs = lhs.into();
        let rhs = rhs.into();
        self.assert_zero(lhs - rhs);
    }

    pub fn multiply<Lhs, Rhs>(&mut self, lhs: Lhs, rhs: Rhs) -> Variable
    where
        Lhs: Into<LinearCombination<F>>,
        Rhs: Into<LinearCombination<F>>,
    {
        let lhs = lhs.into();
        let rhs = rhs.into();
        let value = lhs
            .evaluate(&self.witness)
            .ok()
            .zip(rhs.evaluate(&self.witness).ok())
            .map(|(lhs, rhs)| lhs * rhs);
        let output = self.alloc_witness(value);
        self.assert_product(lhs, rhs, LinearCombination::variable(output));
        output
    }

    pub fn into_matrices(self) -> ConstraintMatrices<F> {
        ConstraintMatrices::new(self.a.len(), self.witness.len(), self.a, self.b, self.c)
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn builder_checks_satisfied_product() {
        let mut builder = R1csBuilder::<Fr>::new();
        let x = builder.alloc(Fr::from_u64(3));
        let y = builder.alloc(Fr::from_u64(9));

        builder.assert_product(x, x, y);

        let witness = builder.witness().expect("witness is assigned");
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());
    }

    #[test]
    fn multiply_allocates_intermediate_witness() {
        let mut builder = R1csBuilder::<Fr>::new();
        let x = builder.alloc(Fr::from_u64(4));
        let y = builder.alloc(Fr::from_u64(5));

        let product = builder.multiply(x, y);

        let witness = builder.witness().expect("witness is assigned");
        assert_eq!(witness[product.0], Fr::from_u64(20));
        assert!(builder.into_matrices().check_witness(&witness).is_ok());
    }

    #[test]
    fn missing_witness_delays_intermediate_assignment() {
        let mut builder = R1csBuilder::<Fr>::new();
        let x = builder.alloc(Fr::from_u64(4));
        let y = builder.alloc_unknown();

        let product = builder.multiply(x, y);

        assert_eq!(
            builder.witness(),
            Err(R1csBuilderError::MissingWitnessValue { variable: y })
        );
        assert_eq!(
            LinearCombination::<Fr>::variable(product).evaluate(&builder.witness),
            Err(R1csBuilderError::MissingWitnessValue { variable: product })
        );
    }

    #[test]
    fn assign_fills_unknown_witness_value() {
        let mut builder = R1csBuilder::<Fr>::new();
        let variable = builder.alloc_unknown();

        builder
            .assign(variable, Fr::from_u64(17))
            .expect("assignment succeeds");

        let witness = builder.witness().expect("witness is assigned");
        assert_eq!(witness[variable.0], Fr::from_u64(17));
    }

    #[test]
    fn assign_rejects_constant_column() {
        let mut builder = R1csBuilder::<Fr>::new();

        let error = builder
            .assign(Variable::ONE, Fr::from_u64(2))
            .expect_err("constant column is not assignable");

        assert_eq!(error, R1csBuilderError::CannotAssignOne);
    }

    #[test]
    fn assign_rejects_already_assigned_variable() {
        let mut builder = R1csBuilder::<Fr>::new();
        let variable = builder.alloc(Fr::from_u64(9));

        let error = builder
            .assign(variable, Fr::from_u64(10))
            .expect_err("assigned variable cannot be overwritten");

        assert_eq!(error, R1csBuilderError::WitnessAlreadyAssigned { variable });
    }

    #[test]
    fn assign_rejects_out_of_bounds_variable() {
        let mut builder = R1csBuilder::<Fr>::new();
        let variable = Variable(7);

        let error = builder
            .assign(variable, Fr::from_u64(10))
            .expect_err("variable is out of bounds");

        assert_eq!(
            error,
            R1csBuilderError::VariableOutOfBounds {
                variable,
                num_vars: 1,
            }
        );
    }
}
