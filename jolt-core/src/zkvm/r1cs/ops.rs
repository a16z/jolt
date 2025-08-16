//! Defines the Linear Combination (LC) object and associated operations.
//! A LinearCombination is a vector of Terms, where each Term is a pair of a Variable and a coefficient.

use crate::{
    field::JoltField,
    poly::multilinear_polynomial::MultilinearPolynomial,
    zkvm::r1cs::{
        inputs::{JoltR1CSInputs, WitnessPolyType},
        types::{
            AzType, AzValue, BzType, BzValue, CzType, CzValue
        },
    },
    utils::u64_and_sign::{U128AndSign, U64AndSign},
};
use std::fmt::Debug;
#[cfg(test)]
use std::fmt::Write as _;
use std::hash::Hash;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Variable {
    Input(usize),
    Constant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Term(pub Variable, pub i128);
impl Term {
    #[cfg(test)]
    fn pretty_fmt(&self, f: &mut String) -> std::fmt::Result {
        use super::inputs::JoltR1CSInputs;

        match self.0 {
            Variable::Input(var_index) => match self.1.abs() {
                1 => write!(f, "{:?}", JoltR1CSInputs::from_index(var_index)),
                _ => write!(f, "{}⋅{:?}", self.1, JoltR1CSInputs::from_index(var_index)),
            },
            Variable::Constant => write!(f, "{}", self.1),
        }
    }
}

/// Linear Combination of terms.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LC(Vec<Term>);

impl LC {
    pub fn evaluate_az_typed<F: JoltField>(
        &self,
        flattened_polynomials: &[MultilinearPolynomial<F>],
        row: usize,
        az_type: AzType,
    ) -> AzValue {
        match az_type {
            AzType::U5 => {
                let mut accumulator: i128 = 0;
                for term in self.terms() {
                    let witness_val: i128 = match term.0 {
                        Variable::Input(var_index) => {
                            let input = JoltR1CSInputs::from_index(var_index);
                            let poly = &flattened_polynomials[var_index];
                            // For U5 constraints, all inputs must be small, fitting in a u8.
                            match input.get_witness_poly_type() {
                                WitnessPolyType::U8 => poly.get_coeff_u8(row) as i128,
                                _ => panic!("Unexpected witness poly type for a U5 constraint. Expected U8."),
                            }
                        }
                        Variable::Constant => term.1,
                    };
                    accumulator += witness_val * term.1;
                }
                debug_assert!(accumulator.abs() <= 31, "AzValue::U5 overflow");
                AzValue::U5(accumulator as i8)
            }
            AzType::U64 => {
                let mut accumulator: i128 = 0;
                for term in self.terms() {
                    let witness_val: i128 = match term.0 {
                        Variable::Input(var_index) => {
                            let input = JoltR1CSInputs::from_index(var_index);
                            let poly = &flattened_polynomials[var_index];
                            match input.get_witness_poly_type() {
                                WitnessPolyType::U8 => poly.get_coeff_u8(row) as i128,
                                WitnessPolyType::U64 => poly.get_coeff_u64(row) as i128,
                                 _ => panic!("Unexpected witness poly type for a U64 constraint. Expected U8 or U64."),
                            }
                        }
                        Variable::Constant => term.1,
                    };
                    accumulator += witness_val * term.1;
                }
                debug_assert!(accumulator >= 0 && accumulator <= u64::MAX as i128, "AzValue::U64 overflow");
                AzValue::U64(accumulator as u64)
            }
            AzType::U64AndSign => {
                // This path is for more complex constraints that can result in a signed 64-bit value.
                // We still use the typed getters for efficiency.
                let mut accumulator: i128 = 0;
                for term in self.terms() {
                    let witness_val: i128 = match term.0 {
                        Variable::Input(var_index) => {
                            let input = JoltR1CSInputs::from_index(var_index);
                            let poly = &flattened_polynomials[var_index];
                            match input.get_witness_poly_type() {
                                WitnessPolyType::U8 => poly.get_coeff_u8(row) as i128,
                                WitnessPolyType::U64 => poly.get_coeff_u64(row) as i128,
                                WitnessPolyType::U64AndSign => {
                                    let u64_and_sign = poly.get_coeff_u64_and_sign(row);
                                    if u64_and_sign.is_positive {
                                        u64_and_sign.magnitude as i128
                                    } else {
                                        -(u64_and_sign.magnitude as i128)
                                    }
                                }
                                 _ => panic!("Unexpected witness poly type for a U64AndSign constraint."),
                            }
                        }
                        Variable::Constant => term.1,
                    };
                    accumulator += witness_val * term.1;
                }
                debug_assert!(accumulator.abs() <= u64::MAX as i128, "AzValue::U64AndSign overflow");
                AzValue::U64AndSign(U64AndSign::from(accumulator))
            }
        }
    }

    pub fn evaluate_bz_typed<F: JoltField>(
        &self,
        _flattened_polynomials: &[MultilinearPolynomial<F>],
        _row: usize,
        _bz_type: BzType,
    ) -> BzValue {
        unimplemented!()
    }

    pub fn evaluate_cz_typed<F: JoltField>(
        &self,
        _flattened_polynomials: &[MultilinearPolynomial<F>],
        _row: usize,
        _cz_type: CzType,
    ) -> CzValue {
        unimplemented!()
    }

    pub fn new(terms: Vec<Term>) -> Self {
        #[cfg(test)]
        Self::assert_no_duplicate_terms(&terms);

        let mut sorted_terms = terms;
        sorted_terms.sort_by(|a, b| a.0.cmp(&b.0));
        LC(sorted_terms)
    }

    pub fn zero() -> Self {
        LC::new(vec![])
    }

    pub fn terms(&self) -> &[Term] {
        &self.0
    }

    pub fn constant_term(&self) -> Option<&Term> {
        self.0
            .last()
            .filter(|term| matches!(term.0, Variable::Constant))
    }

    pub fn constant_term_field(&self) -> i128 {
        if let Some(term) = self.constant_term() {
            term.1
        } else {
            0
        }
    }

    pub fn to_field_elements<F: JoltField>(&self) -> Vec<F> {
        self.terms()
            .iter()
            .map(|term| F::from_i128(term.1))
            .collect()
    }

    pub fn num_terms(&self) -> usize {
        self.0.len()
    }

    pub fn num_vars(&self) -> usize {
        self.0
            .iter()
            .filter(|term| matches!(term.0, Variable::Input(_)))
            .count()
    }

    /// Evaluate the LC for a given row, returning an i128
    /// This is to be used for all Az & Bz computations, besides the one(s) associated with the
    /// product constraint, which may overflow i128
    pub fn evaluate_row_i128<F: JoltField>(
        &self,
        flattened_polynomials: &[MultilinearPolynomial<F>],
        row: usize,
    ) -> i128 {
        self.terms()
            .iter()
            .map(|term| match term.0 {
                Variable::Input(var_index) => {
                    term.1 * flattened_polynomials[var_index].get_coeff_i128(row)
                }
                Variable::Constant => term.1,
            })
            .sum()
    }

    // Evaluate the LC for a given row, returning a field element
    pub fn evaluate_row<F: JoltField>(
        &self,
        flattened_polynomials: &[MultilinearPolynomial<F>],
        row: usize,
    ) -> F {
        self.terms()
            .iter()
            .map(|term| match term.0 {
                Variable::Input(var_index) => flattened_polynomials[var_index]
                    .get_coeff(row)
                    .mul_i128(term.1),
                Variable::Constant => F::from_i128(term.1),
            })
            .sum()
    }

    #[cfg(test)]
    pub fn pretty_fmt(&self, f: &mut String) -> std::fmt::Result {
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
                term.pretty_fmt(f)?;
            }
            if self.0.len() > 1 {
                write!(f, ")")?;
            }
            Ok(())
        }
    }

    #[cfg(test)]
    fn assert_no_duplicate_terms(terms: &[Term]) {
        let mut term_vec = Vec::new();
        for term in terms {
            if term_vec.contains(&term.0) {
                panic!("Duplicate variable found in terms: {:?}", term.0);
            } else {
                term_vec.push(term.0);
            }
        }
    }
}

// Arithmetic for LC

impl<T> std::ops::Add<T> for LC
where
    T: Into<LC>,
{
    type Output = Self;

    fn add(self, other: T) -> Self::Output {
        let other_lc: LC = other.into();
        let mut combined_terms = self.0;
        // TODO(sragss): Can be made more efficient by assuming sorted
        for other_term in other_lc.terms() {
            if let Some(term) = combined_terms
                .iter_mut()
                .find(|term| term.0 == other_term.0)
            {
                term.1 += other_term.1;
            } else {
                combined_terms.push(*other_term);
            }
        }
        LC::new(combined_terms)
    }
}

impl<T> std::ops::Add<T> for Term
where
    T: Into<LC>,
{
    type Output = LC;

    fn add(self, other: T) -> Self::Output {
        let lc: LC = self.into();
        let other_lc: LC = other.into();
        lc + other_lc
    }
}

impl<T> std::ops::Add<T> for Variable
where
    T: Into<LC>,
{
    type Output = LC;

    fn add(self, other: T) -> Self::Output {
        let lc: LC = self.into();
        let other_lc: LC = other.into();
        lc + other_lc
    }
}

impl std::ops::Neg for LC {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let negated_terms: Vec<Term> = self.0.into_iter().map(|term| -term).collect();
        LC::new(negated_terms)
    }
}

impl<T: Into<LC>> std::ops::Sub<T> for LC {
    type Output = Self;

    fn sub(self, other: T) -> Self::Output {
        let other: LC = other.into();
        let negated_other = -other;
        self + negated_other
    }
}

// Arithmetic for Term

impl std::ops::Neg for Term {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Term(self.0, -self.1)
    }
}

impl From<i64> for Term {
    fn from(val: i64) -> Self {
        Term(Variable::Constant, val as i128)
    }
}

impl From<i128> for Term {
    fn from(val: i128) -> Self {
        Term(Variable::Constant, val)
    }
}

impl From<Variable> for Term {
    fn from(val: Variable) -> Self {
        Term(val, 1)
    }
}

impl std::ops::Sub for Variable {
    type Output = LC;

    fn sub(self, other: Self) -> Self::Output {
        let lhs: LC = self.into();
        let rhs: LC = other.into();
        lhs - rhs
    }
}

// Into<LC>

impl From<i64> for LC {
    fn from(val: i64) -> Self {
        LC::new(vec![Term(Variable::Constant, val as i128)])
    }
}

impl From<i128> for LC {
    fn from(val: i128) -> Self {
        LC::new(vec![Term(Variable::Constant, val)])
    }
}

impl From<Variable> for LC {
    fn from(val: Variable) -> Self {
        LC::new(vec![Term(val, 1)])
    }
}

impl From<Term> for LC {
    fn from(val: Term) -> Self {
        LC::new(vec![val])
    }
}

impl From<Vec<Term>> for LC {
    fn from(val: Vec<Term>) -> Self {
        LC::new(val)
    }
}

// Generic arithmetic for Variable

impl std::ops::Mul<i64> for Variable {
    type Output = Term;

    fn mul(self, other: i64) -> Self::Output {
        Term(self, other as i128)
    }
}

impl std::ops::Mul<Variable> for i64 {
    type Output = Term;

    fn mul(self, other: Variable) -> Self::Output {
        Term(other, self as i128)
    }
}

/// Conversions and arithmetic for concrete ConstraintInput
#[macro_export]
macro_rules! impl_r1cs_input_lc_conversions {
    ($ConcreteInput:ty) => {
        impl Into<$crate::zkvm::r1cs::ops::Variable> for $ConcreteInput {
            fn into(self) -> $crate::zkvm::r1cs::ops::Variable {
                $crate::zkvm::r1cs::ops::Variable::Input(self.to_index())
            }
        }

        impl Into<$crate::zkvm::r1cs::ops::Term> for $ConcreteInput {
            fn into(self) -> $crate::zkvm::r1cs::ops::Term {
                $crate::zkvm::r1cs::ops::Term(
                    $crate::zkvm::r1cs::ops::Variable::Input(self.to_index()),
                    1,
                )
            }
        }

        impl Into<$crate::zkvm::r1cs::ops::LC> for $ConcreteInput {
            fn into(self) -> $crate::zkvm::r1cs::ops::LC {
                $crate::zkvm::r1cs::ops::Term(
                    $crate::zkvm::r1cs::ops::Variable::Input(self.to_index()),
                    1,
                )
                .into()
            }
        }

        impl Into<$crate::zkvm::r1cs::ops::LC> for Vec<$ConcreteInput> {
            fn into(self) -> $crate::zkvm::r1cs::ops::LC {
                let terms: Vec<$crate::zkvm::r1cs::ops::Term> =
                    self.into_iter().map(Into::into).collect();
                $crate::zkvm::r1cs::ops::LC::new(terms)
            }
        }

        impl<T: Into<$crate::zkvm::r1cs::ops::LC>> std::ops::Add<T> for $ConcreteInput {
            type Output = $crate::zkvm::r1cs::ops::LC;

            fn add(self, rhs: T) -> Self::Output {
                let lhs_lc: $crate::zkvm::r1cs::ops::LC = self.into();
                let rhs_lc: $crate::zkvm::r1cs::ops::LC = rhs.into();
                lhs_lc + rhs_lc
            }
        }

        impl<T: Into<$crate::zkvm::r1cs::ops::LC>> std::ops::Sub<T> for $ConcreteInput {
            type Output = $crate::zkvm::r1cs::ops::LC;

            fn sub(self, rhs: T) -> Self::Output {
                let lhs_lc: $crate::zkvm::r1cs::ops::LC = self.into();
                let rhs_lc: $crate::zkvm::r1cs::ops::LC = rhs.into();
                lhs_lc - rhs_lc
            }
        }

        impl std::ops::Mul<i64> for $ConcreteInput {
            type Output = $crate::zkvm::r1cs::ops::Term;

            fn mul(self, rhs: i64) -> Self::Output {
                $crate::zkvm::r1cs::ops::Term(
                    $crate::zkvm::r1cs::ops::Variable::Input(self.to_index()),
                    rhs as i128,
                )
            }
        }

        impl std::ops::Mul<$ConcreteInput> for i64 {
            type Output = $crate::zkvm::r1cs::ops::Term;

            fn mul(self, rhs: $ConcreteInput) -> Self::Output {
                $crate::zkvm::r1cs::ops::Term(
                    $crate::zkvm::r1cs::ops::Variable::Input(rhs.to_index()),
                    self as i128,
                )
            }
        }

        impl std::ops::Add<$ConcreteInput> for i64 {
            type Output = $crate::zkvm::r1cs::ops::LC;

            fn add(self, rhs: $ConcreteInput) -> Self::Output {
                let term1 = $crate::zkvm::r1cs::ops::Term(
                    $crate::zkvm::r1cs::ops::Variable::Input(rhs.to_index()),
                    1,
                );
                let term2 = $crate::zkvm::r1cs::ops::Term(
                    $crate::zkvm::r1cs::ops::Variable::Constant,
                    self as i128,
                );
                $crate::zkvm::r1cs::ops::LC::new(vec![term1, term2])
            }
        }

        impl std::ops::Sub<$ConcreteInput> for i64 {
            type Output = $crate::zkvm::r1cs::ops::LC;

            fn sub(self, rhs: $ConcreteInput) -> Self::Output {
                let term1 = $crate::zkvm::r1cs::ops::Term(
                    $crate::zkvm::r1cs::ops::Variable::Input(rhs.to_index()),
                    -1,
                );
                let term2 = $crate::zkvm::r1cs::ops::Term(
                    $crate::zkvm::r1cs::ops::Variable::Constant,
                    self as i128,
                );
                $crate::zkvm::r1cs::ops::LC::new(vec![term1, term2])
            }
        }
    };
}
