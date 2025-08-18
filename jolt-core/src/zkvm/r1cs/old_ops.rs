//! Defines the Linear Combination (LC) object and associated operations.
//! A LinearCombination is a vector of OldTerms, where each OldTerm is a pair of a Variable and a coefficient.

use crate::{field::JoltField, poly::multilinear_polynomial::MultilinearPolynomial};
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
pub struct OldTerm(pub Variable, pub i128);
impl OldTerm {
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

/// Linear Combination of OldTerms.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OldLC(Vec<OldTerm>);

impl OldLC {
    pub fn new(terms: Vec<OldTerm>) -> Self {
        #[cfg(test)]
        Self::assert_no_duplicate_terms(&terms);

        let mut sorted_terms = terms;
        sorted_terms.sort_by(|a, b| a.0.cmp(&b.0));
        OldLC(sorted_terms)
    }

    pub fn zero() -> Self {
        OldLC::new(vec![])
    }

    pub fn terms(&self) -> &[OldTerm] {
        &self.0
    }

    pub fn constant_term(&self) -> Option<&OldTerm> {
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
    fn assert_no_duplicate_terms(terms: &[OldTerm]) {
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

// Arithmetic for OldLC

impl<T> std::ops::Add<T> for OldLC
where
    T: Into<OldLC>,
{
    type Output = Self;

    fn add(self, other: T) -> Self::Output {
        let other_lc: OldLC = other.into();
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
        OldLC::new(combined_terms)
    }
}

impl<T> std::ops::Add<T> for OldTerm
where
    T: Into<OldLC>,
{
    type Output = OldLC;

    fn add(self, other: T) -> Self::Output {
        let lc: OldLC = self.into();
        let other_lc: OldLC = other.into();
        lc + other_lc
    }
}

impl<T> std::ops::Add<T> for Variable
where
    T: Into<OldLC>,
{
    type Output = OldLC;

    fn add(self, other: T) -> Self::Output {
        let lc: OldLC = self.into();
        let other_lc: OldLC = other.into();
        lc + other_lc
    }
}

impl std::ops::Neg for OldLC {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let negated_terms: Vec<OldTerm> = self.0.into_iter().map(|term| -term).collect();
        OldLC::new(negated_terms)
    }
}

impl<T: Into<OldLC>> std::ops::Sub<T> for OldLC {
    type Output = Self;

    fn sub(self, other: T) -> Self::Output {
        let other: OldLC = other.into();
        let negated_other = -other;
        self + negated_other
    }
}

// Arithmetic for OldTerm

impl std::ops::Neg for OldTerm {
    type Output = Self;

    fn neg(self) -> Self::Output {
        OldTerm(self.0, -self.1)
    }
}

impl From<i64> for OldTerm {
    fn from(val: i64) -> Self {
        OldTerm(Variable::Constant, val as i128)
    }
}

impl From<i128> for OldTerm {
    fn from(val: i128) -> Self {
        OldTerm(Variable::Constant, val)
    }
}

impl From<Variable> for OldTerm {
    fn from(val: Variable) -> Self {
        OldTerm(val, 1)
    }
}

impl std::ops::Sub for Variable {
    type Output = OldLC;

    fn sub(self, other: Self) -> Self::Output {
        let lhs: OldLC = self.into();
        let rhs: OldLC = other.into();
        lhs - rhs
    }
}

// Into<OldLC>

impl From<i64> for OldLC {
    fn from(val: i64) -> Self {
        OldLC::new(vec![OldTerm(Variable::Constant, val as i128)])
    }
}

impl From<i128> for OldLC {
    fn from(val: i128) -> Self {
        OldLC::new(vec![OldTerm(Variable::Constant, val)])
    }
}

impl From<Variable> for OldLC {
    fn from(val: Variable) -> Self {
        OldLC::new(vec![OldTerm(val, 1)])
    }
}

impl From<OldTerm> for OldLC {
    fn from(val: OldTerm) -> Self {
        OldLC::new(vec![val])
    }
}

impl From<Vec<OldTerm>> for OldLC {
    fn from(val: Vec<OldTerm>) -> Self {
        OldLC::new(val)
    }
}

// Generic arithmetic for Variable

impl std::ops::Mul<i64> for Variable {
    type Output = OldTerm;

    fn mul(self, other: i64) -> Self::Output {
        OldTerm(self, other as i128)
    }
}

impl std::ops::Mul<Variable> for i64 {
    type Output = OldTerm;

    fn mul(self, other: Variable) -> Self::Output {
        OldTerm(other, self as i128)
    }
}

/// Conversions and arithmetic for concrete ConstraintInput
#[macro_export]
macro_rules! impl_r1cs_input_lc_conversions {
    ($ConcreteInput:ty) => {
        impl Into<$crate::zkvm::r1cs::old_ops::Variable> for $ConcreteInput {
            fn into(self) -> $crate::zkvm::r1cs::old_ops::Variable {
                $crate::zkvm::r1cs::old_ops::Variable::Input(self.to_index())
            }
        }

        impl Into<$crate::zkvm::r1cs::old_ops::OldTerm> for $ConcreteInput {
            fn into(self) -> $crate::zkvm::r1cs::old_ops::OldTerm {
                $crate::zkvm::r1cs::old_ops::OldTerm(
                    $crate::zkvm::r1cs::old_ops::Variable::Input(self.to_index()),
                    1,
                )
            }
        }

        impl Into<$crate::zkvm::r1cs::old_ops::OldLC> for $ConcreteInput {
            fn into(self) -> $crate::zkvm::r1cs::old_ops::OldLC {
                $crate::zkvm::r1cs::old_ops::OldTerm(
                    $crate::zkvm::r1cs::old_ops::Variable::Input(self.to_index()),
                    1,
                )
                .into()
            }
        }

        impl Into<$crate::zkvm::r1cs::old_ops::OldLC> for Vec<$ConcreteInput> {
            fn into(self) -> $crate::zkvm::r1cs::old_ops::OldLC {
                let OldTerms: Vec<$crate::zkvm::r1cs::old_ops::OldTerm> =
                    self.into_iter().map(Into::into).collect();
                $crate::zkvm::r1cs::old_ops::OldLC::new(OldTerms)
            }
        }

        impl<T: Into<$crate::zkvm::r1cs::old_ops::OldLC>> std::ops::Add<T> for $ConcreteInput {
            type Output = $crate::zkvm::r1cs::old_ops::OldLC;

            fn add(self, rhs: T) -> Self::Output {
                let lhs_OldLC: $crate::zkvm::r1cs::old_ops::OldLC = self.into();
                let rhs_OldLC: $crate::zkvm::r1cs::old_ops::OldLC = rhs.into();
                lhs_OldLC + rhs_OldLC
            }
        }

        impl<T: Into<$crate::zkvm::r1cs::old_ops::OldLC>> std::ops::Sub<T> for $ConcreteInput {
            type Output = $crate::zkvm::r1cs::old_ops::OldLC;

            fn sub(self, rhs: T) -> Self::Output {
                let lhs_OldLC: $crate::zkvm::r1cs::old_ops::OldLC = self.into();
                let rhs_OldLC: $crate::zkvm::r1cs::old_ops::OldLC = rhs.into();
                lhs_OldLC - rhs_OldLC
            }
        }

        impl std::ops::Mul<i64> for $ConcreteInput {
            type Output = $crate::zkvm::r1cs::old_ops::OldTerm;

            fn mul(self, rhs: i64) -> Self::Output {
                $crate::zkvm::r1cs::old_ops::OldTerm(
                    $crate::zkvm::r1cs::old_ops::Variable::Input(self.to_index()),
                    rhs as i128,
                )
            }
        }

        impl std::ops::Mul<$ConcreteInput> for i64 {
            type Output = $crate::zkvm::r1cs::old_ops::OldTerm;

            fn mul(self, rhs: $ConcreteInput) -> Self::Output {
                $crate::zkvm::r1cs::old_ops::OldTerm(
                    $crate::zkvm::r1cs::old_ops::Variable::Input(rhs.to_index()),
                    self as i128,
                )
            }
        }

        impl std::ops::Add<$ConcreteInput> for i64 {
            type Output = $crate::zkvm::r1cs::old_ops::OldLC;

            fn add(self, rhs: $ConcreteInput) -> Self::Output {
                let OldTerm1 = $crate::zkvm::r1cs::old_ops::OldTerm(
                    $crate::zkvm::r1cs::old_ops::Variable::Input(rhs.to_index()),
                    1,
                );
                let OldTerm2 = $crate::zkvm::r1cs::old_ops::OldTerm(
                    $crate::zkvm::r1cs::old_ops::Variable::Constant,
                    self as i128,
                );
                $crate::zkvm::r1cs::old_ops::OldLC::new(vec![OldTerm1, OldTerm2])
            }
        }

        impl std::ops::Sub<$ConcreteInput> for i64 {
            type Output = $crate::zkvm::r1cs::old_ops::OldLC;

            fn sub(self, rhs: $ConcreteInput) -> Self::Output {
                let OldTerm1 = $crate::zkvm::r1cs::old_ops::OldTerm(
                    $crate::zkvm::r1cs::old_ops::Variable::Input(rhs.to_index()),
                    -1,
                );
                let OldTerm2 = $crate::zkvm::r1cs::old_ops::OldTerm(
                    $crate::zkvm::r1cs::old_ops::Variable::Constant,
                    self as i128,
                );
                $crate::zkvm::r1cs::old_ops::OldLC::new(vec![OldTerm1, OldTerm2])
            }
        }
    };
}

#[cfg(test)]
mod pretty_fmt_tests {
    use super::super::ops::{Term, LC};

    #[test]
    fn test_const_OldTerm_pretty_fmt() {
        let mut output = String::new();

        // Test coefficient 1
        let OldTerm1 = Term::new(0, 1);
        OldTerm1.pretty_fmt(&mut output).unwrap();
        assert!(output.contains("LeftInstructionInput"));

        // Test coefficient -1
        output.clear();
        let OldTerm2 = Term::new(1, -1);
        OldTerm2.pretty_fmt(&mut output).unwrap();
        assert!(output.contains("-RightInstructionInput"));

        // Test other coefficient
        output.clear();
        let OldTerm3 = Term::new(2, 5);
        OldTerm3.pretty_fmt(&mut output).unwrap();
        assert!(output.contains("5⋅Product"));
    }

    #[test]
    fn test_const_lc_pretty_fmt() {
        let mut output = String::new();

        // Test zero
        let zero = LC::Zero;
        zero.pretty_fmt(&mut output).unwrap();
        assert_eq!(output, "0");

        // Test constant
        output.clear();
        let const_lc = LC::Const(42);
        const_lc.pretty_fmt(&mut output).unwrap();
        assert_eq!(output, "42");

        // Test single OldTerm
        output.clear();
        let single = LC::Terms1([Term::new(0, 1)]);
        single.pretty_fmt(&mut output).unwrap();
        assert!(output.contains("LeftInstructionInput"));

        // Test multiple Terms with parentheses
        output.clear();
        let multi = LC::Terms2([Term::new(0, 1), Term::new(1, -2)]);
        multi.pretty_fmt(&mut output).unwrap();
        assert!(output.starts_with("("));
        assert!(output.ends_with(")"));
        assert!(output.contains("LeftInstructionInput"));
        assert!(output.contains("- 2⋅RightInstructionInput"));

        // Test Terms with constant
        output.clear();
        let with_const = LC::Terms1Const([Term::new(0, 1)], 10);
        with_const.pretty_fmt(&mut output).unwrap();
        assert!(output.contains("LeftInstructionInput"));
        assert!(output.contains("+ 10"));
    }

    #[test]
    fn test_const_lc_assert_no_duplicate_Terms() {
        // This should pass - no duplicates
        let valid = LC::Terms2([Term::new(0, 1), Term::new(1, 2)]);
        valid.assert_no_duplicate_terms(); // Should not panic

        // This should panic - has duplicates
        let invalid = LC::Terms2([
            Term::new(0, 1),
            Term::new(0, 2), // Duplicate index 0
        ]);

        std::panic::catch_unwind(|| {
            invalid.assert_no_duplicate_terms();
        })
        .expect_err("Should have panicked due to duplicate Terms");
    }
}
