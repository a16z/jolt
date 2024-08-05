//! Defines the Linear Combination (LC) object and associated operations.
//! A LinearCombination is a vector of Terms, where each Term is a pair of a Variable and a coefficient.

use crate::field::JoltField;
use std::fmt::Debug;
use strum::{EnumCount, IntoEnumIterator};

pub trait ConstraintInput:
    Clone
    + Copy
    + Debug
    + PartialEq
    + Eq
    + PartialOrd
    + Ord
    + IntoEnumIterator
    + EnumCount
    + Into<usize>
    + Sync
    + 'static
{
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Variable<I: ConstraintInput> {
    Input(I),
    Auxiliary(usize),
    Constant,
}

#[derive(Clone, Copy, PartialEq)]
pub struct Term<I: ConstraintInput>(pub Variable<I>, pub i64);

/// Linear Combination of terms.
#[derive(Clone)]
pub struct LC<I: ConstraintInput>(Vec<Term<I>>);

impl<I: ConstraintInput> LC<I> {
    pub fn new(terms: Vec<Term<I>>) -> Self {
        #[cfg(test)]
        Self::assert_no_duplicate_terms(&terms);

        LC(terms)
    }

    pub fn zero() -> Self {
        LC::new(vec![])
    }

    pub fn terms(&self) -> &[Term<I>] {
        &self.0
    }

    pub fn terms_in_field<F: JoltField>(&self) -> Vec<F> {
        self.terms()
            .iter()
            .map(|term| from_i64::<F>(term.1))
            .collect()
    }

    pub fn sorted_terms(&self) -> Vec<Term<I>> {
        let mut sorted_terms = self.0.clone();
        sorted_terms.sort_by(|a, b| a.0.cmp(&b.0));
        sorted_terms
    }

    pub fn num_terms(&self) -> usize {
        self.0.len()
    }

    pub fn num_vars(&self) -> usize {
        self.0
            .iter()
            .filter(|term| matches!(term.0, Variable::Auxiliary(_) | Variable::Input(_)))
            .count()
    }

    /// LC(a) + LC(b) -> LC(a + b)
    pub fn sum2(a: impl Into<Term<I>>, b: impl Into<Term<I>>) -> Self {
        LC(vec![a.into(), b.into()])
    }

    /// LC(a) - LC(b) -> LC(a - b)
    pub fn sub2(a: impl Into<LC<I>>, b: impl Into<LC<I>>) -> Self {
        let a: LC<I> = a.into();
        let b: LC<I> = b.into();

        a - b
    }

    pub fn evaluate<F: JoltField>(&self, values: &[F]) -> F {
        let num_vars = self.num_vars();
        assert_eq!(num_vars, values.len());

        let mut var_index = 0;
        let mut result = F::zero();
        for term in self.terms().iter() {
            match term.0 {
                Variable::Input(_) => {
                    result += values[var_index] * from_i64::<F>(term.1);
                    var_index += 1;
                }
                Variable::Auxiliary(_) => {
                    result += values[var_index] * from_i64::<F>(term.1);
                    var_index += 1;
                }
                Variable::Constant => result += from_i64::<F>(term.1),
            }
        }
        result
    }

    #[cfg(test)]
    fn assert_no_duplicate_terms(terms: &[Term<I>]) {
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

impl<I: ConstraintInput> std::fmt::Debug for LC<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LC(")?;
        for (index, term) in self.0.iter().enumerate() {
            if index > 0 {
                write!(f, " + ")?;
            }
            write!(f, "{:?}", term)?;
        }
        write!(f, ")")
    }
}

impl<I: ConstraintInput> std::fmt::Debug for Term<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}*{:?}", self.1, self.0)
    }
}

// TODO(sragss): Move this onto JoltField
pub fn from_i64<F: JoltField>(val: i64) -> F {
    if val > 0 {
        F::from_u64(val as u64).unwrap()
    } else {
        // TODO(sragss): THIS DOESN'T WORK FOR BINIUS
        F::zero() - F::from_u64(-(val) as u64).unwrap()
    }
}

// Arithmetic for LC

impl<I: ConstraintInput> std::ops::Add for LC<I> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let mut combined_terms = self.0;
        // TODO(sragss): Can be made more efficient by assuming sorted
        for other_term in other.0 {
            if let Some(term) = combined_terms
                .iter_mut()
                .find(|term| term.0 == other_term.0)
            {
                term.1 += other_term.1;
            } else {
                combined_terms.push(other_term);
            }
        }
        LC::new(combined_terms)
    }
}

impl<I: ConstraintInput> std::ops::Neg for LC<I> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let neg_terms = self.0.into_iter().map(|term| -term).collect();
        LC::new(neg_terms)
    }
}

impl<I: ConstraintInput> std::ops::Sub for LC<I> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        let negated_other = -other;
        self + negated_other
    }
}

// Arithmetic for Term<I>

impl<I: ConstraintInput> std::ops::Add for Term<I> {
    type Output = LC<I>;

    fn add(self, other: Self) -> Self::Output {
        if self.0 == other.0 {
            LC::new(vec![Term(self.0, self.1 + other.1)])
        } else {
            LC::new(vec![self, other])
        }
    }
}

impl<I: ConstraintInput> std::ops::Sub for Term<I> {
    type Output = LC<I>;

    fn sub(self, other: Self) -> Self::Output {
        LC::new(vec![self, -other])
    }
}

impl<I: ConstraintInput> std::ops::Neg for Term<I> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Term(self.0, -self.1)
    }
}

impl<I: ConstraintInput> From<i64> for Term<I> {
    fn from(val: i64) -> Self {
        Term(Variable::Constant, val)
    }
}

impl<I: ConstraintInput> From<Variable<I>> for Term<I> {
    fn from(val: Variable<I>) -> Self {
        Term(val, 1)
    }
}

impl<I: ConstraintInput> From<(Variable<I>, i64)> for Term<I> {
    fn from(val: (Variable<I>, i64)) -> Self {
        Term(val.0, val.1)
    }
}

impl<I: ConstraintInput> std::ops::Add for Variable<I> {
    type Output = LC<I>;

    fn add(self, other: Self) -> Self::Output {
        LC::new(vec![Term(self, 1), Term(other, 1)])
    }
}
impl<I: ConstraintInput> std::ops::Sub for Variable<I> {
    type Output = LC<I>;

    fn sub(self, other: Self) -> Self::Output {
        LC::new(vec![Term(self, 1), Term(other, -1)])
    }
}

impl<I: ConstraintInput> std::ops::Add<i64> for LC<I> {
    type Output = Self;

    fn add(self, other: i64) -> Self::Output {
        let lc: LC<I> = other.into();
        self + lc
    }
}

impl<I: ConstraintInput> std::ops::Add<i64> for Term<I> {
    type Output = LC<I>;

    fn add(self, other: i64) -> Self::Output {
        let mut terms = vec![self];
        terms.push(Term(Variable::Constant, other));
        LC::new(terms)
    }
}

impl<I: ConstraintInput> std::ops::Add<Variable<I>> for Term<I> {
    type Output = LC<I>;

    fn add(self, other: Variable<I>) -> Self::Output {
        let terms = vec![self, Term(other, 1)];
        LC::new(terms)
    }
}

// Into<LC<I>>

impl<I: ConstraintInput> From<i64> for LC<I> {
    fn from(val: i64) -> Self {
        LC::new(vec![Term(Variable::Constant, val)])
    }
}

impl<I: ConstraintInput> From<Variable<I>> for LC<I> {
    fn from(val: Variable<I>) -> Self {
        LC::new(vec![Term(val, 1)])
    }
}

impl<I: ConstraintInput> From<Term<I>> for LC<I> {
    fn from(val: Term<I>) -> Self {
        LC::new(vec![val])
    }
}

impl<I: ConstraintInput> From<Vec<Term<I>>> for LC<I> {
    fn from(val: Vec<Term<I>>) -> Self {
        LC::new(val)
    }
}

// Generic arithmetic for Variable<I>

impl<I: ConstraintInput> std::ops::Mul<i64> for Variable<I> {
    type Output = Term<I>;

    fn mul(self, other: i64) -> Self::Output {
        Term(self, other)
    }
}

impl<I: ConstraintInput> std::ops::Mul<Variable<I>> for i64 {
    type Output = Term<I>;

    fn mul(self, other: Variable<I>) -> Self::Output {
        Term(other, self)
    }
}

impl<I: ConstraintInput> std::ops::Add<LC<I>> for Variable<I> {
    type Output = LC<I>;

    fn add(self, other: LC<I>) -> Self::Output {
        let mut terms = other.terms().to_vec();
        terms.push(Term(self, 1));
        LC::new(terms)
    }
}

impl<I: ConstraintInput> std::ops::Add<Variable<I>> for LC<I> {
    type Output = LC<I>;

    fn add(self, other: Variable<I>) -> Self::Output {
        let mut terms = self.terms().to_vec();
        terms.push(Term(other, 1));
        LC::new(terms)
    }
}

/// Conversions and arithmetic for concrete ConstraintInput
#[macro_export]
macro_rules! impl_r1cs_input_lc_conversions {
    ($ConcreteInput:ty) => {
        impl Into<usize> for $ConcreteInput {
            fn into(self) -> usize {
                self as usize
            }
        }
        impl Into<$crate::r1cs::ops::Variable<$ConcreteInput>> for $ConcreteInput {
            fn into(self) -> $crate::r1cs::ops::Variable<$ConcreteInput> {
                $crate::r1cs::ops::Variable::Input(self)
            }
        }

        impl Into<($crate::r1cs::ops::Variable<$ConcreteInput>, i64)> for $ConcreteInput {
            fn into(self) -> ($crate::r1cs::ops::Variable<$ConcreteInput>, i64) {
                ($crate::r1cs::ops::Variable::Input(self), 1)
            }
        }
        impl Into<$crate::r1cs::ops::Term<$ConcreteInput>> for $ConcreteInput {
            fn into(self) -> $crate::r1cs::ops::Term<$ConcreteInput> {
                $crate::r1cs::ops::Term($crate::r1cs::ops::Variable::Input(self), 1)
            }
        }

        impl Into<$crate::r1cs::ops::Term<$ConcreteInput>> for ($ConcreteInput, i64) {
            fn into(self) -> $crate::r1cs::ops::Term<$ConcreteInput> {
                $crate::r1cs::ops::Term($crate::r1cs::ops::Variable::Input(self.0), self.1)
            }
        }

        impl Into<$crate::r1cs::ops::LC<$ConcreteInput>> for $ConcreteInput {
            fn into(self) -> $crate::r1cs::ops::LC<$ConcreteInput> {
                $crate::r1cs::ops::Term($crate::r1cs::ops::Variable::Input(self), 1).into()
            }
        }

        impl Into<$crate::r1cs::ops::LC<$ConcreteInput>> for Vec<$ConcreteInput> {
            fn into(self) -> $crate::r1cs::ops::LC<$ConcreteInput> {
                let terms: Vec<$crate::r1cs::ops::Term<$ConcreteInput>> =
                    self.into_iter().map(Into::into).collect();
                $crate::r1cs::ops::LC::new(terms)
            }
        }

        impl std::ops::Add for $ConcreteInput {
            type Output = $crate::r1cs::ops::LC<$ConcreteInput>;

            fn add(self, other: Self) -> Self::Output {
                $crate::r1cs::ops::LC::sum2(self, other)
            }
        }

        impl std::ops::Add<$ConcreteInput> for $crate::r1cs::ops::Term<$ConcreteInput> {
            type Output = $crate::r1cs::ops::LC<$ConcreteInput>;

            fn add(self, other: $ConcreteInput) -> Self::Output {
                let other_term: $crate::r1cs::ops::Term<$ConcreteInput> = other.into();
                $crate::r1cs::ops::LC::sum2(self, other_term)
            }
        }

        impl std::ops::Add<$crate::r1cs::ops::Term<$ConcreteInput>> for $ConcreteInput {
            type Output = $crate::r1cs::ops::LC<$ConcreteInput>;

            fn add(self, other: $crate::r1cs::ops::Term<$ConcreteInput>) -> Self::Output {
                other + self
            }
        }

        impl std::ops::Add<$ConcreteInput> for $crate::r1cs::ops::LC<$ConcreteInput> {
            type Output = $crate::r1cs::ops::LC<$ConcreteInput>;

            fn add(self, other: $ConcreteInput) -> Self::Output {
                let other_term: $crate::r1cs::ops::Term<$ConcreteInput> = other.into();
                let mut combined_terms: Vec<$crate::r1cs::ops::Term<$ConcreteInput>> =
                    self.terms().to_vec();
                combined_terms.push(other_term);
                $crate::r1cs::ops::LC::new(combined_terms)
            }
        }

        impl std::ops::Mul<i64> for $ConcreteInput {
            type Output = $crate::r1cs::ops::Term<$ConcreteInput>;

            fn mul(self, rhs: i64) -> Self::Output {
                $crate::r1cs::ops::Term($crate::r1cs::ops::Variable::Input(self), rhs)
            }
        }
        impl<T: Into<$crate::r1cs::ops::LC<$ConcreteInput>>> std::ops::Sub<T> for $ConcreteInput {
            type Output = $crate::r1cs::ops::LC<$ConcreteInput>;

            fn sub(self, rhs: T) -> Self::Output {
                let lhs_lc: $crate::r1cs::ops::LC<$ConcreteInput> = self.into();
                let rhs_lc: $crate::r1cs::ops::LC<$ConcreteInput> = rhs.into();
                lhs_lc - rhs_lc
            }
        }

        impl std::ops::Mul<$ConcreteInput> for i64 {
            type Output = $crate::r1cs::ops::Term<$ConcreteInput>;

            fn mul(self, rhs: $ConcreteInput) -> Self::Output {
                $crate::r1cs::ops::Term($crate::r1cs::ops::Variable::Input(rhs), self)
            }
        }

        impl std::ops::Add<i64> for $ConcreteInput {
            type Output = $crate::r1cs::ops::LC<$ConcreteInput>;

            fn add(self, rhs: i64) -> Self::Output {
                let term1 = $crate::r1cs::ops::Term($crate::r1cs::ops::Variable::Input(self), 1);
                let term2 = $crate::r1cs::ops::Term($crate::r1cs::ops::Variable::Constant, rhs);
                $crate::r1cs::ops::LC::new(vec![term1, term2])
            }
        }

        impl std::ops::Add<$ConcreteInput> for i64 {
            type Output = $crate::r1cs::ops::LC<$ConcreteInput>;

            fn add(self, rhs: $ConcreteInput) -> Self::Output {
                let term1 = $crate::r1cs::ops::Term($crate::r1cs::ops::Variable::Input(rhs), 1);
                let term2 = $crate::r1cs::ops::Term($crate::r1cs::ops::Variable::Constant, self);
                $crate::r1cs::ops::LC::new(vec![term1, term2])
            }
        }
    };
}

/// ```rust
/// use jolt_core::enum_range;
/// #[derive(Debug, PartialEq, Clone, Copy)]
/// #[repr(usize)]
/// pub enum Ex {
///     A,
///     B,
///     C,
///     D
/// }
///
///
/// let range = enum_range!(Ex::B, Ex::D);
/// assert_eq!(range, [Ex::B, Ex::C, Ex::D]);
/// ```
#[macro_export]
macro_rules! enum_range {
    ($start:path, $end:path) => {{
        let mut arr = [$start; ($end as usize) - ($start as usize) + 1];
        for i in ($start as usize)..=($end as usize) {
            arr[i - ($start as usize)] = i.into();
        }
        arr
    }};
}

/// ```rust
/// use jolt_core::input_range;
/// use jolt_core::r1cs::ops::{ConstraintInput, Variable};
/// # use strum_macros::{EnumCount, EnumIter};
///
/// # #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter)]
/// #[repr(usize)]
/// pub enum Inputs {
///     A,
///     B,
///     C,
///     D
/// }
/// #
/// # impl Into<usize> for Inputs {
/// #   fn into(self) -> usize {
/// #       self as usize
/// #   }
/// # }
/// #
/// impl ConstraintInput for Inputs {};
///
/// let range = input_range!(Inputs::B, Inputs::D);
/// let expected_range = [Variable::Input(Inputs::B), Variable::Input(Inputs::C), Variable::Input(Inputs::D)];
/// assert_eq!(range, expected_range);
/// ```
#[macro_export]
macro_rules! input_range {
    ($start:path, $end:path) => {{
        let mut arr = [Variable::Input($start); ($end as usize) - ($start as usize) + 1];
        #[allow(clippy::missing_transmute_annotations)]
        for i in ($start as usize)..=($end as usize) {
            arr[i - ($start as usize)] =
                Variable::Input(unsafe { std::mem::transmute::<usize, _>(i) });
        }
        arr
    }};
    ($start:path, $end:path) => {{
        let mut arr =
            [$crate::r1cs::ops::Variable::Input($start); ($end as usize) - ($start as usize) + 1];
        #[allow(clippy::missing_transmute_annotations)]
        for i in ($start as usize)..=($end as usize) {
            #[allow(clippy::missing_transmute_annotations)]
            arr[i - ($start as usize)] =
                Variable::Input(unsafe { std::mem::transmute::<usize, _>(i) });
        }
        arr
    }};
}

/// Used to fix an aux variable to a specific index to ensure the aux index can be used elsewhere statically.
#[macro_export]
macro_rules! assert_static_aux_index {
    ($var:expr, $index:expr) => {{
        if let Variable::Auxiliary(aux_index) = $var {
            assert_eq!(
                aux_index, $index,
                "Unexpected auxiliary index {:?}",
                aux_index
            );
        } else {
            panic!("Variable is not of variant type Variable::Auxiliary");
        }
    }};
}

#[cfg(test)]
mod test {
    use strum_macros::{EnumCount, EnumIter};

    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter)]
    #[repr(usize)]
    enum Inputs {
        A,
        B,
        C,
        D,
    }

    impl From<Inputs> for usize {
        fn from(val: Inputs) -> Self {
            val as usize
        }
    }
    impl ConstraintInput for Inputs {}

    #[test]
    fn variable_ordering() {
        let mut variables: Vec<Variable<Inputs>> = vec![
            Variable::Auxiliary(10),
            Variable::Auxiliary(5),
            Variable::Constant,
            Variable::Input(Inputs::C),
            Variable::Input(Inputs::B),
        ];
        let expected_sort: Vec<Variable<Inputs>> = vec![
            Variable::Input(Inputs::B),
            Variable::Input(Inputs::C),
            Variable::Auxiliary(5),
            Variable::Auxiliary(10),
            Variable::Constant,
        ];
        variables.sort();
        assert_eq!(variables, expected_sort);
    }

    #[test]
    fn lc_sorting() {
        let variables: Vec<Variable<Inputs>> = vec![
            Variable::Auxiliary(10),
            Variable::Auxiliary(5),
            Variable::Constant,
            Variable::Input(Inputs::C),
            Variable::Input(Inputs::B),
        ];

        let expected_sort: Vec<Variable<Inputs>> = vec![
            Variable::Input(Inputs::B),
            Variable::Input(Inputs::C),
            Variable::Auxiliary(5),
            Variable::Auxiliary(10),
            Variable::Constant,
        ];
        let expected_sorted_terms: Vec<Term<Inputs>> = expected_sort
            .into_iter()
            .map(|variable| variable.into())
            .collect();

        let terms = variables
            .into_iter()
            .map(|variable| variable.into())
            .collect();
        let lc = LC::new(terms);
        assert_eq!(lc.sorted_terms(), expected_sorted_terms);
    }
}
