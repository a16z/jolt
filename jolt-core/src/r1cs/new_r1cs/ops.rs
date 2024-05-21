use crate::poly::field::JoltField;
/// Defines the Linear Combination (LC) object and associated operations.
/// A LinearCombination is a vector of Terms, where each Term is a pair of a Variable and a coefficient.
use std::fmt::Debug;
use strum::{EnumCount, IntoEnumIterator};

pub trait ConstraintInput:
    Clone + Copy + Debug + PartialEq + IntoEnumIterator + EnumCount + Into<usize> + 'static
{
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Variable<I: ConstraintInput> {
    Input(I),
    Auxiliary(usize),
    Constant,
}

#[derive(Clone, Copy, Debug)]
pub struct Term<I: ConstraintInput>(pub Variable<I>, pub i64);

/// Linear Combination of terms.
#[derive(Clone, Debug)]
pub struct LC<I: ConstraintInput>(Vec<Term<I>>);

impl<I: ConstraintInput> LC<I> {
    pub fn new(terms: Vec<Term<I>>) -> Self {
        LC(terms)
    }

    pub fn zero() -> Self {
        LC(vec![])
    }

    pub fn terms(&self) -> &[Term<I>] {
        &self.0
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
        for (term_index, term) in self.terms().iter().enumerate() {
            match term.0 {
                Variable::Input(_) => {
                    result += values[var_index] * from_i64::<F>(term.1);
                    var_index += 1;
                }
                Variable::Auxiliary(_) => {
                    result += values[term_index] * from_i64::<F>(term.1);
                    var_index += 1;
                }
                Variable::Constant => result += from_i64::<F>(term.1),
            }
        }
        result
    }
}

// TODO(sragss): Move this onto JoltFiel
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
        combined_terms.extend(other.0);
        LC(combined_terms)
    }
}

impl<I: ConstraintInput> std::ops::Neg for LC<I> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let neg_terms = self.0.into_iter().map(|term| -term).collect();
        LC(neg_terms)
    }
}

impl<I: ConstraintInput> std::ops::Sub for LC<I> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        let mut combined_terms = self.0;
        combined_terms.extend((-other).0);
        LC(combined_terms)
    }
}

// Arithmetic for Term<I>

impl<I: ConstraintInput> std::ops::Add for Term<I> {
    type Output = LC<I>;

    fn add(self, other: Self) -> Self::Output {
        LC(vec![self, other])
    }
}

impl<I: ConstraintInput> std::ops::Sub for Term<I> {
    type Output = LC<I>;

    fn sub(self, other: Self) -> Self::Output {
        LC(vec![self, -other])
    }
}

impl<I: ConstraintInput> std::ops::Neg for Term<I> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Term(self.0, self.1 * -1)
    }
}

impl<I: ConstraintInput> Into<Term<I>> for i64 {
    fn into(self) -> Term<I> {
        Term(Variable::Constant, self)
    }
}

impl<I: ConstraintInput> Into<Term<I>> for Variable<I> {
    fn into(self) -> Term<I> {
        Term(self, 1)
    }
}

impl<I: ConstraintInput> Into<Term<I>> for (Variable<I>, i64) {
    fn into(self) -> Term<I> {
        Term(self.0, self.1)
    }
}

impl<I: ConstraintInput> std::ops::Add for Variable<I> {
    type Output = LC<I>;

    fn add(self, other: Self) -> Self::Output {
        LC(vec![Term(self, 1), Term(other, 1)])
    }
}
impl<I: ConstraintInput> std::ops::Sub for Variable<I> {
    type Output = LC<I>;

    fn sub(self, other: Self) -> Self::Output {
        LC(vec![Term(self, 1), Term(other, -1)])
    }
}

// Into<LC<I>>

impl<I: ConstraintInput> Into<LC<I>> for i64 {
    fn into(self) -> LC<I> {
        LC(vec![Term(Variable::Constant, self)])
    }
}

impl<I: ConstraintInput> Into<LC<I>> for Variable<I> {
    fn into(self) -> LC<I> {
        LC(vec![Term(self, 1)])
    }
}

impl<I: ConstraintInput> Into<LC<I>> for Term<I> {
    fn into(self) -> LC<I> {
        LC(vec![self])
    }
}

impl<I: ConstraintInput> Into<LC<I>> for Vec<Term<I>> {
    fn into(self) -> LC<I> {
        LC(self)
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

/// Conversions and arithmetic for concrete ConstraintInput
#[macro_export]
macro_rules! impl_r1cs_input_lc_conversions {
    ($ConcreteInput:ty) => {
        impl Into<usize> for $ConcreteInput {
            fn into(self) -> usize {
                self as usize
            }
        }
        impl Into<crate::r1cs::new_r1cs::ops::Variable<$ConcreteInput>> for $ConcreteInput {
            fn into(self) -> crate::r1cs::new_r1cs::ops::Variable<$ConcreteInput> {
                crate::r1cs::new_r1cs::ops::Variable::Input(self)
            }
        }

        impl Into<(crate::r1cs::new_r1cs::ops::Variable<$ConcreteInput>, i64)> for $ConcreteInput {
            fn into(self) -> (crate::r1cs::new_r1cs::ops::Variable<$ConcreteInput>, i64) {
                (crate::r1cs::new_r1cs::ops::Variable::Input(self), 1)
            }
        }
        impl Into<crate::r1cs::new_r1cs::ops::Term<$ConcreteInput>> for $ConcreteInput {
            fn into(self) -> crate::r1cs::new_r1cs::ops::Term<$ConcreteInput> {
                crate::r1cs::new_r1cs::ops::Term(
                    crate::r1cs::new_r1cs::ops::Variable::Input(self),
                    1,
                )
            }
        }

        impl Into<crate::r1cs::new_r1cs::ops::Term<$ConcreteInput>> for ($ConcreteInput, i64) {
            fn into(self) -> crate::r1cs::new_r1cs::ops::Term<$ConcreteInput> {
                crate::r1cs::new_r1cs::ops::Term(
                    crate::r1cs::new_r1cs::ops::Variable::Input(self.0),
                    self.1,
                )
            }
        }

        impl Into<LC<$ConcreteInput>> for $ConcreteInput {
            fn into(self) -> LC<$ConcreteInput> {
                crate::r1cs::new_r1cs::ops::Term(
                    crate::r1cs::new_r1cs::ops::Variable::Input(self),
                    1,
                )
                .into()
            }
        }

        impl Into<LC<$ConcreteInput>> for Vec<$ConcreteInput> {
            fn into(self) -> LC<$ConcreteInput> {
                let terms: Vec<crate::r1cs::new_r1cs::ops::Term<$ConcreteInput>> =
                    self.into_iter().map(Into::into).collect();
                LC::new(terms)
            }
        }

        impl std::ops::Mul<i64> for $ConcreteInput {
            type Output = crate::r1cs::new_r1cs::ops::Term<$ConcreteInput>;

            fn mul(self, rhs: i64) -> Self::Output {
                crate::r1cs::new_r1cs::ops::Term(
                    crate::r1cs::new_r1cs::ops::Variable::Input(self),
                    rhs,
                )
            }
        }

        impl std::ops::Mul<$ConcreteInput> for i64 {
            type Output = crate::r1cs::new_r1cs::ops::Term<$ConcreteInput>;

            fn mul(self, rhs: $ConcreteInput) -> Self::Output {
                crate::r1cs::new_r1cs::ops::Term(
                    crate::r1cs::new_r1cs::ops::Variable::Input(rhs),
                    self,
                )
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
            arr[i - ($start as usize)] = unsafe { std::mem::transmute::<usize, _>(i) };
        }
        arr
    }};
}

/// ```rust
/// use jolt_core::input_range;
/// use jolt_core::r1cs::new_r1cs::ops::{ConstraintInput, Variable};
/// # use strum_macros::{EnumCount, EnumIter};
///
/// # #[derive(Clone, Copy, Debug, PartialEq, EnumCount, EnumIter)]
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
        for i in ($start as usize)..=($end as usize) {
            arr[i - ($start as usize)] =
                Variable::Input(unsafe { std::mem::transmute::<usize, _>(i) });
        }
        arr
    }};
    ($start:path, $end:path) => {{
        let mut arr = [crate::r1cs::new_r1cs::ops::Variable::Input($start);
            ($end as usize) - ($start as usize) + 1];
        for i in ($start as usize)..=($end as usize) {
            arr[i - ($start as usize)] =
                Variable::Input(unsafe { std::mem::transmute::<usize, _>(i) });
        }
        arr
    }};
}
