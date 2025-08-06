#![allow(
    clippy::len_without_is_empty,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

use crate::jolt::execution_trace::{JoltONNXR1CSInputs, WitnessGenerator};
use jolt_core::r1cs::ops::{LC, Term, Variable};

impl From<JoltONNXR1CSInputs> for Variable {
    fn from(input: JoltONNXR1CSInputs) -> Variable {
        Variable::Input(input.to_index())
    }
}

impl From<JoltONNXR1CSInputs> for Term {
    fn from(input: JoltONNXR1CSInputs) -> Term {
        Term(Variable::Input(input.to_index()), 1)
    }
}

impl From<JoltONNXR1CSInputs> for LC {
    fn from(input: JoltONNXR1CSInputs) -> LC {
        Term(Variable::Input(input.to_index()), 1).into()
    }
}

/// Newtype wrapper to allow conversion from a vector of inputs to LC.
pub struct InputVec(pub Vec<JoltONNXR1CSInputs>);

impl From<InputVec> for LC {
    fn from(input_vec: InputVec) -> LC {
        let terms: Vec<Term> = input_vec.0.into_iter().map(Into::into).collect();
        LC::new(terms)
    }
}

impl<T: Into<LC>> std::ops::Add<T> for JoltONNXR1CSInputs {
    type Output = LC;
    fn add(self, rhs: T) -> Self::Output {
        let lhs_lc: LC = self.into();
        let rhs_lc: LC = rhs.into();
        lhs_lc + rhs_lc
    }
}
impl<T: Into<LC>> std::ops::Sub<T> for JoltONNXR1CSInputs {
    type Output = LC;
    fn sub(self, rhs: T) -> Self::Output {
        let lhs_lc: LC = self.into();
        let rhs_lc: LC = rhs.into();
        lhs_lc - rhs_lc
    }
}
impl std::ops::Mul<i64> for JoltONNXR1CSInputs {
    type Output = Term;
    fn mul(self, rhs: i64) -> Self::Output {
        Term(Variable::Input(self.to_index()), rhs)
    }
}
impl std::ops::Mul<JoltONNXR1CSInputs> for i64 {
    type Output = Term;
    fn mul(self, rhs: JoltONNXR1CSInputs) -> Self::Output {
        Term(Variable::Input(rhs.to_index()), self)
    }
}
impl std::ops::Add<JoltONNXR1CSInputs> for i64 {
    type Output = LC;
    fn add(self, rhs: JoltONNXR1CSInputs) -> Self::Output {
        let term1 = Term(Variable::Input(rhs.to_index()), 1);
        let term2 = Term(Variable::Constant, self);
        LC::new(vec![term1, term2])
    }
}
impl std::ops::Sub<JoltONNXR1CSInputs> for i64 {
    type Output = LC;
    fn sub(self, rhs: JoltONNXR1CSInputs) -> Self::Output {
        let term1 = Term(Variable::Input(rhs.to_index()), -1);
        let term2 = Term(Variable::Constant, self);
        LC::new(vec![term1, term2])
    }
}

#[cfg(test)]
mod tests {
    use crate::jolt::execution_trace::ALL_R1CS_INPUTS;

    use super::*;

    #[test]
    fn from_index_to_index() {
        for i in 0..JoltONNXR1CSInputs::len() {
            assert_eq!(i, JoltONNXR1CSInputs::from_index(i).to_index());
        }
        for var in ALL_R1CS_INPUTS {
            assert_eq!(
                var,
                JoltONNXR1CSInputs::from_index(JoltONNXR1CSInputs::to_index(&var))
            );
        }
    }
}
