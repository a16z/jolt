use std::{
    iter::Sum,
    ops::{Add, AddAssign, Mul, MulAssign, Sub},
};

use jolt_lookup_tables::{tables::and::AndTable, LookupEval, LookupTable};

/// A small symbolic multilinear polynomial used to verify that lookup
/// evaluation does not require a cryptographic field implementation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Symbolic([i128; 16]);

impl Symbolic {
    fn variable(index: usize) -> Self {
        let mut coefficients = [0; 16];
        coefficients[1 << index] = 1;
        Self(coefficients)
    }

    fn add_ref(mut self, rhs: &Self) -> Self {
        for (left, right) in self.0.iter_mut().zip(rhs.0) {
            *left += right;
        }
        self
    }

    fn sub_ref(mut self, rhs: &Self) -> Self {
        for (left, right) in self.0.iter_mut().zip(rhs.0) {
            *left -= right;
        }
        self
    }

    fn mul_ref(self, rhs: &Self) -> Self {
        let mut coefficients = [0; 16];
        for (left_mask, left) in self.0.into_iter().enumerate() {
            for (right_mask, right) in rhs.0.into_iter().enumerate() {
                coefficients[left_mask | right_mask] += left * right;
            }
        }
        Self(coefficients)
    }
}

impl LookupEval for Symbolic {
    fn zero() -> Self {
        Self([0; 16])
    }

    fn one() -> Self {
        Self::from_u64(1)
    }

    fn from_u64(value: u64) -> Self {
        let mut coefficients = [0; 16];
        coefficients[0] = value.into();
        Self(coefficients)
    }

    fn from_u128(value: u128) -> Self {
        let mut coefficients = [0; 16];
        coefficients[0] = value as i128;
        Self(coefficients)
    }
}

impl Add for Symbolic {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.add_ref(&rhs)
    }
}

impl Add<&Self> for Symbolic {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        self.add_ref(rhs)
    }
}

impl AddAssign for Symbolic {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.add_ref(&rhs);
    }
}

impl Sub for Symbolic {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub_ref(&rhs)
    }
}

impl Sub<&Self> for Symbolic {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        self.sub_ref(rhs)
    }
}

impl Mul for Symbolic {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_ref(&rhs)
    }
}

impl Mul<&Self> for Symbolic {
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        self.mul_ref(rhs)
    }
}

impl MulAssign for Symbolic {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_ref(&rhs);
    }
}

impl Sum for Symbolic {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |sum, value| sum + value)
    }
}

#[test]
fn and_evaluate_mle_accepts_a_symbolic_algebra() {
    let variables = [
        Symbolic::variable(0),
        Symbolic::variable(1),
        Symbolic::variable(2),
        Symbolic::variable(3),
    ];

    let result = AndTable::<2>.evaluate_mle::<Symbolic, Symbolic>(&variables);

    let mut expected = [0; 16];
    expected[0b0011] = 2;
    expected[0b1100] = 1;
    assert_eq!(result, Symbolic(expected));
}
