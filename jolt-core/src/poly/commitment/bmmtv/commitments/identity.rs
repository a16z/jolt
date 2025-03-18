use ark_ec::CurveGroup;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::ops::Mul;
use std::ops::{Add, MulAssign};

/// Simplest commitment to a Vec<T>, simply send the Vec<T>
///
/// This is just used to fill the gaps in other traits
#[derive(Clone)]
pub enum IdentityCommitment {}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Default, Eq, PartialEq)]
pub struct IdentityOutput<T>(pub Vec<T>)
where
    T: CanonicalSerialize + CanonicalDeserialize + Clone + Default + Eq;

impl<T> Add for IdentityOutput<T>
where
    T: Add<T, Output = T> + CanonicalSerialize + CanonicalDeserialize + Clone + Default + Eq,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        IdentityOutput(
            self.0
                .iter()
                .zip(&rhs.0)
                .map(|(a, b)| a.clone() + b.clone())
                .collect::<Vec<T>>(),
        )
    }
}

impl<T, F> Mul<F> for IdentityOutput<T>
where
    T: MulAssign<F>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Clone
        + Copy
        + Default
        + Eq
        + std::ops::Mul<F, Output = T>,
    F: Clone + Copy,
{
    type Output = IdentityOutput<T>;

    fn mul(self, rhs: F) -> Self::Output {
        IdentityOutput(self.0.iter().map(|a| *a * rhs).collect())
    }
}

impl IdentityCommitment {
    pub fn verify<G: CurveGroup>(a: &[G], b: &IdentityOutput<G>) -> bool {
        b.0 == a
    }
}
