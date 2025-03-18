use ark_ec::pairing::Pairing;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::{
    marker::PhantomData,
    ops::{Add, MulAssign},
};

/// Simplest commitment to a Vec<T>, simply send the Vec<T>
///
/// This is just used to fill the gaps in other traits
#[derive(Clone)]
pub struct IdentityCommitment<P> {
    _pairing: PhantomData<P>,
}

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

impl<T, F> MulAssign<F> for IdentityOutput<T>
where
    T: MulAssign<F> + CanonicalSerialize + CanonicalDeserialize + Clone + Default + Eq,
    F: Clone,
{
    fn mul_assign(&mut self, rhs: F) {
        self.0.iter_mut().for_each(|a| a.mul_assign(rhs.clone()))
    }
}

impl<P: Pairing> IdentityCommitment<P> {
    pub fn verify(a: &[P::G1], b: &IdentityOutput<P::G1>) -> bool {
        b.0 == a
    }
}
