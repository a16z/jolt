use std::{
    marker::PhantomData,
    ops::{Add, MulAssign},
};

use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use eyre::Error;
use rand::Rng;

use super::HomomorphicPlaceholderValue;

#[derive(Clone)]
pub struct IdentityCommitment<T, F: PrimeField> {
    _t: PhantomData<T>,
    _field: PhantomData<F>,
}

impl<T, F> IdentityCommitment<T, F>
where
    T: CanonicalSerialize
        + CanonicalDeserialize
        + Clone
        + Default
        + Eq
        + Add<T, Output = T>
        + MulAssign<F>
        + Send
        + Sync,
    F: PrimeField,
{
    // type Scalar = F;
    // type Message = T;
    // type Key = HomomorphicPlaceholderValue;
    // type Output = IdentityOutput<T>;

    fn setup<R: Rng>(_rng: &mut R, size: usize) -> Result<Vec<HomomorphicPlaceholderValue>, Error> {
        Ok(vec![HomomorphicPlaceholderValue {}; size])
    }

    fn commit(_k: &[HomomorphicPlaceholderValue], m: &[T]) -> Result<IdentityOutput<T>, Error> {
        Ok(IdentityOutput(m.to_vec()))
    }

    fn verify(
        k: &[HomomorphicPlaceholderValue],
        m: &[T],
        com: &IdentityOutput<T>,
    ) -> Result<bool, Error> {
        Ok(Self::commit(k, m)? == *com)
    }
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
