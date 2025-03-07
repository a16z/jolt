use ark_ff::fields::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::Rng;
use std::{
    marker::PhantomData,
    ops::{Add, MulAssign},
};

use super::{Dhc, Error};

/// Simplest commitment to a Vec<T>, simply send the Vec<T>
///
/// This is just used to fill the gaps in other traits
#[derive(Clone)]
pub struct IdentityCommitment<T, F: PrimeField> {
    _t: PhantomData<T>,
    _field: PhantomData<F>,
}

/// Dummy Key param used when you don't need any setup output like [`IdentityCommitment`] does
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Default, Eq, PartialEq)]
pub struct DummyParam;

impl Add for DummyParam {
    type Output = Self;

    fn add(self, _rhs: Self) -> Self::Output {
        DummyParam
    }
}

impl<T> MulAssign<T> for DummyParam {
    fn mul_assign(&mut self, _rhs: T) {}
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

impl<T, F> Dhc for IdentityCommitment<T, F>
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
    type Scalar = F;
    type Message = T;
    type Param = DummyParam;
    type Output = IdentityOutput<T>;

    fn setup<R: Rng>(_rng: &mut R, size: usize) -> Result<Vec<Self::Param>, Error> {
        Ok(vec![DummyParam; size])
    }

    fn commit(_k: &[Self::Param], m: &[Self::Message]) -> Result<Self::Output, Error> {
        Ok(IdentityOutput(m.to_vec()))
    }
}
