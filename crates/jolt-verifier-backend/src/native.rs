use std::marker::PhantomData;

use jolt_field::Field;

use crate::backend::{FieldBackend, ScalarOrigin};
use crate::error::BackendError;

/// Zero-overhead [`FieldBackend`] backed by the underlying field directly.
///
/// `Scalar = F`. Every method is `#[inline(always)]` and forwards to a single
/// field operator, so monomorphization erases the trait machinery and the
/// generated code is byte-identical to handwritten `F` arithmetic.
///
/// Use this for production verification where the verifier is executed in
/// the clear, on real hardware, with no recording or constraint generation.
#[derive(Copy, Clone, Debug, Default)]
pub struct Native<F: Field>(PhantomData<F>);

impl<F: Field> Native<F> {
    /// Constructs a new `Native` backend.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<F: Field> FieldBackend for Native<F> {
    type F = F;
    type Scalar = F;

    #[inline(always)]
    fn wrap(&mut self, value: F, _origin: ScalarOrigin, _label: &'static str) -> F {
        value
    }

    #[inline(always)]
    fn const_i128(&mut self, v: i128) -> F {
        F::from_i128(v)
    }

    #[inline(always)]
    fn const_zero(&mut self) -> F {
        F::zero()
    }

    #[inline(always)]
    fn const_one(&mut self) -> F {
        F::one()
    }

    #[inline(always)]
    fn add(&mut self, a: &F, b: &F) -> F {
        *a + *b
    }

    #[inline(always)]
    fn sub(&mut self, a: &F, b: &F) -> F {
        *a - *b
    }

    #[inline(always)]
    fn mul(&mut self, a: &F, b: &F) -> F {
        *a * *b
    }

    #[inline(always)]
    fn neg(&mut self, a: &F) -> F {
        -*a
    }

    #[inline(always)]
    fn square(&mut self, a: &F) -> F {
        a.square()
    }

    #[inline(always)]
    fn inverse(&mut self, a: &F, ctx: &'static str) -> Result<F, BackendError> {
        a.inverse().ok_or(BackendError::InverseOfZero(ctx))
    }

    #[inline(always)]
    fn assert_eq(&mut self, a: &F, b: &F, ctx: &'static str) -> Result<(), BackendError> {
        if a == b {
            Ok(())
        } else {
            Err(BackendError::AssertionFailed(ctx))
        }
    }

    #[inline(always)]
    fn unwrap(&self, scalar: &F) -> Option<F> {
        Some(*scalar)
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use super::*;
    use jolt_field::Fr;

    #[test]
    fn native_arithmetic_roundtrip() {
        let mut b = Native::<Fr>::new();
        let two = b.const_i128(2);
        let three = b.const_i128(3);
        let six = b.mul(&two, &three);
        let expected = b.const_i128(6);
        b.assert_eq(&six, &expected, "2*3==6").unwrap();
    }

    #[test]
    fn native_assert_eq_failure_returns_error() {
        let mut b = Native::<Fr>::new();
        let a = b.const_i128(2);
        let bv = b.const_i128(3);
        let err = b.assert_eq(&a, &bv, "ne").unwrap_err();
        assert!(matches!(err, BackendError::AssertionFailed("ne")));
    }

    #[test]
    fn native_inverse_of_zero_errors() {
        let mut b = Native::<Fr>::new();
        let z = b.const_zero();
        let err = b.inverse(&z, "z").unwrap_err();
        assert!(matches!(err, BackendError::InverseOfZero("z")));
    }

    #[test]
    fn native_square_matches_mul() {
        let mut b = Native::<Fr>::new();
        let v = b.wrap_proof(Fr::from_u64(7), "x");
        let s = b.square(&v);
        let m = b.mul(&v, &v);
        b.assert_eq(&s, &m, "square==mul").unwrap();
    }
}
