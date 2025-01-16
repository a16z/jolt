use std::ops::Index;

use crate::utils::math::Math;
use crate::{field::JoltField, utils};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use num_integer::Integer;

use super::multilinear_polynomial::{BindingOrder, PolynomialBinding};

pub trait SmallScalar: Copy + Integer + Sync {
    /// Performs a field multiplication. Uses `JoltField::mul_u64_unchecked` under the hood.
    /// WARNING: Does not convert the small scalar into Montgomery form before performing
    /// the multiplication, so the other operand should probably have an additional R^2
    /// factor (see `JoltField::montgomery_r2`).
    fn field_mul<F: JoltField>(&self, n: F) -> F;
    /// Converts a small scalar into a (potentially Montgomery form) `JoltField` type
    fn to_field<F: JoltField>(self) -> F;
}

impl SmallScalar for u8 {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        n.mul_u64_unchecked(*self as u64)
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        F::from_u8(self)
    }
}
impl SmallScalar for u16 {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        n.mul_u64_unchecked(*self as u64)
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        F::from_u16(self)
    }
}
impl SmallScalar for u32 {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        n.mul_u64_unchecked(*self as u64)
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        F::from_u32(self)
    }
}
impl SmallScalar for u64 {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        n.mul_u64_unchecked(*self)
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        F::from_u64(self)
    }
}
impl SmallScalar for i64 {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        if self.is_negative() {
            -n.mul_u64_unchecked(-self as u64)
        } else {
            n.mul_u64_unchecked(*self as u64)
        }
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        F::from_i64(self)
    }
}

#[derive(Default, Debug, PartialEq)]
pub struct CompactPolynomial<T: SmallScalar, F: JoltField> {
    num_vars: usize,
    len: usize,
    pub coeffs: Vec<T>,
    pub bound_coeffs: Vec<F>,
}

impl<T: SmallScalar, F: JoltField> Valid for CompactPolynomial<T, F> {
    fn check(&self) -> Result<(), SerializationError> {
        unimplemented!("Unused; needed to satisfy trait bounds for StructuredPolynomialData")
    }
}

impl<T: SmallScalar, F: JoltField> CanonicalDeserialize for CompactPolynomial<T, F> {
    fn deserialize_with_mode<R: std::io::Read>(
        _reader: R,
        _compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        unimplemented!("Unused; needed to satisfy trait bounds for StructuredPolynomialData")
    }
}

impl<T: SmallScalar, F: JoltField> CanonicalSerialize for CompactPolynomial<T, F> {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        _writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        unimplemented!("Unused; needed to satisfy trait bounds for StructuredPolynomialData")
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        unimplemented!("Unused; needed to satisfy trait bounds for StructuredPolynomialData")
    }
}

impl<T: SmallScalar, F: JoltField> CompactPolynomial<T, F> {
    pub fn from_coeffs(coeffs: Vec<T>) -> Self {
        assert!(
            utils::is_power_of_two(coeffs.len()),
            "Multilinear polynomials must be made from a power of 2 (not {})",
            coeffs.len()
        );

        CompactPolynomial {
            num_vars: coeffs.len().log_2(),
            len: coeffs.len(),
            coeffs,
            bound_coeffs: vec![],
        }
    }

    pub fn get_num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.coeffs.iter()
    }
}

impl<T: SmallScalar, F: JoltField> PolynomialBinding<F> for CompactPolynomial<T, F> {
    fn is_bound(&self) -> bool {
        !self.bound_coeffs.is_empty()
    }

    #[tracing::instrument(skip_all, name = "CompactPolynomial::bind")]
    fn bind(&mut self, r: F, order: BindingOrder) {
        let n = self.len() / 2;
        if self.is_bound() {
            match order {
                BindingOrder::LowToHigh => {
                    for i in 0..n {
                        if self.bound_coeffs[2 * i + 1] == self.bound_coeffs[2 * i] {
                            self.bound_coeffs[i] = self.bound_coeffs[2 * i];
                        } else {
                            self.bound_coeffs[i] = self.bound_coeffs[2 * i]
                                + r * (self.bound_coeffs[2 * i + 1] - self.bound_coeffs[2 * i]);
                        }
                    }
                }
                BindingOrder::HighToLow => {
                    let (left, right) = self.bound_coeffs.split_at_mut(n);
                    left.iter_mut()
                        .zip(right.iter())
                        .filter(|(a, b)| a != b)
                        .for_each(|(a, b)| {
                            *a += r * (*b - *a);
                        });
                }
            }
        } else {
            let r_r2 = r * F::montgomery_r2().unwrap_or(F::one());
            let one_minus_r_r2 = (F::one() - r) * F::montgomery_r2().unwrap_or(F::one());
            match order {
                BindingOrder::LowToHigh => {
                    self.bound_coeffs = (0..n)
                        .map(|i| {
                            if self.coeffs[2 * i] == self.coeffs[2 * i + 1] {
                                self.coeffs[2 * i].to_field()
                            } else {
                                self.coeffs[2 * i].field_mul(one_minus_r_r2)
                                    + self.coeffs[2 * i + 1].field_mul(r_r2)
                            }
                        })
                        .collect();
                }
                BindingOrder::HighToLow => {
                    let (left, right) = self.coeffs.split_at(n);
                    self.bound_coeffs = left
                        .iter()
                        .zip(right.iter())
                        .map(|(&a, &b)| {
                            if a == b {
                                a.to_field()
                            } else {
                                a.field_mul(one_minus_r_r2) + b.field_mul(r_r2)
                            }
                        })
                        .collect();
                }
            }
        }
        self.num_vars -= 1;
        self.len = n;
    }

    fn final_sumcheck_claim(&self) -> F {
        assert_eq!(self.len, 1);
        self.bound_coeffs[0]
    }
}

impl<T: SmallScalar, F: JoltField> Clone for CompactPolynomial<T, F> {
    fn clone(&self) -> Self {
        Self::from_coeffs(self.coeffs.to_vec())
    }
}

impl<T: SmallScalar, F: JoltField> Index<usize> for CompactPolynomial<T, F> {
    type Output = T;

    #[inline(always)]
    fn index(&self, _index: usize) -> &T {
        &(self.coeffs[_index])
    }
}
