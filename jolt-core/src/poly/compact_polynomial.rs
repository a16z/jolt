use std::ops::Index;

use super::multilinear_polynomial::{BindingOrder, PolynomialBinding};
use crate::utils::math::Math;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::{field::JoltField, utils};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use num_integer::Integer;
use rayon::prelude::*;

pub trait SmallScalar: Copy + Integer + Sync + CanonicalSerialize + CanonicalDeserialize {
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

#[derive(Default, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct CompactPolynomial<T: SmallScalar, F: JoltField> {
    num_vars: usize,
    len: usize,
    pub coeffs: Vec<T>,
    pub bound_coeffs: Vec<F>,
    binding_scratch_space: Option<Vec<F>>,
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
            binding_scratch_space: None,
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

    pub fn coeffs_as_field_elements(&self) -> Vec<F> {
        self.coeffs.par_iter().map(|x| x.to_field()).collect()
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

    #[tracing::instrument(skip_all, name = "CompactPolynomial::bind")]
    fn bind_parallel(&mut self, r: F, order: BindingOrder) {
        let n = self.len() / 2;
        if self.is_bound() {
            match order {
                BindingOrder::LowToHigh => {
                    if self.binding_scratch_space.is_none() {
                        self.binding_scratch_space = Some(unsafe_allocate_zero_vec(n));
                    }
                    let binding_scratch_space = self.binding_scratch_space.as_mut().unwrap();

                    binding_scratch_space
                        .par_iter_mut()
                        .take(n)
                        .enumerate()
                        .for_each(|(i, new_coeff)| {
                            if self.bound_coeffs[2 * i + 1] == self.bound_coeffs[2 * i] {
                                *new_coeff = self.bound_coeffs[2 * i];
                            } else {
                                *new_coeff = self.bound_coeffs[2 * i]
                                    + r * (self.bound_coeffs[2 * i + 1] - self.bound_coeffs[2 * i]);
                            }
                        });
                    std::mem::swap(&mut self.bound_coeffs, binding_scratch_space);
                }
                BindingOrder::HighToLow => {
                    let (left, right) = self.bound_coeffs.split_at_mut(n);
                    left.par_iter_mut()
                        .zip(right.par_iter())
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
                        .into_par_iter()
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
                        .par_iter()
                        .zip(right.par_iter())
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
