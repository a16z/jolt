use super::multilinear_polynomial::{BindingOrder, PolynomialBinding};
use crate::field::{JoltField, OptimizedMul};
use crate::utils::math::Math;
use crate::utils::small_scalar::SmallScalar;
use allocative::Allocative;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::ops::Index;

/// Compact polynomials are used to store coefficients of small scalars.
/// They have two representations:
/// 1. `coeffs` is a vector of small scalars
/// 2. `bound_coeffs` is a vector of field elements (e.g. big scalars)
///
/// They are often initialized with `coeffs` and then converted to `bound_coeffs`
/// when binding the polynomial.
#[derive(
    Clone, Default, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize, Allocative,
)]
pub struct CompactPolynomial<T: SmallScalar, F: JoltField> {
    num_vars: usize,
    len: usize,
    pub coeffs: Vec<T>,
    pub bound_coeffs: Vec<F>,
}

impl<T: SmallScalar, F: JoltField> CompactPolynomial<T, F> {
    pub fn from_coeffs(coeffs: Vec<T>) -> Self {
        assert!(
            coeffs.len().is_power_of_two(),
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

    pub fn coeffs_as_field_elements(&self) -> Vec<F> {
        self.coeffs.par_iter().map(|x| x.to_field()).collect()
    }

    pub fn split_eq_evaluate(&self, r_len: usize, eq_one: &[F], eq_two: &[F]) -> F {
        const PARALLEL_THRESHOLD: usize = 16;
        if r_len < PARALLEL_THRESHOLD {
            self.evaluate_split_eq_serial(eq_one, eq_two)
        } else {
            self.evaluate_split_eq_parallel(eq_one, eq_two)
        }
    }

    fn evaluate_split_eq_parallel(&self, eq_one: &[F], eq_two: &[F]) -> F {
        let eval: F = (0..eq_one.len())
            .into_par_iter()
            .map(|x1| {
                let partial_sum = (0..eq_two.len())
                    .into_par_iter()
                    .map(|x2| {
                        let idx = x1 * eq_two.len() + x2;
                        // field_mul now already checks for 0 and 1 optimisation
                        // via Jolfield mul_64 method
                        self.coeffs[idx].field_mul(eq_two[x2])
                    })
                    .reduce(|| F::zero(), |acc, val| acc + val);
                OptimizedMul::mul_01_optimized(partial_sum, eq_one[x1])
            })
            .reduce(|| F::zero(), |acc, val| acc + val);
        eval
    }
    fn evaluate_split_eq_serial(&self, eq_one: &[F], eq_two: &[F]) -> F {
        let eval: F = (0..eq_one.len())
            .map(|x1| {
                let partial_sum = (0..eq_two.len())
                    .map(|x2| {
                        let idx = x1 * eq_two.len() + x2;
                        self.coeffs[idx].field_mul(eq_two[x2])
                    })
                    .fold(F::zero(), |acc, val| acc + val);
                OptimizedMul::mul_01_optimized(partial_sum, eq_one[x1])
            })
            .fold(F::zero(), |acc, val| acc + val);
        eval
    }

    // Faster evaluation based on
    // https://randomwalks.xyz/publish/fast_polynomial_evaluation.html
    // Shaves a factor of 2 from run time.
    pub fn inside_out_evaluate(&self, r: &[F]) -> F {
        // Copied over from eq_poly
        // If the number of variables are greater
        // than 2^16 -- use parallel evaluate
        // Below that it's better to just do things linearly.
        const PARALLEL_THRESHOLD: usize = 16;
        // r must have a value for each variable
        assert_eq!(r.len(), self.get_num_vars());
        let m = r.len();
        if m < PARALLEL_THRESHOLD {
            self.inside_out_serial(r)
        } else {
            self.inside_out_parallel(r)
        }
    }

    fn inside_out_serial(&self, r: &[F]) -> F {
        // coeffs is a vector small scalars
        let mut current: Vec<F> = self.coeffs.iter().map(|&c| c.to_field()).collect();
        let m = r.len();
        for i in (0..m).rev() {
            let stride = 1 << i;
            let r_val = r[m - 1 - i];
            for j in 0..stride {
                let f0 = current[j];
                let f1 = current[j + stride];
                let slope = f1 - f0;
                if slope.is_zero() {
                    current[j] = f0;
                }
                if slope.is_one() {
                    current[j] = f0 + r_val;
                } else {
                    current[j] = f0 + slope * (r_val);
                }
            }
        }
        current[0]
    }

    fn inside_out_parallel(&self, r: &[F]) -> F {
        let mut current: Vec<F> = self.coeffs.par_iter().map(|&c| c.to_field()).collect();
        let m = r.len();
        for i in (0..m).rev() {
            let stride = 1 << i;
            let r_val = r[m - 1 - i];
            let (evals_left, evals_right) = current.split_at_mut(stride);
            let (evals_right, _) = evals_right.split_at_mut(stride);

            evals_left
                .par_iter_mut()
                .zip(evals_right.par_iter())
                .for_each(|(x, y)| {
                    //*x = *x + r_val * (*y - *x);
                    let slope = *y - *x;
                    if slope.is_zero() {
                        return;
                    }
                    if slope.is_one() {
                        *x += r_val;
                    } else {
                        *x += r_val * slope;
                    }
                });
        }
        current[0]
    }
}

impl<T: SmallScalar, F: JoltField> PolynomialBinding<F> for CompactPolynomial<T, F> {
    fn is_bound(&self) -> bool {
        !self.bound_coeffs.is_empty()
    }

    #[tracing::instrument(skip_all, name = "CompactPolynomial::bind")]
    fn bind(&mut self, r: F::Challenge, order: BindingOrder) {
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
            // We want to compute `a * (1 - r) + b * r` where `a` and `b` are small scalars
            // If `a == b`, we can just return `a`
            // If `a < b`, we can compute `a + r * (b - a)`
            // If `a > b`, we can compute `a - r * (a - b)`
            match order {
                BindingOrder::LowToHigh => {
                    self.bound_coeffs = (0..n)
                        .map(|i| {
                            let a = self.coeffs[2 * i];
                            let b = self.coeffs[2 * i + 1];
                            match a.cmp(&b) {
                                Ordering::Equal => a.to_field(),
                                // a < b: Compute a + r * (b - a)
                                Ordering::Less => {
                                    a.to_field::<F>() + b.diff_mul_field::<F>(a, r.into())
                                }
                                // a > b: Compute a - r * (a - b)
                                Ordering::Greater => {
                                    a.to_field::<F>() - a.diff_mul_field::<F>(b, r.into())
                                }
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
                            match a.cmp(&b) {
                                Ordering::Equal => a.to_field(),
                                // a < b: Compute a + r * (b - a)
                                Ordering::Less => {
                                    a.to_field::<F>() + b.diff_mul_field::<F>(a, r.into())
                                }
                                // a > b: Compute a - r * (a - b)
                                Ordering::Greater => {
                                    a.to_field::<F>() - a.diff_mul_field::<F>(b, r.into())
                                }
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
    fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        let n = self.len() / 2;
        if self.is_bound() {
            match order {
                BindingOrder::LowToHigh => {
                    let mut bound_coeffs = Vec::with_capacity(n);
                    (
                        bound_coeffs.spare_capacity_mut(),
                        self.bound_coeffs.par_chunks_exact(2),
                    )
                        .into_par_iter()
                        .with_min_len(512)
                        .for_each(|(bound_coeff, coeffs)| {
                            bound_coeff.write(if coeffs[1] == coeffs[0] {
                                coeffs[0]
                            } else {
                                (coeffs[1] - coeffs[0]) * r + coeffs[0]
                            });
                        });
                    unsafe { bound_coeffs.set_len(n) };
                    self.bound_coeffs = bound_coeffs;
                }
                BindingOrder::HighToLow => {
                    let (left, right) = self.bound_coeffs.split_at_mut(n);
                    left.par_iter_mut()
                        .zip(right.par_iter())
                        .with_min_len(4096)
                        .filter(|(a, b)| a != b)
                        .for_each(|(a, b)| {
                            *a += r * (*b - *a);
                        });
                }
            }
        } else {
            match order {
                BindingOrder::LowToHigh => {
                    self.bound_coeffs = (0..n)
                        .into_par_iter()
                        .map(|i| {
                            let a = self.coeffs[2 * i];
                            let b = self.coeffs[2 * i + 1];
                            match a.cmp(&b) {
                                Ordering::Equal => a.to_field(),
                                // a < b: Compute a + r * (b - a)
                                Ordering::Less => {
                                    a.to_field::<F>() + b.diff_mul_field::<F>(a, r.into())
                                }
                                // a > b: Compute a - r * (a - b)
                                Ordering::Greater => {
                                    a.to_field::<F>() - a.diff_mul_field::<F>(b, r.into())
                                }
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
                            match a.cmp(&b) {
                                Ordering::Equal => a.to_field(),
                                // a < b: Compute a + r * (b - a)
                                Ordering::Less => {
                                    a.to_field::<F>() + b.diff_mul_field::<F>(a, r.into())
                                }
                                // a > b: Compute a - r * (a - b)
                                Ordering::Greater => {
                                    a.to_field::<F>() - a.diff_mul_field::<F>(b, r.into())
                                }
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

impl<T: SmallScalar, F: JoltField> Index<usize> for CompactPolynomial<T, F> {
    type Output = T;

    #[inline(always)]
    fn index(&self, _index: usize) -> &T {
        &(self.coeffs[_index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::JoltField;
    use crate::poly::multilinear_polynomial::{BindingOrder, PolynomialBinding};
    use ark_bn254::Fr;

    #[test]
    fn clone_preserves_bound_state_u8_single_bind() {
        let coeffs: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut poly: CompactPolynomial<u8, Fr> = CompactPolynomial::from_coeffs(coeffs);
        let original_len = poly.len();
        let original_num_vars = poly.get_num_vars();

        let r = <Fr as JoltField>::Challenge::from(5u128);
        poly.bind(r, BindingOrder::LowToHigh);

        assert!(poly.is_bound());
        assert_eq!(poly.len(), original_len / 2);
        assert_eq!(poly.get_num_vars(), original_num_vars - 1);

        let clone = poly.clone();
        assert_eq!(poly, clone);
        assert!(clone.is_bound());
        assert_eq!(clone.len(), poly.len());
        assert_eq!(clone.get_num_vars(), poly.get_num_vars());
        assert_eq!(clone.bound_coeffs, poly.bound_coeffs);
    }

    #[test]
    fn clone_preserves_bound_state_u8_multiple_binds() {
        let coeffs: Vec<u8> = (0..8).collect();
        let mut poly: CompactPolynomial<u8, Fr> = CompactPolynomial::from_coeffs(coeffs);

        let r1 = <Fr as JoltField>::Challenge::from(3u128);
        let r2 = <Fr as JoltField>::Challenge::from(7u128);
        poly.bind(r1, BindingOrder::LowToHigh);
        poly.bind(r2, BindingOrder::LowToHigh);

        let clone = poly.clone();
        assert_eq!(poly, clone);
        assert!(clone.is_bound());
        assert_eq!(clone.len(), poly.len());
        assert_eq!(clone.get_num_vars(), poly.get_num_vars());
        assert_eq!(clone.bound_coeffs, poly.bound_coeffs);
    }
}
