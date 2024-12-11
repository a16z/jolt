// #![allow(clippy::too_many_arguments)]
// use crate::poly::eq_poly::EqPolynomial;
// use crate::utils::thread::{drop_in_background_thread, unsafe_allocate_zero_vec};
// use crate::utils::{self, compute_dotproduct, compute_dotproduct_low_optimized};

// use crate::field::JoltField;
// use crate::utils::math::Math;
// use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
// use core::ops::Index;
// use num_integer::Integer;
// use rand_core::{CryptoRng, RngCore};
// use rayon::prelude::*;

// use super::multilinear_polynomial::PolynomialBinding;

// #[derive(Default, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
// pub struct BooleanPolynomial {
//     num_vars: usize, // the number of variables in the multilinear polynomial
//     len: usize,
//     pub coeffs: Vec<bool>, // evaluations of the polynomial in the num_vars-dimensional Boolean hypercube
//                            // pub bound_coeffs: Vec<F>,
// }

// impl BooleanPolynomial {
//     pub fn from_coeffs(coeffs: Vec<bool>) -> Self {
//         assert!(
//             utils::is_power_of_two(coeffs.len()),
//             "Dense multilinear polynomials must be made from a power of 2 (not {})",
//             coeffs.len()
//         );

//         BooleanPolynomial {
//             num_vars: coeffs.len().log_2(),
//             len: coeffs.len(),
//             coeffs,
//         }
//     }

//     pub fn from_coeffs_padded(mut coeffs: Vec<bool>) -> Self {
//         // Pad non-power-2 evaluations to fill out the dense multilinear polynomial
//         while !(utils::is_power_of_two(coeffs.len())) {
//             coeffs.push(false);
//         }

//         BooleanPolynomial {
//             num_vars: coeffs.len().log_2(),
//             len: coeffs.len(),
//             coeffs,
//         }
//     }

//     pub fn get_num_vars(&self) -> usize {
//         self.num_vars
//     }

//     pub fn len(&self) -> usize {
//         self.len
//     }

//     pub fn is_empty(&self) -> bool {
//         self.len == 0
//     }

//     pub fn random<R: RngCore + CryptoRng>(num_vars: usize, rng: &mut R) -> Self {
//         Self::from_coeffs(
//             std::iter::from_fn(|| Some(rng.next_u32().is_odd()))
//                 .take(1 << num_vars)
//                 .collect(),
//         )
//     }
// }

// impl<F: JoltField> PolynomialBinding<F> for BooleanPolynomial {
//     fn bind(&mut self, r: F) {}
//     fn bind_parallel(&mut self, r: F) {}
// }

// impl Clone for BooleanPolynomial {
//     fn clone(&self) -> Self {
//         Self::from_coeffs(self.coeffs[0..self.len].to_vec())
//     }
// }

// impl Index<usize> for BooleanPolynomial {
//     type Output = bool;

//     #[inline(always)]
//     fn index(&self, _index: usize) -> &bool {
//         &(self.coeffs[_index])
//     }
// }
