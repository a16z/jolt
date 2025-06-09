use std::ops::Index;

use super::multilinear_polynomial::{BindingOrder, PolynomialBinding};
use crate::poly::sparse_matrix_polynomial::SparseMatrixPolynomial;
use crate::utils::math::Math;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::{field::JoltField, utils};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use num_integer::Integer;
use num_traits::MulAdd;
use rayon::prelude::*;
use std::cmp::Ordering;

#[derive(Default, Clone, Debug, PartialEq)]
pub struct OneHotPolynomial<F: JoltField> {
    pub num_rows: usize,
    pub num_cols: usize,
    pub nonzero_indices: Vec<usize>,
    pub nonzero_coeffs: Vec<F>,
}

impl<F: JoltField> OneHotPolynomial<F> {
    fn evaluate(&self, r: &[F]) -> F {
        todo!()
    }
}

impl<F: JoltField> PolynomialBinding<F> for OneHotPolynomial<F> {
    fn is_bound(&self) -> bool {
        todo!()
    }

    fn bind(&mut self, r: F, order: BindingOrder) {
        todo!()
    }

    fn bind_parallel(&mut self, r: F, order: BindingOrder) {
        todo!()
    }

    fn final_sumcheck_claim(&self) -> F {
        todo!()
    }
}
