use jolt_field::Field;
use jolt_poly::EqPolynomial;

use crate::dense::bind_dense_evals_reuse;

#[derive(Clone)]
pub(crate) struct SplitEqState<F: Field> {
    low_point: Vec<F>,
    high_point: Vec<F>,
    e_in: Vec<F>,
    e_out: Vec<F>,
    e_in_scratch: Vec<F>,
    e_out_scratch: Vec<F>,
}

impl<F: Field> SplitEqState<F> {
    #[inline]
    pub(crate) fn new_low_to_high(point: &[F], scaling: Option<F>) -> Self {
        let (high_point, low_point) = point.split_at(point.len() / 2);
        Self {
            low_point: low_point.to_vec(),
            high_point: high_point.to_vec(),
            e_in: EqPolynomial::<F>::evals(low_point, scaling),
            e_out: EqPolynomial::<F>::evals(high_point, None),
            e_in_scratch: Vec::new(),
            e_out_scratch: Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn e_in(&self) -> &[F] {
        &self.e_in
    }

    #[inline]
    pub(crate) fn e_out(&self) -> &[F] {
        &self.e_out
    }

    #[inline]
    pub(crate) fn current_target(&self) -> F {
        debug_assert!(self.e_in.len() > 1 || self.e_out.len() > 1);
        if self.e_in.len() > 1 {
            let remaining = self.e_in.len().trailing_zeros() as usize;
            self.low_point[remaining - 1]
        } else {
            let remaining = self.e_out.len().trailing_zeros() as usize;
            self.high_point[remaining - 1]
        }
    }

    #[inline]
    pub(crate) fn eval(&self) -> F {
        self.e_in[0] * self.e_out[0]
    }

    #[inline]
    pub(crate) fn bind(&mut self, challenge: F) {
        if self.e_in.len() > 1 {
            bind_dense_evals_reuse(&mut self.e_in, &mut self.e_in_scratch, challenge);
        } else {
            bind_dense_evals_reuse(&mut self.e_out, &mut self.e_out_scratch, challenge);
        }
    }
}
