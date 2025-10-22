use std::iter::zip;

use allocative::Allocative;

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{OpeningPoint, BIG_ENDIAN},
    },
};

#[derive(Allocative)]
pub struct LtPolynomial<F: JoltField> {
    lt_lo: MultilinearPolynomial<F>,
    lt_hi: MultilinearPolynomial<F>,
    eq_hi: MultilinearPolynomial<F>,
    n_lo_vars: usize,
}

impl<F: JoltField> LtPolynomial<F> {
    pub fn new(r_cycle: &OpeningPoint<BIG_ENDIAN, F>) -> Self {
        let (r_hi, r_lo) = r_cycle.split_at(r_cycle.len() / 2);
        Self {
            lt_lo: MultilinearPolynomial::from(lt_evals::<F>(&r_lo)),
            lt_hi: MultilinearPolynomial::from(lt_evals::<F>(&r_hi)),
            eq_hi: MultilinearPolynomial::from(EqPolynomial::<F>::evals(&r_hi.r)),
            n_lo_vars: r_lo.len(),
        }
    }

    pub fn bind_high_to_low(&mut self, r_j: F::Challenge) {
        let n_hi_vars = self.lt_hi.get_num_vars();
        if n_hi_vars != 0 {
            self.lt_hi.bind_parallel(r_j, BindingOrder::HighToLow);
            self.eq_hi.bind_parallel(r_j, BindingOrder::HighToLow);
        } else {
            self.lt_lo.bind_parallel(r_j, BindingOrder::HighToLow);
        }
    }

    pub fn get_bound_coeff(&self, i: usize) -> F {
        let i_hi = i >> self.n_lo_vars;
        let i_lo = i & ((1 << self.n_lo_vars) - 1);
        // LT(i) = LT_hi(i_hi) + EQ_hi(i_hi) * LT_lo(i_lo)
        self.lt_hi.get_bound_coeff(i_hi)
            + self.eq_hi.get_bound_coeff(i_hi) * self.lt_lo.get_bound_coeff(i_lo)
    }
}

/// Evaluates `LT(r, j)` for all `j` in the hypercube.
fn lt_evals<F: JoltField>(r: &OpeningPoint<BIG_ENDIAN, F>) -> Vec<F> {
    let mut evals: Vec<F> = vec![F::zero(); 1 << r.len()];
    for (i, r) in r.r.iter().rev().enumerate() {
        let (evals_left, evals_right) = evals.split_at_mut(1 << i);
        zip(evals_left, evals_right).for_each(|(x, y)| {
            *y = *x * r;
            *x += *r - *y;
        });
    }
    evals
}
