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

    pub fn bind(&mut self, r_j: F::Challenge, bind_order: BindingOrder) {
        match bind_order {
            BindingOrder::HighToLow => self.bind_high_to_low(r_j),
            BindingOrder::LowToHigh => self.bind_low_to_high(r_j),
        }
    }

    fn bind_high_to_low(&mut self, r_j: F::Challenge) {
        let n_hi_vars = self.lt_hi.get_num_vars();
        if n_hi_vars != 0 {
            self.lt_hi.bind_parallel(r_j, BindingOrder::HighToLow);
            self.eq_hi.bind_parallel(r_j, BindingOrder::HighToLow);
        } else {
            self.lt_lo.bind_parallel(r_j, BindingOrder::HighToLow);
            self.n_lo_vars -= 1;
        }
    }

    fn bind_low_to_high(&mut self, r_j: F::Challenge) {
        if self.n_lo_vars != 0 {
            self.lt_lo.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.n_lo_vars -= 1;
        } else {
            self.lt_hi.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.eq_hi.bind_parallel(r_j, BindingOrder::LowToHigh);
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

/// Returns the MLE of `LT(j, r)` evaluated at all Boolean `j ∈ {0,1}^n`.
///
/// The less-than MLE is defined as:
///   `LT(x, y) = Σ_i (1 - x_i) · y_i · eq(x[i+1:], y[i+1:])`
/// where the sum runs from MSB to LSB.
///
/// This function computes `[LT(0, r), LT(1, r), ..., LT(2^n - 1, r)]`.
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

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;

    use crate::{
        field::challenge::MontU128Challenge,
        poly::{
            multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialEvaluation},
            opening_proof::{OpeningPoint, BIG_ENDIAN},
        },
    };

    use super::{lt_evals, LtPolynomial};

    #[test]
    fn test_bind_low_to_high_works() {
        let r_cycle = OpeningPoint::new([9, 5, 7, 1].map(MontU128Challenge::from).to_vec());
        let mut lt_poly = LtPolynomial::<Fr>::new(&r_cycle);
        let lt_poly_gt: MultilinearPolynomial<Fr> = lt_evals(&r_cycle).into();
        let r0 = MontU128Challenge::from(2);
        let r1 = MontU128Challenge::from(6);
        let r2 = MontU128Challenge::from(3);
        let r3 = MontU128Challenge::from(9);
        let r = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![r3, r2, r1, r0]);

        lt_poly.bind(r0, BindingOrder::LowToHigh);
        lt_poly.bind(r1, BindingOrder::LowToHigh);
        lt_poly.bind(r2, BindingOrder::LowToHigh);
        lt_poly.bind(r3, BindingOrder::LowToHigh);

        assert_eq!(lt_poly.get_bound_coeff(0), lt_poly_gt.evaluate(&r.r));
    }

    #[test]
    fn test_bind_high_to_low_works() {
        let r_cycle = OpeningPoint::new([9, 5, 7, 1].map(MontU128Challenge::from).to_vec());
        let mut lt_poly = LtPolynomial::<Fr>::new(&r_cycle);
        let lt_poly_gt: MultilinearPolynomial<Fr> = lt_evals(&r_cycle).into();
        let r0 = MontU128Challenge::from(2);
        let r1 = MontU128Challenge::from(6);
        let r2 = MontU128Challenge::from(3);
        let r3 = MontU128Challenge::from(9);
        let r = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![r0, r1, r2, r3]);

        lt_poly.bind(r0, BindingOrder::HighToLow);
        lt_poly.bind(r1, BindingOrder::HighToLow);
        lt_poly.bind(r2, BindingOrder::HighToLow);
        lt_poly.bind(r3, BindingOrder::HighToLow);

        assert_eq!(lt_poly.get_bound_coeff(0), lt_poly_gt.evaluate(&r.r));
    }
}
