use std::marker::PhantomData;

use crate::{field::JoltField, utils::lookup_bits::LookupBits};
use crate::field::MontU128;

pub struct RangeMaskPolynomial<F: JoltField> {
    range_start: u64,
    range_end: u64,
    _field: PhantomData<F>,
}

impl<F: JoltField> RangeMaskPolynomial<F> {
    pub fn new(range_start: u64, range_end: u64) -> Self {
        Self {
            range_start,
            range_end,
            _field: PhantomData,
        }
    }

    /// For r in the Boolean hypercube, this MLE should evaluate to 1
    /// if r falls in the range [range_start, range_end) and 0 otherwise
    /// In other words, LT(r, range_end) - LT(r, range_start)
    pub fn evaluate_mle(&self, r: &[MontU128]) -> F {
        // Compute LT(r, range_end)
        let mut range_end = LookupBits::new(self.range_end, r.len());
        let mut lt_range_end = F::zero();
        let mut eq_range_end = F::one();
        for r_i in r.iter() {
            let range_end_bit = range_end.pop_msb();
            let r_i_f = F::from_u128_mont(*r_i);
            if range_end_bit == 1 {
                lt_range_end += eq_range_end * (F::one() - r_i_f);
                eq_range_end = eq_range_end.mul_u128_mont_form(*r_i);
            } else {
                eq_range_end *= F::one() - r_i_f;
            }
        }

        // Compute LT(r, start)
        let mut range_start = LookupBits::new(self.range_start, r.len());
        let mut lt_range_start = F::zero();
        let mut eq_range_start = F::one();
        for r_i in r.iter() {
            let range_start_bit = range_start.pop_msb();
            let r_i_f = F::from_u128_mont(*r_i);
            if range_start_bit == 1 {
                lt_range_start += eq_range_start * (F::one() - r_i_f);
                eq_range_start = eq_range_start.mul_u128_mont_form(*r_i);
            } else {
                eq_range_start *= F::one() - r_i_f;
            }
        }

        lt_range_end - lt_range_start
    }
}
