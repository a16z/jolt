use std::marker::PhantomData;

use crate::field::JoltField;

pub struct RangeMaskPolynomial<F: JoltField> {
    range_start: u64,
    range_end: u64,
    _field: PhantomData<F>,
}

pub fn is_valid_range(range_start: u64, range_end: u64) -> bool {
    if range_start == 0 || range_end == 0 {
        return false;
    }

    // range_start must have the form: 0b11...100...0
    // Step 1: Find the smallest set bit (rightmost 1)
    let lowest = range_start.trailing_zeros();
    // Step 2: Shift x to remove trailing zeros
    let shifted = range_start >> lowest;
    // Step 3: The pattern is now like 0b111...1
    // If shifted is of the form 0b111...1, then shifted & (shifted + 1) == 0
    if (shifted & (shifted + 1)) != 0 {
        return false;
    }

    // range_end must be the next power of 2
    range_end == range_start.next_power_of_two()
}

impl<F: JoltField> RangeMaskPolynomial<F> {
    pub fn new(range_start: u64, range_end: u64) -> Self {
        println!("Range: [{range_start:b}, {range_end:b})");
        assert!(is_valid_range(range_start, range_end));
        Self {
            range_start,
            range_end,
            _field: PhantomData,
        }
    }

    /// For r in the Boolean hypercube, this MLE should evaluate to 1
    /// if r falls in the range [range_start, range_end) and 0 otherwise
    pub fn evaluate_mle(&self, r: &[F]) -> F {
        let num_leading_zeros = r.len() - self.range_end.trailing_zeros() as usize;
        let mut result = F::one();
        for r_i in r[..num_leading_zeros].iter() {
            result *= F::one() - r_i;
        }

        let num_ones = (self.range_start >> self.range_start.trailing_zeros()).trailing_ones();
        for r_i in r[num_leading_zeros..num_leading_zeros + num_ones as usize].iter() {
            result *= *r_i;
        }

        result
    }
}
