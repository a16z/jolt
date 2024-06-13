use crate::field::JoltField;
use rayon::prelude::*;

pub struct SparsePolynomial<F: JoltField> {
    num_vars: usize,
    Z: Vec<(usize, F)>,
}

impl<Scalar: JoltField> SparsePolynomial<Scalar> {
    pub fn new(num_vars: usize, Z: Vec<(usize, Scalar)>) -> Self {
        SparsePolynomial { num_vars, Z }
    }

    /// Computes the $\tilde{eq}$ extension polynomial.
    /// return 1 when a == r, otherwise return 0.
    fn compute_chi(a: &[bool], r: &[Scalar]) -> Scalar {
        assert_eq!(a.len(), r.len());
        let mut chi_i = Scalar::one();
        for j in 0..r.len() {
            if a[j] {
                chi_i *= r[j];
            } else {
                chi_i *= Scalar::one() - r[j];
            }
        }
        chi_i
    }

    // Takes O(n log n)
    pub fn evaluate(&self, r: &[Scalar]) -> Scalar {
        assert_eq!(self.num_vars, r.len());

        (0..self.Z.len())
            .into_par_iter()
            .map(|i| {
                let bits = get_bits(self.Z[0].0, r.len());
                SparsePolynomial::compute_chi(&bits, r) * self.Z[i].1
            })
            .sum()
    }
}

/// Returns the `num_bits` from n in a canonical order
fn get_bits(operand: usize, num_bits: usize) -> Vec<bool> {
    (0..num_bits)
        .map(|shift_amount| ((operand & (1 << (num_bits - shift_amount - 1))) > 0))
        .collect::<Vec<bool>>()
}

/* This MLE is 1 if y = x + 1 for x in the range [0... 2^l-2].
That is, it ignores the case where x is all 1s, outputting 0.
Assumes x and y are provided big-endian. */
pub fn eq_plus_one<F: JoltField>(x: &[F], y: &[F], l: usize) -> F {
    let one = F::from_u64(1_u64).unwrap();

    /* If y+1 = x, then the two bit vectors are of the following form.
        Let k be the longest suffix of 1s in x.
        In y, those k bits are 0.
        Then, the next bit in x is 0 and the next bit in y is 1.
        The remaining higher bits are the same in x and y.
    */
    (0..l)
        .into_par_iter()
        .map(|k| {
            let lower_bits_product = (0..k)
                .map(|i| x[l - 1 - i] * (F::one() - y[l - 1 - i]))
                .product::<F>();
            let kth_bit_product = (F::one() - x[l - 1 - k]) * y[l - 1 - k];
            let higher_bits_product = ((k + 1)..l)
                .map(|i| x[l - 1 - i] * y[l - 1 - i] + (one - x[l - 1 - i]) * (one - y[l - 1 - i]))
                .product::<F>();
            lower_bits_product * kth_bit_product * higher_bits_product
        })
        .sum()
}
