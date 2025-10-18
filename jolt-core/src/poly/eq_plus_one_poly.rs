use allocative::Allocative;
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

use crate::field::JoltField;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::prefix_suffix::{
    CachedPolynomial, DynamicPrefixRegistry, PrefixSuffixPolynomialFieldDyn,
};
use crate::utils::lookup_bits::LookupBits;

/// Field-valued prefix–suffix construction for EqPlusOne(y; x)
/// x_bits are big-endian challenges: x_bits[0] is MSB, x_bits[ell-1] is LSB
#[derive(Allocative)]
pub struct EqPlusOnePS<F: JoltField> {
    x_bits: Vec<F::Challenge>,
    ell: usize,
    cutoff: usize,
    // Cached field versions of x for convenience
    x_field: Vec<F>,
}

impl<F: JoltField> EqPlusOnePS<F> {
    pub fn new(x_bits: Vec<F::Challenge>, ell: usize, cutoff: usize) -> Self {
        debug_assert_eq!(x_bits.len(), ell);
        let x_field = x_bits.iter().map(|c| (*c).into()).collect();
        Self {
            x_bits,
            ell,
            cutoff,
            x_field,
        }
    }

    #[inline(always)]
    pub fn order(&self) -> usize {
        self.ell
    }

    #[inline(always)]
    fn msb_flip_index(&self, k_lsb: usize) -> usize {
        // Convert LSB-oriented k to MSB position
        self.ell - 1 - k_lsb
    }

    #[inline(always)]
    fn ypref_bit(idx: usize, msb_pos: usize, cutoff: usize) -> bool {
        // idx ranges over 0..2^cutoff, interpret as cutoff-bit big-endian
        // bit at MSB position m maps to bit (cutoff - 1 - m)
        ((idx >> (cutoff - 1 - msb_pos)) & 1) == 1
    }
}

impl<F: JoltField> PrefixSuffixPolynomialFieldDyn<F> for EqPlusOnePS<F> {
    fn order(&self) -> usize {
        self.ell
    }

    fn prefixes(
        &self,
        chunk_len: usize,
        phase: usize,
        _prefix_registry: &mut DynamicPrefixRegistry<F>,
    ) -> Vec<Option<Arc<RwLock<CachedPolynomial<F>>>>> {
        let order = self.order();
        let poly_len = 1usize << chunk_len;
        let cache_capacity = poly_len / 2;
        let start = if phase == 0 { 0 } else { self.cutoff };

        (0..order)
            .into_par_iter()
            .map(|k_lsb| {
                // MSB index of the flip bit for this term
                let m_flip = self.msb_flip_index(k_lsb);
                let evals: Vec<F> = (0..poly_len)
                    .into_par_iter()
                    .map(|idx| {
                        let mut acc = F::one();
                        for off in 0..chunk_len {
                            let m = start + off; // absolute MSB index
                            let y_bit = Self::ypref_bit(idx, off, chunk_len);
                            let y = if y_bit { F::one() } else { F::zero() };
                            let x = self.x_field[m]; // MSB index m
                            let term = if m < m_flip {
                                // eq: x*y + (1-x)*(1-y)
                                x * y + (F::one() - x) * (F::one() - y)
                            } else if m == m_flip {
                                // flip: (1 - x) * y
                                (F::one() - x) * y
                            } else {
                                // carry: x * (1 - y)
                                x * (F::one() - y)
                            };
                            acc *= term;
                            if acc.is_zero() {
                                break;
                            }
                        }
                        acc
                    })
                    .collect();

                let inner: MultilinearPolynomial<F> = MultilinearPolynomial::from(evals);
                Some(Arc::new(RwLock::new(CachedPolynomial::new(
                    inner,
                    cache_capacity,
                ))))
            })
            .collect()
    }

    fn suffix_eval(&self, k_lsb: usize, suffix: LookupBits) -> F {
        let m_flip = self.msb_flip_index(k_lsb);
        let m_len = suffix.len();
        if m_len == 0 {
            return F::one();
        }
        // number of suffix MSB indices expected equals ell - cutoff
        let bits_val: u128 = (&suffix).into();

        let mut acc = F::one();
        for j_msb in 0..m_len {
            // absolute MSB index m
            let m = self.cutoff + j_msb;
            let bit = ((bits_val >> (m_len - 1 - j_msb)) & 1) == 1;
            let y = if bit { F::one() } else { F::zero() };
            let x = self.x_field[m];
            let term = if m < m_flip {
                x * y + (F::one() - x) * (F::one() - y)
            } else if m == m_flip {
                (F::one() - x) * y
            } else {
                x * (F::one() - y)
            };
            acc *= term;
            if acc.is_zero() {
                break;
            }
        }
        acc
    }
}
