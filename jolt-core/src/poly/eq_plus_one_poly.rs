use allocative::Allocative;
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

use crate::field::JoltField;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::prefix_suffix::{
    CachedPolynomial, DynamicPrefixRegistry, PrefixSuffixPolynomialFieldDyn,
};
use crate::utils::{lookup_bits::LookupBits, math::Math, thread::unsafe_allocate_zero_vec};

pub struct EqPlusOnePolynomial<F: JoltField> {
    x: Vec<F::Challenge>,
}

impl<F: JoltField> EqPlusOnePolynomial<F> {
    pub fn new(x: Vec<F::Challenge>) -> Self {
        EqPlusOnePolynomial { x }
    }

    /* This MLE is 1 if y = x + 1 for x in the range [0... 2^l-2].
    That is, it ignores the case where x is all 1s, outputting 0.
    Assumes x and y are provided big-endian. */
    pub fn evaluate(&self, y: &[F::Challenge]) -> F {
        let l = self.x.len();
        let x = &self.x;
        assert!(y.len() == l);

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
                    .map(|i| {
                        x[l - 1 - i] * y[l - 1 - i] + (F::one() - x[l - 1 - i]) * (F::one() - y[l - 1 - i])
                    })
                    .product::<F>();
                lower_bits_product * kth_bit_product * higher_bits_product
            })
            .sum()
    }

    #[tracing::instrument(skip_all, "EqPlusOnePolynomial::evals")]
    pub fn evals(r: &[F::Challenge], scaling_factor: Option<F>) -> (Vec<F>, Vec<F>) {
        let ell = r.len();
        let mut eq_evals: Vec<F> = unsafe_allocate_zero_vec(ell.pow2());
        eq_evals[0] = scaling_factor.unwrap_or(F::one());
        let mut eq_plus_one_evals: Vec<F> = unsafe_allocate_zero_vec(ell.pow2());

        // i indicates the LENGTH of the prefix of r for which the eq_table is calculated
        let eq_evals_helper = |eq_evals: &mut Vec<F>, r: &[F::Challenge], i: usize| {
            debug_assert!(i != 0);
            let step = 1 << (ell - i); // step = (full / size)/2

            let mut selected: Vec<_> = eq_evals.par_iter_mut().step_by(step).collect();

            selected.par_chunks_mut(2).for_each(|chunk| {
                *chunk[1] = *chunk[0] * r[i - 1];
                *chunk[0] -= *chunk[1];
            });
        };

        for i in 0..ell {
            let step = 1 << (ell - i);
            let half_step = step / 2;

            let mut r_lower_product = F::one();
            for &x in r.iter().skip(i + 1) {
                r_lower_product = r_lower_product * x; // To get the benefits of multiplication
            }
            r_lower_product *= F::one() - r[i];

            eq_plus_one_evals
                .par_iter_mut()
                .enumerate()
                .skip(half_step)
                .step_by(step)
                .for_each(|(index, v)| {
                    *v = eq_evals[index - half_step] * r_lower_product;
                });

            eq_evals_helper(&mut eq_evals, r, i + 1);
        }

        (eq_evals, eq_plus_one_evals)
    }
}

/// Field-valued prefix–suffix construction for EqPlusOne(y; x)
/// x is the challenge (sorted big-endian): x[0] is MSB, x[ell-1] is LSB
#[derive(Allocative)]
pub struct EqPlusOnePS<F: JoltField> {
    x: Vec<F::Challenge>,
    ell: usize,
    cutoff: usize,
}

impl<F: JoltField> EqPlusOnePS<F> {
    pub fn new(x: Vec<F::Challenge>, ell: usize, cutoff: usize) -> Self {
        debug_assert_eq!(x.len(), ell);
        Self {
            x,
            ell,
            cutoff,
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

    /// Build per-term prefix polynomials for the current phase and chunk length.
    /// Each returned entry corresponds to the prefix contribution of a single k term
    /// in the EqPlusOne(y; x) decomposition, evaluated over the current chunk's variables.
    fn prefixes(
        &self,
        chunk_len: usize,
        phase: usize,
        prefix_registry: &mut DynamicPrefixRegistry<F>,
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
                            let x = self.x[m]; // MSB index m
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

                // Scale by prior-phase checkpoint to carry over prefix contribution
                let scaled_evals = if phase == 0 {
                    evals
                } else {
                    let scale = prefix_registry.checkpoints[k_lsb].unwrap_or(F::one());
                    evals
                        .into_par_iter()
                        .map(|v| v * scale)
                        .collect()
                };

                let inner: MultilinearPolynomial<F> = MultilinearPolynomial::from(scaled_evals);
                Some(Arc::new(RwLock::new(CachedPolynomial::new(
                    inner,
                    cache_capacity,
                ))))
            })
            .collect()
    }

    /// Evaluate the suffix contribution for term k on the provided suffix bits.
    /// The suffix covers variables strictly after the current cutoff (MSB indexing).
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
            let x = self.x[m];
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

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::{AdditiveGroup, Field};

    use ark_std::test_rng;

    use crate::poly::prefix_suffix::PrefixSuffixDecompositionFieldDyn;

    // Evaluate the EqPlusOne MLE at arbitrary field point y (big-endian),
    // using the field-cast version of x provided in tests.
    fn eq_plus_one_mle_field<G: JoltField>(x_field: &[G], y_field: &[G]) -> G {
        let l = x_field.len();
        debug_assert_eq!(y_field.len(), l);

        let one = G::from_u64(1);
        let zero = G::from_u64(0);
        let mut acc_sum = zero;
        for k in 0..l {
            // Lower bits: product over i in [0..k) of x[i_lsb] * (1 - y[i_lsb]) in big-endian
            let mut lower = one;
            for i in 0..k {
                let xi = x_field[l - 1 - i];
                let yi = y_field[l - 1 - i];
                lower *= xi * (one - yi);
                if lower.is_zero() {
                    break;
                }
            }

            // k-th bit: (1 - x_k) * y_k
            let xk = x_field[l - 1 - k];
            let yk = y_field[l - 1 - k];
            let kth = (one - xk) * yk;
            if kth.is_zero() {
                // Early skip; keeps branches similar to production path
                continue;
            }

            // Higher bits: eq(x_i, y_i) for i in (k+1..l)
            let mut higher = one;
            for i in (k + 1)..l {
                let xi = x_field[l - 1 - i];
                let yi = y_field[l - 1 - i];
                let eq_term = xi * yi + (one - xi) * (one - yi);
                higher *= eq_term;
                if higher.is_zero() {
                    break;
                }
            }

            acc_sum += lower * kth * higher;
        }
        acc_sum
    }

    #[test]
    fn test_eq_plus_one_ps_field_dyn_matches_direct_evals() {
        let mut rng = test_rng();

        // Try a few sizes and random parameters
        for ell in 3..9 {
            for cutoff in 1..ell {
                // Random big-endian x bits as challenges
                let x_bits: Vec<<Fr as JoltField>::Challenge> = (0..ell)
                    .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
                    .collect();

                // Build dynamic prefix–suffix decomposition for EqPlusOne(y; x)
                let ps_poly = EqPlusOnePS::<Fr>::new(x_bits.clone(), ell, cutoff);
                let mut ps = PrefixSuffixDecompositionFieldDyn::new(Box::new(ps_poly), cutoff, ell);
                let mut prefix_registry = DynamicPrefixRegistry::<Fr>::new(ell);

                // Precompute lookup structures once
                let indices: Vec<usize> = (0..(1usize << ell)).collect();
                let lookup_bits: Vec<LookupBits> = (0..(1u128 << ell))
                    .map(|i| LookupBits::new(i, ell))
                    .collect();

                // Keep track of previously bound r values (as field elements)
                let mut rr_field: Vec<Fr> = Vec::new();

                // Phase 0
                {
                    ps.init_P(&mut prefix_registry);
                    let suffix_len = ps.suffix_len();
                    let u_evals = vec![Fr::ONE; 1usize << ell];
                    ps.init_Q(&u_evals, &indices, &lookup_bits);

                    for round in (0..cutoff).rev() {
                        let max_b = 1usize << round;
                        for b in 0..max_b {
                            // eval at t = 0
                            let mut y_prefix_0 = rr_field.clone();
                            y_prefix_0.push(Fr::ZERO);
                            for i in (0..round).rev() {
                                let bit = if ((b >> i) & 1) == 1 {
                                    Fr::ONE
                                } else {
                                    Fr::ZERO
                                };
                                y_prefix_0.push(bit);
                            }

                            // Sum over all suffix assignments
                            let mut direct_0 = Fr::ZERO;
                            for s in 0..(1usize << suffix_len) {
                                let mut y = y_prefix_0.clone();
                                for j in (0..suffix_len).rev() {
                                    let bit = if ((s >> j) & 1) == 1 {
                                        Fr::ONE
                                    } else {
                                        Fr::ZERO
                                    };
                                    y.push(bit);
                                }
                                let x_field: Vec<Fr> = x_bits.iter().map(|c| (*c).into()).collect();
                                direct_0 += eq_plus_one_mle_field::<Fr>(&x_field, &y);
                            }

                            // eval at t = 1 (for computing value at 2 via interpolation)
                            let mut y_prefix_1 = rr_field.clone();
                            y_prefix_1.push(Fr::ONE);
                            for i in (0..round).rev() {
                                let bit = if ((b >> i) & 1) == 1 {
                                    Fr::ONE
                                } else {
                                    Fr::ZERO
                                };
                                y_prefix_1.push(bit);
                            }
                            let mut direct_1 = Fr::ZERO;
                            for s in 0..(1usize << suffix_len) {
                                let mut y = y_prefix_1.clone();
                                for j in (0..suffix_len).rev() {
                                    let bit = if ((s >> j) & 1) == 1 {
                                        Fr::ONE
                                    } else {
                                        Fr::ZERO
                                    };
                                    y.push(bit);
                                }
                                let x_field: Vec<Fr> = x_bits.iter().map(|c| (*c).into()).collect();
                                direct_1 += eq_plus_one_mle_field::<Fr>(&x_field, &y);
                            }

                            // eval at t = 2
                            let mut y_prefix_2 = rr_field.clone();
                            y_prefix_2.push(Fr::ONE + Fr::ONE);
                            for i in (0..round).rev() {
                                let bit = if ((b >> i) & 1) == 1 {
                                    Fr::ONE
                                } else {
                                    Fr::ZERO
                                };
                                y_prefix_2.push(bit);
                            }
                            // For degree-2 polynomial in t, f(2) = 2*f(1) - f(0)
                            let direct_2 = direct_1 + direct_1 - direct_0;

                            let (eval_0, eval_2) = ps.sumcheck_evals(b);
                            assert_eq!(direct_0, eval_0);
                            assert_eq!(direct_2, eval_2);
                        }

                        // Bind next challenge r
                        let r_chal = <Fr as JoltField>::Challenge::random(&mut rng);
                        let r_field: Fr = r_chal.into();
                        rr_field.push(r_field);
                        ps.bind(r_chal);
                    }

                    prefix_registry.update_checkpoints();
                }

                // Phase 1 (remaining variables)
                let rem = ell - cutoff;
                if rem > 0 {
                    ps.init_P(&mut prefix_registry);
                    let suffix_len = ps.suffix_len();
                    debug_assert_eq!(suffix_len, 0);
                    let u_evals = vec![Fr::ONE; 1usize << ell];
                    ps.init_Q(&u_evals, &indices, &lookup_bits);

                    for round in (0..rem).rev() {
                        let max_b = 1usize << round;
                        for b in 0..max_b {
                            // t = 0
                            let mut y_prefix_0 = rr_field.clone();
                            y_prefix_0.push(Fr::ZERO);
                            for i in (0..round).rev() {
                                let bit = if ((b >> i) & 1) == 1 {
                                    Fr::ONE
                                } else {
                                    Fr::ZERO
                                };
                                y_prefix_0.push(bit);
                            }
                            let x_field: Vec<Fr> = x_bits.iter().map(|c| (*c).into()).collect();
                            let direct_0 = eq_plus_one_mle_field::<Fr>(&x_field, &y_prefix_0);

                            // t = 1
                            let mut y_prefix_1 = rr_field.clone();
                            y_prefix_1.push(Fr::ONE);
                            for i in (0..round).rev() {
                                let bit = if ((b >> i) & 1) == 1 {
                                    Fr::ONE
                                } else {
                                    Fr::ZERO
                                };
                                y_prefix_1.push(bit);
                            }
                            // f(2) = 2*f(1) - f(0)
                            let direct_2 = {
                                let f1 = eq_plus_one_mle_field::<Fr>(&x_field, &y_prefix_1);
                                f1 + f1 - direct_0
                            };

                            let (eval_0, eval_2) = ps.sumcheck_evals(b);
                            assert_eq!(direct_0, eval_0);
                            assert_eq!(direct_2, eval_2);
                        }

                        // Bind next challenge r
                        let r_chal = <Fr as JoltField>::Challenge::random(&mut rng);
                        let r_field: Fr = r_chal.into();
                        rr_field.push(r_field);
                        ps.bind(r_chal);
                    }
                }
            }
        }
    }
}
