use crate::{field::JoltField, poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial}, utils::{compute_dotproduct_low_optimized, math::Math, thread::{drop_in_background_thread, unsafe_allocate_zero_vec}}};
use num_integer::Integer;
use rayon::prelude::*;

#[derive(Clone)]
pub struct SparsePolynomial<F: JoltField> {
    num_vars: usize,
    
    Z: Vec<(usize, F)>,
}

impl<F: JoltField> SparsePolynomial<F> {
    pub fn new(num_vars: usize, Z: Vec<(usize, F)>) -> Self {
        SparsePolynomial { num_vars, Z }
    }

    // TODO(sragss): rm
    #[tracing::instrument(skip_all)]
    pub fn from_dense_evals(num_vars: usize, evals: Vec<F>) -> Self {
        assert!(num_vars.pow2() >= evals.len());
        let non_zero_count: usize = evals.par_iter().filter(|f| !f.is_zero()).count();
        let mut sparse: Vec<(usize, F)> = Vec::with_capacity(non_zero_count);
        evals.into_iter().enumerate().for_each(|(dense_index, f)| {
            if !f.is_zero() {
                sparse.push((dense_index, f));
            }
        });
        Self::new(num_vars, sparse)
    }

    /// Computes the $\tilde{eq}$ extension polynomial.
    /// return 1 when a == r, otherwise return 0.
    fn compute_chi(a: &[bool], r: &[F]) -> F {
        assert_eq!(a.len(), r.len());
        let mut chi_i = F::one();
        for j in 0..r.len() {
            if a[j] {
                chi_i *= r[j];
            } else {
                chi_i *= F::one() - r[j];
            }
        }
        chi_i
    }

    // Takes O(n log n)
    pub fn evaluate(&self, r: &[F]) -> F {
        assert_eq!(self.num_vars, r.len());

        (0..self.Z.len())
            .into_par_iter()
            .map(|i| {
                let bits = get_bits(self.Z[0].0, r.len());
                SparsePolynomial::compute_chi(&bits, r) * self.Z[i].1
            })
            .sum()
    }

    /// Returns n chunks of roughly even size without orphaning siblings (adjacent dense indices). Additionally returns a vector of (low, high] dense index ranges.
    fn chunk_no_orphans(&self, n: usize) -> (Vec<&[(usize, F)]>, Vec<(usize, usize)>) {
        if self.Z.len() < n * 2 {
            return (vec![(&self.Z)], vec![(0, self.num_vars.pow2())]);
        }

        let target_chunk_size = self.Z.len() / n;
        let mut chunks: Vec<&[(usize, F)]> = Vec::with_capacity(n);
        let mut dense_ranges: Vec<(usize, usize)> = Vec::with_capacity(n);
        let mut dense_start_index = 0;
        let mut sparse_start_index = 0;
        let mut sparse_end_index = target_chunk_size;
        for _ in 1..n {
            let mut dense_end_index = self.Z[sparse_end_index].0;
            if dense_end_index % 2 != 0 {
                dense_end_index += 1;
                sparse_end_index += 1;
            }
            chunks.push(&self.Z[sparse_start_index..sparse_end_index]);
            dense_ranges.push((dense_start_index, dense_end_index));
            dense_start_index = dense_end_index;

            sparse_start_index = sparse_end_index;
            sparse_end_index = std::cmp::min(sparse_end_index + target_chunk_size, self.Z.len() - 1);
        }
        chunks.push(&self.Z[sparse_start_index..]);
        // TODO(sragss): likely makes more sense to return full range then truncate when needed (triple iterator)
        let highest_non_zero = self.Z.last().map(|&(index, _)| index).unwrap();
        dense_ranges.push((dense_start_index, highest_non_zero + 1));
        assert_eq!(chunks.len(), n);
        assert_eq!(dense_ranges.len(), n);

        // TODO(sragss): To use chunk_no_orphans in the triple iterator, we have to overwrite the top of the dense_ranges.

        (chunks, dense_ranges)
    }

    #[tracing::instrument(skip_all)]
    pub fn bound_poly_var_bot(&mut self, r: &F) {
        // TODO(sragss): Do this with a scan instead.
        let n = self.Z.len();
        let span = tracing::span!(tracing::Level::DEBUG, "allocate");
        let _enter = span.enter();
        let mut new_Z: Vec<(usize, F)> = Vec::with_capacity(n);
        drop(_enter);
        for (sparse_index, (dense_index, value)) in self.Z.iter().enumerate() {
            if dense_index.is_even() {
                let new_dense_index = dense_index / 2;
                // TODO(sragss): Can likely combine these conditions for better speculative execution.
                if self.Z.len() >= 2 && sparse_index <= self.Z.len() - 2 && self.Z[sparse_index + 1].0 == dense_index + 1 {
                    let upper = self.Z[sparse_index + 1].1;
                    let eval = *value + *r  * (upper - value);
                    new_Z.push((new_dense_index, eval));
                } else {
                    new_Z.push((new_dense_index, (F::one() - r) * value));
                }
            } else {
                if sparse_index > 0 && self.Z[sparse_index - 1].0 == dense_index - 1 {
                    continue;
                } else {
                    let new_dense_index = (dense_index - 1) / 2;
                    new_Z.push((new_dense_index, *r * value));
                }
            }
        }
        self.Z = new_Z;
        self.num_vars -= 1;
    }

    #[tracing::instrument(skip_all)]
    pub fn bound_poly_var_bot_par(&mut self, r: &F) {
        // TODO(sragss): Do this with a scan instead.
        let n = self.Z.len();
        // let mut new_Z: Vec<(usize, F)> = Vec::with_capacity(n);

        let (chunks, _range) = self.chunk_no_orphans(rayon::current_num_threads() * 8);
        // TODO(sragsss): We can scan up front and collect directly into the thing.
        let new_Z: Vec<(usize, F)> = chunks.into_par_iter().map(|chunk| {
            // TODO(sragss): Do this with a scan instead;
            let mut chunk_Z: Vec<(usize, F)> = Vec::with_capacity(chunk.len());
            for (sparse_index, (dense_index, value)) in chunk.iter().enumerate() {
                if dense_index.is_even() {
                    let new_dense_index = dense_index / 2;
                    // TODO(sragss): Can likely combine these conditions for better speculative execution.
                    if self.Z.len() >= 2 && sparse_index <= self.Z.len() - 2 && self.Z[sparse_index + 1].0 == dense_index + 1 {
                        let upper = self.Z[sparse_index + 1].1;
                        let eval = *value + *r  * (upper - value);
                        chunk_Z.push((new_dense_index, eval));
                    } else {
                        chunk_Z.push((new_dense_index, (F::one() - r) * value));
                    }
                } else {
                    if sparse_index > 0 && self.Z[sparse_index - 1].0 == dense_index - 1 {
                        continue;
                    } else {
                        let new_dense_index = (dense_index - 1) / 2;
                        chunk_Z.push((new_dense_index, *r * value));
                    }
                }
            }

            chunk_Z
        }).flatten().collect();

        // for (sparse_index, (dense_index, value)) in self.Z.iter().enumerate() {
        //     if dense_index.is_even() {
        //         let new_dense_index = dense_index / 2;
        //         // TODO(sragss): Can likely combine these conditions for better speculative execution.
        //         if self.Z.len() >= 2 && sparse_index <= self.Z.len() - 2 && self.Z[sparse_index + 1].0 == dense_index + 1 {
        //             let upper = self.Z[sparse_index + 1].1;
        //             let eval = *value + *r  * (upper - value);
        //             new_Z.push((new_dense_index, eval));
        //         } else {
        //             new_Z.push((new_dense_index, (F::one() - r) * value));
        //         }
        //     } else {
        //         if sparse_index > 0 && self.Z[sparse_index - 1].0 == dense_index - 1 {
        //             continue;
        //         } else {
        //             let new_dense_index = (dense_index - 1) / 2;
        //             new_Z.push((new_dense_index, *r * value));
        //         }
        //     }
        // }
        self.Z = new_Z;
        self.num_vars -= 1;
    }

    pub fn final_eval(&self) -> F {
        assert_eq!(self.num_vars, 0);
        if self.Z.len() == 0 {
            F::zero()
        } else {
            assert_eq!(self.Z.len(), 1);
            let item = self.Z[0];
            assert_eq!(item.0, 0);
            item.1
        }
    }

    #[cfg(test)]
    #[tracing::instrument(skip_all)]
    pub fn to_dense(self) -> DensePolynomial<F> {
        use crate::utils::math::Math;

        let mut evals = unsafe_allocate_zero_vec(self.num_vars.pow2());

        for (index, value) in self.Z {
            evals[index] = value;
        }

        DensePolynomial::new(evals)
    }
}

pub struct SparseTripleIterator<'a, F: JoltField> {
    dense_index: usize,
    end_index: usize,
    a: &'a [(usize, F)],
    b: &'a [(usize, F)],
    c: &'a [(usize, F)],
}

impl<'a, F: JoltField> SparseTripleIterator<'a, F> {
    #[tracing::instrument(skip_all)]
    pub fn chunks(a: &'a SparsePolynomial<F>, b: &'a SparsePolynomial<F>, c: &'a SparsePolynomial<F>, n: usize) -> Vec<Self> {
        // When the instance is small enough, don't worry about parallelism
        let total_len = a.num_vars.pow2();
        if n * 2 > b.Z.len() {
            return vec![SparseTripleIterator {
                dense_index: 0,
                end_index: total_len,
                a: &a.Z,
                b: &b.Z,
                c: &c.Z
            }];
        }
        // Can be made more generic, but this is an optimization / simplification.
        assert!(b.Z.len() >= a.Z.len() && b.Z.len() >= c.Z.len(), "b.Z.len() assumed to be longest of a, b, and c");

        // TODO(sragss): Explain the strategy

        let target_chunk_size = b.Z.len() / n;
        let mut b_chunks: Vec<&[(usize, F)]> = Vec::with_capacity(n);
        let mut dense_ranges: Vec<(usize, usize)> = Vec::with_capacity(n);
        let mut dense_start_index = 0;
        let mut sparse_start_index = 0;
        let mut sparse_end_index = target_chunk_size;
        for _ in 1..n {
            let mut dense_end_index = b.Z[sparse_end_index].0;
            if dense_end_index % 2 != 0 {
                dense_end_index += 1;
                sparse_end_index += 1;
            }
            b_chunks.push(&b.Z[sparse_start_index..sparse_end_index]);
            dense_ranges.push((dense_start_index, dense_end_index));
            dense_start_index = dense_end_index;

            sparse_start_index = sparse_end_index;
            sparse_end_index = std::cmp::min(sparse_end_index + target_chunk_size, b.Z.len() - 1);
        }
        b_chunks.push(&b.Z[sparse_start_index..]);
        let highest_non_zero = {
            let a_last = a.Z.last().map(|&(index, _)| index);
            let b_last = b.Z.last().map(|&(index, _)| index);
            let c_last = c.Z.last().map(|&(index, _)| index);
            *a_last.iter().chain(b_last.iter()).chain(c_last.iter()).max().unwrap()
        };
        dense_ranges.push((dense_start_index, highest_non_zero + 1));
        assert_eq!(b_chunks.len(), n);
        assert_eq!(dense_ranges.len(), n);

        // Create chunks which overlap with b's sparse indices
        let mut a_chunks: Vec<&[(usize, F)]> = vec![&[]; n];
        let mut c_chunks: Vec<&[(usize, F)]> = vec![&[]; n];
        let mut a_i = 0;
        let mut c_i = 0;
        let span = tracing::span!(tracing::Level::DEBUG, "a, c scanning");
        let _enter = span.enter();
        for (chunk_index, range) in dense_ranges.iter().enumerate().skip(1) {
            // Find the corresponding a, c chunks
            let prev_chunk_end = range.0;

            if a_i < a.Z.len() && a.Z[a_i].0 < prev_chunk_end {
                let a_start = a_i;
                while a_i < a.Z.len() && a.Z[a_i].0 < prev_chunk_end {
                    a_i += 1;
                }

                a_chunks[chunk_index - 1] = &a.Z[a_start..a_i];
            }

            if c_i < c.Z.len() && c.Z[c_i].0 < prev_chunk_end {
                let c_start = c_i;
                while c_i < c.Z.len() && c.Z[c_i].0 < prev_chunk_end {
                    c_i += 1;
                }

                c_chunks[chunk_index - 1] = &c.Z[c_start..c_i];
            }

        }
        drop(_enter);
        a_chunks[n-1] = &a.Z[a_i..];
        c_chunks[n-1] = &c.Z[c_i..];

        #[cfg(test)]
        {
            assert_eq!(a_chunks.concat(), a.Z);
            assert_eq!(b_chunks.concat(), b.Z);
            assert_eq!(c_chunks.concat(), c.Z);
        }

        let mut iterators: Vec<SparseTripleIterator<'a, F>> = Vec::with_capacity(n);
        for (((a_chunk, b_chunk), c_chunk), range) in a_chunks.iter().zip(b_chunks.iter()).zip(c_chunks.iter()).zip(dense_ranges.iter()) {
            #[cfg(test)]
            {
                assert!(a_chunk.iter().all(|(index, _)| *index >= range.0 && *index <= range.1));
                assert!(b_chunk.iter().all(|(index, _)| *index >= range.0 && *index <= range.1));
                assert!(c_chunk.iter().all(|(index, _)| *index >= range.0 && *index <= range.1));
            }
            let iter = SparseTripleIterator {
                dense_index: range.0,
                end_index: range.1,
                a: a_chunk,
                b: b_chunk,
                c: c_chunk
            };
            iterators.push(iter);
        }

        iterators
    }

    pub fn has_next(&self) -> bool {
        self.dense_index < self.end_index
    }

    pub fn next_pairs(&mut self) -> (usize, F, F, F, F, F, F) {
        // TODO(sragss): We can store a map of big ranges of zeros and skip them rather than hitting each dense index.
        let low_index = self.dense_index;
        let match_and_advance = |slice: &mut &[(usize, F)], index: usize| -> F {
            if let Some(first_item) = slice.first() {
                if first_item.0 == index {
                    let ret = first_item.1;
                    *slice = &slice[1..];
                    ret
                } else {
                    F::zero()
                }
            } else {
                F::zero()
            }
        };

        let a_lower_val = match_and_advance(&mut self.a, self.dense_index);
        let b_lower_val = match_and_advance(&mut self.b, self.dense_index);
        let c_lower_val = match_and_advance(&mut self.c, self.dense_index);
        self.dense_index += 1;
        let a_upper_val = match_and_advance(&mut self.a, self.dense_index);
        let b_upper_val = match_and_advance(&mut self.b, self.dense_index);
        let c_upper_val = match_and_advance(&mut self.c, self.dense_index);
        self.dense_index += 1;

        (low_index, a_lower_val, a_upper_val, b_lower_val, b_upper_val, c_lower_val, c_upper_val)
    }
}

pub trait IndexablePoly<F: JoltField>: std::ops::Index<usize, Output = F> + Sync {
    fn len(&self) -> usize;
}

impl<F: JoltField> IndexablePoly<F> for DensePolynomial<F> {
    fn len(&self) -> usize {
        self.Z.len()
    }
}

// TODO: Rather than use these adhoc virtual indexable polys – create a DensePolynomial which takes any impl Index<usize> inner
// and can run all the normal DensePolynomial ops.
#[derive(Clone)]
pub struct SegmentedPaddedWitness<F: JoltField> {
    total_len: usize,
    pub segments: Vec<Vec<F>>,
    pub segment_len: usize,
    zero: F,
}

impl<F: JoltField> SegmentedPaddedWitness<F> {
    pub fn new(total_len: usize, segments: Vec<Vec<F>>) -> Self {
        let segment_len = segments[0].len();
        assert!(segment_len.is_power_of_two());
        for segment in &segments {
            assert_eq!(
                segment.len(),
                segment_len,
                "All segments must be the same length"
            );
        }
        SegmentedPaddedWitness {
            total_len,
            segments,
            segment_len,
            zero: F::zero(),
        }
    }

    pub fn len(&self) -> usize {
        self.total_len
    }

    #[tracing::instrument(skip_all, name = "SegmentedPaddedWitness::evaluate_all")]
    pub fn evaluate_all(&self, point: Vec<F>) -> Vec<F> {
        let chi = EqPolynomial::evals(&point);
        assert!(chi.len() >= self.segment_len);

        let evals = self
            .segments
            .par_iter()
            .map(|segment| compute_dotproduct_low_optimized(&chi[0..self.segment_len], segment))
            .collect();
        drop_in_background_thread(chi);
        evals
    }

    pub fn into_dense_polys(self) -> Vec<DensePolynomial<F>> {
        self.segments
            .into_iter()
            .map(|poly| DensePolynomial::new(poly))
            .collect()
    }
}

impl<F: JoltField> std::ops::Index<usize> for SegmentedPaddedWitness<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.segments.len() * self.segment_len {
            &self.zero
        } else if index >= self.total_len {
            panic!("index too high");
        } else {
            let segment_index = index / self.segment_len;
            let inner_index = index % self.segment_len;
            &self.segments[segment_index][inner_index]
        }
    }
}

impl<F: JoltField> IndexablePoly<F> for SegmentedPaddedWitness<F> {
    fn len(&self) -> usize {
        self.total_len
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

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::Zero;

    #[test]
    fn sparse_bound_bot_all_left() {
        let dense_evals = vec![Fr::from(10), Fr::zero(), Fr::from(20), Fr::zero()];
        let sparse_evals = vec![(0, Fr::from(10)), (2, Fr::from(20))];

        let mut dense = DensePolynomial::new(dense_evals);
        let mut sparse = SparsePolynomial::new(2, sparse_evals);

        assert_eq!(sparse.clone().to_dense(), dense);

        let r = Fr::from(121);
        sparse.bound_poly_var_bot(&r);
        dense.bound_poly_var_bot(&r);
        assert_eq!(sparse.to_dense(), dense);
    }

    #[test]
    fn sparse_bound_bot_all_right() {
        let dense_evals = vec![Fr::zero(), Fr::from(10), Fr::zero(), Fr::from(20)];
        let sparse_evals = vec![(1, Fr::from(10)), (3, Fr::from(20))];

        let mut dense = DensePolynomial::new(dense_evals);
        let mut sparse = SparsePolynomial::new(2, sparse_evals);

        assert_eq!(sparse.clone().to_dense(), dense);

        let r = Fr::from(121);
        sparse.bound_poly_var_bot(&r);
        dense.bound_poly_var_bot(&r);
        assert_eq!(sparse.to_dense(), dense);
    }

    #[test]
    fn sparse_bound_bot_mixed() {
        let dense_evals = vec![Fr::zero(), Fr::from(10), Fr::zero(), Fr::from(20), Fr::from(30), Fr::from(40), Fr::zero(), Fr::from(50)];
        let sparse_evals = vec![(1, Fr::from(10)), (3, Fr::from(20)), (4, Fr::from(30)), (5, Fr::from(40)), (7, Fr::from(50))];

        let mut dense = DensePolynomial::new(dense_evals);
        let mut sparse = SparsePolynomial::new(3, sparse_evals);

        assert_eq!(sparse.clone().to_dense(), dense);

        let r = Fr::from(121);
        sparse.bound_poly_var_bot(&r);
        dense.bound_poly_var_bot(&r);
        assert_eq!(sparse.to_dense(), dense);
    }

    #[test]
    fn sparse_triple_iterator() {
        let a = vec![(9, Fr::from(9)), (10, Fr::from(10)), (12, Fr::from(12))];
        let b = vec![(0, Fr::from(100)), (1, Fr::from(1)), (2, Fr::from(2)), (3, Fr::from(3)), (4, Fr::from(4)), (5, Fr::from(5)), (6, Fr::from(6)), (7, Fr::from(7)), (8, Fr::from(8)), (9, Fr::from(9)), (10, Fr::from(10)), (11, Fr::from(11)), (12, Fr::from(12)), (13, Fr::from(13)), (14, Fr::from(14)), (15, Fr::from(15))];
        let c = vec![(0, Fr::from(12)), (3, Fr::from(3))];

        let a_poly = SparsePolynomial::new(4, a);
        let b_poly = SparsePolynomial::new(4, b);
        let c_poly = SparsePolynomial::new(4, c);

        let iterators = SparseTripleIterator::chunks(&a_poly, &b_poly, &c_poly, 4);
        assert_eq!(iterators.len(), 4);
    }

    #[test]
    fn sparse_triple_iterator_random() {
        use rand::Rng;

        let mut rng = rand::thread_rng();

        let prob_exists = 0.32;
        let num_vars = 10;
        let total_len = 1 << num_vars;

        let mut a = vec![];
        let mut b = vec![];
        let mut c = vec![];

        for i in 0usize..total_len {
            if rng.gen::<f64>() < prob_exists {
                a.push((i, Fr::from(i as u64)));
            }
            if rng.gen::<f64>() < prob_exists * 2f64 {
                b.push((i, Fr::from(i as u64)));
            }
            if rng.gen::<f64>() < prob_exists {
                c.push((i, Fr::from(i as u64)));
            }
        }

        let a_poly = SparsePolynomial::new(num_vars, a);
        let b_poly = SparsePolynomial::new(num_vars, b);
        let c_poly = SparsePolynomial::new(num_vars, c);

        let mut iterators = SparseTripleIterator::chunks(&a_poly, &b_poly, &c_poly, 8);

        let mut new_a = vec![Fr::zero(); total_len];
        let mut new_b = vec![Fr::zero(); total_len];
        let mut new_c = vec![Fr::zero(); total_len];
        let mut expected_dense_index = 0;
        for iterator in iterators.iter_mut() {
            while iterator.has_next() {
                let (dense_index, a_low, a_high, b_low, b_high, c_low, c_high) = iterator.next_pairs();

                new_a[dense_index] = a_low;
                new_a[dense_index+1] = a_high;

                new_b[dense_index] = b_low;
                new_b[dense_index+1] = b_high;

                new_c[dense_index] = c_low;
                new_c[dense_index+1] = c_high;

                assert_eq!(dense_index, expected_dense_index);
                expected_dense_index += 2;
            }
        }

        assert_eq!(a_poly.to_dense().Z, new_a);
        assert_eq!(b_poly.to_dense().Z, new_b);
        assert_eq!(c_poly.to_dense().Z, new_c);
    }
}