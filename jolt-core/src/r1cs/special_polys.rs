use crate::{
    field::JoltField,
    utils::{
        math::Math,
        mul_0_1_optimized,
        thread::{drop_in_background_thread, unsafe_allocate_sparse_zero_vec},
    },
};
use num_integer::Integer;
use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq)]
pub struct SparsePolynomial<F: JoltField> {
    num_vars: usize,

    Z: Vec<(F, usize)>,
}

impl<F: JoltField> SparsePolynomial<F> {
    pub fn new(num_vars: usize, Z: Vec<(F, usize)>) -> Self {
        SparsePolynomial { num_vars, Z }
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
                let bits = get_bits(self.Z[0].1, r.len());
                SparsePolynomial::compute_chi(&bits, r) * self.Z[i].0
            })
            .sum()
    }

    /// Returns `n` chunks of roughly even size without separating siblings (adjacent dense indices). Additionally returns a vector of [low, high) dense index ranges.
    #[tracing::instrument(skip_all)]
    #[allow(clippy::type_complexity)]
    fn chunk_no_split_siblings(&self, n: usize) -> (Vec<&[(F, usize)]>, Vec<(usize, usize)>) {
        if self.Z.len() < n * 2 {
            return (vec![(&self.Z)], vec![(0, self.num_vars.pow2())]);
        }

        let target_chunk_size = self.Z.len() / n;
        let mut chunks: Vec<&[(F, usize)]> = Vec::with_capacity(n);
        let mut dense_ranges: Vec<(usize, usize)> = Vec::with_capacity(n);
        let mut dense_start_index = 0;
        let mut sparse_start_index = 0;
        let mut sparse_end_index = target_chunk_size;
        for _ in 1..n {
            let mut dense_end_index = self.Z[sparse_end_index].1;
            if dense_end_index % 2 != 0 {
                dense_end_index += 1;
                sparse_end_index += 1;
            }
            chunks.push(&self.Z[sparse_start_index..sparse_end_index]);
            dense_ranges.push((dense_start_index, dense_end_index));
            dense_start_index = dense_end_index;

            sparse_start_index = sparse_end_index;
            sparse_end_index =
                std::cmp::min(sparse_end_index + target_chunk_size, self.Z.len() - 1);
        }
        chunks.push(&self.Z[sparse_start_index..]);
        let highest_non_zero = self.Z.last().map(|&(_, index)| index).unwrap();
        dense_ranges.push((dense_start_index, highest_non_zero + 1));
        assert_eq!(chunks.len(), n);
        assert_eq!(dense_ranges.len(), n);

        (chunks, dense_ranges)
    }

    #[tracing::instrument(skip_all)]
    pub fn bound_poly_var_bot(&mut self, r: &F) {
        // TODO(sragss): Do this with a scan instead.
        let n = self.Z.len();
        let span = tracing::span!(tracing::Level::DEBUG, "allocate");
        let _enter = span.enter();
        let mut new_Z: Vec<(F, usize)> = Vec::with_capacity(n);
        drop(_enter);
        for (sparse_index, (value, dense_index)) in self.Z.iter().enumerate() {
            if dense_index.is_even() {
                let new_dense_index = dense_index / 2;
                if self.Z.len() >= 2
                    && sparse_index <= self.Z.len() - 2
                    && self.Z[sparse_index + 1].1 == dense_index + 1
                {
                    let upper = self.Z[sparse_index + 1].0;
                    let eval = *value + *r * (upper - value);
                    new_Z.push((eval, new_dense_index));
                } else {
                    new_Z.push(((F::one() - r) * value, new_dense_index));
                }
            } else if sparse_index > 0 && self.Z[sparse_index - 1].1 == dense_index - 1 {
                continue;
            } else {
                let new_dense_index = (dense_index - 1) / 2;
                new_Z.push((*r * value, new_dense_index));
            }
        }
        self.Z = new_Z;
        self.num_vars -= 1;
    }

    #[tracing::instrument(skip_all)]
    pub fn bound_poly_var_bot_par(&mut self, r: &F) {
        // TODO(sragss): better parallelism.
        let (chunks, _range) = self.chunk_no_split_siblings(rayon::current_num_threads() * 8);

        // Calc chunk sizes post-binding for pre-allocation.
        let count_span = tracing::span!(tracing::Level::DEBUG, "counting");
        let count_enter = count_span.enter();
        let chunk_sizes: Vec<usize> = chunks
            .par_iter()
            .map(|chunk| {
                // Count each pair of siblings if at least one is present.
                chunk
                    .iter()
                    .enumerate()
                    .filter(|(i, (_value, index))| {
                        // Always count odd, only count even indices when the paired odd index is not present.
                        !index.is_even() || i + 1 >= chunk.len() || index + 1 != chunk[i + 1].1
                    })
                    .count()
            })
            .collect();
        drop(count_enter);

        let alloc_span = tracing::span!(tracing::Level::DEBUG, "alloc");
        let alloc_enter = alloc_span.enter();
        let total_len: usize = chunk_sizes.iter().sum();
        let mut new_Z: Vec<(F, usize)> = unsafe_allocate_sparse_zero_vec(total_len);
        drop(alloc_enter);

        let mut mutable_chunks: Vec<&mut [(F, usize)]> = Vec::with_capacity(chunk_sizes.len());
        let mut remainder = new_Z.as_mut_slice();
        for chunk_size in chunk_sizes {
            let (first, second) = remainder.split_at_mut(chunk_size);
            mutable_chunks.push(first);
            remainder = second;
        }
        assert_eq!(mutable_chunks.len(), chunks.len());

        // Bind each chunk in parallel
        chunks
            .into_par_iter()
            .zip(mutable_chunks.par_iter_mut())
            .for_each(|(chunk, mutable)| {
                let span = tracing::span!(tracing::Level::DEBUG, "chunk");
                let _enter = span.enter();
                let mut write_index = 0;
                for (sparse_index, (value, dense_index)) in chunk.iter().enumerate() {
                    if dense_index.is_even() {
                        let new_dense_index = dense_index / 2;

                        if chunk.len() >= 2
                            && sparse_index <= chunk.len() - 2
                            && chunk[sparse_index + 1].1 == dense_index + 1
                        {
                            // (low, high) present
                            let upper = chunk[sparse_index + 1].0;
                            let eval = *value + mul_0_1_optimized(r, &(upper - value));
                            mutable[write_index] = (eval, new_dense_index);
                            write_index += 1;
                        } else {
                            // (low, _) present
                            mutable[write_index] =
                                (mul_0_1_optimized(&(F::one() - r), value), new_dense_index);
                            write_index += 1;
                        }
                    } else if sparse_index > 0 && chunk[sparse_index - 1].1 == dense_index - 1 {
                        // (low, high) present, but handeled prior
                        continue;
                    } else {
                        // (_, high) present
                        let new_dense_index = (dense_index - 1) / 2;
                        mutable[write_index] = (mul_0_1_optimized(r, value), new_dense_index);
                        write_index += 1;
                    }
                }
            });

        let old_Z = std::mem::replace(&mut self.Z, new_Z);
        drop_in_background_thread(old_Z);
        self.num_vars -= 1;
    }

    pub fn final_eval(&self) -> F {
        assert_eq!(self.num_vars, 0);
        if self.Z.is_empty() {
            F::zero()
        } else {
            assert_eq!(self.Z.len(), 1);
            let item = self.Z[0];
            assert_eq!(item.1, 0);
            item.0
        }
    }

    #[cfg(test)]
    #[tracing::instrument(skip_all)]
    pub fn to_dense(self) -> crate::poly::dense_mlpoly::DensePolynomial<F> {
        use crate::utils::{math::Math, thread::unsafe_allocate_zero_vec};

        let mut evals = unsafe_allocate_zero_vec(self.num_vars.pow2());

        for (value, index) in self.Z {
            evals[index] = value;
        }

        crate::poly::dense_mlpoly::DensePolynomial::new(evals)
    }
}

pub struct SparseTripleIterator<'a, F: JoltField> {
    dense_index: usize,
    end_index: usize,
    a: &'a [(F, usize)],
    b: &'a [(F, usize)],
    c: &'a [(F, usize)],
}

impl<'a, F: JoltField> SparseTripleIterator<'a, F> {
    #[tracing::instrument(skip_all)]
    pub fn chunks(
        a: &'a SparsePolynomial<F>,
        b: &'a SparsePolynomial<F>,
        c: &'a SparsePolynomial<F>,
        n: usize,
    ) -> Vec<Self> {
        // Don't chunk for small instances
        let total_len = a.num_vars.pow2();
        if b.Z.len() < n * 2 {
            return vec![SparseTripleIterator {
                dense_index: 0,
                end_index: total_len,
                a: &a.Z,
                b: &b.Z,
                c: &c.Z,
            }];
        }

        // B is assumed most dense. Parallelism depends on evenly distributing B across threads.
        assert!(b.Z.len() >= a.Z.len() && b.Z.len() >= c.Z.len());

        // We'd like to scan over 3 SparsePolynomials (a,b,c) in `n` chunks for parallelism.
        // With dense polynomials we could split directly by index, with SparsePolynomials we don't
        // know the distribution of indices in the polynomials in advance.
        // Further, the dense indices do not match: a[i].dense_index != b[i].dense_index != c[i].dense_index
        // We expect b.len() >> max(a.len(), c.len()), so we'll split b first and use as a guide for (a,c).
        // We'll split it into `n` chunks of roughly even length, but we will not split "sibling" dense indices across
        // chunks as the presence of the pair is relevant to downstream algos.
        // Dense siblings: (0,1), (2,3), ...

        let (b_chunks, mut dense_ranges) = b.chunk_no_split_siblings(n);
        let highest_non_zero = [&a.Z, &b.Z, &c.Z]
            .iter()
            .filter_map(|z| z.last().map(|&(_, index)| index))
            .max()
            .unwrap();
        dense_ranges.last_mut().unwrap().1 = highest_non_zero + 1;
        assert_eq!(b_chunks.len(), n);
        assert_eq!(dense_ranges.len(), n);

        // Create chunks of (a, c) which overlap with b's sparse indices
        let mut a_chunks: Vec<&[(F, usize)]> = vec![&[]; n];
        let mut c_chunks: Vec<&[(F, usize)]> = vec![&[]; n];
        let mut a_sparse_i = 0;
        let mut c_sparse_i = 0;
        let span = tracing::span!(tracing::Level::DEBUG, "a_c_chunking");
        let _enter = span.enter();
        // Using b's dense_ranges as a guide, fill out (a_chunks, c_chunks)
        for (chunk_index, range) in dense_ranges.iter().enumerate().skip(1) {
            // Find the corresponding a, c chunks
            let dense_range_end = range.0;

            if a_sparse_i < a.Z.len() && a.Z[a_sparse_i].1 < dense_range_end {
                let a_start = a_sparse_i;
                // Scan over a until the corresponding dense index is out of range
                while a_sparse_i < a.Z.len() && a.Z[a_sparse_i].1 < dense_range_end {
                    a_sparse_i += 1;
                }

                a_chunks[chunk_index - 1] = &a.Z[a_start..a_sparse_i];
            }

            if c_sparse_i < c.Z.len() && c.Z[c_sparse_i].1 < dense_range_end {
                let c_start = c_sparse_i;
                // Scan over c until the corresponding dense index is out of range
                while c_sparse_i < c.Z.len() && c.Z[c_sparse_i].1 < dense_range_end {
                    c_sparse_i += 1;
                }

                c_chunks[chunk_index - 1] = &c.Z[c_start..c_sparse_i];
            }
        }
        drop(_enter);
        a_chunks[n - 1] = &a.Z[a_sparse_i..];
        c_chunks[n - 1] = &c.Z[c_sparse_i..];

        #[cfg(test)]
        {
            assert_eq!(a_chunks.concat(), a.Z);
            assert_eq!(b_chunks.concat(), b.Z);
            assert_eq!(c_chunks.concat(), c.Z);
        }

        // Assemble the triple iterator objects
        let mut iterators: Vec<SparseTripleIterator<'a, F>> = Vec::with_capacity(n);
        for (((a_chunk, b_chunk), c_chunk), range) in a_chunks
            .iter()
            .zip(b_chunks.iter())
            .zip(c_chunks.iter())
            .zip(dense_ranges.iter())
        {
            #[cfg(test)]
            for chunk in &[a_chunk, b_chunk, c_chunk] {
                for (_, index) in chunk.iter() {
                    assert!(*index >= range.0 && *index <= range.1);
                }
            }

            let iter = SparseTripleIterator {
                dense_index: range.0,
                end_index: range.1,
                a: a_chunk,
                b: b_chunk,
                c: c_chunk,
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
        let match_and_advance = |slice: &mut &[(F, usize)], index: usize| -> F {
            if let Some(first_item) = slice.first() {
                if first_item.1 == index {
                    let ret = first_item.0;
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

        (
            low_index,
            a_lower_val,
            a_upper_val,
            b_lower_val,
            b_upper_val,
            c_lower_val,
            c_upper_val,
        )
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
    use crate::poly::dense_mlpoly::DensePolynomial;

    use super::*;
    use ark_bn254::Fr;
    use ark_std::Zero;

    #[test]
    fn sparse_bound_bot_all_left() {
        let dense_evals = vec![Fr::from(10), Fr::zero(), Fr::from(20), Fr::zero()];
        let sparse_evals = vec![(Fr::from(10), 0), (Fr::from(20), 2)];

        let mut dense = DensePolynomial::new(dense_evals);
        let mut sparse = SparsePolynomial::new(2, sparse_evals);

        assert_eq!(sparse.clone().to_dense(), dense);

        let r = Fr::from(121);
        sparse.bound_poly_var_bot(&r);
        dense.bound_poly_var_bot_01_optimized(&r);
        assert_eq!(sparse.to_dense(), dense);
    }

    #[test]
    fn sparse_bound_bot_all_right() {
        let dense_evals = vec![Fr::zero(), Fr::from(10), Fr::zero(), Fr::from(20)];
        let sparse_evals = vec![(Fr::from(10), 1), (Fr::from(20), 3)];

        let mut dense = DensePolynomial::new(dense_evals);
        let mut sparse = SparsePolynomial::new(2, sparse_evals);

        assert_eq!(sparse.clone().to_dense(), dense);

        let r = Fr::from(121);
        sparse.bound_poly_var_bot(&r);
        dense.bound_poly_var_bot_01_optimized(&r);
        assert_eq!(sparse.to_dense(), dense);
    }

    #[test]
    fn sparse_bound_bot_mixed() {
        let dense_evals = vec![
            Fr::zero(),
            Fr::from(10),
            Fr::zero(),
            Fr::from(20),
            Fr::from(30),
            Fr::from(40),
            Fr::zero(),
            Fr::from(50),
        ];
        let sparse_evals = vec![
            (Fr::from(10), 1),
            (Fr::from(20), 3),
            (Fr::from(30), 4),
            (Fr::from(40), 5),
            (Fr::from(50), 7),
        ];

        let mut dense = DensePolynomial::new(dense_evals);
        let mut sparse = SparsePolynomial::new(3, sparse_evals);

        assert_eq!(sparse.clone().to_dense(), dense);

        let r = Fr::from(121);
        sparse.bound_poly_var_bot(&r);
        dense.bound_poly_var_bot_01_optimized(&r);
        assert_eq!(sparse.to_dense(), dense);
    }

    #[test]
    fn sparse_triple_iterator() {
        let a = vec![(Fr::from(9), 9), (Fr::from(10), 10), (Fr::from(12), 12)];
        let b = vec![
            (Fr::from(100), 0),
            (Fr::from(1), 1),
            (Fr::from(2), 2),
            (Fr::from(3), 3),
            (Fr::from(4), 4),
            (Fr::from(5), 5),
            (Fr::from(6), 6),
            (Fr::from(7), 7),
            (Fr::from(8), 8),
            (Fr::from(9), 9),
            (Fr::from(10), 10),
            (Fr::from(11), 11),
            (Fr::from(12), 12),
            (Fr::from(13), 13),
            (Fr::from(14), 14),
            (Fr::from(15), 15),
        ];
        let c = vec![(Fr::from(12), 0), (Fr::from(3), 3)];

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
        let num_vars = 5;
        let total_len = 1 << num_vars;

        let mut a = vec![];
        let mut b = vec![];
        let mut c = vec![];

        for i in 0usize..total_len {
            if rng.gen::<f64>() < prob_exists {
                a.push((Fr::from(i as u64), i));
            }
            if rng.gen::<f64>() < prob_exists * 2f64 {
                b.push((Fr::from(i as u64), i));
            }
            if rng.gen::<f64>() < prob_exists {
                c.push((Fr::from(i as u64), i));
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
                let (dense_index, a_low, a_high, b_low, b_high, c_low, c_high) =
                    iterator.next_pairs();

                new_a[dense_index] = a_low;
                new_a[dense_index + 1] = a_high;

                new_b[dense_index] = b_low;
                new_b[dense_index + 1] = b_high;

                new_c[dense_index] = c_low;
                new_c[dense_index + 1] = c_high;

                assert_eq!(dense_index, expected_dense_index);
                expected_dense_index += 2;
            }
        }

        assert_eq!(a_poly.to_dense().Z, new_a);
        assert_eq!(b_poly.to_dense().Z, new_b);
        assert_eq!(c_poly.to_dense().Z, new_c);
    }

    #[test]
    fn binding() {
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let prob_exists = 0.32;
        let num_vars = 6;
        let total_len = 1 << num_vars;
        let mut a = vec![];
        for i in 0usize..total_len {
            if rng.gen::<f64>() < prob_exists {
                a.push((Fr::from(i as u64), i));
            }
        }

        let a_poly = SparsePolynomial::new(num_vars, a);
        let r = Fr::from(100);

        let mut regular = a_poly.clone();
        regular.bound_poly_var_bot(&r);

        let mut par = a_poly.clone();
        par.bound_poly_var_bot(&r);
        assert_eq!(regular, par);
    }
}
