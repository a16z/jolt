use ark_ff::PrimeField;

use crate::{poly::dense_mlpoly::DensePolynomial, utils::math::Math};


#[derive(Debug, Clone)]
pub struct SparseEntry<F> {
    value: F,
    index: usize
}

impl<F> SparseEntry<F> {
    pub fn new(value: F, index: usize) -> Self {
        Self { value, index }
    }
}

#[derive(Debug, Clone)]
pub struct SparsePoly<F> {
    low_entries: Vec<SparseEntry<F>>,
    high_entries: Vec<SparseEntry<F>>,

    num_vars: usize,
    dense_len: usize,

    low_sparse_index: usize,
    high_sparse_index: usize
}

impl<F> std::ops::Index<usize> for SparsePoly<F> {
    type Output = SparseEntry<F>;

    fn index(&self, index: usize) -> &Self::Output {
        if index < self.low_entries.len() {
            &self.low_entries[index]
        } else {
            &self.high_entries[index - self.low_entries.len()]
        }
    }
}

impl<F: PrimeField> SparsePoly<F> {
    pub fn new(low_entries: Vec<SparseEntry<F>>, high_entries: Vec<SparseEntry<F>>, dense_len: usize) -> Self {
        let mid = dense_len / 2;
        assert!(low_entries.len() < dense_len);
        assert!(high_entries.len() < dense_len);
        let num_vars = dense_len.log_2();

        Self { low_entries, high_entries, num_vars, dense_len, low_sparse_index: 0, high_sparse_index: 0 }
    }

    pub fn final_eval(&self) -> F {
        assert_eq!(self.low_entries.len(), 1);
        assert_eq!(self.high_entries.len(), 0);
        assert_eq!(self.num_vars, 0);
        let entry = self.low_entries[0].clone();
        assert_eq!(entry.index, 0 );
        entry.value
    }

    pub fn mid(&self) -> usize {
        self.dense_len / 2
    }

    #[inline]
    pub fn low_high_iter(&mut self, index: usize) -> (Option<&F>, Option<&F>) {
        assert!(index < self.mid());

        let low = if self.low_sparse_index < self.low_entries.len() {
            let entry = &self.low_entries[self.low_sparse_index];
            if entry.index == index {
                self.low_sparse_index += 1;
                Some(&entry.value)
            } else {
                None
            }
        } else {
            None
        };

        let high = if self.high_sparse_index < self.high_entries.len() {
            let entry = &self.high_entries[self.high_sparse_index];
            if entry.index == index {
                self.high_sparse_index += 1;
                Some(&entry.value)
            } else {
                None
            }
        } else {
            None
        };

        (low, high)
    }

    // TODO(sragss): RM IN FAVOR OF ITER
    pub fn reset_iter(&mut self) {
        self.low_sparse_index = 0;
        self.high_sparse_index = 0;
    }

    pub fn bound_poly_var_top(&mut self, r: &F) {
        let span_alloc = tracing::span!(tracing::Level::TRACE, "bound::allocation");
        let _enter_alloc = span_alloc.enter();

        let num_entries = std::cmp::max(self.low_entries.len(), self.high_entries.len());
        let mut new_low_entries: Vec<SparseEntry<F>> = Vec::with_capacity(num_entries);
        let mut new_high_entries: Vec<SparseEntry<F>> = Vec::with_capacity(num_entries);
        drop(_enter_alloc);
        drop(span_alloc);


        let span = tracing::span!(tracing::Level::TRACE, "bound::inner_loop");
        let _enter = span.enter();
        for i in 0..self.mid() {
            let new_value = match self.low_high_iter(i){
                (None, None) => continue,
                (Some(low), None) => {
                    let m = F::one() - low;
                    *low + m * r
                },
                (None, Some(high)) => {
                    let m = *high - F::one();
                    F::one() + m * r
                },
                (Some(low), Some(high)) => {
                    let m = *high - low;
                    *low + m * r
                },
            };

            if i < self.mid() / 2  || self.mid() == 1 {
                new_low_entries.push(SparseEntry { value: new_value, index: i});
            } else {
                let index = i - self.mid() / 2;
                new_high_entries.push(SparseEntry { value: new_value, index });
            }
        }

        drop(_enter);
        drop(span);

        self.low_entries = new_low_entries;
        self.high_entries = new_high_entries;
        self.num_vars -= 1;
        self.dense_len /= 2;
        self.low_sparse_index = 0;
        self.high_sparse_index = 0;
    }

    pub fn sparse_len(&self) -> usize {
        self.low_entries.len() + self.high_entries.len()
    }

    pub fn to_dense(self) -> DensePolynomial<F> {
        let half = self.mid();
        let mut dense_evals: Vec<F> = vec![F::one(); self.dense_len];
        for low_entry in self.low_entries {
            dense_evals[low_entry.index] = low_entry.value;
        } 
        for high_entry in self.high_entries {
            dense_evals[high_entry.index + half] = high_entry.value;
        } 

        DensePolynomial::new(dense_evals)
    }
}

#[derive(Debug, Clone)]
pub struct SparseGrandProductCircuit<F> {
    left: Vec<SparsePoly<F>>,
    right: Vec<SparsePoly<F>>,
}

impl<F: PrimeField> SparseGrandProductCircuit<F> {
    pub fn construct(leaves: Vec<F>, flags: Vec<bool>) -> Self {
        let num_leaves = leaves.len();
        let num_layers = num_leaves.log_2(); 
        let leaf_half = num_leaves / 2;

        let mut lefts: Vec<SparsePoly<F>> = Vec::with_capacity(num_layers);
        let mut rights: Vec<SparsePoly<F>> = Vec::with_capacity(num_layers);

        // TODO(sragss): Attempt rough capacity planning here? We could construct metadata from initial flag scan.
        let mut left_low: Vec<SparseEntry<F>> = Vec::new();
        let mut left_high: Vec<SparseEntry<F>> = Vec::new();
        let mut right_low: Vec<SparseEntry<F>> = Vec::new();
        let mut right_high: Vec<SparseEntry<F>> = Vec::new();

        // First layer
        for leaf_index in 0..num_leaves {
            if flags[leaf_index] { 
                if leaf_index < leaf_half / 2 {
                    left_low.push(SparseEntry::new(leaves[leaf_index], leaf_index));
                } else if leaf_index < leaf_half {
                    left_high.push(SparseEntry::new(leaves[leaf_index], leaf_index));
                } else if leaf_index < leaf_half + (leaf_half / 2) {
                    right_low.push(SparseEntry::new(leaves[leaf_index], leaf_index));
                } else {
                    right_high.push(SparseEntry::new(leaves[leaf_index], leaf_index));
                }
            }
        }
        lefts.push(SparsePoly::new(left_low, left_high, leaf_half));
        rights.push(SparsePoly::new(right_low, right_high, leaf_half));

        let mut layer_len = num_leaves;
        for layer in 0..num_layers - 1 {
            let (left, right) = 
                Self::compute_layer(&lefts[layer], &rights[layer], layer_len);

            lefts.push(left);
            rights.push(right);

            layer_len /= 2;
        }

        Self { 
            left: lefts, 
            right: rights
        }
    }

    fn compute_layer(
            prior_left: &SparsePoly<F>, 
            prior_right: &SparsePoly<F>, 
            prior_len: usize) -> (SparsePoly<F>, SparsePoly<F>) {
        // Capacity has the potential to overshoot by a factor 2
        let max_capacity = prior_left.sparse_len();
        let mut left_low: Vec<SparseEntry<F>> = Vec::with_capacity(max_capacity);
        let mut left_high: Vec<SparseEntry<F>> = Vec::with_capacity(max_capacity);
        let max_capacity = prior_right.sparse_len();
        let mut right_low: Vec<SparseEntry<F>> = Vec::with_capacity(max_capacity);
        let mut right_high: Vec<SparseEntry<F>> = Vec::with_capacity(max_capacity);

        let mut left_sparse_index: usize = 0;
        let mut right_sparse_index: usize = 0;

        while left_sparse_index < prior_left.sparse_len() && right_sparse_index < prior_right.sparse_len() {
            // Mere existence of these indices means they're "non-sparse": not equal to 1.
            let left_index = prior_left[left_sparse_index].index;
            let right_index = prior_right[right_sparse_index].index;

            todo!("sam deal with the index updates");

            let entry = if left_index == right_index {
                let value = prior_left[left_sparse_index].value * prior_right[right_sparse_index].value;
                left_sparse_index += 1;
                right_sparse_index += 1;
                SparseEntry::new(value, left_index)
            } else if left_index < right_index {
                let entry = prior_left[left_sparse_index].clone();
                left_sparse_index += 1;
                entry
            } else if right_index < left_index {
                let entry = prior_right[right_sparse_index].clone();
                right_sparse_index += 1;
                entry
            } else {
                unreachable!();
            };

            // prior_left.dense_len() + prior_right.dense_len() == prior_len
            // dense_left.len() == dense_right.len() == prior_len / 4
            // dense_left_low.len() == dense_left_high.len() == ... == prior.len / 8
            if entry.index < prior_len / 8 {
                left_low.push(entry);
            } else if entry.index < prior_len / 4 {
                left_high.push(entry);
            } else if entry.index < (prior_len / 4) + (prior_len / 8) {
                right_low.push(entry);
            } else {
                right_high.push(entry);
            }
        }
        let new_len = prior_len / 2;
        let left = SparsePoly::new(left_low, left_high, new_len);
        let right = SparsePoly::new(right_low, right_high, new_len);
        (left, right)
    }
}


#[cfg(test)]
mod tests {
    use super::{*, bench::init_bind_bench};
    use ark_curve25519::Fr;
    use ark_std::{One, test_rng};

    #[test]
    fn sparse_conversion() {
        let dense = DensePolynomial::new(vec![Fr::from(4), Fr::from(5), Fr::from(6), Fr::from(7)]);
        let sparse = SparsePoly::new(
            vec![
                SparseEntry::new(Fr::from(4), 0),
                SparseEntry::new(Fr::from(5), 1),
            ],
            vec![
                SparseEntry::new(Fr::from(6), 0),
                SparseEntry::new(Fr::from(7), 1),
            ],
            4);
        assert_eq!(dense, sparse.to_dense());

        let dense = DensePolynomial::new(vec![Fr::one(), Fr::one(), Fr::from(3), Fr::one()]);
        let sparse = SparsePoly::new(vec![], vec![SparseEntry::new(Fr::from(3), 0)], 4);
        assert_eq!(dense, sparse.to_dense());

        let dense = DensePolynomial::new(vec![Fr::from(2), Fr::from(2), Fr::one(), Fr::one()]);
        let sparse = SparsePoly::new(vec![SparseEntry::new(Fr::from(2), 0), SparseEntry::new(Fr::from(2), 1)], vec![], 4);
        assert_eq!(dense, sparse.to_dense());

        let dense = DensePolynomial::new(vec![Fr::one(), Fr::one(), Fr::from(8), Fr::from(8)]);
        let sparse = SparsePoly::new(vec![], vec![SparseEntry::new(Fr::from(8), 0), SparseEntry::new(Fr::from(8), 1)], 4);
        assert_eq!(dense, sparse.to_dense());
    }

    #[test]
    fn bound_poly_var_top() {
        let mut dense = DensePolynomial::new(vec![Fr::from(4), Fr::from(5), Fr::from(6), Fr::from(7)]);
        let mut sparse = SparsePoly::new(
            vec![
                SparseEntry::new(Fr::from(4), 0),
                SparseEntry::new(Fr::from(5), 1),
            ],
            vec![
                SparseEntry::new(Fr::from(6), 0),
                SparseEntry::new(Fr::from(7), 1),
            ],
            4);
        assert_eq!(dense, sparse.clone().to_dense());
        let r = Fr::from(12);
        dense.bound_poly_var_top(&r);
        sparse.bound_poly_var_top(&r);
        assert_eq!(dense.evals_ref()[0..2], sparse.clone().to_dense().evals_ref()[0..2]);
    }

    #[test]
    fn bound_poly_var_top_sparse_left() {
        let mut dense = DensePolynomial::new(vec![Fr::from(1), Fr::from(1), Fr::from(6), Fr::from(7)]);
        let mut sparse = SparsePoly::new(
            vec![], 
            vec![
                SparseEntry::new(Fr::from(6), 0),
                SparseEntry::new(Fr::from(7), 1)
            ], 4);
        assert_eq!(dense, sparse.clone().to_dense());
        let r = Fr::from(12);
        dense.bound_poly_var_top(&r);
        sparse.bound_poly_var_top(&r);
        assert_eq!(dense.evals_ref()[0..2], sparse.clone().to_dense().evals_ref()[0..2]);
    }

    #[test]
    fn bound_poly_var_top_sparse_right() {
        let mut dense = DensePolynomial::new(vec![Fr::from(4), Fr::from(5), Fr::from(1), Fr::from(1)]);
        let mut sparse = SparsePoly::new(
            vec![
                SparseEntry::new(Fr::from(4), 0),
                SparseEntry::new(Fr::from(5), 1),
            ],
            vec![
            ], 4);
        assert_eq!(dense, sparse.clone().to_dense());
        let r = Fr::from(12);
        dense.bound_poly_var_top(&r);
        sparse.bound_poly_var_top(&r);
        assert_eq!(dense.evals_ref()[0..2], sparse.clone().to_dense().evals_ref()[0..2]);
    }

    #[test]
    fn random() {
        use ark_std::UniformRand;
        use ark_std::{test_rng, rand::Rng};

        let log_size = 8;
        let size = 1 << log_size;
        let mut sparse = init_bind_bench(log_size, 0.93);
        let mut dense = sparse.clone().to_dense();

        let mut rng = test_rng();
        let r_1 = Fr::rand(&mut rng);
        let r_2 = Fr::rand(&mut rng);

        sparse.bound_poly_var_top(&r_1);
        sparse.bound_poly_var_top(&r_2);
        dense.bound_poly_var_top(&r_1);
        dense.bound_poly_var_top(&r_2);

        assert_eq!(dense.evals_ref()[0..(size/4)], sparse.to_dense().evals_ref()[0..(size/4)]);
    }

    #[test]
    fn full_bind() {
        use ark_std::UniformRand;
        use ark_std::{test_rng, rand::Rng};

        let log_size = 3;
        let size = 1 << log_size;
        let mut sparse = init_bind_bench(log_size, 0.93);
        let mut dense = sparse.clone().to_dense();

        let mut rng = test_rng();
        let r_1 = Fr::rand(&mut rng);
        let r_2 = Fr::rand(&mut rng);
        let r_3 = Fr::rand(&mut rng);

        sparse.bound_poly_var_top(&r_1);
        sparse.bound_poly_var_top(&r_2);
        sparse.bound_poly_var_top(&r_3);
        dense.bound_poly_var_top(&r_1);
        dense.bound_poly_var_top(&r_2);
        dense.bound_poly_var_top(&r_3);

        assert_eq!(dense[0], sparse.final_eval());
    }
}

pub mod bench {
    use super::*;
    use ark_std::{test_rng, rand::Rng};

    pub fn init_bind_bench<F: PrimeField>(log_size: usize, pct_ones: f64) -> SparsePoly<F> {
        let mut rng = test_rng();

        let size = 1usize << log_size;
        let half = size / 2;

        let mut low_entries = Vec::new();
        let mut high_entries = Vec::new();

        for i in 0..half {
            if rng.gen::<f64>() > pct_ones {
                low_entries.push(SparseEntry::new(F::rand(&mut rng), i));
            }
        }
        for i in 0..half {
            if rng.gen::<f64>() > pct_ones {
                high_entries.push(SparseEntry::new(F::rand(&mut rng), i));
            }
        }
        SparsePoly::new(low_entries, high_entries, size)
    }
}