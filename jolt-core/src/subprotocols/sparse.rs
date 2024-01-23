use ark_ff::PrimeField;

use crate::{poly::dense_mlpoly::DensePolynomial, utils::math::Math};

#[derive(Debug, Clone)]
pub struct SparseEntry<F> {
    index: usize,
    value: F
}

impl<F: PrimeField> SparseEntry<F> {
    fn new(value: F, index: usize) -> Self {
        SparseEntry {
            index,
            value
        }
    }
}

#[derive(Debug, Clone)]
pub struct SparsePoly<F> {
    entries: Vec<SparseEntry<F>>,

    num_vars: usize,

    /// Start of 'high values' where index > 50%
    upper_index: usize 
}

pub struct SparsePolyIter<'a, F> {
    poly: &'a SparsePoly<F>,
    low_sparse_index: usize,
    high_sparse_index: usize,
}

impl<'a, F: PrimeField> SparsePolyIter<'a, F> {
    fn new(poly: &'a SparsePoly<F>) -> Self {
        SparsePolyIter {
            poly,
            low_sparse_index: 0,
            high_sparse_index: poly.upper_index
        }
    }

    fn has_next(&self) -> bool {
        self.low_sparse_index < self.poly.upper_index || self.high_sparse_index < self.poly.entries.len()
    }

    fn next_unchecked(&mut self) -> (Option<&SparseEntry<F>>, Option<&SparseEntry<F>>) {
        let index_half = self.index_half();
        let low_sparse_index = self.low_sparse_index;
        let high_sparse_index = self.high_sparse_index;

        // Base cases, one half is exhausted.
        if low_sparse_index >= self.poly.upper_index {
            let result = (None, Some(&self.poly.entries[high_sparse_index]));
            self.high_sparse_index += 1;
            return result;
        } else if high_sparse_index >= self.poly.entries.len() {
            let result = (Some(&self.poly.entries[low_sparse_index]), None);
            self.low_sparse_index += 1;
            return result;
        }

        let low_index = self.poly.entries[low_sparse_index].index;
        let high_index = self.poly.entries[high_sparse_index].index;

        if low_index == (high_index - index_half) {
            let result = (Some(&self.poly.entries[low_sparse_index]), Some(&self.poly.entries[high_sparse_index]));
            self.low_sparse_index += 1;
            self.high_sparse_index += 1;
            return result;
        } else if low_index < (high_index - index_half) {
            let result = (Some(&self.poly.entries[low_sparse_index]), None);
            self.low_sparse_index += 1;
            return result;
        } else if (high_index - index_half) < low_index {
            let result = (None, Some(&self.poly.entries[low_sparse_index]));
            self.high_sparse_index += 1;
            return result;
        } else {
            unreachable!("plz")
        }
    }

    fn index_half(&self) -> usize {
        (1 << self.poly.num_vars) / 2
    }
}

impl<F: PrimeField> SparsePoly<F> {
    pub fn new(entries: Vec<SparseEntry<F>>, num_vars: usize, upper_index: usize) -> Self {
        Self { entries, num_vars, upper_index }
    }
    pub fn bound_poly_var_top(&mut self, r: &F) {
        // `sparse_index` vars are [0, ... 2^num_vars]
        // `index` vars are [0, ... 2^entries.len()]
        let mut low_sparse_index: usize = 0;
        let mut high_sparse_index: usize = self.upper_index;

        let index_half: usize = (1 << self.num_vars) / 2;
        let new_index_half = index_half / 2;
        let mut new_entries = Vec::with_capacity(self.entries.len()); // May overshoot by a factor of 2
        let mut new_upper_sparse_index: Option<usize> = None;

        // TODO(sragss): Figure out how to swap in place. Queue system?
        // TODO(sragss): upper_index compute doesn't work.

        while low_sparse_index < self.upper_index || high_sparse_index < self.entries.len() {
            // Mere existence of these indices means they're "non-sparse": not equal to 1.
            let low_index = self.entries[low_sparse_index].index;
            let high_index = self.entries[high_sparse_index].index;

            if low_index == (high_index - index_half) {
                // If indices match, the parent has a non-sparse low and high
                let m = self.entries[high_sparse_index].value - self.entries[low_sparse_index].value;
                let value: F = self.entries[low_sparse_index].value + *r * m;
                let entry = SparseEntry::new(value, low_index);
                new_entries.push(entry);
                low_sparse_index += 1;
                high_sparse_index += 1;

                if new_upper_sparse_index.is_none() && low_index >= new_index_half {
                    new_upper_sparse_index = Some(new_entries.len() - 1);
                }
            } else if low_index < (high_index - index_half) {
                new_entries.push(self.entries[low_sparse_index].clone()); 
                low_sparse_index += 1;

                if new_upper_sparse_index.is_none() && low_index >= new_index_half {
                    new_upper_sparse_index = Some(new_entries.len() - 1);
                }
            } else if (high_index - index_half) < low_index { // TODO(sragss): Do by default and gate on low_sparse_index < self.upper_index
                let mut entry = self.entries[high_sparse_index].clone();
                entry.index = high_index / 2;
                new_entries.push(entry);
                high_sparse_index += 1;

                if new_upper_sparse_index.is_none() && high_index >= new_index_half {
                    new_upper_sparse_index = Some(new_entries.len() - 1);
                }
            } else {
                unreachable!();
            }
        }
        self.entries = new_entries;
        self.num_vars -= 1;
        self.upper_index = new_upper_sparse_index.unwrap_or(self.entries.len() - 1);
    }

    pub fn bound_poly_var_top_iter(&mut self, r: &F) {
        let mut iter = SparsePolyIter::new(&self);

        let mut new_entries = Vec::with_capacity(self.entries.len()); // may overshoot by factor 2
        let mut new_upper_sparse_index: Option<usize> = None;

        while iter.has_next() {
            let (low, high) = iter.next_unchecked();

            match (low, high) {
                (Some(low_entry), Some(high_entry)) => {
                    let m = high_entry.value - low_entry.value;
                    let value: F = low_entry.value + *r * m;
                    let entry = SparseEntry::new(value, low_entry.index);
                    new_entries.push(entry);
                },
                (Some(low_entry), None) => {
                    let m = F::one() - low_entry.value;
                    let value: F = low_entry.value + *r * m;
                    let entry = SparseEntry::new(value, low_entry.index);
                    new_entries.push(entry); 
                },
                (None, Some(high_entry)) => {
                    let m = high_entry.value - F::one();
                    let value: F = F::one() + *r * m;
                    let index = high_entry.index - iter.index_half();
                    let entry = SparseEntry::new(value, index);
                    new_entries.push(entry);
                },
                _ => {
                    unreachable!("plz");
                }
            }
        }
        self.entries = new_entries;
        self.num_vars -= 1;
        self.upper_index = new_upper_sparse_index.unwrap_or(self.entries.len() - 1);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn to_dense(self) -> DensePolynomial<F> {
        let n = 1 << self.num_vars;

        let mut dense_evals: Vec<F> = Vec::with_capacity(n);

        let mut curr_index = 0;
        for entry in self.entries.iter() {
            for i in (curr_index..entry.index) {
                dense_evals.push(F::one());
            }
            dense_evals.push(entry.value);
            curr_index = entry.index + 1;
        }
        for i in curr_index..n {
            dense_evals.push(F::one());
        }

        DensePolynomial::new(dense_evals)
    }
}

#[derive(Debug, Clone)]
pub struct SparseGrandProductCircuit<F> {
    left_vec: Vec<Vec<SparseEntry<F>>>,
    right_vec: Vec<Vec<SparseEntry<F>>>,
}

impl<F: PrimeField> SparseGrandProductCircuit<F> {
    pub fn construct(leaves: Vec<F>, flags: Vec<bool>) -> Self {
        let num_leaves = leaves.len();
        let num_layers = num_leaves.log_2(); 
        let leaf_half = num_leaves / 2;

        let mut lefts: Vec<Vec<SparseEntry<F>>> = Vec::with_capacity(num_layers);
        let mut rights: Vec<Vec<SparseEntry<F>>> = Vec::with_capacity(num_layers);

        // TODO(sragss): Attempt rough capacity planning here? We could construct metadata from initial flag scan.
        let mut left: Vec<SparseEntry<F>> = Vec::new();
        let mut right: Vec<SparseEntry<F>> = Vec::new();

        // First layer
        for leaf_index in 0..num_leaves {
            if flags[leaf_index] { 
                if leaf_index < leaf_half {
                    left.push(SparseEntry::new(leaves[leaf_index], leaf_index));
                } else {
                    right.push(SparseEntry::new(leaves[leaf_index], leaf_index - leaf_half));
                }
            }
        }
        lefts.push(left);
        rights.push(right);

        let mut layer_len = num_leaves;
        for layer in 0..num_layers - 1 {
            let (left, right) = 
                Self::compute_layer(&lefts[layer], &rights[layer], layer_len);

            lefts.push(left);
            rights.push(right);

            layer_len /= 2;
        }

        Self { 
            left_vec: lefts, 
            right_vec: rights
        }
    }

    fn compute_layer(
            prior_left: &Vec<SparseEntry<F>>, 
            prior_right: &Vec<SparseEntry<F>>, 
            prior_len: usize) -> (Vec<SparseEntry<F>>, Vec<SparseEntry<F>>) {
        // TODO(sragss): Attempt capacity planning?
        let mut left: Vec<SparseEntry<F>> = Vec::new();
        let mut right: Vec<SparseEntry<F>> = Vec::new();

        let mut left_sparse_index: usize = 0;
        let mut right_sparse_index: usize = 0;

        while left_sparse_index < prior_left.len() && right_sparse_index < prior_right.len() {
            // Mere existence of these indices means they're "non-sparse": not equal to 1.
            let left_index = prior_left[left_sparse_index].index;
            let right_index = prior_right[right_sparse_index].index;

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

            if entry.index < prior_len / 4 {
                left.push(entry);
            } else {
                right.push(entry);
            }
        }
        (left, right)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ark_curve25519::Fr;
    use ark_std::One;

    #[test]
    fn sparse_conversion() {
        let dense = DensePolynomial::new(vec![Fr::from(4), Fr::from(5), Fr::from(6), Fr::from(7)]);
        let sparse = SparsePoly::new(
            vec![
                SparseEntry::new(Fr::from(4), 0),
                SparseEntry::new(Fr::from(5), 1),
                SparseEntry::new(Fr::from(6), 2),
                SparseEntry::new(Fr::from(7), 3),
            ], 2, 2);
        assert_eq!(dense, sparse.to_dense());

        let dense = DensePolynomial::new(vec![Fr::one(), Fr::one(), Fr::from(3), Fr::one()]);
        let sparse = SparsePoly::new(vec![SparseEntry::new(Fr::from(3), 2)], 2, 0);
        assert_eq!(dense, sparse.to_dense());

        let dense = DensePolynomial::new(vec![Fr::from(2), Fr::from(2), Fr::one(), Fr::one()]);
        let sparse = SparsePoly::new(vec![SparseEntry::new(Fr::from(2), 0), SparseEntry::new(Fr::from(2), 1)], 2, 2);
        assert_eq!(dense, sparse.to_dense());

        let dense = DensePolynomial::new(vec![Fr::one(), Fr::one(), Fr::from(8), Fr::from(8)]);
        let sparse = SparsePoly::new(vec![SparseEntry::new(Fr::from(8), 2), SparseEntry::new(Fr::from(8), 3)], 2, 0);
        assert_eq!(dense, sparse.to_dense());
    }

    #[test]
    fn bound_poly_var_top() {
        let mut dense = DensePolynomial::new(vec![Fr::from(4), Fr::from(5), Fr::from(6), Fr::from(7)]);
        let mut sparse = SparsePoly::new(
            vec![
                SparseEntry::new(Fr::from(4), 0),
                SparseEntry::new(Fr::from(5), 1),
                SparseEntry::new(Fr::from(6), 2),
                SparseEntry::new(Fr::from(7), 3),
            ], 2, 2);
        assert_eq!(dense, sparse.clone().to_dense());
        let r = Fr::from(12);
        dense.bound_poly_var_top(&r);
        sparse.bound_poly_var_top_iter(&r);
        assert_eq!(dense.evals_ref()[0..2], sparse.clone().to_dense().evals_ref()[0..2]);
    }

    #[test]
    fn bound_poly_var_top_sparse_left() {
        let mut dense = DensePolynomial::new(vec![Fr::from(1), Fr::from(1), Fr::from(6), Fr::from(7)]);
        let mut sparse = SparsePoly::new(
            vec![
                SparseEntry::new(Fr::from(6), 2),
                SparseEntry::new(Fr::from(7), 3),
            ], 2, 0);
        assert_eq!(dense, sparse.clone().to_dense());
        let r = Fr::from(12);
        dense.bound_poly_var_top(&r);
        sparse.bound_poly_var_top_iter(&r);
        assert_eq!(dense.evals_ref()[0..2], sparse.clone().to_dense().evals_ref()[0..2]);
    }

    #[test]
    fn bound_poly_var_top_sparse_right() {
        let mut dense = DensePolynomial::new(vec![Fr::from(4), Fr::from(5), Fr::from(1), Fr::from(1)]);
        let mut sparse = SparsePoly::new(
            vec![
                SparseEntry::new(Fr::from(4), 0),
                SparseEntry::new(Fr::from(5), 1),
            ], 2, 2);
        assert_eq!(dense, sparse.clone().to_dense());
        let r = Fr::from(12);
        dense.bound_poly_var_top(&r);
        sparse.bound_poly_var_top_iter(&r);
        assert_eq!(dense.evals_ref()[0..2], sparse.clone().to_dense().evals_ref()[0..2]);
    }
}

pub mod bench {
    use super::*;
    use ark_std::UniformRand;
    use ark_std::{test_rng, rand::Rng};

    pub fn init_bind_bench<F: PrimeField>(log_size: usize, pct_ones: f64) -> SparsePoly<F> {
        let mut rng = test_rng();

        let size = 1 << log_size;
        let mut entries = Vec::new();
        for i in 0..size {
            if rng.gen::<f64>() > pct_ones {
                entries.push(SparseEntry::new(F::rand(&mut rng), i));
            }
        }
        let max_index = entries.len() - 1;
        SparsePoly::new(entries, log_size, (max_index as f64 * 0.5) as usize)
    }
}