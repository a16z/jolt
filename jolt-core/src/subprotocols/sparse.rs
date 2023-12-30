use ark_ff::PrimeField;

use crate::{poly::dense_mlpoly::DensePolynomial, utils::math::Math};

#[derive(Debug, Clone)]
pub struct SparseEntry<F> {
    index: usize,
    value: F
}

#[derive(Debug, Clone)]
pub struct SparsePoly<F> {
    entries: Vec<SparseEntry<F>>,

    len: usize,

    num_vars: usize,

    /// Start of "high values"
    upper_index: usize 
}

impl<F: PrimeField> SparsePoly<F> {
    pub fn bound_poly_var_top(&mut self, r: &F) {
        let mut low_sparse_index: usize = 0;
        let mut high_sparse_index: usize = 0;
        let mut new_entries = Vec::with_capacity(self.entries.len()); // May overshoot by a factor of 2

        // TODO(sragss): Figure out how to swap in place. Queue system?

        while low_sparse_index < self.upper_index && high_sparse_index < self.len {
            // Mere existence of these indices means they're "non-sparse": not equal to 1.
            let bottom_index = self.entries[low_sparse_index].index;
            let top_index = self.entries[high_sparse_index].index;

            if bottom_index == top_index {
                let m = self.entries[low_sparse_index].value - self.entries[low_sparse_index].value;
                let value: F = self.entries[low_sparse_index].value + *r * m;
                // TODO: index, take
                new_entries.push(value);
                low_sparse_index += 1;
                high_sparse_index += 1;
            } else if low_sparse_index < high_sparse_index {
                // TODO: index, take
                new_entries.push(self.entries[low_sparse_index].clone()); 
                low_sparse_index += 1;
            } else if high_sparse_index < low_sparse_index {
                // TODO: index = high_sparse_index / 2, take
                new_entries.push(self.entries[high_sparse_index].clone());
                high_sparse_index += 1;
            } else {
                unreachable!();
            }
        }
        self.entries = new_entries;
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

        DensePolynomial::new(dense_evals)
    }
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