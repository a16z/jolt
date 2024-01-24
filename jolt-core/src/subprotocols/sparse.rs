use ark_ff::PrimeField;

use crate::{poly::dense_mlpoly::DensePolynomial, utils::math::Math};

#[derive(Debug, Clone)]
pub struct SparsePoly<F> {
    pub entries: Vec<F>,
    indices: Vec<Option<u32>>,

    num_vars: usize,
    mid: usize
}

impl<F: PrimeField> SparsePoly<F> {
    pub fn new(entries: Vec<F>, indices: Vec<Option<u32>>, num_vars: usize) -> Self {
        let mut interleaved_indices = vec![None; indices.len()];
        let (first_half, second_half) = indices.split_at(indices.len() / 2);
        for (i, (first, second)) in first_half.iter().zip(second_half.iter()).enumerate() {
            interleaved_indices[2 * i] = *first;
            interleaved_indices[2 * i + 1] = *second;
        }
        let indices = interleaved_indices;
        assert_eq!(indices.len(), 1 << num_vars);
        let mid = if num_vars > 0 { 1 << (num_vars - 1) } else { 0 };
        Self { entries, indices, num_vars, mid }
    }

    pub fn low_high_iter(&self, index: usize) -> (Option<&F>, Option<&F>) {
        assert!(index < self.mid);
        let low_i = index;
        let high_i = index + self.mid;

        let low = self.indices[low_i].and_then(|index| Some(&self.entries[index as usize]));
        let high = self.indices[high_i].and_then(|index| Some(&self.entries[index as usize]));
        (low, high)
    }

    pub fn bound_poly_var_top(&mut self, r: &F) {
        let mut entry_index = 0;
        let mut new_entries = Vec::with_capacity(self.entries.len());
        let mut new_indices = vec![None; self.mid];

        let span = tracing::span!(tracing::Level::TRACE, "inner_loop");
        let _enter = span.enter();
        for i in 0..self.mid {
            let (low_i, high_i) = (self.indices[2*i], self.indices[2*i + 1]);

            let new_entry = match (low_i, high_i) {
                (None, None) => continue,
                (Some(low_i), None) => {
                    let low = &self.entries[low_i as usize];
                    let m = F::one() - low;
                    *low + m * r
                },
                (None, Some(high_i)) => {
                    let high = &self.entries[high_i as usize];
                    let m = *high - F::one();
                    F::one() + m * r
                },
                (Some(low_i), Some(high_i)) => {
                    let low = &self.entries[low_i as usize];
                    let high = &self.entries[high_i as usize];
                    let m = *high - low;
                    *low + m * r
                },
            };

            if i >= self.mid / 2  {
                let dense_index = 2*i + 1 - self.mid;
                new_indices[dense_index] = Some(entry_index);
            } else {
                let dense_index = 2*i;
                new_indices[dense_index] = Some(entry_index);
            }

            new_entries.push(new_entry);
            entry_index += 1;
        }

        drop(_enter);
        drop(span);

        self.entries = new_entries;
        self.indices = new_indices;
        self.num_vars -= 2;
        self.mid /= 2;
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn to_dense(self) -> DensePolynomial<F> {
        let n = self.len();
        let half = n/2;

        let mut dense_low: Vec<F> = Vec::with_capacity(half);
        let mut dense_high: Vec<F> = Vec::with_capacity(half);
        for index in 0..half {
            let opt_low_sparse_index = self.indices[2*index];
            let opt_high_sparse_index = self.indices[2*index + 1];

            if let Some(low_sparse_index) = opt_low_sparse_index {
                dense_low.push(self.entries[low_sparse_index as usize]);
            } else {
                dense_low.push(F::one());
            }

            if let Some(high_sparse_index) = opt_high_sparse_index {
                dense_high.push(self.entries[high_sparse_index as usize]);
            } else {
                dense_high.push(F::one());
            }
        }

        let mut dense_evals = dense_low;
        dense_evals.extend(dense_high);

        DensePolynomial::new(dense_evals)
    }
}

// #[derive(Debug, Clone)]
// pub struct SparseGrandProductCircuit<F> {
//     left_vec: Vec<Vec<SparseEntry<F>>>,
//     right_vec: Vec<Vec<SparseEntry<F>>>,
// }

// impl<F: PrimeField> SparseGrandProductCircuit<F> {
//     pub fn construct(leaves: Vec<F>, flags: Vec<bool>) -> Self {
//         let num_leaves = leaves.len();
//         let num_layers = num_leaves.log_2(); 
//         let leaf_half = num_leaves / 2;

//         let mut lefts: Vec<Vec<SparseEntry<F>>> = Vec::with_capacity(num_layers);
//         let mut rights: Vec<Vec<SparseEntry<F>>> = Vec::with_capacity(num_layers);

//         // TODO(sragss): Attempt rough capacity planning here? We could construct metadata from initial flag scan.
//         let mut left: Vec<SparseEntry<F>> = Vec::new();
//         let mut right: Vec<SparseEntry<F>> = Vec::new();

//         // First layer
//         for leaf_index in 0..num_leaves {
//             if flags[leaf_index] { 
//                 if leaf_index < leaf_half {
//                     left.push(SparseEntry::new(leaves[leaf_index], leaf_index));
//                 } else {
//                     right.push(SparseEntry::new(leaves[leaf_index], leaf_index - leaf_half));
//                 }
//             }
//         }
//         lefts.push(left);
//         rights.push(right);

//         let mut layer_len = num_leaves;
//         for layer in 0..num_layers - 1 {
//             let (left, right) = 
//                 Self::compute_layer(&lefts[layer], &rights[layer], layer_len);

//             lefts.push(left);
//             rights.push(right);

//             layer_len /= 2;
//         }

//         Self { 
//             left_vec: lefts, 
//             right_vec: rights
//         }
//     }

//     fn compute_layer(
//             prior_left: &Vec<SparseEntry<F>>, 
//             prior_right: &Vec<SparseEntry<F>>, 
//             prior_len: usize) -> (Vec<SparseEntry<F>>, Vec<SparseEntry<F>>) {
//         // TODO(sragss): Attempt capacity planning?
//         let mut left: Vec<SparseEntry<F>> = Vec::new();
//         let mut right: Vec<SparseEntry<F>> = Vec::new();

//         let mut left_sparse_index: usize = 0;
//         let mut right_sparse_index: usize = 0;

//         while left_sparse_index < prior_left.len() && right_sparse_index < prior_right.len() {
//             // Mere existence of these indices means they're "non-sparse": not equal to 1.
//             let left_index = prior_left[left_sparse_index].index;
//             let right_index = prior_right[right_sparse_index].index;

//             let entry = if left_index == right_index {
//                 let value = prior_left[left_sparse_index].value * prior_right[right_sparse_index].value;
//                 left_sparse_index += 1;
//                 right_sparse_index += 1;
//                 SparseEntry::new(value, left_index)
//             } else if left_index < right_index {
//                 let entry = prior_left[left_sparse_index].clone();
//                 left_sparse_index += 1;
//                 entry
//             } else if right_index < left_index {
//                 let entry = prior_right[right_sparse_index].clone();
//                 right_sparse_index += 1;
//                 entry
//             } else {
//                 unreachable!();
//             };

//             if entry.index < prior_len / 4 {
//                 left.push(entry);
//             } else {
//                 right.push(entry);
//             }
//         }
//         (left, right)
//     }
// }


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
                Fr::from(4),
                Fr::from(5),
                Fr::from(6),
                Fr::from(7),
            ],
            vec![Some(0), Some(1), Some(2), Some(3)],
            2);
        assert_eq!(dense, sparse.to_dense());

        let dense = DensePolynomial::new(vec![Fr::one(), Fr::one(), Fr::from(3), Fr::one()]);
        let sparse = SparsePoly::new(vec![Fr::from(3)], vec![None, None, Some(0), None], 2);
        assert_eq!(dense, sparse.to_dense());

        let dense = DensePolynomial::new(vec![Fr::from(2), Fr::from(2), Fr::one(), Fr::one()]);
        let sparse = SparsePoly::new(vec![Fr::from(2), Fr::from(2)], vec![Some(0), Some(1), None, None], 2);
        assert_eq!(dense, sparse.to_dense());

        let dense = DensePolynomial::new(vec![Fr::one(), Fr::one(), Fr::from(8), Fr::from(8)]);
        let sparse = SparsePoly::new(vec![Fr::from(8), Fr::from(8)], vec![None, None, Some(0), Some(1)], 2);
        assert_eq!(dense, sparse.to_dense());
    }

    #[test]
    fn bound_poly_var_top() {
        let mut dense = DensePolynomial::new(vec![Fr::from(4), Fr::from(5), Fr::from(6), Fr::from(7)]);
        let mut sparse = SparsePoly::new(
            vec![
                Fr::from(4),
                Fr::from(5),
                Fr::from(6),
                Fr::from(7),
            ],
            vec![
                Some(0),
                Some(1),
                Some(2),
                Some(3),
            ],
            2);
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
            vec![
                Fr::from(6),
                Fr::from(7),
            ], 
            vec![
                None,
                None,
                Some(0),
                Some(1),
            ], 2);
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
                Fr::from(4),
                Fr::from(5),
            ],
            vec![
                Some(0),
                Some(1),
                None,
                None
            ], 2);
        assert_eq!(dense, sparse.clone().to_dense());
        let r = Fr::from(12);
        dense.bound_poly_var_top(&r);
        sparse.bound_poly_var_top(&r);
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
        let mut indices = Vec::new();
        let mut sparse_index = 0;
        for _ in 0..size {
            if rng.gen::<f64>() > pct_ones {
                entries.push(F::rand(&mut rng));
                indices.push(Some(sparse_index));
                sparse_index += 1;
            } else {
                indices.push(None);
            }
        }
        SparsePoly::new(entries, indices, log_size)
    }
}