use ark_ff::PrimeField;

use crate::{poly::dense_mlpoly::DensePolynomial, utils::math::Math};

#[derive(Debug, Clone)]
pub struct SparsePoly<F> {
    entries: Vec<F>,
    indices: Vec<Option<usize>>,

    num_vars: usize,
    mid: usize
}

impl<F: PrimeField> SparsePoly<F> {
    pub fn new(entries: Vec<F>, indices: Vec<Option<usize>>, num_vars: usize) -> Self {
        assert_eq!(indices.len(), 1 << num_vars);
        let mid = if num_vars > 0 { 1 << (num_vars - 1) } else { 0 };
        Self { entries, indices, num_vars, mid }
    }

    pub fn final_eval(&self) -> F {
        assert_eq!(self.entries.len(), 1);
        assert_eq!(self.num_vars, 0);
        assert!(self.indices[0].is_some());
        self.entries[0]
    }

    #[inline]
    pub fn low_high_iter(&self, index: usize) -> (Option<&F>, Option<&F>) {
        assert!(index < self.mid);
        let low_i = index;
        let high_i = index + self.mid;

        let low = self.indices[low_i].and_then(|index| Some(&self.entries[index]));
        let high = self.indices[high_i].and_then(|index| Some(&self.entries[index]));
        (low, high)
    }

    pub fn bound_poly_var_top(&mut self, r: &F) {
        let span_alloc = tracing::span!(tracing::Level::TRACE, "bound::allocation");
        let _enter_alloc = span_alloc.enter();

        let mut new_indices: Vec<Option<usize>> = vec![None; self.mid];
        let mut new_entries: Vec<F> = Vec::with_capacity(self.entries.len());
        drop(_enter_alloc);
        drop(span_alloc);


        let span = tracing::span!(tracing::Level::TRACE, "bound::inner_loop");
        let _enter = span.enter();
        let mut entry_index = 0;
        for i in 0..self.mid {
            let new_entry = match self.low_high_iter(i){
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

            new_entries.push(new_entry);
            new_indices[i] = Some(entry_index);
            entry_index += 1;
        }

        drop(_enter);
        drop(span);

        self.entries = new_entries;
        self.indices = new_indices;
        self.num_vars -= 1;
        self.mid /= 2;
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn to_dense(self) -> DensePolynomial<F> {
        let n = self.len();

        let mut dense_evals: Vec<F> = Vec::with_capacity(n);
        for index in 0..n {
            let opt_sparse_index = self.indices[index];
            if let Some(sparse_index) = opt_sparse_index {
                dense_evals.push(self.entries[sparse_index as usize]);
            } else {
                dense_evals.push(F::one());
            }
        }

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
    use super::{*, bench::init_bind_bench};
    use ark_curve25519::Fr;
    use ark_std::{One, test_rng};

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
    use ark_std::UniformRand;
    use ark_std::{test_rng, rand::Rng};

    pub fn init_bind_bench<F: PrimeField>(log_size: usize, pct_ones: f64) -> SparsePoly<F> {
        let mut rng = test_rng();

        let size = 1 << log_size;
        let mut entries = Vec::new();
        let mut indices = vec![None; size];
        let mut sparse_index = 0;
        for i in 0..size {
            if rng.gen::<f64>() > pct_ones {
                entries.push(F::rand(&mut rng));
                indices[i] = Some(sparse_index);
                sparse_index += 1;
            }
        }
        SparsePoly::new(entries, indices, log_size)
    }
}