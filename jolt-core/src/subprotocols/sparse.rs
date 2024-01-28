use ark_ff::PrimeField;

use crate::{poly::dense_mlpoly::DensePolynomial, utils::math::Math};


#[derive(Debug, Clone, PartialEq)]
pub struct SparseEntry<F> {
    value: F,
    index: usize
}

impl<F> SparseEntry<F> {
    pub fn new(value: F, index: usize) -> Self {
        Self { value, index }
    }
}

#[derive(Debug, Clone, PartialEq)]
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
        assert!(low_entries.len() <= dense_len);
        assert!(high_entries.len() <= dense_len);
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
    pub fn construct(leaves: &[F], flags: &[bool]) -> Self {
        assert_eq!(leaves.len(), flags.len());
        let num_leaves = leaves.len();
        let num_layers = num_leaves.log_2(); 
        let leaf_half = num_leaves / 2;

        let mut lefts: Vec<SparsePoly<F>> = Vec::with_capacity(num_layers);
        let mut rights: Vec<SparsePoly<F>> = Vec::with_capacity(num_layers);

        let max_capacity = num_leaves / 4;
        let mut left_low: Vec<SparseEntry<F>> = Vec::with_capacity(max_capacity);
        let mut left_high: Vec<SparseEntry<F>> = Vec::with_capacity(max_capacity);
        let mut right_low: Vec<SparseEntry<F>> = Vec::with_capacity(max_capacity);
        let mut right_high: Vec<SparseEntry<F>> = Vec::with_capacity(max_capacity);

        // First layer
        let new_leaf_half = leaf_half / 2;
        for leaf_index in 0..num_leaves {
            if flags[leaf_index] { 
                if leaf_index < new_leaf_half {
                    left_low.push(SparseEntry::new(leaves[leaf_index], leaf_index));
                } else if leaf_index < 2 * new_leaf_half {
                    left_high.push(SparseEntry::new(leaves[leaf_index], leaf_index - new_leaf_half));
                } else if leaf_index < 3 * new_leaf_half{
                    right_low.push(SparseEntry::new(leaves[leaf_index], leaf_index - 2 * new_leaf_half));
                } else {
                    right_high.push(SparseEntry::new(leaves[leaf_index], leaf_index - 3 * new_leaf_half));
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

        // TODO(sragss): Dedupe below -- it's gross.

        // Capacity has the potential to overshoot by a factor 2
        let max_capacity = prior_left.sparse_len();
        let mut left_low: Vec<SparseEntry<F>> = Vec::with_capacity(max_capacity);
        let mut left_high: Vec<SparseEntry<F>> = Vec::with_capacity(max_capacity);

        let mut left_sparse_index: usize = 0;
        let mut right_sparse_index: usize = 0;
        while left_sparse_index < prior_left.low_entries.len() || right_sparse_index < prior_right.low_entries.len() {
            let left_index = if left_sparse_index == prior_left.low_entries.len() { prior_left.dense_len } else { prior_left.low_entries[left_sparse_index].index };
            let right_index = if right_sparse_index == prior_right.low_entries.len() { prior_right.dense_len } else { prior_right.low_entries[right_sparse_index].index };

            let mut entry = if left_index == right_index {
                let value = prior_left.low_entries[left_sparse_index].value * prior_right.low_entries[right_sparse_index].value;
                left_sparse_index += 1;
                right_sparse_index += 1;
                SparseEntry::new(value, left_index)
            } else if left_index < right_index {
                let entry = prior_left.low_entries[left_sparse_index].clone();
                left_sparse_index += 1;
                entry
            } else if right_index < left_index {
                let entry = prior_right.low_entries[right_sparse_index].clone();
                right_sparse_index += 1;
                entry
            } else {
                unreachable!();
            };

            // prior_left.dense_len() + prior_right.dense_len() == prior_len
            // dense_left.len() == dense_right.len() == prior_len / 4
            // dense_left_low.len() == dense_left_high.len() == ... == prior.len / 8
            if entry.index < prior_len / 8 || prior_len == 4 {
                left_low.push(entry);
            } else {
                entry.index -= prior_len / 8;
                left_high.push(entry);
            }
        }

        let max_capacity = prior_right.sparse_len();
        let mut right_low: Vec<SparseEntry<F>> = Vec::with_capacity(max_capacity);
        let mut right_high: Vec<SparseEntry<F>> = Vec::with_capacity(max_capacity);
        let mut left_sparse_index: usize = 0;
        let mut right_sparse_index: usize = 0;
        while left_sparse_index < prior_left.high_entries.len() || right_sparse_index < prior_right.high_entries.len() {
            let left_index = if left_sparse_index == prior_left.high_entries.len() { prior_left.dense_len } else { prior_left.high_entries[left_sparse_index].index };
            let right_index = if right_sparse_index == prior_right.high_entries.len() { prior_right.dense_len } else { prior_right.high_entries[right_sparse_index].index };

            let mut entry = if left_index == right_index {
                let value = prior_left.high_entries[left_sparse_index].value * prior_right.high_entries[right_sparse_index].value;
                left_sparse_index += 1;
                right_sparse_index += 1;
                SparseEntry::new(value, left_index)
            } else if left_index < right_index {
                let entry = prior_left.high_entries[left_sparse_index].clone();
                left_sparse_index += 1;
                entry
            } else if right_index < left_index {
                let entry = prior_right.high_entries[right_sparse_index].clone();
                right_sparse_index += 1;
                entry
            } else {
                unreachable!();
            };

            // prior_left.dense_len() + prior_right.dense_len() == prior_len
            // dense_left.len() == dense_right.len() == prior_len / 4
            // dense_left_low.len() == dense_left_high.len() == ... == prior.len / 8
            if entry.index < prior_len / 8 || prior_len == 4 {
                right_low.push(entry);
            } else {
                entry.index -= prior_len / 8;
                right_high.push(entry);
            }
        }

        let new_len = prior_len / 4;
        let left = SparsePoly::new(left_low, left_high, new_len);
        let right = SparsePoly::new(right_low, right_high, new_len);
        (left, right)
    }

    pub fn num_layers(&self) -> usize {
        self.left.len()
    }

    pub fn evaluate(&self) -> F {
        let num_layers = self.num_layers();
        let left = &self.left[num_layers - 1];
        let right = &self.right[num_layers - 1];
        assert_eq!(left.num_vars, 0);
        assert_eq!(right.num_vars, 0);

        assert_eq!(left.high_entries.len(), 0);
        assert_eq!(right.high_entries.len(), 0);

        // It's possible that an entire side of the GKR circuit evaluates
        // to one, in which case the sparse representation will be empty.
        let left_val = if left.low_entries.len() == 1 {
            left.low_entries[0].value
        } else if left.low_entries.len() == 0 {
            println!("hit the left one case");
            F::one() 
        } else {
            panic!("shouldn't happen");
        };

        let right_val = if right.low_entries.len() == 1 {
            right.low_entries[0].value
        } else if right.low_entries.len() == 0 {
            println!("hit the right one case - left: {left:?}");
            F::one()
        } else {
            panic!("shouldn't happen");
        };

        left_val * right_val
    }
}


#[cfg(test)]
mod tests {
    use crate::subprotocols::grand_product::GrandProductCircuit;

    use super::{*, bench::init_bind_bench};
    use ark_curve25519::Fr;
    use ark_std::{One, test_rng, rand::Rng};

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

    #[test]
    fn sparse_circuit_construction_dense() {
        let leaves: Vec<Fr> = vec![Fr::from(0), Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4), Fr::from(5), Fr::from(6), Fr::from(7)];
        let flags: Vec<bool> = vec![true, true, true, true, true, true, true, true];

        let circuit = SparseGrandProductCircuit::construct(&leaves, &flags);

        // Example:
        // 0: LEFT = A, B, C, D     RIGHT = E, F, G, H
        // 1: LEFT = AE, BF         RIGHT = CG, DH
        // 2: LEFT = AEBF           RIGHT = CGDH

        // (indices) 0  1  0  1             0  1  0  1
        // 0: LEFT = A, B, C, D     RIGHT = E, F, G, H
        //           0   0                  0   0 
        // 1: LEFT = AE, BF         RIGHT = CG, DH
        // 2: LEFT = AEBF           RIGHT = CGDH

        // layer 0
        let expected_left = SparsePoly::new(
            vec![SparseEntry::new(Fr::from(0), 0), SparseEntry::new(Fr::from(1), 1)], vec![SparseEntry::new(Fr::from(2), 0), SparseEntry::new(Fr::from(3), 1)], 
            4
        );
        assert_eq!(circuit.left[0], expected_left);
        let expected_right= SparsePoly::new(
            vec![SparseEntry::new(Fr::from(4), 0), SparseEntry::new(Fr::from(5), 1)], vec![SparseEntry::new(Fr::from(6), 0), SparseEntry::new(Fr::from(7), 1)], 
            4
        );
        assert_eq!(circuit.right[0], expected_right);

        // layer 1
        let expected_left = SparsePoly::new(
            vec![SparseEntry::new(Fr::from(0) * Fr::from(4), 0)], 
            vec![SparseEntry::new(Fr::from(1) * Fr::from(5), 0)], 
            2
        );
        let expected_right = SparsePoly::new(
            vec![SparseEntry::new(Fr::from(2) * Fr::from(6), 0)], 
            vec![SparseEntry::new(Fr::from(3) * Fr::from(7), 0)], 
            2
        );
        assert_eq!(circuit.left[1], expected_left);
        assert_eq!(circuit.right[1], expected_right);

        let combo: Fr = Fr::from(0) * Fr::from(4) * Fr::from(2) * Fr::from(6);
        let expected_left = SparsePoly::new(
            vec![SparseEntry::new(combo, 0)],
            vec![],
            1
        );
        let combo: Fr = Fr::from(1) * Fr::from(5) * Fr::from(3) * Fr::from(7);
        let expected_right= SparsePoly::new(
            vec![SparseEntry::new(combo, 0)],
            vec![],
            1
        );
        assert_eq!(circuit.left[2], expected_left);
        assert_eq!(circuit.right[2], expected_right);

    }

    fn gen_leaves_flags<F: PrimeField>(log_size: usize, pct_ones: f64) -> (Vec<F>, Vec<bool>) {
        let mut rng = test_rng();
        let size = 1 << log_size;
        let mut leaves = Vec::new();
        let mut flags = Vec::new();

        for _ in 0..size {
            leaves.push(F::rand(&mut rng));

            if rng.gen::<f64>() > pct_ones {
                flags.push(true);
            } else {
                flags.push(false);
            }
        }
        (leaves, flags)
    }

    #[test]
    fn sparse_circuit_parity() {
        let log_size = 8;
        let (leaves, flags) = gen_leaves_flags::<Fr>(log_size, 0.8);

        let dense_leaves_toggled: Vec<Fr> = leaves.iter().zip(flags.iter()).map(|(leaf, flag)| {
            if *flag {
                *leaf
            } else {
                Fr::one()
            }
        }).collect();

        let mut dense_circuit = GrandProductCircuit::new(&DensePolynomial::new(dense_leaves_toggled));
        let sparse_circuit = SparseGrandProductCircuit::construct(&leaves, &flags);

        let (dense_layer_0_left, dense_layer_0_right) = dense_circuit.take_layer(0);
        assert_eq!(dense_layer_0_left, sparse_circuit.left[0].clone().to_dense());
        assert_eq!(dense_layer_0_right, sparse_circuit.right[0].clone().to_dense());
    }

    #[test]
    fn sparse_circuit_evaluation() {
        let leaves = vec![Fr::from(10), Fr::from(20), Fr::from(30), Fr::from(40), Fr::from(50), Fr::from(60), Fr::from(70), Fr::from(80)];
        let flags = vec![true, true, true, true, true, true, true, true];
        let circuit = SparseGrandProductCircuit::construct(&leaves, &flags);

        let product = leaves.iter().product();
        assert_eq!(circuit.evaluate(), product);

        let flags = vec![false, false, false, false, true, true, true, true];
        let half_leaves = &leaves[4..8];
        let product = half_leaves.iter().product();
        let circuit = SparseGrandProductCircuit::construct(&leaves, &flags);
        assert_eq!(circuit.evaluate(), product);

        let flags = vec![true, true, true, true, false, false, false, false];
        let half_leaves = &leaves[0..4];
        let product = half_leaves.iter().product();
        let circuit = SparseGrandProductCircuit::construct(&leaves, &flags);
        assert_eq!(circuit.evaluate(), product);
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