use allocative::Allocative;
use rayon::prelude::*;
use std::ops::Index;

use crate::field::JoltField;
use crate::poly::multilinear_polynomial::BindingOrder;
use crate::utils::thread::unsafe_allocate_zero_vec;

/// Table containing the evaluations `EQ(x_1, ..., x_j, r_1, ..., r_j)`,
/// built up incrementally as we receive random challenges `r_j` over the
/// course of sumcheck.
#[derive(Clone, Debug, Allocative, Default)]
pub struct ExpandingTable<F: JoltField> {
    binding_order: BindingOrder,
    len: usize,
    values: Vec<F>,
    scratch_space: Vec<F>,
}

impl<F: JoltField> ExpandingTable<F> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn order(&self) -> BindingOrder {
        self.binding_order
    }

    /// Initializes an `ExpandingTable` with the given `capacity`.
    #[tracing::instrument(skip_all, name = "ExpandingTable::new")]
    pub fn new(capacity: usize, binding_order: BindingOrder) -> Self {
        let (values, scratch_space) = rayon::join(
            || unsafe_allocate_zero_vec(capacity),
            || match binding_order {
                BindingOrder::LowToHigh => Vec::with_capacity(0),
                BindingOrder::HighToLow => unsafe_allocate_zero_vec(capacity),
            },
        );
        Self {
            binding_order,
            len: 0,
            values,
            scratch_space,
        }
    }

    /// Resets this table to be length 1, containing only the given `value`.
    pub fn reset(&mut self, value: F) {
        self.values[0] = value;
        self.len = 1;
    }

    pub fn clone_values(&self) -> Vec<F> {
        self.values[..self.len].to_vec()
    }

    /// Updates this table (expanding it by a factor of 2) to incorporate
    /// the new random challenge `r_j`.
    /// TODO: this is bad parallelisation.
    #[tracing::instrument(skip_all, name = "ExpandingTable::update")]
    pub fn update(&mut self, r_j: F::Challenge) {
        match self.binding_order {
            BindingOrder::LowToHigh => {
                let (values_left, values_right) = self.values.split_at_mut(self.len);
                values_left
                    .par_iter_mut()
                    .zip(values_right.par_iter_mut())
                    .for_each(|(x, y)| {
                        *y = *x * r_j;
                        *x -= *y;
                    });
            }
            BindingOrder::HighToLow => {
                self.values[..self.len]
                    .par_iter()
                    .zip(self.scratch_space.par_chunks_mut(2))
                    .for_each(|(&v_i, dest)| {
                        let eval_1 = r_j * v_i;
                        dest[0] = v_i - eval_1;
                        dest[1] = eval_1;
                    });
                std::mem::swap(&mut self.values, &mut self.scratch_space);
            }
        }
        self.len *= 2;
    }
}

impl<F: JoltField> Index<usize> for ExpandingTable<F> {
    type Output = F;

    fn index(&self, index: usize) -> &F {
        assert!(index < self.len, "index: {}, len: {}", index, self.len);
        &self.values[index]
    }
}

impl<'data, F: JoltField> IntoParallelIterator for &'data ExpandingTable<F> {
    type Item = &'data F;
    type Iter = rayon::slice::Iter<'data, F>;

    fn into_par_iter(self) -> Self::Iter {
        self.values[..self.len].into_par_iter()
    }
}
