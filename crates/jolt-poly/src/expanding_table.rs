//! Incrementally materialized equality tables.

use std::ops::Index;

use jolt_field::Field;

use crate::{thread::unsafe_allocate_zero_vec, BindingOrder};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Table containing the evaluations of `eq(x, r)` as challenges are streamed in.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExpandingTable<F: Field> {
    binding_order: BindingOrder,
    len: usize,
    values: Vec<F>,
    scratch_space: Vec<F>,
}

impl<F: Field> ExpandingTable<F> {
    #[tracing::instrument(skip_all, name = "ExpandingTable::new")]
    pub fn new(capacity: usize, binding_order: BindingOrder) -> Self {
        assert!(capacity > 0, "expanding table capacity must be positive");
        let (values, scratch_space) = join_or_serial(
            || unsafe_allocate_zero_vec(capacity),
            || match binding_order {
                BindingOrder::LowToHigh => Vec::new(),
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

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn order(&self) -> BindingOrder {
        self.binding_order
    }

    #[inline]
    pub fn values(&self) -> &[F] {
        &self.values[..self.len]
    }

    pub fn reset(&mut self, value: F) {
        assert!(!self.values.is_empty(), "expanding table has zero capacity");
        self.values[0] = value;
        self.len = 1;
    }

    pub fn clone_values(&self) -> Vec<F> {
        self.values().to_vec()
    }

    #[tracing::instrument(skip_all, name = "ExpandingTable::update")]
    pub fn update(&mut self, challenge: F) {
        assert!(self.len > 0, "expanding table must be reset before update");
        assert!(
            self.len * 2 <= self.values.len(),
            "expanding table capacity exceeded"
        );
        match self.binding_order {
            BindingOrder::LowToHigh => self.update_low_to_high(challenge),
            BindingOrder::HighToLow => self.update_high_to_low(challenge),
        }
        self.len *= 2;
    }

    fn update_low_to_high(&mut self, challenge: F) {
        #[cfg(feature = "parallel")]
        {
            let (left, right) = self.values.split_at_mut(self.len);
            left.par_iter_mut()
                .zip(right.par_iter_mut())
                .for_each(|(left, right)| {
                    *right = *left * challenge;
                    *left -= *right;
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            let (left, right) = self.values.split_at_mut(self.len);
            for (left, right) in left.iter_mut().zip(right.iter_mut()) {
                *right = *left * challenge;
                *left -= *right;
            }
        }
    }

    fn update_high_to_low(&mut self, challenge: F) {
        #[cfg(feature = "parallel")]
        {
            self.values[..self.len]
                .par_iter()
                .zip(self.scratch_space.par_chunks_mut(2))
                .for_each(|(&value, dest)| {
                    let eval_1 = value * challenge;
                    dest[0] = value - eval_1;
                    dest[1] = eval_1;
                });
            std::mem::swap(&mut self.values, &mut self.scratch_space);
        }

        #[cfg(not(feature = "parallel"))]
        {
            for (index, &value) in self.values[..self.len].iter().enumerate() {
                let eval_1 = value * challenge;
                self.scratch_space[2 * index] = value - eval_1;
                self.scratch_space[2 * index + 1] = eval_1;
            }
            std::mem::swap(&mut self.values, &mut self.scratch_space);
        }
    }
}

impl<F: Field> Index<usize> for ExpandingTable<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(
            index < self.len,
            "expanding table index {index} out of bounds for len {}",
            self.len
        );
        &self.values[index]
    }
}

#[cfg(feature = "parallel")]
fn join_or_serial<A: Send, B: Send>(
    left: impl FnOnce() -> A + Send,
    right: impl FnOnce() -> B + Send,
) -> (A, B) {
    rayon::join(left, right)
}

#[cfg(not(feature = "parallel"))]
fn join_or_serial<A, B>(left: impl FnOnce() -> A, right: impl FnOnce() -> B) -> (A, B) {
    (left(), right())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EqPolynomial;
    use jolt_field::{Fr, RandomSampling};
    use num_traits::One;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn high_to_low_matches_eq_table_prefixes() {
        let mut rng = ChaCha20Rng::seed_from_u64(710);
        let point: Vec<Fr> = (0..8).map(|_| Fr::random(&mut rng)).collect();
        let mut table = ExpandingTable::new(1 << point.len(), BindingOrder::HighToLow);
        table.reset(Fr::one());

        for prefix_len in 0..=point.len() {
            let expected = EqPolynomial::<Fr>::evals(&point[..prefix_len], None);
            assert_eq!(table.values(), expected);
            if prefix_len < point.len() {
                table.update(point[prefix_len]);
            }
        }
    }

    #[test]
    fn low_to_high_matches_reversed_eq_prefixes() {
        let mut rng = ChaCha20Rng::seed_from_u64(711);
        let point: Vec<Fr> = (0..8).map(|_| Fr::random(&mut rng)).collect();
        let mut reversed_prefix = Vec::new();
        let mut table = ExpandingTable::new(1 << point.len(), BindingOrder::LowToHigh);
        table.reset(Fr::one());

        for (prefix_len, &challenge) in point.iter().enumerate() {
            let expected = EqPolynomial::<Fr>::evals(&reversed_prefix, None);
            assert_eq!(table.values(), expected, "prefix_len={prefix_len}");
            reversed_prefix.insert(0, challenge);
            table.update(challenge);
        }
        let expected = EqPolynomial::<Fr>::evals(&reversed_prefix, None);
        assert_eq!(table.values(), expected);
    }

    #[test]
    fn clone_and_index_expose_active_prefix_only() {
        let mut table = ExpandingTable::new(8, BindingOrder::HighToLow);
        table.reset(Fr::from_u64(3));
        table.update(Fr::from_u64(5));

        assert_eq!(table.len(), 2);
        assert_eq!(table[0], Fr::from_u64(3) - Fr::from_u64(15));
        assert_eq!(table[1], Fr::from_u64(15));
        assert_eq!(table.clone_values(), table.values());
    }
}
