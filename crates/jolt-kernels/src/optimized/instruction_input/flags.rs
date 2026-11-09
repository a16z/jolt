//! Bound Boolean columns represented by their original bit patterns.
//!
//! All five columns share a lookup table for the multilinear evaluation of a
//! bit pattern at the challenges bound so far. A row holds five patterns in
//! one word. After four binds, a further lookup table would be too large, so
//! the caller switches to ordinary field columns.

use jolt_field::JoltField;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) struct PackedFlags<F: JoltField> {
    words: Vec<u128>,
    table: FlagTable<F>,
}

/// Challenge-dependent values and constant pattern shared by CPU and device
/// words. Device ownership does not require allocating a host word mirror.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) struct FlagTable<F: JoltField> {
    pub(super) bits: u32,
    pub(super) tail: u128,
    pub(super) values: Vec<F>,
}

impl<F: JoltField> FlagTable<F> {
    pub(super) const FIRST_BITS: u32 = 2;
    pub(super) const SATURATED_BITS: u32 = 16;

    pub(super) fn new(tail: [bool; 5], challenge: F) -> Self {
        Self {
            bits: Self::FIRST_BITS,
            tail: PackedFlags::<F>::encode(tail, tail),
            values: vec![F::zero(), F::one() - challenge, challenge, F::one()],
        }
    }

    pub(super) fn merge(&self, even: u128, odd: u128) -> u128 {
        let bits = self.bits;
        let mask = (1u128 << bits) - 1;
        (0..5).fold(0, |word, column| {
            let shift = column * bits;
            word | ((((even >> shift) & mask) | (((odd >> shift) & mask) << bits)) << (2 * shift))
        })
    }

    pub(super) fn bind(&mut self, challenge: F) {
        debug_assert!(self.bits < Self::SATURATED_BITS);
        let size = self.values.len();
        let entry = |index: usize| {
            let lo = self.values[index % size];
            challenge.mul_add(self.values[index / size] - lo, lo)
        };
        #[cfg(feature = "parallel")]
        let values = (0..size * size).into_par_iter().map(entry).collect();
        #[cfg(not(feature = "parallel"))]
        let values = (0..size * size).map(entry).collect();
        self.tail = self.merge(self.tail, self.tail);
        self.bits *= 2;
        self.values = values;
    }
}

impl<F: JoltField> PackedFlags<F> {
    pub(super) fn new(words: Vec<u128>, tail: [bool; 5], challenge: F) -> Self {
        Self {
            words,
            table: FlagTable::new(tail, challenge),
        }
    }

    #[inline]
    pub(super) fn encode(even: [bool; 5], odd: [bool; 5]) -> u128 {
        (0..5).fold(0, |word, i| {
            word | ((u128::from(even[i]) | (u128::from(odd[i]) << 1)) << (2 * i))
        })
    }

    #[inline]
    pub(super) fn value(&self, column: usize, row: usize) -> F {
        let word = self.words.get(row).copied().unwrap_or(self.table.tail);
        self.table.values
            [((word >> (column as u32 * self.table.bits)) & ((1 << self.table.bits) - 1)) as usize]
    }

    pub(super) fn saturated(&self) -> bool {
        self.table.bits == FlagTable::<F>::SATURATED_BITS
    }

    /// Concatenate the two bit patterns; only the small shared table needs
    /// field arithmetic. The caller materializes before this table saturates.
    pub(super) fn bind(&mut self, challenge: F) {
        debug_assert!(!self.saturated());
        let pair = |row: usize| {
            self.table.merge(
                self.words[2 * row],
                self.words
                    .get(2 * row + 1)
                    .copied()
                    .unwrap_or(self.table.tail),
            )
        };
        let half = self.words.len().div_ceil(2);
        #[cfg(feature = "parallel")]
        let words = (0..half).into_par_iter().map(pair).collect();
        #[cfg(not(feature = "parallel"))]
        let words = (0..half).map(pair).collect();
        self.table.bind(challenge);
        self.words = words;
    }

    pub(super) fn materialize_bound(&self, challenge: F) -> [Vec<F>; 5] {
        let half = self.words.len().div_ceil(2);
        core::array::from_fn(|column| {
            let entry = |row: usize| {
                let lo = self.value(column, 2 * row);
                challenge.mul_add(self.value(column, 2 * row + 1) - lo, lo)
            };
            #[cfg(feature = "parallel")]
            {
                (0..half).into_par_iter().map(entry).collect()
            }
            #[cfg(not(feature = "parallel"))]
            {
                (0..half).map(entry).collect()
            }
        })
    }
}
