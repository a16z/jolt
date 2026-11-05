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
    bits: u32,
    tail: u128,
    values: Vec<F>,
}

impl<F: JoltField> PackedFlags<F> {
    /// Pattern bits per column after the first bind; each later bind doubles
    /// them and squares the `2^bits`-entry lookup table.
    pub(super) const FIRST_BITS: u32 = 2;

    /// Pattern bits at which the caller materializes field columns.
    pub(super) const SATURATED_BITS: u32 = 16;

    pub(super) fn new(words: Vec<u128>, tail: [bool; 5], challenge: F) -> Self {
        Self {
            words,
            bits: Self::FIRST_BITS,
            tail: Self::encode(tail, tail),
            values: vec![F::zero(), F::one() - challenge, challenge, F::one()],
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
        let word = self.words.get(row).copied().unwrap_or(self.tail);
        self.values[((word >> (column as u32 * self.bits)) & ((1 << self.bits) - 1)) as usize]
    }

    pub(super) fn saturated(&self) -> bool {
        self.bits == Self::SATURATED_BITS
    }

    /// Concatenate the two bit patterns; only the small shared table needs
    /// field arithmetic. The caller materializes before this table saturates.
    pub(super) fn bind(&mut self, challenge: F) {
        debug_assert!(!self.saturated());
        let bits = self.bits;
        let mask = (1u128 << bits) - 1;
        let merge = |even: u128, odd: u128| {
            (0..5).fold(0, |word, column| {
                let shift = column * bits;
                word | ((((even >> shift) & mask) | (((odd >> shift) & mask) << bits))
                    << (2 * shift))
            })
        };
        let pair = |row: usize| {
            merge(
                self.words[2 * row],
                self.words.get(2 * row + 1).copied().unwrap_or(self.tail),
            )
        };
        let half = self.words.len().div_ceil(2);
        #[cfg(feature = "parallel")]
        let words = (0..half).into_par_iter().map(pair).collect();
        #[cfg(not(feature = "parallel"))]
        let words = (0..half).map(pair).collect();
        let size = self.values.len();
        let entry = |index: usize| {
            let lo = self.values[index % size];
            challenge.mul_add(self.values[index / size] - lo, lo)
        };
        #[cfg(feature = "parallel")]
        let values = (0..size * size).into_par_iter().map(entry).collect();
        #[cfg(not(feature = "parallel"))]
        let values = (0..size * size).map(entry).collect();
        self.tail = merge(self.tail, self.tail);
        self.bits *= 2;
        self.words = words;
        self.values = values;
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
