//! Output-stationary commit accumulation for K<D trace one-hot rings.
//!
//! A K<D ring packs `D/K` trace rows, so each column adds several shifts of
//! the same `A` entry into one destination. Loading the entry once as
//! negacyclic windows lets each destination tile sum all of its shifts in
//! registers and touch memory once, instead of once per shift.

use akita_algebra::CyclotomicRing;
use jolt_field::{Fp128x8i32, Unreduced};

use crate::AkitaField;

/// Coefficients per register tile. Four keeps the tile's `u32` sums within
/// half of baseline x86-64's SSE2 register file and costs nothing on AVX2 or
/// NEON.
const TILE: usize = 4;

/// One destination ring element as unreduced [`Fp128x8i32`] lanes.
pub(super) type DigitAccumulator<const D: usize> = [Fp128x8i32; D];

/// Every negacyclic shift of one `A` entry as canonical 16-bit digits.
///
/// Holds the digits of `[-a_0, …, -a_{D-1}, a_0, …, a_{D-1}]`, so coefficient
/// `j` of `a · X^k` is entry `D + j - k` for every `k < D`. The digits are the
/// non-negative [`Fp128x8i32`] lanes of each canonical value, so a shift reads
/// half the bytes of a wide ring element and adds a value below `2^16` to
/// each destination lane. At most `2^15` shifts per destination between
/// flushes keep every lane inside `reduce_wide`'s `i32` range.
pub(super) struct DigitWindows<const D: usize> {
    digits: Vec<[u16; 8]>,
}

impl<const D: usize> DigitWindows<D> {
    pub(super) fn new() -> Self {
        const { assert!(D.is_multiple_of(TILE)) };
        Self {
            digits: vec![[0; 8]; 2 * D],
        }
    }

    /// Replaces the held entry with `src`.
    pub(super) fn load(&mut self, src: &CyclotomicRing<AkitaField, D>) {
        let (negative, positive) = self.digits.split_at_mut(D);
        for ((negative, positive), &value) in negative.iter_mut().zip(positive).zip(&src.coeffs) {
            *negative = canonical_digits(-value);
            *positive = canonical_digits(value);
        }
    }

    /// `dst += a · Σ_k X^k` over `shifts`, each `< D`.
    pub(super) fn accumulate(&self, dst: &mut DigitAccumulator<D>, shifts: &[usize]) {
        debug_assert!(shifts.iter().all(|&shift| shift < D));
        for (tile, out) in dst.chunks_exact_mut(TILE).enumerate() {
            let base = D + tile * TILE;
            let mut sums = [[0u32; 8]; TILE];
            for &shift in shifts {
                for (sum, digits) in sums.iter_mut().zip(&self.digits[base - shift..][..TILE]) {
                    for (sum, &digit) in sum.iter_mut().zip(digits) {
                        *sum += u32::from(digit);
                    }
                }
            }
            for (out, sum) in out.iter_mut().zip(sums) {
                for (lane, sum) in out.0.iter_mut().zip(sum) {
                    *lane += sum as i32;
                }
            }
        }
    }
}

fn canonical_digits(value: AkitaField) -> [u16; 8] {
    Fp128x8i32::from(value).0.map(|lane| lane as u16)
}

/// Adds every accumulator into its reduced ring element and clears it.
pub(super) fn flush_digit_accumulators<const D: usize>(
    accumulators: &mut [DigitAccumulator<D>],
    reduced: &mut [CyclotomicRing<AkitaField, D>],
) {
    for (accumulator, reduced) in accumulators.iter_mut().zip(reduced) {
        for (lanes, coefficient) in accumulator.iter_mut().zip(&mut reduced.coeffs) {
            *coefficient += AkitaField::reduce_wide(std::mem::replace(lanes, Fp128x8i32([0; 8])));
        }
    }
}
