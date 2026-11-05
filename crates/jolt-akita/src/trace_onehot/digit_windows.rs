//! Output-stationary commit accumulation for K<D trace one-hot rings.
//!
//! A K<D ring packs `D/K` trace rows, so each column adds several shifts of
//! the same `A` entry into one destination. Loading the entry once as
//! negacyclic windows lets each destination tile sum all of its shifts in
//! registers and touch memory once, instead of once per shift.

use akita_algebra::CyclotomicRing;
use jolt_field::{Fp128x8i32, Unreduced};

use crate::AkitaField;

/// Eight coefficients expose the larger NEON register file to the shift sum.
/// Keep four on other targets to avoid spilling on baseline SSE2.
const TILE: usize = if cfg!(target_arch = "aarch64") { 8 } else { 4 };

/// Shifted views resolved per pass over a destination. Every production batch
/// fits in one pass: at most `D / one_hot_k` (commit.rs) and below
/// `DOT_BATCH * NTT_MIN_SHIFTS_PER_RING` (ntt_commit.rs).
const VIEW_BATCH: usize = 64;

/// One destination ring element as unreduced [`Fp128x8i32`] lanes.
pub(super) type DigitAccumulator<const D: usize> = [Fp128x8i32; D];

/// Every negacyclic shift of one `A` entry as canonical 16-bit digits.
///
/// Holds the digits of `[-a_0, …, -a_{D-1}, a_0, …, a_{D-1}]`, so coefficient
/// `j` of `a · X^k` is entry `D + j - k` for every `k < D`. The digits are the
/// non-negative [`Fp128x8i32`] lanes of each canonical value, so a shift reads
/// half the bytes of a wide ring element and adds a value below `2^16` to
/// each destination lane. At most `MAX_WIDE_ACCUMULATIONS` shifts per
/// destination between flushes keep every lane inside `reduce_wide`'s `i32`
/// range.
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

    pub(super) fn load(&mut self, src: &CyclotomicRing<AkitaField, D>) {
        let (negative, positive) = self.digits.split_at_mut(D);
        for ((negative, positive), &value) in negative.iter_mut().zip(positive).zip(&src.coeffs) {
            *negative = canonical_digits(-value);
            *positive = canonical_digits(value);
        }
    }

    /// `dst += a · Σ_k X^k` over `shifts`, each `< D`.
    #[inline]
    pub(super) fn accumulate(&self, dst: &mut DigitAccumulator<D>, shifts: &[usize]) {
        Self::accumulate_slots(std::slice::from_ref(self), dst, &[shifts]);
    }

    /// `dst += Σ_s a_s · Σ_k X^k` over each window's `shifts`, each `< D`.
    /// All windows' shifts of a destination tile sum in registers, so a
    /// sparse batch touches `dst` once rather than once per window.
    pub(super) fn accumulate_slots(
        windows: &[Self],
        dst: &mut DigitAccumulator<D>,
        shifts: &[impl AsRef<[usize]>],
    ) {
        debug_assert_eq!(windows.len(), shifts.len());
        debug_assert!(shifts
            .iter()
            .flat_map(AsRef::as_ref)
            .all(|&shift| shift < D));
        let Some(first) = windows.first() else {
            return;
        };
        let mut views = [first.shifted(0); VIEW_BATCH];
        let mut len = 0;
        for (window, shifts) in windows.iter().zip(shifts) {
            for &shift in shifts.as_ref() {
                if len == VIEW_BATCH {
                    Self::accumulate_views(dst, &views);
                    len = 0;
                }
                views[len] = window.shifted(shift);
                len += 1;
            }
        }
        Self::accumulate_views(dst, &views[..len]);
    }

    /// Coefficient `j` of `a · X^shift` is entry `j` of the view.
    #[expect(
        clippy::expect_used,
        reason = "new() sizes digits to 2 * D, so the tail from D - shift holds D + shift entries"
    )]
    fn shifted(&self, shift: usize) -> &[[u16; 8]; D] {
        self.digits[D - shift..]
            .first_chunk::<D>()
            .expect("digit windows hold 2 * D entries")
    }

    // Full-length views let the tile loop skip the window/shift walk and
    // index each tile at a constant offset into a fixed-length array.
    fn accumulate_views(dst: &mut DigitAccumulator<D>, views: &[&[[u16; 8]; D]]) {
        if views.is_empty() {
            return;
        }
        for (tile, out) in dst.chunks_exact_mut(TILE).enumerate() {
            // Start with the destination so the tile needs no final addition pass.
            // Each digit is non-negative, so every intermediate sum lies between
            // the original destination and the bounded final accumulator.
            let mut sums: [[i32; 8]; TILE] = std::array::from_fn(|i| out[i].0);
            for view in views {
                for (sum, digits) in sums.iter_mut().zip(&view[tile * TILE..][..TILE]) {
                    for (sum, &digit) in sum.iter_mut().zip(digits) {
                        *sum += i32::from(digit);
                    }
                }
            }
            for (out, sum) in out.iter_mut().zip(sums) {
                out.0 = sum;
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
