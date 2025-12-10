//! Two-pointer merge utilities for sparse matrix binding.
//!
//! This module provides a generic iterator for merging two sorted sequences
//! (typically "even" and "odd" entries) during sumcheck variable binding.
//!
//! # The Merge Pattern
//!
//! When binding a variable in a sparse read-write matrix, we need to merge
//! adjacent entries (either row-pairs or column-pairs). At each step:
//!
//! - **Both present, same key**: Merge the two entries explicitly.
//! - **Only even present**: Use the "odd" implicit value from a checkpoint.
//! - **Only odd present**: Use the "even" implicit value from a checkpoint.
//!
//! This pattern appears in:
//! - `RegisterMatrixCycleMajor::bind` (merging by column within row-pairs)
//! - `RegisterMatrixAddressMajor::bind` (merging by row within column-pairs)
//! - Sumcheck message computation (iterating over merge pairs)
//!
//! # Usage
//!
//! ```ignore
//! let merge = TwoPointerMerge::new(even_slice, odd_slice, |e| e.key);
//! for item in merge {
//!     match item {
//!         MergeItem::Both(e, o) => { /* both present */ }
//!         MergeItem::EvenOnly(e) => { /* only even, odd implicit */ }
//!         MergeItem::OddOnly(o) => { /* only odd, even implicit */ }
//!     }
//! }
//! ```

use crate::field::{JoltField, OptimizedMul};

/// Result of a two-pointer merge step.
///
/// Each variant indicates which entries are explicitly present at the
/// current merge key. The caller must handle implicit entries appropriately
/// (typically using a "checkpoint" value that tracks the running state).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeItem<'a, T> {
    /// Both even and odd entries are present at the same key.
    Both { even: &'a T, odd: &'a T },
    /// Only the even entry is present; odd is implicit.
    EvenOnly(&'a T),
    /// Only the odd entry is present; even is implicit.
    OddOnly(&'a T),
}

impl<'a, T> MergeItem<'a, T> {
    /// Returns `true` if this is a `Both` variant.
    #[inline]
    pub fn is_both(&self) -> bool {
        matches!(self, MergeItem::Both { .. })
    }

    /// Returns the even entry if present.
    #[inline]
    pub fn even(&self) -> Option<&'a T> {
        match self {
            MergeItem::Both { even, .. } | MergeItem::EvenOnly(even) => Some(even),
            MergeItem::OddOnly(_) => None,
        }
    }

    /// Returns the odd entry if present.
    #[inline]
    pub fn odd(&self) -> Option<&'a T> {
        match self {
            MergeItem::Both { odd, .. } | MergeItem::OddOnly(odd) => Some(odd),
            MergeItem::EvenOnly(_) => None,
        }
    }
}

/// A two-pointer merge iterator over two sorted slices.
///
/// Given two slices sorted by a key function, this iterator yields
/// `MergeItem` variants that indicate which entries are present at
/// each unique key value.
///
/// # Invariants
///
/// - Both input slices must be sorted by the key function in ascending order.
/// - The iterator yields items in ascending key order.
///
/// # Example
///
/// ```ignore
/// struct Entry { row: usize, value: i32 }
///
/// let even = vec![Entry { row: 0, value: 10 }, Entry { row: 2, value: 30 }];
/// let odd = vec![Entry { row: 1, value: 20 }, Entry { row: 2, value: 35 }];
///
/// let merge = TwoPointerMerge::new(&even, &odd, |e| e.row);
/// for item in merge {
///     // row 0: EvenOnly
///     // row 1: OddOnly
///     // row 2: Both
/// }
/// ```
pub struct TwoPointerMerge<'a, T, K, F>
where
    F: Fn(&T) -> K,
    K: Ord,
{
    even: &'a [T],
    odd: &'a [T],
    ei: usize,
    oi: usize,
    key_fn: F,
}

impl<'a, T, K, F> TwoPointerMerge<'a, T, K, F>
where
    F: Fn(&T) -> K,
    K: Ord,
{
    /// Create a new merge iterator.
    ///
    /// # Arguments
    ///
    /// - `even`: Slice of "even" entries, sorted by key.
    /// - `odd`: Slice of "odd" entries, sorted by key.
    /// - `key_fn`: Function to extract the merge key from an entry.
    #[inline]
    pub fn new(even: &'a [T], odd: &'a [T], key_fn: F) -> Self {
        Self {
            even,
            odd,
            ei: 0,
            oi: 0,
            key_fn,
        }
    }

    /// Returns `true` if there are more items to yield.
    #[inline]
    pub fn has_next(&self) -> bool {
        self.ei < self.even.len() || self.oi < self.odd.len()
    }

    /// Peek at the next even entry without consuming it.
    #[inline]
    pub fn peek_even(&self) -> Option<&'a T> {
        self.even.get(self.ei)
    }

    /// Peek at the next odd entry without consuming it.
    #[inline]
    pub fn peek_odd(&self) -> Option<&'a T> {
        self.odd.get(self.oi)
    }
}

impl<'a, T, K, F> Iterator for TwoPointerMerge<'a, T, K, F>
where
    F: Fn(&T) -> K,
    K: Ord,
{
    type Item = MergeItem<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let even_entry = self.even.get(self.ei);
        let odd_entry = self.odd.get(self.oi);

        match (even_entry, odd_entry) {
            (Some(e), Some(o)) => {
                let key_e = (self.key_fn)(e);
                let key_o = (self.key_fn)(o);

                match key_e.cmp(&key_o) {
                    std::cmp::Ordering::Equal => {
                        self.ei += 1;
                        self.oi += 1;
                        Some(MergeItem::Both { even: e, odd: o })
                    }
                    std::cmp::Ordering::Less => {
                        self.ei += 1;
                        Some(MergeItem::EvenOnly(e))
                    }
                    std::cmp::Ordering::Greater => {
                        self.oi += 1;
                        Some(MergeItem::OddOnly(o))
                    }
                }
            }
            (Some(e), None) => {
                self.ei += 1;
                Some(MergeItem::EvenOnly(e))
            }
            (None, Some(o)) => {
                self.oi += 1;
                Some(MergeItem::OddOnly(o))
            }
            (None, None) => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining_even = self.even.len() - self.ei;
        let remaining_odd = self.odd.len() - self.oi;
        // Minimum: all are Both (max of the two)
        // Maximum: none are Both (sum of the two)
        (
            remaining_even.max(remaining_odd),
            Some(remaining_even + remaining_odd),
        )
    }
}

// ============================================================================
// Binding Helpers
// ============================================================================

/// Linear interpolation: `even + r * (odd - even)`.
///
/// This is the standard binding operation for sumcheck variables.
/// Computes the evaluation of the linear polynomial passing through
/// `(0, even)` and `(1, odd)` at point `r`.
#[inline]
pub fn linear_interpolate<F: JoltField>(even: F, odd: F, r: F::Challenge) -> F {
    even + r.mul_0_optimized(odd - even)
}

/// Bind two optional field elements.
///
/// This handles the four cases of binding optional RA/WA coefficients:
///
/// - `(Some(e), Some(o))`: Standard interpolation `e + r*(o - e)`
/// - `(Some(e), None)`: Implicit odd = 0, result is `(1-r)*e`
/// - `(None, Some(o))`: Implicit even = 0, result is `r*o`
/// - `(None, None)`: Both implicit 0, result is `None`
///
/// # Important
///
/// Using `None` vs `Some(F::zero())` is semantically meaningful:
/// - `None` means "no access of this type occurred"
/// - `Some(F::zero())` could arise from binding and is a valid coefficient
///
/// Using `F::zero()` as a sentinel value would INVALIDATE the sumcheck.
#[inline]
pub fn bind_optional<F: JoltField>(even: Option<F>, odd: Option<F>, r: F::Challenge) -> Option<F> {
    match (even, odd) {
        (Some(e), Some(o)) => Some(e + r.mul_0_optimized(o - e)),
        (Some(e), None) => Some((F::one() - r).mul_1_optimized(e)),
        (None, Some(o)) => Some(r.mul_1_optimized(o)),
        (None, None) => None,
    }
}

/// Scale an optional field element by a factor.
///
/// Used when binding a present entry with an implicit zero partner.
///
/// - For even-only (odd implicit 0): use `scale = 1 - r`
/// - For odd-only (even implicit 0): use `scale = r`
#[inline]
pub fn scale_optional<F: JoltField>(value: Option<F>, scale: F) -> Option<F> {
    value.map(|v| scale * v)
}

/// Compute evaluations at x=0 and x=2 for a polynomial passing through (0, f0) and (1, f1).
///
/// Used in sumcheck message computation where we need evaluations at multiple points.
///
/// Returns `[f(0), f(2)]` where `f(x) = f0 + x*(f1 - f0)` (linear interpolation).
/// - `f(0) = f0`
/// - `f(2) = f1 + (f1 - f0) = 2*f1 - f0`
#[inline]
pub fn eval_at_0_and_2<F: JoltField>(f0: F, f1: F) -> [F; 2] {
    [f0, f1 + f1 - f0]
}

/// Compute evaluations at x=0, x=2, and x=3 for a polynomial through (0, f0) and (1, f1).
///
/// Returns `[f(0), f(2), f(3)]`.
/// - `f(0) = f0`
/// - `f(2) = 2*f1 - f0`
/// - `f(3) = 3*f1 - 2*f0`
#[inline]
pub fn eval_at_0_2_3<F: JoltField>(f0: F, f1: F) -> [F; 3] {
    let delta = f1 - f0;
    [f0, f1 + delta, f1 + delta + delta]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::One;

    type F = Fr;

    #[derive(Debug, Clone, PartialEq)]
    struct TestEntry {
        key: usize,
        value: i32,
    }

    #[test]
    fn test_merge_both_present() {
        let even = vec![
            TestEntry { key: 0, value: 10 },
            TestEntry { key: 2, value: 30 },
        ];
        let odd = vec![
            TestEntry { key: 0, value: 15 },
            TestEntry { key: 2, value: 35 },
        ];

        let items: Vec<_> = TwoPointerMerge::new(&even, &odd, |e| e.key).collect();

        assert_eq!(items.len(), 2);
        assert!(matches!(items[0], MergeItem::Both { .. }));
        assert!(matches!(items[1], MergeItem::Both { .. }));
    }

    #[test]
    fn test_merge_interleaved() {
        let even = vec![
            TestEntry { key: 0, value: 10 },
            TestEntry { key: 2, value: 30 },
        ];
        let odd = vec![
            TestEntry { key: 1, value: 20 },
            TestEntry { key: 3, value: 40 },
        ];

        let items: Vec<_> = TwoPointerMerge::new(&even, &odd, |e| e.key).collect();

        assert_eq!(items.len(), 4);
        assert!(matches!(items[0], MergeItem::EvenOnly(_)));
        assert!(matches!(items[1], MergeItem::OddOnly(_)));
        assert!(matches!(items[2], MergeItem::EvenOnly(_)));
        assert!(matches!(items[3], MergeItem::OddOnly(_)));
    }

    #[test]
    fn test_merge_mixed() {
        let even = vec![
            TestEntry { key: 0, value: 10 },
            TestEntry { key: 2, value: 30 },
            TestEntry { key: 4, value: 50 },
        ];
        let odd = vec![
            TestEntry { key: 1, value: 20 },
            TestEntry { key: 2, value: 35 },
        ];

        let items: Vec<_> = TwoPointerMerge::new(&even, &odd, |e| e.key).collect();

        assert_eq!(items.len(), 4);
        assert!(matches!(items[0], MergeItem::EvenOnly(_))); // key 0
        assert!(matches!(items[1], MergeItem::OddOnly(_))); // key 1
        assert!(matches!(items[2], MergeItem::Both { .. })); // key 2
        assert!(matches!(items[3], MergeItem::EvenOnly(_))); // key 4
    }

    #[test]
    fn test_merge_empty_slices() {
        let even: Vec<TestEntry> = vec![];
        let odd: Vec<TestEntry> = vec![];

        let items: Vec<_> = TwoPointerMerge::new(&even, &odd, |e| e.key).collect();
        assert!(items.is_empty());
    }

    #[test]
    fn test_merge_one_empty() {
        let even = vec![TestEntry { key: 0, value: 10 }];
        let odd: Vec<TestEntry> = vec![];

        let items: Vec<_> = TwoPointerMerge::new(&even, &odd, |e| e.key).collect();
        assert_eq!(items.len(), 1);
        assert!(matches!(items[0], MergeItem::EvenOnly(_)));

        let items: Vec<_> = TwoPointerMerge::new(&odd, &even, |e| e.key).collect();
        assert_eq!(items.len(), 1);
        assert!(matches!(items[0], MergeItem::OddOnly(_)));
    }

    #[test]
    fn test_linear_interpolate() {
        let even = F::from(10u64);
        let odd = F::from(20u64);
        let r: <F as JoltField>::Challenge = 3u128.into();

        // f(r) = even + r*(odd - even)
        let result = linear_interpolate(even, odd, r);
        let expected = even + r * (odd - even);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bind_optional_both_present() {
        let even = Some(F::from(3u64));
        let odd = Some(F::from(7u64));
        let r: <F as JoltField>::Challenge = 2u128.into();

        // e + r*(o - e)
        let result = bind_optional(even, odd, r);
        let expected = F::from(3u64) + r * (F::from(7u64) - F::from(3u64));
        assert_eq!(result, Some(expected));
    }

    #[test]
    fn test_bind_optional_even_only() {
        let even = Some(F::from(5u64));
        let odd = None;
        let r: <F as JoltField>::Challenge = 3u128.into();

        // (1-r)*e
        let result = bind_optional::<F>(even, odd, r);
        let expected = (F::one() - r) * F::from(5u64);
        assert_eq!(result, Some(expected));
    }

    #[test]
    fn test_bind_optional_odd_only() {
        let even = None;
        let odd = Some(F::from(4u64));
        let r: <F as JoltField>::Challenge = 2u128.into();

        // r*o
        let result = bind_optional::<F>(even, odd, r);
        let expected = r * F::from(4u64);
        assert_eq!(result, Some(expected));
    }

    #[test]
    fn test_bind_optional_both_none() {
        let even: Option<F> = None;
        let odd: Option<F> = None;
        let r: <F as JoltField>::Challenge = 5u128.into();

        let result = bind_optional::<F>(even, odd, r);
        assert_eq!(result, None);
    }

    #[test]
    fn test_eval_at_0_and_2() {
        let f0 = F::from(10u64);
        let f1 = F::from(30u64);

        let [e0, e2] = eval_at_0_and_2(f0, f1);

        // f(0) = 10
        assert_eq!(e0, F::from(10u64));
        // f(2) = 2*30 - 10 = 50
        assert_eq!(e2, F::from(50u64));
    }

    #[test]
    fn test_eval_at_0_2_3() {
        let f0 = F::from(10u64);
        let f1 = F::from(30u64);

        let [e0, e2, e3] = eval_at_0_2_3(f0, f1);

        // f(0) = 10
        assert_eq!(e0, F::from(10u64));
        // f(2) = 2*30 - 10 = 50
        assert_eq!(e2, F::from(50u64));
        // f(3) = 3*30 - 2*10 = 70
        assert_eq!(e3, F::from(70u64));
    }

    #[test]
    fn test_size_hint() {
        let even = vec![
            TestEntry { key: 0, value: 10 },
            TestEntry { key: 2, value: 30 },
        ];
        let odd = vec![TestEntry { key: 1, value: 20 }];

        let merge = TwoPointerMerge::new(&even, &odd, |e| e.key);
        let (min, max) = merge.size_hint();

        // min = max(2, 1) = 2 (if all were Both)
        // max = 2 + 1 = 3 (if none are Both)
        assert_eq!(min, 2);
        assert_eq!(max, Some(3));
    }
}
