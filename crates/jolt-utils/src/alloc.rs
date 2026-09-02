//! Zeroed vector allocation via `alloc_zeroed`.
//!
//! `vec![T::zero(); n]` clone-fills element by element, which at witness-grid
//! scale is a full serial memory pass. `alloc_zeroed` gets zero pages from
//! the OS instead.

use std::alloc::Layout;

use num_traits::Zero;

/// Allocates `vec![T::zero(); size]` through `alloc_zeroed`.
///
/// # Safety contract
///
/// The caller must ensure that the all-zero byte pattern is a valid `T` equal
/// to `T::zero()`.
#[expect(
    clippy::unwrap_used,
    reason = "Layout::array only fails on overflow, which callers' size arithmetic already rules out"
)]
pub fn unsafe_allocate_zero_vec<T: Sized + Zero>(size: usize) -> Vec<T> {
    // `alloc_zeroed` requires a nonzero layout size: use safe construction
    // for empty vectors and zero-sized `T` instead of violating that contract.
    if size == 0 || std::mem::size_of::<T>() == 0 {
        return std::iter::repeat_with(T::zero).take(size).collect();
    }

    // SAFETY: `size` and `size_of::<T>()` are nonzero (checked above), so the
    // layout satisfies `alloc_zeroed`'s nonzero-size requirement. The caller
    // guarantees the allocation contains `size` initialized `T` values.
    unsafe {
        let layout = Layout::array::<T>(size).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout).cast::<T>();
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        #[expect(clippy::same_length_and_capacity)]
        Vec::from_raw_parts(ptr, size, size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, Ring};

    #[test]
    fn zero_vec_u64() {
        let v: Vec<u64> = unsafe_allocate_zero_vec(1024);
        assert_eq!(v.len(), 1024);
        assert!(v.iter().all(|&x| x == 0));
    }

    #[test]
    fn zero_vec_f64() {
        let v: Vec<f64> = unsafe_allocate_zero_vec(256);
        assert_eq!(v.len(), 256);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn zero_vec_field_elements() {
        let table: Vec<Fr> = unsafe_allocate_zero_vec(1 << 10);
        assert_eq!(table, vec![Fr::from_u64(0); 1 << 10]);
        assert!(unsafe_allocate_zero_vec::<Fr>(0).is_empty());
    }

    #[test]
    fn zero_vec_zero_sized_type() {
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct Zst;
        impl std::ops::Add for Zst {
            type Output = Self;
            fn add(self, _: Self) -> Self {
                Self
            }
        }
        impl Zero for Zst {
            fn zero() -> Self {
                Self
            }
            fn is_zero(&self) -> bool {
                true
            }
        }
        let v: Vec<Zst> = unsafe_allocate_zero_vec(8);
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn zero_vec_padded_type() {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct PaddedZero {
            byte: u8,
            word: u32,
        }
        impl std::ops::Add for PaddedZero {
            type Output = Self;
            fn add(self, _: Self) -> Self {
                self
            }
        }
        impl Zero for PaddedZero {
            fn zero() -> Self {
                Self { byte: 0, word: 0 }
            }
            fn is_zero(&self) -> bool {
                self.byte == 0 && self.word == 0
            }
        }
        let values: Vec<PaddedZero> = unsafe_allocate_zero_vec(4);
        assert!(values.iter().all(Zero::is_zero));
    }
}
