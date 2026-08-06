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
/// `T::zero()` must be represented as all-zero bytes and `T` must have no
/// padding bytes (true for Montgomery-form prime fields, where zero's
/// representative is 0). The all-zero-bytes property is asserted at runtime
/// in ALL builds — one `size_of::<T>()` byte-compare per call, nothing next
/// to the allocation — so a nonzero-repr `Zero` implementation fails loudly
/// instead of exposing invalid values in release; the assertion reads the
/// bytes of one `T::zero()` value, so it cannot detect padding.
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

    // SAFETY: reads the zero element's bytes to verify the all-zeros
    // invariant `alloc_zeroed` relies on. Runs in all builds: a release-only
    // wrong-repr instantiation would otherwise construct invalid `T` values.
    unsafe {
        let value = &T::zero();
        let ptr = std::ptr::from_ref::<T>(value).cast::<u8>();
        let bytes = std::slice::from_raw_parts(ptr, std::mem::size_of::<T>());
        assert!(
            bytes.iter().all(|&byte| byte == 0),
            "T::zero() is not all-zero bytes — unsafe_allocate_zero_vec is invalid for this type"
        );
    }

    // SAFETY: `size` and `size_of::<T>()` are nonzero (checked above), so the
    // layout satisfies `alloc_zeroed`'s nonzero-size requirement, and the
    // assertion above guarantees all-zero bytes are a valid `T` (the zero
    // element).
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
    use jolt_field::{Fr, FromPrimitiveInt};

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
    #[should_panic(expected = "not all-zero bytes")]
    fn zero_vec_rejects_non_zero_repr() {
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct WeirdZero(u8);
        impl std::ops::Add for WeirdZero {
            type Output = Self;
            fn add(self, _: Self) -> Self {
                Self(1)
            }
        }
        impl Zero for WeirdZero {
            fn zero() -> Self {
                Self(1)
            }
            fn is_zero(&self) -> bool {
                self.0 == 1
            }
        }
        let _: Vec<WeirdZero> = unsafe_allocate_zero_vec(4);
    }
}
