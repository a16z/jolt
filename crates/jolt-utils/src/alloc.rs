//! Zeroed vector allocation via `alloc_zeroed`.
//!
//! `vec![T::zero(); n]` clone-fills element by element, which at witness-grid
//! scale is a full serial memory pass. `alloc_zeroed` gets zero pages from
//! the OS instead.

use std::alloc::Layout;

use num_traits::Zero;

/// Types whose all-zero byte representation is a valid additive identity.
///
/// # Safety
/// All-zero bytes must represent `Self::zero()`, including for every field of
/// the type. No reference or nonzero niche may require another representation.
pub unsafe trait ZeroBytesValid: Zero {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZeroAllocError {
    CapacityOverflow,
    AllocFailed { layout: Layout },
}

/// Allocates zero elements without requiring a serial initialization pass.
#[inline]
pub fn try_zeroed_vec<T: ZeroBytesValid>(len: usize) -> Result<Vec<T>, ZeroAllocError> {
    // SAFETY: ZeroBytesValid guarantees that zero bytes initialize T.
    unsafe { raw_zeroed(len) }
}

#[inline]
pub fn zeroed_vec<T: ZeroBytesValid>(len: usize) -> Vec<T> {
    infallible_zeroed(try_zeroed_vec(len))
}

/// Allocates `vec![T::zero(); size]` through `alloc_zeroed`.
///
/// # Safety contract
/// The caller must ensure that the all-zero byte pattern is a valid `T` equal
/// to `T::zero()`.
#[inline]
pub fn unsafe_allocate_zero_vec<T: Sized + Zero>(size: usize) -> Vec<T> {
    // SAFETY: this function's caller guarantees that zero bytes initialize T.
    infallible_zeroed(unsafe { raw_zeroed(size) })
}

#[inline]
fn infallible_zeroed<T>(result: Result<Vec<T>, ZeroAllocError>) -> Vec<T> {
    match result {
        Ok(values) => values,
        #[expect(
            clippy::panic,
            reason = "preserves the infallible allocator's capacity-overflow contract"
        )]
        Err(ZeroAllocError::CapacityOverflow) => panic!("zeroed vector capacity overflow"),
        Err(ZeroAllocError::AllocFailed { layout }) => std::alloc::handle_alloc_error(layout),
    }
}

/// # Safety
/// All-zero bytes must be a valid T equal to T::zero().
#[inline]
unsafe fn raw_zeroed<T: Zero>(len: usize) -> Result<Vec<T>, ZeroAllocError> {
    let layout = Layout::array::<T>(len).map_err(|_| ZeroAllocError::CapacityOverflow)?;
    if layout.size() == 0 {
        return Ok(std::iter::repeat_with(T::zero).take(len).collect());
    }
    // SAFETY: layout has nonzero size; the caller guarantees valid zero values.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) }.cast::<T>();
    if ptr.is_null() {
        return Err(ZeroAllocError::AllocFailed { layout });
    }
    // SAFETY: ptr owns len initialized elements and capacity is exactly len.
    #[expect(clippy::same_length_and_capacity)]
    Ok(unsafe { Vec::from_raw_parts(ptr, len, len) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, Ring};

    #[test]
    fn zero_vec_field_elements() {
        let table: Vec<Fr> = unsafe_allocate_zero_vec(1 << 10);
        assert_eq!(table, vec![Fr::from_u64(0); 1 << 10]);
        assert_eq!(unsafe_allocate_zero_vec::<Fr>(0), [] as [Fr; 0]);
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
