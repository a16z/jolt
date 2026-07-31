//! Threading utilities for polynomial operations.

use num_traits::Zero;

/// Drops `data` in a background rayon task to avoid blocking the caller.
#[cfg(feature = "parallel")]
pub fn drop_in_background_thread<T: Send + 'static>(data: T) {
    rayon::spawn(move || drop(data));
}

/// Allocates a zeroed `Vec<T>` of `size` elements using `alloc_zeroed`.
///
/// # Safety contract
///
/// `T::zero()` must be represented as all-zero bytes and `T` must have no
/// padding bytes. The all-zero-bytes property is asserted at runtime in all
/// builds (one byte-compare of `size_of::<T>()` bytes per call); the assertion
/// reads the bytes of a `T::zero()` value, so it cannot detect padding.
#[expect(clippy::unwrap_used)]
pub fn unsafe_allocate_zero_vec<T: Sized + Zero>(size: usize) -> Vec<T> {
    // `alloc_zeroed` requires a nonzero layout size: use safe construction for
    // empty vectors and zero-sized `T` instead of violating that contract.
    if size == 0 || std::mem::size_of::<T>() == 0 {
        return std::iter::repeat_with(T::zero).take(size).collect();
    }

    // SAFETY: We read the zero representation as raw bytes to verify the
    // all-zeros invariant that `alloc_zeroed` relies on. Runs in all builds:
    // a release-only wrong-`Zero`-impl instantiation would otherwise produce
    // invalid `T` values, and the check costs one small compare per call.
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
    // layout satisfies `alloc_zeroed`'s nonzero-size requirement. The assertion
    // above guarantees the all-zero byte pattern is a valid `T` (`T::zero()`),
    // so the resulting `Vec<T>` contains `size` initialized `T` values.
    unsafe {
        let layout = std::alloc::Layout::array::<T>(size).unwrap();
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
    fn zero_vec_empty() {
        let v: Vec<u64> = unsafe_allocate_zero_vec(0);
        assert!(v.is_empty());
    }

    #[test]
    fn zero_vec_zst() {
        #[derive(Clone, Copy, PartialEq, Debug)]
        struct Zst;
        impl std::ops::Add for Zst {
            type Output = Self;
            fn add(self, _: Self) -> Self {
                Zst
            }
        }
        impl Zero for Zst {
            fn zero() -> Self {
                Zst
            }
            fn is_zero(&self) -> bool {
                true
            }
        }

        let v: Vec<Zst> = unsafe_allocate_zero_vec(17);
        assert_eq!(v.len(), 17);
    }

    #[test]
    #[should_panic(expected = "not all-zero bytes")]
    fn zero_vec_rejects_non_zero_repr() {
        #[derive(Clone, Copy, PartialEq, Debug)]
        struct WeirdZero(u8);
        impl std::ops::Add for WeirdZero {
            type Output = Self;
            fn add(self, _: Self) -> Self {
                WeirdZero(1)
            }
        }
        impl Zero for WeirdZero {
            fn zero() -> Self {
                WeirdZero(1)
            }
            fn is_zero(&self) -> bool {
                self.0 == 1
            }
        }

        let _: Vec<WeirdZero> = unsafe_allocate_zero_vec(4);
    }
}
