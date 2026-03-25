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
/// The caller must ensure that `T::zero()` is represented as all-zero bytes.
/// This is verified in debug/test builds via an assertion.
#[allow(clippy::all)]
pub fn unsafe_allocate_zero_vec<T: Sized + Zero>(size: usize) -> Vec<T> {
    #[cfg(test)]
    {
        // SAFETY: We read the zero representation as raw bytes to verify the
        // all-zeros invariant that `alloc_zeroed` relies on.
        unsafe {
            let value = &T::zero();
            let ptr = std::ptr::from_ref::<T>(value).cast::<u8>();
            let bytes = std::slice::from_raw_parts(ptr, std::mem::size_of::<T>());
            assert!(
                bytes.iter().all(|&byte| byte == 0),
                "T::zero() is not all-zero bytes — unsafe_allocate_zero_vec is invalid for this type"
            );
        }
    }

    // SAFETY: `alloc_zeroed` produces a valid zero-initialized allocation.
    // The caller guarantees that `T::zero()` is all-zero bytes, so the
    // resulting `Vec<T>` contains valid `T` values.
    unsafe {
        let layout = std::alloc::Layout::array::<T>(size).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout).cast::<T>();
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        #[allow(clippy::same_length_and_capacity)]
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
}
