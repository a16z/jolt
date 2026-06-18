use core::slice;
use rand::{rngs::StdRng, RngCore, SeedableRng};

// JOLT in ASCII padded with 1s
const SEED: u64 = 0x11114A4F4C541111;
static mut RNG: Option<StdRng> = None;

// Warning: This is a deterministic PRNG implementation. We could allow the prover/verifier to agree on a seed in the future.
// No atomics support so we're using a static mutable variable

/// Fills `dest` with `len` pseudo-random bytes from a deterministic PRNG.
///
/// # Safety
///
/// - `dest` must point to a valid, writable buffer of at least `len` bytes.
/// - `dest` must be properly aligned for `u8`.
/// - The memory region `[dest, dest + len)` must not overlap with the internal
///   `RNG` static.
/// - This function is **not** thread-safe; it mutates a global static without
///   synchronization. Callers must ensure exclusive access.
#[no_mangle]
pub unsafe fn sys_rand(dest: *mut u8, len: usize) {
    crate::println!("Warning: sys_rand is a deterministic PRNG, not a cryptographically secure RNG. Use with caution.");

    let rng_ptr: *mut Option<StdRng> = &raw mut RNG;

    unsafe {
        if (*rng_ptr).is_none() {
            *rng_ptr = Some(StdRng::seed_from_u64(SEED));
        }

        let dest = slice::from_raw_parts_mut(dest, len);

        // SAFETY: Direct field access, no shared reference created
        match RNG {
            Some(ref mut rng) => rng.fill_bytes(dest),
            None => unreachable!(), // just in case
        }
    }
}

#[no_mangle]
pub fn _getrandom_v02(s: &mut [u8]) -> Result<(), getrandom_v02::Error> {
    // SAFETY: s is a valid mutable slice, so its pointer and length satisfy sys_rand's requirements.
    // Not thread-safe (sys_rand uses a global static), but this is acceptable in the single-threaded
    // guest environment where getrandom is called.
    unsafe {
        sys_rand(s.as_mut_ptr(), s.len());
    }

    Ok(())
}

/// Custom `getrandom` v0.3 backend for the Jolt guest environment.
///
/// # Safety
///
/// - `dest` must point to a valid, writable buffer of at least `len` bytes.
/// - `dest` must be properly aligned for `u8`.
/// - This function is **not** thread-safe; it delegates to [`sys_rand`] which
///   mutates a global static without synchronization.
#[unsafe(no_mangle)]
unsafe extern "Rust" fn __getrandom_v03_custom(
    dest: *mut u8,
    len: usize,
) -> Result<(), getrandom_v03::Error> {
    // SAFETY: Caller guarantees dest and len satisfy sys_rand's safety requirements.
    sys_rand(dest, len);

    Ok(())
}

// Register the custom getrandom implementation for getrandom v0.2 on non-wasm targets
// For wasm targets we use the "js" feature which uses the browser's crypto API
#[cfg(not(any(target_arch = "wasm32")))]
getrandom_v02::register_custom_getrandom!(_getrandom_v02);
