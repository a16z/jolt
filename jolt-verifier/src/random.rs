use getrandom::Error;
use rand::{RngCore, SeedableRng, rngs::StdRng};
use std::ptr::{slice_from_raw_parts, slice_from_raw_parts_mut};
use std::slice;

const SEED: u64 = 0x1111222233334444;
static mut RNG: Option<StdRng> = None;

// Warning: This is a deterministic PRNG implementation. We could allow the prover/verifier to agree on a seed in the future.
// No atomics support so we're using a static mutable variable.
#[unsafe(no_mangle)]
pub unsafe extern "Rust" fn __getrandom_custom(dest: *mut u8, len: usize) -> Result<(), Error> {
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

    Ok(())
}

// register_custom_getrandom!(__getrandom_v03_custom);
