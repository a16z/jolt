use crate::jolt_println;
use getrandom::Error;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use std::slice;

// JOLT in ASCII padded with 1s
const SEED: u64 = 0x11114A4F4C541111;
static mut RNG: Option<StdRng> = None;

// Warning: This is a deterministic PRNG implementation. We could allow the prover/verifier to agree on a seed in the future.
// No atomics support so we're using a static mutable variable
#[allow(clippy::missing_safety_doc)]
#[no_mangle]
pub unsafe fn sys_rand(dest: *mut u8, len: usize) {
    jolt_println!("Warning: sys_rand is a deterministic PRNG, not a cryptographically secure RNG. Use with caution.");

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

#[allow(clippy::missing_safety_doc)]
#[no_mangle]
pub unsafe extern "Rust" fn __getrandom_custom(dest: *mut u8, len: usize) -> Result<(), Error> {
    sys_rand(dest, len);

    Ok(())
}

// register_custom_getrandom!(__getrandom_v03_custom);
