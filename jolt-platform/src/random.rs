use crate::jolt_println;
use getrandom::Error;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use std::slice;
use std::sync::{Mutex, OnceLock};

// JOLT in ASCII padded with 1s
const SEED: u64 = 0x11114A4F4C541111;
static RNG: OnceLock<Mutex<StdRng>> = OnceLock::new();

// Warning: This is a deterministic PRNG implementation. We could allow the prover/verifier to agree on a seed in the future.
// No atomics support so we're using a static mutable variable
#[allow(clippy::missing_safety_doc)]
#[no_mangle]
pub unsafe fn sys_rand(dest: *mut u8, len: usize) {
    jolt_println!("Warning: sys_rand is a deterministic PRNG, not a cryptographically secure RNG. Use with caution.");
    let dest = slice::from_raw_parts_mut(dest, len);
    let rng_mutex = RNG.get_or_init(|| Mutex::new(StdRng::seed_from_u64(SEED)));
    // lock poisoning here is unlikely; if it happens, propagate panic rather than UB
    rng_mutex.lock().expect("RNG mutex poisoned").fill_bytes(dest);
}

#[allow(clippy::missing_safety_doc)]
#[no_mangle]
pub unsafe extern "C" fn __getrandom_custom(dest: *mut u8, len: usize) -> Result<(), Error> {
    sys_rand(dest, len);

    Ok(())
}

// register_custom_getrandom!(__getrandom_v03_custom);
