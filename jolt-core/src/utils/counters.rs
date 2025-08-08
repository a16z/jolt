use std::sync::atomic::{AtomicUsize, Ordering};

// A global counter that tracks how many times we've use a*b where a, b are ark_bn254::Fr types
// see jolt-core::field::tracked_fr::TrackedFr for more details
pub static MULT_COUNT: AtomicUsize = AtomicUsize::new(0);
pub static INVERSE_COUNT: AtomicUsize = AtomicUsize::new(0);

pub fn reset_inverse_count() {
    INVERSE_COUNT.store(0, Ordering::Relaxed);
}

pub fn get_inverse_count() -> usize {
    INVERSE_COUNT.load(Ordering::Relaxed)
}
pub fn reset_mult_count() {
    MULT_COUNT.store(0, Ordering::Relaxed);
}

pub fn get_mult_count() -> usize {
    MULT_COUNT.load(Ordering::Relaxed)
}
