//! Sampling ring elements with a prescribed constant term.

use jolt_field::{CyclotomicRing, Field};
use rand_core::{CryptoRng, RngCore};

/// A uniform element of `{a ∈ Z_q[X]/(X^D + 1) : ct(a) = 0}`.
pub fn sample_zero_constant_term<F: Field, const D: usize, R: RngCore + CryptoRng>(
    rng: &mut R,
) -> CyclotomicRing<F, D> {
    let mut a = CyclotomicRing::random(rng);
    if let Some(c0) = a.coefficients_mut().first_mut() {
        *c0 = F::zero();
    }
    a
}
