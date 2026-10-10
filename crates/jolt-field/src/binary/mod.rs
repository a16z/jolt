//! Binary fields for Jolt matching Akita's binary-field commitment (LaBinius).
//!
//! Raw words and canonical bytes are polynomial coefficients; integer ring
//! maps retain only parity. Challenges require a full field-width squeeze for
//! uniform sampling, in particular 24 bytes for `F192`.
//!
//! `F8`, `F64`, `F128`, and `F192` implement the field spine. `From<F8>`
//! embeds into `F64` and `F128` by sending x to the smallest raw root of
//! x^8 + x^4 + x^3 + x + 1; the `F192` embedding lifts through `F64`.
//!
//! `F64`, `F128`, and `F192` defer reduction through `WithAccumulator`:
//! products accumulate by XOR in fixed-width polynomial state with no term
//! limit. `F192` folds the extension modulus before accumulating and reduces
//! its three base-field coefficients only when the accumulator is finalized.

mod accumulator;
mod embed;
mod f128;
mod f192;
mod f64;
mod f8;

#[cfg(any(
    all(target_arch = "aarch64", target_feature = "aes"),
    all(target_arch = "x86_64", target_feature = "pclmulqdq")
))]
mod kernels;
#[cfg_attr(
    all(
        not(test),
        any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq")
        )
    ),
    expect(
        dead_code,
        reason = "portable arithmetic stays compiled for kernel differential tests"
    )
)]
mod portable;

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[path = "arch/aarch64.rs"]
mod arch;
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[path = "arch/x86_64.rs"]
mod arch;

#[cfg(any(
    all(target_arch = "aarch64", target_feature = "aes"),
    all(target_arch = "x86_64", target_feature = "pclmulqdq")
))]
use kernels as arithmetic;
#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "aes"),
    all(target_arch = "x86_64", target_feature = "pclmulqdq")
)))]
use portable as arithmetic;

pub use accumulator::F128Accumulator;
pub use f128::F128;
pub use f192::F192;
pub use f64::F64;
pub use f8::F8;

use crate::{CanonicalEncoding, Field};

fn inverse<F: Field + CanonicalEncoding>(value: F) -> Option<F> {
    if value.is_zero() {
        return None;
    }
    // After n - 2 iterations the exponent is 2^(n - 1) - 1; square to 2^n - 2.
    let mut result = value;
    for _ in 0..F::MODULUS_BITS - 3 {
        result = result.square() * value;
    }
    Some(result.square())
}

#[cfg(test)]
mod tests;
