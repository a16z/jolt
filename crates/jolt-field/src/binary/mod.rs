//! Binary fields for Jolt matching Akita's binary-field commitment (LaBinius).
//!
//! Raw words and canonical bytes are polynomial coefficients; integer ring
//! maps retain only parity. Challenges require a full field-width squeeze for
//! uniform sampling, in particular 24 bytes for `F192`.

mod f128;
mod f192;
mod f64;

pub use f128::F128;
pub use f192::F192;
pub use f64::F64;

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
