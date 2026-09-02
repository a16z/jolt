use std::{
    fmt::{Debug, Display},
    hash::Hash,
};

use jolt_field::{CanonicalEncoding, Field, Ring};

/// Scalar capabilities used by the verifier-side sumcheck crate.
pub trait SumcheckScalar:
    Field
    + Ring
    + CanonicalEncoding
    + Copy
    + Default
    + Eq
    + Debug
    + Display
    + Hash
    + Send
    + Sync
    + 'static
{
}

impl<F> SumcheckScalar for F where
    F: Field
        + Ring
        + Ring
        + CanonicalEncoding
        + CanonicalEncoding
        + CanonicalEncoding
        + Copy
        + Default
        + Eq
        + Debug
        + Display
        + Hash
        + Send
        + Sync
        + 'static
{
}
