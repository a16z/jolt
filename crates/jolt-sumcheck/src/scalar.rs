use std::{
    fmt::{Debug, Display},
    hash::Hash,
};

use jolt_field::{CanonicalRepr, FieldCore, FromPrimitiveInt};

/// Scalar capabilities used by the verifier-side sumcheck crate.
pub trait SumcheckScalar:
    FieldCore
    + FromPrimitiveInt
    + CanonicalRepr
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
    F: FieldCore
        + FromPrimitiveInt
        + FromPrimitiveInt
        + CanonicalRepr
        + CanonicalRepr
        + CanonicalRepr
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
