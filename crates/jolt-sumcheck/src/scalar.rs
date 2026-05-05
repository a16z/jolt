use std::{
    fmt::{Debug, Display},
    hash::Hash,
};

use jolt_field::{
    CanonicalBytes, FieldCore, FixedByteSize, FromPrimitiveInt, MulPow2, TranscriptChallenge,
};

/// Scalar capabilities used by the verifier-side sumcheck crate.
pub trait SumcheckScalar:
    FieldCore
    + FromPrimitiveInt
    + MulPow2
    + CanonicalBytes
    + FixedByteSize
    + TranscriptChallenge
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
        + MulPow2
        + CanonicalBytes
        + FixedByteSize
        + TranscriptChallenge
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
