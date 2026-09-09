//! The Blake2b-256 hasher behind the lattice layout digests: the `blake2`
//! crate, or the jolt-inlines hasher on a `blake2-inline` build (a RISC-V
//! guest verifier). Same bytes either way.

pub(crate) use blake2::Digest;

#[cfg(not(feature = "blake2-inline"))]
pub(crate) type Blake2b256 = blake2::Blake2b<blake2::digest::consts::U32>;
#[cfg(feature = "blake2-inline")]
pub(crate) type Blake2b256 =
    jolt_inlines_blake2::digest_adapter::Blake2b<jolt_inlines_blake2::digest_adapter::U32>;
