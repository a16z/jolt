//! The Blake2b-256 hasher behind the lattice layout digests: the `blake2`
//! crate, or the jolt-inlines hasher on a `blake2-inline` build (a RISC-V
//! guest verifier). Same bytes either way.

pub(crate) use blake2::Digest;
#[cfg(not(feature = "blake2-inline"))]
use blake2::{digest::consts::U32, Blake2b};
#[cfg(feature = "blake2-inline")]
use jolt_inlines_blake2::digest_adapter::{Blake2b, U32};

pub(crate) type Blake2b256 = Blake2b<U32>;
