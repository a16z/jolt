//! The Blake2b-256 hasher behind the protocol's layout and preprocessing
//! digests: the `blake2` crate, or the jolt-inlines hasher on a RISC-V guest
//! built with `blake2-inline`. Same bytes either way; host builds keep the
//! crate, since the inline has no host compression without its `host` feature.

pub use blake2::Digest;
#[cfg(not(all(feature = "blake2-inline", target_arch = "riscv64")))]
use blake2::{digest::consts::U32, Blake2b};
#[cfg(all(feature = "blake2-inline", target_arch = "riscv64"))]
use jolt_inlines_blake2::digest_adapter::{Blake2b, U32};

pub type Blake2b256 = Blake2b<U32>;
