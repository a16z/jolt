//! Pseudo-Mersenne prime fields (`p = 2^k − c`) and their named instances.
//!
//! The leaf layer of the field DAG: the `u32`/`u64`/`u128`-backed
//! `Fp{32,64,128}` field types, the `2^k − offset` registry
//! (`pseudo_mersenne`), and the shared low-level arithmetic helpers (`util`).
//! The extension towers (`ext`), packing (`packed`), and wide accumulators
//! (`unreduced`) all build on top of this module.

#![expect(
    clippy::unreadable_literal,
    reason = "ported modulus and regression constants retain their audited spelling"
)]

pub(crate) mod fp128;
pub(crate) mod fp32;
pub(crate) mod fp64;
pub(crate) mod native_capability;
pub(crate) mod pseudo_mersenne;
pub(crate) mod util;

pub use fp128::{
    Fp128, Prime128Offset159, Prime128Offset2355, Prime128Offset275, Prime128OffsetA7F7,
};
pub use fp32::Fp32;
pub use fp64::Fp64;
pub use pseudo_mersenne::{
    is_registered_prime_offset, pseudo_mersenne_modulus, registered_prime_offset_spec,
    Prime24Offset3, Prime30Offset35, Prime31Offset19, Prime32Offset99, Prime40Offset195,
    Prime48Offset59, Prime56Offset27, Prime64Offset59, PrimeOffsetSpec,
    PRIME_OFFSET_IMPLEMENTED_MAX_BITS, PRIME_OFFSET_MAX, PRIME_OFFSET_SPECS,
};
