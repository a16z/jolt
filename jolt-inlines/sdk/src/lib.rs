#![cfg_attr(not(feature = "host"), no_std)]

mod hcf;
pub use hcf::hcf;

mod spoil;
pub use spoil::UnwrapOrSpoilProof;

pub mod ec;

#[cfg(feature = "host")]
pub mod host;
