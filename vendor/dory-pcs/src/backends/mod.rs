//! Backend implementations for Dory primitives
//!
//! This module provides concrete implementations of the abstract traits
//! defined in the primitives module. Currently supports:
//! - arkworks: BN254 pairing curve implementation using Arkworks

#[cfg(feature = "arkworks")]
pub mod arkworks;

#[cfg(feature = "arkworks")]
pub use arkworks::*;
