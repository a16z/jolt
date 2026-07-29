//! ZK verifier stage construction and committed-proof boundary helpers.

/// BlindFold applies only to the homomorphic build: no zk protocol exists
/// over the packed axis (`akita` and `zk` are mutually exclusive features).
#[cfg(not(feature = "akita"))]
#[doc(hidden)]
pub mod blindfold;
pub(crate) mod committed;
#[cfg(not(feature = "akita"))]
#[doc(hidden)]
pub mod inputs;
#[doc(hidden)]
pub mod outputs;
