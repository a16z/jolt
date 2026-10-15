/// BlindFold applies only to the homomorphic build: no zk protocol exists
/// over the Akita axis (`akita` and `zk` are mutually exclusive features).
#[cfg(not(feature = "akita"))]
#[doc(hidden)]
pub mod blindfold;
pub(crate) mod committed;
#[cfg(not(feature = "akita"))]
#[doc(hidden)]
pub mod inputs;
#[doc(hidden)]
pub mod outputs;
