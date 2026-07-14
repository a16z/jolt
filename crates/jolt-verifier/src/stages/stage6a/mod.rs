//! Stage 6a (address-phase) verifier entry point.

pub mod booleanity;
pub mod bytecode_read_raf;
/// The inc-virtualization phase that opens the stage-6 region on the packed
/// path: a single-instance sumcheck virtualizing the four reduced `Inc`
/// claims into the committed `FusedInc` stream and its `OpFlags(Store)`
/// selector, before the address batch. Its store claim feeds this stage's
/// read-raf fold, so it is co-located here and runs immediately before
/// [`verify`].
#[cfg(feature = "akita")]
pub mod inc_virtualization;
pub mod outputs;
pub mod verify;

pub use outputs::{Stage6aClearOutput, Stage6aOutput, Stage6aZkOutput};
pub use verify::verify;
