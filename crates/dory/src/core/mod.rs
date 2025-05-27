//! # Core
//! This is the beef of the Dory inner-product arguments.
//! That is, it collectively implements the `Extended Dory Reduce` algorithm prover and verifier side
//! Which can be used as a generic interactive argument for inner pairing products of vectors.
//!
//! This by itself does not constitute a PCS, but using machinery under `VMV` we can turn it into one.

pub mod builder;
pub mod inner_product;
pub mod interactive_protocol;
pub mod messages;
pub mod setup;
pub mod state;

pub use builder::*;
pub use inner_product::*;
pub use messages::*;
pub use setup::*;
pub use state::*;
