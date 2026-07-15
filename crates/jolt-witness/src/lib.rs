//! Witness oracle infrastructure for modular Jolt proving.
//!
//! This crate owns data-access abstractions for witness material, not protocol
//! order or backend allocation policy.

mod descriptor;
mod dimensions;
mod encoding;
mod error;
mod namespace;
mod provider;
mod streaming;

pub mod protocols;
pub mod witnesses;

pub use descriptor::OracleDescriptor;
pub use dimensions::WitnessDimensions;
pub use encoding::PolynomialEncoding;
pub use error::WitnessError;
pub use namespace::{NamespaceId, OracleRef, WitnessNamespace};
pub use provider::{CommittedWitnessProvider, WitnessProvider};
pub use streaming::{
    PolynomialBatchChunk, PolynomialBatchStream, PolynomialChunk, PolynomialChunkKind,
    PolynomialStream,
};
