//! Witness oracle infrastructure for modular Jolt proving.
//!
//! This crate owns data-access abstractions for witness material, not protocol
//! order or backend allocation policy.

mod dimensions;
mod encoding;
mod error;
mod namespace;
mod opening;
mod polynomial;
mod provider;
mod public;
mod streaming;

pub mod protocols;

pub use dimensions::WitnessDimensions;
pub use encoding::PolynomialEncoding;
pub use error::WitnessError;
pub use namespace::{NamespaceId, OracleKind, OracleRef, WitnessNamespace};
pub use opening::OpeningWitness;
pub use polynomial::{
    MaterializationPolicy, OracleDescriptor, PolynomialView, RetentionHint, ViewRequirement,
};
pub use protocols::jolt_vm::{
    RaFamilyCycleIndexSource, RaFamilyCycleIndices, RA_FAMILY_MAX_BYTECODE_CHUNKS,
    RA_FAMILY_MAX_INSTRUCTION_CHUNKS, RA_FAMILY_MAX_RAM_CHUNKS,
};
pub use provider::{CommittedWitnessProvider, WitnessProvider};
pub use public::PublicValue;
pub use streaming::{
    PolynomialBatchChunk, PolynomialBatchStream, PolynomialChunk, PolynomialChunkKind,
    PolynomialStream,
};
