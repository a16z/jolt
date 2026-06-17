//! Error types for HyperKZG operations.

use std::path::PathBuf;

use crate::types::{HyperKZGProofKind, HyperKZGSrsKind};

/// Errors produced by the HyperKZG commitment scheme.
#[derive(Debug, thiserror::Error)]
pub enum HyperKZGError {
    #[error("SRS too small: have {have} powers, need {need}")]
    SrsTooSmall { have: usize, need: usize },

    #[error("expected {expected} intermediate commitments, got {got}")]
    WrongCommitmentCount { expected: usize, got: usize },

    #[error("each evaluation row must have {expected} entries")]
    WrongEvaluationWidth { expected: usize },

    #[error("expected {expected:?} proof payload, got {got:?}")]
    WrongProofPayload {
        expected: HyperKZGProofKind,
        got: HyperKZGProofKind,
    },

    #[error("expected {expected:?} SRS setup, got {got:?}")]
    WrongSrsSetupKind {
        expected: HyperKZGSrsKind,
        got: HyperKZGSrsKind,
    },

    #[error("ZK HyperKZG requires an SRS with hiding G1 powers")]
    MissingZkSrs,

    #[error("polynomial must have at least 1 variable")]
    EmptyPoint,

    #[error("folding consistency check failed at level {level}")]
    FoldingConsistencyFailed { level: usize },

    #[error("batched folding consistency check failed")]
    BatchedFoldingConsistencyFailed,

    #[error("batch KZG pairing check failed")]
    PairingCheckFailed,

    #[error("degenerate Fiat-Shamir challenge")]
    DegenerateChallenge,

    #[error("SRS exponent too large: k = {k}")]
    SrsExponentTooLarge { k: usize },

    #[error("SRS capacity must be a power of two for file export, got {capacity}")]
    SrsFileCapacityNotPowerOfTwo { capacity: usize },

    #[error(
        "SRS too small for hyperkzg_{k}.srs: supports {supported} evaluations, needs {required}"
    )]
    SrsFileCapacityMismatch {
        k: usize,
        supported: usize,
        required: usize,
    },

    #[error("SRS file {path} has invalid name {name:?}")]
    SrsFileNameMismatch { path: PathBuf, name: String },

    #[error("SRS file {path} has unsupported version {version}")]
    SrsFileVersionUnsupported { path: PathBuf, version: u32 },

    #[error("SRS file {path} has kind {got:?}, expected {expected:?}")]
    SrsFileKindMismatch {
        path: PathBuf,
        expected: HyperKZGSrsKind,
        got: HyperKZGSrsKind,
    },

    #[error("failed to read SRS file {path}: {source}")]
    SrsFileRead {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("failed to write SRS file {path}: {source}")]
    SrsFileWrite {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("failed to encode SRS file {path}: {source}")]
    SrsFileEncode {
        path: PathBuf,
        source: bincode::error::EncodeError,
    },

    #[error("failed to decode SRS file {path}: {source}")]
    SrsFileDecode {
        path: PathBuf,
        source: bincode::error::DecodeError,
    },
}
