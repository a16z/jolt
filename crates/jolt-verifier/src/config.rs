//! Prover configuration embedded in proofs.
//!
//! The verifier deserializes this from the proof to reconstruct the
//! exact parameters used during proving (memory layout, chunk sizes,
//! strategy choices).

use serde::{Deserialize, Serialize};

/// Configuration parameters that the prover embeds in the proof.
///
/// The verifier needs these to reconstruct sumcheck claims, R1CS keys,
/// and opening structure. Serialized as part of [`JoltProof`](crate::JoltProof).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProverConfig {
    /// Number of registers (typically 64 for RV64).
    pub register_count: usize,
    /// Log₂ of the RAM address space size.
    pub log_ram_size: usize,
    /// Chunk size for one-hot decomposition.
    pub one_hot_chunk_size: usize,
    /// Number of chunks per one-hot decomposition.
    pub one_hot_num_chunks: usize,
    /// Number of committed RA polynomials per virtual RA polynomial.
    pub n_committed_per_virtual_ra: usize,
}
