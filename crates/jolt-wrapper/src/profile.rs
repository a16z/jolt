//! The wrapped verifier's shape: every quantity `relation::build_relation`
//! needs to lay out the R1CS without seeing a proof. Derived from the
//! verifier preprocessing plus the proof-shape fields the prover fixes
//! (`trace_length`, `ram_K`, the chunking configs, the trace order).

use common::jolt_device::MemoryLayout;
use jolt_claims::protocols::jolt::{
    geometry::dimensions::TracePolynomialOrder, JoltOneHotConfig, JoltReadWriteConfig,
};
use jolt_crypto::VectorCommitment;
use jolt_openings::CommitmentScheme;
use jolt_verifier::{JoltProof, JoltVerifierPreprocessing, ZkConfig};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProfileError {
    #[error("the relation supports full-program preprocessing only")]
    CommittedProgram,
    #[error("the relation supports advice-free proofs only")]
    Advice,
    #[error("the relation supports clear (non-ZK) proofs only")]
    Zk,
    #[error("trace length {0} is not a power of two")]
    TraceLength(usize),
    #[error("bytecode length {0} is not a power of two")]
    BytecodeLength(usize),
    #[error("profile serialization: {0}")]
    Encode(#[from] bincode::error::EncodeError),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WrapperProfile {
    /// `log2(trace_length)`.
    pub log_t: usize,
    /// `log2(ram_K)`.
    pub log_k_ram: usize,
    /// `log2(bytecode_len)` (the bytecode table is padded to a power of two).
    pub log_k_bytecode: usize,
    pub rw_config: JoltReadWriteConfig,
    pub one_hot_config: JoltOneHotConfig,
    pub trace_polynomial_order: TracePolynomialOrder,
    pub memory_layout: MemoryLayout,
    pub program_image_len_words: usize,
    pub entry_bytecode_index: usize,
}

impl WrapperProfile {
    pub fn new<PCS, VC>(
        preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
        proof: &JoltProof<PCS, VC>,
    ) -> Result<Self, ProfileError>
    where
        PCS: CommitmentScheme,
        VC: VectorCommitment<Field = PCS::Field>,
    {
        if preprocessing.program.as_full().is_none() {
            return Err(ProfileError::CommittedProgram);
        }
        // Trusted advice is a `verify` argument (the relation replays with
        // `None`); untrusted advice announces itself on the proof.
        if proof.untrusted_advice_commitment.is_some() {
            return Err(ProfileError::Advice);
        }
        if proof.protocol.zk != ZkConfig::Transparent {
            return Err(ProfileError::Zk);
        }
        let log_t =
            exact_log2(proof.trace_length).ok_or(ProfileError::TraceLength(proof.trace_length))?;
        let log_k_ram = exact_log2(proof.ram_K).ok_or(ProfileError::TraceLength(proof.ram_K))?;
        let bytecode_len = preprocessing.program.bytecode_len();
        let log_k_bytecode =
            exact_log2(bytecode_len).ok_or(ProfileError::BytecodeLength(bytecode_len))?;
        Ok(Self {
            log_t,
            log_k_ram,
            log_k_bytecode,
            rw_config: proof.rw_config,
            one_hot_config: proof.one_hot_config,
            trace_polynomial_order: proof.trace_polynomial_order,
            memory_layout: preprocessing.program.memory_layout().clone(),
            program_image_len_words: preprocessing.program.program_image_len_words(),
            entry_bytecode_index: preprocessing
                .program
                .entry_bytecode_index()
                .ok_or(ProfileError::CommittedProgram)?,
        })
    }

    /// The verifier-key digest: the relation is a deterministic function of
    /// the profile, so binding the profile binds the matrices.
    pub fn digest(&self) -> Result<[u8; 32], ProfileError> {
        let bytes = bincode::serde::encode_to_vec(self, bincode::config::standard())?;
        Ok(*blake3::hash(&bytes).as_bytes())
    }

    pub fn ram_k(&self) -> usize {
        1 << self.log_k_ram
    }

    pub fn bytecode_len(&self) -> usize {
        1 << self.log_k_bytecode
    }

    pub fn trace_length(&self) -> usize {
        1 << self.log_t
    }
}

fn exact_log2(value: usize) -> Option<usize> {
    (value.is_power_of_two()).then(|| value.trailing_zeros() as usize)
}
