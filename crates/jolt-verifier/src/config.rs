//! Verifier-selected protocol configuration.

use common::{
    constants::{RAM_START_ADDRESS, REGISTER_COUNT, XLEN},
    jolt_device::MemoryLayout,
};
use jolt_claims::protocols::field_inline::FieldInlineConfig;
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig};
use jolt_program::preprocess::RAMPreprocessing;
use serde::{Deserialize, Serialize};

use crate::{preprocessing::JoltVerifierPreprocessing, proof::JoltProof, VerifierError};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZkConfig {
    Transparent,
    BlindFold,
}

impl ZkConfig {
    pub const fn from_bool(zk: bool) -> Self {
        if zk {
            Self::BlindFold
        } else {
            Self::Transparent
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltProtocolConfig {
    pub zk: ZkConfig,
    pub field_inline: FieldInlineConfig,
}

impl JoltProtocolConfig {
    pub const fn for_zk(zk: bool) -> Self {
        Self {
            zk: ZkConfig::from_bool(zk),
            field_inline: SELECTED_FIELD_INLINE_CONFIG,
        }
    }
}

#[cfg(feature = "field-inline")]
pub const SELECTED_FIELD_INLINE_CONFIG: FieldInlineConfig = FieldInlineConfig::native_v1();

#[cfg(not(feature = "field-inline"))]
pub const SELECTED_FIELD_INLINE_CONFIG: FieldInlineConfig = FieldInlineConfig::disabled();

#[cfg(feature = "zk")]
pub const SELECTED_ZK_CONFIG: ZkConfig = ZkConfig::BlindFold;

#[cfg(not(feature = "zk"))]
pub const SELECTED_ZK_CONFIG: ZkConfig = ZkConfig::Transparent;

pub const JOLT_VERIFIER_CONFIG: JoltProtocolConfig = JoltProtocolConfig {
    zk: SELECTED_ZK_CONFIG,
    field_inline: SELECTED_FIELD_INLINE_CONFIG,
};

pub fn validate_proof_config<PCS, VC, ZkProof>(
    config: &JoltProtocolConfig,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    zk: bool,
) -> Result<(), VerifierError>
where
    PCS: jolt_openings::CommitmentScheme,
    VC: jolt_crypto::VectorCommitment<Field = PCS::Field>,
{
    if !proof.trace_length.is_power_of_two()
        || proof.trace_length > preprocessing.program.max_padded_trace_length
    {
        return Err(VerifierError::InvalidTraceLength {
            got: proof.trace_length,
            max: preprocessing.program.max_padded_trace_length,
        });
    }

    if proof.protocol != *config {
        return Err(VerifierError::ProtocolConfigMismatch {
            expected: *config,
            got: proof.protocol,
        });
    }

    let runtime_config = JoltProtocolConfig::for_zk(zk);
    if proof.protocol != runtime_config {
        return Err(VerifierError::ProtocolConfigMismatch {
            expected: runtime_config,
            got: proof.protocol,
        });
    }

    validate_one_hot_config(proof.one_hot_config)?;

    let min_ram_k = compute_min_ram_K(
        &preprocessing.program.ram,
        &preprocessing.program.memory_layout,
    );
    let max_ram_k = compute_max_ram_K(&preprocessing.program.memory_layout);
    if !proof.ram_K.is_power_of_two() || proof.ram_K < min_ram_k || proof.ram_K > max_ram_k {
        return Err(VerifierError::InvalidRamK {
            got: proof.ram_K,
            min: min_ram_k,
            max: max_ram_k,
        });
    }

    validate_read_write_config(
        proof.rw_config,
        proof.trace_length.ilog2() as usize,
        proof.ram_K.ilog2() as usize,
    )?;

    Ok(())
}

pub fn validate_one_hot_config(config: JoltOneHotConfig) -> Result<(), VerifierError> {
    if config.log_k_chunk != 4 && config.log_k_chunk != 8 {
        return Err(VerifierError::InvalidOneHotConfig(format!(
            "log_k_chunk ({}) must be either 4 or 8",
            config.log_k_chunk
        )));
    }

    let log_k_chunk = config.log_k_chunk as usize;
    let lookups_chunk = config.lookups_ra_virtual_log_k_chunk as usize;
    let instruction_address_bits = 2 * XLEN;

    if lookups_chunk < log_k_chunk {
        return Err(VerifierError::InvalidOneHotConfig(format!(
            "lookups_ra_virtual_log_k_chunk ({lookups_chunk}) must be >= log_k_chunk ({log_k_chunk})"
        )));
    }

    if lookups_chunk > instruction_address_bits {
        return Err(VerifierError::InvalidOneHotConfig(format!(
            "lookups_ra_virtual_log_k_chunk ({lookups_chunk}) must be <= LOG_K ({instruction_address_bits})"
        )));
    }

    if !lookups_chunk.is_multiple_of(log_k_chunk) {
        return Err(VerifierError::InvalidOneHotConfig(format!(
            "lookups_ra_virtual_log_k_chunk ({lookups_chunk}) must be a multiple of log_k_chunk ({log_k_chunk})"
        )));
    }

    if !instruction_address_bits.is_multiple_of(lookups_chunk) {
        return Err(VerifierError::InvalidOneHotConfig(format!(
            "LOG_K ({instruction_address_bits}) must be divisible by lookups_ra_virtual_log_k_chunk ({lookups_chunk})"
        )));
    }

    Ok(())
}

pub fn validate_read_write_config(
    config: JoltReadWriteConfig,
    log_t: usize,
    ram_log_k: usize,
) -> Result<(), VerifierError> {
    let log_register_count = REGISTER_COUNT.ilog2() as usize;
    if (config.ram_rw_phase1_num_rounds as usize) > log_t {
        return Err(VerifierError::InvalidReadWriteConfig(format!(
            "ram_rw_phase1_num_rounds ({}) exceeds log_T ({log_t})",
            config.ram_rw_phase1_num_rounds
        )));
    }
    if (config.ram_rw_phase2_num_rounds as usize) > ram_log_k {
        return Err(VerifierError::InvalidReadWriteConfig(format!(
            "ram_rw_phase2_num_rounds ({}) exceeds log_ram_K ({ram_log_k})",
            config.ram_rw_phase2_num_rounds
        )));
    }
    if (config.registers_rw_phase1_num_rounds as usize) > log_t {
        return Err(VerifierError::InvalidReadWriteConfig(format!(
            "registers_rw_phase1_num_rounds ({}) exceeds log_T ({log_t})",
            config.registers_rw_phase1_num_rounds
        )));
    }
    if (config.registers_rw_phase2_num_rounds as usize) > log_register_count {
        return Err(VerifierError::InvalidReadWriteConfig(format!(
            "registers_rw_phase2_num_rounds ({}) exceeds log_register_count ({log_register_count})",
            config.registers_rw_phase2_num_rounds
        )));
    }
    Ok(())
}

#[expect(non_snake_case, reason = "Matches current jolt-core proof field name.")]
pub fn compute_min_ram_K(
    ram_preprocessing: &RAMPreprocessing,
    memory_layout: &MemoryLayout,
) -> usize {
    let bytecode_end = memory_layout
        .remap_word_address(ram_preprocessing.min_bytecode_address)
        .ok()
        .flatten()
        .unwrap_or(0) as usize
        + ram_preprocessing.bytecode_words.len();

    let io_end = memory_layout
        .remap_word_address(RAM_START_ADDRESS)
        .ok()
        .flatten()
        .unwrap_or(0) as usize;

    bytecode_end.max(io_end).next_power_of_two()
}

#[expect(non_snake_case, reason = "Matches current jolt-core proof field name.")]
pub fn compute_max_ram_K(memory_layout: &MemoryLayout) -> usize {
    let total_words = (memory_layout.heap_end - memory_layout.get_lowest_address()) / 8;
    (total_words as usize).next_power_of_two()
}
