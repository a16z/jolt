//! These minimal types are serialized inside [`JoltProof`](crate::proof::JoltProof)
//! so the verifier can reconstruct claim structure without external context.
//! The prover computes them from trace parameters; the verifier validates and
//! expands them during verification.

use serde::{Deserialize, Serialize};

/// One-hot decomposition configuration (serialized in proof).
///
/// Controls how large lookup indices are chunked into one-hot committed
/// polynomials. Two valid configurations:
/// - `log_k_chunk = 4` (chunk size 16) — for shorter traces
/// - `log_k_chunk = 8` (chunk size 256) — for longer traces
///
/// The threshold is `ONEHOT_CHUNK_THRESHOLD_LOG_T = 25`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OneHotConfig {
    /// Log₂ of one-hot chunk size for committed RA polynomials.
    pub log_k_chunk: u8,
    /// Log₂ of one-hot chunk size for RA virtual sumcheck.
    pub lookups_ra_virtual_log_k_chunk: u8,
}

impl OneHotConfig {
    /// Compute default config from trace length.
    #[allow(non_snake_case)]
    pub fn new(log_T: usize) -> Self {
        const THRESHOLD: usize = 25;
        if log_T < THRESHOLD {
            Self {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 16,
            }
        } else {
            Self {
                log_k_chunk: 8,
                lookups_ra_virtual_log_k_chunk: 32,
            }
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.log_k_chunk != 4 && self.log_k_chunk != 8 {
            return Err(format!(
                "log_k_chunk must be 4 or 8, got {}",
                self.log_k_chunk
            ));
        }
        let log_k = self.log_k_chunk as usize;
        let ra_log_k = self.lookups_ra_virtual_log_k_chunk as usize;
        if !ra_log_k.is_multiple_of(log_k) {
            return Err(format!(
                "lookups_ra_virtual_log_k_chunk ({ra_log_k}) must be divisible by log_k_chunk ({log_k})"
            ));
        }
        Ok(())
    }
}

/// Read-write checking phase configuration (serialized in proof).
///
/// Controls how many sumcheck rounds are allocated to each phase of
/// RAM and register read-write checking.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadWriteConfig {
    pub ram_rw_phase1_num_rounds: u8,
    pub ram_rw_phase2_num_rounds: u8,
    pub registers_rw_phase1_num_rounds: u8,
    pub registers_rw_phase2_num_rounds: u8,
}

impl ReadWriteConfig {
    /// Compute default config from trace dimensions.
    ///
    /// `log_T` is log₂ of the padded trace length. `ram_log_K` is
    /// log₂ of the RAM address space size.
    #[allow(non_snake_case)]
    pub fn new(log_T: usize, ram_log_K: usize) -> Self {
        const LOG_REGISTER_COUNT: u8 = 7; // log₂(128)
        Self {
            ram_rw_phase1_num_rounds: log_T as u8,
            ram_rw_phase2_num_rounds: ram_log_K as u8,
            registers_rw_phase1_num_rounds: log_T as u8,
            registers_rw_phase2_num_rounds: LOG_REGISTER_COUNT,
        }
    }

    #[allow(non_snake_case)]
    pub fn validate(&self, log_T: usize, ram_log_K: usize) -> Result<(), String> {
        if self.ram_rw_phase1_num_rounds as usize > log_T {
            return Err(format!(
                "ram_rw_phase1_num_rounds ({}) > log_T ({log_T})",
                self.ram_rw_phase1_num_rounds
            ));
        }
        if self.ram_rw_phase2_num_rounds as usize > ram_log_K {
            return Err(format!(
                "ram_rw_phase2_num_rounds ({}) > ram_log_K ({ram_log_K})",
                self.ram_rw_phase2_num_rounds
            ));
        }
        Ok(())
    }
}

/// Expanded one-hot parameters reconstructed from [`OneHotConfig`].
///
/// Not serialized — the verifier reconstructs this from the minimal
/// config carried in the proof plus preprocessing parameters (bytecode_K).
#[derive(Clone, Debug)]
pub struct OneHotParams {
    pub log_k_chunk: usize,
    pub lookups_ra_virtual_log_k_chunk: usize,
    /// Chunk size: `2^log_k_chunk`.
    pub k_chunk: usize,
    /// Bytecode address space size (from preprocessing, not proof).
    pub bytecode_k: usize,
    /// RAM address space size (from proof).
    pub ram_k: usize,
    /// Number of instruction RA chunks: `LOG_K / log_k_chunk`.
    pub instruction_d: usize,
    /// Number of bytecode RA chunks: `ceil(log₂(bytecode_k)) / log_k_chunk`.
    pub bytecode_d: usize,
    /// Number of RAM RA chunks: `ceil(log₂(ram_k)) / log_k_chunk`.
    pub ram_d: usize,
}

/// Total instruction lookup index bit width (RV64: XLEN × 2 = 128).
const LOG_K: usize = 128;

impl OneHotParams {
    /// Reconstruct full parameters from a minimal config.
    pub fn from_config(config: &OneHotConfig, bytecode_k: usize, ram_k: usize) -> Self {
        let log_k_chunk = config.log_k_chunk as usize;
        let lookups_ra_virtual_log_k_chunk = config.lookups_ra_virtual_log_k_chunk as usize;
        let k_chunk = 1usize << log_k_chunk;

        let instruction_d = LOG_K / log_k_chunk;
        let bytecode_d = bytecode_k
            .next_power_of_two()
            .trailing_zeros()
            .div_ceil(log_k_chunk as u32) as usize;
        let ram_d = ram_k
            .next_power_of_two()
            .trailing_zeros()
            .div_ceil(log_k_chunk as u32) as usize;

        Self {
            log_k_chunk,
            lookups_ra_virtual_log_k_chunk,
            k_chunk,
            bytecode_k,
            ram_k,
            instruction_d,
            bytecode_d,
            ram_d,
        }
    }

    /// Total number of committed polynomials: 2 (inc) + instruction_d + bytecode_d + ram_d.
    pub fn num_committed_polynomials(&self) -> usize {
        2 + self.instruction_d + self.bytecode_d + self.ram_d
    }

    /// Extract the `idx`-th chunk from a 128-bit instruction lookup index.
    #[inline]
    pub fn lookup_index_chunk(&self, index: u128, idx: usize) -> u8 {
        let shift = self.log_k_chunk * (self.instruction_d - 1 - idx);
        ((index >> shift) & (self.k_chunk as u128 - 1)) as u8
    }

    /// Extract the `idx`-th chunk from a bytecode PC.
    #[inline]
    pub fn bytecode_pc_chunk(&self, pc: usize, idx: usize) -> u8 {
        let shift = self.log_k_chunk * (self.bytecode_d - 1 - idx);
        ((pc >> shift) & (self.k_chunk - 1)) as u8
    }

    /// Extract the `idx`-th chunk from a RAM address.
    #[inline]
    pub fn ram_address_chunk(&self, address: u64, idx: usize) -> u8 {
        let shift = self.log_k_chunk * (self.ram_d - 1 - idx);
        ((address as usize >> shift) & (self.k_chunk - 1)) as u8
    }
}

/// Prover configuration bundled into the proof for self-contained verification.
///
/// All fields needed by the verifier to reconstruct claim structure and
/// validate the proof without external context (aside from the verifying key).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProverConfig {
    /// Padded trace length (power of 2).
    pub trace_length: usize,
    /// RAM address space size (power of 2).
    #[serde(rename = "ram_K")]
    pub ram_k: usize,
    /// Bytecode address space size (power of 2).
    pub bytecode_k: usize,
    /// One-hot decomposition configuration.
    pub one_hot_config: OneHotConfig,
    /// Read-write checking phase configuration.
    pub rw_config: ReadWriteConfig,
}

impl ProverConfig {
    pub fn log_trace_length(&self) -> usize {
        self.trace_length.trailing_zeros() as usize
    }

    pub fn ram_log_k(&self) -> usize {
        self.ram_k.trailing_zeros() as usize
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.trace_length.is_power_of_two() {
            return Err(format!(
                "trace_length {} is not a power of two",
                self.trace_length
            ));
        }
        if !self.ram_k.is_power_of_two() {
            return Err(format!("ram_k {} is not a power of two", self.ram_k));
        }
        self.one_hot_config.validate()?;
        self.rw_config
            .validate(self.log_trace_length(), self.ram_log_k())?;
        Ok(())
    }

    /// `bytecode_k` comes from the verifying key / preprocessing, not the proof.
    pub fn one_hot_params(&self, bytecode_k: usize) -> OneHotParams {
        OneHotParams::from_config(&self.one_hot_config, bytecode_k, self.ram_k)
    }

    /// Compute [`OneHotParams`] using the `bytecode_k` stored in the config.
    pub fn one_hot_params_from_config(&self) -> OneHotParams {
        OneHotParams::from_config(&self.one_hot_config, self.bytecode_k, self.ram_k)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_hot_config_default_small_trace() {
        let config = OneHotConfig::new(20);
        assert_eq!(config.log_k_chunk, 4);
        assert_eq!(config.lookups_ra_virtual_log_k_chunk, 16);
        config.validate().unwrap();
    }

    #[test]
    fn one_hot_config_default_large_trace() {
        let config = OneHotConfig::new(26);
        assert_eq!(config.log_k_chunk, 8);
        assert_eq!(config.lookups_ra_virtual_log_k_chunk, 32);
        config.validate().unwrap();
    }

    #[test]
    fn one_hot_params_instruction_d() {
        let config = OneHotConfig::new(20);
        let params = OneHotParams::from_config(&config, 1024, 65536);
        assert_eq!(params.instruction_d, 32); // 128 / 4
        assert_eq!(params.bytecode_d, 3); // ceil(10/4)
        assert_eq!(params.ram_d, 4); // ceil(16/4)
    }

    #[test]
    fn rw_config_default() {
        let config = ReadWriteConfig::new(20, 16);
        assert_eq!(config.ram_rw_phase1_num_rounds, 20);
        assert_eq!(config.ram_rw_phase2_num_rounds, 16);
        assert_eq!(config.registers_rw_phase1_num_rounds, 20);
        assert_eq!(config.registers_rw_phase2_num_rounds, 7);
        config.validate(20, 16).unwrap();
    }

    #[test]
    fn prover_config_validate() {
        let config = ProverConfig {
            trace_length: 1 << 20,
            ram_k: 1 << 16,
            bytecode_k: 1 << 10,
            one_hot_config: OneHotConfig::new(20),
            rw_config: ReadWriteConfig::new(20, 16),
        };
        config.validate().unwrap();
    }

    #[test]
    fn prover_config_rejects_non_power_of_two() {
        let config = ProverConfig {
            trace_length: 100,
            ram_k: 1 << 16,
            bytecode_k: 1 << 10,
            one_hot_config: OneHotConfig::new(20),
            rw_config: ReadWriteConfig::new(20, 16),
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn lookup_index_chunk_extraction() {
        let config = OneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
        };
        let params = OneHotParams::from_config(&config, 256, 256);

        // 128-bit index with known pattern
        let index: u128 = 0xAB;
        // With d=32 chunks of 4 bits each, chunk 31 (last) holds lowest nibble
        assert_eq!(params.lookup_index_chunk(index, 31), 0xB);
        assert_eq!(params.lookup_index_chunk(index, 30), 0xA);
    }
}
