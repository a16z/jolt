use allocative::Allocative;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::field::JoltField;
use crate::utils::math::Math;
use crate::zkvm::instruction_lookups::LOG_K;
use common::constants::{
    INSTRUCTION_PHASES_THRESHOLD_LOG_T, ONEHOT_CHUNK_THRESHOLD_LOG_T, REGISTER_COUNT,
};

/// Returns the number of phases for instruction sumcheck based on trace length.
///
/// For shorter traces (log_T < threshold), uses 16 phases for better parallelism.
/// For longer traces, uses 8 phases to reduce overhead.
pub fn get_instruction_sumcheck_phases(log_t: usize) -> usize {
    if log_t < INSTRUCTION_PHASES_THRESHOLD_LOG_T {
        16
    } else {
        8
    }
}

/// Configuration for read-write checking sumchecks.
///
/// Contains parameters that control phase structure for RAM and register
/// read-write checking sumchecks. All fields are `u8` to minimize proof size.
#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct ReadWriteConfig {
    /// RAM read-write checking: number of cycle variables to bind in phase 1.
    pub ram_rw_phase1_num_rounds: u8,

    /// RAM read-write checking: number of address variables to bind in phase 2.
    pub ram_rw_phase2_num_rounds: u8,

    /// Registers read-write checking: number of cycle variables to bind in phase 1.
    pub registers_rw_phase1_num_rounds: u8,

    /// Registers read-write checking: number of address variables to bind in phase 2.
    pub registers_rw_phase2_num_rounds: u8,
}

impl ReadWriteConfig {
    /// Create a ReadWriteConfig for the given trace parameters.
    /// This is the default configuration for the prover; other configurations are possible
    /// (as long as they satisfy the validation constraints).
    pub fn new(log_T: usize, ram_log_K: usize) -> Self {
        let log_register_count = REGISTER_COUNT.ilog2() as usize;
        let config = Self {
            ram_rw_phase1_num_rounds: log_T as u8,
            ram_rw_phase2_num_rounds: ram_log_K as u8,
            registers_rw_phase1_num_rounds: log_T as u8,
            registers_rw_phase2_num_rounds: log_register_count as u8,
        };

        // Validate the configuration
        config
            .validate(log_T, ram_log_K)
            .expect("invalid proof configuration");
        config
    }

    /// Validates that the read-write checking configuration is consistent with
    /// the given trace parameters.
    ///
    /// This is called by the verifier to ensure the prover hasn't provided
    /// an invalid configuration that would break soundness of the sumchecks.
    pub fn validate(&self, log_T: usize, ram_log_K: usize) -> Result<(), String> {
        let log_register_count = REGISTER_COUNT.ilog2() as usize;
        if (self.ram_rw_phase1_num_rounds as usize) > log_T {
            return Err(format!(
                "ram_rw_phase1_num_rounds ({}) exceeds log_T ({log_T})",
                self.ram_rw_phase1_num_rounds
            ));
        }
        if (self.ram_rw_phase2_num_rounds as usize) > ram_log_K {
            return Err(format!(
                "ram_rw_phase2_num_rounds ({}) exceeds log_ram_K ({ram_log_K})",
                self.ram_rw_phase2_num_rounds
            ));
        }
        if (self.registers_rw_phase1_num_rounds as usize) > log_T {
            return Err(format!(
                "registers_rw_phase1_num_rounds ({}) exceeds log_T ({log_T})",
                self.registers_rw_phase1_num_rounds
            ));
        }
        if (self.registers_rw_phase2_num_rounds as usize) > log_register_count {
            return Err(format!(
                "registers_rw_phase2_num_rounds ({}) exceeds log_register_count ({log_register_count})",
                self.registers_rw_phase2_num_rounds
            ));
        }
        Ok(())
    }

    /// Returns true if all cycle variables are bound in phase 1.
    ///
    /// When this returns true, the advice opening points for `RamValEvaluation` and
    /// `RamValFinalEvaluation` are identical, so we only need one advice opening.
    #[inline]
    pub fn needs_single_advice_opening(&self, log_T: usize) -> bool {
        self.ram_rw_phase1_num_rounds as usize == log_T
    }
}

/// Minimal configuration for one-hot encoding that gets serialized in the proof.
///
/// Contains only the prover's choices. All fields are `u8` to minimize proof size.
/// The verifier validates these choices and reconstructs the full `OneHotParams`.
#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct OneHotConfig {
    /// Log₂ of chunk size for one-hot encoding of address variables.
    ///
    /// This determines how the address space is decomposed into committed RA polynomials.
    /// Each committed RA polynomial handles 2^log_k_chunk addresses. The total number
    /// of committed RA polynomials is LOG_K / log_k_chunk (e.g., 128/8 = 16 for RV64).
    ///
    /// Must be either 4 or 8 currently
    pub log_k_chunk: u8,

    /// Log₂ of chunk size for virtual RA polynomials in instruction lookups.
    ///
    /// In the instruction lookups Read+RAF sumcheck, the RA polynomial over LOG_K address
    /// bits is decomposed into virtual RA polynomials, each covering `lookups_ra_virtual_log_k_chunk`
    /// address bits. Each virtual RA poly is the product of `lookups_ra_virtual_log_k_chunk / log_k_chunk`
    /// committed RA polynomials.
    ///
    /// Must be a multiple of `log_k_chunk` and divide LOG_K evenly.
    /// Valid range: [log_k_chunk, LOG_K] (e.g., 4-128 or 8-128 depending on log_k_chunk).
    pub lookups_ra_virtual_log_k_chunk: u8,
}

impl OneHotConfig {
    /// Create a OneHotConfig with default values based on trace length.
    pub fn new(log_T: usize) -> Self {
        let log_k_chunk = if log_T < ONEHOT_CHUNK_THRESHOLD_LOG_T {
            4
        } else {
            8
        };
        let lookups_ra_virtual_log_k_chunk = if log_T < ONEHOT_CHUNK_THRESHOLD_LOG_T {
            LOG_K / 8
        } else {
            LOG_K / 4
        };

        Self {
            log_k_chunk: log_k_chunk as u8,
            lookups_ra_virtual_log_k_chunk: lookups_ra_virtual_log_k_chunk as u8,
        }
    }

    /// Validates that the one-hot configuration is valid.
    ///
    /// This is called by the verifier to ensure the prover hasn't provided
    /// an invalid configuration that would break soundness.
    pub fn validate(&self) -> Result<(), String> {
        // log_k_chunk must be either 4 or 8
        if self.log_k_chunk != 4 && self.log_k_chunk != 8 {
            return Err(format!(
                "log_k_chunk ({}) must be either 4 or 8",
                self.log_k_chunk
            ));
        }

        let log_k_chunk = self.log_k_chunk as usize;
        let lookups_chunk = self.lookups_ra_virtual_log_k_chunk as usize;

        // lookups_ra_virtual_log_k_chunk must be at least log_k_chunk
        if lookups_chunk < log_k_chunk {
            return Err(format!(
                "lookups_ra_virtual_log_k_chunk ({lookups_chunk}) must be >= log_k_chunk ({log_k_chunk})"
            ));
        }

        // lookups_ra_virtual_log_k_chunk must be at most LOG_K (128 for RV64)
        if lookups_chunk > LOG_K {
            return Err(format!(
                "lookups_ra_virtual_log_k_chunk ({lookups_chunk}) must be <= LOG_K ({LOG_K})"
            ));
        }

        // lookups_ra_virtual_log_k_chunk must be a multiple of log_k_chunk
        if lookups_chunk % log_k_chunk != 0 {
            return Err(format!(
                "lookups_ra_virtual_log_k_chunk ({lookups_chunk}) must be a multiple of log_k_chunk ({log_k_chunk})"
            ));
        }

        // LOG_K must be divisible by lookups_ra_virtual_log_k_chunk
        if LOG_K % lookups_chunk != 0 {
            return Err(format!(
                "LOG_K ({LOG_K}) must be divisible by lookups_ra_virtual_log_k_chunk ({lookups_chunk})"
            ));
        }

        Ok(())
    }
}

/// Full one-hot parameters with cached derived values.
///
/// This struct is NOT serialized in the proof. It is constructed by the prover
/// and verifier from `OneHotConfig` plus the proof parameters (bytecode_K, ram_K).
#[derive(Allocative, Clone, Debug, Default)]
pub struct OneHotParams {
    pub log_k_chunk: usize,
    pub lookups_ra_virtual_log_k_chunk: usize,
    pub k_chunk: usize,

    pub bytecode_k: usize,
    pub ram_k: usize,

    pub instruction_d: usize,
    pub bytecode_d: usize,
    pub ram_d: usize,

    instruction_shifts: Vec<usize>,
    ram_shifts: Vec<usize>,
    bytecode_shifts: Vec<usize>,
}

impl OneHotParams {
    /// Construct full OneHotParams from a config and proof parameters.
    ///
    /// This is used by the verifier to reconstruct the full params from
    /// the minimal config stored in the proof.
    pub fn from_config(config: &OneHotConfig, bytecode_k: usize, ram_k: usize) -> Self {
        let log_k_chunk = config.log_k_chunk as usize;
        let lookups_ra_virtual_log_k_chunk = config.lookups_ra_virtual_log_k_chunk as usize;

        let instruction_d = LOG_K.div_ceil(log_k_chunk);
        let bytecode_d = bytecode_k.log_2().div_ceil(log_k_chunk);
        let ram_d = ram_k.log_2().div_ceil(log_k_chunk);

        let instruction_shifts = (0..instruction_d)
            .map(|i| log_k_chunk * (instruction_d - 1 - i))
            .collect();
        let ram_shifts = (0..ram_d).map(|i| log_k_chunk * (ram_d - 1 - i)).collect();
        let bytecode_shifts = (0..bytecode_d)
            .map(|i| log_k_chunk * (bytecode_d - 1 - i))
            .collect();

        Self {
            log_k_chunk,
            lookups_ra_virtual_log_k_chunk,
            k_chunk: 1 << log_k_chunk,
            bytecode_k,
            ram_k,
            instruction_d,
            bytecode_d,
            ram_d,
            instruction_shifts,
            ram_shifts,
            bytecode_shifts,
        }
    }

    /// Create OneHotParams for the given trace parameters using default config.
    ///
    /// This is a convenience constructor for the prover.
    pub fn new(log_T: usize, bytecode_k: usize, ram_k: usize) -> Self {
        let config = OneHotConfig::new(log_T);
        Self::from_config(&config, bytecode_k, ram_k)
    }

    /// Extract the minimal config for serialization in the proof.
    pub fn to_config(&self) -> OneHotConfig {
        OneHotConfig {
            log_k_chunk: self.log_k_chunk as u8,
            lookups_ra_virtual_log_k_chunk: self.lookups_ra_virtual_log_k_chunk as u8,
        }
    }

    pub fn ram_address_chunk(&self, address: u64, idx: usize) -> u8 {
        ((address >> self.ram_shifts[idx]) & (self.k_chunk - 1) as u64) as u8
    }

    pub fn bytecode_pc_chunk(&self, pc: usize, idx: usize) -> u8 {
        ((pc >> self.bytecode_shifts[idx]) & (self.k_chunk - 1)) as u8
    }

    pub fn lookup_index_chunk(&self, index: u128, idx: usize) -> u8 {
        ((index >> self.instruction_shifts[idx]) & (self.k_chunk - 1) as u128) as u8
    }

    pub fn compute_r_address_chunks<F: JoltField>(
        &self,
        r_address: &[F::Challenge],
    ) -> Vec<Vec<F::Challenge>> {
        let r_address = if r_address.len().is_multiple_of(self.log_k_chunk) {
            r_address.to_vec()
        } else {
            [
                &vec![
                    F::Challenge::from(0_u128);
                    self.log_k_chunk - (r_address.len() & (self.log_k_chunk - 1))
                ],
                r_address,
            ]
            .concat()
        };

        let r_address_chunks: Vec<Vec<F::Challenge>> = r_address
            .chunks(self.log_k_chunk)
            .map(|chunk| chunk.to_vec())
            .collect();

        r_address_chunks
    }
}
