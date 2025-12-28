use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::field::JoltField;
use crate::utils::math::Math;
use crate::zkvm::instruction_lookups::LOG_K;
use common::constants::REGISTER_COUNT;

const LOG_REGISTER_COUNT: usize = REGISTER_COUNT.ilog2() as usize;

/// Returns the number of phases for instruction sumcheck based on trace length.
pub fn get_instruction_sumcheck_phases(log_t: usize) -> usize {
    if log_t < 23 {
        16
    } else {
        8
    }
}

#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct ProofConfig {
    /// Log of chunk size for one-hot encoding (e.g., 4 or 8).
    /// Affects RA polynomial chunking and Dory opening structure.
    pub log_k_chunk: usize,

    /// Log of chunk size for lookups RA virtual.
    pub lookups_ra_virtual_log_k_chunk: usize,

    /// RAM read-write checking: number of cycle variables to bind in phase 1.
    pub ram_rw_phase1_num_rounds: usize,

    /// RAM read-write checking: number of address variables to bind in phase 2.
    pub ram_rw_phase2_num_rounds: usize,

    /// Registers read-write checking: number of cycle variables to bind in phase 1.
    pub registers_rw_phase1_num_rounds: usize,

    /// Registers read-write checking: number of address variables to bind in phase 2.
    pub registers_rw_phase2_num_rounds: usize,
}

impl ProofConfig {
    /// Create a ProofConfig for the given trace parameters.
    pub fn new(log_T: usize, ram_log_K: usize) -> Self {
        let config = Self {
            log_k_chunk: if log_T < 25 { 4 } else { 8 },
            lookups_ra_virtual_log_k_chunk: if log_T < 25 { LOG_K / 8 } else { LOG_K / 4 },
            ram_rw_phase1_num_rounds: log_T,
            ram_rw_phase2_num_rounds: ram_log_K,
            registers_rw_phase1_num_rounds: log_T,
            registers_rw_phase2_num_rounds: LOG_REGISTER_COUNT,
        };

        // Validate the configuration
        config
            .validate(log_T, ram_log_K)
            .expect("invalid proof configuration");
        config
    }

    pub fn validate(&self, log_T: usize, ram_log_K: usize) -> Result<(), String> {
        if self.ram_rw_phase1_num_rounds > log_T {
            return Err(format!(
                "ram_rw_phase1_num_rounds ({}) exceeds log_T ({log_T})",
                self.ram_rw_phase1_num_rounds
            ));
        }
        if self.ram_rw_phase2_num_rounds > ram_log_K {
            return Err(format!(
                "ram_rw_phase2_num_rounds ({}) exceeds log_ram_K ({ram_log_K})",
                self.ram_rw_phase2_num_rounds
            ));
        }
        if self.registers_rw_phase1_num_rounds > log_T {
            return Err(format!(
                "registers_rw_phase1_num_rounds ({}) exceeds log_T ({log_T})",
                self.registers_rw_phase1_num_rounds
            ));
        }
        if self.registers_rw_phase2_num_rounds > LOG_REGISTER_COUNT {
            return Err(format!(
                "registers_rw_phase2_num_rounds ({}) exceeds LOG_REGISTER_COUNT ({LOG_REGISTER_COUNT})",
                self.registers_rw_phase2_num_rounds
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
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
    /// Create OneHotParams from a ProofConfig.
    pub fn new(config: &ProofConfig, bytecode_k: usize, ram_k: usize) -> Self {
        let log_k_chunk = config.log_k_chunk;
        let lookups_ra_virtual_log_k_chunk = config.lookups_ra_virtual_log_k_chunk;

        // log_k_chunk must be at most 8 so that chunk indices fit in u8
        assert!(
            log_k_chunk <= 8,
            "log_k_chunk must be <= 8 to fit in u8, got {log_k_chunk}",
        );

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
