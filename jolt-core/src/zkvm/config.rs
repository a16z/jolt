use crate::field::JoltField;
use crate::utils::math::Math;
use crate::zkvm::instruction_lookups::LOG_K;

// =============================================================================
// ProofConfig - Configuration that affects proof structure
// =============================================================================

/// Configuration that affects proof structure and MUST match between prover and verifier.
///
/// These parameters determine polynomial layouts, binding orders, and opening structures.
/// If prover and verifier use different ProofConfig values, verification will fail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofConfig {
    /// Log of chunk size for one-hot encoding (e.g., 4 or 8).
    /// Affects RA polynomial chunking and Dory opening structure.
    pub log_k_chunk: usize,

    /// Log of chunk size for lookups RA virtual.
    pub lookups_ra_virtual_log_k_chunk: usize,

    /// RAM read-write checking: number of cycle variables to bind in phase 1.
    /// If None, defaults to T.log_2() (bind all cycle vars in phase 1).
    pub ram_rw_phase1_num_rounds: Option<usize>,

    /// RAM read-write checking: number of address variables to bind in phase 2.
    /// If None, defaults to K.log_2() (bind all address vars in phase 2).
    pub ram_rw_phase2_num_rounds: Option<usize>,

    /// Registers read-write checking: number of cycle variables to bind in phase 1.
    /// If None, defaults to T.log_2() (bind all cycle vars in phase 1).
    pub registers_rw_phase1_num_rounds: Option<usize>,

    /// Registers read-write checking: number of address variables to bind in phase 2.
    /// If None, defaults to LOG_K (7 for 128 registers).
    pub registers_rw_phase2_num_rounds: Option<usize>,
}

impl Default for ProofConfig {
    fn default() -> Self {
        Self {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: LOG_K / 8,
            ram_rw_phase1_num_rounds: None,
            ram_rw_phase2_num_rounds: None,
            registers_rw_phase1_num_rounds: None,
            registers_rw_phase2_num_rounds: None,
        }
    }
}

impl ProofConfig {
    /// Default configuration based on trace length (current behavior).
    pub fn default_for_trace(log_T: usize) -> Self {
        Self {
            log_k_chunk: if log_T < 25 { 4 } else { 8 },
            lookups_ra_virtual_log_k_chunk: if log_T < 25 { LOG_K / 8 } else { LOG_K / 4 },
            ram_rw_phase1_num_rounds: None,
            ram_rw_phase2_num_rounds: None,
            registers_rw_phase1_num_rounds: None,
            registers_rw_phase2_num_rounds: None,
        }
    }

    // -------------------------------------------------------------------------
    // RAM read-write checking phase helpers
    // -------------------------------------------------------------------------

    /// Number of cycle variables to bind in RAM RW phase 1.
    #[inline]
    pub fn ram_rw_phase1_num_rounds(&self, _ram_K: usize, T: usize) -> usize {
        self.ram_rw_phase1_num_rounds.unwrap_or_else(|| T.log_2())
    }

    /// Number of address variables to bind in RAM RW phase 2.
    #[inline]
    pub fn ram_rw_phase2_num_rounds(&self, ram_K: usize, _T: usize) -> usize {
        self.ram_rw_phase2_num_rounds
            .unwrap_or_else(|| ram_K.log_2())
    }

    /// Returns true if all cycle variables are bound in RAM RW phase 1.
    #[inline]
    pub fn ram_rw_all_cycle_in_phase1(&self, ram_K: usize, T: usize) -> bool {
        self.ram_rw_phase1_num_rounds(ram_K, T) == T.log_2()
    }

    /// Returns true if all address variables are bound in RAM RW phase 2.
    #[inline]
    pub fn ram_rw_all_address_in_phase2(&self, ram_K: usize, T: usize) -> bool {
        self.ram_rw_phase2_num_rounds(ram_K, T) == ram_K.log_2()
    }

    // -------------------------------------------------------------------------
    // Registers read-write checking phase helpers
    // -------------------------------------------------------------------------

    /// Number of cycle variables to bind in Registers RW phase 1.
    #[inline]
    pub fn registers_rw_phase1_num_rounds(&self, T: usize) -> usize {
        self.registers_rw_phase1_num_rounds
            .unwrap_or_else(|| T.log_2())
    }

    /// Number of address variables to bind in Registers RW phase 2.
    #[inline]
    pub fn registers_rw_phase2_num_rounds(&self, _T: usize) -> usize {
        // Default to 7 for 128 virtual registers (including temporaries)
        const LOG_VIRTUAL_REGISTERS: usize = 7;
        self.registers_rw_phase2_num_rounds
            .unwrap_or(LOG_VIRTUAL_REGISTERS)
    }

    /// Returns true if all cycle variables are bound in Registers RW phase 1.
    #[inline]
    pub fn registers_rw_all_cycle_in_phase1(&self, T: usize) -> bool {
        self.registers_rw_phase1_num_rounds(T) == T.log_2()
    }

    /// Returns true if all address variables are bound in Registers RW phase 2.
    #[inline]
    pub fn registers_rw_all_address_in_phase2(&self, T: usize) -> bool {
        const LOG_VIRTUAL_REGISTERS: usize = 7;
        self.registers_rw_phase2_num_rounds(T) == LOG_VIRTUAL_REGISTERS
    }
}

// =============================================================================
// ProverOnlyConfig - Configuration that only affects prover performance
// =============================================================================

/// Configuration that only affects prover performance, NOT proof structure.
///
/// The verifier does not need these values. Changing them will not break verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProverOnlyConfig {
    /// Number of phases for instruction lookups batching.
    /// Must be a divisor of 128. Common values: 8 or 16.
    pub instruction_sumcheck_phases: usize,
}

impl Default for ProverOnlyConfig {
    fn default() -> Self {
        Self {
            instruction_sumcheck_phases: 8,
        }
    }
}

impl ProverOnlyConfig {
    /// Default configuration based on trace length.
    pub fn default_for_trace(log_T: usize) -> Self {
        Self {
            instruction_sumcheck_phases: if log_T < 23 { 16 } else { 8 },
        }
    }
}

// =============================================================================
// Legacy helper functions (for backward compatibility)
// =============================================================================

/// Helper to get log_k_chunk based on log_T.
#[inline]
pub const fn get_log_k_chunk(log_T: usize) -> usize {
    if log_T < 25 {
        4
    } else {
        8
    }
}

/// Helper to get lookups_ra_virtual_log_k_chunk based on log_T.
pub const fn get_lookups_ra_virtual_log_k_chunk(log_T: usize) -> usize {
    if log_T < 25 {
        LOG_K / 8
    } else {
        LOG_K / 4
    }
}

/// Compute the number of phases for instruction lookups based on trace length.
/// For traces below 2^23 cycles we want to use 16 phases, otherwise 8.
/// NOTE: currently only divisors of 128 are supported
#[inline]
pub const fn instruction_sumcheck_phases(log_T: usize) -> usize {
    if log_T < 23 {
        16
    } else {
        8
    }
}

// =============================================================================
// OneHotParams
// =============================================================================

/// Helper to compute d (number of chunks) from log_k and log_k_chunk.
#[inline]
fn compute_d(log_k: usize, log_chunk: usize) -> usize {
    log_k.div_ceil(log_chunk)
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
    /// Create OneHotParams using default configuration based on trace length.
    pub fn new(log_T: usize, bytecode_k: usize, ram_k: usize) -> Self {
        let config = ProofConfig::default_for_trace(log_T);
        Self::new_with_config(&config, bytecode_k, ram_k)
    }

    /// Create OneHotParams using the provided ProofConfig.
    pub fn new_with_config(config: &ProofConfig, bytecode_k: usize, ram_k: usize) -> Self {
        Self::new_with_log_k_chunk(
            config.log_k_chunk,
            config.lookups_ra_virtual_log_k_chunk,
            bytecode_k,
            ram_k,
        )
    }

    /// Create OneHotParams with explicit log_k_chunk values.
    pub fn new_with_log_k_chunk(
        log_k_chunk: usize,
        lookups_ra_virtual_log_k_chunk: usize,
        bytecode_k: usize,
        ram_k: usize,
    ) -> Self {
        // log_k_chunk must be at most 8 so that chunk indices fit in u8
        assert!(
            log_k_chunk <= 8,
            "log_k_chunk must be <= 8 to fit in u8, got {log_k_chunk}",
        );
        let instruction_d = compute_d(LOG_K, log_k_chunk);
        let bytecode_d = compute_d(bytecode_k.log_2(), log_k_chunk);
        let ram_d = compute_d(ram_k.log_2(), log_k_chunk);

        let instruction_shifts = (0..instruction_d)
            .map(|i| log_k_chunk * (instruction_d - 1 - i))
            .collect();
        let ram_shifts = (0..ram_d).map(|i| log_k_chunk * (ram_d - 1 - i)).collect();
        let bytecode_shifts = (0..bytecode_d)
            .map(|i| log_k_chunk * (bytecode_d - 1 - i))
            .collect();

        let k_chunk = 1 << log_k_chunk;
        Self {
            log_k_chunk,
            lookups_ra_virtual_log_k_chunk,
            k_chunk,
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

    #[inline(always)]
    pub fn ram_address_chunk(&self, address: u64, idx: usize) -> u8 {
        ((address >> self.ram_shifts[idx]) & (self.k_chunk - 1) as u64) as u8
    }

    #[inline(always)]
    pub fn bytecode_pc_chunk(&self, pc: usize, idx: usize) -> u8 {
        ((pc >> self.bytecode_shifts[idx]) & (self.k_chunk - 1)) as u8
    }

    #[inline(always)]
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
