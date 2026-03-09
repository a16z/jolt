//! Witness generation configuration.
//!
//! [`WitnessConfig`] holds the one-hot decomposition parameters that
//! determine how address spaces are chunked into committed polynomials.
//! It is constructed from the proof's `OneHotConfig` plus runtime parameters
//! (`bytecode_K`, `ram_K`).

/// Polynomial decomposition parameters for witness generation.
///
/// Determines how 128-bit instruction indices, bytecode PCs, and RAM addresses
/// are decomposed into `d` one-hot committed polynomials, each covering a
/// `log_k_chunk`-bit window.
///
/// # Construction
///
/// Use [`WitnessConfig::new`] with the log₂ chunk size and the total address
/// space sizes. The shift tables are pre-computed for fast chunk extraction.
#[derive(Clone, Debug)]
pub struct WitnessConfig {
    /// Log₂ of one-hot chunk size (4 or 8).
    pub log_k_chunk: usize,
    /// One-hot chunk size: `2^log_k_chunk` (16 or 256).
    pub k_chunk: usize,

    /// Number of committed RA polynomials for instruction lookups.
    ///
    /// `instruction_d = ceil(log_k / log_k_chunk)` where `log_k` is the
    /// instruction lookup index bit width.
    pub instruction_d: usize,
    /// Number of committed RA polynomials for bytecode PC.
    pub bytecode_d: usize,
    /// Number of committed RA/WA polynomials for RAM.
    pub ram_d: usize,

    instruction_shifts: Vec<usize>,
    bytecode_shifts: Vec<usize>,
    ram_shifts: Vec<usize>,
}

impl WitnessConfig {
    /// Construct a witness config from decomposition parameters.
    ///
    /// # Arguments
    ///
    /// * `log_k_chunk` — Log₂ of one-hot chunk size (must be 4 or 8).
    /// * `log_k_instruction` — Bit width of instruction lookup indices (e.g. 128 for RV64).
    /// * `log_k_bytecode` — Log₂ of bytecode address space size.
    /// * `log_k_ram` — Log₂ of RAM address space size.
    pub fn new(
        log_k_chunk: usize,
        log_k_instruction: usize,
        log_k_bytecode: usize,
        log_k_ram: usize,
    ) -> Self {
        let k_chunk = 1 << log_k_chunk;
        let instruction_d = log_k_instruction.div_ceil(log_k_chunk);
        let bytecode_d = log_k_bytecode.div_ceil(log_k_chunk);
        let ram_d = log_k_ram.div_ceil(log_k_chunk);

        let instruction_shifts = compute_shifts(log_k_chunk, instruction_d);
        let bytecode_shifts = compute_shifts(log_k_chunk, bytecode_d);
        let ram_shifts = compute_shifts(log_k_chunk, ram_d);

        Self {
            log_k_chunk,
            k_chunk,
            instruction_d,
            bytecode_d,
            ram_d,
            instruction_shifts,
            bytecode_shifts,
            ram_shifts,
        }
    }

    /// Total number of committed polynomials (2 dense + all one-hot chunks).
    #[inline]
    pub fn num_committed_polynomials(&self) -> usize {
        2 + self.instruction_d + self.bytecode_d + self.ram_d
    }

    /// Extract the `idx`-th chunk from a 128-bit instruction lookup index.
    #[inline]
    pub fn lookup_index_chunk(&self, index: u128, idx: usize) -> u8 {
        ((index >> self.instruction_shifts[idx]) & (self.k_chunk - 1) as u128) as u8
    }

    /// Extract the `idx`-th chunk from a bytecode PC index.
    #[inline]
    pub fn bytecode_pc_chunk(&self, pc: u32, idx: usize) -> u8 {
        ((pc as usize >> self.bytecode_shifts[idx]) & (self.k_chunk - 1)) as u8
    }

    /// Extract the `idx`-th chunk from a remapped RAM address.
    #[inline]
    pub fn ram_address_chunk(&self, address: u64, idx: usize) -> u8 {
        ((address >> self.ram_shifts[idx]) & (self.k_chunk - 1) as u64) as u8
    }
}

/// Pre-compute shift table: `shifts[i] = log_k_chunk * (d - 1 - i)`.
///
/// Chunk 0 covers the most significant bits, chunk d-1 covers the least significant.
fn compute_shifts(log_k_chunk: usize, d: usize) -> Vec<usize> {
    (0..d).map(|i| log_k_chunk * (d - 1 - i)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shifts_cover_full_range() {
        let config = WitnessConfig::new(4, 128, 16, 24);
        assert_eq!(config.instruction_d, 32); // 128 / 4
        assert_eq!(config.bytecode_d, 4); // 16 / 4
        assert_eq!(config.ram_d, 6); // 24 / 4
        assert_eq!(config.k_chunk, 16);
    }

    #[test]
    fn chunk_8_bit() {
        let config = WitnessConfig::new(8, 128, 16, 24);
        assert_eq!(config.instruction_d, 16); // 128 / 8
        assert_eq!(config.bytecode_d, 2); // 16 / 8
        assert_eq!(config.ram_d, 3); // 24 / 8
        assert_eq!(config.k_chunk, 256);
    }

    #[test]
    fn lookup_index_chunk_extraction() {
        let config = WitnessConfig::new(4, 16, 8, 8);
        // 16-bit index = 0xABCD → chunks (MSB first): A, B, C, D
        let index: u128 = 0xABCD;
        assert_eq!(config.lookup_index_chunk(index, 0), 0xA);
        assert_eq!(config.lookup_index_chunk(index, 1), 0xB);
        assert_eq!(config.lookup_index_chunk(index, 2), 0xC);
        assert_eq!(config.lookup_index_chunk(index, 3), 0xD);
    }

    #[test]
    fn bytecode_pc_chunk_extraction() {
        let config = WitnessConfig::new(4, 16, 12, 8);
        // 12-bit PC = 0x7F3 → chunks: 7, F, 3
        assert_eq!(config.bytecode_pc_chunk(0x7F3, 0), 0x7);
        assert_eq!(config.bytecode_pc_chunk(0x7F3, 1), 0xF);
        assert_eq!(config.bytecode_pc_chunk(0x7F3, 2), 0x3);
    }

    #[test]
    fn ram_address_chunk_extraction() {
        let config = WitnessConfig::new(8, 128, 16, 24);
        // 24-bit address = 0x123456 → chunks: 0x12, 0x34, 0x56
        assert_eq!(config.ram_address_chunk(0x12_3456, 0), 0x12);
        assert_eq!(config.ram_address_chunk(0x12_3456, 1), 0x34);
        assert_eq!(config.ram_address_chunk(0x12_3456, 2), 0x56);
    }

    #[test]
    fn num_committed_polynomials() {
        let config = WitnessConfig::new(4, 128, 16, 24);
        // 2 dense (RdInc, RamInc) + 32 instruction + 4 bytecode + 6 ram = 44
        assert_eq!(config.num_committed_polynomials(), 44);
    }

    #[test]
    fn non_aligned_dimensions() {
        // 10-bit bytecode with 4-bit chunks → ceil(10/4) = 3
        let config = WitnessConfig::new(4, 128, 10, 7);
        assert_eq!(config.bytecode_d, 3);
        assert_eq!(config.ram_d, 2); // ceil(7/4)
    }
}
