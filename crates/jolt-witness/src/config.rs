//! One-hot decomposition configuration for polynomial generation.
//!
//! [`PolynomialConfig`] determines how per-cycle addresses are decomposed
//! into one-hot committed polynomial buffers. Constructed at preprocessing
//! time and shared between prover and verifier.

/// One-hot decomposition parameters for committed polynomial generation.
///
/// Determines how 128-bit instruction indices, bytecode PCs, and RAM
/// addresses are decomposed into `d` one-hot polynomial buffers, each
/// covering a `log_k_chunk`-bit window.
#[derive(Clone, Debug)]
pub struct PolynomialConfig {
    /// Log₂ of one-hot chunk size (typically 4 or 8).
    pub log_k_chunk: usize,
    /// One-hot chunk size: `2^log_k_chunk`.
    pub k_chunk: usize,

    /// Number of one-hot polynomials for instruction lookups.
    pub instruction_d: usize,
    /// Number of one-hot polynomials for bytecode PC.
    pub bytecode_d: usize,
    /// Number of one-hot polynomials for RAM addresses.
    pub ram_d: usize,

    instruction_shifts: Vec<usize>,
    bytecode_shifts: Vec<usize>,
    ram_shifts: Vec<usize>,
}

impl PolynomialConfig {
    /// Constructs a config from decomposition parameters.
    ///
    /// # Arguments
    ///
    /// * `log_k_chunk` — Log₂ of one-hot chunk size.
    /// * `log_k_instruction` — Bit width of instruction lookup indices.
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

        Self {
            log_k_chunk,
            k_chunk,
            instruction_d,
            bytecode_d,
            ram_d,
            instruction_shifts: compute_shifts(log_k_chunk, instruction_d),
            bytecode_shifts: compute_shifts(log_k_chunk, bytecode_d),
            ram_shifts: compute_shifts(log_k_chunk, ram_d),
        }
    }

    /// Total number of committed polynomial buffers (2 dense + all one-hot).
    #[inline]
    pub fn num_polynomials(&self) -> usize {
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

/// Chunk 0 = most significant bits, chunk d-1 = least significant.
fn compute_shifts(log_k_chunk: usize, d: usize) -> Vec<usize> {
    (0..d).map(|i| log_k_chunk * (d - 1 - i)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions() {
        let c = PolynomialConfig::new(4, 128, 16, 24);
        assert_eq!(c.instruction_d, 32);
        assert_eq!(c.bytecode_d, 4);
        assert_eq!(c.ram_d, 6);
        assert_eq!(c.k_chunk, 16);
    }

    #[test]
    fn chunk_8_bit() {
        let c = PolynomialConfig::new(8, 128, 16, 24);
        assert_eq!(c.instruction_d, 16);
        assert_eq!(c.bytecode_d, 2);
        assert_eq!(c.ram_d, 3);
        assert_eq!(c.k_chunk, 256);
    }

    #[test]
    fn lookup_index_chunk_extraction() {
        let c = PolynomialConfig::new(4, 16, 8, 8);
        let index: u128 = 0xABCD;
        assert_eq!(c.lookup_index_chunk(index, 0), 0xA);
        assert_eq!(c.lookup_index_chunk(index, 1), 0xB);
        assert_eq!(c.lookup_index_chunk(index, 2), 0xC);
        assert_eq!(c.lookup_index_chunk(index, 3), 0xD);
    }

    #[test]
    fn bytecode_pc_chunk_extraction() {
        let c = PolynomialConfig::new(4, 16, 12, 8);
        assert_eq!(c.bytecode_pc_chunk(0x7F3, 0), 0x7);
        assert_eq!(c.bytecode_pc_chunk(0x7F3, 1), 0xF);
        assert_eq!(c.bytecode_pc_chunk(0x7F3, 2), 0x3);
    }

    #[test]
    fn ram_address_chunk_extraction() {
        let c = PolynomialConfig::new(8, 128, 16, 24);
        assert_eq!(c.ram_address_chunk(0x12_3456, 0), 0x12);
        assert_eq!(c.ram_address_chunk(0x12_3456, 1), 0x34);
        assert_eq!(c.ram_address_chunk(0x12_3456, 2), 0x56);
    }

    #[test]
    fn num_polynomials() {
        let c = PolynomialConfig::new(4, 128, 16, 24);
        assert_eq!(c.num_polynomials(), 2 + 32 + 4 + 6);
    }

    #[test]
    fn non_aligned_dimensions() {
        let c = PolynomialConfig::new(4, 128, 10, 7);
        assert_eq!(c.bytecode_d, 3);
        assert_eq!(c.ram_d, 2);
    }
}
