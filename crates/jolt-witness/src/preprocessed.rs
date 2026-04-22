//! Static preprocessed polynomial data.
//!
//! [`PreprocessedSource`] holds polynomial data known before proving:
//! bytecode tables, initial memory images, I/O masks, etc.
//!
//! Handles `PolySource::Preprocessed`: IoMask, ValIo, RamUnmap,
//! RamInit, LookupTable, BytecodeTable(i).

use std::collections::HashMap;

use jolt_compiler::PolynomialId;
use jolt_field::Field;
use jolt_verifier::ProverConfig;

/// Stores preprocessed polynomial data for the prover.
///
/// Data is inserted during setup (before proving begins) and served
/// as borrowed slices during execution.
pub struct PreprocessedSource<F> {
    data: HashMap<PolynomialId, Vec<F>>,
}

impl<F: Field> PreprocessedSource<F> {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Build RAM-related preprocessed polynomials from prover config and
    /// initial memory state.
    ///
    /// Inserts: `IoMask`, `ValIo`, `RamUnmap`, `RamInit`.
    pub fn populate_ram(&mut self, config: &ProverConfig, initial_state: &[u64]) {
        let k = config.ram_k;
        let lowest = config.ram_lowest_address;

        // RamUnmap: unmap[k] = k * 8 + lowest_address
        let unmap: Vec<F> = (0..k).map(|i| F::from_u64(i as u64 * 8 + lowest)).collect();
        self.insert(PolynomialId::RamUnmap, unmap);

        // IoMask: 1 for the full I/O region [input_start, RAM_START_ADDRESS).
        // Must match jolt-core's range which covers inputs, outputs, panic,
        // termination, AND advice regions (even when advice is empty, the mask
        // MLE affects sumcheck round polynomials at extrapolated points).
        let io_start = config.input_word_offset;
        let io_end = ((config.memory_start - config.ram_lowest_address) / 8) as usize;
        let mut mask = vec![F::zero(); k];
        for m in mask.iter_mut().take(io_end.min(k)).skip(io_start) {
            *m = F::one();
        }
        self.insert(PolynomialId::IoMask, mask);

        // ValIo: I/O region values (inputs, outputs, panic, termination)
        let mut val_io = vec![F::zero(); k];
        for (i, chunk) in config.inputs.chunks(8).enumerate() {
            let idx = config.input_word_offset + i;
            if idx < k {
                val_io[idx] = F::from_u64(pack_le_word(chunk));
            }
        }
        for (i, chunk) in config.outputs.chunks(8).enumerate() {
            let idx = config.output_word_offset + i;
            if idx < k {
                val_io[idx] = F::from_u64(pack_le_word(chunk));
            }
        }
        if config.panic_word_offset < k {
            val_io[config.panic_word_offset] = F::from_u64(config.panic as u64);
        }
        if config.termination_word_offset < k && !config.panic {
            val_io[config.termination_word_offset] = F::one();
        }
        self.insert(PolynomialId::ValIo, val_io);

        // RamInit: initial memory state as field elements
        debug_assert_eq!(initial_state.len(), k);
        let ram_init: Vec<F> = initial_state.iter().map(|&v| F::from_u64(v)).collect();
        self.insert(PolynomialId::RamInit, ram_init);
    }

    /// Build FieldReg limb→Fr bridge preprocessed polynomials.
    ///
    /// Inserts the four constant K_reg-element polynomials used by the Stage 2
    /// bridge kernel to reconstruct Fr operand values from four 64-bit scalar-
    /// register limbs:
    ///
    /// | PolynomialId | Nonzero coefficients |
    /// |---|---|
    /// | `BridgeValWeightA` | `[10] = 1`, `[11] = 2^64`, `[12] = 2^128`, `[13] = 2^192` |
    /// | `BridgeValWeightB` | `[14] = 1`, `[15] = 2^64`, `[16] = 2^128`, `[17] = 2^192` |
    /// | `BridgeAnchorA` | `[10] = 1` |
    /// | `BridgeAnchorB` | `[14] = 1` |
    ///
    /// `K_reg = 128` (7-bit register index, matches the scalar register file).
    /// The weight `2^(64·k)` is built by repeatedly multiplying `2^64` —
    /// `F::from_u128(1u128 << 64)` does not fit in a u128 via shift, so we
    /// build it as `F::from_u128(1) * (2^64)` chains; simpler: use
    /// `F::from_u128(1u128 << 63) * F::from_u128(2)` for 2^64, then square
    /// for 2^128 and multiply for 2^192.
    pub fn populate_bridge(&mut self) {
        const K_REG: usize = 128;
        // Build 2^64 as `(1u128 << 63) * 2` to stay within u128 range.
        let two_pow_64 = F::from_u128(1u128 << 63) * F::from_u128(2);
        let two_pow_128 = two_pow_64 * two_pow_64;
        let two_pow_192 = two_pow_128 * two_pow_64;

        let mut weight_a = vec![F::zero(); K_REG];
        weight_a[10] = F::one();
        weight_a[11] = two_pow_64;
        weight_a[12] = two_pow_128;
        weight_a[13] = two_pow_192;
        self.insert(PolynomialId::BridgeValWeightA, weight_a);

        let mut weight_b = vec![F::zero(); K_REG];
        weight_b[14] = F::one();
        weight_b[15] = two_pow_64;
        weight_b[16] = two_pow_128;
        weight_b[17] = two_pow_192;
        self.insert(PolynomialId::BridgeValWeightB, weight_b);

        let mut anchor_a = vec![F::zero(); K_REG];
        anchor_a[10] = F::one();
        self.insert(PolynomialId::BridgeAnchorA, anchor_a);

        let mut anchor_b = vec![F::zero(); K_REG];
        anchor_b[14] = F::one();
        self.insert(PolynomialId::BridgeAnchorB, anchor_b);
    }

    /// Insert a preprocessed polynomial.
    pub fn insert(&mut self, poly_id: PolynomialId, values: Vec<F>) {
        let _ = self.data.insert(poly_id, values);
    }

    /// Borrow preprocessed polynomial data.
    ///
    /// Panics if `poly_id` hasn't been inserted.
    pub fn get(&self, poly_id: PolynomialId) -> &[F] {
        self.data
            .get(&poly_id)
            .unwrap_or_else(|| panic!("PreprocessedSource: {poly_id:?} not loaded"))
    }
}

/// Pack up to 8 bytes into a little-endian u64 word.
fn pack_le_word(bytes: &[u8]) -> u64 {
    let mut word = 0u64;
    for (j, &b) in bytes.iter().enumerate() {
        word |= (b as u64) << (j * 8);
    }
    word
}

impl<F: Field> Default for PreprocessedSource<F> {
    fn default() -> Self {
        Self::new()
    }
}
