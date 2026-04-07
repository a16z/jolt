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

impl<F: Field> Default for PreprocessedSource<F> {
    fn default() -> Self {
        Self::new()
    }
}
