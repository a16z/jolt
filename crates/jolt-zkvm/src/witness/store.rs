//! Witness polynomial storage for the proving pipeline.
//!
//! [`WitnessStore`] owns the evaluation tables produced during witness
//! generation. Stages borrow data via [`get()`](WitnessStore::get) and
//! consume it via [`take()`](WitnessStore::take) when constructing opening
//! claims for stage 8.

use std::collections::HashMap;

use jolt_field::Field;

/// Opaque polynomial identity tag.
///
/// Maps to constants in [`jolt_ir::zkvm::tags::poly`]. The WitnessStore treats
/// these as opaque keys — all semantic interpretation happens in the
/// jolt-ir tags module and stage implementations.
pub type PolynomialTag = u64;

/// Owns all committed polynomial evaluation tables for the duration of proving.
///
/// Created during witness generation, holds data until stages consume it
/// for opening proofs. Stages borrow slices via [`get()`](Self::get) during
/// sumcheck witness construction and move data out via [`take()`](Self::take)
/// when building [`ProverClaim`](jolt_openings::ProverClaim)s.
///
/// # Lifecycle
///
/// 1. Witness generation produces `WitnessStore<F>` + streaming commitment outputs.
/// 2. Stages 1–7 borrow from the store via `get()`. Working copies for
///    `SumcheckCompute` are cloned from these borrows.
/// 3. `extract_claims()` calls `take()` to move evaluation tables into
///    `ProverClaim.evaluations`. After all stages complete, the store is empty.
/// 4. Stage 8 consumes `ProverClaim.evaluations` for `RlcReduction` + `PCS::open`.
pub struct WitnessStore<F: Field> {
    tables: HashMap<PolynomialTag, Vec<F>>,
}

impl<F: Field> WitnessStore<F> {
    pub fn new() -> Self {
        Self {
            tables: HashMap::new(),
        }
    }

    pub fn from_tables(tables: HashMap<PolynomialTag, Vec<F>>) -> Self {
        Self { tables }
    }

    pub fn insert(&mut self, tag: PolynomialTag, evaluations: Vec<F>) -> Option<Vec<F>> {
        self.tables.insert(tag, evaluations)
    }

    /// Borrows the evaluation table for `tag`.
    ///
    /// # Panics
    ///
    /// Panics if no table exists for `tag`.
    pub fn get(&self, tag: PolynomialTag) -> &[F] {
        self.tables
            .get(&tag)
            .unwrap_or_else(|| panic!("polynomial tag {tag} not found in witness store"))
    }

    pub fn try_get(&self, tag: PolynomialTag) -> Option<&[F]> {
        self.tables.get(&tag).map(Vec::as_slice)
    }

    /// Moves the evaluation table out of the store.
    ///
    /// After this call, `get(tag)` will panic. Used by `extract_claims()`
    /// to transfer ownership into [`ProverClaim`](jolt_openings::ProverClaim)s.
    ///
    /// # Panics
    ///
    /// Panics if no table exists for `tag`.
    pub fn take(&mut self, tag: PolynomialTag) -> Vec<F> {
        self.tables
            .remove(&tag)
            .unwrap_or_else(|| panic!("polynomial tag {tag} not found in witness store"))
    }

    pub fn contains(&self, tag: PolynomialTag) -> bool {
        self.tables.contains_key(&tag)
    }

    pub fn len(&self) -> usize {
        self.tables.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }
}

impl<F: Field> Default for WitnessStore<F> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;

    #[test]
    fn insert_and_get() {
        let mut store = WitnessStore::<Fr>::new();
        let evals = vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
        let _ = store.insert(100, evals.clone());
        assert_eq!(store.get(100), &evals);
    }

    #[test]
    fn take_removes_entry() {
        let mut store = WitnessStore::<Fr>::new();
        let evals = vec![Fr::from_u64(42)];
        let _ = store.insert(100, evals.clone());
        assert!(store.contains(100));
        let taken = store.take(100);
        assert_eq!(taken, evals);
        assert!(!store.contains(100));
    }

    #[test]
    #[should_panic(expected = "not found")]
    fn get_missing_panics() {
        let store = WitnessStore::<Fr>::new();
        let _ = store.get(999);
    }

    #[test]
    fn try_get_returns_none_for_missing() {
        let store = WitnessStore::<Fr>::new();
        assert!(store.try_get(999).is_none());
    }

    #[test]
    fn from_tables_constructor() {
        let mut tables = HashMap::new();
        let _ = tables.insert(100u64, vec![Fr::from_u64(1)]);
        let _ = tables.insert(200u64, vec![Fr::from_u64(2)]);
        let store = WitnessStore::from_tables(tables);
        assert_eq!(store.len(), 2);
        assert!(store.contains(100));
        assert!(store.contains(200));
    }

    #[test]
    fn default_is_empty() {
        let store = WitnessStore::<Fr>::default();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }
}
