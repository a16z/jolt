//! Witness polynomial storage for the proving pipeline.
//!
//! [`WitnessStore`] owns the evaluation tables produced during witness
//! generation. Stages borrow data via [`get()`](WitnessStore::get) and
//! consume it via [`take()`](WitnessStore::take) when constructing opening
//! claims for stage 8.
//!
//! One-hot polynomials are stored in both dense form (for sumcheck stages)
//! and sparse [`OneHotPolynomial`] form (for PCS commit via generator lookup
//! instead of full MSM). The commit path checks [`get_one_hot()`](WitnessStore::get_one_hot)
//! to select the fast sparse commit when available.

use std::collections::BTreeMap;

use jolt_field::Field;
use jolt_ir::PolynomialId;
use jolt_poly::OneHotPolynomial;

/// Owns all committed polynomial evaluation tables for the duration of proving.
///
/// Created during witness generation, holds data until stages consume it
/// for opening proofs. Stages borrow slices via [`get()`](Self::get) during
/// sumcheck witness construction and move data out via [`take()`](Self::take)
/// when building [`ProverClaim`](jolt_openings::ProverClaim)s.
///
/// One-hot polynomials (RA witnesses) are stored in two forms:
/// - Dense `Vec<F>` in `tables` -- for sumcheck stage access via `get()`
/// - Sparse [`OneHotPolynomial`] in `one_hot` -- for PCS commit via `get_one_hot()`
///
/// The sparse form enables O(T) generator-lookup commits instead of
/// O(T x K x 254) full MSM. The commit path should prefer `get_one_hot()`
/// when available, falling back to the dense table otherwise.
///
/// # Lifecycle
///
/// 1. Witness generation produces `WitnessStore<F>` + streaming commitment outputs.
/// 2. Stages 1-7 borrow from the store via `get()`. Working copies for
///    `SumcheckCompute` are cloned from these borrows.
/// 3. `extract_claims()` calls `take()` to move evaluation tables into
///    `ProverClaim.evaluations`. After all stages complete, the store is empty.
/// 4. Stage 8 consumes `ProverClaim.evaluations` for `RlcReduction` + `PCS::open`.
pub struct WitnessStore<F: Field> {
    tables: BTreeMap<PolynomialId, Vec<F>>,
    one_hot: BTreeMap<PolynomialId, OneHotPolynomial>,
}

impl<F: Field> WitnessStore<F> {
    pub fn new() -> Self {
        Self {
            tables: BTreeMap::new(),
            one_hot: BTreeMap::new(),
        }
    }

    pub fn from_tables(tables: BTreeMap<PolynomialId, Vec<F>>) -> Self {
        Self {
            tables,
            one_hot: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, id: PolynomialId, evaluations: Vec<F>) -> Option<Vec<F>> {
        self.tables.insert(id, evaluations)
    }

    /// Stores a one-hot polynomial's sparse representation alongside its dense table.
    ///
    /// The dense table must already be inserted via [`insert()`](Self::insert).
    /// The sparse representation is used by the PCS commit path for
    /// generator-lookup instead of full MSM.
    pub fn insert_one_hot(&mut self, id: PolynomialId, poly: OneHotPolynomial) {
        let _ = self.one_hot.insert(id, poly);
    }

    /// Borrows the evaluation table for `id`.
    ///
    /// # Panics
    ///
    /// Panics if no table exists for `id`.
    pub fn get(&self, id: PolynomialId) -> &[F] {
        self.tables
            .get(&id)
            .unwrap_or_else(|| panic!("polynomial {id:?} not found in witness store"))
    }

    pub fn try_get(&self, id: PolynomialId) -> Option<&[F]> {
        self.tables.get(&id).map(Vec::as_slice)
    }

    /// Returns the sparse one-hot representation if this polynomial was one-hot.
    ///
    /// The commit path should prefer this over `get()` when available:
    /// ```ignore
    /// let (commitment, hint) = if let Some(oh) = store.get_one_hot(id) {
    ///     PCS::commit(oh, setup)  // sparse: O(T) generator lookups
    /// } else {
    ///     PCS::commit(store.get(id), setup)  // dense: O(T×K×254) MSM
    /// };
    /// ```
    pub fn get_one_hot(&self, id: PolynomialId) -> Option<&OneHotPolynomial> {
        self.one_hot.get(&id)
    }

    /// Moves the evaluation table out of the store.
    ///
    /// After this call, `get(id)` will panic. Used by `extract_claims()`
    /// to transfer ownership into [`ProverClaim`](jolt_openings::ProverClaim)s.
    /// Also removes the one-hot representation if present.
    ///
    /// # Panics
    ///
    /// Panics if no table exists for `id`.
    pub fn take(&mut self, id: PolynomialId) -> Vec<F> {
        let _ = self.one_hot.remove(&id);
        self.tables
            .remove(&id)
            .unwrap_or_else(|| panic!("polynomial {id:?} not found in witness store"))
    }

    pub fn contains(&self, id: PolynomialId) -> bool {
        self.tables.contains_key(&id)
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
        let _ = store.insert(PolynomialId::RamInc, evals.clone());
        assert_eq!(store.get(PolynomialId::RamInc), &evals);
    }

    #[test]
    fn take_removes_entry() {
        let mut store = WitnessStore::<Fr>::new();
        let evals = vec![Fr::from_u64(42)];
        let _ = store.insert(PolynomialId::RamInc, evals.clone());
        assert!(store.contains(PolynomialId::RamInc));
        let taken = store.take(PolynomialId::RamInc);
        assert_eq!(taken, evals);
        assert!(!store.contains(PolynomialId::RamInc));
    }

    #[test]
    #[should_panic(expected = "not found")]
    fn get_missing_panics() {
        let store = WitnessStore::<Fr>::new();
        let _ = store.get(PolynomialId::RamInc);
    }

    #[test]
    fn try_get_returns_none_for_missing() {
        let store = WitnessStore::<Fr>::new();
        assert!(store.try_get(PolynomialId::RamInc).is_none());
    }

    #[test]
    fn from_tables_constructor() {
        let mut tables = BTreeMap::new();
        let _ = tables.insert(PolynomialId::RamInc, vec![Fr::from_u64(1)]);
        let _ = tables.insert(PolynomialId::RdInc, vec![Fr::from_u64(2)]);
        let store = WitnessStore::from_tables(tables);
        assert_eq!(store.len(), 2);
        assert!(store.contains(PolynomialId::RamInc));
        assert!(store.contains(PolynomialId::RdInc));
    }

    #[test]
    fn default_is_empty() {
        let store = WitnessStore::<Fr>::default();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }
}
