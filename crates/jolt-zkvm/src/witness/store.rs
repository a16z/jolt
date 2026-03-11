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
use jolt_poly::OneHotPolynomial;

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
/// One-hot polynomials (RA witnesses) are stored in two forms:
/// - Dense `Vec<F>` in `tables` — for sumcheck stage access via `get()`
/// - Sparse [`OneHotPolynomial`] in `one_hot` — for PCS commit via `get_one_hot()`
///
/// The sparse form enables O(T) generator-lookup commits instead of
/// O(T × K × 254) full MSM. The commit path should prefer `get_one_hot()`
/// when available, falling back to the dense table otherwise.
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
    tables: BTreeMap<PolynomialTag, Vec<F>>,
    one_hot: BTreeMap<PolynomialTag, OneHotPolynomial>,
}

impl<F: Field> WitnessStore<F> {
    pub fn new() -> Self {
        Self {
            tables: BTreeMap::new(),
            one_hot: BTreeMap::new(),
        }
    }

    pub fn from_tables(tables: BTreeMap<PolynomialTag, Vec<F>>) -> Self {
        Self {
            tables,
            one_hot: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, tag: PolynomialTag, evaluations: Vec<F>) -> Option<Vec<F>> {
        self.tables.insert(tag, evaluations)
    }

    /// Stores a one-hot polynomial's sparse representation alongside its dense table.
    ///
    /// The dense table must already be inserted via [`insert()`](Self::insert).
    /// The sparse representation is used by the PCS commit path for
    /// generator-lookup instead of full MSM.
    pub fn insert_one_hot(&mut self, tag: PolynomialTag, poly: OneHotPolynomial) {
        let _ = self.one_hot.insert(tag, poly);
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

    /// Returns the sparse one-hot representation if this polynomial was one-hot.
    ///
    /// The commit path should prefer this over `get()` when available:
    /// ```ignore
    /// let (commitment, hint) = if let Some(oh) = store.get_one_hot(tag) {
    ///     PCS::commit(oh, setup)  // sparse: O(T) generator lookups
    /// } else {
    ///     PCS::commit(store.get(tag), setup)  // dense: O(T×K×254) MSM
    /// };
    /// ```
    pub fn get_one_hot(&self, tag: PolynomialTag) -> Option<&OneHotPolynomial> {
        self.one_hot.get(&tag)
    }

    /// Moves the evaluation table out of the store.
    ///
    /// After this call, `get(tag)` will panic. Used by `extract_claims()`
    /// to transfer ownership into [`ProverClaim`](jolt_openings::ProverClaim)s.
    /// Also removes the one-hot representation if present.
    ///
    /// # Panics
    ///
    /// Panics if no table exists for `tag`.
    pub fn take(&mut self, tag: PolynomialTag) -> Vec<F> {
        let _ = self.one_hot.remove(&tag);
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
        let mut tables = BTreeMap::new();
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
