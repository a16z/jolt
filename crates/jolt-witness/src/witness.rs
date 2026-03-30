//! Witness polynomial storage and device buffer provision.
//!
//! [`Witness`] owns committed polynomial evaluation tables produced during
//! witness generation. It implements [`WitnessSink`] to absorb builder output
//! directly, and provides [`WitnessProvider`] for the runtime to load
//! polynomials as device buffers on demand.

use std::collections::BTreeMap;

use jolt_compute::{BufferProvider, ComputeBackend};
use jolt_field::Field;
use jolt_poly::OneHotPolynomial;

use crate::polynomial_id::PolynomialId;
use crate::sink::{ChunkData, PolynomialKind, WitnessSink};

/// Owns all committed polynomial evaluation tables for the duration of proving.
///
/// Created during witness generation via the [`WitnessSink`] protocol, holds
/// data until the prover runtime consumes it. Stages borrow slices via
/// [`get()`](Self::get) during sumcheck witness construction and move data out
/// via [`take()`](Self::take) when building opening claims.
///
/// One-hot polynomials (RA witnesses) are stored in two forms:
/// - Dense `Vec<F>` in `tables` — for sumcheck stage access via `get()`
/// - Sparse [`OneHotPolynomial`] in `one_hot` — for PCS commit via generator lookup
pub struct Witness<F: Field> {
    tables: BTreeMap<PolynomialId, Vec<F>>,
    one_hot: BTreeMap<PolynomialId, OneHotPolynomial>,
    pending: BTreeMap<PolynomialId, PendingPoly<F>>,
}

enum PendingPoly<F: Field> {
    Dense(Vec<F>),
    OneHot { k: usize, indices: Vec<Option<u8>> },
}

impl<F: Field> Witness<F> {
    pub fn new() -> Self {
        Self {
            tables: BTreeMap::new(),
            one_hot: BTreeMap::new(),
            pending: BTreeMap::new(),
        }
    }

    pub fn from_tables(tables: BTreeMap<PolynomialId, Vec<F>>) -> Self {
        Self {
            tables,
            one_hot: BTreeMap::new(),
            pending: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, id: PolynomialId, evaluations: Vec<F>) -> Option<Vec<F>> {
        self.tables.insert(id, evaluations)
    }

    /// Stores a one-hot polynomial's sparse representation alongside its dense table.
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
            .unwrap_or_else(|| panic!("polynomial {id:?} not found in witness"))
    }

    pub fn try_get(&self, id: PolynomialId) -> Option<&[F]> {
        self.tables.get(&id).map(Vec::as_slice)
    }

    /// Returns the sparse one-hot representation if this polynomial was one-hot.
    pub fn get_one_hot(&self, id: PolynomialId) -> Option<&OneHotPolynomial> {
        self.one_hot.get(&id)
    }

    /// Moves the evaluation table out of the witness.
    ///
    /// After this call, `get(id)` will panic. Also removes the one-hot
    /// representation if present.
    ///
    /// # Panics
    ///
    /// Panics if no table exists for `id`.
    pub fn take(&mut self, id: PolynomialId) -> Vec<F> {
        let _ = self.one_hot.remove(&id);
        self.tables
            .remove(&id)
            .unwrap_or_else(|| panic!("polynomial {id:?} not found in witness"))
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

    /// Creates a [`WitnessProvider`] for the runtime to load polynomials as device buffers.
    ///
    /// `index_map` maps compiler-assigned polynomial indices to [`PolynomialId`]s.
    /// The mapping is determined by the protocol graph's polynomial ordering.
    pub fn provider<'a>(&'a mut self, index_map: &'a [PolynomialId]) -> WitnessProvider<'a, F> {
        WitnessProvider {
            witness: self,
            index_map,
        }
    }
}

impl<F: Field> Default for Witness<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field> WitnessSink<F> for Witness<F> {
    fn on_polynomial_start(
        &mut self,
        poly_id: PolynomialId,
        total_len: usize,
        kind: PolynomialKind,
    ) {
        let _ = self.pending.entry(poly_id).or_insert_with(|| match kind {
            PolynomialKind::Dense => PendingPoly::Dense(Vec::with_capacity(total_len)),
            PolynomialKind::OneHot { k } => PendingPoly::OneHot {
                k,
                indices: Vec::with_capacity(total_len),
            },
        });
    }

    fn on_chunk(&mut self, poly_id: PolynomialId, data: ChunkData<'_, F>) {
        match (self.pending.get_mut(&poly_id), data) {
            (Some(PendingPoly::Dense(v)), ChunkData::Dense(chunk)) => {
                v.extend_from_slice(chunk);
            }
            (Some(PendingPoly::OneHot { indices, .. }), ChunkData::OneHot(chunk)) => {
                indices.extend_from_slice(chunk);
            }
            _ => panic!("chunk type mismatch or unknown polynomial {poly_id:?}"),
        }
    }

    fn on_polynomial_end(&mut self, poly_id: PolynomialId) {
        if let Some(pending) = self.pending.remove(&poly_id) {
            match pending {
                PendingPoly::Dense(v) => {
                    let _ = self.tables.insert(poly_id, v);
                }
                PendingPoly::OneHot { k, indices } => {
                    let dense = expand_one_hot::<F>(k, &indices);
                    let _ = self.tables.insert(poly_id, dense);
                    let _ = self
                        .one_hot
                        .insert(poly_id, OneHotPolynomial::new(k, indices));
                }
            }
        }
    }

    fn finish(&mut self) {}
}

/// Expands one-hot indices into a flat evaluation table.
fn expand_one_hot<F: Field>(k: usize, indices: &[Option<u8>]) -> Vec<F> {
    let n = indices.len();
    let mut table = vec![F::zero(); n * k];
    for (cycle, &idx) in indices.iter().enumerate() {
        if let Some(i) = idx {
            table[cycle * k + i as usize] = F::one();
        }
    }
    table
}

/// Adapter that implements [`BufferProvider`] by mapping compiler polynomial
/// indices to [`PolynomialId`]s and uploading data from a [`Witness`].
///
/// Created via [`Witness::provider`].
pub struct WitnessProvider<'a, F: Field> {
    witness: &'a mut Witness<F>,
    index_map: &'a [PolynomialId],
}

impl<B: ComputeBackend, F: Field> BufferProvider<B, F> for WitnessProvider<'_, F> {
    fn load(&mut self, poly_index: usize, backend: &B) -> B::Buffer<F> {
        let id = self.index_map[poly_index];
        let data = self.witness.take(id);
        backend.upload(&data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use num_traits::{One, Zero};

    #[test]
    fn insert_and_get() {
        let mut w = Witness::<Fr>::new();
        let evals = vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
        let _ = w.insert(PolynomialId::RamInc, evals.clone());
        assert_eq!(w.get(PolynomialId::RamInc), &evals);
    }

    #[test]
    fn take_removes_entry() {
        let mut w = Witness::<Fr>::new();
        let evals = vec![Fr::from_u64(42)];
        let _ = w.insert(PolynomialId::RamInc, evals.clone());
        assert!(w.contains(PolynomialId::RamInc));
        let taken = w.take(PolynomialId::RamInc);
        assert_eq!(taken, evals);
        assert!(!w.contains(PolynomialId::RamInc));
    }

    #[test]
    #[should_panic(expected = "not found")]
    fn get_missing_panics() {
        let w = Witness::<Fr>::new();
        let _ = w.get(PolynomialId::RamInc);
    }

    #[test]
    fn try_get_returns_none_for_missing() {
        let w = Witness::<Fr>::new();
        assert!(w.try_get(PolynomialId::RamInc).is_none());
    }

    #[test]
    fn from_tables_constructor() {
        let mut tables = BTreeMap::new();
        let _ = tables.insert(PolynomialId::RamInc, vec![Fr::from_u64(1)]);
        let _ = tables.insert(PolynomialId::RdInc, vec![Fr::from_u64(2)]);
        let w = Witness::from_tables(tables);
        assert_eq!(w.len(), 2);
        assert!(w.contains(PolynomialId::RamInc));
        assert!(w.contains(PolynomialId::RdInc));
    }

    #[test]
    fn default_is_empty() {
        let w = Witness::<Fr>::default();
        assert!(w.is_empty());
        assert_eq!(w.len(), 0);
    }

    #[test]
    fn sink_dense_chunks() {
        let mut w = Witness::<Fr>::new();
        w.on_polynomial_start(PolynomialId::RamInc, 3, PolynomialKind::Dense);
        w.on_chunk(
            PolynomialId::RamInc,
            ChunkData::Dense(&[Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)]),
        );
        w.on_polynomial_end(PolynomialId::RamInc);
        w.finish();

        assert_eq!(w.get(PolynomialId::RamInc).len(), 3);
        assert_eq!(w.get(PolynomialId::RamInc)[0], Fr::from_u64(1));
        assert!(w.get_one_hot(PolynomialId::RamInc).is_none());
    }

    #[test]
    fn sink_onehot_expanded() {
        let mut w = Witness::<Fr>::new();
        w.on_polynomial_start(
            PolynomialId::InstructionRa(0),
            2,
            PolynomialKind::OneHot { k: 4 },
        );
        w.on_chunk(
            PolynomialId::InstructionRa(0),
            ChunkData::OneHot(&[Some(1), Some(3)]),
        );
        w.on_polynomial_end(PolynomialId::InstructionRa(0));
        w.finish();

        let table = w.get(PolynomialId::InstructionRa(0));
        assert_eq!(table.len(), 8);
        assert_eq!(table[1], Fr::one());
        assert_eq!(table[0], Fr::zero());
        assert_eq!(table[7], Fr::one());
        assert_eq!(table[4], Fr::zero());

        assert!(w.get_one_hot(PolynomialId::InstructionRa(0)).is_some());
    }

    #[test]
    fn sink_none_onehot_all_zeros() {
        let mut w = Witness::<Fr>::new();
        w.on_polynomial_start(PolynomialId::RamRa(0), 1, PolynomialKind::OneHot { k: 4 });
        w.on_chunk(PolynomialId::RamRa(0), ChunkData::OneHot(&[None]));
        w.on_polynomial_end(PolynomialId::RamRa(0));
        w.finish();

        let table = w.get(PolynomialId::RamRa(0));
        assert_eq!(table.len(), 4);
        assert!(table.iter().all(|&v| v == Fr::zero()));
    }
}
