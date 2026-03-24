//! [`WitnessSink`] adapter that stores polynomial evaluation tables in a [`WitnessStore`].
//!
//! [`StoreSink`] receives chunks from [`jolt_witness::WitnessBuilder`] and
//! materializes them as dense field-element vectors in the [`WitnessStore`].
//! One-hot chunks are expanded into full evaluation tables AND stored as
//! [`OneHotPolynomial`] for sparse PCS commit (generator lookup instead of MSM).

use jolt_field::Field;
use jolt_ir::PolynomialId;
use jolt_poly::OneHotPolynomial;
use jolt_witness::{ChunkData, PolynomialKind, WitnessSink};

use crate::witness::store::WitnessStore;

/// A [`WitnessSink`] that collects polynomial evaluation tables into a [`WitnessStore`].
///
/// Dense polynomial chunks are appended directly. One-hot chunks are both:
/// - Expanded into dense evaluation tables (for sumcheck stage access)
/// - Stored as [`OneHotPolynomial`] (for sparse PCS commit via generator lookup)
pub struct StoreSink<'a, F: Field> {
    store: &'a mut WitnessStore<F>,
    pending: std::collections::BTreeMap<PolynomialId, PendingPoly<F>>,
}

enum PendingPoly<F: Field> {
    Dense(Vec<F>),
    OneHot { k: usize, indices: Vec<Option<u8>> },
}

impl<'a, F: Field> StoreSink<'a, F> {
    pub fn new(store: &'a mut WitnessStore<F>) -> Self {
        Self {
            store,
            pending: std::collections::BTreeMap::new(),
        }
    }
}

impl<F: Field> WitnessSink<F> for StoreSink<'_, F> {
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
                    let _ = self.store.insert(poly_id, v);
                }
                PendingPoly::OneHot { k, indices } => {
                    let dense = expand_one_hot::<F>(k, &indices);
                    let _ = self.store.insert(poly_id, dense);
                    self.store
                        .insert_one_hot(poly_id, OneHotPolynomial::new(k, indices));
                }
            }
        }
    }

    fn finish(&mut self) {}
}

/// Expands one-hot indices into a flat evaluation table.
///
/// For each cycle, the one-hot index `Some(i)` contributes `1` at position
/// `cycle * k + i`, and `None` contributes all zeros. The resulting vector
/// has length `num_cycles * k`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use jolt_witness::PolynomialKind;
    use num_traits::{One, Zero};

    #[test]
    fn dense_chunks_stored() {
        let mut store = WitnessStore::<Fr>::new();
        {
            let mut sink = StoreSink::new(&mut store);
            sink.on_polynomial_start(PolynomialId::RamInc, 3, PolynomialKind::Dense);
            sink.on_chunk(
                PolynomialId::RamInc,
                ChunkData::Dense(&[Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)]),
            );
            sink.on_polynomial_end(PolynomialId::RamInc);
            sink.finish();
        }
        assert_eq!(store.get(PolynomialId::RamInc).len(), 3);
        assert_eq!(store.get(PolynomialId::RamInc)[0], Fr::from_u64(1));
        assert!(
            store.get_one_hot(PolynomialId::RamInc).is_none(),
            "dense poly should not have one-hot repr"
        );
    }

    #[test]
    fn onehot_chunks_expanded() {
        let mut store = WitnessStore::<Fr>::new();
        {
            let mut sink = StoreSink::new(&mut store);
            sink.on_polynomial_start(
                PolynomialId::InstructionRa(0),
                2,
                PolynomialKind::OneHot { k: 4 },
            );
            sink.on_chunk(
                PolynomialId::InstructionRa(0),
                ChunkData::OneHot(&[Some(1), Some(3)]),
            );
            sink.on_polynomial_end(PolynomialId::InstructionRa(0));
            sink.finish();
        }
        // Dense expansion: 2 cycles x 4 positions = 8 elements
        let table = store.get(PolynomialId::InstructionRa(0));
        assert_eq!(table.len(), 8);
        // Cycle 0: index 1 -> position 1 is hot
        assert_eq!(table[1], Fr::one());
        assert_eq!(table[0], Fr::zero());
        // Cycle 1: index 3 -> position 4+3=7 is hot
        assert_eq!(table[7], Fr::one());
        assert_eq!(table[4], Fr::zero());

        // Sparse representation also stored
        let oh = store
            .get_one_hot(PolynomialId::InstructionRa(0))
            .expect("one-hot representation must be stored");
        assert!(jolt_poly::MultilinearPoly::<Fr>::is_sparse(oh));
    }

    #[test]
    fn none_onehot_all_zeros() {
        let mut store = WitnessStore::<Fr>::new();
        {
            let mut sink = StoreSink::new(&mut store);
            sink.on_polynomial_start(PolynomialId::RamRa(0), 1, PolynomialKind::OneHot { k: 4 });
            sink.on_chunk(PolynomialId::RamRa(0), ChunkData::OneHot(&[None]));
            sink.on_polynomial_end(PolynomialId::RamRa(0));
            sink.finish();
        }
        let table = store.get(PolynomialId::RamRa(0));
        assert_eq!(table.len(), 4);
        assert!(table.iter().all(|&v| v == Fr::zero()));
    }
}
