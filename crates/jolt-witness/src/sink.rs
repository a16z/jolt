//! Witness sink abstraction.
//!
//! [`WitnessSink`] defines the push-based output interface for witness
//! generation. As the witness builder processes trace rows, it emits
//! polynomial evaluation table chunks to the sink.
//!
//! The sink decides what to do with each chunk:
//! - **Commit:** Call `StreamingCommitment::feed()` to build commitments
//! - **Store:** Append to a `WitnessStore` for later use by proving stages
//! - **Both:** The typical case in jolt-zkvm's `CommitAndStoreSink`
//!
//! This decouples witness generation from PCS — `jolt-witness` never
//! touches commitment schemes.

use crate::PolynomialId;
use jolt_field::Field;

/// Describes the representation of a polynomial's evaluation data.
///
/// Communicated at polynomial start so the sink can allocate appropriately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolynomialKind {
    /// Dense evaluation table: `total_len` field elements.
    Dense,
    /// One-hot encoded address polynomial.
    ///
    /// `total_len` cycles, each with a single nonzero index in `[0, k)`.
    /// The PCS (e.g. Dory) has a specialized commitment path for this representation.
    OneHot {
        /// Address chunk size (e.g. 16 or 256).
        k: usize,
    },
}

/// A chunk of polynomial evaluation data, either dense or one-hot sparse.
pub enum ChunkData<'a, F> {
    /// Dense field element evaluations.
    Dense(&'a [F]),
    /// Sparse one-hot indices: `Some(i)` means position `i` is 1 (rest 0),
    /// `None` means the entire one-hot vector is zero.
    OneHot(&'a [Option<u8>]),
}

/// Push-based callback for streaming witness polynomial chunks.
///
/// As the witness builder processes trace rows, it yields chunks of
/// evaluation data to the sink. The sink implementation decides what
/// to do with each chunk (commit, store, or both).
///
/// # Chunk ordering
///
/// For a given `poly_id`, chunks arrive in order (chunk 0, chunk 1, ...).
/// Different polynomials may be interleaved — the sink should use `poly_id`
/// to route each chunk to the correct destination.
///
/// # Lifecycle
///
/// 1. Builder calls [`on_polynomial_start`](Self::on_polynomial_start) when a new polynomial begins.
/// 2. Builder calls [`on_chunk`](Self::on_chunk) for each chunk of that polynomial.
/// 3. Builder calls [`on_polynomial_end`](Self::on_polynomial_end) when all chunks have been emitted.
/// 4. After all polynomials: builder calls [`finish`](Self::finish).
///
/// This lifecycle allows the sink to perform per-polynomial setup (e.g.,
/// `StreamingCommitment::begin()`) and finalization (e.g., `finish()`).
pub trait WitnessSink<F: Field> {
    /// Called when a new polynomial begins emission.
    ///
    /// `poly_id` identifies the polynomial via [`PolynomialId`].
    /// `total_len` is the number of cycles (for one-hot) or field elements
    /// (for dense) that will be emitted across all chunks.
    /// `kind` describes the polynomial's representation.
    fn on_polynomial_start(
        &mut self,
        poly_id: PolynomialId,
        total_len: usize,
        kind: PolynomialKind,
    );

    /// Called with a chunk of evaluation data for a polynomial.
    ///
    /// The chunk variant (Dense or OneHot) matches the `kind` passed to
    /// [`on_polynomial_start`](Self::on_polynomial_start).
    fn on_chunk(&mut self, poly_id: PolynomialId, data: ChunkData<'_, F>);

    /// Called when all chunks for a polynomial have been emitted.
    fn on_polynomial_end(&mut self, poly_id: PolynomialId);

    /// Called after all polynomials have been emitted.
    fn finish(&mut self);
}

/// Collected polynomial data, either dense field elements or sparse one-hot indices.
#[cfg(any(test, feature = "test-utils"))]
#[derive(Debug)]
pub enum CollectedPoly<F: Field> {
    Dense(Vec<F>),
    OneHot { k: usize, indices: Vec<Option<u8>> },
}

/// A sink that collects evaluation tables for testing.
///
/// Stores complete polynomial data keyed by [`PolynomialId`], preserving the
/// dense vs one-hot representation.
#[cfg(any(test, feature = "test-utils"))]
pub struct CollectingSink<F: Field> {
    polys: std::collections::BTreeMap<PolynomialId, CollectedPoly<F>>,
}

#[cfg(any(test, feature = "test-utils"))]
impl<F: Field> CollectingSink<F> {
    pub fn new() -> Self {
        Self {
            polys: std::collections::BTreeMap::new(),
        }
    }

    /// Returns the dense evaluation table for a polynomial, or `None`.
    ///
    /// Panics if the polynomial was emitted as one-hot.
    pub fn dense_table(&self, poly_id: PolynomialId) -> Option<&[F]> {
        self.polys.get(&poly_id).map(|p| match p {
            CollectedPoly::Dense(v) => v.as_slice(),
            CollectedPoly::OneHot { .. } => panic!("poly {poly_id:?} is one-hot, not dense"),
        })
    }

    /// Returns the one-hot indices for a polynomial, or `None`.
    ///
    /// Panics if the polynomial was emitted as dense.
    pub fn onehot_table(&self, poly_id: PolynomialId) -> Option<(usize, &[Option<u8>])> {
        self.polys.get(&poly_id).map(|p| match p {
            CollectedPoly::OneHot { k, indices } => (*k, indices.as_slice()),
            CollectedPoly::Dense(_) => panic!("poly {poly_id:?} is dense, not one-hot"),
        })
    }

    /// Consumes the sink and returns all collected polynomial data.
    pub fn into_polys(self) -> std::collections::BTreeMap<PolynomialId, CollectedPoly<F>> {
        self.polys
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl<F: Field> Default for CollectingSink<F> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl<F: Field> WitnessSink<F> for CollectingSink<F> {
    fn on_polynomial_start(
        &mut self,
        poly_id: PolynomialId,
        total_len: usize,
        kind: PolynomialKind,
    ) {
        let _ = self.polys.entry(poly_id).or_insert_with(|| match kind {
            PolynomialKind::Dense => CollectedPoly::Dense(Vec::with_capacity(total_len)),
            PolynomialKind::OneHot { k } => CollectedPoly::OneHot {
                k,
                indices: Vec::with_capacity(total_len),
            },
        });
    }

    fn on_chunk(&mut self, poly_id: PolynomialId, data: ChunkData<'_, F>) {
        match (self.polys.get_mut(&poly_id), data) {
            (Some(CollectedPoly::Dense(v)), ChunkData::Dense(chunk)) => {
                v.extend_from_slice(chunk);
            }
            (Some(CollectedPoly::OneHot { indices, .. }), ChunkData::OneHot(chunk)) => {
                indices.extend_from_slice(chunk);
            }
            (None, ChunkData::Dense(chunk)) => {
                let _ = self
                    .polys
                    .insert(poly_id, CollectedPoly::Dense(chunk.to_vec()));
            }
            (None, ChunkData::OneHot(chunk)) => {
                let _ = self.polys.insert(
                    poly_id,
                    CollectedPoly::OneHot {
                        k: 0,
                        indices: chunk.to_vec(),
                    },
                );
            }
            _ => panic!("chunk type mismatch for poly {poly_id:?}"),
        }
    }

    fn on_polynomial_end(&mut self, _poly_id: PolynomialId) {}

    fn finish(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;

    #[test]
    fn collecting_sink_dense_chunks() {
        let mut sink = CollectingSink::<Fr>::new();
        let id = PolynomialId::RamInc;

        sink.on_polynomial_start(id, 4, PolynomialKind::Dense);
        sink.on_chunk(id, ChunkData::Dense(&[Fr::from_u64(1), Fr::from_u64(2)]));
        sink.on_chunk(id, ChunkData::Dense(&[Fr::from_u64(3), Fr::from_u64(4)]));
        sink.on_polynomial_end(id);
        sink.finish();

        let table = sink.dense_table(id).unwrap();
        assert_eq!(table.len(), 4);
        assert_eq!(table[0], Fr::from_u64(1));
        assert_eq!(table[3], Fr::from_u64(4));
    }

    #[test]
    fn collecting_sink_onehot_chunks() {
        let mut sink = CollectingSink::<Fr>::new();
        let id = PolynomialId::RamReadValue;

        sink.on_polynomial_start(id, 4, PolynomialKind::OneHot { k: 16 });
        sink.on_chunk(id, ChunkData::OneHot(&[Some(3), Some(15)]));
        sink.on_chunk(id, ChunkData::OneHot(&[None, Some(0)]));
        sink.on_polynomial_end(id);
        sink.finish();

        let (k, indices) = sink.onehot_table(id).unwrap();
        assert_eq!(k, 16);
        assert_eq!(indices.len(), 4);
        assert_eq!(indices[0], Some(3));
        assert_eq!(indices[2], None);
    }

    #[test]
    fn collecting_sink_mixed_polynomials() {
        let mut sink = CollectingSink::<Fr>::new();
        let dense_id = PolynomialId::RamInc;
        let onehot_id = PolynomialId::RamReadValue;

        sink.on_polynomial_start(dense_id, 2, PolynomialKind::Dense);
        sink.on_chunk(
            dense_id,
            ChunkData::Dense(&[Fr::from_u64(10), Fr::from_u64(20)]),
        );
        sink.on_polynomial_end(dense_id);

        sink.on_polynomial_start(onehot_id, 3, PolynomialKind::OneHot { k: 256 });
        sink.on_chunk(onehot_id, ChunkData::OneHot(&[Some(1), Some(2), Some(3)]));
        sink.on_polynomial_end(onehot_id);

        sink.finish();

        assert_eq!(sink.dense_table(dense_id).unwrap().len(), 2);
        let (k, indices) = sink.onehot_table(onehot_id).unwrap();
        assert_eq!(k, 256);
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn collecting_sink_into_polys() {
        let mut sink = CollectingSink::<Fr>::new();
        let id = PolynomialId::LookupOutput;

        sink.on_polynomial_start(id, 1, PolynomialKind::Dense);
        sink.on_chunk(id, ChunkData::Dense(&[Fr::from_u64(7)]));
        sink.on_polynomial_end(id);
        sink.finish();

        let polys = sink.into_polys();
        assert_eq!(polys.len(), 1);
        assert!(polys.contains_key(&id));
    }
}
