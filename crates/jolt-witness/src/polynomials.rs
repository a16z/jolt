//! Committed polynomial buffer collection.
//!
//! [`Polynomials`] holds evaluation buffers for all committed polynomials
//! in the Jolt protocol. Cycle data is pushed in via [`push`](Polynomials::push),
//! one-hot decomposition is applied, and the resulting buffers are available
//! via [`get`](Polynomials::get) / [`take`](Polynomials::take) or as a
//! [`BufferProvider`] for the compute runtime.
//!
//! Allocation and population are driven by [`PolynomialDescriptor::witness_slot`]
//! — no hardcoded polynomial identity matching in [`new`](Polynomials::new)
//! or [`push`](Polynomials::push).

use std::collections::BTreeMap;

use jolt_compiler::WitnessSlot;
use jolt_compute::{Buf, BufferProvider, ComputeBackend, DeviceBuffer};
use jolt_field::Field;
use jolt_poly::OneHotPolynomial;

use crate::config::PolynomialConfig;
use crate::cycle_input::CycleInput;
use crate::polynomial_id::PolynomialId;

/// Pending one-hot polynomial accumulation state.
struct OneHotBuffer {
    k: usize,
    indices: Vec<Option<u8>>,
}

/// Collection of committed polynomial evaluation buffers.
///
/// # Lifecycle
///
/// 1. [`new`](Self::new) — allocate with decomposition config.
/// 2. [`push`](Self::push) — feed cycle data (one call or many).
/// 3. [`finish`](Self::finish) — finalize one-hot buffers into dense + sparse form.
/// 4. [`get`](Self::get) / [`take`](Self::take) — read buffers out.
///
/// Batch usage: `new → push(all_cycles) → finish`.
/// Streaming: `new → push(chunk₁) → push(chunk₂) → … → finish`.
pub struct Polynomials<F: Field> {
    config: PolynomialConfig,
    /// Dense polynomial buffers.
    dense: BTreeMap<PolynomialId, Vec<F>>,
    /// One-hot accumulation buffers (not yet expanded).
    one_hot_pending: BTreeMap<PolynomialId, OneHotBuffer>,
    /// Sparse one-hot representations (available after finish).
    one_hot: BTreeMap<PolynomialId, OneHotPolynomial>,
    finished: bool,
}

impl<F: Field> Polynomials<F> {
    /// Creates an empty polynomial buffer collection.
    ///
    /// Allocation strategy is driven by each polynomial's
    /// [`descriptor().witness_slot`](PolynomialId::descriptor):
    /// - [`WitnessSlot::Dense`] → allocate a `Vec<F>`
    /// - [`WitnessSlot::OneHotChunk`] → allocate a `OneHotBuffer`
    /// - `None` → skip (inserted separately or computed on demand)
    pub fn new(config: PolynomialConfig) -> Self {
        let mut dense = BTreeMap::new();
        let mut one_hot_pending = BTreeMap::new();

        for id in config.witness_polynomial_ids() {
            match id.descriptor().witness_slot {
                Some(WitnessSlot::Dense(_)) => {
                    let _ = dense.insert(id, Vec::new());
                }
                Some(WitnessSlot::OneHotChunk { .. }) => {
                    let _ = one_hot_pending.insert(
                        id,
                        OneHotBuffer {
                            k: config.k_chunk,
                            indices: Vec::new(),
                        },
                    );
                }
                None => {}
            }
        }

        Self {
            config,
            dense,
            one_hot_pending,
            one_hot: BTreeMap::new(),
            finished: false,
        }
    }

    /// Pushes a batch of cycle data, appending to all polynomial buffers.
    ///
    /// Routing is driven by [`WitnessSlot`] descriptors — dense slots read
    /// from `CycleInput::dense[i]`, one-hot slots chunk from
    /// `CycleInput::one_hot[source]`.
    ///
    /// Can be called once with the full trace (batch) or repeatedly with
    /// chunks (streaming). Must not be called after [`finish`](Self::finish).
    pub fn push(&mut self, cycles: &[CycleInput]) {
        assert!(!self.finished, "push() called after finish()");

        for (id, buf) in &mut self.dense {
            if let Some(WitnessSlot::Dense(slot)) = id.descriptor().witness_slot {
                buf.extend(cycles.iter().map(|c| F::from_i128(c.dense[slot])));
            }
        }

        for (id, buf) in &mut self.one_hot_pending {
            if let Some(WitnessSlot::OneHotChunk { source, dim }) = id.descriptor().witness_slot {
                buf.indices.extend(
                    cycles
                        .iter()
                        .map(|c| c.one_hot[source].map(|v| self.config.chunk(source, v, dim))),
                );
            }
        }
    }

    /// Finalizes one-hot buffers into dense evaluation tables.
    ///
    /// After this call, all polynomial buffers are available via
    /// [`get`](Self::get) / [`take`](Self::take). No more [`push`](Self::push)
    /// calls are allowed.
    pub fn finish(&mut self) {
        assert!(!self.finished, "finish() called twice");
        self.finished = true;

        for (id, buf) in std::mem::take(&mut self.one_hot_pending) {
            let dense = expand_one_hot::<F>(buf.k, &buf.indices);
            let _ = self.dense.insert(id, dense);
            let _ = self
                .one_hot
                .insert(id, OneHotPolynomial::new(buf.k, buf.indices));
        }
    }

    /// Returns the evaluation buffer for a polynomial.
    ///
    /// # Panics
    ///
    /// Panics if `id` is not present or [`finish`](Self::finish) has not been
    /// called (for one-hot polynomials).
    pub fn get(&self, id: PolynomialId) -> &[F] {
        self.dense
            .get(&id)
            .unwrap_or_else(|| panic!("polynomial {id:?} not found"))
    }

    /// Returns the evaluation buffer if present.
    pub fn try_get(&self, id: PolynomialId) -> Option<&[F]> {
        self.dense.get(&id).map(Vec::as_slice)
    }

    /// Moves the evaluation buffer out.
    ///
    /// After this call, `get(id)` will panic. Also removes the sparse
    /// one-hot representation if present.
    pub fn take(&mut self, id: PolynomialId) -> Vec<F> {
        let _ = self.one_hot.remove(&id);
        self.dense
            .remove(&id)
            .unwrap_or_else(|| panic!("polynomial {id:?} not found"))
    }

    /// Returns the sparse one-hot representation, if this polynomial is one-hot.
    pub fn get_one_hot(&self, id: PolynomialId) -> Option<&OneHotPolynomial> {
        self.one_hot.get(&id)
    }

    /// Release the evaluation buffer for `id`, freeing host memory.
    ///
    /// After this call, `get(id)` will panic. Idempotent — releasing an
    /// already-released polynomial is a no-op.
    pub fn release(&mut self, id: PolynomialId) {
        let _ = self.dense.remove(&id);
        let _ = self.one_hot.remove(&id);
    }

    /// Inserts a pre-computed evaluation buffer (e.g. SpartanWitness, advice).
    pub fn insert(&mut self, id: PolynomialId, buffer: Vec<F>) -> Option<Vec<F>> {
        self.dense.insert(id, buffer)
    }

    pub fn contains(&self, id: PolynomialId) -> bool {
        self.dense.contains_key(&id)
    }

    pub fn len(&self) -> usize {
        self.dense.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }

    /// Returns the decomposition config.
    pub fn config(&self) -> &PolynomialConfig {
        &self.config
    }
}

impl<F: Field> Default for Polynomials<F> {
    fn default() -> Self {
        Self::new(PolynomialConfig::new(4, 128, 1, 1))
    }
}

impl<B: ComputeBackend, F: Field> BufferProvider<B, F> for Polynomials<F> {
    fn load(&mut self, poly_id: PolynomialId, backend: &B) -> Buf<B, F> {
        DeviceBuffer::Field(backend.upload(self.get(poly_id)))
    }

    fn as_slice(&self, poly_id: PolynomialId) -> &[F] {
        self.get(poly_id)
    }

    fn release(&mut self, poly_id: PolynomialId) {
        self.release(poly_id);
    }
}

/// Expands one-hot indices into a flat evaluation buffer.
fn expand_one_hot<F: Field>(k: usize, indices: &[Option<u8>]) -> Vec<F> {
    let n = indices.len();
    let mut buf = vec![F::zero(); n * k];
    for (cycle, &idx) in indices.iter().enumerate() {
        if let Some(i) = idx {
            buf[cycle * k + i as usize] = F::one();
        }
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_compiler::WitnessSlot;
    use jolt_field::Fr;
    use num_traits::{One, Zero};

    fn test_config() -> PolynomialConfig {
        PolynomialConfig::new(4, 8, 4, 4)
    }

    fn test_cycles() -> Vec<CycleInput> {
        vec![
            CycleInput {
                dense: [5, -3],
                one_hot: [Some(0xAB), Some(0xC), Some(0xD)],
            },
            CycleInput {
                dense: [-1, 0],
                one_hot: [Some(0x12), Some(0x3), None],
            },
            CycleInput::PADDING,
            CycleInput::PADDING,
        ]
    }

    #[test]
    fn batch_dense_polynomials() {
        let mut polys = Polynomials::<Fr>::new(test_config());
        polys.push(&test_cycles());
        polys.finish();

        let rd = polys.get(PolynomialId::RdInc);
        assert_eq!(rd.len(), 4);
        assert_eq!(rd[0], Fr::from_i128(5));
        assert_eq!(rd[1], Fr::from_i128(-1));
        assert_eq!(rd[2], Fr::from_i128(0));

        let ram = polys.get(PolynomialId::RamInc);
        assert_eq!(ram[0], Fr::from_i128(-3));
        assert_eq!(ram[1], Fr::from_i128(0));
    }

    #[test]
    fn batch_instruction_ra_chunks() {
        let mut polys = Polynomials::<Fr>::new(test_config());
        polys.push(&test_cycles());
        polys.finish();

        // 8-bit instruction, 4-bit chunks → 2 polynomials
        // lookup_index 0xAB → chunk 0 = 0xA, chunk 1 = 0xB
        let ra0 = polys.get(PolynomialId::InstructionRa(0));
        let k = test_config().k_chunk; // 16
        let c0 = 0;
        let c1 = 1;
        assert_eq!(ra0[c0 * k + 0xA], Fr::one());
        assert_eq!(ra0[c0 * k], Fr::zero());
        assert_eq!(ra0[c1 * k + 0x1], Fr::one());

        let ra1 = polys.get(PolynomialId::InstructionRa(1));
        assert_eq!(ra1[c0 * k + 0xB], Fr::one());
        assert_eq!(ra1[c1 * k + 0x2], Fr::one());
    }

    #[test]
    fn batch_ram_ra_none_for_no_access() {
        let mut polys = Polynomials::<Fr>::new(test_config());
        polys.push(&test_cycles());
        polys.finish();

        let k = test_config().k_chunk;
        let c0 = 0;
        let c1 = 1;
        let c2 = 2;
        let ra0 = polys.get(PolynomialId::RamRa(0));
        assert_eq!(ra0[c0 * k + 0xD], Fr::one());
        assert!(ra0[c1 * k..(c1 + 1) * k].iter().all(|v| *v == Fr::zero()));
        assert!(ra0[c2 * k..(c2 + 1) * k].iter().all(|v| *v == Fr::zero()));
    }

    #[test]
    fn streaming_matches_batch() {
        let cycles = test_cycles();
        let config = test_config();

        let mut batch = Polynomials::<Fr>::new(config.clone());
        batch.push(&cycles);
        batch.finish();

        let mut stream = Polynomials::<Fr>::new(config.clone());
        stream.push(&cycles[..2]);
        stream.push(&cycles[2..]);
        stream.finish();

        for id in config.witness_polynomial_ids() {
            assert_eq!(batch.get(id), stream.get(id), "mismatch for {id:?}");
        }
    }

    #[test]
    fn single_cycle_push() {
        let config = test_config();
        let cycles = test_cycles();

        let mut polys = Polynomials::<Fr>::new(config);
        for c in &cycles {
            polys.push(std::slice::from_ref(c));
        }
        polys.finish();

        let rd = polys.get(PolynomialId::RdInc);
        assert_eq!(rd.len(), 4);
        assert_eq!(rd[0], Fr::from_i128(5));
    }

    #[test]
    fn polynomial_count_matches_config() {
        let config = test_config();
        let expected = config.num_polynomials();

        let mut polys = Polynomials::<Fr>::new(config);
        polys.push(&[CycleInput::PADDING; 4]);
        polys.finish();

        assert_eq!(polys.len(), expected);
    }

    #[test]
    fn insert_and_take() {
        let mut polys = Polynomials::<Fr>::new(test_config());
        let buf = vec![Fr::from_u64(42)];
        let _ = polys.insert(PolynomialId::SpartanWitness, buf.clone());
        assert!(polys.contains(PolynomialId::SpartanWitness));
        let taken = polys.take(PolynomialId::SpartanWitness);
        assert_eq!(taken, buf);
        assert!(!polys.contains(PolynomialId::SpartanWitness));
    }

    #[test]
    #[should_panic(expected = "not found")]
    fn get_missing_panics() {
        let polys = Polynomials::<Fr>::new(test_config());
        let _ = polys.get(PolynomialId::SpartanWitness);
    }

    #[test]
    fn try_get_returns_none() {
        let polys = Polynomials::<Fr>::new(test_config());
        assert!(polys.try_get(PolynomialId::SpartanWitness).is_none());
    }

    #[test]
    fn one_hot_sparse_available_after_finish() {
        let mut polys = Polynomials::<Fr>::new(test_config());
        polys.push(&test_cycles());
        polys.finish();

        assert!(polys.get_one_hot(PolynomialId::InstructionRa(0)).is_some());
        assert!(polys.get_one_hot(PolynomialId::RdInc).is_none());
    }

    #[test]
    fn push_after_finish_panics() {
        let mut polys = Polynomials::<Fr>::new(test_config());
        polys.push(&[CycleInput::PADDING]);
        polys.finish();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            polys.push(&[CycleInput::PADDING]);
        }));
        assert!(result.is_err());
    }

    #[test]
    fn descriptor_drives_allocation() {
        let config = test_config();
        let polys = Polynomials::<Fr>::new(config.clone());

        // Dense witness slots are pre-allocated
        assert!(polys.try_get(PolynomialId::RdInc).is_some());
        assert!(polys.try_get(PolynomialId::RamInc).is_some());

        // Non-trace polynomials are not pre-allocated
        assert!(polys.try_get(PolynomialId::SpartanWitness).is_none());
        assert!(polys.try_get(PolynomialId::TrustedAdvice).is_none());

        // Total pre-allocated = 2 dense (one-hot not in dense until finish)
        assert_eq!(polys.len(), WitnessSlot::NUM_DENSE);
    }
}
