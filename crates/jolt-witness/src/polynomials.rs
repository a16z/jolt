//! Committed polynomial buffer collection.
//!
//! [`Polynomials`] holds evaluation buffers for all committed polynomials
//! in the Jolt protocol. Cycle data is pushed in via [`push`](Polynomials::push),
//! one-hot decomposition is applied, and the resulting buffers are available
//! via [`get`](Polynomials::get) / [`take`](Polynomials::take) or as a
//! [`BufferProvider`] for the compute runtime.

use std::collections::BTreeMap;

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
    /// Dense polynomial buffers (RdInc, RamInc).
    dense: BTreeMap<PolynomialId, Vec<F>>,
    /// One-hot accumulation buffers (not yet expanded).
    one_hot_pending: BTreeMap<PolynomialId, OneHotBuffer>,
    /// Sparse one-hot representations (available after finish).
    one_hot: BTreeMap<PolynomialId, OneHotPolynomial>,
    finished: bool,
}

impl<F: Field> Polynomials<F> {
    /// Creates an empty polynomial buffer collection.
    pub fn new(config: PolynomialConfig) -> Self {
        let mut dense = BTreeMap::new();
        let _ = dense.insert(PolynomialId::RdInc, Vec::new());
        let _ = dense.insert(PolynomialId::RamInc, Vec::new());

        let mut one_hot_pending = BTreeMap::new();
        for i in 0..config.instruction_d {
            let _ = one_hot_pending.insert(
                PolynomialId::InstructionRa(i),
                OneHotBuffer {
                    k: config.k_chunk,
                    indices: Vec::new(),
                },
            );
        }
        for i in 0..config.bytecode_d {
            let _ = one_hot_pending.insert(
                PolynomialId::BytecodeRa(i),
                OneHotBuffer {
                    k: config.k_chunk,
                    indices: Vec::new(),
                },
            );
        }
        for i in 0..config.ram_d {
            let _ = one_hot_pending.insert(
                PolynomialId::RamRa(i),
                OneHotBuffer {
                    k: config.k_chunk,
                    indices: Vec::new(),
                },
            );
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
    /// Can be called once with the full trace (batch) or repeatedly with
    /// chunks (streaming). Must not be called after [`finish`](Self::finish).
    pub fn push(&mut self, cycles: &[CycleInput]) {
        assert!(!self.finished, "push() called after finish()");

        // Dense: RdInc, RamInc
        self.dense
            .get_mut(&PolynomialId::RdInc)
            .unwrap()
            .extend(cycles.iter().map(|c| F::from_i128(c.rd_inc)));
        self.dense
            .get_mut(&PolynomialId::RamInc)
            .unwrap()
            .extend(cycles.iter().map(|c| F::from_i128(c.ram_inc)));

        // One-hot: InstructionRa
        for i in 0..self.config.instruction_d {
            self.one_hot_pending
                .get_mut(&PolynomialId::InstructionRa(i))
                .unwrap()
                .indices
                .extend(
                    cycles
                        .iter()
                        .map(|c| Some(self.config.lookup_index_chunk(c.lookup_index, i))),
                );
        }

        // One-hot: BytecodeRa
        for i in 0..self.config.bytecode_d {
            self.one_hot_pending
                .get_mut(&PolynomialId::BytecodeRa(i))
                .unwrap()
                .indices
                .extend(
                    cycles
                        .iter()
                        .map(|c| Some(self.config.bytecode_pc_chunk(c.pc_index, i))),
                );
        }

        // One-hot: RamRa
        for i in 0..self.config.ram_d {
            self.one_hot_pending
                .get_mut(&PolynomialId::RamRa(i))
                .unwrap()
                .indices
                .extend(
                    cycles
                        .iter()
                        .map(|c| c.ram_address.map(|a| self.config.ram_address_chunk(a, i))),
                );
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

impl<B: ComputeBackend, F: Field> BufferProvider<PolynomialId, B, F> for Polynomials<F> {
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
    use jolt_field::Fr;
    use num_traits::{One, Zero};

    fn test_config() -> PolynomialConfig {
        PolynomialConfig::new(4, 8, 4, 4)
    }

    fn test_cycles() -> Vec<CycleInput> {
        vec![
            CycleInput {
                rd_inc: 5,
                ram_inc: -3,
                lookup_index: 0xAB,
                pc_index: 0xC,
                ram_address: Some(0xD),
            },
            CycleInput {
                rd_inc: -1,
                ram_inc: 0,
                lookup_index: 0x12,
                pc_index: 0x3,
                ram_address: None,
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
        let c0 = 0; // cycle 0 offset
        let c1 = 1; // cycle 1 offset
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
        // Cycle 0: ram_address = Some(0xD) → chunk 0 = 0xD
        assert_eq!(ra0[c0 * k + 0xD], Fr::one());
        // Cycle 1: ram_address = None → all zeros
        assert!(ra0[c1 * k..(c1 + 1) * k].iter().all(|v| *v == Fr::zero()));
        // Padding: all zeros
        assert!(ra0[c2 * k..(c2 + 1) * k].iter().all(|v| *v == Fr::zero()));
    }

    #[test]
    fn streaming_matches_batch() {
        let cycles = test_cycles();
        let config = test_config();

        // Batch
        let mut batch = Polynomials::<Fr>::new(config.clone());
        batch.push(&cycles);
        batch.finish();

        // Streaming: 2 chunks of 2
        let mut stream = Polynomials::<Fr>::new(config.clone());
        stream.push(&cycles[..2]);
        stream.push(&cycles[2..]);
        stream.finish();

        // Dense polynomials must match
        for id in [PolynomialId::RdInc, PolynomialId::RamInc] {
            assert_eq!(batch.get(id), stream.get(id), "mismatch for {id:?}");
        }

        // One-hot polynomials must match
        for i in 0..config.instruction_d {
            let id = PolynomialId::InstructionRa(i);
            assert_eq!(batch.get(id), stream.get(id), "mismatch for {id:?}");
        }
        for i in 0..config.bytecode_d {
            let id = PolynomialId::BytecodeRa(i);
            assert_eq!(batch.get(id), stream.get(id), "mismatch for {id:?}");
        }
        for i in 0..config.ram_d {
            let id = PolynomialId::RamRa(i);
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
    #[should_panic(expected = "push() called after finish()")]
    fn push_after_finish_panics() {
        let mut polys = Polynomials::<Fr>::new(test_config());
        polys.push(&[CycleInput::PADDING]);
        polys.finish();
        polys.push(&[CycleInput::PADDING]);
    }
}
