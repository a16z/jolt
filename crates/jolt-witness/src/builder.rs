//! Core witness generation algorithm.
//!
//! [`WitnessBuilder`] processes [`CycleData`] rows and emits committed
//! polynomial evaluation tables to a [`WitnessSink`]. It supports both
//! batch mode (entire trace at once) and streaming mode (chunk by chunk).
//!
//! The builder is **stateless** — all state for streaming lives in the
//! [`StreamingSession`] returned by [`WitnessBuilder::streaming`].

use std::marker::PhantomData;

use jolt_field::Field;
use jolt_ir::PolynomialId;

use crate::config::WitnessConfig;
use crate::cycle::CycleData;
use crate::sink::{ChunkData, PolynomialKind, WitnessSink};

/// Generates committed witness polynomials from pre-extracted cycle data.
///
/// # Two modes of operation
///
/// 1. **Batch** ([`build`](Self::build)): Process the entire trace at once.
///    Emits one chunk per polynomial containing all evaluations.
///    Suitable for non-streaming (AddressMajor) PCS layouts.
///
/// 2. **Streaming** ([`streaming`](Self::streaming)): Process the trace in
///    row-sized chunks. Emits interleaved chunks for all polynomials.
///    Suitable for streaming (CycleMajor) PCS layouts like Dory.
///
/// # Polynomials emitted
///
/// | Tag | Kind | Description |
/// |-----|------|-------------|
/// | `RdInc` | Dense | Register write increments (i128 -> F) |
/// | `RamInc` | Dense | RAM write increments (i128 -> F) |
/// | `InstructionRa(0..d)` | OneHot | Instruction lookup index chunks |
/// | `BytecodeRa(0..d)` | OneHot | Bytecode PC chunks |
/// | `RamRa(0..d)` | OneHot | RAM address chunks |
pub struct WitnessBuilder {
    config: WitnessConfig,
}

impl WitnessBuilder {
    pub fn new(config: WitnessConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &WitnessConfig {
        &self.config
    }

    /// Process an entire trace at once, emitting one chunk per polynomial.
    ///
    /// The trace must already be padded to the required length.
    /// Polynomials are emitted sequentially: first RdInc, then RamInc,
    /// then all InstructionRa chunks, BytecodeRa chunks, and RamRa chunks.
    pub fn build<F: Field>(&self, trace: &[CycleData], sink: &mut impl WitnessSink<F>) {
        let len = trace.len();

        emit_rd_inc(trace, len, sink);
        emit_ram_inc(trace, len, sink);

        for i in 0..self.config.instruction_d {
            emit_instruction_ra(&self.config, trace, i, len, sink);
        }
        for i in 0..self.config.bytecode_d {
            emit_bytecode_ra(&self.config, trace, i, len, sink);
        }
        for i in 0..self.config.ram_d {
            emit_ram_ra(&self.config, trace, i, len, sink);
        }

        sink.finish();
    }

    /// Begin streaming witness generation.
    ///
    /// Returns a [`StreamingSession`] that processes chunks of the trace
    /// incrementally. All polynomial starts are signaled immediately.
    ///
    /// `total_len` is the padded trace length (number of cycles across
    /// all chunks). Each call to [`StreamingSession::process_chunk`]
    /// emits interleaved chunks for all polynomials.
    pub fn streaming<'a, F: Field, S: WitnessSink<F>>(
        &'a self,
        total_len: usize,
        sink: &'a mut S,
    ) -> StreamingSession<'a, F, S> {
        StreamingSession::new(&self.config, total_len, sink)
    }
}

// Free functions to avoid generic parameter issues on `self` methods.

fn emit_rd_inc<F: Field>(trace: &[CycleData], len: usize, sink: &mut impl WitnessSink<F>) {
    sink.on_polynomial_start(PolynomialId::RdInc, len, PolynomialKind::Dense);
    let values: Vec<F> = trace.iter().map(|c| F::from_i128(c.rd_inc)).collect();
    sink.on_chunk(PolynomialId::RdInc, ChunkData::Dense(&values));
    sink.on_polynomial_end(PolynomialId::RdInc);
}

fn emit_ram_inc<F: Field>(trace: &[CycleData], len: usize, sink: &mut impl WitnessSink<F>) {
    sink.on_polynomial_start(PolynomialId::RamInc, len, PolynomialKind::Dense);
    let values: Vec<F> = trace.iter().map(|c| F::from_i128(c.ram_inc)).collect();
    sink.on_chunk(PolynomialId::RamInc, ChunkData::Dense(&values));
    sink.on_polynomial_end(PolynomialId::RamInc);
}

fn emit_instruction_ra<F: Field>(
    config: &WitnessConfig,
    trace: &[CycleData],
    idx: usize,
    len: usize,
    sink: &mut impl WitnessSink<F>,
) {
    let id = PolynomialId::InstructionRa(idx);
    let k = config.k_chunk;
    sink.on_polynomial_start(id, len, PolynomialKind::OneHot { k });
    let indices: Vec<Option<u8>> = trace
        .iter()
        .map(|c| Some(config.lookup_index_chunk(c.lookup_index, idx)))
        .collect();
    sink.on_chunk(id, ChunkData::OneHot(&indices));
    sink.on_polynomial_end(id);
}

fn emit_bytecode_ra<F: Field>(
    config: &WitnessConfig,
    trace: &[CycleData],
    idx: usize,
    len: usize,
    sink: &mut impl WitnessSink<F>,
) {
    let id = PolynomialId::BytecodeRa(idx);
    let k = config.k_chunk;
    sink.on_polynomial_start(id, len, PolynomialKind::OneHot { k });
    let indices: Vec<Option<u8>> = trace
        .iter()
        .map(|c| Some(config.bytecode_pc_chunk(c.pc_index, idx)))
        .collect();
    sink.on_chunk(id, ChunkData::OneHot(&indices));
    sink.on_polynomial_end(id);
}

fn emit_ram_ra<F: Field>(
    config: &WitnessConfig,
    trace: &[CycleData],
    idx: usize,
    len: usize,
    sink: &mut impl WitnessSink<F>,
) {
    let id = PolynomialId::RamRa(idx);
    let k = config.k_chunk;
    sink.on_polynomial_start(id, len, PolynomialKind::OneHot { k });
    let indices: Vec<Option<u8>> = trace
        .iter()
        .map(|c| {
            c.ram_address
                .map(|addr| config.ram_address_chunk(addr, idx))
        })
        .collect();
    sink.on_chunk(id, ChunkData::OneHot(&indices));
    sink.on_polynomial_end(id);
}

/// Emit all polynomial chunks for a single chunk of cycles (streaming helper).
fn emit_chunk_all<F: Field>(
    config: &WitnessConfig,
    chunk: &[CycleData],
    sink: &mut impl WitnessSink<F>,
) {
    // Dense polynomials
    let rd_inc: Vec<F> = chunk.iter().map(|c| F::from_i128(c.rd_inc)).collect();
    sink.on_chunk(PolynomialId::RdInc, ChunkData::Dense(&rd_inc));

    let ram_inc: Vec<F> = chunk.iter().map(|c| F::from_i128(c.ram_inc)).collect();
    sink.on_chunk(PolynomialId::RamInc, ChunkData::Dense(&ram_inc));

    // One-hot: instruction RA
    for i in 0..config.instruction_d {
        let indices: Vec<Option<u8>> = chunk
            .iter()
            .map(|c| Some(config.lookup_index_chunk(c.lookup_index, i)))
            .collect();
        sink.on_chunk(PolynomialId::InstructionRa(i), ChunkData::OneHot(&indices));
    }

    // One-hot: bytecode RA
    for i in 0..config.bytecode_d {
        let indices: Vec<Option<u8>> = chunk
            .iter()
            .map(|c| Some(config.bytecode_pc_chunk(c.pc_index, i)))
            .collect();
        sink.on_chunk(PolynomialId::BytecodeRa(i), ChunkData::OneHot(&indices));
    }

    // One-hot: RAM RA
    for i in 0..config.ram_d {
        let indices: Vec<Option<u8>> = chunk
            .iter()
            .map(|c| c.ram_address.map(|addr| config.ram_address_chunk(addr, i)))
            .collect();
        sink.on_chunk(PolynomialId::RamRa(i), ChunkData::OneHot(&indices));
    }
}

/// Streaming witness generation session.
///
/// Processes the trace in chunks, emitting interleaved polynomial data
/// to the sink. Created by [`WitnessBuilder::streaming`].
///
/// # Lifecycle
///
/// 1. Construction signals `on_polynomial_start` for all polynomials.
/// 2. Each [`process_chunk`](Self::process_chunk) call emits one chunk
///    per polynomial (interleaved).
/// 3. [`finish`](Self::finish) signals `on_polynomial_end` for all
///    polynomials and calls `sink.finish()`.
pub struct StreamingSession<'a, F: Field, S: WitnessSink<F>> {
    config: &'a WitnessConfig,
    sink: &'a mut S,
    _marker: PhantomData<F>,
}

impl<'a, F: Field, S: WitnessSink<F>> StreamingSession<'a, F, S> {
    fn new(config: &'a WitnessConfig, total_len: usize, sink: &'a mut S) -> Self {
        let k = config.k_chunk;

        sink.on_polynomial_start(PolynomialId::RdInc, total_len, PolynomialKind::Dense);
        sink.on_polynomial_start(PolynomialId::RamInc, total_len, PolynomialKind::Dense);

        for i in 0..config.instruction_d {
            sink.on_polynomial_start(
                PolynomialId::InstructionRa(i),
                total_len,
                PolynomialKind::OneHot { k },
            );
        }
        for i in 0..config.bytecode_d {
            sink.on_polynomial_start(
                PolynomialId::BytecodeRa(i),
                total_len,
                PolynomialKind::OneHot { k },
            );
        }
        for i in 0..config.ram_d {
            sink.on_polynomial_start(
                PolynomialId::RamRa(i),
                total_len,
                PolynomialKind::OneHot { k },
            );
        }

        Self {
            config,
            sink,
            _marker: PhantomData,
        }
    }

    /// Process one chunk of cycles, emitting data for all polynomials.
    ///
    /// Chunks must be provided in trace order and their lengths must sum
    /// to the `total_len` passed to [`WitnessBuilder::streaming`].
    pub fn process_chunk(&mut self, chunk: &[CycleData]) {
        emit_chunk_all(self.config, chunk, self.sink);
    }

    /// Finish streaming: signal polynomial end for all polynomials and finalize.
    pub fn finish(self) {
        // Move into self so Drop doesn't double-finish
        let Self { config, sink, .. } = self;

        sink.on_polynomial_end(PolynomialId::RdInc);
        sink.on_polynomial_end(PolynomialId::RamInc);
        for i in 0..config.instruction_d {
            sink.on_polynomial_end(PolynomialId::InstructionRa(i));
        }
        for i in 0..config.bytecode_d {
            sink.on_polynomial_end(PolynomialId::BytecodeRa(i));
        }
        for i in 0..config.ram_d {
            sink.on_polynomial_end(PolynomialId::RamRa(i));
        }
        sink.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::CollectingSink;
    use jolt_field::Fr;

    fn test_config() -> WitnessConfig {
        // Small config: 4-bit chunks, 8-bit instruction, 4-bit bytecode, 4-bit RAM
        WitnessConfig::new(4, 8, 4, 4)
    }

    fn test_trace() -> Vec<CycleData> {
        vec![
            CycleData {
                rd_inc: 5,
                ram_inc: -3,
                lookup_index: 0xAB,
                pc_index: 0xC,
                ram_address: Some(0xD),
            },
            CycleData {
                rd_inc: -1,
                ram_inc: 0,
                lookup_index: 0x12,
                pc_index: 0x3,
                ram_address: None,
            },
            CycleData::PADDING,
            CycleData::PADDING,
        ]
    }

    #[test]
    fn batch_dense_polynomials() {
        let config = test_config();
        let builder = WitnessBuilder::new(config);
        let trace = test_trace();
        let mut sink = CollectingSink::<Fr>::new();

        builder.build(&trace, &mut sink);

        let rd = sink.dense_table(PolynomialId::RdInc).unwrap();
        assert_eq!(rd.len(), 4);
        assert_eq!(rd[0], Fr::from_i128(5));
        assert_eq!(rd[1], Fr::from_i128(-1));
        assert_eq!(rd[2], Fr::from_i128(0));

        let ram = sink.dense_table(PolynomialId::RamInc).unwrap();
        assert_eq!(ram[0], Fr::from_i128(-3));
        assert_eq!(ram[1], Fr::from_i128(0));
    }

    #[test]
    fn batch_instruction_ra_chunks() {
        let config = test_config();
        let builder = WitnessBuilder::new(config);
        let trace = test_trace();
        let mut sink = CollectingSink::<Fr>::new();

        builder.build(&trace, &mut sink);

        // 8-bit instruction, 4-bit chunks -> 2 polynomials
        // lookup_index 0xAB -> chunk 0 = 0xA, chunk 1 = 0xB
        let (k, idx0) = sink.onehot_table(PolynomialId::InstructionRa(0)).unwrap();
        assert_eq!(k, 16);
        assert_eq!(idx0[0], Some(0xA));
        assert_eq!(idx0[1], Some(0x1)); // 0x12 -> chunk 0 = 0x1

        let (_, idx1) = sink.onehot_table(PolynomialId::InstructionRa(1)).unwrap();
        assert_eq!(idx1[0], Some(0xB));
        assert_eq!(idx1[1], Some(0x2)); // 0x12 -> chunk 1 = 0x2
    }

    #[test]
    fn batch_ram_ra_none_for_no_access() {
        let config = test_config();
        let builder = WitnessBuilder::new(config);
        let trace = test_trace();
        let mut sink = CollectingSink::<Fr>::new();

        builder.build(&trace, &mut sink);

        let (_, indices) = sink.onehot_table(PolynomialId::RamRa(0)).unwrap();
        // Cycle 0: ram_address = Some(0xD) -> chunk 0 = 0xD (4-bit, d=1, shift=0)
        assert_eq!(indices[0], Some(0xD));
        // Cycle 1: ram_address = None
        assert_eq!(indices[1], None);
        // Padding: ram_address = None
        assert_eq!(indices[2], None);
    }

    #[test]
    fn streaming_matches_batch() {
        let config = test_config();
        let builder = WitnessBuilder::new(config);
        let trace = test_trace();

        // Batch
        let mut batch_sink = CollectingSink::<Fr>::new();
        builder.build(&trace, &mut batch_sink);

        // Streaming: 2 chunks of 2
        let mut stream_sink = CollectingSink::<Fr>::new();
        let mut session = builder.streaming::<Fr, _>(trace.len(), &mut stream_sink);
        session.process_chunk(&trace[..2]);
        session.process_chunk(&trace[2..]);
        session.finish();

        // Compare all dense polynomials
        for id in [PolynomialId::RdInc, PolynomialId::RamInc] {
            let batch = batch_sink.dense_table(id).unwrap();
            let stream = stream_sink.dense_table(id).unwrap();
            assert_eq!(batch, stream, "mismatch for dense poly {id:?}");
        }

        // Compare all one-hot polynomials
        let cfg = builder.config();
        for i in 0..cfg.instruction_d {
            let id = PolynomialId::InstructionRa(i);
            let (bk, bi) = batch_sink.onehot_table(id).unwrap();
            let (sk, si) = stream_sink.onehot_table(id).unwrap();
            assert_eq!(bk, sk);
            assert_eq!(bi, si, "mismatch for InstructionRa({i})");
        }
        for i in 0..cfg.bytecode_d {
            let id = PolynomialId::BytecodeRa(i);
            let (_, bi) = batch_sink.onehot_table(id).unwrap();
            let (_, si) = stream_sink.onehot_table(id).unwrap();
            assert_eq!(bi, si, "mismatch for BytecodeRa({i})");
        }
        for i in 0..cfg.ram_d {
            let id = PolynomialId::RamRa(i);
            let (_, bi) = batch_sink.onehot_table(id).unwrap();
            let (_, si) = stream_sink.onehot_table(id).unwrap();
            assert_eq!(bi, si, "mismatch for RamRa({i})");
        }
    }

    #[test]
    fn streaming_single_cycle_chunks() {
        let config = test_config();
        let builder = WitnessBuilder::new(config);
        let trace = test_trace();

        let mut sink = CollectingSink::<Fr>::new();
        let mut session = builder.streaming::<Fr, _>(trace.len(), &mut sink);
        for cycle in &trace {
            session.process_chunk(std::slice::from_ref(cycle));
        }
        session.finish();

        let rd = sink.dense_table(PolynomialId::RdInc).unwrap();
        assert_eq!(rd.len(), 4);
        assert_eq!(rd[0], Fr::from_i128(5));
    }

    #[test]
    fn polynomial_count_matches_config() {
        let config = test_config();
        let num_polys = config.num_committed_polynomials();
        let builder = WitnessBuilder::new(config);
        let trace = vec![CycleData::PADDING; 4];
        let mut sink = CollectingSink::<Fr>::new();

        builder.build(&trace, &mut sink);

        let polys = sink.into_polys();
        assert_eq!(polys.len(), num_polys);
    }
}
