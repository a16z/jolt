//! The byte link on Metal: the messages of [`crate::byte_link::reference`], computed on the GPU
//! (protocol: [`crate::byte_link`]).
//!
//! Benches (ignored tests, through the shared-box GPU gate):
//! `gpu-window.sh cargo nextest run -p jolt-kernels --features metal,akita-byte-link --lib
//! --run-ignored ignored-only --no-capture -E 'test(/byte_link::tests::bench/)'` with
//! `LINK_LOG` (default 26), `LINK_REPS`, `LINK_COOL_SECONDS`, `LINK_FULL=1` (no zero tail),
//! `LINK_HOT=1` (256 tuples per pack) and `LINK_CHECK=1`; it prints per-phase wall and GPU
//! seconds, the process footprint and the arena size.

mod gkr;
mod gpu;
mod reduce;
mod sort;
#[cfg(test)]
mod tests;

use std::marker::PhantomData;

use jolt_claims::protocols::jolt::lattice::byte_link::{
    ByteLinkBatch, ByteLinkInputs, HistogramGroup,
};
use jolt_field::Prime128OffsetA7F7 as F;
use jolt_sumcheck::SumcheckError;
use jolt_verifier::stages::byte_link::{ByteLinkCompression, ByteLinkOpenings};
use metal::{Buffer, MTLResourceOptions};

use super::{Fp128, MetalError, SolinasMetal};
use crate::byte_link::reference::{ByteTrace, Denominators, ReferenceByteLink};
use crate::byte_link::{
    ByteLinkKernel, ByteLinkMessage, ByteLinkRun, ByteLinkTranscript, HistogramTables,
};
use crate::metal::MetalBackend;
use crate::KernelError;
use gkr::{TailForm, TraceLeaves, Trees};
use gpu::Gpu;

pub(super) const SOURCE: &str = include_str!("shader.metal");

/// Columns of `Q`.
const SLOTS: usize = 32;
const PACKS: usize = 7;
/// The RAM pack: two address bytes and the activity bit (`LINK_RAM`).
const RAM: usize = 6;
/// `Q` slots of each pack (`LinkColumns`); the RAM pack's last slot is the activity bit.
const PACK_SLOTS: [[usize; 3]; PACKS] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [9, 10, 11],
    [12, 13, 14],
    [15, 25, 26],
    [27, 28, 29],
];
/// `Q` slots of the fused increment `F = Σ_{i<8} 256^i D_{16+i} + 2^64 D_24`.
const INCREMENT_SLOTS: [usize; 9] = [16, 17, 18, 19, 20, 21, 22, 23, 24];
/// Index variables of a triple table and of the RAM table.
const TRIPLE_BITS: u32 = 24;
const RAM_BITS: u32 = 17;
/// Sumcheck arrays of at most `2^HOST_LOG` records per tree finish on the host.
const HOST_LOG: u32 = 10;
/// Lowest trace-tree level kept in memory; the rounds reading lower levels rebuild them from `Q`.
const STORED_HEIGHT: u32 = 3;
const MIN_LOG_ROWS: u32 = 16;
const MAX_LOG_ROWS: u32 = 30;

const fn table_bits(pack: usize) -> u32 {
    if pack == RAM {
        RAM_BITS
    } else {
        TRIPLE_BITS
    }
}

/// The byte trace on the device: [`SLOTS`] signed-byte columns of `2^log_rows` rows, column `c`
/// of row `t` at byte `c · 2^log_rows + t`. Columns 30 and 31 and every row at or above
/// `active_rows` must be zero: the prover reads none of those rows and leaves the zero-slot term
/// out of the `Q` reduction, so a nonzero byte there makes the proof reject.
pub struct ByteLinkSource<'a> {
    pub bytes: &'a Buffer,
    pub log_rows: u32,
    pub active_rows: usize,
}

/// The eq-weighted tuple histograms on the device, canonical `Fp128` cells: `W` of pack `p` at
/// cell `p << 24` onwards, `2^24` cells per triple, `2^17` for the RAM pack. The `W` commitments
/// and stage 8 read them here.
pub struct ByteLinkHistograms {
    w: Buffer,
}

impl ByteLinkHistograms {
    pub fn buffer(&self) -> &Buffer {
        &self.w
    }

    /// Byte offset in [`Self::buffer`] of `group`'s first table; the group's tables follow back to
    /// back.
    pub fn group_offset(group: HistogramGroup) -> usize {
        (group.packs().start << TRIPLE_BITS) * size_of::<Fp128>()
    }
}

/// The prover's view of a source: the explicit prefix covers whole `2^(n − HOST_LOG)`-row blocks
/// so every GPU level holds whole host records.
#[derive(Clone, Copy, Debug)]
struct Shape {
    log_n: u32,
    explicit: usize,
}

impl Shape {
    fn new(source: &ByteLinkSource<'_>) -> Result<Self, MetalError> {
        let log_n = source.log_rows;
        let invalid = MetalError::ByteLinkShape {
            log_rows: log_n,
            active_rows: source.active_rows,
        };
        if !(MIN_LOG_ROWS..=MAX_LOG_ROWS).contains(&log_n)
            || source.active_rows > 1 << log_n
            || source.bytes.length() < (SLOTS as u64) << log_n
        {
            return Err(invalid);
        }
        let block = 1usize << (log_n - HOST_LOG);
        Ok(Self {
            log_n,
            explicit: source.active_rows.next_multiple_of(block).max(block),
        })
    }

    /// The stage-6b cycle point, LSB-first.
    fn cycle_point(self, inputs: &ByteLinkInputs<F>) -> Result<Vec<F>, MetalError> {
        let point = inputs.cycle_point();
        if point.len() == self.log_n as usize {
            Ok(point.iter().rev().copied().collect())
        } else {
            Err(MetalError::ByteLinkPoint {
                log_rows: self.log_n,
                len: point.len(),
            })
        }
    }

    const fn n(self) -> usize {
        1 << self.log_n
    }
}

/// The byte-link prover. One per proof; its arena is sized for the largest phase and every
/// phase's device buffers come from it.
pub struct ByteLinkProver {
    gpu: Gpu,
}

impl ByteLinkProver {
    pub fn new(metal: &SolinasMetal) -> Result<Self, MetalError> {
        Ok(Self {
            gpu: Gpu::new(metal)?,
        })
    }

    /// The challenge-free key sort of every pack and `W` at the stage-6b cycle point.
    pub fn histograms(
        &mut self,
        source: &ByteLinkSource<'_>,
        inputs: &ByteLinkInputs<F>,
    ) -> Result<ByteLinkHistograms, MetalError> {
        let shape = Shape::new(source)?;
        let r_t = shape.cycle_point(inputs)?;
        self.gpu.reserve_arena(self.arena_bytes(shape))?;
        self.gpu.phase("histograms");
        let w = self.histogram_tables(source, shape, &r_t)?;
        self.gpu.end_phase();
        Ok(ByteLinkHistograms { w })
    }

    /// Everything after the `W` commitments and the compression challenges: both fraction trees
    /// of every pack, the three GKR batches, the histogram query reductions and the `Q`
    /// reduction.
    pub fn prove(
        &mut self,
        source: &ByteLinkSource<'_>,
        histograms: &ByteLinkHistograms,
        inputs: &ByteLinkInputs<F>,
        compression: &ByteLinkCompression<F>,
        transcript: &mut impl ByteLinkTranscript<F>,
    ) -> Result<ByteLinkOpenings<F>, MetalError> {
        let shape = Shape::new(source)?;
        let r_t = &shape.cycle_point(inputs)?;
        let tables = self.gpu.fields(Denominators::new(compression).flat());
        let w = &histograms.w;

        self.gpu.phase("table trees");
        let table_roots = self
            .table_trees(w, &tables)?
            .iter()
            .flat_map(Trees::roots)
            .collect::<Vec<_>>();
        self.gpu.phase("trace trees");
        let tail = TailForm::new(r_t, compression.beta);
        let trace = self.trace_trees(source, shape, r_t, &tables, &tail)?;
        let roots = trace
            .roots()
            .into_iter()
            .chain(table_roots)
            .collect::<Vec<_>>();
        transcript.append(ByteLinkMessage::Roots(&roots));

        self.gpu.phase("trace gkr");
        let leaves = TraceLeaves {
            source,
            shape,
            tables: &tables,
            r_t,
        };
        let (z, trace_leaf) = self.gkr(
            transcript,
            ByteLinkBatch::Trace,
            &trace,
            Some((&leaves, &tail)),
        )?;
        drop(trace);
        self.gpu.phase("table trees rebuilt");
        let [triples, ram] = self.table_trees(w, &tables)?;
        self.gpu.phase("table gkr");
        let (y, triple_leaf) = self.gkr(transcript, ByteLinkBatch::Triples, &triples, None)?;
        let (y_ram, ram_leaf) = self.gkr(transcript, ByteLinkBatch::Ram, &ram, None)?;
        drop((triples, ram));

        self.gpu.phase("histogram queries");
        let triples = self.triple_queries(transcript, w, inputs, &y, &triple_leaf)?;
        let ram = reduce::ram_query(transcript, w, inputs, &y_ram, ram_leaf[0]);
        self.gpu.phase("source reduction");
        let source = self.source_reduction(
            transcript,
            source,
            shape,
            inputs,
            compression,
            r_t,
            &z,
            &trace_leaf,
        )?;
        self.gpu.end_phase();
        Ok(ByteLinkOpenings {
            triples,
            ram,
            source,
        })
    }

    /// Heap bytes of the largest phase.
    fn arena_bytes(&self, shape: Shape) -> usize {
        [
            self.sort_bytes(shape),
            gkr::table_tree_bytes(&self.gpu),
            gkr::trace_tree_bytes(&self.gpu, shape),
            reduce::product_bytes(&self.gpu, RAM, 1 << TRIPLE_BITS, 2),
            reduce::product_bytes(&self.gpu, 1, shape.explicit, 4),
        ]
        .into_iter()
        .max()
        .unwrap_or_default()
    }
}

/// Traces below the GPU prover's smallest shape prove on the host.
impl ByteLinkKernel<F> for MetalBackend {
    fn histograms<'a>(
        &self,
        trace: ByteTrace<'a>,
        inputs: &ByteLinkInputs<F>,
    ) -> Result<Box<dyn ByteLinkRun<F> + 'a>, KernelError<F>> {
        let log_rows = trace.plan.packing().logical_num_vars() as u32;
        if log_rows < MIN_LOG_ROWS {
            return ReferenceByteLink.histograms(trace, inputs);
        }
        let start = || {
            let bytes = trace_view(&self.context, trace.bytes)?;
            let mut prover = ByteLinkProver::new(&self.context)?;
            let source = ByteLinkSource {
                bytes: &bytes,
                log_rows,
                active_rows: trace.active_rows,
            };
            let histograms = prover.histograms(&source, inputs)?;
            Ok(MetalRun {
                prover,
                histograms,
                bytes,
                log_rows,
                active_rows: trace.active_rows,
                trace: PhantomData,
            })
        };
        Ok(Box::new(start().map_err(link_error)?))
    }
}

/// A shared-storage view of `bytes` without a copy; Metal maps only whole pages.
fn trace_view(metal: &SolinasMetal, bytes: &[i8]) -> Result<Buffer, MetalError> {
    const PAGE: usize = 16 << 10;
    if !bytes.as_ptr().addr().is_multiple_of(PAGE) || !bytes.len().is_multiple_of(PAGE) {
        return Err(MetalError::ByteLinkSourcePages {
            address: bytes.as_ptr().addr(),
            bytes: bytes.len(),
        });
    }
    metal.validate_buffer_length(bytes.len() as u64)?;
    Ok(metal.device.new_buffer_with_bytes_no_copy(
        bytes.as_ptr().cast_mut().cast(),
        bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    ))
}

/// A GPU link proof once `W` exists. `bytes` views the borrowed trace without owning it: every
/// command reading it completes before [`ByteLinkProver::histograms`] or
/// [`ByteLinkProver::prove`] returns, and the run drops the view before the borrow ends.
struct MetalRun<'a> {
    prover: ByteLinkProver,
    histograms: ByteLinkHistograms,
    bytes: Buffer,
    log_rows: u32,
    active_rows: usize,
    trace: PhantomData<&'a [i8]>,
}

impl ByteLinkRun<F> for MetalRun<'_> {
    fn tables(&self) -> HistogramTables<'_, F> {
        HistogramTables::Device(&self.histograms)
    }

    fn prove(
        self: Box<Self>,
        inputs: &ByteLinkInputs<F>,
        compression: &ByteLinkCompression<F>,
        mut transcript: &mut dyn ByteLinkTranscript<F>,
    ) -> Result<ByteLinkOpenings<F>, KernelError<F>> {
        let Self {
            mut prover,
            histograms,
            bytes,
            log_rows,
            active_rows,
            ..
        } = *self;
        let source = ByteLinkSource {
            bytes: &bytes,
            log_rows,
            active_rows,
        };
        prover
            .prove(&source, &histograms, inputs, compression, &mut transcript)
            .map_err(link_error)
    }
}

fn link_error(error: MetalError) -> KernelError<F> {
    KernelError::Sumcheck(SumcheckError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    })
}
