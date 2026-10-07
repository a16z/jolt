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

use jolt_claims::protocols::jolt::lattice::byte_link::{ByteLinkBatch, ByteLinkInputs};
use jolt_field::Prime128OffsetA7F7 as F;
use jolt_verifier::stages::byte_link::{ByteLinkCompression, ByteLinkOpenings};
use metal::{Buffer, Heap};

use super::{MetalError, SolinasMetal};
use crate::byte_link::reference::Denominators;
use crate::byte_link::{ByteLinkMessage, ByteLinkTranscript};
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

    /// The link's device arena (`None` before [`Self::histograms`]), for stage 8 to allocate from
    /// with `StorageModeShared | HazardTrackingModeTracked`, the heap's modes: the link has touched
    /// its pages, while a released heap leaves the process footprint only 130–300 ms later, so
    /// freeing it and allocating anew would count both
    /// (`/private/tmp/pika-scratch/akita-p0/link-mem/notes.md` §7).
    pub fn into_arena(mut self) -> Option<Heap> {
        self.gpu.take_arena()
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
