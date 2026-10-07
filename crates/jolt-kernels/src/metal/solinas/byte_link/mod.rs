//! The byte link on Metal: proves that the twenty one-hot claims stage 6b leaves on the
//! byte trace `Q` are the eq-weighted tuple histograms `W` of seven byte packs (spec revision 2
//! §2, W-only). For each pack, `Σ_t eq(r, t) / (β − γ·D(t)) = Σ_h W(h) / (β − γ·h)` is proved by
//! a binary fraction-tree GKR on both sides; the histogram queries reduce `W`'s marginal and
//! leaf claims to one point per group, and a degree-two reduction takes every claim on `Q` to
//! one point.
//!
//! The stage code drives Fiat–Shamir through [`ByteLinkTranscript`]: the prover hands it every
//! message in protocol order and takes every challenge from it, so labels, wire and verifier live
//! with the stage. Points crossing this boundary are canonical MSB-first; inside, round `i` of a
//! sumcheck binds bit `i` of the remaining index, so a layer's child point is
//! `(reverse(s), µ)`.
//!
//! Benches (ignored tests, through the shared-box GPU gate):
//! `gpu-window.sh cargo nextest run -p jolt-kernels --features metal,akita-byte-link --lib
//! --run-ignored ignored-only --no-capture -E 'test(/byte_link::tests::bench/)'` with
//! `LINK_LOG` (default 26), `LINK_REPS`, `LINK_FULL=1` (no zero tail), `LINK_HOT=1` (256 tuples
//! per pack) and `LINK_CHECK=1`; it prints per-phase wall and GPU seconds, the process
//! footprint and the arena size.

mod gkr;
mod gpu;
mod reduce;
mod sort;
#[cfg(test)]
mod tests;

use jolt_field::Prime128OffsetA7F7 as F;
use jolt_poly::UnivariatePoly;
use metal::{Buffer, Heap};

use super::{MetalError, SolinasMetal};
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

/// What stage 6b fixed: the cycle point `r`, each one-hot column's address point `k_c` and claim
/// `v_c` in pack order (`Q` slots 0–15, 25–28), and the fused increment claim `F(r)`. Points are
/// canonical MSB-first.
pub struct ByteLinkStatement<'a> {
    pub cycle_point: &'a [F],
    pub address_points: &'a [[F; 8]; 20],
    pub one_hot_claims: &'a [F; 20],
    pub fused_increment: F,
}

/// The compression challenges drawn after the `W` commitments: `γ` per pack slot, then `β`.
#[derive(Clone, Copy, Debug)]
pub struct ByteLinkCompression {
    pub gamma: [[F; 3]; PACKS],
    pub beta: F,
}

/// A GKR batch, in proof order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ByteLinkBatch {
    /// The seven trace trees.
    Trace,
    /// The six triple-table trees.
    Triples,
    /// The RAM-table tree.
    Ram,
}

/// A histogram query group, in proof order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ByteLinkQueryGroup {
    Triples,
    Ram,
}

/// A prover message, in protocol order. `layer` counts the variables of the layer's parent
/// index, so the top layer of a batch is 0; `round` indexes the rounds of its sumcheck. Messages
/// marked derived carry values the verifier computes itself, so the stage absorbs them without
/// putting them on the wire.
#[derive(Clone, Copy, Debug)]
pub enum ByteLinkMessage<'a> {
    /// `(P, B)` roots of the seven trace trees, then of the seven table trees.
    Roots(&'a [(F, F)]),
    /// Derived: the `(P, B)` claims every tree of the batch brings into `layer`, at `point`.
    LayerClaims {
        batch: ByteLinkBatch,
        layer: usize,
        point: &'a [F],
        claims: &'a [(F, F)],
    },
    /// A cubic round polynomial of a layer sumcheck.
    LayerRound {
        batch: ByteLinkBatch,
        layer: usize,
        round: usize,
        poly: &'a UnivariatePoly<F>,
    },
    /// The bound children `[P_0, B_0, P_1, B_1]` of every tree after the layer's last round.
    Children {
        batch: ByteLinkBatch,
        layer: usize,
        children: &'a [[F; 4]],
    },
    /// Derived: the values a group's query reduction batches, per pack its marginal values
    /// `v_c / 2^16` (RAM `v_c / 2^8`), then its table leaf `W(y)`.
    QueryValues {
        group: ByteLinkQueryGroup,
        values: &'a [F],
    },
    /// A quadratic round polynomial of a query reduction.
    QueryRound {
        group: ByteLinkQueryGroup,
        round: usize,
        poly: &'a UnivariatePoly<F>,
    },
    /// `W` of every pack of the group at the reduction's final point.
    QueryFinals {
        group: ByteLinkQueryGroup,
        values: &'a [F],
    },
    /// Derived: the trace leaf point `z` and the seven denominator leaf claims `B_j(z)`.
    SourceClaims {
        point: &'a [F],
        denominators: &'a [F],
    },
    /// A quadratic round polynomial of the `Q` reduction.
    SourceRound {
        round: usize,
        poly: &'a UnivariatePoly<F>,
    },
    /// `D_c(x)` of the [`SLOTS`] columns at the reduction's final point.
    SourceFinals(&'a [F]),
}

/// A challenge draw.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ByteLinkDraw {
    /// Independent `(λ_P, λ_B)` per tree of the batch, as `2 · trees` scalars.
    LayerWeights {
        batch: ByteLinkBatch,
        layer: usize,
    },
    LayerChallenge {
        batch: ByteLinkBatch,
        layer: usize,
        round: usize,
    },
    /// The child selector `µ` after a layer.
    ChildSelector {
        batch: ByteLinkBatch,
        layer: usize,
    },
    /// One independent weight per query value.
    QueryWeights {
        group: ByteLinkQueryGroup,
    },
    QueryChallenge {
        group: ByteLinkQueryGroup,
        round: usize,
    },
    /// The point `θ` of the zero-slot claims `D_30(θ) = D_31(θ) = 0`, `log_rows` scalars
    /// (canonical order); the prover never reads it, as both columns are zero.
    ZeroSlotPoint,
    /// Ten scalars: `α_j` of the seven denominators, `α_F`, then the zero slots' `α_30, α_31`.
    SourceWeights,
    SourceChallenge {
        round: usize,
    },
}

/// The Fiat–Shamir side of the link, implemented by the stage code.
pub trait ByteLinkTranscript {
    fn append(&mut self, message: ByteLinkMessage<'_>);
    fn challenges(&mut self, draw: ByteLinkDraw, count: usize) -> Vec<F>;
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

/// One evaluation claim the link leaves for stage 8.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ByteLinkOpening {
    /// Canonical MSB-first.
    pub point: Vec<F>,
    pub values: Vec<F>,
}

/// The link's openings: `W` of the six triples, `W` of the RAM pack, and all [`SLOTS`] columns
/// of `Q`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ByteLinkOpenings {
    pub triples: ByteLinkOpening,
    pub ram: ByteLinkOpening,
    pub source: ByteLinkOpening,
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
        cycle_point: &[F],
    ) -> Result<ByteLinkHistograms, MetalError> {
        let shape = Shape::new(source)?;
        assert_eq!(cycle_point.len(), shape.log_n as usize);
        self.gpu.reserve_arena(self.arena_bytes(shape))?;
        let r_t = cycle_point.iter().rev().copied().collect::<Vec<_>>();
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
        statement: &ByteLinkStatement<'_>,
        compression: &ByteLinkCompression,
        transcript: &mut impl ByteLinkTranscript,
    ) -> Result<ByteLinkOpenings, MetalError> {
        let shape = Shape::new(source)?;
        assert_eq!(statement.cycle_point.len(), shape.log_n as usize);
        let r_t = statement
            .cycle_point
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();
        let tables = self.gpu.fields(&gkr::compression_tables(compression));
        let w = &histograms.w;

        self.gpu.phase("table trees");
        let table_roots = self
            .table_trees(w, &tables)?
            .iter()
            .flat_map(Trees::roots)
            .collect::<Vec<_>>();
        self.gpu.phase("trace trees");
        let tail = TailForm::new(&r_t, compression.beta);
        let trace = self.trace_trees(source, shape, &r_t, &tables, &tail)?;
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
            r_t: &r_t,
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
        let triples = self.triple_queries(transcript, w, statement, &y, &triple_leaf)?;
        let ram = reduce::ram_query(transcript, w, statement, &y_ram, ram_leaf[0]);
        self.gpu.phase("source reduction");
        let source = self.source_reduction(
            transcript,
            source,
            shape,
            statement,
            compression,
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

    /// The link's device arena (`None` before [`Self::histograms`]), for stage 8 to allocate from:
    /// the link has touched its pages, while a released heap leaves the process footprint only
    /// 130–300 ms later, so freeing it and allocating anew would count both
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
