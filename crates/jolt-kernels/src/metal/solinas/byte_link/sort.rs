//! The challenge-free key sort and the eq-weighted histograms `W`, one pack at a time: the sort
//! buffers of a pack are dead once its `W` is written.

use metal::Buffer;

use super::{
    gkr::{eq_factors, suffix_sum, Columns, SplitEq},
    gpu::{
        bind, bytes, field, limbs, Grid, SORT_BUCKETS, SORT_COUNT, SORT_SCAN, SORT_SCATTER,
        WEIGHTS_FINISH, WEIGHTS_PREPARE, WEIGHTS_RUNS,
    },
    table_bits, ByteLinkProver, ByteLinkSource, Shape, F, PACKS, RAM, RAM_BITS, TRIPLE_BITS,
};
use crate::metal::solinas::{Fp128, MetalError};

const SORT_TILE: usize = 1024 * 16;
const SORT_BINS: usize = 1 << 12;
/// Sorted positions per thread of the run-sum pass.
const RUN_CHUNK: u32 = 32;

#[repr(C)]
#[derive(Clone, Copy)]
struct SortParams {
    entries: u32,
    n: u32,
    pack: u32,
    low_bits: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct WeightParams {
    entries: u32,
    keys: u32,
    chunk: u32,
    lo_bits: u32,
}

/// Cells of the `W` buffer: six triple tables then the RAM table, pack `p` at `p << 24`.
pub(super) const W_CELLS: usize = RAM * (1 << TRIPLE_BITS) + (1 << RAM_BITS);

impl ByteLinkProver {
    /// Arena bytes of the sort phase: the staging and sorted pairs of one pack, its CSR offsets
    /// and run accumulators.
    pub(super) fn sort_bytes(&self, shape: Shape) -> usize {
        2 * self.gpu.arena_size(shape.explicit * 8)
            + self.gpu.arena_size(((1 << TRIPLE_BITS) + 1) * 4)
            + self.gpu.arena_size((1 << TRIPLE_BITS) * 20)
    }

    pub(super) fn histogram_tables(
        &mut self,
        source: &ByteLinkSource<'_>,
        shape: Shape,
        r_t: &[F],
    ) -> Result<Buffer, MetalError> {
        let entries = shape.explicit;
        let tiles = entries.div_ceil(SORT_TILE);
        let split = SplitEq::new(r_t);
        let eq_lo = self.gpu.fields(&split.lo);
        let eq_hi = self.gpu.fields(&split.hi);
        let w = self.gpu.resident(W_CELLS * 16);
        let staging = self.gpu.arena(entries * 8)?;
        let pairs = self.gpu.arena(entries * 8)?;
        let offsets = self.gpu.arena(((1 << TRIPLE_BITS) + 1) * 4)?;
        let accumulators = self.gpu.arena((1 << TRIPLE_BITS) * 20)?;
        let counts = self.gpu.scratch(SORT_BINS * 4);
        let starts = self.gpu.scratch((SORT_BINS + 1) * 4);
        let cursors = self.gpu.scratch(SORT_BINS * 4);
        // [key-0 count, key-0 scatter cursor]
        let zeros = self.gpu.scratch(8);
        let columns = Columns::new();
        for pack in 0..PACKS {
            let keys = 1usize << table_bits(pack);
            let sort = SortParams {
                entries: entries as u32,
                n: shape.n() as u32,
                pack: pack as u32,
                low_bits: table_bits(pack) - SORT_BINS.ilog2(),
            };
            let weights = WeightParams {
                entries: entries as u32,
                keys: keys as u32,
                chunk: RUN_CHUNK,
                lo_bits: split.lo_bits,
            };
            let w_offset = (pack << TRIPLE_BITS) * 16;
            self.gpu.run("byte link histogram", |r| {
                r.zero(&counts);
                r.zero(&zeros);
                r.dispatch(SORT_COUNT, Grid::Groups(tiles, 1024), |e| {
                    bind(e, 0, source.bytes, 0);
                    bind(e, 1, &counts, 0);
                    bytes(e, 2, &sort);
                    bind(e, 3, &zeros, 0);
                    bytes(e, 4, &columns);
                });
                r.dispatch(SORT_SCAN, Grid::Groups(1, 1024), |e| {
                    bind(e, 0, &counts, 0);
                    bind(e, 1, &starts, 0);
                    bind(e, 2, &cursors, 0);
                    bytes(e, 3, &(entries as u32));
                    bind(e, 4, &zeros, 0);
                });
                r.dispatch(SORT_SCATTER, Grid::Groups(tiles, 1024), |e| {
                    bind(e, 0, source.bytes, 0);
                    bind(e, 1, &cursors, 0);
                    bind(e, 2, &staging, 0);
                    bytes(e, 3, &sort);
                    bind(e, 4, &zeros, 4);
                    bind(e, 5, &pairs, 0);
                    bytes(e, 6, &columns);
                });
                r.dispatch(SORT_BUCKETS, Grid::Groups(SORT_BINS, 1024), |e| {
                    bind(e, 0, &staging, 0);
                    bind(e, 1, &starts, 0);
                    bind(e, 2, &offsets, 0);
                    bind(e, 3, &pairs, 0);
                    bytes(e, 4, &sort);
                });
                r.dispatch(WEIGHTS_PREPARE, Grid::Threads(keys, 256), |e| {
                    bind(e, 0, &offsets, 0);
                    bind(e, 1, &accumulators, 0);
                    bytes(e, 2, &weights);
                });
                r.dispatch(
                    WEIGHTS_RUNS,
                    Grid::Threads(entries.div_ceil(RUN_CHUNK as usize), 256),
                    |e| {
                        bind(e, 0, &pairs, 0);
                        bind(e, 1, &offsets, 0);
                        bind(e, 2, &eq_lo, 0);
                        bind(e, 3, &eq_hi, 0);
                        bind(e, 4, &w, w_offset);
                        bind(e, 5, &accumulators, 0);
                        bytes(e, 6, &weights);
                    },
                );
                r.dispatch(WEIGHTS_FINISH, Grid::Threads(keys, 256), |e| {
                    bind(e, 0, &offsets, 0);
                    bind(e, 1, &accumulators, 0);
                    bind(e, 2, &w, w_offset);
                    bytes(e, 3, &weights);
                });
            })?;
        }
        // Every row past the explicit prefix has key 0 in every pack.
        let tail = suffix_sum(&eq_factors(r_t), entries);
        for pack in 0..PACKS {
            let cell = pack << TRIPLE_BITS;
            // SAFETY: shared storage of W_CELLS values, cell < W_CELLS; no command is in flight.
            unsafe {
                let w0 = w.contents().cast::<Fp128>().add(cell);
                *w0 = limbs(field(*w0) + tail);
            }
        }
        Ok(w)
    }
}
