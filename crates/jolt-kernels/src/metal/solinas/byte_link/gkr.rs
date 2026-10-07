//! Fraction trees and their batched layer sumchecks. Rows at or above the explicit prefix are
//! zero in every column, so their tree nodes, records and round sums are closed forms
//! ([`TailForm`]); records of at most `2^HOST_LOG` per tree finish on the host.

use jolt_field::{One, Ring, Zero};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use metal::Buffer;

use jolt_claims::protocols::jolt::lattice::byte_link::ByteLinkBatch;
use jolt_verifier::stages::byte_link::ByteLinkCompression;

use super::{
    gpu::{
        bind, bytes, field, limbs, upload, view, write, Gpu, Grid, Recorder, BOTTOM_BIND_EVAL,
        BOTTOM_STREAM, ROUND_BIND_EVAL, ROUND_EVAL, ROUND_STREAM, SUMS, SUM_PARTIALS, THREADS,
        TREE_LEAVES, TREE_UPPER,
    },
    ByteLinkProver, ByteLinkSource, Shape, F, HOST_LOG, PACKS, PACK_SLOTS, RAM, RAM_BITS,
    STORED_HEIGHT, TRIPLE_BITS,
};
use crate::byte_link::{ByteLinkDraw, ByteLinkMessage, ByteLinkTranscript};
use crate::metal::solinas::{Fp128, MetalError};

/// Levels a tree pass stores per threadgroup of 256 nodes.
const PASS_LEVELS: u32 = 9;

/// One tree level of every pack.
type PackLevel = Vec<Vec<(F, F)>>;

/// A leaf point and the leaf claims of every tree.
pub(super) type LeafClaims = (Vec<F>, Vec<(F, F)>);

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Layout {
    offset: [u32; 32],
    stride: [u32; 32],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TreeParams {
    table: u32,
    count: u32,
    first: u32,
    levels: u32,
    n: u32,
    lo_bits: u32,
    pack_base: u32,
    reserved: u32,
    layout: Layout,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct RoundParams {
    pub pairs: u32,
    pub groups: u32,
    pub pairs_per_thread: u32,
    pub lo_bits: u32,
    pub in_stride: u32,
    pub out_stride: u32,
    pub e_lo_bits: u32,
    pub stream_log: u32,
    pub n: u32,
    pub materialize: u32,
    pub bound: u32,
    pub shift: u32,
    pub height: u32,
    pub reserved: [u32; 3],
    pub challenge: Fp128,
}

/// `LinkColumns`: the `Q` columns of every pack.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Columns {
    c: [[u32; 4]; PACKS],
}

const _: () = assert!(size_of::<TreeParams>() == 288);
const _: () = assert!(size_of::<RoundParams>() == 80);
const _: () = assert!(size_of::<Columns>() == 112);

impl Columns {
    pub fn new() -> Self {
        Self {
            c: PACK_SLOTS.map(|[a, b, c]| [a as u32, b as u32, c as u32, 0]),
        }
    }
}

/// `eq(point, x)` for every `x`, bit `i` of `x` paired with `point[i]`.
pub(super) fn eq_lsb(point: &[F]) -> Vec<F> {
    let reversed = point.iter().rev().copied().collect::<Vec<_>>();
    EqPolynomial::evals(&reversed, None)
}

pub(super) fn eq(a: &[F], b: &[F]) -> F {
    EqPolynomial::mle(a, b)
}

/// `eq(point, ·)` as two tables over the low half of the bits and the rest.
pub(super) struct SplitEq {
    pub lo_bits: u32,
    pub lo: Vec<F>,
    pub hi: Vec<F>,
}

impl SplitEq {
    pub fn new(point: &[F]) -> Self {
        let lo_bits = point.len() / 2;
        Self {
            lo_bits: lo_bits as u32,
            lo: eq_lsb(&point[..lo_bits]),
            hi: eq_lsb(&point[lo_bits..]),
        }
    }
}

/// `Σ_{x ≥ start} Π_l factor_l(x_l)` over `x < 2^factors.len()`.
pub(super) fn suffix_sum(factors: &[[F; 2]], start: usize) -> F {
    if start >> factors.len() != 0 {
        return F::zero();
    }
    let below = factors
        .iter()
        .scan(F::one(), |acc, f| {
            let current = *acc;
            *acc *= f[0] + f[1];
            Some(current)
        })
        .collect::<Vec<_>>();
    let mut prefix = F::one();
    let mut total = F::zero();
    for l in (0..factors.len()).rev() {
        let bit = (start >> l) & 1;
        if bit == 0 {
            total += prefix * factors[l][1] * below[l];
        }
        prefix *= factors[l][bit];
    }
    total + prefix
}

pub(super) fn eq_factors(point: &[F]) -> Vec<[F; 2]> {
    point.iter().map(|&x| [F::one() - x, x]).collect()
}

/// `[pack][slot][code]`: `β − γ_0 σ(code)` in slot 0, `γ_i σ(code)` in slots 1 and 2.
pub(super) fn compression_tables(compression: &ByteLinkCompression<F>) -> Vec<F> {
    compression
        .gamma
        .iter()
        .flat_map(|gamma| {
            (0..3).flat_map(move |slot| {
                (0..256u32).map(move |code| {
                    let term = gamma[slot] * F::from_i64(i64::from(code as u8 as i8));
                    if slot == 0 {
                        compression.beta - term
                    } else {
                        term
                    }
                })
            })
        })
        .collect()
}

/// Closed forms of the all-zero suffix of the trace trees: a node of height `h` there is
/// `(β^(B−1) eq(r_t[h..], x), β^B)`, `B = 2^h`, in every pack.
pub(super) struct TailForm {
    pub r_t: Vec<F>,
    beta: F,
}

impl TailForm {
    pub fn new(r_t: &[F], beta: F) -> Self {
        Self {
            r_t: r_t.to_vec(),
            beta,
        }
    }

    /// `β^(2^h − 1)`.
    fn pow2_minus_one(&self, height: u32) -> F {
        (0..height).fold(F::one(), |acc, _| acc * acc * self.beta)
    }

    fn node(&self, height: u32, x: usize, eq_high: &[F]) -> (F, F) {
        let scale = self.pow2_minus_one(height);
        (scale * eq_high[x], scale * self.beta)
    }

    /// Records `[start, total)` of the layer whose children have height `child`, after the
    /// layer's challenges `bound`: child `b` of record `y` has numerator `head_b fold eq_y`.
    fn records(&self, child: u32, bound: &[F], start: usize, total: usize) -> Vec<[F; 4]> {
        let scale = self.pow2_minus_one(child);
        let q = scale * self.beta;
        let c = child as usize;
        let i = bound.len();
        let head = [F::one() - self.r_t[c], self.r_t[c]];
        let fold = scale * eq(&self.r_t[c + 1..c + 1 + i], bound);
        let eq_y = eq_lsb(&self.r_t[c + 1 + i..]);
        (start..total)
            .map(|y| [head[0] * fold * eq_y[y], q, head[1] * fold * eq_y[y], q])
            .collect()
    }

    /// The tail pairs' `(ΣwG_P, ΣwG_Q)` at `X = 0` (their `X²` coefficient is zero) in round `i`
    /// of the layer whose children have height `child`; the explicit pairs are `< first_pair`.
    fn round_sums(&self, child: u32, rho: &[F], bound: &[F], first_pair: usize) -> (F, F) {
        let scale = self.pow2_minus_one(child);
        let q = scale * self.beta;
        let parent = child as usize + 1;
        let i = bound.len();
        let alpha = scale * eq(&self.r_t[parent..parent + i], bound);
        let rest = eq_factors(&rho[i + 1..]);
        let s0 = suffix_sum(&rest, first_pair);
        let joint = rest
            .iter()
            .zip(eq_factors(&self.r_t[parent + i + 1..]))
            .map(|(a, b)| [a[0] * b[0], a[1] * b[1]])
            .collect::<Vec<_>>();
        let s1 = suffix_sum(&joint, first_pair);
        let eq_x = F::one() - self.r_t[parent + i];
        (q * alpha * eq_x * s1, q * q * s0)
    }
}

/// The trace leaves of the link: `Q`, the compression tables and the split `eq(r_t, ·)`.
pub(super) struct TraceLeaves<'a> {
    pub source: &'a ByteLinkSource<'a>,
    pub shape: Shape,
    pub tables: &'a Buffer,
    pub r_t: &'a [F],
}

/// One fraction-tree set: heights `lowest..=split` stored on the device for the explicit
/// leaves, heights `split..` of every pack on the host.
pub(super) struct Trees {
    packs: usize,
    log_leaves: u32,
    split: u32,
    lowest: u32,
    explicit: usize,
    nodes: Buffer,
    layout: Layout,
    host: Vec<Option<PackLevel>>,
}

impl Trees {
    fn level_offset(&self, height: u32) -> usize {
        self.layout.offset[height as usize] as usize * 32
    }

    pub fn roots(&self) -> Vec<(F, F)> {
        self.host[self.log_leaves as usize]
            .as_ref()
            .map_or_else(Vec::new, |level| {
                level.iter().map(|nodes| nodes[0]).collect()
            })
    }
}

/// Layout and node count of `packs` trees over `explicit` leaves storing heights
/// `lowest..=split`. The node buffer also holds the records that the streamed layers below
/// `STORED_HEIGHT` materialize, `explicit / 4` nodes per pack.
fn node_layout(packs: usize, log_leaves: u32, explicit: usize, lowest: u32) -> (Layout, usize) {
    let split = log_leaves - HOST_LOG;
    let mut layout = Layout::default();
    let mut count = 0usize;
    for h in lowest..=split {
        layout.offset[h as usize] = count as u32;
        layout.stride[h as usize] = (explicit >> h) as u32;
        count += packs * (explicit >> h);
    }
    if lowest == STORED_HEIGHT {
        count = count.max(packs * (explicit >> 2));
    }
    (layout, count)
}

pub(super) fn trace_tree_bytes(gpu: &Gpu, shape: Shape) -> usize {
    gpu.arena_size(32 * node_layout(PACKS, shape.log_n, shape.explicit, STORED_HEIGHT).1)
}

pub(super) fn table_tree_bytes(gpu: &Gpu) -> usize {
    let nodes = |packs, bits| gpu.arena_size(32 * node_layout(packs, bits, 1 << bits, 0).1);
    nodes(RAM, TRIPLE_BITS) + nodes(1, RAM_BITS)
}

/// Pairs per thread and threadgroups per pack for `pairs` pairs.
pub(super) fn tiling(pairs: usize) -> (usize, usize) {
    let per_thread = (pairs / (THREADS * 256)).clamp(1, 16);
    let per_thread = 1 << per_thread.ilog2();
    (per_thread, pairs.div_ceil(THREADS * per_thread))
}

/// The children `[P_0, Q_0, P_1, Q_1]` of every parent of one layer, per tree.
struct Layer {
    records: Vec<Vec<[F; 4]>>,
}

fn gate(l: (F, F), r: (F, F)) -> (F, F) {
    (l.0 * r.1 + r.0 * l.1, l.1 * r.1)
}

fn record_gate(record: &[F; 4]) -> (F, F) {
    gate((record[0], record[1]), (record[2], record[3]))
}

/// The running claim of one layer sumcheck.
struct Rounds {
    weights: Vec<F>,
    claim: F,
    prefix: F,
    challenges: Vec<F>,
}

impl Rounds {
    fn new(weights: Vec<F>, claims: &[(F, F)]) -> Self {
        let claim = claims
            .iter()
            .zip(weights.chunks(2))
            .map(|(&(p, q), w)| w[0] * p + w[1] * q)
            .sum();
        Self {
            weights,
            claim,
            prefix: F::one(),
            challenges: Vec::new(),
        }
    }

    /// The weighted sum over trees of per-tree `(G_P, G_Q)`.
    fn batch(&self, sums: impl Iterator<Item = (F, F)>) -> F {
        sums.zip(self.weights.chunks(2))
            .map(|((p, q), w)| w[0] * p + w[1] * q)
            .sum()
    }

    /// Weights summed over trees, for the pack-independent tail.
    fn total_weights(&self) -> (F, F) {
        self.weights
            .chunks(2)
            .fold((F::zero(), F::zero()), |(p, q), w| (p + w[0], q + w[1]))
    }

    /// Sends round `s(X) = prefix · eq(ρ, X) · t(X)` from `t(0)` and `t`'s `X²` coefficient.
    fn send(
        &mut self,
        transcript: &mut impl ByteLinkTranscript<F>,
        (batch, layer): (ByteLinkBatch, usize),
        rho: F,
        t0: F,
        t2: F,
    ) {
        let round = self.challenges.len();
        let poly = UnivariatePoly::from_linear_times_quadratic_with_hint(
            [
                self.prefix * (F::one() - rho),
                self.prefix * (rho + rho - F::one()),
            ],
            t0,
            t2,
            self.claim,
        );
        transcript.append(ByteLinkMessage::LayerRound {
            batch,
            layer,
            round,
            poly: &poly,
        });
        let r = transcript.challenges(
            ByteLinkDraw::LayerChallenge {
                batch,
                layer,
                round,
            },
            1,
        )[0];
        self.claim = poly.evaluate(r);
        self.prefix *= F::one() - rho - r + (rho + rho) * r;
        self.challenges.push(r);
    }

    /// Host rounds until every tree holds one record.
    fn finish_on_host(
        &mut self,
        transcript: &mut impl ByteLinkTranscript<F>,
        context: (ByteLinkBatch, usize),
        layer: &mut Layer,
        rho: &[F],
    ) {
        while layer.records[0].len() > 1 {
            let round = self.challenges.len();
            let w = eq_lsb(&rho[round + 1..]);
            let sums = layer
                .records
                .iter()
                .map(|records| {
                    records
                        .chunks(2)
                        .zip(&w)
                        .fold([F::zero(); 4], |mut acc, (pair, &w)| {
                            let (lo, hi) = (&pair[0], &pair[1]);
                            let d = std::array::from_fn(|k| hi[k] - lo[k]);
                            let (p0, q0) = record_gate(lo);
                            let (p2, q2) = record_gate(&d);
                            acc[0] += w * p0;
                            acc[1] += w * q0;
                            acc[2] += w * p2;
                            acc[3] += w * q2;
                            acc
                        })
                })
                .collect::<Vec<_>>();
            let t0 = self.batch(sums.iter().map(|s| (s[0], s[1])));
            let t2 = self.batch(sums.iter().map(|s| (s[2], s[3])));
            self.send(transcript, context, rho[round], t0, t2);
            let r = self.challenges[round];
            for records in &mut layer.records {
                let bound = records
                    .chunks(2)
                    .map(|pair| std::array::from_fn(|k| pair[0][k] + r * (pair[1][k] - pair[0][k])))
                    .collect();
                *records = bound;
            }
        }
    }
}

/// Sends the bound children, draws `µ` and returns the next layer's point and claims.
fn close_layer(
    transcript: &mut impl ByteLinkTranscript<F>,
    (batch, layer): (ByteLinkBatch, usize),
    records: &Layer,
    rounds: &Rounds,
) -> LeafClaims {
    let children = records.records.iter().map(|r| r[0]).collect::<Vec<_>>();
    transcript.append(ByteLinkMessage::Children {
        batch,
        layer,
        children: &children,
    });
    let mu = transcript.challenges(ByteLinkDraw::ChildSelector { batch, layer }, 1)[0];
    let claims = children
        .iter()
        .map(|c| (c[0] + mu * (c[2] - c[0]), c[1] + mu * (c[3] - c[1])))
        .collect();
    let point = std::iter::once(mu)
        .chain(rounds.challenges.iter().copied())
        .collect();
    (point, claims)
}

pub(super) fn canonical(point: &[F]) -> Vec<F> {
    point.iter().rev().copied().collect()
}

/// Per-round device tables of one batch.
struct RoundBuffers {
    w_lo: Buffer,
    w_hi: Buffer,
    e_lo: Buffer,
    e_hi: Buffer,
    coeffs: Buffer,
    k: Buffer,
    partials: Buffer,
    sums: Buffer,
}

impl ByteLinkProver {
    pub(super) fn trace_trees(
        &mut self,
        source: &ByteLinkSource<'_>,
        shape: Shape,
        r_t: &[F],
        tables: &Buffer,
        tail: &TailForm,
    ) -> Result<Trees, MetalError> {
        let split = SplitEq::new(r_t);
        let eq_lo = self.gpu.fields(&split.lo);
        let eq_hi = self.gpu.fields(&split.hi);
        let leaves = TreeLeaves::Trace {
            q: source.bytes,
            eq: (&eq_lo, &eq_hi, split.lo_bits),
            n: shape.n(),
        };
        self.trees(
            &leaves,
            tables,
            PACKS,
            0,
            shape.log_n,
            shape.explicit,
            STORED_HEIGHT,
            Some(tail),
        )
    }

    /// The six triple-table trees and the RAM-table tree. Built once for their roots and again
    /// after the trace GKR, so they are never live next to the trace trees.
    pub(super) fn table_trees(
        &mut self,
        w: &Buffer,
        tables: &Buffer,
    ) -> Result<[Trees; 2], MetalError> {
        let leaves = TreeLeaves::Table { w };
        Ok([
            self.trees(
                &leaves,
                tables,
                RAM,
                0,
                TRIPLE_BITS,
                1 << TRIPLE_BITS,
                0,
                None,
            )?,
            self.trees(&leaves, tables, 1, RAM, RAM_BITS, 1 << RAM_BITS, 0, None)?,
        ])
    }

    #[expect(clippy::too_many_arguments, reason = "one tree-set geometry")]
    fn trees(
        &mut self,
        leaves: &TreeLeaves<'_>,
        tables: &Buffer,
        packs: usize,
        pack_base: usize,
        log_leaves: u32,
        explicit: usize,
        lowest: u32,
        tail: Option<&TailForm>,
    ) -> Result<Trees, MetalError> {
        let split = log_leaves - HOST_LOG;
        let (layout, count) = node_layout(packs, log_leaves, explicit, lowest);
        let nodes = self.gpu.arena(count * 32)?;
        let columns = Columns::new();
        let mut passes = Vec::new();
        let mut first = lowest.max(1);
        while first <= split {
            let levels = PASS_LEVELS.min(split - first + 1);
            passes.push((first, levels, 1usize << (levels - 1)));
            first += levels;
        }
        let (table, n, lo_bits) = match *leaves {
            TreeLeaves::Trace {
                n,
                eq: (_, _, lo_bits),
                ..
            } => (0, n, lo_bits),
            TreeLeaves::Table { .. } => (1, 0, 0),
        };
        self.gpu.run("byte link trees", |r| {
            for (index, &(first, levels, width)) in passes.iter().enumerate() {
                let count = explicit >> first;
                let params = TreeParams {
                    table,
                    count: count as u32,
                    first,
                    levels,
                    n: n as u32,
                    lo_bits,
                    pack_base: pack_base as u32,
                    reserved: 0,
                    layout,
                };
                let grid = Grid::Groups(packs * count / width, width);
                if index == 0 {
                    r.dispatch(TREE_LEAVES, grid, |e| {
                        let (q, w, eq_lo, eq_hi) = match *leaves {
                            TreeLeaves::Trace {
                                q, eq: (lo, hi, _), ..
                            } => (q, lo, lo, hi),
                            TreeLeaves::Table { w } => (w, w, w, w),
                        };
                        bind(e, 0, q, 0);
                        bind(e, 1, w, 0);
                        bind(e, 2, tables, 0);
                        bind(e, 3, eq_lo, 0);
                        bind(e, 4, eq_hi, 0);
                        bind(e, 5, &nodes, 0);
                        bytes(e, 6, &params);
                        bytes(e, 7, &columns);
                    });
                } else {
                    r.dispatch(TREE_UPPER, grid, |e| {
                        bind(e, 0, &nodes, 0);
                        bytes(e, 1, &params);
                    });
                }
            }
        })?;
        let full = 1usize << HOST_LOG;
        let count = explicit >> split;
        let stored = view::<[Fp128; 2]>(
            &nodes,
            layout.offset[split as usize] as usize,
            packs * count,
        );
        let eq_high = tail.map(|t| eq_lsb(&t.r_t[split as usize..]));
        let level = (0..packs)
            .map(|pack| {
                let mut level = stored[pack * count..(pack + 1) * count]
                    .iter()
                    .map(|&[p, q]| (field(p), field(q)))
                    .collect::<Vec<_>>();
                if let (Some(tail), Some(eq_high)) = (tail, &eq_high) {
                    level.extend((count..full).map(|x| tail.node(split, x, eq_high)));
                }
                level
            })
            .collect::<Vec<_>>();
        let mut host = vec![None; log_leaves as usize + 1];
        host[split as usize] = Some(level);
        for height in split..log_leaves {
            let next = host[height as usize].as_ref().map(|level: &PackLevel| {
                level
                    .iter()
                    .map(|nodes| nodes.chunks(2).map(|c| gate(c[0], c[1])).collect())
                    .collect()
            });
            host[height as usize + 1] = next;
        }
        Ok(Trees {
            packs,
            log_leaves,
            split,
            lowest,
            explicit,
            nodes,
            layout,
            host,
        })
    }

    fn round_buffers(&self, trees: &Trees) -> RoundBuffers {
        let rest = trees.log_leaves - 2;
        let table = (1usize << rest.div_ceil(2))
            .max((1 << rest) / THREADS)
            .max(THREADS * 16);
        let pairs = trees.explicit / 4;
        let groups = tiling(pairs.max(1)).1;
        RoundBuffers {
            w_lo: self.gpu.scratch(table * 16),
            w_hi: self.gpu.scratch(table * 16),
            e_lo: self.gpu.scratch(table * 16),
            e_hi: self.gpu.scratch(table * 16),
            coeffs: self.gpu.scratch(2 * trees.packs * 16),
            k: self.gpu.scratch(4 * 16),
            partials: self.gpu.scratch(trees.packs * groups * SUMS * 16),
            sums: self.gpu.scratch(trees.packs * SUMS * 16),
        }
    }

    /// The batched GKR over `trees`; returns the leaf point (internal order) and leaf claims.
    pub(super) fn gkr(
        &mut self,
        transcript: &mut impl ByteLinkTranscript<F>,
        batch: ByteLinkBatch,
        trees: &Trees,
        trace: Option<(&TraceLeaves<'_>, &TailForm)>,
    ) -> Result<LeafClaims, MetalError> {
        let buffers = self.round_buffers(trees);
        let mut point = Vec::new();
        let mut claims = trees.roots();
        for child in (0..trees.log_leaves).rev() {
            let layer = point.len();
            let context = (batch, layer);
            transcript.append(ByteLinkMessage::LayerClaims {
                batch,
                layer,
                point: &canonical(&point),
                claims: &claims,
            });
            let weights =
                transcript.challenges(ByteLinkDraw::LayerWeights { batch, layer }, 2 * trees.packs);
            let mut rounds = Rounds::new(weights, &claims);
            write(&buffers.coeffs, &upload(&rounds.weights));
            let mut records = match &trees.host[child as usize] {
                Some(level) if child >= trees.split => Layer {
                    records: level
                        .iter()
                        .map(|nodes| {
                            nodes
                                .chunks(2)
                                .map(|c| [c[0].0, c[0].1, c[1].0, c[1].1])
                                .collect()
                        })
                        .collect(),
                },
                _ => match trace {
                    Some((leaves, tail)) if child == 0 => self.bottom_layer(
                        transcript,
                        context,
                        &mut rounds,
                        trees,
                        &buffers,
                        &point,
                        leaves,
                        tail,
                    )?,
                    _ => self.stored_layer(
                        transcript,
                        context,
                        &mut rounds,
                        trees,
                        &buffers,
                        child,
                        &point,
                        trace,
                    )?,
                },
            };
            rounds.finish_on_host(transcript, context, &mut records, &point);
            (point, claims) = close_layer(transcript, context, &records, &rounds);
        }
        Ok((point, claims))
    }

    fn reduce_partials(r: &mut Recorder, buffers: &RoundBuffers, packs: usize, groups: usize) {
        r.dispatch(SUM_PARTIALS, Grid::Groups(packs, 1024), |e| {
            bind(e, 0, &buffers.partials, 0);
            bind(e, 1, &buffers.sums, 0);
            bytes(e, 2, &(groups as u32));
        });
    }

    /// The pack-summed `t(0)` and `X²` coefficient of a GPU round plus the tail's.
    fn read_t(buffers: &RoundBuffers, packs: usize, rounds: &Rounds, tail: (F, F)) -> (F, F) {
        let sums = view::<Fp128>(&buffers.sums, 0, packs * SUMS);
        let (sum0, sum2) = (0..packs).fold((F::zero(), F::zero()), |(a, b), pack| {
            (
                a + field(sums[pack * SUMS]),
                b + field(sums[pack * SUMS + 1]),
            )
        });
        let (c_p, c_q) = rounds.total_weights();
        (sum0 + c_p * tail.0 + c_q * tail.1, sum2)
    }

    /// Writes the pair weights `eq(rho[i + 1..], p)` split at the tile size; returns `lo_bits`.
    fn weights(buffers: &RoundBuffers, rho: &[F], i: usize, per_thread: usize) -> u32 {
        let rest = &rho[i + 1..];
        let lo_bits = ((THREADS * per_thread).ilog2() as usize).min(rest.len());
        write(&buffers.w_lo, &upload(&eq_lsb(&rest[..lo_bits])));
        write(&buffers.w_hi, &upload(&eq_lsb(&rest[lo_bits..])));
        lo_bits as u32
    }

    /// GPU rounds of a layer whose children are tree nodes, then its host records. Children at
    /// stored heights are bound in place inside their level; the trace layers whose children
    /// lie below `STORED_HEIGHT` rebuild them from `Q` (`byte_link_round_stream`): children at
    /// height 2 materialize in round 0, children at height 1 in round 1, both at the start of
    /// the node buffer, whose stored levels the layers above have consumed.
    #[expect(clippy::too_many_arguments, reason = "one layer of one batch")]
    fn stored_layer(
        &mut self,
        transcript: &mut impl ByteLinkTranscript<F>,
        context: (ByteLinkBatch, usize),
        rounds: &mut Rounds,
        trees: &Trees,
        buffers: &RoundBuffers,
        child: u32,
        rho: &[F],
        trace: Option<(&TraceLeaves<'_>, &TailForm)>,
    ) -> Result<Layer, MetalError> {
        let packs = trees.packs;
        let k = rho.len();
        let explicit = trees.explicit >> (child + 1);
        let gpu_rounds = k.saturating_sub(HOST_LOG as usize);
        let stream = trace.filter(|_| child < trees.lowest);
        let materialized = match stream {
            Some(_) => 2 - child as usize,
            None => 0,
        };
        assert!(stream.is_none() || gpu_rounds > materialized);
        let eq_split = stream.map(|(leaves, _)| SplitEq::new(leaves.r_t));
        let eq_tables = eq_split
            .as_ref()
            .map(|split| (self.gpu.fields(&split.lo), self.gpu.fields(&split.hi)));
        let columns = Columns::new();
        let tail = trace.map(|(_, tail)| tail);
        let (buffer_offset, mut in_stride) = match stream {
            Some(_) => (0, 0),
            None => (trees.level_offset(child), explicit),
        };
        for i in 0..gpu_rounds + usize::from(gpu_rounds > 0) {
            let last = i == gpu_rounds;
            let bound = explicit >> i;
            let pairs = if last { bound.div_ceil(2) } else { bound / 2 };
            let (per_thread, groups) = tiling(pairs);
            let lo_bits = if last {
                0
            } else {
                Self::weights(buffers, rho, i, per_thread)
            };
            let streaming = stream.is_some() && i <= materialized;
            let kernel = match () {
                () if streaming => ROUND_STREAM,
                () if i == 0 => ROUND_EVAL,
                () => ROUND_BIND_EVAL,
            };
            let out_stride = if streaming { 2 * pairs } else { bound };
            let params = RoundParams {
                pairs: pairs as u32,
                groups: groups as u32,
                pairs_per_thread: per_thread as u32,
                lo_bits,
                in_stride: in_stride as u32,
                out_stride: out_stride as u32,
                e_lo_bits: eq_split.as_ref().map_or(0, |split| split.lo_bits),
                stream_log: i as u32,
                n: stream.map_or(0, |(leaves, _)| leaves.shape.n() as u32),
                materialize: u32::from(streaming && i == materialized),
                bound: bound as u32,
                shift: i.saturating_sub(materialized + 1) as u32,
                height: child,
                challenge: limbs(rounds.challenges.last().copied().unwrap_or_else(F::zero)),
                ..RoundParams::default()
            };
            self.gpu.run(kernel, |r| {
                r.dispatch(kernel, Grid::Groups(packs * groups, THREADS), |e| {
                    bind(e, 0, &trees.nodes, buffer_offset);
                    bind(e, 1, &buffers.w_lo, 0);
                    bind(e, 2, &buffers.w_hi, 0);
                    bind(e, 3, &buffers.partials, 0);
                    bytes(e, 5, &params);
                    bind(e, 6, &buffers.coeffs, 0);
                    if let (Some((leaves, _)), Some((eq_lo, eq_hi))) = (stream, &eq_tables) {
                        bind(e, 7, leaves.source.bytes, 0);
                        bind(e, 8, leaves.tables, 0);
                        bind(e, 9, eq_lo, 0);
                        bind(e, 10, eq_hi, 0);
                        bytes(e, 11, &columns);
                    }
                });
                if !last {
                    Self::reduce_partials(r, buffers, packs, groups);
                }
            })?;
            if streaming && i == materialized {
                in_stride = out_stride;
            }
            if !last {
                let tail_sums = tail.map_or((F::zero(), F::zero()), |t| {
                    t.round_sums(child, rho, &rounds.challenges, pairs)
                });
                let (t0, t2) = Self::read_t(buffers, packs, rounds, tail_sums);
                rounds.send(transcript, context, rho[i], t0, t2);
            }
        }
        let bound = explicit >> gpu_rounds;
        let spread = gpu_rounds.saturating_sub(materialized);
        let total = 1usize << (k - gpu_rounds);
        let raw = view::<[Fp128; 4]>(&trees.nodes, buffer_offset / 64, packs * in_stride);
        let tail_records = tail.map(|t| t.records(child, &rounds.challenges, bound, total));
        Ok(Layer {
            records: (0..packs)
                .map(|pack| {
                    let mut records = (0..bound)
                        .map(|j| {
                            let slot = pack * in_stride + (((j & !1) << spread) | (j & 1));
                            raw[slot].map(field)
                        })
                        .collect::<Vec<_>>();
                    if let Some(tail) = &tail_records {
                        records.extend_from_slice(tail);
                    }
                    records
                })
                .collect(),
        })
    }

    /// The bottom layer of the trace trees: rounds 0 and 1 read `Q` (round 1 stores the bound
    /// denominators at the start of the node buffer), later rounds bind them in place; the
    /// numerators are closed forms.
    #[expect(clippy::too_many_arguments, reason = "one layer of one batch")]
    fn bottom_layer(
        &mut self,
        transcript: &mut impl ByteLinkTranscript<F>,
        context: (ByteLinkBatch, usize),
        rounds: &mut Rounds,
        trees: &Trees,
        buffers: &RoundBuffers,
        rho: &[F],
        leaves: &TraceLeaves<'_>,
        tail: &TailForm,
    ) -> Result<Layer, MetalError> {
        const MATERIALIZED: usize = 1;
        let packs = trees.packs;
        let k = rho.len();
        let explicit = trees.explicit >> 1;
        let gpu_rounds = k - HOST_LOG as usize;
        assert!(gpu_rounds > MATERIALIZED);
        let r_t = &tail.r_t;
        let head = [F::one() - r_t[0], r_t[0]];
        let columns = Columns::new();
        let mut stride = 0;
        for i in 0..=gpu_rounds {
            let last = i == gpu_rounds;
            let bound = explicit >> i;
            let pairs = if last { bound.div_ceil(2) } else { bound / 2 };
            let (per_thread, groups) = tiling(pairs);
            let lo_bits = if last {
                0
            } else {
                Self::weights(buffers, rho, i, per_thread)
            };
            let rest = &r_t[i + 2..];
            let e_lo_bits = rest.len() / 2;
            write(&buffers.e_lo, &upload(&eq_lsb(&rest[..e_lo_bits])));
            write(&buffers.e_hi, &upload(&eq_lsb(&rest[e_lo_bits..])));
            let fold = eq(&r_t[1..=i], &rounds.challenges);
            let x = r_t[1 + i];
            let mut k_table = [Fp128::ZERO; 4];
            for b in 0..2 {
                let lo = head[b] * fold * (F::one() - x);
                k_table[b] = limbs(lo);
                k_table[2 + b] = limbs(head[b] * fold * x - lo);
            }
            write(&buffers.k, &k_table);
            let streaming = i <= MATERIALIZED;
            let out_stride = if streaming { 2 * pairs } else { bound };
            let params = RoundParams {
                pairs: pairs as u32,
                groups: groups as u32,
                pairs_per_thread: per_thread as u32,
                lo_bits,
                in_stride: stride as u32,
                out_stride: out_stride as u32,
                e_lo_bits: e_lo_bits as u32,
                stream_log: i as u32,
                n: leaves.shape.n() as u32,
                materialize: u32::from(i == MATERIALIZED),
                bound: bound as u32,
                shift: i.saturating_sub(MATERIALIZED + 1) as u32,
                challenge: limbs(rounds.challenges.last().copied().unwrap_or_else(F::zero)),
                ..RoundParams::default()
            };
            let kernel = if streaming {
                BOTTOM_STREAM
            } else {
                BOTTOM_BIND_EVAL
            };
            self.gpu.run(kernel, |r| {
                r.dispatch(kernel, Grid::Groups(packs * groups, THREADS), |e| {
                    bind(e, 0, &trees.nodes, 0);
                    bind(e, 1, &buffers.w_lo, 0);
                    bind(e, 2, &buffers.w_hi, 0);
                    bind(e, 3, &buffers.partials, 0);
                    bytes(e, 5, &params);
                    bind(e, 6, &buffers.coeffs, 0);
                    bind(e, 7, &buffers.e_lo, 0);
                    bind(e, 8, &buffers.e_hi, 0);
                    bind(e, 9, &buffers.k, 0);
                    if streaming {
                        bind(e, 10, leaves.source.bytes, 0);
                        bind(e, 11, leaves.tables, 0);
                        bytes(e, 12, &columns);
                    }
                });
                if !last {
                    Self::reduce_partials(r, buffers, packs, groups);
                }
            })?;
            if i == MATERIALIZED {
                stride = out_stride;
            }
            if !last {
                let tail_sums = tail.round_sums(0, rho, &rounds.challenges, pairs);
                let (t0, t2) = Self::read_t(buffers, packs, rounds, tail_sums);
                rounds.send(transcript, context, rho[i], t0, t2);
            }
        }
        let bound = explicit >> gpu_rounds;
        let total = 1usize << (k - gpu_rounds);
        let spread = gpu_rounds - MATERIALIZED;
        let stored = view::<[Fp128; 2]>(&trees.nodes, 0, packs * stride);
        let closed = tail.records(0, &rounds.challenges, 0, total);
        Ok(Layer {
            records: (0..packs)
                .map(|pack| {
                    closed
                        .iter()
                        .enumerate()
                        .map(|(y, record)| {
                            let mut record = *record;
                            if y < bound {
                                let slot = pack * stride + (((y & !1) << spread) | (y & 1));
                                record[1] = field(stored[slot][0]);
                                record[3] = field(stored[slot][1]);
                            }
                            record
                        })
                        .collect()
                })
                .collect(),
        })
    }
}

enum TreeLeaves<'a> {
    /// Trace rows: `(eq(r_t, t), β − Σ γ_i σ(byte_i))` from `Q`.
    Trace {
        q: &'a Buffer,
        eq: (&'a Buffer, &'a Buffer, u32),
        n: usize,
    },
    /// Table codes: `(W, β − Σ γ_i σ(code_i))`.
    Table { w: &'a Buffer },
}
