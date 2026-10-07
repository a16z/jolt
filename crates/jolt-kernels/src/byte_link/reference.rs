//! The byte link on the host: every message computed directly from its
//! definition. On a valid trace (slots 30 and 31 zero) the Metal prover sends
//! the same messages, so this prover is its oracle.

use jolt_claims::protocols::jolt::lattice::byte_link::{
    ByteLinkBatch, ByteLinkInputs, HistogramGroup, HistogramQuery, BYTE_BITS, BYTE_LINK_PACKS,
};
use jolt_claims::protocols::jolt::lattice::geometry::BalancedIncChunking;
use jolt_claims::protocols::jolt::lattice::ByteTraceLayoutPlan;
use jolt_claims::protocols::jolt::JoltCommittedPolynomial as Poly;
use jolt_field::JoltField;
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_verifier::stages::byte_link::{ByteLinkCompression, ByteLinkOpening, ByteLinkOpenings};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{
    ByteLinkDraw, ByteLinkKernel, ByteLinkMessage, ByteLinkRun, ByteLinkTranscript, HistogramTables,
};
use crate::KernelError;

const STORED_HEIGHT: usize = 3;

/// The byte trace `Q`: every slot of `plan` over its `2^log_t` cycles, slot `c`
/// of cycle `t` at `bytes[c · 2^log_t + t]`. The caller guarantees that every
/// slot is zero from cycle `active_rows` on.
#[derive(Clone, Copy, Debug)]
pub struct ByteTrace<'a> {
    pub plan: &'a ByteTraceLayoutPlan,
    pub bytes: &'a [i8],
    pub active_rows: usize,
}

/// The host prover as a backend's [`ByteLinkKernel`].
pub struct ReferenceByteLink;

impl<F: JoltField> ByteLinkKernel<F> for ReferenceByteLink {
    fn histograms<'a>(
        &self,
        trace: ByteTrace<'a>,
        inputs: &ByteLinkInputs<F>,
    ) -> Result<Box<dyn ByteLinkRun<F> + 'a>, KernelError<F>> {
        let histograms = histograms(&trace, inputs)?;
        Ok(Box::new(Run { trace, histograms }))
    }
}

/// The host prover over `trace` with `histograms` as `W`.
pub struct Run<'a, F> {
    pub trace: ByteTrace<'a>,
    pub histograms: Histograms<F>,
}

impl<F: JoltField> ByteLinkRun<F> for Run<'_, F> {
    fn tables(&self) -> HistogramTables<'_, F> {
        HistogramTables::Host(&self.histograms)
    }

    fn prove(
        self: Box<Self>,
        inputs: &ByteLinkInputs<F>,
        compression: &ByteLinkCompression<F>,
        mut transcript: &mut dyn ByteLinkTranscript<F>,
    ) -> Result<ByteLinkOpenings<F>, KernelError<F>> {
        prove(
            &self.trace,
            &self.histograms,
            inputs,
            compression,
            &mut transcript,
        )
    }
}

/// `W` of every pack in pack order: `W_j(h) = Σ_{t : h_j(t) = h} eq(r, t)` over
/// the pack's table index ([`HistogramGroup::table_index`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Histograms<F> {
    pub tables: Vec<Vec<F>>,
}

impl<F> Histograms<F> {
    /// The tables `group` commits, in group order.
    pub fn group(&self, group: HistogramGroup) -> &[Vec<F>] {
        &self.tables[group.packs()]
    }
}

/// The eq-weighted tuple histograms of `trace` at the stage-6b cycle point.
pub fn histograms<F: JoltField>(
    trace: &ByteTrace<'_>,
    inputs: &ByteLinkInputs<F>,
) -> Result<Histograms<F>, KernelError<F>> {
    let columns = Columns::new(trace, inputs)?;
    let eq_r = EqPolynomial::<F>::evals(inputs.cycle_point(), None);
    let tables = collect(BYTE_LINK_PACKS.len(), |pack| {
        let group = HistogramGroup::of_pack(pack);
        let mut table = vec![F::zero(); 1 << group.num_vars()];
        for (t, eq) in eq_r.iter().enumerate() {
            table[group.table_index(columns.codes(pack, t))] += *eq;
        }
        table
    });
    Ok(Histograms { tables })
}

/// Everything after the `W` commitments and the compression challenges: the
/// fraction trees of every pack, the three GKR batches, the histogram query
/// reductions and the `Q` reduction.
pub fn prove<F: JoltField>(
    trace: &ByteTrace<'_>,
    histograms: &Histograms<F>,
    inputs: &ByteLinkInputs<F>,
    compression: &ByteLinkCompression<F>,
    transcript: &mut impl ByteLinkTranscript<F>,
) -> Result<ByteLinkOpenings<F>, KernelError<F>> {
    let columns = Columns::new(trace, inputs)?;
    let shaped = histograms.tables.len() == BYTE_LINK_PACKS.len()
        && histograms
            .tables
            .iter()
            .enumerate()
            .all(|(pack, table)| table.len() == 1 << HistogramGroup::of_pack(pack).num_vars());
    if !shaped {
        return Err(geometry(
            "the histograms do not have the link's table shapes",
        ));
    }
    let denominators = Denominators::new(compression);
    let trace_trees = Trees::new(
        ByteLinkBatch::Trace,
        TraceLeaves {
            eq_r: EqPolynomial::evals(inputs.cycle_point(), None),
            columns: &columns,
            denominators: &denominators,
        },
    );
    let [triples, ram] = HistogramGroup::ALL.map(|group| {
        Trees::new(
            match group {
                HistogramGroup::Triples => ByteLinkBatch::Triples,
                HistogramGroup::Ram => ByteLinkBatch::Ram,
            },
            TableLeaves {
                group,
                tables: histograms.group(group),
                denominators: &denominators,
            },
        )
    });
    let roots = [trace_trees.roots(), triples.roots(), ram.roots()].concat();
    transcript.append(ByteLinkMessage::Roots(&roots));

    let (z, trace_leaves) = trace_trees.gkr(transcript);
    let (y, triple_leaves) = triples.gkr(transcript);
    let (y_ram, ram_leaves) = ram.gkr(transcript);
    let queries = inputs.histogram_queries();
    let triples = query(
        transcript,
        HistogramGroup::Triples,
        histograms,
        &queries,
        &y,
        &triple_leaves,
    );
    let ram = query(
        transcript,
        HistogramGroup::Ram,
        histograms,
        &queries,
        &y_ram,
        &ram_leaves,
    );
    let source = source(transcript, &columns, inputs, compression, &z, &trace_leaves);
    Ok(ByteLinkOpenings {
        triples,
        ram,
        source,
    })
}

fn geometry<F: JoltField>(reason: impl Into<String>) -> KernelError<F> {
    KernelError::InvalidGeometry {
        reason: reason.into(),
    }
}

struct Columns<'a> {
    log_rows: usize,
    slots: Vec<&'a [i8]>,
    packs: Vec<[&'a [i8]; 3]>,
    /// In [`BalancedIncChunking::column_place_values`] order: the digits, then
    /// the carry.
    increment: Vec<&'a [i8]>,
    zero_slots: [&'a [i8]; 2],
    chunking: BalancedIncChunking,
}

impl<'a> Columns<'a> {
    fn new<F: JoltField>(
        trace: &ByteTrace<'a>,
        inputs: &ByteLinkInputs<F>,
    ) -> Result<Self, KernelError<F>> {
        let packing = trace.plan.packing();
        let log_rows = packing.logical_num_vars();
        if trace.bytes.len() != packing.ids().len() << log_rows {
            return Err(geometry(format!(
                "a byte trace of {} slots over 2^{log_rows} cycles has {} bytes",
                packing.ids().len(),
                trace.bytes.len()
            )));
        }
        if inputs.cycle_point().len() != log_rows {
            return Err(geometry(format!(
                "the stage-6b cycle point has {} coordinates, the byte trace 2^{log_rows} cycles",
                inputs.cycle_point().len()
            )));
        }
        let slots = trace.bytes.chunks_exact(1 << log_rows).collect::<Vec<_>>();
        let column = |column: Poly| {
            packing
                .slot_index(&column)
                .map(|slot| slots[slot])
                .ok_or_else(|| geometry(format!("the byte trace has no {column:?} slot")))
        };
        let packs = BYTE_LINK_PACKS
            .iter()
            .map(|pack| Ok([column(pack[0])?, column(pack[1])?, column(pack[2])?]))
            .collect::<Result<_, KernelError<F>>>()?;
        let chunking =
            BalancedIncChunking::new(BYTE_BITS).map_err(|error| geometry(error.to_string()))?;
        let increment = (0..chunking.chunk_count())
            .map(Poly::BalancedIncDigit)
            .chain([Poly::BalancedIncCarry])
            .map(&column)
            .collect::<Result<_, _>>()?;
        let zero_slots = [column(Poly::ZeroSlot(0))?, column(Poly::ZeroSlot(1))?];
        Ok(Self {
            log_rows,
            slots,
            packs,
            increment,
            zero_slots,
            chunking,
        })
    }

    fn codes(&self, pack: usize, t: usize) -> [u8; 3] {
        self.packs[pack].map(|column| column[t] as u8)
    }
}

fn sigma<F: JoltField>(byte: i8) -> F {
    F::from_i64(i64::from(byte))
}

/// Each pack's leaf denominator `β − Σ_i γ_i σ(code_i)` by tuple position:
/// `β − γ_0 σ` at position 0, `γ_i σ` at positions 1 and 2.
pub(crate) struct Denominators<F> {
    tables: Vec<[[F; 256]; 3]>,
}

impl<F: JoltField> Denominators<F> {
    pub(crate) fn new(compression: &ByteLinkCompression<F>) -> Self {
        let tables = compression
            .gamma
            .iter()
            .map(|gamma| {
                std::array::from_fn(|position| {
                    std::array::from_fn(|code| {
                        let term = gamma[position] * sigma::<F>(code as u8 as i8);
                        if position == 0 {
                            compression.beta - term
                        } else {
                            term
                        }
                    })
                })
            })
            .collect();
        Self { tables }
    }

    fn at(&self, pack: usize, [c0, c1, c2]: [u8; 3]) -> F {
        let [t0, t1, t2] = &self.tables[pack];
        t0[c0 as usize] - t1[c1 as usize] - t2[c2 as usize]
    }

    /// Every table, `[pack][position][code]`: the Metal shader's layout.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(crate) fn flat(&self) -> &[F] {
        self.tables.as_flattened().as_flattened()
    }
}

trait Leaves<F>: Sync {
    fn trees(&self) -> usize;
    fn log_leaves(&self) -> usize;
    fn leaf(&self, tree: usize, index: usize) -> (F, F);
}

struct TraceLeaves<'a, F> {
    eq_r: Vec<F>,
    columns: &'a Columns<'a>,
    denominators: &'a Denominators<F>,
}

impl<F: JoltField> Leaves<F> for TraceLeaves<'_, F> {
    fn trees(&self) -> usize {
        BYTE_LINK_PACKS.len()
    }

    fn log_leaves(&self) -> usize {
        self.columns.log_rows
    }

    fn leaf(&self, tree: usize, t: usize) -> (F, F) {
        let codes = self.columns.codes(tree, t);
        (self.eq_r[t], self.denominators.at(tree, codes))
    }
}

struct TableLeaves<'a, F> {
    group: HistogramGroup,
    tables: &'a [Vec<F>],
    denominators: &'a Denominators<F>,
}

impl<F: JoltField> Leaves<F> for TableLeaves<'_, F> {
    fn trees(&self) -> usize {
        self.group.packs().len()
    }

    fn log_leaves(&self) -> usize {
        self.group.num_vars()
    }

    fn leaf(&self, tree: usize, h: usize) -> (F, F) {
        let pack = self.group.packs().start + tree;
        let codes = self.group.table_codes(h);
        (self.tables[tree][h], self.denominators.at(pack, codes))
    }
}

fn gate<F: JoltField>((p0, b0): (F, F), (p1, b1): (F, F)) -> (F, F) {
    (p0 * b1 + p1 * b0, b0 * b1)
}

fn node_from_leaves<F: JoltField>(
    leaves: &impl Leaves<F>,
    tree: usize,
    height: usize,
    index: usize,
) -> (F, F) {
    if height == 0 {
        leaves.leaf(tree, index)
    } else {
        gate(
            node_from_leaves(leaves, tree, height - 1, 2 * index),
            node_from_leaves(leaves, tree, height - 1, 2 * index + 1),
        )
    }
}

struct Trees<F, L> {
    batch: ByteLinkBatch,
    leaves: L,
    /// `levels[tree][height]`, empty below `STORED_HEIGHT` and once consumed.
    levels: Vec<Vec<Vec<(F, F)>>>,
}

impl<F: JoltField, L: Leaves<F>> Trees<F, L> {
    fn new(batch: ByteLinkBatch, leaves: L) -> Self {
        let log_leaves = leaves.log_leaves();
        let lowest = STORED_HEIGHT.min(log_leaves);
        let levels = (0..leaves.trees())
            .map(|tree| {
                let mut levels = vec![Vec::new(); log_leaves + 1];
                levels[lowest] = collect(1 << (log_leaves - lowest), |index| {
                    node_from_leaves(&leaves, tree, lowest, index)
                });
                for height in lowest..log_leaves {
                    let above = {
                        let below = &levels[height];
                        collect(below.len() / 2, |index| {
                            gate(below[2 * index], below[2 * index + 1])
                        })
                    };
                    levels[height + 1] = above;
                }
                levels
            })
            .collect();
        Self {
            batch,
            leaves,
            levels,
        }
    }

    fn roots(&self) -> Vec<(F, F)> {
        let top = self.leaves.log_leaves();
        self.levels.iter().map(|levels| levels[top][0]).collect()
    }

    fn node(&self, tree: usize, height: usize, index: usize) -> (F, F) {
        match self.levels[tree].get(height) {
            Some(level) if !level.is_empty() => level[index],
            _ => node_from_leaves(&self.leaves, tree, height, index),
        }
    }

    /// `[P_0, B_0, P_1, B_1]` of the children of parent `x`, at `height`.
    fn children(&self, tree: usize, height: usize, x: usize) -> [F; 4] {
        let (p0, b0) = self.node(tree, height, 2 * x);
        let (p1, b1) = self.node(tree, height, 2 * x + 1);
        [p0, b0, p1, b1]
    }

    /// The batched GKR from the roots down; returns the leaf point and the
    /// leaf claims of every tree.
    fn gkr(mut self, transcript: &mut impl ByteLinkTranscript<F>) -> (Vec<F>, Vec<(F, F)>) {
        let batch = self.batch;
        let trees = self.leaves.trees();
        let mut point = Vec::new();
        let mut claims = self.roots();
        for layer in 0..self.leaves.log_leaves() {
            let height = self.leaves.log_leaves() - layer - 1;
            transcript.append(ByteLinkMessage::LayerClaims {
                batch,
                layer,
                point: &point,
                claims: &claims,
            });
            let weights = transcript
                .challenges(ByteLinkDraw::LayerWeights { batch, layer }, 2 * trees)
                .chunks_exact(2)
                .map(|weights| (weights[0], weights[1]))
                .collect::<Vec<_>>();
            let gate_sum = |[p0, b0, p1, b1]: [F; 4], (w_p, w_b): (F, F)| {
                w_p * (p0 * b1 + p1 * b0) + w_b * b0 * b1
            };
            let rho = point.iter().rev().copied().collect::<Vec<_>>();
            let mut prefix = F::one();
            let mut challenges = Vec::with_capacity(layer);
            let mut bound: Vec<Vec<[F; 4]>> = Vec::new();
            for (round, &rho_i) in rho.iter().enumerate() {
                let children = |tree: usize, x: usize| {
                    if bound.is_empty() {
                        self.children(tree, height, x)
                    } else {
                        bound[tree][x]
                    }
                };
                let pairs = 1 << (layer - round - 1);
                let eq = eq_lsb(&rho[round + 1..]);
                let [t0, t1, t2] = sum(pairs, |y| {
                    let mut terms = [F::zero(); 3];
                    for (tree, &w) in weights.iter().enumerate() {
                        let (lo, hi) = (children(tree, 2 * y), children(tree, 2 * y + 1));
                        let slope = std::array::from_fn(|k| hi[k] - lo[k]);
                        terms[0] += gate_sum(lo, w);
                        terms[1] += gate_sum(hi, w);
                        terms[2] += gate_sum(slope, w);
                    }
                    terms.map(|term| eq[y] * term)
                });
                let poly = layer_round(prefix, rho_i, t0, t1, t2);
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
                prefix *= F::one() - rho_i - r + (rho_i + rho_i) * r;
                challenges.push(r);
                let next = (0..trees)
                    .map(|tree| {
                        collect(pairs, |y| {
                            let (lo, hi) = (children(tree, 2 * y), children(tree, 2 * y + 1));
                            std::array::from_fn(|k| lo[k] + r * (hi[k] - lo[k]))
                        })
                    })
                    .collect();
                bound = next;
            }
            let children = (0..trees)
                .map(|tree| {
                    if bound.is_empty() {
                        self.children(tree, height, 0)
                    } else {
                        bound[tree][0]
                    }
                })
                .collect::<Vec<_>>();
            transcript.append(ByteLinkMessage::Children {
                batch,
                layer,
                children: &children,
            });
            let mu = transcript.challenges(ByteLinkDraw::ChildSelector { batch, layer }, 1)[0];
            claims = children
                .iter()
                .map(|c| (c[0] + mu * (c[2] - c[0]), c[1] + mu * (c[3] - c[1])))
                .collect();
            point = challenges.into_iter().rev().chain([mu]).collect();
            for levels in &mut self.levels {
                levels.truncate(height);
            }
        }
        (point, claims)
    }
}

/// `s(X) = prefix · eq(ρ, X) · t(X)` from `t(0)`, `t(1)` and `t`'s `X²`
/// coefficient.
fn layer_round<F: JoltField>(prefix: F, rho: F, t0: F, t1: F, t2: F) -> UnivariatePoly<F> {
    let a = prefix * (F::one() - rho);
    let b = prefix * (rho + rho - F::one());
    let t1 = t1 - t0 - t2;
    UnivariatePoly::new(vec![a * t0, a * t1 + b * t0, a * t2 + b * t1, b * t2])
}

/// `eq(point, x)` for every `x`, bit `i` of `x` paired with `point[i]`.
fn eq_lsb<F: JoltField>(point: &[F]) -> Vec<F> {
    let reversed = point.iter().rev().copied().collect::<Vec<_>>();
    EqPolynomial::evals(&reversed, None)
}

/// The rounds of `Σ_x Σ_k f_k(x) g_k(x)` over `2^num_vars` points, low bit
/// first: `entry(k, x)` is `(f_k(x), g_k(x))` before the first round, and
/// `send` emits a round and returns its challenge. Returns the challenges and
/// every `(f_k, g_k)` at the final point.
fn product_rounds<F, T>(
    transcript: &mut T,
    num_vars: usize,
    products: usize,
    entry: impl Fn(usize, usize) -> (F, F) + Sync,
    send: impl Fn(&mut T, usize, &UnivariatePoly<F>) -> F,
) -> (Vec<F>, Vec<(F, F)>)
where
    F: JoltField,
{
    let mut challenges = Vec::with_capacity(num_vars);
    let mut bound: Vec<Vec<(F, F)>> = Vec::new();
    for round in 0..num_vars {
        let at = |k: usize, x: usize| {
            if bound.is_empty() {
                entry(k, x)
            } else {
                bound[k][x]
            }
        };
        let pairs = 1 << (num_vars - round - 1);
        let [s0, s1, leading] = sum(pairs, |y| {
            let mut terms = [F::zero(); 3];
            for k in 0..products {
                let ((f0, g0), (f1, g1)) = (at(k, 2 * y), at(k, 2 * y + 1));
                terms[0] += f0 * g0;
                terms[1] += f1 * g1;
                terms[2] += (f1 - f0) * (g1 - g0);
            }
            terms
        });
        let poly = UnivariatePoly::new(vec![s0, s1 - s0 - leading, leading]);
        let r = send(transcript, round, &poly);
        challenges.push(r);
        let next = (0..products)
            .map(|k| {
                collect(pairs, |y| {
                    let ((f0, g0), (f1, g1)) = (at(k, 2 * y), at(k, 2 * y + 1));
                    (f0 + r * (f1 - f0), g0 + r * (g1 - g0))
                })
            })
            .collect();
        bound = next;
    }
    let finals = if bound.is_empty() {
        (0..products).map(|k| entry(k, 0)).collect()
    } else {
        bound.iter().map(|values| values[0]).collect()
    };
    (challenges, finals)
}

/// `group`'s histogram query reduction: per pack its marginal queries, then its
/// table leaf `W(y)`, reduced to one point of every `W` of the group.
fn query<F: JoltField>(
    transcript: &mut impl ByteLinkTranscript<F>,
    group: HistogramGroup,
    histograms: &Histograms<F>,
    queries: &[HistogramQuery<F>],
    leaf_point: &[F],
    leaves: &[(F, F)],
) -> ByteLinkOpening<F> {
    let mut values = Vec::new();
    let mut terms = Vec::new();
    for (tree, (pack, &(leaf, _))) in group.packs().zip(leaves).enumerate() {
        for query in queries.iter().filter(|query| query.pack == pack) {
            values.push(query.value);
            terms.push((tree, query.point.as_slice()));
        }
        values.push(leaf);
        terms.push((tree, leaf_point));
    }
    transcript.append(ByteLinkMessage::QueryValues {
        group,
        values: &values,
    });
    let weights = transcript.challenges(ByteLinkDraw::QueryWeights { group }, values.len());
    let mut byte_eqs = vec![Vec::new(); leaves.len()];
    for (&(tree, point), &weight) in terms.iter().zip(&weights) {
        let bytes = point
            .chunks(BYTE_BITS)
            .map(|chunk| EqPolynomial::<F>::evals(chunk, None))
            .collect::<Vec<_>>();
        byte_eqs[tree].push((weight, bytes));
    }
    let query_weight = |tree: usize, h: usize| {
        let codes = group.table_codes(h);
        byte_eqs[tree]
            .iter()
            .map(|(weight, bytes)| {
                bytes
                    .iter()
                    .zip(codes)
                    .fold(*weight, |product, (eq, code)| product * eq[code as usize])
            })
            .sum::<F>()
    };
    let tables = histograms.group(group);
    let (challenges, finals) = product_rounds(
        transcript,
        group.num_vars(),
        tables.len(),
        |tree, h| (tables[tree][h], query_weight(tree, h)),
        |transcript, round, poly| {
            transcript.append(ByteLinkMessage::QueryRound { group, round, poly });
            transcript.challenges(ByteLinkDraw::QueryChallenge { group, round }, 1)[0]
        },
    );
    let values = finals.iter().map(|&(w, _)| w).collect::<Vec<_>>();
    transcript.append(ByteLinkMessage::QueryFinals {
        group,
        values: &values,
    });
    ByteLinkOpening {
        point: challenges.into_iter().rev().collect(),
        values,
    }
}

/// The `Q` reduction:
/// `Σ_j α_j (β − b_j) + α_F F(r) = Σ_t eq(z, t) G(t) + eq(r, t) F'(t) + eq(θ, t) Z(t)`,
/// `G` the `α_j γ`-weighted pack columns, `F'` the `α_F`-weighted fused
/// increment, `Z` the weighted zero slots; then every column at its final point.
fn source<F: JoltField>(
    transcript: &mut impl ByteLinkTranscript<F>,
    columns: &Columns<'_>,
    inputs: &ByteLinkInputs<F>,
    compression: &ByteLinkCompression<F>,
    leaf_point: &[F],
    leaves: &[(F, F)],
) -> ByteLinkOpening<F> {
    let log_rows = columns.log_rows;
    let denominators = leaves.iter().map(|&(_, b)| b).collect::<Vec<_>>();
    transcript.append(ByteLinkMessage::SourceClaims {
        point: leaf_point,
        denominators: &denominators,
    });
    let theta = transcript.challenges(ByteLinkDraw::ZeroSlotPoint, log_rows);
    let weights = transcript.challenges(ByteLinkDraw::SourceWeights, 10);
    let (pack_weights, rest) = weights.split_at(BYTE_LINK_PACKS.len());
    let (fused_weight, zero_weights) = (rest[0], &rest[1..]);
    let packs = collect(1 << log_rows, |t| {
        pack_weights
            .iter()
            .zip(&compression.gamma)
            .zip(&columns.packs)
            .map(|((weight, gamma), pack)| {
                *weight
                    * gamma
                        .iter()
                        .zip(pack)
                        .map(|(gamma, column)| *gamma * sigma::<F>(column[t]))
                        .sum::<F>()
            })
            .sum::<F>()
    });
    let places = columns
        .chunking
        .column_place_values::<F>()
        .collect::<Vec<_>>();
    let fused = collect(1 << log_rows, |t| {
        fused_weight
            * columns
                .increment
                .iter()
                .zip(&places)
                .map(|(column, place)| *place * sigma::<F>(column[t]))
                .sum::<F>()
    });
    let zeros = collect(1 << log_rows, |t| {
        zero_weights
            .iter()
            .zip(&columns.zero_slots)
            .map(|(weight, column)| *weight * sigma::<F>(column[t]))
            .sum::<F>()
    });
    let eqs = [leaf_point, inputs.cycle_point(), theta.as_slice()]
        .map(|point| EqPolynomial::<F>::evals(point, None));
    let products = [packs, fused, zeros];
    let (challenges, _) = product_rounds(
        transcript,
        log_rows,
        products.len(),
        |k, t| (eqs[k][t], products[k][t]),
        |transcript, round, poly| {
            transcript.append(ByteLinkMessage::SourceRound { round, poly });
            transcript.challenges(ByteLinkDraw::SourceChallenge { round }, 1)[0]
        },
    );
    let point = challenges.into_iter().rev().collect::<Vec<_>>();
    let eq_x = EqPolynomial::<F>::evals(&point, None);
    let values = columns
        .slots
        .iter()
        .map(|column| sum(column.len(), |t| [eq_x[t] * sigma::<F>(column[t])])[0])
        .collect::<Vec<_>>();
    transcript.append(ByteLinkMessage::SourceFinals(&values));
    ByteLinkOpening { point, values }
}

fn collect<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    #[cfg(feature = "parallel")]
    let items = (0..n).into_par_iter().map(f).collect();
    #[cfg(not(feature = "parallel"))]
    let items = (0..n).map(f).collect();
    items
}

fn sum<F: JoltField, const K: usize>(
    n: usize,
    f: impl Fn(usize) -> [F; K] + Sync + Send,
) -> [F; K] {
    let add = |a: [F; K], b: [F; K]| std::array::from_fn(|k| a[k] + b[k]);
    #[cfg(feature = "parallel")]
    let total = (0..n).into_par_iter().map(f).reduce(|| [F::zero(); K], add);
    #[cfg(not(feature = "parallel"))]
    let total = (0..n).map(f).fold([F::zero(); K], add);
    total
}
