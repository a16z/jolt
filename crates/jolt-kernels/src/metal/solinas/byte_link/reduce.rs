//! Degree-two product reductions: the histogram queries of each group and the `Q` point
//! reduction, plus the column evaluations at its final point.

use jolt_field::{Field, One, Ring, Zero};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use metal::{Buffer, ComputeCommandEncoderRef};

use super::{
    gkr::{canonical, eq, eq_lsb, tiling, RoundParams},
    gpu::{
        bind, bytes, field, limbs, view, Gpu, Grid, Recorder, COLUMN_MLE, PRODUCT2_BIND_EVAL,
        PRODUCT2_EVAL, PRODUCT4_BIND_EVAL, PRODUCT4_EVAL, QUERY_BIND, QUERY_EVAL, SOURCE_BIND,
        SOURCE_EVAL, SUMS, SUM_COLUMNS, SUM_PARTIALS, THREADS,
    },
    ByteLinkCompression, ByteLinkDraw, ByteLinkMessage, ByteLinkOpening, ByteLinkProver,
    ByteLinkQueryGroup, ByteLinkSource, ByteLinkStatement, ByteLinkTranscript, Shape, F, HOST_LOG,
    INCREMENT_SLOTS, PACKS, PACK_SLOTS, RAM, RAM_BITS, SLOTS, TRIPLE_BITS,
};
use crate::metal::solinas::{Fp128, MetalError};

/// Rows per threadgroup of the column evaluations: `LINK_THREADS` × 16.
const MLE_LO_BITS: u32 = 12;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawParams {
    pairs: u32,
    groups: u32,
    pairs_per_thread: u32,
    bound: u32,
    out_stride: u32,
    n: u32,
    z_lo_bits: u32,
    r_lo_bits: u32,
    challenge: Fp128,
}

/// `LinkSourceColumns`: the `Q` columns of `G` (pack order) and of the fused increment.
#[repr(C)]
#[derive(Clone, Copy)]
struct SourceColumns {
    g: [u32; 24],
    f: [u32; 12],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MleParams {
    n: u32,
    lo_bits: u32,
    rows_per_thread: u32,
    columns: u32,
}

const _: () = assert!(size_of::<RawParams>() == 48);
const _: () = assert!(size_of::<SourceColumns>() == 144);

/// The two factor arrays of one term of a degree-2 product sum.
type Factors = (Vec<F>, Vec<F>);

/// Arena bytes of a reduction over `entries` records of `width` field elements per pack.
pub(super) fn product_bytes(gpu: &Gpu, packs: usize, entries: usize, width: usize) -> usize {
    gpu.arena_size(packs * (entries / 2) * width * 16)
        + gpu.arena_size(packs * (entries / 4) * width * 16)
}

#[derive(Clone, Copy)]
enum Reduction {
    Query(ByteLinkQueryGroup),
    Source,
}

impl Reduction {
    /// Sends the round polynomial of `s(0)` and leading coefficient `leading` (`s(1)` follows from
    /// the claim) and returns its challenge.
    fn round(
        self,
        transcript: &mut impl ByteLinkTranscript,
        claim: &mut F,
        round: usize,
        s0: F,
        leading: F,
    ) -> F {
        let s1 = *claim - s0;
        let poly = UnivariatePoly::new(vec![s0, s1 - s0 - leading, leading]);
        let draw = match self {
            Self::Query(group) => {
                transcript.append(ByteLinkMessage::QueryRound {
                    group,
                    round,
                    poly: &poly,
                });
                ByteLinkDraw::QueryChallenge { group, round }
            }
            Self::Source => {
                transcript.append(ByteLinkMessage::SourceRound { round, poly: &poly });
                ByteLinkDraw::SourceChallenge { round }
            }
        };
        let r = transcript.challenges(draw, 1)[0];
        *claim = poly.evaluate(r);
        r
    }

    /// Host rounds until every array holds one value; returns their challenges.
    fn finish_on_host(
        self,
        transcript: &mut impl ByteLinkTranscript,
        pairs: &mut [Factors],
        claim: &mut F,
        first_round: usize,
    ) -> Vec<F> {
        let mut challenges = Vec::new();
        while pairs[0].0.len() > 1 {
            let (s0, leading) = pairs
                .iter()
                .flat_map(|(f, g)| f.chunks(2).zip(g.chunks(2)))
                .fold((F::zero(), F::zero()), |(s0, leading), (f, g)| {
                    (s0 + f[0] * g[0], leading + (f[1] - f[0]) * (g[1] - g[0]))
                });
            let r = self.round(
                transcript,
                claim,
                first_round + challenges.len(),
                s0,
                leading,
            );
            for (f, g) in pairs.iter_mut() {
                for array in [f, g] {
                    let half = array.len() / 2;
                    for k in 0..half {
                        array[k] = array[2 * k] + r * (array[2 * k + 1] - array[2 * k]);
                    }
                    array.truncate(half);
                }
            }
            challenges.push(r);
        }
        challenges
    }
}

/// The raw-entry kernels of one reduction and how to bind their inputs.
struct RawEntries<'a> {
    kernels: (&'static str, &'static str),
    params: RawParams,
    bind: &'a dyn Fn(&ComputeCommandEncoderRef),
}

/// The query weights of triple pack `pack`: `U_s[code] = α_s 2^-16 eq(k_c, code)` for its three
/// slots, then `Z_s[code] = eq(y_s, code)` of the leaf point's bytes, `Z_2` carrying `α_y`.
fn query_tables(statement: &ByteLinkStatement<'_>, pack: usize, alpha: &[F], y: &[F]) -> Vec<F> {
    let scale = F::from_u64(1 << 16).inv_or_zero();
    let u = (0..3).flat_map(|s| {
        let a = alpha[s] * scale;
        EqPolynomial::<F>::evals(&statement.address_points[3 * pack + s], None)
            .into_iter()
            .map(move |e| a * e)
    });
    let z = (0..3).flat_map(|s| {
        let a = if s == 2 { alpha[3] } else { F::one() };
        EqPolynomial::<F>::evals(&y[8 * s..8 * s + 8], None)
            .into_iter()
            .map(move |e| a * e)
    });
    u.chain(z).collect()
}

/// `eq(point, y)` at the hypercube point `y`, bit `l` of `y` paired with `point[l]`.
fn eq_at(point: &[F], y: usize) -> F {
    point
        .iter()
        .enumerate()
        .map(|(l, &x)| if y >> l & 1 == 1 { x } else { F::one() - x })
        .product()
}

impl ByteLinkProver {
    /// GPU rounds of a degree-2 reduction over `entries` explicit entries of `2^log` per pack.
    /// Round 0 evaluates the raw entries, round 1 binds them into stored records and evaluates
    /// those, later rounds bind and evaluate stored records. Returns the challenges and the host
    /// arrays per pack after them; entries past `entries` come from `tail`.
    ///
    /// Round 1 is a bind pass plus an evaluate pass, not one fused bind-and-evaluate pass over the
    /// raw entries: on M5 Max with macOS 27.0.1 the fused kernel returned run-to-run different
    /// sums on identical inputs, also in a race-free form whose threads write only their own
    /// outputs, while its `optimizationLevel = size` compile was deterministic, so the defect sits
    /// below the shader source (`/private/tmp/pika-scratch/akita-p0/link-mem/repro/` `run1.log`,
    /// `run2.log`, `run2-size.log`; `link-mem/notes.md` §1).
    #[expect(clippy::too_many_arguments, reason = "one reduction geometry")]
    fn product_gpu(
        &mut self,
        transcript: &mut impl ByteLinkTranscript,
        reduction: Reduction,
        claim: &mut F,
        packs: usize,
        log: u32,
        entries: usize,
        width: usize,
        raw: &RawEntries<'_>,
        tail: &dyn Fn(&[F], usize) -> [F; 4],
    ) -> Result<(Vec<F>, Vec<Factors>), MetalError> {
        let gpu_rounds = (log - HOST_LOG) as usize;
        assert!(gpu_rounds >= 2);
        let (stored_eval, stored_bind_eval) = if width == 2 {
            (PRODUCT2_EVAL, PRODUCT2_BIND_EVAL)
        } else {
            (PRODUCT4_EVAL, PRODUCT4_BIND_EVAL)
        };
        let partials = self.gpu.scratch(packs * tiling(entries / 2).1 * SUMS * 16);
        let sums = self.gpu.scratch(packs * SUMS * 16);
        let scratch = [
            self.gpu.arena(packs * (entries / 2) * width * 16)?,
            self.gpu.arena(packs * (entries / 4) * width * 16)?,
        ];
        let mut challenges = Vec::new();
        let mut stride = entries;
        for i in 0..=gpu_rounds {
            let last = i == gpu_rounds;
            let bound = entries >> i;
            let pairs = if last { bound.div_ceil(2) } else { bound / 2 };
            let (per_thread, groups) = tiling(pairs);
            let input = &scratch[i % 2];
            let output = &scratch[(i + 1) % 2];
            let challenge = limbs(challenges.last().copied().unwrap_or_else(F::zero));
            let stored = RoundParams {
                pairs: pairs as u32,
                groups: groups as u32,
                pairs_per_thread: per_thread as u32,
                in_stride: stride as u32,
                out_stride: bound as u32,
                bound: bound as u32,
                challenge,
                ..RoundParams::default()
            };
            let sum = |r: &mut Recorder| {
                r.dispatch(SUM_PARTIALS, Grid::Groups(packs, 1024), |e| {
                    bind(e, 0, &partials, 0);
                    bind(e, 1, &sums, 0);
                    bytes(e, 2, &(groups as u32));
                });
            };
            let label = match i {
                0 => raw.kernels.0,
                1 => raw.kernels.1,
                _ => stored_bind_eval,
            };
            self.gpu.run(label, |r| {
                match i {
                    0 => {
                        let params = RawParams {
                            pairs: pairs as u32,
                            groups: groups as u32,
                            pairs_per_thread: per_thread as u32,
                            ..raw.params
                        };
                        r.dispatch(raw.kernels.0, Grid::Groups(packs * groups, THREADS), |e| {
                            (raw.bind)(e);
                            bind(e, 7, &partials, 0);
                            bytes(e, 9, &params);
                        });
                    }
                    1 => {
                        let (bind_per_thread, bind_groups) = tiling(bound);
                        let params = RawParams {
                            groups: bind_groups as u32,
                            pairs_per_thread: bind_per_thread as u32,
                            bound: bound as u32,
                            out_stride: bound as u32,
                            challenge,
                            ..raw.params
                        };
                        r.dispatch(
                            raw.kernels.1,
                            Grid::Groups(packs * bind_groups, THREADS),
                            |e| {
                                (raw.bind)(e);
                                bind(e, 8, output, 0);
                                bytes(e, 9, &params);
                            },
                        );
                        if !last {
                            r.dispatch(stored_eval, Grid::Groups(packs * groups, THREADS), |e| {
                                bind(e, 0, output, 0);
                                bind(e, 3, &partials, 0);
                                bytes(
                                    e,
                                    5,
                                    &RoundParams {
                                        in_stride: bound as u32,
                                        ..stored
                                    },
                                );
                            });
                        }
                    }
                    _ => {
                        r.dispatch(
                            stored_bind_eval,
                            Grid::Groups(packs * groups, THREADS),
                            |e| {
                                bind(e, 0, input, 0);
                                bind(e, 3, &partials, 0);
                                bind(e, 4, output, 0);
                                bytes(e, 5, &stored);
                            },
                        );
                    }
                }
                if !last {
                    sum(r);
                }
            })?;
            if i > 0 {
                stride = bound;
            }
            if !last {
                let values = view::<Fp128>(&sums, 0, packs * SUMS);
                let (s0, leading) = (0..packs).fold((F::zero(), F::zero()), |(a, b), pack| {
                    (
                        a + field(values[pack * SUMS]),
                        b + field(values[pack * SUMS + 1]),
                    )
                });
                challenges.push(reduction.round(transcript, claim, i, s0, leading));
            }
        }
        let output = &scratch[(gpu_rounds + 1) % 2];
        let bound = entries >> gpu_rounds;
        let total = 1usize << (log as usize - gpu_rounds);
        let stored = view::<Fp128>(output, 0, packs * stride * width);
        let arrays = (0..packs)
            .flat_map(|pack| {
                let rows = (0..total)
                    .map(|y| {
                        if y < bound {
                            std::array::from_fn(|k| {
                                if k < width {
                                    field(stored[(pack * stride + y) * width + k])
                                } else {
                                    F::zero()
                                }
                            })
                        } else {
                            tail(&challenges, y)
                        }
                    })
                    .collect::<Vec<[F; 4]>>();
                (0..width / 2)
                    .map(|j| {
                        (
                            rows.iter().map(|e| e[2 * j]).collect(),
                            rows.iter().map(|e| e[2 * j + 1]).collect(),
                        )
                    })
                    .collect::<Vec<Factors>>()
            })
            .collect();
        Ok((challenges, arrays))
    }

    /// The query reduction of the six triple histograms: per pack its three marginal values and
    /// its table leaf, against `W` of all six packs at once.
    pub(super) fn triple_queries(
        &mut self,
        transcript: &mut impl ByteLinkTranscript,
        w: &Buffer,
        statement: &ByteLinkStatement<'_>,
        leaf_point: &[F],
        leaves: &[(F, F)],
    ) -> Result<ByteLinkOpening, MetalError> {
        let group = ByteLinkQueryGroup::Triples;
        let scale = F::from_u64(1 << 16).inv_or_zero();
        let values = (0..RAM)
            .flat_map(|pack| {
                (0..3)
                    .map(move |s| statement.one_hot_claims[3 * pack + s] * scale)
                    .chain([leaves[pack].0])
            })
            .collect::<Vec<_>>();
        transcript.append(ByteLinkMessage::QueryValues {
            group,
            values: &values,
        });
        let alpha = transcript.challenges(ByteLinkDraw::QueryWeights { group }, values.len());
        let mut claim = values.iter().zip(&alpha).map(|(&v, &a)| v * a).sum();
        let y = canonical(leaf_point);
        let tables = (0..RAM)
            .flat_map(|pack| query_tables(statement, pack, &alpha[4 * pack..4 * pack + 4], &y))
            .collect::<Vec<_>>();
        let tables = self.gpu.fields(&tables);
        let bind_query = |e: &ComputeCommandEncoderRef| {
            bind(e, 0, w, 0);
            bind(e, 1, &tables, 0);
        };
        let raw = RawEntries {
            kernels: (QUERY_EVAL, QUERY_BIND),
            params: RawParams::default(),
            bind: &bind_query,
        };
        let reduction = Reduction::Query(group);
        let (mut point, mut arrays) = self.product_gpu(
            transcript,
            reduction,
            &mut claim,
            RAM,
            TRIPLE_BITS,
            1 << TRIPLE_BITS,
            2,
            &raw,
            &|_, _| [F::zero(); 4],
        )?;
        let first = point.len();
        point.extend(reduction.finish_on_host(transcript, &mut arrays, &mut claim, first));
        let finals = arrays.iter().map(|(f, _)| f[0]).collect::<Vec<_>>();
        transcript.append(ByteLinkMessage::QueryFinals {
            group,
            values: &finals,
        });
        Ok(ByteLinkOpening {
            point: canonical(&point),
            values: finals,
        })
    }

    /// The degree-2 reduction of every claim on `Q` to one point:
    /// `Σ_j α_j (β − B_j(z)) + α_F F(r) = Σ_t eq(z, t) G(t) + eq(r, t) F'(t)`, `G` the
    /// `α_j γ`-weighted pack columns, `F'` the `α_F`-weighted fused increment. The zero slots'
    /// `eq(θ, ·)` term is zero on every row of a valid source ([`ByteLinkSource`]), so no round
    /// computes it; their terminal values are still evaluated from `Q`.
    #[expect(clippy::too_many_arguments, reason = "the reduction's fixed claims")]
    pub(super) fn source_reduction(
        &mut self,
        transcript: &mut impl ByteLinkTranscript,
        source: &ByteLinkSource<'_>,
        shape: Shape,
        statement: &ByteLinkStatement<'_>,
        compression: &ByteLinkCompression,
        leaf_point: &[F],
        leaves: &[(F, F)],
    ) -> Result<ByteLinkOpening, MetalError> {
        let z = leaf_point;
        let denominators = leaves.iter().map(|&(_, b)| b).collect::<Vec<_>>();
        transcript.append(ByteLinkMessage::SourceClaims {
            point: &canonical(z),
            denominators: &denominators,
        });
        let _ = transcript.challenges(ByteLinkDraw::ZeroSlotPoint, shape.log_n as usize);
        let alpha = transcript.challenges(ByteLinkDraw::SourceWeights, 10);
        let alpha_f = alpha[PACKS];
        let mut claim = denominators
            .iter()
            .zip(&alpha)
            .map(|(&b, &a)| a * (compression.beta - b))
            .sum::<F>()
            + alpha_f * statement.fused_increment;
        let sigma = |coefficient: F| {
            (0..256u32).map(move |code| coefficient * F::from_i64(i64::from(code as u8 as i8)))
        };
        let g_tables = alpha[..PACKS]
            .iter()
            .zip(&compression.gamma)
            .flat_map(|(&a, gamma)| gamma.map(|g| a * g))
            .flat_map(sigma)
            .collect::<Vec<_>>();
        let place = F::from_u64(256);
        let f_tables = (0..INCREMENT_SLOTS.len())
            .scan(alpha_f, |weight, _| {
                let current = *weight;
                *weight *= place;
                Some(current)
            })
            .flat_map(sigma)
            .collect::<Vec<_>>();
        let (g_tables, f_tables) = (self.gpu.fields(&g_tables), self.gpu.fields(&f_tables));
        let r_t = statement
            .cycle_point
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();
        let lo_bits = shape.log_n as usize / 2;
        let eq_tables = [
            &z[..lo_bits],
            &z[lo_bits..],
            &r_t[..lo_bits],
            &r_t[lo_bits..],
        ]
        .iter()
        .flat_map(|half| eq_lsb(half))
        .collect::<Vec<_>>();
        let eq_tables = self.gpu.fields(&eq_tables);
        let mut columns = SourceColumns {
            g: [0; 24],
            f: [0; 12],
        };
        for (c, &slot) in PACK_SLOTS.iter().flatten().enumerate() {
            columns.g[c] = slot as u32;
        }
        for (c, &slot) in INCREMENT_SLOTS.iter().enumerate() {
            columns.f[c] = slot as u32;
        }
        let bind_source = |e: &ComputeCommandEncoderRef| {
            bind(e, 0, source.bytes, 0);
            bind(e, 1, &g_tables, 0);
            bind(e, 2, &f_tables, 0);
            bind(e, 4, &eq_tables, 0);
            bytes(e, 5, &columns);
        };
        let raw = RawEntries {
            kernels: (SOURCE_EVAL, SOURCE_BIND),
            params: RawParams {
                n: shape.n() as u32,
                z_lo_bits: lo_bits as u32,
                r_lo_bits: lo_bits as u32,
                ..RawParams::default()
            },
            bind: &bind_source,
        };
        let tail = |bound: &[F], y: usize| {
            let g = bound.len();
            [
                eq(&z[..g], bound) * eq_at(&z[g..], y),
                F::zero(),
                eq(&r_t[..g], bound) * eq_at(&r_t[g..], y),
                F::zero(),
            ]
        };
        let reduction = Reduction::Source;
        let (mut point, mut arrays) = self.product_gpu(
            transcript,
            reduction,
            &mut claim,
            1,
            shape.log_n,
            shape.explicit,
            4,
            &raw,
            &tail,
        )?;
        let first = point.len();
        point.extend(reduction.finish_on_host(transcript, &mut arrays, &mut claim, first));
        let values = self.column_mle(source, shape, &point)?;
        transcript.append(ByteLinkMessage::SourceFinals(&values));
        Ok(ByteLinkOpening {
            point: canonical(&point),
            values,
        })
    }

    /// `D̃_c(point)` of every column of `Q`, `point` in internal order.
    fn column_mle(
        &mut self,
        source: &ByteLinkSource<'_>,
        shape: Shape,
        point: &[F],
    ) -> Result<Vec<F>, MetalError> {
        let lo = MLE_LO_BITS as usize;
        let eq_lo = self.gpu.fields(&eq_lsb(&point[..lo]));
        let eq_hi = self.gpu.fields(&eq_lsb(&point[lo..]));
        let blocks = shape.explicit.div_ceil(1 << MLE_LO_BITS);
        let partials = self.gpu.scratch(blocks * SLOTS * 16);
        let sums = self.gpu.scratch(SLOTS * 16);
        let params = MleParams {
            n: shape.n() as u32,
            lo_bits: MLE_LO_BITS,
            rows_per_thread: (1 << MLE_LO_BITS) / THREADS as u32,
            columns: SLOTS as u32,
        };
        self.gpu.run("byte link column evaluations", |r| {
            r.dispatch(COLUMN_MLE, Grid::Groups(blocks, THREADS), |e| {
                bind(e, 0, source.bytes, 0);
                bind(e, 1, &eq_lo, 0);
                bind(e, 2, &eq_hi, 0);
                bind(e, 3, &partials, 0);
                bytes(e, 4, &params);
            });
            r.dispatch(SUM_COLUMNS, Grid::Groups(SLOTS, 1024), |e| {
                bind(e, 0, &partials, 0);
                bind(e, 1, &sums, 0);
                bytes(e, 2, &[blocks as u32, SLOTS as u32]);
            });
        })?;
        Ok(view::<Fp128>(&sums, 0, SLOTS)
            .iter()
            .map(|&v| field(v))
            .collect())
    }
}

/// The RAM histogram's query reduction, on the host: `2^17` cells, its two marginal values and
/// its table leaf.
pub(super) fn ram_query(
    transcript: &mut impl ByteLinkTranscript,
    w: &Buffer,
    statement: &ByteLinkStatement<'_>,
    leaf_point: &[F],
    leaf: (F, F),
) -> ByteLinkOpening {
    let group = ByteLinkQueryGroup::Ram;
    let scale = F::from_u64(1 << 8).inv_or_zero();
    let values = [
        statement.one_hot_claims[18] * scale,
        statement.one_hot_claims[19] * scale,
        leaf.0,
    ];
    transcript.append(ByteLinkMessage::QueryValues {
        group,
        values: &values,
    });
    let alpha = transcript.challenges(ByteLinkDraw::QueryWeights { group }, values.len());
    let mut claim = values.iter().zip(&alpha).map(|(&v, &a)| v * a).sum();
    let y = canonical(leaf_point);
    let k0 = EqPolynomial::<F>::evals(&statement.address_points[18], None);
    let k1 = EqPolynomial::<F>::evals(&statement.address_points[19], None);
    let y0 = EqPolynomial::<F>::evals(&y[..8], None);
    let y1 = EqPolynomial::<F>::evals(&y[8..16], None);
    let y2 = [F::one() - y[16], y[16]];
    let omega = (0..1usize << RAM_BITS)
        .map(|h| {
            let (b0, b1, a) = (h >> 9, (h >> 1) & 255, h & 1);
            let marginal = if a == 1 {
                (alpha[0] * k0[b0] + alpha[1] * k1[b1]) * scale
            } else {
                F::zero()
            };
            marginal + alpha[2] * y0[b0] * y1[b1] * y2[a]
        })
        .collect();
    let cells = view::<Fp128>(w, RAM << TRIPLE_BITS, 1 << RAM_BITS)
        .iter()
        .map(|&v| field(v))
        .collect();
    let mut pairs = [(cells, omega)];
    let point = Reduction::Query(group).finish_on_host(transcript, &mut pairs, &mut claim, 0);
    let finals = vec![pairs[0].0[0]];
    transcript.append(ByteLinkMessage::QueryFinals {
        group,
        values: &finals,
    });
    ByteLinkOpening {
        point: canonical(&point),
        values: finals,
    }
}
