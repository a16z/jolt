//! The oracle of the Metal link: a verifier that replays the prover's transcript and checks
//! every protocol relation, plus every opened value recomputed from the source bytes. The GPU
//! tests run at 2^16; the benches are ignored and documented in the module docs.
#![expect(clippy::unwrap_used, reason = "test oracle")]

use std::{
    collections::HashMap,
    fmt::Debug,
    str::FromStr,
    time::{Duration, Instant},
};

use jolt_field::{Field, One, Ring, Zero};
use jolt_poly::EqPolynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use libc::{rusage_info_v4, RUSAGE_INFO_V4};
use metal::{Buffer, MTLResourceOptions};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use rayon::prelude::*;

use super::{
    gkr::LeafClaims,
    gpu::{field, view, Phase},
    sort::W_CELLS,
    ByteLinkBatch, ByteLinkCompression, ByteLinkDraw, ByteLinkHistograms, ByteLinkMessage,
    ByteLinkOpening, ByteLinkOpenings, ByteLinkProver, ByteLinkQueryGroup, ByteLinkSource,
    ByteLinkStatement, ByteLinkTranscript, F, INCREMENT_SLOTS, PACKS, PACK_SLOTS, RAM, RAM_BITS,
    SLOTS, TRIPLE_BITS,
};
use crate::metal::solinas::{Fp128, SolinasMetal};

type T = Blake2bTranscript<F>;

/// Executed rows of the 2^29 target trace.
const U29: usize = 402_654_183;

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn row_hash(seed: u64, column: u64, t: usize) -> u64 {
    let mut state = seed ^ column.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (t as u64).rotate_left(29);
    splitmix(&mut state)
}

fn sigma(byte: i8) -> F {
    F::from_i64(i64::from(byte))
}

/// The byte trace on the host: column `c` of row `t` at `c · n + t`.
struct Source {
    log_n: u32,
    active: usize,
    q: Vec<i8>,
}

impl Source {
    fn n(&self) -> usize {
        1 << self.log_n
    }

    fn at(&self, slot: usize, t: usize) -> i8 {
        self.q[slot * self.n() + t]
    }

    /// Uniform bytes below `active` (about a third of the rows touch RAM), or with `hot` 256
    /// tuples per pack; increment digits uniform, the carry in {-1, 0, 1}, slots 30 and 31 zero.
    fn synthetic(log_n: u32, active: usize, seed: u64, hot: bool) -> Self {
        let n = 1usize << log_n;
        let mut q = vec![0i8; SLOTS * n];
        let ram_active = |t: usize| row_hash(seed, 1000, t).is_multiple_of(3);
        let tuples = (0..PACKS)
            .map(|pack| {
                (0..256)
                    .map(|k| {
                        let x = row_hash(seed, 2000 + pack as u64, k);
                        if pack == RAM && !x.is_multiple_of(3) {
                            [0, 0, 0]
                        } else {
                            [
                                x as i8,
                                (x >> 16) as i8,
                                if pack == RAM { 1 } else { (x >> 32) as i8 },
                            ]
                        }
                    })
                    .collect::<Vec<[i8; 3]>>()
            })
            .collect::<Vec<_>>();
        q.par_chunks_mut(n).enumerate().for_each(|(slot, column)| {
            let pack_slot = PACK_SLOTS
                .iter()
                .enumerate()
                .find_map(|(pack, slots)| slots.iter().position(|&s| s == slot).map(|i| (pack, i)));
            for (t, value) in column[..active].iter_mut().enumerate() {
                *value = match pack_slot {
                    Some((pack, i)) if hot => {
                        tuples[pack][(row_hash(seed, 3000 + pack as u64, t) & 255) as usize][i]
                    }
                    Some((RAM, 2)) => i8::from(ram_active(t)),
                    Some((RAM, _)) if !ram_active(t) => 0,
                    Some(_) => row_hash(seed, slot as u64, t) as i8,
                    None if slot == 24 => (row_hash(seed, 24, t) % 3) as i8 - 1,
                    None if INCREMENT_SLOTS.contains(&slot) => row_hash(seed, slot as u64, t) as i8,
                    None => 0,
                };
            }
        });
        Self { log_n, active, q }
    }

    /// Rows the semantics single out: both byte extremes, an active RAM access to address bytes
    /// zero, zero non-RAM bytes (row zero of their one-hot tables, inside the claims), and
    /// inactive RAM rows with zero and nonzero address bytes.
    fn with_edge_rows(mut self) -> Self {
        let rows: [(i8, i8, i8); 5] = [
            (-128, -128, 1),
            (127, 127, 1),
            (0, 0, 1),
            (0, 0, 0),
            (-128, 127, 0),
        ];
        let n = self.n();
        for (t, &(byte, ram, activity)) in rows.iter().enumerate() {
            for slots in &PACK_SLOTS[..RAM] {
                for &slot in slots {
                    self.q[slot * n + t] = if t == 2 { 0 } else { byte };
                }
            }
            self.q[27 * n + t] = ram;
            self.q[28 * n + t] = ram;
            self.q[29 * n + t] = activity;
        }
        self
    }

    fn key(&self, pack: usize, t: usize) -> usize {
        let [a, b, c] = PACK_SLOTS[pack].map(|slot| usize::from(self.at(slot, t) as u8));
        if pack == RAM {
            (a << 9) | (b << 1) | c
        } else {
            (a << 16) | (b << 8) | c
        }
    }

    fn fused_increment(&self, t: usize) -> F {
        INCREMENT_SLOTS.iter().rev().fold(F::zero(), |acc, &slot| {
            acc * F::from_u64(256) + sigma(self.at(slot, t))
        })
    }
}

fn random_point(rng: &mut ChaCha20Rng, len: usize) -> Vec<F> {
    (0..len).map(|_| F::random(rng)).collect()
}

/// `eq(point, t)` for every row, `point` canonical MSB-first.
fn eq_rows(point: &[F]) -> Vec<F> {
    EqPolynomial::evals(point, None)
}

/// An honest stage-6b statement for `source`, computed from the bytes.
struct Statement {
    cycle_point: Vec<F>,
    address_points: [[F; 8]; 20],
    one_hot_claims: [F; 20],
    fused_increment: F,
}

fn one_hot_slots() -> impl Iterator<Item = (usize, usize, usize)> {
    PACK_SLOTS
        .iter()
        .enumerate()
        .flat_map(|(pack, slots)| {
            slots
                .iter()
                .enumerate()
                .map(move |(i, &slot)| (pack, i, slot))
        })
        .filter(|&(pack, i, _)| !(pack == RAM && i == 2))
}

impl Statement {
    fn honest(source: &Source, seed: u64) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let cycle_point = random_point(&mut rng, source.log_n as usize);
        let address_points: [[F; 8]; 20] =
            std::array::from_fn(|_| std::array::from_fn(|_| F::random(&mut rng)));
        let low = source.log_n as usize / 2;
        let (hi, lo) = cycle_point.split_at(cycle_point.len() - low);
        let (eq_hi, eq_lo) = (eq_rows(hi), eq_rows(lo));
        let tables = one_hot_slots()
            .zip(&address_points)
            .map(|((pack, _, slot), point)| (pack, slot, EqPolynomial::<F>::evals(point, None)))
            .collect::<Vec<_>>();
        let (one_hot_claims, fused_increment) = (0..source.n())
            .into_par_iter()
            .fold(
                || ([F::zero(); 20], F::zero()),
                |(mut claims, fused), t| {
                    let e = eq_hi[t >> low] * eq_lo[t & ((1 << low) - 1)];
                    for (claim, (pack, slot, table)) in claims.iter_mut().zip(&tables) {
                        if *pack != RAM || source.at(29, t) != 0 {
                            *claim += e * table[usize::from(source.at(*slot, t) as u8)];
                        }
                    }
                    (claims, fused + e * source.fused_increment(t))
                },
            )
            .reduce(
                || ([F::zero(); 20], F::zero()),
                |(a, fa), (b, fb)| (std::array::from_fn(|c| a[c] + b[c]), fa + fb),
            );
        Self {
            cycle_point,
            address_points,
            one_hot_claims,
            fused_increment,
        }
    }

    fn view(&self) -> ByteLinkStatement<'_> {
        ByteLinkStatement {
            cycle_point: &self.cycle_point,
            address_points: &self.address_points,
            one_hot_claims: &self.one_hot_claims,
            fused_increment: self.fused_increment,
        }
    }
}

/// Sparse `W` of every pack: `W_j(h) = Σ_{t: key_j(t) = h} eq(r, t)`.
fn histograms(source: &Source, cycle_point: &[F]) -> Vec<HashMap<usize, F>> {
    let eq_t = eq_rows(cycle_point);
    (0..PACKS)
        .into_par_iter()
        .map(|pack| {
            let mut w = HashMap::new();
            for (t, &e) in eq_t.iter().enumerate() {
                *w.entry(source.key(pack, t)).or_insert_with(F::zero) += e;
            }
            w
        })
        .collect()
}

/// The `W` commitment stand-in: Blake2b over the device cells in 16 MiB chunks.
fn histogram_digest(w: &Buffer) -> Vec<F> {
    let chunks = view::<u8>(w, 0, W_CELLS * 16)
        .par_chunks(1 << 24)
        .map(|chunk| {
            let mut transcript = T::new(b"byte-link-test-histogram-chunk");
            transcript.append_bytes(chunk);
            transcript.state()
        })
        .collect::<Vec<_>>();
    let mut transcript = T::new(b"byte-link-test-histograms");
    for chunk in chunks {
        transcript.append_bytes(&chunk);
    }
    vec![transcript.challenge_scalar()]
}

/// The stage stand-in: labels every message and draw and keeps an owned log.
struct Recorder {
    transcript: T,
    log: Vec<(Kind, Vec<F>)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Kind {
    Roots,
    LayerClaims(ByteLinkBatch, usize),
    LayerRound(ByteLinkBatch, usize, usize),
    Children(ByteLinkBatch, usize),
    QueryValues(ByteLinkQueryGroup),
    QueryRound(ByteLinkQueryGroup, usize),
    QueryFinals(ByteLinkQueryGroup),
    SourceClaims,
    SourceRound(usize),
    SourceFinals,
}

fn absorb(transcript: &mut T, kind: &Kind, values: &[F]) {
    transcript.append_bytes(format!("{kind:?}").as_bytes());
    transcript.append_values(b"values", values);
}

fn draw(transcript: &mut T, draw: ByteLinkDraw, count: usize) -> Vec<F> {
    transcript.append_bytes(format!("{draw:?}").as_bytes());
    (0..count).map(|_| transcript.challenge_scalar()).collect()
}

fn flatten(message: ByteLinkMessage<'_>) -> (Kind, Vec<F>) {
    let pairs = |pairs: &[(F, F)]| pairs.iter().flat_map(|&(p, q)| [p, q]).collect::<Vec<_>>();
    match message {
        ByteLinkMessage::Roots(roots) => (Kind::Roots, pairs(roots)),
        ByteLinkMessage::LayerClaims {
            batch,
            layer,
            point,
            claims,
        } => (
            Kind::LayerClaims(batch, layer),
            point.iter().copied().chain(pairs(claims)).collect(),
        ),
        ByteLinkMessage::LayerRound {
            batch,
            layer,
            round,
            poly,
        } => (
            Kind::LayerRound(batch, layer, round),
            poly.coefficients().to_vec(),
        ),
        ByteLinkMessage::Children {
            batch,
            layer,
            children,
        } => (Kind::Children(batch, layer), children.concat()),
        ByteLinkMessage::QueryValues { group, values } => {
            (Kind::QueryValues(group), values.to_vec())
        }
        ByteLinkMessage::QueryRound { group, round, poly } => {
            (Kind::QueryRound(group, round), poly.coefficients().to_vec())
        }
        ByteLinkMessage::QueryFinals { group, values } => {
            (Kind::QueryFinals(group), values.to_vec())
        }
        ByteLinkMessage::SourceClaims {
            point,
            denominators,
        } => (
            Kind::SourceClaims,
            point.iter().chain(denominators).copied().collect(),
        ),
        ByteLinkMessage::SourceRound { round, poly } => {
            (Kind::SourceRound(round), poly.coefficients().to_vec())
        }
        ByteLinkMessage::SourceFinals(values) => (Kind::SourceFinals, values.to_vec()),
    }
}

impl Recorder {
    /// Absorbs the statement and the `W` commitment stand-in, then draws `γ` and `β`.
    fn new(statement: &Statement, digest: &[F]) -> (Self, ByteLinkCompression) {
        let mut transcript = T::new(b"byte-link-test");
        let compression = open_link(&mut transcript, statement, digest);
        (
            Self {
                transcript,
                log: Vec::new(),
            },
            compression,
        )
    }
}

fn open_link(transcript: &mut T, statement: &Statement, digest: &[F]) -> ByteLinkCompression {
    transcript.append_values(b"cycle", &statement.cycle_point);
    for point in &statement.address_points {
        transcript.append_values(b"address", point);
    }
    transcript.append_values(b"one-hot", &statement.one_hot_claims);
    transcript.append_values(b"fused", &[statement.fused_increment]);
    transcript.append_values(b"histograms", digest);
    let gamma = std::array::from_fn(|_| std::array::from_fn(|_| transcript.challenge_scalar()));
    ByteLinkCompression {
        gamma,
        beta: transcript.challenge_scalar(),
    }
}

impl ByteLinkTranscript for Recorder {
    fn append(&mut self, message: ByteLinkMessage<'_>) {
        let (kind, values) = flatten(message);
        absorb(&mut self.transcript, &kind, &values);
        self.log.push((kind, values));
    }

    fn challenges(&mut self, kind: ByteLinkDraw, count: usize) -> Vec<F> {
        draw(&mut self.transcript, kind, count)
    }
}

/// Replays a recorded log, deriving every challenge itself.
struct Reader<'a> {
    transcript: T,
    log: &'a [(Kind, Vec<F>)],
    next: usize,
}

impl Reader<'_> {
    fn recv(&mut self, kind: Kind, len: usize) -> Result<Vec<F>, String> {
        match self.log.get(self.next) {
            Some((k, values)) if *k == kind && values.len() == len => {
                self.next += 1;
                absorb(&mut self.transcript, k, values);
                Ok(values.clone())
            }
            other => Err(format!(
                "entry {}: expected {kind:?} x{len}, got {other:?}",
                self.next
            )),
        }
    }

    fn draw(&mut self, kind: ByteLinkDraw, count: usize) -> Vec<F> {
        draw(&mut self.transcript, kind, count)
    }

    /// One sumcheck of `rounds` rounds of degree at most `degree`; returns the challenges and the
    /// final claim.
    fn sumcheck(
        &mut self,
        mut claim: F,
        rounds: usize,
        degree: usize,
        round: impl Fn(usize) -> (Kind, ByteLinkDraw),
    ) -> Result<(Vec<F>, F), String> {
        let mut challenges = Vec::new();
        for i in 0..rounds {
            let (kind, draw) = round(i);
            let poly = self.recv(kind.clone(), degree + 1)?;
            let at = |x: F| poly.iter().rev().fold(F::zero(), |acc, &c| acc * x + c);
            if at(F::zero()) + at(F::one()) != claim {
                return Err(format!("{kind:?}: round sum"));
            }
            let r = self.draw(draw, 1)[0];
            claim = at(r);
            challenges.push(r);
        }
        Ok((challenges, claim))
    }

    /// A batched GKR over trees of `log_leaves` levels; returns the canonical leaf point and the
    /// leaf claims.
    fn gkr(
        &mut self,
        batch: ByteLinkBatch,
        log_leaves: usize,
        roots: &[(F, F)],
    ) -> Result<LeafClaims, String> {
        let trees = roots.len();
        let mut point: Vec<F> = Vec::new();
        let mut claims = roots.to_vec();
        for layer in 0..log_leaves {
            let sent = self.recv(Kind::LayerClaims(batch, layer), layer + 2 * trees)?;
            if sent[..layer] != point[..]
                || sent[layer..] != claims.iter().flat_map(|&(p, q)| [p, q]).collect::<Vec<_>>()[..]
            {
                return Err(format!("{batch:?} layer {layer}: claims"));
            }
            let weights = self.draw(ByteLinkDraw::LayerWeights { batch, layer }, 2 * trees);
            let claim = claims
                .iter()
                .zip(weights.chunks(2))
                .map(|(&(p, q), w)| w[0] * p + w[1] * q)
                .sum();
            let (s, claim) = self.sumcheck(claim, layer, 3, |round| {
                (
                    Kind::LayerRound(batch, layer, round),
                    ByteLinkDraw::LayerChallenge {
                        batch,
                        layer,
                        round,
                    },
                )
            })?;
            let children = self.recv(Kind::Children(batch, layer), 4 * trees)?;
            let parent = s.iter().rev().copied().collect::<Vec<_>>();
            let gates = children
                .chunks(4)
                .zip(weights.chunks(2))
                .map(|(c, w)| w[0] * (c[0] * c[3] + c[2] * c[1]) + w[1] * c[1] * c[3])
                .sum::<F>();
            if claim != EqPolynomial::<F>::mle(&point, &parent) * gates {
                return Err(format!("{batch:?} layer {layer}: gate"));
            }
            let mu = self.draw(ByteLinkDraw::ChildSelector { batch, layer }, 1)[0];
            claims = children
                .chunks(4)
                .map(|c| (c[0] + mu * (c[2] - c[0]), c[1] + mu * (c[3] - c[1])))
                .collect();
            point = parent.into_iter().chain([mu]).collect();
        }
        Ok((point, claims))
    }
}

/// `σ̃` of one byte's canonical (MSB-first) coordinates.
fn sigma_mle(x: &[F]) -> F {
    x.iter().fold(F::zero(), |acc, &bit| acc + acc + bit) - F::from_u64(256) * x[0]
}

fn eq_byte(point: &[F; 8], code: &[F]) -> F {
    EqPolynomial::<F>::mle(point, code)
}

/// Verifies a recorded link; returns the openings it leaves for stage 8.
fn verify(
    log: &[(Kind, Vec<F>)],
    statement: &Statement,
    digest: &[F],
) -> Result<ByteLinkOpenings, String> {
    let log_n = statement.cycle_point.len();
    let mut transcript = T::new(b"byte-link-test");
    let compression = open_link(&mut transcript, statement, digest);
    let mut rd = Reader {
        transcript,
        log,
        next: 0,
    };
    let (gamma, beta) = (compression.gamma, compression.beta);
    let roots = rd.recv(Kind::Roots, 4 * PACKS)?;
    let roots = roots.chunks(2).map(|r| (r[0], r[1])).collect::<Vec<_>>();
    let (trace_roots, table_roots) = roots.split_at(PACKS);
    for (pack, (&(p, b), &(pt, bt))) in trace_roots.iter().zip(table_roots).enumerate() {
        if b.is_zero() || bt.is_zero() || p * bt != pt * b {
            return Err(format!("pack {pack}: trace and table roots differ"));
        }
    }
    let (z, trace_leaves) = rd.gkr(ByteLinkBatch::Trace, log_n, trace_roots)?;
    let (y, triple_leaves) = rd.gkr(
        ByteLinkBatch::Triples,
        TRIPLE_BITS as usize,
        &table_roots[..RAM],
    )?;
    let (y_ram, ram_leaves) = rd.gkr(ByteLinkBatch::Ram, RAM_BITS as usize, &table_roots[RAM..])?;
    let eq_r_z = EqPolynomial::<F>::mle(&statement.cycle_point, &z);
    if trace_leaves.iter().any(|&(p, _)| p != eq_r_z) {
        return Err("trace leaf numerator".into());
    }
    for (pack, &(_, b)) in triple_leaves.iter().chain(&ram_leaves).enumerate() {
        let point = if pack == RAM { &y_ram } else { &y };
        let last = if pack == RAM {
            point[16]
        } else {
            sigma_mle(&point[16..24])
        };
        let expected = beta
            - gamma[pack][0] * sigma_mle(&point[..8])
            - gamma[pack][1] * sigma_mle(&point[8..16])
            - gamma[pack][2] * last;
        if b != expected {
            return Err(format!("pack {pack}: table leaf denominator"));
        }
    }

    let half16 = F::from_u64(1 << 16).inv_or_zero();
    let half8 = F::from_u64(1 << 8).inv_or_zero();
    let claims = &statement.one_hot_claims;
    let mut queries = Vec::new();
    for group in [ByteLinkQueryGroup::Triples, ByteLinkQueryGroup::Ram] {
        let (packs, point, leaves, bits) = match group {
            ByteLinkQueryGroup::Triples => (0..RAM, &y, &triple_leaves, TRIPLE_BITS as usize),
            ByteLinkQueryGroup::Ram => (RAM..PACKS, &y_ram, &ram_leaves, RAM_BITS as usize),
        };
        let expected = packs
            .clone()
            .zip(leaves)
            .flat_map(|(pack, &(w, _))| match group {
                ByteLinkQueryGroup::Triples => (0..3)
                    .map(|s| claims[3 * pack + s] * half16)
                    .chain([w])
                    .collect::<Vec<_>>(),
                ByteLinkQueryGroup::Ram => vec![claims[18] * half8, claims[19] * half8, w],
            })
            .collect::<Vec<_>>();
        if rd.recv(Kind::QueryValues(group), expected.len())? != expected {
            return Err(format!("{group:?}: query values"));
        }
        let alpha = rd.draw(ByteLinkDraw::QueryWeights { group }, expected.len());
        let claim = expected.iter().zip(&alpha).map(|(&v, &a)| v * a).sum();
        let (s, claim) = rd.sumcheck(claim, bits, 2, |round| {
            (
                Kind::QueryRound(group, round),
                ByteLinkDraw::QueryChallenge { group, round },
            )
        })?;
        let finals = rd.recv(Kind::QueryFinals(group), packs.len())?;
        let s = s.iter().rev().copied().collect::<Vec<_>>();
        let eq_y = EqPolynomial::<F>::mle(point, &s);
        let total = packs
            .clone()
            .zip(alpha.chunks(expected.len() / packs.len()))
            .zip(&finals)
            .map(|((pack, a), &w)| {
                let omega = match group {
                    ByteLinkQueryGroup::Triples => {
                        (0..3)
                            .map(|i| {
                                a[i] * eq_byte(
                                    &statement.address_points[3 * pack + i],
                                    &s[8 * i..8 * i + 8],
                                )
                            })
                            .sum::<F>()
                            * half16
                            + a[3] * eq_y
                    }
                    ByteLinkQueryGroup::Ram => {
                        (a[0] * eq_byte(&statement.address_points[18], &s[..8])
                            + a[1] * eq_byte(&statement.address_points[19], &s[8..16]))
                            * half8
                            * s[16]
                            + a[2] * eq_y
                    }
                };
                w * omega
            })
            .sum::<F>();
        if claim != total {
            return Err(format!("{group:?}: query reduction"));
        }
        queries.push(ByteLinkOpening {
            point: s,
            values: finals,
        });
    }

    let sent = rd.recv(Kind::SourceClaims, log_n + PACKS)?;
    let denominators = trace_leaves.iter().map(|&(_, b)| b).collect::<Vec<_>>();
    if sent[..log_n] != z[..] || sent[log_n..] != denominators[..] {
        return Err("source claims".into());
    }
    let theta = rd.draw(ByteLinkDraw::ZeroSlotPoint, log_n);
    let alpha = rd.draw(ByteLinkDraw::SourceWeights, 10);
    let claim = denominators
        .iter()
        .zip(&alpha)
        .map(|(&b, &a)| a * (beta - b))
        .sum::<F>()
        + alpha[PACKS] * statement.fused_increment;
    let (s, claim) = rd.sumcheck(claim, log_n, 2, |round| {
        (
            Kind::SourceRound(round),
            ByteLinkDraw::SourceChallenge { round },
        )
    })?;
    let d = rd.recv(Kind::SourceFinals, SLOTS)?;
    let x = s.iter().rev().copied().collect::<Vec<_>>();
    let g = (0..PACKS)
        .map(|pack| {
            alpha[pack]
                * (0..3)
                    .map(|i| gamma[pack][i] * d[PACK_SLOTS[pack][i]])
                    .sum::<F>()
        })
        .sum::<F>();
    let f = INCREMENT_SLOTS
        .iter()
        .rev()
        .fold(F::zero(), |acc, &slot| acc * F::from_u64(256) + d[slot])
        * alpha[PACKS];
    let zeros = alpha[8] * d[30] + alpha[9] * d[31];
    let expected = EqPolynomial::<F>::mle(&z, &x) * g
        + EqPolynomial::<F>::mle(&statement.cycle_point, &x) * f
        + EqPolynomial::<F>::mle(&theta, &x) * zeros;
    if claim != expected {
        return Err("source point reduction".into());
    }
    if rd.next != log.len() {
        return Err("trailing messages".into());
    }
    let ram = queries.pop().unwrap();
    let triples = queries.pop().unwrap();
    Ok(ByteLinkOpenings {
        triples,
        ram,
        source: ByteLinkOpening {
            point: x,
            values: d,
        },
    })
}

/// `eq(point, bits_msb(h))` over `bits` index bits.
fn eq_index(point: &[F], h: usize) -> F {
    let bits = point.len();
    point
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            if h >> (bits - 1 - i) & 1 == 1 {
                x
            } else {
                F::one() - x
            }
        })
        .product()
}

/// Every opened value and the device `W` recomputed from the source.
fn check_openings(
    source: &Source,
    histograms: &[HashMap<usize, F>],
    w: &Buffer,
    openings: &ByteLinkOpenings,
) -> Result<(), String> {
    let device = view::<Fp128>(w, 0, W_CELLS);
    for (pack, histogram) in histograms.iter().enumerate() {
        let cells = 1usize << if pack == RAM { RAM_BITS } else { TRIPLE_BITS };
        let first = pack << TRIPLE_BITS;
        let wrong = (0..cells).into_par_iter().find_any(|&h| {
            field(device[first + h]) != histogram.get(&h).copied().unwrap_or_else(F::zero)
        });
        if let Some(h) = wrong {
            return Err(format!("pack {pack}: W cell {h}"));
        }
    }
    for (opening, packs) in [(&openings.triples, 0..RAM), (&openings.ram, RAM..PACKS)] {
        for (pack, &value) in packs.zip(&opening.values) {
            let expected = histograms[pack]
                .iter()
                .map(|(&h, &w)| w * eq_index(&opening.point, h))
                .sum::<F>();
            if value != expected {
                return Err(format!("pack {pack}: W opening"));
            }
        }
    }
    let eq_x = eq_rows(&openings.source.point);
    for slot in 0..SLOTS {
        let expected = (0..source.n())
            .map(|t| eq_x[t] * sigma(source.at(slot, t)))
            .sum::<F>();
        if openings.source.values[slot] != expected {
            return Err(format!("slot {slot}: Q opening"));
        }
    }
    Ok(())
}

fn device_source(metal: &SolinasMetal, source: &Source) -> Buffer {
    metal.device.new_buffer_with_data(
        source.q.as_ptr().cast(),
        source.q.len() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// One proof: the histograms, their commitment stand-in, the recorded link and its openings.
struct Proof {
    histograms: ByteLinkHistograms,
    digest: Vec<F>,
    log: Vec<(Kind, Vec<F>)>,
    openings: ByteLinkOpenings,
}

fn prove(prover: &mut ByteLinkProver, view: &ByteLinkSource<'_>, statement: &Statement) -> Proof {
    let histograms = prover.histograms(view, &statement.cycle_point).unwrap();
    let digest = histogram_digest(histograms.buffer());
    let (mut recorder, compression) = Recorder::new(statement, &digest);
    let openings = prover
        .prove(
            view,
            &histograms,
            &statement.view(),
            &compression,
            &mut recorder,
        )
        .unwrap();
    Proof {
        histograms,
        digest,
        log: recorder.log,
        openings,
    }
}

/// The Metal link at 2^16, with and without the zero tail, on uniform bytes with the edge rows
/// and on 256 hot tuples per pack: the verifier accepts, every opening and every `W` cell equals
/// the source's, a second proof repeats the transcript, and altered messages are rejected.
#[test]
fn metal_link_is_accepted_and_opens_the_source() {
    let Ok(metal) = SolinasMetal::for_akita() else {
        return;
    };
    let log_n = 16;
    for (active, hot) in [(scaled_active(log_n), false), (1 << log_n, true)] {
        let source = Source::synthetic(log_n, active, 5, hot).with_edge_rows();
        let statement = Statement::honest(&source, 11);
        let bytes = device_source(&metal, &source);
        let view = ByteLinkSource {
            bytes: &bytes,
            log_rows: log_n,
            active_rows: source.active,
        };
        let mut prover = ByteLinkProver::new(&metal).unwrap();
        let proof = prove(&mut prover, &view, &statement);
        let log = &proof.log;
        assert_eq!(
            verify(log, &statement, &proof.digest).unwrap(),
            proof.openings,
            "hot={hot}"
        );
        let cpu = histograms(&source, &statement.cycle_point);
        check_openings(&source, &cpu, proof.histograms.buffer(), &proof.openings).unwrap();
        assert!(
            prove(&mut prover, &view, &statement).log == *log,
            "hot={hot}: the second proof differs"
        );
        if hot {
            continue;
        }
        let round = log
            .iter()
            .position(|(k, _)| *k == Kind::LayerRound(ByteLinkBatch::Trace, 14, 3))
            .unwrap();
        let finals = log
            .iter()
            .position(|(k, _)| *k == Kind::QueryFinals(ByteLinkQueryGroup::Triples))
            .unwrap();
        for (entry, error) in [
            (0, "roots differ"),
            (round, "round sum"),
            (finals, "query reduction"),
        ] {
            let mut altered = log.clone();
            altered[entry].1[0] += F::one();
            let rejected = verify(&altered, &statement, &proof.digest).unwrap_err();
            assert!(rejected.contains(error), "{rejected}");
        }
    }
}

/// Active rows of the 2^29 target trace scaled to `2^log_n` rows.
fn scaled_active(log_n: u32) -> usize {
    U29.div_ceil(1 << (29 - log_n))
}

fn env_or<V: FromStr<Err: Debug>>(name: &str, default: V) -> V {
    std::env::var(name).map_or(default, |value| value.parse().unwrap())
}

extern "C" {
    /// libsystem_kernel (private libproc header): restarts `ri_interval_max_phys_footprint`.
    fn proc_reset_footprint_interval(pid: libc::c_int) -> libc::c_int;
}

/// `(ri_phys_footprint, ri_interval_max_phys_footprint)` of this process.
fn footprint() -> (u64, u64) {
    // SAFETY: proc_pid_rusage writes one complete rusage_info_v4 for RUSAGE_INFO_V4.
    unsafe {
        let mut info: rusage_info_v4 = std::mem::zeroed();
        let status = libc::proc_pid_rusage(libc::getpid(), RUSAGE_INFO_V4, (&raw mut info).cast());
        assert_eq!(status, 0);
        (info.ri_phys_footprint, info.ri_interval_max_phys_footprint)
    }
}

/// The Metal link at `2^LINK_LOG` rows (default 26): `LINK_REPS` proofs (default 3) on U-scaled
/// uniform bytes, or `LINK_FULL=1` without the zero tail, `LINK_HOT=1` with 256 tuples per pack.
/// Prints a `phase` row per phase and proof (wall, GPU seconds, command buffers), a `total` row
/// (wall and GPU seconds of the phases after the histograms, transcript digest, footprint at the
/// proof start and its peak during the proof, arena bytes) and verifies the first proof; with
/// `LINK_CHECK=1` also every opening against the source.
#[test]
#[ignore = "GPU bench; run through gpu-window.sh"]
#[expect(clippy::print_stdout, reason = "bench report")]
fn bench_link() {
    let log_n: u32 = env_or("LINK_LOG", 26);
    let active = if env_or("LINK_FULL", 0) == 1 {
        1 << log_n
    } else {
        scaled_active(log_n)
    };
    let reps: usize = env_or("LINK_REPS", 3);
    let build = Instant::now();
    let source = Source::synthetic(log_n, active, 0x6c69_6e6b, env_or("LINK_HOT", 0) == 1);
    let statement = Statement::honest(&source, 7);
    let metal = SolinasMetal::for_akita().unwrap();
    let bytes = device_source(&metal, &source);
    println!(
        "shape\tlog_n={log_n}\tactive={active}\tsetup_s={:.1}",
        build.elapsed().as_secs_f64()
    );
    let view = ByteLinkSource {
        bytes: &bytes,
        log_rows: log_n,
        active_rows: active,
    };
    let mut prover = ByteLinkProver::new(&metal).unwrap();
    for rep in 0..reps {
        std::thread::sleep(Duration::from_millis(200));
        let start = footprint().0;
        // SAFETY: takes only a pid and writes no caller memory.
        let _ = unsafe { proc_reset_footprint_interval(libc::getpid()) };
        let proof = prove(&mut prover, &view, &statement);
        let peak = footprint().1;
        let phases = prover.gpu.take_phases();
        let (mut wall, mut gpu) = (0.0, 0.0);
        for Phase {
            name,
            wall: w,
            gpu: g,
            commands,
        } in &phases
        {
            println!("phase\t{rep}\t{name}\t{w:.6}\t{g:.6}\t{commands}");
            if *name != "histograms" {
                wall += w;
                gpu += g;
            }
        }
        let mut transcript = T::new(b"byte-link-bench-digest");
        for (kind, values) in &proof.log {
            absorb(&mut transcript, kind, values);
        }
        let state = transcript.state();
        println!(
            "total\t{rep}\t{wall:.6}\t{gpu:.6}\t{:016x}\t{start}\t{peak}\t{}",
            u64::from_be_bytes(state[..8].try_into().unwrap()),
            prover.gpu.arena_bytes()
        );
        if rep == 0 {
            assert_eq!(
                verify(&proof.log, &statement, &proof.digest).unwrap(),
                proof.openings
            );
            println!("verified");
            if env_or("LINK_CHECK", 0) == 1 {
                let cpu = histograms(&source, &statement.cycle_point);
                check_openings(&source, &cpu, proof.histograms.buffer(), &proof.openings).unwrap();
                println!("checked");
            }
        }
    }
}
