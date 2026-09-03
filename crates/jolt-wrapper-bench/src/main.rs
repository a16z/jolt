//! Layer-1 size/time microbench for the curve wrapper (lanes M3/N3): the
//! Blake3 transcript table in its word layout (163 committed bit columns per
//! half-G row) or bit-sliced layouts (16 committed bits per bit position),
//! packed HyperKZG commitments (`k` columns per polynomial), the degree-3
//! row sumcheck with compressed round messages, an optional column sumcheck
//! that replaces the per-column claims, and one combined opening. The
//! `batched` mode folds the limb-relation table (lane M2) and synthetic
//! Spartan outer/inner members into the same sumcheck stream and opening.
//!
//! Usage: `jolt-wrapper-bench [layout=word|bits:<b>] [k=<n>] [col] [batched]
//! [s=<n>] [t=<n>] [spartan=<log2 rows>] [log_rows...]`.

#![expect(
    clippy::print_stdout,
    clippy::expect_used,
    clippy::too_many_lines,
    reason = "benchmark binary: reports to stdout, aborts on setup failure"
)]

mod commit;
mod relation;
mod row;
mod spartan;
mod sumcheck;
mod table;

use std::time::Instant;

use jolt_crypto::{Bn254, PairingGroup};
use jolt_field::{Field, Fr, One, Ring, Zero};
use jolt_hyperkzg::{HyperKZGCommitment, HyperKZGProverSetup, HyperKZGScheme};
use jolt_limb_bench::pack;
use jolt_limb_bench::relation::{Prover as LimbProver, Public as LimbPublic, TABLE_LOG};
use jolt_limb_bench::table::Table as LimbTable;
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};
use jolt_poly::{EqPolynomial, Polynomial};
use jolt_transcript::{Blake3Transcript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use rayon::prelude::*;

use commit::commit_bit_columns;
use relation::Layout;
use row::RowInstance;
use spartan::DenseInstance;
use sumcheck::{prove_stream, verify_stream, ColumnInstance, Instance, Member};
use table::Table;

type Scheme = HyperKZGScheme<Bn254>;

struct Config {
    layout: Layout,
    k: usize,
    column_sumcheck: bool,
    batched: bool,
    s: usize,
    t: usize,
    spartan_log: usize,
    sizes: Vec<usize>,
}

fn parse() -> Config {
    let mut cfg = Config {
        layout: Layout::Word,
        k: 1,
        column_sumcheck: false,
        batched: false,
        s: 6,
        t: 12,
        spartan_log: 14,
        sizes: Vec::new(),
    };
    for arg in std::env::args().skip(1) {
        if let Some(v) = arg.strip_prefix("layout=") {
            cfg.layout = match v {
                "word" => Layout::Word,
                other => Layout::Bits(
                    other
                        .strip_prefix("bits:")
                        .expect("layout=word|bits:<b>")
                        .parse()
                        .expect("bits per row"),
                ),
            };
        } else if let Some(v) = arg.strip_prefix("k=") {
            cfg.k = v.parse().expect("k");
        } else if let Some(v) = arg.strip_prefix("s=") {
            cfg.s = v.parse().expect("s");
        } else if let Some(v) = arg.strip_prefix("t=") {
            cfg.t = v.parse().expect("t");
        } else if let Some(v) = arg.strip_prefix("spartan=") {
            cfg.spartan_log = v.parse().expect("spartan");
        } else if arg == "col" {
            cfg.column_sumcheck = true;
        } else if arg == "batched" {
            cfg.batched = true;
        } else {
            cfg.sizes.push(arg.parse().expect("log_rows"));
        }
    }
    if cfg.sizes.is_empty() {
        cfg.sizes.push(17);
    }
    cfg
}

fn secs(start: Instant) -> f64 {
    start.elapsed().as_secs_f64()
}

/// HyperKZG proof bytes at `ell` variables: `ell − 1` fold commitments, one
/// witness commitment, `3·ell` evaluations.
fn opening_bytes(ell: usize) -> usize {
    32 * (ell + 3 * ell)
}

fn powers(base: Fr, count: usize) -> Vec<Fr> {
    std::iter::successors(Some(Fr::one()), |p| Some(*p * base))
        .take(count)
        .collect()
}

fn eq_weights(s_hi: &[Fr], groups: usize) -> Vec<Fr> {
    if s_hi.is_empty() {
        vec![Fr::one(); groups]
    } else {
        EqPolynomial::<Fr>::evals(s_hi, None)[..groups].to_vec()
    }
}

/// Instance decorator accumulating the time spent in its rounds.
struct Timed<'a> {
    inner: &'a mut dyn Instance,
    secs: f64,
}

impl Instance for Timed<'_> {
    fn rounds(&self) -> usize {
        self.inner.rounds()
    }
    fn input_claim(&self) -> Fr {
        self.inner.input_claim()
    }
    fn round_poly(&mut self) -> Vec<Fr> {
        let start = Instant::now();
        let poly = self.inner.round_poly();
        self.secs += secs(start);
        poly
    }
    fn bind(&mut self, r: Fr) {
        let start = Instant::now();
        self.inner.bind(r);
        self.secs += secs(start);
    }
}

/// Lane M2's limb-relation prover as a stream member.
struct LimbMember<'a> {
    prover: LimbProver,
    public: &'a LimbPublic,
    last: Vec<Fr>,
    claim: Fr,
}

impl<'a> LimbMember<'a> {
    fn new(prover: LimbProver, public: &'a LimbPublic) -> Self {
        let claim = prover.input_claim(public);
        Self {
            prover,
            public,
            last: Vec::new(),
            claim,
        }
    }
}

impl Instance for LimbMember<'_> {
    fn rounds(&self) -> usize {
        self.public.log_rows
    }
    fn input_claim(&self) -> Fr {
        self.claim
    }
    fn round_poly(&mut self) -> Vec<Fr> {
        self.last = self.prover.round_poly(self.public);
        self.last.clone()
    }
    fn bind(&mut self, r: Fr) {
        self.prover.bind(self.public, &self.last, r);
    }
}

/// One table's slice of the combined opening: `columns` real columns padded
/// to whole groups, in group order.
struct Segment<'a> {
    columns: usize,
    claims: Vec<Fr>,
    column: Box<dyn Fn(usize, usize) -> Fr + Sync + 'a>,
}

impl Segment<'_> {
    fn groups(&self, k: usize) -> usize {
        pack::groups(self.columns, k)
    }
}

/// Combined opening of several segments: one polynomial of `rows·k` entries.
struct Combined {
    weights: Vec<Fr>,
    claimed: Fr,
    poly: Vec<Fr>,
}

fn combine_segments(
    rows: usize,
    k: usize,
    segments: &[Segment<'_>],
    weights: &[Fr],
    hash_claim: Option<Fr>,
    s_lo: &[Fr],
) -> Combined {
    let total_groups: usize = segments.iter().map(|s| s.groups(k)).sum();
    assert_eq!(weights.len(), total_groups);
    let mut first_group = Vec::with_capacity(segments.len());
    let mut g = 0;
    for segment in segments {
        first_group.push(g);
        g += segment.groups(k);
    }
    let poly = pack::combine(rows, k, total_groups * k, weights, |column, row| {
        let group = column / k;
        let seg = first_group
            .iter()
            .rposition(|&f| f <= group)
            .expect("group in range");
        let local = (group - first_group[seg]) * k + column % k;
        if local < segments[seg].columns {
            (segments[seg].column)(local, row)
        } else {
            Fr::zero()
        }
    });
    let mut claimed = Fr::zero();
    for (i, segment) in segments.iter().enumerate() {
        let w = &weights[first_group[i]..first_group[i] + segment.groups(k)];
        claimed += match (i, hash_claim) {
            (0, Some(claim)) => claim,
            _ => pack::combined_claim(&segment.claims, k, w, s_lo),
        };
    }
    Combined {
        weights: weights.to_vec(),
        claimed,
        poly,
    }
}

fn setup(entries: usize) -> HyperKZGProverSetup<Bn254> {
    let mut rng = ChaCha20Rng::seed_from_u64(0xb3);
    let beta = Fr::random(&mut rng);
    Scheme::setup_from_secret(beta, entries, Bn254::g1_generator(), Bn254::g2_generator())
}

fn check_packed_commitment(
    setup: &HyperKZGProverSetup<Bn254>,
    rows: usize,
    k: usize,
    columns: &[Vec<u8>],
    commitment: &HyperKZGCommitment<Bn254>,
) {
    if rows * k > 1 << 18 {
        return;
    }
    let mut dense = vec![Fr::zero(); rows * k];
    for j in 0..k {
        if let Some(column) = columns.get(j) {
            for (slot, bit) in dense[j * rows..(j + 1) * rows].iter_mut().zip(column) {
                *slot = Fr::from_u8(*bit);
            }
        }
    }
    let (expected, ()) = Scheme::commit(&dense, setup).expect("kzg commit");
    assert_eq!(
        *commitment, expected,
        "packed bit commitment matches the SRS commit"
    );
}

fn run_single(cfg: &Config, log_rows: usize, setup: &HyperKZGProverSetup<Bn254>) {
    let rows = 1usize << log_rows;
    let (k, log_k) = (cfg.k, pack::log2_exact(cfg.k));
    let layout = cfg.layout;
    let committed = layout.committed();
    let wired = layout.wired_bits().len() + layout.wired_ints().len();
    let groups = pack::groups(committed, k);
    let table = Table::random(layout, log_rows, 0x3a5e);
    println!(
        "\n== {layout:?} rows 2^{log_rows}, committed {committed}, wired {wired}, constraints {}, k={k}: {groups} commitments, column space 2^{}, column sumcheck {}",
        layout.constraints(),
        layout.log_columns(),
        cfg.column_sumcheck
    );

    let start = Instant::now();
    let commitments = commit_bit_columns(setup, rows, k, &table.bits);
    let commit_secs = secs(start);
    check_packed_commitment(setup, rows, k, &table.bits, &commitments[0]);
    let bit_columns = table.bits.clone();

    let mut tr = Blake3Transcript::new(b"wrapper-bench");
    for c in &commitments {
        tr.append(c);
    }
    let tau: Vec<Fr> = tr.challenge_vector(log_rows);
    let gammas: Vec<Fr> = tr.challenge_scalar_powers(layout.constraints());
    let relation = layout.relation(&gammas);

    let start = Instant::now();
    let mut instance = RowInstance::new(&relation, table, tau.clone());
    let stream = prove_stream(
        &mut [Member {
            instance: &mut instance,
            offset: 0,
        }],
        &mut tr,
    );
    let sumcheck_secs = secs(start);
    let (v, w) = instance.claims();
    let r_be: Vec<Fr> = stream.challenges.iter().rev().copied().collect();

    let start = Instant::now();
    let split = relation.log_columns - log_k;
    let (weights, s_lo, hash_claim, column_stream, finals) = if cfg.column_sumcheck {
        let (vf, wf) = RowInstance::column_space(&relation, &v, &w);
        let mut column = ColumnInstance::new(&relation, vf, wf);
        let cs = prove_stream(
            &mut [Member {
                instance: &mut column,
                offset: 0,
            }],
            &mut tr,
        );
        let (t1, t2) = column.finals();
        tr.append(&t1);
        tr.append(&t2);
        let s_be: Vec<Fr> = cs.challenges.iter().rev().copied().collect();
        (
            eq_weights(&s_be[..split], groups),
            s_be[split..].to_vec(),
            Some(t1),
            Some(cs),
            Some((t1, t2)),
        )
    } else {
        for c in v.iter().chain(&w) {
            tr.append(c);
        }
        let rho: Fr = tr.challenge();
        let s_lo: Vec<Fr> = tr.challenge_vector(log_k);
        (powers(rho, groups), s_lo, None, None, None)
    };
    let segments = [Segment {
        columns: committed,
        claims: v.clone(),
        column: Box::new(|c, row| Fr::from_u8(bit_columns[c][row])),
    }];
    let combined = combine_segments(rows, k, &segments, &weights, hash_claim, &s_lo);
    let point = pack::point(&s_lo, &r_be);
    let reduce_secs = secs(start);
    let start = Instant::now();
    let opening = Scheme::open(setup, &combined.poly, &point, &mut tr).expect("open");
    let open_secs = secs(start);
    assert_eq!(
        Polynomial::new(combined.poly.clone()).evaluate(&point),
        combined.claimed,
        "combined polynomial evaluates to the claimed value"
    );

    // Verifier.
    let start = Instant::now();
    let mut vt = Blake3Transcript::new(b"wrapper-bench");
    for c in &commitments {
        vt.append(c);
    }
    let vtau: Vec<Fr> = vt.challenge_vector(log_rows);
    let vgammas: Vec<Fr> = vt.challenge_scalar_powers(layout.constraints());
    let vrel = layout.relation(&vgammas);
    let (challenges, _, final_claim) = verify_stream(
        &stream.rounds,
        &stream.input_claims,
        &[(log_rows, 0)],
        &mut vt,
    );
    let vr_be: Vec<Fr> = challenges.iter().rev().copied().collect();
    let eq_tau = EqPolynomial::<Fr>::mle(&vtau, &vr_be);
    let (vweights, vs_lo, vclaimed) = if let (Some(cs), Some((t1, t2))) = (&column_stream, finals) {
        let q = final_claim * eq_tau.inverse().expect("nonzero eq");
        assert_eq!(cs.input_claims[0], q, "column sumcheck claim is derivable");
        let (s, _, col_final) = verify_stream(&cs.rounds, &[q], &[(vrel.log_columns, 0)], &mut vt);
        let s_be: Vec<Fr> = s.iter().rev().copied().collect();
        assert_eq!(
            ColumnInstance::check(&vrel, &s_be, t1, t2),
            col_final,
            "column sumcheck final check"
        );
        vt.append(&t1);
        vt.append(&t2);
        (
            eq_weights(&s_be[..split], groups),
            s_be[split..].to_vec(),
            t1,
        )
    } else {
        for c in v.iter().chain(&w) {
            vt.append(c);
        }
        let (vf, wf) = RowInstance::column_space(&vrel, &v, &w);
        assert_eq!(
            eq_tau * vrel.evaluate(&vf, &wf),
            final_claim,
            "row sumcheck final check"
        );
        let rho: Fr = vt.challenge();
        let s_lo: Vec<Fr> = vt.challenge_vector(log_k);
        let weights = powers(rho, groups);
        let claimed = pack::combined_claim(&v, k, &weights, &s_lo);
        (weights, s_lo, claimed)
    };
    assert_eq!(vweights, combined.weights);
    let vcombined = Scheme::combine(&commitments, &vweights);
    let vk = Scheme::verifier_setup(setup);
    Scheme::verify(
        &vk,
        &vcombined,
        &pack::point(&vs_lo, &vr_be),
        &vclaimed,
        &opening,
        &mut vt,
    )
    .expect("opening verifies");
    let verify_secs = secs(start);

    let commit_bytes = 32 * groups;
    let round_bytes = stream.wire_bytes();
    let claims_a = 32 * (committed + wired);
    let claims_b = 32 * (3 * relation.log_columns + 2);
    let claims_bytes = if cfg.column_sumcheck {
        claims_b
    } else {
        claims_a
    };
    let open_bytes = opening_bytes(log_rows + log_k);
    let total = commit_bytes + round_bytes + claims_bytes + open_bytes;
    println!(
        "prover: commit {commit_secs:.3} s | row sumcheck {sumcheck_secs:.3} s | claim reduction + combine {reduce_secs:.3} s | open {open_secs:.3} s | total {:.3} s | verify {verify_secs:.3} s",
        commit_secs + sumcheck_secs + reduce_secs + open_secs
    );
    println!(
        "bytes: commitments {commit_bytes} + rounds {round_bytes} (compressed, {} rounds × deg {}) + claims {claims_bytes} [(a) all claims {claims_a} / (b) column sumcheck {claims_b}] + opening {open_bytes} (2^{}) = {total}; bench-only input claim +32",
        stream.rounds.len(),
        stream.max_degree(),
        log_rows + log_k
    );
}

fn run_batched(cfg: &Config, log_rows: usize, setup: &HyperKZGProverSetup<Bn254>) {
    let rows = 1usize << log_rows;
    let (k, log_k) = (cfg.k, pack::log2_exact(cfg.k));
    let layout = cfg.layout;
    let (t, s_group, sp_log) = (cfg.t, cfg.s, cfg.spartan_log);
    let h_committed = layout.committed();
    let h_wired = layout.wired_bits().len() + layout.wired_ints().len();
    let h_groups = pack::groups(h_committed, k);
    println!(
        "\n== batched stream: {layout:?} 2^{log_rows} ({h_committed} committed, k={k} → {h_groups}) + limb 2^{log_rows} (t={t}, s={s_group}, degree {}) + Spartan outer/inner 2^{sp_log}",
        s_group + 2
    );

    let table = Table::random(layout, log_rows, 0x3a5e);
    let start = Instant::now();
    let h_commitments = commit_bit_columns(setup, rows, k, &table.bits);
    let hash_commit = secs(start);
    let bit_columns = table.bits.clone();

    let limb_table = LimbTable::generate(rows, t, false, 0x00c0_ffee);
    let chunk_columns = limb_table.chunks.len();
    let helper_columns = chunk_columns.div_ceil(s_group) + 1;
    let start = Instant::now();
    let chunk_commitments: Vec<HyperKZGCommitment<Bn254>> = (0..chunk_columns)
        .into_par_iter()
        .map(|c| {
            HyperKZGCommitment::new(Bn254::g1_affine_msm_small(
                pack::slot_bases(setup, rows, k, c),
                &limb_table.chunks[c],
            ))
        })
        .collect();
    let chunk_groups = pack::group_commitments(&chunk_commitments, k);
    let limb_commit_chunks = secs(start);

    let abc = spartan::random_columns(3, sp_log, 0xabc);
    let mz = spartan::random_columns(2, sp_log, 0x3e7);
    let start = Instant::now();
    let (w_commitment, ()) = Scheme::commit(&mz[1], setup).expect("commit W");
    let w_commit = secs(start);

    let mut tr = Blake3Transcript::new(b"wrapper-bench-batched");
    for c in h_commitments.iter().chain(&chunk_groups) {
        tr.append(c);
    }
    tr.append(&w_commitment);
    let alpha: Fr = tr.challenge();
    let (inverses, multiplicities) = limb_table.logup_columns(alpha, s_group);
    let start = Instant::now();
    let helper_commitments: Vec<HyperKZGCommitment<Bn254>> = inverses
        .par_iter()
        .enumerate()
        .map(|(i, column)| {
            HyperKZGCommitment::new(Bn254::g1_affine_msm(
                pack::slot_bases(setup, rows, k, i),
                column,
            ))
        })
        .chain(rayon::iter::once(HyperKZGCommitment::new(
            Bn254::g1_affine_msm_small(
                &pack::slot_bases(setup, rows, k, inverses.len())[..1 << TABLE_LOG],
                &multiplicities,
            ),
        )))
        .collect();
    let helper_groups = pack::group_commitments(&helper_commitments, k);
    let limb_commit_helpers = secs(start);
    for c in &helper_groups {
        tr.append(c);
    }

    let tau: Vec<Fr> = tr.challenge_vector(log_rows);
    let gammas: Vec<Fr> = tr.challenge_scalar_powers(layout.constraints());
    let relation = layout.relation(&gammas);
    let l_public = LimbPublic::draw(&mut tr, log_rows, t, chunk_columns, s_group, alpha, false);
    let sp_tau: Vec<Fr> = tr.challenge_vector(sp_log);

    let start = Instant::now();
    let mut hash = RowInstance::new(&relation, table, tau.clone());
    let mut limb = LimbMember::new(
        LimbProver::new(limb_table, inverses, multiplicities, &l_public),
        &l_public,
    );
    let mut outer = DenseInstance::new(abc, Some(sp_tau.clone()), spartan::outer, 2);
    let mut inner = DenseInstance::new(mz.clone(), None, spartan::inner, 2);
    let build_secs = secs(start);
    let start = Instant::now();
    let mut th = Timed {
        inner: &mut hash,
        secs: 0.0,
    };
    let mut tl = Timed {
        inner: &mut limb,
        secs: 0.0,
    };
    let mut to = Timed {
        inner: &mut outer,
        secs: 0.0,
    };
    let mut ti = Timed {
        inner: &mut inner,
        secs: 0.0,
    };
    let stream = prove_stream(
        &mut [
            Member {
                instance: &mut th,
                offset: 0,
            },
            Member {
                instance: &mut tl,
                offset: 0,
            },
            Member {
                instance: &mut to,
                offset: 0,
            },
            Member {
                instance: &mut ti,
                offset: 0,
            },
        ],
        &mut tr,
    );
    let stream_secs = secs(start);
    let (hash_secs, limb_secs, spartan_secs) = (th.secs, tl.secs, to.secs + ti.secs);
    let (hv, hw) = hash.claims();
    let l_claims = limb.prover.claims();
    let outer_f = outer.finals();
    let inner_f = inner.finals();
    let r_be: Vec<Fr> = stream.challenges.iter().rev().copied().collect();

    // Claims and the hash table's claim reduction.
    let start = Instant::now();
    let split = relation.log_columns - log_k;
    let (hash_weights, s_lo, hash_claim, column_stream, finals) = if cfg.column_sumcheck {
        let (vf, wf) = RowInstance::column_space(&relation, &hv, &hw);
        let mut column = ColumnInstance::new(&relation, vf, wf);
        let cs = prove_stream(
            &mut [Member {
                instance: &mut column,
                offset: 0,
            }],
            &mut tr,
        );
        let (t1, t2) = column.finals();
        tr.append(&t1);
        tr.append(&t2);
        let s_be: Vec<Fr> = cs.challenges.iter().rev().copied().collect();
        (
            eq_weights(&s_be[..split], h_groups),
            s_be[split..].to_vec(),
            Some(t1),
            Some(cs),
            Some((t1, t2)),
        )
    } else {
        for c in hv.iter().chain(&hw) {
            tr.append(c);
        }
        (Vec::new(), Vec::new(), None, None, None)
    };
    for c in l_claims
        .committed
        .iter()
        .chain(&l_claims.operand_limbs)
        .chain(&outer_f)
        .chain(&inner_f)
    {
        tr.append(c);
    }
    let rho: Fr = tr.challenge();
    let s_lo = if cfg.column_sumcheck {
        s_lo
    } else {
        tr.challenge_vector(log_k)
    };
    let l_chunk_groups = chunk_groups.len();
    let l_helper_groups = helper_groups.len();
    let total_groups = h_groups + l_chunk_groups + l_helper_groups + 1;
    let rho_powers = powers(rho, total_groups);
    let weights: Vec<Fr> = if cfg.column_sumcheck {
        hash_weights
            .iter()
            .chain(&rho_powers[h_groups..])
            .copied()
            .collect()
    } else {
        rho_powers
    };
    let w_pad = |row: usize| mz[1].get(row).copied().unwrap_or(Fr::zero());
    let w_claim = EqPolynomial::<Fr>::zero_selector(&r_be[..log_rows - sp_log]) * inner_f[1];
    let limb_prover = &limb.prover;
    let segments = [
        Segment {
            columns: h_committed,
            claims: hv.clone(),
            column: Box::new(|c, row| Fr::from_u8(bit_columns[c][row])),
        },
        Segment {
            columns: chunk_columns,
            claims: l_claims.committed[..chunk_columns].to_vec(),
            column: Box::new(|c, row| limb_prover.committed(c, row)),
        },
        Segment {
            columns: helper_columns,
            claims: l_claims.committed[chunk_columns..].to_vec(),
            column: Box::new(|c, row| limb_prover.committed(chunk_columns + c, row)),
        },
        Segment {
            columns: 1,
            claims: vec![w_claim],
            column: Box::new(|_, row| w_pad(row)),
        },
    ];
    let combined = combine_segments(rows, k, &segments, &weights, hash_claim, &s_lo);
    let point = pack::point(&s_lo, &r_be);
    let reduce_secs = secs(start);
    let start = Instant::now();
    let opening = Scheme::open(setup, &combined.poly, &point, &mut tr).expect("open");
    let open_secs = secs(start);
    assert_eq!(
        Polynomial::new(combined.poly.clone()).evaluate(&point),
        combined.claimed,
        "combined polynomial evaluates to the claimed value"
    );

    // Verifier.
    let start = Instant::now();
    let mut vt = Blake3Transcript::new(b"wrapper-bench-batched");
    for c in h_commitments.iter().chain(&chunk_groups) {
        vt.append(c);
    }
    vt.append(&w_commitment);
    let valpha: Fr = vt.challenge();
    for c in &helper_groups {
        vt.append(c);
    }
    let vtau: Vec<Fr> = vt.challenge_vector(log_rows);
    let vgammas: Vec<Fr> = vt.challenge_scalar_powers(layout.constraints());
    let vrel = layout.relation(&vgammas);
    let vl_public = LimbPublic::draw(&mut vt, log_rows, t, chunk_columns, s_group, valpha, false);
    let vsp_tau: Vec<Fr> = vt.challenge_vector(sp_log);
    let members = [(log_rows, 0), (log_rows, 0), (sp_log, 0), (sp_log, 0)];
    let (challenges, betas, final_claim) =
        verify_stream(&stream.rounds, &stream.input_claims, &members, &mut vt);
    let vr_be: Vec<Fr> = challenges.iter().rev().copied().collect();
    let vsp_be: Vec<Fr> = challenges[..sp_log].iter().rev().copied().collect();
    let limb_expected = vl_public
        .final_value(&l_claims, &vr_be)
        .expect("limb claims");
    let outer_expected = DenseInstance::expected(Some(&vsp_tau), spartan::outer, &outer_f, &vsp_be);
    let inner_expected = DenseInstance::expected(None, spartan::inner, &inner_f, &vsp_be);
    let rest = betas[1] * limb_expected + betas[2] * outer_expected + betas[3] * inner_expected;
    let hash_final = (final_claim - rest) * betas[0].inverse().expect("nonzero beta");
    let eq_tau = EqPolynomial::<Fr>::mle(&vtau, &vr_be);
    let (vhash_weights, vs_lo, vhash_claim) = if let (Some(cs), Some((t1, t2))) =
        (&column_stream, finals)
    {
        let q = hash_final * eq_tau.inverse().expect("nonzero eq");
        assert_eq!(cs.input_claims[0], q, "column sumcheck claim is derivable");
        let (s, _, col_final) = verify_stream(&cs.rounds, &[q], &[(vrel.log_columns, 0)], &mut vt);
        let s_be: Vec<Fr> = s.iter().rev().copied().collect();
        assert_eq!(
            ColumnInstance::check(&vrel, &s_be, t1, t2),
            col_final,
            "column sumcheck final check"
        );
        vt.append(&t1);
        vt.append(&t2);
        (
            eq_weights(&s_be[..split], h_groups),
            s_be[split..].to_vec(),
            Some(t1),
        )
    } else {
        for c in hv.iter().chain(&hw) {
            vt.append(c);
        }
        let (vf, wf) = RowInstance::column_space(&vrel, &hv, &hw);
        assert_eq!(
            eq_tau * vrel.evaluate(&vf, &wf),
            hash_final,
            "hash final check"
        );
        (Vec::new(), Vec::new(), None)
    };
    for c in l_claims
        .committed
        .iter()
        .chain(&l_claims.operand_limbs)
        .chain(&outer_f)
        .chain(&inner_f)
    {
        vt.append(c);
    }
    let vrho: Fr = vt.challenge();
    let vs_lo = if cfg.column_sumcheck {
        vs_lo
    } else {
        vt.challenge_vector(log_k)
    };
    let vrho_powers = powers(vrho, total_groups);
    let vweights: Vec<Fr> = if cfg.column_sumcheck {
        vhash_weights
            .iter()
            .chain(&vrho_powers[h_groups..])
            .copied()
            .collect()
    } else {
        vrho_powers
    };
    assert_eq!(vweights, combined.weights);
    let vw_claim = EqPolynomial::<Fr>::zero_selector(&vr_be[..log_rows - sp_log]) * inner_f[1];
    let mut vclaimed = Fr::zero();
    let mut g = 0;
    for (i, segment) in segments.iter().enumerate() {
        let w = &vweights[g..g + segment.groups(k)];
        let claims = if i == 3 {
            vec![vw_claim]
        } else {
            segment.claims.clone()
        };
        vclaimed += match (i, vhash_claim) {
            (0, Some(c)) => c,
            _ => pack::combined_claim(&claims, k, w, &vs_lo),
        };
        g += segment.groups(k);
    }
    let all_commitments: Vec<HyperKZGCommitment<Bn254>> = h_commitments
        .iter()
        .chain(&chunk_groups)
        .chain(&helper_groups)
        .chain(std::iter::once(&w_commitment))
        .copied()
        .collect();
    let vcombined = Scheme::combine(&all_commitments, &vweights);
    let vk = Scheme::verifier_setup(setup);
    Scheme::verify(
        &vk,
        &vcombined,
        &pack::point(&vs_lo, &vr_be),
        &vclaimed,
        &opening,
        &mut vt,
    )
    .expect("opening verifies");
    let verify_secs = secs(start);

    let commit_bytes = 32 * all_commitments.len();
    let round_bytes = stream.wire_bytes();
    let separate_round_bytes =
        32 * (log_rows * 3 + log_rows * (s_group + 2) + sp_log * 3 + sp_log * 2);
    let hash_claims_a = 32 * (h_committed + h_wired);
    let hash_claims_b = 32 * (3 * relation.log_columns + 2);
    let hash_claims = if cfg.column_sumcheck {
        hash_claims_b
    } else {
        hash_claims_a
    };
    let limb_claims = 32 * (l_claims.committed.len() + l_claims.operand_limbs.len());
    let spartan_claims = 32 * (outer_f.len() + inner_f.len());
    let open_bytes = opening_bytes(log_rows + log_k);
    let total =
        commit_bytes + round_bytes + hash_claims + limb_claims + spartan_claims + open_bytes;
    println!(
        "prover: commits {:.3} s (hash {hash_commit:.3}, limb chunks {limb_commit_chunks:.3}, limb helpers {limb_commit_helpers:.3}, W {w_commit:.3}) | instance build {build_secs:.3} s | stream {stream_secs:.3} s (hash {hash_secs:.3}, limb {limb_secs:.3}, spartan {spartan_secs:.3}) | claim reduction + combine {reduce_secs:.3} s | open {open_secs:.3} s | total {:.3} s | verify {verify_secs:.3} s",
        hash_commit + limb_commit_chunks + limb_commit_helpers + w_commit,
        hash_commit + limb_commit_chunks + limb_commit_helpers + w_commit + build_secs + stream_secs + reduce_secs + open_secs
    );
    println!(
        "bytes: commitments {commit_bytes} ({} hash + {} limb + 1 W) + rounds {round_bytes} ({} rounds × deg {}; separate streams {separate_round_bytes}) + hash claims {hash_claims} [(a) {hash_claims_a} / (b) {hash_claims_b}] + limb claims {limb_claims} ({} columns + {} operand limbs) + Spartan claims {spartan_claims} + opening {open_bytes} (2^{}) = {total}; bench-only input claims +{}",
        h_groups,
        l_chunk_groups + l_helper_groups,
        stream.rounds.len(),
        stream.max_degree(),
        l_claims.committed.len(),
        l_claims.operand_limbs.len(),
        log_rows + log_k,
        32 * stream.input_claims.len()
    );
}

fn main() {
    let cfg = parse();
    let max_entries = (1usize << cfg.sizes.iter().copied().max().expect("one size")) * cfg.k;
    let start = Instant::now();
    let setup = setup(max_entries);
    println!(
        "setup: SRS 2^{} in {:.1} s, rayon threads {}",
        max_entries.ilog2(),
        secs(start),
        rayon::current_num_threads()
    );
    for &log_rows in &cfg.sizes {
        if cfg.batched {
            run_batched(&cfg, log_rows, &setup);
        } else {
            run_single(&cfg, log_rows, &setup);
        }
    }
}
