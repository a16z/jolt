//! Committed-bit-table microbench for the wrapper's Blake3 transcript
//! relation (lane M3): the half-G row shape, random data.
//!
//! Per row: 163 committed bit columns (A', D', C', B' = 32 each, 3 add
//! carries, 32 message bits), 66 wired inputs (bin, din bits; a_in, c_in
//! integers) and one public selector (the row's top index bit). The 229
//! constraints are all degree ≤ 2 in the columns: booleanity of every
//! committed column, 64 XOR rows against wired bits, one ternary and one
//! binary 32-bit add with power-of-two weights (the binary add's rotation
//! chosen by the selector).
//!
//! Timed phases: batch-addition HyperKZG commits of the 163 bit columns; the
//! degree-3 sumcheck `eq(τ,row)·Σ_j γ^j C_j(row)`; RLC of the committed
//! columns + one HyperKZG opening at the sumcheck point; verification.
//! Usage: `jolt-wrapper-bench [log_rows...]` (default `16 17 18`).

#![expect(
    clippy::print_stdout,
    clippy::expect_used,
    reason = "benchmark binary: reports to stdout, aborts on setup failure"
)]

use std::time::Instant;

use jolt_crypto::ec::bn254::batch_addition::batch_g1_additions_multi;
use jolt_crypto::{Bn254, Bn254G1, JoltGroup};
use jolt_field::{Field, Fr, One, Ring, Zero};
use jolt_hyperkzg::{HyperKZGCommitment, HyperKZGProverSetup, HyperKZGScheme};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};
use jolt_poly::{EqPolynomial, Polynomial};
use jolt_transcript::{Blake3Transcript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};
use rayon::prelude::*;

const WORD: usize = 32;
const A: usize = 0;
const D: usize = 32;
const C: usize = 64;
const B: usize = 96;
const CARRY: usize = 128;
const M: usize = 131;
const COMMITTED: usize = 163;
const BIN: usize = 163;
const DIN: usize = 195;
const AIN: usize = 227;
const CIN: usize = 228;
const SEL: usize = 229;
const COLUMNS: usize = 230;
const XOR_D: usize = 163;
const XOR_B: usize = 195;
const ADD_A: usize = 227;
const ADD_C: usize = 228;
const CONSTRAINTS: usize = 229;

type Scheme = HyperKZGScheme<Bn254>;

/// `Σ_k 2^((k - rot) mod 32) · v[base + k]` by doubling from the top weight.
fn word_int(v: &[Fr], base: usize, rot: usize) -> Fr {
    let mut acc = Fr::zero();
    for w in (0..WORD).rev() {
        acc = acc + acc + v[base + (w + rot) % WORD];
    }
    acc
}

fn word_u64(v: &[u8], base: usize, rot: usize) -> u64 {
    (0..WORD).fold(0u64, |acc, k| {
        acc | (u64::from(v[base + k]) << ((k + WORD - rot) % WORD))
    })
}

/// The batched constraint value `Σ_j γ_j C_j` at one row.
fn q_value(v: &[Fr], g: &[Fr]) -> Fr {
    let mut acc = Fr::zero();
    for (c, gamma) in v[..COMMITTED].iter().zip(&g[..COMMITTED]) {
        acc += *gamma * (*c * *c - *c);
    }
    for k in 0..WORD {
        let (x, y) = (v[DIN + k], v[A + k]);
        let xy = x * y;
        acc += g[XOR_D + k] * (v[D + k] - x - y + xy + xy);
        let (x, y) = (v[BIN + k], v[C + k]);
        let xy = x * y;
        acc += g[XOR_B + k] * (v[B + k] - x - y + xy + xy);
    }
    let lhs = word_int(v, A, 0) + v[CARRY].mul_pow_2(32) + v[CARRY + 1].mul_pow_2(33);
    let rhs = v[AIN] + word_int(v, BIN, 0) + word_int(v, M, 0);
    acc += g[ADD_A] * (lhs - rhs);
    let (r16, r8) = (word_int(v, D, 16), word_int(v, D, 8));
    let lhs = word_int(v, C, 0) + v[CARRY + 2].mul_pow_2(32);
    let rhs = v[CIN] + r16 + v[SEL] * (r8 - r16);
    acc += g[ADD_C] * (lhs - rhs);
    acc
}

/// The `X²` coefficient of `Σ_j γ_j C_j(lo + X·d)`.
fn q_quad(d: &[Fr], g: &[Fr]) -> Fr {
    let mut acc = Fr::zero();
    for (c, gamma) in d[..COMMITTED].iter().zip(&g[..COMMITTED]) {
        acc += *gamma * c.square();
    }
    for k in 0..WORD {
        let xy = d[DIN + k] * d[A + k];
        acc += g[XOR_D + k] * (xy + xy);
        let xy = d[BIN + k] * d[C + k];
        acc += g[XOR_B + k] * (xy + xy);
    }
    let (r16, r8) = (word_int(d, D, 16), word_int(d, D, 8));
    acc - g[ADD_C] * (d[SEL] * (r8 - r16))
}

/// `q_value` at a Boolean row (bits satisfy booleanity; XOR rows are in
/// {-1, 0, 1}; the adds are small integers).
fn q_bits(v: &[u8], ain: u64, cin: u64, g: &[Fr]) -> Fr {
    let mut plus = Fr::zero();
    let mut minus = Fr::zero();
    for k in 0..WORD {
        match i8::try_from(v[D + k]).expect("bit")
            - i8::try_from(v[DIN + k] ^ v[A + k]).expect("bit")
        {
            1 => plus += g[XOR_D + k],
            -1 => minus += g[XOR_D + k],
            _ => {}
        }
        match i8::try_from(v[B + k]).expect("bit")
            - i8::try_from(v[BIN + k] ^ v[C + k]).expect("bit")
        {
            1 => plus += g[XOR_B + k],
            -1 => minus += g[XOR_B + k],
            _ => {}
        }
    }
    let lhs = word_u64(v, A, 0) + (u64::from(v[CARRY]) << 32) + (u64::from(v[CARRY + 1]) << 33);
    let rhs = ain + word_u64(v, BIN, 0) + word_u64(v, M, 0);
    let add_a = i128::from(lhs) - i128::from(rhs);
    let rot = if v[SEL] == 1 { 8 } else { 16 };
    let lhs = word_u64(v, C, 0) + (u64::from(v[CARRY + 2]) << 32);
    let rhs = cin + word_u64(v, D, rot);
    let add_c = i128::from(lhs) - i128::from(rhs);
    plus - minus + g[ADD_A] * Fr::from_i128(add_a) + g[ADD_C] * Fr::from_i128(add_c)
}

/// `q_quad` for a Boolean pair: `d = hi - lo ∈ {-1, 0, 1}`.
fn q2_bits(lo: &[u8], hi: &[u8], g: &[Fr]) -> Fr {
    let d = |j: usize| i64::from(hi[j]) - i64::from(lo[j]);
    let mut acc = Fr::zero();
    for (j, gamma) in g[..COMMITTED].iter().enumerate() {
        if lo[j] != hi[j] {
            acc += *gamma;
        }
    }
    let mut plus = Fr::zero();
    let mut minus = Fr::zero();
    for k in 0..WORD {
        match d(DIN + k) * d(A + k) {
            1 => plus += g[XOR_D + k],
            -1 => minus += g[XOR_D + k],
            _ => {}
        }
        match d(BIN + k) * d(C + k) {
            1 => plus += g[XOR_B + k],
            -1 => minus += g[XOR_B + k],
            _ => {}
        }
    }
    let xor = plus - minus;
    acc += xor + xor;
    let ds = d(SEL);
    if ds != 0 {
        let rot = |rot: usize| -> i64 {
            (0..WORD)
                .map(|k| d(D + k) << ((k + WORD - rot) % WORD))
                .sum()
        };
        acc -= g[ADD_C] * Fr::from_i64(ds * (rot(8) - rot(16)));
    }
    acc
}

fn eq_scalar(a: Fr, b: Fr) -> Fr {
    let ab = a * b;
    Fr::one() - a - b + ab + ab
}

fn horner(coeffs: &[Fr], x: Fr) -> Fr {
    coeffs.iter().rev().fold(Fr::zero(), |acc, c| acc * x + *c)
}

struct Table {
    /// Round-1 view: 230 bit columns (`AIN`/`CIN` empty) and the two integer
    /// wired inputs.
    bits: Vec<Vec<u8>>,
    ints: [Vec<u64>; 2],
    /// Bound view after round 1.
    cols: Vec<Vec<Fr>>,
}

impl Table {
    fn random(log_rows: usize, rng: &mut ChaCha20Rng) -> Self {
        let rows = 1usize << log_rows;
        let mut bits: Vec<Vec<u8>> = (0..COLUMNS)
            .map(|_| (0..rows).map(|_| (rng.next_u32() & 1) as u8).collect())
            .collect();
        bits[AIN].clear();
        bits[CIN].clear();
        bits[SEL] = (0..rows).map(|row| (row >> (log_rows - 1)) as u8).collect();
        let ints = [
            (0..rows).map(|_| u64::from(rng.next_u32())).collect(),
            (0..rows).map(|_| u64::from(rng.next_u32())).collect(),
        ];
        Self {
            bits,
            ints,
            cols: Vec::new(),
        }
    }
}

struct RoundPoly([Fr; 4]);

struct SumcheckProof {
    claim: Fr,
    rounds: Vec<RoundPoly>,
    claims: Vec<Fr>,
}

/// Round polynomial `s(X) = c·l(X)·t(X)` from `t(0)`, `t(1)`, the quadratic
/// coefficient of `t`, and the eq factor `c·l(X) = l0 + (l1 - l0)·X`.
fn round_poly(l0: Fr, l1: Fr, t0: Fr, t1: Fr, t2: Fr) -> RoundPoly {
    let m = l1 - l0;
    let u = t1 - t0 - t2;
    RoundPoly([l0 * t0, l0 * u + m * t0, l0 * t2 + m * u, m * t2])
}

fn prove(
    table: &mut Table,
    tau: &[Fr],
    gammas: &[Fr],
    transcript: &mut Blake3Transcript,
) -> (SumcheckProof, Vec<Fr>, Vec<f64>) {
    let n = tau.len();
    let mut round_secs = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);
    let mut c = Fr::one();
    let mut rounds = Vec::with_capacity(n);

    // Round 0 on bits: t(0), t(1) directly (the claim is their l-weighted sum).
    let start = Instant::now();
    let tau_v = tau[n - 1];
    let e = EqPolynomial::<Fr>::evals(&tau[..n - 1], None);
    let (t0, t1, t2) = e
        .par_iter()
        .enumerate()
        .fold(
            || {
                (
                    Fr::zero(),
                    Fr::zero(),
                    Fr::zero(),
                    vec![0u8; COLUMNS],
                    vec![0u8; COLUMNS],
                )
            },
            |(mut s0, mut s1, mut s2, mut lo, mut hi), (i, &w)| {
                for (j, column) in table.bits.iter().enumerate() {
                    if j != AIN && j != CIN {
                        lo[j] = column[2 * i];
                        hi[j] = column[2 * i + 1];
                    }
                }
                let (a_lo, a_hi) = (table.ints[0][2 * i], table.ints[0][2 * i + 1]);
                let (c_lo, c_hi) = (table.ints[1][2 * i], table.ints[1][2 * i + 1]);
                s0 += w * q_bits(&lo, a_lo, c_lo, gammas);
                s1 += w * q_bits(&hi, a_hi, c_hi, gammas);
                s2 += w * q2_bits(&lo, &hi, gammas);
                (s0, s1, s2, lo, hi)
            },
        )
        .map(|(s0, s1, s2, _, _)| (s0, s1, s2))
        .reduce(
            || (Fr::zero(), Fr::zero(), Fr::zero()),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
        );
    let l0 = Fr::one() - tau_v;
    let l1 = tau_v;
    let claim = l0 * t0 + l1 * t1;
    transcript.append(&claim);
    let poly = round_poly(l0, l1, t0, t1, t2);
    for coefficient in &poly.0 {
        transcript.append(coefficient);
    }
    let r: Fr = transcript.challenge();
    let mut running = horner(&poly.0, r);
    rounds.push(poly);
    c *= eq_scalar(tau_v, r);
    challenges.push(r);
    // Bind bits → field: code lo + 2·hi ∈ {0, 1, 2, 3} ↦ {0, 1 - r, r, 1}.
    let lut = [Fr::zero(), Fr::one() - r, r, Fr::one()];
    let half = e.len();
    table.cols = table
        .bits
        .par_iter()
        .enumerate()
        .map(|(j, column)| {
            if j == AIN || j == CIN {
                let ints = &table.ints[usize::from(j == CIN)];
                (0..half)
                    .map(|i| {
                        let (lo, hi) = (Fr::from_u64(ints[2 * i]), Fr::from_u64(ints[2 * i + 1]));
                        lo + r * (hi - lo)
                    })
                    .collect()
            } else {
                (0..half)
                    .map(|i| lut[usize::from(column[2 * i] | (column[2 * i + 1] << 1))])
                    .collect()
            }
        })
        .collect();
    table.bits = Vec::new();
    table.ints = [Vec::new(), Vec::new()];
    round_secs.push(start.elapsed().as_secs_f64());

    for round in 1..n {
        let start = Instant::now();
        let tau_v = tau[n - 1 - round];
        let e = EqPolynomial::<Fr>::evals(&tau[..n - 1 - round], None);
        let (t0, t2) = e
            .par_iter()
            .enumerate()
            .fold(
                || {
                    (
                        Fr::zero(),
                        Fr::zero(),
                        vec![Fr::zero(); COLUMNS],
                        vec![Fr::zero(); COLUMNS],
                    )
                },
                |(mut s0, mut s2, mut lo, mut d), (i, &w)| {
                    for (j, column) in table.cols.iter().enumerate() {
                        lo[j] = column[2 * i];
                        d[j] = column[2 * i + 1] - column[2 * i];
                    }
                    s0 += w * q_value(&lo, gammas);
                    s2 += w * q_quad(&d, gammas);
                    (s0, s2, lo, d)
                },
            )
            .map(|(s0, s2, _, _)| (s0, s2))
            .reduce(|| (Fr::zero(), Fr::zero()), |a, b| (a.0 + b.0, a.1 + b.1));
        let l0 = c * (Fr::one() - tau_v);
        let l1 = c * tau_v;
        let t1 = (running - l0 * t0) * l1.inverse().expect("nonzero eq factor");
        let poly = round_poly(l0, l1, t0, t1, t2);
        for coefficient in &poly.0 {
            transcript.append(coefficient);
        }
        let r: Fr = transcript.challenge();
        running = horner(&poly.0, r);
        rounds.push(poly);
        c *= eq_scalar(tau_v, r);
        challenges.push(r);
        table.cols.par_iter_mut().for_each(|column| {
            let half = column.len() / 2;
            for i in 0..half {
                let (lo, hi) = (column[2 * i], column[2 * i + 1]);
                column[i] = lo + r * (hi - lo);
            }
            column.truncate(half);
        });
        round_secs.push(start.elapsed().as_secs_f64());
    }

    let claims: Vec<Fr> = table.cols.iter().map(|column| column[0]).collect();
    assert_eq!(
        claims[SEL],
        challenges[n - 1],
        "selector is the last-bound row bit"
    );
    for claim in &claims[..SEL] {
        transcript.append(claim);
    }
    (
        SumcheckProof {
            claim,
            rounds,
            claims,
        },
        challenges,
        round_secs,
    )
}

fn verify_sumcheck(
    proof: &SumcheckProof,
    tau: &[Fr],
    gammas: &[Fr],
    transcript: &mut Blake3Transcript,
) -> Vec<Fr> {
    let n = tau.len();
    transcript.append(&proof.claim);
    let mut running = proof.claim;
    let mut c = Fr::one();
    let mut challenges = Vec::with_capacity(n);
    for (round, poly) in proof.rounds.iter().enumerate() {
        let at_one: Fr = poly.0.iter().sum();
        assert_eq!(poly.0[0] + at_one, running, "round {round}: s(0) + s(1)");
        for coefficient in &poly.0 {
            transcript.append(coefficient);
        }
        let r: Fr = transcript.challenge();
        running = horner(&poly.0, r);
        c *= eq_scalar(tau[n - 1 - round], r);
        challenges.push(r);
    }
    let mut values = proof.claims.clone();
    values[SEL] = challenges[n - 1];
    assert_eq!(
        running,
        c * q_value(&values, gammas),
        "final sumcheck check"
    );
    for claim in &proof.claims[..SEL] {
        transcript.append(claim);
    }
    challenges
}

fn bench(log_rows: usize, setup: &HyperKZGProverSetup<Bn254>, bases: &[Bn254G1]) {
    let rows = 1usize << log_rows;
    let mut rng = ChaCha20Rng::seed_from_u64(log_rows as u64);
    let mut table = Table::random(log_rows, &mut rng);
    println!(
        "\n== rows 2^{log_rows} = {rows}, committed columns {COMMITTED}, wired {}, constraints {CONSTRAINTS}, committed bits {}",
        COLUMNS - COMMITTED - 1,
        rows * COMMITTED
    );

    // Commit: one batch addition per bit column over the SRS powers.
    let start = Instant::now();
    let index_sets: Vec<Vec<usize>> = table.bits[..COMMITTED]
        .par_iter()
        .map(|column| {
            column
                .iter()
                .enumerate()
                .filter_map(|(row, bit)| (*bit == 1).then_some(row))
                .collect()
        })
        .collect();
    let commitments: Vec<HyperKZGCommitment<Bn254>> =
        batch_g1_additions_multi(&bases[..rows], &index_sets)
            .into_iter()
            .map(HyperKZGCommitment::new)
            .collect();
    let commit_secs = start.elapsed().as_secs_f64();
    println!("commit (batch additions, {COMMITTED} columns): {commit_secs:.3} s");

    let column0: Vec<Fr> = table.bits[0].iter().map(|b| Fr::from_u8(*b)).collect();
    let (expected, ()) = Scheme::commit(&column0, setup).expect("kzg commit");
    assert_eq!(
        commitments[0], expected,
        "batch addition matches the SRS commit"
    );

    let bit_columns: Vec<Vec<u8>> = table.bits[..COMMITTED].to_vec();
    let mut transcript = Blake3Transcript::new(b"wrapper-bench");
    for commitment in &commitments {
        transcript.append(commitment);
    }
    let tau: Vec<Fr> = transcript.challenge_vector(log_rows);
    let gammas: Vec<Fr> = transcript.challenge_scalar_powers(CONSTRAINTS);

    let start = Instant::now();
    let (proof, challenges, round_secs) = prove(&mut table, &tau, &gammas, &mut transcript);
    let sumcheck_secs = start.elapsed().as_secs_f64();
    println!(
        "sumcheck ({log_rows} rounds, degree 3): {sumcheck_secs:.3} s  [round 0 (bits) {:.3} s, round 1 {:.3} s]",
        round_secs[0], round_secs[1]
    );

    // Opening point in HyperKZG/`evals` order: last-bound variable first.
    let point: Vec<Fr> = challenges.iter().rev().copied().collect();
    let start = Instant::now();
    let rho: Fr = transcript.challenge();
    let rho_powers: Vec<Fr> = std::iter::successors(Some(Fr::one()), |p| Some(*p * rho))
        .take(COMMITTED)
        .collect();
    let rlc: Vec<Fr> = (0..rows)
        .into_par_iter()
        .map(|row| {
            bit_columns
                .iter()
                .zip(&rho_powers)
                .filter(|(column, _)| column[row] == 1)
                .fold(Fr::zero(), |acc, (_, p)| acc + *p)
        })
        .collect();
    let rlc_secs = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let combined = Scheme::combine(&commitments, &rho_powers);
    let combined_eval = proof.claims[..COMMITTED]
        .iter()
        .zip(&rho_powers)
        .fold(Fr::zero(), |acc, (v, p)| acc + *v * *p);
    let opening = Scheme::open(setup, &rlc, &point, &mut transcript).expect("open");
    let open_secs = start.elapsed().as_secs_f64();
    println!("rlc {rlc_secs:.3} s, combine + open: {open_secs:.3} s");
    assert_eq!(
        Polynomial::new(rlc).evaluate(&point),
        combined_eval,
        "RLC evaluation matches the batched claims"
    );

    let start = Instant::now();
    let mut vt = Blake3Transcript::new(b"wrapper-bench");
    for commitment in &commitments {
        vt.append(commitment);
    }
    let vtau: Vec<Fr> = vt.challenge_vector(log_rows);
    let vgammas: Vec<Fr> = vt.challenge_scalar_powers(CONSTRAINTS);
    let vchallenges = verify_sumcheck(&proof, &vtau, &vgammas, &mut vt);
    let vpoint: Vec<Fr> = vchallenges.iter().rev().copied().collect();
    let vrho: Fr = vt.challenge();
    let vrho_powers: Vec<Fr> = std::iter::successors(Some(Fr::one()), |p| Some(*p * vrho))
        .take(COMMITTED)
        .collect();
    let vcombined = Scheme::combine(&commitments, &vrho_powers);
    let veval = proof.claims[..COMMITTED]
        .iter()
        .zip(&vrho_powers)
        .fold(Fr::zero(), |acc, (v, p)| acc + *v * *p);
    let vk = Scheme::verifier_setup(setup);
    Scheme::verify(&vk, &vcombined, &vpoint, &veval, &opening, &mut vt).expect("opening verifies");
    assert_eq!(vcombined, combined);
    let verify_secs = start.elapsed().as_secs_f64();
    println!("verify (sumcheck + HyperKZG): {verify_secs:.3} s");

    let opening_bytes = bincode::serde::encode_to_vec(&opening, bincode::config::standard())
        .expect("encode opening")
        .len();
    let round_bytes = log_rows * 4 * 32;
    let claim_bytes = (COLUMNS - 1) * 32;
    println!(
        "proof bytes: rounds {round_bytes} + claims {claim_bytes} ({} committed + {} wired) + opening {opening_bytes} = {}",
        COMMITTED,
        COLUMNS - COMMITTED - 1,
        round_bytes + claim_bytes + opening_bytes + 32
    );
    println!(
        "prover total: {:.3} s (commit {commit_secs:.3} + sumcheck {sumcheck_secs:.3} + rlc {rlc_secs:.3} + open {open_secs:.3})",
        commit_secs + sumcheck_secs + rlc_secs + open_secs
    );
}

fn main() {
    let sizes: Vec<usize> = {
        let args: Vec<usize> = std::env::args()
            .skip(1)
            .map(|a| a.parse().expect("log_rows"))
            .collect();
        if args.is_empty() {
            vec![16, 17, 18]
        } else {
            args
        }
    };
    let max_rows = 1usize << sizes.iter().copied().max().expect("one size");

    let start = Instant::now();
    let mut rng = ChaCha20Rng::seed_from_u64(0xb3);
    let beta = Fr::random(&mut rng);
    let g1 = Bn254::g1_generator();
    let setup = Scheme::setup_from_secret(beta, max_rows, g1, Bn254::g2_generator());
    let mut powers = Vec::with_capacity(max_rows);
    let mut cur = Fr::one();
    for _ in 0..max_rows {
        powers.push(cur);
        cur *= beta;
    }
    let bases: Vec<Bn254G1> = powers.par_iter().map(|p| g1.scalar_mul(p)).collect();
    println!(
        "setup: SRS 2^{} in {:.1} s, rayon threads {}",
        max_rows.ilog2(),
        start.elapsed().as_secs_f64(),
        rayon::current_num_threads()
    );

    for log_rows in sizes {
        bench(log_rows, &setup, &bases);
    }
}
