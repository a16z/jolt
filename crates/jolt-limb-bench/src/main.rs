//! Lane M2 measurement harness: the non-native limb relation table
//! (one row = one `z ≡ Σ x_i·y_i (mod q)` over BN254 Fq), committed with
//! HyperKZG (small-scalar MSMs for 16-bit chunks) and proven by one
//! degree-3 sumcheck + one batched HyperKZG opening.
//!
//! Usage: `limb-relation <log2 rows> <t products per row> [commit-operands] [tamper]`.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::print_stdout,
    reason = "measurement harness"
)]

mod relation;
mod table;

use std::time::Instant;

use jolt_crypto::{Bn254, PairingGroup};
use jolt_field::{Fr, One, Ring, Zero};
use jolt_hyperkzg::{
    HyperKZGCommitment, HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup,
};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use rayon::prelude::*;

use relation::{Prover, Public, TABLE_LOG};
use table::{Table, CHUNK_COLUMNS};

type Scheme = HyperKZGScheme<Bn254>;

/// Wall-clock milliseconds per phase.
struct Timings {
    generate: f64,
    commit_chunks: f64,
    commit_inverses: f64,
    commit_multiplicities: f64,
    sumcheck: f64,
    open: f64,
    verify: f64,
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

fn setup(rows: usize) -> (HyperKZGProverSetup<Bn254>, HyperKZGVerifierSetup<Bn254>) {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5e7);
    let pk = Scheme::setup(&mut rng, rows, Bn254::g1_generator(), Bn254::g2_generator());
    let vk = Scheme::verifier_setup(&pk);
    (pk, vk)
}

fn powers(base: Fr, count: usize) -> Vec<Fr> {
    let mut out = Vec::with_capacity(count);
    let mut acc = Fr::one();
    for _ in 0..count {
        out.push(acc);
        acc *= base;
    }
    out
}

/// One full run: generate, commit, prove, open, verify. `tamper` mutates the
/// table after generation; the prover then runs without self-checks and the
/// verifier must reject.
fn run(
    log_rows: usize,
    t: usize,
    commit_operands: bool,
    pk: &HyperKZGProverSetup<Bn254>,
    vk: &HyperKZGVerifierSetup<Bn254>,
    tamper: Option<fn(&mut Table)>,
) -> (Result<(), &'static str>, Timings, usize) {
    let rows = 1usize << log_rows;
    assert!(
        log_rows >= TABLE_LOG,
        "rows must cover the 2^16 range table"
    );
    let bases = &pk.g1_powers()[..rows];

    let start = Instant::now();
    let mut table = Table::generate(rows, t, commit_operands, 0x00c0_ffee);
    let generate = ms(start);
    if let Some(tamper) = tamper {
        tamper(&mut table);
    }

    let mut transcript = Blake2bTranscript::new(b"m2-limb-relation");

    // Phase 1: commit the 16-bit chunk columns (u16 MSM; a tampered column
    // holds a non-u16 value and goes through the full-width path).
    let start = Instant::now();
    let chunk_commitments: Vec<HyperKZGCommitment<Bn254>> = (0..table.chunks.len())
        .into_par_iter()
        .map(|column| {
            let point = if table.column_has_override(column) {
                Bn254::g1_affine_msm(bases, &table.chunk_column_fr(column))
            } else {
                Bn254::g1_affine_msm_small(bases, &table.chunks[column])
            };
            HyperKZGCommitment::new(point)
        })
        .collect();
    let commit_chunks = ms(start);
    for commitment in &chunk_commitments {
        transcript.append(commitment);
    }
    let alpha: Fr = transcript.challenge();

    // Phase 2: LogUp inverses h = 1/(α − chunk) (full-width columns) and the
    // range-table multiplicities.
    let (inverses, multiplicities) = table.logup_columns(alpha);
    let start = Instant::now();
    let inverse_commitments: Vec<HyperKZGCommitment<Bn254>> = inverses
        .par_iter()
        .map(|column| HyperKZGCommitment::new(Bn254::g1_affine_msm(bases, column)))
        .collect();
    let commit_inverses = ms(start);
    let start = Instant::now();
    let multiplicity_commitment = HyperKZGCommitment::new(Bn254::g1_affine_msm_small(
        &bases[..1 << TABLE_LOG],
        &multiplicities,
    ));
    let commit_multiplicities = ms(start);
    for commitment in &inverse_commitments {
        transcript.append(commitment);
    }
    transcript.append(&multiplicity_commitment);

    let num_chunk_columns = table.chunks.len();
    let public = Public::draw(
        &mut transcript,
        log_rows,
        t,
        num_chunk_columns,
        alpha,
        commit_operands,
    );

    // Phase 3: the sumcheck.
    let start = Instant::now();
    let mut prover = Prover::new(table, inverses, multiplicities, &public);
    let (round_polys, point, claims) = prover.prove(&public, &mut transcript, tamper.is_none());
    let sumcheck = ms(start);

    // Phase 4: one HyperKZG opening of the RLC of every committed column.
    let start = Instant::now();
    let rho: Fr = transcript.challenge();
    let rho_powers = powers(rho, prover.num_committed());
    let rlc = prover.rlc(&rho_powers);
    let all_commitments: Vec<HyperKZGCommitment<Bn254>> = chunk_commitments
        .iter()
        .chain(&inverse_commitments)
        .chain(std::iter::once(&multiplicity_commitment))
        .copied()
        .collect();
    assert_eq!(all_commitments.len(), prover.num_committed());
    let hyperkzg_point: Vec<Fr> = point.iter().rev().copied().collect();
    let opening = Scheme::open(pk, &rlc, &hyperkzg_point, &mut transcript).unwrap();
    let open = ms(start);
    drop(prover);

    // Verifier (native).
    let start = Instant::now();
    let mut verifier_transcript = Blake2bTranscript::new(b"m2-limb-relation");
    for commitment in &chunk_commitments {
        verifier_transcript.append(commitment);
    }
    let alpha_v: Fr = verifier_transcript.challenge();
    for commitment in &inverse_commitments {
        verifier_transcript.append(commitment);
    }
    verifier_transcript.append(&multiplicity_commitment);
    let public_v = Public::draw(
        &mut verifier_transcript,
        log_rows,
        t,
        num_chunk_columns,
        alpha_v,
        commit_operands,
    );
    let accepted = public_v
        .verify(&round_polys, &claims, &mut verifier_transcript)
        .and_then(|point_v| {
            let rho_v: Fr = verifier_transcript.challenge();
            let weights = powers(rho_v, all_commitments.len());
            let combined = Scheme::combine(&all_commitments, &weights);
            let claimed: Fr = claims
                .committed
                .iter()
                .zip(&weights)
                .fold(Fr::zero(), |acc, (claim, weight)| acc + *claim * *weight);
            let point_v: Vec<Fr> = point_v.iter().rev().copied().collect();
            Scheme::verify(
                vk,
                &combined,
                &point_v,
                &claimed,
                &opening,
                &mut verifier_transcript,
            )
            .map_err(|_| "hyperkzg")
        });
    let verify = ms(start);

    let proof_bytes = 32
        * (round_polys
            .iter()
            .map(|coefficients| coefficients.len() - 1)
            .sum::<usize>()
            + claims.committed.len()
            + claims.operand_limbs.len()
            + all_commitments.len()
            + opening.com.len()
            + 1
            + opening.v.iter().map(Vec::len).sum::<usize>());

    (
        accepted,
        Timings {
            generate,
            commit_chunks,
            commit_inverses,
            commit_multiplicities,
            sumcheck,
            open,
            verify,
        },
        proof_bytes,
    )
}

/// Flips one bit of a z chunk: the CRT identity of that row fails.
fn tamper_crt(table: &mut Table) {
    table.chunks[0][7] ^= 1;
}

/// Moves 2^16 from chunk 1 into chunk 0 of one row: the limb value (hence every
/// CRT identity) is unchanged, but chunk 0 is out of range.
fn tamper_range(table: &mut Table) {
    let row = (0..table.rows)
        .find(|&row| table.chunks[1][row] >= 1)
        .unwrap();
    table.chunks[1][row] -= 1;
    let value = Fr::from_u64(u64::from(table.chunks[0][row])) + Fr::pow2(16);
    table.overrides.push((0, row, value));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let log_rows: usize = args.get(1).map_or(16, |s| s.parse().unwrap());
    let t: usize = args.get(2).map_or(24, |s| s.parse().unwrap());
    let commit_operands = args.iter().any(|s| s == "commit-operands");
    let tamper = args.iter().any(|s| s == "tamper");
    let rows = 1usize << log_rows;

    let start = Instant::now();
    let (pk, vk) = setup(rows);
    println!("setup 2^{log_rows}: {:.1} ms", ms(start));

    if tamper {
        for (name, tamper) in [
            ("crt (flip z chunk bit)", tamper_crt as fn(&mut Table)),
            ("range (chunk += 2^16, neighbour -= 1)", tamper_range),
        ] {
            let (accepted, _, _) = run(log_rows, t, commit_operands, &pk, &vk, Some(tamper));
            println!("tamper {name}: verifier says {accepted:?}");
            assert!(accepted.is_err(), "tampered run must be rejected");
        }
        return;
    }

    let columns = CHUNK_COLUMNS + if commit_operands { 32 * t } else { 0 };
    println!(
        "rows=2^{log_rows} t={t} commit_operands={commit_operands} chunk_columns={columns} inverse_columns={columns} operand_limb_polys={} degree=3",
        6 * t
    );
    let (accepted, timings, proof_bytes) = run(log_rows, t, commit_operands, &pk, &vk, None);
    assert!(accepted.is_ok(), "honest run must verify: {accepted:?}");
    let prover_ms = timings.commit_chunks
        + timings.commit_inverses
        + timings.commit_multiplicities
        + timings.sumcheck
        + timings.open;
    println!("generate            {:>9.1} ms", timings.generate);
    println!(
        "commit chunks (u16) {:>9.1} ms  ({:.2} ms/column)",
        timings.commit_chunks,
        timings.commit_chunks / columns as f64
    );
    println!(
        "commit inverses     {:>9.1} ms  ({:.2} ms/column)",
        timings.commit_inverses,
        timings.commit_inverses / columns as f64
    );
    println!(
        "commit multiplicity {:>9.1} ms",
        timings.commit_multiplicities
    );
    println!("sumcheck            {:>9.1} ms", timings.sumcheck);
    println!("rlc + open          {:>9.1} ms", timings.open);
    println!("prover total        {:>9.1} ms", prover_ms);
    println!("verify              {:>9.1} ms", timings.verify);
    println!("proof bytes         {proof_bytes}");
}
