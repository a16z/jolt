//! Lane M2 measurement harness: the non-native limb relation table
//! (one row = one `z ≡ Σ x_i·y_i (mod q)` over BN254 Fq), committed with
//! HyperKZG (small-scalar MSMs for 16-bit chunks) and proven by one
//! degree-3 sumcheck + one batched HyperKZG opening.
//!
//! Usage: `limb-relation <log2 rows> <t products per row> [s=<group size>] [k=<columns per commitment>] [commit-operands] [tamper]`.

#![expect(
    clippy::cast_precision_loss,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::print_stdout,
    reason = "measurement harness"
)]

use std::time::Instant;

use jolt_crypto::{Bn254, PairingGroup};
use jolt_field::{Fr, One, Ring, Zero};
use jolt_hyperkzg::{
    HyperKZGCommitment, HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup,
};
use jolt_limb_bench::pack;
use jolt_limb_bench::relation::{Prover, Public, TABLE_LOG};
use jolt_limb_bench::table::{Table, CHUNK_COLUMNS};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use rayon::prelude::*;

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

/// Proof size itemized in bytes.
struct ProofBytes {
    commitments: usize,
    rounds: usize,
    claims: usize,
    operand_claims: usize,
    opening: usize,
}

impl ProofBytes {
    fn total(&self) -> usize {
        self.commitments + self.rounds + self.claims + self.operand_claims + self.opening
    }
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

fn setup(entries: usize) -> (HyperKZGProverSetup<Bn254>, HyperKZGVerifierSetup<Bn254>) {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5e7);
    let pk = Scheme::setup(
        &mut rng,
        entries,
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    );
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
#[expect(clippy::too_many_arguments, reason = "measurement knobs")]
fn run(
    log_rows: usize,
    t: usize,
    group_size: usize,
    k: usize,
    commit_operands: bool,
    pk: &HyperKZGProverSetup<Bn254>,
    vk: &HyperKZGVerifierSetup<Bn254>,
    tamper: Option<fn(&mut Table)>,
) -> (Result<(), &'static str>, Timings, ProofBytes) {
    let rows = 1usize << log_rows;
    assert!(
        log_rows >= TABLE_LOG,
        "rows must cover the 2^16 range table"
    );
    let log_k = pack::log2_exact(k);
    let slot_bases = |column: usize| pack::slot_bases(pk, rows, k, column);

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
            let bases = slot_bases(column);
            let point = if table.column_has_override(column) {
                Bn254::g1_affine_msm(bases, &table.chunk_column_fr(column))
            } else {
                Bn254::g1_affine_msm_small(bases, &table.chunks[column])
            };
            HyperKZGCommitment::new(point)
        })
        .collect();
    let chunk_groups = pack::group_commitments(&chunk_commitments, k);
    let commit_chunks = ms(start);
    for commitment in &chunk_groups {
        transcript.append(commitment);
    }
    let alpha: Fr = transcript.challenge();

    // Phase 2: grouped LogUp helpers h_g = 1/Π_{i∈g}(α − chunk_i) (full-width
    // columns) and the range-table multiplicities.
    let (inverses, multiplicities) = table.logup_columns(alpha, group_size);
    let num_chunk_columns = table.chunks.len();
    let start = Instant::now();
    let inverse_commitments: Vec<HyperKZGCommitment<Bn254>> = inverses
        .par_iter()
        .enumerate()
        .map(|(i, column)| HyperKZGCommitment::new(Bn254::g1_affine_msm(slot_bases(i), column)))
        .collect();
    let commit_inverses = ms(start);
    let start = Instant::now();
    let multiplicity_commitment = HyperKZGCommitment::new(Bn254::g1_affine_msm_small(
        &slot_bases(inverses.len())[..1 << TABLE_LOG],
        &multiplicities,
    ));
    let commit_multiplicities = ms(start);
    let helper_columns: Vec<HyperKZGCommitment<Bn254>> = inverse_commitments
        .iter()
        .chain(std::iter::once(&multiplicity_commitment))
        .copied()
        .collect();
    let helper_groups = pack::group_commitments(&helper_columns, k);
    for commitment in &helper_groups {
        transcript.append(commitment);
    }
    let packed_commitments: Vec<HyperKZGCommitment<Bn254>> =
        chunk_groups.iter().chain(&helper_groups).copied().collect();
    let num_committed = chunk_commitments.len() + helper_columns.len();

    let public = Public::draw(
        &mut transcript,
        log_rows,
        t,
        num_chunk_columns,
        group_size,
        alpha,
        commit_operands,
    );

    // Phase 3: the sumcheck.
    let start = Instant::now();
    let mut prover = Prover::new(table, inverses, multiplicities, &public);
    let (round_polys, point, claims) = prover.prove(&public, &mut transcript, tamper.is_none());
    let sumcheck = ms(start);
    assert_eq!(num_committed, prover.num_committed());

    // Phase 4: one HyperKZG opening of `Σ_g ρ^g·P_g` at `(s_lo, r)`; chunk
    // columns and helper columns are packed in separate groups.
    let start = Instant::now();
    for claim in claims.committed.iter().chain(&claims.operand_limbs) {
        transcript.append(claim);
    }
    let rho: Fr = transcript.challenge();
    let weights = powers(rho, packed_commitments.len());
    let s_lo: Vec<Fr> = (0..log_k).map(|_| transcript.challenge()).collect();
    let chunk_slots = chunk_groups.len() * k;
    let padded_column = |column: usize| {
        if column < chunk_slots {
            (column < num_chunk_columns).then_some(column)
        } else {
            let helper = column - chunk_slots;
            (helper < helper_columns.len()).then_some(num_chunk_columns + helper)
        }
    };
    let combined = pack::combine(
        rows,
        k,
        packed_commitments.len() * k,
        &weights,
        |column, row| padded_column(column).map_or(Fr::zero(), |c| prover.committed(c, row)),
    );
    let padded_claims: Vec<Fr> = (0..packed_commitments.len() * k)
        .map(|column| padded_column(column).map_or(Fr::zero(), |c| claims.committed[c]))
        .collect();
    let row_point: Vec<Fr> = point.iter().rev().copied().collect();
    let hyperkzg_point = pack::point(&s_lo, &row_point);
    let opening = Scheme::open(pk, &combined, &hyperkzg_point, &mut transcript).unwrap();
    let open = ms(start);
    drop(prover);

    // Verifier (native).
    let start = Instant::now();
    let mut verifier_transcript = Blake2bTranscript::new(b"m2-limb-relation");
    for commitment in &chunk_groups {
        verifier_transcript.append(commitment);
    }
    let alpha_v: Fr = verifier_transcript.challenge();
    for commitment in &helper_groups {
        verifier_transcript.append(commitment);
    }
    let public_v = Public::draw(
        &mut verifier_transcript,
        log_rows,
        t,
        num_chunk_columns,
        group_size,
        alpha_v,
        commit_operands,
    );
    let accepted = public_v
        .verify(&round_polys, &claims, &mut verifier_transcript)
        .and_then(|point_v| {
            for claim in claims.committed.iter().chain(&claims.operand_limbs) {
                verifier_transcript.append(claim);
            }
            let rho_v: Fr = verifier_transcript.challenge();
            let weights = powers(rho_v, packed_commitments.len());
            let s_lo: Vec<Fr> = (0..log_k)
                .map(|_| verifier_transcript.challenge())
                .collect();
            let combined = Scheme::combine(&packed_commitments, &weights);
            let claimed = pack::combined_claim(&padded_claims, k, &weights, &s_lo);
            let row_point: Vec<Fr> = point_v.iter().rev().copied().collect();
            Scheme::verify(
                vk,
                &combined,
                &pack::point(&s_lo, &row_point),
                &claimed,
                &opening,
                &mut verifier_transcript,
            )
            .map_err(|_| "hyperkzg")
        });
    let verify = ms(start);

    let bytes = ProofBytes {
        commitments: 32 * packed_commitments.len(),
        rounds: 32
            * round_polys
                .iter()
                .map(|coefficients| coefficients.len() - 1)
                .sum::<usize>(),
        claims: 32 * claims.committed.len(),
        operand_claims: 32 * claims.operand_limbs.len(),
        opening: 32 * (opening.com.len() + 1 + opening.v.iter().map(Vec::len).sum::<usize>()),
    };

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
        bytes,
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
    let group_size: usize = args
        .iter()
        .find_map(|s| s.strip_prefix("s="))
        .map_or(1, |s| s.parse().unwrap());
    let k: usize = args
        .iter()
        .find_map(|s| s.strip_prefix("k="))
        .map_or(1, |s| s.parse().unwrap());
    let tamper = args.iter().any(|s| s == "tamper");
    let rows = 1usize << log_rows;

    let start = Instant::now();
    let (pk, vk) = setup(rows * k);
    println!(
        "setup 2^{}: {:.1} ms",
        log_rows + pack::log2_exact(k),
        ms(start)
    );

    if tamper {
        for (name, tamper) in [
            ("crt (flip z chunk bit)", tamper_crt as fn(&mut Table)),
            ("range (chunk += 2^16, neighbour -= 1)", tamper_range),
        ] {
            let (accepted, _, _) = run(
                log_rows,
                t,
                group_size,
                k,
                commit_operands,
                &pk,
                &vk,
                Some(tamper),
            );
            println!("tamper {name}: verifier says {accepted:?}");
            assert!(accepted.is_err(), "tampered run must be rejected");
        }
        return;
    }

    let columns = CHUNK_COLUMNS + if commit_operands { 32 * t } else { 0 };
    let helper_columns = columns.div_ceil(group_size);
    println!(
        "rows=2^{log_rows} t={t} s={group_size} k={k} commit_operands={commit_operands} chunk_columns={columns} helper_columns={helper_columns} operand_limb_polys={} degree={} commitments={}",
        6 * t,
        (group_size + 2).max(3),
        pack::groups(columns, k) + pack::groups(helper_columns + 1, k)
    );
    let (accepted, timings, bytes) =
        run(log_rows, t, group_size, k, commit_operands, &pk, &vk, None);
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
        "commit helpers      {:>9.1} ms  ({:.2} ms/column)",
        timings.commit_inverses,
        timings.commit_inverses / helper_columns as f64
    );
    println!(
        "commit multiplicity {:>9.1} ms",
        timings.commit_multiplicities
    );
    println!("sumcheck            {:>9.1} ms", timings.sumcheck);
    println!("rlc + open          {:>9.1} ms", timings.open);
    println!("prover total        {:>9.1} ms", prover_ms);
    println!("verify              {:>9.1} ms", timings.verify);
    println!(
        "proof bytes         {} = commitments {} + rounds {} + column claims {} + operand claims {} + opening {}",
        bytes.total(),
        bytes.commitments,
        bytes.rounds,
        bytes.claims,
        bytes.operand_claims,
        bytes.opening
    );
}
