//! Microbench for the CPU trace one-hot commit accumulate at the production
//! D=512 / K=256 geometry, reporting nanoseconds per committed hot entry.
//!
//! `ROWS_LOG2` (default 20) sets the trace length, `POSITIONS` (default
//! 262144) the positions per root block, `REPS` (default 3) the timed
//! repetitions, `DENSITY_PCT` (default 56, Fibonacci-like) the per-column hot
//! probability. `VERIFY=1` additionally checks the kernel output against a
//! canonical shift-accumulate oracle (use a small `ROWS_LOG2` for that).
#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "benchmark harness with fixed synthetic geometry that reports to the terminal"
)]

use std::sync::Arc;
use std::time::Instant;

use akita_algebra::CyclotomicRing;
use akita_prover::compute::{CommitInnerPlan, RootCommitKernel};
use akita_prover::{
    AkitaProverSetup, ComputeBackendSetup, CpuBackend, RootCommitSource, RootPolyMeta,
};
use akita_types::SetupMatrixCapacity;
use jolt_akita::{AkitaField, OwnedTraceOneHotRows, TracePackedOneHot};

const D: usize = 512;
const K: usize = 256;
const COLUMNS: usize = 29;
const CAPACITY: usize = 32;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map_or(default, |v| v.parse().expect("numeric env var"))
}

fn mix(row: usize, column: usize) -> u64 {
    let mut x = (row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((column as u64) << 40);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 29;
    x
}

fn main() {
    let rows_log2 = env_usize("ROWS_LOG2", 20);
    let positions = env_usize("POSITIONS", 262_144);
    let reps = env_usize("REPS", 3);
    let density = env_usize("DENSITY_PCT", 56) as u64;
    let verify = std::env::var("VERIFY").is_ok();
    let num_rows = 1usize << rows_log2;

    let rows = OwnedTraceOneHotRows::from_row_fn(K, CAPACITY, COLUMNS, num_rows, |row, lanes| {
        for (column, lane) in lanes.iter_mut().enumerate() {
            let h = mix(row, column);
            *lane = if h % 100 < density {
                1 + (h >> 8) as u8 % 255
            } else {
                0
            };
        }
    })
    .unwrap();
    let hot_entries = rows.lanes().iter().filter(|&&lane| lane != 0).count();
    let lanes = rows.lanes().to_vec();
    let source = TracePackedOneHot::new(K, CAPACITY, Arc::new(rows)).unwrap();

    let plan = CommitInnerPlan {
        n_a: 1,
        num_positions_per_block: positions,
        num_digits_inner: 1,
        log_basis_inner: 8,
    };
    let setup_start = Instant::now();
    let setup = AkitaProverSetup::<AkitaField>::generate_with_capacity(
        RootPolyMeta::<AkitaField>::num_vars(&source),
        1,
        SetupMatrixCapacity {
            num_field_elements: plan.n_a * positions * D,
        },
    )
    .unwrap();
    let cpu = CpuBackend::DEFAULT;
    let prepared = cpu.prepare_setup(&setup).unwrap();
    eprintln!(
        "setup ready in {:.2}s; rows=2^{rows_log2} columns={COLUMNS} hot_entries={hot_entries} positions={positions}",
        setup_start.elapsed().as_secs_f64()
    );

    let mut best = f64::INFINITY;
    let mut witness = None;
    for rep in 0..reps {
        let view = RootCommitSource::<AkitaField, D>::commit_view(&source).unwrap();
        let start = Instant::now();
        let out = cpu.commit_inner_group(&prepared, vec![view], plan).unwrap();
        let secs = start.elapsed().as_secs_f64();
        best = best.min(secs);
        println!(
            "rep {rep}: {secs:.3}s  {:.2} ns/hot_entry",
            secs * 1e9 / hot_entries as f64
        );
        witness = Some(out);
    }
    println!(
        "BEST {best:.3}s  {:.2} ns/hot_entry  threads={}",
        best * 1e9 / hot_entries as f64,
        rayon::current_num_threads()
    );

    if verify {
        let witness = witness.unwrap();
        let output = witness[0].inner_rows.as_ring_slice::<D>().unwrap();
        let expanded = cpu.prepared_expanded_setup(&prepared);
        let a_view = expanded
            .shared_matrix()
            .ring_view::<D>(plan.n_a, positions * plan.num_digits_inner)
            .unwrap();
        let a_row = a_view.rows().next().unwrap();
        let rows_per_ring = D / K;
        let rings_per_column = num_rows / rows_per_ring;
        let blocks_per_column = rings_per_column / positions;
        assert_eq!(blocks_per_column * positions, rings_per_column);
        let mut expected =
            vec![CyclotomicRing::<AkitaField, D>::zero(); CAPACITY * blocks_per_column];
        for column in 0..COLUMNS {
            for block in 0..blocks_per_column {
                let acc = &mut expected[column * blocks_per_column + block];
                for (position, source) in a_row.iter().enumerate().take(positions) {
                    let ring = block * positions + position;
                    for row_offset in 0..rows_per_ring {
                        let row = ring * rows_per_ring + row_offset;
                        let hot = lanes[row * COLUMNS + column];
                        if hot != 0 {
                            source.shift_accumulate_into(acc, row_offset * K + usize::from(hot));
                        }
                    }
                }
            }
        }
        assert_eq!(output.len(), expected.len(), "row count");
        let mismatches = output.iter().zip(&expected).filter(|(a, b)| a != b).count();
        assert_eq!(
            mismatches, 0,
            "kernel output differs from the canonical oracle"
        );
        println!(
            "VERIFY OK: {} rows match the canonical oracle",
            expected.len()
        );
    }
}
