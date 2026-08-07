//! Chunked capture/replay gate for two-pass parallel tracing.
//!
//! For each golden guest: run pass-1 (execute-only, trace-equivalent state),
//! cut chunk checkpoints every N ticks with pooled full-memory snapshots,
//! replay every chunk through a single resident trace-mode worker
//! (interleaved with pass-1, one-stage pipeline), and require:
//!   (a) the worker's end state to match the next checkpoint bit-exactly at
//!       every boundary (and pass-1's final state after the last chunk),
//!   (b) the concatenated replayed rows to equal the serial trace() rows,
//!   (c) final memory and JoltDevice outputs/panic to match trace()'s.
//!
//! Usage:
//!   cargo run --release -p tracer --example trace_chunk_replay -- [filter]

// Link inline crates so their inventory registrations reach the tracer.
extern crate jolt_inlines_keccak256 as _;
extern crate jolt_inlines_sha2 as _;

#[path = "support/mod.rs"]
mod support;

use support::chain_input;
use tracer::instruction::Cycle;
use tracer::parallel::{ChunkWorker, PassOne, SnapshotPool};

/// Same (guest, input) pairs as the golden-trace gate, plus per-guest chunk
/// sizes in ticks (chosen to force multiple chunks; muldiv gets a tiny one).
fn golden_cases() -> Vec<(&'static str, Vec<u8>, usize)> {
    vec![
        ("sha2-chain-guest", chain_input(300), 5_000),
        ("sha3-chain-guest", chain_input(235), 2_500),
        (
            "fibonacci-guest",
            postcard::to_stdvec(&84_000u32).unwrap(),
            100_000,
        ),
        (
            "btreemap-guest",
            postcard::to_stdvec(&650u32).unwrap(),
            50_000,
        ),
        (
            "muldiv-guest",
            postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap(),
            37,
        ),
    ]
}

fn run_case(guest: &str, input: &[u8], chunk_ticks: usize) -> Result<(usize, usize), String> {
    let (elf, _, memory_config) = support::build_guest(guest);

    // Serial reference.
    let (_, serial_rows, serial_memory, serial_device, _) =
        tracer::trace(&elf, None, input, &[], &[], &memory_config, None);

    // Pass-1 + interleaved single-worker replay.
    let mut pass1 = PassOne::new(tracer::create_emulator(
        &elf,
        None,
        input,
        &[],
        &[],
        &memory_config,
        None,
    ));
    let mut pool = SnapshotPool::new();
    let mut worker = ChunkWorker::new(pass1.emulator());
    let mut replayed: Vec<Cycle> = Vec::new();
    let mut chunk_count = 0usize;
    loop {
        let checkpoint = pass1.checkpoint();
        let image = pool.capture(&pass1.emulator().get_cpu().mmu.memory.memory);
        let mut ticks = 0;
        while ticks < chunk_ticks && pass1.step() {
            ticks += 1;
        }
        if ticks == 0 {
            pool.put(image);
            break;
        }
        // Boundary reference: pass-1 state right after this chunk (its final
        // state when the program just terminated).
        let boundary = pass1.checkpoint();

        let previous = worker.install_chunk(&checkpoint, image);
        pool.put(previous);
        worker.run_ticks(ticks, &mut replayed);
        if let Some(diff) = boundary.diff_vs_cpu(worker.cpu()) {
            return Err(format!(
                "boundary mismatch after chunk {chunk_count}: {diff}"
            ));
        }
        chunk_count += 1;
        if pass1.is_done() {
            break;
        }
    }

    if replayed != serial_rows {
        let first_diff = replayed
            .iter()
            .zip(serial_rows.iter())
            .position(|(a, b)| a != b);
        return Err(format!(
            "rows differ: {} replayed vs {} serial, first diff at {:?}",
            replayed.len(),
            serial_rows.len(),
            first_diff
        ));
    }

    let worker_memory = worker.cpu().mmu.memory.memory.materialized_nonzero_bytes();
    if worker_memory != serial_memory.materialized_nonzero_bytes() {
        return Err("final memory diverged".to_string());
    }
    let worker_device = worker
        .cpu()
        .mmu
        .jolt_device
        .as_ref()
        .ok_or("worker lost its JoltDevice")?;
    if worker_device.outputs != serial_device.outputs || worker_device.panic != serial_device.panic
    {
        return Err("final device outputs/panic diverged".to_string());
    }
    Ok((chunk_count, replayed.len()))
}

fn main() {
    let filter = std::env::args().nth(1);
    let selected = |name: &str| filter.as_deref().is_none_or(|f| name.contains(f));

    let mut failures = 0usize;
    let mut ran = false;
    for (guest, input, chunk_ticks) in golden_cases() {
        if !selected(guest) {
            continue;
        }
        ran = true;
        match run_case(guest, &input, chunk_ticks) {
            Ok((chunks, rows)) => {
                println!("PASS {guest} ({chunks} chunks @ {chunk_ticks} ticks, {rows} rows)")
            }
            Err(report) => {
                failures += 1;
                println!("FAIL {guest}: {report}");
            }
        }
    }
    if !ran {
        eprintln!("No guest matched filter {filter:?}");
        std::process::exit(2);
    }
    if failures > 0 {
        println!("{failures} guest(s) FAILED chunked replay");
        std::process::exit(1);
    }
    println!("all chunked-replay checks passed");
}
