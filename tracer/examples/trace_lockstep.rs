//! Trace/execute mode-equivalence gate for tracer changes.
//!
//! Runs each golden guest twice in lockstep — one emulator ticking in trace
//! mode, one in execute mode — and asserts bit-identical architectural CPU
//! state (registers incl. virtual, CSRs, pc, reservation, advice tape, device
//! outputs) at every tick boundary, plus identical final memory. This is the
//! foundation of two-pass parallel tracing: pass-1 runs execute mode and its
//! checkpoints seed trace-mode chunk replays.
//!
//! Usage:
//!   cargo run --release -p tracer --example trace_lockstep -- [filter]

// Link inline crates so their inventory registrations reach the tracer.
extern crate jolt_inlines_keccak256 as _;
extern crate jolt_inlines_sha2 as _;

#[path = "support/mod.rs"]
mod support;

use support::chain_input;
use tracer::instruction::Cycle;

/// Same (guest, input) pairs as the golden-trace gate.
fn golden_cases() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("sha2-chain-guest", chain_input(300)),
        ("sha3-chain-guest", chain_input(235)),
        ("fibonacci-guest", postcard::to_stdvec(&84_000u32).unwrap()),
        ("btreemap-guest", postcard::to_stdvec(&650u32).unwrap()),
        (
            "muldiv-guest",
            postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap(),
        ),
    ]
}

/// Lockstep-runs the guest in both modes; returns the tick count on success,
/// or the first divergence report.
fn run_case(guest: &str, input: &[u8]) -> Result<usize, String> {
    let (elf, _, memory_config) = support::build_guest(guest);

    let mut em_trace = tracer::create_emulator(&elf, None, input, &[], &[], &memory_config, None);
    let mut em_exec = tracer::create_emulator(&elf, None, input, &[], &[], &memory_config, None);

    let mut rows: Vec<Cycle> = Vec::new();
    let mut prev_pc: u64 = 0;
    let mut tick_idx: usize = 0;
    loop {
        let pc = em_trace.get_cpu().read_pc();
        let pc_exec = em_exec.get_cpu().read_pc();
        if pc != pc_exec {
            return Err(format!("tick {tick_idx}: pc {pc:#x} vs {pc_exec:#x}"));
        }
        if pc == prev_pc {
            break;
        }
        rows.clear();
        em_trace.tick(Some(&mut rows));
        em_exec.tick(None);
        if rows.is_empty() {
            return Err(format!("tick {tick_idx}: trace mode emitted zero rows"));
        }
        if let Some(diff) = em_trace.get_cpu().arch_state_diff(em_exec.get_cpu()) {
            return Err(format!("tick {tick_idx}: {diff}"));
        }
        prev_pc = pc;
        tick_idx += 1;
    }
    if tick_idx == 0 {
        return Err("program did not execute".to_string());
    }

    let mem_trace = em_trace
        .get_cpu()
        .mmu
        .memory
        .memory
        .materialized_nonzero_bytes();
    let mem_exec = em_exec
        .get_cpu()
        .mmu
        .memory
        .memory
        .materialized_nonzero_bytes();
    if mem_trace != mem_exec {
        return Err(format!(
            "final memory diverged ({} vs {} nonzero bytes)",
            mem_trace.len(),
            mem_exec.len()
        ));
    }
    Ok(tick_idx)
}

fn main() {
    let filter = std::env::args().nth(1);
    let selected = |name: &str| filter.as_deref().is_none_or(|f| name.contains(f));

    let mut failures = 0usize;
    let mut ran = false;
    for (guest, input) in golden_cases() {
        if !selected(guest) {
            continue;
        }
        ran = true;
        match run_case(guest, &input) {
            Ok(ticks) => println!("PASS {guest} ({ticks} ticks in lockstep)"),
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
        println!("{failures} guest(s) FAILED lockstep equivalence");
        std::process::exit(1);
    }
    println!("all lockstep equivalence checks passed");
}
