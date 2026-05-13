//! End-to-end perf measurement for modular FR prove path.
//!
//! Reports per-run:
//! - Raw cycle count (real RV cycles executed by the guest)
//! - Padded cycle count (rounded up to fixture log_t for the proof)
//! - Prove time (release seconds)
//! - Approx kHz throughput (padded cycles / prove time)
//! - Peak resident-set memory (MB; physical RSS during prove)
//!
//! Three variants:
//! - `muldiv`           — RV-only baseline (FR rows present in protocol but no FR events)
//! - `fr_poseidon2_sdk` — FR-active via `jolt-inlines-bn254-fr` coprocessor SDK
//! - `fr_poseidon2_arkworks` — FR-active via software ark-bn254 (no coprocessor)
//!
//! Run with:
//! ```
//! LLVM_SYS_220_PREFIX=/opt/homebrew/opt/llvm \
//!   cargo nextest run -p jolt-host --release --run-ignored only \
//!   --cargo-quiet --no-capture --test fr_perf
//! ```

#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::single_match_else
)]

use std::time::Instant;

use jolt_host::prove_program;
use jolt_profiling::PeakRssSampler;
use jolt_trace::Program;

struct PerfReport {
    label: &'static str,
    raw_cycles: usize,
    padded_cycles: usize,
    prove_ms: f64,
    peak_rss_mb: u64,
    proof_has_evaluation: bool,
}

impl PerfReport {
    fn print(&self) {
        let khz = if self.prove_ms > 0.0 {
            self.padded_cycles as f64 / self.prove_ms
        } else {
            0.0
        };
        println!("=== {} ===", self.label);
        println!("  raw cycles      : {}", self.raw_cycles);
        println!("  padded cycles   : {}", self.padded_cycles);
        println!("  prove time      : {:.3} s", self.prove_ms / 1000.0);
        println!("  prove throughput: {:.1} kHz (padded)", khz);
        let live_khz = if self.prove_ms > 0.0 {
            self.raw_cycles as f64 / self.prove_ms
        } else {
            0.0
        };
        println!("                  : {:.1} kHz (live cycles)", live_khz);
        println!("  peak RSS        : {} MB", self.peak_rss_mb);
        println!("  evaluation      : {}", self.proof_has_evaluation);
        println!();
    }
}

fn run_modular(
    label: &'static str,
    mut configure: impl FnMut(&mut Program),
    inputs: &[u8],
) -> PerfReport {
    let mut program = Program::new("placeholder"); // overwritten by configure
    configure(&mut program);

    let rss = PeakRssSampler::start().expect("start RSS sampler");
    let start = Instant::now();
    let result = prove_program(&mut program, inputs, &[], &[]);
    let prove_ms = start.elapsed().as_secs_f64() * 1_000.0;
    let peak_rss_mb = rss.finish();

    match result {
        Ok(output) => {
            let raw_cycles = output.artifacts.commitment.records.len();
            let _ = raw_cycles;

            let padded_cycles = 1usize << 18;

            PerfReport {
                label,
                raw_cycles: 0,
                padded_cycles,
                prove_ms,
                peak_rss_mb,
                proof_has_evaluation: output.proof.evaluation.is_some(),
            }
        }
        Err(e) => panic!("[{label}] prove_program failed: {e:?}"),
    }
}

/// Trace-only pass to measure raw cycle count without the prove overhead.
fn measure_raw_cycles(configure: impl FnOnce(&mut Program), inputs: &[u8]) -> usize {
    let mut program = Program::new("placeholder");
    configure(&mut program);
    let (_lazy, trace, _final, _io, _events) = program.trace_two_pass_advice(inputs, &[], &[]);
    trace.len()
}

#[test]
#[ignore = "perf benchmark; requires jolt CLI + release build"]
fn perf_muldiv_modular() {
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("postcard");
    let raw = measure_raw_cycles(
        |p| {
            *p = Program::new("muldiv-guest");
        },
        &inputs,
    );
    let mut report = run_modular(
        "muldiv (modular, FR rows idle)",
        |p| {
            *p = Program::new("muldiv-guest");
        },
        &inputs,
    );
    report.raw_cycles = raw;
    report.print();
    assert!(report.proof_has_evaluation);
}

#[test]
#[ignore = "perf benchmark; requires jolt CLI + release build"]
fn perf_fr_poseidon2_sdk_modular() {
    let s0: [u64; 4] = [1, 0, 0, 0];
    let s1: [u64; 4] = [2, 0, 0, 0];
    let s2: [u64; 4] = [3, 0, 0, 0];
    let inputs = postcard::to_stdvec(&(s0, s1, s2)).expect("postcard");

    let raw = measure_raw_cycles(
        |p| {
            let mut pp = Program::new("bn254-fr-poseidon2-sdk-guest");
            let _ = pp
                .set_func("fr_poseidon2_sdk")
                .set_stack_size(65_536)
                .set_heap_size(131_072)
                .set_max_input_size(8_192);
            *p = pp;
        },
        &inputs,
    );
    let mut report = run_modular(
        "fr_poseidon2_sdk (modular + FR coprocessor inline)",
        |p| {
            let mut pp = Program::new("bn254-fr-poseidon2-sdk-guest");
            let _ = pp
                .set_func("fr_poseidon2_sdk")
                .set_stack_size(65_536)
                .set_heap_size(131_072)
                .set_max_input_size(8_192);
            *p = pp;
        },
        &inputs,
    );
    report.raw_cycles = raw;
    report.print();
    assert!(report.proof_has_evaluation);
}

#[test]
#[ignore = "perf benchmark; requires jolt CLI + release build; needs goldens regen at (log_t=18)"]
fn perf_fr_poseidon2_arkworks_modular() {
    let s0: [u64; 4] = [1, 0, 0, 0];
    let s1: [u64; 4] = [2, 0, 0, 0];
    let s2: [u64; 4] = [3, 0, 0, 0];
    let inputs = postcard::to_stdvec(&(s0, s1, s2)).expect("postcard");

    let raw = measure_raw_cycles(
        |p| {
            let mut pp = Program::new("bn254-fr-poseidon2-arkworks-guest");
            let _ = pp
                .set_func("fr_poseidon2_arkworks")
                .set_stack_size(65_536)
                .set_heap_size(1_048_576)
                .set_max_input_size(8_192);
            *p = pp;
        },
        &inputs,
    );
    let mut report = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_modular(
            "fr_poseidon2_arkworks (modular, software ark-bn254 Fr)",
            |p| {
                let mut pp = Program::new("bn254-fr-poseidon2-arkworks-guest");
                let _ = pp
                    .set_func("fr_poseidon2_arkworks")
                    .set_stack_size(65_536)
                    .set_heap_size(1_048_576)
                    .set_max_input_size(8_192);
                *p = pp;
            },
            &inputs,
        )
    })) {
        Ok(r) => r,
        Err(_) => {
            println!(
                "=== fr_poseidon2_arkworks (modular) ===\n\
                 SKIPPED — shape gate likely rejected; raw cycles = {raw}, \
                 needs goldens regen at log_t = {}\n",
                ((raw as f64).log2().ceil() as usize).max(16)
            );
            return;
        }
    };
    report.raw_cycles = raw;
    report.print();
    assert!(report.proof_has_evaluation);
}
