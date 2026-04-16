//! `jolt-bench` — side-by-side prove+verify benchmarks for jolt-core and
//! the modular stack. See `crates/jolt-bench/README` (none; brief only) or
//! the T1 task in `worker-base.md` for the measurement contract.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::process::ExitCode;

use clap::Parser;

mod baseline;
mod cli;
mod measure;
mod output;
mod programs;
mod stacks;

// Force-link the inline crates so their `#[ctor::ctor]` startup functions run
// and register SHA2/Keccak inline instruction sequence builders in the tracer.
// Without these, tracing `sha2-ex` / `sha3-ex` panics in
// `tracer::instruction::inline` ("No inline sequence builder registered").
use jolt_inlines_keccak256 as _;
use jolt_inlines_sha2 as _;

use cli::{Cli, StackSelection};
use measure::{median, median_u64};
use output::{BenchReport, Run, StackLabel};
use stacks::core::CoreStack;
use stacks::modular::ModularStack;
use stacks::{IterMetrics, StackOutcome, StackRunner};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn aggregate(label: StackLabel, metrics: &[IterMetrics]) -> Run {
    let prove: Vec<f64> = metrics.iter().map(|m| m.prove_ms).collect();
    let verify: Vec<f64> = metrics.iter().map(|m| m.verify_ms).collect();
    let rss: Vec<u64> = metrics.iter().map(|m| m.peak_rss_mb).collect();
    let bytes: Vec<u64> = metrics.iter().map(|m| m.proof_bytes).collect();

    let (encoding, verify_note) = match label {
        StackLabel::Core => (Some("ark-compressed".to_string()), None),
        StackLabel::Modular => (
            Some("bincode-serde".to_string()),
            Some(
                "verify measured via proof transplant into jolt-core verifier \
                 (no native modular+Dory verifier wired today)"
                    .to_string(),
            ),
        ),
    };

    Run {
        stack: label.as_str().to_string(),
        prove_ms: Some(median(&prove)),
        verify_ms: Some(median(&verify)),
        peak_rss_mb: Some(median_u64(&rss)),
        proof_bytes: Some(median_u64(&bytes)),
        proof_encoding: encoding,
        unsupported: false,
        reason: None,
        verify_note,
    }
}

fn run_stack<R: StackRunner>(runner: R, label: StackLabel, cli: &Cli) -> Run {
    match runner.run(cli.program, cli.iters, cli.warmup, cli.log_t) {
        StackOutcome::Metrics(iters) => aggregate(label, &iters),
        StackOutcome::Unsupported(run) => run,
    }
}

fn main() -> ExitCode {
    let mut cli = Cli::parse();

    // When emitting a Perfetto trace, single-shot the run so the timeline is
    // readable (no repeated iter+warmup layers on top of each other).
    let _tracing_guards = cli.trace_chrome.as_deref().map(|name| {
        cli.iters = 1;
        cli.warmup = 0;
        jolt_profiling::setup_tracing(&[jolt_profiling::TracingFormat::Chrome], name)
    });

    let mut runs = Vec::new();
    if matches!(cli.stack, StackSelection::Core | StackSelection::Both) {
        runs.push(run_stack(CoreStack, StackLabel::Core, &cli));
    }
    if matches!(cli.stack, StackSelection::Modular | StackSelection::Both) {
        runs.push(run_stack(ModularStack, StackLabel::Modular, &cli));
    }

    let report = BenchReport {
        program: cli.program.cli_name().to_string(),
        iters: cli.iters,
        warmup: cli.warmup,
        runs,
    };

    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    if let Some(path) = &cli.json {
        std::fs::write(path, &json).expect("write json output");
    } else {
        println!("{json}");
    }

    if let Some(baseline_path) = &cli.baseline {
        let baseline = baseline::load_baseline(baseline_path);
        let regressions = baseline::check_regressions(&baseline, &report, cli.threshold);
        if !regressions.is_empty() {
            eprintln!(
                "\njolt-bench: {} regression(s) vs baseline (threshold {:.2}x):",
                regressions.len(),
                cli.threshold
            );
            for r in &regressions {
                eprintln!(
                    "  {}: baseline={:.3} observed={:.3} ratio={:.3}x",
                    r.metric, r.baseline, r.observed, r.ratio
                );
            }
            return ExitCode::from(2);
        }
    }

    ExitCode::SUCCESS
}
