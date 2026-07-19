//! CPU vs CUDA end-to-end prove timing, with per-stage wall-time breakdown.
//!
//! Uses the prover's existing `bolt.stageN` tracing spans (no prover edits) to attribute the
//! host remainder to prove stages — the breakdown that decides which port to pursue.
//!
//!   cargo build -p jolt-equivalence --features cuda --example cuda_single_prove --release
//!   target/release/examples/cuda_single_prove <log_t>
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![expect(clippy::print_stdout, reason = "profiling harness prints its measured ms")]

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use jolt_equivalence::core_oracle::core_sha2_chain_commitment_fixture;
use jolt_equivalence::cuda_backend_oracle::{all_cpu_programs, all_cuda_programs, run_bolt_prover};
use jolt_inlines_sha2 as _;
use tracing::span::{Attributes, Id};
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

static STAGE_NS: OnceLock<Mutex<HashMap<String, u128>>> = OnceLock::new();

fn stage_ns() -> &'static Mutex<HashMap<String, u128>> {
    STAGE_NS.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Default)]
struct Timing {
    name: String,
    entered: Option<Instant>,
}

struct StageTimer;

impl<S> Layer<S> for StageTimer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let mut ext = span.extensions_mut();
            ext.insert(Timing {
                name: attrs.metadata().name().to_string(),
                entered: None,
            });
        }
    }

    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            if let Some(t) = span.extensions_mut().get_mut::<Timing>() {
                t.entered = Some(Instant::now());
            }
        }
    }

    fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            if let Some(t) = span.extensions_mut().get_mut::<Timing>() {
                if let Some(start) = t.entered.take() {
                    let ns = start.elapsed().as_nanos();
                    let keep = t.name.starts_with("bolt.")
                        || t.name.contains("::new")
                        || t.name.contains("_state")
                        || t.name.contains("::round_poly")
                        || t.name.contains("::bind")
                        || t.name.contains("::instance.")
                        || t.name.contains("::prove_batched");
                    if keep {
                        *stage_ns().lock().unwrap().entry(t.name.clone()).or_insert(0) += ns;
                    }
                }
            }
        }
    }
}

fn main() {
    let log_t: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let fixture = core_sha2_chain_commitment_fixture(log_t);

    if std::env::var_os("JOLT_KERNEL_TIMING_LOG").is_some() {
        tracing_subscriber::registry()
            .with(StageTimer)
            .with(tracing_subscriber::fmt::layer().with_target(false))
            .init();
    } else {
        tracing_subscriber::registry().with(StageTimer).init();
    }

    let _ = run_bolt_prover(&fixture, all_cpu_programs(&fixture));
    let _ = run_bolt_prover(&fixture, all_cuda_programs(&fixture));

    let dump = |label: &str| {
        let mut m = stage_ns().lock().unwrap();
        let mut rows: Vec<(String, u128)> = m.drain().collect();
        rows.sort_by_key(|(name, _)| name.clone());
        println!("  {label} per-stage (ms):");
        let mut kernels_1_7 = 0u128;
        for (name, ns) in &rows {
            if name.matches('.').count() == 1 {
                println!("    {name:<24} {:.0}", *ns as f64 / 1e6);
                let is_1_7 = ["1", "2", "3", "4", "5", "6", "7"]
                    .iter()
                    .any(|n| name == &format!("bolt.stage{n}"));
                if is_1_7 {
                    kernels_1_7 += ns;
                }
            }
        }
        println!("    {:<24} {:.0}", "[stages 1-7 kernels]", kernels_1_7 as f64 / 1e6);
        println!("  {label} per-relation (ms):");
        for (name, ns) in &rows {
            if !name.starts_with("bolt.") {
                println!("    {name:<44} {:.0}", *ns as f64 / 1e6);
            }
        }
    };

    stage_ns().lock().unwrap().clear();
    let (_cpu_state, cpu_ms) = run_bolt_prover(&fixture, all_cpu_programs(&fixture));
    dump("cpu");

    stage_ns().lock().unwrap().clear();
    jolt_kernels::cuda::xfer_stats::reset();
    let before = jolt_kernels::cuda::xfer_stats::snapshot();
    let (_cuda_state, cuda_ms) = run_bolt_prover(&fixture, all_cuda_programs(&fixture));
    let after = jolt_kernels::cuda::xfer_stats::snapshot();
    dump("cuda");

    let d = |i: usize| after[i].saturating_sub(before[i]);
    let h2d_raw_mb = d(16) as f64 / (1024.0 * 1024.0);
    let h2d_raw_ms = d(18) as f64 / 1e6;
    let h2d_all_mb = d(2) as f64 / (1024.0 * 1024.0);
    let upload_ms = d(11) as f64 / 1e6;
    println!(
        "  cuda xfer: h2d_raw={h2d_raw_mb:.1} MB in {h2d_raw_ms:.1} ms ({} calls); \
         h2d_total={h2d_all_mb:.1} MB; upload_span={upload_ms:.1} ms",
        d(17)
    );
    println!(
        "  h2d_raw share of cuda prove: {:.1}%",
        100.0 * h2d_raw_ms / cuda_ms
    );

    println!(
        "log_t={log_t}: cpu={cpu_ms:.1} ms  cuda={cuda_ms:.1} ms  speedup={:.3}x",
        cpu_ms / cuda_ms
    );
}
