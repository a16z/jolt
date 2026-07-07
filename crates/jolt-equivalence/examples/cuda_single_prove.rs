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
                    if t.name.starts_with("bolt.") {
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

    tracing_subscriber::registry().with(StageTimer).init();

    let _ = run_bolt_prover(&fixture, all_cpu_programs(&fixture));
    let _ = run_bolt_prover(&fixture, all_cuda_programs(&fixture));

    let dump = |label: &str| {
        let mut m = stage_ns().lock().unwrap();
        let mut rows: Vec<(String, u128)> = m.drain().collect();
        rows.sort_by_key(|(name, _)| name.clone());
        println!("  {label} per-stage (ms):");
        for (name, ns) in rows {
            if name.matches('.').count() == 1 {
                println!("    {name:<20} {:.0}", ns as f64 / 1e6);
            }
        }
    };

    stage_ns().lock().unwrap().clear();
    let (_cpu_state, cpu_ms) = run_bolt_prover(&fixture, all_cpu_programs(&fixture));
    dump("cpu");

    stage_ns().lock().unwrap().clear();
    let (_cuda_state, cuda_ms) = run_bolt_prover(&fixture, all_cuda_programs(&fixture));
    dump("cuda");

    println!(
        "log_t={log_t}: cpu={cpu_ms:.1} ms  cuda={cuda_ms:.1} ms  speedup={:.3}x",
        cpu_ms / cuda_ms
    );
}
