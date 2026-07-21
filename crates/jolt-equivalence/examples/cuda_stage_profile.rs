//! Single CUDA prove with per-stage bottleneck attribution: accumulates the wall-time
//! of every `bolt.stageN` span and every `State::method` tracing span (init / round_poly
//! / bind, across all stages) that fires inside the monolithic `bolt.prove`, plus the
//! jolt-kernels xfer_stats buckets (materialize / upload / d2h / bind ns).
//!
//! Set JOLT_STAGE{N}_TRACE_INSTANCES=1 to enable the per-instance init/round/bind spans
//! for stages that gate them behind that env var (stages 4-7); stage2/3 spans are always on.
//!
//! xfer_stats does NOT record on-device kernel time (ns_kernel is unused), so run
//! this under nsys to attribute the GPU-compute remainder:
//!
//!   cargo build -p jolt-equivalence --features cuda --example cuda_stage_profile --release
//!   JOLT_CUDA_XFER_STATS=1 JOLT_STAGE6_TRACE_INSTANCES=1 \
//!     target/release/examples/cuda_stage_profile <log_t>
//!   nsys profile -o prof --trace=cuda target/release/examples/cuda_stage_profile <log_t>
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![expect(clippy::print_stdout, reason = "profiling harness prints its measured ms")]

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use jolt_equivalence::core_oracle::core_sha2_chain_commitment_fixture;
use jolt_equivalence::cuda_backend_oracle::{all_cuda_programs, run_bolt_prover};
use jolt_inlines_sha2 as _;
use tracing::span::{Attributes, Id};
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

static SPAN_NS: OnceLock<Mutex<HashMap<String, u128>>> = OnceLock::new();

fn span_ns() -> &'static Mutex<HashMap<String, u128>> {
    SPAN_NS.get_or_init(|| Mutex::new(HashMap::new()))
}

// Only the monolithic prover wraps stage work in a `bolt.prove` span. The oracle
// also runs staged proves (CPU backend, no `bolt.prove` parent) for artifact
// derivation, so gate accumulation on being inside `bolt.prove` to attribute
// spans to the measured cuda prove only.
static IN_PROVE: AtomicUsize = AtomicUsize::new(0);

#[derive(Default)]
struct Timing {
    name: String,
    entered: Option<Instant>,
}

struct SpanTimer;

impl<S> Layer<S> for SpanTimer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(Timing {
                name: attrs.metadata().name().to_string(),
                entered: None,
            });
        }
    }

    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            if let Some(t) = span.extensions_mut().get_mut::<Timing>() {
                t.entered = Some(Instant::now());
                if t.name == "bolt.prove" {
                    let _ = IN_PROVE.fetch_add(1, Ordering::SeqCst);
                }
            }
        }
    }

    fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            if let Some(t) = span.extensions_mut().get_mut::<Timing>() {
                if let Some(start) = t.entered.take() {
                    let ns = start.elapsed().as_nanos();
                    if t.name == "bolt.prove" {
                        let _ = IN_PROVE.fetch_sub(1, Ordering::SeqCst);
                    }
                    let keep = (t.name.starts_with("bolt.stage")
                        || t.name.contains("::"))
                        && IN_PROVE.load(Ordering::SeqCst) > 0;
                    if keep {
                        *span_ns().lock().unwrap().entry(t.name.clone()).or_insert(0) += ns;
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
        .unwrap_or(22);
    let fixture = core_sha2_chain_commitment_fixture(log_t);

    tracing_subscriber::registry().with(SpanTimer).init();

    // Warmup prove (NVRTC compile + GPU spin-up + resident-cache fill).
    let _ = run_bolt_prover(&fixture, all_cuda_programs(&fixture));

    span_ns().lock().unwrap().clear();
    jolt_kernels::cuda::xfer_stats::reset();

    let start = Instant::now();
    let _ = run_bolt_prover(&fixture, all_cuda_programs(&fixture));
    let total_ms = start.elapsed().as_secs_f64() * 1e3;

    let mut rows: Vec<(String, u128)> = span_ns().lock().unwrap().drain().collect();
    rows.sort_by_key(|(name, _)| name.clone());
    println!("log_t={log_t} cuda prove total={total_ms:.0} ms");
    println!("  stage5 spans (ms):");
    for (name, ns) in rows {
        println!("    {name:<52} {:.1}", ns as f64 / 1e6);
    }

    let s = jolt_kernels::cuda::xfer_stats::snapshot();
    let ms = |ns: u64| ns as f64 / 1e6;
    let mb = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
    println!("  xfer_stats (whole prove, host-attributed):");
    println!(
        "    materialize={:.1}ms upload={:.1}ms d2h={:.1}ms bind={:.1}ms ({} bind calls)",
        ms(s[8]),
        ms(s[9]),
        ms(s[11]),
        ms(s[12]),
        s[13],
    );
    println!(
        "    h2d={:.1}MB ({} calls)  d2h={:.1}MB ({} calls)",
        mb(s[0]),
        s[1],
        mb(s[2]),
        s[3],
    );
}
