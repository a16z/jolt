//! End-to-end sumcheck benchmark: CPU-only vs Hybrid (Metal->CPU switchover).
//!
//! Runs `SumcheckProver::prove()` through the full protocol loop with
//! `KernelEvaluator` on random data, starting at 2^24 elements (24 rounds).
//! Each scenario compares CPU-only against Hybrid with a fixed 2^12 switchover:
//! Metal handles rounds while buffers are large, CPU takes over at 4K elements.
//!
//! Toom-Cook benchmarks use HighToLow binding so the fused interpolate+reduce
//! kernel fires on every H2L round — one GPU dispatch per round instead of two.
//!
//! Run all:     cargo bench -p jolt-metal --bench sumcheck_e2e -q
//! Run subset:  cargo bench -p jolt-metal --bench sumcheck_e2e -q -- toom_D4

#![cfg(target_os = "macos")]
#![allow(unused_results)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_compiler::Formula;
use jolt_compute::{BindingOrder, ComputeBackend, HybridBackend};
use jolt_cpu::CpuBackend;
use jolt_field::{Field, Fr};
use jolt_metal::MetalBackend;
use jolt_poly::EqPolynomial;
use jolt_sumcheck::{SumcheckClaim, SumcheckProver};
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_zkvm::evaluators::{catalog, kernel::KernelEvaluator};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// When `JOLT_TRACE=1`, writes a Chrome trace to `target/trace-sumcheck-e2e.json`.
/// View with `chrome://tracing` or Perfetto UI.
fn init_tracing() -> Option<tracing_chrome::FlushGuard> {
    if std::env::var("JOLT_TRACE").as_deref() != Ok("1") {
        return None;
    }
    let trace_path = std::env::var("JOLT_TRACE_FILE")
        .unwrap_or_else(|_| "/tmp/trace-sumcheck-e2e.json".to_string());
    let (layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
        .file(trace_path)
        .include_args(true)
        .build();
    use tracing_subscriber::layer::SubscriberExt;
    let subscriber = tracing_subscriber::registry().with(layer);
    tracing::subscriber::set_global_default(subscriber).expect("failed to set tracing subscriber");
    Some(guard)
}

/// Fixed switchover: buffers migrate from Metal to CPU at 4K elements.
/// Below this size, GPU dispatch overhead dominates the actual compute.
const HYBRID_THRESHOLD: usize = 1 << 12;

/// All benchmarks start at 2^20 (1M elements, 20 sumcheck rounds).
const NUM_VARS: usize = 20;

fn random_fr(rng: &mut StdRng, n: usize) -> Vec<Fr> {
    (0..n).map(|_| Fr::random(rng)).collect()
}

struct ToomCookData {
    polys: Vec<Vec<Fr>>,
    eq_w: Vec<Fr>,
    formula: Formula,
    degree: usize,
}

fn prepare_toom_cook(d: usize, num_products: usize) -> ToomCookData {
    let n = 1usize << NUM_VARS;
    let total_inputs = d * num_products;
    let mut rng = StdRng::seed_from_u64(42 + d as u64);

    ToomCookData {
        polys: (0..total_inputs).map(|_| random_fr(&mut rng, n)).collect(),
        eq_w: (0..NUM_VARS).map(|_| Fr::random(&mut rng)).collect(),
        formula: catalog::product_sum(d, num_products),
        degree: d + 1,
    }
}

fn build_toom_cook_witness<B: ComputeBackend>(
    data: &ToomCookData,
    backend: &Arc<B>,
) -> (KernelEvaluator<Fr, B>, SumcheckClaim<Fr>) {
    let kernel = backend.compile_kernel::<Fr>(&data.formula);
    let inputs: Vec<_> = data.polys.iter().map(|p| backend.upload(p)).collect();

    let mut eq_w_h2l = data.eq_w.clone();
    eq_w_h2l.reverse();

    let claimed_sum = Fr::from_u64(42);
    let witness = KernelEvaluator::with_toom_cook_eq(
        inputs,
        kernel,
        data.formula.degree(),
        eq_w_h2l,
        claimed_sum,
        Arc::clone(backend),
        BindingOrder::HighToLow,
    );
    let claim = SumcheckClaim {
        num_vars: NUM_VARS,
        degree: data.degree,
        claimed_sum,
    };
    (witness, claim)
}

struct StandardGridData {
    eq_table: Vec<Fr>,
    polys: Vec<Vec<Fr>>,
    formula: Formula,
}

fn prepare_standard_grid(formula: Formula, num_polys: usize) -> StandardGridData {
    let n = 1usize << NUM_VARS;
    let mut rng = StdRng::seed_from_u64(99);

    let r: Vec<Fr> = (0..NUM_VARS).map(|_| Fr::random(&mut rng)).collect();
    StandardGridData {
        eq_table: EqPolynomial::new(r).evaluations(),
        polys: (0..num_polys).map(|_| random_fr(&mut rng, n)).collect(),
        formula,
    }
}

fn build_standard_grid_witness<B: ComputeBackend>(
    data: &StandardGridData,
    backend: &Arc<B>,
) -> (KernelEvaluator<Fr, B>, SumcheckClaim<Fr>) {
    let kernel = backend.compile_kernel::<Fr>(&data.formula);

    let mut inputs: Vec<_> = vec![backend.upload(&data.eq_table)];
    inputs.extend(data.polys.iter().map(|p| backend.upload(p)));

    let claimed_sum = Fr::from_u64(42);
    let witness = KernelEvaluator::with_unit_weights(
        inputs,
        kernel,
        data.formula.degree(),
        Arc::clone(backend),
    );
    let claim = SumcheckClaim {
        num_vars: NUM_VARS,
        degree: data.formula.degree(),
        claimed_sum,
    };
    (witness, claim)
}

fn time_prove<B: ComputeBackend>(
    witness: &mut KernelEvaluator<Fr, B>,
    claim: &SumcheckClaim<Fr>,
) -> Duration {
    let mut transcript = Blake2bTranscript::new(b"bench");
    let start = Instant::now();
    let _ = SumcheckProver::prove(claim, witness, &mut transcript);
    start.elapsed()
}

fn bench_toom_cook(c: &mut Criterion, name: &str, d: usize, num_products: usize) {
    let mut group = c.benchmark_group(format!("e2e/{name}"));
    let data = prepare_toom_cook(d, num_products);

    let cpu: Arc<CpuBackend> = Arc::new(CpuBackend);
    group.bench_function(BenchmarkId::new("cpu", ""), |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let (mut w, cl) = build_toom_cook_witness(&data, &cpu);
                total += time_prove(&mut w, &cl);
            }
            total
        });
    });

    let hybrid: Arc<HybridBackend<MetalBackend, CpuBackend>> = Arc::new(HybridBackend::new(
        MetalBackend::new(),
        CpuBackend,
        HYBRID_THRESHOLD,
    ));
    group.bench_function(BenchmarkId::new("hybrid", ""), |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let (mut w, cl) = build_toom_cook_witness(&data, &hybrid);
                total += time_prove(&mut w, &cl);
            }
            total
        });
    });

    group.finish();
}

fn bench_standard_grid(c: &mut Criterion, name: &str, formula: Formula, num_polys: usize) {
    let mut group = c.benchmark_group(format!("e2e/{name}"));
    let data = prepare_standard_grid(formula, num_polys);

    let cpu: Arc<CpuBackend> = Arc::new(CpuBackend);
    group.bench_function(BenchmarkId::new("cpu", ""), |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let (mut w, cl) = build_standard_grid_witness(&data, &cpu);
                total += time_prove(&mut w, &cl);
            }
            total
        });
    });

    let hybrid: Arc<HybridBackend<MetalBackend, CpuBackend>> = Arc::new(HybridBackend::new(
        MetalBackend::new(),
        CpuBackend,
        HYBRID_THRESHOLD,
    ));
    group.bench_function(BenchmarkId::new("hybrid", ""), |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let (mut w, cl) = build_standard_grid_witness(&data, &hybrid);
                total += time_prove(&mut w, &cl);
            }
            total
        });
    });

    group.finish();
}

fn bench_toom_d4_p1(c: &mut Criterion) {
    bench_toom_cook(c, "toom_D4_P1", 4, 1);
}

fn bench_toom_d4_p3(c: &mut Criterion) {
    bench_toom_cook(c, "toom_D4_P3", 4, 3);
}

fn bench_toom_d8_p1(c: &mut Criterion) {
    bench_toom_cook(c, "toom_D8_P1", 8, 1);
}

fn bench_eq_product(c: &mut Criterion) {
    bench_standard_grid(c, "eq_product", catalog::eq_product(), 1);
}

fn bench_hamming(c: &mut Criterion) {
    bench_standard_grid(c, "hamming", catalog::hamming_booleanity(), 1);
}

/// Single-iteration tracing run: `JOLT_TRACE=1 cargo bench -p jolt-metal --bench sumcheck_e2e -q -- --profile-time=5`
/// produces `target/trace-sumcheck-e2e.json` viewable in Perfetto.
static TRACE_GUARD: std::sync::Mutex<Option<tracing_chrome::FlushGuard>> =
    std::sync::Mutex::new(None);

fn ensure_tracing() {
    let mut guard = TRACE_GUARD.lock().unwrap();
    if guard.is_none() {
        *guard = init_tracing();
    }
}

fn bench_toom_d4_p1_traced(c: &mut Criterion) {
    ensure_tracing();
    bench_toom_d4_p1(c);
}
fn bench_toom_d4_p3_traced(c: &mut Criterion) {
    ensure_tracing();
    bench_toom_d4_p3(c);
}
fn bench_toom_d8_p1_traced(c: &mut Criterion) {
    ensure_tracing();
    bench_toom_d8_p1(c);
}
fn bench_eq_product_traced(c: &mut Criterion) {
    ensure_tracing();
    bench_eq_product(c);
}
fn bench_hamming_traced(c: &mut Criterion) {
    ensure_tracing();
    bench_hamming(c);
}

criterion_group! {
    name = sumcheck_e2e;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(10));
    targets =
        bench_toom_d4_p1_traced,
        bench_toom_d4_p3_traced,
        bench_toom_d8_p1_traced,
        bench_eq_product_traced,
        bench_hamming_traced,
}
criterion_main!(sumcheck_e2e);
