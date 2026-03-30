//! Binary-search benchmark to find the GPU→CPU crossover threshold.
//!
//! For each operation (`pairwise_reduce`, `interpolate_pairs_inplace`), measures
//! Metal vs CPU latency across buffer sizes from 2^6 to 2^22 and reports the
//! crossover point where Metal first becomes faster.
//!
//! Run with:
//! ```sh
//! cargo bench -p jolt-metal --bench threshold -- --output-format bencher
//! ```

#![cfg(target_os = "macos")]
#![allow(unused_results, clippy::print_stdout)]

use std::time::Instant;

use jolt_compute::{BindingOrder, ComputeBackend, EqInput};
use jolt_cpu::CpuBackend;
use jolt_field::{Field, Fr};
use jolt_compiler::{CompositionFormula, Factor, ProductTerm};
use jolt_metal::MetalBackend;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn random_fr(rng: &mut StdRng, n: usize) -> Vec<Fr> {
    (0..n).map(|_| Fr::random(rng)).collect()
}

fn product_sum_formula(d: usize, p: usize) -> CompositionFormula {
    let terms: Vec<_> = (0..p)
        .map(|g| ProductTerm {
            coefficient: 1,
            factors: (0..d).map(|j| Factor::Input((g * d + j) as u32)).collect(),
        })
        .collect();
    CompositionFormula::from_terms(terms)
}

/// Warmup iterations before timing.
const WARMUP: usize = 5;
/// Timed iterations for each measurement.
const ITERS: usize = 20;

/// Measure median latency of `pairwise_reduce` for a given backend and buffer size.
fn bench_reduce_latency<B: ComputeBackend>(
    backend: &B,
    kernel: &B::CompiledKernel<Fr>,
    inputs_raw: &[Vec<Fr>],
    weights_raw: &[Fr],
    num_evals: usize,
) -> f64 {
    let bufs: Vec<B::Buffer<Fr>> = inputs_raw.iter().map(|v| backend.upload(v)).collect();
    let w_buf = backend.upload(weights_raw);
    let refs: Vec<&B::Buffer<Fr>> = bufs.iter().collect();

    // Warmup
    for _ in 0..WARMUP {
        let _ = backend.pairwise_reduce(
            &refs,
            EqInput::Weighted(&w_buf),
            kernel,
            num_evals,
            BindingOrder::LowToHigh,
        );
    }

    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        let _ = backend.pairwise_reduce(
            &refs,
            EqInput::Weighted(&w_buf),
            kernel,
            num_evals,
            BindingOrder::LowToHigh,
        );
        times.push(start.elapsed().as_nanos() as f64);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[ITERS / 2]
}

/// Measure median latency of `interpolate_pairs_inplace` for a given backend.
fn bench_bind_latency<B: ComputeBackend>(backend: &B, data: &[Fr], scalar: Fr) -> f64 {
    // Warmup
    for _ in 0..WARMUP {
        let mut buf = backend.upload(data);
        backend.interpolate_pairs_inplace(&mut buf, scalar, BindingOrder::LowToHigh);
    }

    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let mut buf = backend.upload(data);
        let start = Instant::now();
        backend.interpolate_pairs_inplace(&mut buf, scalar, BindingOrder::LowToHigh);
        times.push(start.elapsed().as_nanos() as f64);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[ITERS / 2]
}

fn main() {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;

    let log_sizes: Vec<u32> = (6..=22).collect();
    let mut rng = StdRng::seed_from_u64(42);

    for &d in &[4usize, 8] {
        let formula = product_sum_formula(d, 1);
        let mtl_k = metal.compile_kernel::<Fr>(&formula);
        let cpu_k = cpu.compile_kernel::<Fr>(&formula);

        println!("\n=== pairwise_reduce D={d} ===");
        println!(
            "{:<12} {:>14} {:>14} {:>10}",
            "n", "metal_ns", "cpu_ns", "winner"
        );
        println!("{}", "-".repeat(54));

        let mut crossover: Option<usize> = None;

        for &log_n in &log_sizes {
            let n = 1usize << log_n;
            let inputs: Vec<Vec<Fr>> = (0..d).map(|_| random_fr(&mut rng, n)).collect();
            let weights = random_fr(&mut rng, n / 2);

            let mtl_ns = bench_reduce_latency(&metal, &mtl_k, &inputs, &weights, formula.degree());
            let cpu_ns = bench_reduce_latency(&cpu, &cpu_k, &inputs, &weights, formula.degree());

            let winner = if mtl_ns < cpu_ns { "METAL" } else { "CPU" };
            println!("2^{log_n:<8} {mtl_ns:>14.0} {cpu_ns:>14.0} {winner:>10}");

            if crossover.is_none() && mtl_ns < cpu_ns {
                crossover = Some(n);
            }
        }

        match crossover {
            Some(n) => println!(">>> Crossover at n={n} (2^{})", n.trailing_zeros()),
            None => println!(">>> Metal never faster in tested range"),
        }
    }

    {
        println!("\n=== interpolate_pairs_inplace ===");
        println!(
            "{:<12} {:>14} {:>14} {:>10}",
            "n", "metal_ns", "cpu_ns", "winner"
        );
        println!("{}", "-".repeat(54));

        let scalar = Fr::random(&mut rng);
        let mut crossover: Option<usize> = None;

        for &log_n in &log_sizes {
            let n = 1usize << log_n;
            let data = random_fr(&mut rng, n);

            let mtl_ns = bench_bind_latency(&metal, &data, scalar);
            let cpu_ns = bench_bind_latency(&cpu, &data, scalar);

            let winner = if mtl_ns < cpu_ns { "METAL" } else { "CPU" };
            println!("2^{log_n:<8} {mtl_ns:>14.0} {cpu_ns:>14.0} {winner:>10}");

            if crossover.is_none() && mtl_ns < cpu_ns {
                crossover = Some(n);
            }
        }

        match crossover {
            Some(n) => println!(">>> Crossover at n={n} (2^{})", n.trailing_zeros()),
            None => println!(">>> Metal never faster in tested range"),
        }
    }

    {
        let d = 4usize;
        let formula = product_sum_formula(d, 1);
        let mtl_k = metal.compile_kernel::<Fr>(&formula);
        let cpu_k = cpu.compile_kernel::<Fr>(&formula);

        println!("\n=== sumcheck_round (reduce + bind) D={d} ===");
        println!(
            "{:<12} {:>14} {:>14} {:>10}",
            "n", "metal_ns", "cpu_ns", "winner"
        );
        println!("{}", "-".repeat(54));

        let scalar = Fr::random(&mut rng);
        let mut crossover: Option<usize> = None;

        for &log_n in &log_sizes {
            let n = 1usize << log_n;
            let inputs: Vec<Vec<Fr>> = (0..d).map(|_| random_fr(&mut rng, n)).collect();

            // Warmup
            for _ in 0..WARMUP {
                let mut bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
                let refs: Vec<_> = bufs.iter().collect();
                let _ = metal.pairwise_reduce(
                    &refs,
                    EqInput::Unit,
                    &mtl_k,
                    formula.degree(),
                    BindingOrder::LowToHigh,
                );
                bufs = metal.interpolate_pairs_batch(bufs, scalar);
                drop(bufs);
            }

            // Metal
            let mut mtl_times = Vec::with_capacity(ITERS);
            for _ in 0..ITERS {
                let mut bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
                let start = Instant::now();
                let refs: Vec<_> = bufs.iter().collect();
                let _ = metal.pairwise_reduce(
                    &refs,
                    EqInput::Unit,
                    &mtl_k,
                    formula.degree(),
                    BindingOrder::LowToHigh,
                );
                bufs = metal.interpolate_pairs_batch(bufs, scalar);
                drop(bufs);
                mtl_times.push(start.elapsed().as_nanos() as f64);
            }
            mtl_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mtl_ns = mtl_times[ITERS / 2];

            // CPU
            let mut cpu_times = Vec::with_capacity(ITERS);
            for _ in 0..ITERS {
                let mut bufs: Vec<_> = inputs.iter().map(|v| cpu.upload(v)).collect();
                let start = Instant::now();
                let refs: Vec<_> = bufs.iter().collect();
                let _ = cpu.pairwise_reduce(
                    &refs,
                    EqInput::Unit,
                    &cpu_k,
                    formula.degree(),
                    BindingOrder::LowToHigh,
                );
                bufs = cpu.interpolate_pairs_batch(bufs, scalar);
                drop(bufs);
                cpu_times.push(start.elapsed().as_nanos() as f64);
            }
            cpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let cpu_ns = cpu_times[ITERS / 2];

            let winner = if mtl_ns < cpu_ns { "METAL" } else { "CPU" };
            println!("2^{log_n:<8} {mtl_ns:>14.0} {cpu_ns:>14.0} {winner:>10}");

            if crossover.is_none() && mtl_ns < cpu_ns {
                crossover = Some(n);
            }
        }

        match crossover {
            Some(n) => println!(">>> Crossover at n={n} (2^{})", n.trailing_zeros()),
            None => println!(">>> Metal never faster in tested range"),
        }
    }
}
