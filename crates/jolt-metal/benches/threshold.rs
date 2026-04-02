//! Binary-search benchmark to find the GPU->CPU crossover threshold.
//!
//! For each operation (`reduce`, `interpolate_inplace`), measures Metal vs CPU
//! latency across buffer sizes from 2^6 to 2^22 and reports the crossover
//! point where Metal first becomes faster.
//!
//! Run with:
//! ```sh
//! cargo bench -p jolt-metal --bench threshold -- --output-format bencher
//! ```

#![cfg(target_os = "macos")]
#![allow(unused_results, clippy::print_stdout)]

use std::time::Instant;

use jolt_compiler::kernel_spec::Iteration;
use jolt_compiler::{BindingOrder, Factor, Formula, KernelSpec, ProductTerm};
use jolt_compute::{Buf, ComputeBackend, DeviceBuffer};
use jolt_cpu::CpuBackend;
use jolt_field::{Field, Fr};
use jolt_metal::MetalBackend;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn random_fr(rng: &mut StdRng, n: usize) -> Vec<Fr> {
    (0..n).map(|_| Fr::random(rng)).collect()
}

fn product_sum_formula(d: usize, p: usize) -> Formula {
    let terms: Vec<_> = (0..p)
        .map(|g| ProductTerm {
            coefficient: 1,
            factors: (0..d).map(|j| Factor::Input((g * d + j) as u32)).collect(),
        })
        .collect();
    Formula::from_terms(terms)
}

fn make_spec(formula: &Formula) -> KernelSpec {
    KernelSpec::new(formula.clone(), Iteration::Dense, BindingOrder::LowToHigh)
}

const WARMUP: usize = 5;
const ITERS: usize = 20;

/// Measure median latency of `reduce` for a given backend and buffer size.
fn bench_reduce_latency<B: ComputeBackend>(
    backend: &B,
    kernel: &B::CompiledKernel<Fr>,
    inputs_raw: &[Vec<Fr>],
) -> f64 {
    let bufs: Vec<B::Buffer<Fr>> = inputs_raw.iter().map(|v| backend.upload(v)).collect();
    let dev_bufs: Vec<Buf<B, Fr>> = bufs.into_iter().map(DeviceBuffer::Field).collect();
    let refs: Vec<&Buf<B, Fr>> = dev_bufs.iter().collect();

    for _ in 0..WARMUP {
        let _ = backend.reduce(kernel, &refs, &[]);
    }

    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        let _ = backend.reduce(kernel, &refs, &[]);
        times.push(start.elapsed().as_nanos() as f64);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[ITERS / 2]
}

/// Measure median latency of `interpolate_inplace` for a given backend.
fn bench_bind_latency<B: ComputeBackend>(backend: &B, data: &[Fr], scalar: Fr) -> f64 {
    for _ in 0..WARMUP {
        let mut buf = backend.upload(data);
        backend.interpolate_inplace(&mut buf, scalar, BindingOrder::LowToHigh);
    }

    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let mut buf = backend.upload(data);
        let start = Instant::now();
        backend.interpolate_inplace(&mut buf, scalar, BindingOrder::LowToHigh);
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
        let spec = make_spec(&formula);
        let mtl_k = metal.compile::<Fr>(&spec);
        let cpu_k = cpu.compile::<Fr>(&spec);

        println!("\n=== reduce D={d} ===");
        println!(
            "{:<12} {:>14} {:>14} {:>10}",
            "n", "metal_ns", "cpu_ns", "winner"
        );
        println!("{}", "-".repeat(54));

        let mut crossover: Option<usize> = None;

        for &log_n in &log_sizes {
            let n = 1usize << log_n;
            let inputs: Vec<Vec<Fr>> = (0..d).map(|_| random_fr(&mut rng, n)).collect();

            let mtl_ns = bench_reduce_latency(&metal, &mtl_k, &inputs);
            let cpu_ns = bench_reduce_latency(&cpu, &cpu_k, &inputs);

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
        println!("\n=== interpolate_inplace ===");
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
        let spec = make_spec(&formula);
        let mtl_k = metal.compile::<Fr>(&spec);
        let cpu_k = cpu.compile::<Fr>(&spec);

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
                let mut bufs: Vec<Buf<MetalBackend, Fr>> = inputs
                    .iter()
                    .map(|v| DeviceBuffer::Field(metal.upload(v)))
                    .collect();
                let refs: Vec<_> = bufs.iter().collect();
                let _ = metal.reduce(&mtl_k, &refs, &[]);
                for buf in &mut bufs {
                    metal.interpolate_inplace(buf.as_field_mut(), scalar, BindingOrder::LowToHigh);
                }
                drop(bufs);
            }

            // Metal
            let mut mtl_times = Vec::with_capacity(ITERS);
            for _ in 0..ITERS {
                let mut bufs: Vec<Buf<MetalBackend, Fr>> = inputs
                    .iter()
                    .map(|v| DeviceBuffer::Field(metal.upload(v)))
                    .collect();
                let start = Instant::now();
                let refs: Vec<_> = bufs.iter().collect();
                let _ = metal.reduce(&mtl_k, &refs, &[]);
                for buf in &mut bufs {
                    metal.interpolate_inplace(buf.as_field_mut(), scalar, BindingOrder::LowToHigh);
                }
                drop(bufs);
                mtl_times.push(start.elapsed().as_nanos() as f64);
            }
            mtl_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mtl_ns = mtl_times[ITERS / 2];

            // CPU
            let mut cpu_times = Vec::with_capacity(ITERS);
            for _ in 0..ITERS {
                let mut bufs: Vec<Buf<CpuBackend, Fr>> = inputs
                    .iter()
                    .map(|v| DeviceBuffer::Field(cpu.upload(v)))
                    .collect();
                let start = Instant::now();
                let refs: Vec<_> = bufs.iter().collect();
                let _ = cpu.reduce(&cpu_k, &refs, &[]);
                for buf in &mut bufs {
                    cpu.interpolate_inplace(buf.as_field_mut(), scalar, BindingOrder::LowToHigh);
                }
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
