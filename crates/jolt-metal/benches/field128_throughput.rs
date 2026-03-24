//! Benchmarks comparing 4-limb (128-bit) vs 8-limb (256-bit) GPU field
//! arithmetic throughput.
//!
//! Covers raw kernel dispatch (mul, add, fmadd), upload cost, FMA depth
//! sweep (register pressure analysis), and pairwise_reduce (the actual
//! sumcheck hot path).
//!
//! Run all:     cargo bench -p jolt-metal --bench field128_throughput -q
//! Run subset:  cargo bench -p jolt-metal --bench field128_throughput -q -- reduce

#![cfg(target_os = "macos")]
#![allow(unused_results)]

#[path = "../tests/test_field128.rs"]
#[allow(dead_code, unused_imports, clippy::needless_range_loop)]
mod test_field128;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_field::{Field, Fr, MontgomeryConstants};
use jolt_metal::compiler::{CompileMode, GeneratedMsl};
use jolt_metal::field::MetalFieldElement;
use jolt_metal::field_config::MslFieldParams;
use jolt_metal::metal_device_config::MetalDeviceConfig;
use jolt_metal::shaders::{build_source_with_preamble, make_pipeline};
use metal::{ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use num_traits::Zero;
use std::ffi::c_void;
use test_field128::{F128Config, F128};

use jolt_field::{FieldAccumulator, WideAccumulator};

type MF128 = MetalFieldElement<4>;
type MF256 = MetalFieldElement<8>;

struct FieldKernels {
    mul: ComputePipelineState,
    add: ComputePipelineState,
    fmadd: ComputePipelineState,
    fmadd_param: ComputePipelineState,
}

impl FieldKernels {
    fn compile<F: MontgomeryConstants>(device: &Device) -> Self {
        let config = MslFieldParams::new::<F>();
        let source = build_source_with_preamble(&config.msl_preamble, &[&config.msl_test_kernels]);
        let opts = metal::CompileOptions::new();
        let lib = device
            .new_library_with_source(&source, &opts)
            .expect("MSL compilation failed");
        Self {
            mul: make_pipeline(device, &lib, "fr_mul_kernel"),
            add: make_pipeline(device, &lib, "fr_add_kernel"),
            fmadd: make_pipeline(device, &lib, "fr_fmadd_kernel"),
            fmadd_param: make_pipeline(device, &lib, "fr_fmadd_param_kernel"),
        }
    }
}

struct ReduceKernels {
    l2h_unw: ComputePipelineState,
    l2h: ComputePipelineState,
    num_evals: usize,
}

impl ReduceKernels {
    fn compile_product_sum<F: MontgomeryConstants>(device: &Device, d: usize, p: usize) -> Self {
        let field_config = MslFieldParams::new::<F>();
        let device_config = MetalDeviceConfig::detect(device);
        let descriptor = jolt_ir::KernelDescriptor {
            shape: jolt_ir::KernelShape::ProductSum {
                num_inputs_per_product: d,
                num_products: p,
            },
            degree: d,
            tensor_split: None,
        };

        let generated: GeneratedMsl = jolt_metal::compiler::generate_msl(
            &descriptor,
            CompileMode::FastCompile,
            &field_config,
            &device_config,
        );

        let opts = metal::CompileOptions::new();
        let lib = device
            .new_library_with_source(&generated.source, &opts)
            .expect("reduce MSL compilation failed");

        Self {
            l2h_unw: make_pipeline(device, &lib, "reduce_kernel_l2h_unw"),
            l2h: make_pipeline(device, &lib, "reduce_kernel_l2h"),
            num_evals: generated.num_evals,
        }
    }
}

fn upload<T>(device: &Device, data: &[T]) -> metal::Buffer {
    device.new_buffer_with_data(
        data.as_ptr().cast::<c_void>(),
        std::mem::size_of_val(data) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn dispatch_binary(
    queue: &metal::CommandQueue,
    pipeline: &ComputePipelineState,
    a: &metal::Buffer,
    b: &metal::Buffer,
    out: &metal::Buffer,
    n: usize,
) {
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(a), 0);
    enc.set_buffer(1, Some(b), 0);
    enc.set_buffer(2, Some(out), 0);
    let tpg = pipeline.max_total_threads_per_threadgroup().min(n as u64);
    enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(tpg, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

fn dispatch_fmadd(
    queue: &metal::CommandQueue,
    pipeline: &ComputePipelineState,
    a: &metal::Buffer,
    b: &metal::Buffer,
    out: &metal::Buffer,
    params: &metal::Buffer,
    n_threads: usize,
) {
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(a), 0);
    enc.set_buffer(1, Some(b), 0);
    enc.set_buffer(2, Some(out), 0);
    enc.set_buffer(3, Some(params), 0);
    let tpg = pipeline
        .max_total_threads_per_threadgroup()
        .min(n_threads as u64);
    enc.dispatch_threads(
        MTLSize::new(n_threads as u64, 1, 1),
        MTLSize::new(tpg, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

/// Dispatch a weighted reduce kernel (threadgroup-based).
///
/// Buffer layout: [input_0, ..., input_{D-1}, weights, partials, params].
#[allow(clippy::too_many_arguments)]
fn dispatch_reduce_weighted(
    queue: &metal::CommandQueue,
    pipeline: &ComputePipelineState,
    input_bufs: &[&metal::Buffer],
    weights: &metal::Buffer,
    partials: &metal::Buffer,
    params: &metal::Buffer,
    n_pairs: usize,
    device_config: &MetalDeviceConfig,
) {
    let gs = device_config.reduce_group_size;
    let max_groups = device_config.max_reduce_groups;
    let n_groups = n_pairs.div_ceil(gs).min(max_groups);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    for (i, buf) in input_bufs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(buf), 0);
    }
    let next = input_bufs.len() as u64;
    enc.set_buffer(next, Some(weights), 0);
    enc.set_buffer(next + 1, Some(partials), 0);
    enc.set_buffer(next + 2, Some(params), 0);
    enc.dispatch_thread_groups(
        MTLSize::new(n_groups as u64, 1, 1),
        MTLSize::new(gs as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

/// Dispatch a reduce kernel (threadgroup-based).
///
/// Buffer layout: [input_0, ..., input_{D-1}, partials, params].
/// Params: [n_pairs].
fn dispatch_reduce(
    queue: &metal::CommandQueue,
    pipeline: &ComputePipelineState,
    input_bufs: &[&metal::Buffer],
    partials: &metal::Buffer,
    params: &metal::Buffer,
    n_pairs: usize,
    device_config: &MetalDeviceConfig,
) {
    let gs = device_config.reduce_group_size;
    let max_groups = device_config.max_reduce_groups;
    let n_groups = n_pairs.div_ceil(gs).min(max_groups);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    for (i, buf) in input_bufs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(buf), 0);
    }
    let next = input_bufs.len() as u64;
    enc.set_buffer(next, Some(partials), 0);
    enc.set_buffer(next + 1, Some(params), 0);
    enc.dispatch_thread_groups(
        MTLSize::new(n_groups as u64, 1, 1),
        MTLSize::new(gs as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

struct Xorshift64(u64);

impl Xorshift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn random_f128(&mut self) -> F128 {
        let lo = self.next();
        let hi = self.next() & 0x7FFF_FFFF_FFFF_FFFF;
        let limbs = [lo as u32, (lo >> 32) as u32, hi as u32, (hi >> 32) as u32];
        let raw = F128 { limbs };
        let r2 = F128 {
            limbs: [4, 0, 0, 0],
        };
        raw.mul(r2)
    }

    fn random_mf128_vec(&mut self, n: usize) -> Vec<MF128> {
        (0..n)
            .map(|_| {
                let f = self.random_f128();
                MF128 { limbs: f.limbs }
            })
            .collect()
    }

    fn random_mf256_vec(&mut self, n: usize) -> Vec<MF256> {
        (0..n)
            .map(|_| {
                let mut limbs = [0u32; 8];
                for pair in limbs.chunks_exact_mut(2) {
                    let v = self.next();
                    pair[0] = v as u32;
                    pair[1] = (v >> 32) as u32;
                }
                limbs[7] &= 0x1FFF_FFFF;
                MF256 { limbs }
            })
            .collect()
    }

    fn random_fr(&mut self) -> Fr {
        Fr::from_u64(self.next())
    }

    fn random_fr_vec(&mut self, n: usize) -> Vec<Fr> {
        (0..n).map(|_| self.random_fr()).collect()
    }
}

fn cpu_mul_f128(a: &[F128], b: &[F128]) -> Vec<F128> {
    a.iter().zip(b.iter()).map(|(x, y)| x.mul(*y)).collect()
}

fn cpu_fmadd_f128(a: &[F128], b: &[F128], n_fmadd: usize, n_threads: usize) -> Vec<F128> {
    let stride = a.len();
    (0..n_threads)
        .map(|tid| {
            let base = tid * n_fmadd;
            let mut acc = F128::ZERO;
            for i in 0..n_fmadd {
                let idx = (base + i) % stride;
                acc = acc.add(a[idx].mul(b[idx]));
            }
            acc
        })
        .collect()
}

fn cpu_mul_fr(a: &[Fr], b: &[Fr]) -> Vec<Fr> {
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).collect()
}

fn cpu_add_fr(a: &[Fr], b: &[Fr]) -> Vec<Fr> {
    a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect()
}

fn cpu_fmadd_fr(a: &[Fr], b: &[Fr], n_fmadd: usize, n_threads: usize) -> Vec<Fr> {
    let stride = a.len();
    (0..n_threads)
        .map(|tid| {
            let base = tid * n_fmadd;
            let mut acc = WideAccumulator::default();
            for i in 0..n_fmadd {
                let idx = (base + i) % stride;
                acc.fmadd(a[idx], b[idx]);
            }
            acc.reduce()
        })
        .collect()
}

/// CPU pairwise_reduce reference: ProductSum D P=1 weighted.
fn cpu_reduce_fr_weighted(inputs: &[Vec<Fr>], weights: &[Fr], n_pairs: usize, d: usize) -> Vec<Fr> {
    let mut evals = vec![Fr::zero(); d];
    for i in 0..n_pairs {
        let w = weights[i];
        let lo: Vec<Fr> = (0..d).map(|k| inputs[k][2 * i]).collect();
        let hi: Vec<Fr> = (0..d).map(|k| inputs[k][2 * i + 1]).collect();
        let diff: Vec<Fr> = (0..d).map(|k| hi[k] - lo[k]).collect();

        let mut cur = lo.clone();

        evals[0] += w * cur.iter().copied().reduce(|a, b| a * b).unwrap();

        for eval in evals.iter_mut().take(d - 1).skip(1) {
            for (c, d) in cur.iter_mut().zip(diff.iter()) {
                *c += *d;
            }
            *eval += w * cur.iter().copied().reduce(|a, b| a * b).unwrap();
        }

        evals[d - 1] += w * diff.iter().copied().reduce(|a, b| a * b).unwrap();
    }
    evals
}

/// CPU pairwise_reduce reference: ProductSum D P=1 unweighted.
/// Evaluates Σ_i Π_{k=0}^{D-1} p_k(t) at grid points {1, ..., D-1, ∞}
fn cpu_reduce_fr(inputs: &[Vec<Fr>], n_pairs: usize, d: usize) -> Vec<Fr> {
    let mut evals = vec![Fr::zero(); d];
    for i in 0..n_pairs {
        let lo: Vec<Fr> = (0..d).map(|k| inputs[k][2 * i]).collect();
        let hi: Vec<Fr> = (0..d).map(|k| inputs[k][2 * i + 1]).collect();
        let diff: Vec<Fr> = (0..d).map(|k| hi[k] - lo[k]).collect();

        // Incremental grid evaluation: cur starts at lo, steps by diff
        let mut cur = lo.clone();

        // t=1: Π cur[k] (= Π lo[k])
        evals[0] += cur.iter().copied().reduce(|a, b| a * b).unwrap();

        // t=2..D-1
        for eval in evals.iter_mut().take(d - 1).skip(1) {
            for (c, d) in cur.iter_mut().zip(diff.iter()) {
                *c += *d;
            }
            *eval += cur.iter().copied().reduce(|a, b| a * b).unwrap();
        }

        // t=∞: Π diff[k]
        evals[d - 1] += diff.iter().copied().reduce(|a, b| a * b).unwrap();
    }
    evals
}

const SIZES: [usize; 5] = [1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20];

fn fast_config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(2))
}

fn bench_upload(c: &mut Criterion) {
    let device = Device::system_default().expect("no Metal device");
    let mut group = c.benchmark_group("upload");

    for &n in &SIZES {
        let mut rng = Xorshift64(0xaaaa_0001 + n as u64);
        let label = format!("2^{}", n.trailing_zeros());

        let a128 = rng.random_mf128_vec(n);
        group.throughput(Throughput::Bytes((n * std::mem::size_of::<MF128>()) as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("128/{label}"), n),
            &n,
            |bench, _| {
                bench.iter(|| upload(&device, &a128));
            },
        );

        let a256 = rng.random_mf256_vec(n);
        group.throughput(Throughput::Bytes((n * std::mem::size_of::<MF256>()) as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("256/{label}"), n),
            &n,
            |bench, _| {
                bench.iter(|| upload(&device, &a256));
            },
        );
    }
    group.finish();
}

fn bench_mul(c: &mut Criterion) {
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();
    let k128 = FieldKernels::compile::<F128Config>(&device);
    let k256 = FieldKernels::compile::<Fr>(&device);

    let mut group = c.benchmark_group("field_mul");

    for &n in &SIZES {
        let mut rng = Xorshift64(0xdead_0001 + n as u64);
        let label = format!("2^{}", n.trailing_zeros());
        group.throughput(Throughput::Elements(n as u64));

        // GPU 128
        let a128 = rng.random_mf128_vec(n);
        let b128 = rng.random_mf128_vec(n);
        let buf_a = upload(&device, &a128);
        let buf_b = upload(&device, &b128);
        let buf_out = device.new_buffer(
            (n * std::mem::size_of::<MF128>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        group.bench_with_input(
            BenchmarkId::new(format!("gpu_128/{label}"), n),
            &n,
            |bench, _| {
                bench.iter(|| dispatch_binary(&queue, &k128.mul, &buf_a, &buf_b, &buf_out, n));
            },
        );

        // CPU 128 (naive CIOS — included for completeness, not a fair comparison)
        let a128_cpu: Vec<F128> = a128.iter().map(|m| F128 { limbs: m.limbs }).collect();
        let b128_cpu: Vec<F128> = b128.iter().map(|m| F128 { limbs: m.limbs }).collect();
        group.bench_with_input(
            BenchmarkId::new(format!("cpu_128_naive/{label}"), n),
            &n,
            |bench, _| {
                bench.iter(|| cpu_mul_f128(&a128_cpu, &b128_cpu));
            },
        );

        // GPU 256
        let a256 = rng.random_mf256_vec(n);
        let b256 = rng.random_mf256_vec(n);
        let buf_a = upload(&device, &a256);
        let buf_b = upload(&device, &b256);
        let buf_out = device.new_buffer(
            (n * std::mem::size_of::<MF256>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        group.bench_with_input(
            BenchmarkId::new(format!("gpu_256/{label}"), n),
            &n,
            |bench, _| {
                bench.iter(|| dispatch_binary(&queue, &k256.mul, &buf_a, &buf_b, &buf_out, n));
            },
        );

        // CPU 256 (arkworks optimized)
        let a256_cpu = rng.random_fr_vec(n);
        let b256_cpu = rng.random_fr_vec(n);
        group.bench_with_input(
            BenchmarkId::new(format!("cpu_256/{label}"), n),
            &n,
            |bench, _| {
                bench.iter(|| cpu_mul_fr(&a256_cpu, &b256_cpu));
            },
        );
    }
    group.finish();
}

fn bench_add(c: &mut Criterion) {
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();
    let k128 = FieldKernels::compile::<F128Config>(&device);
    let k256 = FieldKernels::compile::<Fr>(&device);

    let mut group = c.benchmark_group("field_add");

    for &n in &SIZES {
        let mut rng = Xorshift64(0xbeef_0001 + n as u64);
        let label = format!("2^{}", n.trailing_zeros());
        group.throughput(Throughput::Elements(n as u64));

        // GPU 128
        let a128 = rng.random_mf128_vec(n);
        let b128 = rng.random_mf128_vec(n);
        let buf_a = upload(&device, &a128);
        let buf_b = upload(&device, &b128);
        let buf_out = device.new_buffer(
            (n * std::mem::size_of::<MF128>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        group.bench_with_input(
            BenchmarkId::new(format!("gpu_128/{label}"), n),
            &n,
            |bench, _| {
                bench.iter(|| dispatch_binary(&queue, &k128.add, &buf_a, &buf_b, &buf_out, n));
            },
        );

        // GPU 256
        let a256 = rng.random_mf256_vec(n);
        let b256 = rng.random_mf256_vec(n);
        let buf_a = upload(&device, &a256);
        let buf_b = upload(&device, &b256);
        let buf_out = device.new_buffer(
            (n * std::mem::size_of::<MF256>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        group.bench_with_input(
            BenchmarkId::new(format!("gpu_256/{label}"), n),
            &n,
            |bench, _| {
                bench.iter(|| dispatch_binary(&queue, &k256.add, &buf_a, &buf_b, &buf_out, n));
            },
        );

        // CPU 256 (arkworks)
        let a256_cpu = rng.random_fr_vec(n);
        let b256_cpu = rng.random_fr_vec(n);
        group.bench_with_input(
            BenchmarkId::new(format!("cpu_256/{label}"), n),
            &n,
            |bench, _| {
                bench.iter(|| cpu_add_fr(&a256_cpu, &b256_cpu));
            },
        );
    }
    group.finish();
}

fn bench_fmadd(c: &mut Criterion) {
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();
    let k128 = FieldKernels::compile::<F128Config>(&device);
    let k256 = FieldKernels::compile::<Fr>(&device);

    let mut group = c.benchmark_group("field_fmadd");

    let stride = 1024;
    let n_fmadd: usize = 256;
    let n_threads = 4096;
    group.throughput(Throughput::Elements((n_threads * n_fmadd) as u64));

    let mut rng = Xorshift64(0xcafe_0001);

    // GPU 128
    let a128 = rng.random_mf128_vec(stride);
    let b128 = rng.random_mf128_vec(stride);
    let buf_a128 = upload(&device, &a128);
    let buf_b128 = upload(&device, &b128);
    let buf_out128 = device.new_buffer(
        (n_threads * std::mem::size_of::<MF128>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buf_params = upload(&device, &[stride as u32]);

    group.bench_function("gpu_128", |bench| {
        bench.iter(|| {
            dispatch_fmadd(
                &queue,
                &k128.fmadd,
                &buf_a128,
                &buf_b128,
                &buf_out128,
                &buf_params,
                n_threads,
            );
        });
    });

    // CPU 128 (naive)
    let a128_cpu: Vec<F128> = a128.iter().map(|m| F128 { limbs: m.limbs }).collect();
    let b128_cpu: Vec<F128> = b128.iter().map(|m| F128 { limbs: m.limbs }).collect();
    group.bench_function("cpu_128_naive", |bench| {
        bench.iter(|| cpu_fmadd_f128(&a128_cpu, &b128_cpu, n_fmadd, n_threads));
    });

    // GPU 256
    let a256 = rng.random_mf256_vec(stride);
    let b256 = rng.random_mf256_vec(stride);
    let buf_a256 = upload(&device, &a256);
    let buf_b256 = upload(&device, &b256);
    let buf_out256 = device.new_buffer(
        (n_threads * std::mem::size_of::<MF256>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    group.bench_function("gpu_256", |bench| {
        bench.iter(|| {
            dispatch_fmadd(
                &queue,
                &k256.fmadd,
                &buf_a256,
                &buf_b256,
                &buf_out256,
                &buf_params,
                n_threads,
            );
        });
    });

    // CPU 256 (WideAccumulator — optimized deferred reduction)
    let a256_cpu = rng.random_fr_vec(stride);
    let b256_cpu = rng.random_fr_vec(stride);
    group.bench_function("cpu_256", |bench| {
        bench.iter(|| cpu_fmadd_fr(&a256_cpu, &b256_cpu, n_fmadd, n_threads));
    });

    group.finish();
}

fn bench_fmadd_depth(c: &mut Criterion) {
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();
    let k128 = FieldKernels::compile::<F128Config>(&device);
    let k256 = FieldKernels::compile::<Fr>(&device);

    let mut group = c.benchmark_group("fmadd_depth");

    let stride = 4096;
    let n_threads = 4096;
    let mut rng = Xorshift64(0xface_0001);

    let a128 = rng.random_mf128_vec(stride);
    let b128 = rng.random_mf128_vec(stride);
    let buf_a128 = upload(&device, &a128);
    let buf_b128 = upload(&device, &b128);
    let buf_out128 = device.new_buffer(
        (n_threads * std::mem::size_of::<MF128>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let a256 = rng.random_mf256_vec(stride);
    let b256 = rng.random_mf256_vec(stride);
    let buf_a256 = upload(&device, &a256);
    let buf_b256 = upload(&device, &b256);
    let buf_out256 = device.new_buffer(
        (n_threads * std::mem::size_of::<MF256>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    for &depth in &[32usize, 128, 256, 512, 1024] {
        let buf_params = upload(&device, &[stride as u32, depth as u32]);
        group.throughput(Throughput::Elements((n_threads * depth) as u64));

        group.bench_with_input(BenchmarkId::new("gpu_128", depth), &depth, |bench, _| {
            bench.iter(|| {
                dispatch_fmadd(
                    &queue,
                    &k128.fmadd_param,
                    &buf_a128,
                    &buf_b128,
                    &buf_out128,
                    &buf_params,
                    n_threads,
                );
            });
        });

        group.bench_with_input(BenchmarkId::new("gpu_256", depth), &depth, |bench, _| {
            bench.iter(|| {
                dispatch_fmadd(
                    &queue,
                    &k256.fmadd_param,
                    &buf_a256,
                    &buf_b256,
                    &buf_out256,
                    &buf_params,
                    n_threads,
                );
            });
        });
    }
    group.finish();
}

fn bench_reduce(c: &mut Criterion, group_name: &str, d: usize, p: usize) {
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();
    let device_config = MetalDeviceConfig::detect(&device);
    let rk128 = ReduceKernels::compile_product_sum::<F128Config>(&device, d, p);
    let rk256 = ReduceKernels::compile_product_sum::<Fr>(&device, d, p);
    let num_inputs = d * p;

    let mut group = c.benchmark_group(group_name);

    for &n in &SIZES {
        let n_pairs = n / 2;
        let mut rng = Xorshift64(0xd400_0001 + n as u64 + d as u64);
        let label = format!("2^{}", n.trailing_zeros());
        group.throughput(Throughput::Elements(n_pairs as u64));

        // GPU 128
        let inputs_128: Vec<Vec<MF128>> =
            (0..num_inputs).map(|_| rng.random_mf128_vec(n)).collect();
        let bufs_128: Vec<_> = inputs_128.iter().map(|v| upload(&device, v)).collect();
        let max_groups = device_config.max_reduce_groups;
        let partials_128 = device.new_buffer(
            (max_groups * rk128.num_evals * std::mem::size_of::<MF128>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params = upload(&device, &[n_pairs as u32]);
        let buf_refs_128: Vec<&metal::Buffer> = bufs_128.iter().collect();
        group.bench_with_input(
            BenchmarkId::new(format!("gpu_128/{label}"), n),
            &n,
            |bench, _| {
                bench.iter(|| {
                    dispatch_reduce(
                        &queue,
                        &rk128.l2h_unw,
                        &buf_refs_128,
                        &partials_128,
                        &params,
                        n_pairs,
                        &device_config,
                    );
                });
            },
        );

        // GPU 256
        let inputs_256: Vec<Vec<MF256>> =
            (0..num_inputs).map(|_| rng.random_mf256_vec(n)).collect();
        let bufs_256: Vec<_> = inputs_256.iter().map(|v| upload(&device, v)).collect();
        let partials_256 = device.new_buffer(
            (max_groups * rk256.num_evals * std::mem::size_of::<MF256>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_refs_256: Vec<&metal::Buffer> = bufs_256.iter().collect();
        group.bench_with_input(
            BenchmarkId::new(format!("gpu_256/{label}"), n),
            &n,
            |bench, _| {
                bench.iter(|| {
                    dispatch_reduce(
                        &queue,
                        &rk256.l2h_unw,
                        &buf_refs_256,
                        &partials_256,
                        &params,
                        n_pairs,
                        &device_config,
                    );
                });
            },
        );

        // CPU 256 (arkworks, single-threaded)
        let cpu_inputs_256: Vec<Vec<Fr>> = (0..num_inputs).map(|_| rng.random_fr_vec(n)).collect();
        group.bench_with_input(
            BenchmarkId::new(format!("cpu_256/{label}"), n),
            &n,
            |bench, _| {
                bench.iter(|| cpu_reduce_fr(&cpu_inputs_256, n_pairs, d));
            },
        );
    }
    group.finish();
}

fn bench_reduce_d4(c: &mut Criterion) {
    bench_reduce(c, "reduce_d4", 4, 1);
}

fn bench_reduce_d8(c: &mut Criterion) {
    bench_reduce(c, "reduce_d8", 8, 1);
}

fn bench_reduce_weighted(c: &mut Criterion, group_name: &str, d: usize, p: usize) {
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();
    let device_config = MetalDeviceConfig::detect(&device);
    let rk256 = ReduceKernels::compile_product_sum::<Fr>(&device, d, p);
    let num_inputs = d * p;

    let mut group = c.benchmark_group(group_name);

    for &n in &SIZES {
        let n_pairs = n / 2;
        let mut rng = Xorshift64(0xe400_0001 + n as u64 + d as u64);
        let label = format!("2^{}", n.trailing_zeros());
        group.throughput(Throughput::Elements(n_pairs as u64));

        // GPU 256 weighted
        let inputs_256: Vec<Vec<MF256>> =
            (0..num_inputs).map(|_| rng.random_mf256_vec(n)).collect();
        let weights_256 = rng.random_mf256_vec(n_pairs);
        let bufs_256: Vec<_> = inputs_256.iter().map(|v| upload(&device, v)).collect();
        let buf_weights = upload(&device, &weights_256);
        let max_groups = device_config.max_reduce_groups;
        let partials_256 = device.new_buffer(
            (max_groups * rk256.num_evals * std::mem::size_of::<MF256>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params = upload(&device, &[n_pairs as u32]);
        let buf_refs_256: Vec<&metal::Buffer> = bufs_256.iter().collect();
        group.bench_with_input(
            BenchmarkId::new(format!("gpu_256_w/{label}"), n),
            &n,
            |bench, _| {
                bench.iter(|| {
                    dispatch_reduce_weighted(
                        &queue,
                        &rk256.l2h,
                        &buf_refs_256,
                        &buf_weights,
                        &partials_256,
                        &params,
                        n_pairs,
                        &device_config,
                    );
                });
            },
        );

        // CPU 256 weighted (single-threaded)
        let cpu_inputs_256: Vec<Vec<Fr>> = (0..num_inputs).map(|_| rng.random_fr_vec(n)).collect();
        let cpu_weights_256 = rng.random_fr_vec(n_pairs);
        group.bench_with_input(
            BenchmarkId::new(format!("cpu_256_w/{label}"), n),
            &n,
            |bench, _| {
                bench
                    .iter(|| cpu_reduce_fr_weighted(&cpu_inputs_256, &cpu_weights_256, n_pairs, d));
            },
        );
    }
    group.finish();
}

fn bench_reduce_weighted_d4(c: &mut Criterion) {
    bench_reduce_weighted(c, "reduce_w_d4", 4, 1);
}

fn bench_reduce_weighted_d8(c: &mut Criterion) {
    bench_reduce_weighted(c, "reduce_w_d8", 8, 1);
}

fn bench_reduce_d8_groupsize(c: &mut Criterion) {
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();
    let d: usize = 8;
    let p: usize = 1;
    let num_inputs = d * p;

    let mut group = c.benchmark_group("reduce_d8_gs");

    for &gs in &[64usize, 128, 256] {
        let device_config = MetalDeviceConfig {
            reduce_group_size: gs,
            ..MetalDeviceConfig::detect(&device)
        };
        let field_config = MslFieldParams::new::<Fr>();
        let descriptor = jolt_ir::KernelDescriptor {
            shape: jolt_ir::KernelShape::ProductSum {
                num_inputs_per_product: d,
                num_products: p,
            },
            degree: d,
            tensor_split: None,
        };
        let generated: GeneratedMsl = jolt_metal::compiler::generate_msl(
            &descriptor,
            CompileMode::FastCompile,
            &field_config,
            &device_config,
        );
        let opts = metal::CompileOptions::new();
        let lib = device
            .new_library_with_source(&generated.source, &opts)
            .expect("gs sweep MSL compilation failed");
        let pipeline = make_pipeline(&device, &lib, "reduce_kernel_l2h_unw");

        for &n in &[1usize << 18, 1 << 20] {
            let n_pairs = n / 2;
            let mut rng = Xorshift64(0xf800_0001 + n as u64 + gs as u64);
            let label = format!("gs{gs}/2^{}", n.trailing_zeros());
            group.throughput(Throughput::Elements(n_pairs as u64));

            let inputs: Vec<Vec<MF256>> =
                (0..num_inputs).map(|_| rng.random_mf256_vec(n)).collect();
            let bufs: Vec<_> = inputs.iter().map(|v| upload(&device, v)).collect();
            let max_groups = device_config.max_reduce_groups;
            let partials = device.new_buffer(
                (max_groups * generated.num_evals * std::mem::size_of::<MF256>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let params = upload(&device, &[n_pairs as u32]);
            let buf_refs: Vec<&metal::Buffer> = bufs.iter().collect();
            group.bench_with_input(BenchmarkId::new(&label, n), &n, |bench, _| {
                bench.iter(|| {
                    dispatch_reduce(
                        &queue,
                        &pipeline,
                        &buf_refs,
                        &partials,
                        &params,
                        n_pairs,
                        &device_config,
                    );
                });
            });
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = fast_config();
    targets =
        bench_upload,
        bench_mul,
        bench_add,
        bench_fmadd,
        bench_fmadd_depth,
        bench_reduce_d4,
        bench_reduce_d8,
        bench_reduce_weighted_d4,
        bench_reduce_weighted_d8,
        bench_reduce_d8_groupsize,
}
criterion_main!(benches);
