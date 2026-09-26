//! Throughput of `jolt::Fp128` on the GPU against `jolt_field` on the CPU.
//!
//! GPU samples are GPU execution time from command-buffer timestamps, which
//! exclude host submission and wake-up; CPU samples are wall time over all
//! cores with rayon, using `jolt_field`'s `asm` multiply as Akita does.
//! Every kernel's output is checked against the CPU once before it is
//! timed.
//!
//! Groups, all for `Prime128OffsetA7F7`:
//! - `chain/{add,mul,mul4,square}`: dependent operations per thread in
//!   registers, reported as operations per second;
//! - `stream/{add,mul,square}`: elementwise over 2^16 to 2^26 elements;
//! - `inner_product`: sum of `a[i] * b[i]` with a threadgroup reduction on the
//!   GPU, over 2^16 to 2^26 elements.

#[cfg(target_os = "macos")]
#[expect(
    unused_results,
    clippy::expect_used,
    reason = "a benchmark aborts on any failure; criterion builders return `&mut Self`"
)]
mod metal {
    use std::time::Duration;

    use criterion::{BenchmarkId, Criterion, Throughput};
    use jolt_field::solinas::Prime128OffsetA7F7;
    use jolt_field::{Ring, Zero};
    use jolt_metal::runtime::{
        host_name, Batch, Binding, Device, DeviceBuffer, Grid, LibrarySpec, Pipeline, ShaderLibrary,
    };
    use jolt_metal::shaders::FIELD_HEADERS;
    use rayon::prelude::*;

    type F = Prime128OffsetA7F7;

    const FIELD_OPS: &str = include_str!("../tests/shaders/field_ops.metal");
    const FIELD_BENCH: &str = include_str!("shaders/field_bench.metal");

    const ADD: &str = "jolt_test_field_add";
    const MUL: &str = "jolt_test_field_mul";
    const SQUARE: &str = "jolt_test_field_square";
    const ADD_CHAIN: &str = "jolt_bench_field_add_chain";
    const MUL_CHAIN: &str = "jolt_bench_field_mul_chain";
    const MUL_CHAIN4: &str = "jolt_bench_field_mul_chain4";
    const SQUARE_CHAIN: &str = "jolt_bench_field_square_chain";
    const INNER_PRODUCT: &str = "jolt_bench_field_inner_product";
    const KERNELS: [&str; 8] = [
        ADD,
        MUL,
        SQUARE,
        ADD_CHAIN,
        MUL_CHAIN,
        MUL_CHAIN4,
        SQUARE_CHAIN,
        INNER_PRODUCT,
    ];

    /// `INNER_PRODUCT_GROUP` in `field_bench.metal`.
    const INNER_PRODUCT_GROUP: usize = 256;
    /// Threadgroups of the inner-product grid: enough to occupy every core
    /// of the largest Apple GPU several times over.
    const INNER_PRODUCT_GROUPS: usize = 1024;

    /// Threads and dependent operations per thread of the chain kernels.
    const CHAIN_THREADS: usize = 1 << 20;
    const CHAIN_ROUNDS: u32 = 256;

    const LOG_SIZES: [u32; 6] = [16, 18, 20, 22, 24, 26];

    pub fn benches(c: &mut Criterion) {
        let device = Device::system_default().expect("a supported Metal device");
        let library = library(&device);
        chains(c, &device, &library);
        streams(c, &device, &library);
        inner_products(c, &device, &library);
    }

    fn library(device: &Device) -> ShaderLibrary {
        let spec = FIELD_HEADERS
            .iter()
            .fold(LibrarySpec::new(), |spec, (name, text)| {
                spec.source(name, text)
            })
            .source("field_ops.metal", FIELD_OPS)
            .source("field_bench.metal", FIELD_BENCH);
        let spec = KERNELS
            .iter()
            .fold(spec, |spec, kernel| spec.instantiate::<F>(kernel));
        ShaderLibrary::compile(device, &spec).expect("benchmark library compiles")
    }

    fn pipeline<'l>(library: &'l ShaderLibrary, kernel: &str) -> &'l Pipeline {
        library
            .pipeline(&host_name::<F>(kernel))
            .expect("kernel is instantiated")
    }

    /// The elementwise threadgroup size used by the tests.
    fn threadgroup(pipeline: &Pipeline) -> usize {
        (pipeline.thread_execution_width() * 8).min(pipeline.max_total_threads_per_threadgroup())
    }

    /// Fixed-seed field elements.
    fn elements(seed: u64, len: usize) -> Vec<F> {
        let mut state = seed;
        let mut next = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        (0..len)
            .map(|_| F::from_u128((u128::from(next()) << 64) | u128::from(next())))
            .collect()
    }

    /// Runs one dispatch and returns its GPU time.
    fn dispatch(
        device: &Device,
        pipeline: &Pipeline,
        bindings: &[Binding<'_>],
        grid: Grid,
    ) -> Duration {
        let mut batch = Batch::new(device).expect("command batch");
        batch
            .dispatch(pipeline, bindings, grid)
            .expect("valid dispatch");
        batch.commit_and_wait().expect("batch completes")
    }

    /// Sums GPU time over `iters` runs of one dispatch, for `iter_custom`.
    fn gpu_time(iters: u64, run: impl Fn() -> Duration) -> Duration {
        (0..iters).map(|_| run()).sum()
    }

    fn read(buffer: &mut DeviceBuffer<F>) -> Vec<F> {
        buffer.read().expect("canonical output").to_vec()
    }

    fn chains(c: &mut Criterion, device: &Device, library: &ShaderLibrary) {
        let a = elements(1, CHAIN_THREADS);
        let b = elements(2, CHAIN_THREADS);
        let (a_dev, b_dev) = (
            DeviceBuffer::from_slice(device, &a).expect("upload"),
            DeviceBuffer::from_slice(device, &b).expect("upload"),
        );
        let mut out = DeviceBuffer::<F>::zeroed(device, CHAIN_THREADS).expect("allocate");
        let rounds = CHAIN_ROUNDS;

        let mut group = c.benchmark_group("fp128_a7f7/chain");
        group.sample_size(10);
        let operations = CHAIN_THREADS as u64 * u64::from(rounds);

        type Chain = fn(F, F, u32) -> F;
        let cases: [(&str, &str, u64, Chain); 4] = [
            ("add", ADD_CHAIN, 1, |x, y, r| (0..r).fold(x, |x, _| x + y)),
            ("mul", MUL_CHAIN, 1, |x, y, r| (0..r).fold(x, |x, _| x * y)),
            ("mul4", MUL_CHAIN4, 4, |x, y, r| {
                let chain = |x: F| (0..r).fold(x, |x, _| x * y);
                let (x1, x2, x3) = (x + y, x + y + y, x + y + y + y);
                (chain(x) + chain(x1)) + (chain(x2) + chain(x3))
            }),
            ("square", SQUARE_CHAIN, 1, |x, _, r| {
                (0..r).fold(x, |x, _| x.square())
            }),
        ];
        for (name, kernel, chains, cpu) in cases {
            let pipeline = pipeline(library, kernel);
            let grid = Grid::linear(CHAIN_THREADS, threadgroup(pipeline));
            let unary = kernel == SQUARE_CHAIN;
            let run = |out: &DeviceBuffer<F>| {
                let bindings = if unary {
                    vec![
                        Binding::buffer(&a_dev),
                        Binding::value(&rounds),
                        Binding::buffer(out),
                    ]
                } else {
                    vec![
                        Binding::buffer(&a_dev),
                        Binding::buffer(&b_dev),
                        Binding::value(&rounds),
                        Binding::buffer(out),
                    ]
                };
                dispatch(device, pipeline, &bindings, grid)
            };
            let cpu_all = || -> Vec<F> {
                a.par_iter()
                    .zip(&b)
                    .map(|(x, y)| cpu(*x, *y, rounds))
                    .collect()
            };
            run(&out);
            assert_eq!(read(&mut out), cpu_all(), "{kernel} disagrees with the CPU");

            group.throughput(Throughput::Elements(operations * chains));
            group.bench_function(BenchmarkId::new("gpu", name), |bench| {
                bench.iter_custom(|iters| gpu_time(iters, || run(&out)));
            });
            group.bench_function(BenchmarkId::new("cpu", name), |bench| {
                bench.iter(cpu_all);
            });
        }
        group.finish();
    }

    fn streams(c: &mut Criterion, device: &Device, library: &ShaderLibrary) {
        let mut group = c.benchmark_group("fp128_a7f7/stream");
        group.sample_size(10);
        type Op = fn(F, F) -> F;
        let cases: [(&str, &str, Op); 3] = [
            ("add", ADD, |x, y| x + y),
            ("mul", MUL, |x, y| x * y),
            ("square", SQUARE, |x, _| x.square()),
        ];
        for log in LOG_SIZES {
            let len = 1usize << log;
            let a = elements(3, len);
            let b = elements(4, len);
            let (a_dev, b_dev) = (
                DeviceBuffer::from_slice(device, &a).expect("upload"),
                DeviceBuffer::from_slice(device, &b).expect("upload"),
            );
            let mut out = DeviceBuffer::<F>::zeroed(device, len).expect("allocate");
            let mut cpu_out = vec![F::zero(); len];
            group.throughput(Throughput::Elements(len as u64));
            for (name, kernel, op) in cases {
                let pipeline = pipeline(library, kernel);
                let grid = Grid::linear(len, threadgroup(pipeline));
                let unary = kernel == SQUARE;
                let run = |out: &DeviceBuffer<F>| {
                    let bindings = if unary {
                        vec![Binding::buffer(&a_dev), Binding::buffer(out)]
                    } else {
                        vec![
                            Binding::buffer(&a_dev),
                            Binding::buffer(&b_dev),
                            Binding::buffer(out),
                        ]
                    };
                    dispatch(device, pipeline, &bindings, grid)
                };
                let cpu = |cpu_out: &mut [F]| {
                    cpu_out
                        .par_iter_mut()
                        .zip(a.par_iter().zip(&b))
                        .for_each(|(o, (x, y))| *o = op(*x, *y));
                };
                run(&out);
                cpu(&mut cpu_out);
                assert_eq!(read(&mut out), cpu_out, "{kernel} disagrees with the CPU");

                let size = format!("2^{log}");
                group.bench_function(BenchmarkId::new(format!("gpu/{name}"), &size), |bench| {
                    bench.iter_custom(|iters| gpu_time(iters, || run(&out)));
                });
                group.bench_function(BenchmarkId::new(format!("cpu/{name}"), &size), |bench| {
                    bench.iter(|| cpu(&mut cpu_out));
                });
            }
        }
        group.finish();
    }

    fn inner_products(c: &mut Criterion, device: &Device, library: &ShaderLibrary) {
        let pipeline = pipeline(library, INNER_PRODUCT);
        assert!(
            pipeline.max_total_threads_per_threadgroup() >= INNER_PRODUCT_GROUP,
            "the inner-product kernel needs {INNER_PRODUCT_GROUP} threads per threadgroup",
        );
        let grid = Grid::linear(
            INNER_PRODUCT_GROUPS * INNER_PRODUCT_GROUP,
            INNER_PRODUCT_GROUP,
        );
        let mut group = c.benchmark_group("fp128_a7f7/inner_product");
        group.sample_size(10);
        for log in LOG_SIZES {
            let len = 1usize << log;
            let a = elements(5, len);
            let b = elements(6, len);
            let (a_dev, b_dev) = (
                DeviceBuffer::from_slice(device, &a).expect("upload"),
                DeviceBuffer::from_slice(device, &b).expect("upload"),
            );
            let n = u32::try_from(len).expect("benchmark sizes fit u32");
            let mut partials =
                DeviceBuffer::<F>::zeroed(device, INNER_PRODUCT_GROUPS).expect("allocate");
            let run = |partials: &DeviceBuffer<F>| {
                let bindings = [
                    Binding::buffer(&a_dev),
                    Binding::buffer(&b_dev),
                    Binding::value(&n),
                    Binding::buffer(partials),
                ];
                dispatch(device, pipeline, &bindings, grid)
            };
            let cpu = || -> F {
                a.par_iter()
                    .zip(&b)
                    .map(|(x, y)| *x * *y)
                    .reduce(F::zero, |x, y| x + y)
            };
            run(&partials);
            let gpu_sum = read(&mut partials)
                .into_iter()
                .fold(F::zero(), |x, y| x + y);
            assert_eq!(gpu_sum, cpu(), "inner product disagrees with the CPU");

            let size = format!("2^{log}");
            group.throughput(Throughput::Elements(len as u64));
            group.bench_function(BenchmarkId::new("gpu", &size), |bench| {
                bench.iter_custom(|iters| gpu_time(iters, || run(&partials)));
            });
            group.bench_function(BenchmarkId::new("cpu", &size), |bench| {
                bench.iter(cpu);
            });
        }
        group.finish();
    }
}

#[cfg(target_os = "macos")]
criterion::criterion_group!(benches, metal::benches);
#[cfg(target_os = "macos")]
criterion::criterion_main!(benches);

/// Metal exists only on macOS; elsewhere the benchmark has nothing to run.
#[cfg(not(target_os = "macos"))]
fn main() {}
