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
//!   GPU, over 2^16 to 2^26 elements;
//! - `accum/{fmadd,fmadd4,fmadd_i64}`: deferred-reduction terms per thread in
//!   registers, reported as terms per second, against `jolt_field`'s
//!   accumulators;
//! - `accum_inner_product`: `inner_product` with unreduced products and
//!   merged accumulators on both sides.

#[cfg(target_os = "macos")]
mod support;

#[cfg(target_os = "macos")]
#[expect(
    unused_results,
    clippy::expect_used,
    reason = "a benchmark aborts on any failure; criterion builders return `&mut Self`"
)]
mod metal {
    use std::time::Duration;

    use criterion::{BenchmarkId, Criterion, Throughput};
    use jolt_field::{Accumulator, Ring, WithAccumulator, Zero};
    use jolt_metal::runtime::{Binding, Device, DeviceBuffer, Grid, ShaderLibrary};
    use rayon::prelude::*;

    use super::support::{dispatch, elements, library, pipeline, threadgroup, words, F};

    const FIELD_OPS: &str = include_str!("../tests/shaders/field_ops.metal");
    const FIELD_BENCH: &str = include_str!("shaders/field_bench.metal");
    const ACCUM_BENCH: &str = include_str!("shaders/accum_bench.metal");

    const ADD: &str = "jolt_test_field_add";
    const MUL: &str = "jolt_test_field_mul";
    const SQUARE: &str = "jolt_test_field_square";
    const ADD_CHAIN: &str = "jolt_bench_field_add_chain";
    const MUL_CHAIN: &str = "jolt_bench_field_mul_chain";
    const MUL_CHAIN4: &str = "jolt_bench_field_mul_chain4";
    const SQUARE_CHAIN: &str = "jolt_bench_field_square_chain";
    const INNER_PRODUCT: &str = "jolt_bench_field_inner_product";
    const ACCUM_FMADD: &str = "jolt_bench_accum_fmadd";
    const ACCUM_FMADD4: &str = "jolt_bench_accum_fmadd4";
    const ACCUM_FMADD_I64: &str = "jolt_bench_accum_fmadd_i64";
    const ACCUM_INNER_PRODUCT: &str = "jolt_bench_accum_inner_product";
    const KERNELS: [&str; 12] = [
        ADD,
        MUL,
        SQUARE,
        ADD_CHAIN,
        MUL_CHAIN,
        MUL_CHAIN4,
        SQUARE_CHAIN,
        INNER_PRODUCT,
        ACCUM_FMADD,
        ACCUM_FMADD4,
        ACCUM_FMADD_I64,
        ACCUM_INNER_PRODUCT,
    ];

    type Acc = <F as WithAccumulator>::Accumulator;
    type SmallScalarAcc = <F as WithAccumulator>::SmallScalarAccumulator;

    /// `INNER_PRODUCT_GROUP` in `field_bench.metal`.
    const INNER_PRODUCT_GROUP: usize = 256;
    /// Threadgroups of the inner-product grid: enough to occupy every core
    /// of the largest Apple GPU several times over.
    const INNER_PRODUCT_GROUPS: usize = 1024;

    /// Threads and dependent operations per thread of the chain kernels.
    const CHAIN_THREADS: usize = 1 << 20;
    const CHAIN_ROUNDS: u32 = 256;
    /// `FMADD_ROUNDS` in `accum_bench.metal`: each accumulator kernel does
    /// four terms per round.
    const FMADD_ROUNDS: u32 = 64;
    /// The additive step of the `fmadd_i64` scalars.
    const SCALAR_STEP: u64 = 0x9E37_79B9_7F4A_7C15;

    const LOG_SIZES: [u32; 6] = [16, 18, 20, 22, 24, 26];

    pub fn benches(c: &mut Criterion) {
        let device = Device::system_default().expect("a supported Metal device");
        let library = library(
            &device,
            &[
                ("field_ops.metal", FIELD_OPS),
                ("field_bench.metal", FIELD_BENCH),
                ("accum_bench.metal", ACCUM_BENCH),
            ],
            &KERNELS,
            &[],
        );
        chains(c, &device, &library);
        streams(c, &device, &library);
        inner_products(
            c,
            &device,
            &library,
            "inner_product",
            INNER_PRODUCT,
            |a, b| {
                a.par_iter()
                    .zip(b)
                    .map(|(x, y)| *x * *y)
                    .reduce(F::zero, |x, y| x + y)
            },
        );
        accumulators(c, &device, &library);
        inner_products(
            c,
            &device,
            &library,
            "accum_inner_product",
            ACCUM_INNER_PRODUCT,
            |a, b| {
                a.par_iter()
                    .zip(b)
                    .fold(Acc::default, |mut acc, (x, y)| {
                        acc.fmadd(*x, *y);
                        acc
                    })
                    .reduce(Acc::default, |mut acc, other| {
                        acc.merge(other);
                        acc
                    })
                    .reduce()
            },
        );
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
                dispatch(device, pipeline, &bindings, grid, 1)
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
                    dispatch(device, pipeline, &bindings, grid, 1)
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

    /// Inner products with `kernel`, whose threadgroups of
    /// `INNER_PRODUCT_GROUP` threads each write one partial sum, against the
    /// CPU inner product `cpu`.
    fn inner_products(
        c: &mut Criterion,
        device: &Device,
        library: &ShaderLibrary,
        name: &str,
        kernel: &str,
        cpu: fn(&[F], &[F]) -> F,
    ) {
        let pipeline = pipeline(library, kernel);
        assert!(
            pipeline.max_total_threads_per_threadgroup() >= INNER_PRODUCT_GROUP,
            "{kernel} needs {INNER_PRODUCT_GROUP} threads per threadgroup",
        );
        let grid = Grid::linear(
            INNER_PRODUCT_GROUPS * INNER_PRODUCT_GROUP,
            INNER_PRODUCT_GROUP,
        );
        let mut group = c.benchmark_group(format!("fp128_a7f7/{name}"));
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
                dispatch(device, pipeline, &bindings, grid, 1)
            };
            run(&partials);
            let gpu_sum = read(&mut partials)
                .into_iter()
                .fold(F::zero(), |x, y| x + y);
            assert_eq!(gpu_sum, cpu(&a, &b), "{kernel} disagrees with the CPU");

            let size = format!("2^{log}");
            group.throughput(Throughput::Elements(len as u64));
            group.bench_function(BenchmarkId::new("gpu", &size), |bench| {
                bench.iter_custom(|iters| gpu_time(iters, || run(&partials)));
            });
            group.bench_function(BenchmarkId::new("cpu", &size), |bench| {
                bench.iter(|| cpu(&a, &b));
            });
        }
        group.finish();
    }

    /// The CPU computation of one GPU thread, by thread index.
    type ThreadMirror<'a> = &'a (dyn Fn(usize) -> F + Sync);

    /// Four fmadd terms per round, as in `accum_bench.metal`.
    fn fmadd(x0: F, y: F) -> F {
        let x1 = x0 + y;
        let x2 = x1 + y;
        let x3 = x2 + y;
        let mut y = y;
        let mut acc = Acc::default();
        for _ in 0..FMADD_ROUNDS {
            for x in [x0, x1, x2, x3] {
                acc.fmadd(x, y);
            }
            y += x0;
        }
        acc.reduce()
    }

    fn fmadd4(x0: F, y: F) -> F {
        let x1 = x0 + y;
        let x2 = x1 + y;
        let x3 = x2 + y;
        let mut y = y;
        let mut accs = [Acc::default(); 4];
        for _ in 0..FMADD_ROUNDS {
            for (acc, x) in accs.iter_mut().zip([x0, x1, x2, x3]) {
                acc.fmadd(x, y);
            }
            y += x0;
        }
        let [a0, a1, a2, a3] = accs.map(Accumulator::reduce);
        (a0 + a1) + (a2 + a3)
    }

    fn fmadd_i64(x0: F, s: u64) -> F {
        let x1 = x0 + x0;
        let x2 = x1 + x0;
        let x3 = x2 + x0;
        let mut z = s;
        let mut acc = SmallScalarAcc::default();
        for _ in 0..FMADD_ROUNDS {
            for x in [x0, x1, x2, x3] {
                acc.fmadd_i64(x, z as i64);
                z = z.wrapping_add(SCALAR_STEP);
            }
        }
        acc.reduce()
    }

    fn accumulators(c: &mut Criterion, device: &Device, library: &ShaderLibrary) {
        let a = elements(7, CHAIN_THREADS);
        let b = elements(8, CHAIN_THREADS);
        let s = words(9, CHAIN_THREADS);
        let (a_dev, b_dev, s_dev) = (
            DeviceBuffer::from_slice(device, &a).expect("upload"),
            DeviceBuffer::from_slice(device, &b).expect("upload"),
            DeviceBuffer::from_slice(device, &s).expect("upload"),
        );
        let mut out = DeviceBuffer::<F>::zeroed(device, CHAIN_THREADS).expect("allocate");

        let mut group = c.benchmark_group("fp128_a7f7/accum");
        group.sample_size(10);
        group.throughput(Throughput::Elements(
            CHAIN_THREADS as u64 * u64::from(FMADD_ROUNDS) * 4,
        ));
        let b_elements = Binding::buffer(&b_dev);
        let scalars = Binding::buffer(&s_dev);
        let cases: [(&str, &str, Binding<'_>, ThreadMirror<'_>); 3] = [
            ("fmadd", ACCUM_FMADD, b_elements, &|i| fmadd(a[i], b[i])),
            ("fmadd4", ACCUM_FMADD4, b_elements, &|i| fmadd4(a[i], b[i])),
            ("fmadd_i64", ACCUM_FMADD_I64, scalars, &|i| {
                fmadd_i64(a[i], s[i])
            }),
        ];
        for (name, kernel, second, cpu) in cases {
            let pipeline = pipeline(library, kernel);
            let grid = Grid::linear(CHAIN_THREADS, threadgroup(pipeline));
            let run = |out: &DeviceBuffer<F>| {
                let bindings = [Binding::buffer(&a_dev), second, Binding::buffer(out)];
                dispatch(device, pipeline, &bindings, grid, 1)
            };
            let cpu_all = || -> Vec<F> { (0..CHAIN_THREADS).into_par_iter().map(cpu).collect() };
            run(&out);
            assert_eq!(read(&mut out), cpu_all(), "{kernel} disagrees with the CPU");

            group.bench_function(BenchmarkId::new("gpu", name), |bench| {
                bench.iter_custom(|iters| gpu_time(iters, || run(&out)));
            });
            group.bench_function(BenchmarkId::new("cpu", name), |bench| {
                bench.iter(cpu_all);
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
