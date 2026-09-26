//! Throughput of the MSL field types on the GPU against `jolt_field` on the
//! CPU.
//!
//! GPU samples are GPU execution time from command-buffer timestamps, which
//! exclude host submission and wake-up; CPU samples are wall time over all
//! cores with rayon, using `jolt_field`'s `asm` multiply as Akita does.
//! Every kernel's output is checked against the CPU once before it is
//! timed.
//!
//! Fields, each a group prefix `metal/{field}`: `fp128_a7f7`
//! (`Prime128OffsetA7F7`), `fp64_59` (`Prime64Offset59`) and `ext2_fp64_59`
//! (its `Ext2`, Akita's fp64 extension field). Groups, for every field:
//! - `chain/{add,mul,mul4,square}`: dependent operations per thread in
//!   registers, reported as operations per second;
//! - `stream/{add,mul,square}`: elementwise over 2^16 to 2^26 elements;
//! - `inner_product`: sum of `a[i] * b[i]` with a threadgroup reduction on the
//!   GPU, over 2^16 to 2^26 elements.
//!
//! For `fp128_a7f7`, whose accumulators are the only ones in MSL so far:
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
    use jolt_field::solinas::{Ext2, Prime128OffsetA7F7, Prime64Offset59};
    use jolt_field::{Accumulator, WithAccumulator};
    use jolt_metal::runtime::{Binding, Device, DeviceBuffer, Grid, ShaderLibrary};
    use rayon::prelude::*;

    use super::support::{dispatch, elements, library, pipeline, threadgroup, words, Sample};

    /// The field of the accumulator benchmarks.
    type F = Prime128OffsetA7F7;

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
    const FIELD_KERNELS: [&str; 8] = [
        ADD,
        MUL,
        SQUARE,
        ADD_CHAIN,
        MUL_CHAIN,
        MUL_CHAIN4,
        SQUARE_CHAIN,
        INNER_PRODUCT,
    ];
    const ACCUM_KERNELS: [&str; 4] = [
        ACCUM_FMADD,
        ACCUM_FMADD4,
        ACCUM_FMADD_I64,
        ACCUM_INNER_PRODUCT,
    ];
    const SOURCES: [(&str, &str); 3] = [
        ("field_ops.metal", FIELD_OPS),
        ("field_bench.metal", FIELD_BENCH),
        ("accum_bench.metal", ACCUM_BENCH),
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
        let fp128 = library::<F>(
            &device,
            &SOURCES,
            &[&FIELD_KERNELS[..], &ACCUM_KERNELS[..]].concat(),
            &[],
        );
        field::<F>(c, &device, &fp128, "fp128_a7f7");
        accumulators(c, &device, &fp128);
        inner_products(
            c,
            &device,
            &fp128,
            "fp128_a7f7/accum_inner_product",
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
        field_with_library::<Prime64Offset59>(c, &device, "fp64_59");
        field_with_library::<Ext2<Prime64Offset59>>(c, &device, "ext2_fp64_59");
    }

    fn field_with_library<T: Sample>(c: &mut Criterion, device: &Device, id: &str) {
        let library = library::<T>(device, &SOURCES, &FIELD_KERNELS, &[]);
        field::<T>(c, device, &library, id);
    }

    /// The chain, stream, and inner-product groups of one field.
    fn field<T: Sample>(c: &mut Criterion, device: &Device, library: &ShaderLibrary, id: &str) {
        chains::<T>(c, device, library, id);
        streams::<T>(c, device, library, id);
        inner_products::<T>(
            c,
            device,
            library,
            &format!("{id}/inner_product"),
            INNER_PRODUCT,
            |a, b| {
                a.par_iter()
                    .zip(b)
                    .map(|(x, y)| *x * *y)
                    .reduce(T::zero, |x, y| x + y)
            },
        );
    }

    /// A chain's CPU mirror: one thread's inputs and rounds to its output.
    type Chain<T> = fn(T, T, u32) -> T;
    /// An elementwise operation; unary ones ignore the second operand.
    type Op<T> = fn(T, T) -> T;

    /// Sums GPU time over `iters` runs of one dispatch, for `iter_custom`.
    fn gpu_time(iters: u64, run: impl Fn() -> Duration) -> Duration {
        (0..iters).map(|_| run()).sum()
    }

    fn read<T: Sample>(buffer: &mut DeviceBuffer<T>) -> Vec<T> {
        buffer.read().expect("canonical output").to_vec()
    }

    fn chains<T: Sample>(c: &mut Criterion, device: &Device, library: &ShaderLibrary, id: &str) {
        let a = elements::<T>(1, CHAIN_THREADS);
        let b = elements::<T>(2, CHAIN_THREADS);
        let (a_dev, b_dev) = (
            DeviceBuffer::from_slice(device, &a).expect("upload"),
            DeviceBuffer::from_slice(device, &b).expect("upload"),
        );
        let mut out = DeviceBuffer::<T>::zeroed(device, CHAIN_THREADS).expect("allocate");
        let rounds = CHAIN_ROUNDS;

        let mut group = c.benchmark_group(format!("metal/{id}/chain"));
        group.sample_size(10);
        let operations = CHAIN_THREADS as u64 * u64::from(rounds);

        let cases: [(&str, &str, u64, Chain<T>); 4] = [
            ("add", ADD_CHAIN, 1, |x, y, r| (0..r).fold(x, |x, _| x + y)),
            ("mul", MUL_CHAIN, 1, |x, y, r| (0..r).fold(x, |x, _| x * y)),
            ("mul4", MUL_CHAIN4, 4, |x, y, r| {
                let chain = |x: T| (0..r).fold(x, |x, _| x * y);
                let (x1, x2, x3) = (x + y, x + y + y, x + y + y + y);
                (chain(x) + chain(x1)) + (chain(x2) + chain(x3))
            }),
            ("square", SQUARE_CHAIN, 1, |x, _, r| {
                (0..r).fold(x, |x, _| x.square())
            }),
        ];
        for (name, kernel, chains, cpu) in cases {
            let pipeline = pipeline::<T>(library, kernel);
            let grid = Grid::linear(CHAIN_THREADS, threadgroup(pipeline));
            let unary = kernel == SQUARE_CHAIN;
            let run = |out: &DeviceBuffer<T>| {
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
            let cpu_all = || -> Vec<T> {
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

    fn streams<T: Sample>(c: &mut Criterion, device: &Device, library: &ShaderLibrary, id: &str) {
        let mut group = c.benchmark_group(format!("metal/{id}/stream"));
        group.sample_size(10);
        let cases: [(&str, &str, Op<T>); 3] = [
            ("add", ADD, |x, y| x + y),
            ("mul", MUL, |x, y| x * y),
            ("square", SQUARE, |x, _| x.square()),
        ];
        for log in LOG_SIZES {
            let len = 1usize << log;
            let a = elements::<T>(3, len);
            let b = elements::<T>(4, len);
            let (a_dev, b_dev) = (
                DeviceBuffer::from_slice(device, &a).expect("upload"),
                DeviceBuffer::from_slice(device, &b).expect("upload"),
            );
            let mut out = DeviceBuffer::<T>::zeroed(device, len).expect("allocate");
            let mut cpu_out = vec![T::zero(); len];
            group.throughput(Throughput::Elements(len as u64));
            for (name, kernel, op) in cases {
                let pipeline = pipeline::<T>(library, kernel);
                let grid = Grid::linear(len, threadgroup(pipeline));
                let unary = kernel == SQUARE;
                let run = |out: &DeviceBuffer<T>| {
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
                let cpu = |cpu_out: &mut [T]| {
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
    fn inner_products<T: Sample>(
        c: &mut Criterion,
        device: &Device,
        library: &ShaderLibrary,
        name: &str,
        kernel: &str,
        cpu: fn(&[T], &[T]) -> T,
    ) {
        let pipeline = pipeline::<T>(library, kernel);
        assert!(
            pipeline.max_total_threads_per_threadgroup() >= INNER_PRODUCT_GROUP,
            "{kernel} needs {INNER_PRODUCT_GROUP} threads per threadgroup",
        );
        let grid = Grid::linear(
            INNER_PRODUCT_GROUPS * INNER_PRODUCT_GROUP,
            INNER_PRODUCT_GROUP,
        );
        let mut group = c.benchmark_group(format!("metal/{name}"));
        group.sample_size(10);
        for log in LOG_SIZES {
            let len = 1usize << log;
            let a = elements::<T>(5, len);
            let b = elements::<T>(6, len);
            let (a_dev, b_dev) = (
                DeviceBuffer::from_slice(device, &a).expect("upload"),
                DeviceBuffer::from_slice(device, &b).expect("upload"),
            );
            let n = u32::try_from(len).expect("benchmark sizes fit u32");
            let mut partials =
                DeviceBuffer::<T>::zeroed(device, INNER_PRODUCT_GROUPS).expect("allocate");
            let run = |partials: &DeviceBuffer<T>| {
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
                .fold(T::zero(), |x, y| x + y);
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
        let a = elements::<F>(7, CHAIN_THREADS);
        let b = elements::<F>(8, CHAIN_THREADS);
        let s = words(9, CHAIN_THREADS);
        let (a_dev, b_dev, s_dev) = (
            DeviceBuffer::from_slice(device, &a).expect("upload"),
            DeviceBuffer::from_slice(device, &b).expect("upload"),
            DeviceBuffer::from_slice(device, &s).expect("upload"),
        );
        let mut out = DeviceBuffer::<F>::zeroed(device, CHAIN_THREADS).expect("allocate");

        let mut group = c.benchmark_group("metal/fp128_a7f7/accum");
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
            let pipeline = pipeline::<F>(library, kernel);
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
