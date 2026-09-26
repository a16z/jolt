//! The machine limits a kernel can be bound by, and kernels' fractions of
//! them (specs/jolt-metal-field.md, Performance model). Prints Markdown for
//! the local report.
//!
//! Limits, from GPU timestamps unless noted:
//! - field multiply: `jolt_bench_field_mul_chain4`, four independent `mul`
//!   chains in each of 2^20 threads;
//! - multiply-accumulate: `jolt_bench_accum_fmadd`, 256 `fmadd` terms per
//!   reduction in each of 2^20 threads;
//! - memory bandwidth: `jolt_bench_copy` and `jolt_bench_read` over 2^18 to
//!   2^26 16 B elements, counting bytes read and written. A sample moves at
//!   least 2^24 elements, by repeating smaller passes within one batch, so
//!   in-cache sizes are not measuring launch cost alone. A kernel that only
//!   reads is bound by the read rate, which is above the copy rate;
//! - threadgroup memory bandwidth: `jolt_bench_threadgroup_load`;
//! - round trip, in wall time: from committing a batch to the host holding
//!   its result, for an empty batch and for one threadgroup reducing an
//!   inner product of 1024 elements to one element that the host reads.
//!
//! Each figure is the median over `ROUNDS` rounds with its p10–p90 spread,
//! and a round's sample is the median of `REPS` runs. A fraction pairs a
//! kernel with its bounding limit: each round samples both back to back, in
//! alternating order, and the fraction is the median per-round ratio of
//! their rates.
//!
//! The field kernels are checked against the CPU in `benches/field.rs`; the
//! kernels only this benchmark runs are checked here before they are timed.

#[cfg(target_os = "macos")]
mod support;

#[cfg(target_os = "macos")]
#[expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "a benchmark aborts on any failure and prints its report"
)]
mod metal {
    use std::process::Command;
    use std::time::{Duration, Instant};

    use jolt_field::solinas::Prime128OffsetA7F7;
    use jolt_field::Zero;
    use jolt_metal::runtime::{Batch, Binding, Device, DeviceBuffer, Grid};

    use super::support::{dispatch, elements, library, pipeline, threadgroup, words};

    /// The field whose kernels the limits bound.
    type F = Prime128OffsetA7F7;

    const FIELD_OPS: &str = include_str!("../tests/shaders/field_ops.metal");
    const FIELD_BENCH: &str = include_str!("shaders/field_bench.metal");
    const ACCUM_BENCH: &str = include_str!("shaders/accum_bench.metal");
    const LIMITS: &str = include_str!("shaders/limits.metal");

    const MUL: &str = "jolt_test_field_mul";
    const MUL_CHAIN4: &str = "jolt_bench_field_mul_chain4";
    const INNER_PRODUCT: &str = "jolt_bench_field_inner_product";
    const ACCUM_FMADD: &str = "jolt_bench_accum_fmadd";
    const ACCUM_INNER_PRODUCT: &str = "jolt_bench_accum_inner_product";
    const TEMPLATES: [&str; 5] = [
        MUL,
        MUL_CHAIN4,
        INNER_PRODUCT,
        ACCUM_FMADD,
        ACCUM_INNER_PRODUCT,
    ];
    const COPY: &str = "jolt_bench_copy";
    const READ: &str = "jolt_bench_read";
    const THREADGROUP_LOAD: &str = "jolt_bench_threadgroup_load";

    const ROUNDS: usize = 31;
    const REPS: usize = 5;
    const ROUND_TRIPS: usize = 201;

    /// Threads of the register-bound kernels.
    const THREADS: usize = 1 << 20;
    const CHAIN_ROUNDS: u32 = 256;
    /// `FMADD_ROUNDS` in `accum_bench.metal`, four terms per round.
    const FMADD_ROUNDS: usize = 64;
    /// `INNER_PRODUCT_GROUP` in `field_bench.metal`; the accumulator inner
    /// product runs in the same shape.
    const INNER_PRODUCT_GROUP: usize = 256;
    const INNER_PRODUCT_GROUPS: usize = 1024;
    /// `TILE_WORDS` and `TILE_ROUNDS` in `limits.metal`.
    const TILE_WORDS: usize = 256;
    const TILE_ROUNDS: usize = 1024;
    /// `READ_WORDS` in `limits.metal`.
    const READ_WORDS: usize = 4;
    /// Elements of the round-trip reduction, one threadgroup.
    const ROUND_TRIP_ELEMENTS: usize = 1024;

    const MEMORY_LOG_SIZES: [u32; 5] = [18, 20, 22, 24, 26];
    const MIN_SAMPLE_ELEMENTS: usize = 1 << 24;
    /// Size of the fraction cases.
    const FRACTION_LOG: u32 = 24;

    /// A 16 B word, the device `uint4`.
    type Word = [u32; 4];
    const WORD_BYTES: usize = 16;
    const ELEMENT_BYTES: usize = 16;

    /// The memory-bandwidth kernels.
    #[derive(Clone, Copy)]
    enum Memory {
        Copy,
        Read,
    }

    impl Memory {
        fn name(self) -> &'static str {
            match self {
                Self::Copy => "memory copy",
                Self::Read => "memory read",
            }
        }

        fn kernel(self) -> &'static str {
            match self {
                Self::Copy => COPY,
                Self::Read => READ,
            }
        }

        /// Threads for `len` words.
        fn threads(self, len: usize) -> usize {
            match self {
                Self::Copy => len,
                Self::Read => len / READ_WORDS,
            }
        }

        /// Bytes read and written for `len` words.
        fn bytes(self, len: usize) -> usize {
            WORD_BYTES * (len + self.threads(len))
        }
    }

    /// A timed workload of `work` units per run.
    struct Case<'a> {
        work: f64,
        run: Box<dyn Fn() -> Duration + 'a>,
    }

    impl Case<'_> {
        /// Units per second: `work` over the median of `REPS` runs.
        fn sample(&self) -> f64 {
            let times = (0..REPS).map(|_| (self.run)().as_secs_f64()).collect();
            self.work / Spread::of(times).median
        }
    }

    /// Median and 10th–90th percentile spread.
    struct Spread {
        median: f64,
        p10: f64,
        p90: f64,
    }

    impl Spread {
        fn of(mut values: Vec<f64>) -> Self {
            values.sort_by(f64::total_cmp);
            let at = |q: f64| values[((values.len() - 1) as f64 * q).round() as usize];
            Self {
                median: at(0.5),
                p10: at(0.1),
                p90: at(0.9),
            }
        }

        fn row(&self, scale: f64, digits: usize) -> String {
            format!(
                "{:.digits$} | {:.digits$}–{:.digits$}",
                self.median * scale,
                self.p10 * scale,
                self.p90 * scale,
            )
        }
    }

    /// Rates of `case` over `ROUNDS` rounds, after one warm-up run.
    fn rate(case: &Case<'_>) -> Spread {
        let _ = (case.run)();
        Spread::of((0..ROUNDS).map(|_| case.sample()).collect())
    }

    /// Per-round ratios of `kernel`'s rate to `limit`'s, sampled back to back
    /// in alternating order.
    fn fraction(kernel: &Case<'_>, limit: &Case<'_>) -> Spread {
        let _ = ((kernel.run)(), (limit.run)());
        let ratios = (0..ROUNDS)
            .map(|round| {
                let (k, l) = if round % 2 == 0 {
                    let l = limit.sample();
                    (kernel.sample(), l)
                } else {
                    let k = kernel.sample();
                    (k, limit.sample())
                };
                k / l
            })
            .collect();
        Spread::of(ratios)
    }

    /// Wall time of `ROUND_TRIPS` runs of `run`.
    fn wall(mut run: impl FnMut() -> Duration) -> Spread {
        let _ = run();
        Spread::of((0..ROUND_TRIPS).map(|_| run().as_secs_f64()).collect())
    }

    fn command(program: &str, args: &[&str]) -> String {
        Command::new(program).args(args).output().map_or_else(
            |error| format!("unavailable ({error})"),
            |out| String::from_utf8_lossy(&out.stdout).trim().to_string(),
        )
    }

    fn to_words(values: &[u64]) -> Vec<Word> {
        values
            .chunks_exact(2)
            .map(|pair| {
                let [a, b] = [pair[0], pair[1]];
                [a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32]
            })
            .collect()
    }

    pub fn main() {
        let device = Device::system_default().expect("a supported Metal device");
        let library = library::<F>(
            &device,
            &[
                ("field_ops.metal", FIELD_OPS),
                ("field_bench.metal", FIELD_BENCH),
                ("accum_bench.metal", ACCUM_BENCH),
                ("limits.metal", LIMITS),
            ],
            &TEMPLATES,
            &[COPY, READ, THREADGROUP_LOAD],
        );
        println!("### Machine limits: {}", device.name());
        println!();
        println!(
            "- load average before: {}",
            command("sysctl", &["-n", "vm.loadavg"])
        );
        println!("- {ROUNDS} rounds, each sample the median of {REPS} runs");
        println!();

        let len = 1usize << FRACTION_LOG;
        let (a, b) = (elements::<F>(1, len), elements::<F>(2, len));
        let a_dev = DeviceBuffer::from_slice(&device, &a).expect("upload");
        let b_dev = DeviceBuffer::from_slice(&device, &b).expect("upload");
        let field_out = DeviceBuffer::<F>::zeroed(&device, len).expect("allocate");
        let partials = DeviceBuffer::<F>::zeroed(&device, INNER_PRODUCT_GROUPS).expect("allocate");
        let max_words = 1usize << MEMORY_LOG_SIZES[MEMORY_LOG_SIZES.len() - 1];
        let source = to_words(&words(3, 2 * max_words));
        let source_dev = DeviceBuffer::from_slice(&device, &source).expect("upload");
        let mut copy_dev = DeviceBuffer::<Word>::zeroed(&device, max_words).expect("allocate");
        let mut tile_out = DeviceBuffer::<Word>::zeroed(&device, THREADS).expect("allocate");

        // The kernels in limits.metal exist only here: check them.
        let memory_grid = |kind: Memory, len: usize| {
            let pipeline = pipeline::<F>(&library, kind.kernel());
            Grid::linear(kind.threads(len), threadgroup(pipeline))
        };
        for kind in [Memory::Copy, Memory::Read] {
            let bindings = [Binding::buffer(&source_dev), Binding::buffer(&copy_dev)];
            let grid = memory_grid(kind, max_words);
            let _ = dispatch(
                &device,
                pipeline::<F>(&library, kind.kernel()),
                &bindings,
                grid,
                1,
            );
            let threads = kind.threads(max_words);
            let expected: Vec<Word> = match kind {
                Memory::Copy => source.clone(),
                Memory::Read => (0..threads)
                    .map(|i| {
                        (0..READ_WORDS).fold([0u32; 4], |acc, k| {
                            let word = source[i + k * threads];
                            std::array::from_fn(|lane| acc[lane].wrapping_add(word[lane]))
                        })
                    })
                    .collect(),
            };
            assert!(
                copy_dev.read().expect("read back")[..threads] == expected[..],
                "{} disagrees with the CPU",
                kind.kernel()
            );
        }
        let tile_pipeline = pipeline::<F>(&library, THREADGROUP_LOAD);
        assert!(tile_pipeline.max_total_threads_per_threadgroup() >= TILE_WORDS);
        let tile_grid = Grid::linear(THREADS, TILE_WORDS);
        let tile_bindings = [Binding::buffer(&source_dev), Binding::buffer(&tile_out)];
        let _ = dispatch(&device, tile_pipeline, &tile_bindings, tile_grid, 1);
        let expected: Vec<Word> = (0..TILE_WORDS)
            .map(|t| {
                (0..TILE_ROUNDS).fold([0u32; 4], |acc, r| {
                    let word = source[(t + 32 * r) % TILE_WORDS];
                    std::array::from_fn(|k| acc[k].wrapping_add(word[k]))
                })
            })
            .collect();
        assert!(
            tile_out
                .read()
                .expect("read back")
                .chunks(TILE_WORDS)
                .all(|group| group == expected.as_slice()),
            "{THREADGROUP_LOAD} disagrees with the CPU"
        );

        let (device, a_dev, b_dev, field_out, partials) =
            (&device, &a_dev, &b_dev, &field_out, &partials);
        let (source_dev, copy_dev, tile_out) = (&source_dev, &copy_dev, &tile_out);
        let multiply = {
            let pipeline = pipeline::<F>(&library, MUL_CHAIN4);
            let grid = Grid::linear(THREADS, threadgroup(pipeline));
            Case {
                work: (THREADS * 4) as f64 * f64::from(CHAIN_ROUNDS),
                run: Box::new(move || {
                    let bindings = [
                        Binding::buffer(a_dev),
                        Binding::buffer(b_dev),
                        Binding::value(&CHAIN_ROUNDS),
                        Binding::buffer(field_out),
                    ];
                    dispatch(device, pipeline, &bindings, grid, 1)
                }),
            }
        };
        let fmadd = {
            let pipeline = pipeline::<F>(&library, ACCUM_FMADD);
            let grid = Grid::linear(THREADS, threadgroup(pipeline));
            Case {
                work: (THREADS * FMADD_ROUNDS * 4) as f64,
                run: Box::new(move || {
                    let bindings = [
                        Binding::buffer(a_dev),
                        Binding::buffer(b_dev),
                        Binding::buffer(field_out),
                    ];
                    dispatch(device, pipeline, &bindings, grid, 1)
                }),
            }
        };
        let memory = |kind: Memory, log: u32| {
            let len = 1usize << log;
            let repeats = (MIN_SAMPLE_ELEMENTS / len).max(1);
            let (pipeline, grid) = (
                pipeline::<F>(&library, kind.kernel()),
                memory_grid(kind, len),
            );
            Case {
                work: (kind.bytes(len) * repeats) as f64,
                run: Box::new(move || {
                    let bindings = [Binding::buffer(source_dev), Binding::buffer(copy_dev)];
                    dispatch(device, pipeline, &bindings, grid, repeats)
                }),
            }
        };
        let tile = Case {
            work: (THREADS * TILE_ROUNDS * WORD_BYTES) as f64,
            run: Box::new(move || {
                let bindings = [Binding::buffer(source_dev), Binding::buffer(tile_out)];
                dispatch(device, tile_pipeline, &bindings, tile_grid, 1)
            }),
        };

        println!("| limit | case | median | p10–p90 |");
        println!("|---|---|---:|---:|");
        let multiply_rate = rate(&multiply);
        println!(
            "| field multiply (G/s) | 4 chains × 2^20 threads | {} |",
            multiply_rate.row(1e-9, 1)
        );
        let fmadd_rate = rate(&fmadd);
        println!(
            "| multiply-accumulate (G/s) | 256 terms per reduction × 2^20 threads | {} |",
            fmadd_rate.row(1e-9, 1)
        );
        let mut beyond_cache = 0.0;
        for kind in [Memory::Copy, Memory::Read] {
            for log in MEMORY_LOG_SIZES {
                let spread = rate(&memory(kind, log));
                println!(
                    "| {} (GB/s) | 2^{log} elements ({} MiB read) | {} |",
                    kind.name(),
                    (WORD_BYTES << log) >> 20,
                    spread.row(1e-9, 0)
                );
                beyond_cache = spread.median;
            }
        }
        println!(
            "| threadgroup memory (GB/s) | 16 B loads, {TILE_WORDS}-thread groups | {} |",
            rate(&tile).row(1e-9, 0)
        );

        let empty = wall(|| {
            let batch = Batch::new(device).expect("command batch");
            let start = Instant::now();
            let _ = batch.commit_and_wait().expect("batch completes");
            start.elapsed()
        });
        println!(
            "| round trip (µs, wall) | empty batch | {} |",
            empty.row(1e6, 0)
        );
        let reduce_pipeline = pipeline::<F>(&library, ACCUM_INNER_PRODUCT);
        let n = ROUND_TRIP_ELEMENTS as u32;
        let mut sum_dev = DeviceBuffer::<F>::zeroed(device, 1).expect("allocate");
        let expected = a[..ROUND_TRIP_ELEMENTS]
            .iter()
            .zip(&b)
            .fold(F::zero(), |sum, (x, y)| sum + *x * *y);
        let mut gpu_times = Vec::with_capacity(ROUND_TRIPS + 1);
        let reduce = wall(|| {
            let mut batch = Batch::new(device).expect("command batch");
            let bindings = [
                Binding::buffer(a_dev),
                Binding::buffer(b_dev),
                Binding::value(&n),
                Binding::buffer(&sum_dev),
            ];
            let grid = Grid::linear(ROUND_TRIP_ELEMENTS, ROUND_TRIP_ELEMENTS);
            batch
                .dispatch(reduce_pipeline, &bindings, grid)
                .expect("valid dispatch");
            let start = Instant::now();
            gpu_times.push(
                batch
                    .commit_and_wait()
                    .expect("batch completes")
                    .as_secs_f64(),
            );
            let sum = sum_dev.read().expect("canonical output")[0];
            let elapsed = start.elapsed();
            assert_eq!(
                sum, expected,
                "{ACCUM_INNER_PRODUCT} disagrees with the CPU"
            );
            elapsed
        });
        println!(
            "| round trip (µs, wall) | inner product of {ROUND_TRIP_ELEMENTS} elements to one, read back | {} |",
            reduce.row(1e6, 0)
        );
        // The first run is `wall`'s warm-up.
        println!(
            "| of which GPU time (µs) | the same dispatch | {} |",
            Spread::of(gpu_times.split_off(1)).row(1e6, 0)
        );
        println!();
        println!(
            "Ridge at the 2^{} read rate: {:.2} multiplies and {:.2} multiply-accumulates \
             per 16 B element read.",
            MEMORY_LOG_SIZES[MEMORY_LOG_SIZES.len() - 1],
            multiply_rate.median * ELEMENT_BYTES as f64 / beyond_cache,
            fmadd_rate.median * ELEMENT_BYTES as f64 / beyond_cache,
        );
        println!();

        println!("### Fractions of limits (paired, 2^{FRACTION_LOG} elements)");
        println!();
        println!("| kernel | work per element | bound | fraction | p10–p90 |");
        println!("|---|---|---|---:|---:|");
        let limits = [
            memory(Memory::Copy, FRACTION_LOG),
            memory(Memory::Read, FRACTION_LOG),
        ];
        let stream_mul = {
            let pipeline = pipeline::<F>(&library, MUL);
            let grid = Grid::linear(len, threadgroup(pipeline));
            Case {
                work: (3 * ELEMENT_BYTES * len) as f64,
                run: Box::new(move || {
                    let bindings = [
                        Binding::buffer(a_dev),
                        Binding::buffer(b_dev),
                        Binding::buffer(field_out),
                    ];
                    dispatch(device, pipeline, &bindings, grid, 1)
                }),
            }
        };
        let inner_product = |kernel: &str| {
            let pipeline = pipeline::<F>(&library, kernel);
            let grid = Grid::linear(
                INNER_PRODUCT_GROUPS * INNER_PRODUCT_GROUP,
                INNER_PRODUCT_GROUP,
            );
            let n = len as u32;
            Case {
                work: (2 * ELEMENT_BYTES * len) as f64,
                run: Box::new(move || {
                    let bindings = [
                        Binding::buffer(a_dev),
                        Binding::buffer(b_dev),
                        Binding::value(&n),
                        Binding::buffer(partials),
                    ];
                    dispatch(device, pipeline, &bindings, grid, 1)
                }),
            }
        };
        let cases = [
            (
                "stream `mul`",
                "1 mul; 32 B read, 16 B written",
                stream_mul,
                Memory::Copy,
            ),
            (
                "inner product",
                "1 mul, 1 add; 32 B read",
                inner_product(INNER_PRODUCT),
                Memory::Read,
            ),
            (
                "accumulator inner product",
                "1 fmadd; 32 B read",
                inner_product(ACCUM_INNER_PRODUCT),
                Memory::Read,
            ),
        ];
        for (name, work, kernel, bound) in &cases {
            println!(
                "| {name} | {work} | {} | {} |",
                bound.name(),
                fraction(kernel, &limits[*bound as usize]).row(1.0, 3)
            );
        }
        println!();
        println!(
            "- load average after: {}",
            command("sysctl", &["-n", "vm.loadavg"])
        );
    }
}

#[cfg(target_os = "macos")]
fn main() {
    metal::main();
}

/// Metal exists only on macOS; elsewhere the benchmark has nothing to run.
#[cfg(not(target_os = "macos"))]
fn main() {}
