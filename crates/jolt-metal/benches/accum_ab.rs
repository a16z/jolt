//! Accumulator-representation A/B for `jolt::Fp128` (evidence for step 3 of
//! `specs/jolt-metal-field.md`; this branch is not for merge).
//!
//! Variants are the structs of `benches/accum_ab/variants.h`, each bound to
//! `jolt::WithAccumulator` in its own library and run through the kernels of
//! `benches/shaders/accum_bench.metal`. Product cases compare `Carried9`,
//! `Slots8`, `Columns8`, and `NaiveProduct`; the signed case compares
//! `Signed7`, `PosNeg7`, and `NaiveSigned`.
//!
//! Method, as for the limb-layout A/B: every variant's output is checked
//! against the CPU first. Each round then times every variant on a case, in
//! alternating order, as the median GPU time of `REPS` dispatches. The report
//! gives the median over rounds, its p10–p90 spread, and the median and
//! p10–p90 of the per-round ratio against the case's first variant.
//!
//! Decision rule, fixed before the run: the product accumulator is the variant
//! with the lowest `fmadd` time, if it beats every other variant by at least
//! 3% there and is at most 3% slower than the best on `fmadd4`. Otherwise it
//! is the variant with the fewest words among those within 3% of the best on
//! `fmadd`, since consumer kernels spend registers on other state. The same
//! rule picks the small-scalar accumulator on `fmadd_i64`.

#[cfg(target_os = "macos")]
#[expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "a benchmark aborts on any failure and prints its report"
)]
mod metal {
    use std::process::Command;
    use std::time::Duration;

    use jolt_field::solinas::Prime128OffsetA7F7;
    use jolt_field::{Ring, Zero};
    use jolt_metal::runtime::{
        host_name, Batch, Binding, Device, DeviceBuffer, Grid, LibrarySpec, Pipeline, ShaderLibrary,
    };
    use jolt_metal::shaders::FIELD_HEADERS;
    use rayon::prelude::*;

    type F = Prime128OffsetA7F7;

    const VARIANTS: &str = include_str!("accum_ab/variants.h");
    const ACCUM_BENCH: &str = include_str!("accum_ab/kernels.metal");

    const FMADD: &str = "jolt_bench_accum_fmadd";
    const FMADD4: &str = "jolt_bench_accum_fmadd4";
    const FMADD_I64: &str = "jolt_bench_accum_fmadd_i64";
    const INNER_PRODUCT: &str = "jolt_bench_accum_inner_product";
    const KERNELS: [&str; 4] = [FMADD, FMADD4, FMADD_I64, INNER_PRODUCT];

    /// `FMADD_ROUNDS` and `ACCUM_INNER_PRODUCT_GROUP` in `accum_bench.metal`.
    const FMADD_ROUNDS: u64 = 64;
    const INNER_PRODUCT_GROUP: usize = 256;
    const INNER_PRODUCT_GROUPS: usize = 1024;
    const THREADS: usize = 1 << 20;
    const STEP: u64 = 0x9E37_79B9_7F4A_7C15;

    const ROUNDS: usize = 31;
    const REPS: usize = 5;

    const PRODUCT: [(&str, usize); 4] = [
        ("Carried9", 9),
        ("Slots8", 16),
        ("Columns8", 16),
        ("NaiveProduct", 4),
    ];
    const SIGNED: [(&str, usize); 3] = [("Signed7", 7), ("PosNeg7", 14), ("NaiveSigned", 4)];

    struct Variant {
        name: &'static str,
        words: usize,
        library: ShaderLibrary,
    }

    #[derive(Clone, Copy)]
    enum Case {
        Fmadd,
        Fmadd4,
        FmaddI64,
        InnerProduct { log: u32 },
    }

    impl Case {
        fn name(self) -> String {
            match self {
                Self::Fmadd => "fmadd".to_owned(),
                Self::Fmadd4 => "fmadd4".to_owned(),
                Self::FmaddI64 => "fmadd_i64".to_owned(),
                Self::InnerProduct { log } => format!("inner product 2^{log}"),
            }
        }

        fn kernel(self) -> &'static str {
            match self {
                Self::Fmadd => FMADD,
                Self::Fmadd4 => FMADD4,
                Self::FmaddI64 => FMADD_I64,
                Self::InnerProduct { .. } => INNER_PRODUCT,
            }
        }

        /// Multiply-accumulates per dispatch.
        fn work(self) -> u64 {
            match self {
                Self::InnerProduct { log } => 1 << log,
                _ => THREADS as u64 * FMADD_ROUNDS * 4,
            }
        }
    }

    fn library(device: &Device, product: &str, signed: &str) -> ShaderLibrary {
        let binding = format!(
            "namespace jolt {{\n\
             template <typename F> struct WithAccumulator;\n\
             template <uint C> struct WithAccumulator<Fp128<C>> {{\n\
                 using Accumulator = jolt_ab::{product}<Fp128<C>>;\n\
                 using SmallScalarAccumulator = jolt_ab::{signed}<Fp128<C>>;\n\
             }};\n\
             template <typename F> using Accumulator = typename WithAccumulator<F>::Accumulator;\n\
             template <typename F> using SmallScalarAccumulator =\n\
                 typename WithAccumulator<F>::SmallScalarAccumulator;\n\
             }}\n"
        );
        // Only the field itself: the variants stand in for the accumulator
        // headers, and the pinned kernels do not use reduce.h.
        let spec = FIELD_HEADERS
            .iter()
            .filter(|(name, _)| *name == "jolt/field/fp128.h")
            .fold(LibrarySpec::new(), |spec, (name, text)| {
                spec.source(name, text)
            })
            .source("variants.h", VARIANTS)
            .source("binding.h", &binding)
            .source("accum_bench.metal", ACCUM_BENCH);
        let spec = KERNELS
            .iter()
            .fold(spec, |spec, kernel| spec.instantiate::<F>(kernel));
        ShaderLibrary::compile(device, &spec).expect("variant compiles")
    }

    fn pipeline<'l>(variant: &'l Variant, kernel: &str) -> &'l Pipeline {
        variant
            .library
            .pipeline(&host_name::<F>(kernel))
            .expect("kernel is instantiated")
    }

    fn threadgroup(pipeline: &Pipeline) -> usize {
        (pipeline.thread_execution_width() * 8).min(pipeline.max_total_threads_per_threadgroup())
    }

    fn words(seed: u64, len: usize) -> Vec<u64> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_add(STEP);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^ (z >> 31)
            })
            .collect()
    }

    fn elements(seed: u64, len: usize) -> Vec<F> {
        words(seed, 2 * len)
            .chunks_exact(2)
            .map(|w| F::from_u128((u128::from(w[0]) << 64) | u128::from(w[1])))
            .collect()
    }

    struct Setup {
        a: DeviceBuffer<F>,
        b: DeviceBuffer<F>,
        s: DeviceBuffer<u64>,
        out: DeviceBuffer<F>,
        len: usize,
        expected: F,
    }

    fn setup(device: &Device, case: Case) -> Setup {
        let len = match case {
            Case::InnerProduct { log } => 1 << log,
            _ => THREADS,
        };
        let a = elements(1, len);
        let b = elements(2, len);
        let s = words(3, len);
        let expected = match case {
            Case::Fmadd | Case::Fmadd4 => a
                .par_iter()
                .zip(&b)
                .map(|(&x0, &y)| {
                    let xs = [x0, x0 + y, x0 + y + y, x0 + y + y + y];
                    let x_sum = xs.iter().fold(F::zero(), |acc, x| acc + *x);
                    let mut y = y;
                    let mut sum = F::zero();
                    for _ in 0..FMADD_ROUNDS {
                        sum += x_sum * y;
                        y += x0;
                    }
                    sum
                })
                .reduce(F::zero, |x, y| x + y),
            Case::FmaddI64 => a
                .par_iter()
                .zip(&s)
                .map(|(&x0, &z)| {
                    let xs = [x0, x0 + x0, x0 + x0 + x0, x0 + x0 + x0 + x0];
                    let mut z = z;
                    let mut sum = F::zero();
                    for _ in 0..FMADD_ROUNDS {
                        for x in xs {
                            sum += x * F::from_i64(z as i64);
                            z = z.wrapping_add(STEP);
                        }
                    }
                    sum
                })
                .reduce(F::zero, |x, y| x + y),
            Case::InnerProduct { .. } => a
                .par_iter()
                .zip(&b)
                .map(|(x, y)| *x * *y)
                .reduce(F::zero, |x, y| x + y),
        };
        let out_len = match case {
            Case::InnerProduct { .. } => INNER_PRODUCT_GROUPS,
            _ => len,
        };
        Setup {
            a: DeviceBuffer::from_slice(device, &a).expect("upload"),
            b: DeviceBuffer::from_slice(device, &b).expect("upload"),
            s: DeviceBuffer::from_slice(device, &s).expect("upload"),
            out: DeviceBuffer::zeroed(device, out_len).expect("allocate"),
            len,
            expected,
        }
    }

    fn run(device: &Device, variant: &Variant, case: Case, setup: &Setup) -> Duration {
        let pipeline = pipeline(variant, case.kernel());
        let n = u32::try_from(setup.len).expect("sizes fit u32");
        let (a, b, out) = (
            Binding::buffer(&setup.a),
            Binding::buffer(&setup.b),
            Binding::buffer(&setup.out),
        );
        let (bindings, grid) = match case {
            Case::Fmadd | Case::Fmadd4 => (
                vec![a, b, out],
                Grid::linear(setup.len, threadgroup(pipeline)),
            ),
            Case::FmaddI64 => (
                vec![a, Binding::buffer(&setup.s), out],
                Grid::linear(setup.len, threadgroup(pipeline)),
            ),
            Case::InnerProduct { .. } => (
                vec![a, b, Binding::value(&n), out],
                Grid::linear(
                    INNER_PRODUCT_GROUPS * INNER_PRODUCT_GROUP,
                    INNER_PRODUCT_GROUP,
                ),
            ),
        };
        let mut batch = Batch::new(device).expect("command batch");
        batch
            .dispatch(pipeline, &bindings, grid)
            .expect("valid dispatch");
        batch.commit_and_wait().expect("batch completes")
    }

    fn check(device: &Device, variant: &Variant, case: Case, setup: &mut Setup) {
        let _ = run(device, variant, case, setup);
        let got = setup
            .out
            .read()
            .expect("canonical output")
            .iter()
            .fold(F::zero(), |x, y| x + *y);
        assert!(
            got == setup.expected,
            "{} on {} disagrees with the CPU",
            case.name(),
            variant.name
        );
    }

    fn percentile(sorted: &[f64], q: f64) -> f64 {
        let index = ((sorted.len() - 1) as f64 * q).round() as usize;
        sorted[index]
    }

    fn sorted(mut values: Vec<f64>) -> Vec<f64> {
        values.sort_by(f64::total_cmp);
        values
    }

    fn command(program: &str, args: &[&str]) -> String {
        Command::new(program)
            .args(args)
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
            .unwrap_or_default()
    }

    fn compare(device: &Device, case: Case, variants: &[Variant]) {
        let mut setup = setup(device, case);
        for variant in variants {
            check(device, variant, case, &mut setup);
        }
        let mut samples = vec![Vec::with_capacity(ROUNDS); variants.len()];
        for round in 0..ROUNDS {
            let mut order: Vec<usize> = (0..variants.len()).collect();
            if round % 2 == 1 {
                order.reverse();
            }
            for v in order {
                let reps = sorted(
                    (0..REPS)
                        .map(|_| run(device, &variants[v], case, &setup).as_secs_f64())
                        .collect(),
                );
                samples[v].push(percentile(&reps, 0.5));
            }
        }
        for (v, variant) in variants.iter().enumerate() {
            let times = sorted(samples[v].clone());
            let ratios = sorted(
                samples[v]
                    .iter()
                    .zip(&samples[0])
                    .map(|(t, base)| t / base)
                    .collect(),
            );
            let median = percentile(&times, 0.5);
            println!(
                "| {} | {} | {} | {} | {:.1} µs | {:.1}–{:.1} µs | {:.2} | {:.3} ({:.3}–{:.3}) |",
                case.name(),
                variant.name,
                variant.words,
                pipeline(variant, case.kernel()).max_total_threads_per_threadgroup(),
                median * 1e6,
                percentile(&times, 0.1) * 1e6,
                percentile(&times, 0.9) * 1e6,
                case.work() as f64 / median / 1e9,
                percentile(&ratios, 0.5),
                percentile(&ratios, 0.1),
                percentile(&ratios, 0.9),
            );
        }
    }

    pub fn main() {
        let device = Device::system_default().expect("a supported Metal device");
        let product = PRODUCT.map(|(name, words)| Variant {
            name,
            words,
            library: library(&device, name, SIGNED[0].0),
        });
        let signed = SIGNED.map(|(name, words)| Variant {
            name,
            words,
            library: library(&device, PRODUCT[0].0, name),
        });

        println!("# Fp128 accumulator A/B\n");
        println!(
            "- device: {} (Apple GPU family {})",
            device.name(),
            device.limits().apple_family
        );
        println!("- macOS: {}", command("sw_vers", &["-productVersion"]));
        println!(
            "- power: {}",
            command("pmset", &["-g", "batt"])
                .lines()
                .next()
                .unwrap_or("")
        );
        println!(
            "- load average before: {}",
            command("sysctl", &["-n", "vm.loadavg"])
        );
        println!(
            "- commit: {}",
            command("git", &["rev-parse", "--short", "HEAD"])
        );
        println!("- rounds {ROUNDS}, {REPS} dispatches per sample, alternating order\n");
        println!("| case | variant | words | max threads/group | median | p10–p90 | G fmadd/s | ratio vs first (p10–p90) |");
        println!("|---|---|---|---|---|---|---|---|");
        for case in [
            Case::Fmadd,
            Case::Fmadd4,
            Case::InnerProduct { log: 20 },
            Case::InnerProduct { log: 24 },
        ] {
            compare(&device, case, &product);
        }
        compare(&device, Case::FmaddI64, &signed);
        println!(
            "\n- load average after: {}",
            command("sysctl", &["-n", "vm.loadavg"])
        );
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    metal::main();
}
