//! Limb-layout A/B for `jolt::Fp128` (evidence for step 2 of
//! `specs/jolt-metal-field.md`; this branch is not for merge).
//!
//! Variants, each compiled into its own library from the same kernels:
//! - `u32`: `shaders/jolt/field/fp128.h`, four 32-bit limbs;
//! - `u64`: `benches/limb_ab/fp128_u64.h`, two 64-bit limbs with native
//!   64-bit `*` and `mulhi`, the same algorithm and the same storage;
//! - `u32/sq=mul`, `u64/sq=mul`: the same headers with `square(a)` replaced by
//!   `a * a`;
//! - `u32/sq=loop`: the `u32` header with the triangular-loop `sqr_wide`
//!   ported first (`benches/limb_ab/sqr_wide_loop.h`).
//!
//! Method: every variant's output is checked against the CPU first. Each
//! round then times every variant on a case, in alternating order (forward
//! on even rounds, reversed on odd), as the median GPU time of `REPS`
//! dispatches. The report gives, per case and variant, the median over rounds,
//! the p10–p90 spread of the round medians, and the median and p10–p90 of the
//! per-round ratio against `u32`: a paired comparison, so drift in clocks or
//! temperature that affects a whole round cancels.

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

    const FIELD_OPS: &str = include_str!("../tests/shaders/field_ops.metal");
    const FIELD_BENCH: &str = include_str!("shaders/field_bench.metal");
    const FP128_U64: &str = include_str!("limb_ab/fp128_u64.h");
    const SQR_WIDE_LOOP: &str = include_str!("limb_ab/sqr_wide_loop.h");

    const MUL: &str = "jolt_test_field_mul";
    const ADD_CHAIN: &str = "jolt_bench_field_add_chain";
    const MUL_CHAIN: &str = "jolt_bench_field_mul_chain";
    const MUL_CHAIN4: &str = "jolt_bench_field_mul_chain4";
    const SQUARE_CHAIN: &str = "jolt_bench_field_square_chain";
    const INNER_PRODUCT: &str = "jolt_bench_field_inner_product";
    const KERNELS: [&str; 6] = [
        MUL,
        ADD_CHAIN,
        MUL_CHAIN,
        MUL_CHAIN4,
        SQUARE_CHAIN,
        INNER_PRODUCT,
    ];

    const INNER_PRODUCT_GROUP: usize = 256;
    const INNER_PRODUCT_GROUPS: usize = 1024;
    const CHAIN_THREADS: usize = 1 << 20;
    const CHAIN_ROUNDS: u32 = 256;

    const ROUNDS: usize = 31;
    const REPS: usize = 5;

    struct Variant {
        name: &'static str,
        library: ShaderLibrary,
    }

    #[derive(Clone, Copy)]
    enum Case {
        Chain { kernel: &'static str, chains: u64 },
        Stream { log: u32 },
        InnerProduct { log: u32 },
    }

    impl Case {
        fn name(self) -> String {
            match self {
                Self::Chain { kernel, .. } => {
                    format!("chain {}", kernel.trim_start_matches("jolt_bench_field_"))
                }
                Self::Stream { log } => format!("stream mul 2^{log}"),
                Self::InnerProduct { log } => format!("inner product 2^{log}"),
            }
        }

        fn kernel(self) -> &'static str {
            match self {
                Self::Chain { kernel, .. } => kernel,
                Self::Stream { .. } => MUL,
                Self::InnerProduct { .. } => INNER_PRODUCT,
            }
        }

        /// Field operations (chains) or elements (stream, inner product) per
        /// dispatch.
        fn work(self) -> u64 {
            match self {
                Self::Chain { chains, .. } => {
                    CHAIN_THREADS as u64 * u64::from(CHAIN_ROUNDS) * chains
                }
                Self::Stream { log } | Self::InnerProduct { log } => 1 << log,
            }
        }
    }

    /// `header` with the body of `square` replaced by `a * a`.
    fn square_as_mul(header: &str) -> String {
        let start = header
            .find("friend Fp128 square(Fp128 a) {")
            .expect("header defines square");
        let end = start + header[start..].find("\n    }").expect("square has a body");
        assert_eq!(header.matches("friend Fp128 square(").count(), 1);
        format!(
            "{}friend Fp128 square(Fp128 a) {{\n        return a * a;{}",
            &header[..start],
            &header[end..]
        )
    }

    /// `header` with `sqr_wide` replaced by `limb_ab/sqr_wide_loop.h`.
    fn sqr_wide_loop(header: &str) -> String {
        let signature = "inline Words<8> sqr_wide(uint4 a) {";
        let start = header.find(signature).expect("header defines sqr_wide");
        let end = start + header[start..].find("\n}\n").expect("sqr_wide has a body") + 3;
        format!(
            "{}{}{}",
            &header[..start],
            SQR_WIDE_LOOP,
            &header[end..]
        )
    }

    fn library(device: &Device, header: &str) -> ShaderLibrary {
        let spec = LibrarySpec::new()
            .source("jolt/field/fp128.h", header)
            .source("field_ops.metal", FIELD_OPS)
            .source("field_bench.metal", FIELD_BENCH);
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

    /// Inputs, output, and CPU reference for one case.
    struct Setup {
        a: DeviceBuffer<F>,
        b: DeviceBuffer<F>,
        out: DeviceBuffer<F>,
        len: usize,
        expected: Vec<F>,
    }

    fn setup(device: &Device, case: Case) -> Setup {
        let len = match case {
            Case::Chain { .. } => CHAIN_THREADS,
            Case::Stream { log } | Case::InnerProduct { log } => 1 << log,
        };
        let a = elements(1, len);
        let b = elements(2, len);
        let expected = match case {
            Case::Chain { kernel, .. } => {
                let chain = |x: F, y: F| (0..CHAIN_ROUNDS).fold(x, |x, _| x * y);
                a.par_iter()
                    .zip(&b)
                    .map(|(&x, &y)| match kernel {
                        ADD_CHAIN => (0..CHAIN_ROUNDS).fold(x, |x, _| x + y),
                        MUL_CHAIN => chain(x, y),
                        MUL_CHAIN4 => {
                            let (x1, x2, x3) = (x + y, x + y + y, x + y + y + y);
                            (chain(x, y) + chain(x1, y)) + (chain(x2, y) + chain(x3, y))
                        }
                        _ => (0..CHAIN_ROUNDS).fold(x, |x, _| x.square()),
                    })
                    .collect()
            }
            Case::Stream { .. } => a.par_iter().zip(&b).map(|(x, y)| *x * *y).collect(),
            Case::InnerProduct { .. } => vec![a
                .par_iter()
                .zip(&b)
                .map(|(x, y)| *x * *y)
                .reduce(F::zero, |x, y| x + y)],
        };
        let out_len = match case {
            Case::InnerProduct { .. } => INNER_PRODUCT_GROUPS,
            _ => len,
        };
        Setup {
            a: DeviceBuffer::from_slice(device, &a).expect("upload"),
            b: DeviceBuffer::from_slice(device, &b).expect("upload"),
            out: DeviceBuffer::zeroed(device, out_len).expect("allocate"),
            len,
            expected,
        }
    }

    /// One dispatch of `case` on `variant`; returns its GPU time.
    fn run(device: &Device, variant: &Variant, case: Case, setup: &Setup) -> Duration {
        let pipeline = pipeline(variant, case.kernel());
        let rounds = CHAIN_ROUNDS;
        let n = u32::try_from(setup.len).expect("sizes fit u32");
        let (a, b, out) = (
            Binding::buffer(&setup.a),
            Binding::buffer(&setup.b),
            Binding::buffer(&setup.out),
        );
        let (bindings, grid) = match case {
            Case::Chain {
                kernel: SQUARE_CHAIN,
                ..
            } => (
                vec![a, Binding::value(&rounds), out],
                Grid::linear(setup.len, threadgroup(pipeline)),
            ),
            Case::Chain { .. } => (
                vec![a, b, Binding::value(&rounds), out],
                Grid::linear(setup.len, threadgroup(pipeline)),
            ),
            Case::Stream { .. } => (
                vec![a, b, out],
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
        let got = setup.out.read().expect("canonical output");
        let got = match case {
            Case::InnerProduct { .. } => vec![got.iter().fold(F::zero(), |x, y| x + *y)],
            _ => got.to_vec(),
        };
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

    pub fn main() {
        let device = Device::system_default().expect("a supported Metal device");
        let u32_header = FIELD_HEADERS
            .iter()
            .find(|(name, _)| *name == "jolt/field/fp128.h")
            .expect("fp128.h is a field header")
            .1;
        let variants = [
            ("u32", u32_header.to_owned()),
            ("u64", FP128_U64.to_owned()),
            ("u32/sq=mul", square_as_mul(u32_header)),
            ("u64/sq=mul", square_as_mul(FP128_U64)),
            ("u32/sq=loop", sqr_wide_loop(u32_header)),
        ]
        .map(|(name, header)| Variant {
            name,
            library: library(&device, &header),
        });

        println!("# Fp128 limb-layout A/B\n");
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
            "- commit: {}",
            command("git", &["rev-parse", "--short", "HEAD"])
        );
        println!("- rounds {ROUNDS}, {REPS} dispatches per sample, alternating order\n");

        let cases = [
            Case::Chain {
                kernel: ADD_CHAIN,
                chains: 1,
            },
            Case::Chain {
                kernel: MUL_CHAIN,
                chains: 1,
            },
            Case::Chain {
                kernel: MUL_CHAIN4,
                chains: 4,
            },
            Case::Chain {
                kernel: SQUARE_CHAIN,
                chains: 1,
            },
            Case::Stream { log: 20 },
            Case::Stream { log: 24 },
            Case::InnerProduct { log: 20 },
            Case::InnerProduct { log: 24 },
        ];
        println!("| case | variant | max threads/group | median | p10–p90 | Gop/s or Gelem/s | ratio vs u32 (p10–p90) |");
        println!("|---|---|---|---|---|---|---|");
        for case in cases {
            let mut setup = setup(&device, case);
            for variant in &variants {
                check(&device, variant, case, &mut setup);
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
                            .map(|_| run(&device, &variants[v], case, &setup).as_secs_f64())
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
                    "| {} | {} | {} | {:.1} µs | {:.1}–{:.1} µs | {:.2} | {:.3} ({:.3}–{:.3}) |",
                    case.name(),
                    variant.name,
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
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    metal::main();
}
