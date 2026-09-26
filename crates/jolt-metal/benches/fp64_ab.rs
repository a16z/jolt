//! Fp64 and Ext2 arithmetic A/B (evidence for the Fp64 and Ext2 step of
//! `specs/jolt-metal-field.md`; this branch is not for merge).
//!
//! Each variant is the merged `fp64.h` and `ext2.h` with exactly one function
//! replaced, so a comparison measures that function alone. All cases run over
//! `Prime64Offset59` and its `Ext2`, through the kernels of
//! `benches/shaders/field_bench.metal`.
//!
//! - Fp64 product (`mul_wide`): `cross`, the merged four-product form that
//!   sums the cross products first; `native`, MSL's 64-bit `*` and `mulhi`;
//!   `rows`, the row-by-row schoolbook of `fp128.h`.
//! - Fp64 square (`sqr_wide`): `sqr3`, the merged three-product form;
//!   `mul`, `mul_wide(a, a)`; `native`, `a * a` and `mulhi(a, a)`.
//! - Ext2 multiply: `karatsuba`, the merged generic form (three base
//!   multiplies); `schoolbook` (four); `lazy`, four unreduced 128-bit
//!   products summed per coefficient and reduced once each, valid for
//!   `C < 2^31` only.
//! - Ext2 square: `generic`, the merged form (two base multiplies);
//!   `lazy`, `c0^2 + 2 c1^2` summed unreduced and reduced once.
//!
//! Method, as for the earlier A/Bs: every variant's output is checked against
//! the CPU first. Each round then times every variant on a case, in
//! alternating order, as the median GPU time of `REPS` dispatches. The report
//! gives the median over rounds, its p10–p90 spread, and the median and
//! p10–p90 of the per-round ratio against the case's first variant (the
//! merged code).
//!
//! Decision rule, fixed before the run, for each comparison: the variant with
//! the lowest time on the dependent chain wins if it beats every other
//! variant by at least 3% there and is at most 3% slower than the best on
//! the four-chain case and on the inner product at 2^20. Otherwise the
//! simplest variant within 3% of the best on the chain wins, in the order
//! listed below for each comparison (first is simplest):
//! - product: `native`, `cross`, `rows`;
//! - square: `mul`, `native`, `sqr3`;
//! - Ext2 multiply: `karatsuba`, `schoolbook`, `lazy` (generic over the base
//!   field before Fp64-specific);
//! - Ext2 square: `generic`, `lazy`.
//!
//! Rounds. Round 1 ran on the `cross` product and chose `rows`, which the
//! branch then merged; round 2 reran every comparison on it and chose `mul`
//! for the square and `lazy` for both Ext2 operations. Round 3 runs the
//! Ext2 comparisons on the `mul` square and adds, under the same rule:
//! - Fp64 square: `rows3`, the row-by-row schoolbook with the cross product
//!   computed once (order: `mul`, `sqr3`, `rows3`);
//! - Ext2 multiply and square: `dot2`, a base-field `dot2(x0, y0, x1, y1)`
//!   (two products summed unreduced, reduced once, for any `C < 2^32`)
//!   with `c0 = dot2(a0, b0, 2 a1, b1)`, `c1 = dot2(a0, b1, a1, b0)` and
//!   `c0 = dot2(a0, a0, 2 a1, a1)` for the square (order: `karatsuba`,
//!   `schoolbook`, `dot2`, `lazy`; `generic`, `dot2`, `lazy`).

#[cfg(target_os = "macos")]
#[expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "a benchmark aborts on any failure and prints its report"
)]
mod metal {
    use std::process::Command;
    use std::time::Duration;

    use jolt_field::solinas::{Ext2, Prime64Offset59};
    use jolt_field::Ring;
    use jolt_metal::runtime::{
        host_name, Batch, Binding, Device, DeviceBuffer, Grid, LibrarySpec, Pipeline, ShaderLibrary,
    };
    use jolt_metal::shaders::FIELD_HEADERS;
    use jolt_metal::MetalField;
    use rayon::prelude::*;

    type F = Prime64Offset59;
    type E = Ext2<F>;

    const FIELD_BENCH: &str = include_str!("shaders/field_bench.metal");
    const MUL_CHAIN: &str = "jolt_bench_field_mul_chain";
    const MUL_CHAIN4: &str = "jolt_bench_field_mul_chain4";
    const SQUARE_CHAIN: &str = "jolt_bench_field_square_chain";
    const INNER_PRODUCT: &str = "jolt_bench_field_inner_product";
    const KERNELS: [&str; 4] = [MUL_CHAIN, MUL_CHAIN4, SQUARE_CHAIN, INNER_PRODUCT];

    const INNER_PRODUCT_GROUP: usize = 256;
    const INNER_PRODUCT_GROUPS: usize = 1024;
    const THREADS: usize = 1 << 20;
    const CHAIN_ROUNDS: u32 = 256;
    const STEP: u64 = 0x9E37_79B9_7F4A_7C15;

    const ROUNDS: usize = 31;
    const REPS: usize = 5;

    /// A replacement of one function: the header, the function's first line
    /// in the merged header, and the replacement text for the whole function
    /// (through its closing brace at column 0 or 4).
    type Patch = (&'static str, &'static str, &'static str);

    const ROWS: Patch = (
        "jolt/field/fp64.h",
        "inline Wide mul_wide(ulong a, ulong b) {",
        "",
    );
    const NATIVE: Patch = (
        "jolt/field/fp64.h",
        "inline Wide mul_wide(ulong a, ulong b) {",
        "inline Wide mul_wide(ulong a, ulong b) {\n    return Wide{a * b, metal::mulhi(a, b)};\n}",
    );
    const CROSS: Patch = (
        "jolt/field/fp64.h",
        "inline Wide mul_wide(ulong a, ulong b) {",
        "inline Wide mul_wide(ulong a, ulong b) {
    uint a0 = uint(a), a1 = uint(a >> 32);
    uint b0 = uint(b), b1 = uint(b >> 32);
    ulong p00 = ulong(a0) * b0;
    ulong p01 = ulong(a0) * b1;
    ulong p10 = ulong(a1) * b0;
    ulong p11 = ulong(a1) * b1;
    ulong mid = p01 + p10;
    ulong mid_carry = mid < p01 ? 1ul << 32 : 0ul;
    ulong lo = p00 + (mid << 32);
    ulong hi = p11 + (mid >> 32) + mid_carry + (lo < p00 ? 1ul : 0ul);
    return Wide{lo, hi};
}",
    );

    const SQR3: Patch = ("jolt/field/fp64.h", "inline Wide sqr_wide(ulong a) {", "");
    const SQR_MUL: Patch = (
        "jolt/field/fp64.h",
        "inline Wide sqr_wide(ulong a) {",
        "inline Wide sqr_wide(ulong a) {\n    return mul_wide(a, a);\n}",
    );
    const SQR_ROWS3: Patch = (
        "jolt/field/fp64.h",
        "inline Wide sqr_wide(ulong a) {",
        "inline Wide sqr_wide(ulong a) {
    uint a0 = uint(a), a1 = uint(a >> 32);
    ulong m = ulong(a0) * a1;
    ulong t = ulong(a0) * a0;
    uint w0 = uint(t);
    t = m + (t >> 32);
    uint w1 = uint(t);
    uint w2 = uint(t >> 32);
    t = m + w1;
    w1 = uint(t);
    t = ulong(a1) * a1 + w2 + (t >> 32);
    return Wide{(ulong(w1) << 32) | w0, t};
}",
    );
    const KARATSUBA: Patch = (
        "jolt/field/ext2.h",
        "    friend Ext2 operator*(Ext2 a, Ext2 b) {",
        "",
    );
    const SCHOOLBOOK: Patch = (
        "jolt/field/ext2.h",
        "    friend Ext2 operator*(Ext2 a, Ext2 b) {",
        "    friend Ext2 operator*(Ext2 a, Ext2 b) {
        return Ext2{a.c0 * b.c0 + mul_non_residue(a.c1 * b.c1), a.c0 * b.c1 + a.c1 * b.c0};
    }",
    );
    const LAZY: Patch = (
        "jolt/field/ext2.h",
        "    friend Ext2 operator*(Ext2 a, Ext2 b) {",
        "    friend Ext2 operator*(Ext2 a, Ext2 b) {\n        return ext2_ab_mul(a, b);\n    }",
    );
    const DOT2: Patch = (
        "jolt/field/ext2.h",
        "    friend Ext2 operator*(Ext2 a, Ext2 b) {",
        "    friend Ext2 operator*(Ext2 a, Ext2 b) {
        return Ext2{dot2(a.c0, b.c0, mul_non_residue(a.c1), b.c1), dot2(a.c0, b.c1, a.c1, b.c0)};
    }",
    );
    const SQUARE_DOT2: Patch = (
        "jolt/field/ext2.h",
        "    friend Ext2 square(Ext2 a) {",
        "    friend Ext2 square(Ext2 a) {
        return Ext2{dot2(a.c0, a.c0, mul_non_residue(a.c1), a.c1), (a.c0 + a.c0) * a.c1};
    }",
    );
    const SQUARE_GENERIC: Patch = ("jolt/field/ext2.h", "    friend Ext2 square(Ext2 a) {", "");
    const SQUARE_LAZY: Patch = (
        "jolt/field/ext2.h",
        "    friend Ext2 square(Ext2 a) {",
        "    friend Ext2 square(Ext2 a) {\n        return ext2_ab_square(a);\n    }",
    );

    /// The lazy Ext2 forms, found by argument-dependent lookup when `Ext2` is
    /// instantiated. A 130-bit value `lo + hi 2^64 + top 2^128`
    /// (`top <= 2`) first-folds to `t + t2 2^64` with `t2 <= 3C`, and
    /// `C t2 <= 3 C^2` wraps `t` at most once when `C < 2^31`.
    const LAZY_FORMS: &str = "
namespace jolt {
namespace ext2_ab {
struct Wide3 {
    ulong lo;
    ulong hi;
    uint top;
};
inline Wide3 add(Wide3 x, fp64_detail::Wide y) {
    ulong lo = x.lo + y.lo;
    ulong c0 = lo < y.lo ? 1ul : 0ul;
    ulong hi = x.hi + y.hi;
    uint c1 = hi < y.hi ? 1u : 0u;
    ulong hi2 = hi + c0;
    c1 += hi2 < hi ? 1u : 0u;
    return Wide3{lo, hi2, x.top + c1};
}
template <uint C>
inline ulong reduce3(Wide3 x) {
    static_assert(C < (1u << 31), \"the lazy forms need C < 2^31\");
    ulong u = ulong(uint(x.hi)) * C + uint(x.lo);
    uint t0 = uint(u);
    u = ulong(uint(x.hi >> 32)) * C + uint(x.lo >> 32) + (u >> 32);
    ulong t = (u << 32) | t0;
    ulong t2 = (u >> 32) + ulong(x.top) * C;
    ulong s = t + t2 * C;
    bool overflow = s < t;
    ulong r = s + C;
    bool carry = r < s;
    return (overflow || carry) ? r : s;
}
} // namespace ext2_ab

template <uint C>
Ext2<Fp64<C>> ext2_ab_mul(Ext2<Fp64<C>> a, Ext2<Fp64<C>> b) {
    using namespace fp64_detail;
    Wide p00 = mul_wide(a.c0.word, b.c0.word);
    Wide p11 = mul_wide(a.c1.word, b.c1.word);
    Wide p01 = mul_wide(a.c0.word, b.c1.word);
    Wide p10 = mul_wide(a.c1.word, b.c0.word);
    ext2_ab::Wide3 c0 = ext2_ab::add(ext2_ab::add(ext2_ab::Wide3{p00.lo, p00.hi, 0u}, p11), p11);
    ext2_ab::Wide3 c1 = ext2_ab::add(ext2_ab::Wide3{p01.lo, p01.hi, 0u}, p10);
    return Ext2<Fp64<C>>{Fp64<C>{ext2_ab::reduce3<C>(c0)}, Fp64<C>{ext2_ab::reduce3<C>(c1)}};
}

template <uint C>
Ext2<Fp64<C>> ext2_ab_square(Ext2<Fp64<C>> a) {
    using namespace fp64_detail;
    Wide p00 = sqr_wide(a.c0.word);
    Wide p11 = sqr_wide(a.c1.word);
    ext2_ab::Wide3 c0 = ext2_ab::add(ext2_ab::add(ext2_ab::Wide3{p00.lo, p00.hi, 0u}, p11), p11);
    return Ext2<Fp64<C>>{Fp64<C>{ext2_ab::reduce3<C>(c0)}, (a.c0 + a.c0) * a.c1};
}

// x0 y0 + x1 y1, reduced once. The two products sum to below 2^129, so a
// carry bit top joins the first fold: t2 <= C + C top <= 2C, and
// fold2_canonicalize needs C (t2 + 1) <= p, which 2C + 1 < 2^32 gives
// whenever C < 2^31. Larger offsets reduce each product.
template <typename F>
F dot2(F x0, F y0, F x1, F y1) {
    return x0 * y0 + x1 * y1;
}

template <uint C>
Fp64<C> dot2(Fp64<C> x0, Fp64<C> y0, Fp64<C> x1, Fp64<C> y1) {
    using namespace fp64_detail;
    if constexpr (C < (1u << 31)) {
        Wide p = mul_wide(x0.word, y0.word);
        Wide q = mul_wide(x1.word, y1.word);
        ulong lo = p.lo + q.lo;
        ulong h = p.hi + q.hi;
        ulong top = h < p.hi ? 1ul : 0ul;
        ulong hi = h + (lo < q.lo ? 1ul : 0ul);
        top += hi < h ? 1ul : 0ul;
        ulong u = ulong(uint(hi)) * C + uint(lo);
        uint t0 = uint(u);
        u = ulong(uint(hi >> 32)) * C + uint(lo >> 32) + (u >> 32);
        ulong t = (u << 32) | t0;
        ulong t2 = (u >> 32) + top * C;
        ulong s = t + t2 * C;
        bool overflow = s < t;
        ulong r = s + C;
        bool carry = r < s;
        return Fp64<C>{(overflow || carry) ? r : s};
    } else {
        return x0 * y0 + x1 * y1;
    }
}
} // namespace jolt
";

    /// `text` with the function starting at `first_line` replaced by
    /// `replacement`; an empty replacement keeps it.
    fn patch(text: &str, first_line: &str, replacement: &str) -> String {
        let start = text
            .find(first_line)
            .expect("patched function is in the header");
        if replacement.is_empty() {
            return text.to_owned();
        }
        let indent = &first_line[..first_line.len() - first_line.trim_start().len()];
        let close = format!("\n{indent}}}\n");
        let end = start + text[start..].find(&close).expect("function closes") + close.len() - 1;
        format!("{}{replacement}{}", &text[..start], &text[end..])
    }

    struct Variant {
        name: &'static str,
        library: ShaderLibrary,
    }

    fn variant<T: MetalField>(device: &Device, name: &'static str, patches: &[Patch]) -> Variant {
        let spec = FIELD_HEADERS
            .iter()
            .fold(LibrarySpec::new(), |spec, (header, text)| {
                let text = patches
                    .iter()
                    .filter(|(target, _, _)| target == header)
                    .fold((*text).to_owned(), |text, (_, first, replacement)| {
                        patch(&text, first, replacement)
                    });
                let spec = spec.source(header, &text);
                if *header == "jolt/field/ext2.h" {
                    spec.source("lazy_forms.h", LAZY_FORMS)
                } else {
                    spec
                }
            });
        let spec = spec.source("field_bench.metal", FIELD_BENCH);
        let spec = KERNELS
            .iter()
            .fold(spec, |spec, kernel| spec.instantiate::<T>(kernel));
        Variant {
            name,
            library: ShaderLibrary::compile(device, &spec).expect("variant compiles"),
        }
    }

    fn pipeline<'l, T: MetalField>(variant: &'l Variant, kernel: &str) -> &'l Pipeline {
        variant
            .library
            .pipeline(&host_name::<T>(kernel))
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

    trait Sample: MetalField + Send + Sync {
        fn sample(seed: u64, len: usize) -> Vec<Self>;
    }

    impl Sample for F {
        fn sample(seed: u64, len: usize) -> Vec<Self> {
            words(seed, len).into_iter().map(F::from_u64).collect()
        }
    }

    impl Sample for E {
        fn sample(seed: u64, len: usize) -> Vec<Self> {
            words(seed, 2 * len)
                .chunks_exact(2)
                .map(|w| E::new(F::from_u64(w[0]), F::from_u64(w[1])))
                .collect()
        }
    }

    #[derive(Clone, Copy)]
    enum Case {
        Chain,
        Chain4,
        Square,
        InnerProduct { log: u32 },
    }

    impl Case {
        fn name(self) -> String {
            match self {
                Self::Chain => "mul chain".to_owned(),
                Self::Chain4 => "mul, 4 chains".to_owned(),
                Self::Square => "square chain".to_owned(),
                Self::InnerProduct { log } => format!("inner product 2^{log}"),
            }
        }

        fn kernel(self) -> &'static str {
            match self {
                Self::Chain => MUL_CHAIN,
                Self::Chain4 => MUL_CHAIN4,
                Self::Square => SQUARE_CHAIN,
                Self::InnerProduct { .. } => INNER_PRODUCT,
            }
        }

        /// Operations per dispatch.
        fn work(self) -> u64 {
            let chain = THREADS as u64 * u64::from(CHAIN_ROUNDS);
            match self {
                Self::Chain | Self::Square => chain,
                Self::Chain4 => 4 * chain,
                Self::InnerProduct { log } => 1 << log,
            }
        }
    }

    struct Setup<T: Sample> {
        a: DeviceBuffer<T>,
        b: DeviceBuffer<T>,
        out: DeviceBuffer<T>,
        len: usize,
        expected: Vec<T>,
    }

    fn setup<T: Sample>(device: &Device, case: Case) -> Setup<T> {
        let len = match case {
            Case::InnerProduct { log } => 1 << log,
            _ => THREADS,
        };
        let a = T::sample(1, len);
        let b = T::sample(2, len);
        let chain = |x: T, y: T| (0..CHAIN_ROUNDS).fold(x, |x, _| x * y);
        let expected: Vec<T> = match case {
            Case::Chain => a.par_iter().zip(&b).map(|(&x, &y)| chain(x, y)).collect(),
            Case::Chain4 => a
                .par_iter()
                .zip(&b)
                .map(|(&x, &y)| {
                    let (x1, x2, x3) = (x + y, x + y + y, x + y + y + y);
                    (chain(x, y) + chain(x1, y)) + (chain(x2, y) + chain(x3, y))
                })
                .collect(),
            Case::Square => a
                .par_iter()
                .map(|&x| (0..CHAIN_ROUNDS).fold(x, |x, _| x.square()))
                .collect(),
            Case::InnerProduct { .. } => vec![a
                .par_iter()
                .zip(&b)
                .map(|(x, y)| *x * *y)
                .reduce(T::zero, |x, y| x + y)],
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

    fn run<T: Sample>(
        device: &Device,
        variant: &Variant,
        case: Case,
        setup: &Setup<T>,
    ) -> Duration {
        let pipeline = pipeline::<T>(variant, case.kernel());
        let n = u32::try_from(setup.len).expect("sizes fit u32");
        let rounds = CHAIN_ROUNDS;
        let (a, b, out) = (
            Binding::buffer(&setup.a),
            Binding::buffer(&setup.b),
            Binding::buffer(&setup.out),
        );
        let linear = Grid::linear(setup.len, threadgroup(pipeline));
        let (bindings, grid) = match case {
            Case::Chain | Case::Chain4 => (vec![a, b, Binding::value(&rounds), out], linear),
            Case::Square => (vec![a, Binding::value(&rounds), out], linear),
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

    fn check<T: Sample>(device: &Device, variant: &Variant, case: Case, setup: &mut Setup<T>) {
        let _ = run(device, variant, case, setup);
        let out = setup.out.read().expect("canonical output").to_vec();
        let got = match case {
            Case::InnerProduct { .. } => vec![out.into_iter().fold(T::zero(), |x, y| x + y)],
            _ => out,
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

    fn compare<T: Sample>(device: &Device, label: &str, case: Case, variants: &[Variant]) {
        let mut setup = setup::<T>(device, case);
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
                "| {label} | {} | {} | {} | {:.1} µs | {:.1}–{:.1} µs | {:.2} | {:.3} ({:.3}–{:.3}) |",
                case.name(),
                variant.name,
                pipeline::<T>(variant, case.kernel()).max_total_threads_per_threadgroup(),
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

    const CASES: [Case; 4] = [
        Case::Chain,
        Case::Chain4,
        Case::InnerProduct { log: 20 },
        Case::InnerProduct { log: 24 },
    ];

    pub fn main() {
        let device = Device::system_default().expect("a supported Metal device");

        println!("# Fp64 and Ext2 A/B\n");
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
        println!("| comparison | case | variant | max threads/group | median | p10–p90 | G op/s | ratio vs first (p10–p90) |");
        println!("|---|---|---|---|---|---|---|---|");

        let product = [
            variant::<F>(&device, "rows", &[ROWS]),
            variant::<F>(&device, "cross", &[CROSS]),
            variant::<F>(&device, "native", &[NATIVE]),
        ];
        for case in CASES {
            compare::<F>(&device, "Fp64 product", case, &product);
        }
        let square = [
            variant::<F>(&device, "mul", &[SQR_MUL]),
            variant::<F>(&device, "sqr3", &[SQR3]),
            variant::<F>(&device, "rows3", &[SQR_ROWS3]),
        ];
        compare::<F>(&device, "Fp64 square", Case::Square, &square);

        let ext_mul = [
            variant::<E>(&device, "karatsuba", &[SQR_MUL, KARATSUBA]),
            variant::<E>(&device, "schoolbook", &[SQR_MUL, SCHOOLBOOK]),
            variant::<E>(&device, "dot2", &[SQR_MUL, DOT2]),
            variant::<E>(&device, "lazy", &[SQR_MUL, LAZY]),
        ];
        for case in CASES {
            compare::<E>(&device, "Ext2 multiply", case, &ext_mul);
        }
        let ext_square = [
            variant::<E>(&device, "generic", &[SQR_MUL, SQUARE_GENERIC]),
            variant::<E>(&device, "dot2", &[SQR_MUL, SQUARE_DOT2]),
            variant::<E>(&device, "lazy", &[SQR_MUL, SQUARE_LAZY]),
        ];
        compare::<E>(&device, "Ext2 square", Case::Square, &ext_square);

        println!(
            "\n- load average after: {}",
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
