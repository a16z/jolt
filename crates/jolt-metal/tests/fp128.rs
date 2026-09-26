//! Conformance of `jolt::Fp128<C>` with `jolt_field::solinas::Fp128<P>`.
//!
//! Every MSL operation runs through a generic test kernel and must match the
//! CPU result byte for byte, on fixed edge vectors, on inputs constructed to
//! reach each branch of the reductions, and on 2^20 fixed-seed random inputs.
//! The branch coverage is asserted, not assumed: the test recomputes each
//! reduction's intermediate values in `u128` arithmetic and requires every
//! branch to be taken.

#![cfg(target_os = "macos")]
#![expect(clippy::unwrap_used, reason = "tests may panic on assertion failures")]

mod support;

mod gpu {
    use std::fmt::Debug;

    use jolt_field::solinas::{Prime128Offset275, Prime128OffsetA7F7};
    use jolt_field::{CanonicalEncoding, PseudoMersenne};
    use jolt_metal::runtime::{
        host_name, Batch, Binding, Device, DeviceBuffer, Grid, LibrarySpec, Pipeline, ShaderLibrary,
    };
    use jolt_metal::shaders::FIELD_HEADERS;
    use jolt_metal::{ErrorClass, MetalError, MetalField};

    use super::support::{gpu, SplitMix64};

    const FIELD_OPS: &str = include_str!("shaders/field_ops.metal");
    const ADD: &str = "jolt_test_field_add";
    const SUB: &str = "jolt_test_field_sub";
    const MUL: &str = "jolt_test_field_mul";
    const NEG: &str = "jolt_test_field_neg";
    const SQUARE: &str = "jolt_test_field_square";
    const MUL_U64: &str = "jolt_test_field_mul_u64";
    const MUL_I64: &str = "jolt_test_field_mul_i64";
    const FROM_U64: &str = "jolt_test_field_from_u64";
    const FROM_I64: &str = "jolt_test_field_from_i64";
    const WRITE_NON_CANONICAL: &str = "jolt_test_field_write_non_canonical";
    const KERNELS: [&str; 10] = [
        ADD,
        SUB,
        MUL,
        NEG,
        SQUARE,
        MUL_U64,
        MUL_I64,
        FROM_U64,
        FROM_I64,
        WRITE_NON_CANONICAL,
    ];

    type BinaryOp<F> = fn(F, F) -> F;
    type UnaryOp<F> = fn(F) -> F;

    /// Random inputs per operation.
    const RANDOM: usize = 1 << 20;

    /// The field types under test, with the CPU operations the harness needs.
    trait TestField: MetalField + PseudoMersenne + CanonicalEncoding + Debug {}
    impl<F: MetalField + PseudoMersenne + CanonicalEncoding + Debug> TestField for F {}

    fn library<F: TestField>(device: &Device) -> ShaderLibrary {
        let spec = FIELD_HEADERS
            .iter()
            .fold(LibrarySpec::new(), |spec, (name, text)| {
                spec.source(name, text)
            })
            .source("field_ops.metal", FIELD_OPS);
        let spec = KERNELS
            .iter()
            .fold(spec, |spec, kernel| spec.instantiate::<F>(kernel));
        ShaderLibrary::compile(device, &spec).unwrap()
    }

    fn element<F: TestField>(value: u128) -> F {
        F::from_u128_checked(value).unwrap()
    }

    fn modulus<F: TestField>() -> u128 {
        0u128.wrapping_sub(F::OFFSET)
    }

    /// Runs `kernel` over `len` threads with `inputs` bound first and a fresh
    /// output buffer last, and returns the checked output.
    fn run<F: TestField>(
        device: &Device,
        library: &ShaderLibrary,
        kernel: &str,
        inputs: Vec<Binding<'_>>,
        len: usize,
    ) -> Result<Vec<F>, MetalError> {
        let pipeline: &Pipeline = library.pipeline(&host_name::<F>(kernel)).unwrap();
        let mut out = DeviceBuffer::<F>::zeroed(device, len).unwrap();
        {
            let mut bindings = inputs;
            bindings.push(Binding::buffer(&out));
            let group = (pipeline.thread_execution_width() * 8)
                .min(pipeline.max_total_threads_per_threadgroup());
            let mut batch = Batch::new(device).unwrap();
            batch
                .dispatch(pipeline, &bindings, Grid::linear(len, group))
                .unwrap();
            let _ = batch.commit_and_wait().unwrap();
        }
        out.read().map(<[F]>::to_vec)
    }

    /// Records the first mismatch and the mismatch count for one operation.
    fn compare<F: TestField>(
        failures: &mut Vec<String>,
        op: &str,
        got: &[F],
        want: &[F],
        input: impl Fn(usize) -> String,
    ) {
        assert_eq!(got.len(), want.len());
        let mut wrong = got
            .iter()
            .zip(want)
            .enumerate()
            .filter(|(_, (g, w))| g != w);
        if let Some((index, (g, w))) = wrong.next() {
            failures.push(format!(
                "{op}: {} of {} results differ; first at {index}: {} gave {g:?}, expected {w:?}",
                wrong.count() + 1,
                got.len(),
                input(index),
            ));
        }
    }

    /// Canonical values at every boundary the arithmetic treats specially:
    /// small values, `C` and its neighbours, word and limb boundaries, the
    /// top of the field, and every value whose 32-bit words are each one of
    /// `0`, `1`, `2^31`, `2^32 − 1`.
    fn edges<F: TestField>() -> Vec<u128> {
        let p = modulus::<F>();
        let c = F::OFFSET;
        let mut values = vec![
            0,
            1,
            2,
            3,
            c - 1,
            c,
            c + 1,
            (1 << 32) - 1,
            1 << 32,
            (1 << 63) - 1,
            1 << 63,
            (1 << 64) - 1,
            1 << 64,
            (1 << 96) - 1,
            1 << 96,
            (1 << 127) - 1,
            1 << 127,
            p / 2,
            p / 2 + 1,
            p - c,
            p - 2,
            p - 1,
        ];
        let words = [0u128, 1, 1 << 31, (1 << 32) - 1];
        for pattern in 0..256u32 {
            let value = (0..4).fold(0u128, |value, i| {
                value | words[((pattern >> (2 * i)) & 3) as usize] << (32 * i)
            });
            values.push(value);
        }
        values.retain(|&v| v < p);
        values.sort_unstable();
        values.dedup();
        values
    }

    fn random_elements<F: TestField>(words: &mut SplitMix64, len: usize) -> Vec<u128> {
        let p = modulus::<F>();
        let mut values = Vec::with_capacity(len);
        while values.len() < len {
            let v = u128::from(words.next().unwrap()) | u128::from(words.next().unwrap()) << 64;
            if v < p {
                values.push(v);
            }
        }
        values
    }

    /// `(hi, lo)` of the 256-bit product `a · b`.
    fn mul_256(a: u128, b: u128) -> (u128, u128) {
        let (a0, a1) = (a & u128::from(u64::MAX), a >> 64);
        let (b0, b1) = (b & u128::from(u64::MAX), b >> 64);
        let (p00, p01, p10, p11) = (a0 * b0, a0 * b1, a1 * b0, a1 * b1);
        let mid = (p00 >> 64) + (p01 & u128::from(u64::MAX)) + (p10 & u128::from(u64::MAX));
        let lo = (p00 & u128::from(u64::MAX)) | mid << 64;
        let hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
        (hi, lo)
    }

    /// Which branch `fold2_canonicalize(t, t2)` takes.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Fold2 {
        /// `t + C·t2 < p`: the sum is already canonical.
        Plain,
        /// `p ≤ t + C·t2 < 2^128`: one conditional add of `C`.
        Canonicalize,
        /// `t + C·t2 ≥ 2^128`: the wrap is corrected by adding `C`.
        Overflow,
    }

    fn fold2<F: TestField>(t: u128, t2: u128) -> Fold2 {
        let (ct2_hi, ct2_lo) = mul_256(F::OFFSET, t2);
        let (s, carry) = t.overflowing_add(ct2_lo);
        if carry || ct2_hi != 0 {
            Fold2::Overflow
        } else if s >= modulus::<F>() {
            Fold2::Canonicalize
        } else {
            Fold2::Plain
        }
    }

    /// The fold-2 branch of `a · b`: the first fold `lo + C·hi` is split
    /// into `t` and `t2 ≤ C`.
    fn mul_branch<F: TestField>(a: u128, b: u128) -> Fold2 {
        let (hi, lo) = mul_256(a, b);
        let (c_hi_hi, c_hi_lo) = mul_256(F::OFFSET, hi);
        let (t, carry) = lo.overflowing_add(c_hi_lo);
        fold2::<F>(t, c_hi_hi + u128::from(carry))
    }

    /// The fold-2 branch of `a · s` for a 64-bit `s`: `t2` is the high part.
    fn mul_u64_branch<F: TestField>(a: u128, s: u64) -> Fold2 {
        let (hi, lo) = mul_256(a, u128::from(s));
        fold2::<F>(lo, hi)
    }

    /// `⌊((k + 1) · 2^128 − 1) / d⌋`, the largest `m` with `m · d` below
    /// `(k + 1) · 2^128`.
    fn window(k: u128, d: u128) -> u128 {
        let (q, r) = (u128::MAX / d, u128::MAX % d);
        (k + 1) * q + ((k + 1) * r + k) / d
    }

    /// Pairs whose product reaches the rare fold-2 branches.
    ///
    /// With `a = 2^127` and `b = 2m`, the product is `m · 2^128`, so the first
    /// fold gives `C·m`. Taking `m = window(k, C)` puts `C·m` in
    /// `[(k + 1) 2^128 − C, (k + 1) 2^128)`: for `k = 0` that is `[p, 2^128)`
    /// (canonicalize), and for `k ≥ 1` it is `t2 = k` with
    /// `t ≥ 2^128 − C·k` (overflow).
    fn mul_windows<F: TestField>() -> Vec<(u128, u128)> {
        let p = modulus::<F>();
        [0, 1, 2, 3]
            .into_iter()
            .map(|k| 2 * window(k, F::OFFSET))
            .filter(|&b| b < p)
            .flat_map(|b| [(1 << 127, b), (b, 1 << 127)])
            .collect()
    }

    /// Scalar products that reach the rare fold-2 branches of `mul_u64`:
    /// `a = window(k, s)` puts `a · s` in `[(k + 1) 2^128 − s, (k + 1) 2^128)`,
    /// which is inside the branch's window when `s ≤ C · max(k, 1)`.
    fn mul_u64_windows<F: TestField>() -> Vec<(u128, u64)> {
        let p = modulus::<F>();
        let c = u64::try_from(F::OFFSET).unwrap();
        [(0, 1), (0, 3), (0, c), (1, c), (2, 2 * c), (7, 5 * c)]
            .into_iter()
            .map(|(k, s)| (window(k, u128::from(s)), s))
            .filter(|&(a, _)| a < p)
            .collect()
    }

    const U64_EDGES: [u64; 11] = [
        0,
        1,
        2,
        3,
        (1 << 31) - 1,
        (1 << 32) - 1,
        1 << 32,
        (1 << 63) - 1,
        1 << 63,
        u64::MAX - 1,
        u64::MAX,
    ];

    const I64_EDGES: [i64; 13] = [
        0,
        1,
        -1,
        2,
        -2,
        i32::MAX as i64,
        i32::MIN as i64,
        (1 << 32) - 1,
        -(1 << 32),
        i64::MAX,
        i64::MAX - 1,
        i64::MIN,
        i64::MIN + 1,
    ];

    fn conformance<F: TestField>(test: &'static str, seed: u64) {
        let (_gpu, device) = gpu(test);
        let library = library::<F>(&device);
        let mut words = SplitMix64(seed);
        let mut failures = Vec::new();

        // Binary operations: every pair of edges, the fold windows, random.
        let edges = edges::<F>();
        let mut pairs: Vec<(u128, u128)> = edges
            .iter()
            .flat_map(|&a| edges.iter().map(move |&b| (a, b)))
            .collect();
        pairs.extend(mul_windows::<F>());
        let random = random_elements::<F>(&mut words, 2 * RANDOM);
        pairs.extend(random.chunks_exact(2).map(|pair| (pair[0], pair[1])));

        let p = modulus::<F>();
        let mul_branches: Vec<Fold2> = pairs.iter().map(|&(a, b)| mul_branch::<F>(a, b)).collect();
        for branch in [Fold2::Plain, Fold2::Canonicalize, Fold2::Overflow] {
            assert!(
                mul_branches.contains(&branch),
                "no mul input takes {branch:?}"
            );
        }
        assert!(
            pairs.iter().any(|&(a, b)| a.checked_add(b).is_none()),
            "no add wraps"
        );
        assert!(
            pairs
                .iter()
                .any(|&(a, b)| a.checked_add(b).is_some_and(|s| s >= p)),
            "no add needs canonicalization"
        );
        assert!(pairs.iter().any(|&(a, b)| a < b), "no sub borrows");

        let a: Vec<F> = pairs.iter().map(|&(a, _)| element(a)).collect();
        let b: Vec<F> = pairs.iter().map(|&(_, b)| element(b)).collect();
        let (a_dev, b_dev) = (
            DeviceBuffer::from_slice(&device, &a).unwrap(),
            DeviceBuffer::from_slice(&device, &b).unwrap(),
        );
        let describe = |i: usize| format!("({}, {})", pairs[i].0, pairs[i].1);
        let binary: [(&str, BinaryOp<F>); 3] = [
            (ADD, |x, y| x + y),
            (SUB, |x, y| x - y),
            (MUL, |x, y| x * y),
        ];
        for (kernel, op) in binary {
            let got = run::<F>(
                &device,
                &library,
                kernel,
                vec![Binding::buffer(&a_dev), Binding::buffer(&b_dev)],
                pairs.len(),
            )
            .unwrap();
            let want: Vec<F> = a.iter().zip(&b).map(|(&x, &y)| op(x, y)).collect();
            compare(&mut failures, kernel, &got, &want, describe);
        }

        // Unary operations: every edge, the window operands, random.
        let mut singles = edges.clone();
        singles.extend(mul_windows::<F>().into_iter().map(|(_, b)| b));
        singles.extend(random_elements::<F>(&mut words, RANDOM));
        let x: Vec<F> = singles.iter().map(|&v| element(v)).collect();
        let x_dev = DeviceBuffer::from_slice(&device, &x).unwrap();
        let describe = |i: usize| format!("{}", singles[i]);
        let unary: [(&str, UnaryOp<F>); 2] = [(NEG, |x| -x), (SQUARE, |x| x.square())];
        for (kernel, op) in unary {
            let got = run::<F>(
                &device,
                &library,
                kernel,
                vec![Binding::buffer(&x_dev)],
                singles.len(),
            )
            .unwrap();
            let want: Vec<F> = x.iter().map(|&v| op(v)).collect();
            compare(&mut failures, kernel, &got, &want, describe);
        }

        // Scalar products: every edge against every scalar edge, the
        // mul_u64 windows, random elements against random scalars.
        let mut u64_pairs: Vec<(u128, u64)> = edges
            .iter()
            .flat_map(|&a| U64_EDGES.iter().map(move |&s| (a, s)))
            .collect();
        u64_pairs.extend(mul_u64_windows::<F>());
        let random = random_elements::<F>(&mut words, RANDOM);
        u64_pairs.extend(random.into_iter().map(|a| (a, words.next().unwrap())));
        let u64_branches: Vec<Fold2> = u64_pairs
            .iter()
            .map(|&(a, s)| mul_u64_branch::<F>(a, s))
            .collect();
        for branch in [Fold2::Plain, Fold2::Canonicalize, Fold2::Overflow] {
            assert!(
                u64_branches.contains(&branch),
                "no mul_u64 input takes {branch:?}"
            );
        }
        let a: Vec<F> = u64_pairs.iter().map(|&(a, _)| element(a)).collect();
        let s: Vec<u64> = u64_pairs.iter().map(|&(_, s)| s).collect();
        let (a_dev, s_dev) = (
            DeviceBuffer::from_slice(&device, &a).unwrap(),
            DeviceBuffer::from_slice(&device, &s).unwrap(),
        );
        let got = run::<F>(
            &device,
            &library,
            MUL_U64,
            vec![Binding::buffer(&a_dev), Binding::buffer(&s_dev)],
            s.len(),
        )
        .unwrap();
        let want: Vec<F> = a.iter().zip(&s).map(|(x, &s)| x.mul_u64(s)).collect();
        compare(&mut failures, MUL_U64, &got, &want, |i| {
            format!("({}, {})", u64_pairs[i].0, u64_pairs[i].1)
        });
        let got = run::<F>(
            &device,
            &library,
            FROM_U64,
            vec![Binding::buffer(&s_dev)],
            s.len(),
        )
        .unwrap();
        let want: Vec<F> = s.iter().map(|&s| F::from_u64(s)).collect();
        compare(&mut failures, FROM_U64, &got, &want, |i| {
            format!("{}", s[i])
        });

        let mut i64_pairs: Vec<(u128, i64)> = edges
            .iter()
            .flat_map(|&a| I64_EDGES.iter().map(move |&s| (a, s)))
            .collect();
        let random = random_elements::<F>(&mut words, RANDOM);
        i64_pairs.extend(
            random
                .into_iter()
                .map(|a| (a, words.next().unwrap().cast_signed())),
        );
        let a: Vec<F> = i64_pairs.iter().map(|&(a, _)| element(a)).collect();
        let s: Vec<i64> = i64_pairs.iter().map(|&(_, s)| s).collect();
        let (a_dev, s_dev) = (
            DeviceBuffer::from_slice(&device, &a).unwrap(),
            DeviceBuffer::from_slice(&device, &s).unwrap(),
        );
        let got = run::<F>(
            &device,
            &library,
            MUL_I64,
            vec![Binding::buffer(&a_dev), Binding::buffer(&s_dev)],
            s.len(),
        )
        .unwrap();
        let want: Vec<F> = a.iter().zip(&s).map(|(x, &s)| x.mul_i64(s)).collect();
        compare(&mut failures, MUL_I64, &got, &want, |i| {
            format!("({}, {})", i64_pairs[i].0, i64_pairs[i].1)
        });
        let got = run::<F>(
            &device,
            &library,
            FROM_I64,
            vec![Binding::buffer(&s_dev)],
            s.len(),
        )
        .unwrap();
        let want: Vec<F> = s.iter().map(|&s| F::from_i64(s)).collect();
        compare(&mut failures, FROM_I64, &got, &want, |i| {
            format!("{}", s[i])
        });

        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[test]
    fn fp128_a7f7_matches_jolt_field() {
        conformance::<Prime128OffsetA7F7>("fp128_a7f7_matches_jolt_field", 0xf128_a7f7);
    }

    #[test]
    fn fp128_275_matches_jolt_field() {
        conformance::<Prime128Offset275>("fp128_275_matches_jolt_field", 0xf128_0113);
    }

    #[test]
    fn non_canonical_output_is_rejected_on_read_back() {
        let (_gpu, device) = gpu("non_canonical_output_is_rejected_on_read_back");
        let library = library::<Prime128OffsetA7F7>(&device);
        let error = run::<Prime128OffsetA7F7>(&device, &library, WRITE_NON_CANONICAL, vec![], 3)
            .unwrap_err();
        assert_eq!(error.class(), ErrorClass::Fault, "{error}");
        assert!(error.to_string().contains("element 0"), "{error}");
    }
}
