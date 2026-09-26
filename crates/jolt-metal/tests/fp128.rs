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

#[path = "support/field.rs"]
mod field;
#[path = "support/ops.rs"]
mod ops;
mod support;

mod gpu {
    use jolt_field::solinas::{Prime128Offset275, Prime128OffsetA7F7};
    use jolt_metal::ErrorClass;

    use super::field::{edges, element, modulus, random_elements, TestField};
    use super::ops::{check_ops, library, run, Inputs, I64_EDGES, U64_EDGES, WRITE_NON_CANONICAL};
    use super::support::{gpu, SplitMix64};

    /// Random inputs per operation.
    const RANDOM: usize = 1 << 20;

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

    fn conformance<F: TestField>(test: &'static str, seed: u64) {
        let (_gpu, device) = gpu(test);
        let library = library::<F>(&device, &[], &[]);
        let mut words = SplitMix64(seed);
        let p = modulus::<F>();

        // Binary operations: every pair of edges, the fold windows, random.
        let edges = edges::<F>();
        let mut pairs: Vec<(u128, u128)> = edges
            .iter()
            .flat_map(|&a| edges.iter().map(move |&b| (a, b)))
            .collect();
        pairs.extend(mul_windows::<F>());
        let random = random_elements::<F>(&mut words, 2 * RANDOM);
        pairs.extend(random.chunks_exact(2).map(|pair| (pair[0], pair[1])));
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

        // Unary operations: every edge, the window operands, random.
        let mut singles = edges.clone();
        singles.extend(mul_windows::<F>().into_iter().map(|(_, b)| b));
        singles.extend(random_elements::<F>(&mut words, RANDOM));

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

        let inputs = Inputs::<F> {
            pairs: pairs
                .iter()
                .map(|&(a, b)| (element(a), element(b)))
                .collect(),
            singles: singles.iter().map(|&v| element(v)).collect(),
            u64_pairs: u64_pairs.iter().map(|&(a, s)| (element(a), s)).collect(),
            i64_pairs: i64_pairs.iter().map(|&(a, s)| (element(a), s)).collect(),
        };
        let mut failures = Vec::new();
        check_ops(&device, &library, &inputs, &mut failures);
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
        let library = library::<Prime128OffsetA7F7>(&device, &[], &[]);
        let error = run::<Prime128OffsetA7F7>(&device, &library, WRITE_NON_CANONICAL, vec![], 3)
            .unwrap_err();
        assert_eq!(error.class(), ErrorClass::Fault, "{error}");
        assert!(error.to_string().contains("element 0"), "{error}");
    }
}
