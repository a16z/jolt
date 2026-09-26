//! Conformance of `jolt::Fp64<C>` with `jolt_field::solinas::Fp64<P>`.
//!
//! Every MSL operation runs through a generic test kernel and must match the
//! CPU result byte for byte, on fixed edge vectors, on inputs constructed to
//! reach each branch of the reduction, and on 2^20 fixed-seed random inputs.
//! The branch coverage is asserted, not assumed: the test recomputes the
//! reduction's intermediate values in `u128` arithmetic and requires every
//! branch to be taken.
//!
//! Two moduli: `Prime64Offset59`, the base field of Akita's fp64 presets, and
//! `2^64 − 2^32 + 1`, whose offset `2^32 − 1` is the largest `jolt::Fp64`
//! accepts, so every bound the header argues is reached at its limit.

#![cfg(target_os = "macos")]
#![expect(clippy::unwrap_used, reason = "tests may panic on assertion failures")]

#[path = "support/field.rs"]
mod field;
#[path = "support/ops.rs"]
mod ops;
mod support;

mod gpu {
    use jolt_field::solinas::{Fp64, Prime64Offset59};
    use jolt_metal::ErrorClass;

    use super::field::{edges, element, modulus, random_elements, TestField};
    use super::ops::{check_ops, library, run, Inputs, I64_EDGES, U64_EDGES, WRITE_NON_CANONICAL};
    use super::support::{gpu, SplitMix64};

    /// `2^64 − 2^32 + 1`: offset `2^32 − 1`.
    type MaxOffset = Fp64<0xFFFF_FFFF_0000_0001>;

    /// Random inputs per operation.
    const RANDOM: usize = 1 << 20;

    const WORD: u128 = u64::MAX as u128;

    /// Which branch `fold2_canonicalize(t, t2)` takes.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Fold2 {
        /// `t + C·t2 < p`: the sum is already canonical.
        Plain,
        /// `p ≤ t + C·t2 < 2^64`: one conditional add of `C`.
        Canonicalize,
        /// `t + C·t2 ≥ 2^64`: the wrap is corrected by adding `C`.
        Overflow,
    }

    /// The fold-2 branch of `reduce_product(x)`: the first fold
    /// `lo + C·hi` is split into `t` and `t2 ≤ C`.
    fn reduce_branch<F: TestField>(x: u128) -> Fold2 {
        let c = F::OFFSET;
        let first = (x & WORD) + c * (x >> 64);
        let v = (first & WORD) + c * (first >> 64);
        if v > WORD {
            Fold2::Overflow
        } else if v >= modulus::<F>() {
            Fold2::Canonicalize
        } else {
            Fold2::Plain
        }
    }

    /// `⌊((k + 1) · 2^64 − 1) / d⌋`, the largest `m` with `m · d` below
    /// `(k + 1) · 2^64`.
    fn window(k: u128, d: u128) -> u128 {
        ((k + 1) << 64).div_ceil(d) - 1
    }

    /// Pairs whose product reaches the rare fold-2 branches.
    ///
    /// With `a = 2^63` and `b = 2m`, the product is `m · 2^64`, so the first
    /// fold gives `C·m`. Taking `m = window(k, C)` puts `C·m` in
    /// `[(k + 1) 2^64 − C, (k + 1) 2^64)`: for `k = 0` that is `[p, 2^64)`
    /// (canonicalize), and for `k ≥ 1` it is `t2 = k` with
    /// `t ≥ 2^64 − C·k` (overflow). `b` is below `2^64` for every `k < C/2`,
    /// so `mul_u64` reaches the branches with the same operands.
    fn windows<F: TestField>() -> Vec<(u128, u128)> {
        let p = modulus::<F>();
        [0, 1, 2, 3]
            .into_iter()
            .map(|k| 2 * window(k, F::OFFSET))
            .filter(|&b| b < p)
            .flat_map(|b| [(1 << 63, b), (b, 1 << 63)])
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
        pairs.extend(windows::<F>());
        let random = random_elements::<F>(&mut words, 2 * RANDOM);
        pairs.extend(random.chunks_exact(2).map(|pair| (pair[0], pair[1])));
        let branches: Vec<Fold2> = pairs
            .iter()
            .map(|&(a, b)| reduce_branch::<F>(a * b))
            .collect();
        for branch in [Fold2::Plain, Fold2::Canonicalize, Fold2::Overflow] {
            assert!(branches.contains(&branch), "no mul input takes {branch:?}");
        }
        assert!(pairs.iter().any(|&(a, b)| a + b > WORD), "no add wraps");
        assert!(
            pairs.iter().any(|&(a, b)| (p..=WORD).contains(&(a + b))),
            "no add needs canonicalization"
        );
        assert!(pairs.iter().any(|&(a, b)| a < b), "no sub borrows");

        // Unary operations: every edge, the window operands, random.
        let mut singles = edges.clone();
        singles.extend(windows::<F>().into_iter().map(|(_, b)| b));
        singles.extend(random_elements::<F>(&mut words, RANDOM));

        // Scalar products: every edge against every scalar edge, the
        // windows, random elements against random scalars.
        let mut u64_pairs: Vec<(u128, u64)> = edges
            .iter()
            .flat_map(|&a| U64_EDGES.iter().map(move |&s| (a, s)))
            .collect();
        u64_pairs.extend(
            windows::<F>()
                .into_iter()
                .map(|(a, b)| (a, u64::try_from(b).unwrap())),
        );
        let random = random_elements::<F>(&mut words, RANDOM);
        u64_pairs.extend(random.into_iter().map(|a| (a, words.next().unwrap())));
        let branches: Vec<Fold2> = u64_pairs
            .iter()
            .map(|&(a, s)| reduce_branch::<F>(a * u128::from(s)))
            .collect();
        for branch in [Fold2::Plain, Fold2::Canonicalize, Fold2::Overflow] {
            assert!(
                branches.contains(&branch),
                "no mul_u64 input takes {branch:?}"
            );
        }
        assert!(
            u64_pairs.iter().any(|&(_, s)| u128::from(s) >= p),
            "no from_u64 input needs reduction"
        );

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
    fn fp64_59_matches_jolt_field() {
        conformance::<Prime64Offset59>("fp64_59_matches_jolt_field", 0xf064_003b);
    }

    #[test]
    fn fp64_max_offset_matches_jolt_field() {
        conformance::<MaxOffset>("fp64_max_offset_matches_jolt_field", 0xf064_ffff);
    }

    #[test]
    fn non_canonical_output_is_rejected_on_read_back() {
        let (_gpu, device) = gpu("fp64::non_canonical_output_is_rejected_on_read_back");
        let library = library::<Prime64Offset59>(&device, &[], &[]);
        let error =
            run::<Prime64Offset59>(&device, &library, WRITE_NON_CANONICAL, vec![], 3).unwrap_err();
        assert_eq!(error.class(), ErrorClass::Fault, "{error}");
        assert!(error.to_string().contains("element 0"), "{error}");
    }
}
