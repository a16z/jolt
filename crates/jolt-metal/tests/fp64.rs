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
#[path = "support/fp64.rs"]
mod fp64;
#[path = "support/ops.rs"]
mod ops;
mod support;

mod gpu {
    use jolt_field::solinas::{Fp64, Prime64Offset59};
    use jolt_metal::ErrorClass;

    use super::field::{edges, element, modulus, random_elements, TestField};
    use super::fp64::{fold2_branch, windows, Fold2, BRANCHES};
    use super::ops::{check_ops, library, run, Inputs, I64_EDGES, U64_EDGES, WRITE_NON_CANONICAL};
    use super::support::{gpu, SplitMix64};

    /// `2^64 − 2^32 + 1`: offset `2^32 − 1`.
    type MaxOffset = Fp64<0xFFFF_FFFF_0000_0001>;

    /// Random inputs per operation.
    const RANDOM: usize = 1 << 20;

    const WORD: u128 = u64::MAX as u128;

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
            .map(|&(a, b)| fold2_branch::<F>(&[a * b]))
            .collect();
        for branch in BRANCHES {
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
            .map(|&(a, s)| fold2_branch::<F>(&[a * u128::from(s)]))
            .collect();
        for branch in BRANCHES {
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
