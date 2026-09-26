//! Conformance of `jolt::Ext2<F>` with `jolt_field::solinas::Ext2<F>`.
//!
//! Every MSL operation, and multiplication by a base-field element, runs
//! through a generic test kernel and must match the CPU result byte for
//! byte: on every pair of elements whose coefficients are base-field edges,
//! on products that reach the rare reduction branches, and on 2^20
//! fixed-seed random inputs.
//!
//! Over `Fp64<C>` with `C < 2^31`, multiply and square reduce each
//! coefficient's sum of products once (`fp64_detail::reduce_sum`). The test
//! recomputes that reduction's intermediate values and requires every branch
//! to be taken by both coefficients of the multiply, and one sum to carry
//! into its top word through the low word. The square shares the reduction.
//!
//! Four base fields: `Prime64Offset59`, whose `Ext2` is Akita's fp64
//! extension field; `2^64 − 0x7fffffd3`, whose offset is the largest prime
//! offset below `2^31`, so the `reduce_sum` bound is reached at its limit;
//! and, through the generic Karatsuba forms, `2^64 − 2^32 + 1` and
//! `Prime128Offset275`.

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
    use jolt_field::solinas::{Ext2, Fp64, Prime128Offset275, Prime64Offset59};
    use jolt_field::ExtField;
    use jolt_metal::runtime::{Binding, DeviceBuffer};
    use jolt_metal::MetalField;

    use super::field::{edges, element, modulus, random_elements, TestField};
    use super::fp64::{fold2_branch, windows, Fold2, BRANCHES};
    use super::ops::{check_ops, compare, library, run, Inputs, I64_EDGES, U64_EDGES};
    use super::support::{gpu, SplitMix64};

    /// `2^64 − 0x7fffffd3`: the largest prime offset below `2^31`.
    type MaxSumOffset = Fp64<0xFFFF_FFFF_8000_002D>;
    /// `2^64 − 2^32 + 1`: the largest `jolt::Fp64` offset.
    type MaxOffset = Fp64<0xFFFF_FFFF_0000_0001>;

    const EXT_OPS: &str = include_str!("shaders/ext_ops.metal");
    const MUL_BASE: &str = "jolt_test_ext_mul_base";

    /// Random inputs per operation.
    const RANDOM: usize = 1 << 20;

    /// Base-field edges for the coefficients: small values, `C`, the
    /// half-way point, the top of the field, and a word boundary.
    fn coefficient_edges<F: TestField>() -> Vec<F> {
        let (p, c) = (modulus::<F>(), F::OFFSET);
        [
            0,
            1,
            2,
            c,
            (1 << 32) - 1,
            1 << 63,
            p / 2,
            p / 2 + 1,
            p - c,
            p - 2,
            p - 1,
        ]
        .into_iter()
        .map(element)
        .collect()
    }

    fn random_ext<F: TestField>(words: &mut SplitMix64, len: usize) -> Vec<Ext2<F>> {
        random_elements::<F>(words, 2 * len)
            .chunks_exact(2)
            .map(|pair| Ext2::new(element(pair[0]), element(pair[1])))
            .collect()
    }

    fn value<F: TestField>(x: F) -> u128 {
        x.to_u128_checked().unwrap()
    }

    /// Whether summing `products` as `fp64_detail::add` does, word by word,
    /// carries into the top word through the low word: the high words sum to
    /// exactly `2^64 − 1` without wrapping, and the low words' carry wraps them.
    /// Random operands reach this with probability about `2^−64`.
    fn carries_through_low(products: &[u128]) -> bool {
        let Some((&first, rest)) = products.split_first() else {
            return false;
        };
        let mut sum = first;
        let mut carried = false;
        for &y in rest {
            let (high, wrapped) = ((sum >> 64) as u64).overflowing_add((y >> 64) as u64);
            let (_, low_carry) = (sum as u64).overflowing_add(y as u64);
            carried |= !wrapped && high == u64::MAX && low_carry;
            sum = sum.wrapping_add(y);
        }
        carried
    }

    /// Operands `(a1, b0)` for which `(p − 1)^2 + a1 b0`, the `c1` of the `Ext2`
    /// product `(p − 1, a1) · (b0, p − 1)`, carries through the low word
    /// (`carries_through_low`).
    ///
    /// With `x = (p − 1)^2 = h 2^64 + l`, the second product must lie in
    /// `[start, start + l)` for `start = (2^64 − 1 − h) 2^64 + 2^64 − l`. Any
    /// `a1 ≤ l` has a multiple there; the smallest `a1` with
    /// `b0 = ⌈start / a1⌉ < p` is taken.
    fn low_carry<F: TestField>() -> (u128, u128) {
        let p = modulus::<F>();
        let x = (p - 1) * (p - 1);
        let (h, l) = (x >> 64, x & u128::from(u64::MAX));
        let start = (u128::from(u64::MAX) - h) * (1 << 64) + (1 << 64) - l;
        (start.div_ceil(p - 1)..=l)
            .map(|a1| (a1, start.div_ceil(a1)))
            .find(|&(a1, b0)| b0 < p && a1 * b0 < start + l)
            .unwrap()
    }

    /// Whether `jolt::Ext2<F>` multiplies through `reduce_sum`: the
    /// condition of the Fp64 overloads in `ext2.h`.
    fn sums_products<F: TestField>() -> bool {
        F::MODULUS_BITS == 64 && F::OFFSET < 1 << 31
    }

    fn conformance<F: TestField>(test: &'static str, seed: u64)
    where
        Ext2<F>: MetalField,
    {
        let (_gpu, device) = gpu(test);
        let library = library::<Ext2<F>>(&device, &[("ext_ops.metal", EXT_OPS)], &[MUL_BASE]);
        let mut words = SplitMix64(seed);

        let coefficients = coefficient_edges::<F>();
        let ext_edges: Vec<Ext2<F>> = coefficients
            .iter()
            .flat_map(|&c0| coefficients.iter().map(move |&c1| Ext2::new(c0, c1)))
            .collect();

        // Every pair of edges; then (a, 0) (b, b) for the base field's
        // window operands, so each coefficient of the product is the single
        // product a b; then random.
        let mut pairs: Vec<(Ext2<F>, Ext2<F>)> = ext_edges
            .iter()
            .flat_map(|&a| ext_edges.iter().map(move |&b| (a, b)))
            .collect();
        if F::MODULUS_BITS == 64 {
            pairs.extend(windows::<F>().into_iter().map(|(a, b)| {
                let (a, b) = (element::<F>(a), element::<F>(b));
                (Ext2::new(a, F::zero()), Ext2::new(b, b))
            }));
        }
        let random = random_ext::<F>(&mut words, 2 * RANDOM);
        pairs.extend(random.chunks_exact(2).map(|pair| (pair[0], pair[1])));
        if sums_products::<F>() {
            // A c1 sum whose carry into the top word comes through the low
            // word.
            let (a1, b0) = low_carry::<F>();
            let top = element::<F>(modulus::<F>() - 1);
            pairs.push((Ext2::new(top, element(a1)), Ext2::new(element(b0), top)));
            let coefficient_sums = |product: fn(u128, u128, u128, u128) -> [u128; 3]| {
                pairs
                    .iter()
                    .map(|&(a, b)| {
                        let [a0, a1, b0, b1] = [a.c0(), a.c1(), b.c0(), b.c1()].map(value);
                        product(a0, a1, b0, b1)
                    })
                    .collect::<Vec<[u128; 3]>>()
            };
            let c0 = coefficient_sums(|a0, a1, b0, b1| [a0 * b0, a1 * b1, a1 * b1]);
            let c1 = coefficient_sums(|a0, a1, b0, b1| [a0 * b1, a1 * b0, 0]);
            for (name, sums) in [("c0", &c0), ("c1", &c1)] {
                let branches: Vec<Fold2> = sums.iter().map(|sum| fold2_branch::<F>(sum)).collect();
                for branch in BRANCHES {
                    assert!(
                        branches.contains(&branch),
                        "no {name} of a product takes {branch:?}"
                    );
                }
            }
            assert!(
                c1.iter().any(|sum| carries_through_low(sum)),
                "no c1 of a product carries into its top word through the low word"
            );
        }

        let mut singles = ext_edges.clone();
        singles.extend(random_ext::<F>(&mut words, RANDOM));

        let mut u64_pairs: Vec<(Ext2<F>, u64)> = ext_edges
            .iter()
            .flat_map(|&a| U64_EDGES.iter().map(move |&s| (a, s)))
            .collect();
        let random = random_ext::<F>(&mut words, RANDOM);
        u64_pairs.extend(random.into_iter().map(|a| (a, words.next().unwrap())));

        let mut i64_pairs: Vec<(Ext2<F>, i64)> = ext_edges
            .iter()
            .flat_map(|&a| I64_EDGES.iter().map(move |&s| (a, s)))
            .collect();
        let random = random_ext::<F>(&mut words, RANDOM);
        i64_pairs.extend(
            random
                .into_iter()
                .map(|a| (a, words.next().unwrap().cast_signed())),
        );

        let inputs = Inputs {
            pairs,
            singles,
            u64_pairs,
            i64_pairs,
        };
        let mut failures = Vec::new();
        check_ops(&device, &library, &inputs, &mut failures);

        // Base-field scaling: every edge against every base-field edge, and
        // random.
        let scalars: Vec<F> = edges::<F>().into_iter().map(element).collect();
        let mut base_pairs: Vec<(Ext2<F>, F)> = ext_edges
            .iter()
            .flat_map(|&a| scalars.iter().map(move |&x| (a, x)))
            .collect();
        let random = random_ext::<F>(&mut words, RANDOM);
        let scalars = random_elements::<F>(&mut words, RANDOM);
        base_pairs.extend(random.into_iter().zip(scalars.into_iter().map(element)));
        let (a, x): (Vec<Ext2<F>>, Vec<F>) = base_pairs.iter().copied().unzip();
        let (a_dev, x_dev) = (
            DeviceBuffer::from_slice(&device, &a).unwrap(),
            DeviceBuffer::from_slice(&device, &x).unwrap(),
        );
        let got = run::<Ext2<F>>(
            &device,
            &library,
            MUL_BASE,
            vec![Binding::buffer(&a_dev), Binding::buffer(&x_dev)],
            a.len(),
        )
        .unwrap();
        let want: Vec<Ext2<F>> = a.iter().zip(&x).map(|(&a, &x)| a.mul_base(x)).collect();
        compare(&mut failures, MUL_BASE, &got, &want, |i| {
            format!("{:?}", base_pairs[i])
        });

        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[test]
    fn ext2_fp64_59_matches_jolt_field() {
        conformance::<Prime64Offset59>("ext2_fp64_59_matches_jolt_field", 0xe264_003b);
    }

    #[test]
    fn ext2_fp64_max_sum_offset_matches_jolt_field() {
        conformance::<MaxSumOffset>("ext2_fp64_max_sum_offset_matches_jolt_field", 0xe264_7fff);
    }

    #[test]
    fn ext2_fp64_max_offset_matches_jolt_field() {
        conformance::<MaxOffset>("ext2_fp64_max_offset_matches_jolt_field", 0xe264_ffff);
    }

    #[test]
    fn ext2_fp128_275_matches_jolt_field() {
        conformance::<Prime128Offset275>("ext2_fp128_275_matches_jolt_field", 0xe228_0113);
    }
}
