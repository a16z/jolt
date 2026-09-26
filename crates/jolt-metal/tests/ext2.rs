//! Conformance of `jolt::Ext2<F>` with `jolt_field::solinas::Ext2<F>`.
//!
//! Every MSL operation, and multiplication by a base-field element, runs
//! through a generic test kernel and must match the CPU result byte for
//! byte: on every pair of elements whose coefficients are base-field edges,
//! and on 2^20 fixed-seed random inputs. `Ext2` has no branches of its own;
//! the base fields' suites cover theirs.
//!
//! Two base fields: `Prime64Offset59`, whose `Ext2` is Akita's fp64
//! extension field, and `Prime128Offset275`, so the template is checked over
//! both base-field layouts.

#![cfg(target_os = "macos")]
#![expect(clippy::unwrap_used, reason = "tests may panic on assertion failures")]

#[path = "support/field.rs"]
mod field;
#[path = "support/ops.rs"]
mod ops;
mod support;

mod gpu {
    use jolt_field::solinas::{Ext2, Prime128Offset275, Prime64Offset59};
    use jolt_field::ExtField;
    use jolt_metal::runtime::{Binding, DeviceBuffer};
    use jolt_metal::MetalField;

    use super::field::{element, modulus, random_elements, TestField};
    use super::ops::{check_ops, compare, library, run, Inputs, I64_EDGES, U64_EDGES};
    use super::support::{gpu, SplitMix64};

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

    fn conformance<F: TestField>(test: &'static str, seed: u64)
    where
        Ext2<F>: MetalField,
    {
        let (_gpu, device) = gpu(test);
        let library = library::<Ext2<F>>(&device, &[("ext_ops.metal", EXT_OPS)], &[MUL_BASE]);
        let mut words = SplitMix64(seed);

        let coefficients = coefficient_edges::<F>();
        let edges: Vec<Ext2<F>> = coefficients
            .iter()
            .flat_map(|&c0| coefficients.iter().map(move |&c1| Ext2::new(c0, c1)))
            .collect();

        let mut pairs: Vec<(Ext2<F>, Ext2<F>)> = edges
            .iter()
            .flat_map(|&a| edges.iter().map(move |&b| (a, b)))
            .collect();
        let random = random_ext::<F>(&mut words, 2 * RANDOM);
        pairs.extend(random.chunks_exact(2).map(|pair| (pair[0], pair[1])));

        let mut singles = edges.clone();
        singles.extend(random_ext::<F>(&mut words, RANDOM));

        let mut u64_pairs: Vec<(Ext2<F>, u64)> = edges
            .iter()
            .flat_map(|&a| U64_EDGES.iter().map(move |&s| (a, s)))
            .collect();
        let random = random_ext::<F>(&mut words, RANDOM);
        u64_pairs.extend(random.into_iter().map(|a| (a, words.next().unwrap())));

        let mut i64_pairs: Vec<(Ext2<F>, i64)> = edges
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

        // Base-field scaling: every edge against every coefficient edge, and
        // random.
        let mut base_pairs: Vec<(Ext2<F>, F)> = edges
            .iter()
            .flat_map(|&a| coefficients.iter().map(move |&x| (a, x)))
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
    fn ext2_fp128_275_matches_jolt_field() {
        conformance::<Prime128Offset275>("ext2_fp128_275_matches_jolt_field", 0xe228_0113);
    }
}
