use super::{
    accumulator::{F128Accumulator, F192Accumulator, F64Accumulator},
    portable, F128, F192, F64,
};
#[cfg(any(
    all(target_arch = "aarch64", target_feature = "aes"),
    all(target_arch = "x86_64", target_feature = "pclmulqdq")
))]
use super::{arch::Word, kernels};
use crate::{Accumulator, ExtField, WithAccumulator};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_core::RngCore;

#[test]
fn kernel_matches_portable() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x6269_6e61_7279_0012);
    let boundaries64 = [0, 1, 1 << 63, u64::MAX];
    let mut acc64 = F64Accumulator::default();
    let mut expected64 = 0;
    for (i, (a, b)) in boundaries64
        .into_iter()
        .flat_map(|a| boundaries64.map(|b| (a, b)))
        .chain((0..10_000).map(|_| (rng.next_u64(), rng.next_u64())))
        .enumerate()
    {
        if i < 1000 {
            acc64.fmadd(F64::from_raw(a), F64::from_raw(b));
            expected64 ^= portable::multiply64(a, b);
        }
        assert_eq!(portable::reduce64(portable::embed64(a)), a);
        let product = portable::product64(a, b);
        assert_eq!(portable::reduce64(product), portable::multiply64(a, b));
        assert_eq!(
            portable::reduce_accumulator64(product),
            portable::multiply64(a, b)
        );
        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq")
        ))]
        {
            assert_eq!(
                kernels::canonical64(kernels::embed64(a)),
                portable::embed64(a)
            );
            assert_eq!(kernels::canonical64(kernels::product64(a, b)), product);
            assert_eq!(
                kernels::reduce64(kernels::product64(a, b)),
                portable::multiply64(a, b)
            );
            assert_eq!(
                kernels::reduce_accumulator64(kernels::product64(a, b)),
                portable::multiply64(a, b)
            );
            assert_eq!(kernels::multiply64(a, b), portable::multiply64(a, b));
            assert_eq!(kernels::square64(a), portable::square64(a));
        }
    }

    assert_eq!(acc64.reduce().to_raw(), expected64);

    let boundaries128 = [0, 1, 1 << 127, u128::MAX];
    let mut acc128 = F128Accumulator::default();
    let mut expected128 = 0;
    for (i, (a, b)) in boundaries128
        .into_iter()
        .flat_map(|a| boundaries128.map(|b| (a, b)))
        .chain((0..10_000).map(|_| {
            let [a0, a1, b0, b1] = std::array::from_fn(|_| rng.next_u64());
            (
                u128::from(a0) | (u128::from(a1) << 64),
                u128::from(b0) | (u128::from(b1) << 64),
            )
        }))
        .enumerate()
    {
        if i < 1000 {
            acc128.fmadd(F128::from_raw(a), F128::from_raw(b));
            expected128 ^= portable::multiply128(a, b);
        }
        assert_eq!(portable::reduce128(portable::embed128(a)), a);
        let product = portable::product128(a, b);
        assert_eq!(portable::accumulate128([0; 2], a, b), product);
        assert_eq!(portable::reduce128(product), portable::multiply128(a, b));
        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq")
        ))]
        {
            assert_eq!(
                kernels::canonical128(kernels::embed128(a)),
                portable::embed128(a)
            );
            assert_eq!(kernels::canonical128(kernels::product128(a, b)), product);
            assert_eq!(
                kernels::canonical128(kernels::accumulate128(Default::default(), a, b)),
                product
            );
            for (raw, reduced) in kernels::variants128(a, b) {
                assert_eq!(raw, product);
                for value in reduced {
                    assert_eq!(value, portable::multiply128(a, b));
                }
            }
            assert_eq!(
                kernels::reduce128(kernels::product128(a, b)),
                portable::multiply128(a, b)
            );
            assert_eq!(kernels::multiply128(a, b), portable::multiply128(a, b));
            assert_eq!(kernels::square128(a), portable::square128(a));
        }
    }

    assert_eq!(acc128.reduce().to_raw(), expected128);

    let boundaries192 = [[0; 3], [1, 0, 0], [0, 0, 1 << 63], [u64::MAX; 3]];
    let mut acc192 = F192Accumulator::default();
    let mut expected192 = [0; 3];
    for (i, (a, b)) in boundaries192
        .into_iter()
        .flat_map(|a| boundaries192.map(|b| (a, b)))
        .chain((0..10_000).map(|_| {
            (
                std::array::from_fn(|_| rng.next_u64()),
                std::array::from_fn(|_| rng.next_u64()),
            )
        }))
        .enumerate()
    {
        if i < 1000 {
            acc192.fmadd(
                F192::from_base_fn(|i| F64::from_raw(a[i])),
                F192::from_base_fn(|i| F64::from_raw(b[i])),
            );
            for (word, product) in expected192.iter_mut().zip(portable::multiply192(a, b)) {
                *word ^= product;
            }
        }
        assert_eq!(portable::reduce192(portable::embed192(a)), a);
        let product = portable::product192(a, b);
        assert_eq!(portable::reduce192(product), portable::multiply192(a, b));
        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq")
        ))]
        {
            assert_eq!(
                kernels::canonical192(kernels::embed192(a)),
                portable::embed192(a)
            );
            assert_eq!(kernels::canonical192(kernels::product192(a, b)), product);
            assert_eq!(
                kernels::reduce192(kernels::product192(a, b)),
                portable::multiply192(a, b)
            );
            assert_eq!(kernels::multiply192(a, b), portable::multiply192(a, b));
            assert_eq!(kernels::square192(a), portable::square192(a));
        }
    }
    assert_eq!(
        acc192.reduce(),
        F192::from_base_fn(|i| F64::from_raw(expected192[i]))
    );

    // (2^64-1)*0x1b = (0x9<<64)^0x9 and 0x9*0x1b = 0xc3 in GF(2)[x].
    let all64 = 0xffff_ffff_ffff_ff35;
    // (2^128-1)*0x87 = (0x7d<<128)^0x7d and 0x7d*0x87 = 0x3ff3.
    let all128 = 0xffff_ffff_ffff_ffff_ffff_ffff_ffff_c071;
    assert_eq!(portable::reduce64(u128::MAX), all64);
    assert_eq!(portable::reduce_accumulator64(u128::MAX), all64);
    assert_eq!(portable::reduce128([u128::MAX; 2]), all128);
    assert_eq!(portable::reduce192([u128::MAX; 3]), [all64; 3]);
    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq")
    ))]
    {
        let ones = Word::from_u128(u128::MAX);
        assert_eq!(kernels::reduce64(ones.into_unreduced64()), all64);
        assert_eq!(
            kernels::reduce_accumulator64(ones.into_unreduced64()),
            all64
        );
        for value in kernels::reductions128([ones, Word::from_u64(0), ones]) {
            assert_eq!(value, all128);
        }
        assert_eq!(kernels::reduce192([ones; 3]), [all64; 3]);
        let storage_ones = [ones; 3];
        // The overlapping middle 128 bits cancel when all three words are ones.
        let outer_ones = [
            0x0000_0000_0000_0000_ffff_ffff_ffff_ffff,
            0xffff_ffff_ffff_ffff_0000_0000_0000_0000,
        ];
        assert_eq!(kernels::canonical128(storage_ones), outer_ones);
        for value in kernels::reductions128(storage_ones) {
            assert_eq!(value, portable::reduce128(outer_ones));
        }
    }
}

#[test]
fn concrete_accumulator_types() {
    fn assert_types<F, A>()
    where
        F: WithAccumulator<
            Accumulator = A,
            SmallScalarAccumulator = A,
            SignedProductAccumulator = A,
        >,
        A: Accumulator<Element = F>,
    {
    }

    assert_types::<F64, F64Accumulator>();
    assert_types::<F128, F128Accumulator>();
    assert_types::<F192, F192Accumulator>();
}
