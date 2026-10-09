#[cfg(any(
    all(target_arch = "aarch64", target_feature = "aes"),
    all(target_arch = "x86_64", target_feature = "pclmulqdq")
))]
use super::kernels;
use super::{
    accumulator::{F128Accumulator, F192Accumulator, F64Accumulator},
    portable, reduction, F128, F192, F64,
};
use crate::{Accumulator, WithAccumulator};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_core::RngCore;

#[test]
fn kernel_matches_portable() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x6269_6e61_7279_0012);
    let boundaries64 = [0, 1, 1 << 63, u64::MAX];
    for (a, b) in boundaries64
        .into_iter()
        .flat_map(|a| boundaries64.map(|b| (a, b)))
        .chain((0..10_000).map(|_| (rng.next_u64(), rng.next_u64())))
    {
        let product = portable::product64(a, b);
        assert_eq!(reduction::reduce64(product), portable::multiply64(a, b));
        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq")
        ))]
        {
            assert_eq!(kernels::product64(a, b), product);
            assert_eq!(kernels::multiply64(a, b), portable::multiply64(a, b));
            assert_eq!(kernels::square64(a), portable::square64(a));
        }
    }

    let boundaries128 = [0, 1, 1 << 127, u128::MAX];
    for (a, b) in boundaries128
        .into_iter()
        .flat_map(|a| boundaries128.map(|b| (a, b)))
        .chain((0..10_000).map(|_| {
            let [a0, a1, b0, b1] = std::array::from_fn(|_| rng.next_u64());
            (
                u128::from(a0) | (u128::from(a1) << 64),
                u128::from(b0) | (u128::from(b1) << 64),
            )
        }))
    {
        let product = portable::product128(a, b);
        assert_eq!(
            reduction::reduce128(product[0], product[1]),
            portable::multiply128(a, b)
        );
        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq")
        ))]
        {
            assert_eq!(kernels::product128(a, b), product);
            assert_eq!(kernels::multiply128(a, b), portable::multiply128(a, b));
            assert_eq!(kernels::square128(a), portable::square128(a));
        }
    }

    let boundaries192 = [[0; 3], [1, 0, 0], [0, 0, 1 << 63], [u64::MAX; 3]];
    for (a, b) in boundaries192
        .into_iter()
        .flat_map(|a| boundaries192.map(|b| (a, b)))
        .chain((0..10_000).map(|_| {
            (
                std::array::from_fn(|_| rng.next_u64()),
                std::array::from_fn(|_| rng.next_u64()),
            )
        }))
    {
        let product = portable::product192(a, b);
        assert_eq!(
            product.map(reduction::reduce64),
            portable::multiply192(a, b)
        );
        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq")
        ))]
        {
            assert_eq!(kernels::product192(a, b), product);
            assert_eq!(kernels::multiply192(a, b), portable::multiply192(a, b));
            assert_eq!(kernels::square192(a), portable::square192(a));
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
