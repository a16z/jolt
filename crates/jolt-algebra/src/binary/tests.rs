use super::{kernels, portable};
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
        assert_eq!(kernels::multiply64(a, b), portable::multiply64(a, b));
        assert_eq!(kernels::square64(a), portable::square64(a));
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
        assert_eq!(kernels::multiply128(a, b), portable::multiply128(a, b));
        assert_eq!(kernels::square128(a), portable::square128(a));
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
        assert_eq!(kernels::multiply192(a, b), portable::multiply192(a, b));
        assert_eq!(kernels::square192(a), portable::square192(a));
    }
}
