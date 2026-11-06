#![cfg(all(feature = "setup-weight-tiles", feature = "solinas"))]

use jolt_field::{Field, Prime128Offset275, Prime128OffsetA7F7};

struct Words(u64);

impl Words {
    fn scalar<F: Field>(&mut self) -> F {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let low = self.0;
        self.0 = self.0.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        F::from_u128(u128::from(low) | (u128::from(self.0) << 64))
    }
}

fn check<F: Field>() {
    let mut words = Words(0x4850_3235_5449_4c45);
    for size in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 63, 64, 65, 257] {
        for round in 0..67 {
            let scale = match round {
                0 => F::zero(),
                1 => F::one(),
                2 => -F::one(),
                _ => words.scalar::<F>(),
            };
            let src = (0..size).map(|_| words.scalar::<F>()).collect::<Vec<_>>();
            let before = (0..size + 2)
                .map(|_| words.scalar::<F>())
                .collect::<Vec<_>>();
            let mut actual = before.clone();
            let expected = before
                .iter()
                .enumerate()
                .map(|(i, value)| src.get(i).map_or(*value, |src| *value + scale * *src))
                .collect::<Vec<_>>();
            F::add_scaled_slice(scale, &src, &mut actual);
            assert_eq!(actual, expected);
            let mut short = before[..size / 2].to_vec();
            F::add_scaled_slice(scale, &src, &mut short);
            assert_eq!(short, expected[..size / 2]);
        }
    }
    let mut dst = [-F::one(); 5];
    F::add_scaled_slice(-F::one(), &[-F::one(); 5], &mut dst);
    assert_eq!(dst, [F::zero(); 5]);
}

#[test]
fn setup_weight_tiles_randomized_and_boundary_equivalence() {
    check::<Prime128OffsetA7F7>();
    check::<Prime128Offset275>();
}
