//! Zero-constant-term sampling.

#![expect(
    clippy::unwrap_used,
    reason = "sampled coefficients are canonical field elements"
)]
use jolt_field::{CanonicalEncoding, Fp64, Prime128Offset275, Zero};
use jolt_zk::whiteout::sample_zero_constant_term;
use rand::rngs::StdRng;
use rand::SeedableRng;

type F = Prime128Offset275;
const D: usize = 64;

#[test]
fn samples_have_zero_constant_term() {
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..32 {
        let g = sample_zero_constant_term::<F, D, _>(&mut rng);
        assert!(g.constant_term().is_zero());
        assert!(g.coefficients().iter().skip(1).any(|c| !c.is_zero()));
    }
}

#[test]
fn nonconstant_coefficients_look_uniform_over_a_small_field() {
    // Over F_q with q = 4294967197, bucket each coefficient by its residue
    // mod 4 and check that no bucket strays far from a quarter.
    type Small = Fp64<4_294_967_197>;
    let mut rng = StdRng::seed_from_u64(9);
    let mut buckets = [0u32; 4];
    let samples = 256;
    for _ in 0..samples {
        let g = sample_zero_constant_term::<Small, D, _>(&mut rng);
        for c in g.coefficients().iter().skip(1) {
            let bucket = (c.to_u128_checked().unwrap() % 4) as usize;
            buckets[bucket] += 1;
        }
    }
    let expected = f64::from(samples * (D as u32 - 1)) / 4.0;
    for count in buckets {
        let deviation = (f64::from(count) - expected).abs() / expected;
        assert!(deviation < 0.05, "bucket count {count} vs {expected}");
    }
}
