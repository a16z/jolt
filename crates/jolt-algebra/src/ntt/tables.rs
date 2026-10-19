//! Deterministic parameter presets for small-prime CRT arithmetic.
//!
//! Q32: `logq = 32` with two `i32` NTT-friendly primes.
//! Q64: `logq = 64` with three `i32` NTT-friendly primes.
//! Q128: `logq = 128` with five `i32` NTT-friendly primes.

use super::prime::NttPrime;

/// Maximum ring degree for Q128 CRT+NTT profile (challenge-supported paths).
pub const Q128_MAX_RING_D: usize = 1024;
/// Maximum ring degree for Q64 CRT+NTT profile (challenge-supported paths).
pub const Q64_MAX_RING_D: usize = 2048;
/// Maximum ring degree for Q32 CRT+NTT profile (challenge-supported paths).
pub const Q32_MAX_RING_D: usize = 2048;

/// Number of CRT primes for the `logq = 32` parameter set.
pub const Q32_NUM_PRIMES: usize = 2;

/// The modulus `q = 2^32 - 99`.
pub const Q32_MODULUS: u64 = (1u64 << 32) - 99;

/// Number of CRT primes for the `logq = 64` reduced profile.
pub const Q64_NUM_PRIMES: usize = 3;

/// The modulus `q = 2^64 - 59`.
pub const Q64_MODULUS: u64 = u64::MAX - 58;

/// Number of CRT primes for the `logq = 128` parameter set.
pub const Q128_NUM_PRIMES: usize = 6;

/// Protocol modulus `q = 2^128 - 275`.
pub const Q128_MODULUS: u128 = u128::MAX - 274;

/// Universal 14-bit tail prime for exact CRT accumulation.
///
/// `12289 - 1 = 3 * 2^12`, so negacyclic transforms are available through
/// `D = 2048`, covering every protocol ring-degree tier.
pub const I16_TAIL_PRIME: NttPrime<i16> = NttPrime {
    p: 12289,
    pinv: -12287,
    mont: 4091,
    montsq: -1337,
};

/// Raw 30-bit primes for Q128 (`logq = 128`, K=5).
///
/// The six largest primes below `2^30` whose `p - 1` is divisible by `2048`,
/// independently selected for the Q128 `D <= 1024` profile. Q32 and Q64 use
/// separate tables below.
pub const I32_RAW_PRIMES: [i32; Q128_NUM_PRIMES] = [
    1073707009, 1073698817, 1073692673, 1073682433, 1073668097, 1073655809,
];

/// CRT primes and per-prime Montgomery constants for Q32 measured `2xi32` profile.
pub const Q32_PRIMES: [NttPrime<i32>; Q32_NUM_PRIMES] = [
    NttPrime {
        p: 1073692673,
        pinv: 1342226433,
        mont: 196604,
        montsq: 196588,
    },
    NttPrime {
        p: 1073668097,
        pinv: 67182593,
        mont: 294908,
        montsq: 3612607,
    },
];

/// Raw 30-bit primes for Q128.
pub const Q128_RAW_PRIMES: [i32; Q128_NUM_PRIMES] = I32_RAW_PRIMES;

/// The three largest primes below `2^30` whose `p - 1` is divisible by `4096`,
/// with per-prime Montgomery constants for the Q64 `D <= 2048` profile.
pub const Q64_PRIMES: [NttPrime<i32>; Q64_NUM_PRIMES] = [
    NttPrime {
        p: 1073692673,
        pinv: 1342226433,
        mont: 196604,
        montsq: 196588,
    },
    NttPrime {
        p: 1073668097,
        pinv: 67182593,
        mont: 294908,
        montsq: 3612607,
    },
    NttPrime {
        p: 1073655809,
        pinv: 2030129153,
        mont: 344060,
        montsq: 275144610,
    },
];

/// CRT primes and per-prime Montgomery constants for `logq = 128`.
pub fn q128_primes() -> [NttPrime<i32>; Q128_NUM_PRIMES] {
    Q128_RAW_PRIMES.map(NttPrime::new)
}

/// Validate CRT+NTT `ring_d` against a profile-specific maximum.
///
/// Not used for B/D matrix degrees below 64; callers skip validation there.
pub fn validate_profile_crt_ring_degree(
    ring_d: usize,
    max_d: usize,
) -> Result<(), akita_error::AkitaError> {
    if ring_d == 0 || !ring_d.is_power_of_two() {
        return Err(akita_error::AkitaError::InvalidSetup(format!(
            "CRT ring degree must be a non-zero power of two, got {ring_d}"
        )));
    }
    if ring_d > max_d {
        return Err(akita_error::AkitaError::InvalidSetup(format!(
            "ring degree {ring_d} exceeds profile max {max_d}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::crt::GarnerData;
    use crate::ntt::prime::is_prime;

    #[test]
    fn six_product_i32_lazy_reduction_fits_signed_wide() {
        for modulus in I32_RAW_PRIMES {
            let modulus = i128::from(modulus);
            let product_sum_bound = 6 * (modulus - 1).pow(2);
            let correction_bound = (1_i128 << 31) * modulus;
            assert!(product_sum_bound + correction_bound < 1_i128 << 63);

            let seven_product_bound = 7 * (modulus - 1).pow(2) + correction_bound;
            assert!(seven_product_bound >= 1_i128 << 63);
        }
    }

    fn assert_garner_profile<W, const K: usize>(
        label: &str,
        primes: &[NttPrime<W>; K],
        garner: GarnerData<K>,
    ) where
        W: super::super::prime::PrimeWidth,
    {
        for (i, prime_i) in primes.iter().enumerate().skip(1) {
            let pi = prime_i.p.to_i64();
            for (j, prime_j) in primes[..i].iter().enumerate() {
                let pj = prime_j.p.to_i64();
                let g = garner.gamma()[i][j] as i64;
                assert_eq!(
                    (pj * g) % pi,
                    1,
                    "{label} garner gamma[{i}][{j}] not inverse of p_{j} mod p_{i}"
                );
            }
        }
    }

    fn largest_eligible_i32_primes(count: usize, root_order: i64) -> Vec<i32> {
        let mut multiple = ((1_i64 << 30) - 2) / root_order;
        let mut primes = Vec::with_capacity(count);
        while primes.len() < count {
            let candidate = multiple * root_order + 1;
            if is_prime(candidate) {
                primes.push(i32::try_from(candidate).expect("prime below 2^30"));
            }
            multiple -= 1;
        }
        primes
    }

    fn assert_i16_prime_profile(primes: &[NttPrime<i16>]) {
        for prime in primes {
            let p = prime.p as i64;
            assert!(is_prime(p), "p={p} must be prime");
            assert_eq!(
                (p - 1) % 512,
                0,
                "512 must divide p-1 for D=256 NTT (p={p})"
            );
            let recomputed = NttPrime::compute(prime.p);
            assert_eq!(
                prime.pinv, recomputed.pinv,
                "pinv mismatch for p={}",
                prime.p
            );
            assert_eq!(
                prime.mont, recomputed.mont,
                "mont mismatch for p={}",
                prime.p
            );
            assert_eq!(
                prime.montsq, recomputed.montsq,
                "montsq mismatch for p={}",
                prime.p
            );
        }
    }

    fn assert_ntt_order_for_ring_d(prime: &NttPrime<i32>, ring_d: usize) {
        let p = prime.p as i64;
        let order = (2 * ring_d) as i64;
        assert_eq!(
            (p - 1) % order,
            0,
            "2*D={order} must divide p-1 for negacyclic NTT at D={ring_d} (p={p})"
        );
    }

    fn assert_i32_prime_profile(primes: &[NttPrime<i32>]) {
        for prime in primes {
            let p = prime.p as i64;
            assert!(is_prime(p), "p={p} must be prime");
            assert_eq!(
                (p - 1) % 512,
                0,
                "512 must divide p-1 for D=256 NTT (p={p})"
            );
            let recomputed = NttPrime::compute(prime.p);
            assert_eq!(
                prime.pinv, recomputed.pinv,
                "pinv mismatch for p={}",
                prime.p
            );
            assert_eq!(
                prime.mont, recomputed.mont,
                "mont mismatch for p={}",
                prime.p
            );
            assert_eq!(
                prime.montsq, recomputed.montsq,
                "montsq mismatch for p={}",
                prime.p
            );
        }
    }

    #[test]
    fn verify_synthetic_i16_prime_derived_constants() {
        let primes = [
            NttPrime::compute(15361_i16),
            NttPrime::compute(13313_i16),
            NttPrime::compute(12289_i16),
        ];
        assert_i16_prime_profile(&primes);
    }

    #[test]
    fn verify_q32_prime_derived_constants() {
        assert_i32_prime_profile(&Q32_PRIMES);
    }

    #[test]
    fn verify_q128_primes_are_valid() {
        assert_i32_prime_profile(&q128_primes());
    }

    #[test]
    fn verify_q64_primes_are_valid() {
        assert_i32_prime_profile(&Q64_PRIMES);
    }

    #[test]
    fn garner_data_is_consistent() {
        assert_garner_profile("Q32", &Q32_PRIMES, GarnerData::compute(&Q32_PRIMES));
        assert_garner_profile("Q64", &Q64_PRIMES, GarnerData::compute(&Q64_PRIMES));
        let q128_primes = q128_primes();
        assert_garner_profile("Q128", &q128_primes, GarnerData::compute(&q128_primes));
    }

    #[test]
    fn production_primes_support_profile_max_ring_degrees() {
        for prime in &Q32_PRIMES {
            assert_ntt_order_for_ring_d(prime, Q32_MAX_RING_D);
        }
        for prime in &Q64_PRIMES {
            assert_ntt_order_for_ring_d(prime, Q64_MAX_RING_D);
        }
        for prime in q128_primes() {
            assert_ntt_order_for_ring_d(&prime, Q128_MAX_RING_D);
        }
    }

    #[test]
    fn production_profiles_use_largest_eligible_i32_primes() {
        assert_eq!(
            Q32_PRIMES.map(|prime| prime.p).as_slice(),
            largest_eligible_i32_primes(Q32_NUM_PRIMES, (2 * Q32_MAX_RING_D) as i64)
        );
        assert_eq!(
            Q64_PRIMES.map(|prime| prime.p).as_slice(),
            largest_eligible_i32_primes(Q64_NUM_PRIMES, (2 * Q64_MAX_RING_D) as i64)
        );
        assert_eq!(
            I32_RAW_PRIMES.as_slice(),
            largest_eligible_i32_primes(Q128_NUM_PRIMES, (2 * Q128_MAX_RING_D) as i64)
        );
    }
}
