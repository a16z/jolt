//! Ring fold challenge configuration for [`crate::SparseChallenge`].
//!
//! Witness-fold challenges are fixed-weight sparse polynomials: `count_pm1`
//! coefficients with magnitude 1 and `count_pm2` with magnitude 2, each with
//! random sign. When `count_pm2 == 0` every non-zero coefficient is ±1; when
//! `count_pm2 > 0` some coefficients are ±2 (production D=64).
//!
//! The actual sampler lives in [`crate::sampler`]; this file is policy-only.

/// Minimum min-entropy (bits) for every ring fold sparse-challenge transcript draw.
///
/// Every logical block receives an independent draw that must clear this floor.
pub const MIN_FOLD_CHALLENGE_ENTROPY_BITS: u32 = 128;

/// Production D=64 signed sparse ±1 count (LaBRADOR-aligned).
pub const D64_PRODUCTION_PM1_COUNT: usize = 31;
/// Production D=64 signed sparse ±2 count (LaBRADOR-aligned).
pub const D64_PRODUCTION_PM2_COUNT: usize = 10;

/// D=64 exact shell used by the selective-L2 operator-norm route.
///
/// The raw family has about 130.15 bits of support. The independently checked
/// fixed-point certificate retains 128.062439 bits inside the strict runtime
/// threshold 18 predicate.
pub const D64_L2_OP_NORM_PM1_COUNT: usize = 31;
pub const D64_L2_OP_NORM_PM2_COUNT: usize = 11;

/// D=128 selective-L2 route reuses the production `(31, 0)` shell.
pub const D128_L2_OP_NORM_PM1_COUNT: usize = 31;
pub const D128_L2_OP_NORM_PM2_COUNT: usize = 0;

/// Verifier-enforced operator-norm rejection policy for selective L2 folds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperatorNormRejection {
    /// Strict certified predicate threshold used by prover and verifier.
    pub threshold: u32,
    /// Fractional bits used by the certified fixed-point predicate.
    pub fractional_bits: u32,
    /// Certified upper bound on each fixed-point root-coordinate error.
    pub root_coordinate_error_units: u32,
    /// Fixed-point distance below `threshold` of the certified true-norm subset.
    pub rounding_margin_units: u32,
}

impl OperatorNormRejection {
    /// D=64 selective-L2 policy. Its support certificate is
    /// `cert_d64_a31_b11_gamma18.json` from the Akita paper artifacts.
    pub const D64_SELECTIVE_L2: Self = Self {
        threshold: 18,
        fractional_bits: 48,
        root_coordinate_error_units: 4,
        rounding_margin_units: 600,
    };

    /// D=128 selective-L2 policy. Its support certificate is
    /// `scripts/operator_norm/d128/cert_d128_w31_gamma13.json`.
    pub const D128_SELECTIVE_L2: Self = Self {
        threshold: 13,
        fractional_bits: 48,
        root_coordinate_error_units: 4,
        rounding_margin_units: 351,
    };

    /// Validate the exact challenge family covered by the support certificate.
    pub fn validate(
        self,
        ring_d: usize,
        config: &SparseChallengeConfig,
    ) -> Result<(), &'static str> {
        let covered = match self {
            Self::D64_SELECTIVE_L2 => {
                ring_d == 64
                    && config.count_pm1 == D64_L2_OP_NORM_PM1_COUNT
                    && config.count_pm2 == D64_L2_OP_NORM_PM2_COUNT
            }
            Self::D128_SELECTIVE_L2 => {
                ring_d == 128
                    && config.count_pm1 == D128_L2_OP_NORM_PM1_COUNT
                    && config.count_pm2 == D128_L2_OP_NORM_PM2_COUNT
            }
            _ => false,
        };
        if !covered {
            return Err("unsupported operator-norm rejection policy or challenge family");
        }
        Ok(())
    }

    /// Canonical transcript-domain bytes for this rejection policy.
    pub fn domain_separator_bytes(self) -> [u8; 17] {
        let mut out = [0u8; 17];
        out[0] = 2;
        out[1..5].copy_from_slice(&self.threshold.to_le_bytes());
        out[5..9].copy_from_slice(&self.fractional_bits.to_le_bytes());
        out[9..13].copy_from_slice(&self.root_coordinate_error_units.to_le_bytes());
        out[13..17].copy_from_slice(&self.rounding_margin_units.to_le_bytes());
        out
    }
}

/// Exact challenge config paired with
/// [`OperatorNormRejection::D64_SELECTIVE_L2`].
pub const D64_SELECTIVE_L2_CHALLENGE_CONFIG: SparseChallengeConfig = SparseChallengeConfig {
    count_pm1: D64_L2_OP_NORM_PM1_COUNT,
    count_pm2: D64_L2_OP_NORM_PM2_COUNT,
};

/// Exact challenge config paired with
/// [`OperatorNormRejection::D128_SELECTIVE_L2`].
pub const D128_SELECTIVE_L2_CHALLENGE_CONFIG: SparseChallengeConfig = SparseChallengeConfig {
    count_pm1: D128_L2_OP_NORM_PM1_COUNT,
    count_pm2: D128_L2_OP_NORM_PM2_COUNT,
};

/// Challenge family selected by a certified selective-L2 route.
#[inline]
#[must_use]
pub fn selective_l2_challenge_config(ring_d: usize) -> Option<SparseChallengeConfig> {
    match ring_d {
        64 => Some(D64_SELECTIVE_L2_CHALLENGE_CONFIG),
        128 => Some(D128_SELECTIVE_L2_CHALLENGE_CONFIG),
        _ => None,
    }
}

/// Rejection policy selected by a certified selective-L2 schedule, if any.
#[inline]
#[must_use]
pub fn selective_l2_operator_norm_rejection(
    ring_d: usize,
    config: &SparseChallengeConfig,
) -> Option<OperatorNormRejection> {
    let policy = match ring_d {
        64 => OperatorNormRejection::D64_SELECTIVE_L2,
        128 => OperatorNormRejection::D128_SELECTIVE_L2,
        _ => return None,
    };
    policy.validate(ring_d, config).is_ok().then_some(policy)
}

/// Ring degrees with a production fold-challenge ladder entry.
macro_rules! production_fold_challenge_ring_dims {
    ($($dim:literal),+ $(,)?) => {
        pub const PRODUCTION_FOLD_CHALLENGE_RING_DIMS: &[usize] = &[$($dim),+];

        macro_rules! __dispatch_fold_challenge_ring_dim {
            ($self:expr, $d:expr, $required_bits:expr) => {
                match $d {
                    $( $dim => $self.validate_min_entropy::<$dim>($required_bits), )+
                    _ => Err("unsupported ring dimension for fold-challenge entropy audit"),
                }
            };
        }
    };
}

production_fold_challenge_ring_dims!(64, 128, 256, 512, 1024, 2048);

// The last coordinate is floor(log2(support)). It lets the verifier and the
// offline planner validate the fixed production families without repeatedly
// evaluating dozens of floating-point logarithms. The tests below recompute
// every value from the canonical support formula.
const PRODUCTION_FOLD_CHALLENGE_LADDER: &[(usize, usize, usize, u32)] = &[
    (64, D64_PRODUCTION_PM1_COUNT, D64_PRODUCTION_PM2_COUNT, 128),
    (128, 31, 0, 129),
    (256, 23, 0, 131),
    (512, 19, 0, 132),
    (1024, 16, 0, 131),
    (2048, 14, 0, 131),
];

/// Fixed-weight sparse ring fold challenge family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SparseChallengeConfig {
    /// Number of non-zero coefficients with magnitude 1 (random sign).
    pub count_pm1: usize,
    /// Number of non-zero coefficients with magnitude 2 (random sign).
    pub count_pm2: usize,
}

impl SparseChallengeConfig {
    /// ±1-only sparse family with Hamming weight `count_pm1`.
    #[inline]
    #[must_use]
    pub const fn pm1_only(count_pm1: usize) -> Self {
        Self {
            count_pm1,
            count_pm2: 0,
        }
    }

    /// Production ladder entry for ring degree `ring_d`, if defined.
    #[inline]
    #[must_use]
    pub fn production_for_ring_dim(ring_d: usize) -> Option<Self> {
        PRODUCTION_FOLD_CHALLENGE_LADDER
            .iter()
            .find(|(d, _, _, _)| *d == ring_d)
            .map(|(_, pm1, pm2, _)| Self {
                count_pm1: *pm1,
                count_pm2: *pm2,
            })
    }

    /// Total Hamming weight.
    #[inline]
    #[must_use]
    pub fn weight(&self) -> usize {
        self.count_pm1.saturating_add(self.count_pm2)
    }

    /// Worst-case `L1` norm of the sampled coefficients.
    #[inline]
    #[must_use]
    pub fn l1_norm(&self) -> usize {
        self.count_pm1
            .saturating_add(2usize.saturating_mul(self.count_pm2))
    }

    /// Worst-case squared ℓ₂ norm `max ‖c‖_2²` over the challenge family.
    #[inline]
    #[must_use]
    pub fn challenge_l2_sq_max(&self) -> u128 {
        (self.count_pm1 as u128).saturating_add(4u128.saturating_mul(self.count_pm2 as u128))
    }

    /// Worst-case `L_infinity` norm of the sampled coefficients.
    #[inline]
    #[must_use]
    pub fn infinity_norm(&self) -> u32 {
        if self.count_pm2 > 0 {
            2
        } else {
            1
        }
    }

    /// `log2` of the number of distinct challenges this family can emit for ring
    /// degree `D` — the (raw) min-entropy of a single sampled challenge.
    pub fn log2_support_bits<const D: usize>(&self) -> f64 {
        fn log2_binom(n: usize, k: usize) -> f64 {
            if k > n {
                return f64::NEG_INFINITY;
            }
            (1..=k)
                .map(|i| ((n - k + i) as f64 / i as f64).log2())
                .sum()
        }
        let w = self.weight();
        if w > D {
            return f64::NEG_INFINITY;
        }
        log2_binom(D, w) + log2_binom(w, self.count_pm1) + w as f64
    }

    /// Reject challenge families whose single-draw support is below
    /// `required_bits` of min-entropy for ring degree `D`.
    pub fn validate_min_entropy<const D: usize>(
        &self,
        required_bits: u32,
    ) -> Result<(), &'static str> {
        if self.log2_support_bits::<D>() < f64::from(required_bits) {
            return Err("sparse challenge family has insufficient min-entropy for security floor");
        }
        Ok(())
    }

    /// Runtime ring-dimension dispatch for [`Self::validate_min_entropy`].
    pub fn validate_min_entropy_for_ring_dim(
        &self,
        ring_dim: usize,
        required_bits: u32,
    ) -> Result<(), &'static str> {
        if let Some((_, _, _, support_floor_bits)) =
            PRODUCTION_FOLD_CHALLENGE_LADDER
                .iter()
                .find(|(d, pm1, pm2, _)| {
                    *d == ring_dim && *pm1 == self.count_pm1 && *pm2 == self.count_pm2
                })
        {
            return if required_bits <= *support_floor_bits {
                Ok(())
            } else {
                Err("sparse challenge family has insufficient min-entropy for security floor")
            };
        }
        __dispatch_fold_challenge_ring_dim!(self, ring_dim, required_bits)
    }

    /// Structural invariants plus the 128-bit entropy floor at `ring_d`.
    pub fn validate_for_ring_dim(&self, ring_d: usize) -> Result<(), &'static str> {
        self.validate_dyn(ring_d)?;
        self.validate_min_entropy_for_ring_dim(ring_d, MIN_FOLD_CHALLENGE_ENTROPY_BITS)
    }

    /// Canonical byte encoding used for transcript domain separation.
    #[inline]
    pub fn domain_separator_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1 + 16);
        out.push(0);
        out.extend_from_slice(&(self.count_pm1 as u64).to_le_bytes());
        out.extend_from_slice(&(self.count_pm2 as u64).to_le_bytes());
        out
    }

    /// Validate basic invariants for a given ring degree `D`.
    pub fn validate<const D: usize>(&self) -> Result<(), &'static str> {
        self.validate_dyn(D)
    }

    /// Runtime ring-dimension form of [`Self::validate`].
    pub fn validate_dyn(&self, ring_d: usize) -> Result<(), &'static str> {
        if self
            .count_pm1
            .checked_add(self.count_pm2)
            .is_none_or(|w| w > ring_d)
        {
            return Err("count_pm1 + count_pm2 must be <= ring degree D");
        }
        Ok(())
    }
}

#[cfg(test)]
mod entropy_tests {
    use super::*;
    use jolt_field::{
        pseudo_mersenne_modulus, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59,
        PseudoMersenne,
    };

    fn assert_ls18_short_difference_bound<F: PseudoMersenne>(split_count: u32) {
        let modulus =
            pseudo_mersenne_modulus(F::MODULUS_BITS, F::OFFSET).expect("production modulus");
        let split_count = u128::from(split_count);
        for &carrier_dimension in PRODUCTION_FOLD_CHALLENGE_RING_DIMS {
            assert!(carrier_dimension.is_power_of_two());
            assert!(carrier_dimension >= split_count as usize);
        }
        assert_eq!(
            modulus % (4 * split_count),
            2 * split_count + 1,
            "production prime must meet the LS18 partial splitting congruence"
        );

        let max_challenge_coefficient = PRODUCTION_FOLD_CHALLENGE_RING_DIMS
            .iter()
            .map(|&ring_dim| {
                SparseChallengeConfig::production_for_ring_dim(ring_dim)
                    .expect("production challenge")
                    .infinity_norm()
            })
            .max()
            .expect("production challenge ladder");
        assert!(max_challenge_coefficient <= 2);

        let max_difference_coefficient = 2 * u128::from(max_challenge_coefficient);
        let split_count_exponent =
            u32::try_from(split_count).expect("small production split count");
        // For even `ell`, raising
        // `2 c_max < q^(1/ell) / sqrt(ell)` to `ell` gives this integer check.
        let exact_shortness_lhs = max_difference_coefficient.pow(split_count_exponent)
            * split_count.pow(split_count_exponent / 2);
        assert!(
            exact_shortness_lhs < modulus,
            "raising the strict LS18 shortness bound to split_count must remain below q"
        );
    }

    #[test]
    fn production_ladder_matches_proof_optimized_dims() {
        assert_eq!(
            PRODUCTION_FOLD_CHALLENGE_RING_DIMS.len(),
            PRODUCTION_FOLD_CHALLENGE_LADDER.len(),
            "the production dimension and challenge ladders must have identical coverage"
        );
        for (&d, &(_, _, _, support_floor_bits)) in PRODUCTION_FOLD_CHALLENGE_RING_DIMS
            .iter()
            .zip(PRODUCTION_FOLD_CHALLENGE_LADDER)
        {
            let cfg = SparseChallengeConfig::production_for_ring_dim(d).expect("ladder entry");
            assert!(cfg.validate_for_ring_dim(d).is_ok(), "d={d}");
            let computed_floor = match d {
                64 => cfg.log2_support_bits::<64>().floor() as u32,
                128 => cfg.log2_support_bits::<128>().floor() as u32,
                256 => cfg.log2_support_bits::<256>().floor() as u32,
                512 => cfg.log2_support_bits::<512>().floor() as u32,
                1024 => cfg.log2_support_bits::<1024>().floor() as u32,
                2048 => cfg.log2_support_bits::<2048>().floor() as u32,
                _ => unreachable!("production dimension list is exhaustive"),
            };
            assert_eq!(computed_floor, support_floor_bits, "d={d}");
            for required_bits in 0..=support_floor_bits + 2 {
                let generic = match d {
                    64 => cfg.validate_min_entropy::<64>(required_bits),
                    128 => cfg.validate_min_entropy::<128>(required_bits),
                    256 => cfg.validate_min_entropy::<256>(required_bits),
                    512 => cfg.validate_min_entropy::<512>(required_bits),
                    1024 => cfg.validate_min_entropy::<1024>(required_bits),
                    2048 => cfg.validate_min_entropy::<2048>(required_bits),
                    _ => unreachable!("production dimension list is exhaustive"),
                };
                assert_eq!(
                    cfg.validate_min_entropy_for_ring_dim(d, required_bits),
                    generic,
                    "fast and generic entropy checks disagree for d={d}, required={required_bits}"
                );
            }
        }
    }

    #[test]
    fn production_primes_and_challenges_meet_ls18_shortness() {
        assert_ls18_short_difference_bound::<Prime32Offset99>(2);
        assert_ls18_short_difference_bound::<Prime64Offset59>(2);
        assert_ls18_short_difference_bound::<Prime128OffsetA7F7>(4);
    }

    #[test]
    fn tiny_shell_is_rejected_at_128_bits() {
        let tiny = SparseChallengeConfig::pm1_only(2);
        assert!(tiny.log2_support_bits::<32>() < 128.0);
        assert!(tiny.validate_min_entropy::<32>(128).is_err());
    }

    #[test]
    fn production_shell_clears_128_bits() {
        let shell = SparseChallengeConfig {
            count_pm1: D64_PRODUCTION_PM1_COUNT,
            count_pm2: D64_PRODUCTION_PM2_COUNT,
        };
        assert!(shell.log2_support_bits::<64>() >= 128.0);
        assert!(shell.validate_for_ring_dim(64).is_ok());
    }

    #[test]
    fn entropy_floor_is_per_draw() {
        let weak = SparseChallengeConfig::pm1_only(1);
        let per_draw = weak.log2_support_bits::<4>();
        assert!(per_draw < 128.0);
        assert!(weak.validate_min_entropy::<4>(128).is_err());
    }

    #[test]
    fn log2_support_matches_small_closed_form() {
        let cfg = SparseChallengeConfig::pm1_only(1);
        assert!((cfg.log2_support_bits::<4>() - 3.0).abs() < 1e-9);
        let uni = SparseChallengeConfig::pm1_only(2);
        assert!((uni.log2_support_bits::<4>() - 24.0_f64.log2()).abs() < 1e-9);
    }

    #[test]
    fn challenge_l2_sq_max_matches_spec_table() {
        let shell = SparseChallengeConfig {
            count_pm1: D64_PRODUCTION_PM1_COUNT,
            count_pm2: D64_PRODUCTION_PM2_COUNT,
        };
        assert_eq!(shell.l1_norm(), 51);
        assert_eq!(shell.challenge_l2_sq_max(), 71);
        assert_eq!(shell.weight(), 41);

        let uni128 = SparseChallengeConfig::pm1_only(31);
        assert_eq!(uni128.challenge_l2_sq_max(), 31);
        assert_eq!(uni128.weight(), 31);

        let uni256 = SparseChallengeConfig::pm1_only(23);
        assert_eq!(uni256.challenge_l2_sq_max(), 23);
        assert_eq!(uni256.weight(), 23);

        for (d, pm1, pm2, _) in PRODUCTION_FOLD_CHALLENGE_LADDER {
            if *d >= 512 {
                let cfg = SparseChallengeConfig {
                    count_pm1: *pm1,
                    count_pm2: *pm2,
                };
                assert!(cfg.validate_for_ring_dim(*d).is_ok());
            }
        }
    }

    #[test]
    fn selective_l2_operator_norm_policies_are_dimension_and_family_typed() {
        let d64 = D64_SELECTIVE_L2_CHALLENGE_CONFIG;
        let d128 = D128_SELECTIVE_L2_CHALLENGE_CONFIG;
        assert_eq!(selective_l2_challenge_config(64), Some(d64));
        assert_eq!(selective_l2_challenge_config(128), Some(d128));
        assert_eq!(selective_l2_challenge_config(256), None);
        assert_eq!(
            selective_l2_operator_norm_rejection(64, &d64),
            Some(OperatorNormRejection::D64_SELECTIVE_L2)
        );
        assert_eq!(
            selective_l2_operator_norm_rejection(128, &d128),
            Some(OperatorNormRejection::D128_SELECTIVE_L2)
        );
        assert!(selective_l2_operator_norm_rejection(64, &d128).is_none());
        assert!(selective_l2_operator_norm_rejection(128, &d64).is_none());
        assert!(OperatorNormRejection::D128_SELECTIVE_L2
            .validate(128, &SparseChallengeConfig::pm1_only(30))
            .is_err());
    }
}
