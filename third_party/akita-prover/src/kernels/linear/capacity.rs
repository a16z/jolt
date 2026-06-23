use super::*;
use crate::kernels::crt_ntt::{select_crt_ntt_params, ProtocolCrtNttParams};
use crate::validation::MAX_I8_LOG_BASIS;
use akita_algebra::ntt::tables::{Q128_NUM_PRIMES, Q32_NUM_PRIMES, Q64_NUM_PRIMES};

pub(super) const BALANCED_DIGIT_RHS_MAX_ABS: u64 = 1 << (MAX_I8_LOG_BASIS - 1);
pub(super) const I8_RHS_MAX_ABS: u64 = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CrtI8CapacityProfile {
    pub profile_id: &'static str,
    pub num_primes: usize,
    pub limb_bits: u32,
    pub max_i8_log_basis: u32,
    pub balanced_digit_safe_width: usize,
    pub raw_i8_safe_width: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SmallNat {
    limbs: Vec<u32>,
}

impl SmallNat {
    fn one() -> Self {
        Self { limbs: vec![1] }
    }

    fn trim(&mut self) {
        while self.limbs.len() > 1 && self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
    }

    fn mul_u128(&mut self, rhs: u128) {
        if rhs == 0 {
            self.limbs.clear();
            self.limbs.push(0);
            return;
        }

        let mut rhs_limbs = Vec::new();
        let mut x = rhs;
        while x != 0 {
            rhs_limbs.push(x as u32);
            x >>= 32;
        }

        let mut out = vec![0u32; self.limbs.len() + rhs_limbs.len()];
        for (i, &lhs_limb) in self.limbs.iter().enumerate() {
            let mut carry = 0u128;
            for (j, &rhs_limb) in rhs_limbs.iter().enumerate() {
                let idx = i + j;
                let accum =
                    u128::from(out[idx]) + u128::from(lhs_limb) * u128::from(rhs_limb) + carry;
                out[idx] = accum as u32;
                carry = accum >> 32;
            }
            let mut idx = i + rhs_limbs.len();
            while carry != 0 {
                if idx == out.len() {
                    out.push(0);
                }
                let accum = u128::from(out[idx]) + carry;
                out[idx] = accum as u32;
                carry = accum >> 32;
                idx += 1;
            }
        }

        self.limbs = out;
        self.trim();
    }
}

impl Ord for SmallNat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.limbs.len().cmp(&other.limbs.len()) {
            std::cmp::Ordering::Equal => self.limbs.iter().rev().cmp(other.limbs.iter().rev()),
            ordering => ordering,
        }
    }
}

impl PartialOrd for SmallNat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Conservative maximum number of products that may be accumulated in one CRT
/// accumulator before Garner reconstruction.
pub(super) fn max_safe_crt_accumulation_width<
    F: CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    params: &CrtNttParamSet<W, K, D>,
    rhs_abs_bound: u64,
) -> Option<usize> {
    if rhs_abs_bound == 0 {
        return Some(usize::MAX);
    }

    let setup_abs_bound = setup_coeff_abs_bound::<F>();
    if setup_abs_bound == 0 || D == 0 {
        return None;
    }

    let mut crt_product = SmallNat::one();
    for prime in &params.primes {
        crt_product.mul_u128(prime.p.to_i64() as u128);
    }

    if !crt_width_is_safe::<F, D>(&crt_product, 1, rhs_abs_bound) {
        return None;
    }

    let mut lo = 1usize;
    let mut hi = 2usize;
    while crt_width_is_safe::<F, D>(&crt_product, hi, rhs_abs_bound) {
        lo = hi;
        let Some(next) = hi.checked_mul(2) else {
            if crt_width_is_safe::<F, D>(&crt_product, usize::MAX, rhs_abs_bound) {
                return Some(usize::MAX);
            }
            hi = usize::MAX;
            break;
        };
        hi = next;
    }

    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if crt_width_is_safe::<F, D>(&crt_product, mid, rhs_abs_bound) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    Some(lo)
}

fn require_safe_width<F, W, const K: usize, const D: usize>(
    params: &CrtNttParamSet<W, K, D>,
    rhs_abs_bound: u64,
    profile_id: &str,
    role: &str,
) -> Result<usize, AkitaError>
where
    F: CanonicalField,
    W: PrimeWidth,
{
    max_safe_crt_accumulation_width::<F, W, K, D>(params, rhs_abs_bound).ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "{profile_id} CRT capacity cannot fit a single {role} term for D={D} with rhs_abs_bound={rhs_abs_bound}"
        ))
    })
}

fn capacity_profile_from_params<F, W, const K: usize, const D: usize>(
    params: &CrtNttParamSet<W, K, D>,
    profile_id: &'static str,
    limb_bits: u32,
) -> Result<CrtI8CapacityProfile, AkitaError>
where
    F: CanonicalField,
    W: PrimeWidth,
{
    Ok(CrtI8CapacityProfile {
        profile_id,
        num_primes: K,
        limb_bits,
        max_i8_log_basis: MAX_I8_LOG_BASIS,
        balanced_digit_safe_width: require_safe_width::<F, W, K, D>(
            params,
            BALANCED_DIGIT_RHS_MAX_ABS,
            profile_id,
            "balanced i8 digit",
        )?,
        raw_i8_safe_width: require_safe_width::<F, W, K, D>(
            params,
            I8_RHS_MAX_ABS,
            profile_id,
            "raw signed-i8",
        )?,
    })
}

/// Validate and describe the universal i8 CRT capacity for the protocol
/// profile selected by `F,D`.
///
/// The setup artifact stores only an envelope, not the schedule levels that
/// originally produced it. This boundary therefore checks the worst supported
/// balanced digit (`log_basis = 6`) and raw signed-i8 roles for the selected
/// profile. Generated-table tests separately prove committed schedules stay
/// within these universal bounds.
pub(crate) fn selected_crt_i8_capacity_profile<F: CanonicalField, const D: usize>(
) -> Result<CrtI8CapacityProfile, AkitaError> {
    match select_crt_ntt_params::<F, D>()? {
        ProtocolCrtNttParams::Q32(params) => {
            capacity_profile_from_params::<F, _, Q32_NUM_PRIMES, D>(&params, "Q32/2xi32", 32)
        }
        ProtocolCrtNttParams::Q64(params) => {
            capacity_profile_from_params::<F, _, Q64_NUM_PRIMES, D>(&params, "Q64/3xi32", 32)
        }
        ProtocolCrtNttParams::Q128(params) => {
            capacity_profile_from_params::<F, _, Q128_NUM_PRIMES, D>(&params, "Q128/5xi32", 32)
        }
    }
}

/// Maximum absolute value of a setup (LHS matrix) coefficient once lifted to a
/// signed integer for CRT accumulation.
///
/// The cached NTT matrix is built by `CyclotomicCrtNtt::from_ring_with_params`,
/// which interprets every coefficient in centered form `(-q/2, q/2]` before
/// reducing into the CRT primes (so the lift matches the negacyclic subtraction
/// that produces negative values), and `to_ring_with_params` reconstructs with
/// signed Garner. The integer magnitude of any matrix coefficient is therefore
/// bounded by `floor(q/2)`, not the full modulus `q`. Basing the capacity bound
/// on `q` would double the chunking unnecessarily.
#[inline(always)]
fn setup_coeff_abs_bound<F: CanonicalField>() -> u128 {
    let modulus = (-F::one()).to_canonical_u128() + 1;
    modulus / 2
}

fn crt_width_is_safe<F: CanonicalField, const D: usize>(
    crt_product: &SmallNat,
    width: usize,
    rhs_abs_bound: u64,
) -> bool {
    let setup_abs_bound = setup_coeff_abs_bound::<F>();

    let mut lhs = SmallNat::one();
    lhs.mul_u128(2);
    lhs.mul_u128(width as u128);
    lhs.mul_u128(D as u128);
    lhs.mul_u128(setup_abs_bound);
    lhs.mul_u128(u128::from(rhs_abs_bound));
    lhs < *crt_product
}

pub(super) fn safe_crt_chunk_width<
    F: CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    params: &CrtNttParamSet<W, K, D>,
    full_width: usize,
    rhs_abs_bound: u64,
) -> Option<usize> {
    if full_width == 0 {
        return Some(0);
    }
    max_safe_crt_accumulation_width::<F, W, K, D>(params, rhs_abs_bound)
        .map(|safe_width| safe_width.min(full_width))
        .filter(|&chunk_width| chunk_width > 0)
}

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    use super::*;
    use akita_algebra::ntt::tables::{
        q128_primes, Q128_NUM_PRIMES, Q32_NUM_PRIMES, Q32_PRIMES, Q64_NUM_PRIMES, Q64_PRIMES,
    };
    use akita_field::{Fp64, Prime128Offset275, Prime32Offset99, Prime64Offset59};

    #[test]
    fn q128_digit_capacity_matches_expected_scale() {
        const D: usize = 64;
        let params = CrtNttParamSet::<i32, Q128_NUM_PRIMES, D>::new(q128_primes());
        let width = max_safe_crt_accumulation_width::<Prime128Offset275, i32, Q128_NUM_PRIMES, D>(
            &params,
            BALANCED_DIGIT_RHS_MAX_ABS,
        )
        .expect("one i8 term should fit");

        assert_eq!(width, 2047);
    }

    #[test]
    fn q128_balanced_digit_bound_recovers_chunk_width() {
        const D: usize = 64;
        let params = CrtNttParamSet::<i32, Q128_NUM_PRIMES, D>::new(q128_primes());
        let balanced_width =
            max_safe_crt_accumulation_width::<Prime128Offset275, i32, Q128_NUM_PRIMES, D>(
                &params,
                BALANCED_DIGIT_RHS_MAX_ABS,
            )
            .expect("one balanced digit term should fit");
        let full_i8_width =
            max_safe_crt_accumulation_width::<Prime128Offset275, i32, Q128_NUM_PRIMES, D>(
                &params,
                I8_RHS_MAX_ABS,
            )
            .expect("one full i8 term should fit");

        assert_eq!(balanced_width, 4 * full_i8_width + 3);
    }

    #[test]
    fn q128_rejects_unsafe_single_centered_term() {
        const D: usize = 128;
        let params = CrtNttParamSet::<i32, Q128_NUM_PRIMES, D>::new(q128_primes());
        let width = max_safe_crt_accumulation_width::<Prime128Offset275, i32, Q128_NUM_PRIMES, D>(
            &params, 32_768,
        );

        assert_eq!(width, None);
    }

    #[test]
    fn q32_digit_capacity_is_not_artificially_small() {
        const D: usize = 64;
        let params = CrtNttParamSet::<i32, Q32_NUM_PRIMES, D>::new(Q32_PRIMES);
        let width = max_safe_crt_accumulation_width::<Fp64<4294967197>, i32, Q32_NUM_PRIMES, D>(
            &params,
            BALANCED_DIGIT_RHS_MAX_ABS,
        )
        .expect("Q32 i8 path should have headroom");

        assert_eq!(width, 131_062);
    }

    fn assert_profile_widths(
        profile: CrtI8CapacityProfile,
        expected_balanced: usize,
        expected_raw: usize,
    ) {
        assert_eq!(
            profile.balanced_digit_safe_width, expected_balanced,
            "{} balanced safe width drifted",
            profile.profile_id
        );
        assert_eq!(
            profile.raw_i8_safe_width, expected_raw,
            "{} raw-i8 safe width drifted",
            profile.profile_id
        );
    }

    #[test]
    fn selected_capacity_profiles_match_golden_safe_widths() {
        assert_profile_widths(
            selected_crt_i8_capacity_profile::<Prime32Offset99, 256>().unwrap(),
            32_765,
            8_191,
        );
        assert_profile_widths(
            selected_crt_i8_capacity_profile::<Prime64Offset59, 256>().unwrap(),
            8_191,
            2_047,
        );
        assert_profile_widths(
            selected_crt_i8_capacity_profile::<Prime128Offset275, 256>().unwrap(),
            511,
            127,
        );
    }

    #[test]
    fn centered_zpre_capacity_matches_golden_widths() {
        const D: usize = 256;
        let q32_params = CrtNttParamSet::<i32, Q32_NUM_PRIMES, D>::new(Q32_PRIMES);
        assert_eq!(
            max_safe_crt_accumulation_width::<Prime32Offset99, i32, Q32_NUM_PRIMES, D>(
                &q32_params,
                32_768
            ),
            Some(31)
        );

        let q64_params = CrtNttParamSet::<i32, Q64_NUM_PRIMES, D>::new(Q64_PRIMES);
        assert_eq!(
            max_safe_crt_accumulation_width::<Prime64Offset59, i32, Q64_NUM_PRIMES, D>(
                &q64_params,
                32_768
            ),
            Some(7)
        );

        let q128_params = CrtNttParamSet::<i32, Q128_NUM_PRIMES, D>::new(q128_primes());
        assert_eq!(
            max_safe_crt_accumulation_width::<Prime128Offset275, i32, Q128_NUM_PRIMES, D>(
                &q128_params,
                32_768
            ),
            None
        );
    }

    #[test]
    fn reduced_profiles_fit_single_i8_terms_at_direct_ring_dims() {
        for profile in [
            selected_crt_i8_capacity_profile::<Prime32Offset99, 128>().unwrap(),
            selected_crt_i8_capacity_profile::<Prime32Offset99, 256>().unwrap(),
            selected_crt_i8_capacity_profile::<Prime64Offset59, 128>().unwrap(),
            selected_crt_i8_capacity_profile::<Prime64Offset59, 256>().unwrap(),
        ] {
            assert!(profile.balanced_digit_safe_width > 0, "{profile:?}");
            assert!(profile.raw_i8_safe_width > 0, "{profile:?}");
        }
    }

    #[test]
    fn selected_capacity_profile_matches_expected_dispatch_metadata() {
        let q32 = selected_crt_i8_capacity_profile::<Prime32Offset99, 64>().unwrap();
        assert_eq!(q32.profile_id, "Q32/2xi32");
        assert_eq!(q32.num_primes, Q32_NUM_PRIMES);
        assert_eq!(q32.limb_bits, 32);

        let q64 = selected_crt_i8_capacity_profile::<Prime64Offset59, 64>().unwrap();
        assert_eq!(q64.profile_id, "Q64/3xi32");
        assert_eq!(q64.num_primes, Q64_NUM_PRIMES);
        assert_eq!(q64.limb_bits, 32);

        let q128 = selected_crt_i8_capacity_profile::<Prime128Offset275, 64>().unwrap();
        assert_eq!(q128.profile_id, "Q128/5xi32");
        assert_eq!(q128.num_primes, Q128_NUM_PRIMES);
        assert_eq!(q128.limb_bits, 32);
    }

    #[test]
    fn profile_safe_widths_match_manual_params() {
        const D: usize = 256;
        let q64_params = CrtNttParamSet::<i32, Q64_NUM_PRIMES, D>::new(Q64_PRIMES);
        let q64 = selected_crt_i8_capacity_profile::<Prime64Offset59, D>().unwrap();
        assert_eq!(
            q64.balanced_digit_safe_width,
            max_safe_crt_accumulation_width::<Prime64Offset59, i32, Q64_NUM_PRIMES, D>(
                &q64_params,
                BALANCED_DIGIT_RHS_MAX_ABS
            )
            .unwrap()
        );
    }
}
