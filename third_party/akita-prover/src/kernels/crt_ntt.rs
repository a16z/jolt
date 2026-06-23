//! Protocol-facing CRT+NTT parameter dispatch and matrix caching.

use akita_algebra::ntt::prime::PrimeWidth;
use akita_algebra::ntt::tables::{
    q128_primes, Q128_MODULUS, Q128_NUM_PRIMES, Q32_MODULUS, Q32_NUM_PRIMES, Q32_PRIMES,
    Q64_MODULUS, Q64_NUM_PRIMES, Q64_PRIMES,
};
use akita_algebra::ring::{CrtNttParamSet, CyclotomicCrtNtt};
#[allow(unused_imports)]
use akita_field::parallel::*;
use akita_field::{cfg_iter, AkitaError, CanonicalField, FieldCore, PseudoMersenneField};
use akita_field::{Prime128Offset159, Prime128Offset2355, Prime128OffsetA7F7};

use akita_types::RingMatrixView;

/// Supported protocol CRT+NTT parameter families.
#[derive(Clone)]
#[allow(missing_docs, clippy::large_enum_variant)]
pub enum ProtocolCrtNttParams<const D: usize> {
    Q32(CrtNttParamSet<i32, Q32_NUM_PRIMES, D>),
    Q64(CrtNttParamSet<i32, Q64_NUM_PRIMES, D>),
    Q128(CrtNttParamSet<i32, Q128_NUM_PRIMES, D>),
}

/// Select a CRT+NTT parameter set from field modulus and ring degree.
///
/// Dispatch policy:
/// - `q <= 2^32-99` and `D <= 256`: Q32 (`i32`, K=2)
/// - `q <= 2^64-59` and `D <= 256`: Q64 (`i32`, K=3)
/// - `q ∈ { 2^128-275, 2^128-159, 2^128-2355, 2^128-2^32+22537 }` and
///   `D <= 256`: Q128 (`i32`, K=5)
/// - otherwise: explicit setup error
///
/// # Errors
///
/// Returns an error if `D` is unsupported or no CRT/NTT parameter family
/// matches the field modulus.
pub fn select_crt_ntt_params<F: CanonicalField, const D: usize>(
) -> Result<ProtocolCrtNttParams<D>, AkitaError> {
    if !matches!(D, 32 | 64 | 128 | 256) {
        return Err(AkitaError::InvalidSetup(format!(
            "CRT+NTT supports ring degree in {{32, 64, 128, 256}}, got D={D}"
        )));
    }

    let modulus = detect_field_modulus::<F>();
    let split_only_q128_modulus =
        u128::MAX - (<Prime128Offset159 as PseudoMersenneField>::MODULUS_OFFSET - 1);
    let ntt_q128_modulus =
        u128::MAX - (<Prime128Offset2355 as PseudoMersenneField>::MODULUS_OFFSET - 1);
    let a7f7_q128_modulus =
        u128::MAX - (<Prime128OffsetA7F7 as PseudoMersenneField>::MODULUS_OFFSET - 1);

    if modulus <= Q32_MODULUS as u128 {
        return Ok(ProtocolCrtNttParams::Q32(CrtNttParamSet::new(Q32_PRIMES)));
    }

    if modulus <= Q64_MODULUS as u128 {
        return Ok(ProtocolCrtNttParams::Q64(CrtNttParamSet::new(Q64_PRIMES)));
    }

    if modulus == Q128_MODULUS
        || modulus == split_only_q128_modulus
        || modulus == ntt_q128_modulus
        || modulus == a7f7_q128_modulus
    {
        return Ok(ProtocolCrtNttParams::Q128(CrtNttParamSet::new(
            q128_primes(),
        )));
    }

    Err(AkitaError::InvalidSetup(format!(
        "no CRT+NTT parameter set for modulus {modulus} and D={D}; supported ranges: <= {Q64_MODULUS} (with Q32/Q64 dispatch) or q in {{{Q128_MODULUS}, {split_only_q128_modulus}, {ntt_q128_modulus}, {a7f7_q128_modulus}}}"
    )))
}

fn detect_field_modulus<F: CanonicalField>() -> u128 {
    (-F::one()).to_canonical_u128() + 1
}

/// Pre-converted CRT+NTT cache for a flat 1D matrix, keyed by parameter family.
///
/// Stores both negacyclic (for mat-vec) and cyclic (for quotient) representations
/// as flat contiguous vectors. Callers provide `(num_rows, num_cols)` when
/// accessing elements, treating the flat data as a row-major matrix with
/// stride = `num_cols`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs, clippy::large_enum_variant)]
pub enum NttSlotCache<const D: usize> {
    /// 32-bit CRT primes.
    Q32 {
        neg: Vec<CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, D>>,
        cyc: Vec<CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, D>>,
        params: CrtNttParamSet<i32, Q32_NUM_PRIMES, D>,
    },
    /// 64-bit CRT primes.
    Q64 {
        neg: Vec<CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, D>>,
        cyc: Vec<CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, D>>,
        params: CrtNttParamSet<i32, Q64_NUM_PRIMES, D>,
    },
    /// 128-bit CRT primes.
    Q128 {
        neg: Vec<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>,
        cyc: Vec<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>,
        params: CrtNttParamSet<i32, Q128_NUM_PRIMES, D>,
    },
}

fn convert_flat_pair<F, W, const K: usize, const D: usize>(
    mat: RingMatrixView<'_, F, D>,
    params: &CrtNttParamSet<W, K, D>,
) -> (
    Vec<CyclotomicCrtNtt<W, K, D>>,
    Vec<CyclotomicCrtNtt<W, K, D>>,
)
where
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
{
    cfg_iter!(mat.as_slice())
        .map(|ring| CyclotomicCrtNtt::from_ring_pair_with_params(ring, params))
        .unzip()
}

/// Build an NTT slot cache for a matrix view (flat 1D storage).
///
/// # Errors
///
/// Returns an error if no CRT+NTT parameter set matches the field modulus and ring degree.
#[tracing::instrument(skip_all, name = "build_ntt_slot")]
pub fn build_ntt_slot<F: FieldCore + CanonicalField, const D: usize>(
    mat: RingMatrixView<'_, F, D>,
) -> Result<NttSlotCache<D>, AkitaError> {
    let params = select_crt_ntt_params::<F, D>()?;
    Ok(build_ntt_slot_from_params(mat, params))
}

fn build_ntt_slot_from_params<F: FieldCore + CanonicalField, const D: usize>(
    mat: RingMatrixView<'_, F, D>,
    params: ProtocolCrtNttParams<D>,
) -> NttSlotCache<D> {
    match params {
        ProtocolCrtNttParams::Q32(p) => {
            let (neg, cyc) = convert_flat_pair(mat, &p);
            NttSlotCache::Q32 {
                neg,
                cyc,
                params: p,
            }
        }
        ProtocolCrtNttParams::Q64(p) => {
            let (neg, cyc) = convert_flat_pair(mat, &p);
            NttSlotCache::Q64 {
                neg,
                cyc,
                params: p,
            }
        }
        ProtocolCrtNttParams::Q128(p) => {
            let (neg, cyc) = convert_flat_pair(mat, &p);
            NttSlotCache::Q128 {
                neg,
                cyc,
                params: p,
            }
        }
    }
}

impl<const D: usize> NttSlotCache<D> {
    /// Total number of NTT elements stored in this cache.
    pub fn total_elements(&self) -> usize {
        match self {
            NttSlotCache::Q32 { neg, .. } => neg.len(),
            NttSlotCache::Q64 { neg, .. } => neg.len(),
            NttSlotCache::Q128 { neg, .. } => neg.len(),
        }
    }

    /// In-memory byte footprint of the negacyclic plus cyclic NTT slot
    /// vectors. Diagnostic surface for the profiler / bench report; the cache
    /// is the dominant prepared-setup allocation and is much larger than the
    /// plain setup vector it is built from.
    pub fn cache_bytes(&self) -> usize {
        match self {
            NttSlotCache::Q32 { neg, cyc, .. } => {
                (neg.len() + cyc.len())
                    * core::mem::size_of::<CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, D>>()
            }
            NttSlotCache::Q64 { neg, cyc, .. } => {
                (neg.len() + cyc.len())
                    * core::mem::size_of::<CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, D>>()
            }
            NttSlotCache::Q128 { neg, cyc, .. } => {
                (neg.len() + cyc.len())
                    * core::mem::size_of::<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>()
            }
        }
    }
}

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    use super::*;
    use akita_field::{
        Prime128Offset159, Prime128Offset2355, Prime128Offset275, Prime128OffsetA7F7,
        Prime32Offset99, Prime64Offset59,
    };

    fn assert_selects_q32_params<F: CanonicalField, const D: usize>() {
        assert!(matches!(
            select_crt_ntt_params::<F, D>(),
            Ok(ProtocolCrtNttParams::Q32(_))
        ));
    }

    fn assert_selects_q64_params<F: CanonicalField, const D: usize>() {
        assert!(matches!(
            select_crt_ntt_params::<F, D>(),
            Ok(ProtocolCrtNttParams::Q64(_))
        ));
    }

    fn assert_selects_q128_params<F: CanonicalField, const D: usize>() {
        assert!(matches!(
            select_crt_ntt_params::<F, D>(),
            Ok(ProtocolCrtNttParams::Q128(_))
        ));
    }

    #[test]
    fn selects_q32_params_across_supported_ring_dims() {
        assert_selects_q32_params::<Prime32Offset99, 32>();
        assert_selects_q32_params::<Prime32Offset99, 64>();
        assert_selects_q32_params::<Prime32Offset99, 128>();
        assert_selects_q32_params::<Prime32Offset99, 256>();
    }

    #[test]
    fn selects_q64_params_across_supported_ring_dims() {
        assert_selects_q64_params::<Prime64Offset59, 32>();
        assert_selects_q64_params::<Prime64Offset59, 64>();
        assert_selects_q64_params::<Prime64Offset59, 128>();
        assert_selects_q64_params::<Prime64Offset59, 256>();
    }

    #[test]
    fn rejects_ring_degrees_above_256() {
        assert!(matches!(
            select_crt_ntt_params::<Prime32Offset99, 512>(),
            Err(AkitaError::InvalidSetup(_))
        ));
    }

    #[test]
    fn selects_q128_params_for_prime275_across_small_protocol_ring_dims() {
        assert_selects_q128_params::<Prime128Offset275, 32>();
        assert_selects_q128_params::<Prime128Offset275, 64>();
        assert_selects_q128_params::<Prime128Offset275, 128>();
        assert_selects_q128_params::<Prime128Offset275, 256>();
    }

    #[test]
    fn selects_q128_params_for_split_only_prime159() {
        assert_selects_q128_params::<Prime128Offset159, 32>();
    }

    #[test]
    fn selects_q128_params_for_prime_a7f7_across_small_protocol_ring_dims() {
        assert_selects_q128_params::<Prime128OffsetA7F7, 32>();
        assert_selects_q128_params::<Prime128OffsetA7F7, 64>();
        assert_selects_q128_params::<Prime128OffsetA7F7, 128>();
    }

    #[test]
    fn selects_q128_params_for_prime2355_across_small_protocol_ring_dims() {
        assert_selects_q128_params::<Prime128Offset2355, 32>();
        assert_selects_q128_params::<Prime128Offset2355, 64>();
        assert_selects_q128_params::<Prime128Offset2355, 128>();
    }
}
