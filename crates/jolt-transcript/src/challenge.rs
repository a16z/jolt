//! Fixed-tape, little-endian field challenge codecs.
//!
//! Each call is one verifier message: all elements share a conventional total
//! variation (TV) budget of at most 2^-161. The soundness theorem charges twice
//! TV, hence at most 2^-160 per message. Rejection codecs return an explicit
//! failure with probability at most 2^-161; conditioned on success their output
//! is exactly uniform. Equivalently, replacing failure by any fixed output
//! gives TV at most that failure probability. Callers instead abort on failure.
//!
//! Extension implementations cover `Ext2` (the `TwoNr` configuration), `FpExt4`,
//! and `FpExt8` over the nine registered pseudo-Mersenne bases only. In particular,
//! there is no implementation for the non-field towers over `Prime128OffsetA7F7`
//! or for arbitrary quadratic configurations.
//!
//! A protocol's session/version must bind the field, message lengths, and codec
//! rules; changing any of these changes the Fiat-Shamir decoding map.

#[cfg(any(feature = "bn254", feature = "binary"))]
use jolt_field::CanonicalEncoding;
#[cfg(feature = "solinas")]
use jolt_field::{
    Ext2, ExtField, Fp128, Fp32, Fp64, FpExt4, FpExt8, Prime128Offset275, Prime24Offset3,
    Prime30Offset35, Prime31Offset19, Prime32Offset99, Prime40Offset195, Prime48Offset59,
    Prime56Offset27, Prime64Offset59, PseudoMersenne,
};
#[cfg(feature = "bn254")]
use jolt_field::{Fq, Fr};
#[cfg(feature = "binary")]
use jolt_field::{F128, F192, F64, F8};
#[cfg(feature = "spongefish")]
use spongefish::VerificationError;

/// Decodes a fixed-length uniform byte tape into one message of field elements.
///
/// Implementations must budget conventional TV ≤ 2^-161 for the entire message,
/// including rejection-pool exhaustion, rather than for each element separately.
/// No bytes are drawn adaptively. Scalar draws use `len == 1` of this same codec.
/// Empty messages consume zero bytes and decode to an empty vector.
///
/// # Pseudo-Mersenne fields (`solinas`)
///
/// Candidates have `NUM_BYTES` little-endian bytes, masked to `MODULUS_BITS`;
/// checked decoding accepts exactly the representatives below the prime.
/// Masking makes each candidate uniform on [0, 2^b); acceptance gives a
/// uniform element of [0, p). Accepted values are independent of acceptance
/// indicators, so the first `len` accepted values are jointly uniform even
/// conditioned on pool success.
///
/// With rejection probability ρ = c/2^b ≤ 2^-R, R = b - ceil(log2(c)),
/// failure among B = len+k candidates needs at least k+1 rejections. A union
/// bound gives Pr[fail] ≤ C(B,k+1) ρ^(k+1) ≤ 2^(-(k+1)(R-L)), where
/// L = ceil(log2(B)). Choose the least k with (k+1)(R-L) ≥ 161. Once L ≥ R,
/// increasing k cannot make the bound attainable. All arithmetic is integral.
///
/// Extensions use one base pool of `len * DEGREE` coordinates, grouped in the
/// canonical basis via `ExtField::from_base_fn`. Implementations cover `Ext2`
/// with `TwoNr`, `FpExt4`, and `FpExt8` over the nine registered bases only;
/// non-field towers over `Prime128OffsetA7F7` and arbitrary quadratic configs
/// have no implementation.
///
/// # BN254 (`bn254`)
///
/// `Fr` and `Fq` reduce each consecutive w-byte little-endian chunk independently.
/// For M = 2^(8w) and r = M mod p, reduction has exact conventional TV
/// δ = r(p-r)/(pM) ≤ p/(4M), since r(p-r) ≤ p²/4.
/// For len independent reductions, TV ≤ len·δ. With
/// w = ceil((MODULUS_BITS + 159 + ceil(log2(len)))/8),
/// len·2δ ≤ len·2^MODULUS_BITS / 2^(8w+1) ≤ 2^-160,
/// so the whole message has conventional TV ≤ 2^-161. For BN254's 254-bit
/// modulus this gives 52 bytes per element for len 1–8, then 53 for len 9–128.
///
/// # Binary fields (`binary`)
///
/// `F8`, `F64`, `F128`, and `F192` decode consecutive `NUM_BYTES` chunks with
/// checked little-endian decoding. Every correctly sized pattern is canonical,
/// making the map a bijection with zero bias and no exhaustion.
pub trait ChallengeCodec: Sized {
    /// Bytes squeezed for `len` elements; depends only on the type and `len`.
    /// Returns `None` on arithmetic overflow or an unattainable rejection bound.
    fn challenge_bytes(len: usize) -> Option<usize>;

    /// Decodes exactly `challenge_bytes(len)` bytes in little-endian order.
    /// Returns `None` on a wrong/unsupported length or rejection-pool exhaustion.
    /// Allocates only the returned vector, with capacity `len`.
    fn decode_challenges(bytes: &[u8], len: usize) -> Option<Vec<Self>>;
}

#[cfg(any(feature = "solinas", feature = "bn254"))]
fn ceil_log2(n: u128) -> u32 {
    u128::BITS - n.saturating_sub(1).leading_zeros()
}

#[cfg(feature = "solinas")]
#[expect(non_snake_case, reason = "R and L match the pooled-rejection bound")]
fn pool_bytes<F: PseudoMersenne>(len: usize) -> Option<usize> {
    if len == 0 {
        return Some(0);
    }
    let R = F::MODULUS_BITS.checked_sub(ceil_log2(F::OFFSET))?;
    let mut k = 0usize;
    loop {
        let candidates = len.checked_add(k)?;
        let L = ceil_log2(u128::try_from(candidates).ok()?);
        let margin = R.checked_sub(L).filter(|&margin| margin > 0)?;
        if k.checked_add(1)?
            .checked_mul(usize::try_from(margin).ok()?)?
            >= 161
        {
            return candidates.checked_mul(F::NUM_BYTES);
        }
        k = k.checked_add(1)?;
    }
}

#[cfg(feature = "solinas")]
fn decode_pool<F, E, const D: usize>(bytes: &[u8], len: usize) -> Option<Vec<E>>
where
    F: PseudoMersenne,
    E: ExtField<F>,
{
    if D != E::DEGREE || bytes.len() != pool_bytes::<F>(len.checked_mul(D)?)? {
        return None;
    }
    let mask = u128::MAX.checked_shr(128u32.checked_sub(F::MODULUS_BITS)?)?;
    let mut accepted = bytes.chunks_exact(F::NUM_BYTES).filter_map(|candidate| {
        let mut word = [0u8; 16];
        word.get_mut(..F::NUM_BYTES)?.copy_from_slice(candidate);
        let masked = (u128::from_le_bytes(word) & mask).to_le_bytes();
        F::from_bytes_le_checked(masked.get(..F::NUM_BYTES)?)
    });
    let mut result = Vec::with_capacity(len);
    for _ in 0..len {
        let mut coordinates = [F::default(); D];
        for coordinate in &mut coordinates {
            *coordinate = accepted.next()?;
        }
        // ExtField::from_base_fn supplies exactly 0..DEGREE; D is checked above.
        result.push(E::from_base_fn(|i| {
            coordinates.get(i).copied().unwrap_or_default()
        }));
    }
    Some(result)
}

#[cfg(feature = "solinas")]
macro_rules! prime_codec {
    ($field:ident, $word:ty) => {
        impl<const P: $word> ChallengeCodec for $field<P> {
            fn challenge_bytes(len: usize) -> Option<usize> {
                pool_bytes::<Self>(len)
            }

            fn decode_challenges(bytes: &[u8], len: usize) -> Option<Vec<Self>> {
                decode_pool::<Self, Self, 1>(bytes, len)
            }
        }
    };
}

#[cfg(feature = "solinas")]
prime_codec!(Fp32, u32);
#[cfg(feature = "solinas")]
prime_codec!(Fp64, u64);
#[cfg(feature = "solinas")]
prime_codec!(Fp128, u128);

// Registered bases have p ≡ ±3 (mod 8). This sealed list excludes
// Prime128OffsetA7F7 from all towers, and the quadratic impl fixes TwoNr.
#[cfg(feature = "solinas")]
trait TowerBase: PseudoMersenne {}

#[cfg(feature = "solinas")]
macro_rules! tower_bases {
    ($($field:ty),+ $(,)?) => {$(impl TowerBase for $field {})+};
}

#[cfg(feature = "solinas")]
tower_bases!(
    Prime24Offset3,
    Prime30Offset35,
    Prime31Offset19,
    Prime32Offset99,
    Prime40Offset195,
    Prime48Offset59,
    Prime56Offset27,
    Prime64Offset59,
    Prime128Offset275,
);

#[cfg(feature = "solinas")]
macro_rules! extension_codec {
    ($extension:ident, $degree:literal) => {
        impl<F: TowerBase> ChallengeCodec for $extension<F> {
            fn challenge_bytes(len: usize) -> Option<usize> {
                pool_bytes::<F>(len.checked_mul(Self::DEGREE)?)
            }

            fn decode_challenges(bytes: &[u8], len: usize) -> Option<Vec<Self>> {
                decode_pool::<F, Self, $degree>(bytes, len)
            }
        }
    };
}

#[cfg(feature = "solinas")]
extension_codec!(Ext2, 2);
#[cfg(feature = "solinas")]
extension_codec!(FpExt4, 4);
#[cfg(feature = "solinas")]
extension_codec!(FpExt8, 8);

#[cfg(feature = "bn254")]
fn reduction_width<F: CanonicalEncoding>(len: usize) -> Option<usize> {
    let bits = F::MODULUS_BITS
        .checked_add(159)?
        .checked_add(ceil_log2(u128::try_from(len).ok()?))?;
    usize::try_from(bits.checked_add(7)?.checked_div(8)?).ok()
}

#[cfg(feature = "bn254")]
macro_rules! reduction_codec {
    ($($field:ty),+ $(,)?) => {$(
        impl ChallengeCodec for $field {
            fn challenge_bytes(len: usize) -> Option<usize> {
                len.checked_mul(reduction_width::<Self>(len)?)
            }

            fn decode_challenges(bytes: &[u8], len: usize) -> Option<Vec<Self>> {
                if bytes.len() != Self::challenge_bytes(len)? {
                    return None;
                }
                let mut result = Vec::with_capacity(len);
                for chunk in bytes.chunks_exact(reduction_width::<Self>(len)?) {
                    result.push(Self::from_bytes_le_reduced(chunk));
                }
                Some(result)
            }
        }
    )+};
}

#[cfg(feature = "bn254")]
reduction_codec!(Fr, Fq);

// F8/F64/F128 checked decoding wraps the entire little-endian integer;
// F192 checked decoding accepts every 24-byte tuple of three F64 coordinates.
// Thus these maps are bijections and have zero bias (binary/{f8,f64,f128,f192}.rs).
#[cfg(feature = "binary")]
macro_rules! binary_codec {
    ($($field:ty),+ $(,)?) => {$(
        impl ChallengeCodec for $field {
            fn challenge_bytes(len: usize) -> Option<usize> {
                len.checked_mul(<Self as jolt_field::CanonicalBytes>::NUM_BYTES)
            }

            fn decode_challenges(bytes: &[u8], len: usize) -> Option<Vec<Self>> {
                if bytes.len() != Self::challenge_bytes(len)? {
                    return None;
                }
                let mut result = Vec::with_capacity(len);
                for chunk in bytes.chunks_exact(<Self as jolt_field::CanonicalBytes>::NUM_BYTES) {
                    result.push(Self::from_bytes_le_checked(chunk)?);
                }
                Some(result)
            }
        }
    )+};
}

#[cfg(feature = "binary")]
binary_codec!(F8, F64, F128, F192);

#[cfg(feature = "spongefish")]
pub(crate) fn decode_message<F: ChallengeCodec>(
    bytes: &[u8],
    len: usize,
) -> Result<Vec<F>, VerificationError> {
    F::decode_challenges(bytes, len).ok_or(VerificationError)
}

#[cfg(all(test, feature = "spongefish", feature = "solinas"))]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::{decode_message, ChallengeCodec, Prime64Offset59};

    #[test]
    fn exhaustion_is_a_verification_error() {
        let bytes = vec![0xff; Prime64Offset59::challenge_bytes(1).unwrap()];
        assert!(decode_message::<Prime64Offset59>(&bytes, 1).is_err());
    }
}
