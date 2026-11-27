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
//! there is no implementation for the non-field towers over `Prime128Offset6A23801F`
//! or for arbitrary quadratic configurations.
//!
//! A protocol's session/version must bind the field, message lengths, and codec
//! rules; changing any of these changes the Fiat-Shamir decoding map.

use std::alloc::Layout;

#[cfg(feature = "binary")]
use jolt_field::CanonicalBytes;
#[cfg(any(feature = "solinas", feature = "bn254", feature = "binary"))]
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
#[cfg(feature = "transcript-keccak")]
use spongefish::instantiations::Keccak;
use spongefish::{DuplexSpongeInterface, VerificationError};

#[cfg(any(feature = "solinas", feature = "binary"))]
use crate::codec::WORD_BYTES;
#[cfg(feature = "transcript-blake2b")]
use crate::Blake2b512;
use crate::ChallengeSetSize;

use sealed::Sealed;

mod sealed {
    pub trait Sealed {}
}

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
/// non-field towers over `Prime128Offset6A23801F` and arbitrary quadratic configs
/// have no implementation.
///
/// Every message is a whole number of eight-byte words, so each squeeze starts
/// and ends on a word boundary of the sponge's rate. A pool of four-byte
/// candidates is rounded up to an even count. Squeezing is prefix-consistent,
/// so the larger pool starts with the same candidates: its first `len`
/// accepted values are those of the smaller pool whenever that one succeeds,
/// and it runs out only if the smaller pool does. The extra candidate only
/// adds rejection reserve.
///
/// # BN254 (`bn254`)
///
/// `Fr` and `Fq` reduce each consecutive 56-byte little-endian chunk
/// independently, whatever the message length. For M = 2^448 and
/// r = M mod p, reduction has exact conventional TV
/// δ = r(p-r)/(pM) ≤ p/(4M) < 2^254 / 2^450 = 2^-196, since r(p-r) ≤ p²/4.
/// For len independent reductions, TV ≤ len·δ. The [`MAX_CHALLENGE_BYTES`]
/// cap allows at most 299,593 < 2^19 elements, so the whole message has
/// TV < 2^-177, within the 2^-161 budget; the width would meet the budget up
/// to 2^35 elements.
///
/// # Binary fields (`binary`)
///
/// `F8`, `F64`, `F128`, and `F192` decode consecutive `NUM_BYTES` chunks with
/// checked little-endian decoding. Every correctly sized pattern is canonical,
/// making the map a bijection with zero bias and no exhaustion. An `F8`
/// message squeezes `ceil(len / 8)` words and decodes its first `len` bytes.
///
/// # Use outside this crate
///
/// The transcript uses this sealed trait to draw field challenges. External
/// fields with a runtime modulus can implement [`RuntimePrimeChallenge`] to
/// use the fixed-width reduction codec. Each codec's width and decoding are
/// part of the proof's wire format.
pub trait ChallengeCodec: Sized + Sealed {
    /// `|Self|`, the number of values one challenge can take. Proof-of-work
    /// difficulties are priced from it ([`grind_bits_for_loss`]).
    ///
    /// This method replaces the `CHALLENGE_SET_SIZE` associated constant.
    /// Callers must now read the size at runtime.
    ///
    /// [`grind_bits_for_loss`]: crate::grind_bits_for_loss
    fn challenge_set_size() -> ChallengeSetSize;

    /// Bytes squeezed for `len` elements, determined by the type and `len`.
    /// Returns `None` on arithmetic overflow or an unattainable rejection bound.
    /// The runtime-prime codec also returns `None` for an unrepresentable output
    /// vector layout. The transcript rejects binary draws that exceed
    /// [`MAX_CHALLENGE_BYTES`].
    fn challenge_bytes(len: usize) -> Option<usize>;

    /// Decodes exactly `challenge_bytes(len)` bytes in little-endian order.
    /// Returns `None` on a wrong or unsupported length, rejection-pool exhaustion,
    /// or output reservation failure in the runtime-prime codec.
    /// The runtime-prime codec reserves space for the result before reducing
    /// the elements.
    fn decode_challenges(bytes: &[u8], len: usize) -> Option<Vec<Self>>;

    /// Decodes one element as in `decode_challenges(bytes, 1)`.
    /// The runtime-prime codec calls
    /// [`RuntimePrimeChallenge::from_le_bytes_wide`] directly.
    fn decode_challenge(bytes: &[u8]) -> Option<Self>;
}

/// A byte-oriented duplex sponge suitable for drawing Fiat-Shamir challenges.
///
/// # Safety
///
/// At the claimed security level, the sponge must behave as an ideal duplex
/// (random oracle) for the protocol's fixed schedule, including across
/// adversarially chosen continuations of the same prefix. For proof of work,
/// the check bytes for distinct nonces must behave as independent uniform
/// bytes. The tape for the protected challenge must remain uniform conditioned
/// on a successful check and the outcomes of other nonce trials.
///
/// The sponge must bind the session identifier at initialization and preserve
/// domain separation for the transcript's unambiguously encoded statement
/// and message history. Replaying the same session and operations with the
/// same inputs must reproduce the same bytes on every platform. Cloning must
/// create an independent snapshot: operations on either copy leave the other's
/// state unchanged. An override of `squeeze_boxed` must return exactly the
/// requested bytes and advance the stream exactly as `squeeze` would.
///
/// The challenge codecs assume a uniform byte tape when bounding bias. If a
/// sponge biases that tape or lets a prover reuse challenges across domains,
/// the protocol's soundness bound no longer holds. Determinism lets the
/// verifier reproduce the prover's challenges.
#[expect(
    unsafe_code,
    reason = "implementors attest to the Fiat-Shamir sponge contract"
)]
pub unsafe trait UniformBytes: DuplexSpongeInterface<U = u8> + Sync {}

#[cfg(feature = "transcript-keccak")]
#[expect(unsafe_code, reason = "attests to the UniformBytes contract")]
// SAFETY: Spongefish initializes the Keccak duplex with the session identifier.
// The duplex construction models an ideal duplex under the Keccak permutation
// assumption.
unsafe impl UniformBytes for Keccak {}
#[cfg(feature = "transcript-blake2b")]
#[expect(unsafe_code, reason = "attests to the UniformBytes contract")]
// SAFETY: Spongefish's session-initialized hash bridge uses separate absorb
// and squeeze prefixes and emits the bytes of counter-indexed Blake2b digests.
// The transcript models Blake2b as a random oracle for these inputs.
unsafe impl UniformBytes for Blake2b512 {}

/// Largest challenge message, in squeezed bytes. Larger requests are rejected
/// before touching the sponge, so a length taken from the input cannot make
/// the verifier allocate without bound. Callers can compare a length against
/// it before drawing.
pub const MAX_CHALLENGE_BYTES: usize = 1 << 24;

#[cfg(feature = "solinas")]
fn ceil_log2(n: u128) -> u32 {
    n.saturating_sub(1).bit_width()
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
            // Whole words: candidates narrower than a word come in groups
            // that fill one. Registered bases have 4, 8 or 16 bytes.
            let per_word = (WORD_BYTES / F::NUM_BYTES).max(1);
            return candidates
                .checked_next_multiple_of(per_word)?
                .checked_mul(F::NUM_BYTES);
        }
        k = k.checked_add(1)?;
    }
}

/// The `len` challenges of a pooled message, in order; `None` if `bytes` is
/// not exactly that message. An item is `None` once the pool is exhausted.
#[cfg(feature = "solinas")]
fn pooled<F, E, const D: usize>(
    bytes: &[u8],
    len: usize,
) -> Option<impl Iterator<Item = Option<E>> + '_>
where
    F: PseudoMersenne,
    E: ExtField<F>,
{
    if D != E::DEGREE || bytes.len() != pool_bytes::<F>(len.checked_mul(D)?)? {
        return None;
    }
    let mask = u128::MAX.checked_shr(128u32.checked_sub(F::MODULUS_BITS)?)?;
    let mut accepted = bytes
        .chunks_exact(F::NUM_BYTES)
        .filter_map(move |candidate| {
            let mut word = [0u8; 16];
            word.get_mut(..F::NUM_BYTES)?.copy_from_slice(candidate);
            let masked = (u128::from_le_bytes(word) & mask).to_le_bytes();
            F::from_bytes_le_checked(masked.get(..F::NUM_BYTES)?)
        });
    Some((0..len).map(move |_| {
        let mut coordinates = [F::default(); D];
        for coordinate in &mut coordinates {
            *coordinate = accepted.next()?;
        }
        // ExtField::from_base_fn supplies exactly 0..DEGREE; D is checked above.
        Some(E::from_base_fn(|i| {
            coordinates.get(i).copied().unwrap_or_default()
        }))
    }))
}

#[cfg(feature = "solinas")]
fn decode_pool<F, E, const D: usize>(bytes: &[u8], len: usize) -> Option<Vec<E>>
where
    F: PseudoMersenne,
    E: ExtField<F>,
{
    let challenges = pooled::<F, E, D>(bytes, len)?;
    let mut result = Vec::with_capacity(len);
    for challenge in challenges {
        result.push(challenge?);
    }
    Some(result)
}

#[cfg(feature = "solinas")]
macro_rules! prime_codec {
    ($field:ident, $word:ty) => {
        impl<const P: $word> Sealed for $field<P> {}

        impl<const P: $word> ChallengeCodec for $field<P> {
            fn challenge_set_size() -> ChallengeSetSize {
                ChallengeSetSize::from_limbs(<Self as CanonicalEncoding>::CHALLENGE_SET_LIMBS)
            }

            fn challenge_bytes(len: usize) -> Option<usize> {
                pool_bytes::<Self>(len)
            }

            fn decode_challenges(bytes: &[u8], len: usize) -> Option<Vec<Self>> {
                decode_pool::<Self, Self, 1>(bytes, len)
            }

            fn decode_challenge(bytes: &[u8]) -> Option<Self> {
                pooled::<Self, Self, 1>(bytes, 1)?.next().flatten()
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
// Prime128Offset6A23801F from all towers, and the quadratic impl fixes TwoNr.
#[cfg(feature = "solinas")]
trait TowerBase: PseudoMersenne + CanonicalEncoding {}

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
        impl<F: TowerBase> Sealed for $extension<F> {}

        impl<F: TowerBase> ChallengeCodec for $extension<F> {
            fn challenge_set_size() -> ChallengeSetSize {
                ChallengeSetSize::from_limbs(F::CHALLENGE_SET_LIMBS).pow(Self::DEGREE as u32)
            }

            fn challenge_bytes(len: usize) -> Option<usize> {
                pool_bytes::<F>(len.checked_mul(Self::DEGREE)?)
            }

            fn decode_challenges(bytes: &[u8], len: usize) -> Option<Vec<Self>> {
                decode_pool::<F, Self, $degree>(bytes, len)
            }

            fn decode_challenge(bytes: &[u8]) -> Option<Self> {
                pooled::<F, Self, $degree>(bytes, 1)?.next().flatten()
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

/// Bytes per BN254 element in every message: seven words.
#[cfg(feature = "bn254")]
const BN254_ELEMENT_BYTES: usize = 56;

#[cfg(feature = "bn254")]
macro_rules! reduction_codec {
    ($($field:ident),+ $(,)?) => {$(
        impl Sealed for $field {}

        impl ChallengeCodec for $field {
            fn challenge_set_size() -> ChallengeSetSize {
                ChallengeSetSize::from_limbs(<Self as CanonicalEncoding>::CHALLENGE_SET_LIMBS)
            }

            fn challenge_bytes(len: usize) -> Option<usize> {
                len.checked_mul(BN254_ELEMENT_BYTES)
            }

            fn decode_challenge(bytes: &[u8]) -> Option<Self> {
                (bytes.len() == BN254_ELEMENT_BYTES).then(|| Self::from_bytes_le_reduced(bytes))
            }

            fn decode_challenges(bytes: &[u8], len: usize) -> Option<Vec<Self>> {
                if bytes.len() != Self::challenge_bytes(len)? {
                    return None;
                }
                let mut result = Vec::with_capacity(len);
                for chunk in bytes.chunks_exact(BN254_ELEMENT_BYTES) {
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
        impl Sealed for $field {}

        impl ChallengeCodec for $field {
            fn challenge_set_size() -> ChallengeSetSize {
                ChallengeSetSize::from_limbs(<Self as CanonicalEncoding>::CHALLENGE_SET_LIMBS)
            }

            fn challenge_bytes(len: usize) -> Option<usize> {
                len.checked_mul(<Self as CanonicalBytes>::NUM_BYTES)?
                    .checked_next_multiple_of(WORD_BYTES)
            }

            fn decode_challenge(bytes: &[u8]) -> Option<Self> {
                if bytes.len() != Self::challenge_bytes(1)? {
                    return None;
                }
                let width = <Self as CanonicalBytes>::NUM_BYTES;
                Self::from_bytes_le_checked(bytes.get(..width)?)
            }

            fn decode_challenges(bytes: &[u8], len: usize) -> Option<Vec<Self>> {
                if bytes.len() != Self::challenge_bytes(len)? {
                    return None;
                }
                let width = <Self as CanonicalBytes>::NUM_BYTES;
                let decoded = bytes.get(..len.checked_mul(width)?)?;
                let mut result = Vec::with_capacity(len);
                for chunk in decoded.chunks_exact(width) {
                    result.push(Self::from_bytes_le_checked(chunk)?);
                }
                Some(result)
            }
        }
    )+};
}

#[cfg(feature = "binary")]
binary_codec!(F8, F64, F128, F192);

/// Number of bytes reduced modulo the runtime prime for each challenge.
pub const RUNTIME_PRIME_WIDE_BYTES: usize = 40;

/// A field with a prime modulus `q < 2^128` chosen at runtime.
///
/// The protocol must bind `q` to the session or public statement. Both
/// parties must use the same modulus throughout the proof, including when
/// pricing proof of work.
///
/// The codec reduces one 320-bit little-endian integer modulo `q` for each
/// challenge. For uniform input the total variation distance from uniform
/// is at most `q / 2^320 < 2^-192` per element. [`MAX_CHALLENGE_BYTES`]
/// permits fewer than 2^19 elements per message, giving a total distance
/// below 2^-173.
pub trait RuntimePrimeChallenge: Sized {
    /// The current proof's prime modulus.
    fn modulus() -> u128;

    /// Returns the field element represented by `bytes` modulo [`Self::modulus`].
    fn from_le_bytes_wide(bytes: &[u8; RUNTIME_PRIME_WIDE_BYTES]) -> Self;
}

impl<T: RuntimePrimeChallenge> Sealed for T {}

impl<T: RuntimePrimeChallenge> ChallengeCodec for T {
    fn challenge_set_size() -> ChallengeSetSize {
        ChallengeSetSize::field_order(T::modulus(), 1).unwrap_or(ChallengeSetSize::power_of_two(0))
    }

    fn challenge_bytes(len: usize) -> Option<usize> {
        let _layout = Layout::array::<T>(len).ok()?;
        len.checked_mul(RUNTIME_PRIME_WIDE_BYTES)
    }

    fn decode_challenge(bytes: &[u8]) -> Option<Self> {
        Some(T::from_le_bytes_wide(bytes.try_into().ok()?))
    }

    fn decode_challenges(bytes: &[u8], len: usize) -> Option<Vec<Self>> {
        if bytes.len() != Self::challenge_bytes(len)? {
            return None;
        }
        let (elements, _) = bytes.as_chunks::<RUNTIME_PRIME_WIDE_BYTES>();
        let mut result = Vec::new();
        result.try_reserve_exact(len).ok()?;
        result.extend(elements.iter().map(T::from_le_bytes_wide));
        Some(result)
    }
}

/// Bytes squeezed for a message of `len` elements of `F`.
///
/// # Errors
/// Returns `VerificationError` when `F` does not support `len` or the message
/// would exceed [`MAX_CHALLENGE_BYTES`].
pub(crate) fn message_bytes<F: ChallengeCodec>(len: usize) -> Result<usize, VerificationError> {
    F::challenge_bytes(len)
        .filter(|&bytes| bytes <= MAX_CHALLENGE_BYTES)
        .ok_or(VerificationError)
}

#[cfg(all(test, feature = "solinas", feature = "transcript-keccak"))]
pub(crate) mod test_codec {
    use super::{ChallengeCodec, ChallengeSetSize, Prime64Offset59, Sealed};

    pub(crate) struct ExhaustedPrime64 {
        _value: Prime64Offset59,
    }

    impl Sealed for ExhaustedPrime64 {}

    impl ChallengeCodec for ExhaustedPrime64 {
        fn challenge_set_size() -> ChallengeSetSize {
            Prime64Offset59::challenge_set_size()
        }

        fn challenge_bytes(len: usize) -> Option<usize> {
            Prime64Offset59::challenge_bytes(len)
        }

        fn decode_challenges(bytes: &[u8], len: usize) -> Option<Vec<Self>> {
            let values = Prime64Offset59::decode_challenges(&vec![u8::MAX; bytes.len()], len)?;
            Some(
                values
                    .into_iter()
                    .map(|value| Self { _value: value })
                    .collect(),
            )
        }

        fn decode_challenge(bytes: &[u8]) -> Option<Self> {
            let value = Prime64Offset59::decode_challenge(&vec![u8::MAX; bytes.len()])?;
            Some(Self { _value: value })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ChallengeCodec, RuntimePrimeChallenge, RUNTIME_PRIME_WIDE_BYTES};

    struct WideChallenge {
        _padding: [u8; 8191],
        _parity: bool,
    }

    impl RuntimePrimeChallenge for WideChallenge {
        fn modulus() -> u128 {
            2
        }

        fn from_le_bytes_wide(&[low, ..]: &[u8; RUNTIME_PRIME_WIDE_BYTES]) -> Self {
            Self {
                _padding: [0; 8191],
                _parity: low & 1 == 1,
            }
        }
    }

    #[test]
    fn runtime_prime_tape_size_requires_a_representable_output_layout() {
        let max_len = isize::MAX.unsigned_abs() / size_of::<WideChallenge>();
        assert_eq!(
            WideChallenge::challenge_bytes(max_len),
            Some(max_len * RUNTIME_PRIME_WIDE_BYTES)
        );
        assert_eq!(WideChallenge::challenge_bytes(max_len + 1), None);
    }
}
