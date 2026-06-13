//! Local codecs for absorbing / decoding Jolt-native messages.
//!
//! **Byte sponges** (`U = u8`): field-element codecs come from spongefish's
//! `ark-ff` feature (`spongefish::Encoding<[u8]>` is implemented for every
//! `ark_ff::Fp<C, N>` using big-endian canonical encoding per RFC8017).
//! 128-bit-truncated challenges decode through spongefish's built-in `u128`
//! codec directly; see [`crate::prover::OptimizedChallenge`]. [`BytesMsg`] is
//! the length-prefixed byte-string framing spongefish 0.6 does not provide.
//! All `[u8]`-domain codecs here are untouched by the field-aligned redesign.
//!
//! **Field-unit Poseidon sponge** (`U = Fr`, `transcript-poseidon`): spongefish
//! ships no `Encoding<[Fr]>`/`Decoding<[Fr]>` codecs (and the orphan rule
//! blocks implementing them on foreign types like `Fr` itself), so this module
//! defines the typed message vocabulary of
//! `specs/transpiler-optimization-spec.md` §4.2–4.3:
//!
//! - [`RawBytesMsg`] — byte-rule message: `L` bytes ↦ `[Fr(2L), ceil(L/31)`
//!   31-byte-LE chunk units`]` (each chunk < 2²⁴⁸ < r, so injective).
//! - [`FieldFrameMsg`] — count-led field frame: `k` elements ↦
//!   `[Fr(2k+1), e₁, …, e_k]` native units.
//! - [`CommitmentsMsg`] — a NARG frame of `k` commitments absorbed as a
//!   leading **frame count unit** `Fr(2k+1)` followed by `k`
//!   **per-commitment byte-rule groups** (one Dory GT = 384 canonical bytes ↦
//!   `[Fr(768), 13 chunks]` = 14 units); an empty frame is the `Fr(1)`
//!   count-led case, distinct from the empty byte message `[Fr(0)]`, so the
//!   data-dependent advice presence frame stays count-led.
//! - [`NativeChallenge`] — one native `Fr` squeeze (`Decoding<[Fr]>` with a
//!   one-unit `Repr`, identity decode) = exactly one permute.
//!
//! Every encoding zero-pads each tagged group to an even unit count, so each
//! message occupies whole permute pairs and message boundaries bind. The
//! `2L` (even) / `2k+1` (odd) leading-tag split type-separates byte messages
//! from field frames at zero extra permutes.

use spongefish::{Encoding, NargDeserialize, VerificationError, VerificationResult};

/// The one authoritative parser for the crate's NARG framing (8-byte LE length
/// ‖ body), shared by [`BytesMsg`] and every `Fr`-domain message type so the
/// accept/reject behavior cannot drift. Returns the body as a borrowed
/// subslice (no copy); `buf` advances past the frame on success and is
/// untouched on error.
///
/// The length is converted with `usize::try_from`, not `as`: on 32-bit
/// targets an `as` cast truncates a > `usize::MAX` length, making the same
/// NARG accepted on one platform and rejected on another — any overflow is a
/// deserialization error instead.
fn read_length_prefixed_body<'a>(buf: &mut &'a [u8]) -> VerificationResult<&'a [u8]> {
    let (len_bytes, rest) = buf.split_first_chunk::<8>().ok_or(VerificationError)?;
    let len = usize::try_from(u64::from_le_bytes(*len_bytes)).map_err(|_| VerificationError)?;
    if rest.len() < len {
        return Err(VerificationError);
    }
    let (body, tail) = rest.split_at(len);
    *buf = tail;
    Ok(body)
}

/// Length-prefixed byte string. 8-byte LE length keeps `BytesMsg(a) ; BytesMsg(b)`
/// distinguishable from `BytesMsg(a||b)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytesMsg(pub Vec<u8>);

impl BytesMsg {
    /// Returns the inner bytes.
    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

impl From<Vec<u8>> for BytesMsg {
    fn from(v: Vec<u8>) -> Self {
        Self(v)
    }
}

impl Encoding<[u8]> for BytesMsg {
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut out = Vec::with_capacity(8 + self.0.len());
        out.extend_from_slice(&(self.0.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.0);
        out
    }
}

impl NargDeserialize for BytesMsg {
    fn deserialize_from_narg(buf: &mut &[u8]) -> VerificationResult<Self> {
        read_length_prefixed_body(buf).map(|body| BytesMsg(body.to_vec()))
    }
}

#[cfg(feature = "transcript-poseidon")]
mod fr_domain {
    use ark_bn254::Fr;
    use ark_ff::{PrimeField, Zero};
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use spongefish::{
        Decoding, Encoding, NargDeserialize, NargSerialize, VerificationError, VerificationResult,
    };

    use super::read_length_prefixed_body;

    /// Bytes per 31-byte little-endian chunk unit (248 bits < BN254 modulus,
    /// so chunk ↦ `Fr` is injective).
    pub const BYTE_RULE_CHUNK: usize = 31;

    /// Append one complete byte-rule message for `bytes` to `units`:
    /// `[Fr(2L), 31-byte-LE chunks…]`, zero-padded to an even unit count
    /// *relative to the start of this message* so each message occupies whole
    /// permute pairs even when concatenated (e.g. per-GT groups inside
    /// [`CommitmentsMsg`]).
    pub fn push_byte_rule_units(units: &mut Vec<Fr>, bytes: &[u8]) {
        let start = units.len();
        units.push(Fr::from(2 * bytes.len() as u64));
        for chunk in bytes.chunks(BYTE_RULE_CHUNK) {
            units.push(Fr::from_le_bytes_mod_order(chunk));
        }
        if (units.len() - start) % 2 == 1 {
            units.push(Fr::zero());
        }
    }

    /// The byte-rule CHUNK values of `bytes` (no leading length tag, no even
    /// padding): `bytes.chunks(31) ↦ Fr::from_le_bytes_mod_order`, exactly the
    /// payload [`push_byte_rule_units`] pushes between the tag and pad. The
    /// single source of the commitment/byte-string chunking the transpiler's
    /// symbolic mirror re-uses (`commitment_to_field_chunks`, the
    /// `FieldAlignedLayout` differential), kept here next to the native
    /// encoder so the chunking logic cannot drift from the byte-rule encoding.
    pub fn commitment_to_chunks(bytes: &[u8]) -> Vec<Fr> {
        bytes
            .chunks(BYTE_RULE_CHUNK)
            .map(Fr::from_le_bytes_mod_order)
            .collect()
    }

    /// Append one complete count-led field frame for `elems` to `units`:
    /// `[Fr(2k+1), e₁, …, e_k]`, zero-padded to an even unit count relative to
    /// the start of this message (see [`push_byte_rule_units`]). Exported so
    /// the transpiler's symbolic sponge mirror imports the encoding instead of
    /// re-hardcoding it.
    pub fn push_field_frame_units(units: &mut Vec<Fr>, elems: &[Fr]) {
        let start = units.len();
        units.push(Fr::from(2 * elems.len() as u64 + 1));
        units.extend_from_slice(elems);
        if (units.len() - start) % 2 == 1 {
            units.push(Fr::zero());
        }
    }

    /// Append the leading frame-count units of a [`CommitmentsMsg`] of `count`
    /// commitments: the count unit `Fr(2·count+1)` padded with a zero unit to a
    /// whole permute pair, so every per-commitment byte-rule group that follows
    /// stays pair-aligned (review fix F1). The per-commitment groups themselves
    /// are appended with [`push_byte_rule_units`] over each commitment's
    /// canonical compressed bytes. Exported for the same anti-drift reason as
    /// [`push_field_frame_units`].
    pub fn push_commitments_frame_header(units: &mut Vec<Fr>, count: usize) {
        units.push(Fr::from(2 * count as u64 + 1));
        units.push(Fr::zero());
    }

    fn write_length_prefixed_body(
        dst: &mut Vec<u8>,
        body_len: usize,
        write: impl FnOnce(&mut Vec<u8>),
    ) {
        dst.extend_from_slice(&(body_len as u64).to_le_bytes());
        let before = dst.len();
        write(dst);
        debug_assert_eq!(dst.len() - before, body_len, "NARG body length mismatch");
    }

    /// Raw bytes under the byte rule for the `Fr`-unit sponge.
    ///
    /// Sponge encoding: `[Fr(2L), 31-byte-LE chunks…]` (+ even padding). NARG
    /// transport (when used as a prover message): byte-identical to
    /// [`BytesMsg`](super::BytesMsg) — 8-byte LE length ‖ body — so switching
    /// a frame's *absorption* to field units never changes the proof bytes.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct RawBytesMsg(pub Vec<u8>);

    impl Encoding<[Fr]> for RawBytesMsg {
        fn encode(&self) -> impl AsRef<[Fr]> {
            let mut units = Vec::with_capacity(2 + self.0.len() / BYTE_RULE_CHUNK);
            push_byte_rule_units(&mut units, &self.0);
            units
        }
    }

    impl NargSerialize for RawBytesMsg {
        fn serialize_into_narg(&self, dst: &mut Vec<u8>) {
            write_length_prefixed_body(dst, self.0.len(), |dst| dst.extend_from_slice(&self.0));
        }
    }

    impl NargDeserialize for RawBytesMsg {
        fn deserialize_from_narg(buf: &mut &[u8]) -> VerificationResult<Self> {
            read_length_prefixed_body(buf).map(|body| RawBytesMsg(body.to_vec()))
        }
    }

    /// A frame of `k` field elements absorbed as the count-led field frame
    /// `[Fr(2k+1), e₁, …, e_k]` (+ even padding).
    ///
    /// NARG transport: 8-byte LE byte-length ‖ 32-byte-LE canonical elements —
    /// byte-identical to `BytesMsg(serialize_slice(elems))`. Deserialization
    /// rejects bodies that are not a multiple of 32 bytes and non-canonical
    /// (≥ r) elements, mirroring the native `read_all` strictness.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct FieldFrameMsg(pub Vec<Fr>);

    impl Encoding<[Fr]> for FieldFrameMsg {
        fn encode(&self) -> impl AsRef<[Fr]> {
            let mut units = Vec::with_capacity(2 + self.0.len());
            push_field_frame_units(&mut units, &self.0);
            units
        }
    }

    impl NargSerialize for FieldFrameMsg {
        #[expect(
            clippy::expect_used,
            reason = "CanonicalSerialize into a Vec is infallible"
        )]
        fn serialize_into_narg(&self, dst: &mut Vec<u8>) {
            write_length_prefixed_body(dst, self.0.len() * 32, |dst| {
                for e in &self.0 {
                    e.serialize_compressed(&mut *dst)
                        .expect("CanonicalSerialize into a Vec is infallible");
                }
            });
        }
    }

    impl NargDeserialize for FieldFrameMsg {
        fn deserialize_from_narg(buf: &mut &[u8]) -> VerificationResult<Self> {
            // Stage the cursor so `buf` is untouched when body validation fails.
            let mut staged = *buf;
            let body = read_length_prefixed_body(&mut staged)?;
            if !body.len().is_multiple_of(32) {
                return Err(VerificationError);
            }
            let elems = body
                .chunks_exact(32)
                .map(|c| Fr::deserialize_compressed(c).map_err(|_| VerificationError))
                .collect::<VerificationResult<Vec<Fr>>>()?;
            *buf = staged;
            Ok(FieldFrameMsg(elems))
        }
    }

    /// A NARG frame of `k` commitments absorbed as a leading **frame count
    /// unit** `Fr(2k+1)` (padded to a whole permute pair) followed by `k`
    /// per-commitment byte-rule groups (spec §4.2: one Dory GT = 384
    /// canonical bytes ↦ `[Fr(2·384), 13 chunks]` = 14 units = 7 permutes).
    ///
    /// WHY the frame-level count: each per-GT group is a self-delimiting,
    /// even-length unit run, so without it adjacent commitment frames could
    /// be re-partitioned (`[c1,c2]+[c3]` vs `[c1]+[c2,c3]`) with an
    /// IDENTICAL absorbed unit stream — challenges unchanged under NARG
    /// malleation. The count binds the partition. Its odd `2k+1` tag shares
    /// a domain with [`FieldFrameMsg`]'s count; the two kinds are
    /// disambiguated positionally by the fixed absorb/read schedule (each
    /// slot is read with exactly one message type), the same way a scalar
    /// frame is distinguished from the byte-rule absorb of the scalars'
    /// serialization today.
    ///
    /// An **empty** frame is the `Fr(1)` count-led case — distinct from the
    /// empty byte message `[Fr(0)]` — so the data-dependent untrusted-advice
    /// presence frame stays count-led: absent ↦ leading unit `Fr(1)`,
    /// present ↦ `Fr(3)`.
    ///
    /// NARG transport: 8-byte LE total length ‖ concatenated compressed
    /// serializations — byte-identical to `BytesMsg(serialize_slice(values))`.
    /// Deserialization parses elements sequentially from the self-delimiting
    /// body (allocation bounded by the actual frame bytes).
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct CommitmentsMsg<T>(pub Vec<T>);

    impl<T: CanonicalSerialize> Encoding<[Fr]> for CommitmentsMsg<T> {
        #[expect(
            clippy::expect_used,
            reason = "CanonicalSerialize into a Vec is infallible"
        )]
        fn encode(&self) -> impl AsRef<[Fr]> {
            // 2 units per frame (count + pad) + 14 per Dory GT.
            let mut units = Vec::with_capacity(2 + 14 * self.0.len());
            // Count unit padded to a whole permute pair so every per-GT group
            // stays pair-aligned (the per-GT witness alignment spec §4.2
            // pays extra permutes for).
            push_commitments_frame_header(&mut units, self.0.len());
            let mut bytes = Vec::new();
            for value in &self.0 {
                bytes.clear();
                value
                    .serialize_compressed(&mut bytes)
                    .expect("CanonicalSerialize into a Vec is infallible");
                push_byte_rule_units(&mut units, &bytes);
            }
            units
        }
    }

    impl<T: CanonicalSerialize> NargSerialize for CommitmentsMsg<T> {
        #[expect(
            clippy::expect_used,
            reason = "CanonicalSerialize into a Vec is infallible"
        )]
        fn serialize_into_narg(&self, dst: &mut Vec<u8>) {
            let body_len: usize = self.0.iter().map(CanonicalSerialize::compressed_size).sum();
            write_length_prefixed_body(dst, body_len, |dst| {
                for value in &self.0 {
                    value
                        .serialize_compressed(&mut *dst)
                        .expect("CanonicalSerialize into a Vec is infallible");
                }
            });
        }
    }

    impl<T: CanonicalSerialize + CanonicalDeserialize> NargDeserialize for CommitmentsMsg<T> {
        fn deserialize_from_narg(buf: &mut &[u8]) -> VerificationResult<Self> {
            // Stage the cursor so `buf` is untouched when body parsing fails.
            let mut staged = *buf;
            let mut cursor = read_length_prefixed_body(&mut staged)?;
            let mut values = Vec::new();
            while !cursor.is_empty() {
                let before = cursor.len();
                values.push(T::deserialize_compressed(&mut cursor).map_err(|_| VerificationError)?);
                // A `T` whose deserialization consumes zero bytes (e.g. a
                // unit-like type) would loop forever; require progress.
                if cursor.len() >= before {
                    return Err(VerificationError);
                }
            }
            *buf = staged;
            Ok(CommitmentsMsg(values))
        }
    }

    /// One native `Fr` challenge squeeze = exactly one permute.
    ///
    /// The orphan rule blocks `Decoding<[Fr]>` on `Fr` itself, so this local
    /// newtype carries the identity decode with the one-unit `Repr`
    /// `[Fr; 1]` (which already satisfies `Default + AsMut<[Fr]>`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct NativeChallenge(pub Fr);

    impl Decoding<[Fr]> for NativeChallenge {
        type Repr = [Fr; 1];

        fn decode(buf: Self::Repr) -> Self {
            NativeChallenge(buf[0])
        }
    }
}

#[cfg(feature = "transcript-poseidon")]
pub use fr_domain::{
    commitment_to_chunks, push_byte_rule_units, push_commitments_frame_header,
    push_field_frame_units, CommitmentsMsg, FieldFrameMsg, NativeChallenge, RawBytesMsg,
    BYTE_RULE_CHUNK,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_msg_is_length_prefixed() {
        let m = BytesMsg(vec![1, 2, 3, 4]);
        let enc = m.encode();
        let bytes = enc.as_ref();
        assert_eq!(bytes.len(), 8 + 4);
        assert_eq!(&bytes[..8], &4u64.to_le_bytes());
        assert_eq!(&bytes[8..], &[1, 2, 3, 4]);
    }

    #[test]
    fn bytes_msg_narg_rejects_truncation() {
        let m = BytesMsg(vec![9, 8, 7]);
        let mut narg: Vec<u8> = Vec::new();
        narg.extend_from_slice(m.encode().as_ref());
        let _ = narg.pop();
        let mut cursor: &[u8] = &narg;
        let before = cursor.len();
        let result = BytesMsg::deserialize_from_narg(&mut cursor);
        assert!(result.is_err());
        assert_eq!(cursor.len(), before, "cursor must not advance on error");
    }

    #[test]
    fn bytes_msg_narg_rejects_oversized_length() {
        let mut narg = Vec::new();
        narg.extend_from_slice(&u64::MAX.to_le_bytes());
        let mut cursor: &[u8] = &narg;
        let before = cursor.len();
        let result = BytesMsg::deserialize_from_narg(&mut cursor);
        assert!(result.is_err());
        assert_eq!(cursor.len(), before, "cursor must not advance on error");
    }
}
