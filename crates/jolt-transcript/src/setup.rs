//! Native spongefish transcript initialization.
//!
//! The session identifier is derived from [`PROTOCOL_ID`] with SHAKE128. Each
//! transcript absorbs one instance: the caller's 32-byte session tag, then the
//! encoding of the public statement as a [`BytesMsg`].

use spongefish::{DuplexSpongeInit, Encoding, SessionId, VerifierState};

use crate::codec::{protocol_tag, BytesMsg, SpongeUnit, Units, TAG_BYTES};
use crate::digest_hash::{DigestHash, DIGEST_HASH};
use crate::shake128::{ProverState, Shake128};

/// Crate-wide protocol tag, `a16z/jolt-transcript/v2/sha3` zero-padded to 32
/// bytes (`a16z/jolt-transcript/v2/keccak` under `keccak256-digests`).
/// [`session_id`] derives the spongefish session identifier from it, so a
/// change to anything below changes every proof's challenges.
///
/// Version 2 fixes:
/// - **The absorb layout.** The transcript is positional: no message carries
///   a label, and a count or length is absorbed, as one `u64` LE word, only
///   where the prover chooses it. Every absorb and squeeze is a whole number
///   of eight-byte words. Byte strings are a `u64` length, the bytes, and zero
///   padding to a word. Integers are `u64` LE words. The instance is the
///   caller's 32-byte session tag, then the statement as a byte string.
///   Version strings that version-1 protocols absorbed mid-transcript
///   (`spartan-clear-v1`, `hyperkzg-binary-bn254-v1`, `BlindFold`) are
///   covered by this tag.
/// - **The challenge codec.** Each field's [`ChallengeCodec`]: BN254 `Fr` and
///   `Fq` take 56 bytes per element; pseudo-Mersenne pools of four-byte
///   candidates hold an even number of them; `F8` messages pad their squeeze
///   to a word.
/// - **The session identifier.** SHAKE128 (FIPS 202) derives it from this
///   tag.
/// - **The proof-of-work layout.** The nonce is canonical LEB128 in the NARG
///   string and one `u64` LE word in the sponge; the check is eight squeezed
///   bytes whose little-endian `u64` has its low `bits` bits zero.
///
/// The suffix names the digest hash ([`DIGEST_HASH`]) behind every digest a
/// statement or key binds, so a proof made under one never verifies under the
/// other. The tag does not name the sponge: a different sponge is a different
/// hash function and changes every output by itself.
///
/// [`ChallengeCodec`]: crate::ChallengeCodec
pub const PROTOCOL_ID: [u8; TAG_BYTES] = protocol_tag(match DIGEST_HASH {
    DigestHash::Sha3_256 => b"a16z/jolt-transcript/v2/sha3".as_slice(),
    DigestHash::Keccak256 => b"a16z/jolt-transcript/v2/keccak".as_slice(),
});

/// The spongefish instance: the 32-byte session tag, then the statement as a
/// [`BytesMsg`]. Never empty, so spongefish's constructors accept it. The tag
/// has a fixed width and the statement carries its length, so the instance
/// cannot run into the first message, and it ends on a word boundary. The
/// sponge absorbs it as one message ([`SpongeUnit`]).
struct Instance<'a, S: ?Sized> {
    session: &'a [u8; TAG_BYTES],
    statement: &'a S,
}

impl<S: Encoding + ?Sized> Encoding for Instance<'_, S> {
    fn encode(&self) -> impl AsRef<[u8]> {
        let body = self.statement.encode();
        let message = BytesMsg(body.as_ref());
        let framed = message.encode();
        let framed = framed.as_ref();
        let mut out = Vec::with_capacity(TAG_BYTES + framed.len());
        out.extend_from_slice(self.session);
        out.extend_from_slice(framed);
        out
    }
}

/// Build a prover transcript bound to `session` and `statement`.
///
/// `session` is the caller's 32-byte protocol tag ([`protocol_tag`], e.g.
/// `protocol_tag(b"jolt-rv64imac/v1")`).
/// `statement` is the public statement, which the encoding must determine
/// completely: everything the verifier is given rather than reads from the
/// proof. Spongefish initializes a fresh state of backend `H` with
/// [`DuplexSpongeInit::init`].
///
/// # Example
///
/// ```
/// use jolt_transcript::{
///     protocol_tag, prover_transcript, verifier_transcript, BytesMsg, Keccak, ProverTranscript,
///     VerifierTranscript,
/// };
///
/// const SESSION: [u8; 32] = protocol_tag(b"jolt-rv64imac/v1");
/// let statement: [u8; 32] = [0xAB; 32];
///
/// let mut prover = prover_transcript::<Keccak>(&SESSION, &statement);
/// prover.send(&BytesMsg(b"commitment-a".to_vec()));
/// let narg = prover.narg_string().to_vec();
///
/// let mut verifier = verifier_transcript::<Keccak>(&SESSION, &statement, &narg);
/// let _: BytesMsg = verifier.prover_message().unwrap();
/// verifier.check_eof().unwrap();
/// ```
#[must_use]
pub fn prover_transcript<H>(
    session: &[u8; TAG_BYTES],
    statement: &(impl Encoding + ?Sized),
) -> ProverState<H>
where
    H: DuplexSpongeInit<U: SpongeUnit>,
{
    ProverState::new(&session_id(), &Units(Instance { session, statement }))
}

/// Build a verifier transcript bound to `session` and `statement` over
/// `narg`. The verifier MUST pass the same `session`/`statement` as the
/// prover or the transcript states diverge. Spongefish initializes a fresh
/// state of backend `H` with [`DuplexSpongeInit::init`].
#[must_use]
pub fn verifier_transcript<'a, H>(
    session: &[u8; TAG_BYTES],
    statement: &(impl Encoding + ?Sized),
    narg: &'a [u8],
) -> VerifierState<'a, H>
where
    H: DuplexSpongeInit<U: SpongeUnit>,
{
    VerifierState::new(&session_id(), &Units(Instance { session, statement }), narg)
}

/// Derive the spongefish session identifier from [`PROTOCOL_ID`] using
/// SHAKE128. Callers with custom instance encodings can pass this identifier
/// directly to spongefish's prover and verifier constructors.
#[must_use]
pub fn session_id() -> SessionId {
    spongefish::derive_session_id::<Shake128>(&PROTOCOL_ID)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_id_is_the_padded_v2_tag_naming_the_digest_hash() {
        let name: &[u8] = match DIGEST_HASH {
            DigestHash::Sha3_256 => b"a16z/jolt-transcript/v2/sha3",
            DigestHash::Keccak256 => b"a16z/jolt-transcript/v2/keccak",
        };
        assert_eq!(PROTOCOL_ID, protocol_tag(name));
    }

    /// Known answer: the first 32 bytes of SHAKE128 over
    /// `irtf-cfrg-fiat-shamir/session-id`, zeros to 168 bytes, then
    /// [`PROTOCOL_ID`].
    #[test]
    fn session_id_is_shake128_of_the_protocol_id() {
        let expected: [u8; 32] = match DIGEST_HASH {
            DigestHash::Sha3_256 => [
                0xe4, 0xf7, 0xc5, 0x85, 0x2b, 0xca, 0x4d, 0x0a, 0x50, 0x8f, 0xe2, 0x19, 0xaf, 0x73,
                0x80, 0x3a, 0x6b, 0xae, 0xce, 0xc2, 0x10, 0x56, 0xd5, 0xc0, 0x8f, 0x72, 0xf1, 0x6c,
                0x05, 0x0f, 0x87, 0x92,
            ],
            DigestHash::Keccak256 => [
                0x2b, 0x99, 0xbf, 0x2a, 0xa3, 0x90, 0x9c, 0x03, 0x1c, 0xbf, 0xa1, 0x76, 0x93, 0x49,
                0x13, 0x00, 0x0f, 0x67, 0x3e, 0xc9, 0x75, 0x26, 0x9d, 0x9a, 0xfe, 0xd7, 0x67, 0x26,
                0x69, 0x1d, 0x0f, 0xd6,
            ],
        };
        assert_eq!(session_id().as_bytes(), &expected);
    }

    #[test]
    fn instance_encoding_is_the_tag_then_the_statement_as_bytes() {
        let tag = protocol_tag(b"session");
        let digest = [0xA5u8; 32];
        let encoded = Instance {
            session: &tag,
            statement: &digest,
        }
        .encode()
        .as_ref()
        .to_vec();
        let mut expected = tag.to_vec();
        expected.extend_from_slice(&32u64.to_le_bytes());
        expected.extend_from_slice(&digest);
        assert_eq!(encoded, expected);

        let encoded = Instance {
            session: &tag,
            statement: &[0u8; 0],
        }
        .encode()
        .as_ref()
        .to_vec();
        let mut expected = tag.to_vec();
        expected.extend_from_slice(&0u64.to_le_bytes());
        assert_eq!(encoded, expected);

        // A 5-byte statement pads to a word, so the instance ends aligned.
        let encoded = Instance {
            session: &tag,
            statement: b"stmt!",
        }
        .encode()
        .as_ref()
        .to_vec();
        let mut expected = tag.to_vec();
        expected.extend_from_slice(&5u64.to_le_bytes());
        expected.extend_from_slice(b"stmt!\0\0\0");
        assert_eq!(encoded, expected);
        assert_eq!(encoded.len() % 8, 0);
    }
}
