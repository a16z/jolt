//! Native spongefish transcript initialization.
//!
//! The session identifier is derived from [`PROTOCOL_ID`] with SHAKE128. Each
//! transcript absorbs one instance: the length-prefixed caller session
//! followed by the length-prefixed encoding of the public statement.

use spongefish::{DuplexSpongeInit, Encoding, SessionId, VerifierState};

use crate::shake128::{ProverState, Shake128};

/// Crate-wide protocol identifier (ASCII left, zero-padded to 64 bytes).
/// [`session_id`] derives the spongefish session identifier from it.
///
/// Version 2 derives the session identifier with SHAKE128 (FIPS 202) and
/// draws full-width BN254 challenges.
pub const PROTOCOL_ID: [u8; 64] = pad_id(b"a16z/jolt-transcript/v2");

const fn pad_id(src: &[u8]) -> [u8; 64] {
    assert!(src.len() <= 64, "protocol id exceeds 64 bytes");
    let mut buf = [0u8; 64];
    let mut i = 0;
    while i < src.len() {
        #[expect(
            clippy::indexing_slicing,
            reason = "i < src.len() <= 64 == buf.len(); const-evaluated, so an out-of-bounds index fails compilation"
        )]
        {
            buf[i] = src[i];
        }
        i += 1;
    }
    buf
}

/// The spongefish instance: the session and the statement, each with an
/// eight-byte little-endian length. Never empty, so spongefish's constructors
/// accept it, and self-delimiting, so the statement cannot run into the first
/// message.
struct Instance<'a, S: ?Sized> {
    session: &'a [u8],
    statement: &'a S,
}

impl<S: Encoding + ?Sized> Encoding for Instance<'_, S> {
    fn encode(&self) -> impl AsRef<[u8]> {
        let statement = self.statement.encode();
        let statement = statement.as_ref();
        let mut out = Vec::with_capacity(16 + self.session.len() + statement.len());
        out.extend_from_slice(&(self.session.len() as u64).to_le_bytes());
        out.extend_from_slice(self.session);
        out.extend_from_slice(&(statement.len() as u64).to_le_bytes());
        out.extend_from_slice(statement);
        out
    }
}

/// Build a prover transcript bound to `session` and `statement`.
///
/// `session` is protocol-version bytes (e.g. `b"jolt-rv64imac/v1"`).
/// `statement` is the public statement, which the encoding must determine
/// completely: everything the verifier is given rather than reads from the
/// proof. Spongefish initializes a fresh state of backend `H` with
/// [`DuplexSpongeInit::init`].
///
/// # Example
///
/// ```
/// use jolt_transcript::{
///     prover_transcript, verifier_transcript, BytesMsg, Keccak, ProverTranscript,
///     VerifierTranscript,
/// };
///
/// const SESSION: &[u8] = b"jolt-rv64imac/v1";
/// let statement: [u8; 32] = [0xAB; 32];
///
/// let mut prover = prover_transcript::<Keccak>(SESSION, &statement);
/// prover.prover_message(&BytesMsg(b"commitment-a".to_vec()));
/// let narg = prover.narg_string().to_vec();
///
/// let mut verifier = verifier_transcript::<Keccak>(SESSION, &statement, &narg);
/// let _: BytesMsg = verifier.prover_message().unwrap();
/// verifier.check_eof().unwrap();
/// ```
#[must_use]
pub fn prover_transcript<H>(session: &[u8], statement: &(impl Encoding + ?Sized)) -> ProverState<H>
where
    H: DuplexSpongeInit<U = u8>,
{
    ProverState::new(&session_id(), &Instance { session, statement })
}

/// Build a verifier transcript bound to `session` and `statement` over
/// `narg`. The verifier MUST pass the same `session`/`statement` as the
/// prover or the transcript states diverge. Spongefish initializes a fresh
/// state of backend `H` with [`DuplexSpongeInit::init`].
#[must_use]
pub fn verifier_transcript<'a, H>(
    session: &[u8],
    statement: &(impl Encoding + ?Sized),
    narg: &'a [u8],
) -> VerifierState<'a, H>
where
    H: DuplexSpongeInit<U = u8>,
{
    VerifierState::new(&session_id(), &Instance { session, statement }, narg)
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
    fn pad_id_left_aligns_and_zero_pads_to_64_bytes() {
        assert_eq!(
            pad_id(b"a16z/jolt-transcript/v2"),
            PROTOCOL_ID,
            "PROTOCOL_ID must be the padded ASCII identifier"
        );
        let short = pad_id(b"xy");
        assert_eq!(&short[..2], b"xy");
        assert!(short[2..].iter().all(|&byte| byte == 0));
        let full = [0x41u8; 64];
        assert_eq!(pad_id(&full), full, "a 64-byte id is passed through");
    }

    /// Known answer: the first 32 bytes of SHAKE128 over
    /// `irtf-cfrg-fiat-shamir/session-id`, zeros to 168 bytes, then
    /// [`PROTOCOL_ID`].
    #[test]
    fn session_id_is_shake128_of_the_protocol_id() {
        assert_eq!(
            session_id().as_bytes(),
            &[
                0x40, 0x47, 0xcd, 0xdd, 0x77, 0x27, 0xcd, 0x49, 0x52, 0xf2, 0xa5, 0xd4, 0x00, 0x03,
                0x57, 0x54, 0x81, 0x80, 0x5c, 0x50, 0x8b, 0xd1, 0xa0, 0xdd, 0x9a, 0xbb, 0x56, 0x77,
                0x11, 0xb4, 0x79, 0x76,
            ]
        );
    }

    #[test]
    fn instance_encoding_frames_session_and_statement() {
        let digest = [0xA5u8; 32];
        let encoded = Instance {
            session: b"",
            statement: &digest,
        }
        .encode()
        .as_ref()
        .to_vec();
        let mut expected = vec![0; 8];
        expected.extend_from_slice(&32u64.to_le_bytes());
        expected.extend_from_slice(&digest);
        assert_eq!(encoded, expected);

        let encoded = Instance {
            session: b"session",
            statement: &[0u8; 0],
        }
        .encode()
        .as_ref()
        .to_vec();
        let mut expected = 7u64.to_le_bytes().to_vec();
        expected.extend_from_slice(b"session");
        expected.extend_from_slice(&0u64.to_le_bytes());
        assert_eq!(encoded, expected);
    }
}
