//! Spongefish domain-separator entry points for the native split-trait
//! surface.
//!
//! Two factory functions cover every Jolt caller:
//!
//! - [`prover_transcript`] — `(session, instance, sponge) → ProverState`
//! - [`verifier_transcript`] — `(session, instance, sponge, narg) → VerifierState`
//!
//! Both pre-bind `PROTOCOL_ID` and absorb the spongefish
//! `session`/`instance` steps internally. Wire format mirrors spongefish's
//! `DomainSeparator` exactly:
//!
//! ```text
//! PROTOCOL_ID  (fixed, crate-wide)
//!   ↓
//! session(bytes)
//!   ↓
//! instance(32-byte digest)
//!   ↓
//! to_prover(sponge) | to_verifier(sponge, narg)
//! ```

use rand::rngs::StdRng;
use spongefish::{DomainSeparator, DuplexSpongeInterface, Encoding, ProverState, VerifierState};

use crate::codec::BytesMsg;

/// Crate-wide spongefish protocol identifier (ASCII left, zero-padded to 64 bytes).
const PROTOCOL_ID: [u8; 64] = pad_id(b"a16z/jolt-transcript/v1");

const fn pad_id(src: &[u8]) -> [u8; 64] {
    assert!(src.len() <= 64, "protocol id exceeds 64 bytes");
    let mut buf = [0u8; 64];
    let mut i = 0;
    while i < src.len() {
        buf[i] = src[i];
        i += 1;
    }
    buf
}

/// 32-byte instance digest. Internal adapter — exposes `Encoding<[u8]>`
/// over a fixed-size digest so it slots into spongefish's `.instance(...)`
/// step. Public callers pass a plain `[u8; 32]` to [`prover_transcript`] /
/// [`verifier_transcript`]; the factory wraps internally.
#[derive(Clone, Copy, Debug, Default)]
struct InstanceDigest([u8; 32]);

impl Encoding<[u8]> for InstanceDigest {
    fn encode(&self) -> impl AsRef<[u8]> {
        self.0
    }
}

/// Build a prover transcript bound to `session` and `instance`.
///
/// `session` is protocol-version bytes (e.g. `b"jolt-rv64imac/v1"`).
/// `instance` is the 32-byte digest of the public statement (typically
/// `Blake2b(CanonicalSerialize(public_state))`). Both are absorbed under
/// spongefish's domain-separator steps after `PROTOCOL_ID`.
///
/// # Example
///
/// ```
/// use jolt_transcript::{prover_transcript, verifier_transcript, BytesMsg};
/// use spongefish::instantiations::Blake2b512;
///
/// const SESSION: &[u8] = b"jolt-rv64imac/v1";
/// let instance: [u8; 32] = [0xAB; 32];
///
/// let mut prover = prover_transcript(SESSION, instance, Blake2b512::default());
/// prover.prover_message(&BytesMsg(b"commitment-a".to_vec()));
/// let narg = prover.narg_string().to_vec();
///
/// let mut verifier = verifier_transcript(SESSION, instance, Blake2b512::default(), &narg);
/// let _: BytesMsg = verifier.prover_message().unwrap();
/// verifier.check_eof().unwrap();
/// ```
#[must_use]
pub fn prover_transcript<H>(session: &[u8], instance: [u8; 32], sponge: H) -> ProverState<H, StdRng>
where
    H: DuplexSpongeInterface<U = u8>,
{
    DomainSeparator::new(PROTOCOL_ID)
        .session(BytesMsg(session.to_vec()))
        .instance(InstanceDigest(instance))
        .to_prover(sponge)
}

/// Build a verifier transcript bound to `session` and `instance` over
/// `narg`. The verifier MUST pass the same `session`/`instance` as the
/// prover or the transcript states diverge.
#[must_use]
pub fn verifier_transcript<'a, H>(
    session: &[u8],
    instance: [u8; 32],
    sponge: H,
    narg: &'a [u8],
) -> VerifierState<'a, H>
where
    H: DuplexSpongeInterface<U = u8>,
{
    DomainSeparator::new(PROTOCOL_ID)
        .session(BytesMsg(session.to_vec()))
        .instance(InstanceDigest(instance))
        .to_verifier(sponge, narg)
}
