//! Spongefish domain-separator entry points for the native split-trait
//! surface.
//!
//! Two factory functions cover every Jolt caller:
//!
//! - [`prover_transcript`] — `(session, instance, sponge) → ProverState`
//! - [`verifier_transcript`] — `(session, instance, sponge, narg) → VerifierState`
//!
//! Both pre-bind [`PROTOCOL_ID`] and absorb the spongefish
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
//!
//! Power users (e.g. `BlindFold` sub-verifiers, custom Encoding types) can
//! drop down to [`transcript_builder`], which returns spongefish's
//! `DomainSeparator` with PROTOCOL_ID pre-bound — the full type-state
//! builder is then available.
//!
//! The factories dispatch per sponge alphabet via [`TranscriptInit`]: byte
//! sponges (`U = u8`) go through spongefish's `DomainSeparator` unchanged;
//! the `Fr`-unit Poseidon sponge absorbs the same three domain-separator
//! byte strings under the byte rule via
//! [`poseidon_prover_transcript`]/[`poseidon_verifier_transcript`].

use rand::rngs::StdRng;
use spongefish::{
    DomainSeparator, DuplexSpongeInterface, Encoding, ProverState, VerifierState, WithoutInstance,
    WithoutSession,
};

use crate::codec::BytesMsg;

/// Crate-wide spongefish protocol identifier (ASCII left, zero-padded to
/// 64 bytes). Pre-bound by every factory in this module; exposed so callers
/// who reach for [`transcript_builder`] can align on the same identifier.
pub const PROTOCOL_ID: [u8; 64] = pad_id(b"a16z/jolt-transcript/v1");

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

/// Per-sponge domain-separator dispatch behind [`prover_transcript`] /
/// [`verifier_transcript`] — the unit-generic seam that lets jolt-core stay
/// generic over `H` whether the sponge alphabet is bytes or `Fr`.
///
/// - Byte sponges (`U = u8`) get a blanket impl that routes through
///   spongefish's `DomainSeparator` exactly as before — byte behavior is
///   untouched by construction.
/// - The `Fr`-unit [`PoseidonSponge`](crate::PoseidonSponge) cannot use
///   `DomainSeparator::to_prover`/`to_verifier` (`[u8; 64]: Encoding<[Fr]>`
///   is orphan-blocked), so its impl builds the states via the public
///   unit-generic constructors (`ProverState::from`,
///   `VerifierState::from_parts`) and absorbs the same domain-separator
///   content under the byte rule (see [`poseidon_prover_transcript`]).
pub trait TranscriptInit: DuplexSpongeInterface + Sized {
    /// Builds the prover state and absorbs `PROTOCOL_ID ‖ session ‖ instance`.
    fn init_prover(session: &[u8], instance: [u8; 32], sponge: Self) -> ProverState<Self, StdRng>;

    /// Builds the verifier state over `narg` and absorbs the same
    /// domain-separator content as [`init_prover`](Self::init_prover).
    fn init_verifier<'a>(
        session: &[u8],
        instance: [u8; 32],
        sponge: Self,
        narg: &'a [u8],
    ) -> VerifierState<'a, Self>;
}

impl<H> TranscriptInit for H
where
    H: DuplexSpongeInterface<U = u8>,
{
    fn init_prover(session: &[u8], instance: [u8; 32], sponge: Self) -> ProverState<Self, StdRng> {
        DomainSeparator::new(PROTOCOL_ID)
            .session(BytesMsg(session.to_vec()))
            .instance(InstanceDigest(instance))
            .to_prover(sponge)
    }

    fn init_verifier<'a>(
        session: &[u8],
        instance: [u8; 32],
        sponge: Self,
        narg: &'a [u8],
    ) -> VerifierState<'a, Self> {
        DomainSeparator::new(PROTOCOL_ID)
            .session(BytesMsg(session.to_vec()))
            .instance(InstanceDigest(instance))
            .to_verifier(sponge, narg)
    }
}

#[cfg(feature = "transcript-poseidon")]
impl TranscriptInit for crate::PoseidonSponge {
    fn init_prover(session: &[u8], instance: [u8; 32], sponge: Self) -> ProverState<Self, StdRng> {
        poseidon_prover_transcript(session, instance, sponge)
    }

    fn init_verifier<'a>(
        session: &[u8],
        instance: [u8; 32],
        sponge: Self,
        narg: &'a [u8],
    ) -> VerifierState<'a, Self> {
        poseidon_verifier_transcript(session, instance, sponge, narg)
    }
}

/// The domain-separator messages shared by the Poseidon prover/verifier
/// factories, absorbed under the byte rule in this order:
///
/// 1. [`PROTOCOL_ID`] — 64 zero-padded bytes (what `DomainSeparator` absorbs
///    as its protocol step);
/// 2. the session as its `BytesMsg` encoding — 8-byte LE length ‖ session
///    (what the byte path's `.session(BytesMsg(..))` step absorbs);
/// 3. the 32-byte instance digest raw (what `InstanceDigest` encodes to).
///
/// Public so the transpiler's symbolic sponge mirror seeds its in-circuit
/// state from the SAME three byte strings the native factories absorb,
/// instead of re-hardcoding them (a re-hardcoded copy is a drift channel).
#[cfg(feature = "transcript-poseidon")]
pub fn poseidon_domain_separator_msgs(
    session: &[u8],
    instance: [u8; 32],
) -> [crate::codec::RawBytesMsg; 3] {
    use crate::codec::RawBytesMsg;
    let session_bytes = BytesMsg(session.to_vec()).encode().as_ref().to_vec();
    [
        RawBytesMsg(PROTOCOL_ID.to_vec()),
        RawBytesMsg(session_bytes),
        RawBytesMsg(instance.to_vec()),
    ]
}

/// Poseidon-specific sibling of [`prover_transcript`]: builds the
/// `Fr`-unit prover state via the unit-generic `ProverState::from` and
/// absorbs the domain separator under the byte rule (spongefish's
/// `DomainSeparator::to_prover` is unusable for `U = Fr` because
/// `[u8; 64]: Encoding<[Fr]>` is orphan-blocked).
#[cfg(feature = "transcript-poseidon")]
#[must_use]
pub fn poseidon_prover_transcript(
    session: &[u8],
    instance: [u8; 32],
    sponge: crate::PoseidonSponge,
) -> ProverState<crate::PoseidonSponge, StdRng> {
    let mut state = ProverState::from(sponge);
    for msg in &poseidon_domain_separator_msgs(session, instance) {
        state.public_message(msg);
    }
    state
}

/// Poseidon-specific sibling of [`verifier_transcript`] — see
/// [`poseidon_prover_transcript`].
#[cfg(feature = "transcript-poseidon")]
#[must_use]
pub fn poseidon_verifier_transcript<'a>(
    session: &[u8],
    instance: [u8; 32],
    sponge: crate::PoseidonSponge,
    narg: &'a [u8],
) -> VerifierState<'a, crate::PoseidonSponge> {
    let mut state = VerifierState::from_parts(sponge, narg);
    for msg in &poseidon_domain_separator_msgs(session, instance) {
        state.public_message(msg);
    }
    state
}

/// Build a prover transcript bound to `session` and `instance`.
///
/// `session` is protocol-version bytes (e.g. `b"jolt-rv64imac/v1"`).
/// `instance` is the 32-byte digest of the public statement (typically
/// `Blake2b(CanonicalSerialize(public_state))`). Both are absorbed under
/// spongefish's domain-separator steps after [`PROTOCOL_ID`].
///
/// Dispatches per sponge via [`TranscriptInit`]: byte sponges go through
/// spongefish's `DomainSeparator` exactly as before; the `Fr`-unit Poseidon
/// sponge goes through [`poseidon_prover_transcript`].
///
/// # Example
///
/// ```
/// use jolt_transcript::{prover_transcript, verifier_transcript, BytesMsg, ProverTranscript, VerifierTranscript};
/// use spongefish::instantiations::Blake2b512;
///
/// const SESSION: &[u8] = b"jolt-rv64imac/v1";
/// let instance: [u8; 32] = [0xAB; 32];
///
/// let mut prover = prover_transcript(SESSION, instance, Blake2b512::default());
/// ProverTranscript::<Blake2b512>::prover_message(
///     &mut prover,
///     &BytesMsg(b"commitment-a".to_vec()),
/// );
/// let narg = ProverTranscript::<Blake2b512>::narg_string(&prover).to_vec();
///
/// let mut verifier = verifier_transcript(SESSION, instance, Blake2b512::default(), &narg);
/// let _: BytesMsg =
///     VerifierTranscript::<Blake2b512>::prover_message(&mut verifier).unwrap();
/// VerifierTranscript::<Blake2b512>::check_eof(verifier).unwrap();
/// ```
#[must_use]
pub fn prover_transcript<H>(session: &[u8], instance: [u8; 32], sponge: H) -> ProverState<H, StdRng>
where
    H: TranscriptInit,
{
    H::init_prover(session, instance, sponge)
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
    H: TranscriptInit,
{
    H::init_verifier(session, instance, sponge, narg)
}

/// Escape hatch returning spongefish's `DomainSeparator` with
/// [`PROTOCOL_ID`] pre-bound. Use only when the standard
/// `(session, instance)` shape doesn't fit — e.g. `BlindFold` sub-verifiers
/// that compose sub-domain-separators, or callers absorbing a non-32-byte
/// instance.
///
/// Almost every Jolt caller should use [`prover_transcript`] /
/// [`verifier_transcript`] instead.
#[must_use]
pub fn transcript_builder() -> DomainSeparator<WithoutInstance, WithoutSession> {
    DomainSeparator::new(PROTOCOL_ID)
}
