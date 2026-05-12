//! Spongefish domain-separator wiring used by both the new split-trait
//! surface and the source-compatible facade.
//!
//! All three sponges in this crate use byte units (`H::U = u8`); the protocol
//! identifier is fixed crate-wide. Per-construction disambiguation is carried
//! by the spongefish session value (e.g. the legacy `Transcript::new(label)`
//! is mapped to `session(label)`).

use rand::rngs::StdRng;
use spongefish::{DomainSeparator, DuplexSpongeInterface, Encoding, ProverState, VerifierState};

use crate::codec::BytesMsg;

/// Crate-wide spongefish protocol identifier (ASCII left, zero-padded to
/// 64 bytes).
pub const PROTOCOL_ID: [u8; 64] = pad_id(b"a16z/jolt-transcript/spongefish/v1");

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

/// Empty `instance` value so the spongefish builder reaches the `to_prover`
/// stage. Encodes to zero bytes.
#[derive(Clone, Copy, Debug, Default)]
pub struct EmptyInstance;

impl Encoding<[u8]> for EmptyInstance {
    fn encode(&self) -> impl AsRef<[u8]> {
        [0u8; 0]
    }
}

/// Builds a fresh `ProverState` over `sponge` for the given session bytes.
pub fn to_prover<H>(sponge: H, session: &[u8]) -> ProverState<H, StdRng>
where
    H: DuplexSpongeInterface<U = u8>,
{
    DomainSeparator::new(PROTOCOL_ID)
        .session(BytesMsg(session.to_vec()))
        .instance(EmptyInstance)
        .to_prover(sponge)
}

/// Builds a fresh `VerifierState` over `sponge` for the given session bytes
/// and NARG.
pub fn to_verifier<'a, H>(
    sponge: H,
    session: &[u8],
    narg: &'a [u8],
) -> VerifierState<'a, H>
where
    H: DuplexSpongeInterface<U = u8>,
{
    DomainSeparator::new(PROTOCOL_ID)
        .session(BytesMsg(session.to_vec()))
        .instance(EmptyInstance)
        .to_verifier(sponge, narg)
}
