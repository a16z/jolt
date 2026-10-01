//! Sponge state shared by both transcript roles.

use jolt_field::{CanonicalBytes, CanonicalEncoding, Field};
use rand_core::RngCore;

#[cfg(feature = "logging")]
use crate::site::TranscriptEvent;
use crate::site::{Log, TranscriptOp};
use crate::{Preview, ProtocolId, SiteId, Sponge};

/// Bytes squeezed for a [`Channel::challenge_small`](crate::Channel::challenge_small).
pub const SMALL_CHALLENGE_BYTES: usize = 16;

/// Width of the little-endian length prefix on framed public bytes.
const FRAME_LEN_BYTES: usize = 8;

/// The sponge plus its event log. Both roles run every absorb and squeeze
/// through here, so their sponge transitions cannot drift apart.
#[derive(Clone, Debug)]
pub(crate) struct Duplex<H> {
    sponge: H,
    log: Log,
}

impl<H: Sponge> Duplex<H> {
    /// Absorbs the protocol id, then the length-framed session.
    pub(crate) fn new(protocol: &ProtocolId, session: &[u8]) -> Self {
        let mut sponge = H::default();
        let _ = sponge.absorb(protocol.as_bytes());
        absorb_framed(&mut sponge, session);
        Self {
            sponge,
            log: Log::default(),
        }
    }

    pub(crate) fn set_site(&mut self, site: SiteId) {
        self.log.set_site(site);
    }

    pub(crate) fn absorb_public(&mut self, bytes: &[u8]) {
        let _ = self.sponge.absorb(bytes);
        self.log.record(TranscriptOp::Public, bytes.len(), None);
    }

    pub(crate) fn absorb_public_atoms<A: CanonicalBytes>(&mut self, values: &[A]) {
        let mut bytes = vec![0u8; A::NUM_BYTES * values.len()];
        for (value, out) in values.iter().zip(bytes.chunks_exact_mut(A::NUM_BYTES)) {
            value.to_bytes_le(out);
        }
        self.absorb_public(&bytes);
    }

    pub(crate) fn absorb_public_framed(&mut self, bytes: &[u8]) {
        absorb_framed(&mut self.sponge, bytes);
        self.log
            .record(TranscriptOp::Public, FRAME_LEN_BYTES + bytes.len(), None);
    }

    /// Absorbs prover-message bytes found at `narg_start` in the argument string.
    pub(crate) fn absorb_message(&mut self, bytes: &[u8], narg_start: usize) {
        let _ = self.sponge.absorb(bytes);
        self.log.record(
            TranscriptOp::Message,
            bytes.len(),
            Some(narg_start..narg_start + bytes.len()),
        );
    }

    pub(crate) fn squeeze(&mut self, out: &mut [u8]) {
        let _ = self.sponge.squeeze(out);
        self.log.record(TranscriptOp::Challenge, out.len(), None);
    }

    pub(crate) fn squeeze_array<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        self.squeeze(&mut out);
        out
    }

    pub(crate) fn challenge<F: Field>(&mut self) -> F {
        let mut rng = SqueezeRng {
            sponge: &mut self.sponge,
            squeezed: 0,
        };
        let value = F::random(&mut rng);
        let squeezed = rng.squeezed;
        self.log.record(TranscriptOp::Challenge, squeezed, None);
        value
    }

    pub(crate) fn challenge_small<F: CanonicalEncoding>(&mut self) -> F {
        let mut bytes = [0u8; SMALL_CHALLENGE_BYTES];
        self.squeeze(&mut bytes);
        F::from_challenge_bytes(&bytes)
    }

    pub(crate) fn preview(&self) -> Preview<H> {
        Preview::new(self.sponge.clone())
    }

    #[cfg(feature = "logging")]
    pub(crate) fn events(&self) -> &[TranscriptEvent] {
        self.log.events()
    }
}

/// Absorbs `len(bytes)` as a `u64` little-endian prefix, then `bytes`, so
/// consecutive variable-length values cannot run into each other.
fn absorb_framed<H: Sponge>(sponge: &mut H, bytes: &[u8]) {
    let _ = sponge.absorb(&(bytes.len() as u64).to_le_bytes());
    let _ = sponge.absorb(bytes);
}

/// Feeds squeezed sponge output to [`Field::random`], whose rejection
/// sampling contract defines exactly uniform challenges.
struct SqueezeRng<'a, H> {
    sponge: &'a mut H,
    squeezed: usize,
}

impl<H: Sponge> RngCore for SqueezeRng<'_, H> {
    fn next_u32(&mut self) -> u32 {
        let mut bytes = [0u8; 4];
        self.fill_bytes(&mut bytes);
        u32::from_le_bytes(bytes)
    }

    fn next_u64(&mut self) -> u64 {
        let mut bytes = [0u8; 8];
        self.fill_bytes(&mut bytes);
        u64::from_le_bytes(bytes)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let _ = self.sponge.squeeze(dest);
        self.squeezed += dest.len();
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}
