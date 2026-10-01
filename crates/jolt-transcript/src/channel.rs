//! Operations both transcript roles perform identically.

use jolt_field::{CanonicalBytes, CanonicalDecode, CanonicalEncoding, Field};

use crate::{Preview, SiteId, Sponge, TranscriptError};

/// One party's end of a Fiat-Shamir proof channel.
///
/// Protocol code that is the same for prover and verifier (absorbing public
/// statements, drawing challenges, exchanging messages whose shape both sides
/// know) is written once against this trait. A prover message is exchanged in
/// place: the prover sends the value it holds, the verifier overwrites the
/// slot with the value it receives.
///
/// Message shapes are positional. Neither role transmits lengths or labels for
/// atoms; both derive every count from public parameters.
pub trait Channel {
    /// The sponge this transcript runs on.
    type Sponge: Sponge;

    /// Tags subsequent operations with `site` in the event log. Never absorbed.
    fn site(&mut self, site: SiteId);

    /// Absorbs a public atom both sides already hold.
    fn public<A: CanonicalBytes>(&mut self, value: &A);

    /// Absorbs public atoms both sides already hold, in order.
    fn public_all<A: CanonicalBytes>(&mut self, values: &[A]);

    /// Absorbs a public byte string, framed by its `u64` little-endian length.
    fn public_bytes(&mut self, bytes: &[u8]);

    /// Sends (prover) or receives into (verifier) one prover message.
    ///
    /// # Errors
    ///
    /// The verifier fails when the argument string is exhausted or the bytes
    /// are not a canonical encoding. The prover never fails.
    fn exchange<A: CanonicalDecode>(&mut self, value: &mut A) -> Result<(), TranscriptError>;

    /// Sends or receives one prover message per slot, in order.
    ///
    /// # Errors
    ///
    /// As [`exchange`](Self::exchange).
    fn exchange_all<A: CanonicalDecode>(&mut self, values: &mut [A])
        -> Result<(), TranscriptError>;

    /// Draws an exactly uniform element of `F`, per [`Field::random`]'s
    /// rejection-sampling contract.
    fn challenge<F: Field>(&mut self) -> F;

    /// Draws a challenge from `F`'s small challenge set, decoded from 16
    /// squeezed bytes by [`CanonicalEncoding::from_challenge_bytes`]. Cheaper to
    /// multiply by than [`challenge`](Self::challenge); its soundness error is
    /// set by the small set's size, not `|F|`.
    fn challenge_small<F: CanonicalEncoding>(&mut self) -> F;

    /// Squeezes `N` raw challenge bytes.
    fn challenge_bytes<const N: usize>(&mut self) -> [u8; N];

    /// A detached copy of the current sponge state.
    fn preview(&self) -> Preview<Self::Sponge>;
}
