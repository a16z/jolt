//! The private-message seam: witness-dependent field messages, which a
//! masked mode hides behind pads and clear mode absorbs as they are.
//!
//! The sumcheck absorbs route through [`PrivateChannel`]: batch input claims,
//! round coefficients, uni-skip output claims and stage output claims. Clear
//! transcripts implement the seam as plain public messages, so routing a
//! value through it changes no transcript byte; a masked channel implements
//! it by absorbing `value + pad` and recording where the pads sit. Other
//! witness-dependent absorbs still call [`Transcript::append`] or
//! [`Transcript::append_values`] directly, among them the stage 8 opening
//! batch and the prefix-packed opening statement in `jolt-openings`; a masked
//! mode must route those before it relies on this seam alone.

use jolt_field::Field;
use spongefish::Encoding;

use crate::Transcript;

/// Either role's end of the private-message seam.
///
/// Both roles hold the values a message carries: the prover computed them,
/// the verifier read them from the proof. So one method serves both, and
/// code written over this trait runs unchanged on the prover and verifier
/// transcripts.
pub trait PrivateChannel {
    /// Absorbs `values`, in order, as one private message.
    fn append_private<F: Field + Encoding>(&mut self, values: &[F]);

    /// Binds a claim the verifier can already derive from earlier messages.
    /// A clear transcript absorbs it as a public value; a masked channel
    /// binds nothing, since the masked messages it derives from already do.
    ///
    /// The claim must be a function of values already absorbed: under a
    /// masked channel this call adds nothing to the transcript.
    fn bind_derived<F: Field + Encoding>(&mut self, claim: &F);
}

/// Clear mode: a private message is a public message of the same values,
/// and a derived claim is a public value.
///
/// [`append_private`](PrivateChannel::append_private) absorbs one
/// [`append_values`](Transcript::append_values) message. For values whose
/// encodings are whole words, such as every production field, the sponge
/// absorbs the bytes one [`append`](Transcript::append) per value would.
impl<T: Transcript + ?Sized> PrivateChannel for T {
    fn append_private<F: Field + Encoding>(&mut self, values: &[F]) {
        self.append_values(values);
    }

    fn bind_derived<F: Field + Encoding>(&mut self, claim: &F) {
        self.append_values(std::slice::from_ref(claim));
    }
}
