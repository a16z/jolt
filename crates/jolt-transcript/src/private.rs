//! The private-message seam: the witness-dependent field messages that a
//! masked mode will hide behind pads and that clear mode absorbs as they are.
//!
//! The sumcheck absorbs route through [`PrivateChannel`]: batch input claims,
//! round coefficients, uni-skip output claims and stage output claims. Clear
//! transcripts implement the seam as plain public messages, so routing a
//! value through it changes no Keccak or Blake2b transcript byte. In clear
//! mode the trait only marks which absorbs depend on the witness. A masked
//! channel cannot implement it as it stands: its methods are infallible and
//! return nothing, while a masked channel can run out of pads and must report
//! the pad slots each message used, so masking needs a fallible seam that
//! returns slots.
//!
//! Other witness-dependent absorbs still call [`Transcript::append`] or
//! [`Transcript::append_values`] directly, for example the stage 8 opening
//! batch and the prefix-packed opening statement in `jolt-openings`. The
//! absorb inventory (`crates/jolt-verifier/tests/fs_inventory/absorb-sites.inventory`)
//! lists every absorb site; a masked mode must route the witness-dependent
//! ones before it relies on this seam alone.

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
}

/// Clear mode: a private message is a public message of the same values.
///
/// [`append_private`](PrivateChannel::append_private) absorbs one
/// [`append_values`](Transcript::append_values) message. For values whose
/// encodings are whole words, such as every production field, that message
/// carries the bytes one [`append`](Transcript::append) per value would. A
/// sponge that absorbs bytes without framing, such as Keccak or Blake2b,
/// then reaches the same state either way. The Poseidon sponge prefixes
/// every absorb with a length element, so for it one merged message and one
/// absorb per value give different states.
impl<T: Transcript + ?Sized> PrivateChannel for T {
    fn append_private<F: Field + Encoding>(&mut self, values: &[F]) {
        self.append_values(values);
    }
}
