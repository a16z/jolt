//! The private-message seam: witness-dependent field messages, which masked
//! mode hides behind pads and clear mode exchanges as they are.

use jolt_field::{CanonicalDecode, CanonicalEncoding, ExtField, Field};

use crate::{Channel, ProverTranscript, Sponge, TranscriptError, VerifierTranscript};

/// Where a private message's pads sit in the pad schedule: `len` base-field
/// pads from `start`, `E::DEGREE` per extension element. Empty in clear mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PadSlots {
    /// Index of the first pad.
    pub start: usize,
    /// Number of pads.
    pub len: usize,
}

/// One role's end of the private-message seam, written once for both roles.
///
/// The prover sends values (clear) or values plus their next pads (masked);
/// the verifier receives them and, in masked mode, the pad slots the prover
/// used, so both sides index pads identically.
pub trait PrivateChannel {
    /// Why an exchange failed: a rejected proof, or an exhausted pad schedule.
    type Error;

    /// Exchanges `values` in place: the prover sends them, the verifier fills
    /// them from the proof.
    ///
    /// # Errors
    ///
    /// The verifier fails as [`Channel::exchange_all`] does.
    fn exchange_private<F, E>(&mut self, values: &mut [E]) -> Result<PadSlots, Self::Error>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F> + CanonicalDecode;

    /// Binds a claim the verifier can already derive. Clear mode absorbs it as
    /// a public value; masked mode binds nothing, since the masked messages it
    /// derives from already do.
    ///
    /// # Errors
    ///
    /// Clear mode never fails.
    fn bind_derived<F, E>(&mut self, claim: &E) -> Result<(), Self::Error>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F> + CanonicalDecode;

    /// [`exchange_private`](Self::exchange_private) for elements of the pad
    /// field itself, one pad each. Separate because not every field
    /// implements `ExtField` over itself.
    ///
    /// # Errors
    ///
    /// As [`exchange_private`](Self::exchange_private).
    fn exchange_private_base<F>(&mut self, values: &mut [F]) -> Result<PadSlots, Self::Error>
    where
        F: Field + CanonicalEncoding;

    /// [`bind_derived`](Self::bind_derived) for an element of the pad field.
    ///
    /// # Errors
    ///
    /// As [`bind_derived`](Self::bind_derived).
    fn bind_derived_base<F>(&mut self, claim: &F) -> Result<(), Self::Error>
    where
        F: Field + CanonicalEncoding;
}

/// Clear mode: private values are ordinary prover messages, derived claims
/// ordinary public values, and no pads are used.
macro_rules! clear_private_channel {
    ($($transcript:tt)*) => {
        impl<H: Sponge> PrivateChannel for $($transcript)* {
            type Error = TranscriptError;

            fn exchange_private<F, E>(&mut self, values: &mut [E]) -> Result<PadSlots, TranscriptError>
            where
                F: Field + CanonicalEncoding,
                E: ExtField<F> + CanonicalDecode,
            {
                self.exchange_all(values)?;
                Ok(PadSlots::default())
            }

            fn bind_derived<F, E>(&mut self, claim: &E) -> Result<(), TranscriptError>
            where
                F: Field + CanonicalEncoding,
                E: ExtField<F> + CanonicalDecode,
            {
                self.public(claim);
                Ok(())
            }

            fn exchange_private_base<F>(&mut self, values: &mut [F]) -> Result<PadSlots, TranscriptError>
            where
                F: Field + CanonicalEncoding,
            {
                self.exchange_all(values)?;
                Ok(PadSlots::default())
            }

            fn bind_derived_base<F>(&mut self, claim: &F) -> Result<(), TranscriptError>
            where
                F: Field + CanonicalEncoding,
            {
                self.public(claim);
                Ok(())
            }
        }
    };
}

clear_private_channel!(ProverTranscript<H>);
clear_private_channel!(VerifierTranscript<'_, H>);
