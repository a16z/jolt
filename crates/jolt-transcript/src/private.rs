//! The private-message seam: witness-dependent field messages, which masked
//! mode hides behind pads and clear mode exchanges as they are, and the
//! claim domains the checks on those messages are written over.

use jolt_algebra::{CanonicalDecode, CanonicalEncoding, ExtField, Field};

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

    /// Exchanges `values` in place: the prover sends them and leaves `values`
    /// unchanged; the verifier fills them from the proof.
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

/// The values a verifier check is written over, so one formula serves the
/// clear verifier ([`ClearDomain`]), the masked verifier (affine forms over
/// the pads, whose checks become residual rows) and the pad layout (symbolic
/// shape). Akita writes its masked-level stage checks once over this trait
/// (Akita `specs/whiteout-akita.md` §4.4); Jolt's residual lowering
/// implements it in `jolt-zk`.
///
/// Auxiliary slots are allocated from a formula's structure alone, never from
/// values, so every domain allocates the same slots in the same order.
pub trait ClaimDomain<F> {
    /// One value of the domain.
    type Value: Clone;
    /// Why a check or an allocation failed.
    type Error;

    /// The public constant `value`.
    fn constant(&self, value: F) -> Self::Value;

    /// `lhs + rhs`.
    fn add(&self, lhs: &Self::Value, rhs: &Self::Value) -> Self::Value;

    /// `value · scale` for a public `scale`.
    fn scale(&self, value: &Self::Value, scale: F) -> Self::Value;

    /// A new auxiliary slot defined by the affine `value`; returns the slot.
    ///
    /// # Errors
    ///
    /// The domain's error if no slot is left.
    fn define_auxiliary(&mut self, value: Self::Value) -> Result<Self::Value, Self::Error>;

    /// A new auxiliary slot holding `x · y`, where `x` is a single masked or
    /// auxiliary slot; returns the slot.
    ///
    /// # Errors
    ///
    /// The domain's error if `x` is not a single slot or no slot is left.
    fn product(&mut self, x: Self::Value, y: Self::Value) -> Result<Self::Value, Self::Error>;

    /// The check `x · y = target`, with no new slot: the product lands on an
    /// existing value, such as a sumcheck's running claim. `x` is as for
    /// [`product`](Self::product).
    ///
    /// # Errors
    ///
    /// The domain's error if the check fails where it is evaluated, or `x`
    /// is not a single slot.
    fn product_into(
        &mut self,
        x: Self::Value,
        y: Self::Value,
        target: Self::Value,
    ) -> Result<(), Self::Error>;

    /// The check `value = 0`.
    ///
    /// # Errors
    ///
    /// The domain's error if the check fails where it is evaluated.
    fn check_zero(&mut self, value: Self::Value) -> Result<(), Self::Error>;
}

/// A clear check failed: the values it compares differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("a clear claim check failed")]
pub struct ClaimMismatch;

/// The clear verifier's domain: values are field elements, auxiliary slots
/// are the values themselves, and checks compare.
#[derive(Clone, Copy, Debug, Default)]
pub struct ClearDomain;

impl<F: Field> ClaimDomain<F> for ClearDomain {
    type Value = F;
    type Error = ClaimMismatch;

    fn constant(&self, value: F) -> F {
        value
    }

    fn add(&self, lhs: &F, rhs: &F) -> F {
        *lhs + *rhs
    }

    fn scale(&self, value: &F, scale: F) -> F {
        *value * scale
    }

    fn define_auxiliary(&mut self, value: F) -> Result<F, ClaimMismatch> {
        Ok(value)
    }

    fn product(&mut self, x: F, y: F) -> Result<F, ClaimMismatch> {
        Ok(x * y)
    }

    fn product_into(&mut self, x: F, y: F, target: F) -> Result<(), ClaimMismatch> {
        self.check_zero(x * y - target)
    }

    fn check_zero(&mut self, value: F) -> Result<(), ClaimMismatch> {
        if value.is_zero() {
            Ok(())
        } else {
            Err(ClaimMismatch)
        }
    }
}
