//! In-circuit Fiat-Shamir transcript interfaces.
//!
//! The base trait owns transcript initialization and scalar challenge
//! production. Absorption is split by capability: algebraic transcripts absorb
//! field elements directly, while byte-oriented transcripts can later absorb
//! constrained bytes or bits.

use jolt_field::Field;
use jolt_r1cs::{AssignedScalar, R1csBuilder};

#[cfg(feature = "transcript-poseidon")]
mod poseidon;

#[cfg(feature = "transcript-poseidon")]
pub use poseidon::PoseidonR1csTranscript;

/// Transcript operations shared by all in-circuit Fiat-Shamir backends.
pub trait R1csTranscript<F: Field> {
    /// The challenge representation returned by this transcript.
    type Challenge;

    /// Creates a transcript with the provided protocol label.
    fn new(builder: &mut R1csBuilder<F>, label: &'static [u8]) -> Self;

    /// Squeezes a scalar challenge and advances transcript state.
    fn challenge_scalar(&mut self, builder: &mut R1csBuilder<F>) -> Self::Challenge;
}

/// In-circuit transcript backend that absorbs algebraic field elements.
pub trait R1csAlgebraicTranscript<F: Field>:
    R1csTranscript<F, Challenge = AssignedScalar<F>>
{
    /// Absorbs an assigned scalar into the transcript.
    fn absorb_scalar(&mut self, builder: &mut R1csBuilder<F>, value: AssignedScalar<F>);

    /// Absorbs a constant scalar into the transcript.
    fn absorb_constant_scalar(&mut self, builder: &mut R1csBuilder<F>, value: F) {
        self.absorb_scalar(builder, AssignedScalar::constant(value));
    }

    /// Absorbs a constant `u64` domain-separation word.
    fn absorb_u64(&mut self, builder: &mut R1csBuilder<F>, value: u64);

    /// Absorbs a constant protocol label.
    fn absorb_label(&mut self, builder: &mut R1csBuilder<F>, label: &'static [u8]);

    /// Absorbs a packed protocol label and length/count word.
    fn absorb_label_with_len(
        &mut self,
        builder: &mut R1csBuilder<F>,
        label: &'static [u8],
        len: u64,
    );
}

/// Jolt proof transcript operations over algebraic in-circuit backends.
pub trait R1csJoltTranscript<F: Field>: R1csAlgebraicTranscript<F> {
    /// Appends a domain-separation label.
    fn append_label(&mut self, builder: &mut R1csBuilder<F>, label: &'static [u8]) {
        self.absorb_label(builder, label);
    }

    /// Appends a labeled `u64`.
    fn append_u64(&mut self, builder: &mut R1csBuilder<F>, label: &'static [u8], value: u64) {
        self.absorb_label(builder, label);
        self.absorb_u64(builder, value);
    }

    /// Appends one labeled scalar.
    fn append_scalar(
        &mut self,
        builder: &mut R1csBuilder<F>,
        label: &'static [u8],
        value: AssignedScalar<F>,
    ) {
        self.absorb_label(builder, label);
        self.absorb_scalar(builder, value);
    }

    /// Appends a labeled scalar slice.
    fn append_scalars(
        &mut self,
        builder: &mut R1csBuilder<F>,
        label: &'static [u8],
        values: &[AssignedScalar<F>],
    ) {
        self.absorb_label_with_len(builder, label, values.len() as u64);
        for value in values {
            self.absorb_scalar(builder, value.clone());
        }
    }
}

/// In-circuit transcript backend that absorbs byte-oriented values.
pub trait R1csByteTranscript<F: Field>: R1csTranscript<F> {
    /// The in-circuit representation of one byte.
    type Byte;

    /// Absorbs constrained byte values into the transcript.
    fn absorb_bytes(&mut self, builder: &mut R1csBuilder<F>, bytes: &[Self::Byte]);

    /// Absorbs public constant bytes into the transcript.
    fn absorb_constant_bytes(&mut self, builder: &mut R1csBuilder<F>, bytes: &'static [u8]);
}

/// Jolt proof transcript byte operations over in-circuit backends.
pub trait R1csJoltByteTranscript<F: Field>: R1csJoltTranscript<F> + R1csByteTranscript<F> {
    /// Appends labeled byte values.
    fn append_bytes(
        &mut self,
        builder: &mut R1csBuilder<F>,
        label: &'static [u8],
        bytes: &[Self::Byte],
    ) {
        self.absorb_label_with_len(builder, label, bytes.len() as u64);
        self.absorb_bytes(builder, bytes);
    }

    /// Appends labeled constant bytes.
    fn append_constant_bytes(
        &mut self,
        builder: &mut R1csBuilder<F>,
        label: &'static [u8],
        bytes: &'static [u8],
    ) {
        self.absorb_label_with_len(builder, label, bytes.len() as u64);
        self.absorb_constant_bytes(builder, bytes);
    }
}

impl<F, T> R1csJoltTranscript<F> for T
where
    F: Field,
    T: R1csAlgebraicTranscript<F>,
{
}

impl<F, T> R1csJoltByteTranscript<F> for T
where
    F: Field,
    T: R1csJoltTranscript<F> + R1csByteTranscript<F>,
{
}
