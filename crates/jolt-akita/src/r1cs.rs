//! Akita epoch-6 transcript bytes and bounded terminal integer relations.
//! Contracts: `specs/akita-wrapper/duplex-constraints.md` and
//! `specs/akita-wrapper/terminal-constraints.md`.

use akita_transcript::{PROTOCOL_TAG, SESSION_DOMAIN_TAG};
use jolt_field::Fr;
use jolt_r1cs::{bn254_bits::ByteVar, R1csBuilder};
use jolt_transcript::r1cs::{Blake2bDuplexVar, Blake2bR1csError};

/// Byte transcript for a fixed Akita verifier replay schedule.
///
/// Session and instance bytes may be witnesses, but their lengths and all call
/// boundaries are fixed circuit shape. The caller must constrain their canonical
/// encoding, authenticate the instance, and bind the session's Jolt bridge scalar.
/// Labels are intentionally absent: upstream ignores operation labels. This does
/// not implement scalar decoding, rejection sampling, grinding, or proof parsing.
/// ONE must be externally fixed and all byte handles must use the same builder.
#[derive(Clone, Debug)]
pub struct AkitaTranscriptVar {
    sponge: Blake2bDuplexVar,
}

impl AkitaTranscriptVar {
    /// Initialize protocol tag, session tag, and length-prefixed session/instance.
    /// The instance must be the actual bound descriptor, not a placeholder.
    pub fn new(
        builder: &mut R1csBuilder<Fr>,
        session: &[ByteVar],
        instance: &[ByteVar],
    ) -> Result<Self, Blake2bR1csError> {
        let mut transcript = Self {
            sponge: Blake2bDuplexVar::default(),
        };
        for tag in [PROTOCOL_TAG.as_slice(), SESSION_DOMAIN_TAG.as_slice()] {
            let mut padded: Vec<_> = tag.iter().copied().map(ByteVar::constant).collect();
            padded.resize_with(64, || ByteVar::constant(0));
            transcript.sponge.absorb(builder, &padded)?;
        }
        transcript.append_bytes(builder, session)?;
        transcript.append_bytes(builder, instance)?;
        Ok(transcript)
    }

    /// Absorb LE-u64 byte length followed by exactly these bytes.
    pub fn append_bytes(
        &mut self,
        builder: &mut R1csBuilder<Fr>,
        bytes: &[ByteVar],
    ) -> Result<(), Blake2bR1csError> {
        let len = u64::try_from(bytes.len()).map_err(|_| Blake2bR1csError::MessageTooLong)?;
        let mut framed: Vec<_> = len.to_le_bytes().map(ByteVar::constant).into();
        framed.extend_from_slice(bytes);
        self.sponge.absorb(builder, &framed)
    }

    /// Consume whole 32-byte blocks, returning only the requested prefix.
    /// A zero-length request is a no-op, as in Akita's `squeeze_bytes`.
    pub fn challenge_bytes(
        &mut self,
        builder: &mut R1csBuilder<Fr>,
        len: usize,
    ) -> Result<Vec<ByteVar>, Blake2bR1csError> {
        if len == 0 {
            return Ok(Vec::new());
        }
        let rounded = len
            .checked_add(31)
            .map(|n| n / 32 * 32)
            .ok_or(Blake2bR1csError::MessageTooLong)?;
        let mut bytes = self.sponge.squeeze(builder, rounded)?;
        bytes.truncate(len);
        Ok(bytes)
    }
}

/// Bounded terminal integer relations; excludes dynamic routing and full acceptance.
pub mod terminal;

/// Fixed-profile point preparation and scalar-opening constraints.
pub mod scalar;

mod sparse_stream;
pub use sparse_stream::AkitaSparseStreamVar;
