//! q128 operations on the legacy chained transcript, using shared field handles.
use jolt_field::Fr;
use jolt_r1cs::bn254_bits::ByteVar;
use jolt_r1cs::fp128_bn254::{Fp128Error, Fp128Var};
use jolt_r1cs::R1csBuilder;
use thiserror::Error;

use super::{Blake2bR1csError, LegacyBlake2bVar};
use crate::{Label, LabelWithCount};

/// Framing or foreign-field constraint failure.
#[derive(Debug, Error)]
pub enum Fp128TranscriptError {
    /// Invalid framing or exhausted public schedule.
    #[error(transparent)]
    Blake(#[from] Blake2bR1csError),
    /// Invalid foreign-field handle.
    #[error(transparent)]
    Field(#[from] Fp128Error),
}

impl LegacyBlake2bVar {
    /// Native raw challenge: advance the full state and reduce its first 16 LE bytes.
    /// Native q128 scalar challenges have the same decoding, but callers retain
    /// the raw/scalar operation identity of their protocol.
    pub fn challenge_fp128(
        &mut self,
        builder: &mut R1csBuilder<Fr>,
    ) -> Result<Fp128Var, Fp128TranscriptError> {
        self.append_bytes(builder, &[])?;
        let bytes = self.state().iter().take(16).cloned().collect::<Vec<_>>();
        let bytes = bytes
            .try_into()
            .map_err(|_| Blake2bR1csError::InvalidLength)?;
        Ok(Fp128Var::reduce_le_bytes(builder, &bytes)?)
    }

    /// Native q128 scalar decoding uses the same sixteen-byte reduction as raw draws.
    pub fn challenge_scalar_fp128(
        &mut self,
        builder: &mut R1csBuilder<Fr>,
    ) -> Result<Fp128Var, Fp128TranscriptError> {
        self.challenge_fp128(builder)
    }

    /// Absorb a canonical q128 handle in the native big-endian field encoding.
    pub fn append_fp128(
        &mut self,
        builder: &mut R1csBuilder<Fr>,
        value: &Fp128Var,
    ) -> Result<(), Fp128TranscriptError> {
        let mut bytes = value.to_le_bytes(builder)?;
        bytes.reverse();
        Ok(self.append_bytes(builder, &bytes)?)
    }

    /// Absorb the native padded label serializer as one transition.
    pub fn append_label(
        &mut self,
        builder: &mut R1csBuilder<Fr>,
        label: &'static [u8],
    ) -> Result<(), Blake2bR1csError> {
        if label.len() > 32 {
            return Err(Blake2bR1csError::LabelTooLong);
        }
        self.append_bytes(builder, &Label(label).to_bytes().map(ByteVar::constant))
    }

    /// Absorb the native count framing as one transition, before separate fields.
    pub fn append_label_with_count(
        &mut self,
        builder: &mut R1csBuilder<Fr>,
        label: &'static [u8],
        count: u64,
    ) -> Result<(), Blake2bR1csError> {
        if label.len() > 24 {
            return Err(Blake2bR1csError::LabelTooLong);
        }
        self.append_bytes(
            builder,
            &LabelWithCount(label, count)
                .to_bytes()
                .map(ByteVar::constant),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_schedule_and_label_errors_precede_allocation() {
        let mut builder = R1csBuilder::new();
        let mut transcript = LegacyBlake2bVar {
            state: std::array::from_fn(|_| ByteVar::constant(0)),
            round: u32::MAX,
        };
        let before = builder.num_vars();
        assert!(transcript.check_schedule(0).is_ok());
        assert!(transcript.check_schedule(1).is_err());
        assert!(transcript.challenge_fp128(&mut builder).is_err());
        assert!(transcript.append_label(&mut builder, &[0; 33]).is_err());
        assert!(transcript
            .append_label_with_count(&mut builder, &[0; 25], 0)
            .is_err());
        assert_eq!(builder.num_vars(), before);
        assert_eq!(transcript.round, u32::MAX);
    }
}
