//! D64 EvaluationTrace FoldDraw framing on an inherited constrained transcript.
use super::{AkitaTranscriptVar, D64AcceptedVar, D64RetryProfile, SparseRetryError};
use akita_challenges::{
    FoldChallengeDrawDomain, FoldChallengeFrame, OperatorNormRejection,
    D64_SELECTIVE_L2_CHALLENGE_CONFIG,
};
use akita_transcript::FOLD_CHALLENGE_SEED_LEN;
use akita_types::FOLD_RESPONSE_NONCE_BITS;
use jolt_field::{CanonicalEncoding, Fr, Ring};
use jolt_r1cs::{
    bn254_bits::{BitsError, ByteVar},
    LinearCombination, R1csBuilder,
};
use jolt_transcript::r1cs::Blake2bR1csError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FoldDrawError {
    #[error("invalid public fold-draw or nonce-slot shape")]
    Shape,
    #[error("native fold frame: {0}")]
    Native(String),
    #[error(transparent)]
    Bits(#[from] BitsError),
    #[error(transparent)]
    Blake(#[from] Blake2bR1csError),
    #[error(transparent)]
    Retry(#[from] SparseRetryError),
}

/// Canonical zero-extended LE-u32 from one native-width packed nonce slot.
/// Site/order, stream length/padding and slot offset must match the public plan.
pub struct FoldResponseNonceVar([ByteVar; 4]);
impl FoldResponseNonceVar {
    /// Extract low-bit-first native FoldResponse bits at a fixed public bit offset.
    /// Every source handle must belong to this builder; ONE is fixed externally.
    pub fn from_packed(
        builder: &mut R1csBuilder<Fr>,
        stream: &[ByteVar],
        bit_offset: usize,
    ) -> Result<Self, FoldDrawError> {
        let width = usize::from(FOLD_RESPONSE_NONCE_BITS);
        let end = bit_offset.checked_add(width).ok_or(FoldDrawError::Shape)?;
        if width > 32 || stream.len().checked_mul(8).is_none_or(|n| end > n) {
            return Err(FoldDrawError::Shape);
        }
        for byte in stream {
            byte.validate_indices(builder)?;
        }
        let mut bytes = Vec::with_capacity(4);
        for byte_index in 0..4 {
            let mut expression = LinearCombination::zero();
            for bit in 0..8 {
                let local = byte_index * 8 + bit;
                if local < width {
                    let source = bit_offset + local;
                    let byte = stream.get(source / 8).ok_or(FoldDrawError::Shape)?;
                    let bits = byte.bit_expressions();
                    let input = bits.get(source % 8).ok_or(FoldDrawError::Shape)?.clone();
                    expression = expression + input.scale(Fr::from_u64(1 << bit));
                }
            }
            let value = builder
                .evaluate(&expression)
                .ok()
                .and_then(|v| v.to_u64_checked())
                .and_then(|v| u8::try_from(v).ok());
            let byte = ByteVar::allocate(builder, value);
            builder.assert_equal(byte.expression(), expression);
            bytes.push(byte);
        }
        Ok(Self(bytes.try_into().map_err(|_| FoldDrawError::Shape)?))
    }
}

/// Fixed public metadata for the supported D64 selective-L2 evaluation route.
/// This is not a proof that metadata matches an authenticated setup or schedule.
pub struct D64FoldDrawShape(FoldChallengeFrame);
impl D64FoldDrawShape {
    /// Bind checked public group counts to the native D64 selective-L2 frame.
    pub fn new(
        group_index: usize,
        live_blocks: usize,
        claims: usize,
    ) -> Result<Self, FoldDrawError> {
        Ok(Self(
            FoldChallengeFrame::new(
                FoldChallengeDrawDomain::EvaluationTrace,
                64,
                group_index,
                live_blocks,
                claims,
                &D64_SELECTIVE_L2_CHALLENGE_CONFIG,
                Some(OperatorNormRejection::D64_SELECTIVE_L2),
            )
            .map_err(|e| FoldDrawError::Native(e.to_string()))?,
        ))
    }

    /// Append the canonical native payload and consume one native 32-byte root block.
    /// Inherited transcript state must include the actual prior verifier replay.
    /// The nonce read itself has no PoW absorb/squeeze in the native verifier.
    pub fn draw(
        &self,
        builder: &mut R1csBuilder<Fr>,
        transcript: &mut AkitaTranscriptVar,
        nonce: &FoldResponseNonceVar,
    ) -> Result<D64FoldDrawVar, FoldDrawError> {
        let mut payload: Vec<_> = self
            .0
            .prefix()
            .iter()
            .copied()
            .map(ByteVar::constant)
            .collect();
        payload.extend_from_slice(&nonce.0);
        payload.extend(self.0.suffix().iter().copied().map(ByteVar::constant));
        transcript.append_bytes(builder, &payload)?;
        let root = transcript
            .challenge_bytes(builder, FOLD_CHALLENGE_SEED_LEN)?
            .try_into()
            .map_err(|_| FoldDrawError::Shape)?;
        Ok(D64FoldDrawVar {
            root,
            coordinates: self.0.coordinate_count(),
        })
    }
}

/// A FoldDraw root derived from constrained transcript state, not a supplied hint.
pub struct D64FoldDrawVar {
    root: [ByteVar; FOLD_CHALLENGE_SEED_LEN],
    coordinates: usize,
}
impl D64FoldDrawVar {
    /// Constrain one native flat callback index in this group's claim-major range.
    /// Other coordinates and their verifier uses remain separate obligations.
    pub fn sample_coordinate(
        &self,
        builder: &mut R1csBuilder<Fr>,
        coordinate: usize,
        capacity: D64RetryProfile,
    ) -> Result<D64AcceptedVar, FoldDrawError> {
        if coordinate >= self.coordinates {
            return Err(FoldDrawError::Shape);
        }
        let coordinate = u64::try_from(coordinate).map_err(|_| FoldDrawError::Shape)?;
        Ok(capacity.sample(builder, &self.root, coordinate)?)
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "native parity and adversarial witnesses"
)]
mod tests {
    use super::*;
    use crate::AkitaField;
    use akita_challenges::{FoldDraw, LiveFoldDraw};
    use akita_pcs::{AkitaTranscript, Transcript};
    use jolt_r1cs::Variable;

    fn initial(
        builder: &mut R1csBuilder<Fr>,
        value: Option<u8>,
    ) -> (AkitaTranscriptVar, Vec<ByteVar>) {
        let session: Vec<_> = b"fold-draw-binding"
            .iter()
            .copied()
            .map(ByteVar::constant)
            .collect();
        let instance: Vec<_> = b"descriptor fixture"
            .iter()
            .copied()
            .map(ByteVar::constant)
            .collect();
        let mut transcript = AkitaTranscriptVar::new(builder, &session, &instance).unwrap();
        let previous = vec![ByteVar::allocate(builder, value)];
        transcript.append_bytes(builder, &previous).unwrap();
        (transcript, previous)
    }
    fn native_initial() -> AkitaTranscript<AkitaField> {
        let mut transcript = AkitaTranscript::verifier(b"fold-draw-binding", b"descriptor fixture");
        transcript.append_bytes(b"prior proof", &[17]);
        transcript
    }
    fn packed(builder: &mut R1csBuilder<Fr>, nonce: Option<u32>) -> Vec<ByteVar> {
        (0..3)
            .map(|i| {
                ByteVar::allocate(builder, nonce.map(|n| (((n << 5) | 0x15) >> (8 * i)) as u8))
            })
            .collect()
    }
    fn bit_variable(byte: &ByteVar, bit: usize) -> Variable {
        byte.bit_expressions()[bit]
            .terms
            .iter()
            .find(|(v, _)| *v != Variable::ONE)
            .unwrap()
            .0
    }

    #[test]
    fn native_draw_to_indexed_accepted_output_and_transcript_continuation() {
        let mut native = native_initial();
        let nonce_value = 0;
        let expected = LiveFoldDraw::<AkitaField, _>::new(&mut native)
            .draw_folding_challenges_with_rejection(
                FoldChallengeDrawDomain::EvaluationTrace,
                64,
                2,
                2,
                1,
                &D64_SELECTIVE_L2_CHALLENGE_CONFIG,
                nonce_value,
                Some(OperatorNormRejection::D64_SELECTIVE_L2),
            )
            .unwrap();
        let mut builder = R1csBuilder::new();
        let (mut transcript, _) = initial(&mut builder, Some(17));
        let stream = packed(&mut builder, Some(nonce_value));
        let nonce = FoldResponseNonceVar::from_packed(&mut builder, &stream, 5).unwrap();
        let draw = D64FoldDrawShape::new(2, 2, 1)
            .unwrap()
            .draw(&mut builder, &mut transcript, &nonce)
            .unwrap();
        let capacity = D64RetryProfile::new(2, 4).unwrap();
        let before = builder.num_vars();
        assert!(matches!(
            draw.sample_coordinate(&mut builder, 2, capacity),
            Err(FoldDrawError::Shape)
        ));
        assert_eq!(builder.num_vars(), before);
        let accepted = draw.sample_coordinate(&mut builder, 1, capacity).unwrap();
        let challenge = &expected.as_slice()[1];
        let mut dense = [Fr::from_u64(0); 64];
        for (&p, &c) in challenge.positions.iter().zip(&challenge.coeffs) {
            let m = Fr::from_u64(c.unsigned_abs().into());
            dense[p as usize] = if c < 0 { -m } else { m };
        }
        for (&variable, value) in accepted.coefficients().iter().zip(dense) {
            builder.assert_equal(variable, LinearCombination::constant(value));
        }
        let next = native.challenge_bytes(b"continuation", 32);
        for (byte, value) in transcript
            .challenge_bytes(&mut builder, 32)
            .unwrap()
            .iter()
            .zip(next)
        {
            builder.assert_equal(
                byte.expression(),
                LinearCombination::constant(Fr::from_u64(value.into())),
            );
        }
        let witness = builder.witness().unwrap();
        assert!(builder.into_matrices().check_witness(&witness).is_ok());
    }

    #[test]
    fn nonce_and_prior_proof_binding_unknown_shape_and_coherent_tamper() {
        let emit = |value: Option<u32>| {
            let mut builder = R1csBuilder::new();
            let (mut transcript, previous) = initial(&mut builder, value.map(|_| 17));
            let stream = packed(&mut builder, value);
            let nonce = FoldResponseNonceVar::from_packed(&mut builder, &stream, 5).unwrap();
            let _ = D64FoldDrawShape::new(2, 2, 1)
                .unwrap()
                .draw(&mut builder, &mut transcript, &nonce)
                .unwrap();
            let witness = value.map(|_| builder.witness().unwrap());
            let matrices = builder.into_matrices();
            if let Some(witness) = witness {
                assert!(matrices.check_witness(&witness).is_ok());
                let mut coherent = witness.clone();
                for v in [bit_variable(&stream[0], 5), bit_variable(&nonce.0[0], 0)] {
                    coherent[v.index()] = Fr::from_u64(1) - coherent[v.index()];
                }
                assert!(matrices.check_witness(&coherent).is_err());
                for v in [bit_variable(&previous[0], 0), bit_variable(&nonce.0[2], 0)] {
                    let mut bad = witness.clone();
                    bad[v.index()] = Fr::from_u64(1) - bad[v.index()];
                    assert!(matrices.check_witness(&bad).is_err());
                }
            }
            matrices
        };
        let known = emit(Some(0));
        for value in [Some(4095), None] {
            let other = emit(value);
            assert_eq!(known.num_vars, other.num_vars);
            assert_eq!(known.a, other.a);
            assert_eq!(known.b, other.b);
            assert_eq!(known.c, other.c);
        }
    }

    #[test]
    fn domain_binding_and_invalid_public_shapes() {
        let mut builder = R1csBuilder::new();
        let (mut transcript, _) = initial(&mut builder, Some(17));
        let stream = packed(&mut builder, Some(0));
        assert!(FoldResponseNonceVar::from_packed(&mut builder, &stream, usize::MAX).is_err());
        assert!(FoldResponseNonceVar::from_packed(&mut builder, &stream[..1], 0).is_err());
        assert!(D64FoldDrawShape::new(0, 0, 1).is_err());
        let nonce = FoldResponseNonceVar::from_packed(&mut builder, &stream, 5).unwrap();
        let draw = D64FoldDrawShape::new(2, 2, 1)
            .unwrap()
            .draw(&mut builder, &mut transcript, &nonce)
            .unwrap();
        let mut other = native_initial();
        let wrong_domain = FoldChallengeFrame::new(
            FoldChallengeDrawDomain::EvaluationTrace,
            64,
            2,
            2,
            1,
            &D64_SELECTIVE_L2_CHALLENGE_CONFIG,
            None,
        )
        .unwrap()
        .encode(0);
        other.append_bytes(b"native ignored label", &wrong_domain);
        let wrong_root = other.challenge_block(b"root");
        // Coherent alternate-domain root claim; the original constrained frame remains fixed.
        for (byte, value) in draw.root.iter().zip(wrong_root) {
            let claimed = ByteVar::allocate(&mut builder, Some(value));
            builder.assert_equal(byte.expression(), claimed.expression());
        }
        let witness = builder.witness().unwrap();
        assert!(builder.into_matrices().check_witness(&witness).is_err());
    }
}
