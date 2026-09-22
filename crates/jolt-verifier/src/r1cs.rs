//! Conditional stage-1 uni-skip constraints for a private q128 proof.
//! The inherited transcript prefix is NOT authenticated here. Checked public
//! input/setup binding, the remainder and later stages are separate obligations.
use jolt_field::Fr;
use jolt_r1cs::fp128_bn254::{Fp128Error, Fp128Var};
use jolt_r1cs::{LinearCombination, R1csBuilder};
use jolt_sumcheck::r1cs::fp128::{Fp128FullRoundShape, Fp128SumcheckError};
use jolt_sumcheck::{
    CenteredIntegerDomain, OPENING_CLAIM_TRANSCRIPT_LABEL, UNISKIP_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::r1cs::{Blake2bR1csError, Fp128TranscriptError, LegacyBlake2bVar};
use thiserror::Error;

use crate::stages::uniskip::{spartan_outer_tau_count, UniskipParams};

#[derive(Debug, Error)]
pub enum Stage1R1csError {
    #[error("log_t must be below 64")]
    InvalidTraceLog,
    #[error(transparent)]
    Round(#[from] Fp128SumcheckError),
    #[error(transparent)]
    Field(#[from] Fp128Error),
    #[error(transparent)]
    Transcript(#[from] Fp128TranscriptError),
    #[error(transparent)]
    Blake(#[from] Blake2bR1csError),
}

/// One fixed nonempty coefficient encoding of the native outer uni-skip.
/// Empty native zero-polynomial encodings are excluded from this profile.
pub struct Stage1UniskipShape {
    tau_count: u32,
    round: Fp128FullRoundShape,
}

/// Handles carried into the future remainder, all from the same q128 transcript.
pub struct Stage1UniskipVars {
    pub tau: Vec<Fp128Var>,
    pub challenge: Fp128Var,
    pub claim: Fp128Var,
}

impl Stage1UniskipShape {
    pub fn new(log_t: usize, coefficient_count: usize) -> Result<Self, Stage1R1csError> {
        if log_t >= 64 {
            return Err(Stage1R1csError::InvalidTraceLog);
        }
        let tau_count = u32::try_from(spartan_outer_tau_count(log_t))
            .map_err(|_| Stage1R1csError::InvalidTraceLog)?;
        let params = UniskipParams::spartan_outer();
        let round = Fp128FullRoundShape::new(
            coefficient_count,
            params.degree(),
            CenteredIntegerDomain::new(params.domain_size()),
            UNISKIP_ROUND_TRANSCRIPT_LABEL,
        )?;
        Ok(Self { tau_count, round })
    }

    /// Draw native outer tau, check the zero-sum full round, bind and absorb the
    /// supplied opening before any remainder draw. The builder's ONE and all
    /// handles' builder provenance are obligations of the eventual wrapper.
    pub fn constrain(
        &self,
        builder: &mut R1csBuilder<Fr>,
        transcript: &mut LegacyBlake2bVar,
        coefficients: &[Fp128Var],
        output: &Fp128Var,
    ) -> Result<Stage1UniskipVars, Stage1R1csError> {
        self.round.validate(builder, coefficients, output)?;
        let transitions = self
            .tau_count
            .checked_add(self.round.transitions())
            .and_then(|count| count.checked_add(2))
            .ok_or(Blake2bR1csError::RoundOverflow)?;
        transcript.check_schedule(transitions)?;
        let tau = (0..self.tau_count)
            .map(|_| transcript.challenge_fp128(builder))
            .collect::<Result<Vec<_>, _>>()?;
        let zero = Fp128Var::allocate(builder, Some(0))?;
        builder.assert_equal(zero.variable(), LinearCombination::zero());
        let reduction = self
            .round
            .constrain(builder, transcript, coefficients, &zero)?;
        builder.assert_equal(reduction.claim.variable(), output.variable());
        transcript.append_label(builder, OPENING_CLAIM_TRANSCRIPT_LABEL)?;
        transcript.append_fp128(builder, output)?;
        Ok(Stage1UniskipVars {
            tau,
            challenge: reduction.challenge,
            claim: reduction.claim,
        })
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    reason = "tests inspect fixed witnesses and native verifier vectors"
)]
mod tests {
    use super::*;
    use crate::stages::uniskip;
    use jolt_field::{Prime128OffsetA7F7, Ring};
    use jolt_poly::UnivariatePoly;
    use jolt_r1cs::bn254_bits::ByteVar;
    use jolt_r1cs::{ConstraintMatrices, Variable};
    use jolt_sumcheck::{ClearProof, ClearSumcheckProof, SumcheckClaim, SumcheckProof};
    use jolt_transcript::{LegacyBlake2bTranscript, Transcript};

    type Q = Prime128OffsetA7F7;

    fn native(scale: u64) -> (Vec<Q>, Q, Q, [u8; 32]) {
        let mut transcript = LegacyBlake2bTranscript::<Q>::new(b"conditional-uniskip");
        transcript.append_bytes(&[7, 11, 13]);
        let tau = uniskip::draw_spartan_outer_tau(&mut transcript, 0);
        let coefficients = vec![-Q::from_u64(scale), Q::from_u64(2 * scale)];
        let polynomial = UnivariatePoly::new(coefficients.clone());
        let mut oracle = transcript.clone();
        let proof: SumcheckProof<Q, ()> =
            SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof {
                round_polynomials: vec![polynomial.clone()],
            }));
        let params = UniskipParams::spartan_outer();
        let reduction = proof
            .verify(
                &SumcheckClaim::new(1, params.degree(), Q::from_u64(0)),
                CenteredIntegerDomain::new(params.domain_size()),
                UNISKIP_ROUND_TRANSCRIPT_LABEL,
                &mut oracle,
            )
            .unwrap();
        let challenge = uniskip::verify_clear(
            &proof,
            &params,
            Q::from_u64(0),
            reduction.value,
            &mut transcript,
        )
        .unwrap();
        (tau, challenge, reduction.value, transcript.state())
    }

    struct Circuit {
        matrices: ConstraintMatrices<Fr>,
        witness: Option<Vec<Fr>>,
        challenge: Variable,
    }

    fn circuit(scale: u64, known: bool, frozen_output: Q) -> Circuit {
        let mut builder = R1csBuilder::new();
        let coefficients = [-Q::from_u64(scale), Q::from_u64(2 * scale)].map(|value| {
            Fp128Var::allocate(&mut builder, known.then_some(value.to_canonical_u128())).unwrap()
        });
        let output = Fp128Var::allocate(
            &mut builder,
            known.then_some(frozen_output.to_canonical_u128()),
        )
        .unwrap();
        builder.assert_equal(
            output.variable(),
            LinearCombination::constant(Fr::from_u128(frozen_output.to_canonical_u128())),
        );
        let mut transcript = LegacyBlake2bVar::new(&mut builder, b"conditional-uniskip").unwrap();
        let prefix = [7, 11, 13].map(|byte| ByteVar::allocate(&mut builder, known.then_some(byte)));
        transcript.append_bytes(&mut builder, &prefix).unwrap();
        let result = Stage1UniskipShape::new(0, 2)
            .unwrap()
            .constrain(&mut builder, &mut transcript, &coefficients, &output)
            .unwrap();
        if known {
            let (tau, challenge, claim, state) = native(scale);
            for (actual, expected) in result.tau.iter().zip(tau) {
                assert_eq!(
                    builder.evaluate(&actual.variable().into()).unwrap(),
                    Fr::from_u128(expected.to_canonical_u128())
                );
            }
            assert_eq!(
                builder
                    .evaluate(&result.challenge.variable().into())
                    .unwrap(),
                Fr::from_u128(challenge.to_canonical_u128())
            );
            assert_eq!(
                builder.evaluate(&result.claim.variable().into()).unwrap(),
                Fr::from_u128(claim.to_canonical_u128())
            );
            if claim == frozen_output {
                for (actual, expected) in transcript.state().iter().zip(state) {
                    assert_eq!(
                        builder.evaluate(&actual.expression()).unwrap(),
                        Fr::from_u64(u64::from(expected))
                    );
                }
            }
        }
        let witness = builder.witness().ok();
        Circuit {
            matrices: builder.into_matrices(),
            witness,
            challenge: result.challenge.variable(),
        }
    }

    #[test]
    fn upstream_uniskip_native_parity_and_coherent_rejection() {
        let (_, _, output, _) = native(1);
        let honest = circuit(1, true, output);
        let mut witness = honest.witness.unwrap();
        honest.matrices.check_witness(&witness).unwrap();
        witness[honest.challenge.index()] += Fr::from_u64(1);
        assert!(honest.matrices.check_witness(&witness).is_err());
        let altered = circuit(2, true, output);
        assert_eq!(honest.matrices.a, altered.matrices.a);
        assert_eq!(honest.matrices.b, altered.matrices.b);
        assert_eq!(honest.matrices.c, altered.matrices.c);
        assert!(altered
            .matrices
            .check_witness(&altered.witness.unwrap())
            .is_err());
    }

    #[test]
    fn upstream_uniskip_unknown_shape_and_public_validation() {
        let (_, _, output, _) = native(1);
        let honest = circuit(1, true, output);
        let unknown = circuit(1, false, output);
        assert!(unknown.witness.is_none());
        assert_eq!(honest.matrices.num_vars, unknown.matrices.num_vars);
        assert_eq!(honest.matrices.a, unknown.matrices.a);
        assert_eq!(honest.matrices.b, unknown.matrices.b);
        assert_eq!(honest.matrices.c, unknown.matrices.c);
        assert!(Stage1UniskipShape::new(63, 2).is_ok());
        assert!(Stage1UniskipShape::new(64, 2).is_err());
        assert!(Stage1UniskipShape::new(0, 0).is_err());
        let empty: SumcheckProof<Q, ()> =
            SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof {
                round_polynomials: vec![UnivariatePoly::new(vec![])],
            }));
        let mut native = LegacyBlake2bTranscript::<Q>::new(b"empty-native-round");
        assert!(uniskip::verify_clear(
            &empty,
            &UniskipParams::spartan_outer(),
            Q::from_u64(0),
            Q::from_u64(0),
            &mut native
        )
        .is_ok());
        assert!(Stage1UniskipShape::new(0, UniskipParams::spartan_outer().degree() + 2).is_err());
        let mut builder = R1csBuilder::new();
        let output = Fp128Var::allocate(&mut builder, None).unwrap();
        let mut transcript = LegacyBlake2bVar::new(&mut builder, b"shape").unwrap();
        let before = builder.num_vars();
        assert!(Stage1UniskipShape::new(0, 2)
            .unwrap()
            .constrain(&mut builder, &mut transcript, &[], &output)
            .is_err());
        assert_eq!(builder.num_vars(), before);
    }
}

#[cfg(feature = "akita")]
mod boundary;
#[cfg(feature = "akita")]
pub use boundary::{
    AkitaStage1BoundaryShape, AkitaStage1BoundaryVars, AkitaStage1BoundaryWitness, BoundaryError,
    BoundaryPublicInputs,
};

#[cfg(feature = "akita")]
mod remainder;
#[cfg(feature = "akita")]
pub use remainder::{
    Stage1RemainderError, Stage1RemainderShape, Stage1RemainderVars, Stage1RemainderWitness,
};
