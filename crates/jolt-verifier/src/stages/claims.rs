//! Stage-local inputs for evaluating symbolic Jolt claim formulas.

use std::collections::BTreeMap;

use jolt_claims::{
    protocols::jolt::{
        JoltChallengeId, JoltConsistencyClaim, JoltExpr, JoltOpeningId, JoltPublicId,
        JoltStageClaims, JoltSumcheckDomain,
    },
    Source,
};
use jolt_field::Field;
use jolt_sumcheck::{
    BooleanHypercube, CenteredIntegerDomain, ClearProof, EvaluationClaim, SumcheckClaim,
    SumcheckVerifier,
};
use jolt_transcript::Transcript;

use crate::VerifierError;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StageClaimInputs<F: Field> {
    openings: BTreeMap<JoltOpeningId, F>,
    challenges: BTreeMap<JoltChallengeId, F>,
    publics: BTreeMap<JoltPublicId, F>,
}

impl<F: Field> StageClaimInputs<F> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_opening(mut self, id: JoltOpeningId, value: F) -> Self {
        let _ = self.insert_opening(id, value);
        self
    }

    pub fn with_challenge(mut self, id: JoltChallengeId, value: F) -> Self {
        let _ = self.insert_challenge(id, value);
        self
    }

    pub fn with_public(mut self, id: JoltPublicId, value: F) -> Self {
        let _ = self.insert_public(id, value);
        self
    }

    pub fn insert_opening(&mut self, id: JoltOpeningId, value: F) -> Option<F> {
        self.openings.insert(id, value)
    }

    pub fn insert_challenge(&mut self, id: JoltChallengeId, value: F) -> Option<F> {
        self.challenges.insert(id, value)
    }

    pub fn insert_public(&mut self, id: JoltPublicId, value: F) -> Option<F> {
        self.publics.insert(id, value)
    }

    pub fn opening_claim(&self, id: JoltOpeningId) -> Result<F, VerifierError> {
        self.openings
            .get(&id)
            .copied()
            .ok_or(VerifierError::MissingStageClaimOpening { id })
    }

    pub fn challenge(&self, id: JoltChallengeId) -> Result<F, VerifierError> {
        self.challenges
            .get(&id)
            .copied()
            .ok_or(VerifierError::MissingStageClaimChallenge { id })
    }

    pub fn public(&self, id: JoltPublicId) -> Result<F, VerifierError> {
        self.publics
            .get(&id)
            .copied()
            .ok_or(VerifierError::MissingStageClaimPublic { id })
    }

    pub fn evaluate(&self, expression: &JoltExpr<F>) -> Result<F, VerifierError> {
        let mut result = F::zero();
        for term in &expression.terms {
            let mut value = term.coefficient;
            for factor in &term.factors {
                value *= match factor {
                    Source::Opening(id) => self.opening_claim(*id)?,
                    Source::Challenge(id) => self.challenge(*id)?,
                    Source::Public(id) => self.public(*id)?,
                };
            }
            result += value;
        }
        Ok(result)
    }

    pub fn evaluate_input_claim(&self, claims: &JoltStageClaims<F>) -> Result<F, VerifierError> {
        self.evaluate(&claims.input.expression)
    }

    pub fn evaluate_output_claim(&self, claims: &JoltStageClaims<F>) -> Result<F, VerifierError> {
        self.evaluate(&claims.output.expression)
    }

    pub fn check_consistency_claims(
        &self,
        claims: &JoltStageClaims<F>,
    ) -> Result<(), VerifierError> {
        for claim in &claims.consistency {
            match claim {
                JoltConsistencyClaim::SameEvaluation(same_evaluation) => {
                    let left = self.opening_claim(same_evaluation.left)?;
                    let right = self.opening_claim(same_evaluation.right)?;
                    if left != right {
                        return Err(VerifierError::StageClaimOpeningMismatch {
                            stage: claims.id,
                            left: same_evaluation.left,
                            right: same_evaluation.right,
                        });
                    }
                }
                JoltConsistencyClaim::EqualExpressions { left, right } => {
                    if self.evaluate(left)? != self.evaluate(right)? {
                        return Err(VerifierError::StageClaimExpressionMismatch {
                            stage: claims.id,
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluatedStageClaims<F: Field> {
    pub input: F,
    pub output: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStageClaim<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: jolt_poly::Point<F>,
    pub sumcheck_final_claim: F,
    pub expected_output_claim: F,
}

pub fn evaluate_stage_claims<F: Field>(
    claims: &JoltStageClaims<F>,
    inputs: &StageClaimInputs<F>,
) -> Result<EvaluatedStageClaims<F>, VerifierError> {
    inputs.check_consistency_claims(claims)?;
    Ok(EvaluatedStageClaims {
        input: inputs.evaluate_input_claim(claims)?,
        output: inputs.evaluate_output_claim(claims)?,
    })
}

pub fn verify_clear_stage_claim<F, T>(
    claims: &JoltStageClaims<F>,
    inputs: &StageClaimInputs<F>,
    proof: &ClearProof<F>,
    compressed_round_label: &'static [u8],
    transcript: &mut T,
) -> Result<VerifiedStageClaim<F>, VerifierError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    if claims.sumcheck.degree == 0 {
        return Err(VerifierError::InvalidStageSumcheckDegree {
            stage: claims.id,
            degree: claims.sumcheck.degree,
        });
    }

    inputs.check_consistency_claims(claims)?;
    let input_claim = inputs.evaluate_input_claim(claims)?;
    let sumcheck_claim =
        SumcheckClaim::new(claims.sumcheck.rounds, claims.sumcheck.degree, input_claim);

    let sumcheck = match (claims.sumcheck.domain, proof) {
        (JoltSumcheckDomain::BooleanHypercube, ClearProof::Full(proof)) => {
            SumcheckVerifier::verify(
                &sumcheck_claim,
                &proof.round_polynomials,
                BooleanHypercube,
                transcript,
            )
        }
        (JoltSumcheckDomain::BooleanHypercube, ClearProof::Compressed(proof)) => {
            SumcheckVerifier::verify_compressed(
                &sumcheck_claim,
                proof,
                BooleanHypercube,
                compressed_round_label,
                transcript,
            )
        }
        (JoltSumcheckDomain::CenteredInteger { domain_size }, ClearProof::Full(proof)) => {
            SumcheckVerifier::verify(
                &sumcheck_claim,
                &proof.round_polynomials,
                CenteredIntegerDomain::new(domain_size),
                transcript,
            )
        }
        (JoltSumcheckDomain::CenteredInteger { .. }, ClearProof::Compressed(_)) => {
            return Err(VerifierError::CompressedStageClaimRequiresBooleanDomain {
                stage: claims.id,
            });
        }
    }
    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
        stage: claims.id,
        reason: error.to_string(),
    })?;

    let EvaluationClaim {
        point: sumcheck_point,
        value: sumcheck_final_claim,
    } = sumcheck;
    let expected_output_claim = inputs.evaluate_output_claim(claims)?;
    if sumcheck_final_claim != expected_output_claim {
        return Err(VerifierError::StageClaimOutputMismatch { stage: claims.id });
    }

    Ok(VerifiedStageClaim {
        input_claim,
        sumcheck_point,
        sumcheck_final_claim,
        expected_output_claim,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_claims::{
        challenge, constant, opening,
        protocols::jolt::{
            JoltCommittedPolynomial, JoltStageId, JoltSumcheckSpec, JoltVirtualPolynomial,
            RamReadWriteChallenge,
        },
        public, SameEvaluationAs,
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::{CompressedPoly, UnivariatePoly};
    use jolt_sumcheck::{ClearSumcheckProof, CompressedSumcheckProof, RoundMessage};
    use jolt_transcript::{Blake2bTranscript, Transcript};

    fn ram_inc() -> JoltOpeningId {
        JoltOpeningId::committed(
            JoltCommittedPolynomial::RamInc,
            JoltStageId::RamReadWriteChecking,
        )
    }

    fn ram_val() -> JoltOpeningId {
        JoltOpeningId::virtual_polynomial(
            JoltVirtualPolynomial::RamVal,
            JoltStageId::RamReadWriteChecking,
        )
    }

    fn ram_address() -> JoltOpeningId {
        JoltOpeningId::virtual_polynomial(
            JoltVirtualPolynomial::RamAddress,
            JoltStageId::RamReadWriteChecking,
        )
    }

    fn gamma() -> JoltChallengeId {
        JoltChallengeId::from(RamReadWriteChallenge::Gamma)
    }

    fn stage_claims() -> JoltStageClaims<Fr> {
        JoltStageClaims::new(
            JoltStageId::RamReadWriteChecking,
            JoltSumcheckSpec::boolean(4, 3),
            opening(ram_inc()) + challenge(gamma()) * public(JoltPublicId::TraceLength),
            opening(ram_val()) * challenge(gamma()) + constant(Fr::from_u64(9)),
        )
    }

    fn one_round_sumcheck_claim(output_claim: Fr) -> JoltStageClaims<Fr> {
        JoltStageClaims::new(
            JoltStageId::RamReadWriteChecking,
            JoltSumcheckSpec::boolean(1, 1),
            public(JoltPublicId::TraceLength),
            opening(ram_val()) + constant(output_claim - Fr::from_u64(7)),
        )
    }

    #[test]
    fn evaluates_stage_input_and_output_claims() -> Result<(), VerifierError> {
        let claims = stage_claims();
        let inputs = StageClaimInputs::new()
            .with_opening(ram_inc(), Fr::from_u64(5))
            .with_opening(ram_val(), Fr::from_u64(7))
            .with_challenge(gamma(), Fr::from_u64(11))
            .with_public(JoltPublicId::TraceLength, Fr::from_u64(13));

        let evaluated = evaluate_stage_claims(&claims, &inputs)?;

        assert_eq!(evaluated.input, Fr::from_u64(148));
        assert_eq!(evaluated.output, Fr::from_u64(86));
        Ok(())
    }

    #[test]
    fn missing_inputs_are_reported_by_source_kind() {
        let claims = stage_claims();
        let inputs = StageClaimInputs::new()
            .with_opening(ram_inc(), Fr::from_u64(5))
            .with_challenge(gamma(), Fr::from_u64(11));

        assert!(matches!(
            inputs.evaluate_input_claim(&claims),
            Err(VerifierError::MissingStageClaimPublic {
                id: JoltPublicId::TraceLength
            })
        ));
        assert!(matches!(
            inputs.evaluate_output_claim(&claims),
            Err(VerifierError::MissingStageClaimOpening { id }) if id == ram_val()
        ));
    }

    #[test]
    fn same_evaluation_consistency_uses_named_openings() {
        let claims = JoltStageClaims::new(
            JoltStageId::RamReadWriteChecking,
            JoltSumcheckSpec::boolean(4, 3),
            opening(ram_inc()),
            opening(ram_val()),
        )
        .with_consistency([ram_inc().same_evaluation_as(ram_address())]);
        let inputs = StageClaimInputs::new()
            .with_opening(ram_inc(), Fr::from_u64(5))
            .with_opening(ram_val(), Fr::from_u64(7))
            .with_opening(ram_address(), Fr::from_u64(6));

        assert!(matches!(
            inputs.check_consistency_claims(&claims),
            Err(VerifierError::StageClaimOpeningMismatch {
                stage: JoltStageId::RamReadWriteChecking,
                left,
                right,
            }) if left == ram_inc() && right == ram_address()
        ));
    }

    #[test]
    fn equal_expression_consistency_is_checked() {
        let claims = JoltStageClaims::new(
            JoltStageId::RamReadWriteChecking,
            JoltSumcheckSpec::boolean(4, 3),
            opening(ram_inc()),
            opening(ram_val()),
        )
        .with_consistency([JoltConsistencyClaim::equal_expressions(
            opening(ram_inc()) + constant(Fr::from_u64(2)),
            opening(ram_address()) * challenge(gamma()),
        )]);
        let inputs = StageClaimInputs::new()
            .with_opening(ram_inc(), Fr::from_u64(5))
            .with_opening(ram_val(), Fr::from_u64(7))
            .with_opening(ram_address(), Fr::from_u64(3))
            .with_challenge(gamma(), Fr::from_u64(2));

        assert!(matches!(
            inputs.check_consistency_claims(&claims),
            Err(VerifierError::StageClaimExpressionMismatch {
                stage: JoltStageId::RamReadWriteChecking
            })
        ));
    }

    #[test]
    fn verifies_clear_stage_claim_against_sumcheck_output() -> Result<(), VerifierError> {
        let round = UnivariatePoly::new(vec![Fr::from_u64(1), Fr::from_u64(1)]);
        let mut expected_transcript = Blake2bTranscript::<Fr>::new(b"stage_claims");
        round.append_to_transcript(&mut expected_transcript);
        let challenge = expected_transcript.challenge();
        let output_claim = round.evaluate(challenge);

        let claims = one_round_sumcheck_claim(output_claim);
        let inputs = StageClaimInputs::new()
            .with_public(JoltPublicId::TraceLength, Fr::from_u64(3))
            .with_opening(ram_val(), Fr::from_u64(7));
        let proof = ClearProof::Full(ClearSumcheckProof {
            round_polynomials: vec![round],
        });
        let mut transcript = Blake2bTranscript::<Fr>::new(b"stage_claims");

        let verified =
            verify_clear_stage_claim(&claims, &inputs, &proof, b"sumcheck_round", &mut transcript)?;

        assert_eq!(verified.input_claim, Fr::from_u64(3));
        assert_eq!(verified.sumcheck_point.as_slice(), &[challenge]);
        assert_eq!(verified.sumcheck_final_claim, output_claim);
        assert_eq!(verified.expected_output_claim, output_claim);
        Ok(())
    }

    #[test]
    fn rejects_stage_claim_output_mismatch() {
        let claims = one_round_sumcheck_claim(Fr::from_u64(99));
        let inputs = StageClaimInputs::new()
            .with_public(JoltPublicId::TraceLength, Fr::from_u64(3))
            .with_opening(ram_val(), Fr::from_u64(7));
        let proof = ClearProof::Full(ClearSumcheckProof {
            round_polynomials: vec![UnivariatePoly::new(vec![Fr::from_u64(1), Fr::from_u64(1)])],
        });
        let mut transcript = Blake2bTranscript::<Fr>::new(b"stage_claims");

        assert!(matches!(
            verify_clear_stage_claim(&claims, &inputs, &proof, b"sumcheck_round", &mut transcript),
            Err(VerifierError::StageClaimOutputMismatch {
                stage: JoltStageId::RamReadWriteChecking
            })
        ));
    }

    #[test]
    fn rejects_invalid_sumcheck_rounds_before_output_check() {
        let claims = one_round_sumcheck_claim(Fr::from_u64(99));
        let inputs = StageClaimInputs::new()
            .with_public(JoltPublicId::TraceLength, Fr::from_u64(3))
            .with_opening(ram_val(), Fr::from_u64(7));
        let proof = ClearProof::Full(ClearSumcheckProof {
            round_polynomials: Vec::new(),
        });
        let mut transcript = Blake2bTranscript::<Fr>::new(b"stage_claims");

        assert!(matches!(
            verify_clear_stage_claim(&claims, &inputs, &proof, b"sumcheck_round", &mut transcript),
            Err(VerifierError::StageClaimSumcheckFailed {
                stage: JoltStageId::RamReadWriteChecking,
                ..
            })
        ));
    }

    #[test]
    fn rejects_compressed_proof_for_centered_integer_domain() {
        let claims = JoltStageClaims::new(
            JoltStageId::SpartanOuter,
            JoltSumcheckSpec::centered_integer(10, 1, 3),
            JoltExpr::zero(),
            JoltExpr::zero(),
        );
        let proof = ClearProof::Compressed(CompressedSumcheckProof {
            round_polynomials: vec![CompressedPoly::new(vec![Fr::from_u64(1)])],
        });
        let mut transcript = Blake2bTranscript::<Fr>::new(b"stage_claims");

        assert!(matches!(
            verify_clear_stage_claim(
                &claims,
                &StageClaimInputs::new(),
                &proof,
                b"sumcheck_round",
                &mut transcript,
            ),
            Err(VerifierError::CompressedStageClaimRequiresBooleanDomain {
                stage: JoltStageId::SpartanOuter
            })
        ));
    }
}
