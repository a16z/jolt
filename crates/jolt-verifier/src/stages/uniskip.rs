//! The shared uni-skip first-round verification step.
//!
//! Stages 1 and 2 each open with a univariate-skip round — a genuinely
//! different round type from the batched remainder sumchecks (separate wire
//! proof, degree-bounded single round over a centered integer domain) — before
//! their generated batch drivers run. The two stages differ only in their
//! degree/domain constants, error attribution, and how the input claim is
//! produced (stage 1: the constant zero; stage 2: the `ProductUniskip`
//! relation's fold of the stage-1 openings), so the verification core is
//! shared here.

use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_field::Field;
use jolt_r1cs::constraints::jolt::{
    SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
    SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};
use jolt_sumcheck::{
    CenteredIntegerDomain, CommittedSumcheckConsistency, SumcheckClaim, SumcheckProof,
    SumcheckStatement, UNISKIP_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::{AppendToTranscript, Transcript};

use crate::stages::zk::committed::{self, CommittedOutputClaimOutput};
use crate::verifier::CheckedInputs;
use crate::VerifierError;

/// A uni-skip round is always a single round reducing to a single challenge.
const UNISKIP_ROUNDS: usize = 1;

/// The per-stage uni-skip shape: degree/domain constants plus error
/// attribution.
pub(crate) struct UniskipParams {
    stage: JoltRelationId,
    /// The stage number reported by `StageClaimOutputMismatch`.
    stage_number: usize,
    degree: usize,
    domain_size: usize,
    /// The proof field name reported by the ZK commitment-count checks.
    proof_field: &'static str,
}

impl UniskipParams {
    /// The stage-1 Spartan outer uni-skip shape.
    pub(crate) fn spartan_outer() -> Self {
        Self {
            stage: JoltRelationId::SpartanOuter,
            stage_number: 1,
            degree: SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
            domain_size: SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
            proof_field: "stage1_uni_skip_first_round_proof",
        }
    }

    /// The stage-2 Spartan product-virtualization uni-skip shape.
    pub(crate) fn spartan_product() -> Self {
        Self {
            stage: JoltRelationId::SpartanProductVirtualization,
            stage_number: 2,
            degree: SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
            domain_size: SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
            proof_field: "stage2_uni_skip_first_round_proof",
        }
    }

    fn sumcheck_failed(&self, reason: impl ToString) -> VerifierError {
        VerifierError::StageClaimSumcheckFailed {
            stage: self.stage,
            reason: reason.to_string(),
        }
    }
}

/// The ZK uni-skip step's outputs: the committed round consistency and output
/// claim commitments (carried downstream for BlindFold), plus the reduction
/// challenge.
pub(crate) struct UniskipZk<F: Field, C> {
    pub consistency: CommittedSumcheckConsistency<F, C>,
    pub output_claims: CommittedOutputClaimOutput<C>,
    pub challenge: F,
}

/// Verify a clear-mode uni-skip round against its input and output claims and
/// return the reduction challenge.
pub(crate) fn verify_clear<F, C, T>(
    proof: &SumcheckProof<F, C>,
    params: &UniskipParams,
    input_claim: F,
    output_claim: F,
    transcript: &mut T,
) -> Result<F, VerifierError>
where
    F: Field,
    C: Clone + AppendToTranscript,
    T: Transcript<Challenge = F>,
{
    let reduction = proof
        .verify(
            &SumcheckClaim::new(UNISKIP_ROUNDS, params.degree, input_claim),
            CenteredIntegerDomain::new(params.domain_size),
            UNISKIP_ROUND_TRANSCRIPT_LABEL,
            transcript,
        )
        .map_err(|error| params.sumcheck_failed(error))?;
    if reduction.value != output_claim {
        return Err(VerifierError::StageClaimOutputMismatch {
            stage: params.stage_number,
        });
    }

    // Match the prover transcript: the uni-skip output is absorbed as an
    // opening claim before any post-uni-skip draw (the remainder batch's RLC
    // coefficient squeeze in particular).
    transcript.append_labeled(b"opening_claim", &output_claim);

    let [challenge] = reduction.point.as_slice() else {
        return Err(params.sumcheck_failed("uni-skip proof did not reduce to one challenge"));
    };
    Ok(*challenge)
}

/// Verify a ZK-mode uni-skip round: committed round consistency plus the
/// output-claim commitment count. The claims themselves stay committed
/// (BlindFold verifies them at stage 8).
pub(crate) fn verify_zk<F, C, T>(
    checked: &CheckedInputs,
    proof: &SumcheckProof<F, C>,
    params: &UniskipParams,
    transcript: &mut T,
) -> Result<UniskipZk<F, C>, VerifierError>
where
    F: Field,
    C: Clone + AppendToTranscript,
    T: Transcript<Challenge = F>,
{
    let consistency = proof
        .verify_committed_consistency(
            SumcheckStatement::new(UNISKIP_ROUNDS, params.degree),
            transcript,
        )
        .map_err(|error| params.sumcheck_failed(error))?;
    let output_claims = committed::verify_output_claim_commitments(
        checked,
        proof,
        params.proof_field,
        1,
        params.stage,
    )?;
    let [round] = consistency.rounds.as_slice() else {
        return Err(
            params.sumcheck_failed("uni-skip committed consistency did not produce one challenge")
        );
    };
    let challenge = round.challenge;
    Ok(UniskipZk {
        consistency,
        output_claims,
        challenge,
    })
}

/// Twin-transcript lock for the shared uni-skip verification core against
/// `jolt_sumcheck::prove_uniskip_clear`, asserting byte-identical transcript
/// states. Kept beside [`verify_clear`], whose contract it pins; the batched
/// engine twins live in `jolt-prover`.
#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod uniskip_twin_tests {
    use jolt_crypto::Bn254G1;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::{UnivariatePoly, UnivariatePolynomial};
    use jolt_r1cs::constraints::jolt::{
        SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
    };
    use jolt_sumcheck::{prove_uniskip_clear, CenteredIntegerDomain, ClearRound, SumcheckDomain};
    use jolt_transcript::{Blake2bTranscript, Transcript};

    use super::{verify_clear, UniskipParams};

    #[test]
    fn uniskip_prover_twin_matches_uniskip_verify_clear() {
        let degree = SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE;
        let domain_size = SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE;
        let poly = UnivariatePoly::new(
            (0..=degree as u64)
                .map(|k| Fr::from_u64(3 * k + 2))
                .collect(),
        );
        let coefficients = CenteredIntegerDomain::new(domain_size)
            .round_sum_coefficients(UnivariatePolynomial::degree(&poly))
            .unwrap();
        let input_claim = <UnivariatePoly<Fr> as ClearRound<Fr>>::coefficient_linear_combination(
            &poly,
            &coefficients,
        );

        let mut prover_transcript = Blake2bTranscript::new(b"uniskip-stage-twin");
        let proved = prove_uniskip_clear::<Fr, Bn254G1, _>(
            poly,
            input_claim,
            degree,
            domain_size,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::new(b"uniskip-stage-twin");
        let challenge = verify_clear(
            &proved.proof,
            &UniskipParams::spartan_outer(),
            input_claim,
            proved.output_claim,
            &mut verifier_transcript,
        )
        .unwrap();

        assert_eq!(challenge, proved.challenge);
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    }
}
