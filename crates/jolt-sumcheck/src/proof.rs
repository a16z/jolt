//! Proof structures for single and batched sumcheck protocols.

use crate::{
    claim::{EvaluationClaim, SumcheckClaim, SumcheckShape},
    committed::{CommittedSumcheckCheck, CommittedSumcheckProof},
    domain::{BooleanHypercube, SumcheckDomain},
    error::SumcheckError,
    round_proof::LabeledRoundPoly,
    verifier::SumcheckVerifier,
    SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_field::Field;
use jolt_poly::{CompressedPoly, UnivariatePoly};
use jolt_transcript::{AppendToTranscript, Transcript};
use serde::{Deserialize, Serialize};

/// A sumcheck proof consisting of one univariate round polynomial per variable.
///
/// In round $i$ the prover sends a univariate polynomial $s_i(X)$ of degree
/// at most $d$ (the claim's degree bound). The verifier checks that
/// $s_i(0) + s_i(1)$ equals the running sum, then sets the next challenge
/// $r_i$ and updates the running sum to $s_i(r_i)$.
///
/// The proof is complete when all $n$ round polynomials have been sent;
/// the verifier is left with a single evaluation claim at the point
/// $(r_1, \ldots, r_n)$.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ClearSumcheckProof<F: jolt_field::Field> {
    /// Round polynomials $s_1, \ldots, s_n$ in the order they were generated.
    pub round_polynomials: Vec<UnivariatePoly<F>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CompressedSumcheckProof<F: jolt_field::Field> {
    /// Boolean-hypercube round polynomials with the linear coefficient omitted.
    pub round_polynomials: Vec<CompressedPoly<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum ClearProof<F: jolt_field::Field> {
    Full(ClearSumcheckProof<F>),
    Compressed(CompressedSumcheckProof<F>),
}

impl<F: jolt_field::Field> Default for ClearProof<F> {
    fn default() -> Self {
        Self::Full(ClearSumcheckProof::default())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "C: Serialize", deserialize = "C: Deserialize<'de>"))]
pub enum SumcheckProof<F: jolt_field::Field, C> {
    Clear(ClearProof<F>),
    Committed(CommittedSumcheckProof<C>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SumcheckVerification<F: Field, C> {
    Clear(EvaluationClaim<F>),
    Committed(CommittedSumcheckCheck<F, C>),
}

impl<F: jolt_field::Field, C> SumcheckProof<F, C> {
    pub fn is_committed(&self) -> bool {
        matches!(self, Self::Committed(_))
    }

    pub fn is_clear(&self) -> bool {
        matches!(self, Self::Clear(_))
    }

    pub fn as_clear(&self) -> Option<&ClearProof<F>> {
        match self {
            Self::Clear(proof) => Some(proof),
            Self::Committed(_) => None,
        }
    }

    pub fn as_committed(&self) -> Option<&CommittedSumcheckProof<C>> {
        match self {
            Self::Clear(_) => None,
            Self::Committed(proof) => Some(proof),
        }
    }

    /// Verifies a full-round sumcheck proof over `domain`.
    ///
    /// Clear proofs check the round equations and return the scalar reduction.
    /// Committed proofs check public committed-proof structure and transcript
    /// effects; hidden claim relations are deferred to BlindFold.
    pub fn verify<T, D>(
        &self,
        claim: &SumcheckClaim<F>,
        domain: D,
        round_label: &'static [u8],
        transcript: &mut T,
    ) -> Result<SumcheckVerification<F, C>, SumcheckError<F>>
    where
        T: Transcript<Challenge = F>,
        D: SumcheckDomain<F>,
        C: Clone + AppendToTranscript,
    {
        match self {
            Self::Clear(ClearProof::Full(proof)) => {
                let rounds = proof
                    .round_polynomials
                    .iter()
                    .map(|poly| LabeledRoundPoly::new(poly, round_label))
                    .collect::<Vec<_>>();
                SumcheckVerifier::verify(claim, &rounds, domain, transcript)
                    .map(SumcheckVerification::Clear)
            }
            Self::Clear(ClearProof::Compressed(_)) => Err(SumcheckError::WrongProofEncoding {
                expected: "full clear or committed",
                got: "compressed clear",
            }),
            Self::Committed(proof) => proof
                .verify_committed(SumcheckShape::from(claim), transcript)
                .map(SumcheckVerification::Committed),
        }
    }

    /// Verifies a compressed Boolean-hypercube sumcheck proof.
    ///
    /// Clear proofs check the compressed round encoding and return the scalar
    /// reduction. Committed proofs check public committed-proof structure and
    /// transcript effects; hidden claim relations are deferred to BlindFold.
    pub fn verify_compressed_boolean<T>(
        &self,
        claim: &SumcheckClaim<F>,
        transcript: &mut T,
    ) -> Result<SumcheckVerification<F, C>, SumcheckError<F>>
    where
        T: Transcript<Challenge = F>,
        C: Clone + AppendToTranscript,
    {
        match self {
            Self::Clear(ClearProof::Compressed(proof)) => SumcheckVerifier::verify_compressed(
                claim,
                proof,
                BooleanHypercube,
                SUMCHECK_ROUND_TRANSCRIPT_LABEL,
                transcript,
            )
            .map(SumcheckVerification::Clear),
            Self::Clear(ClearProof::Full(_)) => Err(SumcheckError::WrongProofEncoding {
                expected: "compressed clear or committed",
                got: "full clear",
            }),
            Self::Committed(proof) => proof
                .verify_committed(SumcheckShape::from(claim), transcript)
                .map(SumcheckVerification::Committed),
        }
    }
}
