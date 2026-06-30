//! Proof structures for single and batched sumcheck protocols.

use crate::{
    claim::{EvaluationClaim, SumcheckClaim, SumcheckStatement},
    committed::{CommittedSumcheckConsistency, CommittedSumcheckProof},
    domain::{BooleanHypercube, SumcheckDomain},
    error::SumcheckError,
    round_proof::RoundPoly,
    verifier::SumcheckVerifier,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_poly::{CompressedPoly, UnivariatePoly};
use jolt_transcript::{FsNargRead, FsTranscript};
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

    /// Verifies a full-round clear sumcheck proof over `domain`.
    pub fn verify<T, D>(
        &self,
        claim: &SumcheckClaim<F>,
        domain: D,
        transcript: &mut T,
    ) -> Result<EvaluationClaim<F>, SumcheckError<F>>
    where
        T: FsTranscript<F>,
        D: SumcheckDomain<F>,
        C: Clone + CanonicalSerialize,
    {
        match self {
            Self::Clear(ClearProof::Full(proof)) => {
                let rounds = proof
                    .round_polynomials
                    .iter()
                    .map(|poly| RoundPoly::new(poly))
                    .collect::<Vec<_>>();
                SumcheckVerifier::verify(claim, &rounds, domain, transcript)
            }
            Self::Clear(ClearProof::Compressed(_)) => Err(SumcheckError::WrongProofEncoding {
                expected: "full clear",
                got: "compressed clear",
            }),
            Self::Committed(_) => Err(SumcheckError::WrongProofEncoding {
                expected: "full clear",
                got: "committed",
            }),
        }
    }

    /// Verifies a full-round clear sumcheck proof by consuming round
    /// polynomial frames from the NARG.
    pub fn verify_from_narg<T, D>(
        &self,
        claim: &SumcheckClaim<F>,
        domain: D,
        transcript: &mut T,
    ) -> Result<EvaluationClaim<F>, SumcheckError<F>>
    where
        T: FsNargRead<F>,
        D: SumcheckDomain<F>,
        C: Clone + CanonicalSerialize,
    {
        match self {
            Self::Clear(ClearProof::Full(_)) => {
                SumcheckVerifier::verify_from_narg(claim, domain, transcript)
            }
            Self::Clear(ClearProof::Compressed(_)) => Err(SumcheckError::WrongProofEncoding {
                expected: "full clear",
                got: "compressed clear",
            }),
            Self::Committed(_) => Err(SumcheckError::WrongProofEncoding {
                expected: "full clear",
                got: "committed",
            }),
        }
    }

    /// Verifies a compressed clear Boolean-hypercube sumcheck proof.
    pub fn verify_compressed_boolean<T>(
        &self,
        claim: &SumcheckClaim<F>,
        transcript: &mut T,
    ) -> Result<EvaluationClaim<F>, SumcheckError<F>>
    where
        T: FsTranscript<F>,
        C: Clone + CanonicalSerialize,
    {
        match self {
            Self::Clear(ClearProof::Compressed(proof)) => {
                SumcheckVerifier::verify_compressed(claim, proof, BooleanHypercube, transcript)
            }
            Self::Clear(ClearProof::Full(_)) => Err(SumcheckError::WrongProofEncoding {
                expected: "compressed clear",
                got: "full clear",
            }),
            Self::Committed(_) => Err(SumcheckError::WrongProofEncoding {
                expected: "compressed clear",
                got: "committed",
            }),
        }
    }

    /// Verifies a compressed clear Boolean-hypercube proof from NARG frames.
    pub fn verify_compressed_boolean_from_narg<T>(
        &self,
        claim: &SumcheckClaim<F>,
        transcript: &mut T,
    ) -> Result<EvaluationClaim<F>, SumcheckError<F>>
    where
        T: FsNargRead<F>,
        C: Clone + CanonicalSerialize,
    {
        match self {
            Self::Clear(ClearProof::Compressed(_)) => {
                SumcheckVerifier::verify_compressed_from_narg(claim, BooleanHypercube, transcript)
            }
            Self::Clear(ClearProof::Full(_)) => Err(SumcheckError::WrongProofEncoding {
                expected: "compressed clear",
                got: "full clear",
            }),
            Self::Committed(_) => Err(SumcheckError::WrongProofEncoding {
                expected: "compressed clear",
                got: "committed",
            }),
        }
    }

    /// Checks public consistency for a committed sumcheck proof.
    ///
    /// This path intentionally takes only a [`SumcheckStatement`]. Committed
    /// proofs do not reveal the scalar claim, so claim relations are deferred
    /// to the BlindFold verifier rather than represented with placeholder
    /// values.
    pub fn verify_committed_consistency<T>(
        &self,
        statement: SumcheckStatement,
        transcript: &mut T,
    ) -> Result<CommittedSumcheckConsistency<F, C>, SumcheckError<F>>
    where
        T: FsTranscript<F>,
        C: Clone + CanonicalSerialize,
    {
        match self {
            Self::Committed(proof) => proof.verify_committed_consistency(statement, transcript),
            Self::Clear(ClearProof::Full(_)) => Err(SumcheckError::WrongProofEncoding {
                expected: "committed",
                got: "full clear",
            }),
            Self::Clear(ClearProof::Compressed(_)) => Err(SumcheckError::WrongProofEncoding {
                expected: "committed",
                got: "compressed clear",
            }),
        }
    }

    /// Checks public consistency for a committed proof by consuming its
    /// commitments, degrees, and output-claim commitments from the NARG.
    pub fn verify_committed_consistency_from_narg<T>(
        &self,
        statement: SumcheckStatement,
        transcript: &mut T,
    ) -> Result<CommittedSumcheckConsistency<F, C>, SumcheckError<F>>
    where
        T: FsNargRead<F>,
        C: Clone + CanonicalSerialize + CanonicalDeserialize,
    {
        match self {
            Self::Committed(_) => {
                CommittedSumcheckProof::<C>::verify_committed_consistency_from_narg(
                    statement, transcript,
                )
            }
            Self::Clear(ClearProof::Full(_)) => Err(SumcheckError::WrongProofEncoding {
                expected: "committed",
                got: "full clear",
            }),
            Self::Clear(ClearProof::Compressed(_)) => Err(SumcheckError::WrongProofEncoding {
                expected: "committed",
                got: "compressed clear",
            }),
        }
    }
}
