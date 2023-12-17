use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use merlin::Transcript;

use crate::utils::{errors::ProofVerifyError, random::RandomTape};

/// Encapsulates the pattern of a collection of related polynomials (e.g. those used to
/// prove instruction lookups in Jolt) that can be "batched" for more efficient
/// commitments/openings.
pub trait BatchablePolynomials {
    /// The batched commitment to these polynomials.
    type Commitment;

    /// Commits to batched polynomials, typically using `DensePolynomial::combined_commit`.
    fn commit(&self) -> Self::Commitment;
}

/// Encapsulates the pattern of opening a batched polynomial commitment at a single point.
/// Note that there may be a one-to-many mapping from `BatchablePolynomials` to `StructuredOpeningProof`:
/// different subset of the same polynomials may be opened at different points, resulting in
/// different opening proofs.
pub trait StructuredOpeningProof<F, G, Polynomials>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    Polynomials: BatchablePolynomials + ?Sized,
{
    type Openings;

    /// Evaluates each fo the given `polynomials` at the given `opening_point`.
    fn open(polynomials: &Polynomials, opening_point: &Vec<F>) -> Self::Openings;

    /// Proves that the `polynomials`, evaluated at `opening_point`, output the values given
    /// by `openings`. The polynomials should already be committed by the prover (`commitment`).
    fn prove_openings(
        polynomials: &Polynomials,
        commitment: &Polynomials::Commitment,
        opening_point: &Vec<F>,
        openings: Self::Openings,
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) -> Self;

    /// Verifies an opening proof, given the associated polynomial `commitment` and `opening_point`.
    fn verify_openings(
        &self,
        commitment: &Polynomials::Commitment,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError>;
}
