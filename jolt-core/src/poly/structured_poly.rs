use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use merlin::Transcript;

use crate::utils::{errors::ProofVerifyError, random::RandomTape};

/// Encapsulates the pattern of a collection of related polynomials (e.g. those used to
/// prove instruction lookups in Jolt) that can be "batched" for more efficient
/// commitments/openings.
pub trait BatchablePolynomials {
    /// The batched form of these polynomials.
    type BatchedPolynomials;
    /// The batched commitment to these polynomials.
    type Commitment;

    /// Organizes polynomials into a batch, to be subsequently committed. Typically
    /// uses `DensePolynomial::merge` to combine polynomials of the same size.
    fn batch(&self) -> Self::BatchedPolynomials;
    /// Commits to batched polynomials, typically using `DensePolynomial::combined_commit`.
    fn commit(batched_polys: &Self::BatchedPolynomials) -> Self::Commitment;
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
        polynomials: &Polynomials::BatchedPolynomials,
        commitment: &Polynomials::Commitment,
        opening_point: &Vec<F>,
        openings: Self::Openings,
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) -> Self;

    /// Often some of the openings do not require an opening proof provided by the prover, and 
    /// instead can be efficiently computed by the verifier by itself. This function populates 
    /// any such fields in `self`.
    fn compute_verifier_openings(&mut self, _opening_point: &Vec<F>) {}

    /// Verifies an opening proof, given the associated polynomial `commitment` and `opening_point`.
    fn verify_openings(
        &self,
        commitment: &Polynomials::Commitment,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError>;
}
