use jolt_crypto::{Bn254G1, Commitment, JoltGroup};
use jolt_field::{Fr, One, Zero};
use jolt_openings::{CommitmentScheme, OpeningsError};
use jolt_poly::MultilinearPoly;
use jolt_transcript::Transcript;

use crate::{
    HyperKZGError, HyperKZGProof, HyperKZGProverSetup, HyperKZGSetupParams, HyperKZGVerifierSetup,
};

/// Clear binary HyperKZG with high-to-low multilinear coordinates.
///
/// The ordinary PCS API is the standalone external contract. The scheme binds
/// the entire opening statement itself; callers may additionally bind it to
/// their surrounding protocol. No hiding or homomorphic-batching API is exposed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HyperKZGScheme;

impl HyperKZGScheme {
    /// Imports authenticated ceremony material; no secret scalar is generated.
    ///
    /// This validates shape and nonidentity powers. Power consistency and
    /// ceremony provenance are the importer's responsibility; see
    /// [`HyperKZGSetupParams`]. Verification keys are application-authenticated.
    pub fn import_srs(params: HyperKZGSetupParams) -> Result<HyperKZGProverSetup, HyperKZGError> {
        let g1 = *params
            .g1_powers
            .first()
            .ok_or(HyperKZGError::InvalidSetup)?;
        let num_powers =
            u64::try_from(params.g1_powers.len()).map_err(|_| HyperKZGError::InvalidSetup)?;
        let verifier = HyperKZGVerifierSetup {
            num_powers,
            setup_id: params.setup_id,
            max_public_degree: params.max_public_degree,
            g1,
            g2: params.g2,
            beta_g2: params.beta_g2,
        };
        let _ = verifier.validate()?;
        if params.g1_powers.iter().any(JoltGroup::is_identity) {
            return Err(HyperKZGError::InvalidSetup);
        }
        Ok(HyperKZGProverSetup {
            g1_powers: params.g1_powers,
            verifier,
        })
    }

    fn open_table(
        evaluations: &[Fr],
        point: &[Fr],
        evaluation: Fr,
        setup: &HyperKZGProverSetup,
        transcript: &mut impl Transcript<Challenge = Fr>,
    ) -> Result<HyperKZGProof, HyperKZGError> {
        let len = setup.verifier.check_arity(point.len())?;
        if evaluations.len() != len {
            return Err(HyperKZGError::PolynomialShape);
        }
        let mut polynomials = Vec::with_capacity(point.len());
        let mut current = evaluations.to_vec();
        for coordinate in point.iter().rev() {
            let folded = current
                .iter()
                .step_by(2)
                .zip(current.iter().skip(1).step_by(2))
                .map(|(even, odd)| *even + *coordinate * (*odd - even))
                .collect();
            polynomials.push(current);
            current = folded;
        }
        if current.as_slice() != [evaluation] {
            return Err(HyperKZGError::WrongEvaluation);
        }
        let commitment = setup.commit_coefficients(evaluations)?;
        setup
            .verifier
            .append_statement(&commitment, point, evaluation, transcript);
        let com = polynomials
            .iter()
            .skip(1)
            .map(|polynomial| setup.commit_coefficients(polynomial))
            .collect::<Result<Vec<_>, _>>()?;
        for commitment in &com {
            transcript.append(commitment);
        }
        let r = transcript.challenge();
        if r.is_zero() {
            return Err(HyperKZGError::DegenerateChallenge);
        }
        let (w, v) = setup.open_batch(&polynomials, [r, -r, r * r], transcript)?;
        Ok(HyperKZGProof { com, v, w })
    }

    /// Verifies shape, binary fold identities, and the batched KZG equation.
    pub fn verify_opening(
        commitment: &Bn254G1,
        point: &[Fr],
        evaluation: Fr,
        proof: &HyperKZGProof,
        setup: &HyperKZGVerifierSetup,
        transcript: &mut impl Transcript<Challenge = Fr>,
    ) -> Result<(), HyperKZGError> {
        let _ = setup.check_arity(point.len())?;
        if proof.com.len() != point.len() - 1 || proof.v.iter().any(|row| row.len() != point.len())
        {
            return Err(HyperKZGError::ProofShape);
        }
        setup.append_statement(commitment, point, evaluation, transcript);
        for commitment in &proof.com {
            transcript.append(commitment);
        }
        let r = transcript.challenge();
        if r.is_zero() {
            return Err(HyperKZGError::DegenerateChallenge);
        }
        let [positive, negative, squared] = &proof.v;
        let next = squared
            .iter()
            .skip(1)
            .copied()
            .chain(std::iter::once(evaluation));
        for (((pos, neg), next), coordinate) in positive
            .iter()
            .zip(negative)
            .zip(next)
            .zip(point.iter().rev())
        {
            let lhs = (r + r) * next;
            let rhs = r * (Fr::one() - coordinate) * (*pos + neg) + *coordinate * (*pos - neg);
            if lhs != rhs {
                return Err(HyperKZGError::Folding);
            }
        }
        let mut commitments = Vec::with_capacity(point.len());
        commitments.push(*commitment);
        commitments.extend_from_slice(&proof.com);
        setup.verify_batch(&commitments, [r, -r, r * r], &proof.v, &proof.w, transcript)
    }
}

impl Commitment for HyperKZGScheme {
    type Output = Bn254G1;
}

impl CommitmentScheme for HyperKZGScheme {
    type Field = Fr;
    type Proof = HyperKZGProof;
    type ProverSetup = HyperKZGProverSetup;
    type VerifierSetup = HyperKZGVerifierSetup;
    type OpeningHint = ();
    type SetupParams = HyperKZGSetupParams;

    fn setup(
        params: Self::SetupParams,
    ) -> Result<(Self::ProverSetup, Self::VerifierSetup), OpeningsError> {
        let prover = Self::import_srs(params)
            .map_err(|error| OpeningsError::InvalidSetup(error.to_string()))?;
        let verifier = prover.verifier.clone();
        Ok((prover, verifier))
    }

    fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
        prover_setup.verifier.clone()
    }

    fn commit<P: MultilinearPoly<Fr> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError> {
        let len = setup
            .verifier
            .check_arity(poly.num_vars())
            .map_err(|error| OpeningsError::CommitFailed(error.to_string()))?;
        let evaluations = poly.to_dense();
        if evaluations.len() != len {
            return Err(OpeningsError::CommitFailed(
                HyperKZGError::PolynomialShape.to_string(),
            ));
        }
        setup
            .commit_coefficients(&evaluations)
            .map(|commitment| (commitment, ()))
            .map_err(|error| OpeningsError::CommitFailed(error.to_string()))
    }

    fn open<P: MultilinearPoly<Fr> + ?Sized>(
        poly: &P,
        point: &[Fr],
        evaluation: Fr,
        setup: &Self::ProverSetup,
        _hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Fr>,
    ) -> Result<Self::Proof, OpeningsError> {
        if poly.num_vars() != point.len() {
            return Err(OpeningsError::ProveFailed(
                HyperKZGError::PolynomialShape.to_string(),
            ));
        }
        let _ = setup
            .verifier
            .check_arity(point.len())
            .map_err(|error| OpeningsError::ProveFailed(error.to_string()))?;
        Self::open_table(&poly.to_dense(), point, evaluation, setup, transcript)
            .map_err(|error| OpeningsError::ProveFailed(error.to_string()))
    }

    fn verify(
        commitment: &Self::Output,
        point: &[Fr],
        evaluation: Fr,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Fr>,
    ) -> Result<(), OpeningsError> {
        Self::verify_opening(commitment, point, evaluation, proof, setup, transcript)
            .map_err(|_| OpeningsError::VerificationFailed)
    }
}
