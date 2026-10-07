use jolt_crypto::{Bn254G1, Commitment, JoltGroup};
use jolt_field::{Fr, One, Zero};
use jolt_openings::{CommitmentScheme, OpeningsError};
use jolt_poly::MultilinearPoly;
use jolt_transcript::{Channel, ProverTranscript, Sponge, VerifierTranscript};

use crate::kzg::exchange_evaluations;
use crate::{HyperKZGError, HyperKZGProverSetup, HyperKZGSetupParams, HyperKZGVerifierSetup};

/// Clear binary HyperKZG with high-to-low multilinear coordinates.
///
/// The scheme binds the entire opening statement itself; callers may
/// additionally bind it to their surrounding protocol.
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

    /// Writes the opening: the binary fold commitments, the fold evaluations at
    /// `[r, -r, r^2]`, then the KZG witnesses.
    fn open_table<H: Sponge>(
        evaluations: &[Fr],
        point: &[Fr],
        evaluation: Fr,
        setup: &HyperKZGProverSetup,
        hint: Option<Bn254G1>,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), HyperKZGError> {
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
        let commitment = match hint {
            Some(commitment) => commitment,
            None => setup.commit_coefficients(evaluations)?,
        };
        setup
            .verifier
            .bind_statement(&commitment, point, evaluation, transcript);
        let mut com = polynomials
            .iter()
            .skip(1)
            .map(|polynomial| setup.commit_coefficients(polynomial))
            .collect::<Result<Vec<_>, _>>()?;
        let r = Self::exchange_fold_commitments(&mut com, transcript)?;
        setup.open_batch(&polynomials, [r, -r, r * r], transcript)
    }

    /// Exchanges the binary fold commitments and draws the nonzero fold point.
    fn exchange_fold_commitments<C: Channel>(
        com: &mut [Bn254G1],
        channel: &mut C,
    ) -> Result<Fr, HyperKZGError> {
        channel.exchange_all(com)?;
        let r: Fr = channel.challenge();
        if r.is_zero() {
            return Err(HyperKZGError::DegenerateChallenge);
        }
        Ok(r)
    }

    /// Reads an opening of `commitment` at `point` and checks the binary fold
    /// identities and the batched KZG equation. The arity is checked against
    /// the setup before anything is bound or read.
    pub fn verify_opening<H: Sponge>(
        commitment: &Bn254G1,
        point: &[Fr],
        evaluation: Fr,
        setup: &HyperKZGVerifierSetup,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), HyperKZGError> {
        let _ = setup.check_arity(point.len())?;
        setup.bind_statement(commitment, point, evaluation, transcript);
        let mut com = vec![Bn254G1::identity(); point.len() - 1];
        let r = Self::exchange_fold_commitments(&mut com, transcript)?;
        let mut evaluations: [Vec<Fr>; 3] = std::array::from_fn(|_| vec![Fr::zero(); point.len()]);
        let powers = exchange_evaluations(&mut evaluations, transcript)?;
        let [positive, negative, squared] = &evaluations;
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
        commitments.extend_from_slice(&com);
        setup.verify_batch(
            &commitments,
            [r, -r, r * r],
            &evaluations,
            &powers,
            transcript,
        )
    }
}

impl Commitment for HyperKZGScheme {
    type Output = Bn254G1;
}

impl CommitmentScheme for HyperKZGScheme {
    type Field = Fr;
    type ProverSetup = HyperKZGProverSetup;
    type VerifierSetup = HyperKZGVerifierSetup;
    /// The commitment returned by `commit`. `open` binds it into the statement
    /// without recomputing it, so a mismatched hint yields a rejected proof.
    type OpeningHint = Bn254G1;
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
            .map(|commitment| (commitment, commitment))
            .map_err(|error| OpeningsError::CommitFailed(error.to_string()))
    }

    fn send_commitment<H: Sponge>(commitment: &Bn254G1, transcript: &mut ProverTranscript<H>) {
        transcript.send(commitment);
    }

    fn receive_commitment<H: Sponge>(
        _setup: &Self::VerifierSetup,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<Bn254G1, OpeningsError> {
        Ok(transcript.receive()?)
    }

    fn absorb_commitment<C: Channel>(commitment: &Bn254G1, channel: &mut C) {
        channel.public(commitment);
    }

    fn open<P: MultilinearPoly<Fr> + ?Sized, H: Sponge>(
        poly: &P,
        point: &[Fr],
        evaluation: Fr,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), OpeningsError> {
        if poly.num_vars() != point.len() {
            return Err(OpeningsError::ProveFailed(
                HyperKZGError::PolynomialShape.to_string(),
            ));
        }
        let _ = setup
            .verifier
            .check_arity(point.len())
            .map_err(|error| OpeningsError::ProveFailed(error.to_string()))?;
        Self::open_table(&poly.to_dense(), point, evaluation, setup, hint, transcript)
            .map_err(|error| OpeningsError::ProveFailed(error.to_string()))
    }

    fn verify<H: Sponge>(
        commitment: &Self::Output,
        point: &[Fr],
        evaluation: Fr,
        setup: &Self::VerifierSetup,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), OpeningsError> {
        Self::verify_opening(commitment, point, evaluation, setup, transcript)
            .map_err(|_| OpeningsError::VerificationFailed)
    }
}
