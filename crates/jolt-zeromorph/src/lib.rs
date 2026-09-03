//! Zeromorph multilinear polynomial commitments over a KZG SRS.
//!
//! The multilinear evaluation table is used verbatim as coefficients of the
//! committed univariate polynomial, matching `jolt-hyperkzg`. A single-point
//! proof contains `ell + 2` G1 elements and no field elements: `ell` quotient
//! commitments, one raised-degree quotient commitment, and one KZG witness.
//! The middle commitment is required by the degree checks in Zeromorph §6.
//!
//! Multiple polynomials at one point are reduced by an external random linear
//! combination. For `t` points, [`ZeromorphScheme::open_multi`] batches the
//! final KZG check, producing `t * (ell + 1) + 1` G1 elements.

#![forbid(unsafe_code)]
#![deny(
    clippy::indexing_slicing,
    clippy::get_unwrap,
    clippy::string_slice,
    clippy::fallible_impl_from,
    clippy::mem_forget,
    clippy::exit,
    clippy::panic_in_result_fn,
    clippy::let_underscore_must_use,
    clippy::host_endian_bytes,
    clippy::wildcard_enum_match_arm
)]

use std::marker::PhantomData;

use jolt_crypto::{Commitment, JoltGroup, PairingGroup};
use jolt_field::{Field, JoltField};
use jolt_hyperkzg::{HyperKZGCommitment, HyperKZGScheme};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, OpeningsError};
use jolt_poly::MultilinearPoly;
use jolt_transcript::{AppendToTranscript, Label, Transcript, U64Word};
use num_traits::{One, Zero};
use rand_core::OsRng;
use rayon::prelude::*;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// Commitment shared byte-for-byte with HyperKZG.
pub type ZeromorphCommitment<P> = HyperKZGCommitment<P>;

/// Prover setup backed by HyperKZG's monomial KZG SRS.
#[derive(Clone, Debug)]
pub struct ZeromorphProverSetup<P: PairingGroup> {
    g1_powers: Vec<P::G1Affine>,
    num_vars: usize,
    verifier_setup: ZeromorphVerifierSetup<P>,
}

impl<P: PairingGroup> ZeromorphProverSetup<P> {
    /// Exact G1 SRS `[1, beta, ..., beta^(N-1)]`.
    pub fn g1_powers(&self) -> &[P::G1Affine] {
        &self.g1_powers
    }

    /// Supported multilinear arity.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }
}

/// Arity-specific verifier setup.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "P::G1: Serialize, P::G2: Serialize",
    deserialize = "P::G1: for<'a> Deserialize<'a>, P::G2: for<'a> Deserialize<'a>"
))]
pub struct ZeromorphVerifierSetup<P: PairingGroup> {
    g1: P::G1,
    g2: P::G2,
    beta_g2: P::G2,
    num_vars: usize,
}

/// A paper-faithful non-hiding Zeromorph proof.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "P::G1: Serialize",
    deserialize = "P::G1: for<'a> Deserialize<'a>"
))]
pub struct ZeromorphProof<P: PairingGroup> {
    /// Point-major commitments to `U_k(q_k)`, with `k` increasing per point.
    pub quotient_commitments: Vec<P::G1>,
    /// One raised-degree quotient commitment per point.
    pub lifted_degree_quotients: Vec<P::G1>,
    /// KZG witness for the combined degree and evaluation identities.
    pub opening_proof: P::G1,
}

impl<P: PairingGroup> ZeromorphProof<P> {
    /// Compressed group payload size, excluding container framing.
    pub fn compressed_payload_bytes(&self, g1_bytes: usize) -> usize {
        (self.quotient_commitments.len() + self.lifted_degree_quotients.len() + 1) * g1_bytes
    }
}

/// Multi-point proofs use the same flat point-major wire type.
pub type ZeromorphMultiPointProof<P> = ZeromorphProof<P>;

/// Zeromorph failures.
#[derive(Debug, thiserror::Error)]
pub enum ZeromorphError {
    /// Setup or polynomial arity does not match the claim.
    #[error("expected {expected} variables, got {got}")]
    WrongArity { expected: usize, got: usize },
    /// Dense evaluation table has the wrong length.
    #[error("expected {expected} evaluations, got {got}")]
    WrongEvaluationCount { expected: usize, got: usize },
    /// Multi-point claim shape is inconsistent.
    #[error("expected {expected} point claims, got {got}")]
    WrongPointCount { expected: usize, got: usize },
    /// Proof has an invalid number of quotient commitments.
    #[error("expected {expected} quotient commitments, got {got}")]
    WrongQuotientCount { expected: usize, got: usize },
    /// At least one evaluation claim is required.
    #[error("multi-point batch is empty")]
    EmptyBatch,
    /// The SRS does not cover the polynomial.
    #[error("SRS has {have} G1 powers, need {need}")]
    SrsTooSmall { have: usize, need: usize },
    /// A Fiat-Shamir challenge that must be invertible was zero.
    #[error("degenerate Fiat-Shamir challenge: {0} = 0")]
    DegenerateChallenge(&'static str),
    /// The trait opening path did not receive the commitment returned by `commit`.
    #[error("opening requires the commitment hint returned by commit")]
    MissingOpeningHint,
    /// Final KZG equation failed.
    #[error("pairing check failed")]
    PairingCheckFailed,
    /// Number of variables does not fit a platform `usize` coefficient count.
    #[error("number of variables is too large")]
    ArityTooLarge,
}

/// KZG-backed Zeromorph PCS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZeromorphScheme<P: PairingGroup> {
    _marker: PhantomData<P>,
}

impl<P: PairingGroup> ZeromorphScheme<P>
where
    P::ScalarField: AppendToTranscript,
{
    /// Deterministic setup for tests and trusted-setup tooling.
    pub fn setup_from_secret(
        beta: P::ScalarField,
        num_vars: usize,
        g1: P::G1,
        g2: P::G2,
    ) -> Result<(ZeromorphProverSetup<P>, ZeromorphVerifierSetup<P>), ZeromorphError> {
        if num_vars == 0 {
            return Err(ZeromorphError::WrongArity {
                expected: 1,
                got: 0,
            });
        }
        let coefficient_count = coefficient_count(num_vars)?;
        let mut g1_powers = HyperKZGScheme::<P>::setup_from_secret(
            beta,
            coefficient_count,
            g1,
            g2,
        )
        .into_g1_powers();
        g1_powers.truncate(coefficient_count);
        let verifier = ZeromorphVerifierSetup {
            g1,
            g2,
            beta_g2: g2.scalar_mul(&beta),
            num_vars,
        };
        let prover = ZeromorphProverSetup {
            g1_powers,
            num_vars,
            verifier_setup: verifier.clone(),
        };
        Ok((prover, verifier))
    }

    /// Commits to the evaluation table as univariate coefficients in table order.
    #[tracing::instrument(skip_all, name = "Zeromorph::commit")]
    pub fn commit(
        setup: &ZeromorphProverSetup<P>,
        evaluations: &[P::ScalarField],
    ) -> Result<ZeromorphCommitment<P>, ZeromorphError> {
        validate_evaluations(setup, evaluations)?;
        Ok(HyperKZGCommitment::new(commit_coefficients(
            evaluations,
            setup,
        )?))
    }

    /// Opens one multilinear evaluation.
    #[tracing::instrument(skip_all, name = "Zeromorph::open")]
    pub fn open<T: Transcript<Challenge = P::ScalarField>>(
        setup: &ZeromorphProverSetup<P>,
        evaluations: &[P::ScalarField],
        commitment: &ZeromorphCommitment<P>,
        point: &[P::ScalarField],
        claimed_eval: P::ScalarField,
        transcript: &mut T,
    ) -> Result<ZeromorphProof<P>, ZeromorphError> {
        Self::open_multi(
            setup,
            evaluations,
            commitment,
            &[point.to_vec()],
            &[claimed_eval],
            transcript,
        )
    }

    /// Opens one committed polynomial at several multilinear points.
    #[tracing::instrument(skip_all, name = "Zeromorph::open_multi")]
    pub fn open_multi<T: Transcript<Challenge = P::ScalarField>>(
        setup: &ZeromorphProverSetup<P>,
        evaluations: &[P::ScalarField],
        commitment: &ZeromorphCommitment<P>,
        points: &[Vec<P::ScalarField>],
        claimed_evals: &[P::ScalarField],
        transcript: &mut T,
    ) -> Result<ZeromorphMultiPointProof<P>, ZeromorphError> {
        validate_evaluations(setup, evaluations)?;
        validate_claims(setup.num_vars, points, claimed_evals)?;
        absorb_statement::<P, _>(
            transcript,
            setup.num_vars,
            commitment,
            points,
            claimed_evals,
        );

        let quotient_sets = points
            .par_iter()
            .map(|point| multilinear_quotients(evaluations, point))
            .collect::<Vec<_>>();
        let quotient_commitment_rows = quotient_sets
            .par_iter()
            .map(|quotients| {
                quotients
                    .par_iter()
                    .map(|q| commit_coefficients(q, setup))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let quotient_commitments = quotient_commitment_rows.into_iter().flatten().collect::<Vec<_>>();

        absorb_quotient_commitments::<P, _>(transcript, &quotient_commitments);
        let y = nonzero_challenge(transcript, b"zeromorph-y", "y")?;
        let lifted_quotients = quotient_sets
            .par_iter()
            .map(|quotients| lifted_degree_quotient(quotients, y, evaluations.len()))
            .collect::<Vec<_>>();
        let lifted_degree_quotients = lifted_quotients
            .par_iter()
            .map(|q| commit_shifted_coefficients(q, evaluations.len() / 2, setup))
            .collect::<Result<Vec<_>, _>>()?;
        absorb_lifted_commitments::<P, _>(transcript, &lifted_degree_quotients);

        let x = nonzero_challenge(transcript, b"zeromorph-x", "x")?;
        let z = nonzero_challenge(transcript, b"zeromorph-z", "z")?;
        let rho = nonzero_challenge(transcript, b"zeromorph-rho", "rho")?;
        let challenges = ProtocolChallenges { y, x, z, rho };
        let combined = combined_identity(
            evaluations,
            points,
            claimed_evals,
            &quotient_sets,
            &lifted_quotients,
            challenges,
        );
        let witness = divide_by_linear(&combined, x);
        let opening_proof = commit_shifted_coefficients(&witness, 1, setup)?;
        transcript.append(&Label(b"zeromorph-opening"));
        transcript.append(&opening_proof);

        Ok(ZeromorphMultiPointProof {
            quotient_commitments,
            lifted_degree_quotients,
            opening_proof,
        })
    }

    /// Verifies one multilinear evaluation.
    #[tracing::instrument(skip_all, name = "Zeromorph::verify")]
    pub fn verify<T: Transcript<Challenge = P::ScalarField>>(
        setup: &ZeromorphVerifierSetup<P>,
        commitment: &ZeromorphCommitment<P>,
        point: &[P::ScalarField],
        claimed_eval: P::ScalarField,
        proof: &ZeromorphProof<P>,
        transcript: &mut T,
    ) -> Result<(), ZeromorphError> {
        Self::verify_multi(
            setup,
            commitment,
            &[point.to_vec()],
            &[claimed_eval],
            proof,
            transcript,
        )
    }

    /// Verifies a multi-point proof with one two-pair KZG equation.
    #[tracing::instrument(skip_all, name = "Zeromorph::verify_multi")]
    pub fn verify_multi<T: Transcript<Challenge = P::ScalarField>>(
        setup: &ZeromorphVerifierSetup<P>,
        commitment: &ZeromorphCommitment<P>,
        points: &[Vec<P::ScalarField>],
        claimed_evals: &[P::ScalarField],
        proof: &ZeromorphMultiPointProof<P>,
        transcript: &mut T,
    ) -> Result<(), ZeromorphError> {
        validate_claims(setup.num_vars, points, claimed_evals)?;
        validate_proof_shape(setup.num_vars, points.len(), proof)?;
        absorb_statement::<P, _>(
            transcript,
            setup.num_vars,
            commitment,
            points,
            claimed_evals,
        );
        absorb_quotient_commitments::<P, _>(transcript, &proof.quotient_commitments);
        let y = nonzero_challenge(transcript, b"zeromorph-y", "y")?;
        absorb_lifted_commitments::<P, _>(transcript, &proof.lifted_degree_quotients);
        let x = nonzero_challenge(transcript, b"zeromorph-x", "x")?;
        let z = nonzero_challenge(transcript, b"zeromorph-z", "z")?;
        let rho = nonzero_challenge(transcript, b"zeromorph-rho", "rho")?;
        let challenges = ProtocolChallenges { y, x, z, rho };
        transcript.append(&Label(b"zeromorph-opening"));
        transcript.append(&proof.opening_proof);

        let identity_commitment = combined_identity_commitment(
            setup,
            commitment,
            points,
            claimed_evals,
            proof,
            challenges,
        );
        let scaled_opening = proof.opening_proof.scalar_mul(&x);
        let result = P::multi_pairing(
            &[identity_commitment - proof.opening_proof, scaled_opening],
            &[setup.beta_g2, setup.g2],
        );
        if result.is_identity() {
            Ok(())
        } else {
            Err(ZeromorphError::PairingCheckFailed)
        }
    }

    fn verifier_setup(setup: &ZeromorphProverSetup<P>) -> ZeromorphVerifierSetup<P> {
        setup.verifier_setup.clone()
    }
}

fn coefficient_count(num_vars: usize) -> Result<usize, ZeromorphError> {
    1usize
        .checked_shl(u32::try_from(num_vars).map_err(|_| ZeromorphError::ArityTooLarge)?)
        .ok_or(ZeromorphError::ArityTooLarge)
}

fn validate_evaluations<P: PairingGroup>(
    setup: &ZeromorphProverSetup<P>,
    evaluations: &[P::ScalarField],
) -> Result<(), ZeromorphError> {
    let expected = coefficient_count(setup.num_vars)?;
    if evaluations.len() == expected {
        Ok(())
    } else {
        Err(ZeromorphError::WrongEvaluationCount {
            expected,
            got: evaluations.len(),
        })
    }
}

fn validate_claims<F: JoltField>(
    num_vars: usize,
    points: &[Vec<F>],
    claimed_evals: &[F],
) -> Result<(), ZeromorphError> {
    if points.is_empty() {
        return Err(ZeromorphError::EmptyBatch);
    }
    if points.len() != claimed_evals.len() {
        return Err(ZeromorphError::WrongPointCount {
            expected: points.len(),
            got: claimed_evals.len(),
        });
    }
    if let Some(point) = points.iter().find(|point| point.len() != num_vars) {
        return Err(ZeromorphError::WrongArity {
            expected: num_vars,
            got: point.len(),
        });
    }
    Ok(())
}

fn validate_proof_shape<P: PairingGroup>(
    num_vars: usize,
    point_count: usize,
    proof: &ZeromorphMultiPointProof<P>,
) -> Result<(), ZeromorphError> {
    let expected_quotients = point_count * num_vars;
    if proof.quotient_commitments.len() != expected_quotients {
        return Err(ZeromorphError::WrongQuotientCount {
            expected: expected_quotients,
            got: proof.quotient_commitments.len(),
        });
    }
    if proof.lifted_degree_quotients.len() != point_count {
        return Err(ZeromorphError::WrongPointCount {
            expected: point_count,
            got: proof.lifted_degree_quotients.len(),
        });
    }
    Ok(())
}

fn commit_coefficients<P: PairingGroup>(
    coefficients: &[P::ScalarField],
    setup: &ZeromorphProverSetup<P>,
) -> Result<P::G1, ZeromorphError> {
    let bases = setup
        .g1_powers
        .get(..coefficients.len())
        .ok_or(ZeromorphError::SrsTooSmall {
            have: setup.g1_powers.len(),
            need: coefficients.len(),
        })?;
    Ok(P::g1_affine_msm(bases, coefficients))
}

fn commit_shifted_coefficients<P: PairingGroup>(
    coefficients: &[P::ScalarField],
    shift: usize,
    setup: &ZeromorphProverSetup<P>,
) -> Result<P::G1, ZeromorphError> {
    let end = shift + coefficients.len();
    let bases = setup
        .g1_powers
        .get(shift..end)
        .ok_or(ZeromorphError::SrsTooSmall {
            have: setup.g1_powers.len(),
            need: end,
        })?;
    Ok(P::g1_affine_msm(bases, coefficients))
}

fn absorb_statement<P, T>(
    transcript: &mut T,
    num_vars: usize,
    commitment: &ZeromorphCommitment<P>,
    points: &[Vec<P::ScalarField>],
    claimed_evals: &[P::ScalarField],
)
where
    P: PairingGroup,
    T: Transcript<Challenge = P::ScalarField>,
    P::ScalarField: AppendToTranscript,
{
    transcript.append(&Label(b"zeromorph-statement"));
    transcript.append(commitment);
    transcript.append(&U64Word(num_vars as u64));
    transcript.append(&U64Word(points.len() as u64));
    for (point, claimed_eval) in points.iter().zip(claimed_evals) {
        for coordinate in point {
            transcript.append(coordinate);
        }
        transcript.append(claimed_eval);
    }
}

fn absorb_quotient_commitments<P, T>(transcript: &mut T, commitments: &[P::G1])
where
    P: PairingGroup,
    T: Transcript<Challenge = P::ScalarField>,
{
    transcript.append(&Label(b"zeromorph-quotients"));
    for commitment in commitments {
        transcript.append(commitment);
    }
}

fn absorb_lifted_commitments<P, T>(transcript: &mut T, commitments: &[P::G1])
where
    P: PairingGroup,
    T: Transcript<Challenge = P::ScalarField>,
{
    transcript.append(&Label(b"zeromorph-lifted"));
    for commitment in commitments {
        transcript.append(commitment);
    }
}

fn nonzero_challenge<F: JoltField>(
    transcript: &mut impl Transcript<Challenge = F>,
    label: &'static [u8],
    name: &'static str,
) -> Result<F, ZeromorphError> {
    transcript.append(&Label(label));
    let challenge = transcript.challenge();
    if challenge.is_zero() {
        Err(ZeromorphError::DegenerateChallenge(name))
    } else {
        Ok(challenge)
    }
}

/// Quotient order follows the paper. Jolt's first point coordinate is the
/// table's high bit, so eliminating coordinates front-to-back produces q_k
/// back-to-front.
fn multilinear_quotients<F: JoltField>(evaluations: &[F], point: &[F]) -> Vec<Vec<F>> {
    let mut current = evaluations.to_vec();
    let mut quotients = Vec::with_capacity(point.len());
    for &coordinate in point {
        let half = current.len() / 2;
        let (low, high) = current.split_at_mut(half);
        let quotient = low
            .par_iter_mut()
            .zip(high.par_iter())
            .map(|(low, &high)| {
                let difference = high - *low;
                *low += coordinate * difference;
                difference
            })
            .collect();
        current.truncate(half);
        quotients.push(quotient);
    }
    quotients.reverse();
    quotients
}

fn lifted_degree_quotient<F: JoltField>(quotients: &[Vec<F>], y: F, size: usize) -> Vec<F> {
    let mut lifted = vec![F::zero(); size / 2];
    let mut y_power = F::one();
    for quotient in quotients {
        let shift = size / 2 - quotient.len();
        let (_, tail) = lifted.split_at_mut(shift);
        tail
            .par_iter_mut()
            .zip(quotient.par_iter())
            .for_each(|(output, &coefficient)| *output += y_power * coefficient);
        y_power *= y;
    }
    lifted
}

#[derive(Clone, Copy)]
struct ProtocolChallenges<F> {
    y: F,
    x: F,
    z: F,
    rho: F,
}

#[expect(
    clippy::indexing_slicing,
    reason = "power and suffix tables have n + 1 entries by construction"
)]
fn identity_scalars<F: JoltField>(point: &[F], x: F) -> (F, Vec<F>, Vec<F>) {
    let n = point.len();
    let mut powers = Vec::with_capacity(n + 1);
    let mut power = x;
    powers.push(power);
    for _ in 0..n {
        power *= power;
        powers.push(power);
    }
    let mut suffix = vec![F::one(); n + 1];
    for k in (0..n).rev() {
        suffix[k] = suffix[k + 1] * (F::one() + powers[k]);
    }
    let inverse_x = x.inverse().unwrap_or(F::zero());
    let mut inverse_power = inverse_x;
    let mut shifts = Vec::with_capacity(n);
    let mut factors = Vec::with_capacity(n);
    for (k, &u_k) in point.iter().rev().enumerate() {
        shifts.push(powers[n] * inverse_power);
        factors.push(powers[k] * suffix[k + 1] - u_k * suffix[k]);
        inverse_power *= inverse_power;
    }
    (suffix[0], shifts, factors)
}

#[expect(
    clippy::indexing_slicing,
    reason = "validated point, evaluation, quotient, and lifted rows have matching protocol dimensions"
)]
fn combined_identity<F: JoltField>(
    evaluations: &[F],
    points: &[Vec<F>],
    claimed_evals: &[F],
    quotient_sets: &[Vec<Vec<F>>],
    lifted_quotients: &[Vec<F>],
    challenges: ProtocolChallenges<F>,
) -> Vec<F> {
    let ProtocolChallenges { y, x, z, rho } = challenges;
    let mut combined = vec![F::zero(); evaluations.len()];
    let mut rho_power = F::one();
    for point_index in 0..points.len() {
        let (phi, shifts, factors) = identity_scalars(&points[point_index], x);
        let (_, lifted_output) = combined.split_at_mut(evaluations.len() / 2);
        lifted_output
            .par_iter_mut()
            .zip(lifted_quotients[point_index].par_iter())
            .for_each(|(output, &coefficient)| *output += rho_power * coefficient);
        combined
            .par_iter_mut()
            .zip(evaluations.par_iter())
            .for_each(|(output, &coefficient)| *output += rho_power * z * coefficient);
        combined[0] -= rho_power * z * claimed_evals[point_index] * phi;

        let mut y_power = F::one();
        for quotient_index in 0..quotient_sets[point_index].len() {
            let scalar =
                -rho_power * (y_power * shifts[quotient_index] + z * factors[quotient_index]);
            let (prefix, _) =
                combined.split_at_mut(quotient_sets[point_index][quotient_index].len());
            prefix
                .par_iter_mut()
                .zip(quotient_sets[point_index][quotient_index].par_iter())
                .for_each(|(output, &coefficient)| *output += scalar * coefficient);
            y_power *= y;
        }
        rho_power *= rho;
    }
    combined
}

#[expect(
    clippy::indexing_slicing,
    reason = "validated proof rows, points, evaluations, and scalar rows have matching dimensions"
)]
fn combined_identity_commitment<P: PairingGroup>(
    setup: &ZeromorphVerifierSetup<P>,
    commitment: &ZeromorphCommitment<P>,
    points: &[Vec<P::ScalarField>],
    claimed_evals: &[P::ScalarField],
    proof: &ZeromorphMultiPointProof<P>,
    challenges: ProtocolChallenges<P::ScalarField>,
) -> P::G1 {
    let ProtocolChallenges { y, x, z, rho } = challenges;
    let mut bases = Vec::with_capacity(points.len() * (points[0].len() + 1) + 2);
    let mut scalars = Vec::with_capacity(bases.capacity());
    let mut rho_power = P::ScalarField::one();
    let mut commitment_scalar = P::ScalarField::zero();
    let mut generator_scalar = P::ScalarField::zero();
    for point_index in 0..points.len() {
        let (phi, shifts, factors) = identity_scalars(&points[point_index], x);
        bases.push(proof.lifted_degree_quotients[point_index]);
        scalars.push(rho_power);
        commitment_scalar += rho_power * z;
        generator_scalar -= rho_power * z * claimed_evals[point_index] * phi;
        let mut y_power = P::ScalarField::one();
        for quotient_index in 0..points[point_index].len() {
            let flat_index = point_index * points[point_index].len() + quotient_index;
            bases.push(proof.quotient_commitments[flat_index]);
            scalars.push(
                -rho_power * (y_power * shifts[quotient_index] + z * factors[quotient_index]),
            );
            y_power *= y;
        }
        rho_power *= rho;
    }
    bases.push(commitment.point());
    scalars.push(commitment_scalar);
    bases.push(setup.g1);
    scalars.push(generator_scalar);
    P::g1_msm(&bases, &scalars)
}

#[expect(
    clippy::indexing_slicing,
    reason = "non-empty size-2^ell identity polynomial fixes quotient indices"
)]
fn divide_by_linear<F: JoltField>(polynomial: &[F], root: F) -> Vec<F> {
    let mut quotient = vec![F::zero(); polynomial.len() - 1];
    let last = polynomial.len() - 1;
    let mut carry = polynomial[last];
    quotient[last - 1] = carry;
    for index in (1..last).rev() {
        carry = polynomial[index] + root * carry;
        quotient[index - 1] = carry;
    }
    quotient
}

impl<P: PairingGroup> Commitment for ZeromorphScheme<P> {
    type Output = ZeromorphCommitment<P>;
}

impl<P: PairingGroup> CommitmentScheme for ZeromorphScheme<P>
where
    P::ScalarField: AppendToTranscript + Serialize + DeserializeOwned,
    P::G1: Serialize + DeserializeOwned,
    P::G2: Serialize + DeserializeOwned,
{
    type Field = P::ScalarField;
    type Proof = ZeromorphProof<P>;
    type ProverSetup = ZeromorphProverSetup<P>;
    type VerifierSetup = ZeromorphVerifierSetup<P>;
    type OpeningHint = ZeromorphCommitment<P>;
    type SetupParams = (usize, P::G1, P::G2);

    fn setup(
        (num_vars, g1, g2): Self::SetupParams,
    ) -> Result<(Self::ProverSetup, Self::VerifierSetup), OpeningsError> {
        let beta = P::ScalarField::random(&mut OsRng);
        Self::setup_from_secret(beta, num_vars, g1, g2)
            .map_err(|error| OpeningsError::InvalidSetup(error.to_string()))
    }

    fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
        Self::verifier_setup(prover_setup)
    }

    fn commit<S: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &S,
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError> {
        if poly.num_vars() != setup.num_vars {
            return Err(OpeningsError::CommitFailed(
                ZeromorphError::WrongArity {
                    expected: setup.num_vars,
                    got: poly.num_vars(),
                }
                .to_string(),
            ));
        }
        let evaluations = poly.to_dense();
        Self::commit(setup, &evaluations)
            .map(|commitment| (commitment, commitment))
            .map_err(|error| OpeningsError::CommitFailed(error.to_string()))
    }

    fn open<S: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &S,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::Proof, OpeningsError> {
        let commitment = hint.ok_or_else(|| {
            OpeningsError::ProveFailed(ZeromorphError::MissingOpeningHint.to_string())
        })?;
        let evaluations = poly.to_dense();
        Self::open(setup, &evaluations, &commitment, point, eval, transcript)
            .map_err(|error| OpeningsError::ProveFailed(error.to_string()))
    }

    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        Self::verify(setup, commitment, point, eval, proof, transcript)
            .map_err(|_| OpeningsError::VerificationFailed)
    }
}

impl<P: PairingGroup> AdditivelyHomomorphic for ZeromorphScheme<P>
where
    P::ScalarField: AppendToTranscript + Serialize + DeserializeOwned,
    P::G1: Serialize + DeserializeOwned,
    P::G2: Serialize + DeserializeOwned,
{
    fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output {
        assert_eq!(commitments.len(), scalars.len());
        let bases = commitments
            .iter()
            .map(HyperKZGCommitment::point)
            .collect::<Vec<_>>();
        HyperKZGCommitment::new(P::g1_msm(&bases, scalars))
    }

    fn combine_hints(
        hints: Vec<Self::OpeningHint>,
        scalars: &[Self::Field],
    ) -> Self::OpeningHint {
        Self::combine(&hints, scalars)
    }
}

#[cfg(test)]
mod algebra_tests {
    #![expect(
        clippy::indexing_slicing,
        reason = "small deterministic protocol vectors use explicit coordinates"
    )]

    use super::{lifted_degree_quotient, multilinear_quotients};
    use jolt_field::{Fr, Ring};
    use jolt_poly::Polynomial;
    use num_traits::Zero;

    #[test]
    fn quotient_identity_and_constant_tables() {
        for num_vars in 1..=3 {
            let size = 1 << num_vars;
            let evaluations = (0..size)
                .map(|index| Fr::from_u64((index * index + 3) as u64))
                .collect::<Vec<_>>();
            let point = (0..num_vars)
                .map(|index| Fr::from_u64((index + 2) as u64))
                .collect::<Vec<_>>();
            let test_point = (0..num_vars)
                .map(|index| Fr::from_u64((index + 11) as u64))
                .collect::<Vec<_>>();
            let quotients = multilinear_quotients(&evaluations, &point);
            let polynomial = Polynomial::new(evaluations);
            let mut reconstructed = polynomial.evaluate(&point);
            let paper_point = test_point.iter().rev().copied().collect::<Vec<_>>();
            let paper_opening = point.iter().rev().copied().collect::<Vec<_>>();
            for (k, quotient) in quotients.iter().enumerate() {
                let quotient_point = paper_point[..k].iter().rev().copied().collect::<Vec<_>>();
                let quotient_eval = Polynomial::new(quotient.clone()).evaluate(&quotient_point);
                reconstructed += (paper_point[k] - paper_opening[k]) * quotient_eval;
            }
            assert_eq!(reconstructed, polynomial.evaluate(&test_point));

            let constant = multilinear_quotients(&vec![Fr::from_u64(9); size], &point);
            assert!(constant.iter().flatten().all(Zero::is_zero));
        }
    }

    #[test]
    fn lifted_coefficients_match_definition() {
        let num_vars = 3;
        let size = 1 << num_vars;
        let evaluations = (0..size)
            .map(|index| Fr::from_u64((index + 1) as u64))
            .collect::<Vec<_>>();
        let point = [Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let quotients = multilinear_quotients(&evaluations, &point);
        let y = Fr::from_u64(7);
        let lifted = lifted_degree_quotient(&quotients, y, size);
        let mut expected = vec![Fr::zero(); size];
        let mut y_power = Fr::from_u64(1);
        for quotient in &quotients {
            let offset = size - quotient.len();
            for (index, coefficient) in quotient.iter().enumerate() {
                expected[offset + index] += y_power * coefficient;
            }
            y_power *= y;
        }
        assert_eq!(lifted, expected[size / 2..]);
    }
}
