use std::marker::PhantomData;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand_core::CryptoRngCore;

use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::poly::commitment::pedersen::PedersenGenerators;
use crate::poly::lagrange_poly::LagrangePolynomial;
use crate::poly::opening_proof::{
    ProverOpeningAccumulator, UniSkipStageData, VerifierOpeningAccumulator,
};
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcripts::Transcript;
use crate::utils::errors::ProofVerifyError;

/// Returns the interleaved symmetric univariate-skip target indices outside the base window.
///
/// Domain is assumed to be the canonical symmetric window of size DOMAIN_SIZE with
/// base indices from start = -((DOMAIN_SIZE-1)/2) to end = start + DOMAIN_SIZE - 1.
///
/// Targets are the extended points z ∈ {−DEGREE..−1} ∪ {1..DEGREE}, interleaved as
/// [start-1, end+1, start-2, end+2, ...] until DEGREE points are produced.
#[inline]
pub const fn uniskip_targets<const DOMAIN_SIZE: usize, const DEGREE: usize>() -> [i64; DEGREE] {
    let d: i64 = DEGREE as i64;
    let ext_left: i64 = -d;
    let ext_right: i64 = d;
    let base_left: i64 = -((DOMAIN_SIZE as i64 - 1) / 2);
    let base_right: i64 = base_left + (DOMAIN_SIZE as i64) - 1;

    let mut targets: [i64; DEGREE] = [0; DEGREE];
    let mut idx = 0usize;
    let mut n = base_left - 1;
    let mut p = base_right + 1;

    while n >= ext_left && p <= ext_right && idx < DEGREE {
        targets[idx] = n;
        idx += 1;
        if idx >= DEGREE {
            break;
        }
        targets[idx] = p;
        idx += 1;
        n -= 1;
        p += 1;
    }

    while idx < DEGREE && n >= ext_left {
        targets[idx] = n;
        idx += 1;
        n -= 1;
    }

    while idx < DEGREE && p <= ext_right {
        targets[idx] = p;
        idx += 1;
        p += 1;
    }

    targets
}

/// Builds the uni-skip first-round polynomial s1 from base and extended evaluations of t1.
///
/// SPECIFIC: This helper targets the setting where s1(Y) = L(τ_high, Y) · t1(Y), with L the
/// degree-(DOMAIN_SIZE-1) Lagrange kernel over the base window and t1 a univariate of degree
/// at most 2·DEGREE (extended symmetric window size EXTENDED_SIZE = 2·DEGREE + 1).
/// Consequently, the resulting s1 has degree at most 3·DEGREE (NUM_COEFFS = 3·DEGREE + 1).
///
/// Inputs:
/// - base_evals: optional t1 evaluations on the base window (symmetric grid of size DOMAIN_SIZE).
///   When `None`, base evaluations are treated as all zeros.
/// - extended_evals: t1 evaluated on the extended symmetric grid outside the base window,
///   in the order given by `uniskip_targets::<DOMAIN_SIZE, DEGREE>()`.
/// - tau_high: the challenge used in the Lagrange kernel L(τ_high, ·) over the base window.
///
/// Returns: UniPoly s1 with exactly NUM_COEFFS coefficients.
#[inline]
pub fn build_uniskip_first_round_poly<
    F: JoltField,
    const DOMAIN_SIZE: usize,
    const DEGREE: usize,
    const EXTENDED_SIZE: usize,
    const NUM_COEFFS: usize,
>(
    base_evals: Option<&[F; DOMAIN_SIZE]>,
    extended_evals: &[F; DEGREE],
    tau_high: F::Challenge,
) -> UniPoly<F> {
    debug_assert_eq!(EXTENDED_SIZE, 2 * DEGREE + 1);
    debug_assert_eq!(NUM_COEFFS, 3 * DEGREE + 1);

    // Rebuild t1 on the full extended symmetric window
    let targets: [i64; DEGREE] = uniskip_targets::<DOMAIN_SIZE, DEGREE>();
    let mut t1_vals: [F; EXTENDED_SIZE] = [F::zero(); EXTENDED_SIZE];

    // Fill in base window evaluations when provided (otherwise treated as zeros)
    if let Some(base) = base_evals {
        let base_left: i64 = -((DOMAIN_SIZE as i64 - 1) / 2);
        for (i, &val) in base.iter().enumerate() {
            let z = base_left + (i as i64);
            let pos = (z + (DEGREE as i64)) as usize;
            t1_vals[pos] = val;
        }
    }

    // Fill in extended evaluations (outside base window)
    for (idx, &val) in extended_evals.iter().enumerate() {
        let z = targets[idx];
        let pos = (z + (DEGREE as i64)) as usize;
        t1_vals[pos] = val;
    }

    let t1_coeffs = LagrangePolynomial::<F>::interpolate_coeffs::<EXTENDED_SIZE>(&t1_vals);
    let lagrange_values = LagrangePolynomial::<F>::evals::<F::Challenge, DOMAIN_SIZE>(&tau_high);
    let lagrange_coeffs =
        LagrangePolynomial::<F>::interpolate_coeffs::<DOMAIN_SIZE>(&lagrange_values);

    let mut s1_coeffs: [F; NUM_COEFFS] = [F::zero(); NUM_COEFFS];
    for (i, &a) in lagrange_coeffs.iter().enumerate() {
        for (j, &b) in t1_coeffs.iter().enumerate() {
            s1_coeffs[i + j] += a * b;
        }
    }

    UniPoly::from_coeff(s1_coeffs.to_vec())
}

/// Prove-only helper for a uni-skip first round instance (non-ZK mode).
/// Produces the proof object, the uni-skip challenge r0, and the next claim s1(r0).
pub fn prove_uniskip_round<F: JoltField, T: Transcript, I: SumcheckInstanceProver<F, T>>(
    instance: &mut I,
    opening_accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
) -> UniSkipFirstRoundProof<F, T> {
    let input_claim = instance.input_claim(opening_accumulator);
    let uni_poly = instance.compute_message(0, input_claim);
    // Append full polynomial and derive r0
    transcript.append_scalars(b"uniskip_poly", &uni_poly.coeffs);
    let r0: F::Challenge = transcript.challenge_scalar_optimized::<F>();
    instance.cache_openings(opening_accumulator, &[r0]);
    UniSkipFirstRoundProof::new(uni_poly)
}

/// ZK variant: commits to coefficients instead of revealing them.
/// The polynomial coefficients are stored in the accumulator for BlindFold verification.
pub fn prove_uniskip_round_zk<
    F: JoltField,
    C: JoltCurve,
    T: Transcript,
    I: SumcheckInstanceProver<F, T>,
    R: CryptoRngCore,
>(
    instance: &mut I,
    opening_accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
    pedersen_gens: &PedersenGenerators<C>,
    rng: &mut R,
) -> ZkUniSkipFirstRoundProof<F, C, T> {
    let input_claim = instance.input_claim(opening_accumulator);
    let uni_poly = instance.compute_message(0, input_claim);
    let poly_degree = uni_poly.degree();

    // Generate blinding and compute Pedersen commitment to all coefficients
    let blinding = F::random(rng);
    let commitment = pedersen_gens.commit(&uni_poly.coeffs, &blinding);

    // Serialize commitment for transcript
    let mut commitment_bytes = Vec::new();
    commitment
        .serialize_compressed(&mut commitment_bytes)
        .expect("Serialization should not fail");

    // Append commitment to transcript (NOT raw coefficients)
    transcript.append_bytes(b"sumcheck_commitment", &commitment_bytes);

    let r0: F::Challenge = transcript.challenge_scalar_optimized::<F>();
    instance.cache_openings(opening_accumulator, &[r0]);

    // Get input constraint from the instance params
    let input_constraint = instance.get_params().input_claim_constraint();
    let input_constraint_challenge_values = instance
        .get_params()
        .input_constraint_challenge_values(opening_accumulator);

    // Store uni-skip data in accumulator for BlindFold
    opening_accumulator.push_uniskip_stage_data(UniSkipStageData {
        input_claim,
        poly_coeffs: uni_poly.coeffs.clone(),
        blinding_factor: blinding,
        challenge: r0,
        poly_degree,
        commitment_bytes,
        input_constraint,
        input_constraint_challenge_values,
    });

    ZkUniSkipFirstRoundProof::new(commitment, poly_degree)
}

/// The sumcheck proof for a univariate skip round
/// Consists of the (single) univariate polynomial sent in that round, no omission of any coefficient
#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct UniSkipFirstRoundProof<F: JoltField, T: Transcript> {
    pub uni_poly: UniPoly<F>,
    _marker: PhantomData<T>,
}

impl<F: JoltField, T: Transcript> UniSkipFirstRoundProof<F, T> {
    pub fn new(uni_poly: UniPoly<F>) -> Self {
        Self {
            uni_poly,
            _marker: PhantomData,
        }
    }

    /// Verify only the univariate-skip first round.
    /// Returns the challenge derived during verification.
    pub fn verify<const N: usize, const FIRST_ROUND_POLY_NUM_COEFFS: usize>(
        proof: &Self,
        sumcheck_instance: &dyn SumcheckInstanceVerifier<F, T>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<F::Challenge, ProofVerifyError> {
        let degree_bound = sumcheck_instance.degree();
        // Degree check for the high-degree first polynomial
        if proof.uni_poly.degree() > degree_bound {
            return Err(ProofVerifyError::InvalidInputLength(
                degree_bound,
                proof.uni_poly.degree(),
            ));
        }

        // Append full polynomial and derive r0
        transcript.append_scalars(b"uniskip_poly", &proof.uni_poly.coeffs);
        let r0 = transcript.challenge_scalar_optimized::<F>();

        // Check symmetric-domain sum equals zero (initial claim), and compute next claim s1(r0)
        let input_claim = sumcheck_instance.input_claim(opening_accumulator);
        let ok = proof
            .uni_poly
            .check_sum_evals::<N, FIRST_ROUND_POLY_NUM_COEFFS>(input_claim);
        sumcheck_instance.cache_openings(opening_accumulator, &[r0]);

        if !ok {
            Err(ProofVerifyError::UniSkipVerificationError)
        } else {
            Ok(r0)
        }
    }
}

/// ZK variant of uni-skip first round proof.
/// Contains only the Pedersen commitment to polynomial coefficients.
/// Actual verification is deferred to BlindFold R1CS.
#[derive(Debug, Clone)]
pub struct ZkUniSkipFirstRoundProof<F: JoltField, C: JoltCurve, T: Transcript> {
    pub commitment: C::G1,
    pub poly_degree: usize,
    _marker: PhantomData<(F, T)>,
}

impl<F: JoltField, C: JoltCurve, T: Transcript> ZkUniSkipFirstRoundProof<F, C, T> {
    pub fn new(commitment: C::G1, poly_degree: usize) -> Self {
        Self {
            commitment,
            poly_degree,
            _marker: PhantomData,
        }
    }

    /// Verify transcript consistency only.
    /// The actual polynomial verification (sum check + evaluation) is done by BlindFold.
    pub fn verify_transcript<I: SumcheckInstanceVerifier<F, T>>(
        &self,
        sumcheck_instance: &I,
        opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<F::Challenge, ProofVerifyError> {
        let degree_bound = sumcheck_instance.degree();
        if self.poly_degree > degree_bound {
            return Err(ProofVerifyError::InvalidInputLength(
                degree_bound,
                self.poly_degree,
            ));
        }

        // Serialize commitment and append to transcript
        let mut commitment_bytes = Vec::new();
        self.commitment
            .serialize_compressed(&mut commitment_bytes)
            .map_err(|_| ProofVerifyError::SerializationError)?;

        transcript.append_bytes(b"sumcheck_commitment", &commitment_bytes);

        let r0: F::Challenge = transcript.challenge_scalar_optimized::<F>();
        sumcheck_instance.cache_openings(opening_accumulator, &[r0]);

        Ok(r0)
    }
}

impl<F: JoltField, C: JoltCurve, T: Transcript> CanonicalSerialize
    for ZkUniSkipFirstRoundProof<F, C, T>
{
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.commitment.serialize_with_mode(&mut writer, compress)?;
        self.poly_degree.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        self.commitment.serialized_size(compress) + self.poly_degree.serialized_size(compress)
    }
}

impl<F: JoltField, C: JoltCurve, T: Transcript> CanonicalDeserialize
    for ZkUniSkipFirstRoundProof<F, C, T>
{
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let commitment = C::G1::deserialize_with_mode(&mut reader, compress, validate)?;
        let poly_degree = usize::deserialize_with_mode(reader, compress, validate)?;
        Ok(Self::new(commitment, poly_degree))
    }
}

impl<F: JoltField, C: JoltCurve, T: Transcript> ark_serialize::Valid
    for ZkUniSkipFirstRoundProof<F, C, T>
{
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        self.commitment.check()
    }
}

/// Unified proof enum for uni-skip first round (similar to SumcheckInstanceProof).
#[derive(Debug, Clone)]
pub enum UniSkipFirstRoundProofVariant<F: JoltField, C: JoltCurve, T: Transcript> {
    Standard(UniSkipFirstRoundProof<F, T>),
    Zk(ZkUniSkipFirstRoundProof<F, C, T>),
}

impl<F: JoltField, C: JoltCurve, T: Transcript> UniSkipFirstRoundProofVariant<F, C, T> {
    /// Returns the polynomial degree for BlindFold R1CS configuration.
    pub fn poly_degree(&self) -> usize {
        match self {
            Self::Standard(p) => p.uni_poly.degree(),
            Self::Zk(p) => p.poly_degree,
        }
    }
}

impl<F: JoltField, C: JoltCurve, T: Transcript> CanonicalSerialize
    for UniSkipFirstRoundProofVariant<F, C, T>
{
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        match self {
            Self::Standard(proof) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                proof.serialize_with_mode(writer, compress)
            }
            Self::Zk(proof) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                proof.serialize_with_mode(writer, compress)
            }
        }
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        1 + match self {
            Self::Standard(proof) => proof.serialized_size(compress),
            Self::Zk(proof) => proof.serialized_size(compress),
        }
    }
}

impl<F: JoltField, C: JoltCurve, T: Transcript> CanonicalDeserialize
    for UniSkipFirstRoundProofVariant<F, C, T>
{
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let variant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match variant {
            0 => {
                let proof =
                    UniSkipFirstRoundProof::deserialize_with_mode(reader, compress, validate)?;
                Ok(Self::Standard(proof))
            }
            1 => {
                let proof =
                    ZkUniSkipFirstRoundProof::deserialize_with_mode(reader, compress, validate)?;
                Ok(Self::Zk(proof))
            }
            _ => Err(ark_serialize::SerializationError::InvalidData),
        }
    }
}

impl<F: JoltField, C: JoltCurve, T: Transcript> ark_serialize::Valid
    for UniSkipFirstRoundProofVariant<F, C, T>
{
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        match self {
            Self::Standard(p) => p.check(),
            Self::Zk(p) => p.check(),
        }
    }
}
