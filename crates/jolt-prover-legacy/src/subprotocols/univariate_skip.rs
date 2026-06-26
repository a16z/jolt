use std::marker::PhantomData;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
#[cfg(feature = "zk")]
use rand_core::CryptoRngCore;

use crate::curve::JoltCurve;
use crate::field::JoltField;
#[cfg(feature = "zk")]
use crate::poly::commitment::pedersen::PedersenGenerators;
use crate::poly::lagrange_poly::LagrangePolynomial;
#[cfg(feature = "zk")]
use crate::poly::opening_proof::OpeningId;
use crate::poly::opening_proof::{AbstractVerifierOpeningAccumulator, ProverOpeningAccumulator};
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcript_msgs::{ProverFs, VerifierFs};
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
pub fn prove_uniskip_round<F: JoltField, I: SumcheckInstanceProver<F>>(
    instance: &mut I,
    opening_accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut impl ProverFs<F>,
) -> UniSkipFirstRoundProof<F> {
    let input_claim = instance.input_claim(opening_accumulator);
    let uni_poly = instance.compute_message(0, input_claim);
    // Write the clear first-round polynomial into the prover-side NARG. The
    // modular verifier reads the same frame at this transcript position.
    transcript.write_slice(&uni_poly.coeffs);
    let r0: F::Challenge = transcript.challenge_optimized();
    instance.cache_openings(opening_accumulator, &[r0]);
    opening_accumulator.flush_to_transcript(transcript);
    UniSkipFirstRoundProof::new(uni_poly)
}

/// ZK variant: commits to coefficients instead of revealing them.
/// The polynomial coefficients are stored in the accumulator for BlindFold verification.
#[cfg(feature = "zk")]
pub fn prove_uniskip_round_zk<
    F: JoltField,
    C: JoltCurve<F = F>,
    I: SumcheckInstanceProver<F>,
    R: CryptoRngCore,
>(
    instance: &mut I,
    opening_accumulator: &mut ProverOpeningAccumulator<F>,
    blindfold_accumulator: &mut crate::subprotocols::blindfold::BlindFoldAccumulator<F, C>,
    transcript: &mut impl ProverFs<F>,
    pedersen_gens: &PedersenGenerators<C>,
    rng: &mut R,
) -> ZkUniSkipFirstRoundProof<F, C> {
    use crate::subprotocols::blindfold::UniSkipStageData;

    let input_claim = instance.input_claim(opening_accumulator);
    let uni_poly = instance.compute_message(0, input_claim);
    let poly_degree = uni_poly.degree();

    let blinding = F::random(rng);
    let commitment = pedersen_gens.commit(&uni_poly.coeffs, &blinding);

    // Commitment is prover-only, so record it in the prover-side NARG. The modular
    // verifier bridge still carries the same commitment structurally and replays it
    // with `absorb`; for a single commitment this is the same sponge message as
    // `write_slice(&[commitment])`.
    transcript.write_slice(std::slice::from_ref(&commitment));

    let r0: F::Challenge = transcript.challenge_optimized();
    instance.cache_openings(opening_accumulator, &[r0]);
    transcript.write_slice(std::slice::from_ref(&poly_degree));

    let output_claim_values = opening_accumulator.take_pending_claims();
    let output_claim_ids = opening_accumulator.take_pending_claim_ids();
    let oc_committed: Vec<_> = pedersen_gens.commit_chunked(&output_claim_values, rng);
    let output_claims: Vec<(OpeningId, F)> = output_claim_ids
        .into_iter()
        .zip(output_claim_values)
        .collect();
    let output_claims_commitments: Vec<_> = oc_committed.iter().map(|(c, _)| *c).collect();
    let output_claims_blindings: Vec<_> = oc_committed.iter().map(|(_, b)| *b).collect();
    transcript.write_slice(&output_claims_commitments);

    let input_constraint = instance.get_params().input_claim_constraint();
    let input_constraint_challenge_values = instance
        .get_params()
        .input_constraint_challenge_values(opening_accumulator);
    let output_constraint = instance.get_params().output_claim_constraint();
    let output_constraint_challenge_values = instance
        .get_params()
        .output_constraint_challenge_values(&[r0]);

    blindfold_accumulator.push_uniskip_data(UniSkipStageData {
        input_claim,
        poly_coeffs: uni_poly.coeffs.clone(),
        blinding_factor: blinding,
        challenge: r0,
        poly_degree,
        commitment,
        input_constraint,
        input_constraint_challenge_values,
        output_constraint,
        output_constraint_challenge_values,
        output_claims,
        output_claims_blindings,
        output_claims_commitments: output_claims_commitments.clone(),
    });

    ZkUniSkipFirstRoundProof::new(commitment, poly_degree, output_claims_commitments)
}

/// Proof marker for a univariate-skip first round (non-ZK).
///
/// The prover carries the full first-round polynomial so the structured modular
/// verifier bridge can replay the same Spongefish transcript. The prover also
/// writes this polynomial into its internal NARG; the exported modular proof
/// still consumes the retained structured field.
#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct UniSkipFirstRoundProof<F: JoltField> {
    pub uni_poly: UniPoly<F>,
}

impl<F: JoltField> UniSkipFirstRoundProof<F> {
    pub fn new(uni_poly: UniPoly<F>) -> Self {
        Self { uni_poly }
    }

    /// Verify only the univariate-skip first round. Structured proofs replay the
    /// polynomial as a shared absorb; the empty marker is retained as a fallback
    /// for full-NARG readers that read the polynomial from the NARG.
    pub fn verify<
        const N: usize,
        const FIRST_ROUND_POLY_NUM_COEFFS: usize,
        A: AbstractVerifierOpeningAccumulator<F>,
    >(
        &self,
        sumcheck_instance: &dyn SumcheckInstanceVerifier<F, A>,
        opening_accumulator: &mut A,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<F::Challenge, ProofVerifyError> {
        let degree_bound = sumcheck_instance.degree();

        let uni_poly = if self.uni_poly.coeffs.is_empty() {
            let coeffs: Vec<F> = transcript
                .read_slice()
                .map_err(|_| ProofVerifyError::UniSkipVerificationError)?;
            UniPoly::from_coeff(coeffs)
        } else {
            transcript.absorb_slice(&self.uni_poly.coeffs);
            self.uni_poly.clone()
        };

        // The first-round polynomial has a fixed coefficient count; reject a frame of
        // any other length before it reaches `check_sum_evals` (which indexes by
        // `FIRST_ROUND_POLY_NUM_COEFFS` and assumes exactly that many coefficients).
        if uni_poly.coeffs.len() != FIRST_ROUND_POLY_NUM_COEFFS {
            return Err(ProofVerifyError::InvalidInputLength(
                FIRST_ROUND_POLY_NUM_COEFFS,
                uni_poly.coeffs.len(),
            ));
        }
        if uni_poly.degree() > degree_bound {
            return Err(ProofVerifyError::InvalidInputLength(
                degree_bound,
                uni_poly.degree(),
            ));
        }
        let r0 = transcript.challenge_optimized();

        // Check symmetric-domain sum equals zero (initial claim), and compute next claim s1(r0)
        let input_claim = sumcheck_instance.input_claim(opening_accumulator);
        let input_claim_ok =
            uni_poly.check_sum_evals::<N, FIRST_ROUND_POLY_NUM_COEFFS>(input_claim);

        sumcheck_instance.cache_openings(opening_accumulator, &[r0]);
        let expected_output = uni_poly.evaluate(&r0);
        let claimed_output = sumcheck_instance.expected_output_claim(opening_accumulator, &[r0]);
        let output_claim_ok = claimed_output == expected_output;

        opening_accumulator.flush_to_transcript(transcript);

        if !input_claim_ok || !output_claim_ok {
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
pub struct ZkUniSkipFirstRoundProof<F: JoltField, C: JoltCurve<F = F>> {
    pub commitment: C::G1,
    pub poly_degree: usize,
    /// Pedersen commitments to output claims, chunked to fit generator count
    pub output_claims_commitments: Vec<C::G1>,
    _marker: PhantomData<F>,
}

impl<F: JoltField, C: JoltCurve<F = F>> ZkUniSkipFirstRoundProof<F, C> {
    pub fn new(
        commitment: C::G1,
        poly_degree: usize,
        output_claims_commitments: Vec<C::G1>,
    ) -> Self {
        Self {
            commitment,
            poly_degree,
            output_claims_commitments,
            _marker: PhantomData,
        }
    }

    /// Verify transcript consistency only.
    /// The actual polynomial verification (sum check + evaluation) is done by BlindFold.
    pub fn verify_transcript<
        A: AbstractVerifierOpeningAccumulator<F>,
        I: SumcheckInstanceVerifier<F, A>,
    >(
        &self,
        sumcheck_instance: &I,
        opening_accumulator: &mut A,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<F::Challenge, ProofVerifyError> {
        let degree_bound = sumcheck_instance.degree();
        if self.poly_degree > degree_bound {
            return Err(ProofVerifyError::InvalidInputLength(
                degree_bound,
                self.poly_degree,
            ));
        }

        transcript.absorb(&self.commitment);

        let r0: F::Challenge = transcript.challenge_optimized();
        sumcheck_instance.cache_openings(opening_accumulator, &[r0]);

        transcript.absorb(&self.output_claims_commitments);
        opening_accumulator.take_pending_claims();

        Ok(r0)
    }
}

impl<F: JoltField, C: JoltCurve<F = F>> CanonicalSerialize for ZkUniSkipFirstRoundProof<F, C> {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.commitment.serialize_with_mode(&mut writer, compress)?;
        self.poly_degree
            .serialize_with_mode(&mut writer, compress)?;
        self.output_claims_commitments
            .serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        self.commitment.serialized_size(compress)
            + self.poly_degree.serialized_size(compress)
            + self.output_claims_commitments.serialized_size(compress)
    }
}

impl<F: JoltField, C: JoltCurve<F = F>> CanonicalDeserialize for ZkUniSkipFirstRoundProof<F, C> {
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let commitment = C::G1::deserialize_with_mode(&mut reader, compress, validate)?;
        let poly_degree = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let output_claims_commitments =
            Vec::<C::G1>::deserialize_with_mode(reader, compress, validate)?;
        Ok(Self::new(
            commitment,
            poly_degree,
            output_claims_commitments,
        ))
    }
}

impl<F: JoltField, C: JoltCurve<F = F>> ark_serialize::Valid for ZkUniSkipFirstRoundProof<F, C> {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        self.commitment.check()?;
        self.output_claims_commitments.check()
    }
}

/// Unified proof enum for uni-skip first round (similar to SumcheckInstanceProof).
#[derive(Debug, Clone)]
pub enum UniSkipFirstRoundProofVariant<F: JoltField, C: JoltCurve<F = F>> {
    Standard(UniSkipFirstRoundProof<F>),
    Zk(ZkUniSkipFirstRoundProof<F, C>),
}

impl<F: JoltField, C: JoltCurve<F = F>> UniSkipFirstRoundProofVariant<F, C> {
    pub fn is_zk(&self) -> bool {
        matches!(self, Self::Zk(_))
    }

    /// Returns the polynomial degree for BlindFold R1CS configuration.
    ///
    /// Only queried for ZK proofs (`verify_blindfold`, `cfg(feature = "zk")`),
    /// where the variant is always `Zk`. A `Standard` proof's first-round poly
    /// is irrelevant because BlindFold never runs against it, so that arm is
    /// unreachable.
    pub fn poly_degree(&self) -> usize {
        match self {
            Self::Standard(_) => {
                unreachable!("poly_degree is only queried for ZK (BlindFold) proofs")
            }
            Self::Zk(p) => p.poly_degree,
        }
    }
}

impl<F: JoltField, C: JoltCurve<F = F>> CanonicalSerialize for UniSkipFirstRoundProofVariant<F, C> {
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

impl<F: JoltField, C: JoltCurve<F = F>> CanonicalDeserialize
    for UniSkipFirstRoundProofVariant<F, C>
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

impl<F: JoltField, C: JoltCurve<F = F>> ark_serialize::Valid
    for UniSkipFirstRoundProofVariant<F, C>
{
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        match self {
            Self::Standard(p) => p.check(),
            Self::Zk(p) => p.check(),
        }
    }
}
