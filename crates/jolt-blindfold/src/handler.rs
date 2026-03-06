//! Committed round handlers for ZK sumcheck.
//!
//! These handlers replace cleartext polynomial absorption with commitment-based
//! absorption, making sumcheck zero-knowledge. They are generic over
//! [`JoltCommitment`] — any vector commitment scheme (Pedersen, hash-based,
//! lattice-based) can be used.

use std::marker::PhantomData;

use jolt_crypto::JoltCommitment;
use jolt_field::Field;
use jolt_poly::{UnivariatePoly, UnivariatePolynomial};
use jolt_sumcheck::error::SumcheckError;
use jolt_sumcheck::{RoundHandler, RoundVerifier};
use jolt_transcript::{AppendToTranscript, Transcript};
use rand_core::CryptoRngCore;

use crate::proof::{CommittedRoundData, CommittedSumcheckOutput, CommittedSumcheckProof};

/// ZK round handler: commits to round polynomial coefficients via any
/// vector commitment scheme implementing [`JoltCommitment`].
///
/// For each round, instead of appending polynomial coefficients to the
/// transcript (which would leak the witness), the handler:
/// 1. Draws a blinding factor from `rng`.
/// 2. Commits to the coefficients via `VC::commit`.
/// 3. Appends only the commitment to the transcript.
/// 4. Stores the coefficients and blinding factor for later BlindFold use.
///
/// The resulting [`CommittedSumcheckOutput`] contains both the public proof
/// (commitments only) and private data (coefficients + blindings) needed
/// by the [`BlindFoldAccumulator`](crate::BlindFoldAccumulator).
pub struct CommittedRoundHandler<'a, F, VC, R>
where
    F: Field,
    VC: JoltCommitment,
    R: CryptoRngCore,
{
    setup: &'a VC::Setup,
    rng: &'a mut R,
    round_commitments: Vec<VC::Commitment>,
    poly_coeffs: Vec<Vec<F>>,
    blinding_factors: Vec<F>,
    poly_degrees: Vec<usize>,
    challenges: Vec<F>,
}

impl<'a, F, VC, R> CommittedRoundHandler<'a, F, VC, R>
where
    F: Field,
    VC: JoltCommitment,
    R: CryptoRngCore,
{
    /// - `setup`: commitment scheme parameters (generators, public params).
    /// - `rng`: cryptographic RNG for drawing blinding factors.
    pub fn new(setup: &'a VC::Setup, rng: &'a mut R) -> Self {
        Self {
            setup,
            rng,
            round_commitments: Vec::new(),
            poly_coeffs: Vec::new(),
            blinding_factors: Vec::new(),
            poly_degrees: Vec::new(),
            challenges: Vec::new(),
        }
    }

    /// Pre-allocates for `capacity` rounds to avoid reallocation.
    pub fn with_capacity(setup: &'a VC::Setup, rng: &'a mut R, capacity: usize) -> Self {
        Self {
            setup,
            rng,
            round_commitments: Vec::with_capacity(capacity),
            poly_coeffs: Vec::with_capacity(capacity),
            blinding_factors: Vec::with_capacity(capacity),
            poly_degrees: Vec::with_capacity(capacity),
            challenges: Vec::with_capacity(capacity),
        }
    }
}

impl<F, VC, R> RoundHandler<F> for CommittedRoundHandler<'_, F, VC, R>
where
    F: Field,
    VC: JoltCommitment,
    R: CryptoRngCore,
{
    type Proof = CommittedSumcheckOutput<F, VC>;

    fn absorb_round_poly(&mut self, poly: &UnivariatePoly<F>, transcript: &mut impl Transcript) {
        let coeffs: Vec<F> = poly.coefficients().to_vec();
        let blinding = F::random(&mut self.rng);
        let commitment = VC::commit(self.setup, &coeffs, &blinding);

        commitment.append_to_transcript(transcript);

        self.poly_degrees.push(poly.degree());
        self.round_commitments.push(commitment);
        self.poly_coeffs.push(coeffs);
        self.blinding_factors.push(blinding);
    }

    fn on_challenge(&mut self, challenge: F) {
        self.challenges.push(challenge);
    }

    fn finalize(self) -> CommittedSumcheckOutput<F, VC> {
        let proof = CommittedSumcheckProof {
            round_commitments: self.round_commitments.clone(),
            poly_degrees: self.poly_degrees.clone(),
        };

        let round_data = CommittedRoundData {
            round_commitments: self.round_commitments,
            poly_coeffs: self.poly_coeffs,
            blinding_factors: self.blinding_factors,
            poly_degrees: self.poly_degrees,
            challenges: self.challenges,
        };

        CommittedSumcheckOutput { proof, round_data }
    }
}

/// ZK verifier: absorbs commitments into the transcript and defers all
/// consistency checking to the BlindFold protocol.
///
/// In standard (cleartext) mode, the verifier checks `poly(0) + poly(1) ==
/// running_sum` at each round. In committed mode, the verifier cannot evaluate
/// the polynomial (only commitments are available), so these checks are
/// deferred to BlindFold's verifier R1CS.
///
/// Generic over [`JoltCommitment`] to match the prover's commitment scheme.
pub struct CommittedRoundVerifier<VC: JoltCommitment> {
    _marker: PhantomData<VC>,
}

impl<VC: JoltCommitment> CommittedRoundVerifier<VC> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<VC: JoltCommitment> Default for CommittedRoundVerifier<VC> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field, VC: JoltCommitment> RoundVerifier<F> for CommittedRoundVerifier<VC> {
    type RoundProof = VC::Commitment;

    fn absorb_and_check(
        &self,
        proof: &VC::Commitment,
        _running_sum: F,
        _degree_bound: usize,
        _round: usize,
        transcript: &mut impl Transcript,
    ) -> Result<(), SumcheckError> {
        // Only absorb — consistency check is deferred to BlindFold verification.
        proof.append_to_transcript(transcript);
        Ok(())
    }

    fn next_running_sum(&self, _proof: &VC::Commitment, _challenge: F) -> F {
        // Cannot evaluate committed polynomial; BlindFold verifies this.
        F::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_crypto::arkworks::bn254::Bn254G1;
    use jolt_crypto::{JoltGroup, Pedersen, PedersenSetup};
    use jolt_field::{Field, Fr};
    use jolt_sumcheck::{SumcheckClaim, SumcheckProver, SumcheckWitness};
    use jolt_transcript::Blake2bTranscript;
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    struct TestWitness {
        evals: Vec<Fr>,
    }

    impl SumcheckWitness<Fr> for TestWitness {
        fn round_polynomial(&self) -> UnivariatePoly<Fr> {
            let n = self.evals.len();
            let half = n / 2;
            let s0: Fr = self.evals[..half].iter().copied().sum();
            let s1: Fr = self.evals[half..].iter().copied().sum();
            UnivariatePoly::interpolate(&[(Fr::zero(), s0), (Fr::one(), s1)])
        }

        fn bind(&mut self, challenge: Fr) {
            let n = self.evals.len();
            let half = n / 2;
            let mut new_evals = Vec::with_capacity(half);
            for i in 0..half {
                new_evals.push(
                    self.evals[i] * (Fr::one() - challenge) + self.evals[half + i] * challenge,
                );
            }
            self.evals = new_evals;
        }
    }

    fn test_setup(max_message_len: usize) -> PedersenSetup<Bn254G1> {
        let mut rng = ChaCha20Rng::seed_from_u64(0xBAAD);
        let generators: Vec<Bn254G1> = (0..max_message_len)
            .map(|_| {
                let scalar = Fr::random(&mut rng);
                Bn254G1::default().scalar_mul(&scalar)
            })
            .collect();
        let blinding_gen = {
            let scalar = Fr::random(&mut rng);
            Bn254G1::default().scalar_mul(&scalar)
        };
        PedersenSetup::new(generators, blinding_gen)
    }

    #[test]
    fn committed_handler_produces_correct_output_shape() {
        let num_vars = 3;
        let num_evals = 1 << num_vars;
        let evals: Vec<Fr> = (1..=num_evals).map(|i| Fr::from_u64(i as u64)).collect();
        let claimed_sum: Fr = evals.iter().copied().sum();

        let claim = SumcheckClaim {
            num_vars,
            degree: 1,
            claimed_sum,
        };

        let mut witness = TestWitness { evals };
        let setup = test_setup(2);
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut transcript = Blake2bTranscript::new(b"test-committed");

        let handler = CommittedRoundHandler::<Fr, Pedersen<Bn254G1>, _>::with_capacity(
            &setup, &mut rng, num_vars,
        );

        let output = SumcheckProver::prove_with_handler(
            &claim,
            &mut witness,
            &mut transcript,
            |c: u128| Fr::from_u128(c),
            handler,
        );

        assert_eq!(output.proof.round_commitments.len(), num_vars);
        assert_eq!(output.proof.poly_degrees.len(), num_vars);
        assert_eq!(output.round_data.round_commitments.len(), num_vars);
        assert_eq!(output.round_data.poly_coeffs.len(), num_vars);
        assert_eq!(output.round_data.blinding_factors.len(), num_vars);
        assert_eq!(output.round_data.poly_degrees.len(), num_vars);
        assert_eq!(output.round_data.challenges.len(), num_vars);

        for coeffs in &output.round_data.poly_coeffs {
            assert_eq!(coeffs.len(), 2);
        }

        // Challenges must be non-zero (derived from transcript)
        for ch in &output.round_data.challenges {
            assert_ne!(*ch, Fr::zero());
        }
    }

    #[test]
    fn committed_verifier_absorbs_commitments_and_returns_zero() {
        let verifier = CommittedRoundVerifier::<Pedersen<Bn254G1>>::new();
        let commitment = Bn254G1::default();
        let mut transcript = Blake2bTranscript::new(b"test-verifier");

        let result =
            verifier.absorb_and_check(&commitment, Fr::from_u64(42), 2, 0, &mut transcript);
        assert!(result.is_ok());

        let next = verifier.next_running_sum(&commitment, Fr::from_u64(7));
        assert_eq!(next, Fr::zero());
    }

    #[test]
    fn prover_verifier_transcript_sync() {
        let num_vars = 2;
        let evals: Vec<Fr> = (1..=4).map(Fr::from_u64).collect();
        let claimed_sum: Fr = evals.iter().copied().sum();

        let claim = SumcheckClaim {
            num_vars,
            degree: 1,
            claimed_sum,
        };

        let mut witness = TestWitness {
            evals: evals.clone(),
        };
        let setup = test_setup(2);
        let mut rng = ChaCha20Rng::seed_from_u64(99);

        let mut prover_transcript = Blake2bTranscript::new(b"sync-test");
        let handler = CommittedRoundHandler::<Fr, Pedersen<Bn254G1>, _>::new(&setup, &mut rng);

        let output = SumcheckProver::prove_with_handler(
            &claim,
            &mut witness,
            &mut prover_transcript,
            |c: u128| Fr::from_u128(c),
            handler,
        );

        let mut verifier_transcript = Blake2bTranscript::new(b"sync-test");
        let verifier = CommittedRoundVerifier::<Pedersen<Bn254G1>>::new();

        let result = jolt_sumcheck::SumcheckVerifier::verify_with_handler(
            &claim,
            &output.proof.round_commitments,
            &mut verifier_transcript,
            |c: u128| Fr::from_u128(c),
            &verifier,
        );

        assert!(result.is_ok());

        let (final_val, challenges) = result.unwrap();
        // Committed mode always returns zero; actual check is deferred to BlindFold.
        assert_eq!(final_val, Fr::zero());
        assert_eq!(challenges.len(), num_vars);
    }
}
