//! HyperKZG commitment scheme implementing `jolt-openings` traits.
//!
//! [`HyperKZGScheme`] is generic over `P: PairingGroup` — instantiate with
//! `Bn254` for the concrete BN254 curve.

use std::marker::PhantomData;

use jolt_crypto::{Commitment, JoltGroup, PairingGroup, Pedersen, PedersenSetup};
use jolt_field::Field;
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, OpeningsError, VcSetupExtractable};
use jolt_poly::Polynomial;
use jolt_transcript::{AppendToTranscript, Transcript};
use num_traits::{One, Zero};
use rayon::prelude::*;

use crate::error::HyperKZGError;
use crate::kzg::{self, challenge_to_field, kzg_open_batch, kzg_verify_batch};
use crate::types::{
    HyperKZGCommitment, HyperKZGProof, HyperKZGProverSetup, HyperKZGVerifierSetup,
};

/// HyperKZG multilinear polynomial commitment scheme.
///
/// Generic over `P: PairingGroup`. Implements [`CommitmentScheme`] and
/// [`AdditivelyHomomorphic`] from `jolt-openings`.
#[derive(Clone)]
pub struct HyperKZGScheme<P: PairingGroup> {
    _phantom: PhantomData<P>,
}

impl<P: PairingGroup> HyperKZGScheme<P>
where
    P::ScalarField: AppendToTranscript,
    P::G1: AppendToTranscript,
{
    /// Generates an SRS from a random generator and secret scalar.
    ///
    /// `max_degree` is the maximum polynomial length (number of evaluations).
    /// The SRS will contain `max_degree + 1` G1 powers and 2 G2 powers.
    pub fn setup<R: rand_core::RngCore>(
        rng: &mut R,
        max_degree: usize,
        g1: P::G1,
        g2: P::G2,
    ) -> HyperKZGProverSetup<P> {
        let beta = P::ScalarField::random(rng);
        Self::setup_from_secret(beta, max_degree, g1, g2)
    }

    /// Generates SRS from a known secret (for deterministic testing).
    pub fn setup_from_secret(
        beta: P::ScalarField,
        max_degree: usize,
        g1: P::G1,
        g2: P::G2,
    ) -> HyperKZGProverSetup<P> {
        let mut g1_powers = Vec::with_capacity(max_degree + 1);
        let mut cur = g1;
        for _ in 0..=max_degree {
            g1_powers.push(cur);
            cur = cur.scalar_mul(&beta);
        }

        let g2_powers = vec![g2, g2.scalar_mul(&beta)];

        HyperKZGProverSetup {
            g1_powers,
            g2_powers,
        }
    }

    /// Derives the verifier setup from a prover setup.
    pub fn verifier_setup(prover: &HyperKZGProverSetup<P>) -> HyperKZGVerifierSetup<P> {
        HyperKZGVerifierSetup::from(prover)
    }

    /// Phase 1 of the HyperKZG protocol: fold the multilinear polynomial.
    ///
    /// Given polynomial $P$ with $2^\ell$ evaluations and opening point
    /// $x = (x_1, \ldots, x_\ell)$, produces $\ell$ polynomials
    /// $P_0 = P, P_1, \ldots, P_{\ell-1}$ where each $P_i$ has half
    /// the length of $P_{i-1}$.
    ///
    /// The folding relation is:
    /// $P_i[j] = (1 - x_{\ell-i}) \cdot P_{i-1}[2j] + x_{\ell-i} \cdot P_{i-1}[2j+1]$
    fn fold_polynomials(
        evals: &[P::ScalarField],
        point: &[P::ScalarField],
    ) -> Vec<Vec<P::ScalarField>> {
        let ell = point.len();
        let mut polys = Vec::with_capacity(ell);
        polys.push(evals.to_vec());

        for i in 0..ell - 1 {
            let prev = &polys[i];
            let half = prev.len() / 2;
            let xi = point[ell - i - 1];
            let mut pi = vec![P::ScalarField::zero(); half];
            pi.par_iter_mut().enumerate().for_each(|(j, pj)| {
                *pj = prev[2 * j] + xi * (prev[2 * j + 1] - prev[2 * j]);
            });
            polys.push(pi);
        }

        polys
    }

    /// Full HyperKZG opening proof.
    pub fn open<T: Transcript>(
        setup: &HyperKZGProverSetup<P>,
        evals: &[P::ScalarField],
        point: &[P::ScalarField],
        transcript: &mut T,
    ) -> Result<HyperKZGProof<P>, HyperKZGError> {
        let ell = point.len();
        let n = evals.len();
        assert_eq!(n, 1 << ell, "evaluation count must be 2^ell");

        // Phase 1: fold
        let polys = Self::fold_polynomials(evals, point);
        assert_eq!(polys.len(), ell);
        assert_eq!(polys[ell - 1].len(), 2);

        // Commit to intermediate polynomials (skip polys[0] — already committed)
        let com: Vec<P::G1> = polys[1..]
            .par_iter()
            .map(|p| kzg::kzg_commit::<P>(p, setup).expect("SRS large enough for intermediate"))
            .collect();

        // Phase 2: derive challenge r
        for c in &com {
            transcript.append(c);
        }
        let r_challenge = transcript.challenge();
        let r = challenge_to_field::<P::ScalarField, T>(r_challenge);
        let u = vec![r, -r, r * r];

        // Phase 3: batch open all polynomials at the three points
        let (w, v) = kzg_open_batch::<P, T>(&polys, &u, setup, transcript);

        Ok(HyperKZGProof { com, w, v })
    }

    /// HyperKZG verification.
    pub fn verify<T: Transcript>(
        vk: &HyperKZGVerifierSetup<P>,
        commitment: &HyperKZGCommitment<P>,
        point: &[P::ScalarField],
        claimed_eval: &P::ScalarField,
        proof: &HyperKZGProof<P>,
        transcript: &mut T,
    ) -> Result<(), HyperKZGError> {
        let ell = point.len();

        let mut com = proof.com.clone();

        // Absorb intermediate commitments
        for c in &com {
            transcript.append(c);
        }
        let r_challenge = transcript.challenge();
        let r = challenge_to_field::<P::ScalarField, T>(r_challenge);

        if r.is_zero() {
            return Err(HyperKZGError::VerificationFailed);
        }

        // Prepend the original commitment as C_0
        com.insert(0, commitment.point);

        let u = vec![r, -r, r * r];

        // Validate proof dimensions
        let v = &proof.v;
        if v.len() != 3 {
            return Err(HyperKZGError::InvalidProof("v must have 3 evaluation rows"));
        }
        if v[0].len() != ell || v[1].len() != ell || v[2].len() != ell {
            return Err(HyperKZGError::InvalidProof(
                "each v row must have ell entries",
            ));
        }

        let ypos = &v[0]; // evaluations at r
        let yneg = &v[1]; // evaluations at -r
        let mut y_sq = v[2].clone(); // evaluations at r^2
        y_sq.push(*claimed_eval);

        // Consistency check: the folding relation must hold across evaluations
        //
        // For each level i, the polynomial P_i is defined by:
        //   P_i(x) = (1 - x_{ell-i}) * P_{i-1,even}(x) + x_{ell-i} * P_{i-1,odd}(x)
        //
        // This implies:
        //   2*r * P_{i+1}(r^2) = r * (1 - x_{ell-i-1}) * (P_i(r) + P_i(-r))
        //                       + x_{ell-i-1} * (P_i(r) - P_i(-r))
        let two = P::ScalarField::from_u64(2);
        for i in 0..ell {
            let lhs = two * r * y_sq[i + 1];
            let rhs = r * (P::ScalarField::one() - point[ell - i - 1]) * (ypos[i] + yneg[i])
                + point[ell - i - 1] * (ypos[i] - yneg[i]);
            if lhs != rhs {
                return Err(HyperKZGError::VerificationFailed);
            }
        }

        // Batch KZG pairing check
        if !kzg_verify_batch::<P, T>(vk, &com, &proof.w, &u, &proof.v, transcript) {
            return Err(HyperKZGError::VerificationFailed);
        }

        Ok(())
    }
}

impl<P: PairingGroup> Commitment for HyperKZGScheme<P> {
    type Output = HyperKZGCommitment<P>;
}

impl<P: PairingGroup> CommitmentScheme for HyperKZGScheme<P>
where
    P::ScalarField: AppendToTranscript,
    P::G1: AppendToTranscript,
{
    type Field = P::ScalarField;
    type Proof = HyperKZGProof<P>;
    type ProverSetup = HyperKZGProverSetup<P>;
    type VerifierSetup = HyperKZGVerifierSetup<P>;
    type Polynomial = Polynomial<P::ScalarField>;
    type OpeningHint = ();

    fn commit(
        evaluations: &[Self::Field],
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        let point = kzg::kzg_commit::<P>(evaluations, setup)
            .expect("SRS must be large enough for the polynomial");
        (HyperKZGCommitment { point }, ())
    }

    fn open(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        _eval: Self::Field,
        setup: &Self::ProverSetup,
        _hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript,
    ) -> Self::Proof {
        Self::open(setup, poly.evaluations(), point, transcript)
            .expect("HyperKZG open should not fail with valid inputs")
    }

    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript,
    ) -> Result<(), OpeningsError> {
        Self::verify(setup, commitment, point, &eval, proof, transcript)
            .map_err(|_| OpeningsError::VerificationFailed)
    }
}

impl<P: PairingGroup> AdditivelyHomomorphic for HyperKZGScheme<P>
where
    P::ScalarField: AppendToTranscript,
    P::G1: AppendToTranscript,
{
    fn combine(
        commitments: &[Self::Output],
        scalars: &[Self::Field],
    ) -> Self::Output {
        assert_eq!(commitments.len(), scalars.len());
        let combined = commitments
            .iter()
            .zip(scalars.iter())
            .map(|(c, s)| c.point.scalar_mul(s))
            .fold(P::G1::identity(), |acc, x| acc + x);
        HyperKZGCommitment { point: combined }
    }
}

impl<P: PairingGroup> VcSetupExtractable<Pedersen<P::G1>> for HyperKZGScheme<P>
where
    P::ScalarField: AppendToTranscript,
    P::G1: AppendToTranscript,
{
    fn extract_vc_setup(
        setup: &Self::ProverSetup,
        capacity: usize,
    ) -> PedersenSetup<P::G1> {
        // Use the first `capacity` SRS G1 powers as Pedersen message generators.
        // Under the discrete-log assumption, powers [g, β·g, β²·g, ...] are
        // computationally independent and suitable as Pedersen generators.
        //
        // The blinding generator is the next power g^(β^capacity), which is
        // independent from the message generators.
        assert!(
            setup.g1_powers.len() > capacity,
            "SRS has {} G1 powers, need at least {} (capacity + 1 for blinding)",
            setup.g1_powers.len(),
            capacity + 1,
        );

        let message_generators = setup.g1_powers[..capacity].to_vec();
        let blinding_generator = setup.g1_powers[capacity];

        PedersenSetup::new(message_generators, blinding_generator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_crypto::Bn254;
    use jolt_field::Fr;
    use jolt_poly::Polynomial;
    use jolt_transcript::Blake2bTranscript;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    type TestScheme = HyperKZGScheme<Bn254>;

    fn test_setup(max_degree: usize) -> (HyperKZGProverSetup<Bn254>, HyperKZGVerifierSetup<Bn254>) {
        let mut rng = ChaCha20Rng::seed_from_u64(0xdead_beef);
        let g1 = Bn254::g1_generator();
        let g2 = Bn254::g2_generator();
        let prover = TestScheme::setup(&mut rng, max_degree, g1, g2);
        let verifier = TestScheme::verifier_setup(&prover);
        (prover, verifier)
    }

    #[test]
    fn commit_open_verify_roundtrip() {
        for ell in [2, 3, 4, 6, 8] {
            let n = 1 << ell;
            let mut rng = ChaCha20Rng::seed_from_u64(ell as u64);
            let (pk, vk) = test_setup(n);

            let poly = Polynomial::<Fr>::random(ell, &mut rng);
            let point: Vec<Fr> = (0..ell).map(|_| Fr::random(&mut rng)).collect();
            let eval = poly.evaluate(&point);

            let (commitment, ()) = TestScheme::commit(poly.evaluations(), &pk);

            let mut prover_transcript = Blake2bTranscript::new(b"test");
            let proof = <TestScheme as CommitmentScheme>::open(
                &poly,
                &point,
                eval,
                &pk,
                None,
                &mut prover_transcript,
            );

            let mut verifier_transcript = Blake2bTranscript::new(b"test");
            let result = <TestScheme as CommitmentScheme>::verify(
                &commitment,
                &point,
                eval,
                &proof,
                &vk,
                &mut verifier_transcript,
            );
            assert!(result.is_ok(), "ell={ell}: verification failed: {result:?}");
        }
    }

    #[test]
    fn wrong_eval_rejects() {
        let ell = 4;
        let n = 1 << ell;
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let (pk, vk) = test_setup(n);

        let poly = Polynomial::<Fr>::random(ell, &mut rng);
        let point: Vec<Fr> = (0..ell).map(|_| Fr::random(&mut rng)).collect();
        let eval = poly.evaluate(&point);
        let wrong_eval = eval + Fr::from_u64(1);

        let (commitment, ()) = TestScheme::commit(poly.evaluations(), &pk);

        let mut prover_transcript = Blake2bTranscript::new(b"test-bad");
        let proof = <TestScheme as CommitmentScheme>::open(
            &poly,
            &point,
            eval,
            &pk,
            None,
            &mut prover_transcript,
        );

        let mut verifier_transcript = Blake2bTranscript::new(b"test-bad");
        let result = <TestScheme as CommitmentScheme>::verify(
            &commitment,
            &point,
            wrong_eval,
            &proof,
            &vk,
            &mut verifier_transcript,
        );
        assert!(result.is_err(), "wrong evaluation should be rejected");
    }

    #[test]
    fn tampered_proof_rejects() {
        let ell = 4;
        let n = 1 << ell;
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let (pk, vk) = test_setup(n);

        let poly = Polynomial::<Fr>::random(ell, &mut rng);
        let point: Vec<Fr> = (0..ell).map(|_| Fr::random(&mut rng)).collect();
        let eval = poly.evaluate(&point);

        let (commitment, ()) = TestScheme::commit(poly.evaluations(), &pk);

        let mut prover_transcript = Blake2bTranscript::new(b"test-tamper");
        let mut proof = <TestScheme as CommitmentScheme>::open(
            &poly,
            &point,
            eval,
            &pk,
            None,
            &mut prover_transcript,
        );

        // Tamper with proof: swap v[0] and v[1]
        let v1 = proof.v[1].clone();
        proof.v[0].clone_from(&v1);

        let mut verifier_transcript = Blake2bTranscript::new(b"test-tamper");
        let result = <TestScheme as CommitmentScheme>::verify(
            &commitment,
            &point,
            eval,
            &proof,
            &vk,
            &mut verifier_transcript,
        );
        assert!(result.is_err(), "tampered proof should be rejected");
    }

    #[test]
    fn combine_is_homomorphic() {
        let ell = 3;
        let n = 1 << ell;
        let mut rng = ChaCha20Rng::seed_from_u64(300);
        let (pk, _vk) = test_setup(n);

        let poly_a = Polynomial::<Fr>::random(ell, &mut rng);
        let poly_b = Polynomial::<Fr>::random(ell, &mut rng);

        let (ca, ()) = TestScheme::commit(poly_a.evaluations(), &pk);
        let (cb, ()) = TestScheme::commit(poly_b.evaluations(), &pk);

        let sum_evals: Vec<Fr> = poly_a
            .evaluations()
            .iter()
            .zip(poly_b.evaluations().iter())
            .map(|(a, b)| *a + *b)
            .collect();
        let (c_sum_direct, ()) = TestScheme::commit(&sum_evals, &pk);

        let c_sum_combined =
            TestScheme::combine(&[ca, cb], &[Fr::from_u64(1), Fr::from_u64(1)]);

        assert_eq!(
            c_sum_direct, c_sum_combined,
            "combine([1,1]) must match commitment to sum"
        );
    }

    #[test]
    fn combine_with_scalars() {
        let ell = 3;
        let n = 1 << ell;
        let mut rng = ChaCha20Rng::seed_from_u64(400);
        let (pk, _vk) = test_setup(n);

        let poly_a = Polynomial::<Fr>::random(ell, &mut rng);
        let poly_b = Polynomial::<Fr>::random(ell, &mut rng);
        let s_a = Fr::random(&mut rng);
        let s_b = Fr::random(&mut rng);

        let (ca, ()) = TestScheme::commit(poly_a.evaluations(), &pk);
        let (cb, ()) = TestScheme::commit(poly_b.evaluations(), &pk);

        let combined_evals: Vec<Fr> = poly_a
            .evaluations()
            .iter()
            .zip(poly_b.evaluations().iter())
            .map(|(a, b)| s_a * *a + s_b * *b)
            .collect();
        let (c_direct, ()) = TestScheme::commit(&combined_evals, &pk);

        let c_combined = TestScheme::combine(&[ca, cb], &[s_a, s_b]);

        assert_eq!(c_direct, c_combined);
    }

    #[test]
    fn open_verify_with_random_points() {
        let mut rng = ChaCha20Rng::seed_from_u64(0xcafe);

        for _ in 0..5 {
            let ell = 4;
            let n = 1 << ell;
            let (pk, vk) = test_setup(n);

            let poly = Polynomial::<Fr>::random(ell, &mut rng);
            let point: Vec<Fr> = (0..ell).map(|_| Fr::random(&mut rng)).collect();
            let eval = poly.evaluate(&point);

            let (commitment, ()) = TestScheme::commit(poly.evaluations(), &pk);

            let mut pt = Blake2bTranscript::new(b"rand-test");
            let proof = <TestScheme as CommitmentScheme>::open(
                &poly, &point, eval, &pk, None, &mut pt,
            );

            let mut vt = Blake2bTranscript::new(b"rand-test");
            <TestScheme as CommitmentScheme>::verify(
                &commitment, &point, eval, &proof, &vk, &mut vt,
            )
            .expect("random instance should verify");
        }
    }

    #[test]
    fn extract_vc_setup_produces_valid_pedersen() {
        use jolt_crypto::JoltCommitment;

        let n = 1 << 4;
        let (pk, _vk) = test_setup(n);

        let capacity = 5;
        let vc_setup =
            <TestScheme as VcSetupExtractable<Pedersen<jolt_crypto::Bn254G1>>>::extract_vc_setup(
                &pk, capacity,
            );

        assert_eq!(
            <Pedersen<jolt_crypto::Bn254G1> as JoltCommitment>::capacity(&vc_setup),
            capacity,
        );

        // Commit and verify a small vector.
        let values = vec![Fr::one(), Fr::from_u64(2), Fr::from_u64(3)];
        let blinding = Fr::from_u64(42);
        let commitment = <Pedersen<jolt_crypto::Bn254G1> as JoltCommitment>::commit(
            &vc_setup, &values, &blinding,
        );
        assert!(<Pedersen<jolt_crypto::Bn254G1> as JoltCommitment>::verify(
            &vc_setup,
            &commitment,
            &values,
            &blinding,
        ));
    }

    #[test]
    fn trivial_polynomial() {
        // 1-variable polynomial: [a, b]
        let ell = 1;
        let n = 1 << ell;
        let mut rng = ChaCha20Rng::seed_from_u64(777);
        let (pk, vk) = test_setup(n);

        let poly = Polynomial::<Fr>::random(ell, &mut rng);
        let point: Vec<Fr> = (0..ell).map(|_| Fr::random(&mut rng)).collect();
        let eval = poly.evaluate(&point);

        let (commitment, ()) = TestScheme::commit(poly.evaluations(), &pk);

        let mut pt = Blake2bTranscript::new(b"trivial");
        let proof = <TestScheme as CommitmentScheme>::open(
            &poly, &point, eval, &pk, None, &mut pt,
        );

        let mut vt = Blake2bTranscript::new(b"trivial");
        <TestScheme as CommitmentScheme>::verify(
            &commitment, &point, eval, &proof, &vk, &mut vt,
        )
        .expect("trivial polynomial should verify");
    }
}
