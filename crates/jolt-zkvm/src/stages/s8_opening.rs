//! Stage 8: Batch opening proofs.
//!
//! Reduces all polynomial opening claims from stages 1–7 via random linear
//! combination (RLC), then produces PCS opening proofs for the reduced claims.
//!
//! Generic over the polynomial commitment scheme — only the concrete
//! instantiation (e.g., in jolt-zkvm's pipeline) specifies Dory or another PCS.

use std::marker::PhantomData;

use jolt_openings::{
    AdditivelyHomomorphic, CommitmentScheme, OpeningReduction, OpeningsError, ProverClaim,
    RlcReduction, VerifierClaim,
};
use jolt_transcript::Transcript;
use serde::{Deserialize, Serialize};

/// Batch opening stage, generic over an additively homomorphic PCS.
///
/// Collects all `ProverClaim`s from prior sumcheck stages, groups them
/// by evaluation point, combines each group via RLC, and opens the
/// reduced claims. The verifier performs the same reduction on
/// `VerifierClaim`s and checks each opening proof.
pub struct OpeningStage<PCS: AdditivelyHomomorphic> {
    _marker: PhantomData<PCS>,
}

/// Prover-side opening proof bundle.
///
/// One proof per distinct evaluation point group after RLC reduction.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct OpeningProofs<PCS: CommitmentScheme> {
    pub proofs: Vec<PCS::Proof>,
}

impl<PCS: AdditivelyHomomorphic> OpeningStage<PCS> {
    /// Reduces and opens all prover claims.
    ///
    /// 1. Groups claims by evaluation point via [`RlcReduction`].
    /// 2. Opens each reduced claim using `PCS::open`.
    ///
    /// The transcript must be in the same Fiat-Shamir state as the verifier's
    /// transcript at this protocol step.
    #[tracing::instrument(skip_all, name = "OpeningStage::prove")]
    pub fn prove<T: Transcript<Challenge = PCS::Field>>(
        claims: Vec<ProverClaim<PCS::Field>>,
        setup: &PCS::ProverSetup,
        transcript: &mut T,
    ) -> OpeningProofs<PCS> {
        let (reduced, ()) =
            <RlcReduction as OpeningReduction<PCS>>::reduce_prover(claims, transcript);

        let proofs = reduced
            .into_iter()
            .map(|claim| {
                let poly: PCS::Polynomial = claim.evaluations.into();
                PCS::open(&poly, &claim.point, claim.eval, setup, None, transcript)
            })
            .collect();

        OpeningProofs { proofs }
    }

    /// Reduces and verifies all verifier claims against opening proofs.
    ///
    /// # Errors
    ///
    /// Returns [`OpeningsError::VerificationFailed`] if any opening proof
    /// is invalid, or if the reduction detects inconsistency.
    #[tracing::instrument(skip_all, name = "OpeningStage::verify")]
    pub fn verify<T: Transcript<Challenge = PCS::Field>>(
        claims: Vec<VerifierClaim<PCS::Field, PCS::Output>>,
        opening_proofs: &OpeningProofs<PCS>,
        setup: &PCS::VerifierSetup,
        transcript: &mut T,
    ) -> Result<(), OpeningsError> {
        let reduced =
            <RlcReduction as OpeningReduction<PCS>>::reduce_verifier(claims, &(), transcript)?;

        if reduced.len() != opening_proofs.proofs.len() {
            return Err(OpeningsError::VerificationFailed);
        }

        for (claim, proof) in reduced.iter().zip(opening_proofs.proofs.iter()) {
            PCS::verify(
                &claim.commitment,
                &claim.point,
                claim.eval,
                proof,
                setup,
                transcript,
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};
    use jolt_openings::mock::MockCommitmentScheme;
    use jolt_poly::Polynomial;
    use jolt_transcript::Blake2bTranscript;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    type MockPCS = MockCommitmentScheme<Fr>;

    #[allow(clippy::type_complexity)]
    fn random_claims<PCS: CommitmentScheme<Field = Fr>>(
        rng: &mut ChaCha20Rng,
        num_polys: usize,
        num_vars: usize,
        point: &[Fr],
        setup: &PCS::ProverSetup,
    ) -> (Vec<ProverClaim<Fr>>, Vec<VerifierClaim<Fr, PCS::Output>>) {
        let mut prover_claims = Vec::with_capacity(num_polys);
        let mut verifier_claims = Vec::with_capacity(num_polys);

        for _ in 0..num_polys {
            let poly = Polynomial::<Fr>::random(num_vars, rng);
            let eval = poly.evaluate(point);

            let (commitment, _hint) = PCS::commit(poly.evaluations(), setup);

            prover_claims.push(ProverClaim {
                evaluations: poly.evaluations().to_vec(),
                point: point.to_vec(),
                eval,
            });

            verifier_claims.push(VerifierClaim {
                commitment,
                point: point.to_vec(),
                eval,
            });
        }

        (prover_claims, verifier_claims)
    }

    #[test]
    fn single_claim_round_trip() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let num_vars = 4;
        let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        let (prover_claims, verifier_claims) =
            random_claims::<MockPCS>(&mut rng, 1, num_vars, &point, &());

        let mut pt = Blake2bTranscript::new(b"s8_single");
        let proofs = OpeningStage::<MockPCS>::prove(prover_claims, &(), &mut pt);

        let mut vt = Blake2bTranscript::new(b"s8_single");
        OpeningStage::<MockPCS>::verify(verifier_claims, &proofs, &(), &mut vt)
            .expect("single claim should verify");
    }

    #[test]
    fn multiple_claims_same_point() {
        let mut rng = ChaCha20Rng::seed_from_u64(100);
        let num_vars = 3;
        let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        let (prover_claims, verifier_claims) =
            random_claims::<MockPCS>(&mut rng, 5, num_vars, &point, &());

        let mut pt = Blake2bTranscript::new(b"s8_shared");
        let proofs = OpeningStage::<MockPCS>::prove(prover_claims, &(), &mut pt);
        assert_eq!(proofs.proofs.len(), 1, "same point → one reduced claim");

        let mut vt = Blake2bTranscript::new(b"s8_shared");
        OpeningStage::<MockPCS>::verify(verifier_claims, &proofs, &(), &mut vt)
            .expect("shared-point claims should verify");
    }

    #[test]
    fn multiple_claims_distinct_points() {
        let mut rng = ChaCha20Rng::seed_from_u64(200);
        let num_vars = 3;

        let point_a: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let point_b: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        let (mut pc_a, mut vc_a) = random_claims::<MockPCS>(&mut rng, 3, num_vars, &point_a, &());
        let (pc_b, vc_b) = random_claims::<MockPCS>(&mut rng, 2, num_vars, &point_b, &());

        pc_a.extend(pc_b);
        vc_a.extend(vc_b);

        let mut pt = Blake2bTranscript::new(b"s8_distinct");
        let proofs = OpeningStage::<MockPCS>::prove(pc_a, &(), &mut pt);
        assert_eq!(proofs.proofs.len(), 2, "two distinct points → two proofs");

        let mut vt = Blake2bTranscript::new(b"s8_distinct");
        OpeningStage::<MockPCS>::verify(vc_a, &proofs, &(), &mut vt)
            .expect("distinct-point claims should verify");
    }

    #[test]
    fn tampered_eval_rejected() {
        let mut rng = ChaCha20Rng::seed_from_u64(300);
        let num_vars = 3;
        let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        let (prover_claims, mut verifier_claims) =
            random_claims::<MockPCS>(&mut rng, 3, num_vars, &point, &());

        // Tamper with second claim's evaluation
        verifier_claims[1].eval += Fr::from_u64(1);

        let mut pt = Blake2bTranscript::new(b"s8_tamper");
        let proofs = OpeningStage::<MockPCS>::prove(prover_claims, &(), &mut pt);

        let mut vt = Blake2bTranscript::new(b"s8_tamper");
        let result = OpeningStage::<MockPCS>::verify(verifier_claims, &proofs, &(), &mut vt);
        assert!(result.is_err(), "tampered evaluation should be rejected");
    }

    #[test]
    fn empty_claims_is_no_op() {
        let mut pt = Blake2bTranscript::new(b"s8_empty");
        let proofs = OpeningStage::<MockPCS>::prove(vec![], &(), &mut pt);
        assert!(proofs.proofs.is_empty());

        let mut vt = Blake2bTranscript::new(b"s8_empty");
        OpeningStage::<MockPCS>::verify(vec![], &proofs, &(), &mut vt)
            .expect("empty claims should verify trivially");
    }

    #[test]
    fn mixed_point_groups_round_trip() {
        let mut rng = ChaCha20Rng::seed_from_u64(400);
        let num_vars = 4;

        let point_a: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let point_b: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let point_c: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        let (mut pc, mut vc) = random_claims::<MockPCS>(&mut rng, 4, num_vars, &point_a, &());
        let (pc_b, vc_b) = random_claims::<MockPCS>(&mut rng, 2, num_vars, &point_b, &());
        let (pc_c, vc_c) = random_claims::<MockPCS>(&mut rng, 1, num_vars, &point_c, &());

        pc.extend(pc_b);
        pc.extend(pc_c);
        vc.extend(vc_b);
        vc.extend(vc_c);

        let mut pt = Blake2bTranscript::new(b"s8_mixed");
        let proofs = OpeningStage::<MockPCS>::prove(pc, &(), &mut pt);
        assert_eq!(
            proofs.proofs.len(),
            3,
            "three distinct points → three proofs"
        );

        let mut vt = Blake2bTranscript::new(b"s8_mixed");
        OpeningStage::<MockPCS>::verify(vc, &proofs, &(), &mut vt)
            .expect("mixed-group claims should verify");
    }

    mod dory {
        use super::*;
        use jolt_dory::DoryScheme;

        #[test]
        fn dory_single_claim_round_trip() {
            let num_vars = 4;
            let mut rng = ChaCha20Rng::seed_from_u64(42);
            let prover_setup = DoryScheme::setup_prover(num_vars);
            let verifier_setup = DoryScheme::setup_verifier(num_vars);

            let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
            let (prover_claims, verifier_claims) =
                random_claims::<DoryScheme>(&mut rng, 1, num_vars, &point, &prover_setup);

            let mut pt = Blake2bTranscript::new(b"s8_dory_single");
            let proofs = OpeningStage::<DoryScheme>::prove(prover_claims, &prover_setup, &mut pt);

            let mut vt = Blake2bTranscript::new(b"s8_dory_single");
            OpeningStage::<DoryScheme>::verify(verifier_claims, &proofs, &verifier_setup, &mut vt)
                .expect("Dory single claim should verify");
        }

        #[test]
        fn dory_multiple_claims_same_point() {
            let num_vars = 3;
            let mut rng = ChaCha20Rng::seed_from_u64(100);
            let prover_setup = DoryScheme::setup_prover(num_vars);
            let verifier_setup = DoryScheme::setup_verifier(num_vars);

            let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
            let (prover_claims, verifier_claims) =
                random_claims::<DoryScheme>(&mut rng, 4, num_vars, &point, &prover_setup);

            let mut pt = Blake2bTranscript::new(b"s8_dory_shared");
            let proofs = OpeningStage::<DoryScheme>::prove(prover_claims, &prover_setup, &mut pt);
            assert_eq!(proofs.proofs.len(), 1, "same point → one reduced proof");

            let mut vt = Blake2bTranscript::new(b"s8_dory_shared");
            OpeningStage::<DoryScheme>::verify(verifier_claims, &proofs, &verifier_setup, &mut vt)
                .expect("Dory shared-point claims should verify");
        }

        #[test]
        fn dory_distinct_points() {
            let num_vars = 3;
            let mut rng = ChaCha20Rng::seed_from_u64(200);
            let prover_setup = DoryScheme::setup_prover(num_vars);
            let verifier_setup = DoryScheme::setup_verifier(num_vars);

            let point_a: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
            let point_b: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

            let (mut pc, mut vc) =
                random_claims::<DoryScheme>(&mut rng, 2, num_vars, &point_a, &prover_setup);
            let (pc_b, vc_b) =
                random_claims::<DoryScheme>(&mut rng, 2, num_vars, &point_b, &prover_setup);

            pc.extend(pc_b);
            vc.extend(vc_b);

            let mut pt = Blake2bTranscript::new(b"s8_dory_distinct");
            let proofs = OpeningStage::<DoryScheme>::prove(pc, &prover_setup, &mut pt);
            assert_eq!(proofs.proofs.len(), 2, "two distinct points → two proofs");

            let mut vt = Blake2bTranscript::new(b"s8_dory_distinct");
            OpeningStage::<DoryScheme>::verify(vc, &proofs, &verifier_setup, &mut vt)
                .expect("Dory distinct-point claims should verify");
        }

        #[test]
        fn dory_tampered_eval_rejected() {
            let num_vars = 2;
            let mut rng = ChaCha20Rng::seed_from_u64(300);
            let prover_setup = DoryScheme::setup_prover(num_vars);
            let verifier_setup = DoryScheme::setup_verifier(num_vars);

            let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
            let (prover_claims, mut verifier_claims) =
                random_claims::<DoryScheme>(&mut rng, 2, num_vars, &point, &prover_setup);

            verifier_claims[0].eval += Fr::from_u64(1);

            let mut pt = Blake2bTranscript::new(b"s8_dory_tamper");
            let proofs = OpeningStage::<DoryScheme>::prove(prover_claims, &prover_setup, &mut pt);

            let mut vt = Blake2bTranscript::new(b"s8_dory_tamper");
            let result = OpeningStage::<DoryScheme>::verify(
                verifier_claims,
                &proofs,
                &verifier_setup,
                &mut vt,
            );
            assert!(result.is_err(), "Dory should reject tampered evaluation");
        }
    }
}
