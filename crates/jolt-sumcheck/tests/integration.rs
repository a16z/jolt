//! Cross-crate integration tests for the sumcheck protocol.

use jolt_field::{Field, Fr};
use jolt_poly::{EqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::claim::SumcheckClaim;
use jolt_sumcheck::prover::{SumcheckCompute, SumcheckProver};
use jolt_sumcheck::verifier::SumcheckVerifier;
use jolt_transcript::{Blake2bTranscript, KeccakTranscript, Transcript};
use num_traits::Zero;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

struct EqProductWitness {
    poly: Polynomial<Fr>,
    eq_evals: Vec<Fr>,
}

impl EqProductWitness {
    fn new(poly: Polynomial<Fr>, tau: &[Fr]) -> Self {
        let eq_evals = EqPolynomial::new(tau.to_vec()).evaluations();
        Self { poly, eq_evals }
    }
}

impl SumcheckCompute<Fr> for EqProductWitness {
    fn round_polynomial(&self) -> UnivariatePoly<Fr> {
        let half = self.poly.len() / 2;
        let mut evals = [Fr::zero(); 3];
        for i in 0..half {
            let f_lo = self.poly.evaluations()[i];
            let f_hi = self.poly.evaluations()[i + half];
            let eq_lo = self.eq_evals[i];
            let eq_hi = self.eq_evals[i + half];
            evals[0] += f_lo * eq_lo;
            evals[1] += f_hi * eq_hi;
            let f_at_2 = f_lo + (f_hi - f_lo) + (f_hi - f_lo);
            let eq_at_2 = eq_lo + (eq_hi - eq_lo) + (eq_hi - eq_lo);
            evals[2] += f_at_2 * eq_at_2;
        }
        let points: Vec<(Fr, Fr)> = evals
            .iter()
            .enumerate()
            .map(|(t, &y)| (Fr::from_u64(t as u64), y))
            .collect();
        UnivariatePoly::interpolate(&points)
    }

    fn bind(&mut self, challenge: Fr) {
        self.poly.bind(challenge);
        let half = self.eq_evals.len() / 2;
        for i in 0..half {
            let lo = self.eq_evals[i];
            let hi = self.eq_evals[i + half];
            self.eq_evals[i] = lo + challenge * (hi - lo);
        }
        self.eq_evals.truncate(half);
    }
}

#[test]
fn blake2b_and_keccak_both_verify() {
    let mut rng = ChaCha20Rng::seed_from_u64(1000);
    let num_vars = 5;
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let tau: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let claimed_sum = poly.evaluate(&tau);

    let claim = SumcheckClaim {
        num_vars,
        degree: 2,
        claimed_sum,
    };

    // Prove with Blake2b
    let mut w_b = EqProductWitness::new(poly.clone(), &tau);
    let mut pt_b = Blake2bTranscript::new(b"cross_crate");
    let proof_b = SumcheckProver::prove(&claim, &mut w_b, &mut pt_b);

    let mut vt_b = Blake2bTranscript::new(b"cross_crate");
    let (eval_b, challenges_b) = SumcheckVerifier::verify(&claim, &proof_b, &mut vt_b)
        .expect("Blake2b verification should succeed");

    let expected_b =
        poly.evaluate(&challenges_b) * EqPolynomial::new(tau.clone()).evaluate(&challenges_b);
    assert_eq!(eval_b, expected_b);

    // Prove with Keccak
    let mut w_k = EqProductWitness::new(poly.clone(), &tau);
    let mut pt_k = KeccakTranscript::new(b"cross_crate");
    let proof_k = SumcheckProver::prove(&claim, &mut w_k, &mut pt_k);

    let mut vt_k = KeccakTranscript::new(b"cross_crate");
    let (eval_k, challenges_k) = SumcheckVerifier::verify(&claim, &proof_k, &mut vt_k)
        .expect("Keccak verification should succeed");

    let expected_k = poly.evaluate(&challenges_k) * EqPolynomial::new(tau).evaluate(&challenges_k);
    assert_eq!(eval_k, expected_k);

    // Challenges should differ between transcript backends
    assert_ne!(
        challenges_b, challenges_k,
        "Blake2b and Keccak should produce different challenges"
    );
}

#[test]
fn evaluate_then_prove_then_verify() {
    let mut rng = ChaCha20Rng::seed_from_u64(2000);
    let num_vars = 6;
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let tau: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

    // Step 1: evaluate
    let expected_eval = poly.evaluate(&tau);

    // Step 2: prove evaluation via sumcheck
    let claim = SumcheckClaim {
        num_vars,
        degree: 2,
        claimed_sum: expected_eval,
    };

    let mut witness = EqProductWitness::new(poly.clone(), &tau);
    let mut pt = Blake2bTranscript::new(b"eval_prove");
    let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

    // Step 3: verify
    let mut vt = Blake2bTranscript::new(b"eval_prove");
    let (final_eval, challenges) =
        SumcheckVerifier::verify(&claim, &proof, &mut vt).expect("verification should succeed");

    // Final evaluation matches f(r) * eq(r, tau)
    let f_r = poly.evaluate(&challenges);
    let eq_r = EqPolynomial::new(tau).evaluate(&challenges);
    assert_eq!(final_eval, f_r * eq_r);
}
