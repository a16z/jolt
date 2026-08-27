//! Statistical tests for zero-knowledge property of Dory PCS
//!
//! Verifies that proof elements are statistically indistinguishable from uniform
//! random regardless of the witness (polynomial) distribution.
//!
//! ```sh
//! cargo run --release --features "backends zk" --example zk_statistical
//! ```

use ark_serialize::CanonicalSerialize;
use dory_pcs::backends::arkworks::{
    ArkFr, ArkworksPolynomial, Blake2bTranscript, G1Routines, G2Routines, BN254,
};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::poly::Polynomial;
use dory_pcs::{create_evaluation_proof, setup, verify, DoryProof, ZK};
use std::collections::HashMap;
use tracing::info;

const NUM_BUCKETS: usize = 16;
const NUM_TRIALS: usize = 1000;
/// chi-squared critical value: df=15, alpha=0.0001 (Bonferroni-safe for ~360 tests)
const CHI2_CRITICAL: f64 = 43.84;

fn random_polynomial(size: usize) -> ArkworksPolynomial {
    let coefficients: Vec<ArkFr> = (0..size).map(|_| ArkFr::random()).collect();
    ArkworksPolynomial::new(coefficients)
}

fn random_point(num_vars: usize) -> Vec<ArkFr> {
    (0..num_vars).map(|_| ArkFr::random()).collect()
}

fn fresh_transcript() -> Blake2bTranscript<BN254> {
    Blake2bTranscript::new(b"dory-test")
}

struct BucketTracker {
    buckets: HashMap<String, Vec<usize>>,
}

impl BucketTracker {
    fn new() -> Self {
        Self {
            buckets: HashMap::new(),
        }
    }

    fn record(&mut self, name: &str, bucket: usize) {
        self.buckets
            .entry(name.to_string())
            .or_insert_with(|| vec![0; NUM_BUCKETS])[bucket] += 1;
    }

    fn chi_squared(&self, name: &str, expected: f64) -> Option<f64> {
        self.buckets.get(name).map(|buckets| {
            buckets
                .iter()
                .map(|&observed| {
                    let diff = observed as f64 - expected;
                    diff * diff / expected
                })
                .sum()
        })
    }

    fn all_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.buckets.keys().cloned().collect();
        names.sort();
        names
    }
}

fn bucket_from_serializable<T: CanonicalSerialize>(elem: &T) -> usize {
    let mut bytes = Vec::new();
    elem.serialize_compressed(&mut bytes).unwrap();
    (bytes[0] as usize) % NUM_BUCKETS
}

type ArkDoryProof = DoryProof<
    dory_pcs::backends::arkworks::ArkG1,
    dory_pcs::backends::arkworks::ArkG2,
    dory_pcs::backends::arkworks::ArkGT,
>;

fn collect_full_zk_proof_stats(proof: &ArkDoryProof, tracker: &mut BucketTracker) {
    tracker.record("zk_vmv_c", bucket_from_serializable(&proof.vmv_message.c));
    tracker.record("zk_vmv_d2", bucket_from_serializable(&proof.vmv_message.d2));
    tracker.record("zk_vmv_e1", bucket_from_serializable(&proof.vmv_message.e1));

    if let Some(ref e2) = proof.e2 {
        tracker.record("zk_vmv_e2", bucket_from_serializable(e2));
    }
    if let Some(ref y_com) = proof.y_com {
        tracker.record("zk_vmv_y_com", bucket_from_serializable(y_com));
    }

    for (i, msg) in proof.first_messages.iter().enumerate() {
        let prefix = format!("zk_first_{i}");
        tracker.record(
            &format!("{prefix}_d1_left"),
            bucket_from_serializable(&msg.d1_left),
        );
        tracker.record(
            &format!("{prefix}_d1_right"),
            bucket_from_serializable(&msg.d1_right),
        );
        tracker.record(
            &format!("{prefix}_d2_left"),
            bucket_from_serializable(&msg.d2_left),
        );
        tracker.record(
            &format!("{prefix}_d2_right"),
            bucket_from_serializable(&msg.d2_right),
        );
    }

    for (i, msg) in proof.second_messages.iter().enumerate() {
        let prefix = format!("zk_second_{i}");
        tracker.record(
            &format!("{prefix}_c_plus"),
            bucket_from_serializable(&msg.c_plus),
        );
        tracker.record(
            &format!("{prefix}_c_minus"),
            bucket_from_serializable(&msg.c_minus),
        );
        tracker.record(
            &format!("{prefix}_e1_plus"),
            bucket_from_serializable(&msg.e1_plus),
        );
        tracker.record(
            &format!("{prefix}_e1_minus"),
            bucket_from_serializable(&msg.e1_minus),
        );
        tracker.record(
            &format!("{prefix}_e2_plus"),
            bucket_from_serializable(&msg.e2_plus),
        );
        tracker.record(
            &format!("{prefix}_e2_minus"),
            bucket_from_serializable(&msg.e2_minus),
        );
    }

    // ZK proofs must not reveal the folded witness: the clear final message is
    // omitted entirely (the scalar-product Σ-proof below replaces it).
    assert!(
        proof.final_message.is_none(),
        "ZK proof must not carry a clear final message"
    );

    if let Some(ref sigma1) = proof.sigma1_proof {
        tracker.record("sigma1_a1", bucket_from_serializable(&sigma1.a1));
        tracker.record("sigma1_a2", bucket_from_serializable(&sigma1.a2));
        tracker.record("sigma1_z1", bucket_from_serializable(&sigma1.z1));
        tracker.record("sigma1_z2", bucket_from_serializable(&sigma1.z2));
        tracker.record("sigma1_z3", bucket_from_serializable(&sigma1.z3));
    }

    if let Some(ref sigma2) = proof.sigma2_proof {
        tracker.record("sigma2_a", bucket_from_serializable(&sigma2.a));
        tracker.record("sigma2_z1", bucket_from_serializable(&sigma2.z1));
        tracker.record("sigma2_z2", bucket_from_serializable(&sigma2.z2));
    }

    if let Some(ref sp) = proof.scalar_product_proof {
        tracker.record("zk_sp_p1", bucket_from_serializable(&sp.p1));
        tracker.record("zk_sp_p2", bucket_from_serializable(&sp.p2));
        tracker.record("zk_sp_q", bucket_from_serializable(&sp.q));
        tracker.record("zk_sp_r", bucket_from_serializable(&sp.r));
        tracker.record("zk_sp_e1", bucket_from_serializable(&sp.e1));
        tracker.record("zk_sp_e2", bucket_from_serializable(&sp.e2));
        tracker.record("zk_sp_r1", bucket_from_serializable(&sp.r1));
        tracker.record("zk_sp_r2", bucket_from_serializable(&sp.r2));
        tracker.record("zk_sp_r3", bucket_from_serializable(&sp.r3));
    }
}

fn prove_verify_collect(
    poly: &ArkworksPolynomial,
    point: &[ArkFr],
    nu: usize,
    sigma: usize,
    prover_setup: &dory_pcs::ProverSetup<BN254>,
    verifier_setup: &dory_pcs::VerifierSetup<BN254>,
    tracker: &mut BucketTracker,
) {
    let (tier_2, tier_1, commit_blind) = poly
        .commit::<BN254, ZK, G1Routines>(nu, sigma, prover_setup)
        .unwrap();

    let evaluation = poly.evaluate(point);
    let mut transcript = fresh_transcript();
    let (proof, _) = create_evaluation_proof::<_, BN254, G1Routines, G2Routines, _, _, ZK>(
        poly,
        point,
        Some(tier_1),
        commit_blind,
        nu,
        sigma,
        prover_setup,
        &mut transcript,
    )
    .unwrap();

    let mut verifier_transcript = fresh_transcript();
    verify::<_, BN254, G1Routines, G2Routines, _>(
        tier_2,
        evaluation,
        point,
        &proof,
        verifier_setup.clone(),
        &mut verifier_transcript,
    )
    .expect("proof verification failed");

    collect_full_zk_proof_stats(&proof, tracker);
}

fn two_sample_chi_squared(a: &[usize], b: &[usize]) -> f64 {
    let n_a: f64 = a.iter().sum::<usize>() as f64;
    let n_b: f64 = b.iter().sum::<usize>() as f64;
    let n_total = n_a + n_b;

    a.iter()
        .zip(b.iter())
        .map(|(&obs_a, &obs_b)| {
            let pooled = obs_a as f64 + obs_b as f64;
            if pooled < 1.0 {
                return 0.0;
            }
            let expected_a = pooled * n_a / n_total;
            let expected_b = pooled * n_b / n_total;
            let term_a = if expected_a > 0.0 {
                (obs_a as f64 - expected_a).powi(2) / expected_a
            } else {
                0.0
            };
            let term_b = if expected_b > 0.0 {
                (obs_b as f64 - expected_b).powi(2) / expected_b
            } else {
                0.0
            };
            term_a + term_b
        })
        .sum()
}

fn assert_uniformity(trackers: &[(&str, &BucketTracker)], expected: f64) {
    let mut failures = Vec::new();

    for &(label, tracker) in trackers {
        for name in tracker.all_names() {
            if let Some(chi2) = tracker.chi_squared(&name, expected) {
                if chi2 >= CHI2_CRITICAL {
                    failures.push(format!(
                        "{label}/{name}: chi2={chi2:.2} >= {CHI2_CRITICAL:.2}"
                    ));
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "ZK statistical test failed - {} elements showed non-uniform distribution:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn assert_witness_independence(trackers: &[(&str, &BucketTracker)]) {
    let mut failures = Vec::new();

    for i in 0..trackers.len() {
        for j in (i + 1)..trackers.len() {
            let (label_a, tracker_a) = trackers[i];
            let (label_b, tracker_b) = trackers[j];

            for name in tracker_a.all_names() {
                let Some(buckets_a) = tracker_a.buckets.get(&name) else {
                    continue;
                };
                let Some(buckets_b) = tracker_b.buckets.get(&name) else {
                    continue;
                };

                let chi2 = two_sample_chi_squared(buckets_a, buckets_b);
                if chi2 >= CHI2_CRITICAL {
                    failures.push(format!(
                        "{label_a} vs {label_b}/{name}: chi2={chi2:.2} >= {CHI2_CRITICAL:.2}"
                    ));
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "ZK witness independence test failed - {} elements showed witness-dependent distribution:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn test_statistical_indistinguishability(
    prover_setup: &dory_pcs::ProverSetup<BN254>,
    verifier_setup: &dory_pcs::VerifierSetup<BN254>,
) {
    let nu = 2;
    let sigma = 2;
    let poly_size = 1 << (nu + sigma);
    let point = random_point(nu + sigma);

    let mut tracker_zeros = BucketTracker::new();
    let mut tracker_ones = BucketTracker::new();
    let mut tracker_random = BucketTracker::new();

    for _ in 0..NUM_TRIALS {
        let zeros = ArkworksPolynomial::new(vec![ArkFr::zero(); poly_size]);
        prove_verify_collect(
            &zeros,
            &point,
            nu,
            sigma,
            prover_setup,
            verifier_setup,
            &mut tracker_zeros,
        );

        let ones = ArkworksPolynomial::new(vec![ArkFr::one(); poly_size]);
        prove_verify_collect(
            &ones,
            &point,
            nu,
            sigma,
            prover_setup,
            verifier_setup,
            &mut tracker_ones,
        );

        let random = random_polynomial(poly_size);
        prove_verify_collect(
            &random,
            &point,
            nu,
            sigma,
            prover_setup,
            verifier_setup,
            &mut tracker_random,
        );
    }

    let expected = NUM_TRIALS as f64 / NUM_BUCKETS as f64;
    assert_uniformity(
        &[
            ("zeros", &tracker_zeros),
            ("ones", &tracker_ones),
            ("random", &tracker_random),
        ],
        expected,
    );
}

fn test_statistical_indistinguishability_non_square(
    prover_setup: &dory_pcs::ProverSetup<BN254>,
    verifier_setup: &dory_pcs::VerifierSetup<BN254>,
) {
    let nu = 1;
    let sigma = 3;
    let poly_size = 1 << (nu + sigma);
    let point = random_point(nu + sigma);

    let mut tracker_zeros = BucketTracker::new();
    let mut tracker_ones = BucketTracker::new();
    let mut tracker_random = BucketTracker::new();

    for _ in 0..NUM_TRIALS {
        let zeros = ArkworksPolynomial::new(vec![ArkFr::zero(); poly_size]);
        prove_verify_collect(
            &zeros,
            &point,
            nu,
            sigma,
            prover_setup,
            verifier_setup,
            &mut tracker_zeros,
        );

        let ones = ArkworksPolynomial::new(vec![ArkFr::one(); poly_size]);
        prove_verify_collect(
            &ones,
            &point,
            nu,
            sigma,
            prover_setup,
            verifier_setup,
            &mut tracker_ones,
        );

        let random = random_polynomial(poly_size);
        prove_verify_collect(
            &random,
            &point,
            nu,
            sigma,
            prover_setup,
            verifier_setup,
            &mut tracker_random,
        );
    }

    let expected = NUM_TRIALS as f64 / NUM_BUCKETS as f64;
    assert_uniformity(
        &[
            ("zeros", &tracker_zeros),
            ("ones", &tracker_ones),
            ("random", &tracker_random),
        ],
        expected,
    );
}

fn test_witness_independence(
    prover_setup: &dory_pcs::ProverSetup<BN254>,
    verifier_setup: &dory_pcs::VerifierSetup<BN254>,
) {
    let nu = 2;
    let sigma = 2;
    let poly_size = 1 << (nu + sigma);
    let point = random_point(nu + sigma);

    let mut tracker_zeros = BucketTracker::new();
    let mut tracker_ones = BucketTracker::new();
    let mut tracker_skewed = BucketTracker::new();
    let mut tracker_uniform = BucketTracker::new();

    for _ in 0..NUM_TRIALS {
        let zeros = ArkworksPolynomial::new(vec![ArkFr::zero(); poly_size]);
        prove_verify_collect(
            &zeros,
            &point,
            nu,
            sigma,
            prover_setup,
            verifier_setup,
            &mut tracker_zeros,
        );

        let ones = ArkworksPolynomial::new(vec![ArkFr::one(); poly_size]);
        prove_verify_collect(
            &ones,
            &point,
            nu,
            sigma,
            prover_setup,
            verifier_setup,
            &mut tracker_ones,
        );

        let mut skewed_coeffs = vec![ArkFr::zero(); poly_size];
        skewed_coeffs[0] = ArkFr::from_u64(42);
        let skewed = ArkworksPolynomial::new(skewed_coeffs);
        prove_verify_collect(
            &skewed,
            &point,
            nu,
            sigma,
            prover_setup,
            verifier_setup,
            &mut tracker_skewed,
        );

        let uniform = random_polynomial(poly_size);
        prove_verify_collect(
            &uniform,
            &point,
            nu,
            sigma,
            prover_setup,
            verifier_setup,
            &mut tracker_uniform,
        );
    }

    assert_witness_independence(&[
        ("zeros", &tracker_zeros),
        ("ones", &tracker_ones),
        ("skewed", &tracker_skewed),
        ("uniform", &tracker_uniform),
    ]);
}

fn main() {
    tracing_subscriber::fmt::init();

    let (prover_setup, verifier_setup) = setup::<BN254>(6);

    info!("[1/3] statistical indistinguishability (square, nu=2 sigma=2)...");
    test_statistical_indistinguishability(&prover_setup, &verifier_setup);
    info!("      PASS");

    info!("[2/3] statistical indistinguishability (non-square, nu=1 sigma=3)...");
    test_statistical_indistinguishability_non_square(&prover_setup, &verifier_setup);
    info!("      PASS");

    info!("[3/3] witness independence (4 distributions, pairwise chi-squared)...");
    test_witness_independence(&prover_setup, &verifier_setup);
    info!("      PASS");
}
