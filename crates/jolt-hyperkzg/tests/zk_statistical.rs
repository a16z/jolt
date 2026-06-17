//! Statistical tests for ZK HyperKZG hiding.

#![cfg(feature = "zk")]
#![expect(
    clippy::cast_precision_loss,
    clippy::expect_used,
    reason = "statistical tests compute empirical floating-point statistics and fail loudly"
)]

use std::collections::BTreeMap;

use jolt_crypto::{Bn254, JoltGroup};
use jolt_field::{FixedBytes, Fr, FromPrimitiveInt};
use jolt_hyperkzg::{HyperKZGCommitment, HyperKZGProof, HyperKZGScheme};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, ZkOpeningScheme};
use jolt_poly::Polynomial;
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use serde::Serialize;

const DEFAULT_SAMPLES: usize = 128;
const NUM_BUCKETS: usize = 16;
const MIN_SAMPLES: usize = NUM_BUCKETS * 4;
const CHI2_CRITICAL: f64 = 43.84;
const NUM_VARS: usize = 4;

type KzgPCS = HyperKZGScheme<Bn254>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum PolyFamily {
    Zero,
    Sparse,
    Structured,
    Random,
    Combined,
}

impl PolyFamily {
    fn all() -> [Self; 5] {
        [
            Self::Zero,
            Self::Sparse,
            Self::Structured,
            Self::Random,
            Self::Combined,
        ]
    }

    fn name(self) -> &'static str {
        match self {
            Self::Zero => "zero",
            Self::Sparse => "sparse",
            Self::Structured => "structured",
            Self::Random => "random",
            Self::Combined => "combined",
        }
    }
}

struct ZkSample {
    commitment: HyperKZGCommitment<Bn254>,
    proof: HyperKZGProof<Bn254>,
    output_blind: Fr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StableProofShape {
    point_len: usize,
    fold_commitments: usize,
    y_rows: [usize; 3],
    witnesses: usize,
}

impl StableProofShape {
    fn from_proof(point: &[Fr], proof: &HyperKZGProof<Bn254>) -> Self {
        let (y, _) = proof
            .hidden_evaluation_commitments()
            .expect("ZK proof should expose hidden evaluation commitments");
        Self {
            point_len: point.len(),
            fold_commitments: proof.com.len(),
            y_rows: [y[0].len(), y[1].len(), y[2].len()],
            witnesses: proof.w.len(),
        }
    }
}

#[test]
fn zk_statistical_smoke_components_vary() {
    let setup = make_zk_setup(1 << NUM_VARS);
    let verifier_setup = KzgPCS::verifier_setup(&setup);
    let point = fixed_point(NUM_VARS);
    let mut tracker = BucketTracker::new();
    let mut baseline_shape = None;

    for sample_index in 0usize..16 {
        let mut rng = ChaCha20Rng::seed_from_u64(10_000 + sample_index as u64);
        let poly = polynomial_for_family(PolyFamily::Zero, sample_index, &mut rng);
        let sample = prove_family(PolyFamily::Zero, &poly, &point, &setup, sample_index);

        let mut verifier_transcript = Blake2bTranscript::new(b"hyperkzg-zk-stat");
        let _verified_y_out = KzgPCS::verify_zk(
            &sample.commitment,
            &point,
            &sample.proof,
            &verifier_setup,
            &mut verifier_transcript,
        )
        .expect("ZK statistical smoke sample should verify");

        let shape = StableProofShape::from_proof(&point, &sample.proof);
        if let Some(baseline) = &baseline_shape {
            assert_eq!(baseline, &shape);
        } else {
            baseline_shape = Some(shape);
        }
        collect_sample_statistics("", &sample, &mut tracker);
    }

    for name in tracker.names() {
        assert!(
            unique_sample_count(&tracker.samples[&name]) > 1,
            "{name} should vary across repeated ZK samples"
        );
    }
}

#[test]
#[ignore = "run with --release: cargo nextest run -p jolt-hyperkzg hyperkzg_zk_proof_components_are_statistically_independent --features zk --run-ignored ignored-only --cargo-quiet"]
fn hyperkzg_zk_proof_components_are_statistically_independent() {
    require_release_build();

    let samples = statistical_sample_count();
    assert!(
        samples >= MIN_SAMPLES,
        "JOLT_HYPERKZG_ZK_STAT_SAMPLES must be at least {MIN_SAMPLES}"
    );

    let setup = make_zk_setup(1 << NUM_VARS);
    let verifier_setup = KzgPCS::verifier_setup(&setup);
    let point = fixed_point(NUM_VARS);
    let mut family_trackers = BTreeMap::new();
    let mut even = BucketTracker::new();
    let mut odd = BucketTracker::new();
    let mut baseline_shape = None;

    for family in PolyFamily::all() {
        assert!(family_trackers
            .insert(family, BucketTracker::new())
            .is_none());
    }

    for sample_index in 0..samples {
        for family in PolyFamily::all() {
            let mut rng =
                ChaCha20Rng::seed_from_u64(20_000 + sample_index as u64 * 17 + family as u64);
            let poly = polynomial_for_family(family, sample_index, &mut rng);
            let sample = prove_family(family, &poly, &point, &setup, sample_index);

            let mut verifier_transcript = Blake2bTranscript::new(b"hyperkzg-zk-stat");
            let _verified_y_out = KzgPCS::verify_zk(
                &sample.commitment,
                &point,
                &sample.proof,
                &verifier_setup,
                &mut verifier_transcript,
            )
            .expect("ZK statistical sample should verify");

            let shape = StableProofShape::from_proof(&point, &sample.proof);
            if let Some(baseline) = &baseline_shape {
                assert_eq!(
                    baseline, &shape,
                    "all samples should have a stable public statement and proof shape"
                );
            } else {
                baseline_shape = Some(shape);
            }

            let tracker = family_trackers
                .get_mut(&family)
                .expect("family tracker should exist");
            collect_sample_statistics(family.name(), &sample, tracker);

            if family == PolyFamily::Zero {
                if sample_index.is_multiple_of(2) {
                    collect_sample_statistics("", &sample, &mut even);
                } else {
                    collect_sample_statistics("", &sample, &mut odd);
                }
            }
        }
    }

    for (family, tracker) in &family_trackers {
        assert_uniformity(tracker, samples as f64 / NUM_BUCKETS as f64, family.name());
    }

    let baseline = family_trackers
        .get(&PolyFamily::Zero)
        .expect("zero family tracker should exist");
    for family in [
        PolyFamily::Sparse,
        PolyFamily::Structured,
        PolyFamily::Random,
        PolyFamily::Combined,
    ] {
        assert_same_distribution(
            baseline,
            family_trackers
                .get(&family)
                .expect("family tracker should exist"),
            family.name(),
        );
    }
    assert_same_distribution(&even, &odd, "zero split-half");
}

fn make_zk_setup(max_degree: usize) -> jolt_hyperkzg::HyperKZGProverSetup<Bn254> {
    let g1 = Bn254::g1_generator();
    let hiding_g1 = g1.scalar_mul(&Fr::from_u64(17));
    let g2 = Bn254::g2_generator();
    KzgPCS::setup_zk_from_secret(Fr::from_u64(12345), max_degree, g1, hiding_g1, g2)
}

fn fixed_point(num_vars: usize) -> Vec<Fr> {
    (0..num_vars)
        .map(|i| Fr::from_u64((i as u64 + 3) * 11))
        .collect()
}

fn polynomial_for_family(
    family: PolyFamily,
    sample_index: usize,
    rng: &mut ChaCha20Rng,
) -> Polynomial<Fr> {
    let len = 1 << NUM_VARS;
    match family {
        PolyFamily::Zero => Polynomial::new(vec![Fr::from_u64(0); len]),
        PolyFamily::Sparse => {
            let mut evals = vec![Fr::from_u64(0); len];
            evals[0] = Fr::from_u64(1);
            evals[len / 2] = Fr::from_u64(2);
            evals[len - 1] = Fr::from_u64(3);
            Polynomial::new(evals)
        }
        PolyFamily::Structured => Polynomial::new(
            (0..len)
                .map(|i| {
                    let x = i as u64;
                    Fr::from_u64((x * x + 3 * x + 7) % 97)
                })
                .collect(),
        ),
        PolyFamily::Random => Polynomial::<Fr>::random(NUM_VARS, rng),
        PolyFamily::Combined => {
            let mut left_rng = ChaCha20Rng::seed_from_u64(30_000 + sample_index as u64);
            let mut right_rng = ChaCha20Rng::seed_from_u64(40_000 + sample_index as u64);
            let left = Polynomial::<Fr>::random(NUM_VARS, &mut left_rng);
            let right = polynomial_for_family(PolyFamily::Structured, sample_index, &mut right_rng);
            let scalar_left = Fr::from_u64(13);
            let scalar_right = Fr::from_u64(29);
            let evals = left
                .evaluations()
                .iter()
                .zip(right.evaluations())
                .map(|(&lhs, &rhs)| scalar_left * lhs + scalar_right * rhs)
                .collect();
            Polynomial::new(evals)
        }
    }
}

fn prove_family(
    family: PolyFamily,
    poly: &Polynomial<Fr>,
    point: &[Fr],
    setup: &jolt_hyperkzg::HyperKZGProverSetup<Bn254>,
    sample_index: usize,
) -> ZkSample {
    let eval = poly.evaluate(point);
    let mut prover_transcript = Blake2bTranscript::new(b"hyperkzg-zk-stat");

    if family == PolyFamily::Combined {
        let mut left_rng = ChaCha20Rng::seed_from_u64(30_000 + sample_index as u64);
        let mut right_rng = ChaCha20Rng::seed_from_u64(40_000 + sample_index as u64);
        let left = Polynomial::<Fr>::random(NUM_VARS, &mut left_rng);
        let right = polynomial_for_family(PolyFamily::Structured, sample_index, &mut right_rng);
        let scalar_left = Fr::from_u64(13);
        let scalar_right = Fr::from_u64(29);

        let (left_commitment, left_hint) =
            <KzgPCS as ZkOpeningScheme>::commit_zk(left.evaluations(), setup);
        let (right_commitment, right_hint) =
            <KzgPCS as ZkOpeningScheme>::commit_zk(right.evaluations(), setup);
        let commitment = <KzgPCS as AdditivelyHomomorphic>::combine(
            &[left_commitment, right_commitment],
            &[scalar_left, scalar_right],
        );
        let hint = <KzgPCS as AdditivelyHomomorphic>::combine_hints(
            vec![left_hint, right_hint],
            &[scalar_left, scalar_right],
        );
        let (proof, _y_out, output_blind) =
            KzgPCS::open_zk(poly, point, eval, setup, hint, &mut prover_transcript);
        return ZkSample {
            commitment,
            proof,
            output_blind,
        };
    }

    let (commitment, hint) = <KzgPCS as ZkOpeningScheme>::commit_zk(poly.evaluations(), setup);
    let (proof, _y_out, output_blind) =
        KzgPCS::open_zk(poly, point, eval, setup, hint, &mut prover_transcript);
    ZkSample {
        commitment,
        proof,
        output_blind,
    }
}

fn collect_sample_statistics(prefix: &str, sample: &ZkSample, tracker: &mut BucketTracker) {
    let component_prefix = |name: &str| {
        if prefix.is_empty() {
            name.to_string()
        } else {
            format!("{prefix}.{name}")
        }
    };

    tracker.record_serde(component_prefix("commitment"), &sample.commitment);
    tracker.record_append_positions(&component_prefix("fold_com"), &sample.proof.com);

    let (y, y_out) = sample
        .proof
        .hidden_evaluation_commitments()
        .expect("ZK proof should expose hidden evaluation commitments");
    tracker.record_append_positions(&component_prefix("y_r"), &y[0]);
    tracker.record_append_positions(&component_prefix("y_neg_r"), &y[1]);
    tracker.record_append_positions(&component_prefix("y_r2"), &y[2]);
    tracker.record_append(component_prefix("y_out"), y_out);
    tracker.record_append_positions(&component_prefix("witness"), &sample.proof.w);
    tracker.record_append(component_prefix("output_blind"), &sample.output_blind);
}

#[derive(Clone, Debug, Default)]
struct BucketTracker {
    buckets: BTreeMap<String, Vec<usize>>,
    samples: BTreeMap<String, Vec<u64>>,
}

impl BucketTracker {
    fn new() -> Self {
        Self::default()
    }

    fn record_append<A: AppendToTranscript>(&mut self, name: impl Into<String>, value: &A) {
        let name = name.into();
        let mut transcript = Blake2bTranscript::<Fr>::new(b"hkzg-zk-stat");
        transcript.append_bytes(name.as_bytes());
        value.append_to_transcript(&mut transcript);
        self.record_projected(name, field_low_u64(transcript.challenge()));
    }

    fn record_append_positions<A: AppendToTranscript>(&mut self, prefix: &str, values: &[A]) {
        for index in selected_positions(values.len()) {
            self.record_append(format!("{prefix}.{index}"), &values[index]);
        }
    }

    fn record_serde<A: Serialize>(&mut self, name: impl Into<String>, value: &A) {
        let name = name.into();
        let bytes = bincode::serde::encode_to_vec(value, bincode::config::standard())
            .expect("statistical component serialization should succeed");
        self.record_bytes(name, &bytes);
    }

    fn record_bytes(&mut self, name: String, bytes: &[u8]) {
        let mut transcript = Blake2bTranscript::<Fr>::new(b"hkzg-zk-stat");
        transcript.append_bytes(name.as_bytes());
        transcript.append_bytes(bytes);
        self.record_projected(name, field_low_u64(transcript.challenge()));
    }

    fn record_projected(&mut self, name: String, value: u64) {
        self.buckets
            .entry(name.clone())
            .or_insert_with(|| vec![0; NUM_BUCKETS])[(value as usize) % NUM_BUCKETS] += 1;
        self.samples.entry(name).or_default().push(value);
    }

    fn names(&self) -> Vec<String> {
        self.buckets.keys().cloned().collect()
    }

    fn chi_squared(&self, name: &str, expected: f64) -> f64 {
        self.buckets[name]
            .iter()
            .map(|&observed| {
                let delta = observed as f64 - expected;
                delta * delta / expected
            })
            .sum()
    }
}

fn selected_positions(len: usize) -> Vec<usize> {
    match len {
        0 => Vec::new(),
        1 => vec![0],
        2 => vec![0, 1],
        _ => {
            let mut positions = vec![0, len / 2, len - 1];
            positions.dedup();
            positions
        }
    }
}

fn field_low_u64(value: Fr) -> u64 {
    let bytes = value.to_bytes_array();
    u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

fn assert_uniformity(tracker: &BucketTracker, expected: f64, context: &str) {
    let mut failures = Vec::new();
    for name in tracker.names() {
        let unique = unique_sample_count(&tracker.samples[&name]);
        let minimum_unique = tracker.samples[&name].len() * 90 / 100;
        if unique < minimum_unique {
            failures.push(format!(
                "{context}:{name}: only {unique} unique projected samples out of {}",
                tracker.samples[&name].len()
            ));
            continue;
        }

        let chi2 = tracker.chi_squared(&name, expected);
        if chi2 >= CHI2_CRITICAL {
            failures.push(format!(
                "{context}:{name}: chi2={chi2:.2} >= {CHI2_CRITICAL:.2}"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "HyperKZG ZK statistical uniformity check failed for {} projected components:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn assert_same_distribution(lhs: &BucketTracker, rhs: &BucketTracker, context: &str) {
    let mut failures = Vec::new();
    for name in lhs.names() {
        let rhs_name = counterpart_name(&name, context);
        let Some(rhs_buckets) = rhs.buckets.get(&rhs_name) else {
            continue;
        };
        let chi2 = two_sample_chi_squared(&lhs.buckets[&name], rhs_buckets);
        if chi2 >= CHI2_CRITICAL {
            failures.push(format!(
                "{context}:{name}: two-sample chi2={chi2:.2} >= {CHI2_CRITICAL:.2}"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "HyperKZG ZK statistical two-sample check failed for {} projected components:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn counterpart_name(lhs_name: &str, rhs_family_name: &str) -> String {
    let Some((_, suffix)) = lhs_name.split_once('.') else {
        return lhs_name.to_string();
    };
    format!("{rhs_family_name}.{suffix}")
}

fn unique_sample_count(values: &[u64]) -> usize {
    let mut values = values.to_vec();
    values.sort_unstable();
    values.dedup();
    values.len()
}

fn two_sample_chi_squared(a: &[usize], b: &[usize]) -> f64 {
    let n_a = a.iter().sum::<usize>() as f64;
    let n_b = b.iter().sum::<usize>() as f64;
    let n_total = n_a + n_b;

    a.iter()
        .zip(b)
        .map(|(&observed_a, &observed_b)| {
            let pooled = observed_a as f64 + observed_b as f64;
            if pooled < 1.0 {
                return 0.0;
            }
            let expected_a = pooled * n_a / n_total;
            let expected_b = pooled * n_b / n_total;
            let term_a = if expected_a > 0.0 {
                (observed_a as f64 - expected_a).powi(2) / expected_a
            } else {
                0.0
            };
            let term_b = if expected_b > 0.0 {
                (observed_b as f64 - expected_b).powi(2) / expected_b
            } else {
                0.0
            };
            term_a + term_b
        })
        .sum()
}

fn require_release_build() {
    assert!(
        !(cfg!(debug_assertions)
            && std::env::var_os("JOLT_HYPERKZG_ALLOW_DEBUG_STAT_TESTS").is_none()),
        "run this statistical test with --release, or set JOLT_HYPERKZG_ALLOW_DEBUG_STAT_TESTS=1"
    );
}

fn statistical_sample_count() -> usize {
    std::env::var("JOLT_HYPERKZG_ZK_STAT_SAMPLES")
        .ok()
        .map_or(DEFAULT_SAMPLES, |value| {
            value
                .parse::<usize>()
                .expect("JOLT_HYPERKZG_ZK_STAT_SAMPLES must be a positive integer")
        })
}
