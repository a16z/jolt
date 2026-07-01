#![cfg_attr(
    all(feature = "prover-fixtures", feature = "zk"),
    expect(
        clippy::cast_precision_loss,
        clippy::expect_used,
        reason = "statistical tests compute empirical floating-point test statistics and fail loudly"
    )
)]

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use std::collections::BTreeMap;

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use ark_serialize::CanonicalSerialize;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_dory::DoryCommitment;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_field::{FixedBytes, Fr};
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_transcript::{prover_transcript, Blake2b512, FsAbsorb, FsChallenge};

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
const DEFAULT_SAMPLES: usize = 64;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
const NUM_BUCKETS: usize = 16;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
const MIN_SAMPLES: usize = NUM_BUCKETS * 4;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
const CHI2_CRITICAL: f64 = 43.84;

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
#[ignore = "run with --release: cargo nextest run -p jolt-verifier --release --features prover-fixtures,zk zk_muldiv_jolt_proof_components_are_statistically_independent --run-ignored ignored-only --cargo-quiet"]
fn zk_muldiv_jolt_proof_components_are_statistically_independent() {
    with_zk_statistical_stack(run_zk_muldiv_jolt_proof_components_are_statistically_independent);
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn run_zk_muldiv_jolt_proof_components_are_statistically_independent() {
    require_release_build();

    let samples = statistical_sample_count();
    assert!(
        samples >= MIN_SAMPLES,
        "JOLT_VERIFIER_ZK_STAT_SAMPLES must be at least {MIN_SAMPLES}"
    );

    let mut all = BucketTracker::new();
    let mut even = BucketTracker::new();
    let mut odd = BucketTracker::new();
    let mut baseline = None;

    for sample_index in 0..samples {
        let case = crate::support::verifier_fixtures::fresh_zk_muldiv_case();
        crate::support::assert_zk_accepts(case.verify());

        let shape = StableZkProofShape::from_case(&case);
        if let Some(baseline) = &baseline {
            assert_eq!(
                baseline, &shape,
                "fresh ZK samples must prove the same public statement and proof shape"
            );
        } else {
            baseline = Some(shape);
        }

        collect_jolt_proof_statistics(&case, &mut all);
        if sample_index.is_multiple_of(2) {
            collect_jolt_proof_statistics(&case, &mut even);
        } else {
            collect_jolt_proof_statistics(&case, &mut odd);
        }
    }

    let expected = samples as f64 / NUM_BUCKETS as f64;
    assert_uniformity(&all, expected);
    assert_same_distribution(&even, &odd);
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn with_zk_statistical_stack(test: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("zk-statistical-independence".to_string())
        .stack_size(128 * 1024 * 1024)
        .spawn(test)
        .expect("spawn ZK statistical-independence test")
        .join()
        .expect("ZK statistical-independence test panicked");
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn require_release_build() {
    assert!(
        !(cfg!(debug_assertions)
            && std::env::var_os("JOLT_VERIFIER_ALLOW_DEBUG_STAT_TESTS").is_none()),
        "run this statistical test with --release, or set JOLT_VERIFIER_ALLOW_DEBUG_STAT_TESTS=1"
    );
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn statistical_sample_count() -> usize {
    std::env::var("JOLT_VERIFIER_ZK_STAT_SAMPLES")
        .ok()
        .map_or(DEFAULT_SAMPLES, |value| {
            value
                .parse::<usize>()
                .expect("JOLT_VERIFIER_ZK_STAT_SAMPLES must be a positive integer")
        })
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[derive(Clone, Debug, PartialEq)]
struct StableZkProofShape {
    public_io: common::jolt_device::JoltDevice,
    preprocessing_digest: [u8; 32],
    trace_length: usize,
    ram_k: usize,
    rw_config: jolt_claims::protocols::jolt::JoltReadWriteConfig,
    one_hot_config: jolt_claims::protocols::jolt::JoltOneHotConfig,
    trace_polynomial_order: jolt_claims::protocols::jolt::TracePolynomialOrder,
    commitment_shape: CommitmentShape,
    dory_shape: DoryOpeningProofShape,
    narg_len: usize,
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
impl StableZkProofShape {
    fn from_case(case: &crate::support::verifier_fixtures::ZkVerifierFixtureCase) -> Self {
        Self {
            public_io: case.public_io.clone(),
            preprocessing_digest: case.preprocessing.preprocessing_digest,
            trace_length: case.proof.trace_length,
            ram_k: case.proof.ram_K,
            rw_config: case.proof.rw_config,
            one_hot_config: case.proof.one_hot_config,
            trace_polynomial_order: case.proof.trace_polynomial_order,
            commitment_shape: CommitmentShape::from_proof(&case.proof),
            dory_shape: DoryOpeningProofShape::from_proof(&case.proof.joint_opening_proof),
            narg_len: case.proof.narg.len(),
        }
    }
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[derive(Clone, Debug, PartialEq, Eq)]
struct CommitmentShape {
    proof_commitments: usize,
    has_untrusted_advice: bool,
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
impl CommitmentShape {
    fn from_proof(proof: &jolt_verifier::JoltProof<jolt_dory::DoryScheme>) -> Self {
        Self {
            proof_commitments: proof_commitments_from_narg(proof).len(),
            has_untrusted_advice: narg_frame_body(&proof.narg, 1)
                .is_some_and(|body| !body.is_empty()),
        }
    }
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[derive(Clone, Debug, PartialEq, Eq)]
struct DoryOpeningProofShape {
    first_messages: usize,
    second_messages: usize,
    nu: usize,
    sigma: usize,
    has_e2: bool,
    has_y_com: bool,
    has_sigma1: bool,
    has_sigma2: bool,
    has_scalar_product: bool,
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
impl DoryOpeningProofShape {
    fn from_proof(proof: &jolt_dory::DoryProof) -> Self {
        let proof = &proof.0;
        Self {
            first_messages: proof.first_messages.len(),
            second_messages: proof.second_messages.len(),
            nu: proof.nu,
            sigma: proof.sigma,
            has_e2: proof.e2.is_some(),
            has_y_com: proof.y_com.is_some(),
            has_sigma1: proof.sigma1_proof.is_some(),
            has_sigma2: proof.sigma2_proof.is_some(),
            has_scalar_product: proof.scalar_product_proof.is_some(),
        }
    }
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn collect_jolt_proof_statistics(
    case: &crate::support::verifier_fixtures::ZkVerifierFixtureCase,
    tracker: &mut BucketTracker,
) {
    let proof = &case.proof;

    let commitments = proof_commitments_from_narg(proof);
    tracker.record_append_positions("pcs.commitment", &commitments);
    let untrusted_advice = untrusted_advice_commitments_from_narg(proof);
    if let Some(commitment) = untrusted_advice.first() {
        tracker.record_append("pcs.commitment.untrusted_advice", commitment);
    }

    collect_narg_statistics(&proof.narg, tracker);
    collect_dory_opening_statistics(&proof.joint_opening_proof, tracker);
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn collect_narg_statistics(narg: &[u8], tracker: &mut BucketTracker) {
    tracker.record_bytes("proof.narg".to_string(), narg);
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn proof_commitments_from_narg(
    proof: &jolt_verifier::JoltProof<jolt_dory::DoryScheme>,
) -> Vec<DoryCommitment> {
    narg_frame_body(&proof.narg, 0)
        .map(|body| jolt_transcript::deserialize_slice(&body).expect("proof commitments decode"))
        .unwrap_or_default()
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn untrusted_advice_commitments_from_narg(
    proof: &jolt_verifier::JoltProof<jolt_dory::DoryScheme>,
) -> Vec<DoryCommitment> {
    narg_frame_body(&proof.narg, 1)
        .map(|body| {
            jolt_transcript::deserialize_slice(&body).expect("untrusted advice commitment decodes")
        })
        .unwrap_or_default()
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn narg_frame_body(narg: &[u8], frame_index: usize) -> Option<Vec<u8>> {
    let range = crate::support::narg_frame_ranges(narg)
        .get(frame_index)?
        .body
        .clone();
    Some(narg[range].to_vec())
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn collect_dory_opening_statistics(proof: &jolt_dory::DoryProof, tracker: &mut BucketTracker) {
    let proof = &proof.0;
    tracker.record_canonical("dory.vmv.c", &proof.vmv_message.c);
    tracker.record_canonical("dory.vmv.d2", &proof.vmv_message.d2);
    tracker.record_canonical("dory.vmv.e1", &proof.vmv_message.e1);
    if let Some(e2) = &proof.e2 {
        tracker.record_canonical("dory.vmv.e2", e2);
    }
    if let Some(y_com) = &proof.y_com {
        tracker.record_canonical("dory.vmv.y_com", y_com);
    }
    for index in selected_positions(proof.first_messages.len()) {
        let message = &proof.first_messages[index];
        let prefix = format!("dory.first.{index}");
        tracker.record_canonical(format!("{prefix}.d1_left"), &message.d1_left);
        tracker.record_canonical(format!("{prefix}.d1_right"), &message.d1_right);
        tracker.record_canonical(format!("{prefix}.d2_left"), &message.d2_left);
        tracker.record_canonical(format!("{prefix}.d2_right"), &message.d2_right);
    }
    for index in selected_positions(proof.second_messages.len()) {
        let message = &proof.second_messages[index];
        let prefix = format!("dory.second.{index}");
        tracker.record_canonical(format!("{prefix}.c_plus"), &message.c_plus);
        tracker.record_canonical(format!("{prefix}.c_minus"), &message.c_minus);
        tracker.record_canonical(format!("{prefix}.e1_plus"), &message.e1_plus);
        tracker.record_canonical(format!("{prefix}.e1_minus"), &message.e1_minus);
        tracker.record_canonical(format!("{prefix}.e2_plus"), &message.e2_plus);
        tracker.record_canonical(format!("{prefix}.e2_minus"), &message.e2_minus);
    }
    tracker.record_canonical("dory.final.e1", &proof.final_message.e1);
    tracker.record_canonical("dory.final.e2", &proof.final_message.e2);

    if let Some(sigma1) = &proof.sigma1_proof {
        tracker.record_canonical("dory.sigma1.a1", &sigma1.a1);
        tracker.record_canonical("dory.sigma1.a2", &sigma1.a2);
        tracker.record_canonical("dory.sigma1.z1", &sigma1.z1);
        tracker.record_canonical("dory.sigma1.z2", &sigma1.z2);
        tracker.record_canonical("dory.sigma1.z3", &sigma1.z3);
    }
    if let Some(sigma2) = &proof.sigma2_proof {
        tracker.record_canonical("dory.sigma2.a", &sigma2.a);
        tracker.record_canonical("dory.sigma2.z1", &sigma2.z1);
        tracker.record_canonical("dory.sigma2.z2", &sigma2.z2);
    }
    if let Some(scalar_product) = &proof.scalar_product_proof {
        tracker.record_canonical("dory.scalar_product.p1", &scalar_product.p1);
        tracker.record_canonical("dory.scalar_product.p2", &scalar_product.p2);
        tracker.record_canonical("dory.scalar_product.q", &scalar_product.q);
        tracker.record_canonical("dory.scalar_product.r", &scalar_product.r);
        tracker.record_canonical("dory.scalar_product.e1", &scalar_product.e1);
        tracker.record_canonical("dory.scalar_product.e2", &scalar_product.e2);
        tracker.record_canonical("dory.scalar_product.r1", &scalar_product.r1);
        tracker.record_canonical("dory.scalar_product.r2", &scalar_product.r2);
        tracker.record_canonical("dory.scalar_product.r3", &scalar_product.r3);
    }
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[derive(Clone, Debug, Default)]
struct BucketTracker {
    buckets: BTreeMap<String, Vec<usize>>,
    samples: BTreeMap<String, Vec<u64>>,
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
impl BucketTracker {
    fn new() -> Self {
        Self::default()
    }

    fn record_append<A: CanonicalSerialize>(&mut self, name: impl Into<String>, value: &A) {
        let name = name.into();
        let mut transcript = prover_transcript(b"jolt-zk-stat", [0u8; 32], Blake2b512::default());
        transcript.absorb_bytes(name.as_bytes());
        transcript.absorb(value);
        self.record_projected(
            name,
            field_low_u64(FsChallenge::<Fr>::challenge(&mut transcript)),
        );
    }

    fn record_append_positions<A: CanonicalSerialize>(&mut self, prefix: &str, values: &[A]) {
        for index in selected_positions(values.len()) {
            self.record_append(format!("{prefix}.{index}"), &values[index]);
        }
    }

    fn record_canonical<A: CanonicalSerialize>(&mut self, name: impl Into<String>, value: &A) {
        let name = name.into();
        let mut bytes = Vec::new();
        value
            .serialize_compressed(&mut bytes)
            .expect("canonical serialization succeeds");
        self.record_bytes(name, &bytes);
    }

    fn record_canonical_positions<A: CanonicalSerialize>(&mut self, prefix: &str, values: &[A]) {
        for index in selected_positions(values.len()) {
            self.record_canonical(format!("{prefix}.{index}"), &values[index]);
        }
    }

    fn record_bytes(&mut self, name: String, bytes: &[u8]) {
        let mut transcript = prover_transcript(b"jolt-zk-stat", [0u8; 32], Blake2b512::default());
        transcript.absorb_bytes(name.as_bytes());
        transcript.absorb_bytes(bytes);
        self.record_projected(
            name,
            field_low_u64(FsChallenge::<Fr>::challenge(&mut transcript)),
        );
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

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
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

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn field_low_u64(value: Fr) -> u64 {
    let bytes = value.to_bytes_array();
    u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn assert_uniformity(tracker: &BucketTracker, expected: f64) {
    let mut failures = Vec::new();
    for name in tracker.names() {
        let unique = unique_sample_count(&tracker.samples[&name]);
        let minimum_unique = tracker.samples[&name].len() * 95 / 100;
        if unique < minimum_unique {
            failures.push(format!(
                "{name}: only {unique} unique projected samples out of {}",
                tracker.samples[&name].len()
            ));
            continue;
        }

        let chi2 = tracker.chi_squared(&name, expected);
        if chi2 >= CHI2_CRITICAL {
            failures.push(format!("{name}: chi2={chi2:.2} >= {CHI2_CRITICAL:.2}"));
        }
    }

    assert!(
        failures.is_empty(),
        "ZK statistical uniformity check failed for {} projected components:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn assert_same_distribution(lhs: &BucketTracker, rhs: &BucketTracker) {
    let mut failures = Vec::new();
    for name in lhs.names() {
        let Some(rhs_buckets) = rhs.buckets.get(&name) else {
            continue;
        };
        let chi2 = two_sample_chi_squared(&lhs.buckets[&name], rhs_buckets);
        if chi2 >= CHI2_CRITICAL {
            failures.push(format!(
                "{name}: split-half chi2={chi2:.2} >= {CHI2_CRITICAL:.2}"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "ZK statistical split-half check failed for {} projected components:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn unique_sample_count(values: &[u64]) -> usize {
    let mut values = values.to_vec();
    values.sort_unstable();
    values.dedup();
    values.len()
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
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

#[cfg(any(not(feature = "prover-fixtures"), not(feature = "zk")))]
#[test]
#[ignore = "enable --features prover-fixtures,zk and run with --release to generate fresh ZK proof samples"]
fn zk_muldiv_jolt_proof_components_are_statistically_independent() {}
