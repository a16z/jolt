#![cfg_attr(
    all(feature = "prover-fixtures", feature = "zk"),
    expect(
        clippy::cast_precision_loss,
        clippy::expect_used,
        clippy::panic,
        reason = "statistical tests compute empirical floating-point test statistics and fail loudly"
    )
)]

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use std::collections::BTreeMap;

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use ark_serialize::CanonicalSerialize;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_field::{FixedBytes, Fr};
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_sumcheck::SumcheckProof;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_transcript::{
    AppendToTranscript, LegacyBlake2bTranscript as Blake2bTranscript, Transcript,
};
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_verifier::JoltProofClaims;

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
    public_io: jolt_common::jolt_device::JoltDevice,
    preprocessing_digest: [u8; 32],
    trace_length: usize,
    ram_k: usize,
    rw_config: jolt_claims::protocols::jolt::JoltReadWriteConfig,
    one_hot_config: jolt_claims::protocols::jolt::JoltOneHotConfig,
    trace_polynomial_order: jolt_claims::protocols::jolt::TracePolynomialOrder,
    commitment_shape: CommitmentShape,
    stage_shapes: Vec<CommittedStageShape>,
    dory_shape: DoryOpeningProofShape,
    blindfold_shape: BlindFoldProofShape,
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
impl StableZkProofShape {
    fn from_case(case: &crate::support::verifier_fixtures::ZkVerifierFixtureCase) -> Self {
        let JoltProofClaims::Zk { blindfold_proof } = &case.proof.claims else {
            panic!("ZK statistical fixture must carry a BlindFold proof");
        };

        Self {
            public_io: case.public_io.clone(),
            preprocessing_digest: case.preprocessing.preprocessing_digest,
            trace_length: case.proof.trace_length,
            ram_k: case.proof.ram_K,
            rw_config: case.proof.rw_config,
            one_hot_config: case.proof.one_hot_config,
            trace_polynomial_order: case.proof.trace_polynomial_order,
            commitment_shape: CommitmentShape {
                instruction_ra: case.proof.commitments.instruction_ra.len(),
                ram_ra: case.proof.commitments.ram_ra.len(),
                bytecode_ra: case.proof.commitments.bytecode_ra.len(),
                has_untrusted_advice: case.proof.untrusted_advice_commitment.is_some(),
            },
            stage_shapes: committed_stage_shapes(&case.proof.stages),
            dory_shape: DoryOpeningProofShape::from_proof(&case.proof.joint_opening_proof),
            blindfold_shape: BlindFoldProofShape::from_proof(blindfold_proof),
        }
    }
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[derive(Clone, Debug, PartialEq, Eq)]
struct CommitmentShape {
    instruction_ra: usize,
    ram_ra: usize,
    bytecode_ra: usize,
    has_untrusted_advice: bool,
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[derive(Clone, Debug, PartialEq, Eq)]
struct CommittedStageShape {
    rounds: usize,
    degrees: Vec<usize>,
    output_claim_rows: usize,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BlindFoldProofShape {
    auxiliary_rows: usize,
    random_round_commitment_rows: usize,
    random_output_claim_rows: usize,
    random_auxiliary_rows: usize,
    random_error_rows: usize,
    random_eval_commitments: usize,
    cross_term_error_rows: usize,
    folded_eval_output_openings: usize,
    folded_eval_blinding_openings: usize,
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
impl BlindFoldProofShape {
    fn from_proof(proof: &jolt_blindfold::BlindFoldProof<Fr, jolt_crypto::Bn254G1>) -> Self {
        Self {
            auxiliary_rows: proof.auxiliary_row_commitments.len(),
            random_round_commitment_rows: proof.random_round_commitments.len(),
            random_output_claim_rows: proof.random_output_claim_row_commitments.len(),
            random_auxiliary_rows: proof.random_auxiliary_row_commitments.len(),
            random_error_rows: proof.random_error_row_commitments.len(),
            random_eval_commitments: proof.random_eval_commitments.len(),
            cross_term_error_rows: proof.cross_term_error_row_commitments.len(),
            folded_eval_output_openings: proof.folded_eval_output_openings.len(),
            folded_eval_blinding_openings: proof.folded_eval_blinding_openings.len(),
        }
    }
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
fn committed_stage_shapes(
    stages: &jolt_verifier::proof::JoltStageProofs<Fr, jolt_crypto::Pedersen<jolt_crypto::Bn254G1>>,
) -> Vec<CommittedStageShape> {
    [
        &stages.stage1_uni_skip_first_round_proof,
        &stages.stage1_sumcheck_proof,
        &stages.stage2_uni_skip_first_round_proof,
        &stages.stage2_sumcheck_proof,
        &stages.stage3_sumcheck_proof,
        &stages.stage4_sumcheck_proof,
        &stages.stage5_sumcheck_proof,
        &stages.stage6a_sumcheck_proof,
        &stages.stage6b_sumcheck_proof,
        &stages.stage7_sumcheck_proof,
    ]
    .into_iter()
    .map(committed_stage_shape)
    .collect()
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn committed_stage_shape<C>(proof: &SumcheckProof<Fr, C>) -> CommittedStageShape {
    let proof = proof
        .as_committed()
        .expect("ZK statistical fixture must use committed sumcheck proofs");
    CommittedStageShape {
        rounds: proof.rounds.len(),
        degrees: proof.rounds.iter().map(|round| round.degree).collect(),
        output_claim_rows: proof.output_claims.commitments.len(),
    }
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn collect_jolt_proof_statistics(
    case: &crate::support::verifier_fixtures::ZkVerifierFixtureCase,
    tracker: &mut BucketTracker,
) {
    let proof = &case.proof;
    let JoltProofClaims::Zk { blindfold_proof } = &proof.claims else {
        panic!("ZK statistical fixture must carry a BlindFold proof");
    };

    tracker.record_append("pcs.commitment.rd_inc", &proof.commitments.rd_inc);
    tracker.record_append("pcs.commitment.ram_inc", &proof.commitments.ram_inc);
    tracker.record_append_positions(
        "pcs.commitment.instruction_ra",
        &proof.commitments.instruction_ra,
    );
    tracker.record_append_positions("pcs.commitment.ram_ra", &proof.commitments.ram_ra);
    tracker.record_append_positions("pcs.commitment.bytecode_ra", &proof.commitments.bytecode_ra);
    if let Some(commitment) = &proof.untrusted_advice_commitment {
        tracker.record_append("pcs.commitment.untrusted_advice", commitment);
    }

    collect_sumcheck_statistics(
        "sumcheck.stage1_uniskip",
        &proof.stages.stage1_uni_skip_first_round_proof,
        tracker,
    );
    collect_sumcheck_statistics(
        "sumcheck.stage1_batch",
        &proof.stages.stage1_sumcheck_proof,
        tracker,
    );
    collect_sumcheck_statistics(
        "sumcheck.stage2_uniskip",
        &proof.stages.stage2_uni_skip_first_round_proof,
        tracker,
    );
    collect_sumcheck_statistics(
        "sumcheck.stage2_batch",
        &proof.stages.stage2_sumcheck_proof,
        tracker,
    );
    collect_sumcheck_statistics(
        "sumcheck.stage3_batch",
        &proof.stages.stage3_sumcheck_proof,
        tracker,
    );
    collect_sumcheck_statistics(
        "sumcheck.stage4_batch",
        &proof.stages.stage4_sumcheck_proof,
        tracker,
    );
    collect_sumcheck_statistics(
        "sumcheck.stage5_batch",
        &proof.stages.stage5_sumcheck_proof,
        tracker,
    );
    collect_sumcheck_statistics(
        "sumcheck.stage6_address_phase",
        &proof.stages.stage6a_sumcheck_proof,
        tracker,
    );
    collect_sumcheck_statistics(
        "sumcheck.stage6_cycle_phase",
        &proof.stages.stage6b_sumcheck_proof,
        tracker,
    );
    collect_sumcheck_statistics(
        "sumcheck.stage7_batch",
        &proof.stages.stage7_sumcheck_proof,
        tracker,
    );

    collect_dory_opening_statistics(&proof.joint_opening_proof, tracker);
    collect_blindfold_statistics(blindfold_proof, tracker);
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn collect_sumcheck_statistics<C>(
    prefix: &'static str,
    proof: &SumcheckProof<Fr, C>,
    tracker: &mut BucketTracker,
) where
    C: AppendToTranscript,
{
    let proof = proof
        .as_committed()
        .expect("ZK statistical fixture must use committed sumcheck proofs");
    for index in selected_positions(proof.rounds.len()) {
        tracker.record_append(
            format!("{prefix}.round.{index}"),
            &proof.rounds[index].commitment,
        );
    }
    tracker.record_append_positions(
        &format!("{prefix}.output_claim"),
        &proof.output_claims.commitments,
    );
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
    // A ZK proof carries no clear final message (it would reveal the folded
    // witness); the hiding scalar-product Σ-proof recorded below replaces it.
    assert!(
        proof.final_message.is_none(),
        "ZK proof must not carry a clear final message"
    );

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
fn collect_blindfold_statistics(
    proof: &jolt_blindfold::BlindFoldProof<Fr, jolt_crypto::Bn254G1>,
    tracker: &mut BucketTracker,
) {
    tracker.record_canonical("blindfold.random.u", &proof.random_u);
    tracker.record_append_positions(
        "blindfold.random.round_commitment",
        &proof.random_round_commitments,
    );
    tracker.record_append_positions(
        "blindfold.random.output_claim_commitment",
        &proof.random_output_claim_row_commitments,
    );
    tracker.record_append_positions(
        "blindfold.random.auxiliary_commitment",
        &proof.random_auxiliary_row_commitments,
    );
    tracker.record_append_positions(
        "blindfold.random.error_commitment",
        &proof.random_error_row_commitments,
    );
    tracker.record_append_positions(
        "blindfold.random.eval_commitment",
        &proof.random_eval_commitments,
    );
    tracker.record_append_positions(
        "blindfold.real.auxiliary_commitment",
        &proof.auxiliary_row_commitments,
    );
    tracker.record_append_positions(
        "blindfold.cross_term_commitment",
        &proof.cross_term_error_row_commitments,
    );
    tracker.record_canonical("blindfold.az_rx", &proof.az_rx);
    tracker.record_canonical("blindfold.bz_rx", &proof.bz_rx);
    tracker.record_canonical("blindfold.cz_rx", &proof.cz_rx);
    tracker.record_vector_opening("blindfold.witness_opening", &proof.witness_opening);
    tracker.record_vector_opening("blindfold.error_opening", &proof.error_opening);
    tracker.record_canonical_positions("blindfold.folded_eval.output", &proof.folded_eval_outputs);
    tracker.record_canonical_positions(
        "blindfold.folded_eval.blinding",
        &proof.folded_eval_blindings,
    );
    // These openings are fixed-coordinate checks; the rows intentionally contain
    // structural zero slots, so the hiding component to sample is the opening
    // blinding.
    for index in selected_positions(proof.folded_eval_output_openings.len()) {
        tracker.record_vector_opening_blinding(
            &format!("blindfold.folded_eval.output_opening.{index}"),
            &proof.folded_eval_output_openings[index],
        );
    }
    for index in selected_positions(proof.folded_eval_blinding_openings.len()) {
        tracker.record_vector_opening_blinding(
            &format!("blindfold.folded_eval.blinding_opening.{index}"),
            &proof.folded_eval_blinding_openings[index],
        );
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

    fn record_append<A: AppendToTranscript>(&mut self, name: impl Into<String>, value: &A) {
        let name = name.into();
        let mut transcript = Blake2bTranscript::<Fr>::new(b"jolt-zk-stat");
        transcript.append_bytes(name.as_bytes());
        value.append_to_transcript(&mut transcript);
        self.record_projected(name, field_low_u64(transcript.challenge()));
    }

    fn record_append_positions<A: AppendToTranscript>(&mut self, prefix: &str, values: &[A]) {
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

    fn record_vector_opening<F>(
        &mut self,
        prefix: &str,
        opening: &jolt_crypto::VectorCommitmentOpening<F>,
    ) where
        F: CanonicalSerialize,
    {
        self.record_canonical_positions(&format!("{prefix}.row"), &opening.combined_vector);
        self.record_canonical(format!("{prefix}.blinding"), &opening.combined_blinding);
    }

    fn record_vector_opening_blinding<F>(
        &mut self,
        prefix: &str,
        opening: &jolt_crypto::VectorCommitmentOpening<F>,
    ) where
        F: CanonicalSerialize,
    {
        self.record_canonical(format!("{prefix}.blinding"), &opening.combined_blinding);
    }

    fn record_bytes(&mut self, name: String, bytes: &[u8]) {
        let mut transcript = Blake2bTranscript::<Fr>::new(b"jolt-zk-stat");
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
