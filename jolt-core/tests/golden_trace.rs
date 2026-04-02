//! Golden trace infrastructure for bit-for-bit stage comparison.
//!
//! # Architecture
//!
//! 1. `generate_golden_fixture` runs muldiv through jolt-core's prover, serializes
//!    the proof + verifier preprocessing + program I/O to `test_fixtures/`.
//!
//! 2. `load_verifier` loads the serialized data and constructs a `JoltVerifier`.
//!
//! 3. Per-stage replay: call `verifier.verify_stage1()`, `verify_stage2()`, etc.
//!    Each advances the transcript + opening accumulator. After replaying through
//!    stage N-1, the verifier's transcript state is exactly what stage N expects.
//!
//! 4. Comparison: extract batching coefficients, round polys, challenges from the
//!    golden proof and compare against the Module runtime's output.
#![allow(non_snake_case)]

use std::path::PathBuf;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_core::field::JoltField;
use jolt_core::poly::commitment::dory::DoryGlobals;
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::zkvm::verifier::{JoltSharedPreprocessing, JoltVerifierPreprocessing};
use jolt_core::zkvm::{RV64IMACProof, RV64IMACVerifier, Serializable};
use tracer::JoltDevice;

type Fr = ark_bn254::Fr;
type Challenge = <Fr as JoltField>::Challenge;

type VerifierPreprocessing = JoltVerifierPreprocessing<
    ark_bn254::Fr,
    jolt_core::curve::Bn254Curve,
    jolt_core::poly::commitment::dory::DoryCommitmentScheme,
>;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_fixtures")
}

fn proof_path() -> PathBuf {
    fixture_dir().join("muldiv_proof.bin")
}

fn preprocessing_path() -> PathBuf {
    fixture_dir().join("muldiv_preprocessing.bin")
}

fn io_path() -> PathBuf {
    fixture_dir().join("muldiv_io.bin")
}

fn fixture_exists() -> bool {
    proof_path().exists() && preprocessing_path().exists() && io_path().exists()
}

/// Loads the golden fixture and constructs a JoltVerifier with preamble already run.
/// The verifier owns the proof — access proof fields via `verifier.proof`.
fn load_verifier() -> RV64IMACVerifier<'static> {
    assert!(
        fixture_exists(),
        "Golden fixture not found. Run: cargo nextest run -p jolt-core --test golden_trace \
         generate_golden_fixture --features host --cargo-quiet"
    );
    DoryGlobals::reset();

    let proof = RV64IMACProof::from_file(proof_path()).expect("load proof");
    let preprocessing: &'static VerifierPreprocessing = Box::leak(Box::new(
        VerifierPreprocessing::from_file(preprocessing_path()).expect("load preprocessing"),
    ));
    let program_io = JoltDevice::from_file(io_path()).expect("load io");

    let mut verifier = RV64IMACVerifier::new(preprocessing, proof, program_io, None, None)
        .expect("build verifier from golden fixture");
    verifier.run_preamble();
    verifier
}

// ============================================================================
// Golden fixture generation
// ============================================================================

/// Generates the golden fixture by running muldiv through jolt-core's prover.
/// Saves proof, verifier preprocessing, and program I/O to `test_fixtures/`.
///
/// Run with: cargo nextest run -p jolt-core --test golden_trace generate_golden_fixture --features host
#[test]
#[cfg(feature = "host")]
fn generate_golden_fixture() {
    use jolt_core::host;
    use jolt_core::zkvm::prover::JoltProverPreprocessing;

    DoryGlobals::reset();

    let mut program = host::Program::new("muldiv-guest");
    let (bytecode, init_memory_state, _, e_entry) = program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode.clone(),
        io_device.memory_layout.clone(),
        init_memory_state,
        1 << 16,
        e_entry,
    );

    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
    let elf_contents_opt = program.get_elf_contents();
    let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
    let prover = jolt_core::zkvm::RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        elf_contents,
        &inputs,
        &[],
        &[],
        None,
        None,
        None,
    );
    let io_device = prover.program_io.clone();
    let (jolt_proof, _debug_info) = prover.prove();

    let dir = fixture_dir();
    std::fs::create_dir_all(&dir).expect("create test_fixtures dir");

    jolt_proof.save_to_file(proof_path()).expect("save proof");

    let verifier_preprocessing = JoltVerifierPreprocessing::from(&prover_preprocessing);
    verifier_preprocessing
        .save_to_file(preprocessing_path())
        .expect("save preprocessing");

    io_device.save_to_file(io_path()).expect("save io");

    eprintln!("Golden fixture saved to {}", fixture_dir().display());
}

// ============================================================================
// Proof data extraction helpers
// ============================================================================

fn count_clear_rounds(
    proof: &SumcheckInstanceProof<
        ark_bn254::Fr,
        jolt_core::curve::Bn254Curve,
        jolt_core::transcripts::Blake2bTranscript,
    >,
) -> usize {
    match proof {
        SumcheckInstanceProof::Clear(p) => p.compressed_polys.len(),
        _ => panic!("expected ClearSumcheckProof (non-ZK mode)"),
    }
}

// ============================================================================
// Stage replay + comparison tests
// ============================================================================

/// Sanity check: golden fixture passes full verification.
/// Uses `verify()` directly (which runs its own preamble), not `load_verifier()`.
#[test]
fn golden_full_verify() {
    if !fixture_exists() {
        eprintln!("Skipping: golden fixture not generated yet");
        return;
    }
    DoryGlobals::reset();
    let proof = RV64IMACProof::from_file(proof_path()).expect("load proof");
    let preprocessing: &'static VerifierPreprocessing = Box::leak(Box::new(
        VerifierPreprocessing::from_file(preprocessing_path()).expect("load preprocessing"),
    ));
    let program_io = JoltDevice::from_file(io_path()).expect("load io");
    let verifier = RV64IMACVerifier::new(preprocessing, proof, program_io, None, None)
        .expect("build verifier");
    verifier.verify().expect("golden fixture must verify");
}

/// Replay stage 1: verify challenges match round count.
#[test]
fn golden_stage1_replay() {
    if !fixture_exists() {
        eprintln!("Skipping: golden fixture not generated yet");
        return;
    }
    let mut verifier = load_verifier();

    let (stage1_result, _uniskip_challenge) = verifier
        .verify_stage1()
        .expect("stage 1 verification failed");

    let n_challenges = stage1_result.challenges.len();
    let n_rounds = count_clear_rounds(&verifier.proof.stage1_sumcheck_proof);

    eprintln!("Stage 1: {n_challenges} challenges, {n_rounds} round polys");
    assert!(n_challenges > 0);
    assert_eq!(n_challenges, n_rounds);
}

/// Replay stages 1-2: verify stage 2 structure.
#[test]
fn golden_stage2_replay() {
    if !fixture_exists() {
        eprintln!("Skipping: golden fixture not generated yet");
        return;
    }
    let mut verifier = load_verifier();

    let _ = verifier
        .verify_stage1()
        .expect("stage 1 verification failed");
    let (stage2_result, _uniskip_challenge) = verifier
        .verify_stage2()
        .expect("stage 2 verification failed");

    let n_challenges = stage2_result.challenges.len();
    let n_rounds = count_clear_rounds(&verifier.proof.stage2_sumcheck_proof);

    eprintln!("Stage 2: {n_challenges} challenges, {n_rounds} round polys");
    assert!(n_challenges > 0);
    assert_eq!(n_challenges, n_rounds);
}

/// Replay all 8 stages sequentially.
#[test]
fn golden_full_stage_replay() {
    if !fixture_exists() {
        eprintln!("Skipping: golden fixture not generated yet");
        return;
    }
    let mut verifier = load_verifier();

    let _ = verifier.verify_stage1().expect("stage 1 failed");
    let _ = verifier.verify_stage2().expect("stage 2 failed");
    let _ = verifier.verify_stage3().expect("stage 3 failed");
    let _ = verifier.verify_stage4().expect("stage 4 failed");
    let _ = verifier.verify_stage5().expect("stage 5 failed");
    let _ = verifier.verify_stage6().expect("stage 6 failed");
    let _ = verifier.verify_stage7().expect("stage 7 failed");
    let _ = verifier.verify_stage8().expect("stage 8 failed");

    eprintln!("All 8 stages replayed successfully");
}

/// Detailed Stage 2 data extraction: round poly structure and challenges.
#[test]
fn golden_stage2_detailed() {
    if !fixture_exists() {
        eprintln!("Skipping: golden fixture not generated yet");
        return;
    }
    let mut verifier = load_verifier();

    let _ = verifier.verify_stage1().expect("stage 1 failed");
    let (stage2_result, _) = verifier.verify_stage2().expect("stage 2 failed");

    let n_challenges = stage2_result.challenges.len();
    eprintln!("Stage 2: {n_challenges} sumcheck challenges");

    // Log per-round degree (number of coefficients in compressed poly)
    match &verifier.proof.stage2_sumcheck_proof {
        SumcheckInstanceProof::Clear(clear) => {
            let mut degree_counts = std::collections::HashMap::new();
            for poly in &clear.compressed_polys {
                let degree = poly.coeffs_except_linear_term.len() + 1;
                *degree_counts.entry(degree).or_insert(0usize) += 1;
            }
            eprintln!("Stage 2 round degree distribution: {:?}", degree_counts);

            // Log first few round poly coefficients for fingerprinting
            for (i, poly) in clear.compressed_polys.iter().take(3).enumerate() {
                let coeffs = &poly.coeffs_except_linear_term;
                eprintln!(
                    "  round {i}: {} coeffs, first={:?}",
                    coeffs.len(),
                    coeffs.first()
                );
            }
        }
        _ => panic!("expected non-ZK proof"),
    }
}

/// Structural overview: challenge count + round poly count per stage.
#[test]
fn golden_all_stages_structure() {
    if !fixture_exists() {
        eprintln!("Skipping: golden fixture not generated yet");
        return;
    }
    let mut verifier = load_verifier();

    let (s1, _) = verifier.verify_stage1().expect("stage 1 failed");
    let n1 = count_clear_rounds(&verifier.proof.stage1_sumcheck_proof);
    eprintln!(
        "Stage 1: {} challenges, {} round polys",
        s1.challenges.len(),
        n1
    );

    let (s2, _) = verifier.verify_stage2().expect("stage 2 failed");
    let n2 = count_clear_rounds(&verifier.proof.stage2_sumcheck_proof);
    eprintln!(
        "Stage 2: {} challenges, {} round polys",
        s2.challenges.len(),
        n2
    );

    let s3 = verifier.verify_stage3().expect("stage 3 failed");
    let n3 = count_clear_rounds(&verifier.proof.stage3_sumcheck_proof);
    eprintln!(
        "Stage 3: {} challenges, {} round polys",
        s3.challenges.len(),
        n3
    );

    let s4 = verifier.verify_stage4().expect("stage 4 failed");
    let n4 = count_clear_rounds(&verifier.proof.stage4_sumcheck_proof);
    eprintln!(
        "Stage 4: {} challenges, {} round polys",
        s4.challenges.len(),
        n4
    );

    let s5 = verifier.verify_stage5().expect("stage 5 failed");
    let n5 = count_clear_rounds(&verifier.proof.stage5_sumcheck_proof);
    eprintln!(
        "Stage 5: {} challenges, {} round polys",
        s5.challenges.len(),
        n5
    );

    let s6 = verifier.verify_stage6().expect("stage 6 failed");
    let n6 = count_clear_rounds(&verifier.proof.stage6_sumcheck_proof);
    eprintln!(
        "Stage 6: {} challenges, {} round polys",
        s6.challenges.len(),
        n6
    );

    let s7 = verifier.verify_stage7().expect("stage 7 failed");
    let n7 = count_clear_rounds(&verifier.proof.stage7_sumcheck_proof);
    eprintln!(
        "Stage 7: {} challenges, {} round polys",
        s7.challenges.len(),
        n7
    );
}

// ============================================================================
// Golden data export — per-stage round poly coefficients + challenges
// ============================================================================

/// Per-stage golden data: decompressed round poly coefficients and challenges.
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct GoldenStageData {
    pub initial_claim: Fr,
    pub challenges: Vec<Challenge>,
    /// Full round polynomial coefficients (linear term recovered).
    /// round_poly_coeffs[i] = [c_0, c_1, ..., c_d] for round i.
    pub round_poly_coeffs: Vec<Vec<Fr>>,
}

/// All stages combined.
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct GoldenData {
    pub stages: Vec<GoldenStageData>,
}

impl GoldenData {
    fn path() -> PathBuf {
        fixture_dir().join("muldiv_golden_data.bin")
    }
}

impl Serializable for GoldenData {}

fn extract_clear_proof(
    proof: &SumcheckInstanceProof<
        Fr,
        jolt_core::curve::Bn254Curve,
        jolt_core::transcripts::Blake2bTranscript,
    >,
) -> &jolt_core::subprotocols::sumcheck::ClearSumcheckProof<
    Fr,
    jolt_core::transcripts::Blake2bTranscript,
> {
    match proof {
        SumcheckInstanceProof::Clear(p) => p,
        _ => panic!("expected ClearSumcheckProof (non-ZK mode)"),
    }
}

/// Export per-stage golden data: decompressed round polys + challenges.
///
/// Run with: cargo nextest run -p jolt-core --test golden_trace golden_export --features host
#[test]
fn golden_export_stage_data() {
    if !fixture_exists() {
        eprintln!("Skipping: golden fixture not generated yet");
        return;
    }
    let mut verifier = load_verifier();

    let mut stages = Vec::new();

    // Stage 1
    let (s1, _) = verifier.verify_stage1().expect("stage 1 failed");
    let clear1 = extract_clear_proof(&verifier.proof.stage1_sumcheck_proof);
    let coeffs1 = clear1.decompress_all_rounds(s1.initial_claim, &s1.challenges);
    eprintln!(
        "Stage 1: {} rounds, initial_claim={:?}",
        coeffs1.len(),
        s1.initial_claim
    );
    stages.push(GoldenStageData {
        initial_claim: s1.initial_claim,
        challenges: s1.challenges,
        round_poly_coeffs: coeffs1,
    });

    // Stage 2
    let (s2, _) = verifier.verify_stage2().expect("stage 2 failed");
    let clear2 = extract_clear_proof(&verifier.proof.stage2_sumcheck_proof);
    let coeffs2 = clear2.decompress_all_rounds(s2.initial_claim, &s2.challenges);
    eprintln!(
        "Stage 2: {} rounds, initial_claim={:?}",
        coeffs2.len(),
        s2.initial_claim
    );
    stages.push(GoldenStageData {
        initial_claim: s2.initial_claim,
        challenges: s2.challenges,
        round_poly_coeffs: coeffs2,
    });

    // Stage 3
    let s3 = verifier.verify_stage3().expect("stage 3 failed");
    let clear3 = extract_clear_proof(&verifier.proof.stage3_sumcheck_proof);
    let coeffs3 = clear3.decompress_all_rounds(s3.initial_claim, &s3.challenges);
    eprintln!("Stage 3: {} rounds", coeffs3.len());
    stages.push(GoldenStageData {
        initial_claim: s3.initial_claim,
        challenges: s3.challenges,
        round_poly_coeffs: coeffs3,
    });

    // Stage 4
    let s4 = verifier.verify_stage4().expect("stage 4 failed");
    let clear4 = extract_clear_proof(&verifier.proof.stage4_sumcheck_proof);
    let coeffs4 = clear4.decompress_all_rounds(s4.initial_claim, &s4.challenges);
    eprintln!("Stage 4: {} rounds", coeffs4.len());
    stages.push(GoldenStageData {
        initial_claim: s4.initial_claim,
        challenges: s4.challenges,
        round_poly_coeffs: coeffs4,
    });

    // Stage 5
    let s5 = verifier.verify_stage5().expect("stage 5 failed");
    let clear5 = extract_clear_proof(&verifier.proof.stage5_sumcheck_proof);
    let coeffs5 = clear5.decompress_all_rounds(s5.initial_claim, &s5.challenges);
    eprintln!("Stage 5: {} rounds", coeffs5.len());
    stages.push(GoldenStageData {
        initial_claim: s5.initial_claim,
        challenges: s5.challenges,
        round_poly_coeffs: coeffs5,
    });

    // Stage 6
    let s6 = verifier.verify_stage6().expect("stage 6 failed");
    let clear6 = extract_clear_proof(&verifier.proof.stage6_sumcheck_proof);
    let coeffs6 = clear6.decompress_all_rounds(s6.initial_claim, &s6.challenges);
    eprintln!("Stage 6: {} rounds", coeffs6.len());
    stages.push(GoldenStageData {
        initial_claim: s6.initial_claim,
        challenges: s6.challenges,
        round_poly_coeffs: coeffs6,
    });

    // Stage 7
    let s7 = verifier.verify_stage7().expect("stage 7 failed");
    let clear7 = extract_clear_proof(&verifier.proof.stage7_sumcheck_proof);
    let coeffs7 = clear7.decompress_all_rounds(s7.initial_claim, &s7.challenges);
    eprintln!("Stage 7: {} rounds", coeffs7.len());
    stages.push(GoldenStageData {
        initial_claim: s7.initial_claim,
        challenges: s7.challenges,
        round_poly_coeffs: coeffs7,
    });

    let golden = GoldenData { stages };
    golden
        .save_to_file(GoldenData::path())
        .expect("save golden data");

    eprintln!("Golden data exported to {}", GoldenData::path().display());
}

/// Verify golden data can be round-tripped: load, decompress, re-verify.
#[test]
fn golden_data_roundtrip() {
    if !GoldenData::path().exists() {
        eprintln!("Skipping: golden data not exported yet");
        return;
    }
    let golden = GoldenData::from_file(GoldenData::path()).expect("load golden data");
    assert_eq!(golden.stages.len(), 7, "expected 7 stages");
    for (i, stage) in golden.stages.iter().enumerate() {
        assert_eq!(
            stage.challenges.len(),
            stage.round_poly_coeffs.len(),
            "stage {} challenge/round mismatch",
            i + 1
        );
        eprintln!(
            "Stage {}: {} rounds, {} coeffs in round 0",
            i + 1,
            stage.round_poly_coeffs.len(),
            stage.round_poly_coeffs.first().map_or(0, |v| v.len()),
        );
    }
}
