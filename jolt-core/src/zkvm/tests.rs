//! End-to-end test infrastructure for Jolt ZKVM.
//!
//! This module provides a unified test runner that reduces boilerplate across e2e tests.
//! Tests can be configured via `E2ETestConfig` to vary:
//! - Program (fibonacci, sha2, etc.)
//! - ProgramMode (Full vs Committed)
//! - DoryLayout (CycleMajor vs AddressMajor)
//! - Trace size
//! - Advice (trusted/untrusted)

use std::sync::Arc;

use ark_bn254::Fr;
use serial_test::serial;

use crate::host;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::dory::{DoryCommitmentScheme, DoryContext, DoryGlobals, DoryLayout};
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::opening_proof::{OpeningAccumulator, SumcheckId};
use crate::zkvm::bytecode::chunks::total_lanes;
use crate::zkvm::claim_reductions::AdviceKind;
use crate::zkvm::config::ProgramMode;
use crate::zkvm::program::ProgramPreprocessing;
use crate::zkvm::prover::JoltProverPreprocessing;
use crate::zkvm::ram::populate_memory_states;
use crate::zkvm::verifier::{JoltSharedPreprocessing, JoltVerifier, JoltVerifierPreprocessing};
use crate::zkvm::witness::CommittedPolynomial;
use crate::zkvm::{RV64IMACProver, RV64IMACVerifier};

/// Configuration for an end-to-end test.
#[derive(Clone)]
pub struct E2ETestConfig {
    /// Guest program name (e.g., "fibonacci-guest", "sha2-guest")
    pub program_name: &'static str,
    /// Serialized inputs to pass to the guest
    pub inputs: Vec<u8>,
    /// Maximum padded trace length (must be power of 2)
    pub max_trace_length: usize,
    /// Whether to use Committed program mode (vs Full)
    pub committed_program: bool,
    /// Dory layout override (None = use default CycleMajor)
    pub dory_layout: Option<DoryLayout>,
    /// Trusted advice bytes
    pub trusted_advice: Vec<u8>,
    /// Untrusted advice bytes
    pub untrusted_advice: Vec<u8>,
    /// Expected output bytes (None = don't verify output)
    pub expected_output: Option<Vec<u8>>,
}

impl Default for E2ETestConfig {
    fn default() -> Self {
        Self {
            program_name: "fibonacci-guest",
            inputs: postcard::to_stdvec(&100u32).unwrap(),
            max_trace_length: 1 << 16,
            committed_program: false,
            dory_layout: None,
            trusted_advice: vec![],
            untrusted_advice: vec![],
            expected_output: None,
        }
    }
}

impl E2ETestConfig {
    // ========================================================================
    // Program Constructors
    // ========================================================================

    /// Create config for fibonacci with custom input.
    pub fn fibonacci(n: u32) -> Self {
        Self {
            inputs: postcard::to_stdvec(&n).unwrap(),
            ..Default::default()
        }
    }

    /// Create config for sha2 (with default 32-byte input).
    pub fn sha2() -> Self {
        Self {
            program_name: "sha2-guest",
            inputs: postcard::to_stdvec(&[5u8; 32]).unwrap(),
            expected_output: Some(vec![
                0x28, 0x9b, 0xdf, 0x82, 0x9b, 0x4a, 0x30, 0x26, 0x7, 0x9a, 0x3e, 0xa0, 0x89, 0x73,
                0xb1, 0x97, 0x2d, 0x12, 0x4e, 0x7e, 0xaf, 0x22, 0x33, 0xc6, 0x3, 0x14, 0x3d, 0xc6,
                0x3b, 0x50, 0xd2, 0x57,
            ]),
            ..Default::default()
        }
    }

    /// Create config for sha3 (with default 32-byte input).
    pub fn sha3() -> Self {
        Self {
            program_name: "sha3-guest",
            inputs: postcard::to_stdvec(&[5u8; 32]).unwrap(),
            expected_output: Some(vec![
                0xd0, 0x3, 0x5c, 0x96, 0x86, 0x6e, 0xe2, 0x2e, 0x81, 0xf5, 0xc4, 0xef, 0xbd, 0x88,
                0x33, 0xc1, 0x7e, 0xa1, 0x61, 0x10, 0x81, 0xfc, 0xd7, 0xa3, 0xdd, 0xce, 0xce, 0x7f,
                0x44, 0x72, 0x4, 0x66,
            ]),
            ..Default::default()
        }
    }

    /// Create config for merkle-tree guest.
    /// Default: 4 leaves with input=[5;32], trusted=[6;32,7;32], untrusted=[8;32]
    pub fn merkle_tree() -> Self {
        let inputs = postcard::to_stdvec(&[5u8; 32].as_slice()).unwrap();
        let untrusted_advice = postcard::to_stdvec(&[8u8; 32]).unwrap();
        let mut trusted_advice = postcard::to_stdvec(&[6u8; 32]).unwrap();
        trusted_advice.extend(postcard::to_stdvec(&[7u8; 32]).unwrap());

        Self {
            program_name: "merkle-tree-guest",
            inputs,
            trusted_advice,
            untrusted_advice,
            expected_output: Some(vec![
                0xb4, 0x37, 0x0f, 0x3a, 0xb, 0x3d, 0x38, 0xa8, 0x7a, 0x6c, 0x4c, 0x46, 0x9, 0xe7,
                0x83, 0xb3, 0xcc, 0xb7, 0x1c, 0x30, 0x1f, 0xf8, 0x54, 0xd, 0xf7, 0xdd, 0xc8, 0x42,
                0x32, 0xbb, 0x16, 0xd7,
            ]),
            ..Default::default()
        }
    }

    /// Create config for memory-ops guest (no inputs).
    pub fn memory_ops() -> Self {
        Self {
            program_name: "memory-ops-guest",
            inputs: vec![],
            ..Default::default()
        }
    }

    /// Create config for btreemap guest.
    pub fn btreemap(n: u32) -> Self {
        Self {
            program_name: "btreemap-guest",
            inputs: postcard::to_stdvec(&n).unwrap(),
            ..Default::default()
        }
    }

    /// Create config for muldiv guest.
    pub fn muldiv(a: u32, b: u32, c: u32) -> Self {
        Self {
            program_name: "muldiv-guest",
            inputs: postcard::to_stdvec(&[a, b, c]).unwrap(),
            ..Default::default()
        }
    }

    // ========================================================================
    // Builder Methods
    // ========================================================================

    /// Set committed program mode.
    pub fn with_committed_program(mut self) -> Self {
        self.committed_program = true;
        self
    }

    /// Set Dory layout.
    pub fn with_dory_layout(mut self, layout: DoryLayout) -> Self {
        self.dory_layout = Some(layout);
        self
    }

    /// Set small trace (256 cycles).
    pub fn with_small_trace(mut self) -> Self {
        self.max_trace_length = 256;
        self
    }

    /// Set custom max trace length.
    #[allow(dead_code)] // API for future tests
    pub fn with_max_trace_length(mut self, len: usize) -> Self {
        self.max_trace_length = len;
        self
    }

    /// Set trusted advice bytes.
    pub fn with_trusted_advice(mut self, advice: Vec<u8>) -> Self {
        self.trusted_advice = advice;
        self
    }

    /// Set untrusted advice bytes.
    pub fn with_untrusted_advice(mut self, advice: Vec<u8>) -> Self {
        self.untrusted_advice = advice;
        self
    }

    /// Set expected output for verification.
    #[allow(dead_code)] // API for future tests
    pub fn expecting_output(mut self, output: Vec<u8>) -> Self {
        self.expected_output = Some(output);
        self
    }

    /// Clear expected output (don't verify).
    #[allow(dead_code)] // API for future tests
    pub fn without_output_check(mut self) -> Self {
        self.expected_output = None;
        self
    }
}

/// Run an end-to-end test with the given configuration.
///
/// This handles all axes of variation:
/// - Program selection
/// - Bytecode mode (Full vs Committed)
/// - Dory layout (CycleMajor vs AddressMajor)
/// - Trusted/untrusted advice (computes commitment if non-empty)
/// - Maximum padded trace length
pub fn run_e2e_test(config: E2ETestConfig) {
    // Setup Dory globals
    DoryGlobals::reset();
    if let Some(layout) = config.dory_layout {
        DoryGlobals::set_layout(layout);
    }

    // Decode and trace program
    let mut program = host::Program::new(config.program_name);
    let (instructions, init_memory_state, _) = program.decode();
    let (_, _, _, io_device) = program.trace(
        &config.inputs,
        &config.untrusted_advice,
        &config.trusted_advice,
    );

    // Preprocess bytecode and program image
    let program_data = Arc::new(ProgramPreprocessing::preprocess(
        instructions,
        init_memory_state,
    ));
    let shared_preprocessing = JoltSharedPreprocessing::new(
        program_data.meta(),
        io_device.memory_layout.clone(),
        config.max_trace_length,
    );

    // Create prover preprocessing (mode-dependent)
    let prover_preprocessing = if config.committed_program {
        JoltProverPreprocessing::new_committed(
            shared_preprocessing.clone(),
            Arc::clone(&program_data),
        )
    } else {
        JoltProverPreprocessing::new(shared_preprocessing.clone(), Arc::clone(&program_data))
    };

    // Verify mode is correct
    assert_eq!(
        prover_preprocessing.is_committed_mode(),
        config.committed_program,
        "Prover mode mismatch"
    );

    // Compute trusted advice commitment if advice is provided
    let (trusted_commitment, trusted_hint) = if !config.trusted_advice.is_empty() {
        let (c, h) =
            commit_trusted_advice_preprocessing_only(&prover_preprocessing, &config.trusted_advice);
        (Some(c), Some(h))
    } else {
        (None, None)
    };

    // Create prover and prove
    let elf_contents = program.get_elf_contents().expect("elf contents is None");
    let program_mode = if config.committed_program {
        ProgramMode::Committed
    } else {
        ProgramMode::Full
    };
    let prover = RV64IMACProver::gen_from_elf_with_program_mode(
        &prover_preprocessing,
        &elf_contents,
        &config.inputs,
        &config.untrusted_advice,
        &config.trusted_advice,
        trusted_commitment,
        trusted_hint,
        program_mode,
    );
    let io_device = prover.program_io.clone();
    let (jolt_proof, debug_info) = prover.prove();
    assert_eq!(jolt_proof.program_mode, program_mode);

    // Create verifier preprocessing from prover (respects mode)
    let verifier_preprocessing = JoltVerifierPreprocessing::from(&prover_preprocessing);

    // Verify mode propagated correctly
    assert_eq!(
        verifier_preprocessing.program.is_committed(),
        config.committed_program,
        "Verifier mode mismatch"
    );

    // Verify
    let verifier = RV64IMACVerifier::new(
        &verifier_preprocessing,
        jolt_proof,
        io_device.clone(),
        trusted_commitment,
        debug_info,
    )
    .expect("Failed to create verifier");
    verifier.verify().expect("Verification failed");

    // Check expected output if specified
    if let Some(expected) = config.expected_output {
        assert_eq!(
            io_device.outputs, expected,
            "Output mismatch for program '{}'",
            config.program_name
        );
    }
}

/// Helper to commit trusted advice during preprocessing.
fn commit_trusted_advice_preprocessing_only(
    preprocessing: &JoltProverPreprocessing<Fr, DoryCommitmentScheme>,
    trusted_advice_bytes: &[u8],
) -> (
    <DoryCommitmentScheme as CommitmentScheme>::Commitment,
    <DoryCommitmentScheme as CommitmentScheme>::OpeningProofHint,
) {
    let max_trusted_advice_size = preprocessing.shared.memory_layout.max_trusted_advice_size;
    let mut trusted_advice_words = vec![0u64; (max_trusted_advice_size as usize) / 8];
    populate_memory_states(
        0,
        trusted_advice_bytes,
        Some(&mut trusted_advice_words),
        None,
    );

    let poly = MultilinearPolynomial::<Fr>::from(trusted_advice_words);
    let advice_len = poly.len().next_power_of_two().max(1);

    let _guard = DoryGlobals::initialize_context(1, advice_len, DoryContext::TrustedAdvice, None);
    let (commitment, hint) = {
        let _ctx = DoryGlobals::with_context(DoryContext::TrustedAdvice);
        DoryCommitmentScheme::commit(&poly, &preprocessing.generators)
    };
    (commitment, hint)
}

#[test]
#[serial]
fn fib_e2e() {
    run_e2e_test(E2ETestConfig::default());
}

#[test]
#[serial]
fn fib_e2e_small_trace() {
    run_e2e_test(E2ETestConfig::fibonacci(5).with_small_trace());
}

#[test]
#[serial]
fn sha2_e2e() {
    #[cfg(feature = "host")]
    use jolt_inlines_sha2 as _;
    run_e2e_test(E2ETestConfig::sha2());
}

#[test]
#[serial]
fn sha3_e2e() {
    #[cfg(feature = "host")]
    use jolt_inlines_keccak256 as _;
    run_e2e_test(E2ETestConfig::sha3());
}

#[test]
#[serial]
fn sha2_with_unused_advice_e2e() {
    // SHA2 guest does not consume advice, but providing both trusted and untrusted advice
    // should still work correctly through the full pipeline.
    #[cfg(feature = "host")]
    use jolt_inlines_sha2 as _;

    run_e2e_test(
        E2ETestConfig::sha2()
            .with_trusted_advice(postcard::to_stdvec(&[7u8; 32]).unwrap())
            .with_untrusted_advice(postcard::to_stdvec(&[9u8; 32]).unwrap()),
    );
}

#[test]
#[serial]
fn advice_merkle_tree_e2e() {
    run_e2e_test(E2ETestConfig::merkle_tree());
}

#[test]
#[serial]
fn memory_ops_e2e() {
    run_e2e_test(E2ETestConfig::memory_ops());
}

#[test]
#[serial]
fn btreemap_e2e() {
    run_e2e_test(E2ETestConfig::btreemap(50));
}

#[test]
#[serial]
fn muldiv_e2e() {
    run_e2e_test(E2ETestConfig::muldiv(9, 5, 3));
}

#[test]
#[serial]
fn fib_e2e_address_major() {
    run_e2e_test(E2ETestConfig::default().with_dory_layout(DoryLayout::AddressMajor));
}

#[test]
#[serial]
fn advice_merkle_tree_e2e_address_major() {
    run_e2e_test(E2ETestConfig::merkle_tree().with_dory_layout(DoryLayout::AddressMajor));
}

// ============================================================================
// New Tests - Committed Program Mode
//
// These tests exercise the end-to-end committed program path (bytecode + program image).
// ============================================================================

#[test]
#[serial]
fn fib_e2e_committed_program() {
    run_e2e_test(E2ETestConfig::default().with_committed_program());
}

#[test]
#[serial]
fn fib_e2e_committed_program_address_major() {
    run_e2e_test(
        E2ETestConfig::default()
            .with_committed_program()
            .with_dory_layout(DoryLayout::AddressMajor),
    );
}

#[test]
#[serial]
fn fib_e2e_committed_small_trace() {
    // Committed mode with minimal trace (256 cycles).
    // Tests program image commitment when trace is smaller than bytecode.
    run_e2e_test(
        E2ETestConfig::fibonacci(5)
            .with_small_trace()
            .with_committed_program(),
    );
}

#[test]
#[serial]
fn fib_e2e_committed_small_trace_address_major() {
    run_e2e_test(
        E2ETestConfig::fibonacci(5)
            .with_small_trace()
            .with_committed_program()
            .with_dory_layout(DoryLayout::AddressMajor),
    );
}

#[test]
#[serial]
fn sha2_e2e_committed_program() {
    // Larger program with committed mode (tests program image commitment with larger ELF).
    #[cfg(feature = "host")]
    use jolt_inlines_sha2 as _;
    run_e2e_test(E2ETestConfig::sha2().with_committed_program());
}

#[test]
#[serial]
fn sha2_e2e_committed_program_address_major() {
    #[cfg(feature = "host")]
    use jolt_inlines_sha2 as _;
    run_e2e_test(
        E2ETestConfig::sha2()
            .with_committed_program()
            .with_dory_layout(DoryLayout::AddressMajor),
    );
}

#[test]
#[serial]
fn sha3_e2e_committed_program() {
    // Another larger program for committed mode coverage.
    #[cfg(feature = "host")]
    use jolt_inlines_keccak256 as _;
    run_e2e_test(E2ETestConfig::sha3().with_committed_program());
}

#[test]
#[serial]
fn merkle_tree_e2e_committed_program() {
    // Committed mode with both trusted and untrusted advice.
    // Tests interaction of program image commitment with advice claim reductions.
    run_e2e_test(E2ETestConfig::merkle_tree().with_committed_program());
}

#[test]
#[serial]
fn merkle_tree_e2e_committed_program_address_major() {
    run_e2e_test(
        E2ETestConfig::merkle_tree()
            .with_committed_program()
            .with_dory_layout(DoryLayout::AddressMajor),
    );
}

#[test]
#[serial]
fn memory_ops_e2e_committed_program() {
    // Memory-ops guest exercises various load/store patterns.
    // Tests committed mode with diverse memory access patterns.
    run_e2e_test(E2ETestConfig::memory_ops().with_committed_program());
}

#[test]
#[serial]
fn btreemap_e2e_committed_program() {
    // BTreeMap guest has complex heap allocations.
    run_e2e_test(E2ETestConfig::btreemap(50).with_committed_program());
}

#[test]
#[serial]
fn muldiv_e2e_committed_program() {
    // Mul/div operations in committed mode.
    run_e2e_test(E2ETestConfig::muldiv(9, 5, 3).with_committed_program());
}

#[test]
#[serial]
fn fib_e2e_committed_large_trace() {
    // Larger trace length (2^17) in committed mode.
    // Tests bytecode chunking with log_k_chunk=8 (256 lanes per chunk).
    run_e2e_test(
        E2ETestConfig::fibonacci(1000)
            .with_max_trace_length(1 << 17)
            .with_committed_program(),
    );
}

#[test]
#[serial]
fn fib_e2e_committed_large_trace_address_major() {
    run_e2e_test(
        E2ETestConfig::fibonacci(1000)
            .with_max_trace_length(1 << 17)
            .with_committed_program()
            .with_dory_layout(DoryLayout::AddressMajor),
    );
}

#[test]
#[serial]
fn sha2_committed_program_with_advice() {
    // SHA2 doesn't consume advice, but providing it should still work in committed mode.
    // Tests that program image + bytecode + advice claim reductions all batch correctly.
    #[cfg(feature = "host")]
    use jolt_inlines_sha2 as _;
    run_e2e_test(
        E2ETestConfig::sha2()
            .with_committed_program()
            .with_trusted_advice(postcard::to_stdvec(&[7u8; 32]).unwrap())
            .with_untrusted_advice(postcard::to_stdvec(&[9u8; 32]).unwrap()),
    );
}

// ============================================================================
// New Tests - Bytecode Lane Ordering / Chunking
// ============================================================================

#[test]
fn bytecode_lane_chunking_counts() {
    // Canonical lane spec (see bytecode-commitment-progress.md):
    // 3*REGISTER_COUNT (rs1/rs2/rd) + 2 scalars + 13 circuit flags + 7 instr flags
    // + 41 lookup selector + 1 raf flag = 448 (with REGISTER_COUNT=128).
    assert_eq!(total_lanes(), 448);
    assert_eq!(total_lanes().div_ceil(16), 28);
    assert_eq!(total_lanes().div_ceil(256), 2);
}

// ============================================================================
// New Tests - Program Mode Detection
// ============================================================================

#[test]
#[serial]
fn program_mode_detection_full() {
    DoryGlobals::reset();
    let mut program = host::Program::new("fibonacci-guest");
    let (instructions, init_memory_state, _) = program.decode();
    let (_, _, _, io_device) = program.trace(&[], &[], &[]);

    let program = Arc::new(ProgramPreprocessing::preprocess(
        instructions,
        init_memory_state,
    ));
    let shared =
        JoltSharedPreprocessing::new(program.meta(), io_device.memory_layout.clone(), 1 << 16);

    // Full mode
    let prover_full: JoltProverPreprocessing<Fr, DoryCommitmentScheme> =
        JoltProverPreprocessing::new(shared.clone(), Arc::clone(&program));
    assert!(!prover_full.is_committed_mode());
    assert!(prover_full.program_commitments.is_none());

    let verifier_full = JoltVerifierPreprocessing::from(&prover_full);
    assert!(verifier_full.program.is_full());
    assert!(!verifier_full.program.is_committed());
    assert!(verifier_full.program.as_full().is_ok());
    assert!(verifier_full.program.as_committed().is_err());
}

#[test]
#[serial]
fn program_mode_detection_committed() {
    DoryGlobals::reset();
    let mut program = host::Program::new("fibonacci-guest");
    let (instructions, init_memory_state, _) = program.decode();
    let (_, _, _, io_device) = program.trace(&[], &[], &[]);

    let program_data = Arc::new(ProgramPreprocessing::preprocess(
        instructions,
        init_memory_state,
    ));
    let shared = JoltSharedPreprocessing::new(
        program_data.meta(),
        io_device.memory_layout.clone(),
        1 << 16,
    );

    // Committed mode
    let prover_committed: JoltProverPreprocessing<Fr, DoryCommitmentScheme> =
        JoltProverPreprocessing::new_committed(shared.clone(), Arc::clone(&program_data));
    assert!(prover_committed.is_committed_mode());
    assert!(prover_committed.program_commitments.is_some());

    let verifier_committed = JoltVerifierPreprocessing::from(&prover_committed);
    assert!(!verifier_committed.program.is_full());
    assert!(verifier_committed.program.is_committed());
    assert!(verifier_committed.program.as_full().is_err());
    assert!(verifier_committed.program.as_committed().is_ok());

    // Verify committed mode doesn't carry full program data
    assert!(
        verifier_committed.program.program_image_words().is_none(),
        "Committed mode should NOT have program image words"
    );
    assert!(
        verifier_committed.program.instructions().is_none(),
        "Committed mode should NOT have instructions"
    );
    assert!(
        verifier_committed.program.full().is_none(),
        "Committed mode should NOT have full preprocessing"
    );

    // But it should have commitments and metadata
    let trusted = verifier_committed.program.as_committed().unwrap();
    assert!(
        !trusted.bytecode_commitments.is_empty(),
        "Should have bytecode commitments"
    );
    assert!(
        trusted.bytecode_len > 0,
        "Should have bytecode length metadata"
    );
    assert!(
        trusted.program_image_num_words > 0,
        "Should have program image num words metadata"
    );
}

// ============================================================================
// Internal and Security Tests
//
// These tests require access to prover internals or manipulate trace/io
// directly for security testing. They cannot use E2ETestConfig.
// ============================================================================

#[test]
#[serial]
fn max_advice_with_small_trace() {
    DoryGlobals::reset();
    // Tests that max-sized advice (4KB = 512 words) works with a minimal trace.
    // With balanced dims (sigma_a=5, nu_a=4 for 512 words), the minimum padded trace
    // (256 cycles -> total_vars=12) is sufficient to embed advice.
    let mut program = host::Program::new("fibonacci-guest");
    let inputs = postcard::to_stdvec(&5u32).unwrap();
    let trusted_advice = vec![7u8; 4096];
    let untrusted_advice = vec![9u8; 4096];

    let (instructions, init_memory_state, _) = program.decode();
    let (lazy_trace, trace, final_memory_state, io_device) =
        program.trace(&inputs, &untrusted_advice, &trusted_advice);

    let program = Arc::new(ProgramPreprocessing::preprocess(
        instructions,
        init_memory_state,
    ));
    let shared_preprocessing =
        JoltSharedPreprocessing::new(program.meta(), io_device.memory_layout.clone(), 256);
    let prover_preprocessing: JoltProverPreprocessing<Fr, DoryCommitmentScheme> =
        JoltProverPreprocessing::new(shared_preprocessing.clone(), Arc::clone(&program));
    tracing::info!(
        "preprocessing.memory_layout.max_trusted_advice_size: {}",
        shared_preprocessing.memory_layout.max_trusted_advice_size
    );

    let (trusted_commitment, trusted_hint) =
        commit_trusted_advice_preprocessing_only(&prover_preprocessing, &trusted_advice);

    let prover = RV64IMACProver::gen_from_trace(
        &prover_preprocessing,
        lazy_trace,
        trace,
        io_device,
        Some(trusted_commitment),
        Some(trusted_hint),
        final_memory_state,
    );

    // Trace is tiny but advice is max-sized
    assert!(prover.unpadded_trace_len < 512);
    assert_eq!(prover.padded_trace_len, 256);

    let io_device = prover.program_io.clone();
    let (jolt_proof, debug_info) = prover.prove();

    let verifier_preprocessing = JoltVerifierPreprocessing::from(&prover_preprocessing);
    RV64IMACVerifier::new(
        &verifier_preprocessing,
        jolt_proof,
        io_device,
        Some(trusted_commitment),
        debug_info,
    )
    .expect("Failed to create verifier")
    .verify()
    .expect("Verification failed");
}

#[test]
#[serial]
fn advice_opening_point_derives_from_unified_point() {
    DoryGlobals::reset();
    // Tests that advice opening points are correctly derived from the unified main opening
    // point using Dory's balanced dimension policy.
    //
    // For a small trace (256 cycles), the advice row coordinates span both Stage 6 (cycle)
    // and Stage 7 (address) challenges, verifying the two-phase reduction works correctly.
    let mut program = host::Program::new("fibonacci-guest");
    let inputs = postcard::to_stdvec(&5u32).unwrap();
    let trusted_advice = postcard::to_stdvec(&[7u8; 32]).unwrap();
    let untrusted_advice = postcard::to_stdvec(&[9u8; 32]).unwrap();

    let (instructions, init_memory_state, _) = program.decode();
    let (lazy_trace, trace, final_memory_state, io_device) =
        program.trace(&inputs, &untrusted_advice, &trusted_advice);

    let program = Arc::new(ProgramPreprocessing::preprocess(
        instructions,
        init_memory_state,
    ));
    let shared_preprocessing =
        JoltSharedPreprocessing::new(program.meta(), io_device.memory_layout.clone(), 1 << 16);
    let prover_preprocessing: JoltProverPreprocessing<Fr, DoryCommitmentScheme> =
        JoltProverPreprocessing::new(shared_preprocessing.clone(), Arc::clone(&program));
    let (trusted_commitment, trusted_hint) =
        commit_trusted_advice_preprocessing_only(&prover_preprocessing, &trusted_advice);

    let prover = RV64IMACProver::gen_from_trace(
        &prover_preprocessing,
        lazy_trace,
        trace,
        io_device,
        Some(trusted_commitment),
        Some(trusted_hint),
        final_memory_state,
    );

    assert_eq!(prover.padded_trace_len, 256, "test expects small trace");

    let io_device = prover.program_io.clone();
    let (jolt_proof, debug_info) = prover.prove();
    let debug_info = debug_info.expect("expected debug_info in tests");

    // Get unified opening point and derive expected advice point
    let (opening_point, _) = debug_info
        .opening_accumulator
        .get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::HammingWeightClaimReduction,
        );
    let mut point_dory_le = opening_point.r.clone();
    point_dory_le.reverse();

    let total_vars = point_dory_le.len();
    let (sigma_main, _nu_main) = DoryGlobals::balanced_sigma_nu(total_vars);
    let (sigma_a, nu_a) = DoryGlobals::advice_sigma_nu_from_max_bytes(
        prover_preprocessing
            .shared
            .memory_layout
            .max_trusted_advice_size as usize,
    );

    // Build expected advice point: [col_bits[0..sigma_a] || row_bits[0..nu_a]]
    let mut expected_advice_le: Vec<_> = point_dory_le[0..sigma_a].to_vec();
    expected_advice_le.extend_from_slice(&point_dory_le[sigma_main..sigma_main + nu_a]);

    // Verify both advice types derive the same opening point
    for (name, kind) in [
        ("trusted", AdviceKind::Trusted),
        ("untrusted", AdviceKind::Untrusted),
    ] {
        let get_fn = debug_info
            .opening_accumulator
            .get_advice_opening(kind, SumcheckId::AdviceClaimReduction);
        assert!(
            get_fn.is_some(),
            "{name} advice opening missing for AdviceClaimReductionPhase2"
        );
        let (point_be, _) = get_fn.unwrap();
        let mut point_le = point_be.r.clone();
        point_le.reverse();
        assert_eq!(point_le, expected_advice_le, "{name} advice point mismatch");
    }

    // Verify end-to-end
    let verifier_preprocessing = JoltVerifierPreprocessing::from(&prover_preprocessing);
    RV64IMACVerifier::new(
        &verifier_preprocessing,
        jolt_proof,
        io_device,
        Some(trusted_commitment),
        Some(debug_info),
    )
    .expect("Failed to create verifier")
    .verify()
    .expect("Verification failed");
}

#[test]
#[serial]
#[should_panic]
fn truncated_trace() {
    let mut program = host::Program::new("fibonacci-guest");
    let (instructions, init_memory_state, _) = program.decode();
    let inputs = postcard::to_stdvec(&9u8).unwrap();
    let (lazy_trace, mut trace, final_memory_state, mut program_io) =
        program.trace(&inputs, &[], &[]);
    trace.truncate(100);
    program_io.outputs[0] = 0; // change the output to 0

    let program = Arc::new(ProgramPreprocessing::preprocess(
        instructions,
        init_memory_state,
    ));
    let shared_preprocessing =
        JoltSharedPreprocessing::new(program.meta(), program_io.memory_layout.clone(), 1 << 16);

    let prover_preprocessing: JoltProverPreprocessing<Fr, DoryCommitmentScheme> =
        JoltProverPreprocessing::new(shared_preprocessing.clone(), Arc::clone(&program));

    let prover = RV64IMACProver::gen_from_trace(
        &prover_preprocessing,
        lazy_trace,
        trace,
        program_io.clone(),
        None,
        None,
        final_memory_state,
    );

    let (proof, _) = prover.prove();

    let verifier_preprocessing = JoltVerifierPreprocessing::new_full(
        prover_preprocessing.shared.clone(),
        prover_preprocessing.generators.to_verifier_setup(),
        Arc::clone(&prover_preprocessing.program),
    );
    let verifier =
        RV64IMACVerifier::new(&verifier_preprocessing, proof, program_io, None, None).unwrap();
    verifier.verify().unwrap();
}

#[test]
#[serial]
#[should_panic]
fn malicious_trace() {
    let mut program = host::Program::new("fibonacci-guest");
    let inputs = postcard::to_stdvec(&1u8).unwrap();
    let (instructions, init_memory_state, _) = program.decode();
    let (lazy_trace, trace, final_memory_state, mut program_io) = program.trace(&inputs, &[], &[]);

    let program = Arc::new(ProgramPreprocessing::preprocess(
        instructions,
        init_memory_state,
    ));

    // Since the preprocessing is done with the original memory layout, the verifier should fail
    let shared_preprocessing =
        JoltSharedPreprocessing::new(program.meta(), program_io.memory_layout.clone(), 1 << 16);
    let prover_preprocessing: JoltProverPreprocessing<Fr, DoryCommitmentScheme> =
        JoltProverPreprocessing::new(shared_preprocessing.clone(), Arc::clone(&program));

    // change memory address of output & termination bit to the same address as input
    // changes here should not be able to spoof the verifier result
    program_io.memory_layout.output_start = program_io.memory_layout.input_start;
    program_io.memory_layout.output_end = program_io.memory_layout.input_end;
    program_io.memory_layout.termination = program_io.memory_layout.input_start;

    let prover = RV64IMACProver::gen_from_trace(
        &prover_preprocessing,
        lazy_trace,
        trace,
        program_io.clone(),
        None,
        None,
        final_memory_state,
    );
    let (proof, _) = prover.prove();

    let verifier_preprocessing = JoltVerifierPreprocessing::new_full(
        prover_preprocessing.shared.clone(),
        prover_preprocessing.generators.to_verifier_setup(),
        Arc::clone(&prover_preprocessing.program),
    );
    let verifier =
        JoltVerifier::new(&verifier_preprocessing, proof, program_io, None, None).unwrap();
    verifier.verify().unwrap();
}
