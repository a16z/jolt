//! Standalone proof generator that loads ELF and inputs from files
//! and generates a Jolt proof without depending on the guest compilation.
//!
//! This demonstrates how to prove a program when the ELF and inputs are
//! provided as separate files (e.g., from `extract_files.rs`).

use common::jolt_device::{MemoryConfig, MemoryLayout};
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryContext, DoryGlobals};
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::zkvm::prover::JoltProverPreprocessing;
use jolt_core::zkvm::ram::populate_memory_states;
use jolt_core::zkvm::verifier::{JoltSharedPreprocessing, JoltVerifierPreprocessing};
use jolt_core::zkvm::RV64IMACProver;
use jolt_sdk::guest::verifier;
use jolt_sdk::host::Program;
use jolt_sdk::{F, PCS, RV64IMACProof};
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::time::Instant;
use tracing::info;

// Import the inline handlers for SHA256 (used by sha2-chain)
// The #[ctor::ctor] attribute in jolt-inlines-sha2 auto-registers these,
// but we need to ensure the crate is linked.
use jolt_inlines_sha2 as _;

/// Memory configuration nested in JSON
#[derive(Debug, Deserialize)]
struct MemConf {
    memory_size: u64,
    stack_size: u64,
    max_input_size: u64,
    max_output_size: u64,
    max_trusted_advice_size: u64,
    max_untrusted_advice_size: u64,
}

/// Program configuration loaded from JSON
#[derive(Debug, Deserialize)]
struct ProgramConfig {
    #[allow(dead_code)]
    compressed_chunk_size: u64,
    max_trace_length: u64,
    mem_conf: MemConf,
}

/// Commit trusted advice in a preprocessing-only Dory context.
///
/// This replicates what `commit_trusted_advice_*` does in the generated SDK code.
fn commit_trusted_advice(
    preprocessing: &JoltProverPreprocessing<F, DoryCommitmentScheme>,
    trusted_advice_bytes: &[u8],
) -> (
    Option<<DoryCommitmentScheme as CommitmentScheme>::Commitment>,
    Option<<DoryCommitmentScheme as CommitmentScheme>::OpeningProofHint>,
) {
    if trusted_advice_bytes.is_empty() {
        return (None, None);
    }

    let max_trusted_advice_size = preprocessing.shared.memory_layout.max_trusted_advice_size;
    let mut trusted_advice_words = vec![0u64; (max_trusted_advice_size as usize) / 8];

    populate_memory_states(0, trusted_advice_bytes, Some(&mut trusted_advice_words), None);

    let poly = MultilinearPolynomial::<F>::from(trusted_advice_words);
    let advice_len = poly.len().next_power_of_two().max(1);

    // Initialize the dedicated Dory context for trusted advice commitment
    let _guard = DoryGlobals::initialize_context(1, advice_len, DoryContext::TrustedAdvice, None);
    let (commitment, hint) = {
        let _ctx = DoryGlobals::with_context(DoryContext::TrustedAdvice);
        DoryCommitmentScheme::commit(&poly, &preprocessing.generators)
    };

    (Some(commitment), Some(hint))
}

/// Data loaded from program files
pub struct LoadedProgram {
    pub elf_contents: Vec<u8>,
    pub prover_preprocessing: JoltProverPreprocessing<F, DoryCommitmentScheme>,
}

/// Load program configuration, ELF, and create preprocessing from files.
pub fn load_program(program_dir: &Path) -> Result<LoadedProgram, Box<dyn std::error::Error>> {
    // Load configuration
    let config_path = program_dir.join("program.json");
    let config_str = fs::read_to_string(&config_path)?;
    let config: ProgramConfig = serde_json::from_str(&config_str)?;

    info!("Loaded config: {:?}", config);
    info!("max_trace_length = {}", config.max_trace_length);

    // Load ELF
    let elf_path = program_dir.join("program.elf");
    let elf_path_str = elf_path.to_str().unwrap();

    // Create Program and set memory configuration from JSON
    let mut program = Program::new("guest");
    program.load_elf(elf_path_str);
    program.set_memory_size(config.mem_conf.memory_size);
    program.set_stack_size(config.mem_conf.stack_size);
    program.set_max_input_size(config.mem_conf.max_input_size);
    program.set_max_output_size(config.mem_conf.max_output_size);
    program.set_max_trusted_advice_size(config.mem_conf.max_trusted_advice_size);
    program.set_max_untrusted_advice_size(config.mem_conf.max_untrusted_advice_size);

    let elf_contents = program.get_elf_contents().expect("Failed to get ELF contents");

    // Decode bytecode and initial memory state
    let (bytecode, init_memory_state, program_size) = program.decode();

    // Create memory config with program_size
    let memory_config = MemoryConfig {
        memory_size: config.mem_conf.memory_size,
        stack_size: config.mem_conf.stack_size,
        max_input_size: config.mem_conf.max_input_size,
        max_output_size: config.mem_conf.max_output_size,
        max_trusted_advice_size: config.mem_conf.max_trusted_advice_size,
        max_untrusted_advice_size: config.mem_conf.max_untrusted_advice_size,
        program_size: Some(program_size),
    };
    let memory_layout = MemoryLayout::new(&memory_config);

    info!("Program size: {} bytes", program_size);
    info!("Bytecode instructions: {}", bytecode.len());

    // Create shared preprocessing
    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode,
        memory_layout,
        init_memory_state,
        config.max_trace_length as usize,
    );

    // Create prover preprocessing
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());

    Ok(LoadedProgram {
        elf_contents,
        prover_preprocessing,
    })
}

/// Loaded advice data with commitment
pub struct LoadedAdvice {
    pub trusted_advice_bytes: Vec<u8>,
    pub untrusted_advice_bytes: Vec<u8>,
    pub trusted_advice_commitment: Option<<DoryCommitmentScheme as CommitmentScheme>::Commitment>,
    pub trusted_advice_hint: Option<<DoryCommitmentScheme as CommitmentScheme>::OpeningProofHint>,
}

/// Load advice from files and commit trusted advice.
pub fn load_advice(
    inputs_dir: &Path,
    prover_preprocessing: &JoltProverPreprocessing<F, DoryCommitmentScheme>,
) -> Result<LoadedAdvice, Box<dyn std::error::Error>> {
    // Load trusted advice
    let trusted_advice_path = inputs_dir.join("trusted_advice.bin");
    let trusted_advice_bytes = if trusted_advice_path.exists() {
        fs::read(&trusted_advice_path)?
    } else {
        Vec::new()
    };
    info!(
        "Loaded trusted_advice.bin: {} bytes",
        trusted_advice_bytes.len()
    );

    // Load untrusted advice
    let untrusted_advice_path = inputs_dir.join("untrusted_advice.bin");
    let untrusted_advice_bytes = if untrusted_advice_path.exists() {
        fs::read(&untrusted_advice_path)?
    } else {
        Vec::new()
    };
    info!(
        "Loaded untrusted_advice.bin: {} bytes",
        untrusted_advice_bytes.len()
    );

    // Commit trusted advice BEFORE proving (in its own Dory context)
    let (trusted_advice_commitment, trusted_advice_hint) =
        commit_trusted_advice(prover_preprocessing, &trusted_advice_bytes);

    if trusted_advice_commitment.is_some() {
        info!("Committed trusted advice");
    }

    Ok(LoadedAdvice {
        trusted_advice_bytes,
        untrusted_advice_bytes,
        trusted_advice_commitment,
        trusted_advice_hint,
    })
}

/// Verify one or more proofs against the preprocessing.
pub fn verify_proofs(
    input: &[u8],
    proofs: Vec<(&str, RV64IMACProof, Vec<u8>)>,
    preprocessing: &JoltProverPreprocessing<F, PCS>,
    trusted_advice_commitment: Option<<PCS as CommitmentScheme>::Commitment>,
) {
    info!("Verifying proofs...");
    let verifier_preprocessing = JoltVerifierPreprocessing::from(preprocessing);
    let mut output_bytes =
        vec![0; verifier_preprocessing.shared.memory_layout.max_output_size as usize];

    for (name, proof, outputs) in proofs {
        info!("Verifying {} proof...", name);
        output_bytes[..outputs.len()].copy_from_slice(&outputs);
        info!("  Output: {}", hex::encode(&output_bytes[..outputs.len()]));

        let is_valid = verifier::verify(
            input,
            trusted_advice_commitment.clone(),
            &output_bytes,
            proof,
            &verifier_preprocessing,
        )
        .is_ok();

        if is_valid {
            println!("✓ {} proof verified successfully!", name);
        } else {
            println!("✗ {} proof verification FAILED!", name);
        }
    }
}

/// Load input data from files and run the prover
pub fn prove_from_files(
    program_dir: &Path,
    inputs_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let LoadedProgram {
        elf_contents,
        prover_preprocessing,
    } = load_program(program_dir)?;

    // Load inputs
    let input_path = inputs_dir.join("input.bin");
    let input_bytes = fs::read(&input_path)?;
    info!("Loaded input.bin: {} bytes", input_bytes.len());

    // Load and commit advice
    let LoadedAdvice {
        trusted_advice_bytes,
        untrusted_advice_bytes,
        trusted_advice_commitment,
        trusted_advice_hint,
    } = load_advice(inputs_dir, &prover_preprocessing)?;

    // Create prover
    info!("Creating prover from ELF...");
    let prover = RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &input_bytes,
        &untrusted_advice_bytes,
        &trusted_advice_bytes,
        trusted_advice_commitment.clone(),
        trusted_advice_hint,
    );

    let program_io = prover.program_io.clone();
    let padded_trace_len = prover.padded_trace_len;

    info!(
        "Trace length: {} (padded to 2^{} = {})",
        prover.unpadded_trace_len,
        padded_trace_len.trailing_zeros(),
        padded_trace_len
    );

    // Prove
    info!("Starting proof generation...");
    let now = Instant::now();
    let (jolt_proof, _debug_info) = prover.prove();
    let prove_duration = now.elapsed();

    let prove_hz = padded_trace_len as f64 / prove_duration.as_secs_f64();
    info!(
        "Proved in {:.3} s ({:.1} Hz over 2^{} instructions)",
        prove_duration.as_secs_f64(),
        prove_hz,
        padded_trace_len.trailing_zeros()
    );

    // Verify using verify_proofs
    let proofs = vec![(
        "Standalone",
        jolt_proof,
        program_io.outputs.clone(),
    )];
    verify_proofs(
        &input_bytes,
        proofs,
        &prover_preprocessing,
        trusted_advice_commitment,
    );

    Ok(())
}

pub fn main() {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();

    let (program_dir, inputs_dir) = if args.len() >= 3 {
        (
            Path::new(&args[1]).to_path_buf(),
            Path::new(&args[2]).to_path_buf(),
        )
    } else {
        // Default paths relative to workspace root
        let base = Path::new("./sha2-chain-export");
        (base.join("program"), base.join("inputs"))
    };

    info!("Program directory: {}", program_dir.display());
    info!("Inputs directory: {}", inputs_dir.display());

    if let Err(e) = prove_from_files(&program_dir, &inputs_dir) {
        tracing::error!("Error: {}", e);
        std::process::exit(1);
    }
}
