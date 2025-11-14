use clap::Parser;
use eyre::Result;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use ark_bn254::Fr;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::transcripts::Blake2bTranscript;
use jolt_core::zkvm::proof_serialization::JoltProof;
use jolt_core::zkvm::prover::JoltProverPreprocessing;
use jolt_core::zkvm::verifier::{JoltVerifier, JoltVerifierPreprocessing};
use jolt_core::zkvm::Serializable;

type RV64IMACProverPreprocessing = JoltProverPreprocessing<Fr, DoryCommitmentScheme>;
type RV64IMACVerifierPreprocessing = JoltVerifierPreprocessing<Fr, DoryCommitmentScheme>;
type RV64IMACProof = JoltProof<Fr, DoryCommitmentScheme, Blake2bTranscript>;
type RV64IMACVerifier<'a> = JoltVerifier<'a, Fr, DoryCommitmentScheme, Blake2bTranscript>;

/// Standalone Jolt verifier for proofs generated via C FFI
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the prover preprocessing file
    #[arg(short, long)]
    preprocessing: PathBuf,

    /// Path to the proof file
    #[arg(short = 'f', long)]
    proof: PathBuf,

    /// Path to the ELF file
    #[arg(short, long)]
    elf: PathBuf,

    /// Program inputs as hex string (optional)
    #[arg(short, long, default_value = "")]
    inputs: String,
}

fn parse_hex_inputs(hex_str: &str) -> Result<Vec<u8>> {
    if hex_str.is_empty() {
        return Ok(Vec::new());
    }

    let hex_str = hex_str.trim_start_matches("0x");
    hex::decode(hex_str).map_err(|e| eyre::eyre!("Failed to parse hex inputs: {}", e))
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    println!("Jolt Standalone Verifier");
    println!("========================\n");

    // Step 1: Load prover preprocessing
    println!("Step 1: Loading prover preprocessing...");
    let prover_preprocessing = RV64IMACProverPreprocessing::from_file(&args.preprocessing)?;
    println!(
        "  ✓ Loaded preprocessing from: {}\n",
        args.preprocessing.display()
    );

    // Step 2: Convert to verifier preprocessing
    println!("Step 2: Converting to verifier preprocessing...");
    let verifier_preprocessing: RV64IMACVerifierPreprocessing = (&prover_preprocessing).into();
    println!("  ✓ Converted to verifier preprocessing\n");

    // Step 3: Load proof
    println!("Step 3: Loading proof...");
    let proof = RV64IMACProof::from_file(&args.proof)?;
    let proof_size = fs::metadata(&args.proof)?.len();
    println!(
        "  ✓ Loaded proof ({} bytes) from: {}\n",
        proof_size,
        args.proof.display()
    );

    // Step 4: Re-execute program to generate io_device
    println!("Step 4: Re-executing program to generate io_device...");
    let elf_contents = fs::read(&args.elf)?;
    let inputs = parse_hex_inputs(&args.inputs)?;

    println!("  - ELF size: {} bytes", elf_contents.len());
    println!("  - Inputs: {} bytes", inputs.len());

    let memory_config = common::jolt_device::MemoryConfig {
        max_untrusted_advice_size: verifier_preprocessing
            .memory_layout
            .max_untrusted_advice_size,
        max_trusted_advice_size: verifier_preprocessing.memory_layout.max_trusted_advice_size,
        max_input_size: verifier_preprocessing.memory_layout.max_input_size,
        max_output_size: verifier_preprocessing.memory_layout.max_output_size,
        stack_size: verifier_preprocessing.memory_layout.stack_size,
        memory_size: verifier_preprocessing.memory_layout.memory_size,
        program_size: Some(verifier_preprocessing.memory_layout.program_size),
    };

    let (_lazy_trace, _trace, _final_memory, io_device) =
        jolt_core::guest::program::trace(&elf_contents, None, &inputs, &[], &[], &memory_config);

    println!("  ✓ Generated io_device:");
    println!("    - Inputs: {} bytes", io_device.inputs.len());
    println!("    - Outputs: {} bytes", io_device.outputs.len());
    println!("    - Panic: {}", io_device.panic);
    println!();

    // Step 5: Verify proof
    println!("Step 5: Verifying proof...");

    let verifier = RV64IMACVerifier::new(&verifier_preprocessing, proof, io_device, None, None)?;
    let verify_start = Instant::now();
    verifier
        .verify()
        .map_err(|e| eyre::eyre!("Verification failed: {}", e))?;

    let verify_duration = verify_start.elapsed();
    println!("  ✓ Proof verified successfully!");
    println!(
        "  ✓ Verification time: {:.3}s\n",
        verify_duration.as_secs_f64()
    );

    println!("Done! Proof is valid.");

    Ok(())
}
