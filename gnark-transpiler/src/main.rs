//! Transpile Jolt verifier stages 1-6 to Gnark circuit
//!
//! Uses TranspilableVerifier with symbolic proof and MleOpeningAccumulator
//! to generate a Gnark circuit for stages 1-6 of the Jolt verifier.

use ark_serialize::CanonicalDeserialize;
use clap::Parser;
use std::collections::HashMap;
use std::path::PathBuf;

use gnark_transpiler::{
    symbolize_proof, extract_witness_values, generate_circuit_from_bundle,
    AstCommitmentScheme, MleOpeningAccumulator, PoseidonAstTranscript, sanitize_go_name,
};
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::transpilable_verifier::TranspilableVerifier;
use jolt_core::zkvm::verifier::JoltVerifierPreprocessing;
use jolt_core::zkvm::RV64IMACProof;
use common::jolt_device::JoltDevice;
use zklean_extractor::mle_ast::{enable_constraint_mode, take_constraints as take_assertions, AstBundle, InputKind, MleAst};

// Output file names
const CIRCUIT_FILENAME: &str = "stages_circuit.go";
const WITNESS_FILENAME: &str = "stages_witness.json";
const BUNDLE_FILENAME: &str = "stages_bundle.json";

/// Transpile Jolt proofs to gnark circuits for Groth16 proving.
///
/// Takes a Jolt proof, io_device, and preprocessing files as input,
/// and generates gnark circuit code and witness data.
#[derive(Parser)]
#[command(name = "gnark-transpiler")]
#[command(version, about)]
struct Args {
    /// Path to the proof file (serialized JoltProof)
    #[arg(long, default_value = "/tmp/fib_proof.bin")]
    proof: PathBuf,

    /// Path to the io_device file (inputs/outputs)
    #[arg(long, default_value = "/tmp/fib_io_device.bin")]
    io_device: PathBuf,

    /// Path to the preprocessing file
    #[arg(long, default_value = "/tmp/jolt_verifier_preprocessing.dat")]
    preprocessing: PathBuf,

    /// Output directory for generated Go files
    #[arg(long, short = 'o', default_value = "go")]
    output_dir: PathBuf,

    /// Enable verbose output
    #[arg(long, short = 'v')]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    println!("=== Transpiling Jolt Verifier Stages 1-6 to Gnark ===\n");

    // Load proof
    println!("Loading proof from: {:?}", args.proof);
    let proof_bytes = std::fs::read(&args.proof)
        .unwrap_or_else(|e| panic!("Failed to read proof file {:?}: {}", args.proof, e));
    let real_proof: RV64IMACProof =
        CanonicalDeserialize::deserialize_compressed(&proof_bytes[..])
            .expect("Failed to deserialize proof");
    println!("  trace_length: {}", real_proof.trace_length);
    println!("  commitments: {}", real_proof.commitments.len());

    if args.verbose {
        // Debug: check commitment serialization size
        if let Some(first_commitment) = real_proof.commitments.first() {
            use ark_serialize::CanonicalSerialize;
            let mut bytes = Vec::new();
            first_commitment.serialize_uncompressed(&mut bytes).expect("serialize failed");
            println!("  first commitment serialized size: {} bytes ({} chunks of 32)", bytes.len(), bytes.len() / 32);
        }
    }

    // Load io_device
    println!("\nLoading io_device from: {:?}", args.io_device);
    let io_device_bytes = std::fs::read(&args.io_device)
        .unwrap_or_else(|e| panic!("Failed to read io_device file {:?}: {}", args.io_device, e));
    let io_device: JoltDevice = CanonicalDeserialize::deserialize_compressed(&io_device_bytes[..])
        .expect("Failed to deserialize io_device");
    println!("  inputs: {} bytes", io_device.inputs.len());
    println!("  outputs: {} bytes", io_device.outputs.len());

    // Load preprocessing (Dory version - matches jolt-sdk)
    println!("\nLoading preprocessing from: {:?}", args.preprocessing);
    let preprocessing_bytes = std::fs::read(&args.preprocessing)
        .unwrap_or_else(|e| panic!("Failed to read preprocessing file {:?}: {}", args.preprocessing, e));
    let real_preprocessing: JoltVerifierPreprocessing<ark_bn254::Fr, DoryCommitmentScheme> =
        CanonicalDeserialize::deserialize_compressed(&preprocessing_bytes[..])
            .expect("Failed to deserialize preprocessing");
    println!("  memory_layout: {:?}", real_preprocessing.shared.memory_layout);

    // Convert preprocessing to AstCommitmentScheme version
    // (only generators change, shared stays the same)
    let symbolic_preprocessing: JoltVerifierPreprocessing<MleAst, AstCommitmentScheme> =
        JoltVerifierPreprocessing {
            generators: gnark_transpiler::ast_commitment_scheme::AstVerifierSetup,
            shared: real_preprocessing.shared.clone(),
        };

    // Symbolize the proof
    println!("\n=== Symbolizing Proof ===");
    let (symbolic_proof, accumulator, var_alloc) = symbolize_proof(&real_proof);
    println!("  Total symbolic variables: {}", var_alloc.next_idx());

    // Create transcript
    let transcript: PoseidonAstTranscript = Transcript::new(b"Jolt");

    // Create TranspilableVerifier with symbolic types
    println!("\n=== Creating TranspilableVerifier ===");
    let verifier = TranspilableVerifier::<
        MleAst,
        AstCommitmentScheme,
        PoseidonAstTranscript,
        MleOpeningAccumulator,
    >::new_with_accumulator(
        &symbolic_preprocessing,
        symbolic_proof,
        io_device,
        None, // trusted_advice_commitment
        transcript,
        accumulator,
    );

    // Enable assertion mode so MleAst comparisons register equality checks
    enable_constraint_mode();

    // Run verification (stages 1-6)
    println!("\n=== Running Symbolic Verification (Stages 1-6) ===");
    match verifier.verify() {
        Ok(()) => println!("  Verification completed successfully"),
        Err(e) => {
            println!("  Verification error: {:?}", e);
            return;
        }
    }

    // Collect accumulated assertions (equality checks that become api.AssertIsEqual calls)
    let assertions = take_assertions();
    println!("\n=== Accumulated Assertions ===");
    println!("  Total assertions: {}", assertions.len());

    // Build AstBundle — the central data structure for circuit generation
    println!("\n=== Building AstBundle ===");
    let mut bundle = AstBundle::new();
    bundle.snapshot_arena();

    // All symbolic variables from symbolize_proof are proof data
    for (idx, name) in var_alloc.descriptions() {
        bundle.add_input(*idx, name.clone(), InputKind::ProofData);
    }
    println!("  Inputs: {}", bundle.inputs.len());

    // All constraints from PartialEq::eq are EqualZero: (lhs - rhs) == 0
    for (i, assertion) in assertions.iter().enumerate() {
        bundle.add_constraint_eq_zero(format!("assertion_{}", i), assertion.root());
    }
    println!("  Constraints: {}", bundle.constraints.len());

    // Determine output directory (resolve relative to manifest dir if relative path)
    let output_dir = if args.output_dir.is_relative() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        manifest_dir.join(&args.output_dir)
    } else {
        args.output_dir.clone()
    };

    // Serialize bundle to JSON for future use (debugging, analysis)
    let bundle_path = output_dir.join(BUNDLE_FILENAME);
    bundle
        .write_json(&bundle_path)
        .unwrap_or_else(|e| panic!("Failed to write bundle file {:?}: {}", bundle_path, e));
    println!("  Bundle written to: {:?}", bundle_path);

    // Generate Gnark circuit from bundle
    println!("\n=== Generating Gnark Circuit ===");
    let circuit_code = generate_circuit_from_bundle(&bundle, "JoltStagesCircuit");

    let circuit_path = output_dir.join(CIRCUIT_FILENAME);
    std::fs::write(&circuit_path, &circuit_code)
        .unwrap_or_else(|e| panic!("Failed to write circuit file {:?}: {}", circuit_path, e));
    println!("  Circuit written to: {:?}", circuit_path);
    println!("  Circuit size: {} bytes", circuit_code.len());

    // === Generate witness data ===
    println!("\n=== Generating Witness Data ===");
    let witness_values = extract_witness_values(&real_proof);

    // Build witness JSON mapping sanitized variable names to values
    let mut witness_map: HashMap<String, String> = HashMap::new();
    for (idx, name) in var_alloc.descriptions() {
        let sanitized = sanitize_go_name(name);
        if let Some(value) = witness_values.get(&(*idx as usize)) {
            witness_map.insert(sanitized, value.clone());
        }
    }

    let witness_json = serde_json::to_string_pretty(&witness_map).expect("Failed to serialize witness");
    let witness_path = output_dir.join(WITNESS_FILENAME);
    std::fs::write(&witness_path, &witness_json)
        .unwrap_or_else(|e| panic!("Failed to write witness file {:?}: {}", witness_path, e));
    println!("  Witness written to: {:?}", witness_path);
    println!("  Witness variables: {}", witness_map.len());

    println!("\n=== SUCCESS ===");
    println!("TranspilableVerifier stages 1-6 transpiled to Gnark circuit.");
}
