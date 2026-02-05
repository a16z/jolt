//! Transpile Jolt verifier stages 1-6 to Gnark circuit
//!
//! Uses TranspilableVerifier with symbolic proof and MleOpeningAccumulator
//! to generate a Gnark circuit for stages 1-6 of the Jolt verifier.

use ark_serialize::CanonicalDeserialize;
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
use std::collections::HashMap;

fn main() {
    println!("=== Transpiling Jolt Verifier Stages 1-6 to Gnark ===\n");

    // Load proof
    let proof_path = "/tmp/fib_proof.bin";
    println!("Loading proof from: {}", proof_path);
    let proof_bytes = std::fs::read(proof_path).expect("Failed to read proof file");
    let real_proof: RV64IMACProof =
        CanonicalDeserialize::deserialize_compressed(&proof_bytes[..])
            .expect("Failed to deserialize proof");
    println!("  trace_length: {}", real_proof.trace_length);
    println!("  commitments: {}", real_proof.commitments.len());

    // Debug: check commitment serialization size
    if let Some(first_commitment) = real_proof.commitments.first() {
        use ark_serialize::CanonicalSerialize;
        let mut bytes = Vec::new();
        first_commitment.serialize_uncompressed(&mut bytes).expect("serialize failed");
        println!("  first commitment serialized size: {} bytes ({} chunks of 32)", bytes.len(), bytes.len() / 32);
    }

    // Load io_device
    let io_device_path = "/tmp/fib_io_device.bin";
    println!("\nLoading io_device from: {}", io_device_path);
    let io_device_bytes = std::fs::read(io_device_path).expect("Failed to read io_device file");
    let io_device: JoltDevice = CanonicalDeserialize::deserialize_compressed(&io_device_bytes[..])
        .expect("Failed to deserialize io_device");
    println!("  inputs: {} bytes", io_device.inputs.len());
    println!("  outputs: {} bytes", io_device.outputs.len());

    // Load preprocessing (Dory version - matches jolt-sdk)
    let preprocessing_path = "/tmp/jolt_verifier_preprocessing.dat";
    println!("\nLoading preprocessing from: {}", preprocessing_path);
    let preprocessing_bytes =
        std::fs::read(preprocessing_path).expect("Failed to read preprocessing file");
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

    // Serialize bundle to JSON for future use (recursion, debugging)
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let bundle_path = format!("{}/go/stages16_bundle.json", manifest_dir);
    bundle
        .write_json(std::path::Path::new(&bundle_path))
        .expect("Failed to write AstBundle JSON");
    println!("  Bundle written to: {}", bundle_path);

    // Generate Gnark circuit from bundle
    println!("\n=== Generating Gnark Circuit ===");
    let circuit_code = generate_circuit_from_bundle(&bundle, "JoltStages16Circuit");

    let output_path = format!("{}/go/stages16_circuit.go", manifest_dir);
    std::fs::write(&output_path, &circuit_code).expect("Failed to write circuit file");
    println!("  Circuit written to: {}", output_path);
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
    let witness_path = format!("{}/go/stages16_witness.json", manifest_dir);
    std::fs::write(&witness_path, &witness_json).expect("Failed to write witness file");
    println!("  Witness written to: {}", witness_path);
    println!("  Witness variables: {}", witness_map.len());

    println!("\n=== SUCCESS ===");
    println!("TranspilableVerifier stages 1-6 transpiled to Gnark circuit.");
}

