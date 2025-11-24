/// Demonstration of Stage 1 Only Verifier using fibonacci
///
/// This example shows how to extract and verify just Stage 1 (R1CS constraints)
/// from a full Jolt proof without modifying the fibonacci example.
///
/// Run with: cargo run --release --example stage1_verifier_demo

use jolt_core::zkvm::stage1_only_verifier::{
    Stage1OnlyPreprocessing, Stage1OnlyProof, Stage1OnlyVerifier,
};

fn main() {
    println!("\n=== Stage 1 Only Verifier Demo ===\n");
    println!("This demo requires the fibonacci example to generate a proof.");
    println!("Please run in a separate terminal:");
    println!("  cd examples/fibonacci && RUST_LOG=info cargo run --release\n");
    println!("This isolated verifier demonstrates:");
    println!("  • Extracting Stage 1 components from full proof");
    println!("  • Verifying R1CS constraints independently");
    println!("  • Minimal verification (no RAM, registers, or Dory opening)\n");
    println!("For actual usage, integrate Stage1OnlyVerifier into your application");
    println!("where you have access to the JoltProof object in memory.\n");

    println!("Example code:");
    println!("```rust");
    println!("// After generating full_proof with fibonacci prover:");
    println!("let (stage1_proof, opening_claims) = Stage1OnlyProof::from_full_proof(&full_proof);");
    println!("let stage1_preprocessing = Stage1OnlyPreprocessing::new(");
    println!("    full_proof.trace_length.next_power_of_two()");
    println!(");");
    println!("let verifier = Stage1OnlyVerifier::new(");
    println!("    stage1_preprocessing,");
    println!("    stage1_proof,");
    println!("    opening_claims,");
    println!(")?;");
    println!("verifier.verify()?; // Verifies ONLY R1CS constraints");
    println!("```\n");
}
