//! Minimal Circuit Test
//!
//! A minimal function that can be transpiled from Rust to Gnark.
//!
//! Circuit: result = (a + b) * c * hash_challenge
//! where hash_challenge comes from a Poseidon transcript.
//!
//! Usage: cargo run --bin minimal_circuit

use ark_bn254::Fr;
use gnark_transpiler::{generate_circuit, PoseidonAstTranscript};
use jolt_core::field::JoltField;
use jolt_core::transcripts::{FrParams, PoseidonTranscript, Transcript};
use zklean_extractor::mle_ast::MleAst;

// ============================================================================
// The function to transpile - generic over JoltField and Transcript
// ============================================================================

/// Computes: (a + b) * c * hash_challenge
///
/// This is a single generic function that works with:
/// - Fr + PoseidonTranscript → concrete computation
/// - MleAst + PoseidonAstTranscript → builds AST for transpilation
fn compute<F: JoltField, T: Transcript>(a: F, b: F, c: F, d: F, transcript: &mut T) -> F {
    // Arithmetic: a + b
    let sum = a + b;

    // Append d to transcript (this triggers a Poseidon hash internally)
    transcript.append_scalar(&d);

    // Get challenge from transcript (another Poseidon hash)
    let hash_challenge: F = transcript.challenge_scalar();

    // Arithmetic: (a + b) * c * hash_challenge
    sum * c * hash_challenge
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("=== Minimal Circuit: (a + b) * c * hash_challenge ===\n");

    // Test values
    let a_val: u64 = 3;
    let b_val: u64 = 7;
    let c_val: u64 = 5;
    let d_val: u64 = 42;

    // ========================================
    // Part 1: Concrete computation with Fr
    // ========================================
    println!("--- Part 1: Concrete Computation (Fr + PoseidonTranscript) ---\n");

    let a_fr = Fr::from(a_val);
    let b_fr = Fr::from(b_val);
    let c_fr = Fr::from(c_val);
    let d_fr = Fr::from(d_val);

    let mut transcript_fr: PoseidonTranscript<Fr, FrParams> = Transcript::new(b"minimal");
    let result_fr = compute(a_fr, b_fr, c_fr, d_fr, &mut transcript_fr);

    println!("Inputs: a={}, b={}, c={}, d={}", a_val, b_val, c_val, d_val);
    println!("Result: {}", fr_to_decimal(&result_fr));

    // ========================================
    // Part 2: Symbolic execution with MleAst
    // ========================================
    println!("\n--- Part 2: Symbolic Execution (MleAst + PoseidonAstTranscript) ---\n");

    let a_ast = MleAst::from_var(0);
    let b_ast = MleAst::from_var(1);
    let c_ast = MleAst::from_var(2);
    let d_ast = MleAst::from_var(3);

    let mut transcript_ast: PoseidonAstTranscript = Transcript::new(b"minimal");
    let result_ast = compute(a_ast, b_ast, c_ast, d_ast, &mut transcript_ast);

    println!("AST root node ID: {}", result_ast.root());

    // ========================================
    // Part 3: Transpile AST → Go code
    // ========================================
    println!("\n--- Part 3: Transpilation (AST → Go) ---\n");

    let go_code = generate_circuit(result_ast.root(), "MinimalCircuit");

    println!("Generated Go code:\n");
    println!("{}", go_code);

    // Write Go code
    let go_path = "/Users/mariogalante/DEV/wonderjolt/jolt/gnark-transpiler/go/minimal_circuit.go";
    std::fs::write(go_path, &go_code).expect("Failed to write Go file");
    println!("\nWritten to: {}", go_path);

    // ========================================
    // Part 4: Witness data for Go test
    // ========================================
    println!("\n--- Part 4: Witness Data ---\n");
    println!("X_0 (a) = {}", a_val);
    println!("X_1 (b) = {}", b_val);
    println!("X_2 (c) = {}", c_val);
    println!("X_3 (d) = {}", d_val);
    println!("Expected output = {}", fr_to_decimal(&result_fr));
}

fn fr_to_decimal(fr: &Fr) -> String {
    format!("{}", fr)
}
