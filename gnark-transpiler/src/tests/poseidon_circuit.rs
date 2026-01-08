//! Pure Poseidon Circuit Test (no ByteReverse)
//!
//! Circuit: result = poseidon(a, b, c)
//!
//! Usage: cargo run --bin poseidon_circuit

use ark_bn254::Fr;
use gnark_transpiler::generate_circuit;
use light_poseidon::{Poseidon, PoseidonHasher};
use zklean_extractor::mle_ast::MleAst;

fn main() {
    println!("=== Pure Poseidon Circuit: poseidon(a, b, c) ===\n");

    // Test values
    let a_val: u64 = 3;
    let b_val: u64 = 7;
    let c_val: u64 = 5;

    // Concrete computation with Fr using light-poseidon (circom parameters)
    let a_fr = Fr::from(a_val);
    let b_fr = Fr::from(b_val);
    let c_fr = Fr::from(c_val);

    // Use light-poseidon with circom parameters (width 3)
    let mut poseidon = Poseidon::<Fr>::new_circom(3).unwrap();
    let result_fr = poseidon.hash(&[a_fr, b_fr, c_fr]).unwrap();

    println!("Concrete poseidon({}, {}, {}) = {}", a_val, b_val, c_val, result_fr);

    // Symbolic execution with MleAst
    let a_ast = MleAst::from_var(0);
    let b_ast = MleAst::from_var(1);
    let c_ast = MleAst::from_var(2);

    let result_ast = MleAst::poseidon(&a_ast, &b_ast, &c_ast);

    // Generate Go code
    let go_code = generate_circuit(result_ast.root(), "PoseidonCircuit");

    println!("Generated Go code:\n");
    println!("{}", go_code);

    // Write Go code
    let go_path = "/Users/mariogalante/DEV/wonderjolt/jolt/gnark-transpiler/go/poseidon_circuit.go";
    std::fs::write(go_path, &go_code).expect("Failed to write Go file");
    println!("\nWritten to: {}", go_path);

    println!("\n--- Witness Data ---\n");
    println!("X_0 (a) = {}", a_val);
    println!("X_1 (b) = {}", b_val);
    println!("X_2 (c) = {}", c_val);
    println!("Expected output = {}", result_fr);
}
