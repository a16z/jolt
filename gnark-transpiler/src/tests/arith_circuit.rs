//! Pure Arithmetic Circuit Test (no Poseidon)
//!
//! Circuit: result = (a + b) * c * d
//!
//! Usage: cargo run --bin arith_circuit

use ark_bn254::Fr;
use gnark_transpiler::generate_circuit;
use zklean_extractor::mle_ast::MleAst;

fn main() {
    println!("=== Pure Arithmetic Circuit: (a + b) * c * d ===\n");

    // Test values
    let a_val: u64 = 3;
    let b_val: u64 = 7;
    let c_val: u64 = 5;
    let d_val: u64 = 42;

    // Concrete computation with Fr
    let a_fr = Fr::from(a_val);
    let b_fr = Fr::from(b_val);
    let c_fr = Fr::from(c_val);
    let d_fr = Fr::from(d_val);

    let result_fr = (a_fr + b_fr) * c_fr * d_fr;
    println!("Concrete result: {}", result_fr);

    // Symbolic execution with MleAst
    let a_ast = MleAst::from_var(0);
    let b_ast = MleAst::from_var(1);
    let c_ast = MleAst::from_var(2);
    let d_ast = MleAst::from_var(3);

    let result_ast = (a_ast + b_ast) * c_ast * d_ast;

    // Generate Go code
    let go_code = generate_circuit(result_ast.root(), "ArithCircuit");

    println!("Generated Go code:\n");
    println!("{}", go_code);

    // Write Go code
    let go_path = "/Users/mariogalante/DEV/wonderjolt/jolt/gnark-transpiler/go/arith_circuit.go";
    std::fs::write(go_path, &go_code).expect("Failed to write Go file");
    println!("\nWritten to: {}", go_path);

    println!("\n--- Witness Data ---\n");
    println!("X_0 (a) = {}", a_val);
    println!("X_1 (b) = {}", b_val);
    println!("X_2 (c) = {}", c_val);
    println!("X_3 (d) = {}", d_val);
    println!("Expected output = {}", result_fr);
}
