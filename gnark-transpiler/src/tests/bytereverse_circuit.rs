//! Pure ByteReverse Circuit Test
//!
//! Circuit: result = byte_reverse(x)
//!
//! Usage: cargo run --bin bytereverse_circuit

use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use gnark_transpiler::generate_circuit;
use zklean_extractor::mle_ast::MleAst;

fn byte_reverse_fr(x: Fr) -> Fr {
    // Serialize to LE bytes
    let mut buf = vec![];
    x.serialize_uncompressed(&mut buf).unwrap();
    println!("Serialized LE bytes: {:?}", buf);

    // Reverse
    buf.reverse();
    println!("Reversed bytes: {:?}", buf);

    // Interpret as LE (from_le_bytes_mod_order)
    Fr::from_le_bytes_mod_order(&buf)
}

fn main() {
    println!("=== Pure ByteReverse Circuit: byte_reverse(x) ===\n");

    // Test value
    let x_val: u64 = 42;

    // Concrete computation
    let x_fr = Fr::from(x_val);
    let result_fr = byte_reverse_fr(x_fr);

    println!("Concrete byte_reverse({}) = {}", x_val, result_fr);

    // Symbolic execution with MleAst
    let x_ast = MleAst::from_var(0);
    let result_ast = MleAst::byte_reverse(&x_ast);

    // Generate Go code
    let go_code = generate_circuit(result_ast.root(), "ByteReverseCircuit");

    println!("\nGenerated Go code:\n");
    println!("{}", go_code);

    // Write Go code
    let go_path = "/Users/mariogalante/DEV/wonderjolt/jolt/gnark-transpiler/go/bytereverse_circuit.go";
    std::fs::write(go_path, &go_code).expect("Failed to write Go file");
    println!("\nWritten to: {}", go_path);

    println!("\n--- Witness Data ---\n");
    println!("X_0 (x) = {}", x_val);
    println!("Expected output = {}", result_fr);
}
