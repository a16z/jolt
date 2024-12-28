pub fn main() {
    let (prove_sha2_chain, verify_sha2_chain) = guest::build_sha2_chain();

    // Constant input data
    const INPUT: [u8; 32] = [5u8; 32];
    const ITERS: usize = 100;

    // Perform the SHA2 chain operation and generate proof
    let native_output = guest::sha2_chain(INPUT, ITERS);
    let (output, proof) = prove_sha2_chain(INPUT, ITERS);
    let is_valid = verify_sha2_chain(proof);

    // Verify if the results match
    assert_eq!(output, native_output, "output mismatch");

    // Encode to hex once and use the results
    let output_hex = hex::encode(output);
    let native_output_hex = hex::encode(native_output);

    // Print the results at once
    println!("output: {}", output_hex);
    println!("native_output: {}", native_output_hex);
    println!("valid: {}", is_valid);
}
