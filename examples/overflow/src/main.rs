use guest::{build_memory_overflow_risk, build_stack_overflow_risk};

fn main() {
    // Stack overflow risk test
    let (prove_stack, verify_stack) = build_stack_overflow_risk();
    let (output, proof) = prove_stack(1000); // Try with different values
    let is_valid = verify_stack(proof);

    println!("Stack overflow risk output: {}", output);
    println!("Stack overflow risk valid: {}", is_valid);

    // Memory overflow risk test
    let (prove_memory, verify_memory) = build_memory_overflow_risk();
    let (output, proof) = prove_memory(100000); // Try with different values
    let is_valid = verify_memory(proof);

    println!("Memory overflow risk output: {}", output);
    println!("Memory overflow risk valid: {}", is_valid);
}
