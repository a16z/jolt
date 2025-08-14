use std::time::Instant;

use ark_bn254::Fr;
use ark_ff::{BigInt, PrimeField};
use jolt_sdk::{host, postcard};

// // Original 256-bit test inputs for BigInteger multiplication (commented out)
// // Two 256-bit inputs as two 128-bit limbs each: (lo, hi)
// // Random-looking 128-bit values for testing
// const A_LOW: u128 = 0xf3a8_9b7c_4d2e_1f0a_8b6c_5d3e_2f1a_0b9c;
// const A_HIGH: u128 = 0x9c8b_7a6f_5e4d_3c2b_1a09_8776_5544_3322;
// const B_LOW: u128 = 0x1234_5678_9abc_def0_fedc_ba98_7654_3210;
// const B_HIGH: u128 = 0xa5b6_c7d8_e9fa_0b1c_2d3e_4f50_6172_8394;

// Test inputs for mul_assign benchmark (field elements as 4 u64 limbs each)
// Using random-looking unstructured values for BN254's Fr field
const A_LIMBS: [u64; 4] = [
    0x73f8a2e6b49c5d1f,
    0x8b6c2a9f4e7d3012,
    0xd5a7c3b8e1f96024,
    0x294f6e8a3c5b7d19,
];

const B_LIMBS: [u64; 4] = [
    0xe9b5f7c2a4816d3e,
    0x47a3c8d6f1e52b90,
    0x3f2e8b7c6a9d5014,
    0x1c7e9a5f2d4b8036,
];

fn print_cycles_and_bytecode() {    
    let program = host::Program::new("big-int-guest");
    let mut inputs = vec![];
    
    // Add the 8 u64 limbs as inputs (4 for each field element)
    for limb in A_LIMBS.iter() {
        inputs.append(&mut postcard::to_stdvec(limb).unwrap());
    }
    for limb in B_LIMBS.iter() {
        inputs.append(&mut postcard::to_stdvec(limb).unwrap());
    }

    // Get the program summary
    let summary = program.trace_analyze::<Fr>(&inputs);
    
    // Print trace length
    println!("Trace length is: {}", summary.trace_len());
    
    // Write comprehensive trace analysis to file
    let filename = "mul_assign_montgomery_arkworks2.txt";
    match summary.write_trace_analysis_ecalls::<Fr>(&filename) {
        Ok(_) => println!("✅ Saved comprehensive trace analysis to: {}", filename),
        Err(e) => println!("❌ Failed to save trace analysis: {}", e),
    }
}

// 6537

// // Original main function for BigInteger multiplication (commented out)
// pub fn main() {
//     // print_cycles_and_bytecode();
//     let target_dir = "/tmp/jolt-guest-targets";
//     let mut program = guest::compile_mul_u256(target_dir);

//     let prover_preprocessing = guest::preprocess_prover_mul_u256(&mut program);
//     let verifier_preprocessing =
//         guest::verifier_preprocessing_from_prover_mul_u256(&prover_preprocessing);

//     let prove_mul_u256 = guest::build_prover_mul_u256(program, prover_preprocessing);
//     let verify_mul_u256 = guest::build_verifier_mul_u256(verifier_preprocessing);

//     let now = Instant::now();
//     let ((res0, res1, res2, res3, res4, res5, res6, res7), proof, program_io) = prove_mul_u256(A_LOW, A_HIGH, B_LOW, B_HIGH);
//     println!("Prover runtime: {} s", now.elapsed().as_secs_f64());

//     let is_valid = verify_mul_u256(A_LOW, A_HIGH, B_LOW, B_HIGH, (res0, res1, res2, res3, res4, res5, res6, res7), program_io.panic, proof);

//     println!("=== Low 256 bits (res0-res3) ===");
//     println!("res0: 0x{:016x}", res0);
//     println!("res1: 0x{:016x}", res1);
//     println!("res2: 0x{:016x}", res2);
//     println!("res3: 0x{:016x}", res3);
//     println!("=== High 256 bits (res4-res7) ===");
//     println!("res4: 0x{:016x}", res4);
//     println!("res5: 0x{:016x}", res5);
//     println!("res6: 0x{:016x}", res6);
//     println!("res7: 0x{:016x}", res7);
    
//     println!("valid: {is_valid}");
//     println!("✓ Result verification passed!");
// }

// New main function for mul_assign benchmark
pub fn main() {
    print_cycles_and_bytecode();
    
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_benchmark_mul_assign(target_dir);

    let prover_preprocessing = guest::preprocess_prover_benchmark_mul_assign(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_benchmark_mul_assign(&prover_preprocessing);

    let prove_mul_assign = guest::build_prover_benchmark_mul_assign(program, prover_preprocessing);
    let verify_mul_assign = guest::build_verifier_benchmark_mul_assign(verifier_preprocessing);

    println!("=== Benchmarking Montgomery mul_assign ===");
    println!("Input A limbs: {:016x}, {:016x}, {:016x}, {:016x}", 
             A_LIMBS[0], A_LIMBS[1], A_LIMBS[2], A_LIMBS[3]);
    println!("Input B limbs: {:016x}, {:016x}, {:016x}, {:016x}", 
             B_LIMBS[0], B_LIMBS[1], B_LIMBS[2], B_LIMBS[3]);
    
    let now = Instant::now();
    let ((res0, res1, res2, res3), proof, program_io) = prove_mul_assign(
        A_LIMBS[0], A_LIMBS[1], A_LIMBS[2], A_LIMBS[3],
        B_LIMBS[0], B_LIMBS[1], B_LIMBS[2], B_LIMBS[3]
    );
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());

    let is_valid = verify_mul_assign(
        A_LIMBS[0], A_LIMBS[1], A_LIMBS[2], A_LIMBS[3],
        B_LIMBS[0], B_LIMBS[1], B_LIMBS[2], B_LIMBS[3],
        (res0, res1, res2, res3), 
        program_io.panic, 
        proof
    );

    println!("=== Result (4 u64 limbs) ===");
    println!("res0: 0x{:016x}", res0);
    println!("res1: 0x{:016x}", res1);
    println!("res2: 0x{:016x}", res2);
    println!("res3: 0x{:016x}", res3);
    
    // Convert result back to Fr to verify correctness
    let result_bigint = BigInt::<4>::new([res0, res1, res2, res3]);
    let result_fr = Fr::from(result_bigint);
    
    // Also compute expected result for verification
    let a_bigint = BigInt::<4>::new(A_LIMBS);
    let a_fr = Fr::from(a_bigint);
    let b_bigint = BigInt::<4>::new(B_LIMBS);
    let b_fr = Fr::from(b_bigint);
    let expected = a_fr * b_fr;
    
    println!("\nValid proof: {}", is_valid);
    println!("Result matches expected: {}", result_fr == expected);
    
    if result_fr == expected {
        println!("✓ mul_assign benchmark verification passed!");
    } else {
        println!("✗ Result mismatch!");
        println!("Expected: {:?}", expected);
        println!("Got: {:?}", result_fr);
    }
}