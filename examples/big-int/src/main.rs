//! Test suite for Montgomery multiplication (mul_assign) in Jolt zkVM
//! 
//! This test suite verifies the correctness of the mul_assign operation
//! for BN254 field elements (Fr) using two complementary approaches:
//! 
//! 1. **Edge Case Testing**: Tests specific boundary conditions including:
//!    - Zero and one values
//!    - Maximum field elements (modulus - 1)
//!    - Values that cause modular reduction
//!    - Inverses and special bit patterns
//!    - Square roots and values close to the modulus
//! 
//! 2. **Random Testing**: Tests with 1000 randomly generated Fr elements
//!    to ensure correctness across the entire field
//!
//! Each test generates a proof using Jolt and verifies both:
//! - The validity of the proof
//! - The correctness of the computed result

use std::time::Instant;

use ark_bn254::Fr;
use ark_ff::{BigInt, Field, One, UniformRand, Zero};
// use jolt_sdk::{host, postcard};  // Only needed for trace analysis
use rand::thread_rng;


// Original test constants (kept for reference)
// const A_LIMBS: [u64; 4] = [
//     0x73f8a2e6b49c5d1f,
//     0x8b6c2a9f4e7d3012,
//     0xd5a7c3b8e1f96024,
//     0x294f6e8a3c5b7d19,
// ];
// 
// const B_LIMBS: [u64; 4] = [
//     0xe9b5f7c2a4816d3e,
//     0x47a3c8d6f1e52b90,
//     0x3f2e8b7c6a9d5014,
//     0x1c7e9a5f2d4b8036,
// ];

// Helper function to print cycle count and bytecode analysis
// #[allow(dead_code)]
// fn print_cycles_and_bytecode() {    
//     let program = host::Program::new("big-int-guest");
//     let mut inputs = vec![];
//     
//     // Add the 8 u64 limbs as inputs (4 for each field element)
//     for limb in A_LIMBS.iter() {
//         inputs.append(&mut postcard::to_stdvec(limb).unwrap());
//     }
//     for limb in B_LIMBS.iter() {
//         inputs.append(&mut postcard::to_stdvec(limb).unwrap());
//     }
// 
//     // Get the program summary
//     let summary = program.trace_analyze::<Fr>(&inputs);
//     
//     // Print trace length
//     println!("Trace length is: {}", summary.trace_len());
//     
//     // Write comprehensive trace analysis to file
//     let filename = "mul_assign_fr_no_SUB.txt";
//     match summary.write_trace_analysis::<Fr>(&filename) {
//         Ok(_) => println!("✅ Saved comprehensive trace analysis to: {}", filename),
//         Err(e) => println!("❌ Failed to save trace analysis: {}", e),
//     }
// }

// Original main function for reference
// pub fn main() {
//     // print_cycles_and_bytecode();
//     
//     let target_dir = "/tmp/jolt-guest-targets";
//     let mut program = guest::compile_benchmark_mul_assign(target_dir);
// 
//     let prover_preprocessing = guest::preprocess_prover_benchmark_mul_assign(&mut program);
//     let verifier_preprocessing =
//         guest::verifier_preprocessing_from_prover_benchmark_mul_assign(&prover_preprocessing);
// 
//     let prove_mul_assign = guest::build_prover_benchmark_mul_assign(program, prover_preprocessing);
//     let verify_mul_assign = guest::build_verifier_benchmark_mul_assign(verifier_preprocessing);
// 
//     println!("=== Benchmarking Montgomery mul_assign ===");
//     println!("Input A limbs: {:016x}, {:016x}, {:016x}, {:016x}", 
//             A_LIMBS[0], A_LIMBS[1], A_LIMBS[2], A_LIMBS[3]);
//     println!("Input B limbs: {:016x}, {:016x}, {:016x}, {:016x}", 
//             B_LIMBS[0], B_LIMBS[1], B_LIMBS[2], B_LIMBS[3]);
//     
//     let now = Instant::now();
//     let ((res0, res1, res2, res3), proof, program_io) = prove_mul_assign(
//         A_LIMBS[0], A_LIMBS[1], A_LIMBS[2], A_LIMBS[3],
//         B_LIMBS[0], B_LIMBS[1], B_LIMBS[2], B_LIMBS[3]
//     );
//     println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
// 
//     let is_valid = verify_mul_assign(
//         A_LIMBS[0], A_LIMBS[1], A_LIMBS[2], A_LIMBS[3],
//         B_LIMBS[0], B_LIMBS[1], B_LIMBS[2], B_LIMBS[3],
//         (res0, res1, res2, res3), 
//         program_io.panic, 
//         proof
//     );
// 
//     println!("=== Result (4 u64 limbs) ===");
//     println!("res0: 0x{:016x}", res0);
//     println!("res1: 0x{:016x}", res1);
//     println!("res2: 0x{:016x}", res2);
//     println!("res3: 0x{:016x}", res3);
//     
//     // Convert result back to Fr to verify correctness
//     let result_bigint = BigInt::<4>::new([res0, res1, res2, res3]);
//     let result_fr = Fr::from(result_bigint);
//     
//     // Also compute expected result for verification
//     let a_bigint = BigInt::<4>::new(A_LIMBS);
//     let mut a_fr = Fr::from(a_bigint);
//     let b_bigint = BigInt::<4>::new(B_LIMBS);
//     let mut b_fr = Fr::from(b_bigint);
// 
//     a_fr *= b_fr;  // This calls mul_assign internally
//     b_fr *= a_fr;  // This calls mul_assign internally
//     a_fr *= b_fr;  // This calls mul_assign internally
//     b_fr *= a_fr;  // This calls mul_assign internally
//     a_fr *= b_fr;  // This calls mul_assign internally
//     b_fr *= a_fr;  // This calls mul_assign internally
//     a_fr *= b_fr;  // This calls mul_assign internally
//     b_fr *= a_fr;  // This calls mul_assign internally
//     a_fr *= a_fr;  // This calls mul_assign internally
//     a_fr *= b_fr;  // This calls mul_assign internally
//     let expected = a_fr;
//     
//     println!("\nValid proof: {}", is_valid);
//     println!("Result matches expected: {}", result_fr == expected);
//     
//     if result_fr == expected {
//         println!("✓ mul_assign benchmark verification passed!");
//     } else {
//         println!("✗ Result mismatch!");
//         println!("Expected: {:?}", expected);
//         println!("Got: {:?}", result_fr);
//     }
// }

/// Test with random Fr elements
fn test_random_fr_elements() {
    println!("=== Testing Montgomery mul_assign with random Fr elements ===");
    
    // Setup once
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_benchmark_mul_assign(target_dir);
    
    let prover_preprocessing = guest::preprocess_prover_benchmark_mul_assign(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_benchmark_mul_assign(&prover_preprocessing);
    
    let prove_mul_assign = guest::build_prover_benchmark_mul_assign(program, prover_preprocessing);
    let verify_mul_assign = guest::build_verifier_benchmark_mul_assign(verifier_preprocessing);
    
    let mut rng = thread_rng();
    let num_iterations = 1000;
    let mut all_passed = true;
    let mut total_prove_time = 0.0;
    let mut total_verify_time = 0.0;
    
    for i in 0..num_iterations {
        // Generate random Fr elements
        let a_fr = Fr::rand(&mut rng);
        let b_fr = Fr::rand(&mut rng);
        
        // Convert Fr elements to limbs
        let a_bigint: BigInt<4> = a_fr.into();
        let b_bigint: BigInt<4> = b_fr.into();
        let a_limbs = a_bigint.0;
        let b_limbs = b_bigint.0;
        
        // Progress indicator every 100 iterations
        if i % 100 == 0 {
            println!("Progress: {}/{} iterations", i, num_iterations);
        }
        
        // Prove
        let prove_start = Instant::now();
        let ((res0, res1, res2, res3), proof, program_io) = prove_mul_assign(
            a_limbs[0], a_limbs[1], a_limbs[2], a_limbs[3],
            b_limbs[0], b_limbs[1], b_limbs[2], b_limbs[3]
        );
        let prove_time = prove_start.elapsed().as_secs_f64();
        total_prove_time += prove_time;
        
        // Verify
        let verify_start = Instant::now();
        let is_valid = verify_mul_assign(
            a_limbs[0], a_limbs[1], a_limbs[2], a_limbs[3],
            b_limbs[0], b_limbs[1], b_limbs[2], b_limbs[3],
            (res0, res1, res2, res3),
            program_io.panic,
            proof
        );
        let verify_time = verify_start.elapsed().as_secs_f64();
        total_verify_time += verify_time;
        
        // Convert result back to Fr
        let result_bigint = BigInt::<4>::new([res0, res1, res2, res3]);
        let result_fr = Fr::from(result_bigint);
        
        let expected = a_fr * b_fr;
        
        // Check results
        if !is_valid {
            println!("❌ Iteration {}: Invalid proof!", i);
            all_passed = false;
        }
        
        if result_fr != expected {
            println!("❌ Iteration {}: Result mismatch!", i);
            println!("  Expected: {:?}", expected);
            println!("  Got: {:?}", result_fr);
            all_passed = false;
        }
        
        // Optional: Show details for failures or first few iterations
        if i < 3 || (!is_valid || result_fr != expected) {
            println!("Iteration {}:", i);
            println!("  Input A: {:?}", a_fr);
            println!("  Input B: {:?}", b_fr);
            println!("  Result: {:?}", result_fr);
            println!("  Valid: {}, Correct: {}", is_valid, result_fr == expected);
        }
    }
    
    // Summary statistics
    println!("\n=== Summary ===");
    println!("Total iterations: {}", num_iterations);
    if all_passed {
        println!("✅ All {} tests passed!", num_iterations);
    } else {
        println!("❌ Some tests failed!");
    }
    println!("Average prove time: {:.4} s", total_prove_time / num_iterations as f64);
    println!("Average verify time: {:.4} s", total_verify_time / num_iterations as f64);
    println!("Total runtime: {:.2} s", total_prove_time + total_verify_time);
}

/// Test edge cases for Montgomery mul_assign
fn test_edge_cases() {
    println!("=== Testing Montgomery mul_assign with edge cases ===");
    
    // Setup once
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_benchmark_mul_assign(target_dir);
    
    let prover_preprocessing = guest::preprocess_prover_benchmark_mul_assign(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_benchmark_mul_assign(&prover_preprocessing);
    
    let prove_mul_assign = guest::build_prover_benchmark_mul_assign(program, prover_preprocessing);
    let verify_mul_assign = guest::build_verifier_benchmark_mul_assign(verifier_preprocessing);
    
    // Define edge cases
    let mut test_cases: Vec<(Fr, Fr, &str)> = Vec::new();
    
    // 1. Zero cases
    test_cases.push((Fr::zero(), Fr::zero(), "zero * zero"));
    test_cases.push((Fr::zero(), Fr::one(), "zero * one"));
    test_cases.push((Fr::one(), Fr::zero(), "one * zero"));
    
    // 2. One cases
    test_cases.push((Fr::one(), Fr::one(), "one * one"));
    
    // 3. Maximum field element (modulus - 1)
    let max_fr = Fr::zero() - Fr::one();  // This gives us modulus - 1
    test_cases.push((max_fr, Fr::one(), "max * one"));
    test_cases.push((max_fr, max_fr, "max * max"));
    test_cases.push((max_fr, Fr::from(2u64), "max * 2"));
    
    // 4. Small values
    test_cases.push((Fr::from(2u64), Fr::from(3u64), "2 * 3"));
    test_cases.push((Fr::from(255u64), Fr::from(256u64), "255 * 256"));
    
    // 5. Powers of 2
    test_cases.push((Fr::from(1u64 << 32), Fr::from(1u64 << 32), "2^32 * 2^32"));
    test_cases.push((Fr::from(1u64 << 63), Fr::from(2u64), "(2^63) * 2"));
    
    // 6. Values that will definitely wrap around the modulus
    let large_val = Fr::from(u64::MAX);
    test_cases.push((large_val, large_val, "u64::MAX * u64::MAX"));
    
    // 7. Inverses (a * a^-1 should equal 1 after all the operations)
    let some_val = Fr::from(12345u64);
    let inv_val = some_val.inverse().unwrap();
    test_cases.push((some_val, inv_val, "value * its_inverse"));
    
    // 8. Half of modulus
    let half_mod = max_fr / Fr::from(2u64);
    test_cases.push((half_mod, Fr::from(2u64), "half_modulus * 2"));
    test_cases.push((half_mod, half_mod, "half_modulus * half_modulus"));
    
    // 9. Values close to modulus
    let near_mod = max_fr - Fr::from(100u64);
    test_cases.push((near_mod, Fr::from(2u64), "(modulus-100) * 2"));
    test_cases.push((near_mod, near_mod, "(modulus-100) * (modulus-100)"));
    
    // 10. Square roots of max (their product should wrap)
    let sqrt_max = max_fr.sqrt().unwrap_or(Fr::from(2u64).pow(&[128u64]));
    test_cases.push((sqrt_max, sqrt_max, "sqrt(max) * sqrt(max)"));
    
    // 11. Negative one (-1 in field arithmetic)
    let neg_one = Fr::zero() - Fr::one();
    test_cases.push((neg_one, neg_one, "(-1) * (-1)"));
    test_cases.push((neg_one, Fr::from(2u64), "(-1) * 2"));
    
    // 12. Special values that test Montgomery reduction edge cases
    // These are designed to stress the reduction algorithm
    let special_val1 = Fr::from((1u64 << 62) - 1);
    let special_val2 = Fr::from((1u64 << 61) + 1);
    test_cases.push((special_val1, special_val2, "(2^62-1) * (2^61+1)"));
    
    // 13. Testing with values that have specific bit patterns
    let all_ones_32 = Fr::from(0xFFFFFFFFu64);
    test_cases.push((all_ones_32, all_ones_32, "0xFFFFFFFF * 0xFFFFFFFF"));
    
    // 14. Random large values that will cause multiple reductions
    let mut rng = thread_rng();
    for i in 0..3 {
        let a = Fr::rand(&mut rng);
        let b = Fr::rand(&mut rng);
        // Scale them up to be large
        let a_large = a * max_fr;
        let b_large = b * max_fr;
        test_cases.push((a_large, b_large, 
            if i == 0 { "random_large_1 * random_large_1" }
            else if i == 1 { "random_large_2 * random_large_2" }
            else { "random_large_3 * random_large_3" }
        ));
    }
    
    // Run all edge case tests
    let mut all_passed = true;
    let mut total_prove_time = 0.0;
    let mut total_verify_time = 0.0;
    
    for (i, (a_fr, b_fr, description)) in test_cases.iter().enumerate() {
        println!("\nTest {}: {}", i + 1, description);
        println!("  Input A: {:?}", a_fr);
        println!("  Input B: {:?}", b_fr);
        
        // Convert Fr elements to limbs
        let a_bigint: BigInt<4> = (*a_fr).into();
        let b_bigint: BigInt<4> = (*b_fr).into();
        let a_limbs = a_bigint.0;
        let b_limbs = b_bigint.0;
        
        // Prove
        let prove_start = Instant::now();
        let ((res0, res1, res2, res3), proof, program_io) = prove_mul_assign(
            a_limbs[0], a_limbs[1], a_limbs[2], a_limbs[3],
            b_limbs[0], b_limbs[1], b_limbs[2], b_limbs[3]
        );
        let prove_time = prove_start.elapsed().as_secs_f64();
        total_prove_time += prove_time;
        
        // Verify
        let verify_start = Instant::now();
        let is_valid = verify_mul_assign(
            a_limbs[0], a_limbs[1], a_limbs[2], a_limbs[3],
            b_limbs[0], b_limbs[1], b_limbs[2], b_limbs[3],
            (res0, res1, res2, res3),
            program_io.panic,
            proof
        );
        let verify_time = verify_start.elapsed().as_secs_f64();
        total_verify_time += verify_time;
        
        // Convert result back to Fr
        let result_bigint = BigInt::<4>::new([res0, res1, res2, res3]);
        let result_fr = Fr::from(result_bigint);
        
        // Compute expected result
        // The guest code does: a *= b (which is just a * b in field arithmetic)
        let expected = *a_fr * *b_fr;
        
        // Check results
        if !is_valid {
            println!("  ❌ Invalid proof!");
            all_passed = false;
        } else {
            println!("  ✅ Valid proof");
        }
        
        if result_fr != expected {
            println!("  ❌ Result mismatch!");
            println!("    Expected: {:?}", expected);
            println!("    Got: {:?}", result_fr);
            all_passed = false;
        } else {
            println!("  ✅ Correct result");
        }
        
        println!("  Prove time: {:.4} s, Verify time: {:.4} s", prove_time, verify_time);
    }
    
    // Summary
    println!("\n=== Edge Cases Summary ===");
    println!("Total edge case tests: {}", test_cases.len());
    if all_passed {
        println!("✅ All edge case tests passed!");
    } else {
        println!("❌ Some edge case tests failed!");
    }
    println!("Average prove time: {:.4} s", total_prove_time / test_cases.len() as f64);
    println!("Average verify time: {:.4} s", total_verify_time / test_cases.len() as f64);
    println!("Total runtime: {:.2} s", total_prove_time + total_verify_time);
}

/// Main function that runs both random and edge case tests
pub fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║     Montgomery Multiplication (mul_assign) Test Suite     ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
    
    // Run edge case tests first (they're faster and more targeted)
    let edge_start = Instant::now();
    test_edge_cases();
    let edge_time = edge_start.elapsed().as_secs_f64();
    
    println!("\n{}\n", "=".repeat(60));
    
    // Then run random tests
    let random_start = Instant::now();
    test_random_fr_elements();
    let random_time = random_start.elapsed().as_secs_f64();
    
    // Overall summary
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║                    OVERALL SUMMARY                        ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("Edge case tests runtime: {:.2} s", edge_time);
    println!("Random tests runtime: {:.2} s", random_time);
    println!("Total test suite runtime: {:.2} s", edge_time + random_time);
}
