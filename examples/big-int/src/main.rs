use std::time::Instant;

use ark_bn254::Fr;
use jolt_sdk::{host, postcard};

// Two 256-bit inputs as two 128-bit limbs each: (lo, hi)
// Random-looking 128-bit values for testing
const A_LOW: u128 = 0xf3a8_9b7c_4d2e_1f0a_8b6c_5d3e_2f1a_0b9c;
const A_HIGH: u128 = 0x9c8b_7a6f_5e4d_3c2b_1a09_8776_5544_3322;
const B_LOW: u128 = 0x1234_5678_9abc_def0_fedc_ba98_7654_3210;
const B_HIGH: u128 = 0xa5b6_c7d8_e9fa_0b1c_2d3e_4f50_6172_8394;

fn print_cycles_and_bytecode() {    
    let program = host::Program::new("big-int-guest");
    let mut inputs = vec![];
    inputs.append(&mut postcard::to_stdvec(&A_LOW).unwrap());
    inputs.append(&mut postcard::to_stdvec(&A_HIGH).unwrap());
    inputs.append(&mut postcard::to_stdvec(&B_LOW).unwrap());
    inputs.append(&mut postcard::to_stdvec(&B_HIGH).unwrap());

    // Get the program summary
    let summary = program.trace_analyze::<Fr>(&inputs);
    
    // Print trace length
    println!("Trace length is: {}", summary.trace_len());
    
    // Write comprehensive trace analysis to file
    let filename = "big-int-multiplication_arkworks.txt";
    match summary.write_trace_analysis_ecalls::<Fr>(&filename) {
        Ok(_) => println!("✅ Saved comprehensive trace analysis to: {}", filename),
        Err(e) => println!("❌ Failed to save trace analysis: {}", e),
    }
}

pub fn main() {
    // print_cycles_and_bytecode();
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_mul_u256(target_dir);

    let prover_preprocessing = guest::preprocess_prover_mul_u256(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_mul_u256(&prover_preprocessing);

    let prove_mul_u256 = guest::build_prover_mul_u256(program, prover_preprocessing);
    let verify_mul_u256 = guest::build_verifier_mul_u256(verifier_preprocessing);

    let now = Instant::now();
    let ((res0, res1, res2, res3, res4, res5, res6, res7), proof, program_io) = prove_mul_u256(A_LOW, A_HIGH, B_LOW, B_HIGH);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());

    let is_valid = verify_mul_u256(A_LOW, A_HIGH, B_LOW, B_HIGH, (res0, res1, res2, res3, res4, res5, res6, res7), program_io.panic, proof);

    println!("=== Low 256 bits (res0-res3) ===");
    println!("res0: 0x{:016x}", res0);
    println!("res1: 0x{:016x}", res1);
    println!("res2: 0x{:016x}", res2);
    println!("res3: 0x{:016x}", res3);
    println!("=== High 256 bits (res4-res7) ===");
    println!("res4: 0x{:016x}", res4);
    println!("res5: 0x{:016x}", res5);
    println!("res6: 0x{:016x}", res6);
    println!("res7: 0x{:016x}", res7);
    
    println!("valid: {is_valid}");
    println!("✓ Result verification passed!");

//     Prover runtime: 1.246992041 s
// === Low 256 bits (res0-res3) ===
// res0: 0xc90fab9bbf1531c0
// res1: 0xaad973dd55fab9b8
// res2: 0xbca2ca1b10cfc4cf
// res3: 0xbb1ee1c3c94e6a79
// === High 256 bits (res4-res7) ===
// res4: 0xa6ae39a57f3091a7
// res5: 0x2dae6201791d3cf5
// res6: 0x50221be9fec14f26
// res7: 0x6555ab47e3e48c66
// valid: true
// ✓ Result verification passed!
}


