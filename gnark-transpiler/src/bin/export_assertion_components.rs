//! Export assertion component values from Rust for comparison with Go
//!
//! This binary extracts the same values that compose each assertion a0...a14
//! and exports them to JSON for comparison with the Go test.
//!
//! Usage:
//!   cargo run -p gnark-transpiler --bin export_assertion_components

use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_serialize::CanonicalDeserialize;
use jolt_core::zkvm::RV64IMACProof;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug)]
struct AssertionComponents {
    name: String,
    components: HashMap<String, String>,
    result: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct ComponentExport {
    source: String,
    assertions: Vec<AssertionComponents>,
}

fn fr_to_string(f: &Fr) -> String {
    format!("{}", f.into_bigint())
}

fn main() {
    println!("=== Exporting Rust Assertion Components ===\n");

    // Load proof
    let proof_path = "/tmp/fib_proof.bin";
    println!("Loading proof from: {}", proof_path);
    let proof_bytes = std::fs::read(proof_path).expect("Failed to read proof file");
    let proof: RV64IMACProof =
        CanonicalDeserialize::deserialize_compressed(&proof_bytes[..])
            .expect("Failed to deserialize proof");
    println!("  trace_length: {}", proof.trace_length);

    let mut assertions: Vec<AssertionComponents> = Vec::new();

    // === a0: Stage1 Univariate Skip ===
    println!("\n=== a0: Stage1 Univariate Skip ===");
    let power_sums_stage1: Vec<i64> = vec![
        10, 5, 85, 125, 1333, 3125, 25405, 78125, 535333, 1953125,
        11982925, 48828125, 278766133, 1220703125, 6649985245, 30517578125,
        161264049733, 762939453125, 3952911584365, 19073486328125,
        97573430562133, 476837158203125, 2419432933612285, 11920928955078125,
        60168159621439333, 298023223876953125, 1499128402505381005, 7450580596923828125,
    ];

    let mut a0_components = HashMap::new();
    let mut a0_result = Fr::from(0u64);

    let coeffs = &proof.stage1_uni_skip_first_round_proof.uni_poly.coeffs;
    for (i, coeff) in coeffs.iter().enumerate() {
        let field_name = format!("Stage1_Uni_Skip_Coeff_{}", i);
        let coeff_str = fr_to_string(coeff);
        a0_components.insert(field_name.clone(), coeff_str.clone());

        let power_sum = Fr::from(power_sums_stage1[i] as u64);
        let term = *coeff * power_sum;
        a0_result += term;

        println!("  {} = {} (× {})", field_name, &coeff_str[..40.min(coeff_str.len())], power_sums_stage1[i]);
    }
    println!("  a0 = {}", fr_to_string(&a0_result));

    assertions.push(AssertionComponents {
        name: "a0".to_string(),
        components: a0_components,
        result: fr_to_string(&a0_result),
    });

    // === a1: Stage1 Sumcheck (round coefficients) ===
    println!("\n=== a1: Stage1 Sumcheck ===");
    let mut a1_components = HashMap::new();

    for (round, poly) in proof.stage1_sumcheck_proof.compressed_polys.iter().enumerate() {
        for (j, coeff) in poly.coeffs_except_linear_term.iter().enumerate() {
            let field_name = format!("Stage1_Sumcheck_R{}_{}", round, j);
            let coeff_str = fr_to_string(coeff);
            a1_components.insert(field_name.clone(), coeff_str.clone());
            if round <= 3 || round >= 11 {
                println!("  {} = {}", field_name, &coeff_str[..30.min(coeff_str.len())]);
            } else if round == 4 {
                println!("  ... (rounds 4-10 omitted) ...");
            }
        }
    }

    assertions.push(AssertionComponents {
        name: "a1".to_string(),
        components: a1_components,
        result: "sumcheck_output".to_string(), // Complex computation
    });

    // === a2: Stage2 Univariate Skip ===
    println!("\n=== a2: Stage2 Univariate Skip ===");
    let mut a2_components = HashMap::new();

    let coeffs2 = &proof.stage2_uni_skip_first_round_proof.uni_poly.coeffs;
    for (i, coeff) in coeffs2.iter().enumerate() {
        let field_name = format!("Stage2_Uni_Skip_Coeff_{}", i);
        let coeff_str = fr_to_string(coeff);
        a2_components.insert(field_name.clone(), coeff_str.clone());
        println!("  {} = {}", field_name, &coeff_str[..40.min(coeff_str.len())]);
    }

    assertions.push(AssertionComponents {
        name: "a2".to_string(),
        components: a2_components,
        result: "see_circuit".to_string(),
    });

    // === a3: Stage2 Sumcheck ===
    println!("\n=== a3: Stage2 Sumcheck ===");
    let mut a3_components = HashMap::new();

    for (round, poly) in proof.stage2_sumcheck_proof.compressed_polys.iter().enumerate() {
        for (j, coeff) in poly.coeffs_except_linear_term.iter().enumerate() {
            let field_name = format!("Stage2_Sumcheck_R{}_{}", round, j);
            let coeff_str = fr_to_string(coeff);
            a3_components.insert(field_name.clone(), coeff_str.clone());
            if round <= 2 || round >= 23 {
                println!("  {} = {}", field_name, &coeff_str[..30.min(coeff_str.len())]);
            } else if round == 3 {
                println!("  ... (rounds 3-22 omitted) ...");
            }
        }
    }

    assertions.push(AssertionComponents {
        name: "a3".to_string(),
        components: a3_components,
        result: "sumcheck_output".to_string(),
    });

    // === a4: Stage3 Sumcheck ===
    println!("\n=== a4: Stage3 Sumcheck ===");
    let mut a4_components = HashMap::new();

    for (round, poly) in proof.stage3_sumcheck_proof.compressed_polys.iter().enumerate() {
        for (j, coeff) in poly.coeffs_except_linear_term.iter().enumerate() {
            let field_name = format!("Stage3_Sumcheck_R{}_{}", round, j);
            let coeff_str = fr_to_string(coeff);
            a4_components.insert(field_name.clone(), coeff_str.clone());
            if round <= 2 || round >= 10 {
                println!("  {} = {}", field_name, &coeff_str[..30.min(coeff_str.len())]);
            } else if round == 3 {
                println!("  ... (rounds 3-9 omitted) ...");
            }
        }
    }

    assertions.push(AssertionComponents {
        name: "a4".to_string(),
        components: a4_components,
        result: "sumcheck_output".to_string(),
    });

    // === a5-a8, a10-a11, a13: Claim differences ===
    // These come from opening_claims - extract the relevant ones
    println!("\n=== Claim differences (a5-a8, a10-a11, a13) ===");

    // Extract claim values from opening_claims
    for (key, (_point, claim)) in &proof.opening_claims.0 {
        let key_str = format!("{:?}", key);
        // Check if it's one of the claims we care about
        if key_str.contains("Rs1Value") || key_str.contains("Rs2Value") ||
           key_str.contains("LookupOutput") || key_str.contains("UnexpandedPC") {
            println!("  {:?} = {}", key, &fr_to_string(claim)[..40.min(fr_to_string(claim).len())]);
        }
    }

    // === a9: Stage4 Sumcheck ===
    println!("\n=== a9: Stage4 Sumcheck ===");
    let mut a9_components = HashMap::new();

    for (round, poly) in proof.stage4_sumcheck_proof.compressed_polys.iter().enumerate() {
        for (j, coeff) in poly.coeffs_except_linear_term.iter().enumerate() {
            let field_name = format!("Stage4_Sumcheck_R{}_{}", round, j);
            let coeff_str = fr_to_string(coeff);
            a9_components.insert(field_name.clone(), coeff_str.clone());
            if round <= 2 || round >= 17 {
                println!("  {} = {}", field_name, &coeff_str[..30.min(coeff_str.len())]);
            } else if round == 3 {
                println!("  ... (rounds 3-16 omitted) ...");
            }
        }
    }

    assertions.push(AssertionComponents {
        name: "a9".to_string(),
        components: a9_components,
        result: "sumcheck_output".to_string(),
    });

    // === a12: Stage5 Sumcheck ===
    println!("\n=== a12: Stage5 Sumcheck ===");
    let mut a12_components = HashMap::new();

    for (round, poly) in proof.stage5_sumcheck_proof.compressed_polys.iter().enumerate() {
        for (j, coeff) in poly.coeffs_except_linear_term.iter().enumerate() {
            let field_name = format!("Stage5_Sumcheck_R{}_{}", round, j);
            let coeff_str = fr_to_string(coeff);
            a12_components.insert(field_name.clone(), coeff_str.clone());
            if round <= 2 || round >= 138 {
                println!("  {} = {}", field_name, &coeff_str[..30.min(coeff_str.len())]);
            } else if round == 3 {
                println!("  ... (rounds 3-137 omitted) ...");
            }
        }
    }

    assertions.push(AssertionComponents {
        name: "a12".to_string(),
        components: a12_components,
        result: "sumcheck_output".to_string(),
    });

    // === a14: Stage6 Sumcheck ===
    println!("\n=== a14: Stage6 Sumcheck ===");
    let mut a14_components = HashMap::new();

    for (round, poly) in proof.stage6_sumcheck_proof.compressed_polys.iter().enumerate() {
        for (j, coeff) in poly.coeffs_except_linear_term.iter().enumerate() {
            let field_name = format!("Stage6_Sumcheck_R{}_{}", round, j);
            let coeff_str = fr_to_string(coeff);
            a14_components.insert(field_name.clone(), coeff_str.clone());
            if round <= 2 || round >= 23 {
                println!("  {} = {}", field_name, &coeff_str[..30.min(coeff_str.len())]);
            } else if round == 3 {
                println!("  ... (rounds 3-22 omitted) ...");
            }
        }
    }

    assertions.push(AssertionComponents {
        name: "a14".to_string(),
        components: a14_components,
        result: "sumcheck_output".to_string(),
    });

    // Export to JSON
    let export = ComponentExport {
        source: "rust_proof".to_string(),
        assertions,
    };

    let json = serde_json::to_string_pretty(&export).expect("Failed to serialize JSON");
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let output_path = format!("{}/go/rust_assertion_components.json", manifest_dir);
    std::fs::write(&output_path, &json).expect("Failed to write JSON");

    println!("\n=== Exported to {} ===", output_path);
    println!("\nRun in Go: go test -v -run TestCompareWithRust");
}
