//! Export assertion values from Rust verifier to JSON
//!
//! This binary runs verify_real with debug output and parses the output to extract
//! the assertion values, then exports them to a JSON file.
//!
//! Usage:
//!   cargo run -p gnark-transpiler --bin export_assertions

use std::process::Command;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct SumcheckAssertion {
    sumcheck_index: usize,
    output_claim: String,
    expected_output_claim: String,
    difference: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct AssertionExport {
    source: String,
    sumcheck_assertions: Vec<SumcheckAssertion>,
    all_pass: bool,
}

fn main() {
    println!("=== Exporting Rust Assertion Values ===\n");

    // Run verify_real with debug output and capture stderr
    println!("Running verify_real with debug-expected-output...");

    let output = Command::new("cargo")
        .args([
            "run",
            "-p", "gnark-transpiler",
            "--bin", "verify_real",
            "--features", "debug-expected-output",
        ])
        .output()
        .expect("Failed to run verify_real");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Parse the debug output to extract assertion values
    let mut assertions: Vec<SumcheckAssertion> = Vec::new();
    let mut current_output_claim: Option<String> = None;
    let mut sumcheck_index = 0;

    for line in stderr.lines() {
        if line.contains("output_claim (from sumcheck) = ") {
            let value = line.split(" = ").nth(1).unwrap_or("").to_string();
            current_output_claim = Some(value);
        } else if line.contains("expected_output_claim (batched) = ") {
            if let Some(output_claim) = current_output_claim.take() {
                let expected = line.split(" = ").nth(1).unwrap_or("").to_string();

                // Calculate difference (should be 0)
                let diff = if output_claim == expected {
                    "0".to_string()
                } else {
                    "MISMATCH".to_string()
                };

                assertions.push(SumcheckAssertion {
                    sumcheck_index,
                    output_claim,
                    expected_output_claim: expected,
                    difference: diff,
                });
                sumcheck_index += 1;
            }
        }
    }

    let all_pass = assertions.iter().all(|a| a.difference == "0");

    let export = AssertionExport {
        source: "rust_verify_real".to_string(),
        sumcheck_assertions: assertions,
        all_pass,
    };

    // Print summary
    println!("\nExtracted {} sumcheck assertions:", export.sumcheck_assertions.len());
    for assertion in &export.sumcheck_assertions {
        let status = if assertion.difference == "0" { "✓" } else { "✗" };
        println!("  Sumcheck {}: {} (diff: {})", assertion.sumcheck_index + 1, status, assertion.difference);
    }

    // Export to JSON
    let json = serde_json::to_string_pretty(&export).expect("Failed to serialize JSON");
    let output_path = concat!(env!("CARGO_MANIFEST_DIR"), "/go/rust_assertion_values.json");
    std::fs::write(output_path, &json).expect("Failed to write JSON file");

    println!("\nAssertion values exported to: {}", output_path);
    println!("\nJSON content:");
    println!("{}", json);

    if !all_pass {
        eprintln!("\nWARNING: Some assertions have mismatches!");
        std::process::exit(1);
    }

    println!("\n✓ All assertions passed (output_claim == expected_output_claim)");
}
