//! Cross-validation test: compare Rust AST evaluation against Go gnark circuit.
//!
//! Each system computes 20 assertions with LHS and RHS = 40 values per side.
//! The test makes 40 comparisons: Rust_LHS==Go_LHS and Rust_RHS==Go_RHS for each assertion.
//!
//! Prerequisites:
//! - Proof files in /tmp (from: cargo run -p fibonacci --features transcript-poseidon -- --save 50)
//! - Generated circuit files (from: cargo run -p transpiler --bin transpiler --features transcript-poseidon)

use ark_bn254::Fr;
use std::collections::HashMap;
use std::str::FromStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use transpiler::ast_evaluator::{evaluate_assertions, fr_to_decimal};
use transpiler::gnark_codegen::sanitize_go_name;
use zklean_extractor::AstBundle;

fn go_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("go")
}

fn bundle_path() -> PathBuf {
    go_dir().join("stages_bundle.json")
}

fn witness_path() -> PathBuf {
    go_dir().join("stages_witness.json")
}

/// Load witness JSON and map to Var indices using the bundle's input descriptions.
fn load_witness(bundle: &AstBundle, witness_path: &Path) -> HashMap<u16, Fr> {
    let json_str = std::fs::read_to_string(witness_path).expect("failed to read witness JSON");
    let witness_map: HashMap<String, String> =
        serde_json::from_str(&json_str).expect("failed to parse witness JSON");

    let mut result = HashMap::new();
    for input in &bundle.inputs {
        let go_name = sanitize_go_name(&input.name);
        if let Some(value_str) = witness_map.get(&go_name) {
            let fr = Fr::from_str(value_str).unwrap_or_else(|_| {
                panic!(
                    "failed to parse Fr for {} = {}",
                    go_name,
                    &value_str[..value_str.len().min(40)]
                )
            });
            result.insert(input.index, fr);
        } else {
            panic!("witness missing value for input '{}' (go: '{}')", input.name, go_name);
        }
    }
    result
}

/// Run the Go crossval test and return the assertion values from JSON.
fn run_go_crossval() -> Vec<GoAssertionValue> {
    let go_dir = go_dir();
    let crossval_dir = go_dir.join("crossval");

    // Run go test in crossval sub-package
    let output = Command::new("go")
        .args(["test", "-run", "TestCrossValidation", "-v", "-count=1", "-timeout", "10m"])
        .current_dir(&crossval_dir)
        .output()
        .expect("failed to run go test");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        panic!(
            "Go crossval test failed!\nstdout:\n{}\nstderr:\n{}",
            stdout, stderr
        );
    }

    // Read the JSON output
    let json_path = go_dir.join("go_crossval_values.json");
    let json_str = std::fs::read_to_string(&json_path).unwrap_or_else(|e| {
        panic!(
            "failed to read go_crossval_values.json: {}\nGo stdout:\n{}",
            e, stdout
        )
    });

    serde_json::from_str(&json_str).expect("failed to parse go_crossval_values.json")
}

#[derive(serde::Deserialize, Debug)]
struct GoAssertionValue {
    name: String,
    lhs: String,
    rhs: String,
}

#[test]
fn test_cross_validation_80_values() {
    // Skip if bundle doesn't exist (not generated yet)
    if !bundle_path().exists() {
        eprintln!("Skipping: {} not found. Run transpiler first.", bundle_path().display());
        return;
    }

    println!("=== Loading AstBundle ===");
    let bundle = AstBundle::read_json(&bundle_path()).expect("failed to load bundle");
    println!(
        "  {} nodes, {} constraints, {} inputs",
        bundle.nodes.len(),
        bundle.constraints.len(),
        bundle.inputs.len()
    );

    println!("\n=== Loading Witness ===");
    let witness = load_witness(&bundle, &witness_path());
    println!("  {} witness values loaded", witness.len());

    // === Rust side: evaluate AST ===
    println!("\n=== Evaluating AST (Rust) ===");
    let rust_values = evaluate_assertions(&bundle.nodes, &bundle.constraints, &witness);
    println!("  {} assertions evaluated", rust_values.len());

    // Build serializable Rust values and persist to JSON
    let rust_json_values: Vec<serde_json::Value> = rust_values
        .iter()
        .map(|v| {
            serde_json::json!({
                "name": v.name,
                "lhs": fr_to_decimal(&v.lhs),
                "rhs": fr_to_decimal(&v.rhs),
            })
        })
        .collect();
    let rust_json_path = go_dir().join("rust_crossval_values.json");
    let rust_json_bytes = serde_json::to_string_pretty(&rust_json_values).unwrap();
    std::fs::write(&rust_json_path, &rust_json_bytes).expect("failed to write rust_crossval_values.json");
    println!("  Wrote Rust values to {}", rust_json_path.display());

    for v in &rust_values {
        let lhs = fr_to_decimal(&v.lhs);
        let rhs = fr_to_decimal(&v.rhs);
        println!(
            "  {}: LHS={} RHS={}",
            v.name,
            &lhs[..lhs.len().min(40)],
            &rhs[..rhs.len().min(40)],
        );
    }

    // === Go side: run crossval test ===
    println!("\n=== Running Go Crossval Test ===");
    let go_values = run_go_crossval();
    println!("  {} assertions from Go", go_values.len());

    // === Compare all 40 pairs (20 LHS + 20 RHS) ===
    println!("\n=== Comparing 40 Value Pairs (20 LHS + 20 RHS) ===");
    assert_eq!(
        rust_values.len(),
        go_values.len(),
        "assertion count mismatch: Rust={} Go={}",
        rust_values.len(),
        go_values.len()
    );

    let mut pass_count = 0;
    let mut fail_count = 0;
    let total_comparisons = rust_values.len() * 2; // LHS + RHS per assertion

    for (i, (rust_val, go_val)) in rust_values.iter().zip(go_values.iter()).enumerate() {
        // Names differ in format (assertion_0 vs a0) but must refer to the same index
        assert!(
            go_val.name == format!("a{i}"),
            "Go assertion order mismatch at index {i}: got {}",
            go_val.name
        );
        let rust_lhs = fr_to_decimal(&rust_val.lhs);
        let rust_rhs = fr_to_decimal(&rust_val.rhs);
        let go_lhs = &go_val.lhs;
        let go_rhs = &go_val.rhs;

        let lhs_match = rust_lhs == *go_lhs;
        let rhs_match = rust_rhs == *go_rhs;

        if lhs_match && rhs_match {
            println!("  {} LHS=MATCH RHS=MATCH", rust_val.name);
            pass_count += 2;
        } else {
            if !lhs_match {
                println!(
                    "  {} LHS MISMATCH: Rust={} Go={}",
                    rust_val.name,
                    &rust_lhs[..rust_lhs.len().min(50)],
                    &go_lhs[..go_lhs.len().min(50)]
                );
                fail_count += 1;
            } else {
                pass_count += 1;
            }
            if !rhs_match {
                println!(
                    "  {} RHS MISMATCH: Rust={} Go={}",
                    rust_val.name,
                    &rust_rhs[..rust_rhs.len().min(50)],
                    &go_rhs[..go_rhs.len().min(50)]
                );
                fail_count += 1;
            } else {
                pass_count += 1;
            }
        }
    }

    println!("\n=== RESULTS ===");
    println!("  {}/{} comparisons PASS", pass_count, total_comparisons);
    if fail_count > 0 {
        println!("  {}/{} comparisons FAIL", fail_count, total_comparisons);
    }

    assert_eq!(fail_count, 0, "{} of {} comparisons mismatched!", fail_count, total_comparisons);

    // === Generate report ===
    let report_path = go_dir().join("crossval_report.txt");
    let mut report = String::new();
    report.push_str("=== Cross-Validation Report ===\n\n");
    report.push_str(&format!("Bundle: {} nodes, {} constraints, {} inputs\n",
        bundle.nodes.len(), bundle.constraints.len(), bundle.inputs.len()));
    report.push_str(&format!("Witness: {} values\n\n", witness.len()));

    report.push_str(&format!("{:<6} {:<6} {:<80} {:<80}\n", "NAME", "SIDE", "RUST", "GO"));
    report.push_str(&format!("{}\n", "-".repeat(174)));

    for (rust_val, go_val) in rust_values.iter().zip(go_values.iter()) {
        let rust_lhs = fr_to_decimal(&rust_val.lhs);
        let rust_rhs = fr_to_decimal(&rust_val.rhs);
        let go_lhs = &go_val.lhs;
        let go_rhs = &go_val.rhs;

        let lhs_ok = if rust_lhs == *go_lhs { "==" } else { "!=" };
        let rhs_ok = if rust_rhs == *go_rhs { "==" } else { "!=" };

        report.push_str(&format!("{:<6} LHS {} {:<80} {:<80}\n", rust_val.name, lhs_ok, rust_lhs, go_lhs));
        report.push_str(&format!("{:<6} RHS {} {:<80} {:<80}\n", "", rhs_ok, rust_rhs, go_rhs));
    }

    report.push_str(&format!("\nResult: {}/{} PASS, {}/{} FAIL\n",
        pass_count, total_comparisons, fail_count, total_comparisons));

    std::fs::write(&report_path, &report).expect("failed to write report");
    println!("\nReport written to {}", report_path.display());
}
