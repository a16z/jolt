use std::{fs::File, io::Read, process::Command};

use serde_json::Value;

#[test]
fn test_jolt2_parsing(){
    let input_file_path = "input.json";
    let script_path = "./../../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "PairingCheck";
    let circom_file_path = "./../jolt2.circom";
    let circom_file = "jolt2";

    let mut params: [usize; 9] = [4, 54, 26,
        4, 4, 4, 11, 4, 16];

    let mut args = [circom_file_path,
    circom_template,
    input_file_path,
    circom_file].to_vec();

    let mut param_strings: Vec<String> = Vec::new();
    for i in 0..params.len() {
        param_strings.push(params[i].to_string());
    }
    for param_str in &param_strings {
        args.push(param_str);
    }

    let output = Command::new(script_path)
    .args(&args)
    .output()
    .expect("Failed to execute shell script");

    if !output.status.success() {
        panic!(
            "Shell script execution failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    println!(
        "Shell script output: {}",
        String::from_utf8_lossy(&output.stdout)
    );

    // Step 4: Verify witness generation
    println!("Witness generated successfully.");
    // Read the witness.json file
    let witness_file_path = "witness.json"; // Make sure this is the correct path to your witness.json
    let mut witness_file = File::open(witness_file_path).expect("Failed to open witness.json");

    // Read the file contents
    let mut witness_contents = String::new();
    witness_file
        .read_to_string(&mut witness_contents)
        .expect("Failed to read witness.json");

    // Parse the JSON contents
    let witness_json: Value =
        serde_json::from_str(&witness_contents).expect("Failed to parse witness.json");
}