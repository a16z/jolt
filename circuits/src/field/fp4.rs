#![allow(unused_imports)]
use ark_bn254::{Fq, Fq2};
use ark_ff::Fp2;
use ark_std::{One, UniformRand};
use matrix::check_inner_product;
use num_bigint::BigUint;
use rand::thread_rng;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::convert::TryInto;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::process::Command;

fn parse_bn254_modulus() -> BigUint {
    let modulus_str =
        "21888242871839275222246405745257275088696311157297823662689037894645226208583";
    let modulus = BigUint::parse_bytes(modulus_str.as_bytes(), 10)
        .expect("Failed to parse the string into a BigUint");
    modulus
}

fn string_for_fp2_input(input: &str) -> String {
    // Define regex to capture the x and y parts of the QuadExtField
    let re = Regex::new(r#"\"QuadExtField\((\d+) \+ (\d+) \* u\)\""#).unwrap();

    // Replace matched patterns with the desired format
    let result = re.replace_all(input, |caps: &regex::Captures| {
        let x = &caps[1]; // First capture group (x)
        let y = &caps[2]; // Second capture group (y)

        // Format the replacement string as per desired output
        format!("{{\"x\": \"{}\", \"y\": \"{}\"}}", x, y)
    });

    result.to_string()
}

#[test]
fn test_Fp4square() {
    let mut rng = rand::thread_rng();

    let a: ark_ff::QuadExtField<ark_ff::Fp2ConfigWrapper<ark_bn254::Fq2Config>> =
        Fp2::rand(&mut rng);
    let b: ark_ff::QuadExtField<ark_ff::Fp2ConfigWrapper<ark_bn254::Fq2Config>> =
        Fp2::rand(&mut rng);

    let input_json = format!(
        r#"{{
    "op1": {{
    "x": "{}",
    "y": "{}"
    }}
}}"#,
        a, b
    );

    let t0 = a * a;
    let t1 = b * b;
    let shi = Fq2::new(Fq::from(9), Fq::from(1));
    let out0 = t1 * shi + t0;
    let c1 = a + b;
    let c2 = c1 * c1;
    let c3 = c2 - t0;
    let out1 = c3 - t1;

    let input_json = string_for_fp2_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Fp4Square";
    let circom_file_path = "./fp4.circom";
    let circom_file = "fp4";

    let output = Command::new(script_path)
        .args(&[
            circom_file_path,
            circom_template,
            input_file_path,
            circom_file,
        ])
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

    if let Some(witness_array) = witness_json.as_array() {
        let result: Vec<String> = witness_array
            .iter()
            .skip(1)
            .take(4)
            .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
            .collect();

        let result: Vec<BigUint> = result
            .into_iter()
            .map(|entry| {
                BigUint::parse_bytes(entry.as_bytes(), 10)
                    .expect("Failed to parse the string into a BigUint")
            })
            .collect();

        let r0 = Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone()));
        let r1 = Fq2::new(Fq::from(result[2].clone()), Fq::from(result[3].clone()));

        assert_eq!(r0, out0);
        assert_eq!(r1, out1);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}
