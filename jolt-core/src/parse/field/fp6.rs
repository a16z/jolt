#![allow(unused_imports)]
use ark_bn254::{Fq, Fq2, Fq6};
use ark_ff::{CyclotomicMultSubgroup, Field, Fp6, PrimeField};
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

fn string_for_fp2_input(input: &str) -> String {
    // Define regex to capture the x and y parts of the QuadExtField
    let re = Regex::new(r#"QuadExtField\((\d+) \+ (\d+) \* u\)"#).unwrap();

    // Replace matched patterns with the desired format
    let result = re.replace_all(input, |caps: &regex::Captures| {
        let x = &caps[1]; // First capture group (x)
        let y = &caps[2]; // Second capture group (y)

        // Format the replacement string as per desired output
        format!("{{\"x\": \"{}\", \"y\": \"{}\"}}", x, y)
    });

    result.to_string()
}

fn string_for_fp6_input(input: &str) -> String {
    // Define regex to capture the x and y parts of the QuadExtField
    let re = Regex::new(r#""CubicExtField\(\{([^}]+)\}, \{([^}]+)\}, \{([^}]+)\}\)""#).unwrap();

    // Replace matched patterns with the desired format
    let result = re.replace_all(input, |caps: &regex::Captures| {
        let x = &caps[1]; // First capture group (x)
        let y = &caps[2]; // Second capture group (y)
        let z = &caps[3];
        format!(r#"{{"x": {{{}}}, "y": {{{}}}, "z": {{{}}}}}"#, x, y, z)
    });

    result.to_string()
}

#[test]
fn test_Fp6add() {
    let mut rng = rand::thread_rng();

    let a = Fq6::rand(&mut rng);
    let b = Fq6::rand(&mut rng);
    let sum = a + b;

    let input_json = format!(
        r#"{{
    "op1": "{}",
    "op2": "{}"
}}"#,
        a, b
    );

    let input_json = string_for_fp2_input(&input_json);
    let input_json = string_for_fp6_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Fp6add";
    let circom_file_path = "./fp6.circom";
    let circom_file = "fp6";

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
            .take(6)
            .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
            .collect();

        let result: Vec<BigUint> = result
            .into_iter()
            .map(|entry| {
                BigUint::parse_bytes(entry.as_bytes(), 10)
                    .expect("Failed to parse the string into a BigUint")
            })
            .collect();

        // let result = Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone()));

        let result = Fp6::new(
            Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone())),
            Fq2::new(Fq::from(result[2].clone()), Fq::from(result[3].clone())),
            Fq2::new(Fq::from(result[4].clone()), Fq::from(result[5].clone())),
        );
        assert_eq!(result, sum);

        println!("Sum: {}", sum);
        println!("Result: {} \n", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_Fp6sub() {
    let mut rng = rand::thread_rng();

    let a = Fq6::rand(&mut rng);
    let b = Fq6::rand(&mut rng);
    let diff = a - b;

    let input_json = format!(
        r#"{{
    "op1": "{}",
    "op2": "{}"
}}"#,
        a, b
    );

    let input_json = string_for_fp2_input(&input_json);
    let input_json = string_for_fp6_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Fp6sub";
    let circom_file_path = "./fp6.circom";
    let circom_file = "fp6";

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
            .take(6)
            .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
            .collect();

        let result: Vec<BigUint> = result
            .into_iter()
            .map(|entry| {
                BigUint::parse_bytes(entry.as_bytes(), 10)
                    .expect("Failed to parse the string into a BigUint")
            })
            .collect();

        // let result = Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone()));

        let result = Fp6::new(
            Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone())),
            Fq2::new(Fq::from(result[2].clone()), Fq::from(result[3].clone())),
            Fq2::new(Fq::from(result[4].clone()), Fq::from(result[5].clone())),
        );
        assert_eq!(result, diff);

        println!("Diff: {}", diff);
        println!("Result: {} \n", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_Fp6mul() {
    let mut rng = thread_rng();

    let a = Fq6::rand(&mut rng);
    let b = Fq6::rand(&mut rng);
    let prod = a * b;

    let input_json = format!(
        r#"{{
    "op1": "{}",
    "op2": "{}"
}}"#,
        a, b
    );

    let input_json = string_for_fp2_input(&input_json);
    let input_json = string_for_fp6_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Fp6mul";
    let circom_file_path = "./fp6.circom";
    let circom_file = "fp6";

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
            .take(6)
            .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
            .collect();

        let result: Vec<BigUint> = result
            .into_iter()
            .map(|entry| {
                BigUint::parse_bytes(entry.as_bytes(), 10)
                    .expect("Failed to parse the string into a BigUint")
            })
            .collect();

        // let result = Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone()));

        let result = Fp6::new(
            Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone())),
            Fq2::new(Fq::from(result[2].clone()), Fq::from(result[3].clone())),
            Fq2::new(Fq::from(result[4].clone()), Fq::from(result[5].clone())),
        );
        assert_eq!(result, prod);

        println!("Product: {}", prod);
        println!("Result: {} \n", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_Fp6neg() {
    let mut rng = rand::thread_rng();

    let a = Fq6::rand(&mut rng);

    let input_json = format!(
        r#"{{
    "op1": "{}"
}}"#,
        a
    );

    let input_json = string_for_fp2_input(&input_json);
    let input_json = string_for_fp6_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Fp6neg";
    let circom_file_path = "./fp6.circom";
    let circom_file = "fp6";

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
            .take(6)
            .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
            .collect();

        let result: Vec<BigUint> = result
            .into_iter()
            .map(|entry| {
                BigUint::parse_bytes(entry.as_bytes(), 10)
                    .expect("Failed to parse the string into a BigUint")
            })
            .collect();

        // let result = Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone()));

        let result = Fp6::new(
            Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone())),
            Fq2::new(Fq::from(result[2].clone()), Fq::from(result[3].clone())),
            Fq2::new(Fq::from(result[4].clone()), Fq::from(result[5].clone())),
        );
        assert_eq!(result, -a);

        println!("Neg: {}", -a);
        println!("Result: {} \n", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_Fp6inv() {
    let mut rng = thread_rng(); //ark_std::test_rng();

    // let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    // rng.reseed(&now);

    let a = Fq6::rand(&mut rng);

    println!("a: {}", a);

    let input_json = format!(
        r#"{{
    "op1": "{}"
}}"#,
        a
    );

    let input_json = string_for_fp2_input(&input_json);
    let input_json = string_for_fp6_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Fp6inv";
    let circom_file_path = "./fp6.circom";
    let circom_file = "fp6";

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
            .take(6)
            .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
            .collect();

        let result: Vec<BigUint> = result
            .into_iter()
            .map(|entry| {
                BigUint::parse_bytes(entry.as_bytes(), 10)
                    .expect("Failed to parse the string into a BigUint")
            })
            .collect();

        // let result = Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone()));
        println!("Result: {:?} \n", result);

        println!("Witness correct!");
        let result = Fp6::new(
            Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone())),
            Fq2::new(Fq::from(result[2].clone()), Fq::from(result[3].clone())),
            Fq2::new(Fq::from(result[4].clone()), Fq::from(result[5].clone())),
        );
        assert_eq!(result, a.inverse().unwrap());

        println!("Inverse: {}", a.inverse().unwrap());
        println!("Result: {} \n", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_all() {
    test_Fp6add();
    test_Fp6sub();
    test_Fp6mul();
    test_Fp6neg();
    test_Fp6inv();
}
