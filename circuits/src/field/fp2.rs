#![allow(unused_imports)]
use ark_bn254::{Fq, Fq2};
use ark_ff::Field;
use ark_std::{One, UniformRand};
use matrix::check_inner_product;
use num_bigint::BigUint;
use rand::{Rng, thread_rng};
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
fn test_Fp2add() {
    let mut rng = rand::thread_rng();
    let a = Fq2::rand(&mut rng);
    let b = Fq2::rand(&mut rng);
    let sum = a + b;

    let input_json = format!(
        r#"{{
    "op1": "{}",
    "op2": "{}"
}}"#,
        a, b
    );

    let input_json = string_for_fp2_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Fp2add";
    let circom_file_path = "./fp2.circom";
    let circom_file = "fp2";

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
            .take(2)
            .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
            .collect();

        let result: Vec<BigUint> = result
            .into_iter()
            .map(|entry| {
                BigUint::parse_bytes(entry.as_bytes(), 10)
                    .expect("Failed to parse the string into a BigUint")
            })
            .collect();

        let result = Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone()));

        assert_eq!(result, sum);

        println!("Sum: {}", sum);
        println!("Result: {}", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_Fp2sub() {
    let mut rng = rand::thread_rng();

    let a = Fq2::rand(&mut rng);
    let b = Fq2::rand(&mut rng);
    let diff = a - b;

    let input_json = format!(
        r#"{{
    "op1": "{}",
    "op2": "{}"
}}"#,
        a, b
    );

    let input_json = string_for_fp2_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Fp2sub";
    let circom_file_path = "./fp2.circom";
    let circom_file = "fp2";

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
            .take(2)
            .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
            .collect();

        let result: Vec<BigUint> = result
            .into_iter()
            .map(|entry| {
                BigUint::parse_bytes(entry.as_bytes(), 10)
                    .expect("Failed to parse the string into a BigUint")
            })
            .collect();

        let result = Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone()));

        assert_eq!(result, diff);

        println!("Difference: {}", diff);
        println!("Result: {}", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_Fp2mulbyfp() {
    let mut rng = rand::thread_rng();

    let a = Fq::rand(&mut rng);
    let b = Fq2::rand(&mut rng);
    // let prod = b.clone().mul_assign_by_basefield(&a);
    let mut prod = b;
    prod.mul_assign_by_basefield(&a);

    let input_json = format!(
        r#"{{
    "op1": "{}",
    "op2": "{}"
}}"#,
        a, b
    );

    let input_json = string_for_fp2_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Fp2mulbyfp";
    let circom_file_path = "./fp2.circom";
    let circom_file = "fp2";

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
            .take(2)
            .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
            .collect();

        let result: Vec<BigUint> = result
            .into_iter()
            .map(|entry| {
                BigUint::parse_bytes(entry.as_bytes(), 10)
                    .expect("Failed to parse the string into a BigUint")
            })
            .collect();

        let result = Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone()));

        assert_eq!(result, prod);

        println!("Product: {}", prod);
        println!("Result: {}", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_Fp2exp() {
    let mut rng = rand::thread_rng();

    let p = parse_bn254_modulus();

    let random_bytes: Vec<u8> = (0..32).map(|_| rng.r#gen()).collect();
    let mut a = BigUint::from_bytes_le(&random_bytes);
    a = a % p;
    let mut a_limbs = a.to_u64_digits();
    // a_limbs.reverse();

    let b = Fq2::rand(&mut rng);
    let mut prod = b.pow(a_limbs);

    let input_json = format!(
        r#"{{
    "op1": "{}",
    "op2": "{}"
}}"#,
        a, b
    );

    let input_json = string_for_fp2_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Fp2exp";
    let circom_file_path = "./fp2.circom";
    let circom_file = "fp2";

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
            .take(2)
            .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
            .collect();

        let result: Vec<BigUint> = result
            .into_iter()
            .map(|entry| {
                BigUint::parse_bytes(entry.as_bytes(), 10)
                    .expect("Failed to parse the string into a BigUint")
            })
            .collect();

        let result = Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone()));

        assert_eq!(result, prod);

        println!("Product: {}", prod);
        println!("Result: {}", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_Fp2mul2() {
    let mut rng = rand::thread_rng();

    let a = Fq2::rand(&mut rng);
    let b = Fq2::rand(&mut rng);
    let prod = a * b;

    let input_json = format!(
        r#"{{
    "op1": "{}",
    "op2": "{}"
}}"#,
        a, b
    );

    let input_json = string_for_fp2_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Fp2mul";
    let circom_file_path = "./fp2.circom";
    let circom_file = "fp2";

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
            .take(2)
            .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
            .collect();

        let result: Vec<BigUint> = result
            .into_iter()
            .map(|entry| {
                BigUint::parse_bytes(entry.as_bytes(), 10)
                    .expect("Failed to parse the string into a BigUint")
            })
            .collect();

        let result = Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone()));

        assert_eq!(result, prod);

        println!("Product: {}", prod);
        println!("Result: {}", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_Fp2muladd() {
    let mut rng = thread_rng();

    let a = Fq2::rand(&mut rng);
    let b = Fq2::rand(&mut rng);
    let c = Fq2::rand(&mut rng);
    let prod = (a * b) + c;

    let input_json = format!(
        r#"{{
    "op1": "{}",
    "op2": "{}",
    "op3": "{}"
}}"#,
        a, b, c
    );

    let input_json = string_for_fp2_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Fp2muladd";
    let circom_file_path = "./fp2.circom";
    let circom_file = "fp2";

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
            .take(2)
            .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
            .collect();

        let result: Vec<BigUint> = result
            .into_iter()
            .map(|entry| {
                BigUint::parse_bytes(entry.as_bytes(), 10)
                    .expect("Failed to parse the string into a BigUint")
            })
            .collect();

        let result = Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone()));

        assert_eq!(result, prod);

        println!("Product: {}", prod);
        println!("Result: {}", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_Fp2inv() {
    let mut rng = rand::thread_rng();

    let a = Fq2::rand(&mut rng);
    let inv = a.inverse().unwrap();

    let input_json = format!(
        r#"{{
    "op1": "{}"
}}"#,
        a
    );

    let input_json = string_for_fp2_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Fp2inv";
    let circom_file_path = "./fp2.circom";
    let circom_file = "fp2";

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
            .take(2)
            .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
            .collect();

        let result: Vec<BigUint> = result
            .into_iter()
            .map(|entry| {
                BigUint::parse_bytes(entry.as_bytes(), 10)
                    .expect("Failed to parse the string into a BigUint")
            })
            .collect();

        let result = Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone()));

        assert_eq!(result, inv);

        println!("Inverse: {}", inv);
        println!("Result: {}", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_Fp2conjugate() {
    let mut rng = rand::thread_rng();

    let a = Fq2::rand(&mut rng);
    let mut conj = a.clone();
    let conj2 = conj.conjugate_in_place();

    let input_json = format!(
        r#"{{
    "op1": "{}"
}}"#,
        a
    );

    let input_json = string_for_fp2_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Fp2conjugate";
    let circom_file_path = "./fp2.circom";
    let circom_file = "fp2";

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
            .take(2)
            .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
            .collect();

        let result: Vec<BigUint> = result
            .into_iter()
            .map(|entry| {
                BigUint::parse_bytes(entry.as_bytes(), 10)
                    .expect("Failed to parse the string into a BigUint")
            })
            .collect();

        let result = Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone()));

        assert_eq!(result, conj);

        println!("conjugate: {}", conj);
        println!("Result: {}", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_all() {
    test_Fp2add();
    test_Fp2sub();
    test_Fp2mulbyfp();
    test_Fp2exp();
    test_Fp2mul2();
    test_Fp2muladd();
    test_Fp2inv();
    test_Fp2conjugate();
}
