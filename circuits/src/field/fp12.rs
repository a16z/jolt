#![allow(unused_imports)]
use ark_bn254::{Fq, Fq2, Fq6, Fq12, Fr};
use ark_ff::{BigInt, CyclotomicMultSubgroup, Field, Fp6, PrimeField};
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
use std::str::FromStr;

fn parse_bn254_modulus() -> BigUint {
    let modulus_str =
        "21888242871839275222246405745257275088696311157297823662689037894645226208583";
    let modulus = BigUint::parse_bytes(modulus_str.as_bytes(), 10)
        .expect("Failed to parse the string into a BigUint");
    modulus
}

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
    let re = Regex::new(r#"CubicExtField\(\{([^}]+)\}, \{([^}]+)\}, \{([^}]+)\}\)"#).unwrap();

    // Replace matched patterns with the desired format
    let result = re.replace_all(input, |caps: &regex::Captures| {
        let x = &caps[1]; // First capture group (x)
        let y = &caps[2]; // Second capture group (y)
        let z = &caps[3];
        format!(r#"{{"x": {{{}}}, "y": {{{}}}, "z": {{{}}}}}"#, x, y, z)
    });

    result.to_string()
}

fn string_for_fp12_input(input: &str) -> String {
    // Define regex to capture the x and y parts of the QuadExtField
    let re = Regex::new(r#""QuadExtField\((.*?) \+ (.*?) \* u\)""#).unwrap();

    // Replace matched patterns with the desired format
    let result = re.replace_all(input, |caps: &regex::Captures| {
        let x = &caps[1]; // First capture group (x)
        let y = &caps[2]; // Second capture group (y)

        // Format the replacement string as per desired output
        format!("{{\"x\": {}, \"y\": {}}}", x, y)
    });

    result.to_string()
}

#[test]
fn test_Fp12add() {
    let mut rng = rand::thread_rng();

    let a = Fq12::rand(&mut rng);
    let b = Fq12::rand(&mut rng);
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
    let input_json = string_for_fp12_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Fp12add";
    let circom_file_path = "./fp12.circom";
    let circom_file = "fp12";

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
            .take(12)
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

        let r0 = Fp6::new(
            Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone())),
            Fq2::new(Fq::from(result[2].clone()), Fq::from(result[3].clone())),
            Fq2::new(Fq::from(result[4].clone()), Fq::from(result[5].clone())),
        );
        let r1 = Fp6::new(
            Fq2::new(Fq::from(result[6].clone()), Fq::from(result[7].clone())),
            Fq2::new(Fq::from(result[8].clone()), Fq::from(result[9].clone())),
            Fq2::new(Fq::from(result[10].clone()), Fq::from(result[11].clone())),
        );

        let result = Fq12::new(r0, r1);

        assert_eq!(result, sum);

        // println!("Sum: {}", sum);
        // println!("Result: {} \n", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    // check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_Fp12mul2() {
    let mut rng = thread_rng();

    let a = Fq12::rand(&mut rng);
    let b = Fq12::rand(&mut rng);
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
    let input_json = string_for_fp12_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Fp12mul";
    let circom_file_path = "./fp12.circom";
    let circom_file = "fp12";

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
            .take(12)
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

        let r0 = Fp6::new(
            Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone())),
            Fq2::new(Fq::from(result[2].clone()), Fq::from(result[3].clone())),
            Fq2::new(Fq::from(result[4].clone()), Fq::from(result[5].clone())),
        );
        let r1 = Fp6::new(
            Fq2::new(Fq::from(result[6].clone()), Fq::from(result[7].clone())),
            Fq2::new(Fq::from(result[8].clone()), Fq::from(result[9].clone())),
            Fq2::new(Fq::from(result[10].clone()), Fq::from(result[11].clone())),
        );

        let result = Fq12::new(r0, r1);

        assert_eq!(result, prod);

        // println!("Product: {}", prod);
        // println!("Result: {} \n", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    // check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_Fp12conjugate() {
    let mut rng = rand::thread_rng();

    let a = Fq12::rand(&mut rng);

    let mut b = a.clone();
    b.conjugate_in_place();

    let input_json = format!(
        r#"{{
    "op1": "{}"
}}"#,
        a
    );

    let input_json = string_for_fp2_input(&input_json);
    let input_json = string_for_fp6_input(&input_json);
    let input_json = string_for_fp12_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Fp12conjugate";
    let circom_file_path = "./fp12.circom";
    let circom_file = "fp12";

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
            .take(12)
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

        let r0 = Fp6::new(
            Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone())),
            Fq2::new(Fq::from(result[2].clone()), Fq::from(result[3].clone())),
            Fq2::new(Fq::from(result[4].clone()), Fq::from(result[5].clone())),
        );
        let r1 = Fp6::new(
            Fq2::new(Fq::from(result[6].clone()), Fq::from(result[7].clone())),
            Fq2::new(Fq::from(result[8].clone()), Fq::from(result[9].clone())),
            Fq2::new(Fq::from(result[10].clone()), Fq::from(result[11].clone())),
        );

        let result = Fq12::new(r0, r1);

        assert_eq!(result, b);

        // println!("Conjugate: {}", b);
        // println!("Result: {} \n", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    // check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_Fp12inv() {
    let mut rng = thread_rng();

    let a = Fq12::rand(&mut rng);
    let mut b = a.clone();
    b.inverse_in_place().unwrap();

    let input_json = format!(
        r#"{{
    "op1": "{}"
}}"#,
        a
    );

    let input_json = string_for_fp2_input(&input_json);
    let input_json = string_for_fp6_input(&input_json);
    let input_json = string_for_fp12_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Fp12inv";
    let circom_file_path = "./fp12.circom";
    let circom_file = "fp12";

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
            .take(12)
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

        let r0 = Fp6::new(
            Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone())),
            Fq2::new(Fq::from(result[2].clone()), Fq::from(result[3].clone())),
            Fq2::new(Fq::from(result[4].clone()), Fq::from(result[5].clone())),
        );
        let r1 = Fp6::new(
            Fq2::new(Fq::from(result[6].clone()), Fq::from(result[7].clone())),
            Fq2::new(Fq::from(result[8].clone()), Fq::from(result[9].clone())),
            Fq2::new(Fq::from(result[10].clone()), Fq::from(result[11].clone())),
        );

        let result = Fq12::new(r0, r1);

        assert_eq!(result, b);

        // println!("Inverse: {}", b);
        // println!("Result: {} \n", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    // check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_Fp12square() {
    let mut rng = thread_rng();

    // let p = parse_bn254_modulus();
    // let mut exp = p.pow(6) - BigUint::from(1u32);
    // exp *= p.pow(2) + BigUint::from(1u32);
    let a = Fq12::rand(&mut rng);
    // a = a.pow(exp.to_u64_digits());
    let mut b = a.clone();
    b.cyclotomic_square_in_place();

    let input_json = format!(
        r#"{{
    "op1": "{}"
}}"#,
        a
    );

    let input_json = string_for_fp2_input(&input_json);
    let input_json = string_for_fp6_input(&input_json);
    let input_json = string_for_fp12_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Fp12square";
    let circom_file_path = "./fp12.circom";
    let circom_file = "fp12";

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
            .take(12)
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

        let r0 = Fp6::new(
            Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone())),
            Fq2::new(Fq::from(result[2].clone()), Fq::from(result[3].clone())),
            Fq2::new(Fq::from(result[4].clone()), Fq::from(result[5].clone())),
        );
        let r1 = Fp6::new(
            Fq2::new(Fq::from(result[6].clone()), Fq::from(result[7].clone())),
            Fq2::new(Fq::from(result[8].clone()), Fq::from(result[9].clone())),
            Fq2::new(Fq::from(result[10].clone()), Fq::from(result[11].clone())),
        );

        let result = Fq12::new(r0, r1);

        assert_eq!(result, b);

        println!("Square: {}", b);
        println!("Result: {} \n", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    // check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_Fp12sq2() {
    let mut rng = thread_rng();

    let p = parse_bn254_modulus();
    let mut exp = p.pow(6) - BigUint::from(1u32);
    exp *= p.pow(2) + BigUint::from(1u32);
    let a = Fq12::rand(&mut rng);
    let a = a.pow(exp.to_u64_digits());
    let mut b = a.clone();
    b.cyclotomic_square_in_place();

    let input_json = format!(
        r#"{{
    "op1": "{}"
}}"#,
        a
    );

    let input_json = string_for_fp2_input(&input_json);
    let input_json = string_for_fp6_input(&input_json);
    let input_json = string_for_fp12_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Fp12sq";
    let circom_file_path = "./fp12.circom";
    let circom_file = "fp12";

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
            .take(12)
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

        let r0 = Fp6::new(
            Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone())),
            Fq2::new(Fq::from(result[2].clone()), Fq::from(result[3].clone())),
            Fq2::new(Fq::from(result[4].clone()), Fq::from(result[5].clone())),
        );
        let r1 = Fp6::new(
            Fq2::new(Fq::from(result[6].clone()), Fq::from(result[7].clone())),
            Fq2::new(Fq::from(result[8].clone()), Fq::from(result[9].clone())),
            Fq2::new(Fq::from(result[10].clone()), Fq::from(result[11].clone())),
        );

        let result = Fq12::new(r0, r1);

        assert_eq!(result, b);

        println!("Square: {}", b);
        println!("Result: {} \n", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    // check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_Fp12mulbyfp() {
    let mut rng = rand::thread_rng();

    let a = Fq::rand(&mut rng);
    let mut c = Fq12::rand(&mut rng);
    // let mut b = c.clone();
    // b.c0.c0.mul_assign_by_basefield(&a);
    // b.c0.c1.mul_assign_by_basefield(&a);
    // b.c0.c2.mul_assign_by_basefield(&a);
    // b.c1.c0.mul_assign_by_basefield(&a);
    // b.c1.c1.mul_assign_by_basefield(&a);
    // b.c1.c2.mul_assign_by_basefield(&a);

    let input_json = format!(
        r#"{{
    "op1": "{}",
    "op2": "{}"
}}"#,
        a, c
    );

    c.mul_by_fp(&a);

    let input_json = string_for_fp2_input(&input_json);
    let input_json = string_for_fp6_input(&input_json);
    let input_json = string_for_fp12_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Fp12mulbyfp";
    let circom_file_path = "./fp12.circom";
    let circom_file = "fp12";

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
            .take(12)
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

        let r0 = Fp6::new(
            Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone())),
            Fq2::new(Fq::from(result[2].clone()), Fq::from(result[3].clone())),
            Fq2::new(Fq::from(result[4].clone()), Fq::from(result[5].clone())),
        );
        let r1 = Fp6::new(
            Fq2::new(Fq::from(result[6].clone()), Fq::from(result[7].clone())),
            Fq2::new(Fq::from(result[8].clone()), Fq::from(result[9].clone())),
            Fq2::new(Fq::from(result[10].clone()), Fq::from(result[11].clone())),
        );

        let result = Fq12::new(r0, r1);

        assert_eq!(result, c);

        println!("Product: {}", c);
        println!("Result: {} \n", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    // check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_Fp12exp() {
    let mut rng = rand::thread_rng();

    // let a = Fq::rand(&mut rng);
    let p = parse_bn254_modulus();

    let random_bytes: Vec<u8> = (0..32).map(|_| rng.r#gen()).collect();
    let mut a = BigUint::from_bytes_le(&random_bytes);
    a = a % p.clone();
    let mut a_limbs = a.to_u64_digits();

    // let b = Fq12::rand(&mut rng);
    let mut exp = p.pow(6) - BigUint::from(1u32);
    exp *= p.pow(2) + BigUint::from(1u32);
    let mut b = Fq12::rand(&mut rng);
    b = b.pow(exp.to_u64_digits());

    let mut c = b.pow(a_limbs);
    // c.cyclotomic_exp_in_place(&a);
    let input_json = format!(
        r#"{{
    "op1": "{}",
    "op2": "{}"
}}"#,
        a, b
    );

    let input_json = string_for_fp2_input(&input_json);
    let input_json = string_for_fp6_input(&input_json);
    let input_json = string_for_fp12_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Fp12exp";
    let circom_file_path = "./fp12.circom";
    let circom_file = "fp12";

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
            .take(12)
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

        let r0 = Fp6::new(
            Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone())),
            Fq2::new(Fq::from(result[2].clone()), Fq::from(result[3].clone())),
            Fq2::new(Fq::from(result[4].clone()), Fq::from(result[5].clone())),
        );
        let r1 = Fp6::new(
            Fq2::new(Fq::from(result[6].clone()), Fq::from(result[7].clone())),
            Fq2::new(Fq::from(result[8].clone()), Fq::from(result[9].clone())),
            Fq2::new(Fq::from(result[10].clone()), Fq::from(result[11].clone())),
        );

        let result = Fq12::new(r0, r1);

        assert_eq!(result, c);
        println!("Test passed!");

        println!("Product: {}", c);
        println!("Result: {} \n", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    // check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_all() {
    for i in 0..1 {
        test_Fp12add();
        test_Fp12mul2();
        test_Fp12conjugate();
        test_Fp12inv();
        test_Fp12square();
        test_Fp12sq2();
        test_Fp12mulbyfp();
        test_Fp12exp();
    }
}
