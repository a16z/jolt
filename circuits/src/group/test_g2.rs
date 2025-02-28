use ark_bn254::{Fq, Fq2, Fq6, G2Projective, Fr};
use ark_ff::{AdditiveGroup, CyclotomicMultSubgroup, Field, Fp6, PrimeField};
use ark_std::{One, UniformRand};
use jolt_core::test_circom_link::link_opening_combiners::{convert_from_3_limbs};
use num_bigint::BigUint;
use parsing::non_native::Fqq;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::convert::TryInto;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::process::Command;
use std::str::FromStr;
use::std::ops::Mul;
use rand::{thread_rng, Rng};

const CIRCOM_FILE: &str = "bn254_g2";
const CIRCOM_FILE_PATH: &str = "./bn254_g2.circom";
const SCRIPT_PATH: &str = "./../../scripts/compile_and_generate_witness_bn_base.sh";
const INPUT_FILE_PATH: &str = "input.json";

struct FQ {
    // element: BigUint,
    limbs: [BigUint; 3],
}

impl fmt::Debug for FQ {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Print the struct fields with appropriate formatting
        write!(
            f,
            "\n\n{{\"limbs\": [\n\"{:?}\", \n\"{:?}\", \n\"{:?}\"] }}",
            self.limbs[0], self.limbs[1], self.limbs[2]
        )
    }
}

fn string_for_G2_input(input: &str) -> String {
    // Define regex to capture the x and y parts of the QuadExtField
    let re = Regex::new(r#""\((\{[^}]+\}), (\{[^}]+\})\)""#).unwrap();

    // Replace matched patterns with the desired format
    let result = re.replace_all(input, |caps: &regex::Captures| {
        let x = &caps[1]; // First capture group (x)
        let y = &caps[2]; // Second capture group (y)
                          // let z = &caps[3];
        format!(r#"{{"x": {}, "y": {}, "z": {{"x": "1", "y": "0"}}}}"#, x, y)
    });

    result.to_string()
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

fn parse_bn254_modulus() -> BigUint {
    let modulus_str =
        "21888242871839275222246405745257275088696311157297823662689037894645226208583";
    let modulus = BigUint::parse_bytes(modulus_str.as_bytes(), 10)
        .expect("Failed to parse the string into a BigUint");
    modulus
}


fn write_input_read_witness(input_json: String, circom_template: &str) -> Value {
    let mut input_file = File::create(INPUT_FILE_PATH).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness

    let output = Command::new(SCRIPT_PATH)
        .args(&[
            CIRCOM_FILE_PATH,
            circom_template,
            INPUT_FILE_PATH,
            CIRCOM_FILE,
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

    witness_json
}

fn check_result(witness_json: Value, exp_result: G2Projective) {
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

        let result = G2Projective::new(
            Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone()))
                * Fq2::new(Fq::from(result[4].clone()), Fq::from(result[5].clone()))
                    .inverse()
                    .unwrap(),
            Fq2::new(Fq::from(result[2].clone()), Fq::from(result[3].clone()))
                * Fq2::new(Fq::from(result[4].clone()), Fq::from(result[5].clone()))
                    .inverse()
                    .unwrap(),
            Fq2::new(Fq::from(1), Fq::from(0)),
        );

        assert_eq!(result, exp_result);

        // println!("Double: {}", exp_result);
        // println!("Result: {} \n", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }
}

#[test]
fn test_G2Double() {
    let mut rng = rand::thread_rng();

    let a = G2Projective::rand(&mut rng);
    let exp_result = a.double();

    let input_json = format!(
        r#"{{
    "op1": "{}"
}}"#,
        a
    );

    let input_json = string_for_fp2_input(&input_json);
    let input_json = string_for_G2_input(&input_json);
    let circom_template = "G2Double";

    let witness_json = write_input_read_witness(input_json, circom_template);
    check_result(witness_json, exp_result);

    //check_inner_product_sparse(vec![CIRCOM_FILE.to_string()]);
}

#[test]
fn test_G2Add() {
    let mut rng = rand::thread_rng();

    let a = G2Projective::rand(&mut rng);
    let b = G2Projective::rand(&mut rng);
    let exp_result = a + b;

    let input_json = format!(
        r#"{{
    "op1": "{}",
    "op2": "{}"
}}"#,
        a, b
    );

    let input_json = string_for_fp2_input(&input_json);
    let input_json = string_for_G2_input(&input_json);
    let circom_template = "G2Add";

    let witness_json = write_input_read_witness(input_json, circom_template);
    check_result(witness_json, exp_result);

    // //check_inner_product_sparse(vec![CIRCOM_FILE.to_string()]);
}

#[test]
fn test_G2Mul() {
    let mut rng = thread_rng();

    let p = parse_bn254_modulus();

    let a = G2Projective::rand(&mut rng);
    let mut scalar_bar = [Fq::from(0u8); 3];

    for i in 0..2 {
        scalar_bar[i] = Fq::from(rng.gen_range(0..(1u128 << 125)));

    }
    scalar_bar[2] = Fq::from(rng.gen_range(0..(1u128 << 4)));

    let scalar = convert_from_3_limbs(scalar_bar.to_vec());
    let op2 = Fqq {
        // element: scalar,
        limbs: scalar_bar,
    };

    let exp_result = a.mul(scalar);

    let input_json = format!(
        r#"{{
            "op1": "{}",
            "op2": {:?}
        }}"#,
        a, op2
    );

    let input_json = string_for_fp2_input(&input_json);
    let input_json = string_for_G2_input(&input_json);
    let circom_template = "G2Mul";

    let witness_json = write_input_read_witness(input_json, circom_template);
    check_result(witness_json, exp_result);

    //check_inner_product_sparse(vec![CIRCOM_FILE.to_string()]);
}

#[test]
fn test_group2(){
    test_G2Double();
    test_G2Mul();
    test_G2Add();
}