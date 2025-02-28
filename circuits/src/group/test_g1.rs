#![allow(unused_imports)]
use ::std::ops::Mul;
use ark_bn254::{G1Projective};
use ark_ec::{AffineRepr, CurveGroup, AdditiveGroup};
use ark_ff::{CyclotomicMultSubgroup, Field, Fp6, PrimeField};
use ark_std::{One, UniformRand};
use jolt_core::test_circom_link::link_joltstuff::G1AffineCircomLink;
use jolt_core::test_circom_link::link_opening_combiners::{convert_from_3_limbs, convert_to_3_limbs};
use matrix::check_inner_product;
use num_bigint::BigUint;
use parsing::non_native::Fqq;
use rand::{thread_rng, Rng};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::convert::TryInto;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::process::Command;
use std::str::FromStr;
use ark_bn254::{Fq, Fr};

struct FQ {
    // element: BigUint,
    limbs: [BigUint; 3],
}

impl fmt::Debug for FQ {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Print the struct fields with appropriate formatting
        write!(
            f,
            "\n\n{{ \"limbs\": [\n\"{:?}\", \n\"{:?}\", \n\"{:?}\"] }}",
             self.limbs[0], self.limbs[1], self.limbs[2]
        )
    }
}

fn string_for_G1_input(input: &str) -> String {
    // Define regex to capture the x and y parts of the QuadExtField
    let re = Regex::new(r#""\((\d+), (\d+)\)""#).unwrap();

    // Replace matched patterns with the desired format
    let result = re.replace_all(input, |caps: &regex::Captures| {
        let x = &caps[1]; // First capture group (x)
        let y = &caps[2]; // Second capture group (y)
                          // let z = &caps[3];
        format!(r#"{{"x": "{}", "y": "{}", "z": "1"}}"#, x, y)
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

#[test]
fn test_G1Double() {
    let mut rng = thread_rng();

    let a = G1Projective::rand(&mut rng);
    let double = a.double();

    let input_json = format!(
        r#"{{
        "op1": "{}"
    }}"#,
        a
    );

    let input_json = string_for_G1_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "G1Double";
    let circom_file_path = "./bn254_g1.circom";
    let circom_file = "bn254_g1";

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
            .take(3)
            .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
            .collect();

        let result: Vec<BigUint> = result
            .into_iter()
            .map(|entry| {
                BigUint::parse_bytes(entry.as_bytes(), 10)
                    .expect("Failed to parse the string into a BigUint")
            })
            .collect();

        let result = G1Projective::new(
            Fq::from(result[0].clone()) * Fq::from(result[2].clone()).inverse().unwrap(),
            Fq::from(result[1].clone()) * Fq::from(result[2].clone()).inverse().unwrap(),
            Fq::from(1),
        );

        assert_eq!(result, double);

        println!("Double: {}", double);
        println!("Result: {} \n", result);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_G1Add() {
    let mut rng = thread_rng();

    let a = G1Projective::rand(&mut rng);
    let b = G1Projective::rand(&mut rng);
    let sum = a + b;
    println!("sum = {:?}", sum);

    let input_json = format!(
        r#"{{
            "op1": "{}",
            "op2": "{}"
        }}"#,
        a, b
    );

    let input_json = string_for_G1_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "G1Add";
    let circom_file_path = "./bn254_g1.circom";
    let circom_file = "bn254_g1";

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
            .take(3)
            .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
            .collect();

        let result: Vec<BigUint> = result
            .into_iter()
            .map(|entry| {
                BigUint::parse_bytes(entry.as_bytes(), 10)
                    .expect("Failed to parse the string into a BigUint")
            })
            .collect();

        let mut computed_sum = G1Projective::default();

        if (result[2] != BigUint::ZERO) {
            computed_sum = G1Projective::new(
                Fq::from(result[0].clone()) * Fq::from(result[2].clone()).inverse().unwrap(),
                Fq::from(result[1].clone()) * Fq::from(result[2].clone()).inverse().unwrap(),
                Fq::from(1),
            );
        } else {
            computed_sum = G1Projective::new(
                Fq::from(result[0].clone()),
                Fq::from(result[1].clone()),
                Fq::from(result[2].clone()),
            );
        }

        assert_eq!(computed_sum, sum);

        println!("Sum: {}", sum);
        println!("Result: {} \n", computed_sum);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}
use ark_std::Zero;

use parsing::bn_group::G1ProjectiveCircom;
#[test]
fn test_G1Mul() {
    let mut rng = thread_rng();

    let p = parse_bn254_modulus();

    let point = G1Projective::rand(&mut rng);

    // let a = a - a;

    
    // Neither of the follwoing work. Will have to find a way to put point at infinity in input.json.
    //
    // let mut a = G1Projective::ZERO;
    //
    // let mut a = G1Projective::rand(&mut rng);
    // a = a - a;

    let mut rng = thread_rng();

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

    let prod = point.mul(scalar);

    // let cirocm_pt = G1ProjectiveCircom{
    //     x: Fq::ZERO,
    //     y: Fq::ONE,
    //     z: Fq::ZERO,
    // };
    let circom_pt = G1ProjectiveCircom{
        x: point.x,
        y: point.y,
        z: point.z,
    };
    
    let input_json = format!(
        r#"{{
            "op1": {:?},
            "op2": {:?}
        }}"#,
        circom_pt, op2
    );

    // let input_json = string_for_G1_input(&input_json);

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "G1Mul";
    let circom_file_path = "./bn254_g1.circom";
    let circom_file = "bn254_g1";

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
            .take(3)
            .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
            .collect();

        let result: Vec<BigUint> = result
            .into_iter()
            .map(|entry| {
                BigUint::parse_bytes(entry.as_bytes(), 10)
                    .expect("Failed to parse the string into a BigUint")
            })
            .collect();

        let mut computed_product = G1Projective::default();

        if (result[2] != BigUint::ZERO) {
            computed_product = G1Projective::new(
                Fq::from(result[0].clone()) * Fq::from(result[2].clone()).inverse().unwrap(),
                Fq::from(result[1].clone()) * Fq::from(result[2].clone()).inverse().unwrap(),
                Fq::from(1),
            );
        } else {
            computed_product = G1Projective::new(
                Fq::from(result[0].clone()),
                Fq::from(result[1].clone()),
                Fq::from(result[2].clone()),
            );
        }

        assert_eq!(computed_product.into_affine(), prod.into_affine());

        println!("Product: {}", prod);
        println!("Result: {} \n", computed_product);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_new_proj(){
    let mut rng = rand::thread_rng();

    let mut proj_pt = G1Projective::rand(&mut rng);
    // let mut proj_pt = proj_pt.mul(Fr::ZERO);
    let mut aff_pt;
    let mut affine_circom;

    if G1Projective::is_zero(&proj_pt){
        proj_pt = proj_pt.mul(Fr::ZERO);
        aff_pt = proj_pt.into_affine();
        affine_circom = G1AffineCircomLink{
        x: Fq::ZERO,
        y: Fq::ZERO
    };
    }
    else{
        aff_pt = proj_pt.into_affine();
        affine_circom = G1AffineCircomLink{
            x: aff_pt.x().unwrap(),
            y: aff_pt.y().unwrap()
        };
    }
    let input_json = format!(
            r#"{{
                    "affine_point": {:?}
            }}"#,
        affine_circom
    );
    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "toProjectiveNew";
    let circom_file_path = "./bn254_g1.circom";
    let circom_file = "bn254_g1";

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
    
            let circom_result = G1Projective{
                x: Fq::from(result[0].clone()),
                y: Fq::from(result[1].clone()),
                z: Fq::from(result[2].clone()),
            };
        assert_eq!(circom_result, proj_pt);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }
    check_inner_product(vec![circom_file.to_string()]);

}

#[test]
fn test_group(){
    test_G1Double();
    test_G1Mul();
    test_G1Add();
    test_new_proj();
}