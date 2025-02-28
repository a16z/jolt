// use ark_bn254::{Bn254, Fq, Fq12, Fq2, Fr, G1Affine, G1Projective, G2Projective};
// use ark_ec::bn::G2Prepared;
// use ark_ec::pairing::Pairing;
// use ark_ec::{pairing, AffineRepr, CurveGroup};
// use ark_ff::{BigInt, CyclotomicMultSubgroup, Field, Fp, MontFp, PrimeField};
// use ark_ff::{Fp6, ToConstraintField};
// use ark_std::{One, UniformRand};
// use group::G1ProjectiveCircom;
// use jolt_core::test_circom_link::link_joltstuff::G1AffineCircomLink;
// // use matrix::check_inner_product;
// // use matrix::print_test;
// use num_bigint::BigUint;
// use rand::thread_rng;
// use rand::Rng;
// use regex::Regex;
// use serde::{Deserialize, Serialize};
// use serde_json::{json, Value};
// use std::convert::TryInto;
// use std::fmt;
// use std::fs::{self, File, OpenOptions};
// use std::io::{Read, Write};
// use std::process::Command;

// #[test]
// fn final_exponentiation() {
//     let mut rng = rand::thread_rng();
//     let a = G1Projective::rand(&mut rng).into_affine().into_group();
//     let b = G2Projective::rand(&mut rng).into_affine().into_group();
//     let c = Bn254::miller_loop(a, b);

//     let f_exp = Bn254::final_exponentiation(c).unwrap();
//     let input_json = format!(
//         r#"{{
//     "f": {{ "x": {{
//         "x": {{"x": "{}", "y": "{}"}},
//         "y": {{"x": "{}", "y": "{}"}},
//         "z": {{"x": "{}", "y": "{}"}}
//         }},
//     "y": {{
//         "x": {{"x": "{}", "y": "{}"}},
//         "y": {{"x": "{}", "y": "{}"}},
//         "z": {{"x": "{}", "y": "{}"}}
//         }}
//     }}
//     }}"#,
//         c.0.c0.c0.c0,
//         c.0.c0.c0.c1,
//         c.0.c0.c1.c0,
//         c.0.c0.c1.c1,
//         c.0.c0.c2.c0,
//         c.0.c0.c2.c1,
//         c.0.c1.c0.c0,
//         c.0.c1.c0.c1,
//         c.0.c1.c1.c0,
//         c.0.c1.c1.c1,
//         c.0.c1.c2.c0,
//         c.0.c1.c2.c1,
//     );
//     let input_file_path = "input.json";
//     let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
//     input_file
//         .write_all(input_json.as_bytes())
//         .expect("Failed to write to input.json");
//     println!("Input JSON file created successfully.");

//     // Step 3: Call shell script to compile and generate witness
//     let script_path = "./../../../scripts/compile_and_generate_witness.sh";
//     let circom_template = "FinalExp";
//     let circom_file_path = "./pairing.circom";
//     let circom_file = "pairing";

//     let output = Command::new(script_path)
//         .args(&[
//             circom_file_path,
//             circom_template,
//             input_file_path,
//             circom_file,
//         ])
//         .output()
//         .expect("Failed to execute shell script");
//     if !output.status.success() {
//         panic!(
//             "Shell script execution failed: {}",
//             String::from_utf8_lossy(&output.stderr)
//         );
//     }

//     println!(
//         "Shell script output: {}",
//         String::from_utf8_lossy(&output.stdout)
//     );

//     // Step 4: Verify witness generation
//     println!("Witness generated successfully.");

//     // Read the witness.json file
//     let witness_file_path = "witness.json"; // Make sure this is the correct path to your witness.json
//     let mut witness_file = File::open(witness_file_path).expect("Failed to open witness.json");

//     // Read the file contents
//     let mut witness_contents = String::new();
//     witness_file
//         .read_to_string(&mut witness_contents)
//         .expect("Failed to read witness.json");

//     // Parse the JSON contents
//     let witness_json: Value =
//         serde_json::from_str(&witness_contents).expect("Failed to parse witness.json");

//     if let Some(witness_array) = witness_json.as_array() {
//         let result: Vec<String> = witness_array
//             .iter()
//             .skip(1)
//             .take(12)
//             .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
//             .collect();

//         let result: Vec<BigUint> = result
//             .into_iter()
//             .map(|entry| {
//                 BigUint::parse_bytes(entry.as_bytes(), 10)
//                     .expect("Failed to parse the string into a BigUint")
//             })
//             .collect();

//         // let result = Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone()));

//         let r0 = Fp6::new(
//             Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone())),
//             Fq2::new(Fq::from(result[2].clone()), Fq::from(result[3].clone())),
//             Fq2::new(Fq::from(result[4].clone()), Fq::from(result[5].clone())),
//         );
//         let r1 = Fp6::new(
//             Fq2::new(Fq::from(result[6].clone()), Fq::from(result[7].clone())),
//             Fq2::new(Fq::from(result[8].clone()), Fq::from(result[9].clone())),
//             Fq2::new(Fq::from(result[10].clone()), Fq::from(result[11].clone())),
//         );

//         let result = Fq12::new(r0, r1);

//         assert_eq!(result, f_exp.0);
//     } else {
//         eprintln!("The JSON is not an array or 'witness' field is missing");
//     }
// }

// #[test]
// fn millerLoop() {
//     let mut rng = rand::thread_rng();
//     let a = G1Projective::rand(&mut rng).into_affine().into_group();
//     let b = G2Projective::rand(&mut rng).into_affine().into_group();

//     let c = Bn254::miller_loop(a, b).0;

//     let input_json = format!(
//         r#"{{
//     "P": {{
//         "x": "{}",
//         "y": "{}",
//         "z": "{}"
//     }},
//     "Q": {{
//         "x": {{ "x": "{}", "y": "{}" }},
//         "y": {{ "x": "{}", "y": "{}" }},
//         "z": {{ "x": "{}", "y": "{}" }}
//         }}
//     }}"#,
//         a.x, a.y, a.z, b.x.c0, b.x.c1, b.y.c0, b.y.c1, b.z.c0, b.z.c1
//     );

//     let input_file_path = "input.json";
//     let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
//     input_file
//         .write_all(input_json.as_bytes())
//         .expect("Failed to write to input.json");
//     println!("Input JSON file created successfully.");

//     // Step 3: Call shell script to compile and generate witness
//     let script_path = "./../../../scripts/compile_and_generate_witness.sh";

//     let circom_template = "MillerLoop";
//     let circom_file_path = "./pairing.circom";
//     let circom_file = "pairing";

//     let output = Command::new(script_path)
//         .args(&[
//             circom_file_path,
//             circom_template,
//             input_file_path,
//             circom_file,
//         ])
//         .output()
//         .expect("Failed to execute shell script");

//     if !output.status.success() {
//         panic!(
//             "Shell script execution failed: {}",
//             String::from_utf8_lossy(&output.stderr)
//         );
//     }

//     println!(
//         "Shell script output: {}",
//         String::from_utf8_lossy(&output.stdout)
//     );

//     // Step 4: Verify witness generation
//     println!("Witness generated successfully.");

//     // Read the witness.json file
//     let witness_file_path = "witness.json"; // Make sure this is the correct path to your witness.json
//     let mut witness_file = File::open(witness_file_path).expect("Failed to open witness.json");

//     // Read the file contents
//     let mut witness_contents = String::new();
//     witness_file
//         .read_to_string(&mut witness_contents)
//         .expect("Failed to read witness.json");

//     // Parse the JSON contents
//     let witness_json: Value =
//         serde_json::from_str(&witness_contents).expect("Failed to parse witness.json");

//     if let Some(witness_array) = witness_json.as_array() {
//         let result: Vec<String> = witness_array
//             .iter()
//             .skip(1)
//             .take(12)
//             .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
//             .collect();

//         let result: Vec<BigUint> = result
//             .into_iter()
//             .map(|entry| {
//                 BigUint::parse_bytes(entry.as_bytes(), 10)
//                     .expect("Failed to parse the string into a BigUint")
//             })
//             .collect();

//         // let result = Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone()));

//         let r0 = Fp6::new(
//             Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone())),
//             Fq2::new(Fq::from(result[2].clone()), Fq::from(result[3].clone())),
//             Fq2::new(Fq::from(result[4].clone()), Fq::from(result[5].clone())),
//         );
//         let r1 = Fp6::new(
//             Fq2::new(Fq::from(result[6].clone()), Fq::from(result[7].clone())),
//             Fq2::new(Fq::from(result[8].clone()), Fq::from(result[9].clone())),
//             Fq2::new(Fq::from(result[10].clone()), Fq::from(result[11].clone())),
//         );

//         let result = Fq12::new(r0, r1);

//         assert_eq!(result, c);
//     } else {
//         eprintln!("The JSON is not an array or 'witness' field is missing");
//     }

//     // check_inner_product(vec![circom_file.to_string()]);
// }

// #[test]
// fn test_pairing() {
//     let mut rng = rand::thread_rng();
//     let a = G1Projective::rand(&mut rng).into_affine().into_group();
//     let b = G2Projective::rand(&mut rng).into_affine().into_group();
//     let c = Bn254::pairing(a, b).0;

//     let input_json = format!(
//         r#"{{
//     "P": {{
//         "x": "{}",
//         "y": "{}",
//         "z": "{}"
//     }},
//     "Q": {{
//         "x": {{ "x": "{}", "y": "{}" }},
//         "y": {{ "x": "{}", "y": "{}" }}
//     }}
// }}"#,
//         a.x, a.y, a.z, b.x.c0, b.x.c1, b.y.c0, b.y.c1
//     );

//     let input_file_path = "input.json";
//     let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
//     input_file
//         .write_all(input_json.as_bytes())
//         .expect("Failed to write to input.json");
//     println!("Input JSON file created successfully.");

//     // Step 3: Call shell script to compile and generate witness
//     let script_path = "./../../../scripts/compile_and_generate_witness.sh";
//     let circom_template = "Pairing";
//     let circom_file_path = "./pairing.circom";
//     let circom_file = "pairing";

//     let output = Command::new(script_path)
//         .args(&[
//             circom_file_path,
//             circom_template,
//             input_file_path,
//             circom_file,
//         ])
//         .output()
//         .expect("Failed to execute shell script");

//     if !output.status.success() {
//         panic!(
//             "Shell script execution failed: {}",
//             String::from_utf8_lossy(&output.stderr)
//         );
//     }

//     println!(
//         "Shell script output: {}",
//         String::from_utf8_lossy(&output.stdout)
//     );

//     // Step 4: Verify witness generation
//     println!("Witness generated successfully.");

//     // Read the witness.json file
//     let witness_file_path = "witness.json"; // Make sure this is the correct path to your witness.json
//     let mut witness_file = File::open(witness_file_path).expect("Failed to open witness.json");

//     // Read the file contents
//     let mut witness_contents = String::new();
//     witness_file
//         .read_to_string(&mut witness_contents)
//         .expect("Failed to read witness.json");

//     // Parse the JSON contents
//     let witness_json: Value =
//         serde_json::from_str(&witness_contents).expect("Failed to parse witness.json");

//     if let Some(witness_array) = witness_json.as_array() {
//         let result: Vec<String> = witness_array
//             .iter()
//             .skip(1)
//             .take(12)
//             .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
//             .collect();

//         let result: Vec<BigUint> = result
//             .into_iter()
//             .map(|entry| {
//                 BigUint::parse_bytes(entry.as_bytes(), 10)
//                     .expect("Failed to parse the string into a BigUint")
//             })
//             .collect();

//         // let result = Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone()));

//         let r0 = Fp6::new(
//             Fq2::new(Fq::from(result[0].clone()), Fq::from(result[1].clone())),
//             Fq2::new(Fq::from(result[2].clone()), Fq::from(result[3].clone())),
//             Fq2::new(Fq::from(result[4].clone()), Fq::from(result[5].clone())),
//         );
//         let r1 = Fp6::new(
//             Fq2::new(Fq::from(result[6].clone()), Fq::from(result[7].clone())),
//             Fq2::new(Fq::from(result[8].clone()), Fq::from(result[9].clone())),
//             Fq2::new(Fq::from(result[10].clone()), Fq::from(result[11].clone())),
//         );

//         let result = Fq12::new(r0, r1);

//         assert_eq!(result, c);
//     } else {
//         eprintln!("The JSON is not an array or 'witness' field is missing");
//     }

//     // check_inner_product(vec![circom_file.to_string()]);
// }

// use ark_std::Zero;

// #[test]
// fn check_g1_to_affine(){
//     // UNcomment the line "let proj_pt = proj_pt - proj_pt;" for runnig test with identity projective point
//     let mut rng = rand::thread_rng();
//     let mut flag= false;
//     let proj_pt = G1Projective::rand(&mut rng);
//     let proj_pt = proj_pt - proj_pt;

//     if proj_pt.is_zero(){
//         flag = true;
//     }

//     let aff_pt = proj_pt.into_affine();
//     let proj_circom = G1ProjectiveCircom{
//         x: proj_pt.x,
//         y: proj_pt.y,
//         z: proj_pt.z
//     };

//     let input_json = format!(
//         r#"{{
//                 "op1": {:?}
//         }}"#,
//         proj_circom
//     );

//     let input_file_path = "input.json";
//     let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
//     input_file
//         .write_all(input_json.as_bytes())
//         .expect("Failed to write to input.json");
//     println!("Input JSON file created successfully.");

//     // Step 3: Call shell script to compile and generate witness
//     let script_path = "./../../../scripts/compile_and_generate_witness.sh";
//     let circom_template = "G1ToAffine";
//     let circom_file_path = "./pairing.circom";
//     let circom_file = "pairing";

//     let output = Command::new(script_path)
//         .args(&[
//             circom_file_path,
//             circom_template,
//             input_file_path,
//             circom_file,
//         ])
//         .output()
//         .expect("Failed to execute shell script");

//     if !output.status.success() {
//         panic!(
//             "Shell script execution failed: {}",
//             String::from_utf8_lossy(&output.stderr)
//         );
//     }

//     println!(
//         "Shell script output: {}",
//         String::from_utf8_lossy(&output.stdout)
//     );

//     // Step 4: Verify witness generation
//     println!("Witness generated successfully.");

//     let witness_file_path = "witness.json"; // Make sure this is the correct path to your witness.json
//     let mut witness_file = File::open(witness_file_path).expect("Failed to open witness.json");

//     // Read the file contents
//     let mut witness_contents = String::new();
//     witness_file
//         .read_to_string(&mut witness_contents)
//         .expect("Failed to read witness.json");

//     // Parse the JSON contents
//     let witness_json: Value =
//     serde_json::from_str(&witness_contents).expect("Failed to parse witness.json");

//     if let Some(witness_array) = witness_json.as_array() {
//         let result: Vec<String> = witness_array
//             .iter()
//             .skip(1)
//             .take(3)
//             .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
//             .collect();

//         let result: Vec<BigUint> = result
//             .into_iter()
//             .map(|entry| {
//                 BigUint::parse_bytes(entry.as_bytes(), 10)
//                     .expect("Failed to parse the string into a BigUint")
//             })
//             .collect();

//         let circom_result = G1Affine{
//             x: Fq::from(result[0].clone()),
//             y: Fq::from(result[1].clone()),
//             infinity: flag
//         };
//     assert_eq!(circom_result, proj_pt);
// } else {
//     eprintln!("The JSON is not an array or 'witness' field is missing");
// }
// }
