// For jolt1, Commitmments are over BN curve, but each coordinate expressed as 3 limbs of BN Fr
use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
use ark_ec::AffineRepr;
use ark_ff::UniformRand;
use jolt_core::{
    test_circom_jolt_1_2::{
        helper_commitms::{convert_rust_fp_to_circom, G1AffineCircom},
        struct_fq::FqCircom,
    },
    utils::{poseidon_transcript::PoseidonTranscript, transcript::Transcript},
};
use num_bigint::{BigInt, BigUint};
use parsing::{
    transcript_bn_scalar::{convert_transcript_fr_fr_to_circom, TestTranscript},
    SCALAR_LEN,
};
use regex::bytes;
use serde_json::Value;
use std::{
    fmt,
    fs::File,
    io::{Read, Write},
    process::Command,
    str::FromStr,
};
const STATE_WIDTH: usize = 5;
use ark_ff::PrimeField;

extern crate jolt_core;
use ark_ff::Field;

#[test]
fn testing_transcript_append_bytes() {
    let x = Fr::from(8u64);
    let mut rng = ark_std::test_rng();
    let mut t = <PoseidonTranscript<Fr, Fr> as Transcript>::new(b"label");
    let state = t.state.state[1];
    let nrounds = Fr::from(t.n_rounds);
    println!("n rounds are {:?}", nrounds);

    let testinput_transcript = convert_transcript_fr_fr_to_circom(t.clone());

    // Change this
    let n = 10;
    let bytes_to_append: Vec<u8> = (0..n).map(|_| u8::rand(&mut rng)).collect();
    let bytes_to_append_str: Vec<String> = bytes_to_append.iter().map(|x| x.to_string()).collect();
    // let scalar_to_append = Fr::rand(&mut rng);

    let input_json = format!(
        r#"{{
        "bytes": {:?},
        "transcript": {:?}
        }}"#,
        bytes_to_append_str, testinput_transcript
    );

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../scripts/compile_and_generate_witness_bn_scalar.sh";
    let circom_template = "AppendBytes";
    let circom_file_path = "./transcript.circom";
    let circom_file = "transcript";

    let output = Command::new(script_path)
        .args(&[
            circom_file_path,
            circom_template,
            input_file_path,
            circom_file,
            &n.to_string(),
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

        t.append_bytes(&bytes_to_append.as_slice());
        // t.append_scalar(&scalar_to_append.clone());

        let final_state_from_rust = t.state.state[1];
        let final_state_from_circom = Fr::from(result[0].clone());
        assert_eq!(final_state_from_rust, final_state_from_circom);

        let finaln_rounds_from_rust = t.n_rounds;
        let finaln_rounds_from_circom = Fr::from(result[1].clone());
        assert_eq!(Fr::from(finaln_rounds_from_rust), finaln_rounds_from_circom);
        println!(
            "finalnRounds_from_circom are {:?}",
            finaln_rounds_from_circom
        );
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    // check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn testing_transcript_append_scalar() {
    let x = Fr::from(8u64);
    let mut rng = ark_std::test_rng();
    let mut t = <PoseidonTranscript<Fr, Fr> as Transcript>::new(b"label");
    let state = t.state.state[1];
    let nrounds = Fr::from(t.n_rounds);
    println!("n rounds are {:?}", nrounds);

    let testinput_transcript = convert_transcript_fr_fr_to_circom(t.clone());

    let scalar_to_append = Fr::rand(&mut rng);

    let input_json = format!(
        r#"{{
        "scalar": "{}",
        "transcript": {:?}
        }}"#,
        scalar_to_append, testinput_transcript
    );

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../scripts/compile_and_generate_witness_bn_scalar.sh";
    let circom_template = "AppendScalar";
    let circom_file_path = "./transcript.circom";
    let circom_file = "transcript";

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

        t.append_scalar(&scalar_to_append.clone());

        let final_state_from_rust = t.state.state[1];
        let final_state_from_circom = Fr::from(result[0].clone());
        assert_eq!(final_state_from_rust, final_state_from_circom);

        let finaln_rounds_from_rust = t.n_rounds;
        let finaln_rounds_from_circom = Fr::from(result[1].clone());
        assert_eq!(Fr::from(finaln_rounds_from_rust), finaln_rounds_from_circom);
        println!(
            "finalnRounds_from_circom are {:?}",
            finaln_rounds_from_circom
        );
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    // check_inner_product(vec![circom_file.to_string()]);
}

// #[test]
// fn testing_transcript_append_scalars() {
//     let x = Fr::from(8u64);
//     let mut rng = ark_std::test_rng();
//     let mut t = <PoseidonTranscript<Fr, Fr> as Transcript>::new(b"label");
//     let state = t.state.state[1];
//     let nrounds = Fr::from(t.n_rounds);
//     println!("n rounds are {:?}", nrounds);

//     let testinput_transcript = convert_transcript_fr_fr_to_circom(t.clone());

//     let mut scalar_to_append = Vec::new();
//     for i in 0..SCALAR_LEN{
//         scalar_to_append.push(Fr::rand(&mut rng));
//     }
//     let mut formatted_scalars_to_append = Vec::new();
//     for i in 0..SCALAR_LEN{
//         formatted_scalars_to_append.push(FqCircom(scalar_to_append[i]));
//     }

//     let input_json = format!(
//         r#"{{
//         "scalars": {:?},
//         "transcript": {:?}
//         }}"#,
//         formatted_scalars_to_append, testinput_transcript
//     );

//     let input_file_path = "input.json";
//     let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
//     input_file
//         .write_all(input_json.as_bytes())
//         .expect("Failed to write to input.json");
//     println!("Input JSON file created successfully.");

//     // Step 3: Call shell script to compile and generate witness
//     let script_path = "./../../scripts/compile_and_generate_witness_bn_scalar.sh";
//     let circom_template = "AppendScalars";
//     let circom_file_path = "./transcript.circom";
//     let circom_file = "transcript";

//     let output = Command::new(script_path)
//         .args(&[
//             circom_file_path,
//             circom_template,
//             input_file_path,
//             circom_file,
//             &SCALAR_LEN.to_string()
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
//             .take(2)
//             .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
//             .collect();

//         let result: Vec<BigUint> = result
//             .into_iter()
//             .map(|entry| {
//                 BigUint::parse_bytes(entry.as_bytes(), 10)
//                     .expect("Failed to parse the string into a BigUint")
//             })
//             .collect();

//         t.append_scalars(&scalar_to_append.clone());

//         let final_state_from_rust = t.state.state[1];
//         let final_state_from_circom = Fr::from(result[0].clone());
//         assert_eq!(final_state_from_rust, final_state_from_circom);

//         let finaln_rounds_from_rust = t.n_rounds;
//         let finaln_rounds_from_circom = Fr::from(result[1].clone());
//         assert_eq!(Fr::from(finaln_rounds_from_rust), finaln_rounds_from_circom);
//         println!(
//             "finalnRounds_from_circom are {:?}",
//             finaln_rounds_from_circom
//         );
//     } else {
//         eprintln!("The JSON is not an array or 'witness' field is missing");
//     }

//     // check_inner_product(vec![circom_file.to_string()]);
// }

// #[test]
// fn test_append_point(){
//     let mut rng = ark_std::test_rng();
//     let mut t = <PoseidonTranscript<Fr, Fr> as Transcript>::new(b"label");
//     let state = t.state.state[1];
//     let nrounds = Fr::from(t.n_rounds);
//     println!("n rounds are {:?}", nrounds);

//     let testinput_transcript = convert_transcript_fr_fr_to_circom(t.clone());

//     let point_to_append = G1Affine::rand(&mut rng);

//     let point_in_circom = G1AffineCircom{
//         x: convert_rust_fp_to_circom(&point_to_append.x().unwrap()),
//         y: convert_rust_fp_to_circom(&point_to_append.y().unwrap()),
//     };

//     let proj_form = G1Projective {
//         x: point_to_append.x,
//         y: point_to_append.y,
//         z: Fq::ONE,
//     };

//     let input_json = format!(
//         r#"{{
//         "point": {:?},
//         "transcript": {:?}
//         }}"#,
//         point_in_circom, testinput_transcript
//     );

//     let input_file_path = "input.json";
//     let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
//     input_file
//         .write_all(input_json.as_bytes())
//         .expect("Failed to write to input.json");
//     println!("Input JSON file created successfully.");

//     // Step 3: Call shell script to compile and generate witness
//     let script_path = "./../../scripts/compile_and_generate_witness_bn_scalar.sh";
//     let circom_template = "AppendPoint";
//     let circom_file_path = "./transcript.circom";
//     let circom_file = "transcript";

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
//             .take(2)
//             .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
//             .collect();

//         let result: Vec<BigUint> = result
//             .into_iter()
//             .map(|entry| {
//                 BigUint::parse_bytes(entry.as_bytes(), 10)
//                     .expect("Failed to parse the string into a BigUint")
//             })
//             .collect();

//         t.append_point(&proj_form);

//         let final_state_from_rust = t.state.state[1];
//         let final_state_from_circom = Fr::from(result[0].clone());
//         assert_eq!(final_state_from_rust, final_state_from_circom);

//         let finaln_rounds_from_rust = t.n_rounds;
//         let finaln_rounds_from_circom = Fr::from(result[1].clone());
//         assert_eq!(Fr::from(finaln_rounds_from_rust), finaln_rounds_from_circom);
//         println!(
//             "finalnRounds_from_circom are {:?}",
//             finaln_rounds_from_circom
//         );
//     } else {
//         eprintln!("The JSON is not an array or 'witness' field is missing");
//     }

// }

// #[test]
// fn test_append_points(){
//     let mut rng = ark_std::test_rng();
//     let mut t = <PoseidonTranscript<Fr, Fr> as Transcript>::new(b"label");
//     let state = t.state.state[1];
//     let nrounds = Fr::from(t.n_rounds);
//     println!("n rounds are {:?}", nrounds);

//     let testinput_transcript = convert_transcript_fr_fr_to_circom(t.clone());

//     let mut points_to_append: [G1Affine; SCALAR_LEN] = [G1Affine::identity(); SCALAR_LEN];
//     for i in 0..SCALAR_LEN {
//         points_to_append[i] = G1Affine::rand(&mut rng);
//     }

//     let mut formatted_points_to_append_in_circom =  Vec::new();
//     for i in 0..points_to_append.len(){
//         formatted_points_to_append_in_circom.push(
//             G1AffineCircom{
//                 x: convert_rust_fp_to_circom(&points_to_append[i].x().unwrap()),
//                 y: convert_rust_fp_to_circom(&points_to_append[i].y().unwrap()),
//             }
//         )
//     }

//     let mut proj_forms = Vec::new();
//     for i in 0..points_to_append.len(){
//         proj_forms.push(
//             G1Projective {
//                 x: points_to_append[i].x,
//                 y: points_to_append[i].y,
//                 z: Fq::ONE,
//             }
//         )
//     }

//     let input_json = format!(
//         r#"{{
//         "points": {:?},
//         "transcript": {:?}
//         }}"#,
//         formatted_points_to_append_in_circom, testinput_transcript
//     );

//     let input_file_path = "input.json";
//     let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
//     input_file
//         .write_all(input_json.as_bytes())
//         .expect("Failed to write to input.json");
//     println!("Input JSON file created successfully.");

//     // Step 3: Call shell script to compile and generate witness
//     let script_path = "./../../scripts/compile_and_generate_witness_bn_scalar.sh";
//     let circom_template = "AppendPoints";
//     let circom_file_path = "./transcript.circom";
//     let circom_file = "transcript";
//     let nPointstr = &SCALAR_LEN.to_string();

//     let output = Command::new(script_path)
//         .args(&[
//             circom_file_path,
//             circom_template,
//             input_file_path,
//             circom_file,
//             nPointstr,
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
//             .take(2)
//             .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
//             .collect();

//         let result: Vec<BigUint> = result
//             .into_iter()
//             .map(|entry| {
//                 BigUint::parse_bytes(entry.as_bytes(), 10)
//                     .expect("Failed to parse the string into a BigUint")
//             })
//             .collect();

//         t.append_points(&proj_forms);

//         let final_state_from_rust = t.state.state[1];
//         let final_state_from_circom = Fr::from(result[0].clone());
//         assert_eq!(final_state_from_rust, final_state_from_circom);

//         let finaln_rounds_from_rust = t.n_rounds;
//         let finaln_rounds_from_circom = Fr::from(result[1].clone());
//         assert_eq!(Fr::from(finaln_rounds_from_rust), finaln_rounds_from_circom);
//         println!(
//             "finalnRounds_from_circom are {:?}",
//             finaln_rounds_from_circom
//         );
//     } else {
//         eprintln!("The JSON is not an array or 'witness' field is missing");
//     }

// }

// // const SCALAR_LEN: usize = 5;

// #[test]
// fn testing_challenge_scalar() {
//     let mut rng = ark_std::test_rng();
//     let mut t = <PoseidonTranscript<Fr, Fr> as Transcript>::new(b"label");
//     let state = t.state.state[1];
//     let nrounds = Fr::from(t.n_rounds);

//     let testinput_transcript = convert_transcript_fr_fr_to_circom(t.clone());

//     let input_json = format!(
//         r#"{{
//         "transcript": {:?}
//         }}"#,
//         testinput_transcript
//     );

//     let input_file_path = "input.json";
//     let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
//     input_file
//         .write_all(input_json.as_bytes())
//         .expect("Failed to write to input.json");
//     println!("Input JSON file created successfully.");

//     // Step 3: Call shell script to compile and generate witness
//     let script_path = "./../../scripts/compile_and_generate_witness_bn_scalar.sh";
//     let circom_template = "ChallengeScalar";
//     let circom_file_path = "./transcript.circom";
//     let circom_file = "transcript";
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

//         let circom_result_transcript = TestTranscript {
//             state: Fr::from(result[0].clone()),
//             nrounds: Fr::from(result[1].clone()),
//         };

//         let circom_output_challenge = Fr::from(result[2].clone());

//         let rust_challenge = t.challenge_scalar::<Fr>();

//         assert_eq!(circom_output_challenge, rust_challenge);
//         // add assert for limbs after correcting

//         let final_state_from_rust = t.state.state[1];
//         let final_state_from_circom = circom_result_transcript.state;
//         assert_eq!(final_state_from_rust, final_state_from_circom);
//     } else {
//         eprintln!("The JSON is not an array or 'witness' field is missing");
//     }
// }

// #[test]
// fn testing_challenge_vector() {
//     let mut rng = ark_std::test_rng();
//     let mut t = <PoseidonTranscript<Fr, Fr> as Transcript>::new(b"label");
//     let state = t.state.state[1];
//     let nrounds = Fr::from(t.n_rounds);

//     let testinput_transcript = convert_transcript_fr_fr_to_circom(t.clone());

//     let input_json = format!(
//         r#"{{
//         "transcript": {:?}
//         }}"#,
//         testinput_transcript
//     );

//     let input_file_path = "input.json";
//     let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
//     input_file
//         .write_all(input_json.as_bytes())
//         .expect("Failed to write to input.json");
//     println!("Input JSON file created successfully.");

//     // Step 3: Call shell script to compile and generate witness
//     let script_path = "./../../scripts/compile_and_generate_witness_bn_scalar.sh";
//     let circom_template = "ChallengeVector";
//     let circom_file_path = "./transcript.circom";
//     let circom_file = "transcript";
//     let output = Command::new(script_path)
//         .args(&[
//             circom_file_path,
//             circom_template,
//             input_file_path,
//             circom_file,
//             &SCALAR_LEN.to_string()
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
//             .take(2 + SCALAR_LEN)
//             .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
//             .collect();

//         let result: Vec<BigUint> = result
//             .into_iter()
//             .map(|entry| {
//                 BigUint::parse_bytes(entry.as_bytes(), 10)
//                     .expect("Failed to parse the string into a BigUint")
//             })
//             .collect();

//         let circom_result_transcript = TestTranscript {
//             state: Fr::from(result[0].clone()),
//             nrounds: Fr::from(result[1].clone()),
//         };

//         let mut circom_output_challenge = Vec::new();
//         for i in 0..SCALAR_LEN{
//             circom_output_challenge.push(Fr::from(result[2 + i].clone()));
//         }

//         let rust_challenge = t.challenge_vector(SCALAR_LEN);

//         for i in 0..SCALAR_LEN{
//             assert_eq!(circom_output_challenge[i], rust_challenge[i]);

//         }
//         // add assert for limbs after correcting

//         let final_state_from_rust = t.state.state[1];
//         let final_state_from_circom = circom_result_transcript.state;
//         assert_eq!(final_state_from_rust, final_state_from_circom);
//     } else {
//         eprintln!("The JSON is not an array or 'witness' field is missing");
//     }
// }

#[test]
fn test_transcript_new() {
    let mut t = <PoseidonTranscript<Fr, Fr> as Transcript>::new(b"Jolt transcript");
    let state = t.state.state[1];

    println!("state is {:?}", state);
    let nrounds = Fr::from(t.n_rounds);
    println!("n rounds are {:?}", nrounds);

    let x = BigUint::from_str("604586419824232873836833680384618314");
    // let scalar = Fr::from();

    let input_json = format!(
        r#"{{
            "scalar": "604586419824232873836833680384618314"
        }}"#,
    );

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../scripts/compile_and_generate_witness_bn_scalar.sh";
    let circom_template = "TranscriptNew";
    let circom_file_path = "./transcript.circom";
    let circom_file = "transcript";

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

        let final_state_from_rust = t.state.state[1];
        let final_state_from_circom = Fr::from(result[0].clone());
        assert_eq!(final_state_from_rust, final_state_from_circom);

        println!("final_state_from_rust is {:?}", final_state_from_rust);

        let finaln_rounds_from_rust = t.n_rounds;
        let finaln_rounds_from_circom = Fr::from(result[1].clone());
        assert_eq!(Fr::from(finaln_rounds_from_rust), finaln_rounds_from_circom);
        println!(
            "finalnRounds_from_circom are {:?}",
            finaln_rounds_from_circom
        );
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }
}

// #[test]
// fn test_all(){
//     testing_transcript_append_scalar();
//     testing_transcript_append_scalars();
//     testing_challenge_vector();
//     testing_challenge_scalar();
//     testing_challenge_scalar_powers();
//     test_append_point();
//     test_append_points();
// }
