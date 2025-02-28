// use ark_ec::CurveGroup;
// use ark_ff::UniformRand;
// use ark_ff::{AdditiveGroup, Field, PrimeField};
// use ark_grumpkin::{Fr as GrumpkinScalar, Fq as GrumpkinBase, Projective};
// use matrix::check_inner_product;
// use num_bigint::BigUint;
// use parsing::grumpkin_group::{convert_from_3_limbs, GrumpkinProjectiveCircom, GrumpkinScalarCircom};
// use std::ops::Mul;
// use rand::{thread_rng, Rng};
// use serde_json::Value;
// use std::{fmt, fs::File, io::{Read, Write}, process::Command};

// #[test]
// fn grumpkin_add(){
//     let mut rng = thread_rng();

//     let a = Projective::rand(&mut rng);
//     let b = Projective::rand(&mut rng);
//     let sum = a + b;

//     let a_circom =  GrumpkinProjectiveCircom{
//         x: a.x,
//         y: a.y,
//         z: a.z,
//     };

//     let b_circom =  GrumpkinProjectiveCircom{
//         x: b.x,
//         y: b.y,
//         z: b.z,
//     };

//     let input_json = format!(
//         r#"{{
//             "op1": {:?},
//             "op2": {:?}
//         }}"#,
//         a_circom, b_circom
//     );

//     let input_file_path = "input.json";
//     let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
//     input_file
//         .write_all(input_json.as_bytes())
//         .expect("Failed to write to input.json");
//     println!("Input JSON file created successfully.");

//     let script_path = "./../../scripts/compile_and_generate_witness_bn_scalar.sh";
//     let circom_template = "G1Add";
//     let circom_file_path = "./grumpkin_g1.circom";
//     let circom_file = "grumpkin_g1";

//     let output = Command::new(script_path)
//     .args(&[
//         circom_file_path,
//         circom_template,
//         input_file_path,
//         circom_file,
//     ])
//     .output()
//     .expect("Failed to execute shell script");

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

//      // Read the witness.json file
//      let witness_file_path = "witness.json"; // Make sure this is the correct path to your witness.json
//      let mut witness_file = File::open(witness_file_path).expect("Failed to open witness.json");
 
//      // Read the file contents
//      let mut witness_contents = String::new();
//      witness_file
//          .read_to_string(&mut witness_contents)
//          .expect("Failed to read witness.json");
 
//      // Parse the JSON contents
//      let witness_json: Value =
//          serde_json::from_str(&witness_contents).expect("Failed to parse witness.json");

         
//          println!("fine till here");
//          if let Some(witness_array) = witness_json.as_array() {
//             let result: Vec<String> = witness_array
//                 .iter()
//                 .skip(1)
//                 .take(3)
//                 .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
//                 .collect();
    
//             let result: Vec<BigUint> = result
//                 .into_iter()
//                 .map(|entry| {
//                     BigUint::parse_bytes(entry.as_bytes(), 10)
//                         .expect("Failed to parse the string into a BigUint")
//                 })
//                 .collect();
    
//             let mut computed_sum = Projective::default();
    
//             if (result[2] != BigUint::ZERO) {
//                 computed_sum = Projective::new(
//                     GrumpkinBase::from(result[0].clone()) * GrumpkinBase::from(result[2].clone()).inverse().unwrap(),
//                     GrumpkinBase::from(result[1].clone()) * GrumpkinBase::from(result[2].clone()).inverse().unwrap(),
//                     GrumpkinBase::from(1),
//                 );
//             } else {
//                 computed_sum = Projective::new(
//                     GrumpkinBase::from(result[0].clone()),
//                     GrumpkinBase::from(result[1].clone()),
//                     GrumpkinBase::from(result[2].clone()),
//                 );
//             }
    
//             assert_eq!(computed_sum, sum);
    
//             println!("Sum: {}", sum);
//             println!("Result: {} \n", computed_sum);
//         } else {
//             eprintln!("The JSON is not an array or 'witness' field is missing");
//         }
//         check_inner_product(vec![circom_file.to_string()]);

// }

// #[test]
// fn grumpkin_double(){
//     let mut rng = thread_rng();

//     let a = Projective::rand(&mut rng);
//     let sum = a.double();

//     let a_circom =  GrumpkinProjectiveCircom{
//         x: a.x,
//         y: a.y,
//         z: a.z,
//     };

//     let input_json = format!(
//         r#"{{
//             "op1": {:?}
//         }}"#,
//         a_circom
//     );

//     let input_file_path = "input.json";
//     let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
//     input_file
//         .write_all(input_json.as_bytes())
//         .expect("Failed to write to input.json");
//     println!("Input JSON file created successfully.");

//     let script_path = "./../../scripts/compile_and_generate_witness_bn_scalar.sh";
//     let circom_template = "G1Double";
//     let circom_file_path = "./grumpkin_g1.circom";
//     let circom_file = "grumpkin_g1";

//     let output = Command::new(script_path)
//     .args(&[
//         circom_file_path,
//         circom_template,
//         input_file_path,
//         circom_file,
//     ])
//     .output()
//     .expect("Failed to execute shell script");

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

//      // Read the witness.json file
//      let witness_file_path = "witness.json"; // Make sure this is the correct path to your witness.json
//      let mut witness_file = File::open(witness_file_path).expect("Failed to open witness.json");
 
//      // Read the file contents
//      let mut witness_contents = String::new();
//      witness_file
//          .read_to_string(&mut witness_contents)
//          .expect("Failed to read witness.json");
 
//      // Parse the JSON contents
//      let witness_json: Value =
//          serde_json::from_str(&witness_contents).expect("Failed to parse witness.json");

         
//          println!("fine till here");
//          if let Some(witness_array) = witness_json.as_array() {
//             let result: Vec<String> = witness_array
//                 .iter()
//                 .skip(1)
//                 .take(3)
//                 .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
//                 .collect();
    
//             let result: Vec<BigUint> = result
//                 .into_iter()
//                 .map(|entry| {
//                     BigUint::parse_bytes(entry.as_bytes(), 10)
//                         .expect("Failed to parse the string into a BigUint")
//                 })
//                 .collect();
    
//             let mut computed_sum = Projective::default();
    
//             if (result[2] != BigUint::ZERO) {
//                 computed_sum = Projective::new(
//                     GrumpkinBase::from(result[0].clone()) * GrumpkinBase::from(result[2].clone()).inverse().unwrap(),
//                     GrumpkinBase::from(result[1].clone()) * GrumpkinBase::from(result[2].clone()).inverse().unwrap(),
//                     GrumpkinBase::from(1),
//                 );
//             } else {
//                 computed_sum = Projective::new(
//                     GrumpkinBase::from(result[0].clone()),
//                     GrumpkinBase::from(result[1].clone()),
//                     GrumpkinBase::from(result[2].clone()),
//                 );
//             }
    
//             assert_eq!(computed_sum, sum);
    
//             println!("Sum: {}", sum);
//             println!("Result: {} \n", computed_sum);
//         } else {
//             eprintln!("The JSON is not an array or 'witness' field is missing");
//         }
//         check_inner_product(vec![circom_file.to_string()]);

// }


// #[test]
// fn test_grumpkin_mul() {
//     let mut rng = thread_rng();

//     let a = Projective::rand(&mut rng);

//     // let a = a - a;

    
//     // Neither of the follwoing work. Will have to find a way to put point at infinity in input.json.
//     //
//     // let mut a = G1Projective::ZERO;
//     //
//     // let mut a = G1Projective::rand(&mut rng);
//     // a = a - a;
//     let a_circom =  GrumpkinProjectiveCircom{
//         x: a.x,
//         y: a.y,
//         z: a.z,
//     };

//     let mut scalar_bar = [GrumpkinBase::from(0u8); 3];

//     for i in 0..2 {
//         scalar_bar[i] = GrumpkinBase::from(rng.gen_range(0..(1u128 << 125)));

//     }
//     scalar_bar[2] = GrumpkinBase::from(rng.gen_range(0..(1u128 << 3)));

//     let scalar = convert_from_3_limbs(scalar_bar.to_vec());
//     let op2 = GrumpkinScalarCircom{
//         // element: scalar,
//         limbs: scalar_bar,
//     };

//     let prod = a.mul(scalar);

    
//     let input_json = format!(
//         r#"{{
//             "op1": {:?},
//             "op2": {:?}
//         }}"#,
//         a_circom, op2
//     );

//     // let input_json = string_for_G1_input(&input_json);

//     let input_file_path = "input.json";
//     let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
//     input_file
//         .write_all(input_json.as_bytes())
//         .expect("Failed to write to input.json");
//     println!("Input JSON file created successfully.");

//     // Step 3: Call shell script to compile and generate witness
//     let script_path = "./../../scripts/compile_and_generate_witness_bn_scalar.sh";
//     let circom_template = "G1Mul";
//     let circom_file_path = "./grumpkin_g1.circom";
//     let circom_file = "grumpkin_g1";

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

//         let mut computed_product = Projective::default();

//         if (result[2] != BigUint::ZERO) {
//             computed_product = Projective::new(
//                 GrumpkinBase::from(result[0].clone()) * GrumpkinBase::from(result[2].clone()).inverse().unwrap(),
//                 GrumpkinBase::from(result[1].clone()) * GrumpkinBase::from(result[2].clone()).inverse().unwrap(),
//                 GrumpkinBase::from(1),
//             );
//         } else {
//             computed_product = Projective::new(
//                 GrumpkinBase::from(result[0].clone()),
//                 GrumpkinBase::from(result[1].clone()),
//                 GrumpkinBase::from(result[2].clone()),
//             );
//         }

//         assert_eq!(computed_product.into_affine(), prod.into_affine());

//         println!("Product: {}", prod);
//         println!("Result: {} \n", computed_product);
//     } else {
//         eprintln!("The JSON is not an array or 'witness' field is missing");
//     }

//     check_inner_product(vec![circom_file.to_string()]);
// }


// #[test]
// fn test_all(){
//     test_grumpkin_mul();
//     grumpkin_add();
//     grumpkin_double();
// }
