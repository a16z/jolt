// extern crate jolt_core;
// use ark_bn254::Fr;
// use ark_ff::AdditiveGroup;
// use ark_ff::Field;
// use ark_ff::UniformRand;
// use jolt_core::poly::unipoly::UniPoly;

// use jolt_core::utils::{poseidon_transcript::PoseidonTranscript, transcript::Transcript};
// use jolt_core::{
//     jolt::vm::rv32i_vm::ProofTranscript, poly::dense_mlpoly::DensePolynomial,
//     subprotocols::sumcheck::SumcheckInstanceProof,
// };
// use num_bigint::BigUint;
// use serde_json::Value;
// use std::{
//     fs::File,
//     io::{Read, Write},
//     process::Command,
// };
// mod helper;
// #[test]
// fn test_non_native_sum_check() {
//     const NO_OF_VARS: usize = 3;
//     const DEGREE: usize = 1;
//     let mut rng = ark_std::test_rng();
//     let mut poly: Vec<Fr> = Vec::new();
//     for i in 0..(1 << NO_OF_VARS) {
//         poly.push(Fr::rand(&mut rng));
//     }

//     let mut initial_claim = Fr::ZERO;
//     for i in 0..8 {
//         initial_claim = initial_claim + poly[i];
//     }
//     let mut polys = DensePolynomial::new(poly.clone());
//     let polys_copy = DensePolynomial::new(poly.clone());

//     let output_check_fn = |vals: &[Fr]| -> Fr { vals[0] };
//     let mut prover_final_transcript_from_rust =
//         <PoseidonTranscript<Fr, Fr> as Transcript>::new(b"label");
//     let (proof, r, final_evals) = SumcheckInstanceProof::prove_arbitrary(
//         &initial_claim,
//         NO_OF_VARS,
//         &mut vec![polys],
//         output_check_fn,
//         1,
//         &mut prover_final_transcript_from_rust,
//     );

//     // verifier
//     let mut transcript = <PoseidonTranscript<Fr, Fr> as Transcript>::new(b"label");
//     let result = proof.verify(initial_claim, 3, 1, &mut transcript);
//     assert_eq!(final_evals[0], polys_copy.evaluate(&r));

//     let proof_instance = convert_sum_check_proof_to_circom(&proof);
//     let transcript = <PoseidonTranscript<Fr, Fr> as Transcript>::new(b"label");

//     let transcript = convert_transcript_to_circom(transcript);

//     let input_json = format!(
//         r#"{{
//         "initialClaim": "{}",
//         "sumcheck_proof": {:?},
//         "transcript": {:?}
//         }}"#,
//         initial_claim, proof_instance, transcript
//     );

//     let input_file_path = "input.json";
//     let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
//     input_file
//         .write_all(input_json.as_bytes())
//         .expect("Failed to write to input.json");
//     println!("Input JSON file created successfully.");

//     // Step 3: Call shell script to compile and generate witness
//     let script_path = "./../../scripts/compile_and_generate_witness.sh";
//     let circom_template = "SumCheck";
//     let circom_file_path = "./sumcheck.circom";
//     let circom_file = "sumcheck";
//     let rounds = NO_OF_VARS.to_string();
//     let degree = DEGREE.to_string();

//     let output = Command::new(script_path)
//         .args(&[
//             circom_file_path,
//             circom_template,
//             input_file_path,
//             circom_file,
//             &rounds,
//             &degree,
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
//             .take(3 + NO_OF_VARS)
//             .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
//             .collect();

//         let result: Vec<BigUint> = result
//             .into_iter()
//             .map(|entry| {
//                 BigUint::parse_bytes(entry.as_bytes(), 10)
//                     .expect("Failed to parse the string into a BigUint")
//             })
//             .collect();

//         let circom_final_result = Fr::from(result[2].clone());
//         assert_eq!(circom_final_result, final_evals[0]);

//         assert_eq!(
//             prover_final_transcript_from_rust.state.state[1],
//             Fr::from(result[0].clone())
//         );
//         assert_eq!(
//             Fr::from(prover_final_transcript_from_rust.n_rounds),
//             Fr::from(result[1].clone())
//         );
//     } else {
//         eprintln!("The JSON is not an array or 'witness' field is missing");
//     }

//     // check_inner_product(vec![circom_file.to_string()]);
// }
