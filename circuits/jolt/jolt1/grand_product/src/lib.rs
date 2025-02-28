// #![allow(unused_imports)]
// extern crate jolt_core;
// use ark_bn254::{Bn254, Fq as Fp, Fr as Scalar};
// use ark_ff::UniformRand;
// use ark_ff::{Field, PrimeField};
// use ark_std::log2;
// use jolt_core::field::JoltField;
// use jolt_core::poly::unipoly::UniPoly;
// use jolt_core::subprotocols::grand_product::{BatchedGrandProduct, BatchedGrandProductProof};
// use jolt_core::test_circom_new::struct_fq::FqCircom;
// use jolt_core::test_circom_new::sum_check_gkr::convert_from_batched_GKRProof_to_circom;
// use jolt_core::test_circom_new::transcript::convert_transcript_to_circom;
// // use helper::{Fqq, SumcheckInstanceProofCircom, UniPolyCircom};
// use jolt_core::utils::{poseidon_transcript::PoseidonTranscript, transcript::Transcript};
// use jolt_core::{
//     jolt::vm::rv32i_vm::ProofTranscript, poly::dense_mlpoly::DensePolynomial,
//     subprotocols::grand_product::BatchedDenseGrandProduct,
//     subprotocols::sumcheck::SumcheckInstanceProof,
// };
// use rayon::prelude::*;

// use itertools::Itertools;
// use jolt_core::poly::commitment::hyperkzg::{self, HyperKZG};
// use num_bigint::BigUint;
// use regex::Regex;
// use serde_json::Value;
// use std::{
//     fs::File,
//     io::{Read, Write},
//     process::Command,
// };
// use transcript::helper::{convert_from_3_limbs, convert_to_3_limbs, TestTranscript};

// #[test]
// fn grand_product() {
//     let mut rng = ark_std::test_rng();
//     let layer_size = 16;
//     let batch_size = 80;

//     assert!(batch_size > 1);
//     let leaves: Vec<Vec<Scalar>> = std::iter::repeat_with(|| {
//         std::iter::repeat_with(|| Scalar::rand(&mut rng))
//             .take(layer_size)
//             .collect::<Vec<_>>()
//     })
//     .take(batch_size)
//     .collect();

//     let mut batched_circuit = <BatchedDenseGrandProduct<Scalar> as BatchedGrandProduct<
//         Scalar,
//         HyperKZG<Bn254, PoseidonTranscript<Scalar>>,
//         PoseidonTranscript<Scalar>,
//     >>::construct((leaves.concat(), batch_size));

//     let mut prover_transcript = <PoseidonTranscript<Scalar> as Transcript>::new(b"label");

//     let claims: Vec<Scalar> = <BatchedDenseGrandProduct<Scalar> as BatchedGrandProduct<
//         Scalar,
//         HyperKZG<Bn254, PoseidonTranscript<Scalar>>,
//         PoseidonTranscript<Scalar>,
//     >>::claimed_outputs(&batched_circuit);
//     // println!("len of claims is {}", claims.len());

//     let (proof, r_prover) =
//         <BatchedDenseGrandProduct<Scalar> as BatchedGrandProduct<
//             Scalar,
//             HyperKZG<Bn254, PoseidonTranscript<Scalar>>,
//             PoseidonTranscript<Scalar>,
//         >>::prove_grand_product(&mut batched_circuit, None, &mut prover_transcript, None);
//     println!("called prove_grand_product");

//     // Verifier
//     let mut verifier_transcript: PoseidonTranscript<Scalar> =
//         <PoseidonTranscript<Scalar> as Transcript>::new(b"label");

//     // verifier_transcript.compare_to(prover_transcript);
//     let (final_claim, r_verifier) = BatchedDenseGrandProduct::verify_grand_product(
//         &proof,
//         &claims,
//         None,
//         &mut verifier_transcript,
//         None,
//     );
//     assert_eq!(r_prover, r_verifier);

//     let prover_transcript = <PoseidonTranscript<Scalar> as Transcript>::new(b"label");

//     let test_transcript = convert_transcript_to_circom(prover_transcript);

//     println!("num of layers = {} ", proof.gkr_layers.len());
//     for i in 0..proof.gkr_layers.len() {
//         println!(
//             "no of coeff {:?} ",
//             proof.gkr_layers[i].proof.uni_polys.len()
//         );
//         for j in 0..proof.gkr_layers[i].proof.uni_polys.len() {
//             println!(
//                 "degree {:?} ",
//                 proof.gkr_layers[i].proof.uni_polys[j].coeffs.len()
//             );
//         }
//     }
//     let num_gkr_layers = proof.gkr_layers.len();
//     let num_coeffs = proof.gkr_layers[num_gkr_layers - 1].proof.uni_polys[0]
//         .coeffs
//         .len();
//     let max_no_polys = proof.gkr_layers[num_gkr_layers - 1].proof.uni_polys.len();
//     println!("max_rounds = {}", max_no_polys);

//     let batched_proof = convert_from_batched_GKRProof_to_circom(&proof);

//     let mut claimed_output = Vec::new();
//     for i in 0..claims.len(){
//         claimed_output.push(FqCircom(Scalar::from(claims[i].clone())));
//     }

//     let input_json = format!(
//         r#"{{
//                 "proof": {:?},
//                 "claimed_outputs": {:?},
//                 "transcript": {:?}
//                 }}"#,
//         batched_proof, claimed_output, test_transcript
//     );

//     let input_file_path = "input.json";

//     let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
//     input_file
//         .write_all(input_json.as_bytes())
//         .expect("Failed to write to input.json");
//     println!("Input JSON file created successfully.");

//     // Step 3: Call shell script to compile and generate witness
//     let script_path = "./../../scripts/compile_and_generate_witness.sh";
//     let circom_template = "VerifyGrandProduct";
//     let circom_file_path = "./grand_product.circom";
//     let circom_file = "grand_product";

//     let max_rounds = max_no_polys.to_string();
//     let proof_layers = num_gkr_layers.to_string();
//     let len_of_claims = claimed_output.len().to_string();

//     let output = Command::new(script_path)
//         .args(&[
//             circom_file_path,
//             circom_template,
//             input_file_path,
//             circom_file,
//             &max_rounds,
//             &proof_layers,
//             &len_of_claims,
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

//     let r_grand_product_len = log2(claims.len().next_power_of_two()) as usize + num_gkr_layers;

//     if let Some(witness_array) = witness_json.as_array() {
//         let result: Vec<String> = witness_array
//             .iter()
//             .skip(1)
//             .take(6 + 4 * r_grand_product_len)
//             .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
//             .collect();

//         let result: Vec<BigUint> = result
//             .into_iter()
//             .map(|entry| {
//                 BigUint::parse_bytes(entry.as_bytes(), 10)
//                     .expect("Failed to parse the string into a BigUint")
//             })
//             .collect();

//         let circom_transcript = TestTranscript {
//             state: Scalar::from(result[0].clone()),
//             nrounds: Scalar::from(result[1].clone()),
//         };

//         let circom_final_claim = Scalar::from(result[2].clone());
//         assert_eq!(circom_transcript.state, verifier_transcript.state.state[1]);
//         assert_eq!(
//             circom_transcript.nrounds,
//             Scalar::from(verifier_transcript.n_rounds)
//         );
//         assert_eq!(
//             circom_final_claim,
//             final_claim
//         );
//         for i in 0..r_grand_product_len {
//             assert_eq!(
//                 r_verifier[i],
//                 Scalar::from(result[3 + i].clone())
//             )
//         }
//     } else {
//         eprintln!("The JSON is not an array or 'witness' field is missing");
//     }
// }
