// use ark_bn254::Fq;
// use ark_crypto_primitives::sponge::{poseidon::{get_poseidon_parameters, PoseidonConfig, PoseidonDefaultConfigEntry, PoseidonSponge}, CryptographicSponge, DuplexSpongeMode};
// use ark_ff::UniformRand;
// use num_bigint::BigUint;
// use std::{fmt, fs::File, io::{Read, Write}, process::Command};
// use serde_json::Value;
// // use matrix::check_inner_product;
// const STATE_WIDTH: usize = 5;

// pub struct ARRFp{
//     pub state: Vec<Fq>
// }
// impl fmt::Debug for ARRFp {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(
//             f,
//             r#"[
//             "{}", "{}", "{}", "{}", "{}"
//             ]"#,
//             self.state[0], self.state[1], self.state[2], self.state[3], self.state[4]
//         )
//     }
// }

// #[test]
// fn poseidon_bn_base(){
//     let mut rng = ark_std::test_rng();
    
//     let mut initial_state: Vec<Fq> = Vec::new();
//     for _ in 0..STATE_WIDTH{
//         initial_state.push(Fq::rand(&mut rng));
//     }
//     let params =
//             get_poseidon_parameters::<Fq>(4, PoseidonDefaultConfigEntry::new(4, 5, 8, 56, 0))
//                 .unwrap();

//     let mut poseidon_sponge = PoseidonSponge::<Fq>::new(&params);
//     poseidon_sponge.state = initial_state.clone();

//     assert_eq!(poseidon_sponge.state.len(), STATE_WIDTH);

//     let initial_state_arr = ARRFp{state: initial_state};
//     let input_json = format!(
//         r#"{{
//         "state": {:?}
//     }}"#,
//         initial_state_arr
//     );

//     let input_file_path = "input.json";
//     let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
//     input_file
//         .write_all(input_json.as_bytes())
//         .expect("Failed to write to input.json");
//     println!("Input JSON file created successfully.");

//     // Step 3: Call shell script to compile and generate witness
//     let script_path = "./../../scripts/compile_and_generate_witness_bn_base.sh";
//     let circom_template = "permute";
//     let circom_file_path = "./poseidon.circom";
//     let circom_file = "poseidon";

//     // println!("script_path is {:?}", script_path);
//     // println!("circom_template is {:?}", circom_template);
//     // println!("circom_file_path is {:?}", circom_file_path);
//     // println!("circom_file is {:?}", circom_file);

//     let output = Command::new(script_path)
//         .args(&[
//             circom_file_path,
//             circom_template,
//             input_file_path,
//             circom_file,
//             "2"
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
//             .take(5)
//             .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
//             .collect();

//         let result: Vec<BigUint> = result
//             .into_iter()
//             .map(|entry| {
//                 BigUint::parse_bytes(entry.as_bytes(), 10)
//                     .expect("Failed to parse the string into a BigUint")
//             })
//             .collect();

//         let mut circom_result = Vec::new();
//         for i in 0..5{
//             circom_result.push(Fq::from(result[i].clone()))
//         }

//         poseidon_sponge.permute();
//         let result_from_rust = poseidon_sponge.state;
//         assert_eq!(result_from_rust, circom_result);
//         assert_eq!(result_from_rust.len(), STATE_WIDTH);
//     } else {
//         eprintln!("The JSON is not an array or 'witness' field is missing");
//     }

//     // check_inner_product(vec![circom_file.to_string()]);
// }