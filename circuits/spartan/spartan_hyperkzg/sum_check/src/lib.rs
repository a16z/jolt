extern crate jolt_core;
use ark_bn254::{Fq, Fr};
use ark_ff::{AdditiveGroup, UniformRand};
use ark_ff::{Field, PrimeField};
use jolt_core::test_circom_link::link_opening_combiners::Fqq;
use jolt_core::utils::{poseidon_transcript::PoseidonTranscript, transcript::Transcript};
use jolt_core::{
    jolt::vm::rv32i_vm::ProofTranscript, poly::dense_mlpoly::DensePolynomial,
    subprotocols::sumcheck::SumcheckInstanceProof,
};
use jolt_core::poly::unipoly::UniPoly;
// use matrix::check_inner_product;
use num_bigint::BigUint;
use parsing::non_native::{convert_from_3_limbs, convert_to_3_limbs};
use parsing::spartan_sumcheck::convert_sum_check_proof_to_circom;
use parsing::transcript_bn_fq::TestTranscript;
use regex::Regex;
use serde_json::Value;
use std::{
    fs::File,
    io::{Read, Write},
    process::Command,
};
// use transcript::helper::{convert_from_3_limbs, convert_to_3_limbs, TestTranscript};

fn format_uni_polys(json_str: &str) -> String {
    // Define a regex pattern to target the double brackets in "uni_polys"
    let re = Regex::new(r#""uni_polys":\s*\[\[(.*?)\]\]"#).unwrap();

    // Replace the double brackets with single brackets
    let formatted_json = re.replace(json_str, |caps: &regex::Captures| {
        format!(r#""uni_polys": [{}]"#, &caps[1])
    });

    formatted_json.to_string()
}



#[test]
fn test_non_native_sum_check() {
    const NO_OF_VARS: usize = 3;
    const DEGREE: usize = 1;
    let mut rng = ark_std::test_rng();
    let mut poly: Vec<Fr> = Vec::new();
    for i in 0..(1 << NO_OF_VARS) {
        poly.push(Fr::rand(&mut rng));
    }

    let mut initial_claim = Fr::ZERO;
    for i in 0..8 {
        initial_claim = initial_claim + poly[i];
    }
    let mut polys = DensePolynomial::new(poly.clone());
    let polys_copy = DensePolynomial::new(poly.clone());

    let output_check_fn = |vals: &[Fr]| -> Fr { vals[0] };
    let mut transcript = <PoseidonTranscript<Fr, Fq> as Transcript>::new(b"label");
    let (proof, r, final_evals) = SumcheckInstanceProof::prove_arbitrary(
        &initial_claim,
        NO_OF_VARS,
        &mut vec![polys],
        output_check_fn,
        1,
        &mut transcript,
    );

    // verifier
    let mut transcript = <PoseidonTranscript<Fr, Fq> as Transcript>::new(b"label");
    let result = proof.verify(initial_claim, 3, 1, &mut transcript);
    assert_eq!(final_evals[0], polys_copy.evaluate(&r));

    let initial_claim = Fqq {
        // element: initial_claim,
        limbs: convert_to_3_limbs(initial_claim),
    };

    let proof_instance = convert_sum_check_proof_to_circom(&proof);
    let transcript = <PoseidonTranscript<Fr, Fq> as Transcript>::new(b"label");

    let transcript = TestTranscript {
        state: transcript.state.state[1],
        nrounds: Fq::from(transcript.n_rounds),
    };

    let input_json = format!(
        r#"{{
        "initialClaim": {:?},
        "sumcheck_proof": {:?},
        "transcript": {:?}
        }}"#,
        initial_claim, proof_instance, transcript
    );

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "NonNativeSumCheck";
    let circom_file_path = "./sumcheck.circom";
    let circom_file = "sumcheck";
    let rounds = NO_OF_VARS.to_string();
    let degree = DEGREE.to_string();

    let output = Command::new(script_path)
        .args(&[
            circom_file_path,
            circom_template,
            input_file_path,
            circom_file,
            &rounds,
            &degree,
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
            .take(9)
            .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
            .collect();

        let result: Vec<BigUint> = result
            .into_iter()
            .map(|entry| {
                BigUint::parse_bytes(entry.as_bytes(), 10)
                    .expect("Failed to parse the string into a BigUint")
            })
            .collect();

        let circom_final_claim = Fqq {
            // element: Fr::from(result[2].clone()),
            limbs: [
                Fq::from(result[3].clone()),
                Fq::from(result[4].clone()),
                Fq::from(result[5].clone()),
            ],
        };
        let circom_final_claim_in_scalar_form =
            convert_from_3_limbs(circom_final_claim.limbs.to_vec());
        assert_eq!(circom_final_claim_in_scalar_form, final_evals[0]);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    // check_inner_product(vec![circom_file.to_string()]);
}


