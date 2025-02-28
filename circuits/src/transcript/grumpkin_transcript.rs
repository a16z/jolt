// pub mod helper;
use ark_ff::{AdditiveGroup, Field, PrimeField, UniformRand};
use num_bigint::{BigInt, BigUint};
use parsing::{grumpkin_group::{convert_from_3_limbs, convert_to_3_limbs, GrumpkinScalarCircom}, grumpkin_transcript::{convert_grumpkin_transcript_to_circom, AffineFormTest, TestTranscript}, SCALAR_LEN};
use serde_json::Value;
use std::{
    fmt,
    fs::File,
    io::{Read, Write},
    process::Command, str::FromStr,
};
use ark_grumpkin::{Affine, Fq as GrumpkinBase, Fr as GrumpkinScalar, GrumpkinConfig, Projective};
use jolt_core::utils::{poseidon_transcript::PoseidonTranscript, transcript::Transcript};

extern crate jolt_core;
#[test]
fn testing_transcript_append_scalar() {
    let mut rng = ark_std::test_rng();
    let mut t = <PoseidonTranscript<GrumpkinScalar, GrumpkinBase> as Transcript>::new(b"label");

    let state = t.state.state[1];

    let scalar_to_append = GrumpkinScalar::rand(&mut rng);
    println!("t n round is {:?}", t.n_rounds);

    let scalar_for_circom = GrumpkinScalarCircom{
        // element: scalar_to_append,
        limbs: convert_to_3_limbs(scalar_to_append)
    };

    let test_transcipt = TestTranscript{
        state: t.state.state[1],
        nrounds: GrumpkinBase::from(t.n_rounds)
    };


    let input_json = format!(
        r#"{{
        "scalar": {:?},
        "transcript": {:?}
        }}"#,
        scalar_for_circom, test_transcipt
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
    let circom_file_path = "./grumpkin_transcript.circom";
    let circom_file = "grumpkin_transcript";

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
        let final_state_from_circom = GrumpkinBase::from(result[0].clone());
        assert_eq!(final_state_from_rust, final_state_from_circom);

        let finaln_rounds_from_rust = t.n_rounds;
        let finaln_rounds_from_circom = GrumpkinBase::from(result[1].clone());
        assert_eq!(GrumpkinBase::from(finaln_rounds_from_rust), finaln_rounds_from_circom);
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
fn testing_transcript_append_scalars() {
    // let x = Scalar::from(8u64);
    let mut rng = ark_std::test_rng();
    let mut t = <PoseidonTranscript<GrumpkinScalar, GrumpkinBase> as Transcript>::new(b"label");

    let state = t.state.state[1];
    let nrounds = GrumpkinScalar::from(t.n_rounds);
    println!("n rounds are {:?}", nrounds);

    let mut scalars_to_append = Vec::new();
    for i in 0..SCALAR_LEN{ 
        scalars_to_append.push(GrumpkinScalar::rand(&mut rng));
    }

    println!("t n round is {:?}", t.n_rounds);

    let mut scalars_for_circom = Vec::new();
    for i in 0..SCALAR_LEN{ 
        scalars_for_circom.push(GrumpkinScalarCircom{
            // element: scalars_to_append[i],
            limbs: convert_to_3_limbs(scalars_to_append[i])
        });
    }
     ;

    let test_transcipt = TestTranscript{
        state: t.state.state[1],
        nrounds: GrumpkinBase::from(t.n_rounds)
    };


    let input_json = format!(
        r#"{{
        "scalars": {:?},
        "transcript": {:?}
        }}"#,
        scalars_for_circom, test_transcipt
    );

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../scripts/compile_and_generate_witness_bn_scalar.sh";
    let circom_template = "AppendScalars";
    let circom_file_path = "./grumpkin_transcript.circom";
    let circom_file = "grumpkin_transcript";
    let nPointstr = &SCALAR_LEN.to_string();

    let output = Command::new(script_path)
        .args(&[
            circom_file_path,
            circom_template,
            input_file_path,
            circom_file,
            nPointstr
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

        t.append_scalars(&scalars_to_append.clone());

        let final_state_from_rust = t.state.state[1];
        let final_state_from_circom: ark_ff::Fp<ark_ff::MontBackend<ark_bn254::FrConfig, 4>, 4> = GrumpkinBase::from(result[0].clone());
        assert_eq!(final_state_from_rust, final_state_from_circom);

        let finaln_rounds_from_rust = t.n_rounds;
        let finaln_rounds_from_circom = GrumpkinBase::from(result[1].clone());
        assert_eq!(GrumpkinBase::from(finaln_rounds_from_rust), finaln_rounds_from_circom);
        println!(
            "finalnRounds_from_circom are {:?}",
            finaln_rounds_from_circom
        );
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

}

// const SCALAR_LEN: usize = 5;


#[test]
fn testing_transcript_append_point() {
    let mut rng = ark_std::test_rng();
    let mut t = <PoseidonTranscript<GrumpkinScalar, GrumpkinBase> as Transcript>::new(b"label");

    let state = t.state.state[1];
    let nrounds = GrumpkinBase::from(t.n_rounds);
    println!("n rounds are {:?}", nrounds);

    let testinput_transcript = TestTranscript {
        state: state,
        nrounds,
    };

    let point_to_append = Affine::rand(&mut rng);
    let formatted_point_to_append = AffineFormTest {
        x: point_to_append.x,
        y: point_to_append.y,
    };
    let proj_form: Projective = Projective {
        x: point_to_append.x,
        y: point_to_append.y,
        z: GrumpkinBase::ONE,
    };
    let input_json = format!(
        r#"{{
        "point": {:?},
        "transcript": {:?}
        }}"#,
        formatted_point_to_append, testinput_transcript
    );

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../scripts/compile_and_generate_witness_bn_scalar.sh";
    let circom_template = "AppendPoint";
    let circom_file_path = "./grumpkin_transcript.circom";
    let circom_file = "grumpkin_transcript";

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

        let circom_result = TestTranscript {
            state: GrumpkinBase::from(result[0].clone()),
            nrounds: GrumpkinBase::from(result[1].clone()),
        };

        t.append_point(&proj_form);

        let final_state_from_rust = t.state.state[1];
        let final_state_from_circom = circom_result.state;
        assert_eq!(final_state_from_rust, final_state_from_circom);

        let finalnRounds_from_rust = t.n_rounds;
        let finalnRounds_from_circom = circom_result.nrounds;
        assert_eq!(GrumpkinBase::from(finalnRounds_from_rust), finalnRounds_from_circom);
        println!("finalnRounds_from_circom are {:?}", finalnRounds_from_circom);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    // check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn testing_transcript_append_points() {
    let mut rng = ark_std::test_rng();
    let mut t = <PoseidonTranscript<GrumpkinScalar, GrumpkinBase> as Transcript>::new(b"label");

    let state = t.state.state[1];
    let nrounds = GrumpkinBase::from(t.n_rounds);
    println!("n rounds are {:?}", nrounds);

    let testinput_transcript =  convert_grumpkin_transcript_to_circom(t.clone());

    let mut points_to_append = Vec::new();
    for i in 0..SCALAR_LEN{
        points_to_append.push(Affine::rand(&mut rng));
    }
    // let point_to_append = Affine::rand(&mut rng);
    let mut formatted_points_to_append = Vec::new();
    for i in 0..SCALAR_LEN {
        formatted_points_to_append.push(AffineFormTest {
            x: points_to_append[i].x,
            y: points_to_append[i].y,
        });
    }

    let mut proj_forms = Vec::new();
    for i in 0..SCALAR_LEN {
        proj_forms.push(Projective {
            x: points_to_append[i].x,
            y: points_to_append[i].y,
            z: GrumpkinBase::ONE,
        });
    }


    let input_json = format!(
        r#"{{
        "points": {:?},
        "transcript": {:?}
        }}"#,
        formatted_points_to_append, testinput_transcript
    );

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../scripts/compile_and_generate_witness_bn_scalar.sh";
    let circom_template = "AppendPoints";
    let circom_file_path = "./grumpkin_transcript.circom";
    let circom_file = "grumpkin_transcript";
    let nPointstr = &SCALAR_LEN.to_string();

    let output = Command::new(script_path)
        .args(&[
            circom_file_path,
            circom_template,
            input_file_path,
            circom_file,
            nPointstr
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

        let circom_result = TestTranscript {
            state: GrumpkinBase::from(result[0].clone()),
            nrounds: GrumpkinBase::from(result[1].clone()),
        };

        t.append_points(&proj_forms);

        let final_state_from_rust = t.state.state[1];
        let final_state_from_circom = circom_result.state;
        assert_eq!(final_state_from_rust, final_state_from_circom);

        let finalnRounds_from_rust = t.n_rounds;
        let finalnRounds_from_circom = circom_result.nrounds;
        assert_eq!(GrumpkinBase::from(finalnRounds_from_rust), finalnRounds_from_circom);
        println!("finalnRounds_from_circom are {:?}", finalnRounds_from_circom);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    // check_inner_product(vec![circom_file.to_string()]);
}


#[test]
fn testing_challenge_scalar() {
    let mut rng = ark_std::test_rng();
    let mut t = <PoseidonTranscript<GrumpkinScalar, GrumpkinBase> as Transcript>::new(b"label");

    let state = t.state.state[1];

    let testinput_transcript =  convert_grumpkin_transcript_to_circom(t.clone());

    let input_json = format!(
        r#"{{
        "transcript": {:?}
        }}"#,
        testinput_transcript
    );

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../scripts/compile_and_generate_witness_bn_scalar.sh";
    let circom_template = "ChallengeScalar";
    let circom_file_path = "./grumpkin_transcript.circom";
    let circom_file = "grumpkin_transcript";
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

        let circom_result_transcript = TestTranscript {
            state: GrumpkinBase::from(result[0].clone()),
            nrounds: GrumpkinBase::from(result[1].clone()),
        };

        let circom_output_challenge = GrumpkinScalarCircom {
            limbs: [
                GrumpkinBase::from(result[2].clone()),
                GrumpkinBase::from(result[3].clone()),
                GrumpkinBase::from(result[4].clone()),
            ],
        };

        let rust_challenge = t.challenge_scalar::<GrumpkinScalar>();

        assert_eq!(convert_from_3_limbs(circom_output_challenge.limbs.to_vec()), rust_challenge);
        // add assert for limbs after correcting

        let final_state_from_rust = t.state.state[1];
        let final_state_from_circom = circom_result_transcript.state;
        assert_eq!(final_state_from_rust, final_state_from_circom);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }
}

#[test]
fn testing_challenge_vector() {
    let mut rng = ark_std::test_rng();
    let mut t = <PoseidonTranscript<GrumpkinScalar, GrumpkinBase> as Transcript>::new(b"label");

    let state = t.state.state[1];

    let testinput_transcript =  convert_grumpkin_transcript_to_circom(t.clone());

    let input_json = format!(
        r#"{{
        "transcript": {:?}
        }}"#,
        testinput_transcript
    );

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../scripts/compile_and_generate_witness_bn_scalar.sh";
    let circom_template = "ChallengeVector";
    let circom_file_path = "./grumpkin_transcript.circom";
    let circom_file = "grumpkin_transcript";
    let nPointstr = &SCALAR_LEN.to_string();

    let output = Command::new(script_path)
        .args(&[
            circom_file_path,
            circom_template,
            input_file_path,
            circom_file,
            nPointstr
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
            .take(2 + 4 * SCALAR_LEN)
            .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
            .collect();

        let result: Vec<BigUint> = result
            .into_iter()
            .map(|entry| {
                BigUint::parse_bytes(entry.as_bytes(), 10)
                    .expect("Failed to parse the string into a BigUint")
            })
            .collect();

        let circom_result_transcript = TestTranscript {
            state: GrumpkinBase::from(result[0].clone()),
            nrounds: GrumpkinBase::from(result[1].clone()),
        };

        let mut circom_output_challenges: [GrumpkinScalarCircom; SCALAR_LEN] = [GrumpkinScalarCircom {
            // element: GrumpkinScalar::ZERO,
            limbs: [GrumpkinBase::ZERO; 3],
        }; SCALAR_LEN];
        for i in 0..SCALAR_LEN {
            circom_output_challenges[i] = GrumpkinScalarCircom {
                // element: GrumpkinScalar::from(result[2 + 4 * i].clone()),
                limbs: [
                    GrumpkinBase::from(result[2 + 3 * i].clone()),
                    GrumpkinBase::from(result[3 + 3 * i].clone()),
                    GrumpkinBase::from(result[4 + 3 * i].clone()),
                ],
            };
        }


        let rust_challenges: Vec<GrumpkinScalar> = t.challenge_vector(SCALAR_LEN);
        for i in 0..SCALAR_LEN{
            assert_eq!(convert_from_3_limbs(circom_output_challenges[i].limbs.to_vec()), rust_challenges[i]);
        }
        // add assert for limbs after correcting

        let final_state_from_rust = t.state.state[1];
        let final_state_from_circom = circom_result_transcript.state;
        assert_eq!(final_state_from_rust, final_state_from_circom);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }
}

#[test]
fn testing_challenge_scalar_powers() {
    let mut rng = ark_std::test_rng();
    let mut t = <PoseidonTranscript<GrumpkinScalar, GrumpkinBase> as Transcript>::new(b"label");

    let state = t.state.state[1];

    let testinput_transcript =  convert_grumpkin_transcript_to_circom(t.clone());

    let input_json = format!(
        r#"{{
        "transcript": {:?}
        }}"#,
        testinput_transcript
    );

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../scripts/compile_and_generate_witness_bn_scalar.sh";
    let circom_template = "ChallengeScalarPowers";
    let circom_file_path = "./grumpkin_transcript.circom";
    let circom_file = "grumpkin_transcript";
    let nPointstr = &SCALAR_LEN.to_string();

    let output = Command::new(script_path)
        .args(&[
            circom_file_path,
            circom_template,
            input_file_path,
            circom_file,
            nPointstr
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
            .take(2 + 4 * SCALAR_LEN)
            .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
            .collect();

        let result: Vec<BigUint> = result
            .into_iter()
            .map(|entry| {
                BigUint::parse_bytes(entry.as_bytes(), 10)
                    .expect("Failed to parse the string into a BigUint")
            })
            .collect();

        let circom_result_transcript = TestTranscript {
            state: GrumpkinBase::from(result[0].clone()),
            nrounds: GrumpkinBase::from(result[1].clone()),
        };

        let mut circom_output_challenges: [GrumpkinScalarCircom; SCALAR_LEN] = [GrumpkinScalarCircom {
            // element: GrumpkinScalar::ZERO,
            limbs: [GrumpkinBase::ZERO; 3],
        }; SCALAR_LEN];
        for i in 0..SCALAR_LEN {
            circom_output_challenges[i] = GrumpkinScalarCircom {
                // element: GrumpkinScalar::from(result[2 + 4 * i].clone()),
                limbs: [
                    GrumpkinBase::from(result[2 + 3 * i].clone()),
                    GrumpkinBase::from(result[3 + 3 * i].clone()),
                    GrumpkinBase::from(result[4 + 3 * i].clone()),
                ],
            };
        }


        let rust_challenges: Vec<GrumpkinScalar> = t.challenge_scalar_powers(SCALAR_LEN);

        for i in 0..SCALAR_LEN {
            assert_eq!(convert_from_3_limbs(circom_output_challenges[i].limbs.to_vec()), rust_challenges[i], "{i}");
        }
        // // add assert for limbs after correcting

        let final_state_from_rust = t.state.state[1];
        let final_state_from_circom = circom_result_transcript.state;
        assert_eq!(final_state_from_rust, final_state_from_circom);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }
}

#[test]
fn test_transcript_new(){
    let mut t = <PoseidonTranscript<GrumpkinScalar, GrumpkinBase> as Transcript>::new(b"Jolt transcript");
    let state = t.state.state[1];

    println!("state is {:?}", state);
    let nrounds = GrumpkinBase::from(t.n_rounds);
    println!("n rounds are {:?}", nrounds);

    let x = BigUint::from_str("604586419824232873836833680384618314");

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
    let circom_file_path = "./grumpkin_transcript.circom";
    let circom_file = "grumpkin_transcript";

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
            let final_state_from_circom = GrumpkinBase::from(result[0].clone());
            assert_eq!(final_state_from_rust, final_state_from_circom);

            println!("final_state_from_rust is {:?}", final_state_from_rust);
    
            let finaln_rounds_from_rust = t.n_rounds;
            let finaln_rounds_from_circom = GrumpkinBase::from(result[1].clone());
            assert_eq!(GrumpkinBase::from(finaln_rounds_from_rust), finaln_rounds_from_circom);
            println!(
                "finalnRounds_from_circom are {:?}",
                finaln_rounds_from_circom
            );
        } else {
            eprintln!("The JSON is not an array or 'witness' field is missing");
        }

}


#[test]
fn test_all(){
    testing_transcript_append_scalar();
    testing_transcript_append_scalars();
    testing_transcript_append_point();
    testing_transcript_append_points();
    testing_challenge_scalar();
    testing_challenge_vector();
    testing_challenge_scalar_powers();
}

