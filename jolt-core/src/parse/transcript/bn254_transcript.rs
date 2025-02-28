use core::fmt;
use std::{fs::File, io::{Read, Write}, process::Command, str::FromStr};

use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
use ark_ff::{AdditiveGroup, Field, PrimeField, UniformRand};
use jolt_core::utils::{poseidon_transcript::PoseidonTranscript, transcript::Transcript};
use num_bigint::BigUint;
use parsing::{bn_group::G1AffineFormTest, non_native::{convert_from_3_limbs, convert_to_3_limbs, Fqq}, transcript_bn_fq::TestTranscript, SCALAR_LEN};
use serde_json::Value;

#[test]
fn testing_transcript_append_scalar() {
    let x = Fr::from(8u64);
    let mut rng = ark_std::test_rng();
    let mut t = <PoseidonTranscript<Fr, Fq> as Transcript>::new(b"label");
    let state = t.state.state[1];
    let nrounds = Fq::from(t.n_rounds);
    println!("n rounds are {:?}", nrounds);

    let testinput_transcript = TestTranscript {
        state: state,
        nrounds,
    };

    let scalar_to_append = Fr::rand(&mut rng);
    let scalar = Fqq {
        // element: scalar_to_append,
        limbs: convert_to_3_limbs(scalar_to_append),
    };

    // println!("scalar_to_append is {:?}", scalar_to_append.to_string());
    let input_json = format!(
        r#"{{
        "scalar": {:?},
        "transcript": {:?}
        }}"#,
        scalar, testinput_transcript
    );

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "AppendScalar";
    let circom_file_path = "./bn254_transcript.circom";
    let circom_file = "bn254_transcript";

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
            state: Fq::from(result[0].clone()),
            nrounds: Fq::from(result[1].clone()),
        };

        t.append_scalar(&scalar_to_append);

        let final_state_from_rust = t.state.state[1];
        let final_state_from_circom = circom_result.state;
        assert_eq!(final_state_from_rust, final_state_from_circom);

        let finaln_rounds_from_rust = t.n_rounds;
        let finaln_rounds_from_circom = circom_result.nrounds;
        assert_eq!(Fq::from(finaln_rounds_from_rust), finaln_rounds_from_circom);
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
fn testing_transcript_appendscalars() {
    let mut rng = ark_std::test_rng();
    let mut t = <PoseidonTranscript<Fr, Fq> as Transcript>::new(b"label");
    let state = t.state.state[1];
    let nrounds = Fq::from(t.n_rounds);
    // println!("n rounds are {:?}", nrounds);

    let testinput_transcript = TestTranscript { state, nrounds };

    let mut scalars_to_append: [Fr; SCALAR_LEN] = [Fr::ZERO; SCALAR_LEN];
    for i in 0..SCALAR_LEN {
        scalars_to_append[i] = Fr::rand(&mut rng);
    }

    // let scalar_to_append = Fr::rand(&mut rng);
    let mut scalars = Vec::new();
    for i in 0..SCALAR_LEN{
        scalars.push(Fqq {
            // element: scalars_to_append[i],
            limbs: convert_to_3_limbs(scalars_to_append[i]),
        })
    } 

    let input_json = format!(
        r#"{{
        "scalars": {:?},
        "transcript": {:?}
        }}"#,
        scalars, testinput_transcript
    );

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "AppendScalars";
    let circom_file_path = "./bn254_transcript.circom";
    let circom_file = "bn254_transcript";
    let nScalars = SCALAR_LEN.to_string();

    let output = Command::new(script_path)
        .args(&[
            circom_file_path,
            circom_template,
            input_file_path,
            circom_file,
            &nScalars,
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
            state: Fq::from(result[0].clone()),
            nrounds: Fq::from(result[1].clone()),
        };

        t.append_scalars(&scalars_to_append);

        let final_state_from_rust = t.state.state[1];
        let final_state_from_circom = circom_result.state;
        assert_eq!(final_state_from_rust, final_state_from_circom);

        let finaln_rounds_from_rust = t.n_rounds;
        let finaln_rounds_from_circom = circom_result.nrounds;
        assert_eq!(Fq::from(finaln_rounds_from_rust), finaln_rounds_from_circom);
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
fn testing_transcript_append_point() {
    let mut rng = ark_std::test_rng();
    let mut t = <PoseidonTranscript<Fr, Fq> as Transcript>::new(b"label");
    let state = t.state.state[1];
    let nrounds = Fq::from(t.n_rounds);

    let testinput_transcript = TestTranscript {
        state: state,
        nrounds,
    };

    let point_to_append = G1Affine::rand(&mut rng);
    let formatted_point_to_append = G1AffineFormTest {
        x: point_to_append.x,
        y: point_to_append.y,
    };
    let proj_form = G1Projective {
        x: point_to_append.x,
        y: point_to_append.y,
        z: Fq::ONE,
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
    let script_path = "./../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "AppendPoint";
    let circom_file_path = "./bn254_transcript.circom";
    let circom_file = "bn254_transcript";

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
            state: Fq::from(result[0].clone()),
            nrounds: Fq::from(result[1].clone()),
        };

        t.append_point(&proj_form);

        let final_state_from_rust = t.state.state[1];
        let final_state_from_circom = circom_result.state;
        assert_eq!(final_state_from_rust, final_state_from_circom);

        let finalnRounds_from_rust = t.n_rounds;
        let finalnRounds_from_circom = circom_result.nrounds;
        assert_eq!(Fq::from(finalnRounds_from_rust), finalnRounds_from_circom);
        // println!("finalnRounds_from_circom are {:?}", finalnRounds_from_circom);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    // check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn testing_transcript_appendpoints() {
    let mut rng = ark_std::test_rng();
    let mut t = <PoseidonTranscript<Fr, Fq> as Transcript>::new(b"label");

    let state = t.state.state[1];
    let nrounds = Fq::from(t.n_rounds);

    let testinput_transcript = TestTranscript {
        state: state,
        nrounds,
    };
    let mut points_to_append: [G1Affine; SCALAR_LEN] = [G1Affine::identity(); SCALAR_LEN];
    for i in 0..SCALAR_LEN {
        points_to_append[i] = G1Affine::rand(&mut rng);
    }
    let mut formatted_points_to_append = Vec::new();
    for i in 0..SCALAR_LEN {
        formatted_points_to_append.push(
            G1AffineFormTest {
                x: points_to_append[i].x,
                y: points_to_append[i].y,
            }
        )
    }

    let mut proj_forms = Vec::new();
    for i in 0..SCALAR_LEN{
        proj_forms.push(G1Projective {
            x: points_to_append[i].x,
            y: points_to_append[i].y,
            z: Fq::ONE,
        })
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
    let script_path = "./../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "AppendPoints";
    let circom_file_path = "./bn254_transcript.circom";
    let circom_file = "bn254_transcript";
    let nPointstr = &SCALAR_LEN.to_string();

    let output = Command::new(script_path)
        .args(&[
            circom_file_path,
            circom_template,
            input_file_path,
            circom_file,
            nPointstr,
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
            state: Fq::from(result[0].clone()),
            nrounds: Fq::from(result[1].clone()),
        };

        t.append_points(&proj_forms);

        let final_state_from_rust = t.state.state[1];
        let final_state_from_circom = circom_result.state;
        assert_eq!(final_state_from_rust, final_state_from_circom);

        let finalnRounds_from_rust = t.n_rounds;
        let finalnRounds_from_circom = circom_result.nrounds;
        assert_eq!(Fq::from(finalnRounds_from_rust), finalnRounds_from_circom);
        // println!("finalnRounds_from_circom are {:?}", finalnRounds_from_circom);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    // check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn testing_challenge_scalar() {
    let mut rng = ark_std::test_rng();
    let mut t = <PoseidonTranscript<Fr, Fq> as Transcript>::new(b"label");
    let state = t.state.state[1];
    let nrounds = Fq::from(t.n_rounds);

    let testinput_transcript = TestTranscript {
        state: state,
        nrounds,
    };

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
    let script_path = "./../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "ChallengeScalar";
    let circom_file_path = "./bn254_transcript.circom";
    let circom_file = "bn254_transcript";
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
            state: Fq::from(result[0].clone()),
            nrounds: Fq::from(result[1].clone()),
        };

        let circom_output_challenge = Fqq {
            // element: Fr::from(result[2].clone()),
            limbs: [
                Fq::from(result[2].clone()),
                Fq::from(result[3].clone()),
                Fq::from(result[4].clone()),
            ],
        };

        let rust_challenge = t.challenge_scalar::<Fr>();

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
    let mut t = <PoseidonTranscript<Fr, Fq> as Transcript>::new(b"label");
    let state = t.state.state[1];
    let nrounds = Fq::from(t.n_rounds);

    let testinput_transcript = TestTranscript {
        state: state,
        nrounds,
    };

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
    let script_path = "./../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "ChallengeVector";
    let circom_file_path = "./bn254_transcript.circom";
    let circom_file = "bn254_transcript";
    let len = SCALAR_LEN.to_string();

    let output = Command::new(script_path)
        .args(&[
            circom_file_path,
            circom_template,
            input_file_path,
            circom_file,
            &len,
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
            state: Fq::from(result[0].clone()),
            nrounds: Fq::from(result[1].clone()),
        };

        let mut circom_output_challenges = Vec::new();

        for i in 0..SCALAR_LEN {
            circom_output_challenges.push(Fqq {
                // element: Fr::from(result[2 + 4 * i].clone()),
                limbs: [
                    Fq::from(result[2 + 3 * i].clone()),
                    Fq::from(result[3 + 3 * i].clone()),
                    Fq::from(result[4 + 3 * i].clone()),
                ]
            }
            )
        };

        let rust_challenge: Vec<Fr> = t.challenge_vector(SCALAR_LEN);

        for i in 0..SCALAR_LEN {
            assert_eq!(convert_from_3_limbs((circom_output_challenges[i].limbs.to_vec())), rust_challenge[i], "at i {}", i);
        }

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
    let mut t = <PoseidonTranscript<Fr, Fq> as Transcript>::new(b"label");
    let state = t.state.state[1];
    let nrounds = Fq::from(t.n_rounds);

    let testinput_transcript = TestTranscript {
        state: state,
        nrounds,
    };

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
    let script_path = "./../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "ChallengeScalarPowers";
    let circom_file_path = "./bn254_transcript.circom";
    let circom_file = "bn254_transcript";
    let len = SCALAR_LEN.to_string();

    let output = Command::new(script_path)
        .args(&[
            circom_file_path,
            circom_template,
            input_file_path,
            circom_file,
            &len,
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
            state: Fq::from(result[0].clone()),
            nrounds: Fq::from(result[1].clone()),
        };

        let mut circom_output_challenges = Vec::new();
        for i in 0..SCALAR_LEN {
            let limbs = [
                Fq::from(result[3 + 4 * i].clone()),
                Fq::from(result[4 + 4 * i].clone()),
                Fq::from(result[5 + 4 * i].clone()),
            ];
            circom_output_challenges.push(Fqq {
                limbs: limbs,
                // element: convert_from_3_limbs(limbs.to_vec()),
            }
            )
        }

        let rust_challenge: Vec<Fr> = t.challenge_scalar_powers(SCALAR_LEN);

        // for i in 0..SCALAR_LEN {
        //     assert_eq!(
        //         circom_output_challenges[i].element, rust_challenge[i],
        //         "failing for {i}"
        //     );
        // }

        let final_state_from_rust = t.state.state[1];
        let final_state_from_circom = circom_result_transcript.state;
        assert_eq!(final_state_from_rust, final_state_from_circom);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }
}

#[test]
fn test_transcript_new(){
    let mut t = <PoseidonTranscript<Fr, Fq> as Transcript>::new(b"Jolt transcript");
    let state = t.state.state[1];

    println!("state is {:?}", state);
    let nrounds = Fq::from(t.n_rounds);
    println!("n rounds are {:?}", nrounds);

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
    let script_path = "./../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "TranscriptNew";
    let circom_file_path = "./bn254_transcript.circom";
    let circom_file = "bn254_transcript";

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
            let final_state_from_circom = Fq::from(result[0].clone());
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



#[test]
fn test_all(){
    testing_transcript_append_scalar();
    testing_transcript_appendscalars();
    testing_challenge_scalar();
    testing_challenge_vector();
    testing_transcript_append_point();
    testing_transcript_appendpoints();
    testing_challenge_scalar_powers();

}