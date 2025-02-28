#![allow(unused_imports)]
use ark_bn254::Fq;
use ark_bn254::Fr;
use ark_ff::AdditiveGroup;
use ark_ff::{BigInteger, Field, MontFp, PrimeField, ToConstraintField, UniformRand};
use ark_std::rand::Rng;
use matrix::check_inner_product;
use num_bigint::{BigUint, ParseBigIntError};
use parsing::non_native::convert_from_3_limbs;
use parsing::non_native::convert_to_3_limbs;
use parsing::non_native::Fqq;
use rand::thread_rng;
use serde_json::Value;
use std::{
    fmt::format,
    fs::File,
    io::{Read, Write},
    process::Command,
    str::FromStr,
};

use crate::helper::return_multiple_of_q;

#[test]
fn test_non_native_addition() {
    let mut rng = thread_rng();

    let mut a_bar = [Fq::from(0u8); 3];

    let mut b_bar = [Fq::from(0u8); 3];
    for i in 0..2 {
        a_bar[i] = Fq::from(rng.gen_range(0..(1u128 << 125)));

        b_bar[i] = Fq::from(rng.gen_range(0..(1u128 << 125)));
    }
    a_bar[2] = Fq::from(rng.gen_range(0..(1u128 << 120)));

    b_bar[2] = Fq::from(rng.gen_range(0..(1u128 << 120)));

    let a = convert_from_3_limbs(a_bar.to_vec());
    let b = convert_from_3_limbs(b_bar.to_vec());

    let op1 = Fqq {
        // element: a,
        limbs: a_bar,
    };
    let op2 = Fqq {
        // element: b,
        limbs: b_bar,
    };

    let input_json = format!(
        r#"{{
        "op1": {:?},
        "op2": {:?}
    }}"#,
        op1, op2
    );

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "NonNativeAdd";
    let circom_file_path = "./non_native_over_bn_base.circom";
    let circom_file = "non_native_over_bn_base";

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

        let circom_result = Fqq {
            // element: Fr::from(result[0].clone()),
            limbs: [
                Fq::from(result[0].clone()),
                Fq::from(result[1].clone()),
                Fq::from(result[2].clone()),
            ],
        };

        let add_res_direct_from_rust = a + b;

        let circom_res_limbs_combined = convert_from_3_limbs(circom_result.limbs.to_vec());
        assert_eq!(circom_res_limbs_combined, add_res_direct_from_rust);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_non_native_modulo() {
    let mut rng = thread_rng();

    let mut a_bar = [Fq::from(0u8); 3];

    for i in 0..2 {
        a_bar[i] = Fq::from(rng.gen_range(0..(1u128 << 125)));
    }
    a_bar[2] = Fq::from(rng.gen_range(0..(1u128 << 120)));

    let a = convert_from_3_limbs(a_bar.to_vec());

    let op1 = Fqq {
        // element: a,
        limbs: a_bar,
    };

    println!("a = {:?}", a);
    println!("a_bar = {:?}", a_bar);

    let input_json = format!(
        r#"{{
        "op": {:?}
    }}"#,
        op1
    );

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "NonNativeModulo";
    let circom_file_path = "./non_native_over_bn_base.circom";
    let circom_file = "non_native_over_bn_base";

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

        let circom_result = Fqq {
            // element: Fr::from(result[0].clone()),
            limbs: [
                Fq::from(result[0].clone()),
                Fq::from(result[1].clone()),
                Fq::from(result[2].clone()),
            ],
        };

        let add_res_direct_from_rust = a;

        let circom_res_limbs_combined = convert_from_3_limbs(circom_result.limbs.to_vec());
        assert_eq!(circom_res_limbs_combined, add_res_direct_from_rust);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn testing() {
    // let x = BigUint::from_str("31190347037942102772907416716342395593441382154865078771838148010163239324186");
    // println!("x is {:?}", x.expect("he"));
    let y = Fr::from(
        BigUint::from_str(
            "31190347037942102772907416716342395593441382154865078771838148010163239324186",
        )
        .clone()
        .unwrap(),
    );
}

#[test]
fn test_non_native_equality_default() {
    let mut rng = thread_rng();

    let mut a_bar = [Fq::from(0u8); 3];
    let mut q_bar = [
        Fq::from(10903342367192220456583066779700428801 as u128),
        Fq::from(4166566524057721139834548734155997929 as u128),
        Fq::from(12),
    ];

    for i in 0..2 {
        a_bar[i] = Fq::from(rng.gen_range(0..(1u128 << 125)));
    }
    a_bar[2] = Fq::from(rng.gen_range(0..(1u128 << 120)));

    let mut a = convert_from_3_limbs(a_bar.to_vec());
    let q = convert_from_3_limbs(q_bar.to_vec());

    let b = a + q;
    let b_bar = convert_to_3_limbs(b);

    println!("a_bar = {:?}", a_bar);
    println!("b_bar = {:?}", b_bar);

    let op1 = Fqq {
        // element: a,
        limbs: a_bar,
    };
    let op2 = Fqq {
        // element: b,
        limbs: b_bar,
    };

    let input_json = format!(
        r#"{{
        "op1": {:?},
        "op2": {:?}
    }}"#,
        op1, op2
    );

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "NonNativeEquality";
    let circom_file_path = "./non_native_over_bn_base.circom";
    let circom_file = "non_native_over_bn_base";

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

        let circom_result = Fqq {
            // element: Fr::from(result[0].clone()),
            limbs: [
                Fq::from(result[0].clone()),
                Fq::from(result[1].clone()),
                Fq::from(result[2].clone()),
            ],
        };
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_non_native_equality_reduced_rhs() {
    let mut rng = thread_rng();

    let mut a_bar = [Fq::from(0u8); 3];
    let mut q_bar = [
        Fq::from(10903342367192220456583066779700428801 as u128),
        Fq::from(4166566524057721139834548734155997929 as u128),
        Fq::from(12),
    ];

    for i in 0..2 {
        a_bar[i] = Fq::from(rng.gen_range(0..(1u128 << 125)));
    }
    a_bar[2] = Fq::from(rng.gen_range(0..(1u128 << 120)));

    let mut a = convert_from_3_limbs(a_bar.to_vec());
    let q = convert_from_3_limbs(q_bar.to_vec());

    let b = a + q;
    let b_bar = convert_to_3_limbs(b);

    println!("a_bar = {:?}", a_bar);
    println!("b_bar = {:?}", b_bar);

    let op1 = Fqq {
        // element: a,
        limbs: a_bar,
    };
    let op2 = Fqq {
        // element: b,
        limbs: b_bar,
    };

    let input_json = format!(
        r#"{{
        "op1": {:?},
        "op2": {:?}
    }}"#,
        op1, op2
    );

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "NonNativeEqualityReducedRHS";
    let circom_file_path = "./non_native_over_bn_base.circom";
    let circom_file = "non_native_over_bn_base";

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

        let circom_result = Fqq {
            // element: Fr::from(result[0].clone()),
            limbs: [
                Fq::from(result[0].clone()),
                Fq::from(result[1].clone()),
                Fq::from(result[2].clone()),
            ],
        };
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_non_native_equality_reduced_lhs() {
    let mut rng = thread_rng();

    let mut a_bar = [Fq::from(0u8); 3];
    let mut q_bar = [
        Fq::from(10903342367192220456583066779700428801 as u128),
        Fq::from(4166566524057721139834548734155997929 as u128),
        Fq::from(12),
    ];

    for i in 0..2 {
        a_bar[i] = Fq::from(rng.gen_range(0..(1u128 << 125)));
    }
    a_bar[2] = Fq::from(rng.gen_range(0..(1u128 << 120)));

    let mut a = convert_from_3_limbs(a_bar.to_vec());
    let q = convert_from_3_limbs(q_bar.to_vec());

    let b = a + q;
    let b_bar = convert_to_3_limbs(b);

    println!("a_bar = {:?}", a_bar);
    println!("b_bar = {:?}", b_bar);

    let op2 = Fqq {
        // element: a,
        limbs: a_bar,
    };
    let op1 = Fqq {
        // element: b,
        limbs: b_bar,
    };

    let input_json = format!(
        r#"{{
        "op1": {:?},
        "op2": {:?}
    }}"#,
        op1, op2
    );

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "NonNativeEqualityReducedLHS";
    let circom_file_path = "./non_native_over_bn_base.circom";
    let circom_file = "non_native_over_bn_base";

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

        let circom_result = Fqq {
            // element: Fr::from(result[0].clone()),
            limbs: [
                Fq::from(result[0].clone()),
                Fq::from(result[1].clone()),
                Fq::from(result[2].clone()),
            ],
        };
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn chekcing_limbs_interconv() {
    let mut rng = ark_std::test_rng();

    let x = Fr::rand(&mut rng);
    let limbs = convert_to_3_limbs(x);
    let y = convert_from_3_limbs(limbs.to_vec());
    assert_eq!(x, y)
}
#[test]
fn test_non_native_mul() {
    let mut rng = ark_std::test_rng();

    let mut a_bar = [Fq::from(0u8); 3];

    let mut b_bar = [Fq::from(0u8); 3];
    for i in 0..2 {
        a_bar[i] = Fq::from(rng.gen_range(0..(1u128 << 125)));

        b_bar[i] = Fq::from(rng.gen_range(0..(1u128 << 125)));
    }
    a_bar[2] = Fq::from(rng.gen_range(0..(1u128 << 120)));

    b_bar[2] = Fq::from(rng.gen_range(0..(1u128 << 120)));

    let a = convert_from_3_limbs(a_bar.to_vec());
    let b = convert_from_3_limbs(b_bar.to_vec());

    let op1 = Fqq {
        // element: a,
        limbs: a_bar,
    };
    let op2 = Fqq {
        // element: b,
        limbs: b_bar,
    };

    let input_json = format!(
        r#"{{
        "op1": {:?},
        "op2": {:?}
    }}"#,
        op1, op2
    );

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "NonNativeMul";
    let circom_file_path = "./non_native_over_bn_base.circom";
    let circom_file = "non_native_over_bn_base";

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

        let circom_result = Fqq {
            // element: Fr::from(result[0].clone()),
            limbs: [
                Fq::from(result[0].clone()),
                Fq::from(result[1].clone()),
                Fq::from(result[2].clone()),
            ],
        };

        let mul_res_from_rust = a * b;

        let circom_res_limbs_combined = convert_from_3_limbs(circom_result.limbs.to_vec());
        assert_eq!(circom_res_limbs_combined, mul_res_from_rust);
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_non_native_additive_inv() {
    let mut rng = ark_std::test_rng();

    let a = Fr::rand(&mut rng);
    let a_bar = convert_to_3_limbs(a);

    let b = -a;
    let b_bar = convert_to_3_limbs(b);

    let constant_mul_scalar_modulus = Fqq {
        // element: Fr::ZERO,
        limbs: return_multiple_of_q(1),
    };

    let op1 = Fqq {
        // element: a,
        limbs: a_bar,
    };

    let input_json = format!(
        r#"{{
    "op": {:?}
    }}"#,
        op1
    );

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "NonNativeAdditiveInverse";
    let circom_file_path = "./non_native_over_bn_base.circom";
    let circom_file = "non_native_over_bn_base";

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
            .take(1)
            .map(|entry| entry.as_str().map(|s| s.to_string()).unwrap())
            .collect();

        let result: Vec<BigUint> = result
            .into_iter()
            .map(|entry| {
                BigUint::parse_bytes(entry.as_bytes(), 10)
                    .expect("Failed to parse the string into a BigUint")
            })
            .collect();
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }

    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_non_native_sub() {
    let mut rng = ark_std::test_rng();

    let mut op1_bar = [Fq::from(0u8); 3];

    for i in 0..2 {
        op1_bar[i] = Fq::from(rng.gen_range(0..(1u128 << 125)));
    }
    op1_bar[2] = Fq::from(rng.gen_range(0..(1u128 << 120)));
    let op1 = convert_from_3_limbs(op1_bar.to_vec());

    let op2 = Fr::rand(&mut rng);
    let op2_bar = convert_to_3_limbs(op2);

    let constant_mul_scalar_modulus = Fqq {
        // element: Fr::ZERO,
        limbs: return_multiple_of_q(1),
    };

    let op1_bar = Fqq {
        // element: op1,
        limbs: op1_bar,
    };
    let op2_bar = Fqq {
        // element: op2,
        limbs: op2_bar,
    };

    let input_json = format!(
        r#"{{
    "op1": {:?},
    "op2": {:?}
    }}"#,
        op1_bar, op2_bar
    );

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "NonNativeSub";
    let circom_file_path = "./non_native_over_bn_base.circom";
    let circom_file = "non_native_over_bn_base";

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

        let circom_result = Fqq {
            limbs: [
                Fq::from(result[0].clone()),
                Fq::from(result[1].clone()),
                Fq::from(result[2].clone()),
            ],
        };
        //TODO:- Add assertion
        assert_eq!(
            convert_from_3_limbs(circom_result.limbs.to_vec()), op1 - op2
        );
    } else {
        eprintln!("The JSON is not an array or 'witness' field is missing");
    }
    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_Num2Limbs() {
    let mut rng = ark_std::test_rng();

    let input = Fq::from(7);
    let input_json = format!(
        r#"{{
        "in": {:?}
        }}"#,
        input,
    );
    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // Step 3: Call shell script to compile and generate witness
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_base.sh";
    let circom_template = "Num2Limbs";
    let circom_file_path = "./non_native_over_bn_base.circom";
    let circom_file = "non_native_over_bn_base";
    let l1 = &1.to_string();
    let l2 = &1.to_string();
    let l3 = &2.to_string();
    // let l3 = &4.to_string();
    let output = Command::new(script_path)
        .args(&[
            circom_file_path,
            circom_template,
            input_file_path,
            circom_file,
            l1,
            l2,
            l3,
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
    check_inner_product(vec![circom_file.to_string()]);
}

#[test]
fn test_all() {
    test_non_native_addition();
    test_non_native_modulo();
    test_non_native_equality_default();
    test_non_native_equality_reduced_rhs();
    test_non_native_equality_reduced_lhs();
    test_non_native_mul();
    test_non_native_additive_inv();
    test_non_native_sub();
    test_Num2Limbs();
}
