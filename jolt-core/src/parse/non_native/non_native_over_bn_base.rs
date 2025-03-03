#[cfg(test)]
mod test {
    use std::env;

    use ark_ff::UniformRand;
    use ark_std::rand::Rng;
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;
    use serde_json::json;

    use crate::{
        parse::{
            generate_circuit_and_witness, get_path, read_witness, spartan2::from_limbs, write_json,
            Parse,
        },
        spartan::spartan_memory_checking::R1CSConstructor,
    };
    type Fr = ark_bn254::Fr;
    type Fq = ark_bn254::Fq;
    const PACKAGE_NAME: &str = "non_native_over_bn_base";

    #[test]
    fn NonNativeAdd() {
        let mut rng = ChaCha8Rng::from_seed([2; 32]);

        let op1 = Fr::rand(&mut rng);
        let op2 = Fr::rand(&mut rng);
        let actual_result = op1 + op2;

        let input = json!(
        {
            "op1": op1.format_non_native(),
            "op2": op2.format_non_native(),
        }
        );
        let circom_template = "NonNativeAdd";
        verify(input, circom_template, actual_result);
    }

    #[test]
    fn NonNativeSub() {
        let mut rng = ChaCha8Rng::from_seed([2; 32]);

        let op1 = Fr::rand(&mut rng);
        let op2 = Fr::rand(&mut rng);
        let actual_result = op1 - op2;

        let input = json!(
        {
            "op1": op1.format_non_native(),
            "op2": op2.format_non_native(),
        }
        );
        let circom_template = "NonNativeSub";
        verify(input, circom_template, actual_result);
    }

    #[test]
    fn NonNativeMul() {
        let mut rng = ChaCha8Rng::from_seed([2; 32]);

        let op1 = Fr::rand(&mut rng);
        let op2 = Fr::rand(&mut rng);
        let actual_result = op1 * op2;

        let input = json!(
        {
            "op1": op1.format_non_native(),
            "op2": op2.format_non_native(),
        }
        );
        let circom_template = "NonNativeMul";
        verify(input, circom_template, actual_result);
    }

    #[test]
    fn NonNativeModulo() {
        let mut rng = ChaCha8Rng::from_seed([2; 32]);

        let mut a = vec![Fq::from(0u8); 3];
        for i in 0..3 {
            a[i] = Fq::from(rng.gen_range(0..(1u128 << 100)));
        }
        let actual_result = from_limbs(a.clone());

        let input = json!(
        {
            "op": {
                "limbs": [
                    a[0].to_string(), a[1].to_string(), a[2].to_string()
                ]
            }
        }

        );
        let circom_template = "NonNativeModulo";
        verify(input, circom_template, actual_result);
    }

    #[test]
    fn NonNativeEquality() {
        let mut rng = ChaCha8Rng::from_seed([2; 32]);

        let mut a = vec![Fq::from(0u8); 3];
        for i in 0..3 {
            a[i] = Fq::from(rng.gen_range(0..(1u128 << 100)));
        }
        let reduced_a: Fr = from_limbs(a.clone());

        let input = json!(
            {
                "op1": {
                    "limbs": [
                        a[0].to_string(), a[1].to_string(), a[2].to_string()
                    ]
                },
                "op2": reduced_a.format_non_native(),
            }
        );

        let circom_template = "NonNativeEquality";
        verify_equality(input, circom_template);
    }

    fn verify_equality(input: serde_json::Value, circom_template: &str) {
        let binding = env::current_dir().unwrap().join("src/parse/requirements");
        let output_dir = binding.to_str().unwrap();
        let package_path = get_path();
        write_json(&input, output_dir, PACKAGE_NAME);

        let file_name = format!("{}/{}.circom", "fields/non_native", PACKAGE_NAME);
        let file_path = package_path.join(file_name);

        let prime = "grumpkin";

        generate_circuit_and_witness(
            &file_path,
            &output_dir,
            circom_template,
            [].to_vec(),
            prime,
            None,
        );

        // // Read the witness.json file
        let witness_file_path = format!("{}/{}_witness.json", output_dir, PACKAGE_NAME);
        let z = read_witness::<Fq>(&witness_file_path.to_string());
        let constraint_path =
            format!("{}/{}_constraints.json", output_dir, PACKAGE_NAME).to_string();

        //To Check Az.Bz = C.z
        let _ = R1CSConstructor::<Fq>::construct(Some(&constraint_path), Some(&z), 0);
    }

    fn verify(
        input: serde_json::Value,
        circom_template: &str,
        actual_result: Fr,
    ) {
        let binding = env::current_dir().unwrap().join("src/parse/requirements");
        let output_dir = binding.to_str().unwrap();
        let package_path = get_path();
        write_json(&input, output_dir, PACKAGE_NAME);

        let file_name = format!("{}/{}.circom", "fields/non_native", PACKAGE_NAME);
        let file_path = package_path.join(file_name);

        let prime = "grumpkin";

        generate_circuit_and_witness(
            &file_path,
            &output_dir,
            circom_template,
            [].to_vec(),
            prime,
            None,
        );

        // // Read the witness.json file
        let witness_file_path = format!("{}/{}_witness.json", output_dir, PACKAGE_NAME);
        let z = read_witness::<Fq>(&witness_file_path.to_string());
        let constraint_path =
            format!("{}/{}_constraints.json", output_dir, PACKAGE_NAME).to_string();
        let expected_result: Fr = from_limbs(vec![z[1], z[2], z[3]]);
        assert_eq!(expected_result, actual_result, "assertion failed");
        //To Check Az.Bz = C.z
        let _ = R1CSConstructor::<Fq>::construct(Some(&constraint_path), Some(&z), 0);
    }
}
