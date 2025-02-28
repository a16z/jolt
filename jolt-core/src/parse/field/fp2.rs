#[cfg(test)]
mod test {
    use crate::parse::{generate_circuit_and_witness, get_path, read_witness, write_json, Parse};
    use crate::spartan::spartan_memory_checking::R1CSConstructor;
    use ark_bn254::{Fq, Fq2, Fr};
    use ark_ff::{Field, PrimeField};
    use ark_std::UniformRand;
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;
    use serde_json::json;
    use std::env;
    impl Parse for Fq2 {
        fn format(&self) -> serde_json::Value {
            json!({
                "x": self.c0.to_string(),
                "y": self.c1.to_string(),
            })
        }
    }
    #[test]
    fn Fp2Add() {
        let mut rng = ChaCha8Rng::from_seed([2; 32]);

        let op1 = Fq2::rand(&mut rng);
        let op2 = Fq2::rand(&mut rng);
        let actual_result = op1 + op2;

        let input = json!(
        {
            "op1": op1.format(),
            "op2": op2.format(),
        }
        );
        let package_name = "fp2";
        let circom_template = "Fp2add";
        verify(input, package_name, circom_template, actual_result);
    }
    #[test]
    fn Fp2Sub() {
        let mut rng = ChaCha8Rng::from_seed([2; 32]);

        let op1 = Fq2::rand(&mut rng);
        let op2 = Fq2::rand(&mut rng);
        let actual_result = op1 - op2;

        let input = json!(
        {
            "op1": op1.format(),
            "op2": op2.format(),
        }
        );
        let package_name = "fp2";
        let circom_template = "Fp2sub";
        verify(input, package_name, circom_template, actual_result);
    }

    #[test]
    fn Fp2mul() {
        let mut rng = ChaCha8Rng::from_seed([2; 32]);

        let op1 = Fq2::rand(&mut rng);
        let op2 = Fq2::rand(&mut rng);
        let actual_result = op1 * op2;

        let input = json!(
        {
            "op1": op1.format(),
            "op2": op2.format(),
        }
        );
        let package_name = "fp2";
        let circom_template = "Fp2mul";
        verify(input, package_name, circom_template, actual_result);
    }

    #[test]
    fn Fp2muladd() {
        let mut rng = ChaCha8Rng::from_seed([2; 32]);

        let op1 = Fq2::rand(&mut rng);
        let op2 = Fq2::rand(&mut rng);
        let op3 = Fq2::rand(&mut rng);
        let actual_result = (op1 * op2) + op3;

        let input = json!(
        {
            "op1": op1.format(),
            "op2": op2.format(),
            "op3": op3.format(),
        }
        );
        let package_name = "fp2";
        let circom_template = "Fp2muladd";
        verify(input, package_name, circom_template, actual_result);
    }

    #[test]
    fn Fp2mulbyfp() {
        let mut rng = ChaCha8Rng::from_seed([2; 32]);

        let op1 = Fq::rand(&mut rng);
        let op2 = Fq2::rand(&mut rng);
        let actual_result = op2.mul_by_base_prime_field(&op1);

        let input = json!(
        {
            "op1": op1.to_string(),
            "op2": op2.format(),
        }
        );
        let package_name = "fp2";
        let circom_template = "Fp2mulbyfp";
        verify(input, package_name, circom_template, actual_result);
    }

    #[test]
    fn Fp2inv() {
        let mut rng = ChaCha8Rng::from_seed([2; 32]);

        let op1 = Fq2::rand(&mut rng);
        let actual_result = op1.inverse().unwrap();

        let input = json!(
        {
            "op1": op1.format(),
        }
        );
        let package_name = "fp2";
        let circom_template = "Fp2inv";
        verify(input, package_name, circom_template, actual_result);
    }

    #[test]
    fn Fp2exp() {
        let mut rng = ChaCha8Rng::from_seed([2; 32]);

        let op1 = Fr::rand(&mut rng);
        let op2 = Fq2::rand(&mut rng);
        let actual_result = op2.pow(op1.into_bigint());

        let input = json!(
        {
            "op1": op1.to_string(),
            "op2": op2.format(),
        }
        );
        let package_name = "fp2";
        let circom_template = "Fp2exp";
        verify(input, package_name, circom_template, actual_result);
    }
    fn verify(
        input: serde_json::Value,
        package_name: &str,
        circom_template: &str,
        actual_result: Fq2,
    ) {
        let binding = env::current_dir().unwrap().join("src/parse/requirements");
        let output_dir = binding.to_str().unwrap();
        let package_path = get_path();
        write_json(&input, output_dir, package_name);

        let file_name = format!("{}/{}.circom", "fields/field", package_name);
        let file_path = package_path.join(file_name);

        let prime = "grumpkin";

        generate_circuit_and_witness(&file_path, &output_dir, circom_template, [].to_vec(), prime);

        // // Read the witness.json file
        let witness_file_path = format!("{}/{}_witness.json", output_dir, package_name);
        let z = read_witness::<ark_bn254::Fq>(&witness_file_path.to_string());

        let constraint_path =
            format!("{}/{}_constraints.json", output_dir, package_name).to_string();
        let expected_result = Fq2::new(z[1], z[2]);

        assert_eq!(expected_result, actual_result, "assertion failed");
        //To Check Az.Bz = C.z
        let _ = R1CSConstructor::<ark_bn254::Fq>::construct(Some(&constraint_path), Some(&z), 0);
    }
}
