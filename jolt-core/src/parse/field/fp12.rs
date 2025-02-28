#[cfg(test)]
mod tests {
    use std::env;
    use ark_ff::{CyclotomicMultSubgroup, Field, UniformRand};
    use rand_chacha::ChaCha8Rng;
    use ark_bn254::{Fq, Fq12, Fq2, Fq6};
    use rand_core::SeedableRng;
    use serde_json::json;

    use crate::{parse::{generate_circuit_and_witness, get_path, read_witness, write_json, Parse}, spartan::spartan_memory_checking::R1CSConstructor};
    impl Parse for Fq12 {
        fn format(&self) -> serde_json::Value {
            json!({
                "x": self.c0.format(),
                "y": self.c1.format(),
            })
        }
    }
    #[test]
    fn Fp12Add() {
        let mut rng = ChaCha8Rng::from_seed([2; 32]);

        let op1 = Fq12::rand(&mut rng);
        let op2 = Fq12::rand(&mut rng);
        let actual_result = op1 + op2;

        let input = json!(
        {
            "op1": op1.format(),
            "op2": op2.format(),
        }
        );
        let package_name = "fp12";
        let circom_template = "Fp12add";
        verify(input, package_name, circom_template, actual_result);
    }

    #[test]
    fn Fp12mul() {
        let mut rng = ChaCha8Rng::from_seed([2; 32]);

        let op1 = Fq12::rand(&mut rng);
        let op2 = Fq12::rand(&mut rng);
        let actual_result = op1 * op2;

        let input = json!(
        {
            "op1": op1.format(),
            "op2": op2.format(),
        }
        );
        let package_name = "fp12";
        let circom_template = "Fp12mul";
        verify(input, package_name, circom_template, actual_result);
    }

    #[test]
    fn Fp12inv() {
        let mut rng = ChaCha8Rng::from_seed([2; 32]);

        let op1 = Fq12::rand(&mut rng);
        let actual_result = op1.inverse().unwrap();

        let input = json!(
        {
            "op1": op1.format()
        }
        );
        let package_name = "fp12";
        let circom_template = "Fp12inv";
        verify(input, package_name, circom_template, actual_result);
    }

    #[test]
    fn Fp12mulbyfp() {
        let mut rng = ChaCha8Rng::from_seed([2; 32]);

        let op1 = Fq::rand(&mut rng);
        let op2 = Fq12::rand(&mut rng);

        let mut actual_result = op2;
        actual_result.mul_by_fp(&op1);

        let input = json!(
        {
            "op1": op1.to_string(),
            "op2": op2.format(),
        }
        );
        let package_name = "fp12";
        let circom_template = "Fp12mulbyfp";


        verify(input, package_name, circom_template, actual_result);
    }
    
    // failing
    #[test]
    fn Fp12square() {
        let mut rng = ChaCha8Rng::from_seed([2; 32]);
        let op1 = Fq12::rand(&mut rng);

        let input = json!(
        {
            "op1": op1.format()
        }
        );
        let actual_result = op1.cyclotomic_square();
        let package_name = "fp12";
        let circom_template = "Fp12square";
        verify(input, package_name, circom_template, actual_result);
    }


    fn verify(
        input: serde_json::Value,
        package_name: &str,
        circom_template: &str,
        actual_result: Fq12,
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
        let expected_result = Fq12::new(Fq6::new(Fq2::new(z[1], z[2]), Fq2::new(z[3], z[4]), Fq2::new(z[5], z[6])), Fq6::new(Fq2::new(z[7], z[8]), Fq2::new(z[9], z[10]), Fq2::new(z[11], z[12])));

        assert_eq!(expected_result, actual_result, "assertion failed");
        // To Check Az.Bz = C.z
        let _ = R1CSConstructor::<ark_bn254::Fq>::construct(Some(&constraint_path), Some(&z), 0);
    }
    
}