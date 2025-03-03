#[cfg(test)]
mod tests {
    use crate::{
        parse::{generate_circuit_and_witness, get_path, read_witness, write_json, Parse},
        spartan::spartan_memory_checking::R1CSConstructor,
    };
    use ark_bn254::{Fq2, Fq6};
    use ark_ff::{Field, UniformRand};
    use serde_json::json;
    use std::env;

    impl Parse for Fq6 {
        fn format(&self) -> serde_json::Value {
            json!({
                "x": self.c0.format(),
                "y": self.c1.format(),
                "z": self.c2.format(),
            })
        }
    }

    #[test]
    fn Fp6Add() {
        let mut rng = ark_std::test_rng();
        let op1 = Fq6::rand(&mut rng);
        let op2 = Fq6::rand(&mut rng);
        let actual_result = op1 + op2;

        let input = json!(
        {
            "op1": op1.format(),
            "op2": op2.format(),
        }
        );
        let package_name = "fp6";
        let circom_template = "Fp6add";
        verify(input, package_name, circom_template, actual_result);
    }

    #[test]
    fn Fp6Sub() {
        let mut rng = ark_std::test_rng();
        let op1 = Fq6::rand(&mut rng);
        let op2 = Fq6::rand(&mut rng);
        let actual_result = op1 - op2;

        let input = json!(
        {
            "op1": op1.format(),
            "op2": op2.format(),
        }
        );
        let package_name = "fp6";
        let circom_template = "Fp6sub";
        verify(input, package_name, circom_template, actual_result);
    }

    #[test]
    fn Fp6Neg() {
        let mut rng = ark_std::test_rng();
        let op1 = Fq6::rand(&mut rng);
        let actual_result = -op1;

        let input = json!(
        {
            "op1": op1.format()
        }
        );
        let package_name = "fp6";
        let circom_template = "Fp6neg";
        verify(input, package_name, circom_template, actual_result);
    }

    #[test]
    fn Fp6mul() {
        let mut rng = ark_std::test_rng();
        let op1 = Fq6::rand(&mut rng);
        let op2 = Fq6::rand(&mut rng);
        let actual_result = op1 * op2;

        let input = json!(
        {
            "op1": op1.format(),
            "op2": op2.format(),
        }
        );
        let package_name = "fp6";
        let circom_template = "Fp6mul";
        verify(input, package_name, circom_template, actual_result);
    }

    #[test]
    fn Fp6inv() {
        let mut rng = ark_std::test_rng();
        let op1 = Fq6::rand(&mut rng);
        let actual_result = op1.inverse().unwrap();

        let input = json!(
        {
            "op1": op1.format()
        }
        );
        let package_name = "fp6";
        let circom_template = "Fp6inv";
        verify(input, package_name, circom_template, actual_result);
    }

    fn verify(
        input: serde_json::Value,
        package_name: &str,
        circom_template: &str,
        actual_result: Fq6,
    ) {
        let binding = env::current_dir().unwrap().join("src/parse/requirements");
        let output_dir = binding.to_str().unwrap();
        let package_path = get_path();
        write_json(&input, output_dir, package_name);

        let file_name = format!("{}/{}.circom", "fields/field", package_name);
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
        let witness_file_path = format!("{}/{}_witness.json", output_dir, package_name);
        let z = read_witness::<ark_bn254::Fq>(&witness_file_path.to_string());

        let constraint_path =
            format!("{}/{}_constraints.json", output_dir, package_name).to_string();
        let expected_result = Fq6::new(
            Fq2::new(z[1], z[2]),
            Fq2::new(z[3], z[4]),
            Fq2::new(z[5], z[6]),
        );

        assert_eq!(expected_result, actual_result, "assertion failed");
        // To Check Az.Bz = C.z
        let _ = R1CSConstructor::<ark_bn254::Fq>::construct(Some(&constraint_path), Some(&z), 0);
    }
}
