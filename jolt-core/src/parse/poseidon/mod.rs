#[cfg(test)]
mod tests {
    const STATE_WIDTH: usize = 5;

    use crate::{
        field::JoltField,
        parse::{generate_circuit_and_witness, get_path, read_witness, write_json},
        spartan::spartan_memory_checking::R1CSConstructor,
    };
    use ark_bn254::{Fq, Fr};
    use ark_crypto_primitives::sponge::{
        poseidon::{get_poseidon_parameters, PoseidonDefaultConfigEntry, PoseidonSponge},
        CryptographicSponge,
    };
    use ark_ff::{PrimeField, UniformRand};
    use serde_json::json;
    use std::env;

    #[test]
    fn poseidon_bn_base() {
        let mut rng = ark_std::test_rng();
        let mut initial_state: Vec<Fq> = Vec::new();
        for _ in 0..STATE_WIDTH {
            initial_state.push(Fq::rand(&mut rng));
        }

        let params =
            get_poseidon_parameters::<Fq>(4, PoseidonDefaultConfigEntry::new(4, 5, 8, 56, 0))
                .unwrap();

        let mut poseidon_sponge = PoseidonSponge::<Fq>::new(&params);
        poseidon_sponge.state = initial_state.clone();

        assert_eq!(poseidon_sponge.state.len(), STATE_WIDTH);

        let input = json!(
            {
                "state": [
                    poseidon_sponge.state[0].to_string(),  poseidon_sponge.state[1].to_string(), poseidon_sponge.state[2].to_string(),  poseidon_sponge.state[3].to_string(), poseidon_sponge.state[4].to_string()
                ]
            }
        );

        poseidon_sponge.permute();
        let actual_result = poseidon_sponge.state;
        let package_name = "poseidon";
        let circom_template = "permute";
        let flag = 2;
        let prime = "grumpkin";
        verify(
            input,
            package_name,
            circom_template,
            actual_result,
            flag,
            prime,
        );
    }

    #[test]
    fn poseidon_bn_scalar() {
        let mut rng = ark_std::test_rng();
        let mut initial_state: Vec<Fr> = Vec::new();
        for _ in 0..STATE_WIDTH {
            initial_state.push(Fr::rand(&mut rng));
        }

        let params =
            get_poseidon_parameters::<Fr>(4, PoseidonDefaultConfigEntry::new(4, 5, 8, 56, 0))
                .unwrap();

        let mut poseidon_sponge = PoseidonSponge::<Fr>::new(&params);
        poseidon_sponge.state = initial_state.clone();

        assert_eq!(poseidon_sponge.state.len(), STATE_WIDTH);

        let input = json!(
            {
                "state": [
                    poseidon_sponge.state[0].to_string(),  poseidon_sponge.state[1].to_string(), poseidon_sponge.state[2].to_string(),  poseidon_sponge.state[3].to_string(), poseidon_sponge.state[4].to_string()
                ]
            }
        );

        poseidon_sponge.permute();
        let actual_result = poseidon_sponge.state;

        let package_name = "poseidon";
        let circom_template = "permute";
        let flag = 1;
        let prime = "bn128";
        verify(
            input,
            package_name,
            circom_template,
            actual_result,
            flag,
            prime,
        );
    }

    fn verify<F: PrimeField + JoltField>(
        input: serde_json::Value,
        package_name: &str,
        circom_template: &str,
        actual_result: Vec<F>,
        flag: usize,
        prime: &str,
    ) {
        let binding = env::current_dir().unwrap().join("src/parse/requirements");
        let output_dir = binding.to_str().unwrap();
        let package_path = get_path();
        write_json(&input, output_dir, package_name);

        let file_name = format!("{}/{}.circom", "transcript", package_name);
        let file_path = package_path.join(file_name);

        generate_circuit_and_witness(
            &file_path,
            &output_dir,
            circom_template,
            [flag].to_vec(),
            prime,
            None,
        );

        // // Read the witness.json file
        let witness_file_path = format!("{}/{}_witness.json", output_dir, package_name);
        let z = read_witness::<F>(&witness_file_path.to_string());

        let constraint_path =
            format!("{}/{}_constraints.json", output_dir, package_name).to_string();
        let mut expected_result = Vec::new();

        for i in 0..STATE_WIDTH {
            expected_result.push(z[i + 1].clone());
        }
        for i in 0..STATE_WIDTH {
            assert_eq!(
                expected_result[i].into_bigint(),
                actual_result[i].into_bigint(),
                "assertion failed"
            );
        }
        //To Check Az.Bz = C.z
        let _ = R1CSConstructor::<F>::construct(Some(&constraint_path), Some(&z), 0);
    }
}
