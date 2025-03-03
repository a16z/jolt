#[cfg(test)]
mod tests {
    use ark_bn254::{Bn254, Fq12, Fq2, Fq6, G1Projective, G2Projective};
    use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
    use ark_ff::UniformRand;
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;
    use serde_json::json;
    use std::env;

    use crate::{
        parse::{generate_circuit_and_witness, get_path, read_witness, write_json},
        spartan::spartan_memory_checking::R1CSConstructor,
    };

    #[test]
    fn pairing() {
        let mut rng = ChaCha8Rng::from_seed([2; 32]);
        let P = G1Projective::rand(&mut rng).into_affine().into_group();
        let Q = G2Projective::rand(&mut rng).into_affine().into_group();
        let actual_result = Bn254::pairing(P, Q).0;

        let input = json!({
            "P": {
                "x": P.x.to_string(),
                "y": P.y.to_string(),
                "z": P.z.to_string()
            },
            "Q": {
                "x": {
                    "x": Q.x.c0.to_string(),
                    "y": Q.x.c1.to_string()
                },
                "y": {
                    "x": Q.y.c0.to_string(),
                    "y": Q.y.c1.to_string()
                }
            }
        });

        let package_name = "pairing";
        let circom_template = "Pairing";

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

        let file_name = format!("{}/{}.circom", "pcs", package_name);
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
        let expected_result = Fq12::new(
            Fq6::new(
                Fq2::new(z[1], z[2]),
                Fq2::new(z[3], z[4]),
                Fq2::new(z[5], z[6]),
            ),
            Fq6::new(
                Fq2::new(z[7], z[8]),
                Fq2::new(z[9], z[10]),
                Fq2::new(z[11], z[12]),
            ),
        );

        assert_eq!(expected_result, actual_result, "assertion failed");
        // To Check Az.Bz = C.z
        let _ = R1CSConstructor::<ark_bn254::Fq>::construct(Some(&constraint_path), Some(&z), 0);
    }
}
