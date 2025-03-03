#[cfg(test)]
mod tests {
    use crate::{
        parse::{generate_circuit_and_witness, get_path, read_witness, write_json, Parse},
        spartan::spartan_memory_checking::R1CSConstructor,
    };
    use ark_bn254::{g1::Config, Fq, Fr, G1Projective};
    use ark_ec::{short_weierstrass::Projective, AffineRepr, CurveGroup};
    use ark_ff::{AdditiveGroup, Field, UniformRand};
    use serde_json::json;
    use std::env;
    use std::ops::Mul;

    #[test]
    fn g1_double() {
        let mut rng = ark_std::test_rng();
        let op = G1Projective::rand(&mut rng);

        let actual_out = op.double();

        let op_projective = op.into_affine().into_group();

        let input = json!({
            "op1": {
                "x": op_projective.x.to_string(),
                "y": op_projective.y.to_string(),
                "z": op_projective.z.to_string()
            }
        });

        let package_name = "bn254_g1";
        let circom_template = "G1Double";

        verify(
            input,
            package_name,
            circom_template,
            actual_out.into_affine().into_group(),
        );
    }

    #[test]
    fn g1_add() {
        let mut rng = ark_std::test_rng();
        let op1 = G1Projective::rand(&mut rng);
        let op2 = G1Projective::rand(&mut rng);

        let actual_out = op1 + op2;

        let op1_projective = op1.into_affine().into_group();
        let op2_projective = op2.into_affine().into_group();

        let input = json!({
            "op1": {
                "x": op1_projective.x.to_string(),
                "y": op1_projective.y.to_string(),
                "z": op1_projective.z.to_string()
            },
            "op2": {
                "x": op2_projective.x.to_string(),
                "y": op2_projective.y.to_string(),
                "z": op2_projective.z.to_string()
            },
        });

        let package_name = "bn254_g1";
        let circom_template = "G1Add";

        verify(
            input,
            package_name,
            circom_template,
            actual_out.into_affine().into_group(),
        );
    }

    #[test]
    fn g1_mul() {
        let mut rng = ark_std::test_rng();
        let point = G1Projective::rand(&mut rng).into_affine().into_group();
        let scalar = Fr::rand(&mut rng);
        let prod = point.mul(scalar);
        let input = json!(
        {
            "op1": {
                "x": point.x.to_string(),
                "y": point.y.to_string(),
                "z": point.z.to_string()
            }  ,
            "op2" :  scalar.format_non_native()
        }
        );
        let package_name = "bn254_g1";
        let circom_template = "G1Mul";
        verify(
            input,
            package_name,
            circom_template,
            prod.into_affine().into_group(),
        );
    }

    fn verify(
        input: serde_json::Value,
        package_name: &str,
        circom_template: &str,
        actual_result: Projective<Config>,
    ) {
        let binding = env::current_dir().unwrap().join("src/parse/requirements");
        let output_dir = binding.to_str().unwrap();
        let package_path = get_path();
        write_json(&input, output_dir, package_name);

        let file_name = format!("{}/{}.circom", "groups", package_name);
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

        let computed_result: G1Projective;
        if z[3] == Fq::ZERO {
            computed_result = G1Projective {
                x: Fq::ZERO,
                y: Fq::ZERO,
                z: Fq::ONE,
            }
            .into_affine()
            .into_group();
        } else {
            computed_result = G1Projective {
                x: z[1] / z[3],
                y: z[2] / z[3],
                z: Fq::ONE,
            }
            .into_affine()
            .into_group();
        }

        assert_eq!(actual_result, computed_result, "assertion failed");
        // To Check Az.Bz = C.z
        let _ = R1CSConstructor::<ark_bn254::Fq>::construct(Some(&constraint_path), Some(&z), 0);
    }
}
