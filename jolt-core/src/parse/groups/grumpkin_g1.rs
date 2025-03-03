#[cfg(test)]
mod test {
    use crate::parse::{generate_circuit_and_witness, get_path, read_witness, write_json, Parse};
    use crate::spartan::spartan_memory_checking::R1CSConstructor;
    use ark_ec::{short_weierstrass::Projective, AffineRepr, CurveGroup};
    use ark_ff::{AdditiveGroup, Field};
    use ark_grumpkin::{Fq, Fr, GrumpkinConfig, Projective as GrumpkinProjective};
    use ark_std::UniformRand;
    use serde_json::json;
    use std::env;
    use std::ops::Mul;

    #[test]
    fn g1_double() {
        let mut rng = ark_std::test_rng();
        let op: Projective<GrumpkinConfig> = GrumpkinProjective::rand(&mut rng)
            .into_affine()
            .into_group();
        let actual_out = op.double();
        let input = json!({
            "op1": {
                "x": op.x.to_string(),
                "y": op.y.to_string(),
                "z": op.z.to_string()
            }
        });
        let package_name = "grumpkin_g1";
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
        let op1 = GrumpkinProjective::rand(&mut rng)
            .into_affine()
            .into_group();
        let op2 = GrumpkinProjective::rand(&mut rng)
            .into_affine()
            .into_group();
        let sum = op1 + op2;
        let input = json!(
        {
            "op1": {
                "x": op1.x.to_string(),
                "y": op1.y.to_string(),
                "z": op1.z.to_string()
            },
            "op2": {
                "x": op2.x.to_string(),
                "y": op2.y.to_string(),
                "z": op2.z.to_string()
            }
        }
        );
        let package_name = "grumpkin_g1";
        let circom_template = "G1Add";
        verify(
            input,
            package_name,
            circom_template,
            sum.into_affine().into_group(),
        );
    }

    #[test]
    fn g1_mul() {
        let mut rng = ark_std::test_rng();
        let point = GrumpkinProjective::rand(&mut rng)
            .into_affine()
            .into_group();
        let scalar = Fr::rand(&mut rng);
        let prod = point.mul(scalar);
        let input = json!(
        {
            "op1": {
                "x": point.x.to_string(),
                "y": point.y.to_string(),
                "z": point.z.to_string()
            }  ,
            "op2" :  scalar.format()
        }
        );
        let package_name = "grumpkin_g1";
        let circom_template = "G1Mul";
        verify(input, package_name, circom_template, prod);
    }

    fn verify(
        input: serde_json::Value,
        package_name: &str,
        circom_template: &str,
        actual_result: Projective<GrumpkinConfig>,
    ) {
        let binding = env::current_dir().unwrap().join("src/parse/requirements");
        let output_dir = binding.to_str().unwrap();
        let package_path = get_path();
        write_json(&input, output_dir, package_name);

        let file_name = format!("{}/{}.circom", "groups", package_name);
        let file_path = package_path.join(file_name);

        let prime = "bn128";

        generate_circuit_and_witness(
            &file_path,
            &output_dir,
            circom_template,
            [].to_vec(),
            prime,
            None,
        );

        // Read the witness.json file
        let witness_file_path = format!("{}/{}_witness.json", output_dir, package_name);
        let z = read_witness::<Fq>(&witness_file_path.to_string());
        let constraint_path =
            format!("{}/{}_constraints.json", output_dir, package_name).to_string();
        let computed_result: GrumpkinProjective;
        if z[3] == Fq::ZERO {
            computed_result = GrumpkinProjective {
                x: Fq::ZERO,
                y: Fq::ZERO,
                z: Fq::ONE,
            }
            .into_affine()
            .into_group();
        } else {
            computed_result = GrumpkinProjective {
                x: z[1] / z[3],
                y: z[2] / z[3],
                z: Fq::ONE,
            }
            .into_affine()
            .into_group();
        }

        assert_eq!(actual_result, computed_result, "assertion failed");

        //To Check Az.Bz = C.z
        let _ = R1CSConstructor::<Fq>::construct(Some(&constraint_path), Some(&z), 0);
    }
}
