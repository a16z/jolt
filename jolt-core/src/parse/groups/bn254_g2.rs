#[cfg(test)]
mod tests {
    use crate::{
        parse::{generate_circuit_and_witness, get_path, read_witness, write_json, Parse},
        spartan::spartan_memory_checking::R1CSConstructor,
    };
    use ark_bn254::{g2::Config, Fq, Fq2, Fr, G2Projective};
    use ark_ec::{short_weierstrass::Projective, AffineRepr, CurveGroup};
    use ark_ff::{AdditiveGroup, Field, UniformRand};
    use serde_json::json;
    use std::env;
    use std::ops::Mul;

    #[test]
    fn g2_double() {
        let mut rng = ark_std::test_rng();
        let op = G2Projective::rand(&mut rng);

        let actual_out = op.double();

        let op_projective = op.into_affine().into_group();

        let input = json!({
                 "op1": { "x": {"x": op_projective.x.c0.to_string(),
                 "y": op_projective.x.c1.to_string()
             },
             "y": {"x": op_projective.y.c0.to_string(),
                  "y": op_projective.y.c1.to_string()
             },
             "z": {"x": op_projective.z.c0.to_string(),
                   "y": op_projective.z.c1.to_string()
        },
         }
             });

        // Change this.
        let package_name = "bn254_g2";
        let circom_template = "G2Double";

        verify(
            input,
            package_name,
            circom_template,
            actual_out.into_affine().into_group(),
        );
    }

    #[test]
    fn g2_add() {
        let mut rng = ark_std::test_rng();
        let op1 = G2Projective::rand(&mut rng);
        let op2 = G2Projective::rand(&mut rng);

        let actual_out = op1 + op2;

        let op1_projective = op1.into_affine().into_group();
        let op2_projective = op2.into_affine().into_group();

        let input = json!({

                "op1": { "x": {"x": op1_projective.x.c0.to_string(),
                                "y": op1_projective.x.c1.to_string()
                            },

                            "y": {"x": op1_projective.y.c0.to_string(),
                                 "y": op1_projective.y.c1.to_string()
                            },
                            "z": {"x": op1_projective.z.c0.to_string(),
                                  "y": op1_projective.z.c1.to_string()
                       },
                        },
                "op2": { "x": {"x": op2_projective.x.c0.to_string(),
                               "y": op2_projective.x.c1.to_string()},

                         "y": {"x": op2_projective.y.c0.to_string(),
                               "y": op2_projective.y.c1.to_string()
                                },
                        "z": {"x": op2_projective.z.c0.to_string(),
                                  "y": op2_projective.z.c1.to_string()
                       },
        }
        });

        let package_name = "bn254_g2";
        let circom_template = "G2Add";

        verify(
            input,
            package_name,
            circom_template,
            actual_out.into_affine().into_group(),
        );
    }

    #[test]
    fn g2_mul() {
        let mut rng = ark_std::test_rng();
        let point = G2Projective::rand(&mut rng).into_affine().into_group();
        let scalar = Fr::rand(&mut rng);
        let prod = point.mul(scalar);
        let input = json!(
        {
            "op1": { "x": {"x": point.x.c0.to_string(),
                                "y": point.x.c1.to_string()
                            },

                            "y": {"x": point.y.c0.to_string(),
                                 "y": point.y.c1.to_string()
                            },
                            "z": {"x": point.z.c0.to_string(),
                                  "y": point.z.c1.to_string()
                       },
                       },
            "op2" :  scalar.format_non_native()
        }
        );
        let package_name = "bn254_g2";
        let circom_template = "G2Mul";
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

        let computed_result: G2Projective;
        if z[3] == Fq::ZERO {
            computed_result = G2Projective {
                x: Fq2::ZERO,
                y: Fq2::ZERO,
                z: Fq2::ONE,
            }
            .into_affine()
            .into_group();
        } else {
            computed_result = G2Projective {
                x: Fq2 { c0: z[1], c1: z[2] } * (Fq2 { c0: z[5], c1: z[6] }).inverse().unwrap(),
                y: Fq2 { c0: z[3], c1: z[4] } * (Fq2 { c0: z[5], c1: z[6] }).inverse().unwrap(),
                z: Fq2::ONE,
            }
            .into_affine()
            .into_group();
        }

        assert_eq!(actual_result, computed_result, "assertion failed");
        // To Check Az.Bz = C.z
        let _ = R1CSConstructor::<ark_bn254::Fq>::construct(Some(&constraint_path), Some(&z), 0);
    }
}
