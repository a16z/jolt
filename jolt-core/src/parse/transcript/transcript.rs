#[cfg(test)]
mod tests {
    const LEN: usize = 7;
    use crate::{
        field::JoltField,
        parse::{generate_circuit_and_witness, get_path, read_witness, write_json, Parse},
        spartan::spartan_memory_checking::R1CSConstructor,
        utils::{poseidon_transcript::PoseidonTranscript, transcript::Transcript},
    };
    use ark_bn254::{Fr, G1Affine, G1Projective};
    use ark_ff::{PrimeField, UniformRand};
    use serde_json::json;
    use std::env;

    #[test]
    fn new() {
        let label = b"Jolt transcript";
        let t = <PoseidonTranscript<Fr, Fr> as Transcript>::new(label);

        let label_scalar = ark_bn254::Fr::from_le_bytes_mod_order(label);

        let input = json!(
            {
                "scalar": label_scalar.to_string(),
            }
        );

        let package_name = "transcript";
        let circom_template = "TranscriptNew";

        let actual_results = vec![t.state.state[1], Fr::from(t.n_rounds)];
        verify(input, package_name, circom_template, actual_results);
    }
    #[test]
    fn append_scalar() {
        let mut rng = ark_std::test_rng();
        let mut t = <PoseidonTranscript<Fr, Fr> as Transcript>::new(b"label");

        let scalar_to_append = Fr::rand(&mut rng);

        let input = json!(
            {
                "scalar": scalar_to_append.to_string(),
                "transcript": {
                    "state": t.state.state[1].to_string(),
                    "nRounds": 0.to_string()
            }
            }
        );

        let package_name = "transcript";
        let circom_template = "AppendScalar";

        t.append_scalar(&scalar_to_append);
        let actual_results = vec![t.state.state[1], Fr::from(t.n_rounds)];

        verify(input, package_name, circom_template, actual_results);
    }

    #[test]
    fn append_scalars() {
        let mut rng = ark_std::test_rng();
        let mut t = <PoseidonTranscript<Fr, Fr> as Transcript>::new(b"label");

        let mut scalars_to_append = Vec::new();

        for _ in 0..LEN {
            scalars_to_append.push(Fr::rand(&mut rng));
        }

        let scalars_str = scalars_to_append
            .iter()
            .map(|x: &Fr| x.to_string())
            .collect::<Vec<String>>();

        let input = json!(
            {
                "scalars": scalars_str,
                "transcript": {
                    "state": t.state.state[1].to_string(),
                    "nRounds": 0.to_string()
            }
            }
        );

        let package_name = "transcript";
        let circom_template = "AppendScalars";

        t.append_scalars(&scalars_to_append);
        let actual_results = vec![t.state.state[1], Fr::from(t.n_rounds)];

        verify(input, package_name, circom_template, actual_results);
    }

    #[test]
    fn append_point() {
        let mut rng = ark_std::test_rng();
        let mut t = <PoseidonTranscript<Fr, Fr> as Transcript>::new(b"label");

        let point_to_append = G1Affine::rand(&mut rng);

        let input = json!(
            {
                "point":{
                    "x": point_to_append.x.format(),
                    "y": point_to_append.y.format()
                },
                "transcript": {
                    "state": t.state.state[1].to_string(),
                    "nRounds": 0.to_string()
            }
            }
        );

        let package_name = "transcript";
        let circom_template = "AppendPoint";

        t.append_point(&G1Projective::from(point_to_append));
        let actual_results = vec![t.state.state[1], Fr::from(t.n_rounds)];

        verify(input, package_name, circom_template, actual_results);
    }

    #[test]
    fn append_points() {
        let mut rng = ark_std::test_rng();
        let mut t = <PoseidonTranscript<Fr, Fr> as Transcript>::new(b"label");

        let mut points_to_append = Vec::new();
        for _ in 0..LEN {
            points_to_append.push(G1Affine::rand(&mut rng));
        }

        let points_str = points_to_append
            .iter()
            .map(|pt: &G1Affine| {
                json!({
                    "x": pt.x.format(),
                    "y": pt.y.format()
                })
            })
            .collect::<Vec<serde_json::Value>>();

        let input = json!(
            {
                "points": points_str,
                "transcript": {
                    "state": t.state.state[1].to_string(),
                    "nRounds": 0.to_string()
            }
            }
        );

        let package_name = "transcript";
        let circom_template = "AppendPoints";

        t.append_points(
            &points_to_append
                .iter()
                .map(|pt| G1Projective::from(*pt))
                .collect::<Vec<G1Projective>>(),
        );

        let actual_results = vec![t.state.state[1], Fr::from(t.n_rounds)];

        verify(input, package_name, circom_template, actual_results);
    }

    #[test]
    fn append_bytes() {
        let mut rng = ark_std::test_rng();
        let mut t = <PoseidonTranscript<Fr, Fr> as Transcript>::new(b"label");

        let mut bytes_to_append: Vec<u8> = Vec::new();
        for _ in 0..LEN {
            bytes_to_append.push(u8::rand(&mut rng));
        }

        let bytes_str = bytes_to_append
            .iter()
            .map(|x: &u8| x.to_string())
            .collect::<Vec<String>>();

        let input = json!(
            {
                "bytes": bytes_str,
                "transcript": {
                    "state": t.state.state[1].to_string(),
                    "nRounds": 0.to_string()
            }
            }
        );

        let package_name = "transcript";
        let circom_template = "AppendBytes";

        t.append_bytes(&bytes_to_append);

        let actual_results = vec![t.state.state[1], Fr::from(t.n_rounds)];

        verify(input, package_name, circom_template, actual_results);
    }

    #[test]
    fn challenge_scalar() {
        let mut t = <PoseidonTranscript<Fr, Fr> as Transcript>::new(b"label");

        let input = json!(
            {
                "transcript": {
                    "state": t.state.state[1].to_string(),
                    "nRounds": 0.to_string()
            }
            }
        );

        let package_name = "transcript";
        let circom_template = "ChallengeScalar";

        let rust_challenge = t.challenge_scalar::<Fr>();

        let actual_results = vec![t.state.state[1], Fr::from(t.n_rounds), rust_challenge];

        verify(input, package_name, circom_template, actual_results);
    }

    #[test]
    fn challenge_vector() {
        let mut t = <PoseidonTranscript<Fr, Fr> as Transcript>::new(b"label");

        let input = json!(
            {
                "transcript": {
                    "state": t.state.state[1].to_string(),
                    "nRounds": 0.to_string()
            }
            }
        );

        let package_name = "transcript";
        let circom_template = "ChallengeVector";

        let rust_challenge = t.challenge_vector(LEN);

        let actual_results =
            [vec![t.state.state[1], Fr::from(t.n_rounds)], rust_challenge].concat();

        verify(input, package_name, circom_template, actual_results);
    }

    #[test]
    fn challenge_powers() {
        let mut t = <PoseidonTranscript<Fr, Fr> as Transcript>::new(b"label");

        let input = json!(
            {
                "transcript": {
                    "state": t.state.state[1].to_string(),
                    "nRounds": 0.to_string()
            }
            }
        );

        let package_name = "transcript";
        let circom_template = "ChallengeScalarPowers";

        let rust_challenge = t.challenge_scalar_powers(LEN);

        let actual_results =
            [vec![t.state.state[1], Fr::from(t.n_rounds)], rust_challenge].concat();

        verify(input, package_name, circom_template, actual_results);
    }

    fn verify<F: JoltField + PrimeField>(
        input: serde_json::Value,
        package_name: &str,
        circom_template: &str,
        actual_results: Vec<F>,
    ) {
        let binding = env::current_dir().unwrap().join("src/parse/requirements");
        let output_dir = binding.to_str().unwrap();
        let package_path = get_path();
        write_json(&input, output_dir, package_name);

        let file_name = format!("{}/{}.circom", "transcript", package_name);
        let file_path = package_path.join(file_name);

        let prime = "bn128";

        let mut params = Vec::new();

        if circom_template == "AppendScalars"
            || circom_template == "AppendPoints"
            || circom_template == "AppendBytes"
            || circom_template == "ChallengeVector"
            || circom_template == "ChallengeScalarPowers"
        {
            params.push(LEN);
        }

        generate_circuit_and_witness(
            &file_path,
            &output_dir,
            circom_template,
            params,
            prime,
            None,
        );

        // Read the witness.json file
        let witness_file_path = format!("{}/{}_witness.json", output_dir, package_name);
        let z = read_witness::<F>(&witness_file_path.to_string());

        let constraint_path =
            format!("{}/{}_constraints.json", output_dir, package_name).to_string();

        let expected_state = F::from(z[1]);
        let expected_nRounds = F::from(z[2]);

        assert_eq!(expected_state, actual_results[0]);
        assert_eq!(expected_nRounds, actual_results[1]);

        if circom_template == "ChallengeScalar" {
            assert_eq!(actual_results[2], F::from(z[3]));
        }
        if circom_template == "ChallengeVector" || circom_template == "ChallengeScalarPowers" {
            for i in 0..LEN {
                assert_eq!(actual_results[2 + i], F::from(z[3 + i]));
            }
        }

        //To Check Az.Bz = C.z
        let _ = R1CSConstructor::<F>::construct(Some(&constraint_path), Some(&z), 0);
    }
}
