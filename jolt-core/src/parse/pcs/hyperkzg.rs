#[cfg(test)]
mod test {
    use std::env;

    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;
    use serde_json::json;

    use crate::{
        field::JoltField,
        parse::{generate_circuit_and_witness, get_path, read_witness, write_json, Parse},
        poly::{
            commitment::{
                commitment_scheme::{BatchType, CommitShape, CommitmentScheme},
                hyperkzg::HyperKZG,
            },
            dense_mlpoly::DensePolynomial,
        },
        spartan::spartan_memory_checking::R1CSConstructor,
        utils::{poseidon_transcript::PoseidonTranscript, transcript::Transcript},
    };
    type Fr = ark_bn254::Fr;
    type Fq = ark_bn254::Fq;
    type ProofTranscript = PoseidonTranscript<Fr, Fq>;
    type Pcs = HyperKZG<ark_bn254::Bn254, ProofTranscript>;
    #[test]
    fn test_hyperkzg() {
        let vars = 5;
        let mut rng = ChaCha8Rng::from_seed([2; 32]);
        let poly = DensePolynomial::random(vars, &mut rng);
        let opening_point: Vec<Fr> = (0..vars).map(|_| Fr::random(&mut rng)).collect();
        let eval = poly.evaluate(&opening_point);
        let commitment_shape = vec![CommitShape::new(1 << vars, BatchType::Small)];

        let srs = Pcs::setup(&commitment_shape);

        let commitment = Pcs::commit(&srs.0, &poly).unwrap();

        // prove an evaluation
        let mut prover_transcript = <ProofTranscript as Transcript>::new(b"label");
        let proof =
            Pcs::open(&srs.0, &poly, &opening_point, &eval, &mut prover_transcript).unwrap();

        // verify the evaluation
        let mut verifier_transcript = <ProofTranscript as Transcript>::new(b"label");

        HyperKZG::verify(
            &srs.1,
            &commitment,
            &opening_point,
            &eval,
            &proof,
            &mut verifier_transcript,
        )
        .unwrap();
        let verifier_transcript = <ProofTranscript as Transcript>::new(b"label");

        let input = json!(
        {
            "vk": srs.1.format(),
            "C": commitment.format(),
            "point": opening_point.iter().map(|point|point.format_non_native()).collect::<Vec<serde_json::Value>>(),
            "pi": proof.format(),
            "P_of_x": eval.format_non_native(),
            "transcript": {"state": verifier_transcript.state.state[1].to_string(), "nRounds": 0.to_string()}
        });

        let binding = env::current_dir().unwrap().join("src/parse/requirements");
        let output_dir = binding.to_str().unwrap();
        let package_path = get_path();

        let package_name = "hyperkzg";
        write_json(&input, output_dir, package_name);

        let file_name = format!("{}/{}.circom", "pcs", package_name);
        let file_path = package_path.join(file_name);

        let prime = "grumpkin";
        let circom_template = "HyperKzgVerifier";

        generate_circuit_and_witness(
            &file_path,
            &output_dir,
            circom_template,
            [vars].to_vec(),
            prime,
        );

        // // Read the witness.json file
        let witness_file_path = format!("{}/{}_witness.json", output_dir, package_name);
        let z = read_witness::<Fq>(&witness_file_path.to_string());

        let constraint_path =
            format!("{}/{}_constraints.json", output_dir, package_name).to_string();

        //To Check Az.Bz = C.z
        let _ = R1CSConstructor::<Fq>::construct(Some(&constraint_path), Some(&z), 0);
    }
}
