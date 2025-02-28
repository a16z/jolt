#[cfg(test)]
mod test {
    use crate::{
        field::JoltField,
        parse::{generate_circuit_and_witness, get_path, read_witness, write_json, Parse},
        poly::{
            commitment::{
                hyrax::{HyraxCommitment, HyraxOpeningProof},
                pedersen::PedersenGenerators,
            },
            dense_mlpoly::DensePolynomial,
        },
        spartan::spartan_memory_checking::R1CSConstructor,
        utils::{poseidon_transcript::PoseidonTranscript, transcript::Transcript},
    };
    use ark_grumpkin::Projective;
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;
    use serde_json::json;
    use std::env;
    type Fr = ark_grumpkin::Fr;
    type Fq = ark_grumpkin::Fq;
    type ProofTranscript = PoseidonTranscript<Fr, Fq>;

    #[test]
    fn hyrax() {
        let vars = 5;
        let gens: PedersenGenerators<Projective> = PedersenGenerators::new(1 << (vars / 2), &[0]);
        let mut transcript: PoseidonTranscript<Fr, Fq> =
            <ProofTranscript as Transcript>::new(b"label");
        let mut rng = ChaCha8Rng::from_seed([2; 32]);

        let poly = DensePolynomial::random(vars, &mut rng);
        let commit = HyraxCommitment::commit(&poly, &gens);
        let opening_point: Vec<Fr> = (0..vars).map(|_| Fr::random(&mut rng)).collect();

        let eval_proof: HyraxOpeningProof<Projective, ProofTranscript> =
            HyraxOpeningProof::prove(&poly, &opening_point, 1, &mut transcript);
        let eval = poly.evaluate(&opening_point);

        let input = json!(
          {
                "commit": commit.format(),
                "proof": eval_proof.format(),
                "setup": gens.format(),
                "evaluation": eval.format(),
                "eval_point": opening_point.iter().map(|point|point.format()).collect::<Vec<serde_json::Value>>()
        });

        let binding = env::current_dir().unwrap().join("src/parse/requirements");
        let output_dir = binding.to_str().unwrap();
        let package_path = get_path();

        let package_name = "hyrax";
        write_json(&input, output_dir, package_name);

        let file_name = format!("{}/{}.circom", "pcs", package_name);
        let file_path = package_path.join(file_name);

        let prime = "bn128";
        let circom_template = "HyraxVerifier";
        generate_circuit_and_witness(
            &file_path,
            &output_dir,
            circom_template,
            [vars].to_vec(),
            prime,
            None,
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
