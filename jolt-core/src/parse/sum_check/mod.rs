// Sum check test for hyperkzg
#[cfg(test)]
mod tests {
    use crate::{
        parse::{
            generate_circuit_and_witness, get_path, read_witness, spartan2::from_limbs, write_json,
            Parse,
        },
        poly::dense_mlpoly::DensePolynomial,
        spartan::spartan_memory_checking::R1CSConstructor,
        subprotocols::sumcheck::SumcheckInstanceProof,
        utils::{poseidon_transcript::PoseidonTranscript, transcript::Transcript},
    };
    use ark_ff::AdditiveGroup;
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;
    use serde_json::json;
    use std::env;
    type Fr = ark_bn254::Fr;
    type Fq = ark_bn254::Fq;
    type ProofTranscript = PoseidonTranscript<Fr, Fq>;
    #[test]
    fn sumcheck() {
        let vars: usize = 3;
        let degree: usize = 1;
        let mut rng = ChaCha8Rng::from_seed([2; 32]);

        let poly = DensePolynomial::random(vars, &mut rng);
        let initial_sum = poly.Z.iter().fold(Fr::ZERO, |acc, elem| acc + elem);

        let output_check_fn = |vals: &[Fr]| -> Fr { vals[0] };
        let mut transcript = <PoseidonTranscript<Fr, Fq> as Transcript>::new(b"label");

        let (proof, r, final_evals) = SumcheckInstanceProof::prove_arbitrary(
            &Fr::ZERO,
            vars,
            &mut vec![poly.clone()],
            output_check_fn,
            1,
            &mut transcript,
        );

        // verifier
        let mut transcript = <PoseidonTranscript<Fr, Fq> as Transcript>::new(b"label");
        proof.verify(initial_sum, vars, 1, &mut transcript).unwrap();

        let transcript = <PoseidonTranscript<Fr, Fq> as Transcript>::new(b"label");
        assert_eq!(final_evals[0], poly.evaluate(&r));

        let input = json!(
            {"initialClaim": initial_sum.format_non_native(),
             "sumcheck_proof": proof.format_non_native(),
             "transcript": {
                    "state": transcript.state.state[1].to_string(),
                    "nRounds": 0.to_string()
             }
        }
        );
        let package_name = "sumcheck";
        let circom_template = "NonNativeSumCheck";

        let binding = env::current_dir().unwrap().join("src/parse/requirements");
        let output_dir = binding.to_str().unwrap();
        let package_path = get_path();
        write_json(&input, output_dir, package_name);

        let file_name = format!(
            "{}/{}.circom",
            "spartan/spartan_hyperkzg/sum_check", package_name
        );
        let file_path = package_path.join(file_name);

        let prime = "grumpkin";
        generate_circuit_and_witness(
            &file_path,
            &output_dir,
            circom_template,
            [vars, degree].to_vec(),
            prime,
            None,
        );

        // // Read the witness.json file
        let witness_file_path = format!("{}/{}_witness.json", output_dir, package_name);
        let z = read_witness::<Fq>(&witness_file_path.to_string());

        let constraint_path =
            format!("{}/{}_constraints.json", output_dir, package_name).to_string();

        assert_eq!(
            from_limbs::<Fq, Fr>([z[3], z[4], z[5]].to_vec()),
            final_evals[0],
            "assertion failed"
        );

        let _ = R1CSConstructor::<Fq>::construct(Some(&constraint_path), Some(&z), 0);
    }
}