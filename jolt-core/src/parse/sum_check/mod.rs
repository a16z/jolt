// Sum check test for hyperkzg
#[cfg(test)]
mod tests{
    use std::env;

    use ark_bn254::{Bn254, Fq, Fq12, Fq2, Fq6, Fr, G1Projective, G2Projective};
    use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
    use ark_ff::{AdditiveGroup, UniformRand};
    use bincode::de;
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;
    use serde_json::json;
    type ProofTranscript = PoseidonTranscript<ark_bn254::Fr, ark_bn254::Fq>;

    use crate::{parse::{generate_circuit_and_witness, get_path, read_witness, spartan2::from_limbs, write_json, Parse}, poly::dense_mlpoly::DensePolynomial, spartan::spartan_memory_checking::R1CSConstructor, subprotocols::sumcheck::SumcheckInstanceProof, utils::{poseidon_transcript::PoseidonTranscript, transcript::Transcript}};

    #[test]
    fn sumcheck(){
        let  vars: usize = 3;
        let degree: usize = 1;
        let mut rng = ChaCha8Rng::from_seed([2; 32]);

        let poly = DensePolynomial::random(vars, &mut rng);
        let sum = poly.Z.iter().fold(Fr::ZERO, |acc, elem |acc + elem) ;

        let output_check_fn = |vals: &[Fr]| -> Fr { vals[0] };
        let mut transcript = <PoseidonTranscript<Fr, Fq> as Transcript>::new(b"label");

        let (proof, r, final_evals) = SumcheckInstanceProof::prove_arbitrary(
            &sum,
            vars,
            &mut vec![poly.clone()],
            output_check_fn,
            1,
            &mut transcript,
        );
        // verifier
        let mut transcript = <PoseidonTranscript<Fr, Fq> as Transcript>::new(b"label");
        let result = proof.verify(sum, 3, 1, &mut transcript);
        assert_eq!(final_evals[0], poly.evaluate(&r));

        let input = json!(
            {"initialClaim": sum.format_non_native(),
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

        let file_name = format!("{}/{}.circom", "spartan/spartan_hyperkzg/sum_check", package_name);
        let file_path = package_path.join(file_name);

        let prime = "grumpkin";
        generate_circuit_and_witness(&file_path, &output_dir, circom_template, [vars, degree].to_vec(), prime);

        // // Read the witness.json file
        let witness_file_path = format!("{}/{}_witness.json", output_dir, package_name);
        let z = read_witness::<ark_bn254::Fq>(&witness_file_path.to_string());

        let constraint_path =
            format!("{}/{}_constraints.json", output_dir, package_name).to_string();

        assert_eq!(transcript.state.state[1], z[1], "assertion failed");
        assert_eq!(from_limbs([z[3], z[4], z[5]].to_vec()), final_evals[0], "assertion failed");
    }

    // fn verify(
    //     input: serde_json::Value,
    //     package_name: &str,
    //     circom_template: &str,
    //     actual_result: Fq12,
    // ) {
        
    //     // let expected_result = Fq12::new(Fq6::new(Fq2::new(z[1], z[2]), Fq2::new(z[3], z[4]), Fq2::new(z[5], z[6])), Fq6::new(Fq2::new(z[7], z[8]), Fq2::new(z[9], z[10]), Fq2::new(z[11], z[12])));

    //     // assert_eq!(expected_result, actual_result, "assertion failed");
    //     // // To Check Az.Bz = C.z
    //     // let _ = R1CSConstructor::<ark_bn254::Fq>::construct(Some(&constraint_path), Some(&z), 0);
    // }
    

    
    // impl Parse for SumcheckInstanceProof<Fr, ProofTranscript> {
    //     fn format(&self) -> serde_json::Value {
    //         let uni_polys: Vec<serde_json::Value> =
    //             self.uni_polys.iter().map(|poly| poly.format()).collect();
    //         json!({
    //             "uni_polys": uni_polys,
    //         })
    //     }
    //     fn format_non_native(&self) -> serde_json::Value {
    //         let uni_polys: Vec<serde_json::Value> = self
    //             .uni_polys
    //             .iter()
    //             .map(|poly| poly.format_non_native())
    //             .collect();
    //         json!({
    //             "uni_polys": uni_polys,
    //         })
    //     }
    // }
}

