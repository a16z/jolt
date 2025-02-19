mod commitments;
mod grand_product;
mod hyrax;
mod memory_check;
mod non_native;
mod reduced_opening_proof;
mod spartan_proof;
mod sum_check;
mod transcript;
use super::spartan_memory_checking::{SpartanPreprocessing, SpartanProof};
use super::*;
use crate::{
    poly::commitment::{
        commitment_scheme::CommitmentScheme, hyperkzg::HyperKZG, hyrax::HyraxScheme,
    },
    utils::{poseidon_transcript::PoseidonTranscript, transcript::Transcript},
};
use ark_crypto_primitives::sponge::poseidon::{
    get_poseidon_parameters, PoseidonDefaultConfigEntry,
};
use ark_ec::{AdditiveGroup, CurveGroup};
use ark_ff::PrimeField;
use ark_grumpkin::{Fq as Fp, Fr as Scalar, Projective};
use commitments::SpartanCommitmentsHyraxCircom;
use hyrax::{hyrax_commitment_to_circom, hyrax_gens_to_circom};
use non_native::convert_vec_to_fqq;
use num_bigint::BigUint;
use spartan_proof::{preprocessing_to_pi_circom, SpartanProofHyraxCircom};
use std::{convert, fs::File, io::Write};
use transcript::convert_transcript_to_circom;

#[test]
fn parse_spartan_hyrax() {
    type ProofTranscript = PoseidonTranscript<Scalar, Fp>;
    type PCS = HyraxScheme<Projective, ProofTranscript>;
    let mut preprocessing = SpartanPreprocessing::<Scalar>::preprocess(None, None, 2);
    let commitment_shapes = SpartanProof::<Scalar, PCS, ProofTranscript>::commitment_shapes(
        preprocessing.inputs.len() + preprocessing.vars.len(),
    );
    let pcs_setup = PCS::setup(&commitment_shapes);
    let (_spartan_polynomials, _spartan_commitments) =
        SpartanProof::<Scalar, PCS, ProofTranscript>::generate_witness(&preprocessing, &pcs_setup);

    let proof = SpartanProof::<Scalar, PCS, ProofTranscript>::prove(&pcs_setup, &mut preprocessing);

    let transcipt_init = <PoseidonTranscript<Scalar, Fp> as Transcript>::new(b"Spartan transcript");

    let input_json = format!(
        r#"{{
        "pub_inp": {:?},
        "setup": {:?},
        "proof": {:?},
        "w_commitment": {:?},
        "transcript": {:?}
    }}"#,
        preprocessing_to_pi_circom(&preprocessing),
        hyrax_gens_to_circom(&pcs_setup, &proof),
        SpartanProofHyraxCircom::parse_spartan_proof(&proof),
        hyrax_commitment_to_circom(&proof.witness_commit),
        convert_transcript_to_circom(transcipt_init)
    );

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    SpartanProof::<Scalar, PCS, ProofTranscript>::verify(&pcs_setup, &preprocessing, proof)
        .unwrap();
}
