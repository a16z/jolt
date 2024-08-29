#![allow(non_snake_case)]

mod test_constants;
mod subprotocols;
mod utils;
mod instruction;

use test_constants::*;

use crate::subprotocols::hyperkzg;
use ark_bn254::{Bn254, Fq, Fq2, Fr, G1Affine, G2Affine};
use ark_ff::PrimeField;
use ark_serialize::CanonicalDeserialize;
use jolt_types::poly::commitment::hyperkzg::{
    HyperKZGCommitment, HyperKZGProof, HyperKZGVerifierKey,
};
use jolt_types::poly::commitment::kzg::KZGVerifierKey;
use jolt_types::utils::transcript::ProofTranscript;
use solana_program::{
    account_info::AccountInfo, entrypoint, entrypoint::ProgramResult, msg, pubkey::Pubkey,
};
use solana_program::program_error::ProgramError;
use crate::instruction::VerifierInstruction;
use crate::subprotocols::grand_product::{verify_grand_product, GrandProductProof};

entrypoint!(process_instruction);

fn process_instruction(
    _program_id: &Pubkey,
    _accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    msg!("Solana WIP Jolt Verifier!");

    let instruction = VerifierInstruction::unpack(instruction_data)?;

    match instruction {
        VerifierInstruction::VerifyHyperKZG => {
            msg!("Running verify_hyperKZG_JOLT");
            verify_hyperKZG_JOLT();
        }
        VerifierInstruction::VerifySumcheck => {
            msg!("Running verify_sumcheck");
            verify_sumcheck();
        }
    }

    Ok(())
}

fn verify_sumcheck() {
    let proof: GrandProductProof<Fr> = CanonicalDeserialize::deserialize_uncompressed(&GRAND_PRODUCT_BATCH_PROOFS[..]).unwrap();
    let claims: Vec<Fr> = CanonicalDeserialize::deserialize_uncompressed(&GRAND_PRODUCT_CLAIMS[..]).unwrap();
    let mut transcript = ProofTranscript::new(b"test_transcript");
    let r_grand_product = verify_grand_product(&proof, &claims, &mut transcript);
    let expected_r_grand_product: Vec<Fr> =
        CanonicalDeserialize::deserialize_uncompressed(&GRAND_PRODUCT_R_PROVER[..]).unwrap();

    assert_eq!(expected_r_grand_product, r_grand_product);
}

fn verify_hyperKZG_JOLT() {
    let eval: Fr = CanonicalDeserialize::deserialize_uncompressed(&EVAL_BYTES[..]).unwrap();
    let point: Vec<Fr> = CanonicalDeserialize::deserialize_uncompressed(&POINT_BYTES[..]).unwrap();

    let proof = readProof();
    let vk = readVK();
    let C = readC();

    let mut transcript = ProofTranscript::new(b"TestEval");
    let res = hyperkzg::verify_hyperkzg(&vk, &C, &point, &eval, &proof, &mut transcript);
    msg!("Verify Result: {:?}", res);
}

fn readProof() -> HyperKZGProof<Bn254> {
    let v_pos: Vec<Vec<u8>> =
        CanonicalDeserialize::deserialize_uncompressed(&V_POS_BYTES[..]).unwrap();
    let v_neg: Vec<Vec<u8>> =
        CanonicalDeserialize::deserialize_uncompressed(&V_NEG_BYTES[..]).unwrap();
    let v_y: Vec<Vec<u8>> = CanonicalDeserialize::deserialize_uncompressed(&V_BYTES[..]).unwrap();

    let pre_com: Vec<Vec<u8>> =
        CanonicalDeserialize::deserialize_uncompressed(&COM_BYTES[..]).unwrap();
    let pre_w: Vec<Vec<u8>> = CanonicalDeserialize::deserialize_uncompressed(&W_BYTES[..]).unwrap();

    let mut com: Vec<G1Affine> = vec![];
    for coords in pre_com.chunks(2) {
        // convert coords into G1Affine
        let x = Fq::from_be_bytes_mod_order(&coords[0]);
        let y = Fq::from_be_bytes_mod_order(&coords[1]);
        com.push(G1Affine::new_unchecked(x, y));
    }

    let mut w: Vec<G1Affine> = vec![];
    for coords in pre_w.chunks(2) {
        // convert coords into G1Affine
        let x = Fq::from_be_bytes_mod_order(&coords[0]);
        let y = Fq::from_be_bytes_mod_order(&coords[1]);
        w.push(G1Affine::new_unchecked(x, y));
    }
    let v_pos = v_pos
        .iter()
        .map(|v| Fr::from_be_bytes_mod_order(v))
        .collect::<Vec<Fr>>();
    let v_neg = v_neg
        .iter()
        .map(|v| Fr::from_be_bytes_mod_order(v))
        .collect::<Vec<Fr>>();
    let v_y = v_y
        .iter()
        .map(|v| Fr::from_be_bytes_mod_order(v))
        .collect::<Vec<Fr>>();

    HyperKZGProof {
        com,
        w,
        v: vec![v_pos, v_neg, v_y],
    }
}

fn readVK() -> HyperKZGVerifierKey<Bn254> {
    let pre_g1: Vec<Vec<u8>> =
        CanonicalDeserialize::deserialize_uncompressed(&VK_G1_BYTES[..]).unwrap();
    let pre_g2: Vec<Vec<u8>> =
        CanonicalDeserialize::deserialize_uncompressed(&VK_G2_BYTES[..]).unwrap();
    let pre_beta_g2: Vec<Vec<u8>> =
        CanonicalDeserialize::deserialize_uncompressed(&VK_BETA_G2_BYTES[..]).unwrap();

    let x = Fq::from_be_bytes_mod_order(&pre_g1[0]);
    let y = Fq::from_be_bytes_mod_order(&pre_g1[1]);
    let g1 = G1Affine::new_unchecked(x, y);

    let x_c0 = Fq::from_be_bytes_mod_order(&pre_g2[0]);
    let y_c0 = Fq::from_be_bytes_mod_order(&pre_g2[1]);
    let x_c1 = Fq::from_be_bytes_mod_order(&pre_g2[2]);
    let y_c1 = Fq::from_be_bytes_mod_order(&pre_g2[3]);
    let g2 = G2Affine::new_unchecked(Fq2::new(x_c0, y_c0), Fq2::new(x_c1, y_c1));

    let x_c0 = Fq::from_be_bytes_mod_order(&pre_beta_g2[0]);
    let y_c0 = Fq::from_be_bytes_mod_order(&pre_beta_g2[1]);
    let x_c1 = Fq::from_be_bytes_mod_order(&pre_beta_g2[2]);
    let y_c1 = Fq::from_be_bytes_mod_order(&pre_beta_g2[3]);
    let beta_g2 = G2Affine::new_unchecked(Fq2::new(x_c0, y_c0), Fq2::new(x_c1, y_c1));

    HyperKZGVerifierKey {
        kzg_vk: KZGVerifierKey { g1, g2, beta_g2 },
    }
}

fn readC() -> HyperKZGCommitment<Bn254> {
    let pre_c: Vec<Vec<u8>> =
        CanonicalDeserialize::deserialize_uncompressed(&C_G1_BYTES[..]).unwrap();
    let x = Fq::from_be_bytes_mod_order(&pre_c[0]);
    let y = Fq::from_be_bytes_mod_order(&pre_c[1]);
    HyperKZGCommitment(G1Affine::new_unchecked(x, y))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_verify() {
        verify_hyperKZG_JOLT();
    }
}
