use core::fmt;

use super::{helper_commitms::{convert_rust_fp_to_circom, G1AffineCircom}, struct_fq::FqCircom, sum_check_gkr::{convert_sum_check_proof_to_circom, SumcheckInstanceProofCircom}};
use crate::{poly::{commitment::hyperkzg::{HyperKZG, HyperKZGProof}, opening_proof::ReducedOpeningProof}, utils::poseidon_transcript::PoseidonTranscript};
use ark_bn254::{Bn254,  Fr as Scalar};


#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReducedOpeningProofCircom{
    pub sumcheck_proof: SumcheckInstanceProofCircom,
    pub sumcheck_claims: Vec<FqCircom>,
    pub joint_opening_proof: HyperKZGProofCircom
}

impl fmt::Debug for ReducedOpeningProofCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
            "sumcheck_proof": {:?},
            "sumcheck_claims": {:?},
            "joint_opening_proof": {:?}
            }}"#,
            self.sumcheck_proof, self.sumcheck_claims, self.joint_opening_proof
        )
    }
}

pub fn convert_reduced_opening_proof_to_circom(red_opening: ReducedOpeningProof<Scalar, HyperKZG<Bn254, PoseidonTranscript<Scalar, Scalar>>, PoseidonTranscript<Scalar, Scalar>>) -> ReducedOpeningProofCircom{
    let mut claims = Vec::new();
    // println!("red_opening.sumcheck_claims.len() is {}", red_opening.sumcheck_claims.len());
    for i in 0..red_opening.sumcheck_claims.len(){
        claims.push(
            FqCircom(red_opening.sumcheck_claims[i])
        )
    }
    ReducedOpeningProofCircom{
        sumcheck_proof: convert_sum_check_proof_to_circom(&red_opening.sumcheck_proof),
        sumcheck_claims: claims,
        joint_opening_proof: hyper_kzg_proof_to_hyper_kzg_circom(red_opening.joint_opening_proof),
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct HyperKZGProofCircom{
    pub com: Vec<G1AffineCircom>,
    pub w: [G1AffineCircom; 3],
    pub v: [Vec<FqCircom>; 3],
}


impl fmt::Debug for HyperKZGProofCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                    "com": {:?},
                    "w": [ {:?}, {:?}, {:?} ],
                    "v": {:?}
            }}"#,
            self.com,
            self.w[0], self.w[1], self.w[2],
            self.v

        )
    }
}

pub fn hyper_kzg_proof_to_hyper_kzg_circom(proof: HyperKZGProof<Bn254>) -> HyperKZGProofCircom {
    let com: Vec<G1AffineCircom> = proof
        .com
        .iter()
        .map(|c| G1AffineCircom { x: convert_rust_fp_to_circom(&c.x), y: convert_rust_fp_to_circom(&c.y) })
        .collect();

    let w = proof
        .w
        .iter()
        .map(|wi| G1AffineCircom { x: convert_rust_fp_to_circom(&wi.x), y: convert_rust_fp_to_circom( &wi.y) })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let mut v: [Vec<FqCircom>; 3] = Default::default();
    for i in 0..proof.v.len() {
        for j in 0..proof.v[i].len() {
            v[i].push(
                FqCircom(proof.v[i][j])
            )
        }
    }
    HyperKZGProofCircom { com, w, v }
}
