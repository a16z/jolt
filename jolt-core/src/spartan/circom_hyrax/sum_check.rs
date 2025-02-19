use core::fmt;
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct SumcheckInstanceProofCircom {
    pub uni_polys: Vec<UniPolyCircom>,
}
use ark_grumpkin::{Fq as Fp, Fr as Scalar};

use crate::{
    poly::unipoly::UniPoly, subprotocols::sumcheck::SumcheckInstanceProof,
    utils::poseidon_transcript::PoseidonTranscript,
};

use super::non_native::{convert_to_3_limbs, Fqq};
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct UniPolyCircom {
    pub coeffs: Vec<Fqq>,
}
impl fmt::Debug for SumcheckInstanceProofCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{\"uni_polys\": {:?}}}", self.uni_polys)
    }
}

impl fmt::Debug for UniPolyCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{\"coeffs\": {:?}}}", self.coeffs)
    }
}

pub fn convert_sum_check_proof_to_circom(
    sum_check_proof: &SumcheckInstanceProof<Scalar, PoseidonTranscript<Scalar, Fp>>,
) -> SumcheckInstanceProofCircom {
    let mut uni_polys_circom = Vec::new();
    for poly in &sum_check_proof.uni_polys {
        let mut temp_coeffs = Vec::new();
        for coeff in &poly.coeffs {
            temp_coeffs.push(Fqq {
                element: *coeff,
                limbs: convert_to_3_limbs(*coeff),
            });
        }
        uni_polys_circom.push(UniPolyCircom {
            coeffs: temp_coeffs,
        });
    }
    SumcheckInstanceProofCircom {
        uni_polys: uni_polys_circom,
    }
}

pub fn convert_uni_polys_to_circom(uni_polys: Vec<UniPoly<Scalar>>) -> SumcheckInstanceProofCircom {
    let mut uni_polys_circom = Vec::new();
    for poly in uni_polys {
        let mut temp_coeffs = Vec::new();
        for coeff in poly.coeffs {
            temp_coeffs.push(Fqq {
                element: coeff,
                limbs: convert_to_3_limbs(coeff),
            });
        }
        uni_polys_circom.push(UniPolyCircom {
            coeffs: temp_coeffs,
        });
    }
    SumcheckInstanceProofCircom {
        uni_polys: uni_polys_circom,
    }
}
