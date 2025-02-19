use super::struct_fq::FqCircom;
use crate::{
    poly::unipoly::UniPoly, subprotocols::sumcheck::SumcheckInstanceProof,
    utils::poseidon_transcript::PoseidonTranscript,
};
use ark_bn254::{Bn254, Fr as Scalar};
use core::fmt;

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct UniPolyCircom {
    pub coeffs: Vec<FqCircom>,
}

impl fmt::Debug for UniPolyCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{\"coeffs\": {:?}}}", self.coeffs)
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct SumcheckInstanceProofCircom {
    pub uni_polys: Vec<UniPolyCircom>,
}

impl fmt::Debug for SumcheckInstanceProofCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{\"uni_polys\": {:?}}}", self.uni_polys)
    }
}

pub fn convert_sum_check_proof_to_circom(
    sum_check_proof: &SumcheckInstanceProof<Scalar, PoseidonTranscript<Scalar, Scalar>>,
) -> SumcheckInstanceProofCircom {
    let mut uni_polys_circom = Vec::new();
    for poly in &sum_check_proof.uni_polys {
        let mut temp_coeffs = Vec::new();
        for coeff in &poly.coeffs {
            temp_coeffs.push(FqCircom(*coeff));
        }
        uni_polys_circom.push(UniPolyCircom {
            coeffs: temp_coeffs,
        });
    }
    SumcheckInstanceProofCircom {
        uni_polys: uni_polys_circom,
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct BatchedGrandProductLayerProofCircom {
    pub proof: SumcheckInstanceProofCircom,
    pub left_claim: FqCircom,
    pub right_claim: FqCircom,
}

impl fmt::Debug for BatchedGrandProductLayerProofCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                    "proof": {:?},
                    "left_claim": {:?},
                    "right_claim": {:?}
                }}"#,
            self.proof, self.left_claim, self.right_claim
        )
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct BatchedGrandProductProofCircom {
    pub gkr_layers: Vec<BatchedGrandProductLayerProofCircom>,
}

impl fmt::Debug for BatchedGrandProductProofCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                "gkr_layers": {:?}
                }}"#,
            self.gkr_layers,
        )
    }
}
use crate::{
    poly::commitment::hyperkzg::HyperKZG, subprotocols::grand_product::BatchedGrandProductProof,
};

pub fn convert_from_batched_GKRProof_to_circom(
    proof: &BatchedGrandProductProof<
        HyperKZG<Bn254, PoseidonTranscript<Scalar, Scalar>>,
        PoseidonTranscript<Scalar, Scalar>,
    >,
) -> BatchedGrandProductProofCircom {
    let num_gkr_layers = proof.gkr_layers.len();

    let num_coeffs = proof.gkr_layers[num_gkr_layers - 1].proof.uni_polys[0]
        .coeffs
        .len();

    let max_no_polys = proof.gkr_layers[num_gkr_layers - 1].proof.uni_polys.len();

    let mut updated_gkr_layers = Vec::new();

    for idx in 0..num_gkr_layers {
        let zero_poly = UniPoly::from_coeff(vec![Scalar::from(0u8); num_coeffs]);
        let len = proof.gkr_layers[idx].proof.uni_polys.len();
        let updated_uni_poly: Vec<_> = proof.gkr_layers[idx]
            .proof
            .uni_polys
            .clone()
            .into_iter()
            .chain(vec![zero_poly; max_no_polys - len].into_iter())
            .collect();

        updated_gkr_layers.push(BatchedGrandProductLayerProofCircom {
            proof: convert_uni_polys_to_circom(updated_uni_poly),
            left_claim: FqCircom(proof.gkr_layers[idx].left_claim),
            right_claim: FqCircom(proof.gkr_layers[idx].right_claim),
        });
    }

    BatchedGrandProductProofCircom {
        gkr_layers: updated_gkr_layers,
    }
}

pub fn convert_uni_polys_to_circom(uni_polys: Vec<UniPoly<Scalar>>) -> SumcheckInstanceProofCircom {
    let mut uni_polys_circom = Vec::new();
    for poly in uni_polys {
        let mut temp_coeffs = Vec::new();
        for coeff in poly.coeffs {
            temp_coeffs.push(FqCircom(coeff));
        }
        uni_polys_circom.push(UniPolyCircom {
            coeffs: temp_coeffs,
        });
    }
    SumcheckInstanceProofCircom {
        uni_polys: uni_polys_circom,
    }
}
