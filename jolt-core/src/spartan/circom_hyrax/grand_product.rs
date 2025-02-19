use core::fmt;

use super::non_native::convert_to_3_limbs;
use super::sum_check::convert_uni_polys_to_circom;
use super::{non_native::Fqq, sum_check::SumcheckInstanceProofCircom};
use crate::poly::commitment::hyrax::HyraxScheme;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::grand_product::BatchedGrandProductProof;
use crate::utils::poseidon_transcript::PoseidonTranscript;
use ark_grumpkin::{Fq as Fp, Fr as Scalar, Projective};

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct BatchedGrandProductLayerProofCircom {
    pub proof: SumcheckInstanceProofCircom,
    pub left_claim: Fqq,
    pub right_claim: Fqq,
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

pub struct VecFqq {
    pub state: Vec<Fqq>,
}
impl fmt::Debug for VecFqq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"[
            "{:?}"
            ]"#,
            self.state
        )
    }
}

pub fn convert_from_batched_GKRProof_to_circom_hyrax(
    proof: &BatchedGrandProductProof<
        HyraxScheme<Projective, PoseidonTranscript<Scalar, Fp>>,
        PoseidonTranscript<Scalar, Fp>,
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
            left_claim: Fqq {
                element: proof.gkr_layers[idx].left_claim,
                limbs: convert_to_3_limbs(proof.gkr_layers[idx].left_claim),
            },
            right_claim: Fqq {
                element: proof.gkr_layers[idx].right_claim,
                limbs: convert_to_3_limbs(proof.gkr_layers[idx].right_claim),
            },
        });
    }

    BatchedGrandProductProofCircom {
        gkr_layers: updated_gkr_layers,
    }
}
