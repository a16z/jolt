use ark_grumpkin::{Affine, Fq as Fp, Fr as Scalar, Projective};
use std::fmt;

use ark_ec::CurveGroup;

use crate::poly::commitment::hyrax::HyraxCommitment;
use crate::poly::commitment::hyrax::HyraxOpeningProof;
use crate::poly::commitment::hyrax::HyraxScheme;
use crate::poly::commitment::pedersen::PedersenGenerators;
use crate::spartan::spartan_memory_checking::SpartanProof;
use crate::utils::poseidon_transcript::PoseidonTranscript;

use super::non_native::convert_vec_to_fqq;
use super::non_native::Fqq;

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct HyraxGensCircom(pub Vec<G1Circom>);
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct HyraxCommitmentCircom(pub Vec<G1AffineCircom>);
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct HyraxEvalProofCircom(pub Vec<Fqq>);

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct G1Circom {
    pub x: Fp,
    pub y: Fp,
    pub z: Fp,
}

impl G1Circom {
    pub fn from_g1(elem: &Projective) -> G1Circom {
        G1Circom {
            x: elem.x,
            y: elem.y,
            z: elem.z,
        }
    }
}
impl fmt::Debug for G1Circom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                "x": "{:}",
                "y": "{:}",
                "z": "{:}"
            }}"#,
            self.x, self.y, self.z,
        )
    }
}

impl fmt::Debug for HyraxGensCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                "gens": {:?}
            }}"#,
            self.0
        )
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct G1AffineCircom {
    pub x: Fp,
    pub y: Fp,
}

impl G1AffineCircom {
    pub fn from_g1(elem: &Affine) -> G1AffineCircom {
        G1AffineCircom {
            x: elem.x,
            y: elem.y,
        }
    }
}
impl fmt::Debug for G1AffineCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                "x": "{:}",
                "y": "{:}"
            }}"#,
            self.x, self.y,
        )
    }
}

impl fmt::Debug for HyraxCommitmentCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                "commitments": {:?}
            }}"#,
            self.0
        )
    }
}

impl fmt::Debug for HyraxEvalProofCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                "tau": {:?}
            }}"#,
            self.0
        )
    }
}

pub fn hyrax_gens_to_circom(
    gens: &PedersenGenerators<Projective>,
    proof: &SpartanProof<
        Scalar,
        HyraxScheme<Projective, PoseidonTranscript<Scalar, Fp>>,
        PoseidonTranscript<Scalar, Fp>,
    >,
) -> HyraxGensCircom {
    let len = proof.pcs_proof.vector_matrix_product.len();
    HyraxGensCircom(
        gens.generators[..len]
            .iter()
            .map(|g| G1Circom::from_g1(&Projective::from(*g)))
            .collect(),
    )
}

pub fn hyrax_commitment_to_circom(commit: &HyraxCommitment<Projective>) -> HyraxCommitmentCircom {
    HyraxCommitmentCircom(
        commit
            .row_commitments
            .iter()
            .map(|g| G1AffineCircom::from_g1(&Projective::from(*g).into_affine()))
            .collect(),
    )
}

pub fn hyrax_eval_proof_to_circom(
    eval_proof: &HyraxOpeningProof<Projective, PoseidonTranscript<Scalar, Fp>>,
) -> HyraxEvalProofCircom {
    HyraxEvalProofCircom(convert_vec_to_fqq(&eval_proof.vector_matrix_product))
}
