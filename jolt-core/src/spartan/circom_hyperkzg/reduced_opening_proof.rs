use core::fmt;

use ark_bn254::Bn254;

use ark_bn254::Fq as Fp;

use super::non_native::convert_to_3_limbs;
use super::non_native::Fqq;
use super::sum_check::SumcheckInstanceProofCircom;
use crate::poly::commitment::hyperkzg::HyperKZGCommitment;
use crate::poly::commitment::hyperkzg::HyperKZGProof;
use crate::poly::commitment::hyperkzg::HyperKZGVerifierKey;

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct G1AffineCircom {
    pub x: Fp,
    pub y: Fp,
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Fp2Circom {
    pub x: Fp,
    pub y: Fp,
}

impl fmt::Debug for Fp2Circom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                                "x": "{}",
                                "y": "{}"
                            }}"#,
            self.x, self.y
        )
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct G2AffineCircom {
    pub x: Fp2Circom,
    pub y: Fp2Circom,
}

impl fmt::Debug for G2AffineCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                            "x": {:?},
                            "y": {:?}
                                }}"#,
            self.x, self.y
        )
    }
}

impl fmt::Debug for G1AffineCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                            "x": "{}",
                            "y": "{}"
                            }}"#,
            self.x, self.y
        )
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct HyperKZGCommitmentCircom {
    pub commitment: G1AffineCircom,
}

impl fmt::Debug for HyperKZGCommitmentCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
            "commitment": {:?}
            }}"#,
            self.commitment,
        )
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct KZGVerifierKeyCircom {
    pub g1: G1AffineCircom,
    pub g2: G2AffineCircom,
    pub beta_g2: G2AffineCircom,
}

impl fmt::Debug for KZGVerifierKeyCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                        "g1": {:?},
                        "g2": {:?},
                        "beta_g2": {:?}
            }}"#,
            self.g1, self.g2, self.beta_g2,
        )
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct HyperKZGVerifierKeyCircom {
    pub kzg_vk: KZGVerifierKeyCircom,
}

impl fmt::Debug for HyperKZGVerifierKeyCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                    "kzg_vk": {:?}
            }}"#,
            self.kzg_vk
        )
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct HyperKZGProofCircom {
    pub com: Vec<G1AffineCircom>,
    pub w: [G1AffineCircom; 3],
    pub v: [Vec<Fqq>; 3],
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
            self.com, self.w[0], self.w[1], self.w[2], self.v
        )
    }
}

pub fn hyper_kzg_proof_to_hyper_kzg_circom(proof: &HyperKZGProof<Bn254>) -> HyperKZGProofCircom {
    let com: Vec<G1AffineCircom> = proof
        .com
        .iter()
        .map(|c| G1AffineCircom { x: c.x, y: c.y })
        .collect();

    let w = proof
        .w
        .iter()
        .map(|wi| G1AffineCircom { x: wi.x, y: wi.y })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let mut v: [Vec<Fqq>; 3] = Default::default();
    for i in 0..proof.v.len() {
        for j in 0..proof.v[i].len() {
            v[i].push(Fqq {
                element: proof.v[i][j],
                limbs: convert_to_3_limbs(proof.v[i][j]),
            })
        }
    }
    HyperKZGProofCircom { com, w, v }
}

pub fn convert_hyperkzg_commitment_to_circom(
    commitment: &HyperKZGCommitment<Bn254>,
) -> HyperKZGCommitmentCircom {
    HyperKZGCommitmentCircom {
        commitment: G1AffineCircom {
            x: commitment.0.x,
            y: commitment.0.y,
        },
    }
}

pub fn convert_hyperkzg_verifier_key_to_hyperkzg_verifier_key_circom(
    vk: HyperKZGVerifierKey<Bn254>,
) -> HyperKZGVerifierKeyCircom {
    HyperKZGVerifierKeyCircom {
        kzg_vk: KZGVerifierKeyCircom {
            g1: G1AffineCircom {
                x: vk.kzg_vk.g1.x,
                y: vk.kzg_vk.g1.y,
            },
            g2: G2AffineCircom {
                x: Fp2Circom {
                    x: vk.kzg_vk.g2.x.c0,
                    y: vk.kzg_vk.g2.x.c1,
                },
                y: Fp2Circom {
                    x: vk.kzg_vk.g2.y.c0,
                    y: vk.kzg_vk.g2.y.c1,
                },
            },
            beta_g2: G2AffineCircom {
                x: Fp2Circom {
                    x: vk.kzg_vk.beta_g2.x.c0,
                    y: vk.kzg_vk.beta_g2.x.c1,
                },
                y: Fp2Circom {
                    x: vk.kzg_vk.beta_g2.y.c0,
                    y: vk.kzg_vk.beta_g2.y.c1,
                },
            },
        },
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReducedOpeningProofCircom {
    pub sumcheck_proof: SumcheckInstanceProofCircom,
    pub sumcheck_claims: Vec<Fqq>,
    pub joint_opening_proof: HyperKZGProofCircom,
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
