use core::fmt;

use super::hyrax::HyraxEvalProofCircom;
use super::non_native::Fqq;
use super::sum_check::SumcheckInstanceProofCircom;
use ark_grumpkin::Fq as Fp;

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

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReducedOpeningProofCircomHyrax {
    pub sumcheck_proof: SumcheckInstanceProofCircom,
    pub sumcheck_claims: Vec<Fqq>,
    pub joint_opening_proof: HyraxEvalProofCircom,
}

impl fmt::Debug for ReducedOpeningProofCircomHyrax {
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
