use super::Parse;
use crate::{
    parse::jolt::{to_limbs, Fq, Fr},
    poly::commitment::{
        hyperkzg::{HyperKZGCommitment, HyperKZGProof, HyperKZGVerifierKey},
        hyrax::{HyraxCommitment, HyraxOpeningProof},
        kzg::KZGVerifierKey,
        pedersen::PedersenGenerators,
    },
    utils::poseidon_transcript::PoseidonTranscript,
};
use ark_ec::AdditiveGroup;
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::Field;
use serde_json::json;

//Parse Group
impl Parse for ark_bn254::G1Affine {
    fn format(&self) -> serde_json::Value {
        json!({
            "x": self.x.to_string(),
            "y": self.y.to_string()
        })
    }

    fn format_non_native(&self) -> serde_json::Value {
        let x_limbs = to_limbs::<Fq, Fr>(self.x);
        let y_limbs = to_limbs::<Fq, Fr>(self.y);
        json!({
            "x":{"limbs": [x_limbs[0].to_string(), x_limbs[1].to_string(), x_limbs[2].to_string()]},
            "y":{"limbs": [y_limbs[0].to_string(), y_limbs[1].to_string(), y_limbs[2].to_string()]}
        })
    }
}

impl Parse for ark_grumpkin::Projective {
    fn format(&self) -> serde_json::Value {
        json!({
            "x": self.x.to_string(),
            "y": self.y.to_string(),
            "z": self.z.to_string()
        })
    }
}
//Parse CommitmentSchems
//Parse Hyperkzg
impl Parse for HyperKZGVerifierKey<ark_bn254::Bn254> {
    fn format(&self) -> serde_json::Value {
        json!({
            "kzg_vk": self.kzg_vk.format()
        })
    }
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "kzg_vk": self.kzg_vk.format_non_native()
        })
    }
}

impl Parse for KZGVerifierKey<ark_bn254::Bn254> {
    fn format(&self) -> serde_json::Value {
        json!({
            "g1": self.g1.format(),
            "g2":{ "x": {"x": self.g2.x.c0.to_string(),
                        "y": self.g2.x.c1.to_string()
                    },

                    "y": {"x": self.g2.y.c0.to_string(),
                         "y": self.g2.y.c1.to_string()
                    },
                },
            "beta_g2": {"x": {"x": self.beta_g2.x.c0.to_string(),
                              "y": self.beta_g2.x.c1.to_string()},

                        "y": {"x": self.beta_g2.y.c0.to_string(),
                              "y": self.beta_g2.y.c1.to_string()}
                        }
        })
    }
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "g1": self.g1.format_non_native(),
            "g2":{ "x": {"x": self.g2.x.c0.format(),
                        "y": self.g2.x.c1.format()
                    },

                    "y": {"x": self.g2.y.c0.format(),
                         "y": self.g2.y.c1.format()
                    },
                },
            "beta_g2": {"x": {"x": self.beta_g2.x.c0.format(),
                              "y": self.beta_g2.x.c1.format()},

                        "y": {"x": self.beta_g2.y.c0.format(),
                              "y": self.beta_g2.y.c1.format()}
                        }
        })
    }
}

impl Parse for HyperKZGCommitment<ark_bn254::Bn254> {
    fn format(&self) -> serde_json::Value {
        json!({
            "commitment": self.0.format(),
        })
    }
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "commitment": self.0.format_non_native(),
        })
    }
}

impl Parse for HyperKZGProof<ark_bn254::Bn254> {
    fn format(&self) -> serde_json::Value {
        let com: Vec<serde_json::Value> = self.com.iter().map(|c| c.format()).collect();
        let w: Vec<serde_json::Value> = self.w.iter().map(|c| c.format()).collect();
        let v: Vec<Vec<serde_json::Value>> = self
            .v
            .iter()
            .map(|v_inner| {
                v_inner
                    .iter()
                    .map(|elem| elem.format_non_native())
                    .collect()
            })
            .collect();
        json!({
            "com": com,
            "w": w,
            "v": v,
        })
    }
    fn format_non_native(&self) -> serde_json::Value {
        let com: Vec<serde_json::Value> = self.com.iter().map(|c| c.format_non_native()).collect();
        let w: Vec<serde_json::Value> = self.w.iter().map(|c| c.format_non_native()).collect();
        let v: Vec<Vec<String>> = self
            .v
            .iter()
            .map(|v_inner| v_inner.iter().map(|elem| elem.to_string()).collect())
            .collect();
        json!({
            "com": com,
            "w": w,
            "v": v,
        })
    }
}

impl Parse for HyraxCommitment<ark_grumpkin::Projective> {
    fn format(&self) -> serde_json::Value {
        let commitments: Vec<serde_json::Value> = self
            .row_commitments
            .iter()
            .map(|commit| {
                if *commit == ark_grumpkin::Projective::ZERO {
                    ark_grumpkin::Projective::new_unchecked(
                        ark_grumpkin::Fq::ZERO,
                        ark_grumpkin::Fq::ONE,
                        ark_grumpkin::Fq::ZERO,
                    )
                    .format()
                } else {
                    commit.into_affine().into_group().format()
                }
            })
            .collect();
        json!({
            "row_commitments": commitments,
        })
    }
}

impl Parse
    for HyraxOpeningProof<
        ark_grumpkin::Projective,
        PoseidonTranscript<ark_grumpkin::Fr, ark_grumpkin::Fq>,
    >
{
    fn format(&self) -> serde_json::Value {
        let vector_matrix_product: Vec<serde_json::Value> = self
            .vector_matrix_product
            .iter()
            .map(|elem| elem.format())
            .collect();
        json!({
            "tau": vector_matrix_product,
        })
    }
}

impl Parse for PedersenGenerators<ark_grumpkin::Projective> {
    fn format(&self) -> serde_json::Value {
        let generators: Vec<serde_json::Value> = self
            .generators
            .iter()
            .map(|gen| gen.into_group().format())
            .collect();
        json!({
            "gens": generators
        })
    }
}
