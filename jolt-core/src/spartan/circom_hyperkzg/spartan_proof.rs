use ark_bn254::Bn254;
use ark_bn254::Fq as Fp;
use ark_bn254::Fr as Scalar;
use ark_bn254::G1Projective;

use ark_std::Zero;

use super::non_native::convert_vec_to_fqq;

use super::{
    non_native::{convert_to_3_limbs, Fqq},
    reduced_opening_proof::{hyper_kzg_proof_to_hyper_kzg_circom, HyperKZGProofCircom},
    sum_check::{convert_sum_check_proof_to_circom, SumcheckInstanceProofCircom},
};

use std::fmt;

use crate::lasso::memory_checking::StructuredPolynomialData;
use crate::poly::commitment::hyperkzg::HyperKZG;
use crate::poly::commitment::hyrax::HyraxScheme;
use crate::spartan::spartan_memory_checking::SpartanPreprocessing;
use crate::spartan::spartan_memory_checking::SpartanProof;
use crate::utils::poseidon_transcript::PoseidonTranscript;

use super::spartan_memory_checking::SpartanOpenings;

pub struct SpartanProofCircom {
    pub outer_sumcheck_proof: SumcheckInstanceProofCircom,
    pub inner_sumcheck_proof: SumcheckInstanceProofCircom,
    pub outer_sumcheck_claims: [Fqq; 3],
    pub inner_sumcheck_claims: [Fqq; 4],
    pub pi_eval: Fqq,

    pub w_opening: HyperKZGProofCircom,
}

impl SpartanProofCircom {
    pub fn new(
        outer_sumcheck_proof: SumcheckInstanceProofCircom,
        inner_sumcheck_proof: SumcheckInstanceProofCircom,
        outer_sumcheck_claims: [Fqq; 3],
        inner_sumcheck_claims: [Fqq; 4],
        pi_eval: Fqq,
        w_opening: HyperKZGProofCircom,
    ) -> Self {
        Self {
            outer_sumcheck_proof,
            inner_sumcheck_proof,
            outer_sumcheck_claims,
            inner_sumcheck_claims,
            pi_eval,
            w_opening,
        }
    }

    pub fn parse_spartan_proof(
        proof: &SpartanProof<
            Scalar,
            HyperKZG<Bn254, PoseidonTranscript<Scalar, Fp>>,
            PoseidonTranscript<Scalar, Fp>,
        >,
    ) -> Self {
        parse_spartan_proof(proof)
    }
}

impl fmt::Debug for SpartanProofCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"
            {{
            "outer_sumcheck_proof": {:?},
            "outer_sumcheck_claims": {:?},
            "inner_sumcheck_proof": {:?},
            "inner_sumcheck_claims": {:?},
            "pi_eval" : {:?},
            "joint_opening_proof": {:?}
            }}"#,
            self.outer_sumcheck_proof,
            self.outer_sumcheck_claims,
            self.inner_sumcheck_proof,
            self.inner_sumcheck_claims,
            self.pi_eval,
            self.w_opening
        )
    }
}

pub fn parse_spartan_proof(
    proof: &SpartanProof<
        Scalar,
        HyperKZG<Bn254, PoseidonTranscript<Scalar, Fp>>,
        PoseidonTranscript<Scalar, Fp>,
    >,
) -> SpartanProofCircom {
    let outer_sumcheck_proof = convert_sum_check_proof_to_circom(&proof.outer_sumcheck_proof);
    let inner_sumcheck_proof = convert_sum_check_proof_to_circom(&proof.inner_sumcheck_proof);
    let outer_sumcheck_claims = [
        Fqq {
            element: proof.outer_sumcheck_claims.0,
            limbs: convert_to_3_limbs(proof.outer_sumcheck_claims.0),
        },
        Fqq {
            element: proof.outer_sumcheck_claims.1,
            limbs: convert_to_3_limbs(proof.outer_sumcheck_claims.1),
        },
        Fqq {
            element: proof.outer_sumcheck_claims.2,
            limbs: convert_to_3_limbs(proof.outer_sumcheck_claims.2),
        },
    ];
    let inner_sumcheck_claims = [
        Fqq {
            element: proof.inner_sumcheck_claims.0,
            limbs: convert_to_3_limbs(proof.inner_sumcheck_claims.0),
        },
        Fqq {
            element: proof.inner_sumcheck_claims.1,
            limbs: convert_to_3_limbs(proof.inner_sumcheck_claims.1),
        },
        Fqq {
            element: proof.inner_sumcheck_claims.2,
            limbs: convert_to_3_limbs(proof.inner_sumcheck_claims.2),
        },
        Fqq {
            element: proof.inner_sumcheck_claims.3,
            limbs: convert_to_3_limbs(proof.inner_sumcheck_claims.3),
        },
    ];

    let pi_eval = Fqq {
        element: proof.pi_eval,
        limbs: convert_to_3_limbs(proof.pi_eval),
    };

    let w_opening = hyper_kzg_proof_to_hyper_kzg_circom(&proof.pcs_proof);
    SpartanProofCircom::new(
        outer_sumcheck_proof,
        inner_sumcheck_proof,
        outer_sumcheck_claims,
        inner_sumcheck_claims,
        pi_eval,
        w_opening,
    )
}

pub struct SpartanProofHyraxCircom {
    pub outer_sumcheck_proof: SumcheckInstanceProofCircom,
    pub inner_sumcheck_proof: SumcheckInstanceProofCircom,
    pub outer_sumcheck_claims: [Fqq; 3],
    pub inner_sumcheck_claims: [Fqq; 4],
}

impl SpartanProofHyraxCircom {
    pub fn new(
        outer_sumcheck_proof: SumcheckInstanceProofCircom,
        inner_sumcheck_proof: SumcheckInstanceProofCircom,
        outer_sumcheck_claims: [Fqq; 3],
        inner_sumcheck_claims: [Fqq; 4],
    ) -> Self {
        Self {
            outer_sumcheck_proof,
            inner_sumcheck_proof,
            outer_sumcheck_claims,
            inner_sumcheck_claims,
        }
    }

    pub fn parse_spartan_proof(
        proof: &SpartanProof<
            Scalar,
            HyraxScheme<G1Projective, PoseidonTranscript<Scalar, Fp>>,
            PoseidonTranscript<Scalar, Fp>,
        >,
    ) -> Self {
        parse_spartan_proof_hyrax(proof)
    }
}

impl fmt::Debug for SpartanProofHyraxCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"
            {{
            "outer_sumcheck_proof": {:?},
            "outer_sumcheck_claims": {:?},
            "inner_sumcheck_proof": {:?},
            "inner_sumcheck_claims": {:?},
            }}"#,
            self.outer_sumcheck_proof,
            self.outer_sumcheck_claims,
            self.inner_sumcheck_proof,
            self.inner_sumcheck_claims,
        )
    }
}

pub fn parse_spartan_proof_hyrax(
    proof: &SpartanProof<
        Scalar,
        HyraxScheme<G1Projective, PoseidonTranscript<Scalar, Fp>>,
        PoseidonTranscript<Scalar, Fp>,
    >,
) -> SpartanProofHyraxCircom {
    let outer_sumcheck_proof = convert_sum_check_proof_to_circom(&proof.outer_sumcheck_proof);
    let inner_sumcheck_proof = convert_sum_check_proof_to_circom(&proof.inner_sumcheck_proof);
    // let spark_sumcheck_proof = convert_sum_check_proof_to_circom(&proof.spark_sumcheck_proof);
    let outer_sumcheck_claims = [
        Fqq {
            element: proof.outer_sumcheck_claims.0,
            limbs: convert_to_3_limbs(proof.outer_sumcheck_claims.0),
        },
        Fqq {
            element: proof.outer_sumcheck_claims.1,
            limbs: convert_to_3_limbs(proof.outer_sumcheck_claims.1),
        },
        Fqq {
            element: proof.outer_sumcheck_claims.2,
            limbs: convert_to_3_limbs(proof.outer_sumcheck_claims.2),
        },
    ];
    let inner_sumcheck_claims = [
        Fqq {
            element: proof.inner_sumcheck_claims.0,
            limbs: convert_to_3_limbs(proof.inner_sumcheck_claims.0),
        },
        Fqq {
            element: proof.inner_sumcheck_claims.1,
            limbs: convert_to_3_limbs(proof.inner_sumcheck_claims.1),
        },
        Fqq {
            element: proof.inner_sumcheck_claims.2,
            limbs: convert_to_3_limbs(proof.inner_sumcheck_claims.2),
        },
        Fqq {
            element: proof.inner_sumcheck_claims.3,
            limbs: convert_to_3_limbs(proof.inner_sumcheck_claims.3),
        },
    ];

    SpartanProofHyraxCircom::new(
        outer_sumcheck_proof,
        inner_sumcheck_proof,
        outer_sumcheck_claims,
        inner_sumcheck_claims,
    )
}

pub fn convert_and_flatten_spark_openings(openings: &SpartanOpenings<Scalar>) -> [Fqq; 24] {
    let mut flattened_opening = [Fqq {
        element: Scalar::zero(),
        limbs: [Fp::zero(); 3],
    }; 24];

    let read_write_opening = openings.read_write_values();
    for i in 0..18 {
        flattened_opening[i] = Fqq {
            element: *read_write_opening[i],
            limbs: convert_to_3_limbs(*read_write_opening[i]),
        };
    }
    let init_final_opening = openings.init_final_values();

    for i in 0..6 {
        flattened_opening[18 + i] = Fqq {
            element: *init_final_opening[i],
            limbs: convert_to_3_limbs(*init_final_opening[i]),
        };
    }
    flattened_opening
}

pub fn preprocessing_to_pi_circom(preprocessing: &SpartanPreprocessing<Scalar>) -> Vec<Fqq> {
    convert_vec_to_fqq(&preprocessing.inputs)
}
