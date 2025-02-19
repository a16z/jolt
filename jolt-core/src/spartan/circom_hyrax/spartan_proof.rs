use super::hyrax::hyrax_eval_proof_to_circom;
use super::hyrax::HyraxEvalProofCircom;
use super::non_native::convert_vec_to_fqq;
use super::{
    non_native::{convert_to_3_limbs, Fqq},
    sum_check::{convert_sum_check_proof_to_circom, SumcheckInstanceProofCircom},
};
use ark_grumpkin::{Fq as Fp, Fr as Scalar, Projective};

use std::fmt;

use crate::poly::commitment::hyrax::HyraxScheme;
use crate::spartan::spartan_memory_checking::SpartanPreprocessing;

use crate::spartan::spartan_memory_checking::SpartanProof;
use crate::utils::poseidon_transcript::PoseidonTranscript;

pub struct SpartanProofHyraxCircom {
    pub outer_sumcheck_proof: SumcheckInstanceProofCircom,
    pub inner_sumcheck_proof: SumcheckInstanceProofCircom,
    pub outer_sumcheck_claims: [Fqq; 3],
    pub inner_sumcheck_claims: [Fqq; 4],
    pub pi_eval: Fqq,
    pub joint_opening_proof: HyraxEvalProofCircom,
}

impl SpartanProofHyraxCircom {
    pub fn new(
        outer_sumcheck_proof: SumcheckInstanceProofCircom,
        inner_sumcheck_proof: SumcheckInstanceProofCircom,
        outer_sumcheck_claims: [Fqq; 3],
        inner_sumcheck_claims: [Fqq; 4],
        pi_eval: Fqq,
        joint_opening_proof: HyraxEvalProofCircom,
    ) -> Self {
        Self {
            outer_sumcheck_proof,
            inner_sumcheck_proof,
            outer_sumcheck_claims,
            inner_sumcheck_claims,
            pi_eval,
            joint_opening_proof,
        }
    }

    pub fn parse_spartan_proof(
        proof: &SpartanProof<
            Scalar,
            HyraxScheme<Projective, PoseidonTranscript<Scalar, Fp>>,
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
            "pi_eval" : {:?},
            "joint_opening_proof": {:?}
            }}"#,
            self.outer_sumcheck_proof,
            self.outer_sumcheck_claims,
            self.inner_sumcheck_proof,
            self.inner_sumcheck_claims,
            self.pi_eval,
            self.joint_opening_proof
        )
    }
}

pub fn parse_spartan_proof_hyrax(
    proof: &SpartanProof<
        Scalar,
        HyraxScheme<Projective, PoseidonTranscript<Scalar, Fp>>,
        PoseidonTranscript<Scalar, Fp>,
    >,
) -> SpartanProofHyraxCircom {
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
    SpartanProofHyraxCircom::new(
        outer_sumcheck_proof,
        inner_sumcheck_proof,
        outer_sumcheck_claims,
        inner_sumcheck_claims,
        pi_eval,
        hyrax_eval_proof_to_circom(&proof.pcs_proof),
    )
}

pub fn preprocessing_to_pi_circom(preprocessing: &SpartanPreprocessing<Scalar>) -> Vec<Fqq> {
    convert_vec_to_fqq(&preprocessing.inputs)
}
