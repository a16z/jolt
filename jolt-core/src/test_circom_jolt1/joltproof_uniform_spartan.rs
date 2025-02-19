use core::fmt;

use super::{struct_fq::FqCircom, sum_check_gkr::{convert_sum_check_proof_to_circom, SumcheckInstanceProofCircom}};
use ark_bn254::Fr as Scalar;
use crate::{r1cs::{inputs::JoltR1CSInputs, spartan::UniformSpartanProof}, utils::poseidon_transcript::PoseidonTranscript};
use crate::jolt::vm::rv32i_vm::C;
use ark_ff::AdditiveGroup;

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct UniformSpartanProofCircom{
    pub outer_sumcheck_proof: SumcheckInstanceProofCircom,
    pub outer_sumcheck_claims: [FqCircom; 3],
    pub inner_sumcheck_proof: SumcheckInstanceProofCircom,
    pub claimed_witness_evals: Vec<FqCircom>
}

impl fmt::Debug for UniformSpartanProofCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                "outer_sumcheck_proof": {:?},
                "outer_sumcheck_claims": {:?},
                "inner_sumcheck_proof": {:?},
                "claimed_witness_evals": {:?}
            }}"#,
            self.outer_sumcheck_proof, self.outer_sumcheck_claims, self.inner_sumcheck_proof, self.claimed_witness_evals,
        )
    }
}


pub fn compute_uniform_spartan_to_circom(uni_spartan_proof: UniformSpartanProof<C, JoltR1CSInputs, Scalar, PoseidonTranscript<Scalar, Scalar>>) -> UniformSpartanProofCircom {

    let mut outer_s_c_claims: [FqCircom; 3] = [FqCircom(
        Scalar::ZERO) ; 3];

    outer_s_c_claims[0] = 
    FqCircom(uni_spartan_proof.outer_sumcheck_claims.0);
    outer_s_c_claims[1] =  FqCircom(uni_spartan_proof.outer_sumcheck_claims.1);

    outer_s_c_claims[2] = FqCircom(uni_spartan_proof.outer_sumcheck_claims.2);



    let mut claimed_witness_evals = Vec::new();
    for i in 0..uni_spartan_proof.claimed_witness_evals.len(){
        claimed_witness_evals.push(
            FqCircom(uni_spartan_proof.claimed_witness_evals[i])
        );
    }

    UniformSpartanProofCircom{
        outer_sumcheck_proof: convert_sum_check_proof_to_circom(&uni_spartan_proof.outer_sumcheck_proof),
        outer_sumcheck_claims: outer_s_c_claims,
        inner_sumcheck_proof: convert_sum_check_proof_to_circom(&uni_spartan_proof.inner_sumcheck_proof),
        claimed_witness_evals: claimed_witness_evals,
    }
}