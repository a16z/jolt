use core::fmt;

use ark_bn254::{Bn254,  Fr as Scalar};

use super::{joltproof_bytecode_proof::{convert_multiset_hashes_to_circom, MultiSethashesCircom}, struct_fq::FqCircom, sum_check_gkr::{convert_from_batched_GKRProof_to_circom, convert_sum_check_proof_to_circom, BatchedGrandProductProofCircom, SumcheckInstanceProofCircom}};

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct InstMemoryCheckingProofCircom{
    pub multiset_hashes: MultiSethashesCircom,
    pub read_write_grand_product: BatchedGrandProductProofCircom,
    pub init_final_grand_product: BatchedGrandProductProofCircom,
    pub openings: Vec<FqCircom>
}

impl fmt::Debug for InstMemoryCheckingProofCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                "multiset_hashes": {:?},
                "read_write_grand_product": {:?},
                "init_final_grand_product": {:?},
                "openings": {:?}
            }}"#,
            self.multiset_hashes, self.read_write_grand_product, self.init_final_grand_product, self.openings,
        )
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct PrimarySumcheckCircom{
    pub sumcheck_proof: SumcheckInstanceProofCircom,
    pub openings: PrimarySumcheckOpeningsCircom
}

impl fmt::Debug for PrimarySumcheckCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                "sumcheck_proof": {:?},
                "openings": {:?}
            }}"#,
            self.sumcheck_proof, self.openings
        )
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct PrimarySumcheckOpeningsCircom{
    pub E_poly_openings: Vec<FqCircom>,
    pub flag_openings: Vec<FqCircom>,
    pub lookup_outputs_opening: FqCircom,
}

impl fmt::Debug for PrimarySumcheckOpeningsCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
            "E_poly_openings": {:?},
            "flag_openings": {:?},
            "lookup_outputs_opening": {:?}
            }}"#,
            self.E_poly_openings, self.flag_openings, self.lookup_outputs_opening
        )
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct InstructionLookupsProofCircom{
    pub primary_sumcheck: PrimarySumcheckCircom,
    pub memory_checking: InstMemoryCheckingProofCircom
}

impl fmt::Debug for InstructionLookupsProofCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                "primary_sumcheck": {:?},
                "memory_checking_proof": {:?}
            }}"#,
            self.primary_sumcheck, self.memory_checking
        )
    }
}
use crate::jolt::vm::instruction_lookups::{InstructionLookupsProof, PrimarySumcheckOpenings};

pub fn convert_from_primary_sum_check_opening_to_circom(prim_s_c_openings: &PrimarySumcheckOpenings<Scalar>) -> PrimarySumcheckOpeningsCircom{
    let mut E_poly_openings = Vec::new();
    for i in 0..prim_s_c_openings.E_poly_openings.len(){
        E_poly_openings.push(
            FqCircom(
                prim_s_c_openings.E_poly_openings[i],
            )
        )
    }
    let mut flag_openings = Vec::new();
    for i in 0..prim_s_c_openings.flag_openings.len(){
        flag_openings.push(
            FqCircom(
                prim_s_c_openings.flag_openings[i],
        )
        );
    }
    


    PrimarySumcheckOpeningsCircom{
        E_poly_openings,
        flag_openings,
        lookup_outputs_opening: FqCircom(
            prim_s_c_openings.lookup_outputs_opening,
        )
    }

}

use crate::jolt::vm::rv32i_vm::{RV32ISubtables, C, M, RV32I};
use crate::lasso::memory_checking::StructuredPolynomialData;
use crate::poly::commitment::hyperkzg::HyperKZG;
use crate::utils::poseidon_transcript::PoseidonTranscript;


pub fn convert_from_inst_lookups_proof_to_circom(inst_lookup_proof: InstructionLookupsProof<{C}, {M}, Scalar, HyperKZG<Bn254, PoseidonTranscript<Scalar, Scalar>>, RV32I, RV32ISubtables<Scalar> ,PoseidonTranscript<Scalar, Scalar>>) -> InstructionLookupsProofCircom{
    let primary_sum_check = PrimarySumcheckCircom{
        sumcheck_proof: convert_sum_check_proof_to_circom(&inst_lookup_proof.primary_sumcheck.sumcheck_proof),
        openings: convert_from_primary_sum_check_opening_to_circom(&inst_lookup_proof.primary_sumcheck.openings),
    };

    let mut openings = Vec::new();
    let lookup_openings = inst_lookup_proof.memory_checking.openings;
    
    for i in 0..lookup_openings.dim.len(){
        openings.push(
            FqCircom(
                lookup_openings.dim[i],
            )
        );
    }
    for i in 0..lookup_openings.read_cts.len(){
        openings.push(
            FqCircom(
                lookup_openings.read_cts[i],
            )
        )
    };
    for i in 0..lookup_openings.final_cts.len(){
        openings.push(
            FqCircom(
                lookup_openings.final_cts[i],
            )
        )
    };
    for i in 0..lookup_openings.E_polys.len(){
        openings.push(
            FqCircom(
                lookup_openings.E_polys[i],
            )
        )
    };
    for i in 0..lookup_openings.instruction_flags.len(){
        openings.push(
            FqCircom(
                lookup_openings.instruction_flags[i],
            )
        )
    };
    openings.push(
        FqCircom(
            lookup_openings.lookup_outputs,
            )
            
    );

    let mem_checking_proof = InstMemoryCheckingProofCircom{
        multiset_hashes: convert_multiset_hashes_to_circom(&inst_lookup_proof.memory_checking.multiset_hashes),
        read_write_grand_product: convert_from_batched_GKRProof_to_circom(&inst_lookup_proof.memory_checking.read_write_grand_product),
        init_final_grand_product: convert_from_batched_GKRProof_to_circom(&inst_lookup_proof.memory_checking.init_final_grand_product),
        openings: openings,
    };

    InstructionLookupsProofCircom{
        primary_sumcheck: primary_sum_check,
        memory_checking: mem_checking_proof,
    }
}