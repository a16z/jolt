use core::fmt;

use super::{joltproof_bytecode_proof::{convert_multiset_hashes_to_circom, MultiSethashesCircom}, struct_fq::FqCircom, sum_check_gkr::{convert_from_batched_GKRProof_to_circom, convert_sum_check_proof_to_circom, BatchedGrandProductProofCircom, SumcheckInstanceProofCircom}};
use crate::{jolt::vm::read_write_memory::ReadWriteMemoryProof, poly::commitment::hyperkzg::HyperKZG, utils::poseidon_transcript::PoseidonTranscript};
use ark_bn254::{Bn254, Fr as Scalar};
use crate::lasso::memory_checking::StructuredPolynomialData;


const MEMORY_OPS_PER_INSTRUCTION: usize = 4;

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReadWriteMemoryCheckingProofCircom{
    pub multiset_hashes: MultiSethashesCircom,
    pub read_write_grand_product: BatchedGrandProductProofCircom,
    pub init_final_grand_product: BatchedGrandProductProofCircom,
    pub openings: Vec<FqCircom>,
    pub exogenous_openings: Vec<FqCircom>
}
impl fmt::Debug for ReadWriteMemoryCheckingProofCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        //             
        write!(
            f,
            r#"{{
                "multiset_hashes": {:?},
                "read_write_grand_product": {:?},
                "init_final_grand_product": {:?},
                "openings": {:?},
                "exogenous_openings": {:?}
            }}"#,
            self.multiset_hashes, self.read_write_grand_product, self.init_final_grand_product,self.openings
            , self.exogenous_openings,
        )
    }
}


#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct TimestampRangeCheckOpenings{
    read_cts_read_timestamp: Vec<FqCircom>,
    read_cts_global_minus_read: Vec<FqCircom>,
    final_cts_read_timestamp: Vec<FqCircom>,
    final_cts_global_minus_read: Vec<FqCircom>,
    identity: FqCircom
}

impl fmt::Debug for TimestampRangeCheckOpenings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                    "read_cts_read_timestamp": {:?},
                    "read_cts_global_minus_read": {:?},
                    "final_cts_read_timestamp": {:?},
                    "final_cts_global_minus_read": {:?},
                    "identity": {:?}
            }}"#,
            self.read_cts_read_timestamp, self.read_cts_global_minus_read, self.final_cts_read_timestamp, self.final_cts_global_minus_read, self.identity
        )
    }
}


#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct TimestampValidityProofCircom{
    pub multiset_hashes: MultiSethashesCircom,
    pub openings: TimestampRangeCheckOpenings,
    pub exogenous_openings: Vec<FqCircom>,
    pub batched_grand_product: BatchedGrandProductProofCircom
}

impl fmt::Debug for TimestampValidityProofCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {  
        write!(
            f,
            r#"{{
                "multiset_hashes": {:?},
                "openings": {:?},
                "exogenous_openings": {:?},
                "batched_grand_product": {:?}
            }}"#,
            self.multiset_hashes, self.openings, self.exogenous_openings, self.batched_grand_product
        )
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct OutputSumcheckProofCircom{
    sumcheck_proof: SumcheckInstanceProofCircom,
    opening: FqCircom
}

impl fmt::Debug for OutputSumcheckProofCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
            "sumcheck_proof": {:?},
            "opening": {:?}
            }}"#,
            self.sumcheck_proof, self.opening
        )
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReadWriteMemoryProofCircom{
    pub memory_checking_proof: ReadWriteMemoryCheckingProofCircom,
    pub timestamp_validity_proof: TimestampValidityProofCircom,
    pub output_proof: OutputSumcheckProofCircom,
}

impl fmt::Debug for ReadWriteMemoryProofCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        
        // 
        write!(
            f,
            r#"{{
            "memory_checking_proof": {:?},
            "timestamp_validity_proof": {:?},
            "output_proof": {:?}
            }}"#,
            self.memory_checking_proof, self.timestamp_validity_proof, self.output_proof
        )
    }
}

pub fn convert_from_read_write_mem_proof_to_circom(rw_mem_proof: ReadWriteMemoryProof<Scalar, HyperKZG<Bn254, PoseidonTranscript<Scalar, Scalar>>, PoseidonTranscript<Scalar, Scalar>>) -> ReadWriteMemoryProofCircom
{
    let mut openings = Vec::new();
    // confirm the 9 required values
    let rw_openings = rw_mem_proof.memory_checking_proof.openings;
    
    openings.push(
        FqCircom(rw_openings.a_ram),
        // Fqq{
        //     element: rw_openings.a_ram,
        //     limbs: convert_to_3_limbs(rw_openings.a_ram),
        // }
    );
    openings.push(
        FqCircom(rw_openings.v_read_rd),
        // Fqq{
        //     element: rw_openings.v_read_rd,
        //     limbs: convert_to_3_limbs(rw_openings.v_read_rd),
        // }
    );
    openings.push(
        FqCircom(rw_openings.v_read_rs1),
        // Fqq{
        //     element: rw_openings.v_read_rs1,
        //     limbs: convert_to_3_limbs(rw_openings.v_read_rs1),
        // }
    );
    openings.push(
        FqCircom(rw_openings.v_read_rs2),
        // Fqq{
        //     element: rw_openings.v_read_rs2,
        //     limbs: convert_to_3_limbs(rw_openings.v_read_rs2),
        // }
    );
    openings.push(
        FqCircom(rw_openings.v_read_ram),
        // Fqq{
        //     element: rw_openings.v_read_ram,
        //     limbs: convert_to_3_limbs(rw_openings.v_read_ram),
        // }
    );
    openings.push(
        FqCircom(rw_openings.v_write_rd),
        // Fqq{
        //     element: rw_openings.v_write_rd,
        //     limbs: convert_to_3_limbs(rw_openings.v_write_rd),
        // }
    );
    openings.push(
        FqCircom(rw_openings.v_write_ram),
        // Fqq{
        //     element: rw_openings.v_write_ram,
        //     limbs: convert_to_3_limbs(rw_openings.v_write_ram),
        // }
    );
    openings.push(
        FqCircom(rw_openings.v_final),
        // Fqq{
        //     element: rw_openings.v_final,
        //     limbs: convert_to_3_limbs(rw_openings.v_final),
        // }
    );
    openings.push( 
        FqCircom(rw_openings.t_read_rd),
        // Fqq{
        //     element: rw_openings.t_read_rd,
        //     limbs: convert_to_3_limbs(rw_openings.t_read_rd),
        // }
    );
    openings.push(
        FqCircom(rw_openings.t_read_rs1),
        // Fqq{
        //     element: rw_openings.t_read_rs1,
        //     limbs: convert_to_3_limbs(rw_openings.t_read_rs1),
        // }
    );
    openings.push(
        FqCircom(rw_openings.t_read_rs2),
        // Fqq{
        //     element: rw_openings.t_read_rs2,
        //     limbs: convert_to_3_limbs(rw_openings.t_read_rs2),
        // }
    );
    openings.push(
        FqCircom(rw_openings.t_read_ram),
        // Fqq{
        //     element: rw_openings.t_read_ram,
        //     limbs: convert_to_3_limbs(rw_openings.t_read_ram),
        // }
    );
    openings.push(
        FqCircom(rw_openings.t_final),
        // Fqq{
        //     element: rw_openings.t_final,
        //     limbs: convert_to_3_limbs(rw_openings.t_final),
        // }
    );
    for i in 0..3{
        openings.push(
            FqCircom(Scalar::from(0u8))
        );
        // openings.push(
        //     Fqq{
        //         element: Scalar::from(0u8),
        //         limbs: [Fp::from(0u8); 3],
        //         }
        // );
    }

    // println!("openings.len() is {}", openings.len());

    let exogenous_openings_from_rust = rw_mem_proof.memory_checking_proof.exogenous_openings;
    let mut exogenous_openings = Vec::new();
    exogenous_openings.push(
        FqCircom(exogenous_openings_from_rust.a_rd),
        // Fqq{
        //     element: exogenous_openings_from_rust.a_rd,
        //     limbs: convert_to_3_limbs(exogenous_openings_from_rust.a_rd),
        // }
    );
    exogenous_openings.push(
        FqCircom(exogenous_openings_from_rust.a_rs1),
    );
    exogenous_openings.push(
        FqCircom(exogenous_openings_from_rust.a_rs2),
    );
    // println!("exogenous_openings.len() is {}", exogenous_openings.len());


    let mem_checking_proof = ReadWriteMemoryCheckingProofCircom {
        multiset_hashes: convert_multiset_hashes_to_circom(&rw_mem_proof.memory_checking_proof.multiset_hashes),
        read_write_grand_product: convert_from_batched_GKRProof_to_circom(&rw_mem_proof.memory_checking_proof.read_write_grand_product),
        init_final_grand_product: convert_from_batched_GKRProof_to_circom(&rw_mem_proof.memory_checking_proof.init_final_grand_product),
        openings,
        exogenous_openings,
    };


    let ts_openings = rw_mem_proof.timestamp_validity_proof.openings;
    let mut openings = Vec::new();
    for opening in ts_openings.read_write_values() {
        openings.push(
            FqCircom(opening.clone()),
    );
    }
    
    let ts_exo_openings = rw_mem_proof.timestamp_validity_proof.exogenous_openings;
    let mut exo_openings: Vec<FqCircom> = Vec::new();
    for opening in ts_exo_openings {
        exo_openings.push(
            FqCircom(opening.clone()),
        //     Fqq {
        //     element: opening.clone(),
        //     limbs: convert_to_3_limbs(opening.clone()),
        // }
    );
    }

    
    let ts_validity_proof = TimestampValidityProofCircom{
        multiset_hashes: convert_multiset_hashes_to_circom(&rw_mem_proof.timestamp_validity_proof.multiset_hashes),
        openings: TimestampRangeCheckOpenings{
            read_cts_read_timestamp: openings[0..MEMORY_OPS_PER_INSTRUCTION].to_vec(),
            read_cts_global_minus_read: openings[MEMORY_OPS_PER_INSTRUCTION..2 * MEMORY_OPS_PER_INSTRUCTION].to_vec(),
            final_cts_read_timestamp: openings[2 * MEMORY_OPS_PER_INSTRUCTION..3 * MEMORY_OPS_PER_INSTRUCTION].to_vec(),
            final_cts_global_minus_read: openings[3 * MEMORY_OPS_PER_INSTRUCTION..4 * MEMORY_OPS_PER_INSTRUCTION].to_vec(),
            identity: FqCircom(Scalar::from(0u8))
        },
        exogenous_openings: exo_openings,
        batched_grand_product: convert_from_batched_GKRProof_to_circom(&rw_mem_proof.timestamp_validity_proof.batched_grand_product),
    };

    let ouput_sum_check_proof: OutputSumcheckProofCircom = OutputSumcheckProofCircom{
        sumcheck_proof: convert_sum_check_proof_to_circom(&rw_mem_proof.output_proof.sumcheck_proof),
        opening: FqCircom(rw_mem_proof.output_proof.opening),
    };

    ReadWriteMemoryProofCircom{
        memory_checking_proof: mem_checking_proof,
        timestamp_validity_proof: ts_validity_proof,
        output_proof: ouput_sum_check_proof,
    }
}


