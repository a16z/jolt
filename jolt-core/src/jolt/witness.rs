use rayon::prelude::*;
use tracer::instruction::RV32IMCycle;

use crate::{
    field::JoltField,
    jolt::vm::{ram::remap_address, JoltProverPreprocessing},
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial, sparse_matrix_polynomial::OneHotPolynomial,
    },
    utils::transcript::Transcript,
};

use super::instruction::{CircuitFlags, InstructionFlags, LookupQuery};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CommittedPolynomials {
    /* R1CS aux variables */
    /// The "left" input to the current instruction. Typically either the
    /// rs1 value or the current program counter.
    LeftInstructionInput,
    /// The "right" input to the current instruction. Typically either the
    /// rs2 value or the immediate value.
    RightInstructionInput,
    /// Product of `LeftInstructionInput` and `RightInstructionInput`
    Product,
    /// Whether the current instruction should write the lookup output to
    /// the destination register
    WriteLookupOutputToRD,
    /// Whether the current instruction should write the program counter to
    /// the destination register
    WritePCtoRD,
    /// Whether the current instruction triggers a branch
    ShouldBranch,
    /*  Twist/Shout witnesses */
    /// One-hot ra polynomial for the bytecode instance of Shout
    BytecodeRa,
    /// One-hot ra/wa polynomial for the RAM instance of Twist
    /// Note that for RAM, ra and wa are the same polynomial because
    /// there is at most one load or store per cycle.
    RamRa,
    /// Inc polynomial for the registers instance of Twist
    RdInc,
    /// Inc polynomial for the RAM instance of Twist
    RamInc,
    /// One-hot ra polynomial for the instruction lookups instance of Shout.
    /// There are four (d=4) of these polynomials, `InstructionRa(0) .. InstructionRa(3)`
    InstructionRa(usize),
}

pub const ALL_COMMITTED_POLYNOMIALS: [CommittedPolynomials; 6] = [
    CommittedPolynomials::LeftInstructionInput,
    CommittedPolynomials::RightInstructionInput,
    CommittedPolynomials::Product,
    CommittedPolynomials::WriteLookupOutputToRD,
    CommittedPolynomials::WritePCtoRD,
    CommittedPolynomials::ShouldBranch,
    // CommittedPolynomials::BytecodeRa,
    // CommittedPolynomials::RamRa,
    // CommittedPolynomials::RdInc,
    // CommittedPolynomials::RamInc,
    // CommittedPolynomials::InstructionRa(0),
    // CommittedPolynomials::InstructionRa(1),
    // CommittedPolynomials::InstructionRa(2),
    // CommittedPolynomials::InstructionRa(3),
];

impl CommittedPolynomials {
    pub fn len() -> usize {
        ALL_COMMITTED_POLYNOMIALS.len()
    }

    pub fn from_index(index: usize) -> Self {
        ALL_COMMITTED_POLYNOMIALS[index]
    }

    pub fn generate_witness<F, PCS, ProofTranscript>(
        &self,
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
        trace: &[RV32IMCycle],
    ) -> MultilinearPolynomial<F>
    where
        F: JoltField,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        match self {
            CommittedPolynomials::LeftInstructionInput => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| LookupQuery::<32>::to_instruction_inputs(cycle).0)
                    .collect();
                coeffs.into()
            }
            CommittedPolynomials::RightInstructionInput => {
                let coeffs: Vec<i64> = trace
                    .par_iter()
                    .map(|cycle| LookupQuery::<32>::to_instruction_inputs(cycle).1)
                    .collect();
                coeffs.into()
            }
            CommittedPolynomials::Product => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| {
                        let (left_input, right_input) =
                            LookupQuery::<32>::to_instruction_inputs(cycle);
                        left_input * right_input as u64
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomials::WriteLookupOutputToRD => {
                let coeffs: Vec<u8> = trace
                    .par_iter()
                    .map(|cycle| {
                        let flag = cycle.instruction().circuit_flags()
                            [CircuitFlags::WriteLookupOutputToRD as usize];
                        (cycle.rd_write().0 as u8) * (flag as u8)
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomials::WritePCtoRD => {
                let coeffs: Vec<u8> = trace
                    .par_iter()
                    .map(|cycle| {
                        let flag = cycle.instruction().circuit_flags()[CircuitFlags::Jump as usize];
                        (cycle.rd_write().0 as u8) * (flag as u8)
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomials::ShouldBranch => {
                let coeffs: Vec<u8> = trace
                    .par_iter()
                    .map(|cycle| {
                        let is_branch =
                            cycle.instruction().circuit_flags()[CircuitFlags::Branch as usize];
                        (LookupQuery::<32>::to_lookup_output(cycle) as u8) * is_branch as u8
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomials::BytecodeRa => {
                let addresses: Vec<usize> = trace
                    .par_iter()
                    .map(|cycle| {
                        let instr = cycle.instruction().normalize();
                        let k = preprocessing
                            .shared
                            .bytecode
                            .virtual_address_map
                            .get(&(instr.address, instr.virtual_sequence_remaining.unwrap_or(0)))
                            .unwrap();
                        *k
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    preprocessing.shared.bytecode.code_size,
                ))
            }
            CommittedPolynomials::RamRa => {
                let addresses: Vec<usize> = trace
                    .par_iter()
                    .map(|cycle| {
                        let ram_op = cycle.ram_access();
                        match ram_op {
                            tracer::instruction::RAMAccess::Read(read) => {
                                remap_address(read.address, &preprocessing.shared.memory_layout)
                                    as usize
                            }
                            tracer::instruction::RAMAccess::Write(write) => {
                                remap_address(write.address, &preprocessing.shared.memory_layout)
                                    as usize
                            }
                            tracer::instruction::RAMAccess::NoOp => 0,
                        }
                    })
                    .collect();
                let K = addresses.par_iter().max().unwrap().next_power_of_two();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(addresses, K))
            }
            CommittedPolynomials::RdInc => {
                todo!()
                // let increments: Vec<(usize, i64)> = trace
                //     .par_iter()
                //     .map(|cycle| {
                //         let (k, pre_value, post_value) = cycle.rd_write();
                //         let increment = post_value as i64 - pre_value as i64;
                //         (k, increment)
                //     })
                //     .collect();
                // MultilinearPolynomial::OneHot(OneHotPolynomial::from_increments(increments))
            }
            CommittedPolynomials::RamInc => {
                todo!()
                // let increments: Vec<(usize, i64)> = trace
                //     .par_iter()
                //     .map(|cycle| {
                //         let ram_op = cycle.ram_access();
                //         match ram_op {
                //             tracer::instruction::RAMAccess::Read(read) => {
                //                 let k = remap_address(
                //                     read.address,
                //                     &preprocessing.shared.memory_layout,
                //                 ) as usize;
                //                 (k, 0)
                //             }
                //             tracer::instruction::RAMAccess::Write(write) => {
                //                 let k = remap_address(
                //                     write.address,
                //                     &preprocessing.shared.memory_layout,
                //                 ) as usize;
                //                 let increment = write.post_value as i64 - write.pre_value as i64;
                //                 (k, increment)
                //             }
                //             tracer::instruction::RAMAccess::NoOp => (0, 0),
                //         }
                //     })
                //     .collect();
                // MultilinearPolynomial::OneHot(OneHotPolynomial::from_increments(increments))
            }
            CommittedPolynomials::InstructionRa(i) => {
                if *i > 3 {
                    panic!("Unexpected i: {i}");
                }
                let addresses: Vec<usize> = trace
                    .par_iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<32>::to_lookup_index(cycle);
                        let k = (lookup_index >> (16 * (3 - i))) % (1 << 16);
                        k as usize
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(addresses, 1 << 16))
            }
        }
    }
}
