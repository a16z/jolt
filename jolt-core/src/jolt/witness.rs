use itertools::Itertools;
use rayon::prelude::*;
use tracer::instruction::RV32IMCycle;

use crate::{
    field::JoltField,
    jolt::vm::{instruction_lookups, ram::remap_address, JoltProverPreprocessing},
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial, one_hot_polynomial::OneHotPolynomial,
    },
    utils::math::Math,
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
    /// Whether the current instruction triggers a jump
    ShouldJump,
    /*  Twist/Shout witnesses */
    /// One-hot ra polynomial for the bytecode instance of Shout
    BytecodeRa(usize),
    /// One-hot ra/wa polynomial for the RAM instance of Twist
    /// Note that for RAM, ra and wa are the same polynomial because
    /// there is at most one load or store per cycle.
    /// d = 1 right now hence we only ever use RamRa(0) for now.
    RamRa(usize),
    /// Inc polynomial for the registers instance of Twist
    RdInc,
    /// Inc polynomial for the RAM instance of Twist
    RamInc,
    /// One-hot ra polynomial for the instruction lookups instance of Shout.
    /// There are four (d=4) of these polynomials, `InstructionRa(0) .. InstructionRa(3)`
    InstructionRa(usize),
}

pub const ALL_COMMITTED_POLYNOMIALS: [CommittedPolynomials; 20] = [
    CommittedPolynomials::LeftInstructionInput,
    CommittedPolynomials::RightInstructionInput,
    CommittedPolynomials::Product,
    CommittedPolynomials::WriteLookupOutputToRD,
    CommittedPolynomials::WritePCtoRD,
    CommittedPolynomials::ShouldBranch,
    CommittedPolynomials::ShouldJump,
    CommittedPolynomials::BytecodeRa(0),
    CommittedPolynomials::BytecodeRa(1),
    CommittedPolynomials::RamRa(0),
    CommittedPolynomials::RdInc,
    CommittedPolynomials::RamInc,
    CommittedPolynomials::InstructionRa(0),
    CommittedPolynomials::InstructionRa(1),
    CommittedPolynomials::InstructionRa(2),
    CommittedPolynomials::InstructionRa(3),
    CommittedPolynomials::InstructionRa(4),
    CommittedPolynomials::InstructionRa(5),
    CommittedPolynomials::InstructionRa(6),
    CommittedPolynomials::InstructionRa(7),
];

impl CommittedPolynomials {
    pub fn len() -> usize {
        ALL_COMMITTED_POLYNOMIALS.len()
    }

    pub fn from_index(index: usize) -> Self {
        ALL_COMMITTED_POLYNOMIALS[index]
    }

    pub fn to_index(&self) -> usize {
        ALL_COMMITTED_POLYNOMIALS
            .iter()
            .find_position(|poly| *poly == self)
            .unwrap()
            .0
    }

    pub fn generate_witness<F, PCS>(
        &self,
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        trace: &[RV32IMCycle],
    ) -> MultilinearPolynomial<F>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
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
            CommittedPolynomials::ShouldJump => {
                let coeffs: Vec<u8> = trace
                    .par_iter()
                    .zip(
                        trace
                            .par_iter()
                            .skip(1)
                            .chain(rayon::iter::once(&RV32IMCycle::NoOp)),
                    )
                    .map(|(cycle, next_cycle)| {
                        let is_jump = cycle.instruction().circuit_flags()[CircuitFlags::Jump];
                        let is_next_noop =
                            next_cycle.instruction().circuit_flags()[CircuitFlags::IsNoop];
                        is_jump as u8 * (1 - is_next_noop as u8)
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomials::BytecodeRa(i) => {
                let d = preprocessing.shared.bytecode.d;
                let log_K = preprocessing.shared.bytecode.code_size.log_2();
                let log_K_chunk = log_K.div_ceil(d);
                let K_chunk = 1 << log_K_chunk;
                if *i > d {
                    panic!("Invalid index for bytecode ra: {i}");
                }
                let addresses: Vec<usize> = trace
                    .par_iter()
                    .map(|cycle| {
                        let pc = preprocessing.shared.bytecode.get_pc(cycle);
                        (pc >> (log_K_chunk * (d - 1 - i))) % K_chunk
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(addresses, K_chunk))
            }
            // TODO(markosg04) logic here needs to be adjusted for when d > 1 is implemented
            CommittedPolynomials::RamRa(i) => {
                if *i > 0 {
                    panic!("RAM is implemented for only d=1 currently.");
                }
                let addresses: Vec<usize> = trace
                    .par_iter()
                    .map(|cycle| {
                        remap_address(
                            cycle.ram_access().address() as u64,
                            &preprocessing.shared.memory_layout,
                        ) as usize
                    })
                    .collect();
                let K = addresses.par_iter().max().unwrap().next_power_of_two();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(addresses, K))
            }
            CommittedPolynomials::RdInc => {
                let coeffs: Vec<i64> = trace
                    .par_iter()
                    .map(|cycle| {
                        let (_, pre_value, post_value) = cycle.rd_write();
                        post_value as i64 - pre_value as i64
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomials::RamInc => {
                let coeffs: Vec<i64> = trace
                    .par_iter()
                    .map(|cycle| {
                        let ram_op = cycle.ram_access();
                        match ram_op {
                            tracer::instruction::RAMAccess::Write(write) => {
                                write.post_value as i64 - write.pre_value as i64
                            }
                            _ => 0,
                        }
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomials::InstructionRa(i) => {
                if *i > instruction_lookups::D {
                    panic!("Unexpected i: {i}");
                }
                let addresses: Vec<usize> = trace
                    .par_iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<32>::to_lookup_index(cycle);
                        let k = (lookup_index
                            >> (instruction_lookups::LOG_K_CHUNK
                                * (instruction_lookups::D - 1 - i)))
                            % instruction_lookups::K_CHUNK as u64;
                        k as usize
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    instruction_lookups::K_CHUNK,
                ))
            }
        }
    }
}
