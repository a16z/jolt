use rayon::prelude::*;
use tracer::instruction::RV32IMCycle;

use crate::{field::JoltField, poly::multilinear_polynomial::MultilinearPolynomial};

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
    /// The program counter for the next cycle in the trace
    NextPC,
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

pub const ALL_COMMITTED_POLYNOMIALS: [CommittedPolynomials; 7] = [
    CommittedPolynomials::LeftInstructionInput,
    CommittedPolynomials::RightInstructionInput,
    CommittedPolynomials::Product,
    CommittedPolynomials::WriteLookupOutputToRD,
    CommittedPolynomials::WritePCtoRD,
    CommittedPolynomials::ShouldBranch,
    CommittedPolynomials::NextPC,
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

    pub fn generate_witness<F>(&self, trace: &[RV32IMCycle]) -> MultilinearPolynomial<F>
    where
        F: JoltField,
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
            CommittedPolynomials::NextPC => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| {
                        let is_branch =
                            cycle.instruction().circuit_flags()[CircuitFlags::Branch as usize];
                        let should_branch =
                            is_branch && LookupQuery::<32>::to_lookup_output(cycle) != 0;
                        let instr = cycle.instruction().normalize();
                        if should_branch {
                            (instr.address as i64 + instr.operands.imm) as u64
                        } else {
                            let is_jump =
                                cycle.instruction().circuit_flags()[CircuitFlags::Jump as usize];
                            let do_not_update_pc = cycle.instruction().circuit_flags()
                                [CircuitFlags::DoNotUpdatePC as usize];
                            if is_jump {
                                LookupQuery::<32>::to_lookup_output(cycle)
                            } else if do_not_update_pc {
                                instr.address as u64
                            } else {
                                instr.address as u64 + 4
                            }
                        }
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomials::BytecodeRa => todo!(),
            CommittedPolynomials::RamRa => todo!(),
            CommittedPolynomials::RdInc => todo!(),
            CommittedPolynomials::RamInc => todo!(),
            CommittedPolynomials::InstructionRa(i) => todo!(),
        }
    }
}
