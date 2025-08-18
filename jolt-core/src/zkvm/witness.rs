#![allow(static_mut_refs)]

use std::sync::LazyLock;

use allocative::Allocative;
use itertools::Itertools;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use strum::IntoEnumIterator;
use tracer::instruction::RV32IMCycle;

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial, one_hot_polynomial::OneHotPolynomial,
    },
    utils::math::Math,
    zkvm::{
        lookup_table::LookupTables,
        {instruction_lookups, ram::remap_address, JoltProverPreprocessing},
    },
};

use super::instruction::{CircuitFlags, InstructionFlags, LookupQuery};

/// K^{1/d}
pub const DTH_ROOT_OF_K: usize = 1 << 8;
pub fn compute_d_parameter(K: usize) -> usize {
    // Calculate D dynamically such that 2^8 = K^(1/D)
    let log_K = K.log_2();
    log_K.div_ceil(DTH_ROOT_OF_K.log_2())
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum CommittedPolynomial {
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
    /// Inc polynomial for the registers instance of Twist
    RdInc,
    /// Inc polynomial for the RAM instance of Twist
    RamInc,
    /// One-hot ra polynomial for the instruction lookups instance of Shout.
    /// There are d=8 of these polynomials, `InstructionRa(0) .. InstructionRa(7)`
    InstructionRa(usize),
    /// One-hot ra polynomial for the bytecode instance of Shout
    BytecodeRa(usize),
    /// One-hot ra/wa polynomial for the RAM instance of Twist
    /// Note that for RAM, ra and wa are the same polynomial because
    /// there is at most one load or store per cycle.
    RamRa(usize),
}

pub static mut ALL_COMMITTED_POLYNOMIALS: OnceCell<Vec<CommittedPolynomial>> = OnceCell::new();

pub struct AllCommittedPolynomials();
impl AllCommittedPolynomials {
    pub fn initialize(ram_d: usize, bytecode_d: usize) -> Self {
        let mut polynomials = vec![
            CommittedPolynomial::LeftInstructionInput,
            CommittedPolynomial::RightInstructionInput,
            CommittedPolynomial::Product,
            CommittedPolynomial::WriteLookupOutputToRD,
            CommittedPolynomial::WritePCtoRD,
            CommittedPolynomial::ShouldBranch,
            CommittedPolynomial::ShouldJump,
            CommittedPolynomial::RdInc,
            CommittedPolynomial::RamInc,
            CommittedPolynomial::InstructionRa(0),
            CommittedPolynomial::InstructionRa(1),
            CommittedPolynomial::InstructionRa(2),
            CommittedPolynomial::InstructionRa(3),
            CommittedPolynomial::InstructionRa(4),
            CommittedPolynomial::InstructionRa(5),
            CommittedPolynomial::InstructionRa(6),
            CommittedPolynomial::InstructionRa(7),
        ];
        for i in 0..ram_d {
            polynomials.push(CommittedPolynomial::RamRa(i));
        }
        for i in 0..bytecode_d {
            polynomials.push(CommittedPolynomial::BytecodeRa(i));
        }

        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .set(polynomials)
                .expect("ALL_COMMITTED_POLYNOMIALS is already initialized");
        }

        AllCommittedPolynomials()
    }

    pub fn iter() -> impl Iterator<Item = &'static CommittedPolynomial> {
        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .get()
                .expect("ALL_COMMITTED_POLYNOMIALS is uninitialized")
                .iter()
        }
    }

    pub fn par_iter() -> impl ParallelIterator<Item = &'static CommittedPolynomial> {
        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .get()
                .expect("ALL_COMMITTED_POLYNOMIALS is uninitialized")
                .par_iter()
        }
    }
}

impl Drop for AllCommittedPolynomials {
    fn drop(&mut self) {
        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .take()
                .expect("ALL_COMMITTED_POLYNOMIALS is uninitialized");
        }
    }
}

impl CommittedPolynomial {
    pub fn len() -> usize {
        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .get()
                .expect("ALL_COMMITTED_POLYNOMIALS is uninitialized")
                .len()
        }
    }

    // TODO(moodlezoup): return Result<Self>
    pub fn from_index(index: usize) -> Self {
        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .get()
                .expect("ALL_COMMITTED_POLYNOMIALS is uninitialized")[index]
        }
    }

    // TODO(moodlezoup): return Result<usize>
    pub fn to_index(&self) -> usize {
        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .get()
                .expect("ALL_COMMITTED_POLYNOMIALS is uninitialized")
                .iter()
                .find_position(|poly| *poly == self)
                .unwrap()
                .0
        }
    }

    fn ram_d(&self) -> usize {
        // this is kind of jank but fine for now ig
        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .get()
                .expect("ALL_COMMITTED_POLYNOMIALS is uninitialized")
                .iter()
                .filter(|poly| matches!(poly, CommittedPolynomial::RamRa(_)))
                .count()
        }
    }

    #[tracing::instrument(skip_all, name = "CommittedPolynomial::generate_witness")]
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
            CommittedPolynomial::LeftInstructionInput => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| LookupQuery::<32>::to_instruction_inputs(cycle).0)
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::RightInstructionInput => {
                let coeffs: Vec<i64> = trace
                    .par_iter()
                    .map(|cycle| LookupQuery::<32>::to_instruction_inputs(cycle).1)
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::Product => {
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
            CommittedPolynomial::WriteLookupOutputToRD => {
                let coeffs: Vec<u8> = trace
                    .par_iter()
                    .map(|cycle| {
                        let flag = cycle.instruction().circuit_flags()
                            [CircuitFlags::WriteLookupOutputToRD as usize];
                        (cycle.rd_write().0) * (flag as u8)
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::WritePCtoRD => {
                let coeffs: Vec<u8> = trace
                    .par_iter()
                    .map(|cycle| {
                        let flag = cycle.instruction().circuit_flags()[CircuitFlags::Jump as usize];
                        (cycle.rd_write().0) * (flag as u8)
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::ShouldBranch => {
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
            CommittedPolynomial::ShouldJump => {
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
            CommittedPolynomial::BytecodeRa(i) => {
                let d = preprocessing.shared.bytecode.d;
                let log_K = preprocessing.shared.bytecode.code_size.log_2();
                let log_K_chunk = log_K.div_ceil(d);
                let K_chunk = 1 << log_K_chunk;
                if *i > d {
                    panic!("Invalid index for bytecode ra: {i}");
                }
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        let pc = preprocessing.shared.bytecode.get_pc(cycle);
                        Some((pc >> (log_K_chunk * (d - 1 - i))) % K_chunk)
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(addresses, K_chunk))
            }
            CommittedPolynomial::RamRa(i) => {
                let d = self.ram_d();
                debug_assert!(*i < d);
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        remap_address(
                            cycle.ram_access().address() as u64,
                            &preprocessing.shared.memory_layout,
                        )
                        .map(|address| {
                            (address as usize >> (DTH_ROOT_OF_K.log_2() * (d - 1 - i)))
                                % DTH_ROOT_OF_K
                        })
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    DTH_ROOT_OF_K,
                ))
            }
            CommittedPolynomial::RdInc => {
                let coeffs: Vec<i64> = trace
                    .par_iter()
                    .map(|cycle| {
                        let (_, pre_value, post_value) = cycle.rd_write();
                        post_value as i64 - pre_value as i64
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::RamInc => {
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
            CommittedPolynomial::InstructionRa(i) => {
                if *i > instruction_lookups::D {
                    panic!("Unexpected i: {i}");
                }
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<32>::to_lookup_index(cycle);
                        let k = (lookup_index
                            >> (instruction_lookups::LOG_K_CHUNK
                                * (instruction_lookups::D - 1 - i)))
                            % instruction_lookups::K_CHUNK as u64;
                        Some(k as usize)
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

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum VirtualPolynomial {
    SpartanAz,
    SpartanBz,
    SpartanCz,
    PC,
    UnexpandedPC,
    NextPC,
    NextUnexpandedPC,
    NextIsNoop,
    LeftLookupOperand,
    RightLookupOperand,
    Rd,
    Imm,
    Rs1Value,
    Rs2Value,
    RdWriteValue,
    Rs1Ra,
    Rs2Ra,
    RdWa,
    LookupOutput,
    InstructionRaf,
    InstructionRafFlag,
    RegistersVal,
    RamAddress,
    RamRa,
    RamReadValue,
    RamWriteValue,
    RamVal,
    RamValInit,
    RamValFinal,
    RamHammingWeight,
    OpFlags(CircuitFlags),
    LookupTableFlag(usize),
}

pub static ALL_VIRTUAL_POLYNOMIALS: LazyLock<Vec<VirtualPolynomial>> = LazyLock::new(|| {
    let mut polynomials = vec![
        VirtualPolynomial::SpartanAz,
        VirtualPolynomial::SpartanBz,
        VirtualPolynomial::SpartanCz,
        VirtualPolynomial::PC,
        VirtualPolynomial::UnexpandedPC,
        VirtualPolynomial::NextPC,
        VirtualPolynomial::NextUnexpandedPC,
        VirtualPolynomial::NextIsNoop,
        VirtualPolynomial::LeftLookupOperand,
        VirtualPolynomial::RightLookupOperand,
        VirtualPolynomial::Rd,
        VirtualPolynomial::Imm,
        VirtualPolynomial::Rs1Value,
        VirtualPolynomial::Rs2Value,
        VirtualPolynomial::RdWriteValue,
        VirtualPolynomial::Rs1Ra,
        VirtualPolynomial::Rs2Ra,
        VirtualPolynomial::RdWa,
        VirtualPolynomial::LookupOutput,
        VirtualPolynomial::InstructionRaf,
        VirtualPolynomial::InstructionRafFlag,
        VirtualPolynomial::RegistersVal,
        VirtualPolynomial::RamAddress,
        VirtualPolynomial::RamRa,
        VirtualPolynomial::RamReadValue,
        VirtualPolynomial::RamWriteValue,
        VirtualPolynomial::RamVal,
        VirtualPolynomial::RamValInit,
        VirtualPolynomial::RamValFinal,
        VirtualPolynomial::RamHammingWeight,
    ];
    for flag in CircuitFlags::iter() {
        polynomials.push(VirtualPolynomial::OpFlags(flag));
    }
    for table in LookupTables::iter() {
        polynomials.push(VirtualPolynomial::LookupTableFlag(
            LookupTables::<32>::enum_index(&table),
        ));
    }

    polynomials
});

impl VirtualPolynomial {
    pub fn from_index(index: usize) -> Self {
        ALL_VIRTUAL_POLYNOMIALS[index]
    }

    pub fn to_index(&self) -> usize {
        ALL_VIRTUAL_POLYNOMIALS
            .iter()
            .find_position(|poly| *poly == self)
            .unwrap()
            .0
    }
}
