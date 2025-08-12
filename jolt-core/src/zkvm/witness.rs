#![allow(static_mut_refs)]

use std::sync::Arc;
use std::sync::LazyLock;

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

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord)]
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

    #[tracing::instrument(skip_all, name = "CommittedPolynomial::generate_witness_batch")]
    pub fn generate_witness_batch<F, PCS>(
        polynomials: &[CommittedPolynomial],
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        trace: &[RV32IMCycle],
    ) -> std::collections::HashMap<CommittedPolynomial, MultilinearPolynomial<F>>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
    {
        use std::collections::HashMap;

        let mut results = HashMap::with_capacity(polynomials.len());

        // Pre-compute common values that multiple polynomials might need
        let trace_len = trace.len();

        // Group polynomials by type to handle similar ones together
        let mut simple_polys = Vec::new();
        let mut one_hot_polys = Vec::new();

        for poly in polynomials {
            match poly {
                CommittedPolynomial::BytecodeRa(_)
                | CommittedPolynomial::RamRa(_)
                | CommittedPolynomial::InstructionRa(_) => one_hot_polys.push(*poly),
                _ => simple_polys.push(*poly),
            }
        }

        // Process simple polynomials in a single pass
        if !simple_polys.is_empty() {
            // Prepare storage for each polynomial type
            let mut left_input_coeffs = Vec::with_capacity(trace_len);
            let mut right_input_coeffs = Vec::with_capacity(trace_len);
            let mut product_coeffs = Vec::with_capacity(trace_len);
            let mut write_lookup_rd_coeffs = Vec::with_capacity(trace_len);
            let mut write_pc_rd_coeffs = Vec::with_capacity(trace_len);
            let mut should_branch_coeffs = Vec::with_capacity(trace_len);
            let mut should_jump_coeffs = Vec::with_capacity(trace_len);
            let mut rd_inc_coeffs = Vec::with_capacity(trace_len);
            let mut ram_inc_coeffs = Vec::with_capacity(trace_len);

            // Flags to track which polynomials we need to compute
            let need_left_input = simple_polys.contains(&CommittedPolynomial::LeftInstructionInput);
            let need_right_input =
                simple_polys.contains(&CommittedPolynomial::RightInstructionInput);
            let need_product = simple_polys.contains(&CommittedPolynomial::Product);
            let need_write_lookup_rd =
                simple_polys.contains(&CommittedPolynomial::WriteLookupOutputToRD);
            let need_write_pc_rd = simple_polys.contains(&CommittedPolynomial::WritePCtoRD);
            let need_should_branch = simple_polys.contains(&CommittedPolynomial::ShouldBranch);
            let need_should_jump = simple_polys.contains(&CommittedPolynomial::ShouldJump);
            let need_rd_inc = simple_polys.contains(&CommittedPolynomial::RdInc);
            let need_ram_inc = simple_polys.contains(&CommittedPolynomial::RamInc);

            // Single pass over trace
            let trace_results: Vec<(u64, i64, u64, u8, u8, u8, u8, i64, i64)> = trace
                .par_iter()
                .zip(
                    trace
                        .par_iter()
                        .skip(1)
                        .chain(rayon::iter::once(&RV32IMCycle::NoOp)),
                )
                .map(|(cycle, next_cycle)| {
                    let mut tuple = (0u64, 0i64, 0u64, 0u8, 0u8, 0u8, 0u8, 0i64, 0i64);

                    // Compute instruction inputs if needed
                    let (left_input, right_input) =
                        if need_left_input || need_right_input || need_product {
                            LookupQuery::<32>::to_instruction_inputs(cycle)
                        } else {
                            (0, 0)
                        };

                    if need_left_input {
                        tuple.0 = left_input;
                    }
                    if need_right_input {
                        tuple.1 = right_input;
                    }
                    if need_product {
                        tuple.2 = left_input * right_input as u64;
                    }

                    if need_write_lookup_rd {
                        let flag = cycle.instruction().circuit_flags()
                            [CircuitFlags::WriteLookupOutputToRD as usize];
                        tuple.3 = (cycle.rd_write().0) * (flag as u8);
                    }

                    if need_write_pc_rd {
                        let flag = cycle.instruction().circuit_flags()[CircuitFlags::Jump as usize];
                        tuple.4 = (cycle.rd_write().0) * (flag as u8);
                    }

                    if need_should_branch {
                        let is_branch =
                            cycle.instruction().circuit_flags()[CircuitFlags::Branch as usize];
                        tuple.5 =
                            (LookupQuery::<32>::to_lookup_output(cycle) as u8) * is_branch as u8;
                    }

                    if need_should_jump {
                        let is_jump = cycle.instruction().circuit_flags()[CircuitFlags::Jump];
                        let is_next_noop =
                            next_cycle.instruction().circuit_flags()[CircuitFlags::IsNoop];
                        tuple.6 = is_jump as u8 * (1 - is_next_noop as u8);
                    }

                    if need_rd_inc {
                        let (_, pre_value, post_value) = cycle.rd_write();
                        tuple.7 = post_value as i64 - pre_value as i64;
                    }

                    if need_ram_inc {
                        let ram_op = cycle.ram_access();
                        tuple.8 = match ram_op {
                            tracer::instruction::RAMAccess::Write(write) => {
                                write.post_value as i64 - write.pre_value as i64
                            }
                            _ => 0,
                        };
                    }

                    tuple
                })
                .collect();

            // Extract results into individual vectors
            if need_left_input {
                left_input_coeffs = trace_results.iter().map(|t| t.0).collect();
            }
            if need_right_input {
                right_input_coeffs = trace_results.iter().map(|t| t.1).collect();
            }
            if need_product {
                product_coeffs = trace_results.iter().map(|t| t.2).collect();
            }
            if need_write_lookup_rd {
                write_lookup_rd_coeffs = trace_results.iter().map(|t| t.3).collect();
            }
            if need_write_pc_rd {
                write_pc_rd_coeffs = trace_results.iter().map(|t| t.4).collect();
            }
            if need_should_branch {
                should_branch_coeffs = trace_results.iter().map(|t| t.5).collect();
            }
            if need_should_jump {
                should_jump_coeffs = trace_results.iter().map(|t| t.6).collect();
            }
            if need_rd_inc {
                rd_inc_coeffs = trace_results.iter().map(|t| t.7).collect();
            }
            if need_ram_inc {
                ram_inc_coeffs = trace_results.iter().map(|t| t.8).collect();
            }

            // Store results
            for poly in simple_polys {
                let multilinear = match poly {
                    CommittedPolynomial::LeftInstructionInput => {
                        MultilinearPolynomial::<F>::from(left_input_coeffs.clone())
                    }
                    CommittedPolynomial::RightInstructionInput => {
                        MultilinearPolynomial::<F>::from(right_input_coeffs.clone())
                    }
                    CommittedPolynomial::Product => {
                        MultilinearPolynomial::<F>::from(product_coeffs.clone())
                    }
                    CommittedPolynomial::WriteLookupOutputToRD => {
                        MultilinearPolynomial::<F>::from(write_lookup_rd_coeffs.clone())
                    }
                    CommittedPolynomial::WritePCtoRD => {
                        MultilinearPolynomial::<F>::from(write_pc_rd_coeffs.clone())
                    }
                    CommittedPolynomial::ShouldBranch => {
                        MultilinearPolynomial::<F>::from(should_branch_coeffs.clone())
                    }
                    CommittedPolynomial::ShouldJump => {
                        MultilinearPolynomial::<F>::from(should_jump_coeffs.clone())
                    }
                    CommittedPolynomial::RdInc => {
                        MultilinearPolynomial::<F>::from(rd_inc_coeffs.clone())
                    }
                    CommittedPolynomial::RamInc => {
                        MultilinearPolynomial::<F>::from(ram_inc_coeffs.clone())
                    }
                    _ => unreachable!(),
                };
                results.insert(poly, multilinear);
            }
        }

        // Process one-hot polynomials (these require special handling)
        for poly in one_hot_polys {
            let multilinear = poly.generate_witness(preprocessing, trace);
            results.insert(poly, multilinear);
        }

        results
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
                MultilinearPolynomial::OneHot(Arc::new(OneHotPolynomial::from_indices(
                    addresses, K_chunk,
                )))
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
                MultilinearPolynomial::OneHot(Arc::new(OneHotPolynomial::from_indices(
                    addresses,
                    DTH_ROOT_OF_K,
                )))
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
                MultilinearPolynomial::OneHot(Arc::new(OneHotPolynomial::from_indices(
                    addresses,
                    instruction_lookups::K_CHUNK,
                )))
            }
        }
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord)]
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
