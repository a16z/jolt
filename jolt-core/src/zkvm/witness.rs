#![allow(static_mut_refs)]

use allocative::Allocative;
use common::constants::XLEN;
use common::jolt_device::MemoryLayout;
use itertools::Itertools;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::LazyLock;
use strum::IntoEnumIterator;
use tracer::instruction::Cycle;

use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::commitment_scheme::StreamingCommitmentScheme;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::config;
use crate::zkvm::instruction::InstructionFlags;
use crate::zkvm::prover::JoltProverPreprocessing;
use crate::{
    field::JoltField,
    poly::{multilinear_polynomial::MultilinearPolynomial, one_hot_polynomial::OneHotPolynomial},
    zkvm::{lookup_table::LookupTables, ram::remap_address},
};

use super::instruction::{CircuitFlags, LookupQuery};

struct SharedWitnessData(UnsafeCell<WitnessData>);
unsafe impl Sync for SharedWitnessData {}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum CommittedPolynomial {
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

struct WitnessData {
    // Simple polynomial coefficients
    left_instruction_input: Vec<u64>,
    right_instruction_input: Vec<i128>,
    rd_inc: Vec<i128>,
    ram_inc: Vec<i128>,

    // One-hot polynomial indices
    instruction_ra: Vec<Vec<Option<u16>>>,
    bytecode_ra: Vec<Vec<Option<u16>>>,
    ram_ra: Vec<Vec<Option<u16>>>,
}

unsafe impl Send for WitnessData {}
unsafe impl Sync for WitnessData {}

impl WitnessData {
    fn new(trace_len: usize, instruction_d: usize, ram_d: usize, bytecode_d: usize) -> Self {
        Self {
            left_instruction_input: vec![0; trace_len],
            right_instruction_input: vec![0; trace_len],
            rd_inc: vec![0; trace_len],
            ram_inc: vec![0; trace_len],

            instruction_ra: (0..instruction_d).map(|_| vec![None; trace_len]).collect(),
            bytecode_ra: (0..bytecode_d).map(|_| vec![None; trace_len]).collect(),
            ram_ra: (0..ram_d).map(|_| vec![None; trace_len]).collect(),
        }
    }
}

pub struct AllCommittedPolynomials();
impl AllCommittedPolynomials {
    pub fn initialize(ram_K: usize, bytecode_d: usize) -> Self {
        let ram_d = Self::ram_d_from_K(ram_K);
        unsafe {
            if let Some(existing) = ALL_COMMITTED_POLYNOMIALS.get() {
                // Check if existing polynomials match requested dimensions
                let existing_ram_d = existing
                    .iter()
                    .filter(|p| matches!(p, CommittedPolynomial::RamRa(_)))
                    .count();
                let existing_bytecode_d = existing
                    .iter()
                    .filter(|p| matches!(p, CommittedPolynomial::BytecodeRa(_)))
                    .count();

                let existing_instruction_d = existing
                    .iter()
                    .filter(|p| matches!(p, CommittedPolynomial::InstructionRa(_)))
                    .count();

                if existing_instruction_d == config::params().instruction.d
                    && existing_ram_d == ram_d
                    && existing_bytecode_d == bytecode_d
                {
                    // Parameters match, reuse existing polynomials
                    return AllCommittedPolynomials();
                } else {
                    // Parameters differ, need to reinitialize
                    ALL_COMMITTED_POLYNOMIALS.take();
                }
            }
        };
        let instruction_d = config::params().instruction.d;

        let mut polynomials = vec![CommittedPolynomial::RdInc, CommittedPolynomial::RamInc];
        for i in 0..instruction_d {
            polynomials.push(CommittedPolynomial::InstructionRa(i));
        }
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

    pub fn len() -> usize {
        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .get()
                .expect("ALL_COMMITTED_POLYNOMIALS is uninitialized")
                .len()
        }
    }
    pub fn ram_d_from_K(ram_K: usize) -> usize {
        config::params().one_hot.compute_d(ram_K)
    }
}

impl CommittedPolynomial {
    /// Generate witness data and compute tier 1 commitment for a single row
    pub fn stream_witness_and_commit_rows<F, PCS>(
        &self,
        setup: &PCS::ProverSetup,
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        row_cycles: &[tracer::instruction::Cycle],
        ram_d: usize,
    ) -> <PCS as StreamingCommitmentScheme>::ChunkState
    where
        F: JoltField,
        PCS: StreamingCommitmentScheme<Field = F>,
    {
        match self {
            CommittedPolynomial::RdInc => {
                let row: Vec<i128> = row_cycles
                    .iter()
                    .map(|cycle| {
                        let (_, pre_value, post_value) = cycle.rd_write();
                        post_value as i128 - pre_value as i128
                    })
                    .collect();
                PCS::process_chunk(setup, &row)
            }
            CommittedPolynomial::RamInc => {
                let row: Vec<i128> = row_cycles
                    .iter()
                    .map(|cycle| match cycle.ram_access() {
                        tracer::instruction::RAMAccess::Write(write) => {
                            write.post_value as i128 - write.pre_value as i128
                        }
                        _ => 0,
                    })
                    .collect();
                PCS::process_chunk(setup, &row)
            }
            CommittedPolynomial::InstructionRa(idx) => {
                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                        let k = (lookup_index
                            >> (config::params().instruction.log_k_chunk
                                * (config::params().instruction.d - 1 - idx)))
                            % config::params().instruction.k_chunk as u128;
                        Some(k as usize)
                    })
                    .collect();
                PCS::process_chunk_onehot(setup, config::params().instruction.k_chunk, &row)
            }
            CommittedPolynomial::BytecodeRa(idx) => {
                let params = &config::params().bytecode;
                let d = preprocessing.bytecode.d;

                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| {
                        let pc = preprocessing.bytecode.get_pc(cycle);
                        Some((pc >> (params.log_chunk * (d - 1 - idx))) % params.chunk_size)
                    })
                    .collect();
                PCS::process_chunk_onehot(setup, params.chunk_size, &row)
            }
            CommittedPolynomial::RamRa(idx) => {
                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| {
                        remap_address(
                            cycle.ram_access().address() as u64,
                            &preprocessing.memory_layout,
                        )
                        .map(|address| {
                            (address as usize
                                >> (config::params().ram.log_chunk * (ram_d - 1 - idx)))
                                % config::params().ram.chunk_size
                        })
                    })
                    .collect();
                PCS::process_chunk_onehot(setup, config::params().ram.chunk_size, &row)
            }
        }
    }

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

    #[tracing::instrument(skip_all, name = "CommittedPolynomial::generate_witness_batch")]
    pub fn generate_witness_batch<F, PCS>(
        polynomials: &[CommittedPolynomial],
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        trace: &[Cycle],
    ) -> HashMap<CommittedPolynomial, MultilinearPolynomial<F>>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
    {
        let mut ram_d = 0;
        let mut bytecode_d = 0;

        for poly in polynomials {
            match poly {
                CommittedPolynomial::BytecodeRa(i) => {
                    bytecode_d = bytecode_d.max(*i + 1);
                }
                CommittedPolynomial::RamRa(i) => {
                    ram_d = ram_d.max(*i + 1);
                }
                _ => {}
            }
        }
        let params = config::params();
        let one_hot = &params.one_hot;
        let one_hot_num_bits = if ram_d > 0 {
            Some(one_hot.log_chunk)
        } else {
            None
        };
        let instruction_params = &params.instruction;
        let instruction_d = instruction_params.d;
        let batch = WitnessData::new(trace.len(), instruction_d, ram_d, bytecode_d);
        let instruction_ra_shifts: Vec<usize> = (0..instruction_d)
            .map(|i| instruction_params.log_k_chunk * (instruction_d - 1 - i))
            .collect();
        let batch_cell = Arc::new(SharedWitnessData(UnsafeCell::new(batch)));

        // #SAFETY: Each thread writes to a unique index of a pre-allocated vector
        (0..trace.len()).into_par_iter().for_each({
            let batch_cell = batch_cell.clone();
            move |i| {
                let cycle = &trace[i];
                let batch_ref = unsafe { &mut *batch_cell.0.get() };
                let (left, right) = LookupQuery::<XLEN>::to_instruction_inputs(cycle);
                let (_, pre_rd, post_rd) = cycle.rd_write();

                batch_ref.left_instruction_input[i] = left;
                batch_ref.right_instruction_input[i] = right;

                batch_ref.rd_inc[i] = post_rd as i128 - pre_rd as i128;

                // RAM inc
                let ram_inc = match cycle.ram_access() {
                    tracer::instruction::RAMAccess::Write(write) => {
                        write.post_value as i128 - write.pre_value as i128
                    }
                    _ => 0,
                };
                batch_ref.ram_inc[i] = ram_inc;

                // InstructionRa indices
                let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                for (j, shift) in instruction_ra_shifts.iter().enumerate() {
                    let k = (lookup_index >> shift) % instruction_params.k_chunk as u128;
                    batch_ref.instruction_ra[j][i] = Some(k as u16);
                }

                // BytecodeRa indices
                if let Some(dth_root_log) = one_hot_num_bits {
                    let pc = preprocessing.bytecode.get_pc(cycle);

                    for j in 0..bytecode_d {
                        let index =
                            (pc >> (dth_root_log * (bytecode_d - 1 - j))) % one_hot.chunk_size;
                        batch_ref.bytecode_ra[j][i] = Some(index as u16);
                    }
                }

                // RamRa indices
                if let Some(dth_log) = one_hot_num_bits {
                    let address_opt = remap_address(
                        cycle.ram_access().address() as u64,
                        &preprocessing.memory_layout,
                    );

                    for j in 0..ram_d {
                        let index = address_opt.map(|address| {
                            ((address as usize >> (dth_log * (ram_d - 1 - j))) % one_hot.chunk_size)
                                as u16
                        });
                        batch_ref.ram_ra[j][i] = index;
                    }
                }
            }
        });

        let mut batch = Arc::try_unwrap(batch_cell)
            .ok()
            .expect("Arc should have single owner")
            .0
            .into_inner();

        // We zero-cost move the data back
        let mut results = HashMap::with_capacity(polynomials.len());

        for poly in polynomials {
            match poly {
                CommittedPolynomial::RdInc => {
                    let coeffs = std::mem::take(&mut batch.rd_inc);
                    results.insert(*poly, MultilinearPolynomial::<F>::from(coeffs));
                }
                CommittedPolynomial::RamInc => {
                    let coeffs = std::mem::take(&mut batch.ram_inc);
                    results.insert(*poly, MultilinearPolynomial::<F>::from(coeffs));
                }
                CommittedPolynomial::InstructionRa(i) => {
                    if *i < batch.instruction_ra.len() {
                        let indices = std::mem::take(&mut batch.instruction_ra[*i]);
                        let one_hot =
                            OneHotPolynomial::from_indices(indices, instruction_params.k_chunk);
                        results.insert(*poly, MultilinearPolynomial::OneHot(one_hot));
                    }
                }
                CommittedPolynomial::BytecodeRa(i) => {
                    if *i < bytecode_d {
                        let indices = std::mem::take(&mut batch.bytecode_ra[*i]);
                        let one_hot = OneHotPolynomial::from_indices(indices, one_hot.chunk_size);
                        results.insert(*poly, MultilinearPolynomial::OneHot(one_hot));
                    }
                }
                CommittedPolynomial::RamRa(i) => {
                    if *i < ram_d {
                        let indices = std::mem::take(&mut batch.ram_ra[*i]);
                        let one_hot = OneHotPolynomial::from_indices(indices, one_hot.chunk_size);
                        results.insert(*poly, MultilinearPolynomial::OneHot(one_hot));
                    }
                }
            }
        }
        results
    }

    #[tracing::instrument(skip_all, name = "CommittedPolynomial::generate_witness")]
    pub fn generate_witness<F>(
        &self,
        bytecode_preprocessing: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
        trace: &[Cycle],
        ram_d: usize,
    ) -> MultilinearPolynomial<F>
    where
        F: JoltField,
    {
        let params = config::params();
        let one_hot = &params.one_hot;

        match self {
            CommittedPolynomial::BytecodeRa(i) => {
                let d = bytecode_preprocessing.d;
                if *i > d {
                    panic!("Invalid index for bytecode ra: {i}");
                }
                let one_hot_length = one_hot.chunk_size;
                let one_hot_num_bits = one_hot.log_chunk;
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        let pc = bytecode_preprocessing.get_pc(cycle);
                        Some(((pc >> (one_hot_num_bits * (d - 1 - i))) % one_hot_length) as u16)
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_length,
                ))
            }
            CommittedPolynomial::RamRa(i) => {
                let d = ram_d;

                debug_assert!(*i < d);
                let one_hot_length = one_hot.chunk_size;
                let one_hot_num_bits = one_hot.log_chunk;
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        remap_address(cycle.ram_access().address() as u64, memory_layout)
                            .map(|address| {
                                ((address as usize
                                    >> (config::params().ram.log_chunk * (d - 1 - i)))
                                    % config::params().ram.chunk_size)
                                    as u8
                            })
                            .map(|address| {
                                ((address as usize >> (one_hot_num_bits * (d - 1 - i)))
                                    % one_hot_length) as u16
                            })
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_length,
                ))
            }
            CommittedPolynomial::RdInc => {
                let coeffs: Vec<i128> = trace
                    .par_iter()
                    .map(|cycle| {
                        let (_, pre_value, post_value) = cycle.rd_write();
                        post_value as i128 - pre_value as i128
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::RamInc => {
                let coeffs: Vec<i128> = trace
                    .par_iter()
                    .map(|cycle| {
                        let ram_op = cycle.ram_access();
                        match ram_op {
                            tracer::instruction::RAMAccess::Write(write) => {
                                write.post_value as i128 - write.pre_value as i128
                            }
                            _ => 0,
                        }
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::InstructionRa(i) => {
                let instruction_params = &params.instruction;
                let instruction_d = instruction_params.d;
                let idx = *i;
                if idx >= instruction_d {
                    panic!("Unexpected i: {i}");
                }
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                        let shift = instruction_params.log_k_chunk * (instruction_d - 1 - idx);
                        let k = (lookup_index >> shift) % instruction_params.k_chunk as u128;
                        Some(k as u16)
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    instruction_params.k_chunk,
                ))
            }
        }
    }

    pub fn get_onehot_k<F, PCS>(
        &self,
        _preprocessing: &JoltProverPreprocessing<F, PCS>,
    ) -> Option<usize>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
    {
        let params = config::params();
        let one_hot = &params.one_hot;

        match self {
            CommittedPolynomial::InstructionRa(_) => Some(params.instruction.k_chunk),
            CommittedPolynomial::BytecodeRa(_) => Some(params.bytecode.chunk_size),
            CommittedPolynomial::RamRa(_) => Some(one_hot.chunk_size),
            _ => None,
        }
    }

    pub fn generate_witness_and_commit_row<F: JoltField, PCS>(
        &self,
        prover_setup: &PCS::ProverSetup,
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        row_cycles: &[Cycle],
        ram_d: usize,
    ) -> PCS::ChunkState
    where
        PCS: StreamingCommitmentScheme<Field = F>,
    {
        let params = config::params();
        let one_hot = &params.one_hot;

        match self {
            CommittedPolynomial::RdInc => {
                let row: Vec<i128> = row_cycles
                    .iter()
                    .map(|cycle| {
                        let (_, pre_value, post_value) = cycle.rd_write();
                        post_value as i128 - pre_value as i128
                    })
                    .collect();
                PCS::process_chunk(prover_setup, &row)
            }
            CommittedPolynomial::RamInc => {
                let row: Vec<i128> = row_cycles
                    .iter()
                    .map(|cycle| {
                        let ram_op = cycle.ram_access();
                        match ram_op {
                            tracer::instruction::RAMAccess::Write(write) => {
                                write.post_value as i128 - write.pre_value as i128
                            }
                            _ => 0,
                        }
                    })
                    .collect();
                PCS::process_chunk(prover_setup, &row)
            }
            CommittedPolynomial::InstructionRa(idx) => {
                let instruction_params = &params.instruction;
                let instruction_d = instruction_params.d;
                debug_assert!(*idx < instruction_d);
                let idx = *idx;
                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                        let shift = instruction_params.log_k_chunk * (instruction_d - 1 - idx);
                        let k = (lookup_index >> shift) % instruction_params.k_chunk as u128;
                        Some(k as usize)
                    })
                    .collect();
                PCS::process_chunk_onehot(prover_setup, instruction_params.k_chunk, &row)
            }
            CommittedPolynomial::BytecodeRa(idx) => {
                let d = preprocessing.bytecode.d;
                let params = &config::params().bytecode;

                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| {
                        let pc = preprocessing.bytecode.get_pc(cycle);
                        Some((pc >> (params.log_chunk * (d - 1 - idx))) % params.chunk_size)
                    })
                    .collect();
                PCS::process_chunk_onehot(prover_setup, params.chunk_size, &row)
            }
            CommittedPolynomial::RamRa(idx) => {
                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| {
                        remap_address(
                            cycle.ram_access().address() as u64,
                            &preprocessing.memory_layout,
                        )
                        .map(|address| {
                            (address as usize >> (one_hot.log_chunk * (ram_d - 1 - idx)))
                                % one_hot.chunk_size
                        })
                    })
                    .collect();
                PCS::process_chunk_onehot(prover_setup, one_hot.chunk_size, &row)
            }
        }
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum VirtualPolynomial {
    PC,
    UnexpandedPC,
    NextPC,
    NextUnexpandedPC,
    NextIsNoop,
    NextIsVirtual,
    NextIsFirstInSequence,
    LeftLookupOperand,
    RightLookupOperand,
    LeftInstructionInput,
    RightInstructionInput,
    Product,
    ShouldJump,
    ShouldBranch,
    WritePCtoRD,
    WriteLookupOutputToRD,
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
    InstructionRa,
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
    InstructionFlags(InstructionFlags),
    LookupTableFlag(usize),
}

pub static ALL_VIRTUAL_POLYNOMIALS: LazyLock<Vec<VirtualPolynomial>> = LazyLock::new(|| {
    let mut polynomials = vec![
        VirtualPolynomial::PC,
        VirtualPolynomial::UnexpandedPC,
        VirtualPolynomial::NextPC,
        VirtualPolynomial::NextUnexpandedPC,
        VirtualPolynomial::NextIsNoop,
        VirtualPolynomial::NextIsVirtual,
        VirtualPolynomial::NextIsFirstInSequence,
        VirtualPolynomial::LeftLookupOperand,
        VirtualPolynomial::RightLookupOperand,
        VirtualPolynomial::LeftInstructionInput,
        VirtualPolynomial::RightInstructionInput,
        VirtualPolynomial::Product,
        VirtualPolynomial::ShouldJump,
        VirtualPolynomial::ShouldBranch,
        VirtualPolynomial::WritePCtoRD,
        VirtualPolynomial::WriteLookupOutputToRD,
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
        VirtualPolynomial::InstructionRa,
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
    for flag in InstructionFlags::iter() {
        polynomials.push(VirtualPolynomial::InstructionFlags(flag));
    }
    for table in LookupTables::iter() {
        polynomials.push(VirtualPolynomial::LookupTableFlag(
            LookupTables::<XLEN>::enum_index(&table),
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
