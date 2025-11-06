#![allow(static_mut_refs)]

use allocative::Allocative;
use common::constants::XLEN;
use itertools::Itertools;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use std::array;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::LazyLock;
use strum::IntoEnumIterator;
use tracer::instruction::Cycle;

use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::commitment_scheme::StreamingCommitmentScheme;
use crate::zkvm::instruction::InstructionFlags;
use crate::{
    field::JoltField,
    poly::{multilinear_polynomial::MultilinearPolynomial, one_hot_polynomial::OneHotPolynomial},
    utils::math::Math,
    zkvm::{
        instruction_lookups, lookup_table::LookupTables, ram::remap_address,
        JoltProverPreprocessing,
    },
};

use super::instruction::{CircuitFlags, LookupQuery};

struct SharedWitnessData(UnsafeCell<WitnessData>);
unsafe impl Sync for SharedWitnessData {}

/// K^{1/d}
pub const DTH_ROOT_OF_K: usize = 1 << 8;

pub fn compute_d_parameter_from_log_K(log_K: usize) -> usize {
    log_K.div_ceil(DTH_ROOT_OF_K.log_2())
}

pub fn compute_d_parameter(K: usize) -> usize {
    // Calculate D dynamically such that 2^8 = K^(1/D)
    let log_K = K.log_2();
    log_K.div_ceil(DTH_ROOT_OF_K.log_2())
}

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
    instruction_ra: [Vec<Option<u8>>; instruction_lookups::D],
    bytecode_ra: Vec<Vec<Option<u8>>>,
    ram_ra: Vec<Vec<Option<u8>>>,
}

unsafe impl Send for WitnessData {}
unsafe impl Sync for WitnessData {}

impl WitnessData {
    fn new(trace_len: usize, ram_d: usize, bytecode_d: usize) -> Self {
        Self {
            left_instruction_input: vec![0; trace_len],
            right_instruction_input: vec![0; trace_len],
            rd_inc: vec![0; trace_len],
            ram_inc: vec![0; trace_len],

            instruction_ra: array::from_fn(|_| vec![None; trace_len]),
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

                if existing_ram_d == ram_d && existing_bytecode_d == bytecode_d {
                    // Parameters match, reuse existing polynomials
                    return AllCommittedPolynomials();
                } else {
                    // Parameters differ, need to reinitialize
                    ALL_COMMITTED_POLYNOMIALS.take();
                }
            }
        };
        let mut polynomials = vec![
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
            CommittedPolynomial::InstructionRa(8),
            CommittedPolynomial::InstructionRa(9),
            CommittedPolynomial::InstructionRa(10),
            CommittedPolynomial::InstructionRa(11),
            CommittedPolynomial::InstructionRa(12),
            CommittedPolynomial::InstructionRa(13),
            CommittedPolynomial::InstructionRa(14),
            CommittedPolynomial::InstructionRa(15),
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

    pub fn len() -> usize {
        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .get()
                .expect("ALL_COMMITTED_POLYNOMIALS is uninitialized")
                .len()
        }
    }
    pub fn ram_d_from_K(ram_K: usize) -> usize {
        compute_d_parameter(ram_K)
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
        let batch = WitnessData::new(trace.len(), ram_d, bytecode_d);

        let dth_root_log = if ram_d > 0 {
            Some(DTH_ROOT_OF_K.log_2())
        } else {
            None
        };

        let instruction_ra_shifts: [usize; instruction_lookups::D] =
            array::from_fn(|i| instruction_lookups::LOG_K_CHUNK * (instruction_lookups::D - 1 - i));
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
                for j in 0..instruction_lookups::D {
                    let k = (lookup_index >> instruction_ra_shifts[j])
                        % instruction_lookups::K_CHUNK as u128;
                    batch_ref.instruction_ra[j][i] = Some(k as u8);
                }

                // BytecodeRa indices
                if let Some(dth_root_log) = dth_root_log {
                    let pc = preprocessing.bytecode.get_pc(cycle);

                    for j in 0..bytecode_d {
                        let index = (pc >> (dth_root_log * (bytecode_d - 1 - j))) % DTH_ROOT_OF_K;
                        batch_ref.bytecode_ra[j][i] = Some(index as u8);
                    }
                }

                // RamRa indices
                if let Some(dth_log) = dth_root_log {
                    let address_opt = remap_address(
                        cycle.ram_access().address() as u64,
                        &preprocessing.memory_layout,
                    );

                    for j in 0..ram_d {
                        let index = address_opt.map(|address| {
                            ((address as usize >> (dth_log * (ram_d - 1 - j))) % DTH_ROOT_OF_K)
                                as u8
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
                    if *i < instruction_lookups::D {
                        let indices = std::mem::take(&mut batch.instruction_ra[*i]);
                        let one_hot =
                            OneHotPolynomial::from_indices(indices, instruction_lookups::K_CHUNK);
                        results.insert(*poly, MultilinearPolynomial::OneHot(one_hot));
                    }
                }
                CommittedPolynomial::BytecodeRa(i) => {
                    if *i < bytecode_d {
                        let indices = std::mem::take(&mut batch.bytecode_ra[*i]);
                        let one_hot = OneHotPolynomial::from_indices(indices, DTH_ROOT_OF_K);
                        results.insert(*poly, MultilinearPolynomial::OneHot(one_hot));
                    }
                }
                CommittedPolynomial::RamRa(i) => {
                    if *i < ram_d {
                        let indices = std::mem::take(&mut batch.ram_ra[*i]);
                        let one_hot = OneHotPolynomial::from_indices(indices, DTH_ROOT_OF_K);
                        results.insert(*poly, MultilinearPolynomial::OneHot(one_hot));
                    }
                }
            }
        }
        results
    }

    #[tracing::instrument(skip_all, name = "CommittedPolynomial::generate_witness")]
    pub fn generate_witness<F, PCS>(
        &self,
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        trace: &[Cycle],
        ram_d: usize,
    ) -> MultilinearPolynomial<F>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
    {
        match self {
            CommittedPolynomial::BytecodeRa(i) => {
                let d = preprocessing.bytecode.d;
                if *i > d {
                    panic!("Invalid index for bytecode ra: {i}");
                }
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        let pc = preprocessing.bytecode.get_pc(cycle);
                        Some(((pc >> (DTH_ROOT_OF_K.log_2() * (d - 1 - i))) % DTH_ROOT_OF_K) as u8)
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    DTH_ROOT_OF_K,
                ))
            }
            CommittedPolynomial::RamRa(i) => {
                let d = ram_d;

                debug_assert!(*i < d);
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        remap_address(
                            cycle.ram_access().address() as u64,
                            &preprocessing.memory_layout,
                        )
                        .map(|address| {
                            ((address as usize >> (DTH_ROOT_OF_K.log_2() * (d - 1 - i)))
                                % DTH_ROOT_OF_K) as u8
                        })
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    DTH_ROOT_OF_K,
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
                if *i > instruction_lookups::D {
                    panic!("Unexpected i: {i}");
                }
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                        let k = (lookup_index
                            >> (instruction_lookups::LOG_K_CHUNK
                                * (instruction_lookups::D - 1 - i)))
                            % instruction_lookups::K_CHUNK as u128;
                        Some(k as u8)
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    instruction_lookups::K_CHUNK,
                ))
            }
        }
    }

    pub fn get_onehot_k<F, PCS>(
        &self,
        preprocessing: &JoltProverPreprocessing<F, PCS>,
    ) -> Option<usize>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
    {
        match self {
            CommittedPolynomial::InstructionRa(_) => Some(instruction_lookups::K_CHUNK),
            CommittedPolynomial::BytecodeRa(_) => {
                // TODO: Compute this up front?
                let d = preprocessing.bytecode.d;
                let log_K = preprocessing.bytecode.code_size.log_2();
                let log_K_chunk = log_K.div_ceil(d);
                let K_chunk = 1 << log_K_chunk;
                Some(K_chunk)
            }
            CommittedPolynomial::RamRa(_) => Some(DTH_ROOT_OF_K),
            _ => None,
        }
    }

    pub fn generate_witness_and_commit_row<F: JoltField, PCS>(
        &self,
        cached_data: &PCS::CachedData,
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        row_cycles: &[Cycle],
        ram_d: usize,
    ) -> PCS::ChunkState
    where
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
                PCS::process_chunk(cached_data, &row)
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
                PCS::process_chunk(cached_data, &row)
            }
            CommittedPolynomial::InstructionRa(idx) => {
                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                        let k = (lookup_index
                            >> (instruction_lookups::LOG_K_CHUNK
                                * (instruction_lookups::D - 1 - idx)))
                            % instruction_lookups::K_CHUNK as u128;
                        Some(k as usize)
                    })
                    .collect();
                PCS::process_chunk_onehot(cached_data, instruction_lookups::K_CHUNK, &row)
            }
            CommittedPolynomial::BytecodeRa(idx) => {
                let d = preprocessing.bytecode.d;
                let log_K = preprocessing.bytecode.code_size.log_2();
                let log_K_chunk = log_K.div_ceil(d);
                let K_chunk = 1 << log_K_chunk;

                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| {
                        let pc = preprocessing.bytecode.get_pc(cycle);
                        Some((pc >> (log_K_chunk * (d - 1 - idx))) % K_chunk)
                    })
                    .collect();
                PCS::process_chunk_onehot(cached_data, K_chunk, &row)
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
                            (address as usize >> (DTH_ROOT_OF_K.log_2() * (ram_d - 1 - idx)))
                                % DTH_ROOT_OF_K
                        })
                    })
                    .collect();
                PCS::process_chunk_onehot(cached_data, DTH_ROOT_OF_K, &row)
            }
        }
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum VirtualPolynomial {
    SpartanAz,
    SpartanBz,
    FusedProductLeft,
    FusedProductRight,
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
        VirtualPolynomial::SpartanAz,
        VirtualPolynomial::SpartanBz,
        VirtualPolynomial::FusedProductLeft,
        VirtualPolynomial::FusedProductRight,
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
