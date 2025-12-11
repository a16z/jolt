#![allow(static_mut_refs)]

use allocative::Allocative;
use common::constants::XLEN;
use common::jolt_device::MemoryLayout;
use rayon::prelude::*;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::sync::Arc;
use tracer::instruction::Cycle;

use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::commitment_scheme::StreamingCommitmentScheme;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::config::OneHotParams;
use crate::zkvm::instruction::InstructionFlags;
use crate::zkvm::prover::JoltProverPreprocessing;
use crate::{
    field::JoltField,
    poly::{multilinear_polynomial::MultilinearPolynomial, one_hot_polynomial::OneHotPolynomial},
    zkvm::ram::remap_address,
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
    fn new(trace_len: usize, one_hot_params: &OneHotParams) -> Self {
        Self {
            left_instruction_input: vec![0; trace_len],
            right_instruction_input: vec![0; trace_len],
            rd_inc: vec![0; trace_len],
            ram_inc: vec![0; trace_len],

            instruction_ra: (0..one_hot_params.instruction_d)
                .map(|_| vec![None; trace_len])
                .collect(),
            bytecode_ra: (0..one_hot_params.bytecode_d)
                .map(|_| vec![None; trace_len])
                .collect(),
            ram_ra: (0..one_hot_params.ram_d)
                .map(|_| vec![None; trace_len])
                .collect(),
        }
    }
}

/// Returns a list of symbols representing all committed polynomials.
pub fn all_committed_polynomials(one_hot_params: &OneHotParams) -> Vec<CommittedPolynomial> {
    let mut polynomials = vec![CommittedPolynomial::RdInc, CommittedPolynomial::RamInc];
    for i in 0..one_hot_params.instruction_d {
        polynomials.push(CommittedPolynomial::InstructionRa(i));
    }
    for i in 0..one_hot_params.ram_d {
        polynomials.push(CommittedPolynomial::RamRa(i));
    }
    for i in 0..one_hot_params.bytecode_d {
        polynomials.push(CommittedPolynomial::BytecodeRa(i));
    }
    polynomials
}

impl CommittedPolynomial {
    /// Generate witness data and compute tier 1 commitment for a single row
    pub fn stream_witness_and_commit_rows<F, PCS>(
        &self,
        setup: &PCS::ProverSetup,
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        row_cycles: &[tracer::instruction::Cycle],
        one_hot_params: &OneHotParams,
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
                        Some(one_hot_params.lookup_index_chunk(lookup_index, *idx) as usize)
                    })
                    .collect();
                PCS::process_chunk_onehot(setup, one_hot_params.k_chunk, &row)
            }
            CommittedPolynomial::BytecodeRa(idx) => {
                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| {
                        let pc = preprocessing.bytecode.get_pc(cycle);
                        Some(one_hot_params.bytecode_pc_chunk(pc, *idx) as usize)
                    })
                    .collect();
                PCS::process_chunk_onehot(setup, one_hot_params.k_chunk, &row)
            }
            CommittedPolynomial::RamRa(idx) => {
                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| {
                        remap_address(
                            cycle.ram_access().address() as u64,
                            &preprocessing.memory_layout,
                        )
                        .map(|address| one_hot_params.ram_address_chunk(address, *idx) as usize)
                    })
                    .collect();
                PCS::process_chunk_onehot(setup, one_hot_params.k_chunk, &row)
            }
        }
    }

    #[tracing::instrument(skip_all, name = "CommittedPolynomial::generate_witness_batch")]
    pub fn generate_witness_batch<F, PCS>(
        polynomials: &[CommittedPolynomial],
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        trace: &[Cycle],
        one_hot_params: &OneHotParams,
    ) -> HashMap<CommittedPolynomial, MultilinearPolynomial<F>>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
    {
        // let one_hot_num_bits = if ram_d > 0 { Some(log_chunk) } else { None };
        let batch = WitnessData::new(trace.len(), one_hot_params);
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
                for j in 0..one_hot_params.instruction_d {
                    let k = one_hot_params.lookup_index_chunk(lookup_index, j);
                    batch_ref.instruction_ra[j][i] = Some(k);
                }

                // BytecodeRa indices
                let pc = preprocessing.bytecode.get_pc(cycle);

                for j in 0..one_hot_params.bytecode_d {
                    let pc = one_hot_params.bytecode_pc_chunk(pc, j);
                    batch_ref.bytecode_ra[j][i] = Some(pc);
                }

                // RamRa indices
                let address = remap_address(
                    cycle.ram_access().address() as u64,
                    &preprocessing.memory_layout,
                );

                for j in 0..one_hot_params.ram_d {
                    let index = address.map(|address| one_hot_params.ram_address_chunk(address, j));
                    batch_ref.ram_ra[j][i] = index;
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
                            OneHotPolynomial::from_indices(indices, one_hot_params.k_chunk);
                        results.insert(*poly, MultilinearPolynomial::OneHot(one_hot));
                    }
                }
                CommittedPolynomial::BytecodeRa(i) => {
                    if *i < batch.bytecode_ra.len() {
                        let indices = std::mem::take(&mut batch.bytecode_ra[*i]);
                        let one_hot =
                            OneHotPolynomial::from_indices(indices, one_hot_params.k_chunk);
                        results.insert(*poly, MultilinearPolynomial::OneHot(one_hot));
                    }
                }
                CommittedPolynomial::RamRa(i) => {
                    if *i < batch.ram_ra.len() {
                        let indices = std::mem::take(&mut batch.ram_ra[*i]);
                        let one_hot =
                            OneHotPolynomial::from_indices(indices, one_hot_params.k_chunk);
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
        one_hot_params: Option<&OneHotParams>,
    ) -> MultilinearPolynomial<F>
    where
        F: JoltField,
    {
        match self {
            CommittedPolynomial::BytecodeRa(i) => {
                let one_hot_params = one_hot_params.unwrap();
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        let pc = bytecode_preprocessing.get_pc(cycle);
                        Some(one_hot_params.bytecode_pc_chunk(pc, *i))
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_params.k_chunk,
                ))
            }
            CommittedPolynomial::RamRa(i) => {
                let one_hot_params = one_hot_params.unwrap();
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        remap_address(cycle.ram_access().address() as u64, memory_layout)
                            .map(|address| one_hot_params.ram_address_chunk(address, *i))
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_params.k_chunk,
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
                let one_hot_params = one_hot_params.unwrap();
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                        Some(one_hot_params.lookup_index_chunk(lookup_index, *i))
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_params.k_chunk,
                ))
            }
        }
    }

    pub fn get_onehot_k(&self, one_hot_params: &OneHotParams) -> Option<usize> {
        match self {
            CommittedPolynomial::InstructionRa(_)
            | CommittedPolynomial::BytecodeRa(_)
            | CommittedPolynomial::RamRa(_) => Some(one_hot_params.k_chunk),
            _ => None,
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
    InstructionRa(usize),
    RegistersVal,
    RamAddress,
    RamRa,
    RamReadValue,
    RamWriteValue,
    RamVal,
    RamValInit,
    RamValFinal,
    RamHammingWeight,
    UnivariateSkip,
    OpFlags(CircuitFlags),
    InstructionFlags(InstructionFlags),
    LookupTableFlag(usize),
}
