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

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::{CommitmentScheme, StreamingCommitmentScheme, StreamingProcessChunk},
        compact_polynomial::StreamingCompactWitness,
        multilinear_polynomial::{Multilinear, MultilinearPolynomial, StreamingWitness},
        one_hot_polynomial::{OneHotPolynomial, StreamingOneHotWitness},
    },
    utils::math::Math,
    zkvm::{
        instruction_lookups, lookup_table::LookupTables, ram::remap_address,
        JoltProverPreprocessing,
    },
};
use ark_ff::biginteger::S128;

use super::instruction::{CircuitFlags, InstructionFlags, LookupQuery};

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

// Types of witness polynomials.
struct LeftInstructionInput;
struct RightInstructionInput;
struct Product;
struct WriteLookupOutputToRD;
struct WritePCtoRD;
struct ShouldBranch;
struct ShouldJump;
struct RdInc;
struct RamInc;
struct InstructionRa(usize);
struct BytecodeRa(usize);
struct RamRa(usize);

trait StreamWitness<F: JoltField> {
    type WitnessType;

    fn generate_streaming_witness<'a, PCS>(
        &self,
        preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        cycle: &RV32IMCycle,
        next_cycle: &RV32IMCycle,
    ) -> Self::WitnessType
    where
        PCS: CommitmentScheme<Field = F>;
}

impl<F: JoltField> StreamWitness<F> for InstructionRa {
    type WitnessType = StreamingOneHotWitness<F>;

    fn generate_streaming_witness<'a, PCS>(
        &self,
        _preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        cycle: &RV32IMCycle,
        _next_cycle: &RV32IMCycle,
    ) -> Self::WitnessType
    where
        PCS: CommitmentScheme<Field = F>
    {
        let i = self.0;
        debug_assert!(i < instruction_lookups::D, "Invalid index for instruction ra: {i}");
        let v = {
            let lookup_index = LookupQuery::<32>::to_lookup_index(cycle);
            let k = (lookup_index
                >> (instruction_lookups::LOG_K_CHUNK * (instruction_lookups::D - 1 - i)))
                % instruction_lookups::K_CHUNK as u64;
            k as usize
        };

        StreamingOneHotWitness::new(Some(v))
    }
}

impl<F: JoltField> StreamWitness<F> for LeftInstructionInput {
    type WitnessType = StreamingCompactWitness<u64, F>;

    fn generate_streaming_witness<'a, PCS>(
        &self,
        _preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        cycle: &RV32IMCycle,
        _next_cycle: &RV32IMCycle,
    ) -> Self::WitnessType
    where PCS: CommitmentScheme<Field = F>
    {
        let v = LookupQuery::<32>::to_instruction_inputs(cycle).0;
        StreamingCompactWitness::new(v)
    }
}

impl<F: JoltField> StreamWitness<F> for RightInstructionInput {
    type WitnessType = StreamingCompactWitness<i64, F>;

    fn generate_streaming_witness<'a, PCS>(
        &self,
        _preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        cycle: &RV32IMCycle,
        _next_cycle: &RV32IMCycle,
    ) -> Self::WitnessType
    where PCS: CommitmentScheme<Field = F> {
        let v = LookupQuery::<32>::to_instruction_inputs(cycle).1;
        StreamingCompactWitness::new(v)
    }
}

impl<F: JoltField> StreamWitness<F> for Product {
    type WitnessType = StreamingCompactWitness<u64, F>;

    fn generate_streaming_witness<'a, PCS>(
        &self,
        _preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        cycle: &RV32IMCycle,
        _next_cycle: &RV32IMCycle,
    ) -> Self::WitnessType
    where PCS: CommitmentScheme<Field = F> {
        let v = {
            let (left_input, right_input) = LookupQuery::<32>::to_instruction_inputs(cycle);
            left_input * right_input as u64
        };
        StreamingCompactWitness::new(v)
    }
}

impl<F: JoltField> StreamWitness<F> for WriteLookupOutputToRD {
    type WitnessType = StreamingCompactWitness<u8, F>;

    fn generate_streaming_witness<'a, PCS>(
        &self,
        _preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        cycle: &RV32IMCycle,
        _next_cycle: &RV32IMCycle,
    ) -> Self::WitnessType
    where PCS: CommitmentScheme<Field = F> {
        let v = {
            let flag = cycle.instruction().circuit_flags()
                [CircuitFlags::WriteLookupOutputToRD as usize];
            (cycle.rd_write().0 as u8) * (flag as u8)
        };
        StreamingCompactWitness::new(v)
    }
}

impl<F: JoltField> StreamWitness<F> for WritePCtoRD {
    type WitnessType = StreamingCompactWitness<u8, F>;

    fn generate_streaming_witness<'a, PCS>(
        &self,
        _preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        cycle: &RV32IMCycle,
        _next_cycle: &RV32IMCycle,
    ) -> Self::WitnessType
    where PCS: CommitmentScheme<Field = F> {
        let v = {
            let flag = cycle.instruction().circuit_flags()[CircuitFlags::Jump as usize];
            (cycle.rd_write().0 as u8) * (flag as u8)
        };
        StreamingCompactWitness::new(v)
    }
}

impl<F: JoltField> StreamWitness<F> for ShouldBranch {
    type WitnessType = StreamingCompactWitness<u8, F>;

    fn generate_streaming_witness<'a, PCS>(
        &self,
        _preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        cycle: &RV32IMCycle,
        _next_cycle: &RV32IMCycle,
    ) -> Self::WitnessType
    where PCS: CommitmentScheme<Field = F> {
        let v = {
            let is_branch =
                cycle.instruction().circuit_flags()[CircuitFlags::Branch as usize];
            (LookupQuery::<32>::to_lookup_output(cycle) as u8) * is_branch as u8
        };
        StreamingCompactWitness::new(v)
    }
}

impl<F: JoltField> StreamWitness<F> for ShouldJump {
    type WitnessType = StreamingCompactWitness<u8, F>;

    fn generate_streaming_witness<'a, PCS>(
        &self,
        _preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        cycle: &RV32IMCycle,
        next_cycle: &RV32IMCycle,
    ) -> Self::WitnessType
    where PCS: CommitmentScheme<Field = F> {
        let v = {
            let is_jump = cycle.instruction().circuit_flags()[CircuitFlags::Jump];
            let is_next_noop =
                next_cycle.instruction().circuit_flags()[CircuitFlags::IsNoop];
            is_jump as u8 * (1 - is_next_noop as u8)
        };
        StreamingCompactWitness::new(v)
    }
}

impl<F: JoltField> StreamWitness<F> for RdInc {
    type WitnessType = StreamingCompactWitness<i64, F>;

    fn generate_streaming_witness<'a, PCS>(
        &self,
        _preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        cycle: &RV32IMCycle,
        _next_cycle: &RV32IMCycle,
    ) -> Self::WitnessType
    where PCS: CommitmentScheme<Field = F> {
        let v = {
            let (_, pre_value, post_value) = cycle.rd_write();
            post_value as i64 - pre_value as i64
        };
        StreamingCompactWitness::new(v)
    }
}

impl<F: JoltField> StreamWitness<F> for RamInc {
    type WitnessType = StreamingCompactWitness<i64, F>;

    fn generate_streaming_witness<'a, PCS>(
        &self,
        _preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        cycle: &RV32IMCycle,
        _next_cycle: &RV32IMCycle,
    ) -> Self::WitnessType
    where PCS: CommitmentScheme<Field = F> {
        let v = {
            let ram_op = cycle.ram_access();
            match ram_op {
                tracer::instruction::RAMAccess::Write(write) => {
                    write.post_value as i64 - write.pre_value as i64
                }
                _ => 0,
            }
        };
        StreamingCompactWitness::new(v)
    }
}

impl<F: JoltField> StreamWitness<F> for BytecodeRa {
    type WitnessType = StreamingOneHotWitness<F>;

    fn generate_streaming_witness<'a, PCS>(
        &self,
        preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        cycle: &RV32IMCycle,
        _next_cycle: &RV32IMCycle,
    ) -> Self::WitnessType
    where PCS: CommitmentScheme<Field = F> {
        let i = self.0;
        // TODO: Compute this up front?
        let d = preprocessing.shared.bytecode.d;
        let log_K = preprocessing.shared.bytecode.code_size.log_2();
        let log_K_chunk = log_K.div_ceil(d);
        let K_chunk = 1 << log_K_chunk;
        debug_assert!(i < d, "Invalid index for bytecode ra: {i}");
        let v = {
                let pc = preprocessing.shared.bytecode.get_pc(cycle);
                (pc >> (log_K_chunk * (d - 1 - i))) % K_chunk
        };
        StreamingOneHotWitness::new(Some(v))
    }
}

impl<F: JoltField> StreamWitness<F> for RamRa {
    type WitnessType = StreamingOneHotWitness<F>;

    fn generate_streaming_witness<'a, PCS>(
        &self,
        preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        cycle: &RV32IMCycle,
        _next_cycle: &RV32IMCycle,
    ) -> Self::WitnessType
    where PCS: CommitmentScheme<Field = F> {
        let i = self.0;
        // TODO: Compute this up front?
        let d = preprocessing.ram_d;
        debug_assert!(i < d, "Invalid index for ram ra: {i}");
        let v = {
            remap_address(
                cycle.ram_access().address() as u64,
                &preprocessing.shared.memory_layout,
            )
            .map(|address| {
                (address as usize >> (DTH_ROOT_OF_K.log_2() * (d - 1 - i)))
                    % DTH_ROOT_OF_K
            })
        };

        StreamingOneHotWitness::new(v)
    }
}

pub static mut ALL_COMMITTED_POLYNOMIALS: OnceCell<Vec<CommittedPolynomial>> = OnceCell::new();

struct WitnessData {
    // Simple polynomial coefficients
    left_instruction_input: Vec<u64>,
    right_instruction_input: Vec<i128>,
    product: Vec<S128>,
    write_lookup_output_to_rd: Vec<u8>,
    write_pc_to_rd: Vec<u8>,
    should_branch: Vec<u8>,
    should_jump: Vec<u8>,
    rd_inc: Vec<i128>,
    ram_inc: Vec<i128>,

    // One-hot polynomial indices
    instruction_ra: [Vec<Option<usize>>; instruction_lookups::D],
    bytecode_ra: Vec<Vec<Option<usize>>>,
    ram_ra: Vec<Vec<Option<usize>>>,
}

unsafe impl Send for WitnessData {}
unsafe impl Sync for WitnessData {}

impl WitnessData {
    fn new(trace_len: usize, ram_d: usize, bytecode_d: usize) -> Self {
        Self {
            left_instruction_input: vec![0; trace_len],
            right_instruction_input: vec![0; trace_len],
            product: vec![S128::zero(); trace_len],
            write_lookup_output_to_rd: vec![0; trace_len],
            write_pc_to_rd: vec![0; trace_len],
            should_branch: vec![0; trace_len],
            should_jump: vec![0; trace_len],
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

    pub fn ram_d() -> usize {
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

    #[tracing::instrument(skip_all, name = "CommittedPolynomial::generate_witness_batch")]
    pub fn generate_witness_batch<F, PCS>(
        polynomials: &[CommittedPolynomial],
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        trace: &[Cycle],
    ) -> std::collections::HashMap<CommittedPolynomial, MultilinearPolynomial<F>>
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

        // Precompute constants per cycle
        let bytecode_constants = if bytecode_d > 0 {
            let d = preprocessing.shared.bytecode.d;
            let log_K = preprocessing.shared.bytecode.code_size.log_2();
            let log_K_chunk = log_K.div_ceil(d);
            let K_chunk = 1 << log_K_chunk;
            Some((d, log_K_chunk, K_chunk))
        } else {
            None
        };

        let dth_root_log = if ram_d > 0 {
            Some(DTH_ROOT_OF_K.log_2())
        } else {
            None
        };

        let instruction_ra_shifts: [usize; instruction_lookups::D] = std::array::from_fn(|i| {
            instruction_lookups::LOG_K_CHUNK * (instruction_lookups::D - 1 - i)
        });
        let batch_cell = Arc::new(SharedWitnessData(UnsafeCell::new(batch)));

        // #SAFETY: Each thread writes to a unique index of a pre-allocated vector
        (0..trace.len()).into_par_iter().for_each({
            let batch_cell = batch_cell.clone();
            move |i| {
                let cycle = &trace[i];
                let batch_ref = unsafe { &mut *batch_cell.0.get() };
                let (left, right) = LookupQuery::<XLEN>::to_instruction_inputs(cycle);
                let circuit_flags = cycle.instruction().circuit_flags();
                let (rd_write_flag, pre_rd, post_rd) = cycle.rd_write();
                let lookup_output = LookupQuery::<XLEN>::to_lookup_output(cycle);

                batch_ref.left_instruction_input[i] = left;
                batch_ref.right_instruction_input[i] = right;
                batch_ref.product[i] = if right >= 0 {
                    S128::from_u128(left as u128 * right.unsigned_abs())
                } else {
                    S128::from_u128_and_sign(left as u128 * right.unsigned_abs(), false)
                };

                batch_ref.write_lookup_output_to_rd[i] = rd_write_flag
                    * (circuit_flags[CircuitFlags::WriteLookupOutputToRD as usize] as u8);

                batch_ref.write_pc_to_rd[i] =
                    rd_write_flag * (circuit_flags[CircuitFlags::Jump as usize] as u8);

                batch_ref.should_branch[i] =
                    (lookup_output as u8) * (circuit_flags[CircuitFlags::Branch as usize] as u8);

                // Handle should_jump
                let is_jump = circuit_flags[CircuitFlags::Jump] as u8;
                let is_next_noop = if i + 1 < trace.len() {
                    trace[i + 1].instruction().circuit_flags()[CircuitFlags::IsNoop] as u8
                } else {
                    1 // Last cycle, treat as if next is NoOp
                };
                batch_ref.should_jump[i] = is_jump * (1 - is_next_noop);

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
                    batch_ref.instruction_ra[j][i] = Some(k as usize);
                }

                // BytecodeRa indices
                if let Some((d, log_K_chunk, K_chunk)) = bytecode_constants {
                    let pc = preprocessing.shared.bytecode.get_pc(cycle);

                    for j in 0..bytecode_d {
                        let index = (pc >> (log_K_chunk * (d - 1 - j))) % K_chunk;
                        batch_ref.bytecode_ra[j][i] = Some(index);
                    }
                }

                // RamRa indices
                if let Some(dth_log) = dth_root_log {
                    let address_opt = remap_address(
                        cycle.ram_access().address() as u64,
                        &preprocessing.shared.memory_layout,
                    );

                    for j in 0..ram_d {
                        let index = address_opt.map(|address| {
                            (address as usize >> (dth_log * (ram_d - 1 - j))) % DTH_ROOT_OF_K
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
                CommittedPolynomial::LeftInstructionInput => {
                    let coeffs = std::mem::take(&mut batch.left_instruction_input);
                    results.insert(*poly, MultilinearPolynomial::<F>::from(coeffs));
                }
                CommittedPolynomial::RightInstructionInput => {
                    let coeffs = std::mem::take(&mut batch.right_instruction_input);
                    results.insert(*poly, MultilinearPolynomial::<F>::from(coeffs));
                }
                CommittedPolynomial::Product => {
                    let coeffs = std::mem::take(&mut batch.product);
                    results.insert(*poly, MultilinearPolynomial::<F>::from(coeffs));
                }
                CommittedPolynomial::WriteLookupOutputToRD => {
                    let coeffs = std::mem::take(&mut batch.write_lookup_output_to_rd);
                    results.insert(*poly, MultilinearPolynomial::<F>::from(coeffs));
                }
                CommittedPolynomial::WritePCtoRD => {
                    let coeffs = std::mem::take(&mut batch.write_pc_to_rd);
                    results.insert(*poly, MultilinearPolynomial::<F>::from(coeffs));
                }
                CommittedPolynomial::ShouldBranch => {
                    let coeffs = std::mem::take(&mut batch.should_branch);
                    results.insert(*poly, MultilinearPolynomial::<F>::from(coeffs));
                }
                CommittedPolynomial::ShouldJump => {
                    let coeffs = std::mem::take(&mut batch.should_jump);
                    results.insert(*poly, MultilinearPolynomial::<F>::from(coeffs));
                }
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
                        let d = preprocessing.shared.bytecode.d;
                        let log_K = preprocessing.shared.bytecode.code_size.log_2();
                        let log_K_chunk = log_K.div_ceil(d);
                        let K_chunk = 1 << log_K_chunk;
                        let one_hot = OneHotPolynomial::from_indices(indices, K_chunk);
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
    ) -> MultilinearPolynomial<F>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
    {
        match self {
            CommittedPolynomial::LeftInstructionInput => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| LookupQuery::<XLEN>::to_instruction_inputs(cycle).0)
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::RightInstructionInput => {
                let coeffs: Vec<i128> = trace
                    .par_iter()
                    .map(|cycle| LookupQuery::<XLEN>::to_instruction_inputs(cycle).1)
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::Product => {
                let coeffs: Vec<S128> = trace
                    .par_iter()
                    .map(|cycle| {
                        let (left_input, right_input) =
                            LookupQuery::<XLEN>::to_instruction_inputs(cycle);
                        // Use the fact that `|right_input|` fits in u64 to avoid overflow
                        if right_input >= 0 {
                            S128::from_u128(left_input as u128 * right_input.unsigned_abs())
                        } else {
                            S128::from_u128_and_sign(
                                left_input as u128 * right_input.unsigned_abs(),
                                false,
                            )
                        }
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
                        (LookupQuery::<XLEN>::to_lookup_output(cycle) as u8) * is_branch as u8
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
                            .chain(rayon::iter::once(&Cycle::NoOp)),
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
                let d = preprocessing.ram_d;
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

    pub fn to_polynomial_type(
        &self,
    ) -> Multilinear {
        match self {
            CommittedPolynomial::LeftInstructionInput => Multilinear::U64Scalars,
            CommittedPolynomial::RightInstructionInput => Multilinear::I64Scalars,
            CommittedPolynomial::Product => Multilinear::U64Scalars,
            CommittedPolynomial::WriteLookupOutputToRD => Multilinear::U8Scalars,
            CommittedPolynomial::WritePCtoRD => Multilinear::U8Scalars,
            CommittedPolynomial::ShouldBranch => Multilinear::U8Scalars,
            CommittedPolynomial::ShouldJump => Multilinear::U8Scalars,
            CommittedPolynomial::RdInc => Multilinear::I64Scalars,
            CommittedPolynomial::RamInc => Multilinear::I64Scalars,
            CommittedPolynomial::InstructionRa(_) => Multilinear::OneHot,
            CommittedPolynomial::BytecodeRa(_) => Multilinear::OneHot,
            CommittedPolynomial::RamRa(_) => Multilinear::OneHot,
        }
    }

    // TODO: Make this more amenable to parallelization.
    pub fn generate_witness_and_commit_row<'a, F: JoltField, PCS: StreamingCommitmentScheme<Field = F>>(
        &self,
        pcs: PCS::State<'a>,
        preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        row_cycles: &[(RV32IMCycle, RV32IMCycle)], // impl Iterator<Item = (RV32IMCycle, RV32IMCycle)>
    ) -> PCS::State<'a> {
        #[inline(always)]
        fn helper<'a, T: StreamWitness<F>, F: JoltField, PCS: StreamingCommitmentScheme<Field = F>>(
            witness_type: T,
            pcs: PCS::State<'a>,
            preprocessing: &'a JoltProverPreprocessing<F, PCS>,
            row_cycles: &[(RV32IMCycle, RV32IMCycle)], // impl Iterator<Item = (RV32IMCycle, RV32IMCycle)>
        ) -> PCS::State<'a>
        where
            PCS::State<'a>: StreamingProcessChunk<T::WitnessType>,
        {
            let row: Vec<_> = row_cycles
                .iter()
                .map(|(cycle, next_cycle)| witness_type.generate_streaming_witness(preprocessing, &cycle, &next_cycle))
                .collect();
            PCS::process_chunk(pcs, &row)
        }
        match self {
            CommittedPolynomial::LeftInstructionInput => {
                helper(LeftInstructionInput, pcs, preprocessing, row_cycles)
            }
            CommittedPolynomial::RightInstructionInput => {
                helper(RightInstructionInput, pcs, preprocessing, row_cycles)
            }
            CommittedPolynomial::Product => {
                helper(Product, pcs, preprocessing, row_cycles)
            }
            CommittedPolynomial::WriteLookupOutputToRD => {
                helper(WriteLookupOutputToRD, pcs, preprocessing, row_cycles)
            }
            CommittedPolynomial::WritePCtoRD => {
                helper(WritePCtoRD, pcs, preprocessing, row_cycles)
            }
            CommittedPolynomial::ShouldBranch => {
                helper(ShouldBranch, pcs, preprocessing, row_cycles)
            }
            CommittedPolynomial::ShouldJump => {
                helper(ShouldJump, pcs, preprocessing, row_cycles)
            }
            CommittedPolynomial::RdInc => {
                helper(RdInc, pcs, preprocessing, row_cycles)
            }
            CommittedPolynomial::RamInc => {
                helper(RamInc, pcs, preprocessing, row_cycles)
            }
            CommittedPolynomial::InstructionRa(i) => {
                helper(InstructionRa(*i), pcs, preprocessing, row_cycles)
            }
            CommittedPolynomial::BytecodeRa(i) => {
                helper(BytecodeRa(*i), pcs, preprocessing, row_cycles)
            }
            CommittedPolynomial::RamRa(i) => {
                helper(RamRa(*i), pcs, preprocessing, row_cycles)
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
