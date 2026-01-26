use crate::field::JoltField;
use crate::poly::commitment::dory::{DoryGlobals, DoryLayout};
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::instruction::{
    Flags, InstructionLookup, InterleavedBitsMarker, NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS,
};
use crate::zkvm::lookup_table::LookupTables;
use common::constants::{REGISTER_COUNT, XLEN};
use rayon::prelude::*;
use tracer::instruction::Instruction;

/// Total number of "lanes" to commit bytecode fields
pub const fn total_lanes() -> usize {
    3 * (REGISTER_COUNT as usize) // rs1, rs2, rd one-hot lanes
        + 2 // unexpanded_pc, imm
        + NUM_CIRCUIT_FLAGS
        + NUM_INSTRUCTION_FLAGS
        + <LookupTables<XLEN> as strum::EnumCount>::COUNT
        + 1 // raf flag
}

/// Canonical lane layout for bytecode chunk polynomials.
///
/// The global lane order matches [`lane_value`] and the weights in
/// `claim_reductions/bytecode.rs::compute_chunk_lane_weights`.
#[derive(Clone, Copy, Debug)]
pub struct BytecodeLaneLayout {
    pub rs1_start: usize,
    pub rs2_start: usize,
    pub rd_start: usize,
    pub unexp_pc_idx: usize,
    pub imm_idx: usize,
    pub circuit_start: usize,
    pub instr_start: usize,
    pub lookup_start: usize,
    pub raf_flag_idx: usize,
}

impl BytecodeLaneLayout {
    pub const fn new() -> Self {
        let reg_count = REGISTER_COUNT as usize;
        let rs1_start = 0usize;
        let rs2_start = rs1_start + reg_count;
        let rd_start = rs2_start + reg_count;
        let unexp_pc_idx = rd_start + reg_count;
        let imm_idx = unexp_pc_idx + 1;
        let circuit_start = imm_idx + 1;
        let instr_start = circuit_start + NUM_CIRCUIT_FLAGS;
        let lookup_start = instr_start + NUM_INSTRUCTION_FLAGS;
        let raf_flag_idx = lookup_start + <LookupTables<XLEN> as strum::EnumCount>::COUNT;
        Self {
            rs1_start,
            rs2_start,
            rd_start,
            unexp_pc_idx,
            imm_idx,
            circuit_start,
            instr_start,
            lookup_start,
            raf_flag_idx,
        }
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub const fn total_lanes(&self) -> usize {
        self.raf_flag_idx + 1
    }

    /// True for all lanes except `unexpanded_pc` and `imm`.
    #[inline(always)]
    #[allow(dead_code)]
    pub const fn is_boolean_lane(&self, global_lane: usize) -> bool {
        global_lane != self.unexp_pc_idx && global_lane != self.imm_idx
    }
}

pub const BYTECODE_LANE_LAYOUT: BytecodeLaneLayout = BytecodeLaneLayout::new();

/// Active lane values for a single instruction.
///
/// Most lanes are boolean/one-hot, so we represent them as `One` to avoid
/// unnecessary field multiplications at call sites (e.g. Dory VMV).
#[derive(Clone, Copy, Debug)]
pub enum ActiveLaneValue<F: JoltField> {
    One,
    Scalar(F),
}

/// Evaluate the weighted lane sum for a single instruction:
/// \( \sum_{\ell} weights[\ell] \cdot lane\_value(\ell, instr) \),
/// without scanning all lanes (uses one-hot and boolean sparsity).
#[inline(always)]
pub fn weighted_lane_sum_for_instruction<F: JoltField>(weights: &[F], instr: &Instruction) -> F {
    debug_assert_eq!(weights.len(), total_lanes());

    let l = BYTECODE_LANE_LAYOUT;

    let normalized = instr.normalize();
    let circuit_flags = <Instruction as Flags>::circuit_flags(instr);
    let instr_flags = <Instruction as Flags>::instruction_flags(instr);
    let lookup_idx = <Instruction as InstructionLookup<XLEN>>::lookup_table(instr)
        .map(|t| LookupTables::<XLEN>::enum_index(&t));
    let raf_flag = !InterleavedBitsMarker::is_interleaved_operands(&circuit_flags);

    let unexpanded_pc = F::from_u64(normalized.address as u64);
    let imm = F::from_i128(normalized.operands.imm);
    let rs1 = normalized.operands.rs1.map(|r| r as usize);
    let rs2 = normalized.operands.rs2.map(|r| r as usize);
    let rd = normalized.operands.rd.map(|r| r as usize);

    let mut acc = F::zero();

    // One-hot register lanes: select weight at the active register (or 0 if None).
    if let Some(r) = rs1 {
        acc += weights[l.rs1_start + r];
    }
    if let Some(r) = rs2 {
        acc += weights[l.rs2_start + r];
    }
    if let Some(r) = rd {
        acc += weights[l.rd_start + r];
    }

    // Scalar lanes.
    acc += weights[l.unexp_pc_idx] * unexpanded_pc;
    acc += weights[l.imm_idx] * imm;

    // Circuit flags (boolean): add weight when flag is true.
    for i in 0..NUM_CIRCUIT_FLAGS {
        if circuit_flags[i] {
            acc += weights[l.circuit_start + i];
        }
    }

    // Instruction flags (boolean): add weight when flag is true.
    for i in 0..NUM_INSTRUCTION_FLAGS {
        if instr_flags[i] {
            acc += weights[l.instr_start + i];
        }
    }

    // Lookup table selector (one-hot / zero-hot).
    if let Some(t) = lookup_idx {
        acc += weights[l.lookup_start + t];
    }

    // RAF flag.
    if raf_flag {
        acc += weights[l.raf_flag_idx];
    }

    acc
}

/// Enumerate the non-zero lanes for a single instruction in canonical global-lane order.
///
/// This is the sparse counterpart to [`lane_value`]: instead of scanning all lanes and
/// branching on zeros, we directly visit only lanes that are 1 (for boolean/one-hot lanes)
/// or have a non-zero scalar value (for `unexpanded_pc` and `imm`).
///
/// This is useful for:
/// - Streaming / VMV computations where the downstream logic needs to map lanes to matrix indices
/// - Any place where per-lane work dominates and the instruction lane vector is sparse
#[inline(always)]
pub fn for_each_active_lane_value<F: JoltField>(
    instr: &Instruction,
    mut visit: impl FnMut(usize, ActiveLaneValue<F>),
) {
    let l = BYTECODE_LANE_LAYOUT;

    let normalized = instr.normalize();
    let circuit_flags = <Instruction as Flags>::circuit_flags(instr);
    let instr_flags = <Instruction as Flags>::instruction_flags(instr);
    let lookup_idx = <Instruction as InstructionLookup<XLEN>>::lookup_table(instr)
        .map(|t| LookupTables::<XLEN>::enum_index(&t));
    let raf_flag = !InterleavedBitsMarker::is_interleaved_operands(&circuit_flags);

    // One-hot register lanes.
    if let Some(r) = normalized.operands.rs1 {
        visit(l.rs1_start + (r as usize), ActiveLaneValue::One);
    }
    if let Some(r) = normalized.operands.rs2 {
        visit(l.rs2_start + (r as usize), ActiveLaneValue::One);
    }
    if let Some(r) = normalized.operands.rd {
        visit(l.rd_start + (r as usize), ActiveLaneValue::One);
    }

    // Scalar lanes (skip if zero).
    let unexpanded_pc = F::from_u64(normalized.address as u64);
    if !unexpanded_pc.is_zero() {
        visit(l.unexp_pc_idx, ActiveLaneValue::Scalar(unexpanded_pc));
    }
    let imm = F::from_i128(normalized.operands.imm);
    if !imm.is_zero() {
        visit(l.imm_idx, ActiveLaneValue::Scalar(imm));
    }

    // Circuit flags.
    for i in 0..NUM_CIRCUIT_FLAGS {
        if circuit_flags[i] {
            visit(l.circuit_start + i, ActiveLaneValue::One);
        }
    }

    // Instruction flags.
    for i in 0..NUM_INSTRUCTION_FLAGS {
        if instr_flags[i] {
            visit(l.instr_start + i, ActiveLaneValue::One);
        }
    }

    // Lookup selector.
    if let Some(t) = lookup_idx {
        visit(l.lookup_start + t, ActiveLaneValue::One);
    }

    // RAF flag.
    if raf_flag {
        visit(l.raf_flag_idx, ActiveLaneValue::One);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn lane_value<F: JoltField>(
    global_lane: usize,
    rs1: Option<u8>,
    rs2: Option<u8>,
    rd: Option<u8>,
    unexpanded_pc: F,
    imm: F,
    circuit_flags: &[bool; NUM_CIRCUIT_FLAGS],
    instr_flags: &[bool; NUM_INSTRUCTION_FLAGS],
    lookup_idx: Option<usize>,
    raf_flag: bool,
) -> F {
    let reg_count = REGISTER_COUNT as usize;
    let rs1_start = 0usize;
    let rs2_start = rs1_start + reg_count;
    let rd_start = rs2_start + reg_count;
    let unexp_pc_idx = rd_start + reg_count;
    let imm_idx = unexp_pc_idx + 1;
    let circuit_start = imm_idx + 1;
    let instr_start = circuit_start + NUM_CIRCUIT_FLAGS;
    let lookup_start = instr_start + NUM_INSTRUCTION_FLAGS;
    let raf_flag_idx = lookup_start + <LookupTables<XLEN> as strum::EnumCount>::COUNT;

    if global_lane < rs2_start {
        // rs1 one-hot
        let r = global_lane as u8;
        return F::from_bool(rs1 == Some(r));
    }
    if global_lane < rd_start {
        // rs2 one-hot
        let r = (global_lane - rs2_start) as u8;
        return F::from_bool(rs2 == Some(r));
    }
    if global_lane < unexp_pc_idx {
        // rd one-hot
        let r = (global_lane - rd_start) as u8;
        return F::from_bool(rd == Some(r));
    }
    if global_lane == unexp_pc_idx {
        return unexpanded_pc;
    }
    if global_lane == imm_idx {
        return imm;
    }
    if global_lane < instr_start {
        let flag_idx = global_lane - circuit_start;
        return F::from_bool(circuit_flags[flag_idx]);
    }
    if global_lane < lookup_start {
        let flag_idx = global_lane - instr_start;
        return F::from_bool(instr_flags[flag_idx]);
    }
    if global_lane < raf_flag_idx {
        let table_idx = global_lane - lookup_start;
        return F::from_bool(lookup_idx == Some(table_idx));
    }
    debug_assert_eq!(global_lane, raf_flag_idx);
    F::from_bool(raf_flag)
}

/// Build bytecode chunk polynomials from a preprocessed instruction slice.
///
/// This avoids constructing a `BytecodePreprocessing` wrapper (and its clones) when callers
/// already have the padded instruction list.
#[tracing::instrument(skip_all, name = "bytecode::build_bytecode_chunks_from_instructions")]
pub fn build_bytecode_chunks_from_instructions<F: JoltField>(
    instructions: &[Instruction],
    log_k_chunk: usize,
) -> Vec<MultilinearPolynomial<F>> {
    let k_chunk = 1usize << log_k_chunk;
    let bytecode_len = instructions.len();
    let total = total_lanes();
    let num_chunks = total.div_ceil(k_chunk);

    (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let mut coeffs = unsafe_allocate_zero_vec(k_chunk * bytecode_len);
            for k in 0..bytecode_len {
                let instr = &instructions[k];
                let normalized = instr.normalize();
                let circuit_flags = <Instruction as Flags>::circuit_flags(instr);
                let instr_flags = <Instruction as Flags>::instruction_flags(instr);
                let lookup_idx = <Instruction as InstructionLookup<XLEN>>::lookup_table(instr)
                    .map(|t| LookupTables::<XLEN>::enum_index(&t));
                let raf_flag = !InterleavedBitsMarker::is_interleaved_operands(&circuit_flags);

                let unexpanded_pc = F::from_u64(normalized.address as u64);
                let imm = F::from_i128(normalized.operands.imm);
                let rs1 = normalized.operands.rs1;
                let rs2 = normalized.operands.rs2;
                let rd = normalized.operands.rd;

                for lane in 0..k_chunk {
                    let global_lane = chunk_idx * k_chunk + lane;
                    if global_lane >= total {
                        break;
                    }
                    let value = lane_value::<F>(
                        global_lane,
                        rs1,
                        rs2,
                        rd,
                        unexpanded_pc,
                        imm,
                        &circuit_flags,
                        &instr_flags,
                        lookup_idx,
                        raf_flag,
                    );
                    let idx = DoryGlobals::get_layout().address_cycle_to_index(
                        lane,
                        k,
                        k_chunk,
                        bytecode_len,
                    );
                    coeffs[idx] = value;
                }
            }
            MultilinearPolynomial::from(coeffs)
        })
        .collect()
}

#[tracing::instrument(skip_all, name = "bytecode::build_bytecode_chunks")]
pub fn build_bytecode_chunks<F: JoltField>(
    bytecode: &BytecodePreprocessing,
    log_k_chunk: usize,
) -> Vec<MultilinearPolynomial<F>> {
    build_bytecode_chunks_from_instructions::<F>(&bytecode.bytecode, log_k_chunk)
}

/// Build bytecode chunk polynomials with main-matrix dimensions for CycleMajor embedding.
///
/// This creates bytecode chunks with `k_chunk * padded_trace_len` coefficients, using
/// main-matrix indexing (`lane * T + cycle`) instead of bytecode indexing (`lane * bytecode_len + cycle`).
///
/// **Why this is needed for CycleMajor:**
/// - In CycleMajor, coefficients are ordered as: lane 0's cycles, lane 1's cycles, ...
/// - Bytecode indexing gives: `lane * bytecode_len + cycle`
/// - Main indexing gives: `lane * T + cycle`
/// - When T > bytecode_len, these differ for lane > 0, causing row-commitment hint mismatch
///
/// **For AddressMajor, this is NOT needed** because both use `cycle * k_chunk + lane`,
/// which gives the same index for cycle < bytecode_len.
///
/// The bytecode values are placed at positions (lane, cycle) for cycle < bytecode_len,
/// with zeros for cycle >= bytecode_len (matching the "extra cycle vars fixed to 0" embedding).
#[tracing::instrument(skip_all, name = "bytecode::build_bytecode_chunks_for_main_matrix")]
pub fn build_bytecode_chunks_for_main_matrix<F: JoltField>(
    bytecode: &BytecodePreprocessing,
    log_k_chunk: usize,
    padded_trace_len: usize,
    layout: DoryLayout,
) -> Vec<MultilinearPolynomial<F>> {
    debug_assert_eq!(
        layout,
        DoryLayout::CycleMajor,
        "build_bytecode_chunks_for_main_matrix should only be used for CycleMajor layout"
    );

    let k_chunk = 1usize << log_k_chunk;
    let bytecode_len = bytecode.bytecode.len();
    let total = total_lanes();
    let num_chunks = total.div_ceil(k_chunk);

    debug_assert!(
        padded_trace_len >= bytecode_len,
        "padded_trace_len ({padded_trace_len}) must be >= bytecode_len ({bytecode_len})"
    );

    (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            // Use padded_trace_len for coefficient array size (main-matrix dimensions)
            let mut coeffs = unsafe_allocate_zero_vec(k_chunk * padded_trace_len);
            for k in 0..bytecode_len {
                let instr = &bytecode.bytecode[k];
                let normalized = instr.normalize();
                let circuit_flags = <Instruction as Flags>::circuit_flags(instr);
                let instr_flags = <Instruction as Flags>::instruction_flags(instr);
                let lookup_idx = <Instruction as InstructionLookup<XLEN>>::lookup_table(instr)
                    .map(|t| LookupTables::<XLEN>::enum_index(&t));
                let raf_flag = !InterleavedBitsMarker::is_interleaved_operands(&circuit_flags);

                let unexpanded_pc = F::from_u64(normalized.address as u64);
                let imm = F::from_i128(normalized.operands.imm);
                let rs1 = normalized.operands.rs1;
                let rs2 = normalized.operands.rs2;
                let rd = normalized.operands.rd;

                for lane in 0..k_chunk {
                    let global_lane = chunk_idx * k_chunk + lane;
                    if global_lane >= total {
                        break;
                    }
                    let value = lane_value::<F>(
                        global_lane,
                        rs1,
                        rs2,
                        rd,
                        unexpanded_pc,
                        imm,
                        &circuit_flags,
                        &instr_flags,
                        lookup_idx,
                        raf_flag,
                    );
                    // Use padded_trace_len (main T) for indexing
                    let idx = layout.address_cycle_to_index(lane, k, k_chunk, padded_trace_len);
                    coeffs[idx] = value;
                }
            }
            MultilinearPolynomial::from(coeffs)
        })
        .collect()
}
