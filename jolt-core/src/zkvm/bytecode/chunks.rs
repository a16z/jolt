use crate::field::JoltField;
use crate::poly::commitment::dory::{DoryGlobals, DoryLayout};
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::instruction::{
    Flags, InstructionLookup, NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS,
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

#[tracing::instrument(skip_all, name = "bytecode::build_bytecode_chunks")]
pub fn build_bytecode_chunks<F: JoltField>(
    bytecode: &BytecodePreprocessing,
    log_k_chunk: usize,
) -> Vec<MultilinearPolynomial<F>> {
    let k_chunk = 1usize << log_k_chunk;
    let bytecode_len = bytecode.bytecode.len();
    let total = total_lanes();
    let num_chunks = total.div_ceil(k_chunk);

    (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let mut coeffs = unsafe_allocate_zero_vec(k_chunk * bytecode_len);
            for k in 0..bytecode_len {
                let instr = &bytecode.bytecode[k];
                let normalized = instr.normalize();
                let circuit_flags = <Instruction as Flags>::circuit_flags(instr);
                let instr_flags = <Instruction as Flags>::instruction_flags(instr);
                let lookup_idx = <Instruction as InstructionLookup<XLEN>>::lookup_table(instr)
                    .map(|t| LookupTables::<XLEN>::enum_index(&t));
                let raf_flag =
                    !crate::zkvm::instruction::InterleavedBitsMarker::is_interleaved_operands(
                        &circuit_flags,
                    );

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
                let raf_flag =
                    !crate::zkvm::instruction::InterleavedBitsMarker::is_interleaved_operands(
                        &circuit_flags,
                    );

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
