use crate::field::JoltField;
use crate::poly::commitment::dory::DoryGlobals;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::instruction::{
    Flags, InstructionLookup, InterleavedBitsMarker, NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS,
};
use crate::zkvm::lookup_table::LookupTables;
use common::constants::{REGISTER_COUNT, XLEN};
use tracer::instruction::Instruction;

/// Total number of lanes encoded by committed-bytecode rows.
pub const fn total_lanes() -> usize {
    3 * (REGISTER_COUNT as usize)
        + 2
        + NUM_CIRCUIT_FLAGS
        + NUM_INSTRUCTION_FLAGS
        + <LookupTables<XLEN> as strum::EnumCount>::COUNT
        + 1
}

/// Fixed lane capacity for committed bytecode rows.
pub const COMMITTED_BYTECODE_LANE_CAPACITY: usize = total_lanes().next_power_of_two();

#[inline(always)]
pub const fn committed_lanes() -> usize {
    COMMITTED_BYTECODE_LANE_CAPACITY
}

pub const DEFAULT_COMMITTED_BYTECODE_CHUNK_COUNT: usize = 1;

#[inline]
pub fn validate_committed_bytecode_chunk_count(chunk_count: usize) {
    assert!(chunk_count > 0, "bytecode chunk count must be non-zero");
    assert!(
        chunk_count.is_power_of_two(),
        "bytecode chunk count must be a power of two"
    );
}

#[inline(always)]
pub fn validate_committed_bytecode_chunking_for_len(bytecode_len: usize, chunk_count: usize) {
    validate_committed_bytecode_chunk_count(chunk_count);
    assert!(
        bytecode_len.is_multiple_of(chunk_count),
        "bytecode length ({bytecode_len}) must be divisible by chunk count ({chunk_count})"
    );
}

#[inline(always)]
pub fn committed_bytecode_chunk_cycle_len(bytecode_len: usize, chunk_count: usize) -> usize {
    validate_committed_bytecode_chunking_for_len(bytecode_len, chunk_count);
    bytecode_len / chunk_count
}

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
}

pub const BYTECODE_LANE_LAYOUT: BytecodeLaneLayout = BytecodeLaneLayout::new();

#[derive(Clone, Copy, Debug)]
pub enum ActiveLaneValue<F: JoltField> {
    One,
    Scalar(F),
}

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

    if let Some(r) = normalized.operands.rs1 {
        visit(l.rs1_start + (r as usize), ActiveLaneValue::One);
    }
    if let Some(r) = normalized.operands.rs2 {
        visit(l.rs2_start + (r as usize), ActiveLaneValue::One);
    }
    if let Some(r) = normalized.operands.rd {
        visit(l.rd_start + (r as usize), ActiveLaneValue::One);
    }

    let unexpanded_pc = F::from_u64(normalized.address as u64);
    if !unexpanded_pc.is_zero() {
        visit(l.unexp_pc_idx, ActiveLaneValue::Scalar(unexpanded_pc));
    }
    let imm = F::from_i128(normalized.operands.imm);
    if !imm.is_zero() {
        visit(l.imm_idx, ActiveLaneValue::Scalar(imm));
    }

    for i in 0..NUM_CIRCUIT_FLAGS {
        if circuit_flags[i] {
            visit(l.circuit_start + i, ActiveLaneValue::One);
        }
    }
    for i in 0..NUM_INSTRUCTION_FLAGS {
        if instr_flags[i] {
            visit(l.instr_start + i, ActiveLaneValue::One);
        }
    }
    if let Some(t) = lookup_idx {
        visit(l.lookup_start + t, ActiveLaneValue::One);
    }
    if raf_flag {
        visit(l.raf_flag_idx, ActiveLaneValue::One);
    }
}

#[tracing::instrument(skip_all, name = "bytecode::build_committed_bytecode_chunk_coeffs")]
pub fn build_committed_bytecode_chunk_coeffs<F: JoltField>(
    instructions: &[Instruction],
    chunk_count: usize,
) -> Vec<Vec<F>> {
    let bytecode_len = instructions.len();
    validate_committed_bytecode_chunking_for_len(bytecode_len, chunk_count);

    let chunk_cycle_len = committed_bytecode_chunk_cycle_len(bytecode_len, chunk_count);
    let lane_capacity = committed_lanes();
    let mut chunk_coeffs: Vec<Vec<F>> = (0..chunk_count)
        .map(|_| unsafe_allocate_zero_vec(lane_capacity * chunk_cycle_len))
        .collect();

    for (cycle, instr) in instructions.iter().enumerate() {
        let cycle_chunk_idx = cycle / chunk_cycle_len;
        let chunk_cycle = cycle % chunk_cycle_len;
        let coeffs = &mut chunk_coeffs[cycle_chunk_idx];

        for_each_active_lane_value::<F>(instr, |global_lane, lane_val| {
            let idx = DoryGlobals::get_layout().address_cycle_to_index(
                global_lane,
                chunk_cycle,
                lane_capacity,
                chunk_cycle_len,
            );
            let lane_value = match lane_val {
                ActiveLaneValue::One => F::one(),
                ActiveLaneValue::Scalar(v) => v,
            };
            coeffs[idx] += lane_value;
        });
    }

    chunk_coeffs
}

#[tracing::instrument(
    skip_all,
    name = "bytecode::build_committed_bytecode_chunk_polynomials"
)]
pub fn build_committed_bytecode_chunk_polynomials<F: JoltField>(
    instructions: &[Instruction],
    chunk_count: usize,
) -> Vec<MultilinearPolynomial<F>> {
    build_committed_bytecode_chunk_coeffs::<F>(instructions, chunk_count)
        .into_iter()
        .map(MultilinearPolynomial::from)
        .collect()
}
