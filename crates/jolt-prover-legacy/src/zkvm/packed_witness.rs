//! Prover-side Akita witness assembly. Wjolt is a native commitment group of
//! uniform row-major one-hot members; auxiliary program/advice objects retain
//! sparse prefix-packed representations.

use jolt_claims::protocols::jolt::lattice::geometry::WORD_BYTES;
use jolt_claims::protocols::jolt::lattice::UnsignedIncChunking;
pub use jolt_claims::protocols::jolt::lattice::UNSIGNED_INC_BITS;
use jolt_claims::protocols::jolt::{BytecodeRegisterLane, JoltCommittedPolynomial};
use jolt_openings::PrefixPacking;
use jolt_poly::OneHotPolynomial;
use jolt_riscv::JoltInstructionRow;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::utils::math::Math;
use crate::zkvm::instruction::{
    CircuitFlags, Flags, InstructionLookup, InterleavedBitsMarker, JoltTraceCycle,
};
use crate::zkvm::lookup_table::LookupTables;
use common::constants::XLEN;

/// Sparse unit-valued multilinear polynomial: value `1` at each listed
/// position, `0` everywhere else — the witness form of a packed one-hot
/// commitment. The union of one-hot columns scattered into prefix slots is
/// exactly a set of unit positions over the packed domain, so it advertises
/// the `MultilinearPoly` unit-sparse contract (`is_one_hot`/`for_each_one`)
/// without `OneHotPolynomial`'s per-row structure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseUnitPolynomial<F> {
    num_vars: usize,
    one_positions: Vec<usize>,
    _field: core::marker::PhantomData<F>,
}

impl<F: jolt_field::Field> SparseUnitPolynomial<F> {
    /// # Panics
    ///
    /// Panics if a position lies outside the `2^num_vars` domain.
    #[must_use]
    pub fn new(num_vars: usize, one_positions: Vec<usize>) -> Self {
        assert!(
            one_positions
                .iter()
                .all(|position| position >> num_vars == 0),
            "one position outside the 2^{num_vars} domain"
        );
        Self {
            num_vars,
            one_positions,
            _field: core::marker::PhantomData,
        }
    }

    #[must_use]
    pub fn one_positions(&self) -> &[usize] {
        &self.one_positions
    }
}

impl<F: jolt_field::Field> jolt_poly::MultilinearPoly<F> for SparseUnitPolynomial<F> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn evaluate(&self, point: &[F]) -> F {
        assert_eq!(point.len(), self.num_vars);
        self.one_positions
            .iter()
            .map(|position| {
                point.iter().enumerate().fold(F::one(), |acc, (bit, r)| {
                    // Big-endian: point[0] is the most significant bit.
                    if (position >> (self.num_vars - 1 - bit)) & 1 == 1 {
                        acc * *r
                    } else {
                        acc * (F::one() - *r)
                    }
                })
            })
            .sum()
    }

    fn for_each_row(&self, sigma: usize, f: &mut dyn FnMut(usize, &[F])) {
        let row_len = 1usize << sigma;
        let num_rows = 1usize << (self.num_vars - sigma);
        let mut row = vec![F::zero(); row_len];
        let mut positions = self.one_positions.clone();
        positions.sort_unstable();
        let mut next = positions.iter().peekable();
        for row_index in 0..num_rows {
            row.fill(F::zero());
            while let Some(&&position) = next.peek() {
                if position >> sigma != row_index {
                    break;
                }
                row[position & (row_len - 1)] = F::one();
                let _ = next.next();
            }
            f(row_index, &row);
        }
    }

    fn is_one_hot(&self) -> bool {
        true
    }

    fn for_each_one(&self, f: &mut dyn FnMut(usize)) {
        for position in &self.one_positions {
            f(*position);
        }
    }
}

/// The per-cycle fused increment stream: the RAM delta on store cycles, the
/// rd delta otherwise. Padding cycles carry `delta = 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FusedIncCycle {
    pub delta: i128,
}

impl FusedIncCycle {
    /// The per-cycle fused delta: the RAM write delta on store cycles, the
    /// rd write delta otherwise.
    pub fn from_cycle(cycle: &tracer::instruction::Cycle) -> Self {
        Self::from_cycle_with_store(cycle).0
    }

    /// [`from_cycle`](Self::from_cycle) plus the store selector itself, so
    /// witness generation and the `IncVirtualization` instance read one
    /// predicate: the same `OpFlags(Store)` circuit flag the sumcheck
    /// selector opens.
    pub fn from_cycle_with_store(cycle: &tracer::instruction::Cycle) -> (Self, bool) {
        let store = JoltTraceCycle::try_new(cycle)
            .expect("Wjolt cycles must be final Jolt instruction rows")
            .circuit_flags()[CircuitFlags::Store];
        let ram_delta = match cycle.ram_access() {
            tracer::instruction::RAMAccess::Write(write) => {
                write.post_value as i128 - write.pre_value as i128
            }
            _ => 0,
        };
        let (_, rd_pre_value, rd_post_value) = cycle.rd_write().unwrap_or_default();
        let rd_delta = rd_post_value as i128 - rd_pre_value as i128;
        // One fused column can serve both inc consumers only because no
        // cycle increments RAM and rd at once (every RMW instruction lowers
        // into a sequence whose RAM-writing step is a plain store). A
        // violation means an instruction shape the fused encoding cannot
        // represent — fail here, not with an opaque sumcheck mismatch.
        debug_assert_eq!(
            store,
            matches!(cycle.ram_access(), tracer::instruction::RAMAccess::Write(_)),
            "Store circuit flag disagrees with the cycle's RAM-write access: {cycle:?}"
        );
        debug_assert!(
            if store { rd_delta == 0 } else { ram_delta == 0 },
            "cycle increments both RAM and rd; the fused inc encoding cannot represent it: {cycle:?}"
        );
        let delta = if store { ram_delta } else { rd_delta };
        (Self { delta }, store)
    }

    /// The shifted unsigned encoding `2^64 + delta`: the msb bit and the
    /// low-64-bit chunk hot_lanes. Padding (`delta = 0`) encodes as msb hot
    /// with every chunk at hot_lane 0 (the jolt-claims padding invariant).
    fn shifted(self) -> u128 {
        debug_assert!(self.delta.unsigned_abs() < 1u128 << UNSIGNED_INC_BITS);
        (self.delta + (1i128 << UNSIGNED_INC_BITS)) as u128
    }

    pub fn msb(self) -> bool {
        self.shifted() >> UNSIGNED_INC_BITS == 1
    }

    pub fn chunk_hot_lane(self, chunking: UnsignedIncChunking, index: usize) -> usize {
        self.chunk_hot_lane_bits(chunking.chunk_width(), index)
    }

    /// Chunk hot_lane from a plain bit width (the shared-final-point invariant
    /// fixes `width == log_k_chunk`).
    pub fn chunk_hot_lane_bits(self, width: usize, index: usize) -> usize {
        let low = self.shifted() & ((1u128 << UNSIGNED_INC_BITS) - 1);
        ((low >> (width * index)) & ((1u128 << width) - 1)) as usize
    }
}

/// One committed column's scatter input: what fills its slot subcube.
enum ColumnCells {
    /// Per-cycle hot address (`None` on padding cycles — no access).
    Ra(Vec<Option<usize>>),
    /// The fused-inc chunk column at this index.
    Chunk(usize),
    /// The fused-inc msb column.
    Msb,
}

/// Builds the strict one-hot Wjolt members in canonical order for the native
/// one-hot commitment format: `K = 2^chunk_width` lanes over `2^log_t`
/// row-major rows (`cycle · K + lane`).
///
/// `Ra` members take their per-cycle hot address, `UnsignedIncChunk(i)` its
/// per-cycle chunk address, and the MSB member encodes its bit as hot address
/// zero or one.
pub fn assemble_one_hot_members(
    members: &[JoltCommittedPolynomial],
    chunking: UnsignedIncChunking,
    log_t: usize,
    ra_indices: &dyn Fn(JoltCommittedPolynomial) -> Option<Vec<Option<usize>>>,
    fused_inc: &[FusedIncCycle],
) -> Result<Vec<OneHotPolynomial>, String> {
    debug_assert!(fused_inc.len() <= 1 << log_t);
    let k = 1usize
        .checked_shl(chunking.chunk_width() as u32)
        .filter(|k| *k <= u8::MAX as usize + 1)
        .ok_or_else(|| {
            format!(
                "one-hot member lanes need chunk width <= 8, got {}",
                chunking.chunk_width()
            )
        })?;
    let rows = 1usize << log_t;

    let mut columns = Vec::new();
    for polynomial in members {
        let cells = match polynomial {
            JoltCommittedPolynomial::InstructionRa(_)
            | JoltCommittedPolynomial::BytecodeRa(_)
            | JoltCommittedPolynomial::RamRa(_) => ColumnCells::Ra(
                ra_indices(*polynomial)
                    .ok_or_else(|| format!("missing ra indices for {polynomial:?}"))?,
            ),
            JoltCommittedPolynomial::UnsignedIncChunk(index) => ColumnCells::Chunk(*index),
            JoltCommittedPolynomial::UnsignedIncMsb => ColumnCells::Msb,
            other => {
                return Err(format!("polynomial {other:?} is not a native Wjolt member"));
            }
        };
        columns.push((*polynomial, cells));
    }

    columns
        .into_par_iter()
        .map(|(polynomial, cells)| {
            let mut indices: Vec<Option<u8>> = vec![None; rows];
            match cells {
                ColumnCells::Ra(addresses) => {
                    for (cycle, address) in addresses.iter().enumerate() {
                        let Some(address) = address else { continue };
                        indices[cycle] = Some(u8::try_from(*address).map_err(|_| {
                            format!("{polynomial:?} address {address} exceeds the {k} lanes")
                        })?);
                    }
                }
                ColumnCells::Chunk(index) => {
                    for (cycle, inc) in fused_inc.iter().enumerate() {
                        indices[cycle] = Some(inc.chunk_hot_lane(chunking, index) as u8);
                    }
                }
                ColumnCells::Msb => {
                    for (cycle, inc) in fused_inc.iter().enumerate() {
                        indices[cycle] = Some(inc.msb() as u8);
                    }
                }
            }
            Ok(OneHotPolynomial::new(k, indices))
        })
        .collect()
}

/// Scatters the precommitted `W_prog` sub-columns (per-chunk bytecode lanes
/// and the program image) into one-positions of the packed precommitted
/// witness, per the canonical `precommitted_packing` slots. Row domain per
/// chunk is `2^log_bytecode_rows` (bytecode rows, zero-padded); byte
/// one-hot columns encode padding as hot_lane-0 hot (never all-zero), the
/// selector/flag columns leave padding rows empty.
///
/// The imm lane decomposes `F::from_i128(imm)`'s canonical little-endian
/// field bytes over `imm_byte_width` limbs — the same value
/// `for_each_active_lane_value` places on the base committed chunk, so the
/// byte reconstruction and the base lane agree exactly (including negative
/// immediates, which wrap to `p − |imm|`).
pub fn assemble_precommitted_witness<F: JoltField>(
    packing: &PrefixPacking<JoltCommittedPolynomial>,
    instructions: &[JoltInstructionRow],
    log_bytecode_rows: usize,
    imm_byte_width: usize,
    program_image_words: Option<&[u64]>,
) -> Result<Vec<usize>, String> {
    let rows = 1usize << log_bytecode_rows;
    let chunk_rows = |chunk: usize| -> &[JoltInstructionRow] {
        let start = (chunk * rows).min(instructions.len());
        let end = ((chunk + 1) * rows).min(instructions.len());
        &instructions[start..end]
    };
    let imm_limb_bits = imm_byte_width.log_2();
    let imm_bytes = |imm: i128| -> Result<Vec<u8>, String> {
        let value = F::from_i128(imm);
        let mut bytes = Vec::new();
        value
            .serialize_compressed(&mut bytes)
            .map_err(|error| format!("imm serialization failed: {error}"))?;
        if bytes.len() < imm_byte_width || bytes[imm_byte_width..].iter().any(|byte| *byte != 0) {
            return Err(format!(
                "imm {imm} does not fit the canonical {imm_byte_width}-byte lane"
            ));
        }
        bytes.truncate(imm_byte_width);
        Ok(bytes)
    };

    let mut one_positions = Vec::new();
    for (column, slot) in packing {
        match column {
            JoltCommittedPolynomial::BytecodeRegisterSelector { chunk, lane } => {
                for (row, instruction) in chunk_rows(*chunk).iter().enumerate() {
                    let register = match lane {
                        BytecodeRegisterLane::Rs1 => instruction.operands.rs1,
                        BytecodeRegisterLane::Rs2 => instruction.operands.rs2,
                        BytecodeRegisterLane::Rd => instruction.operands.rd,
                    };
                    if let Some(register) = register {
                        one_positions.push(
                            slot.packed_index(((register as usize) << log_bytecode_rows) | row),
                        );
                    }
                }
            }
            JoltCommittedPolynomial::BytecodeCircuitFlag { chunk, flag } => {
                for (row, instruction) in chunk_rows(*chunk).iter().enumerate() {
                    if instruction.circuit_flags()[*flag] {
                        one_positions.push(slot.packed_index(row));
                    }
                }
            }
            JoltCommittedPolynomial::BytecodeInstructionFlag { chunk, flag } => {
                for (row, instruction) in chunk_rows(*chunk).iter().enumerate() {
                    if instruction.instruction_flags()[*flag] {
                        one_positions.push(slot.packed_index(row));
                    }
                }
            }
            JoltCommittedPolynomial::BytecodeLookupSelector { chunk } => {
                for (row, instruction) in chunk_rows(*chunk).iter().enumerate() {
                    if let Some(table) = InstructionLookup::<XLEN>::lookup_table(instruction) {
                        let index = LookupTables::<XLEN>::enum_index(&table);
                        one_positions.push(slot.packed_index((index << log_bytecode_rows) | row));
                    }
                }
            }
            JoltCommittedPolynomial::BytecodeRafFlag { chunk } => {
                for (row, instruction) in chunk_rows(*chunk).iter().enumerate() {
                    if !InterleavedBitsMarker::is_interleaved_operands(&instruction.circuit_flags())
                    {
                        one_positions.push(slot.packed_index(row));
                    }
                }
            }
            JoltCommittedPolynomial::BytecodeUnexpandedPcBytes { chunk } => {
                let instructions = chunk_rows(*chunk);
                let limb_bits = WORD_BYTES.log_2();
                for limb in 0..WORD_BYTES {
                    for row in 0..rows {
                        let byte = instructions.get(row).map_or(0, |instruction| {
                            ((instruction.address as u64) >> (8 * limb)) as u8
                        }) as usize;
                        one_positions.push(slot.packed_index(
                            (((byte << limb_bits) | limb) << log_bytecode_rows) | row,
                        ));
                    }
                }
            }
            JoltCommittedPolynomial::BytecodeImmBytes { chunk } => {
                let instructions = chunk_rows(*chunk);
                for row in 0..rows {
                    let bytes = match instructions.get(row) {
                        Some(instruction) => imm_bytes(instruction.operands.imm)?,
                        None => vec![0u8; imm_byte_width],
                    };
                    for (limb, byte) in bytes.into_iter().enumerate() {
                        one_positions.push(slot.packed_index(
                            ((((byte as usize) << imm_limb_bits) | limb) << log_bytecode_rows)
                                | row,
                        ));
                    }
                }
            }
            JoltCommittedPolynomial::ProgramImageBytes => {
                let words = program_image_words
                    .ok_or_else(|| "program image words missing for W_prog".to_string())?;
                let word_vars = slot.num_vars - 8 - WORD_BYTES.log_2();
                let limb_bits = WORD_BYTES.log_2();
                debug_assert!(words.len() <= 1 << word_vars);
                for limb in 0..WORD_BYTES {
                    for word_index in 0..(1usize << word_vars) {
                        let byte = words
                            .get(word_index)
                            .map_or(0, |word| (word >> (8 * limb)) as u8)
                            as usize;
                        one_positions.push(slot.packed_index(
                            (((byte << limb_bits) | limb) << word_vars) | word_index,
                        ));
                    }
                }
            }
            other => {
                return Err(format!(
                    "column {other:?} is not part of the precommitted packed witness"
                ))
            }
        }
    }
    one_positions.sort_unstable();
    Ok(one_positions)
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    const LOG_T: usize = 3;
    const LOG_K_CHUNK: usize = 8;

    fn chunking() -> UnsignedIncChunking {
        UnsignedIncChunking::new(LOG_K_CHUNK).unwrap()
    }

    fn fused_trace() -> Vec<FusedIncCycle> {
        [7i128, -3, 0, (1 << 40) + 5, -(1 << 63), 1, -1, 0]
            .into_iter()
            .map(|delta| FusedIncCycle { delta })
            .collect()
    }

    fn members() -> Vec<(JoltCommittedPolynomial, OneHotPolynomial)> {
        let ra = |polynomial: JoltCommittedPolynomial| {
            let base = match polynomial {
                JoltCommittedPolynomial::InstructionRa(_) => 3usize,
                JoltCommittedPolynomial::BytecodeRa(_) => 5,
                JoltCommittedPolynomial::RamRa(_) => 7,
                _ => return None,
            };
            Some(
                (0..1 << LOG_T)
                    .map(|cycle| (cycle != 2).then_some((base * (cycle + 1)) % 256))
                    .collect(),
            )
        };
        let mut ids = vec![
            JoltCommittedPolynomial::InstructionRa(0),
            JoltCommittedPolynomial::BytecodeRa(0),
            JoltCommittedPolynomial::RamRa(0),
        ];
        ids.extend((0..chunking().chunk_count()).map(JoltCommittedPolynomial::UnsignedIncChunk));
        ids.push(JoltCommittedPolynomial::UnsignedIncMsb);
        let polynomials =
            assemble_one_hot_members(&ids, chunking(), LOG_T, &ra, &fused_trace()).unwrap();
        ids.into_iter().zip(polynomials).collect()
    }

    #[test]
    fn every_wjolt_member_has_the_uniform_native_shape() {
        for (_, member) in members() {
            assert_eq!(member.num_vars(), LOG_K_CHUNK + LOG_T);
            assert_eq!(member.k(), 1 << LOG_K_CHUNK);
            assert_eq!(member.indices().len(), 1 << LOG_T);
        }
    }

    #[test]
    fn chunk_columns_reconstruct_the_shifted_fused_increment() {
        let chunking = chunking();
        let trace = fused_trace();
        let members = members();
        for (cycle, inc) in trace.iter().enumerate() {
            let mut reconstructed = 0u128;
            for index in 0..chunking.chunk_count() {
                let (_, member) = members
                    .iter()
                    .find(|(id, _)| *id == JoltCommittedPolynomial::UnsignedIncChunk(index))
                    .unwrap();
                let hot = member.indices()[cycle].unwrap() as usize;
                reconstructed |= (hot as u128) << (chunking.chunk_width() * index);
            }
            let (_, msb) = members
                .iter()
                .find(|(id, _)| *id == JoltCommittedPolynomial::UnsignedIncMsb)
                .unwrap();
            reconstructed |= (msb.indices()[cycle].unwrap() as u128) << UNSIGNED_INC_BITS;
            assert_eq!(
                reconstructed as i128 - (1i128 << UNSIGNED_INC_BITS),
                inc.delta,
                "cycle {cycle}"
            );
        }
    }

    #[test]
    fn msb_is_one_hot_at_address_zero_or_one() {
        let (_, msb) = members()
            .into_iter()
            .find(|(id, _)| *id == JoltCommittedPolynomial::UnsignedIncMsb)
            .unwrap();
        for (cycle, inc) in fused_trace().iter().enumerate() {
            assert_eq!(msb.indices()[cycle], Some(u8::from(inc.msb())));
        }
    }

    #[test]
    fn padding_cycles_encode_msb_hot_chunks_at_zero() {
        let padding = FusedIncCycle { delta: 0 };
        assert!(padding.msb());
        for index in 0..chunking().chunk_count() {
            assert_eq!(padding.chunk_hot_lane(chunking(), index), 0);
        }
    }

    #[test]
    fn ra_member_retains_absent_rows() {
        let (_, instruction) = members()
            .into_iter()
            .find(|(id, _)| *id == JoltCommittedPolynomial::InstructionRa(0))
            .unwrap();
        for cycle in 0..1usize << LOG_T {
            let expected = (cycle != 2).then_some(((3 * (cycle + 1)) % 256) as u8);
            assert_eq!(instruction.indices()[cycle], expected, "cycle {cycle}");
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod precommitted_tests {
    use super::*;
    use crate::poly::eq_poly::EqPolynomial;
    use crate::zkvm::bytecode::chunks::{for_each_active_lane_value, ActiveLaneValue};
    use ark_bn254::Fr;
    use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::{
        committed_lane_vars, BYTECODE_LANE_LAYOUT,
    };
    use jolt_claims::protocols::jolt::lattice::{precommitted_packing, PrecommittedPackingShape};
    use jolt_field::Fr as ClaimsFr;
    use jolt_field::FromPrimitiveInt;
    use jolt_riscv::{JoltInstructionKind, NormalizedOperands};

    fn row(
        kind: JoltInstructionKind,
        rd: Option<u8>,
        rs1: Option<u8>,
        rs2: Option<u8>,
        imm: i128,
        address: usize,
    ) -> JoltInstructionRow {
        JoltInstructionRow {
            instruction_kind: kind,
            address,
            operands: NormalizedOperands { rd, rs1, rs2, imm },
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        }
    }

    /// Per-cell reconstruction weight of a `W_prog` sub-column: the value the
    /// bytecode chunk reconstruction attributes to that cell against the
    /// chunk's lane-eq table (register/lookup selectors and flags are plain
    /// lane weights; pc/imm bytes carry the `byte · 256^place` decode).
    fn cell_weight(
        eq_lane: &[ClaimsFr],
        column: &JoltCommittedPolynomial,
        cell: usize,
        log_rows: usize,
        slot_num_vars: usize,
    ) -> ClaimsFr {
        let layout = BYTECODE_LANE_LAYOUT;
        let byte_decode = |lane_index: usize| {
            let limb_bits = slot_num_vars - log_rows - 8;
            let byte = cell >> limb_bits;
            let limb = cell & ((1 << limb_bits) - 1);
            let mut place_value = ClaimsFr::from_u64(1);
            for _ in 0..limb {
                place_value *= ClaimsFr::from_u64(256);
            }
            eq_lane[lane_index] * place_value * ClaimsFr::from_u64(byte as u64)
        };
        match column {
            JoltCommittedPolynomial::BytecodeRegisterSelector { lane, .. } => {
                let start = match lane {
                    BytecodeRegisterLane::Rs1 => layout.rs1_start,
                    BytecodeRegisterLane::Rs2 => layout.rs2_start,
                    BytecodeRegisterLane::Rd => layout.rd_start,
                };
                eq_lane[start + cell]
            }
            JoltCommittedPolynomial::BytecodeCircuitFlag { flag, .. } => {
                eq_lane[layout.circuit_start + flag]
            }
            JoltCommittedPolynomial::BytecodeInstructionFlag { flag, .. } => {
                eq_lane[layout.instr_start + flag]
            }
            JoltCommittedPolynomial::BytecodeLookupSelector { .. } => {
                eq_lane[layout.lookup_start + cell]
            }
            JoltCommittedPolynomial::BytecodeRafFlag { .. } => eq_lane[layout.raf_flag_idx],
            JoltCommittedPolynomial::BytecodeUnexpandedPcBytes { .. } => {
                byte_decode(layout.unexp_pc_idx)
            }
            JoltCommittedPolynomial::BytecodeImmBytes { .. } => byte_decode(layout.imm_idx),
            _ => ClaimsFr::from_u64(0),
        }
    }

    /// The precommitted sub-column witness must reconstruct the base
    /// committed bytecode chunk exactly: for a random `(lane ‖ row)` point,
    /// applying the per-cell reconstruction weights to the packed sub-column
    /// cells equals the direct lane-value evaluation
    /// (`for_each_active_lane_value`) — pinning the witness layout, the
    /// canonical imm bytes (including a negative imm), and the lane layout
    /// against each other.
    #[test]
    fn precommitted_witness_reconstructs_the_committed_chunk() {
        const LOG_ROWS: usize = 2;
        const IMM_BYTES: usize = 32;
        let instructions = vec![
            row(
                JoltInstructionKind::ADDI,
                Some(1),
                Some(2),
                None,
                3,
                0x80000000,
            ),
            row(
                JoltInstructionKind::ADD,
                Some(5),
                Some(6),
                Some(7),
                0,
                0x80000004,
            ),
            row(
                JoltInstructionKind::SD,
                None,
                Some(8),
                Some(9),
                -16,
                0x80000008,
            ),
            row(
                JoltInstructionKind::BEQ,
                None,
                Some(3),
                Some(4),
                64,
                0x8000000c,
            ),
        ];

        let shape = PrecommittedPackingShape {
            bytecode_chunks: 1,
            log_bytecode_rows: LOG_ROWS,
            imm_byte_width: IMM_BYTES,
            program_image_log_words: Some(1),
        };
        let packing = precommitted_packing(&shape).unwrap();
        let program_image_words = [0xdeadbeefu64, 0x0102030405060708];
        let one_positions = assemble_precommitted_witness::<Fr>(
            &packing,
            &instructions,
            LOG_ROWS,
            IMM_BYTES,
            Some(&program_image_words),
        )
        .unwrap();
        let witness: std::collections::HashSet<usize> = one_positions.iter().copied().collect();
        assert_eq!(witness.len(), one_positions.len(), "positions are distinct");

        // Random-ish (lane ‖ row) point over the jolt-claims field.
        let lane_point: Vec<ClaimsFr> = (0..committed_lane_vars())
            .map(|i| ClaimsFr::from_u64(3 + 7 * i as u64))
            .collect();
        let row_point: Vec<ClaimsFr> = (0..LOG_ROWS)
            .map(|i| ClaimsFr::from_u64(101 + 13 * i as u64))
            .collect();

        // Reconstructed value: Σ over sub-column cells of
        // weight(column, cell) · W[cell ‖ row] · eq(row), the packed rows
        // evaluated from the one-positions.
        let eq_lane = jolt_poly::EqPolynomial::<ClaimsFr>::evals(&lane_point, None);
        let eq_row = jolt_poly::EqPolynomial::<ClaimsFr>::evals(&row_point, None);
        let mut reconstructed = ClaimsFr::from_u64(0);
        for (column, slot) in &packing {
            if matches!(column, JoltCommittedPolynomial::ProgramImageBytes) {
                continue;
            }
            let cells = 1usize << (slot.num_vars - LOG_ROWS);
            for cell in 0..cells {
                let weight = cell_weight(&eq_lane, column, cell, LOG_ROWS, slot.num_vars);
                for (r, eq) in eq_row.iter().enumerate() {
                    if witness.contains(&slot.packed_index((cell << LOG_ROWS) | r)) {
                        reconstructed += weight * *eq;
                    }
                }
            }
        }

        // Direct value: Σ_row eq(row) Σ_lane eq(lane) · lane_value.
        let eq_row_ark = EqPolynomial::<Fr>::evals(
            &row_point
                .iter()
                .map(|value| {
                    let mut bytes = Vec::new();
                    ark_serialize::CanonicalSerialize::serialize_compressed(value, &mut bytes)
                        .unwrap();
                    <Fr as ark_serialize::CanonicalDeserialize>::deserialize_compressed(
                        bytes.as_slice(),
                    )
                    .unwrap()
                })
                .collect::<Vec<Fr>>(),
        );
        let eq_lane_ark: Vec<Fr> = eq_lane
            .iter()
            .map(|value| {
                let mut bytes = Vec::new();
                ark_serialize::CanonicalSerialize::serialize_compressed(value, &mut bytes).unwrap();
                <Fr as ark_serialize::CanonicalDeserialize>::deserialize_compressed(
                    bytes.as_slice(),
                )
                .unwrap()
            })
            .collect();
        let mut direct = Fr::from(0u64);
        for (r, instruction) in instructions.iter().enumerate() {
            for_each_active_lane_value::<Fr>(instruction, |lane, value| {
                let value = match value {
                    ActiveLaneValue::One => Fr::from(1u64),
                    ActiveLaneValue::Scalar(v) => v,
                };
                direct += eq_lane_ark[lane] * eq_row_ark[r] * value;
            });
        }

        let mut reconstructed_bytes = Vec::new();
        ark_serialize::CanonicalSerialize::serialize_compressed(
            &reconstructed,
            &mut reconstructed_bytes,
        )
        .unwrap();
        let reconstructed_ark =
            <Fr as ark_serialize::CanonicalDeserialize>::deserialize_compressed(
                reconstructed_bytes.as_slice(),
            )
            .unwrap();
        assert_eq!(
            reconstructed_ark, direct,
            "sub-column reconstruction must equal the committed chunk lane evaluation"
        );
    }
}
