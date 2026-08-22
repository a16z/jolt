//! Prover-side Akita witness assembly. `OneHotTrace` contains the uniform
//! row-major one-hot columns derived from the execution trace; program objects
//! retain sparse prefix-packed representations.

use jolt_claims::protocols::jolt::lattice::geometry::WORD_BYTES;
use jolt_claims::protocols::jolt::lattice::PrefixPackedObjectPlan;
pub use jolt_claims::protocols::jolt::lattice::FUSED_INC_BITS;
use jolt_claims::protocols::jolt::{BytecodeRegisterLane, JoltCommittedPolynomial};
use jolt_poly::OneHotPolynomial;
use jolt_riscv::JoltInstructionRow;

use crate::field::JoltField;
use crate::utils::math::Math;
use crate::zkvm::instruction::{
    CircuitFlags, Flags, InstructionLookup, InterleavedBitsMarker, JoltTraceCycle,
};
use crate::zkvm::lookup_table::LookupTables;
use common::constants::XLEN;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DigitZeroRow {
    Committed,
    Virtualized,
}

/// Packs equal-length semantic columns into prefix slots and pads unused
/// slots with zero rows. A virtualized digit-zero row is omitted from the
/// physical polynomial; a committed column retains row zero.
pub fn pack_one_hot_columns(
    k: usize,
    slot_capacity: usize,
    columns: Vec<(Vec<Option<u8>>, DigitZeroRow)>,
) -> OneHotPolynomial {
    assert!(slot_capacity.is_power_of_two());
    assert!(!columns.is_empty() && columns.len() <= slot_capacity);
    let rows = columns[0].0.len();
    assert!(columns.iter().all(|(column, _)| column.len() == rows));

    let mut indices = Vec::with_capacity(slot_capacity * rows);
    for (column, digit_zero_row) in columns {
        indices.extend(column.into_iter().map(|row| match row {
            Some(0) if digit_zero_row == DigitZeroRow::Virtualized => None,
            None => None,
            Some(row) => {
                assert!((row as usize) < k);
                Some(row)
            }
        }));
    }
    indices.resize(slot_capacity * rows, None);
    OneHotPolynomial::new(k, indices)
}

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

impl<F: jolt_field::JoltField> SparseUnitPolynomial<F> {
    /// Sorts the positions ascending once here — the invariant
    /// `for_each_row`'s row scan and `for_each_one`'s yield order rely on.
    /// Duplicates are neither deduplicated nor rejected.
    ///
    /// # Panics
    ///
    /// Panics if a position lies outside the `2^num_vars` domain.
    #[must_use]
    pub fn new(num_vars: usize, mut one_positions: Vec<usize>) -> Self {
        assert!(
            one_positions
                .iter()
                .all(|position| position >> num_vars == 0),
            "one position outside the 2^{num_vars} domain"
        );
        one_positions.sort_unstable();
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

impl<F: jolt_field::JoltField> jolt_poly::MultilinearPoly<F> for SparseUnitPolynomial<F> {
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
        let mut next = self.one_positions.iter().peekable();
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
pub struct FusedIncValue {
    pub delta: i128,
}

impl FusedIncValue {
    /// The per-cycle fused delta: the RAM write delta on store cycles, the
    /// rd write delta otherwise.
    pub fn from_cycle(cycle: &tracer::instruction::Cycle) -> Self {
        Self::from_cycle_with_store(cycle).0
    }

    /// [`from_cycle`](Self::from_cycle) plus the store selector itself, so
    /// witness generation and the read-raf fused stages read one
    /// predicate: the same `OpFlags(Store)` circuit flag the sumcheck
    /// selector opens.
    pub fn from_cycle_with_store(cycle: &tracer::instruction::Cycle) -> (Self, bool) {
        let store = JoltTraceCycle::try_new(cycle)
            .expect("OneHotTrace cycles must be final Jolt instruction rows")
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

    fn balanced_bias(width: usize) -> i128 {
        debug_assert!(width > 0 && FUSED_INC_BITS.is_multiple_of(width));
        let radix = 1i128 << width;
        (radix / 2) * (((1i128 << FUSED_INC_BITS) - 1) / (radix - 1))
    }

    fn biased_for_balanced_digits(self, width: usize) -> i128 {
        debug_assert!(self.delta.unsigned_abs() < 1u128 << FUSED_INC_BITS);
        self.delta + Self::balanced_bias(width)
    }

    /// The centered radix-`2^width` digit encoded modulo the radix.
    pub fn balanced_digit_row(self, width: usize, index: usize) -> usize {
        let radix = 1i128 << width;
        let mask = radix - 1;
        let standard_digit = (self.biased_for_balanced_digits(width) >> (width * index)) & mask;
        ((standard_digit + radix / 2) & mask) as usize
    }

    /// The signed carry above bit 63, encoded modulo the chunk radix.
    pub fn balanced_carry_row(self, width: usize) -> usize {
        let radix = 1i128 << width;
        let carry = self.biased_for_balanced_digits(width) >> FUSED_INC_BITS;
        debug_assert!((-1..=1).contains(&carry));
        carry.rem_euclid(radix) as usize
    }
}

/// Scatters the precommitted `ProgramOneHot` sub-columns (per-chunk bytecode lanes
/// and the program image) into one-positions of the packed precommitted
/// witness, per the canonical `precommitted_packing_plan` slots. Row domain per
/// chunk is `2^log_bytecode_rows` (bytecode rows, zero-padded); byte
/// one-hot columns encode padding with row zero selected (never all-zero), the
/// selector/flag columns leave padding rows empty.
///
/// The imm lane decomposes `F::from_i128(imm)`'s canonical little-endian
/// field bytes over `imm_byte_width` limbs — the same value
/// `for_each_active_lane_value` places on the base committed chunk, so the
/// byte reconstruction and the base lane agree exactly (including negative
/// immediates, which wrap to `p − |imm|`).
pub fn assemble_precommitted_witness<F: JoltField>(
    plan: &PrefixPackedObjectPlan,
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
    let packed_index = |column: &JoltCommittedPolynomial, local: usize| {
        plan.packing()
            .packed_index(column, local)
            .map_err(|error| error.to_string())
    };
    for column in plan.packing().ids() {
        match column {
            JoltCommittedPolynomial::BytecodeRegisterSelector { chunk, lane } => {
                for (row, instruction) in chunk_rows(*chunk).iter().enumerate() {
                    let register = match lane {
                        BytecodeRegisterLane::Rs1 => instruction.operands.rs1,
                        BytecodeRegisterLane::Rs2 => instruction.operands.rs2,
                        BytecodeRegisterLane::Rd => instruction.operands.rd,
                    };
                    if let Some(register) = register {
                        one_positions.push(packed_index(
                            column,
                            ((register as usize) << log_bytecode_rows) | row,
                        )?);
                    }
                }
            }
            JoltCommittedPolynomial::BytecodeCircuitFlag { chunk, flag } => {
                for (row, instruction) in chunk_rows(*chunk).iter().enumerate() {
                    if instruction.circuit_flags()[*flag] {
                        one_positions.push(packed_index(column, row)?);
                    }
                }
            }
            JoltCommittedPolynomial::BytecodeInstructionFlag { chunk, flag } => {
                for (row, instruction) in chunk_rows(*chunk).iter().enumerate() {
                    if instruction.instruction_flags()[*flag] {
                        one_positions.push(packed_index(column, row)?);
                    }
                }
            }
            JoltCommittedPolynomial::BytecodeLookupSelector { chunk } => {
                for (row, instruction) in chunk_rows(*chunk).iter().enumerate() {
                    if let Some(table) = InstructionLookup::<XLEN>::lookup_table(instruction) {
                        let index = LookupTables::<XLEN>::enum_index(&table);
                        one_positions
                            .push(packed_index(column, (index << log_bytecode_rows) | row)?);
                    }
                }
            }
            JoltCommittedPolynomial::BytecodeRafFlag { chunk } => {
                for (row, instruction) in chunk_rows(*chunk).iter().enumerate() {
                    if !InterleavedBitsMarker::is_interleaved_operands(&instruction.circuit_flags())
                    {
                        one_positions.push(packed_index(column, row)?);
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
                        one_positions.push(packed_index(
                            column,
                            (((byte << limb_bits) | limb) << log_bytecode_rows) | row,
                        )?);
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
                        one_positions.push(packed_index(
                            column,
                            ((((byte as usize) << imm_limb_bits) | limb) << log_bytecode_rows)
                                | row,
                        )?);
                    }
                }
            }
            JoltCommittedPolynomial::ProgramImageBytes => {
                let words = program_image_words
                    .ok_or_else(|| "program image words missing for ProgramOneHot".to_string())?;
                let word_vars = plan
                    .logical_num_vars(*column)
                    .ok_or_else(|| "ProgramImageBytes logical arity missing".to_string())?
                    - 8
                    - WORD_BYTES.log_2();
                let limb_bits = WORD_BYTES.log_2();
                debug_assert!(words.len() <= 1 << word_vars);
                for limb in 0..WORD_BYTES {
                    for word_index in 0..(1usize << word_vars) {
                        let byte = words
                            .get(word_index)
                            .map_or(0, |word| (word >> (8 * limb)) as u8)
                            as usize;
                        one_positions.push(packed_index(
                            column,
                            (((byte << limb_bits) | limb) << word_vars) | word_index,
                        )?);
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
    Ok(one_positions)
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_claims::protocols::jolt::lattice::BalancedIncChunking;

    #[test]
    fn physical_one_hot_prefix_order_matches_selector_reduction() {
        use jolt_field::{Fr, Ring};
        use jolt_poly::{eq_index_msb, MultilinearPoly};

        let columns = vec![
            (vec![Some(0), Some(2)], DigitZeroRow::Virtualized),
            (vec![Some(0), None], DigitZeroRow::Committed),
        ];
        let packed = pack_one_hot_columns(4, 4, columns);
        assert_eq!(
            packed.indices(),
            [None, Some(2), Some(0), None, None, None, None, None]
        );

        let selector = [Fr::from_u64(3), Fr::from_u64(5)];
        let logical = [Fr::from_u64(7), Fr::from_u64(11), Fr::from_u64(13)];
        let first = OneHotPolynomial::new(4, vec![None, Some(2)]).evaluate(&logical);
        let second = OneHotPolynomial::new(4, vec![Some(0), None]).evaluate(&logical);
        let expected = eq_index_msb(&selector, 0) * first + eq_index_msb(&selector, 1) * second;
        let point = [selector.as_slice(), logical.as_slice()].concat();
        assert_eq!(packed.evaluate(&point), expected);
    }

    #[test]
    fn balanced_chunks_and_carry_reconstruct_the_fused_increment() {
        let values = [
            -(1i128 << 64) + 1,
            -(1i128 << 63),
            -129,
            -128,
            -127,
            -1,
            0,
            1,
            127,
            128,
            129,
            (1i128 << 63) - 1,
            (1i128 << 64) - 1,
        ];
        for width in [4, 8] {
            let chunking = BalancedIncChunking::new(width).unwrap();
            let radix = 1i128 << width;
            for delta in values {
                let inc = FusedIncValue { delta };
                let digits = (0..chunking.chunk_count()).map(|index| {
                    let row = inc.balanced_digit_row(width, index) as i128;
                    if row < radix / 2 {
                        row
                    } else {
                        row - radix
                    }
                });
                let carry_row = inc.balanced_carry_row(width) as i128;
                let carry = if carry_row < radix / 2 {
                    carry_row
                } else {
                    carry_row - radix
                };
                let reconstructed = digits
                    .enumerate()
                    .fold(0, |sum, (index, digit)| sum + (digit << (width * index)))
                    + (carry << FUSED_INC_BITS);
                assert_eq!(reconstructed, delta, "width={width}, delta={delta}");
            }
        }
    }

    #[test]
    fn zero_fused_increment_uses_digit_zero_for_every_column() {
        let inc = FusedIncValue { delta: 0 };
        for width in [4, 8, 16, 32] {
            let chunking = BalancedIncChunking::new(width).unwrap();
            for index in 0..chunking.chunk_count() {
                assert_eq!(inc.balanced_digit_row(width, index), 0);
            }
            assert_eq!(inc.balanced_carry_row(width), 0);
        }
    }

    #[test]
    fn sparse_unit_positions_sort_ascending_on_construction() {
        use jolt_field::{Fr, Ring};
        use jolt_poly::MultilinearPoly;

        let poly = SparseUnitPolynomial::<Fr>::new(4, vec![9, 2, 11, 0, 2]);
        assert_eq!(poly.one_positions(), [0, 2, 2, 9, 11]);

        let mut yielded = Vec::new();
        poly.for_each_one(&mut |position| yielded.push(position));
        assert_eq!(yielded, [0, 2, 2, 9, 11]);

        let mut rows = vec![Vec::new(); 4];
        poly.for_each_row(2, &mut |row_index, row| rows[row_index] = row.to_vec());
        let expected = |bits: [u64; 4]| bits.map(Fr::from_u64);
        assert_eq!(rows[0], expected([1, 0, 1, 0]));
        assert_eq!(rows[1], expected([0, 0, 0, 0]));
        assert_eq!(rows[2], expected([0, 1, 0, 1]));
        assert_eq!(rows[3], expected([0, 0, 0, 0]));
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
    use jolt_claims::protocols::jolt::lattice::{
        precommitted_packing_plan, PrecommittedPackingShape,
    };
    use jolt_field::Fr as ClaimsFr;
    use jolt_field::Ring;
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

    /// Per-cell reconstruction weight of a `ProgramOneHot` sub-column: the value the
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
        let packing = precommitted_packing_plan(&shape).unwrap();
        let bytecode = packing.bytecode();
        let program_image_words = [0xdeadbeefu64, 0x0102030405060708];
        let one_positions = assemble_precommitted_witness::<Fr>(
            bytecode,
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
        for column in bytecode.packing().ids() {
            let num_vars = bytecode.logical_num_vars(*column).unwrap();
            let cells = 1usize << (num_vars - LOG_ROWS);
            for cell in 0..cells {
                let weight = cell_weight(&eq_lane, column, cell, LOG_ROWS, num_vars);
                for (r, eq) in eq_row.iter().enumerate() {
                    let position = bytecode
                        .packing()
                        .packed_index(column, (cell << LOG_ROWS) | r)
                        .unwrap();
                    if witness.contains(&position) {
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
