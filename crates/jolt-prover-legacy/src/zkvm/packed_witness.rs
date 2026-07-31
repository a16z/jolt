//! Prover-side Akita witness assembly. `OneHotTrace` contains the uniform
//! row-major one-hot columns derived from the execution trace; auxiliary program/advice objects retain
//! sparse prefix-packed representations.

use jolt_claims::protocols::jolt::lattice::geometry::WORD_BYTES;
pub use jolt_claims::protocols::jolt::lattice::UNSIGNED_INC_BITS;
use jolt_claims::protocols::jolt::{BytecodeRegisterLane, JoltCommittedPolynomial};
use jolt_openings::PrefixPacking;
use jolt_riscv::{JoltInstructionRow, JoltTraceRow};
#[cfg(any(feature = "akita", test))]
use rayon::prelude::*;

use crate::field::JoltField;
use crate::utils::math::Math;
use crate::zkvm::instruction::{CircuitFlags, Flags, InstructionLookup, InterleavedBitsMarker};
use crate::zkvm::lookup_table::LookupTables;
#[cfg(any(feature = "akita", test))]
use allocative::Allocative;
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
    pub fn from_trace_row(row: &JoltTraceRow) -> Self {
        Self::from_trace_row_with_store(row).0
    }

    /// [`from_trace_row`](Self::from_trace_row) plus the store selector itself, so
    /// witness generation and the read-raf fused stages read one
    /// predicate: the same `OpFlags(Store)` circuit flag the sumcheck
    /// selector opens.
    pub fn from_trace_row_with_store(row: &JoltTraceRow) -> (Self, bool) {
        let store = Flags::circuit_flags(row)[CircuitFlags::Store];
        let ram_delta = if store {
            row.ram_write_value() as i128 - row.ram_read_value() as i128
        } else {
            0
        };
        let rd_pre_value = row.rd_pre_value();
        let rd_post_value = row.rd_write_value();
        let rd_delta = rd_post_value as i128 - rd_pre_value as i128;
        // One fused column can serve both inc consumers only because no
        // cycle increments RAM and rd at once (every RMW instruction lowers
        // into a sequence whose RAM-writing step is a plain store). A
        // violation means an instruction shape the fused encoding cannot
        // represent — fail here, not with an opaque sumcheck mismatch.
        debug_assert_eq!(
            store,
            row.is_store(),
            "Store circuit flag disagrees with the trace-row class: {row:?}"
        );
        debug_assert!(
            if store { rd_delta == 0 } else { ram_delta == 0 },
            "trace row increments both RAM and rd; the fused inc encoding cannot represent it: {row:?}"
        );
        let delta = if store { ram_delta } else { rd_delta };
        (Self { delta }, store)
    }

    fn balanced_bias(width: usize) -> i128 {
        debug_assert!(width > 0 && UNSIGNED_INC_BITS.is_multiple_of(width));
        let radix = 1i128 << width;
        (radix / 2) * (((1i128 << UNSIGNED_INC_BITS) - 1) / (radix - 1))
    }

    fn biased_for_balanced_digits(self, width: usize) -> i128 {
        debug_assert!(self.delta.unsigned_abs() < 1u128 << UNSIGNED_INC_BITS);
        self.delta + Self::balanced_bias(width)
    }

    /// The centered radix-`2^width` digit encoded modulo the radix.
    pub fn balanced_chunk_hot_lane_bits(self, width: usize, index: usize) -> usize {
        let radix = 1i128 << width;
        let mask = radix - 1;
        let standard_digit = (self.biased_for_balanced_digits(width) >> (width * index)) & mask;
        ((standard_digit + radix / 2) & mask) as usize
    }

    /// The signed carry above bit 63, encoded modulo the chunk radix.
    pub fn balanced_carry_hot_lane_bits(self, width: usize) -> usize {
        let radix = 1i128 << width;
        let carry = self.biased_for_balanced_digits(width) >> UNSIGNED_INC_BITS;
        debug_assert!((-1..=1).contains(&carry));
        carry.rem_euclid(radix) as usize
    }
}

/// Sign-magnitude storage for the per-cycle fused increment stream.
///
/// Every fused delta is the difference of two `u64` values, so its magnitude
/// fits in one limb. Signs are stored separately at one bit per cycle.
#[cfg(any(feature = "akita", test))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Allocative)]
pub struct FusedIncDeltas {
    magnitudes: Vec<u64>,
    negative_words: Vec<u64>,
}

#[cfg(any(feature = "akita", test))]
impl FusedIncDeltas {
    #[cfg(feature = "akita")]
    /// Builds the packed stream directly from trace rows without a full-width
    /// signed temporary.
    pub fn from_trace(trace: &[JoltTraceRow]) -> Self {
        let mut magnitudes = vec![0; trace.len()];
        let mut negative_words = vec![0; trace.len().div_ceil(64)];
        magnitudes
            .par_chunks_mut(64)
            .zip(negative_words.par_iter_mut())
            .zip(trace.par_chunks(64))
            .for_each(|((magnitudes, negative_word), rows)| {
                let mut signs = 0;
                for (index, (magnitude, row)) in magnitudes.iter_mut().zip(rows).enumerate() {
                    let delta = FusedIncValue::from_trace_row(row).delta;
                    let absolute = delta.unsigned_abs();
                    debug_assert!(absolute <= u64::MAX as u128);
                    *magnitude = absolute as u64;
                    if delta.is_negative() {
                        signs |= 1 << index;
                    }
                }
                *negative_word = signs;
            });
        Self {
            magnitudes,
            negative_words,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_values(values: &[i128]) -> Self {
        assert!(values
            .iter()
            .all(|value| value.unsigned_abs() <= u64::MAX as u128));
        let magnitudes = values
            .iter()
            .map(|value| value.unsigned_abs() as u64)
            .collect();
        let negative_words = values
            .chunks(64)
            .map(|chunk| {
                chunk.iter().enumerate().fold(0, |word, (index, value)| {
                    word | (u64::from(value.is_negative()) << index)
                })
            })
            .collect();
        Self {
            magnitudes,
            negative_words,
        }
    }

    #[inline]
    /// Number of cycle coefficients in the stream.
    pub fn len(&self) -> usize {
        self.magnitudes.len()
    }

    #[inline]
    /// Whether the stream contains no cycle coefficients.
    pub fn is_empty(&self) -> bool {
        self.magnitudes.is_empty()
    }

    #[inline]
    /// Decodes one signed coefficient.
    pub fn value(&self, index: usize) -> i128 {
        let magnitude = self.magnitudes[index] as i128;
        if self.negative_words[index / 64] >> (index % 64) & 1 == 1 {
            -magnitude
        } else {
            magnitude
        }
    }

    #[cfg(feature = "akita")]
    pub(crate) fn magnitudes(&self) -> &[u64] {
        &self.magnitudes
    }

    #[cfg(feature = "akita")]
    pub(crate) fn negative_words(&self) -> &[u64] {
        &self.negative_words
    }

    #[cfg(test)]
    pub(crate) fn par_map_values<T, M>(&self, map: M) -> Vec<T>
    where
        T: Send,
        M: Fn(i128) -> T + Send + Sync,
    {
        let mut output = Vec::with_capacity(self.len());
        (
            output.spare_capacity_mut().par_chunks_mut(64),
            self.magnitudes.par_chunks(64),
            self.negative_words.par_iter(),
        )
            .into_par_iter()
            .for_each(|(output, magnitudes, &negative_word)| {
                for (index, (output, &magnitude)) in output.iter_mut().zip(magnitudes).enumerate() {
                    let magnitude = magnitude as i128;
                    let value = if negative_word >> index & 1 == 1 {
                        -magnitude
                    } else {
                        magnitude
                    };
                    output.write(map(value));
                }
            });
        // SAFETY: The parallel chunks cover every spare slot exactly once,
        // and each slot is initialized before the vector length is published.
        unsafe { output.set_len(self.len()) };
        output
    }
}

/// Scatters the precommitted `ProgramOneHot` sub-columns (per-chunk bytecode lanes
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
                    .ok_or_else(|| "program image words missing for ProgramOneHot".to_string())?;
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
    Ok(one_positions)
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_claims::protocols::jolt::lattice::UnsignedIncChunking;

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
            let encoding = UnsignedIncChunking::new(width).unwrap();
            let radix = 1i128 << width;
            let signed = |lane: usize| {
                let lane = lane as i128;
                if lane < radix / 2 {
                    lane
                } else {
                    lane - radix
                }
            };
            for (cycle, delta) in values.into_iter().enumerate() {
                let inc = FusedIncValue { delta };
                let mut reconstructed = 0i128;
                for index in 0..encoding.chunk_count() {
                    let lane = inc.balanced_chunk_hot_lane_bits(width, index);
                    assert!(lane < 1 << width, "cycle {cycle}, width {width}");
                    reconstructed += signed(lane) << (width * index);
                }
                let carry = signed(inc.balanced_carry_hot_lane_bits(width));
                assert!((-1..=1).contains(&carry));
                reconstructed += carry << UNSIGNED_INC_BITS;
                assert_eq!(reconstructed, delta, "cycle {cycle}, width {width}");
            }
        }
    }

    #[test]
    fn zero_increment_uses_only_implicit_default_lanes() {
        let zero = FusedIncValue { delta: 0 };
        for width in [4, 8] {
            let encoding = UnsignedIncChunking::new(width).unwrap();
            for index in 0..encoding.chunk_count() {
                assert_eq!(zero.balanced_chunk_hot_lane_bits(width, index), 0);
            }
            assert_eq!(zero.balanced_carry_hot_lane_bits(width), 0);
        }
    }

    #[test]
    fn packed_fused_inc_deltas_round_trip_boundaries() {
        let values = [
            -(1i128 << 64) + 1,
            -(1i128 << 63),
            -1,
            0,
            1,
            (1i128 << 63) - 1,
            (1i128 << 64) - 1,
        ];
        let packed = FusedIncDeltas::from_values(&values);

        assert_eq!(packed.len(), values.len());
        assert_eq!(
            (0..values.len())
                .map(|index| packed.value(index))
                .collect::<Vec<_>>(),
            values
        );
        assert_eq!(packed.par_map_values(|value| value), values);
    }

    #[test]
    fn sparse_unit_positions_sort_ascending_on_construction() {
        use jolt_field::{Fr, FromPrimitiveInt};
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
