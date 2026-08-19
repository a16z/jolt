//! Prover-side packed (Akita) witness assembly: the `OneHotTrace` columns
//! from the witness plane's typed rows and the sparse unit-valued auxiliary
//! objects (advice byte columns, the precommitted `ProgramOneHot`).

use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
use jolt_claims::protocols::jolt::lattice::geometry::WORD_BYTES;
use jolt_claims::protocols::jolt::lattice::{
    advice_bytes_packing_plan, precommitted_packing_plan, OneHotTraceLayoutPlan,
    PrecommittedPackingShape, PrefixPackedObjectPlan,
};
use jolt_claims::protocols::jolt::{BytecodeRegisterLane, JoltAdviceKind, JoltCommittedPolynomial};
use jolt_field::{Field, FixedByteSize};
use jolt_lookup_tables::{InstructionLookupTable, XLEN};
use jolt_openings::{CommitmentScheme, TransparentObjectSetup};
use jolt_poly::{MultilinearPoly, OneHotPolynomial};
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_riscv::{
    instructions::Noop, Flags, InstructionFlags, InterleavedBitsMarker, JoltInstruction,
    JoltInstructionRow, CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS,
};
use jolt_witness::witnesses::{
    BalancedIncLane, FusedInc, LookupIndex, MappedPc, RaChunkSelector, RemappedRamAddress,
};
use jolt_witness::{collect_bundles, JoltWitnessPlane, WitnessBundle};

use crate::ProverError;

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

impl<F: Field> SparseUnitPolynomial<F> {
    /// Sorts the positions ascending once here — the invariant
    /// `for_each_row`'s row scan and `for_each_one`'s yield order rely on.
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

impl<F: Field> MultilinearPoly<F> for SparseUnitPolynomial<F> {
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

/// The per-cycle sources every `OneHotTrace` column derives from: the
/// instruction's lookup index, the mapped bytecode PC, the remapped RAM word
/// address, and the fused increment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
struct OneHotTraceSourceRow {
    lookup_index: LookupIndex,
    mapped_pc: MappedPc,
    ram_address: RemappedRamAddress,
    fused_inc: FusedInc,
}

/// Digit-zero treatment of one `OneHotTrace` column
/// (`specs/digit-zero-virtualization.md`): virtualized families omit their
/// hot-row-zero cells from the committed polynomial (the reduction
/// reconstructs them from the column activation), RAM commits every row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DigitZeroRow {
    Committed,
    Virtualized,
}

/// Scatters per-column hot rows into the single prefix-packed `OneHotTrace`
/// polynomial: column `i` occupies selector slot `i`, unused capacity slots
/// stay empty, and virtualized columns' row-zero cells are omitted. Mirrors
/// the legacy prover's `pack_one_hot_columns` convention — the byte-diff
/// suite pins the two together.
fn pack_one_hot_columns(
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

/// Builds the physical prefix-packed `OneHotTrace` polynomial from the
/// witness plane's typed per-cycle rows, in the plan's canonical column
/// order. Every hot index fits the `u8` lane domain (`K` is at most 256).
#[tracing::instrument(skip_all, name = "assemble_one_hot_trace")]
pub fn assemble_one_hot_trace<F: Field>(
    witness: &dyn JoltWitnessPlane<F>,
    plan: &OneHotTraceLayoutPlan,
    ra_layout: JoltRaPolynomialLayout,
    log_k_chunk: usize,
    log_t: usize,
) -> Result<OneHotPolynomial, ProverError<F>> {
    let rows: Vec<OneHotTraceSourceRow> = collect_bundles(witness, 1usize << log_t)?;
    let k = 1usize << log_k_chunk;
    let hot_u8 = |hot: usize| -> Result<u8, ProverError<F>> {
        u8::try_from(hot).map_err(|_| ProverError::InvariantViolation {
            reason: "OneHotTrace K is at most the u8 lane domain",
        })
    };
    let mut columns = Vec::with_capacity(plan.packing().ids().len());
    for polynomial in plan.packing().ids() {
        let mut indices: Vec<Option<u8>> = Vec::with_capacity(rows.len());
        let digit_zero_row = match polynomial {
            JoltCommittedPolynomial::InstructionRa(index) => {
                let selector = RaChunkSelector::new(*index, ra_layout.instruction(), log_k_chunk)?;
                for row in &rows {
                    indices.push(Some(hot_u8(selector.chunk_u128(row.lookup_index.0))?));
                }
                DigitZeroRow::Virtualized
            }
            JoltCommittedPolynomial::BytecodeRa(index) => {
                let selector = RaChunkSelector::new(*index, ra_layout.bytecode(), log_k_chunk)?;
                for row in &rows {
                    let pc = row.mapped_pc.0.ok_or(ProverError::InvariantViolation {
                        reason: "OneHotTrace bytecode column requires a mapped PC on every cycle",
                    })?;
                    indices.push(Some(hot_u8(selector.chunk_usize(pc))?));
                }
                DigitZeroRow::Virtualized
            }
            JoltCommittedPolynomial::RamRa(index) => {
                let selector = RaChunkSelector::new(*index, ra_layout.ram(), log_k_chunk)?;
                for row in &rows {
                    indices.push(match row.ram_address.0 {
                        Some(address) => Some(hot_u8(selector.chunk_usize(address as usize))?),
                        None => None,
                    });
                }
                DigitZeroRow::Committed
            }
            JoltCommittedPolynomial::BalancedIncDigit(index) => {
                let lane = BalancedIncLane::Digit {
                    width: log_k_chunk,
                    index: *index,
                };
                for row in &rows {
                    indices.push(Some(hot_u8(row.fused_inc.hot_lane(lane))?));
                }
                DigitZeroRow::Virtualized
            }
            JoltCommittedPolynomial::BalancedIncCarry => {
                let lane = BalancedIncLane::Carry { width: log_k_chunk };
                for row in &rows {
                    indices.push(Some(hot_u8(row.fused_inc.hot_lane(lane))?));
                }
                DigitZeroRow::Virtualized
            }
            _ => {
                return Err(ProverError::InvariantViolation {
                    reason: "OneHotTrace plan contains only canonical columns",
                })
            }
        };
        columns.push((indices, digit_zero_row));
    }
    Ok(pack_one_hot_columns(
        k,
        plan.packing().slot_capacity(),
        columns,
    ))
}

/// A packed advice commitment object (`UntrustedAdviceOneHot` per proof,
/// `TrustedAdviceOneHot` precommitted): the byte one-hot column, its
/// canonical object plan, and its commitment data over the transparent
/// per-object setup.
pub struct AdviceOneHot<PCS: CommitmentScheme> {
    pub plan: PrefixPackedObjectPlan,
    pub byte_column: SparseUnitPolynomial<PCS::Field>,
    pub commitment: PCS::Output,
    pub hint: PCS::OpeningHint,
    pub setup: PCS::ProverSetup,
    pub word_vars: usize,
}

/// Builds a packed advice byte commitment object from raw advice bytes: per
/// `(place ‖ word)` row the hot value is the advice byte, zero-padded past
/// the actual advice length — the same zero padding the base word polynomial
/// carries. The setup is derived from the public advice shape, seeded by the
/// plan's layout digest, so both sides re-derive it byte-identically.
#[tracing::instrument(skip_all, name = "commit_advice_one_hot")]
pub fn commit_advice_one_hot<PCS>(
    kind: JoltAdviceKind,
    advice_bytes: &[u8],
    max_advice_bytes: usize,
) -> Result<AdviceOneHot<PCS>, ProverError<PCS::Field>>
where
    PCS: CommitmentScheme + TransparentObjectSetup,
{
    if advice_bytes.len() > max_advice_bytes {
        return Err(ProverError::Unsupported {
            reason: "advice bytes exceed the configured maximum advice size",
        });
    }
    let words = max_advice_bytes / 8;
    let word_vars = words.next_power_of_two().max(1).ilog2() as usize;
    let plan = advice_bytes_packing_plan(kind, word_vars).map_err(commit_failed)?;
    let cell_vars = plan.packing().packed_num_vars();
    let limb_bits = WORD_BYTES.ilog2() as usize;
    let mut one_positions = Vec::with_capacity(WORD_BYTES * words);
    for limb in 0..WORD_BYTES {
        for word_index in 0..words {
            let byte = advice_bytes
                .get(word_index * 8 + limb)
                .copied()
                .unwrap_or(0) as usize;
            one_positions.push((((byte << limb_bits) | limb) << word_vars) | word_index);
        }
    }
    let byte_column = SparseUnitPolynomial::new(cell_vars, one_positions);
    let (setup, _verifier_setup) =
        PCS::transparent_object_setup(cell_vars, plan.layout_digest()).map_err(commit_failed)?;
    let (commitment, hint) = PCS::commit(&byte_column, &setup).map_err(commit_failed)?;
    Ok(AdviceOneHot {
        plan,
        byte_column,
        commitment,
        hint,
        setup,
        word_vars,
    })
}

fn commit_failed<F: Field>(error: impl ToString) -> ProverError<F> {
    ProverError::Verifier(
        jolt_verifier::VerifierError::FinalOpeningVerificationFailed {
            reason: error.to_string(),
        },
    )
}

/// One precommitted `ProgramOneHot` commitment object: the object's packed
/// sub-column witness, its commitment/hint, the shape-exact transparent
/// setup, and its canonical plan.
#[derive(Clone)]
pub struct ProgramOneHotObject<PCS: CommitmentScheme> {
    pub plan: PrefixPackedObjectPlan,
    pub witness: SparseUnitPolynomial<PCS::Field>,
    pub commitment: PCS::Output,
    pub hint: PCS::OpeningHint,
    pub setup: PCS::ProverSetup,
}

/// The precommitted `ProgramOneHot` commitment objects (committed-program
/// mode) in canonical order: the bytecode-lane object, then the program-image
/// object. Built once at preprocessing time and retained in
/// [`crate::CommittedProgramProverData`], so proving consumes the objects
/// directly.
#[derive(Clone)]
pub struct ProgramOneHot<PCS: CommitmentScheme> {
    pub objects: Vec<ProgramOneHotObject<PCS>>,
}

/// Assembles and commits the `ProgramOneHot` objects from the full (public)
/// program: every bytecode sub-column, and the program image as its own
/// object, packed per the canonical `precommitted_packing_plan`. The imm
/// lane uses the field's canonical byte width, so negative immediates
/// (`p − |imm|`) reconstruct exactly.
#[tracing::instrument(skip_all, name = "commit_program_one_hot")]
pub fn commit_program_one_hot<PCS>(
    program: &JoltProgramPreprocessing,
    bytecode_chunk_count: usize,
) -> Result<ProgramOneHot<PCS>, ProverError<PCS::Field>>
where
    PCS: CommitmentScheme + TransparentObjectSetup,
{
    let imm_byte_width = <PCS::Field as FixedByteSize>::NUM_BYTES;
    let bytecode_len = program.bytecode.bytecode.len();
    if bytecode_chunk_count == 0 || !bytecode_len.is_multiple_of(bytecode_chunk_count) {
        return Err(ProverError::InvariantViolation {
            reason: "bytecode chunk count must divide bytecode length",
        });
    }
    let chunk_rows = bytecode_len / bytecode_chunk_count;
    if !chunk_rows.is_power_of_two() {
        return Err(ProverError::InvariantViolation {
            reason: "bytecode chunk row count must be a power of two",
        });
    }
    let log_bytecode_rows = chunk_rows.ilog2() as usize;
    let image_words = program_image_words_padded(program);
    let shape = PrecommittedPackingShape {
        bytecode_chunks: bytecode_chunk_count,
        log_bytecode_rows,
        imm_byte_width,
        program_image_log_words: Some(image_words.len().ilog2() as usize),
    };
    let plan = precommitted_packing_plan(&shape).map_err(commit_failed)?;
    let objects = plan
        .objects()
        .map(|object_plan| {
            let one_positions = assemble_precommitted_witness::<PCS::Field>(
                object_plan,
                &program.bytecode.bytecode,
                log_bytecode_rows,
                imm_byte_width,
                Some(&image_words),
            )?;
            let witness =
                SparseUnitPolynomial::new(object_plan.packing().packed_num_vars(), one_positions);
            let (setup, _verifier_setup) = PCS::transparent_object_setup(
                object_plan.packing().packed_num_vars(),
                object_plan.layout_digest(),
            )
            .map_err(commit_failed)?;
            let (commitment, hint) = PCS::commit(&witness, &setup).map_err(commit_failed)?;
            Ok(ProgramOneHotObject {
                plan: object_plan.clone(),
                witness,
                commitment,
                hint,
                setup,
            })
        })
        .collect::<Result<Vec<_>, ProverError<PCS::Field>>>()?;
    Ok(ProgramOneHot { objects })
}

/// The padded program-image words: the RAM preprocessing's bytecode words,
/// zero-padded to `committed_program_image_num_words` (the next power of two,
/// at least 2 — the packed word-domain convention legacy shares).
pub fn program_image_words_padded(program: &JoltProgramPreprocessing) -> Vec<u64> {
    let words = &program.ram.bytecode_words;
    let padded_len = words.len().next_power_of_two().max(2);
    let mut padded = words.clone();
    padded.resize(padded_len, 0);
    padded
}

/// Decode one bytecode row like the shared stage-value fold does (unknown
/// rows fall back to a no-op).
pub fn decode_row(row: &JoltInstructionRow) -> JoltInstruction {
    JoltInstruction::try_from(*row).unwrap_or(JoltInstruction::Noop(Noop(*row)))
}

/// The instruction flags in lane order (`flag as usize` — the enum's
/// declaration order, the same indexing `BYTECODE_LANE_LAYOUT` uses).
pub const INSTRUCTION_FLAG_ORDER: [InstructionFlags; NUM_INSTRUCTION_FLAGS] = [
    InstructionFlags::LeftOperandIsPC,
    InstructionFlags::RightOperandIsImm,
    InstructionFlags::LeftOperandIsRs1Value,
    InstructionFlags::RightOperandIsRs2Value,
    InstructionFlags::Branch,
    InstructionFlags::IsNoop,
];

// Compile-time pin: the hand-listed order IS the enum's discriminant order
// (the lane index the layout and the reconstruction verifier both use).
const _: () = {
    let mut index = 0;
    while index < NUM_INSTRUCTION_FLAGS {
        assert!(INSTRUCTION_FLAG_ORDER[index] as usize == index);
        index += 1;
    }
};

/// Scatters one precommitted `ProgramOneHot` object's sub-columns (per-chunk
/// bytecode lanes, or the program image) into one-positions of its packed
/// witness, per the object plan's canonical slots. Row domain per chunk is
/// `2^log_bytecode_rows` (bytecode rows, zero-padded); byte one-hot columns
/// encode padding as hot-lane-0 hot (never all-zero), the selector/flag
/// columns leave padding rows empty.
pub fn assemble_precommitted_witness<F: Field>(
    plan: &PrefixPackedObjectPlan,
    instructions: &[JoltInstructionRow],
    log_bytecode_rows: usize,
    imm_byte_width: usize,
    program_image_words: Option<&[u64]>,
) -> Result<Vec<usize>, ProverError<F>> {
    let rows = 1usize << log_bytecode_rows;
    let chunk_rows = |chunk: usize| -> &[JoltInstructionRow] {
        let start = (chunk * rows).min(instructions.len());
        let end = ((chunk + 1) * rows).min(instructions.len());
        &instructions[start..end]
    };
    let imm_limb_bits = imm_byte_width.ilog2() as usize;
    let imm_bytes = |imm: i128| -> Result<Vec<u8>, ProverError<F>> {
        let value = F::from_i128(imm);
        let bytes = value.to_bytes_le_vec();
        if bytes.len() < imm_byte_width || bytes[imm_byte_width..].iter().any(|byte| *byte != 0) {
            return Err(ProverError::InvariantViolation {
                reason: "immediate does not fit the canonical imm byte lane",
            });
        }
        Ok(bytes[..imm_byte_width].to_vec())
    };

    let mut one_positions = Vec::new();
    for column in plan.packing().ids() {
        let mut push = |local: usize| -> Result<(), ProverError<F>> {
            one_positions.push(
                plan.packing()
                    .packed_index(column, local)
                    .map_err(commit_failed)?,
            );
            Ok(())
        };
        match column {
            JoltCommittedPolynomial::BytecodeRegisterSelector { chunk, lane } => {
                for (row, instruction) in chunk_rows(*chunk).iter().enumerate() {
                    let register = match lane {
                        BytecodeRegisterLane::Rs1 => instruction.operands.rs1,
                        BytecodeRegisterLane::Rs2 => instruction.operands.rs2,
                        BytecodeRegisterLane::Rd => instruction.operands.rd,
                    };
                    if let Some(register) = register {
                        push(((register as usize) << log_bytecode_rows) | row)?;
                    }
                }
            }
            JoltCommittedPolynomial::BytecodeCircuitFlag { chunk, flag } => {
                for (row, instruction) in chunk_rows(*chunk).iter().enumerate() {
                    if decode_row(instruction).circuit_flags()[CIRCUIT_FLAGS[*flag]] {
                        push(row)?;
                    }
                }
            }
            JoltCommittedPolynomial::BytecodeInstructionFlag { chunk, flag } => {
                for (row, instruction) in chunk_rows(*chunk).iter().enumerate() {
                    if decode_row(instruction).instruction_flags()[INSTRUCTION_FLAG_ORDER[*flag]] {
                        push(row)?;
                    }
                }
            }
            JoltCommittedPolynomial::BytecodeLookupSelector { chunk } => {
                for (row, instruction) in chunk_rows(*chunk).iter().enumerate() {
                    if let Some(table) =
                        InstructionLookupTable::<XLEN>::lookup_table(&decode_row(instruction))
                    {
                        push((table.index() << log_bytecode_rows) | row)?;
                    }
                }
            }
            JoltCommittedPolynomial::BytecodeRafFlag { chunk } => {
                for (row, instruction) in chunk_rows(*chunk).iter().enumerate() {
                    if !decode_row(instruction)
                        .circuit_flags()
                        .is_interleaved_operands()
                    {
                        push(row)?;
                    }
                }
            }
            JoltCommittedPolynomial::BytecodeUnexpandedPcBytes { chunk } => {
                let instructions = chunk_rows(*chunk);
                let limb_bits = WORD_BYTES.ilog2() as usize;
                for limb in 0..WORD_BYTES {
                    for row in 0..rows {
                        let byte = instructions.get(row).map_or(0, |instruction| {
                            ((instruction.address as u64) >> (8 * limb)) as u8
                        }) as usize;
                        push((((byte << limb_bits) | limb) << log_bytecode_rows) | row)?;
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
                        push(
                            ((((byte as usize) << imm_limb_bits) | limb) << log_bytecode_rows)
                                | row,
                        )?;
                    }
                }
            }
            JoltCommittedPolynomial::ProgramImageBytes => {
                let words = program_image_words.ok_or(ProverError::InvariantViolation {
                    reason: "program image words missing for ProgramOneHot",
                })?;
                let limb_bits = WORD_BYTES.ilog2() as usize;
                let word_vars =
                    plan.logical_num_vars(*column)
                        .ok_or(ProverError::InvariantViolation {
                            reason: "the object plan carries no ProgramImageBytes arity",
                        })?
                        - 8
                        - limb_bits;
                debug_assert!(words.len() <= 1 << word_vars);
                for limb in 0..WORD_BYTES {
                    for word_index in 0..(1usize << word_vars) {
                        let byte = words
                            .get(word_index)
                            .map_or(0, |word| (word >> (8 * limb)) as u8)
                            as usize;
                        push((((byte << limb_bits) | limb) << word_vars) | word_index)?;
                    }
                }
            }
            _ => {
                return Err(ProverError::InvariantViolation {
                    reason: "column is not part of the precommitted packed witness",
                })
            }
        }
    }
    Ok(one_positions)
}
