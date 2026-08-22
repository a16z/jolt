//! Prover-side packed (Akita) witness assembly: the `OneHotTrace` columns
//! from the witness plane's typed rows, the advice word objects, the
//! sparse unit-valued precommitted `ProgramOneHot`, and the shape-only
//! stand-ins the native openings take.

use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
use jolt_claims::protocols::jolt::lattice::geometry::WORD_BYTES;
use jolt_claims::protocols::jolt::lattice::packing::{
    advice_packing_plan, precommitted_packing_plan, PrecommittedPackingShape,
    PrefixPackedObjectPlan,
};
use jolt_claims::protocols::jolt::lattice::strategy::OneHotTraceLayoutPlan;
use jolt_claims::protocols::jolt::{BytecodeRegisterLane, JoltAdviceKind, JoltCommittedPolynomial};
use jolt_field::{Field, FixedByteSize, FromPrimitiveInt};
use jolt_lookup_tables::{InstructionLookupTable, XLEN};
use jolt_openings::{CommitmentScheme, TransparentObjectSetup};
use jolt_poly::{MultilinearPoly, Polynomial};
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_riscv::{
    instructions::Noop, Flags, InstructionFlags, InterleavedBitsMarker, JoltInstruction,
    JoltInstructionRow, CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS,
};
use jolt_witness::witnesses::{
    BalancedIncColumn, FusedInc, LookupIndex, MappedPc, RaChunkSelector, RemappedRamAddress,
};
use jolt_witness::{collect_bundles, JoltWitnessPlane, WitnessBundle};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

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

#[derive(Clone, Copy)]
enum OneHotTraceColumn {
    Instruction(RaChunkSelector),
    Bytecode(RaChunkSelector),
    Ram(RaChunkSelector),
    Increment(BalancedIncColumn),
}

struct PackedTraceRows {
    num_rows: usize,
    num_columns: usize,
    selected_rows: Vec<u8>,
    ram_active_rows: Vec<u64>,
    ram_digit_zero_mask: u64,
}

impl jolt_akita::TraceOneHotRows for PackedTraceRows {
    fn num_rows(&self) -> usize {
        self.num_rows
    }

    fn num_columns(&self) -> usize {
        self.num_columns
    }

    fn fill_row(&self, row: usize, selected_rows: &mut [u8]) {
        let start = row * self.num_columns;
        selected_rows.copy_from_slice(&self.selected_rows[start..start + self.num_columns]);
    }

    fn fill_rows(&self, row_start: usize, selected_rows: &mut [u8]) {
        debug_assert_eq!(selected_rows.len() % self.num_columns, 0);
        let start = row_start * self.num_columns;
        selected_rows.copy_from_slice(&self.selected_rows[start..start + selected_rows.len()]);
    }

    fn committed_digit_zero_mask(&self, row: usize) -> u64 {
        let active = self.ram_active_rows[row / u64::BITS as usize]
            & (1u64 << (row % u64::BITS as usize))
            != 0;
        if active {
            self.ram_digit_zero_mask
        } else {
            0
        }
    }
}

#[derive(Clone, Copy)]
struct TraceRowStatus {
    bytecode_valid: bool,
    ram_active: bool,
}

fn fill_trace_row(
    row: OneHotTraceSourceRow,
    columns: &[OneHotTraceColumn],
    selected_rows: &mut [u8],
) -> TraceRowStatus {
    debug_assert_eq!(columns.len(), selected_rows.len());
    let mut valid = true;
    for (column, selected_row) in columns.iter().zip(selected_rows) {
        let row_index = match column {
            OneHotTraceColumn::Instruction(selector) => selector.chunk_u128(row.lookup_index.0),
            OneHotTraceColumn::Bytecode(selector) => {
                if let Some(pc) = row.mapped_pc.0 {
                    selector.chunk_usize(pc)
                } else {
                    valid = false;
                    0
                }
            }
            OneHotTraceColumn::Ram(selector) => row
                .ram_address
                .0
                .map_or(0, |address| selector.chunk_usize(address as usize)),
            OneHotTraceColumn::Increment(column) => row.fused_inc.selected_row(*column),
        };
        debug_assert!(row_index <= u8::MAX as usize);
        *selected_row = row_index as u8;
    }
    TraceRowStatus {
        bytecode_valid: valid,
        ram_active: row.ram_address.0.is_some(),
    }
}

/// Builds the row-major source for the native `OneHotTrace` commitment in the
/// plan's canonical semantic-column order.
#[tracing::instrument(skip_all, name = "assemble_one_hot_trace")]
pub fn assemble_one_hot_trace_rows<F: Field>(
    witness: &dyn JoltWitnessPlane<F>,
    plan: &OneHotTraceLayoutPlan,
    ra_layout: JoltRaPolynomialLayout,
    log_k_chunk: usize,
    log_t: usize,
) -> Result<std::sync::Arc<dyn jolt_akita::TraceOneHotRows>, ProverError<F>> {
    let num_rows = 1usize << log_t;
    let num_columns = plan.packing().ids().len();
    let ram_digit_zero_mask = plan
        .ranges()
        .ram
        .clone()
        .fold(0u64, |mask, column| mask | (1u64 << column));
    let mut columns = Vec::with_capacity(num_columns);
    for polynomial in plan.packing().ids() {
        match polynomial {
            JoltCommittedPolynomial::InstructionRa(index) => {
                let selector = RaChunkSelector::new(*index, ra_layout.instruction(), log_k_chunk)?;
                columns.push(OneHotTraceColumn::Instruction(selector));
            }
            JoltCommittedPolynomial::BytecodeRa(index) => {
                let selector = RaChunkSelector::new(*index, ra_layout.bytecode(), log_k_chunk)?;
                columns.push(OneHotTraceColumn::Bytecode(selector));
            }
            JoltCommittedPolynomial::RamRa(index) => {
                let selector = RaChunkSelector::new(*index, ra_layout.ram(), log_k_chunk)?;
                columns.push(OneHotTraceColumn::Ram(selector));
            }
            JoltCommittedPolynomial::BalancedIncDigit(index) => {
                columns.push(OneHotTraceColumn::Increment(BalancedIncColumn::Digit {
                    width: log_k_chunk,
                    index: *index,
                }));
            }
            JoltCommittedPolynomial::BalancedIncCarry => {
                columns.push(OneHotTraceColumn::Increment(BalancedIncColumn::Carry {
                    width: log_k_chunk,
                }));
            }
            _ => {
                return Err(ProverError::InvariantViolation {
                    reason: "OneHotTrace plan contains only canonical columns",
                })
            }
        }
    }

    let mut selected_rows = vec![0u8; num_rows * num_columns];
    let mut ram_active_rows = vec![0u64; num_rows.div_ceil(u64::BITS as usize)];
    #[cfg(feature = "parallel")]
    if let Some(access) = witness.random_access() {
        if num_rows <= access.cycles() {
            let extraction_error = std::sync::Mutex::new(None);
            let bytecode_rows_valid = std::sync::atomic::AtomicBool::new(true);
            selected_rows
                .par_chunks_mut(num_columns * u64::BITS as usize)
                .zip(ram_active_rows.par_iter_mut())
                .enumerate()
                .for_each(|(word_index, (word_rows, ram_active_word))| {
                    for (row_offset, selected_rows) in
                        word_rows.chunks_exact_mut(num_columns).enumerate()
                    {
                        let row_index = word_index * u64::BITS as usize + row_offset;
                        match access.window::<OneHotTraceSourceRow>(row_index) {
                            Ok(row) => {
                                let status = fill_trace_row(row, &columns, selected_rows);
                                if !status.bytecode_valid {
                                    bytecode_rows_valid
                                        .store(false, std::sync::atomic::Ordering::Relaxed);
                                }
                                if status.ram_active {
                                    *ram_active_word |= 1u64 << row_offset;
                                }
                            }
                            Err(error) => {
                                if let Ok(mut guard) = extraction_error.try_lock() {
                                    let _ = guard.get_or_insert(error);
                                }
                            }
                        }
                    }
                });
            #[expect(clippy::unwrap_used, reason = "no lock user can panic")]
            if let Some(error) = extraction_error.into_inner().unwrap() {
                return Err(error.into());
            }
            if !bytecode_rows_valid.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(ProverError::InvariantViolation {
                    reason: "OneHotTrace bytecode column requires a mapped PC on every cycle",
                });
            }
            return Ok(std::sync::Arc::new(PackedTraceRows {
                num_rows,
                num_columns,
                selected_rows,
                ram_active_rows,
                ram_digit_zero_mask,
            }));
        }
    }

    let rows: Vec<OneHotTraceSourceRow> = collect_bundles(witness, num_rows)?;
    for (row_index, (row, selected_rows)) in rows
        .into_iter()
        .zip(selected_rows.chunks_exact_mut(num_columns))
        .enumerate()
    {
        let status = fill_trace_row(row, &columns, selected_rows);
        if !status.bytecode_valid {
            return Err(ProverError::InvariantViolation {
                reason: "OneHotTrace bytecode column requires a mapped PC on every cycle",
            });
        }
        if status.ram_active {
            ram_active_rows[row_index / u64::BITS as usize] |=
                1u64 << (row_index % u64::BITS as usize);
        }
    }
    Ok(std::sync::Arc::new(PackedTraceRows {
        num_rows,
        num_columns,
        selected_rows,
        ram_active_rows,
        ram_digit_zero_mask,
    }))
}

/// One advice-word commitment object: one field coefficient per
/// canonical little-endian `u64`, embedded in slot zero when Akita's dense
/// schedule floor exceeds the logical word arity.
pub struct AdviceObject<PCS: CommitmentScheme> {
    pub plan: PrefixPackedObjectPlan,
    pub polynomial: Polynomial<PCS::Field>,
    pub commitment: PCS::Output,
    pub hint: PCS::OpeningHint,
    pub setup: PCS::ProverSetup,
    pub word_vars: usize,
}

/// Builds the canonical zero-padded advice-word commitment. The setup
/// is derived from the public advice shape with the same fixed seed on both
/// sides (the setup is transparent).
pub fn commit_advice<PCS>(
    kind: JoltAdviceKind,
    advice_bytes: &[u8],
    max_advice_bytes: usize,
) -> Result<AdviceObject<PCS>, ProverError<PCS::Field>>
where
    PCS: CommitmentScheme + TransparentObjectSetup,
{
    let words = common::advice::canonical_advice_words(advice_bytes, max_advice_bytes)
        .map_err(commit_failed)?;
    let word_vars = words.len().ilog2() as usize;
    let plan = advice_packing_plan(kind, word_vars).map_err(commit_failed)?;
    let physical_vars = plan.packing().packed_num_vars();
    let (setup, _) = PCS::transparent_object_setup(physical_vars, plan.layout_digest())
        .map_err(commit_failed)?;
    let mut evaluations = vec![PCS::Field::default(); 1usize << physical_vars];
    for (evaluation, word) in evaluations.iter_mut().zip(words) {
        *evaluation = PCS::Field::from_u64(word);
    }
    let polynomial = Polynomial::new(evaluations);
    let (commitment, hint) = PCS::commit(&polynomial, &setup).map_err(commit_failed)?;
    Ok(AdviceObject {
        plan,
        polynomial,
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
    pub shape: PrecommittedPackingShape,
    pub objects: Vec<ProgramOneHotObject<PCS>>,
}

/// Assembles and commits `ProgramOneHot` from the full (public) program:
/// every bytecode sub-column plus the program image, packed per the canonical
/// `precommitted_packing`. The imm lane uses the field's canonical byte
/// width, so negative immediates (`p − |imm|`) reconstruct exactly.
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
    Ok(ProgramOneHot { shape, objects })
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

/// Scatters the precommitted `ProgramOneHot` sub-columns (per-chunk bytecode
/// lanes and the program image) into one-positions of the packed precommitted
/// witness, per the canonical `precommitted_packing` slots. Row domain per
/// chunk is `2^log_bytecode_rows` (bytecode rows, zero-padded); byte one-hot
/// columns encode padding by selecting row zero (never all-zero), the
/// selector/flag columns leave padding rows empty.
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
    let packed_index = |column: &JoltCommittedPolynomial, local: usize| {
        plan.packing()
            .packed_index(column, local)
            .map_err(commit_failed::<F>)
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
                    if decode_row(instruction).circuit_flags()[CIRCUIT_FLAGS[*flag]] {
                        one_positions.push(packed_index(column, row)?);
                    }
                }
            }
            JoltCommittedPolynomial::BytecodeInstructionFlag { chunk, flag } => {
                for (row, instruction) in chunk_rows(*chunk).iter().enumerate() {
                    if decode_row(instruction).instruction_flags()[INSTRUCTION_FLAG_ORDER[*flag]] {
                        one_positions.push(packed_index(column, row)?);
                    }
                }
            }
            JoltCommittedPolynomial::BytecodeLookupSelector { chunk } => {
                for (row, instruction) in chunk_rows(*chunk).iter().enumerate() {
                    if let Some(table) =
                        InstructionLookupTable::<XLEN>::lookup_table(&decode_row(instruction))
                    {
                        one_positions.push(packed_index(
                            column,
                            (table.index() << log_bytecode_rows) | row,
                        )?);
                    }
                }
            }
            JoltCommittedPolynomial::BytecodeRafFlag { chunk } => {
                for (row, instruction) in chunk_rows(*chunk).iter().enumerate() {
                    if !decode_row(instruction)
                        .circuit_flags()
                        .is_interleaved_operands()
                    {
                        one_positions.push(packed_index(column, row)?);
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
                let words = program_image_words.ok_or(ProverError::InvariantViolation {
                    reason: "program image words missing for ProgramOneHot",
                })?;
                let limb_bits = WORD_BYTES.ilog2() as usize;
                let word_vars =
                    plan.logical_num_vars(*column)
                        .ok_or(ProverError::InvariantViolation {
                            reason: "program image is missing its logical arity",
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
                        one_positions.push(packed_index(
                            column,
                            (((byte << limb_bits) | limb) << word_vars) | word_index,
                        )?);
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
