//! Prover-side packed (Akita) witness assembly: the `OneHotTrace` columns
//! from the witness plane's typed rows, the advice word objects, the
//! direct bounded-dense committed-program objects, and the shape-only
//! stand-ins the native openings take.

use std::sync::Arc;

use jolt_akita::TraceOneHotRows;
use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
use jolt_claims::protocols::jolt::lattice::packing::{
    advice_packing_plan, precommitted_packing_plan, PrecommittedPackingShape,
    PrefixPackedObjectPlan,
};
use jolt_claims::protocols::jolt::lattice::strategy::OneHotTraceLayoutPlan;
use jolt_claims::protocols::jolt::{JoltAdviceKind, JoltCommittedPolynomial, TracePolynomialOrder};
use jolt_field::{JoltField, Ring};
use jolt_openings::{CommitmentScheme, TransparentObjectSetup};
use jolt_poly::{MultilinearPoly, Polynomial};
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_witness::witnesses::{
    BalancedIncColumn, BytecodePc, FusedInc, LookupIndex, RaChunkSelector, RemappedRamAddress,
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

impl<F: JoltField> SparseUnitPolynomial<F> {
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

impl<F: JoltField> MultilinearPoly<F> for SparseUnitPolynomial<F> {
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
    bytecode_pc: BytecodePc,
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

impl PackedTraceRows {
    fn validate_dimensions<F: JoltField>(
        plan: &OneHotTraceLayoutPlan,
        log_k_chunk: usize,
        log_t: usize,
    ) -> Result<(), ProverError<F>> {
        if !matches!(log_k_chunk, 4 | 8) {
            return Err(ProverError::Unsupported {
                reason: "packed one-hot trace chunk width must be 4 or 8 bits",
            });
        }
        let logical_num_vars = log_t
            .checked_add(log_k_chunk)
            .ok_or(ProverError::Unsupported {
                reason: "packed one-hot trace dimensions overflow",
            })?;
        if plan.packing().logical_num_vars() != logical_num_vars {
            return Err(ProverError::InvariantViolation {
                reason: "OneHotTrace plan dimensions disagree with the witness dimensions",
            });
        }
        Ok(())
    }
}

impl TraceOneHotRows for PackedTraceRows {
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

/// Fills one row's selected-row bytes; returns whether the cycle makes a
/// remappable RAM access (the only per-row fact the caller still needs — the
/// bytecode column is total, so no cycle can be missing its slot).
fn fill_trace_row(
    row: OneHotTraceSourceRow,
    columns: &[OneHotTraceColumn],
    selected_rows: &mut [u8],
) -> bool {
    debug_assert_eq!(columns.len(), selected_rows.len());
    for (column, selected_row) in columns.iter().zip(selected_rows) {
        let row_index = match column {
            OneHotTraceColumn::Instruction(selector) => selector.chunk_u128(row.lookup_index.0),
            OneHotTraceColumn::Bytecode(selector) => selector.chunk_usize(row.bytecode_pc.0),
            OneHotTraceColumn::Ram(selector) => row
                .ram_address
                .0
                .map_or(0, |address| selector.chunk_usize(address as usize)),
            OneHotTraceColumn::Increment(column) => row.fused_inc.selected_row(*column),
        };
        debug_assert!(row_index <= u8::MAX as usize);
        *selected_row = row_index as u8;
    }
    row.ram_address.0.is_some()
}

/// Builds the row-major source for the native `OneHotTrace` commitment in the
/// plan's canonical semantic-column order.
#[tracing::instrument(skip_all, name = "assemble_one_hot_trace")]
pub fn assemble_one_hot_trace_rows<F: JoltField>(
    witness: &dyn JoltWitnessPlane<F>,
    plan: &OneHotTraceLayoutPlan,
    ra_layout: JoltRaPolynomialLayout,
    log_k_chunk: usize,
    log_t: usize,
) -> Result<Arc<dyn TraceOneHotRows>, ProverError<F>> {
    PackedTraceRows::validate_dimensions::<F>(plan, log_k_chunk, log_t)?;
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
                                if fill_trace_row(row, &columns, selected_rows) {
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
            return Ok(Arc::new(PackedTraceRows {
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
        if fill_trace_row(row, &columns, selected_rows) {
            ram_active_rows[row_index / u64::BITS as usize] |=
                1u64 << (row_index % u64::BITS as usize);
        }
    }
    Ok(Arc::new(PackedTraceRows {
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

fn commit_failed<F: JoltField>(error: impl ToString) -> ProverError<F> {
    ProverError::Verifier(
        jolt_verifier::VerifierError::FinalOpeningVerificationFailed {
            reason: error.to_string(),
        },
    )
}

/// One direct bounded-dense committed-program object.
#[derive(Clone)]
pub struct DirectProgramObject<PCS: CommitmentScheme> {
    pub plan: PrefixPackedObjectPlan,
    pub commitment: PCS::Output,
    pub hint: PCS::OpeningHint,
}

/// The precommitted direct program objects in canonical order: indexed
/// bytecode chunks followed by the program image. Built once at preprocessing
/// time and retained in
/// [`crate::CommittedProgramProverData`], so proving consumes the objects
/// directly.
#[derive(Clone)]
pub struct DirectProgramObjects<PCS: CommitmentScheme> {
    pub objects: Vec<DirectProgramObject<PCS>>,
}

/// Assembles and commits the direct bytecode chunks and program-image object.
pub fn commit_direct_program<PCS>(
    program: &JoltProgramPreprocessing,
    bytecode_chunk_count: usize,
    trace_order: TracePolynomialOrder,
) -> Result<DirectProgramObjects<PCS>, ProverError<PCS::Field>>
where
    PCS: CommitmentScheme + TransparentObjectSetup,
{
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
        trace_order,
        program_image_log_words: Some(image_words.len().ilog2() as usize),
    };
    let plan = precommitted_packing_plan(&shape).map_err(commit_failed)?;
    let mut chunk_coeffs = jolt_kernels::committed_program::build_committed_bytecode_chunk_coeffs(
        &program.bytecode.bytecode,
        bytecode_chunk_count,
        trace_order,
    )
    .map_err(commit_failed)?
    .into_iter();
    let objects = plan
        .objects()
        .map(|object_plan| {
            let id = object_plan.packing().ids()[0];
            let mut evaluations = match id {
                JoltCommittedPolynomial::BytecodeChunk(_) => {
                    chunk_coeffs.next().ok_or(ProverError::InvariantViolation {
                        reason: "missing direct bytecode chunk witness",
                    })?
                }
                JoltCommittedPolynomial::ProgramImageInit => image_words
                    .iter()
                    .map(|word| PCS::Field::from_u64(*word))
                    .collect(),
                _ => {
                    return Err(ProverError::InvariantViolation {
                        reason: "unexpected direct committed-program object",
                    })
                }
            };
            evaluations.resize(
                1usize << object_plan.packing().packed_num_vars(),
                PCS::Field::default(),
            );
            let witness = Polynomial::new(evaluations);
            let (setup, _verifier_setup) = PCS::transparent_object_setup(
                object_plan.packing().packed_num_vars(),
                object_plan.layout_digest(),
            )
            .map_err(commit_failed)?;
            let (commitment, hint) = PCS::commit(&witness, &setup).map_err(commit_failed)?;
            Ok(DirectProgramObject {
                plan: object_plan.clone(),
                commitment,
                hint,
            })
        })
        .collect::<Result<Vec<_>, ProverError<PCS::Field>>>()?;
    Ok(DirectProgramObjects { objects })
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
