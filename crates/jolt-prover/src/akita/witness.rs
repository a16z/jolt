//! Prover-side packed (Akita) witness assembly: the `OneHotTrace` columns
//! from the witness plane's typed rows, the advice word objects, the
//! direct bounded-dense committed-program objects.

#[cfg(feature = "parallel")]
use std::sync::Mutex;
use std::{collections::HashMap, sync::Arc};

use jolt_akita::TraceOneHotRows;
use jolt_claims::protocols::jolt::lattice::packing::{
    advice_packing_plan, committed_program_packing_plan, PrefixPackedObjectPlan,
};
use jolt_claims::protocols::jolt::lattice::strategy::OneHotTraceLayoutPlan;
use jolt_claims::protocols::jolt::{JoltAdviceKind, JoltCommittedPolynomial, TracePolynomialOrder};
use jolt_field::{JoltField, Ring};
use jolt_openings::{CommitmentScheme, TransparentObjectSetup};
use jolt_poly::Polynomial;
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_witness::witnesses::{
    write_ra_chunks, BytecodePc, FusedInc, LookupIndex, RemappedRamAddress,
};
use jolt_witness::{collect_bundles, JoltWitnessPlane, RandomAccessRows, WitnessBundle};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::ProverError;

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

/// Column counts of one row in the plan's canonical order, which
/// `OneHotTraceLayout::plan` fixes: instruction chunks, balanced-increment
/// digits then their carry, bytecode chunks, and the remaining RAM chunks.
#[derive(Clone, Copy)]
struct OneHotTraceRowLayout {
    chunk_bits: usize,
    instruction: usize,
    increment: usize,
    bytecode: usize,
}

impl OneHotTraceRowLayout {
    fn new(plan: &OneHotTraceLayoutPlan, chunk_bits: usize) -> Self {
        let ranges = plan.ranges();
        Self {
            chunk_bits,
            instruction: ranges.instruction.len(),
            increment: ranges.balanced_inc.len() + 1,
            bytecode: ranges.bytecode.len(),
        }
    }

    /// Fills one row's selected-row bytes; returns whether the cycle makes a
    /// remappable RAM access (the only per-row fact the caller still needs —
    /// the bytecode column is total, so no cycle can be missing its slot).
    fn fill_row(self, row: OneHotTraceSourceRow, selected_rows: &mut [u8]) -> bool {
        let (instruction, rest) = selected_rows.split_at_mut(self.instruction);
        let (increment, rest) = rest.split_at_mut(self.increment);
        let (bytecode, ram) = rest.split_at_mut(self.bytecode);
        write_ra_chunks(row.lookup_index.0, self.chunk_bits, instruction);
        row.fused_inc
            .write_selected_rows(self.chunk_bits, increment);
        write_ra_chunks(row.bytecode_pc.0 as u128, self.chunk_bits, bytecode);
        match row.ram_address.0 {
            Some(address) => write_ra_chunks(u128::from(address), self.chunk_bits, ram),
            None => ram.fill(0),
        }
        row.ram_address.0.is_some()
    }

    /// A filled row's committed entries: its nonzero selected rows, plus every
    /// RAM chunk of an active access, which commits row zero.
    fn committed_entries(self, selected_rows: &[u8], ram_active: bool) -> usize {
        let nonzero = |rows: &[u8]| rows.iter().filter(|&&row| row != 0).count();
        let (rows, ram) = selected_rows.split_at(self.instruction + self.increment + self.bytecode);
        nonzero(rows) + if ram_active { ram.len() } else { nonzero(ram) }
    }
}

struct PackedTraceRows {
    num_rows: usize,
    num_columns: usize,
    selected_rows: Vec<u8>,
    ram_active_rows: Vec<u64>,
    ram_digit_zero_mask: u64,
    hot_entries: usize,
    zero_suffix_start: usize,
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

    fn packed_selectors(&self) -> Option<jolt_akita::TracePackedSelectors<'_>> {
        Some(
            jolt_akita::TracePackedSelectors::new_with_precomputed_metrics(
                &self.selected_rows,
                &self.ram_active_rows,
                self.ram_digit_zero_mask,
                self.hot_entries,
                self.zero_suffix_start,
            ),
        )
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

/// The random access the rows fill straight from: a parallel build whose
/// witness covers every row. Otherwise the assembly first collects every
/// cycle's source row, a second trace-sized allocation.
fn direct_row_access<F: JoltField>(
    witness: &dyn JoltWitnessPlane<F>,
    num_rows: usize,
) -> Option<RandomAccessRows> {
    witness
        .random_access()
        .filter(|access| cfg!(feature = "parallel") && num_rows <= access.cycles())
}

/// Whether [`assemble_one_hot_trace_rows`] fills the `2^log_t` rows straight
/// from the witness, so that the rows are its only trace-sized allocation.
pub fn fills_one_hot_trace_rows_directly<F: JoltField>(
    witness: &dyn JoltWitnessPlane<F>,
    log_t: usize,
) -> bool {
    direct_row_access(witness, 1usize << log_t).is_some()
}

/// Builds the row-major source for the native `OneHotTrace` commitment in the
/// plan's canonical semantic-column order.
#[tracing::instrument(skip_all, name = "assemble_one_hot_trace")]
pub fn assemble_one_hot_trace_rows<F: JoltField>(
    witness: &dyn JoltWitnessPlane<F>,
    plan: &OneHotTraceLayoutPlan,
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
    let layout = OneHotTraceRowLayout::new(plan, log_k_chunk);

    let random_access = witness.random_access();
    let zero_suffix_start = if let Some(access) = random_access.as_ref() {
        let physical_rows = access.physical_rows().min(num_rows);
        if physical_rows < num_rows {
            let padding = access.window::<OneHotTraceSourceRow>(physical_rows)?;
            let mut selected = vec![0u8; num_columns];
            if layout.fill_row(padding, &mut selected) || selected.iter().any(|&row| row != 0) {
                num_rows
            } else {
                physical_rows
            }
        } else {
            num_rows
        }
    } else {
        num_rows
    };
    let mut selected_rows = vec![0u8; num_rows * num_columns];
    let mut ram_active_rows = vec![0u64; num_rows.div_ceil(u64::BITS as usize)];
    #[cfg(feature = "profiling")]
    tracing::info!(
        selected_bytes = selected_rows.capacity(),
        ram_active_bytes = size_of_val(ram_active_rows.as_slice()),
        "one-hot trace rows"
    );
    #[cfg(feature = "parallel")]
    if let Some(access) = direct_row_access(witness, num_rows) {
        let extraction_error = Mutex::new(None);
        let active_lane_count = zero_suffix_start * num_columns;
        let active_word_count = zero_suffix_start.div_ceil(u64::BITS as usize);
        let hot_entries = selected_rows[..active_lane_count]
            .par_chunks_mut(num_columns * u64::BITS as usize)
            .zip(ram_active_rows[..active_word_count].par_iter_mut())
            .enumerate()
            .map(|(word_index, (word_rows, ram_active_word))| {
                let mut hot_entries = 0usize;
                for (row_offset, selected_rows) in
                    word_rows.chunks_exact_mut(num_columns).enumerate()
                {
                    let row_index = word_index * u64::BITS as usize + row_offset;
                    match access.window::<OneHotTraceSourceRow>(row_index) {
                        Ok(row) => {
                            let ram_active = layout.fill_row(row, selected_rows);
                            if ram_active {
                                *ram_active_word |= 1u64 << row_offset;
                            }
                            hot_entries += layout.committed_entries(selected_rows, ram_active);
                        }
                        Err(error) => {
                            if let Ok(mut guard) = extraction_error.try_lock() {
                                let _ = guard.get_or_insert(error);
                            }
                        }
                    }
                }
                hot_entries
            })
            .sum();
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
            hot_entries,
            zero_suffix_start,
        }));
    }

    let rows: Vec<OneHotTraceSourceRow> = collect_bundles(witness, num_rows)?;
    let mut hot_entries = 0usize;
    for (row_index, (row, selected_rows)) in rows
        .into_iter()
        .zip(selected_rows.chunks_exact_mut(num_columns))
        .enumerate()
    {
        let ram_active = layout.fill_row(row, selected_rows);
        if ram_active {
            ram_active_rows[row_index / u64::BITS as usize] |=
                1u64 << (row_index % u64::BITS as usize);
        }
        hot_entries += layout.committed_entries(selected_rows, ram_active);
    }
    Ok(Arc::new(PackedTraceRows {
        num_rows,
        num_columns,
        selected_rows,
        ram_active_rows,
        ram_digit_zero_mask,
        hot_entries,
        zero_suffix_start,
    }))
}

/// One advice-word commitment object: one field coefficient per
/// canonical little-endian `u64`, embedded in slot zero when Akita's dense
/// schedule floor exceeds the logical word arity.
pub struct AdviceObject<PCS: CommitmentScheme> {
    pub plan: PrefixPackedObjectPlan,
    pub commitment: PCS::Output,
    pub hint: PCS::OpeningHint,
}

/// Builds the canonical zero-padded advice-word commitment. The setup
/// is derived from the application-owned immutable setup context plus the
/// public advice shape (the setup is transparent).
pub fn commit_advice<PCS>(
    setup_context: &PCS::SetupContext,
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
    let (setup, _) =
        PCS::transparent_object_setup(setup_context, physical_vars, plan.layout_digest())
            .map_err(commit_failed)?;
    let mut evaluations = vec![PCS::Field::default(); 1usize << physical_vars];
    for (evaluation, word) in evaluations.iter_mut().zip(words) {
        *evaluation = PCS::Field::from_u64(word);
    }
    let polynomial = Polynomial::new(evaluations);
    let (commitment, hint) = PCS::commit(&polynomial, &setup).map_err(commit_failed)?;
    Ok(AdviceObject {
        plan,
        commitment,
        hint,
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
    setup_context: &PCS::SetupContext,
    program: &JoltProgramPreprocessing,
    bytecode_chunk_count: usize,
    trace_order: TracePolynomialOrder,
) -> Result<DirectProgramObjects<PCS>, ProverError<PCS::Field>>
where
    PCS: CommitmentScheme + TransparentObjectSetup,
{
    let bytecode_len = program.bytecode.bytecode.len();
    let image_words = program_image_words_padded(program);
    let plan = committed_program_packing_plan(
        bytecode_len,
        bytecode_chunk_count,
        program.ram.bytecode_words.len(),
        trace_order,
    )
    .map_err(commit_failed)?;
    let mut chunk_coeffs = jolt_kernels::committed_program::build_committed_bytecode_chunk_coeffs(
        &program.bytecode.bytecode,
        bytecode_chunk_count,
        trace_order,
    )
    .map_err(commit_failed)?
    .into_iter();
    let mut setups = HashMap::<usize, PCS::ProverSetup>::new();
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
            let physical_vars = object_plan.packing().packed_num_vars();
            let setup = if let Some(setup) = setups.get(&physical_vars) {
                PCS::retag_transparent_object_setup(setup, object_plan.layout_digest())
                    .map_err(commit_failed)?
                    .0
            } else {
                PCS::transparent_object_setup(
                    setup_context,
                    physical_vars,
                    object_plan.layout_digest(),
                )
                .map_err(commit_failed)?
                .0
            };
            let (commitment, hint) = PCS::commit(&witness, &setup).map_err(commit_failed)?;
            let _ = setups.entry(physical_vars).or_insert_with(|| setup.clone());
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
