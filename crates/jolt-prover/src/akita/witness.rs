//! Prover-side packed (Akita) witness assembly: the `OneHotTrace` columns
//! from the witness plane's typed rows, the advice word objects, the
//! direct bounded-dense committed-program objects.

#[cfg(feature = "field-inline")]
use jolt_kernels::field_inline::FieldIncrementColumn;
#[cfg(all(feature = "field-inline", feature = "parallel"))]
use rayon::prelude::*;
use std::collections::HashMap;
#[cfg(not(feature = "field-inline"))]
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

use jolt_akita::{no_selected_row, TraceOneHotRows};
use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
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
    BalancedIncColumn, BytecodePc, FusedInc, LookupIndex, RaChunkSelector, RemappedRamAddress,
};
use jolt_witness::{
    collect_bundles, JoltWitnessPlane, RandomAccessRows, WitnessBundle, WitnessError,
};

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

/// The RAM access of [`OneHotTraceSourceRow`] alone: whether the RAM columns
/// commit row zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
struct RamAccessRow {
    ram_address: RemappedRamAddress,
}

pub(super) struct AssembledTrace<F: JoltField> {
    pub(super) rows: Arc<OneHotTraceRows>,
    #[cfg(feature = "field-inline")]
    pub(super) increments: FieldIncrementColumn<F>,
    #[cfg(not(feature = "field-inline"))]
    field: PhantomData<F>,
}

#[derive(Clone, Copy)]
enum OneHotTraceColumn {
    Instruction(RaChunkSelector),
    Bytecode(RaChunkSelector),
    Ram(RaChunkSelector),
    Increment(BalancedIncColumn),
}

/// Extracts rows on every read from the witness plane's resident compact
/// trace, so no trace-sized row matrix lives from the commitment to the
/// opening. The row trait cannot return extraction errors, so the first error
/// is retained for [`OneHotTraceRows::check_extraction`]. The commitment reads
/// every row of this immutable source, so it observes any error a later read could.
struct ExtractedRows {
    access: RandomAccessRows,
    extraction_error: OnceLock<WitnessError>,
}

impl ExtractedRows {
    fn window<B: WitnessBundle>(&self, row: usize) -> Option<B> {
        self.access
            .window(row)
            .map_err(|error| {
                let _ = self.extraction_error.set(error);
            })
            .ok()
    }
}

enum SelectedRows {
    Extracted(ExtractedRows),
    /// Materialized for witness planes without random access.
    Packed {
        selected_rows: Vec<u8>,
        ram_active_rows: Vec<u64>,
    },
}

/// Row-major `OneHotTrace` rows in the plan's canonical semantic-column order.
pub(super) struct OneHotTraceRows {
    num_rows: usize,
    columns: Vec<OneHotTraceColumn>,
    ram_digit_zero_mask: u64,
    selected_rows: SelectedRows,
}

impl OneHotTraceRows {
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

    /// Returns the first extraction error any read has hit.
    pub(super) fn check_extraction(&self) -> Result<(), WitnessError> {
        match &self.selected_rows {
            SelectedRows::Extracted(rows) => {
                rows.extraction_error.get().cloned().map_or(Ok(()), Err)
            }
            SelectedRows::Packed { .. } => Ok(()),
        }
    }
}

impl TraceOneHotRows for OneHotTraceRows {
    fn num_rows(&self) -> usize {
        self.num_rows
    }

    fn num_columns(&self) -> usize {
        self.columns.len()
    }

    fn fill_row(&self, row: usize, selected_rows: &mut [u8]) {
        self.fill_rows(row, selected_rows);
    }

    fn fill_rows(&self, row_start: usize, selected_rows: &mut [u8]) {
        let num_columns = self.columns.len();
        debug_assert_eq!(selected_rows.len() % num_columns, 0);
        match &self.selected_rows {
            SelectedRows::Extracted(rows) => {
                for (row_offset, selected_rows) in
                    selected_rows.chunks_exact_mut(num_columns).enumerate()
                {
                    match rows.window(row_start + row_offset) {
                        Some(row) => {
                            let _ = fill_trace_row(row, &self.columns, selected_rows);
                        }
                        None => selected_rows.fill(no_selected_row()),
                    }
                }
            }
            SelectedRows::Packed {
                selected_rows: packed,
                ..
            } => {
                let start = row_start * num_columns;
                selected_rows.copy_from_slice(&packed[start..start + selected_rows.len()]);
            }
        }
    }

    fn committed_digit_zero_mask(&self, row: usize) -> u64 {
        let ram_active = match &self.selected_rows {
            SelectedRows::Extracted(rows) => rows
                .window::<RamAccessRow>(row)
                .is_some_and(|row| row.ram_address.0.is_some()),
            SelectedRows::Packed {
                ram_active_rows, ..
            } => {
                ram_active_rows[row / u64::BITS as usize] & (1u64 << (row % u64::BITS as usize))
                    != 0
            }
        };
        if ram_active {
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
/// plan's canonical semantic-column order. Witness planes with random access
/// yield a view that extracts rows on demand; others are materialized once.
#[tracing::instrument(skip_all, name = "assemble_one_hot_trace")]
pub(super) fn assemble_one_hot_trace_rows<F: JoltField>(
    witness: &dyn JoltWitnessPlane<F>,
    plan: &OneHotTraceLayoutPlan,
    ra_layout: JoltRaPolynomialLayout,
    log_k_chunk: usize,
    log_t: usize,
) -> Result<AssembledTrace<F>, ProverError<F>> {
    OneHotTraceRows::validate_dimensions::<F>(plan, log_k_chunk, log_t)?;
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

    #[cfg(feature = "field-inline")]
    let field_oracle = witness.field_inline().ok_or(ProverError::Unsupported {
        reason: "field-inline trace assembly requires its witness oracle",
    })?;
    #[cfg(feature = "field-inline")]
    let mut increments = vec![F::zero(); num_rows];

    let random_access = witness
        .random_access()
        .filter(|access| num_rows <= access.cycles());
    let selected_rows = if let Some(access) = random_access {
        // Increment commitments precede the trace commitment, so retain only
        // their shared column; the one-hot rows stay lazy.
        #[cfg(feature = "field-inline")]
        {
            #[cfg(feature = "parallel")]
            let values = increments.par_iter_mut();
            #[cfg(not(feature = "parallel"))]
            let values = increments.iter_mut();
            values.enumerate().try_for_each(|(index, value)| {
                *value = field_oracle.rd_increment_at(index)?;
                Ok::<_, WitnessError>(())
            })?;
        }
        SelectedRows::Extracted(ExtractedRows {
            access,
            extraction_error: OnceLock::new(),
        })
    } else {
        let mut selected_rows = vec![0u8; num_rows * num_columns];
        let mut ram_active_rows = vec![0u64; num_rows.div_ceil(u64::BITS as usize)];
        let rows: Vec<OneHotTraceSourceRow> = collect_bundles(witness, num_rows)?;
        for (row_index, (row, selected_rows)) in rows
            .into_iter()
            .zip(selected_rows.chunks_exact_mut(num_columns))
            .enumerate()
        {
            #[cfg(feature = "field-inline")]
            {
                increments[row_index] = field_oracle.rd_increment_at(row_index)?;
            }
            if fill_trace_row(row, &columns, selected_rows) {
                ram_active_rows[row_index / u64::BITS as usize] |=
                    1u64 << (row_index % u64::BITS as usize);
            }
        }
        SelectedRows::Packed {
            selected_rows,
            ram_active_rows,
        }
    };
    Ok(AssembledTrace {
        rows: Arc::new(OneHotTraceRows {
            num_rows,
            columns,
            ram_digit_zero_mask,
            selected_rows,
        }),
        #[cfg(feature = "field-inline")]
        increments: FieldIncrementColumn::from_values(increments),
        #[cfg(not(feature = "field-inline"))]
        field: PhantomData,
    })
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
