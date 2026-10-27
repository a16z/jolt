//! Packed trace assembly for the reference Akita commitment kernel.

#[cfg(feature = "field-inline")]
use crate::field_inline::FieldIncrementColumn;
use crate::KernelError;
use jolt_akita::{no_selected_row, TraceOneHotRows};
use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
use jolt_claims::protocols::jolt::lattice::geometry::BalancedIncChunking;
use jolt_claims::protocols::jolt::lattice::strategy::OneHotTraceLayoutPlan;
use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_field::JoltField;
use jolt_witness::witnesses::{
    BalancedIncColumn, BytecodePc, FusedInc, LookupIndex, RemappedRamAddress,
};
use jolt_witness::{
    collect_bundles, JoltWitnessPlane, RandomAccessRows, WitnessBundle, WitnessError,
};
#[cfg(all(feature = "field-inline", feature = "parallel"))]
use rayon::prelude::*;
#[cfg(not(feature = "field-inline"))]
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

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
enum TraceChunkBits {
    Four,
    Eight,
}

/// Canonical column order: instruction RA, balanced increment digits and
/// carry, bytecode RA, then RAM RA.
struct OneHotTraceColumns {
    layout: JoltRaPolynomialLayout,
    increment_digits: usize,
    bits: TraceChunkBits,
}

impl OneHotTraceColumns {
    fn new<F: JoltField>(
        plan: &OneHotTraceLayoutPlan,
        layout: JoltRaPolynomialLayout,
        log_k_chunk: usize,
    ) -> Result<Self, KernelError<F>> {
        let increment_digits = BalancedIncChunking::new(log_k_chunk)
            .map_err(|_| KernelError::Unsupported {
                reason: "native one-hot trace chunk width must divide the increment width",
            })?
            .chunk_count();
        let expected = layout
            .committed_polynomials()
            .take(layout.instruction())
            .chain((0..increment_digits).map(JoltCommittedPolynomial::BalancedIncDigit))
            .chain(std::iter::once(JoltCommittedPolynomial::BalancedIncCarry))
            .chain(layout.committed_polynomials().skip(layout.instruction()));
        if !plan.ids().iter().copied().eq(expected) {
            return Err(KernelError::InvariantViolation {
                reason: "OneHotTrace plan disagrees with the canonical RA and increment layout",
            });
        }
        let bits = match log_k_chunk {
            4 => TraceChunkBits::Four,
            8 => TraceChunkBits::Eight,
            _ => {
                return Err(KernelError::Unsupported {
                    reason: "native one-hot trace chunk width must be 4 or 8 bits",
                })
            }
        };
        if [layout.instruction(), layout.bytecode(), layout.ram()]
            .into_iter()
            .any(|chunks| chunks > 128 / log_k_chunk)
        {
            return Err(KernelError::InvariantViolation {
                reason: "OneHotTrace address decomposition exceeds 128 bits",
            });
        }
        Ok(Self {
            layout,
            increment_digits,
            bits,
        })
    }

    fn len(&self) -> usize {
        self.layout.total() + self.increment_digits + 1
    }

    fn fill(&self, row: OneHotTraceSourceRow, selected_rows: &mut [u8]) -> bool {
        match self.bits {
            TraceChunkBits::Four => self.fill_with_width::<4>(row, selected_rows),
            TraceChunkBits::Eight => self.fill_with_width::<8>(row, selected_rows),
        }
    }

    fn fill_with_width<const BITS: usize>(
        &self,
        row: OneHotTraceSourceRow,
        selected_rows: &mut [u8],
    ) -> bool {
        let (instruction, rest) = selected_rows.split_at_mut(self.layout.instruction());
        let (digits, rest) = rest.split_at_mut(self.increment_digits);
        let (carry, rest) = rest.split_at_mut(1);
        let (bytecode, ram) = rest.split_at_mut(self.layout.bytecode());
        for (index, digit) in digits.iter_mut().enumerate() {
            *digit = row
                .fused_inc
                .selected_row(BalancedIncColumn::Digit { width: BITS, index })
                as u8;
        }
        carry[0] = row
            .fused_inc
            .selected_row(BalancedIncColumn::Carry { width: BITS }) as u8;
        for (mut value, columns) in [
            (row.lookup_index.0, instruction),
            (row.bytecode_pc.0 as u128, bytecode),
            (row.ram_address.0.unwrap_or(0) as u128, ram),
        ] {
            for column in columns.iter_mut().rev() {
                *column = (value & ((1 << BITS) - 1)) as u8;
                value >>= BITS;
            }
        }
        row.ram_address.0.is_some()
    }
}

/// Extracts rows on every read from the witness plane's compact
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
    columns: OneHotTraceColumns,
    ram_digit_zero_mask: u64,
    selected_rows: SelectedRows,
}

impl OneHotTraceRows {
    fn validate_dimensions<F: JoltField>(
        plan: &OneHotTraceLayoutPlan,
        log_k_chunk: usize,
        log_t: usize,
    ) -> Result<(), KernelError<F>> {
        if !matches!(log_k_chunk, 4 | 8) {
            return Err(KernelError::Unsupported {
                reason: "native one-hot trace chunk width must be 4 or 8 bits",
            });
        }
        let logical_num_vars = log_t
            .checked_add(log_k_chunk)
            .ok_or(KernelError::Unsupported {
                reason: "native one-hot trace dimensions overflow",
            })?;
        if plan.num_vars() != logical_num_vars {
            return Err(KernelError::InvariantViolation {
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
    fn nonzero_prefix_len(&self) -> usize {
        match &self.selected_rows {
            SelectedRows::Extracted(rows) => rows.access.physical_cycles(),
            SelectedRows::Packed { .. } => self.num_rows,
        }
    }

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
                            let _ = self.columns.fill(row, selected_rows);
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

    fn fill_rows_with_masks(&self, row_start: usize, selected_rows: &mut [u8], masks: &mut [u64]) {
        match &self.selected_rows {
            SelectedRows::Extracted(rows) => {
                for (offset, (selected, mask)) in selected_rows
                    .chunks_exact_mut(self.columns.len())
                    .zip(masks)
                    .enumerate()
                {
                    let ram_active = if let Some(row) = rows.window(row_start + offset) {
                        self.columns.fill(row, selected)
                    } else {
                        selected.fill(no_selected_row());
                        false
                    };
                    *mask = if ram_active {
                        self.ram_digit_zero_mask
                    } else {
                        0
                    };
                }
            }
            SelectedRows::Packed { .. } => {
                self.fill_rows(row_start, selected_rows);
                self.fill_committed_digit_zero_masks(row_start, masks);
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
) -> Result<AssembledTrace<F>, KernelError<F>> {
    OneHotTraceRows::validate_dimensions::<F>(plan, log_k_chunk, log_t)?;
    let num_rows = 1usize << log_t;
    let num_columns = plan.ids().len();
    let ram_digit_zero_mask = plan
        .ranges()
        .ram
        .clone()
        .fold(0u64, |mask, column| mask | (1u64 << column));
    let columns = OneHotTraceColumns::new(plan, ra_layout, log_k_chunk)?;

    #[cfg(feature = "field-inline")]
    let field_oracle = witness.field_inline().ok_or(KernelError::Unsupported {
        reason: "field-inline trace assembly requires its witness oracle",
    })?;
    #[cfg(feature = "field-inline")]
    let increments = {
        const CHUNK_SIZE: usize = 1 << 14;
        let mut increments = jolt_utils::unsafe_allocate_zero_vec(num_rows);
        #[cfg(feature = "parallel")]
        let chunks = increments.par_chunks_mut(CHUNK_SIZE);
        #[cfg(not(feature = "parallel"))]
        let chunks = increments.chunks_mut(CHUNK_SIZE);
        chunks.enumerate().try_for_each(|(chunk, values)| {
            field_oracle.fill_rd_increments(chunk * CHUNK_SIZE, values)
        })?;
        increments
    };

    let random_access = witness
        .random_access()
        .filter(|access| num_rows <= access.cycles());
    let selected_rows = if let Some(access) = random_access {
        // Increment commitments precede the trace commitment, so retain only
        // their shared column; the one-hot rows stay lazy.
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
            if columns.fill(row, selected_rows) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn chunk_order_and_ram_presence_match_frozen_encodings() -> Result<(), Box<dyn Error>> {
        let lookup = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210;
        for (bits, width, instruction, bytecode, ram, ra_expected) in [
            (
                TraceChunkBits::Four,
                4,
                32,
                3,
                4,
                vec![
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10,
                    9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7,
                ],
            ),
            (
                TraceChunkBits::Eight,
                8,
                16,
                2,
                2,
                vec![
                    0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76,
                    0x54, 0x32, 0x10, 0x01, 0x23, 0x45, 0x67,
                ],
            ),
        ] {
            let increment_digits = 64 / width;
            let columns = OneHotTraceColumns {
                layout: JoltRaPolynomialLayout::new(instruction, bytecode, ram)?,
                increment_digits,
                bits,
            };
            let increments = instruction..instruction + increment_digits + 1;
            // A zero increment selects the digit-zero row of every digit and the carry.
            let mut expected = ra_expected.clone();
            drop(expected.splice(instruction..instruction, vec![0; increments.len()]));
            let mut output = vec![0xff; columns.len()];
            let mut row = OneHotTraceSourceRow {
                lookup_index: LookupIndex(lookup),
                bytecode_pc: BytecodePc(0x123),
                ram_address: RemappedRamAddress(Some(0x4567)),
                fused_inc: FusedInc(0),
            };
            assert!(columns.fill(row, &mut output));
            assert_eq!(output, expected);
            for address in [None, Some(0)] {
                row.ram_address = RemappedRamAddress(address);
                assert_eq!(columns.fill(row, &mut output), address.is_some());
                assert_eq!(
                    &output[..expected.len() - ram],
                    &expected[..expected.len() - ram]
                );
                assert_eq!(&output[expected.len() - ram..], vec![0; ram]);
            }
            for delta in [1, -1, i128::from(u64::MAX), -i128::from(u64::MAX)] {
                row.fused_inc = FusedInc(delta);
                let _ = columns.fill(row, &mut output);
                let (digits, carry) = output[increments.clone()].split_at(increment_digits);
                for (index, &digit) in digits.iter().enumerate() {
                    assert_eq!(
                        usize::from(digit),
                        row.fused_inc
                            .selected_row(BalancedIncColumn::Digit { width, index })
                    );
                }
                assert_eq!(
                    usize::from(carry[0]),
                    row.fused_inc
                        .selected_row(BalancedIncColumn::Carry { width })
                );
                let mut ra_output = output.clone();
                drop(ra_output.drain(increments.clone()));
                assert_eq!(&ra_output[..instruction], &ra_expected[..instruction]);
            }
        }
        Ok(())
    }
}
