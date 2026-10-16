//! The owned handle over a witness's resident execution rows.

use std::ops::Range;
use std::sync::Arc;

use jolt_program::execution::TraceData;
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_riscv::JoltTraceRow as TraceRow;

use crate::bundle::WitnessBundle;
use crate::consumer::ChunkVisitor;
use crate::witnesses::WitnessEnv;
use crate::{WitnessError, JOLT_VM_LABEL};

/// Owned, cloneable access to a witness's execution rows over the padded
/// cycle domain `[0, cycles)`. Rows at or past the proof prefix are canonical
/// padding (`TraceRow::default()`); a row's lookahead is the next domain row,
/// absent only for row `cycles - 1`. Clones share the rows.
#[derive(Clone)]
pub struct TraceRows {
    rows: Arc<TraceData>,
    cycles: usize,
    preprocessing: Arc<JoltProgramPreprocessing>,
    padding: TraceRow,
}

impl TraceRows {
    pub(crate) fn new(
        rows: Arc<TraceData>,
        cycles: usize,
        preprocessing: Arc<JoltProgramPreprocessing>,
    ) -> Result<Self, WitnessError> {
        Self::check_domain(&rows, cycles)?;
        Ok(Self {
            rows,
            cycles,
            preprocessing,
            padding: TraceRow::default(),
        })
    }

    /// Rejects rows that overrun a padded domain of `cycles` rows.
    pub(crate) fn check_domain(rows: &TraceData, cycles: usize) -> Result<(), WitnessError> {
        let proof_len = rows.proof_len();
        if proof_len > cycles {
            return Err(WitnessError::InvalidWitnessData {
                label: JOLT_VM_LABEL,
                reason: format!(
                    "physical trace has {proof_len} rows but the cycle domain has {cycles}"
                ),
            });
        }
        Ok(())
    }

    /// Padded cycle-domain size.
    pub fn cycles(&self) -> usize {
        self.cycles
    }

    /// Visits `range` in order: callback `i` gets the `min(chunk, rows left)`
    /// rows from `range.start + i * chunk` and the next domain row as
    /// lookahead (`None` only at the end of the cycle domain, even when
    /// `range` ends earlier). An empty range makes no callback; a visitor
    /// error ends the visit and is returned.
    ///
    /// Physical rows are borrowed in place; a buffer reaching into padding is
    /// copied into one reused visit-owned buffer.
    pub fn visit(
        &self,
        range: Range<usize>,
        chunk: usize,
        visitor: &mut ChunkVisitor<'_>,
    ) -> Result<(), WitnessError> {
        if chunk == 0 {
            return Err(WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: "visit chunk size must be nonzero".to_owned(),
            });
        }
        if range.start > range.end || range.end > self.cycles {
            return Err(WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: format!(
                    "cycle range [{}, {}) exceeds the domain of {} cycles",
                    range.start, range.end, self.cycles
                ),
            });
        }
        let env = WitnessEnv::new(&self.preprocessing);
        let physical = self.rows.proof_rows();
        let mut padded = Vec::new();
        let mut start = range.start;
        while start < range.end {
            let end = start + chunk.min(range.end - start);
            let next = (end < self.cycles).then(|| physical.get(end).unwrap_or(&self.padding));
            if let Some(rows) = physical.get(start..end) {
                visitor(rows, next, &env)?;
            } else {
                padded.clear();
                padded.reserve_exact(end - start);
                padded.extend_from_slice(physical.get(start..).unwrap_or_default());
                padded.resize(end - start, TraceRow::default());
                visitor(&padded, next, &env)?;
            }
            start = end;
        }
        Ok(())
    }

    /// [`Self::visit`] with every row extracted serially into one visit-owned
    /// buffer of at most `min(chunk, range.len())` bundles; `visit(first,
    /// bundles)` receives each buffer with its first cycle. The callback
    /// partition is exactly [`Self::visit`]'s.
    pub fn visit_bundles<B: WitnessBundle>(
        &self,
        range: Range<usize>,
        chunk: usize,
        mut visit: impl FnMut(usize, &[B]) -> Result<(), WitnessError>,
    ) -> Result<(), WitnessError> {
        let mut bundles = Vec::new();
        let mut first = range.start;
        self.visit(range, chunk, &mut |rows, next, env| {
            bundles.clear();
            bundles.reserve_exact(rows.len());
            for (index, row) in rows.iter().enumerate() {
                bundles.push(B::from_row(row, rows.get(index + 1).or(next), env)?);
            }
            visit(first, &bundles)?;
            first += rows.len();
            Ok(())
        })
    }

    /// Extracts the bundle of row `index` with padding and lookahead.
    #[inline]
    pub fn window<B: WitnessBundle>(&self, index: usize) -> Result<B, WitnessError> {
        let physical = self.rows.proof_rows();
        let current = physical.get(index).unwrap_or(&self.padding);
        let next =
            (index + 1 < self.cycles).then(|| physical.get(index + 1).unwrap_or(&self.padding));
        B::from_row(current, next, &WitnessEnv::new(&self.preprocessing))
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::panic, reason = "test module")]
mod tests {
    use jolt_program::preprocess::{BytecodePreprocessing, RAMPreprocessing};
    use jolt_riscv::{
        JoltInstructionKind, JoltInstructionRow, NormalizedOperands, RamAccess, RegisterState,
    };

    use super::*;
    use crate::witnesses::{NextUnexpandedPc, UnexpandedPc};

    const CYCLES: usize = 16;
    const PROOF_LEN: usize = 11;

    #[derive(Debug, PartialEq, Eq, WitnessBundle)]
    struct PcWindow {
        pc: UnexpandedPc,
        next_pc: NextUnexpandedPc,
    }

    /// Distinct non-padding rows; row `i` sits at address `0x8000_0000 + 4i`.
    fn physical_rows() -> Vec<TraceRow> {
        (0..PROOF_LEN)
            .map(|index| {
                let instruction = JoltInstructionRow {
                    instruction_kind: JoltInstructionKind::ADDI,
                    address: 0x8000_0000 + 4 * index,
                    operands: NormalizedOperands {
                        rd: Some(1),
                        rs1: Some(2),
                        rs2: None,
                        imm: 3,
                    },
                    ..Default::default()
                };
                TraceRow::new(
                    instruction,
                    RegisterState::default(),
                    RamAccess::NoOp,
                    u32::try_from(index + 1).unwrap(),
                )
                .unwrap()
            })
            .collect()
    }

    fn domain_row(index: usize) -> TraceRow {
        physical_rows().get(index).copied().unwrap_or_default()
    }

    fn preprocessing() -> Arc<JoltProgramPreprocessing> {
        Arc::new(JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing::default(),
            ram: RAMPreprocessing::default(),
            memory_layout: Default::default(),
            max_padded_trace_length: CYCLES,
        })
    }

    fn resident() -> TraceRows {
        let rows = Arc::new(TraceData::new(physical_rows()));
        TraceRows::new(rows, CYCLES, preprocessing()).unwrap()
    }

    /// Every callback as (first cycle, rows, lookahead).
    type Callback = (usize, Vec<TraceRow>, Option<TraceRow>);

    fn record(rows: &TraceRows, range: Range<usize>, chunk: usize) -> Vec<Callback> {
        let mut callbacks = Vec::new();
        let mut first = range.start;
        rows.visit(range, chunk, &mut |buffer, next, _| {
            callbacks.push((first, buffer.to_vec(), next.copied()));
            first += buffer.len();
            Ok(())
        })
        .unwrap();
        callbacks
    }

    #[test]
    fn visits_partition_ranges_into_exact_chunks_with_lookahead() {
        let all_singles: Vec<(usize, usize)> = (0..CYCLES).map(|start| (start, 1)).collect();
        let cases = [
            (3..13, 4, vec![(3, 4), (7, 4), (11, 2)]),
            (5..10, 3, vec![(5, 3), (8, 2)]),
            (9..16, 5, vec![(9, 5), (14, 2)]),
            (12..16, 16, vec![(12, 4)]),
            (6..6, 4, vec![]),
            (0..16, 1, all_singles),
        ];
        for (range, chunk, partition) in cases {
            let expected: Vec<Callback> = partition
                .iter()
                .map(|&(start, len)| {
                    let end = start + len;
                    let rows = (start..end).map(domain_row).collect();
                    (start, rows, (end < CYCLES).then(|| domain_row(end)))
                })
                .collect();
            assert_eq!(
                record(&resident(), range.clone(), chunk),
                expected,
                "{range:?} by {chunk}"
            );

            // Random access extracts each window independently.
            let mut lengths = Vec::new();
            resident()
                .visit_bundles(range.clone(), chunk, |first, bundles: &[PcWindow]| {
                    for (offset, bundle) in bundles.iter().enumerate() {
                        assert_eq!(*bundle, resident().window(first + offset)?);
                    }
                    lengths.push((first, bundles.len()));
                    Ok(())
                })
                .unwrap();
            assert_eq!(lengths, partition, "{range:?} by {chunk}");
        }
    }

    #[test]
    fn visitor_failures_end_the_visit() {
        let stop = WitnessError::InvalidWitnessData {
            label: "test visitor",
            reason: "stop".to_owned(),
        };
        let mut calls = 0;
        let result = resident().visit(0..CYCLES, 3, &mut |_, _, _| {
            calls += 1;
            if calls == 2 {
                return Err(stop.clone());
            }
            Ok(())
        });
        assert_eq!(result, Err(stop));
        assert_eq!(calls, 2);
    }

    #[test]
    fn invalid_requests_are_rejected_before_any_callback() {
        let rows = resident();
        for (range, chunk) in [
            (4..4, 0),
            (0..CYCLES + 1, 4),
            (Range { start: 5, end: 4 }, 1),
        ] {
            let result = rows.visit(range, chunk, &mut |_, _, _| panic!("no callback"));
            assert!(
                matches!(result, Err(WitnessError::InvalidDimensions { .. })),
                "{result:?}"
            );
        }
        assert!(matches!(
            TraceRows::new(
                Arc::new(TraceData::new(physical_rows())),
                PROOF_LEN - 1,
                preprocessing()
            ),
            Err(WitnessError::InvalidWitnessData { .. })
        ));
    }
}
