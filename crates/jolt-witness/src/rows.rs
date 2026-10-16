//! The owned handle over a witness's execution rows, resident or replayed.

use std::ops::Range;
use std::sync::Arc;

use jolt_program::execution::{RowReplay, TraceData, TraceError};
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_riscv::JoltTraceRow as TraceRow;

use crate::bundle::WitnessBundle;
use crate::consumer::ChunkVisitor;
use crate::witnesses::WitnessEnv;
use crate::{WitnessError, JOLT_VM_LABEL};

/// Where a witness's execution rows live.
#[derive(Clone)]
pub enum TraceStorage {
    /// The retained rows of the execution.
    Resident(Arc<TraceData>),
    /// Rows re-executed from checkpoints by every visit that reads them.
    Replay(Arc<dyn RowReplay>),
}

impl TraceStorage {
    /// Rejects rows that overrun a padded domain of `cycles` rows and replay
    /// checkpoint spacing that is not a power of two.
    pub(crate) fn check_domain(&self, cycles: usize) -> Result<(), WitnessError> {
        let proof_len = match self {
            Self::Resident(rows) => rows.proof_len(),
            Self::Replay(replay) => {
                let spacing = replay.segment_rows();
                if !spacing.is_power_of_two() {
                    return Err(WitnessError::InvalidWitnessData {
                        label: JOLT_VM_LABEL,
                        reason: format!(
                            "replay checkpoint spacing {spacing} is not a power of two"
                        ),
                    });
                }
                replay.proof_len()
            }
        };
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
}

/// Owned, cloneable access to a witness's execution rows over the padded
/// cycle domain `[0, cycles)`. Rows at or past the proof prefix are canonical
/// padding (`TraceRow::default()`); a row's lookahead is the next domain row,
/// absent only for row `cycles - 1`. Clones share the storage.
#[derive(Clone)]
pub struct TraceRows {
    storage: TraceStorage,
    cycles: usize,
    preprocessing: Arc<JoltProgramPreprocessing>,
    padding: TraceRow,
}

impl TraceRows {
    pub(crate) fn new(
        storage: TraceStorage,
        cycles: usize,
        preprocessing: Arc<JoltProgramPreprocessing>,
    ) -> Result<Self, WitnessError> {
        storage.check_domain(cycles)?;
        Ok(Self {
            storage,
            cycles,
            preprocessing,
            padding: TraceRow::default(),
        })
    }

    /// Padded cycle-domain size.
    pub fn cycles(&self) -> usize {
        self.cycles
    }

    /// The alignment unit for parallel visits: the replay checkpoint spacing,
    /// 1 for resident rows. An aligned replayed visit restores the checkpoint
    /// at its start, but a multi-row instruction spanning that row still
    /// re-executes from an earlier checkpoint ([`RowReplay::segment_rows`]).
    pub fn segment_len(&self) -> usize {
        match &self.storage {
            TraceStorage::Resident(_) => 1,
            TraceStorage::Replay(replay) => replay.segment_rows(),
        }
    }

    /// Visits `range` in order: callback `i` gets the `min(chunk, rows left)`
    /// rows from `range.start + i * chunk` and the next domain row as
    /// lookahead (`None` only at the end of the cycle domain, even when
    /// `range` ends earlier). An empty range makes no callback; a source or
    /// visitor error ends the visit and is returned.
    ///
    /// A replayed visit runs on the calling thread and owns one reader and one
    /// `min(chunk, range.len()) + 1`-row buffer until it returns; its visitor
    /// must not enter Rayon, or work stealing could stack several live visits
    /// on one thread. Nothing enforces this caller contract. Resident visits
    /// borrow physical rows in place.
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
        if range.is_empty() {
            return Ok(());
        }
        let env = WitnessEnv::new(&self.preprocessing);
        match &self.storage {
            TraceStorage::Resident(rows) => {
                self.visit_resident(rows.proof_rows(), range, chunk, &env, visitor)
            }
            TraceStorage::Replay(replay) => {
                self.visit_replay(replay.as_ref(), range, chunk, &env, visitor)
            }
        }
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
    /// Random access is a resident capability: replayed rows return
    /// [`WitnessError::UnavailableView`] instead of re-executing per row.
    #[inline]
    pub fn window<B: WitnessBundle>(&self, index: usize) -> Result<B, WitnessError> {
        let TraceStorage::Resident(rows) = &self.storage else {
            return Err(WitnessError::UnavailableView {
                label: "replayed trace rows",
            });
        };
        let physical = rows.proof_rows();
        let current = physical.get(index).unwrap_or(&self.padding);
        let next =
            (index + 1 < self.cycles).then(|| physical.get(index + 1).unwrap_or(&self.padding));
        B::from_row(current, next, &WitnessEnv::new(&self.preprocessing))
    }

    /// Physical rows are borrowed in place; a buffer reaching into padding is
    /// copied into one reused visit-owned buffer.
    fn visit_resident(
        &self,
        physical: &[TraceRow],
        range: Range<usize>,
        chunk: usize,
        env: &WitnessEnv<'_>,
        visitor: &mut ChunkVisitor<'_>,
    ) -> Result<(), WitnessError> {
        let mut padded = Vec::new();
        let mut start = range.start;
        while start < range.end {
            let end = start + chunk.min(range.end - start);
            let next = (end < self.cycles).then(|| physical.get(end).unwrap_or(&self.padding));
            if let Some(rows) = physical.get(start..end) {
                visitor(rows, next, env)?;
            } else {
                padded.clear();
                padded.reserve_exact(end - start);
                padded.extend_from_slice(physical.get(start..).unwrap_or_default());
                padded.resize(end - start, TraceRow::default());
                visitor(&padded, next, env)?;
            }
            start = end;
        }
        Ok(())
    }

    /// Reads every row once through one reader opened at `range.start`: each
    /// buffer's lookahead is carried to the front of the next buffer instead
    /// of being replayed again. Rows past the proof prefix are padding, so a
    /// range starting there opens no reader.
    fn visit_replay(
        &self,
        replay: &dyn RowReplay,
        range: Range<usize>,
        chunk: usize,
        env: &WitnessEnv<'_>,
        visitor: &mut ChunkVisitor<'_>,
    ) -> Result<(), WitnessError> {
        let proof_len = replay.proof_len();
        let mut physical = if range.start < proof_len {
            let reader = replay
                .reader(range.start)
                .map_err(|error| replay_error(range.start, error))?;
            Some((reader, proof_len - range.start))
        } else {
            None
        };
        let mut rows = vec![TraceRow::default(); chunk.min(range.len()) + 1];
        let mut carried = 0;
        let mut start = range.start;
        while start < range.end {
            let len = chunk.min(range.end - start);
            let end = start + len;
            let has_next = end < self.cycles;
            // Replay while the reader has proof rows left, then pad; the
            // reader is dropped with the last proof row.
            let pending = &mut rows[carried..len + usize::from(has_next)];
            let mut read = 0;
            if let Some((reader, left)) = &mut physical {
                read = (*left).min(pending.len());
                if read > 0 {
                    reader
                        .read(&mut pending[..read])
                        .map_err(|error| replay_error(start + carried, error))?;
                    *left -= read;
                }
                if *left == 0 {
                    physical = None;
                }
            }
            pending[read..].fill(TraceRow::default());
            visitor(&rows[..len], has_next.then(|| &rows[len]), env)?;
            if has_next {
                rows[0] = rows[len];
                carried = 1;
            }
            start = end;
        }
        Ok(())
    }
}

fn replay_error(row: usize, error: TraceError) -> WitnessError {
    WitnessError::Replay {
        row,
        reason: error.to_string(),
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
    use crate::testing::{LoggedReplay, ReplayEvent};
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
        TraceRows::new(TraceStorage::Resident(rows), CYCLES, preprocessing()).unwrap()
    }

    fn replayed(replay: &Arc<LoggedReplay>) -> TraceRows {
        let storage = TraceStorage::Replay(Arc::clone(replay) as Arc<dyn RowReplay>);
        TraceRows::new(storage, CYCLES, preprocessing()).unwrap()
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

    /// A visit reads the physical rows it needs once, in order, through one
    /// reader: a nonempty `range` plus its lookahead row, clipped to the proof
    /// prefix.
    fn assert_single_pass(log: &[ReplayEvent], range: &Range<usize>) {
        let needed_end = if range.is_empty() {
            range.start
        } else {
            (range.end + 1).min(CYCLES).min(PROOF_LEN).max(range.start)
        };
        let mut events = log.iter();
        if range.start < needed_end {
            assert_eq!(events.next(), Some(&ReplayEvent::Open(range.start)));
        }
        let mut next = range.start;
        for event in events {
            let ReplayEvent::Read(read) = event else {
                panic!("{range:?}: unexpected {event:?}");
            };
            assert_eq!(read.start, next, "{range:?}: rows reread or skipped");
            next = read.end;
        }
        assert_eq!(next, needed_end, "{range:?}: needed physical rows unread");
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
        let replay = Arc::new(LoggedReplay::new(physical_rows(), 4));
        for (range, chunk, partition) in cases {
            let expected: Vec<Callback> = partition
                .iter()
                .map(|&(start, len)| {
                    let end = start + len;
                    let rows = (start..end).map(domain_row).collect();
                    (start, rows, (end < CYCLES).then(|| domain_row(end)))
                })
                .collect();
            assert_eq!(record(&resident(), range.clone(), chunk), expected);
            assert_eq!(
                record(&replayed(&replay), range.clone(), chunk),
                expected,
                "{range:?} by {chunk}"
            );
            assert_single_pass(&replay.take_log(), &range);

            // Resident random access extracts each window independently.
            let mut lengths = Vec::new();
            replayed(&replay)
                .visit_bundles(range.clone(), chunk, |first, bundles: &[PcWindow]| {
                    for (offset, bundle) in bundles.iter().enumerate() {
                        assert_eq!(*bundle, resident().window(first + offset)?);
                    }
                    lengths.push((first, bundles.len()));
                    Ok(())
                })
                .unwrap();
            assert_eq!(lengths, partition, "{range:?} by {chunk}");
            assert_single_pass(&replay.take_log(), &range);
        }
    }

    #[test]
    fn source_and_visitor_failures_end_the_visit() {
        let replay = Arc::new(LoggedReplay::new(physical_rows(), 4).failing_at(8));
        let mut starts = Vec::new();
        let mut first = 0;
        let result = replayed(&replay).visit(0..CYCLES, 3, &mut |rows, _, _| {
            starts.push(first);
            first += rows.len();
            Ok(())
        });
        assert_eq!(starts, [0, 3]);
        assert!(
            matches!(result, Err(WitnessError::Replay { row: 7, .. })),
            "{result:?}"
        );

        let stop = WitnessError::InvalidWitnessData {
            label: "test visitor",
            reason: "stop".to_owned(),
        };
        let healthy = Arc::new(LoggedReplay::new(physical_rows(), 4));
        for rows in [resident(), replayed(&healthy)] {
            let mut calls = 0;
            let result = rows.visit(0..CYCLES, 3, &mut |_, _, _| {
                calls += 1;
                if calls == 2 {
                    return Err(stop.clone());
                }
                Ok(())
            });
            assert_eq!(result, Err(stop.clone()));
            assert_eq!(calls, 2);
        }
    }

    #[test]
    fn invalid_requests_are_rejected_before_any_callback() {
        let replay = Arc::new(LoggedReplay::new(physical_rows(), 4));
        for rows in [resident(), replayed(&replay)] {
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
        }
        assert!(replay.take_log().is_empty());
        assert!(matches!(
            replayed(&replay).window::<PcWindow>(0),
            Err(WitnessError::UnavailableView { .. })
        ));

        let storage = |segment_rows| {
            TraceStorage::Replay(Arc::new(LoggedReplay::new(physical_rows(), segment_rows)))
        };
        for (storage, cycles) in [(storage(4), PROOF_LEN - 1), (storage(3), CYCLES)] {
            assert!(matches!(
                TraceRows::new(storage, cycles, preprocessing()),
                Err(WitnessError::InvalidWitnessData { .. })
            ));
        }
    }
}
