//! Large commitment windows share row-aligned tiles across devices.
//!
//! Canonical witness windows remain cached for later sumchecks, while either
//! device can commit a ready tile. This balances active rows against padding
//! without changing the PCS row order. Optimized streams rows to its PCS
//! consumer; CUDA retains them on-device and assigns whole columns to tier-two
//! workers. Full equal-height Miller batches amortize G2 preparation; pairing
//! individual tiles reduced throughput on two L4s despite better overlap.

use std::collections::VecDeque;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use cudarc::driver::CudaSlice;
use jolt_field::JoltField;
use jolt_witness::backend::cuda::DeviceTrace;
use tracing::Span;

use super::partition::{ColumnPlan, PipelineResult};
use super::{
    dense_columns_from_trace, hot_column_views, hot_columns_from_trace, retain_hot_columns,
    tier1_order, DeviceTier1Commitment, WindowHotColumns,
};
use crate::cuda::common::context::{context_for, enter_device, CudaKernelContext};
use crate::cuda::common::devices::{
    device_selections, fan_out_bound, CycleWindow, DeviceTask, MIN_COMMIT_TILE_WINDOW,
};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::msm::{AffineLimbs, DeviceG1Bases, OneHotMode, SignedColumn, FQ_LIMBS};
use crate::KernelError;

struct PreparedWindow {
    hot: WindowHotColumns,
    dense: Vec<Option<SignedColumn>>,
}

struct Work {
    window: usize,
    local: CycleWindow,
    input: Arc<PreparedWindow>,
}

struct State {
    ready: VecDeque<Work>,
    inputs: Vec<Arc<PreparedWindow>>,
    unfinished: usize,
    cancelled: bool,
}

struct Queue {
    state: Mutex<State>,
    changed: Condvar,
}

impl Queue {
    fn new(windows: usize) -> Self {
        Self {
            state: Mutex::new(State {
                ready: VecDeque::new(),
                inputs: Vec::new(),
                unfinished: windows,
                cancelled: false,
            }),
            changed: Condvar::new(),
        }
    }

    fn cancel(&self) {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .cancelled = true;
        self.changed.notify_all();
    }

    fn take(&self) -> Option<Work> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if state.cancelled || state.unfinished == 0 {
                return None;
            }
            if let Some(work) = state.ready.pop_front() {
                return Some(work);
            }
            state = self.changed.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn prepared(&self, window: usize, len: usize, input: Arc<PreparedWindow>) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.inputs.push(Arc::clone(&input));
        let tile_len = len / 2;
        for start in [0, tile_len] {
            state.ready.push_back(Work {
                window,
                local: CycleWindow {
                    start,
                    len: tile_len,
                },
                input: Arc::clone(&input),
            });
        }
        state.unfinished += 1;
        self.changed.notify_all();
    }

    fn finished(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.unfinished -= 1;
        self.changed.notify_all();
    }
}

struct RowTile {
    first_block: usize,
    blocks: usize,
    columns: Vec<CudaSlice<u64>>,
}

fn complete_work<F: JoltField, T>(
    context: &CudaKernelContext,
    work: impl FnOnce() -> Result<T, KernelError<F>>,
) -> Result<T, KernelError<F>> {
    let result = catch_unwind(AssertUnwindSafe(work));
    let completion = context.stream().synchronize().map_err(CudaError::from);
    match result {
        Ok(result) => result.and_then(|value| completion.map(|()| value).map_err(Into::into)),
        Err(payload) => resume_unwind(payload),
    }
}

impl PreparedWindow {
    fn commit<F: JoltField>(
        &self,
        context: &CudaKernelContext,
        bases: &DeviceG1Bases,
        plan: &ColumnPlan<'_>,
        global_start: usize,
        local: &CycleWindow,
    ) -> Result<RowTile, KernelError<F>> {
        let hot = hot_column_views(&self.hot);
        let mut columns: Vec<Option<CudaSlice<u64>>> =
            (0..plan.kinds.len()).map(|_| None).collect();
        for index in tier1_order(&hot) {
            let output = if let Some(column) = hot[index].as_ref() {
                let source = column.slice(local.start..local.end());
                let remote = (column.ordinal() != context.ordinal())
                    .then(|| context.stream().clone_dtod(&source))
                    .transpose()
                    .map_err(CudaError::from)?;
                let view = remote.as_ref().map_or(source, |buffer| buffer.slice(..));
                context.one_hot_rows_resident(
                    bases,
                    &view,
                    local.len,
                    plan.one_hot_k,
                    plan.row_width,
                    OneHotMode::Auto,
                )?
            } else {
                let column = self.dense[index]
                    .as_ref()
                    .ok_or(KernelError::InvariantViolation {
                        reason: "a commitment tile has no dense column",
                    })?;
                let tile = column.copy_window(context, local.start, local.len)?;
                context.msm_rows_signed_resident(bases, &tile, plan.row_width)?
            };
            columns[index] = Some(output);
        }
        // Cross-context source-read events cannot be attached to the consuming
        // stream by cudarc. Finish those reads before releasing the input Arc.
        context.stream().synchronize().map_err(CudaError::from)?;
        Ok(RowTile {
            first_block: (global_start + local.start) / plan.row_width,
            blocks: local.len / plan.row_width,
            columns: columns
                .into_iter()
                .map(|column| {
                    column.ok_or(KernelError::InvariantViolation {
                        reason: "a commitment tile did not produce every column",
                    })
                })
                .collect::<Result<_, _>>()?,
        })
    }
}

pub(super) fn suitable(windows: &[CycleWindow], row_width: usize) -> bool {
    row_width > 0
        && windows.len() > 1
        && windows.iter().all(|window| {
            window.len >= MIN_COMMIT_TILE_WINDOW
                && window.len.is_multiple_of(row_width)
                && (window.len / row_width).is_multiple_of(2)
        })
}

pub(super) fn commit<F, PCS>(
    setup: &PCS::ProverSetup,
    plan: &ColumnPlan<'_>,
    host_bases: &[AffineLimbs],
    windows: &[CycleWindow],
    mut prepare: impl FnMut(usize) -> Result<Arc<DeviceTrace>, KernelError<F>>,
) -> Result<PipelineResult<PCS, WindowHotColumns>, KernelError<F>>
where
    F: JoltField,
    PCS: DeviceTier1Commitment,
{
    if windows.is_empty() {
        return Err(KernelError::InvariantViolation {
            reason: "commitment tiles require at least one window",
        });
    }
    let mut end = 0usize;
    for window in windows {
        if plan.row_width == 0
            || plan.one_hot_k == 0
            || window.start != end
            || window.len == 0
            || !window.len.is_multiple_of(plan.row_width)
            || !(window.len / plan.row_width).is_multiple_of(2)
        {
            return Err(KernelError::InvariantViolation {
                reason: "commitment tiles must cover contiguous, complete row pairs",
            });
        }
        end = window
            .start
            .checked_add(window.len)
            .ok_or(KernelError::InvariantViolation {
                reason: "a commitment window end overflowed",
            })?;
    }
    let queue_owner = Queue::new(windows.len());
    let queue = &queue_owner;
    let tasks: Vec<DeviceTask<'_, _, KernelError<F>>> = windows
        .iter()
        .enumerate()
        .map(|(ordinal, _)| {
            let task: DeviceTask<'_, _, KernelError<F>> = Box::new(move || {
                let _device = enter_device(ordinal);
                let run = || -> Result<_, KernelError<F>> {
                    let context = context_for(ordinal).ok_or(KernelError::InvariantViolation {
                        reason: "a commitment worker has no device",
                    })?;
                    complete_work(context, || {
                        let bases = context.upload_g1_bases(host_bases)?;
                        let mut tiles = Vec::new();
                        while let Some(Work {
                            window,
                            local,
                            input,
                        }) = queue.take()
                        {
                            let tile = tracing::info_span!(
                                "cuda_commit_tile",
                                device = ordinal,
                                start = windows[window].start + local.start,
                                cycles = local.len
                            )
                            .in_scope(|| {
                                input.commit::<F>(
                                    context,
                                    &bases,
                                    plan,
                                    windows[window].start,
                                    &local,
                                )
                            })?;
                            tiles.push(tile);
                            queue.finished();
                        }
                        Ok(tiles)
                    })
                };
                match catch_unwind(AssertUnwindSafe(run)) {
                    Ok(result) => {
                        if result.is_err() {
                            queue.cancel();
                        }
                        result
                    }
                    Err(payload) => {
                        queue.cancel();
                        resume_unwind(payload)
                    }
                }
            });
            task
        })
        .collect();
    let mut retained = Vec::with_capacity(windows.len());
    let parent = Span::current();
    let completed = thread::scope(|scope| {
        let workers = scope.spawn(|| parent.in_scope(|| fan_out_bound(tasks)));
        let prepared = catch_unwind(AssertUnwindSafe(|| {
            for (ordinal, window) in windows.iter().enumerate() {
                let trace = prepare(ordinal)?;
                let context = context_for(ordinal).ok_or(KernelError::InvariantViolation {
                    reason: "a commitment window has no device",
                })?;
                let local = CycleWindow {
                    start: 0,
                    len: window.len,
                };
                let hot = retain_hot_columns(hot_columns_from_trace::<F>(
                    &trace,
                    plan.kinds,
                    plan.one_hot_k,
                    &local,
                )?);
                let dense = dense_columns_from_trace::<F>(context, &trace, plan.kinds, local.len)?;
                retained.push(hot.clone());
                queue.prepared(ordinal, local.len, Arc::new(PreparedWindow { hot, dense }));
            }
            Ok::<_, KernelError<F>>(())
        }));
        if !matches!(&prepared, Ok(Ok(()))) {
            queue.cancel();
        }
        let completed = workers.join();
        match prepared {
            Ok(result) => result?,
            Err(payload) => resume_unwind(payload),
        }
        match completed {
            Ok(result) => result,
            Err(payload) => resume_unwind(payload),
        }
    })?;
    // Peer reads are complete after joining; release dense inputs before
    // allocating the assembled columns. The session still owns cached hot rows.
    drop(queue_owner);
    let tiles: Vec<_> = completed.into_iter().flatten().collect();
    let total_blocks = end / plan.row_width;
    let counts: Vec<usize> = plan
        .kinds
        .iter()
        .map(|kind| {
            total_blocks
                .checked_mul(if kind.is_one_hot() { plan.one_hot_k } else { 1 })
                .ok_or(KernelError::InvariantViolation {
                    reason: "a commitment column row count overflowed",
                })
        })
        .collect::<Result<_, _>>()?;
    let selections = device_selections(&counts, windows.len());
    let tiles = &tiles;
    let counts = &counts;
    let tasks: Vec<DeviceTask<'_, _, KernelError<F>>> = selections
        .into_iter()
        .enumerate()
        .map(|(ordinal, selection)| {
            let task: DeviceTask<'_, _, KernelError<F>> = Box::new(move || {
                let _device = enter_device(ordinal);
                let context = context_for(ordinal).ok_or(KernelError::InvariantViolation {
                    reason: "a tier-two commitment worker has no device",
                })?;
                complete_work(context, || {
                    let mut columns = Vec::with_capacity(selection.len());
                    for &index in &selection {
                        let words = counts[index].checked_mul(3 * FQ_LIMBS).ok_or(
                            KernelError::InvariantViolation {
                                reason: "a commitment column allocation size overflowed",
                            },
                        )?;
                        let mut rows = context.alloc_u64_unset(words)?;
                        for tile in tiles {
                            let source = &tile.columns[index];
                            let remote = (source.ordinal() != ordinal)
                                .then(|| context.stream().clone_dtod(source))
                                .transpose()
                                .map_err(CudaError::from)?;
                            context.stitch_commit_rows(
                                remote.as_ref().unwrap_or(source),
                                &mut rows,
                                tile.blocks,
                                total_blocks,
                                tile.first_block,
                            )?;
                        }
                        columns.push(rows);
                    }
                    let finished = tracing::info_span!(
                        "cuda_commit_tier2_window",
                        device = ordinal,
                        columns = columns.len()
                    )
                    .in_scope(|| PCS::tier2_resident(context, setup, &columns))?;
                    if finished.len() != selection.len() {
                        return Err(KernelError::InvariantViolation {
                            reason: "tier two did not finalize every selected column",
                        });
                    }
                    Ok(selection.into_iter().zip(finished).collect::<Vec<_>>())
                })
            });
            task
        })
        .collect();
    // Keep every source tile alive until all peer copies and consumers finish.
    let finished = fan_out_bound(tasks)?;
    let mut placed: Vec<_> = finished.into_iter().flatten().collect();
    placed.sort_unstable_by_key(|(index, _)| *index);
    Ok((
        placed.into_iter().map(|(_, value)| value).collect(),
        retained,
    ))
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test operations must fail loudly")]
mod tests {
    use super::{commit, ColumnPlan, CycleWindow};
    use crate::cuda::commitment::DeviceTier1Commitment;
    use crate::cuda::common::context::context_for;
    use crate::cuda::common::testing::with_r1cs_witness;
    use crate::cuda::witness::session_row_trace_window;
    use crate::reference::commitment::ColumnKind;
    use crate::{KernelError, ProofSession};
    use jolt_claims::protocols::jolt::JoltOneHotConfig;
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_witness::ProgramSource;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    #[test]
    fn preparation_failure_drains_tiled_workers() {
        if context_for(1).is_none() {
            return;
        }
        let config = JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 32,
        };
        let setup = DoryScheme::setup_prover(14);
        let bases = DoryScheme::tier1_bases(&setup, 128).expect("bases");
        let plan = ColumnPlan {
            kinds: &[ColumnKind::RamInc],
            one_hot_k: 16,
            row_width: 128,
        };
        let windows = [
            CycleWindow { start: 0, len: 512 },
            CycleWindow {
                start: 512,
                len: 512,
            },
        ];
        with_r1cs_witness(10, 1024, config, 29, |witness| {
            for panic in [false, true] {
                let mut session = ProofSession::default();
                let result = catch_unwind(AssertUnwindSafe(|| {
                    commit::<Fr, DoryScheme>(&setup, &plan, &bases, &windows, |ordinal| {
                        if ordinal == 1 {
                            assert!(!panic, "injected preparation panic");
                            return Err(KernelError::InvariantViolation {
                                reason: "injected preparation failure",
                            });
                        }
                        session_row_trace_window(
                            context_for(ordinal).expect("device"),
                            &mut session,
                            witness,
                            witness.program_preprocessing(),
                            1024,
                            &windows[ordinal].residency(1024),
                        )
                    })
                }));
                if panic {
                    assert!(result.is_err());
                } else {
                    assert!(matches!(
                        result.expect("preparation returns an error"),
                        Err(KernelError::InvariantViolation {
                            reason: "injected preparation failure"
                        })
                    ));
                }
            }
        });
    }
}
