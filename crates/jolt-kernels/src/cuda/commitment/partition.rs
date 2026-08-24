use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

use jolt_field::Field;
use jolt_program::execution::TraceRow;
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_witness::backend::cuda::DeviceTrace;

use super::{
    dense_columns_from_trace, hot_column_views, hot_columns_from_trace, retain_hot_columns,
    tier1_order, ColumnKind, DeviceTier1Commitment, FinishedColumn, RetainedHotColumns,
    WindowHotColumns,
};
use crate::cuda::common::context::{context_for, enter_device, CudaKernelContext};
use crate::cuda::common::devices::{device_selections, fan_out_bound, CycleWindow, DeviceTask};
use crate::cuda::common::msm::{AffineLimbs, DeviceG1Bases, JacobianLimbs, SignedColumn};
use crate::KernelError;

pub(super) struct ColumnPlan<'a> {
    pub(super) kinds: &'a [ColumnKind],
    pub(super) cycles: usize,
    pub(super) one_hot_k: usize,
    pub(super) row_width: usize,
}

pub(super) struct TraceSource<'a> {
    pub(super) rows: &'a [TraceRow],
    pub(super) preprocessing: &'a JoltProgramPreprocessing,
}

type ColumnSink<'a, F> = &'a mut dyn FnMut(usize, Vec<JacobianLimbs>) -> Result<(), KernelError<F>>;

pub(super) fn window_columns<F: Field>(
    context: &CudaKernelContext,
    bases: &DeviceG1Bases,
    hot: &RetainedHotColumns,
    dense: &[Option<SignedColumn>],
    plan: &ColumnPlan<'_>,
    view: &CycleWindow,
    emit: ColumnSink<'_, F>,
) -> Result<(), KernelError<F>> {
    for index in tier1_order(hot) {
        let rows = if let Some(column) = hot.get(index).and_then(Option::as_ref) {
            context.require_owned(column.ordinal())?;
            let slice = column.slice(view.start..view.end());
            tracing::info_span!("cuda_commit_tier1_one_hot").in_scope(|| {
                context.one_hot_rows_device(bases, &slice, view.len, plan.one_hot_k, plan.row_width)
            })
        } else {
            let column = dense.get(index).and_then(Option::as_ref).ok_or(
                KernelError::InvariantViolation {
                    reason: "the commit pipeline has no increment column for a dense id",
                },
            )?;
            tracing::info_span!("cuda_commit_tier1_dense")
                .in_scope(|| context.msm_rows_signed(bases, column, plan.row_width))
        }?;
        emit(index, rows)?;
    }
    Ok(())
}

fn stitch_column<F: Field>(
    parts: &[&[JacobianLimbs]],
    one_hot: bool,
    one_hot_k: usize,
    blocks: &[usize],
) -> Result<Vec<JacobianLimbs>, KernelError<F>> {
    let mut rows = Vec::with_capacity(parts.iter().map(|part| part.len()).sum());
    if !one_hot {
        for part in parts {
            rows.extend_from_slice(part);
        }
        return Ok(rows);
    }
    for address in 0..one_hot_k {
        for (part, &blocks) in parts.iter().zip(blocks) {
            let segment = part.get(address * blocks..(address + 1) * blocks).ok_or(
                KernelError::InvariantViolation {
                    reason: "a one-hot commit window returned fewer segments than its address \
                             count",
                },
            )?;
            rows.extend_from_slice(segment);
        }
    }
    Ok(rows)
}

pub(super) struct Tier1Part {
    pub(super) window: usize,
    pub(super) index: usize,
    pub(super) rows: Vec<JacobianLimbs>,
}

pub(super) struct PartBoard {
    parts: Vec<Vec<Option<Vec<JacobianLimbs>>>>,
    done: Vec<bool>,
    one_hot: Vec<bool>,
    blocks: Vec<usize>,
    one_hot_k: usize,
}

impl PartBoard {
    pub(super) fn new(plan: &ColumnPlan<'_>, blocks: &[usize]) -> Self {
        let one_hot: Vec<bool> = plan.kinds.iter().map(|kind| kind.is_one_hot()).collect();
        Self {
            parts: (0..one_hot.len())
                .map(|_| (0..blocks.len()).map(|_| None).collect())
                .collect(),
            done: vec![false; one_hot.len()],
            one_hot,
            blocks: blocks.to_vec(),
            one_hot_k: plan.one_hot_k,
        }
    }

    pub(super) fn place<F: Field>(
        &mut self,
        part: Tier1Part,
    ) -> Result<Option<(usize, Vec<JacobianLimbs>)>, KernelError<F>> {
        let index = part.index;
        let outside = || KernelError::InvariantViolation {
            reason: "a commit cycle window produced a column outside the id list",
        };
        if *self.done.get(index).ok_or_else(outside)? {
            return Err(KernelError::InvariantViolation {
                reason: "a commit cycle window delivered a column that was already stitched",
            });
        }
        let complete = {
            let slots = self.parts.get_mut(index).ok_or_else(outside)?;
            let slot = slots
                .get_mut(part.window)
                .ok_or(KernelError::InvariantViolation {
                    reason: "a commit cycle window delivered a column outside the window list",
                })?;
            if slot.is_some() {
                return Err(KernelError::InvariantViolation {
                    reason: "a commit cycle window delivered the same column twice",
                });
            }
            *slot = Some(part.rows);
            slots.iter().all(Option::is_some)
        };
        if !complete {
            return Ok(None);
        }
        let one_hot = *self.one_hot.get(index).ok_or_else(outside)?;
        let slots = self.parts.get(index).ok_or_else(outside)?;
        let mut views: Vec<&[JacobianLimbs]> = Vec::with_capacity(slots.len());
        for slot in slots {
            views.push(slot.as_deref().ok_or(KernelError::InvariantViolation {
                reason: "a commit cycle window produced no rows for a column",
            })?);
        }
        let rows = stitch_column::<F>(&views, one_hot, self.one_hot_k, &self.blocks)?;
        if let Some(slots) = self.parts.get_mut(index) {
            for slot in slots.iter_mut() {
                *slot = None;
            }
        }
        if let Some(done) = self.done.get_mut(index) {
            *done = true;
        }
        Ok(Some((index, rows)))
    }
}

fn column_rows(plan: &ColumnPlan<'_>, blocks: &[usize]) -> Vec<usize> {
    let total: usize = blocks.iter().sum();
    plan.kinds
        .iter()
        .map(|kind| {
            if kind.is_one_hot() {
                plan.one_hot_k * total
            } else {
                total
            }
        })
        .collect()
}

pub(super) type PipelineResult<PCS, W> = (Vec<FinishedColumn<PCS>>, Vec<W>);

pub(super) type Tier1Producer<'a, F, W> = Box<
    dyn FnOnce(&'static CudaKernelContext, usize, &Sender<Tier1Part>) -> Result<W, KernelError<F>>
        + Send
        + 'a,
>;

type StitchedColumn = (usize, Vec<JacobianLimbs>);

type WindowTask<'a, F, PCS, W> =
    DeviceTask<'a, (Vec<(usize, FinishedColumn<PCS>)>, W), KernelError<F>>;

fn coordinate<F: Field>(
    parts: Receiver<Tier1Part>,
    plan: &ColumnPlan<'_>,
    blocks: &[usize],
    owner: &[usize],
    senders: Vec<Sender<StitchedColumn>>,
    parent: &tracing::Span,
) -> Result<(), KernelError<F>> {
    let mut board = PartBoard::new(plan, blocks);
    for part in parts {
        let placed = tracing::info_span!(parent: parent, "cuda_commit_stitch")
            .in_scope(|| board.place::<F>(part))?;
        if let Some((index, rows)) = placed {
            let ordinal = *owner.get(index).ok_or(KernelError::InvariantViolation {
                reason: "the commit pipeline stitched a column no device owns",
            })?;
            let sender = senders
                .get(ordinal)
                .ok_or(KernelError::InvariantViolation {
                    reason: "the commit pipeline stitched a column for an absent device",
                })?;
            let _ = sender.send((index, rows));
        }
    }
    Ok(())
}

fn tier2_groups(selection: &[usize], counts: &[usize]) -> Vec<(usize, usize)> {
    let mut groups: Vec<(usize, usize)> = Vec::new();
    for &index in selection {
        let rows = counts.get(index).copied().unwrap_or(0);
        match groups.iter_mut().find(|(count, _)| *count == rows) {
            Some((_, members)) => *members += 1,
            None => groups.push((rows, 1)),
        }
    }
    groups
}

fn consume_tier2<F, PCS>(
    context: &'static CudaKernelContext,
    setup: &PCS::ProverSetup,
    stitched: &Receiver<StitchedColumn>,
    selection: &[usize],
    counts: &[usize],
    ordinal: usize,
) -> Result<Vec<(usize, FinishedColumn<PCS>)>, KernelError<F>>
where
    F: Field,
    PCS: DeviceTier1Commitment,
{
    let mut groups = tier2_groups(selection, counts);
    let mut ready: Vec<Vec<usize>> = groups.iter().map(|_| Vec::new()).collect();
    let mut columns: Vec<Vec<JacobianLimbs>> = (0..counts.len()).map(|_| Vec::new()).collect();
    let mut finished = Vec::with_capacity(selection.len());
    let mut outstanding = groups.len();
    while outstanding > 0 {
        let Ok((index, rows)) = stitched.recv() else {
            return Err(KernelError::InvariantViolation {
                reason: "a commit cycle window failed before every column was stitched",
            });
        };
        let group = groups
            .iter()
            .position(|&(count, _)| count == rows.len())
            .ok_or(KernelError::InvariantViolation {
                reason: "the commit pipeline stitched a column whose row count no tier-2 group \
                         expects",
            })?;
        let slot = columns
            .get_mut(index)
            .ok_or(KernelError::InvariantViolation {
                reason: "the commit pipeline stitched a column outside the id list",
            })?;
        *slot = rows;
        let members = ready
            .get_mut(group)
            .ok_or(KernelError::InvariantViolation {
                reason: "the commit pipeline lost a tier-2 group",
            })?;
        members.push(index);
        let (rows, expected) = groups.get(group).copied().unwrap_or((0, 0));
        if members.len() < expected {
            continue;
        }
        members.sort_unstable();
        let batch = tracing::info_span!(
            "cuda_commit_tier2_window",
            device = ordinal,
            columns = members.len(),
            rows
        )
        .in_scope(|| PCS::tier2_selected(context, setup, &columns, members))?;
        finished.extend(batch);
        for &index in members.iter() {
            if let Some(slot) = columns.get_mut(index) {
                *slot = Vec::new();
            }
        }
        members.clear();
        if let Some(group) = groups.get_mut(group) {
            group.1 = usize::MAX;
        }
        outstanding -= 1;
    }
    Ok(finished)
}

pub(super) fn pipeline_columns<F, PCS, W, C>(
    setup: &PCS::ProverSetup,
    plan: &ColumnPlan<'_>,
    blocks: &[usize],
    context_of: &C,
    producers: Vec<Tier1Producer<'_, F, W>>,
) -> Result<PipelineResult<PCS, W>, KernelError<F>>
where
    F: Field,
    PCS: DeviceTier1Commitment,
    W: Send,
    C: Fn(usize) -> Option<&'static CudaKernelContext> + Sync,
{
    if blocks.len() != producers.len() {
        return Err(KernelError::InvariantViolation {
            reason: "the commit pipeline was given a window count that does not match its \
                     tier-1 producers",
        });
    }
    if plan.kinds.is_empty() {
        return Err(KernelError::InvariantViolation {
            reason: "the commit pipeline was given no columns to commit",
        });
    }
    let counts = column_rows(plan, blocks);
    if counts.contains(&0) {
        return Err(KernelError::InvariantViolation {
            reason: "the commit pipeline was given a column with no rows",
        });
    }
    let mut contexts = Vec::with_capacity(producers.len());
    for ordinal in 0..producers.len() {
        contexts.push(context_of(ordinal).ok_or(KernelError::Unsupported {
            reason: "a CUDA commit window was scheduled onto an absent device",
        })?);
    }
    let selections = device_selections(&counts, producers.len());
    let mut owner = vec![0usize; counts.len()];
    for (ordinal, selection) in selections.iter().enumerate() {
        for &index in selection {
            if let Some(slot) = owner.get_mut(index) {
                *slot = ordinal;
            }
        }
    }

    let (parts_tx, parts_rx) = mpsc::channel::<Tier1Part>();
    let mut senders = Vec::with_capacity(producers.len());
    let mut receivers = Vec::with_capacity(producers.len());
    for _ in 0..producers.len() {
        let (tx, rx) = mpsc::channel::<StitchedColumn>();
        senders.push(tx);
        receivers.push(Some(rx));
    }
    let mut parts: Vec<Option<Sender<Tier1Part>>> = (0..producers.len())
        .map(|_| Some(parts_tx.clone()))
        .collect();
    drop(parts_tx);

    let parent = tracing::Span::current();
    let stitching = parent.clone();
    std::thread::scope(|scope| {
        let coordinator = scope
            .spawn(move || coordinate::<F>(parts_rx, plan, blocks, &owner, senders, &stitching));
        let mut tasks: Vec<WindowTask<'_, F, PCS, W>> = Vec::with_capacity(producers.len());
        for (ordinal, producer) in producers.into_iter().enumerate() {
            let context = contexts.get(ordinal).copied();
            let stitched = receivers.get_mut(ordinal).and_then(Option::take);
            let sender = parts.get_mut(ordinal).and_then(Option::take);
            let selection = selections.get(ordinal);
            let counts = &counts;
            let parent = parent.clone();
            tasks.push(Box::new(move || {
                let absent = || KernelError::InvariantViolation {
                    reason: "the commit pipeline lost a window's device wiring",
                };
                let context = context.ok_or_else(absent)?;
                let stitched = stitched.ok_or_else(absent)?;
                let sender = sender.ok_or_else(absent)?;
                let selection = selection.ok_or_else(absent)?;
                let _device = enter_device(context.ordinal());
                let window = tracing::info_span!(
                    parent: &parent,
                    "cuda_commit_tier1_window",
                    device = context.ordinal()
                )
                .in_scope(|| producer(context, ordinal, &sender))?;
                drop(sender);
                let finished = consume_tier2::<F, PCS>(
                    context,
                    setup,
                    &stitched,
                    selection,
                    counts,
                    context.ordinal(),
                )?;
                Ok((finished, window))
            }));
        }
        let outcome = tracing::info_span!("cuda_commit_tier1_fanout", windows = tasks.len())
            .in_scope(|| fan_out_bound(tasks));
        match coordinator.join() {
            Ok(stitch) => {
                let done = outcome?;
                stitch?;
                Ok(done)
            }
            Err(payload) => std::panic::resume_unwind(payload),
        }
    })
    .and_then(|done| {
        let mut placed: Vec<Option<FinishedColumn<PCS>>> =
            (0..counts.len()).map(|_| None).collect();
        let mut windows = Vec::with_capacity(done.len());
        for (finished, window) in done {
            for (index, column) in finished {
                let slot = placed
                    .get_mut(index)
                    .ok_or(KernelError::InvariantViolation {
                        reason: "the commit pipeline finished a column outside the id list",
                    })?;
                *slot = Some(column);
            }
            windows.push(window);
        }
        let columns = placed.into_iter().collect::<Option<Vec<_>>>().ok_or(
            KernelError::InvariantViolation {
                reason: "the commit pipeline finished fewer columns than it was given",
            },
        )?;
        Ok((columns, windows))
    })
}

type WindowState = (Option<Arc<DeviceTrace>>, WindowHotColumns);

type SplitColumns<PCS> = (
    Vec<FinishedColumn<PCS>>,
    Vec<Option<Arc<DeviceTrace>>>,
    Vec<WindowHotColumns>,
);

fn window_trace<F: Field>(
    context: &CudaKernelContext,
    plan: &ColumnPlan<'_>,
    source: &TraceSource<'_>,
    window: &CycleWindow,
    ordinal: usize,
) -> Result<Arc<DeviceTrace>, KernelError<F>> {
    let trace = tracing::info_span!(
        "cuda_commit_window_residency",
        device = ordinal,
        cycles = window.len
    )
    .in_scope(|| {
        DeviceTrace::upload_window(
            Arc::clone(context.stream()),
            source.rows,
            plan.cycles,
            window.start,
            window.residency(plan.cycles).len,
            source.preprocessing,
        )
    })?;
    Ok(Arc::new(trace))
}

pub(super) fn split_columns<F, PCS>(
    setup: &PCS::ProverSetup,
    bases: &DeviceG1Bases,
    resident: Option<Arc<DeviceTrace>>,
    plan: &ColumnPlan<'_>,
    source: &TraceSource<'_>,
    host_bases: &[AffineLimbs],
    windows: &[CycleWindow],
) -> Result<SplitColumns<PCS>, KernelError<F>>
where
    F: Field,
    PCS: DeviceTier1Commitment,
{
    let blocks: Vec<usize> = windows
        .iter()
        .map(|window| window.len / plan.row_width)
        .collect();
    let mut producers: Vec<Tier1Producer<'_, F, WindowState>> = Vec::with_capacity(windows.len());
    for (ordinal, window) in windows.iter().enumerate() {
        let resident = (ordinal == 0).then(|| resident.clone()).flatten();
        producers.push(Box::new(move |context, ordinal, sink| {
            let (trace, parked) = if let Some(trace) = resident {
                (trace, None)
            } else {
                let trace = window_trace::<F>(context, plan, source, window, ordinal)?;
                (Arc::clone(&trace), Some(trace))
            };
            let built = tracing::info_span!(
                "cuda_commit_park_hot",
                device = ordinal,
                cycles = window.len
            )
            .in_scope(|| {
                hot_columns_from_trace::<F>(
                    &trace,
                    plan.kinds,
                    plan.one_hot_k,
                    &CycleWindow {
                        start: 0,
                        len: window.len,
                    },
                )
            })?;
            let built = retain_hot_columns(built);
            let hot: RetainedHotColumns = hot_column_views(&built);
            let dense = tracing::info_span!(
                "cuda_commit_increments",
                device = ordinal,
                cycles = window.len
            )
            .in_scope(|| dense_columns_from_trace::<F>(context, &trace, plan.kinds, window.len))?;
            let uploaded = (ordinal > 0)
                .then(|| context.upload_g1_bases(host_bases))
                .transpose()?;
            window_columns::<F>(
                context,
                uploaded.as_ref().unwrap_or(bases),
                &hot,
                &dense,
                plan,
                &CycleWindow {
                    start: 0,
                    len: window.len,
                },
                &mut |index, rows| {
                    sink.send(Tier1Part {
                        window: ordinal,
                        index,
                        rows,
                    })
                    .map_err(|_| KernelError::InvariantViolation {
                        reason: "the commit pipeline closed its stitch channel while tier-1 was \
                                 still producing rows",
                    })
                },
            )?;
            Ok((parked, built))
        }));
    }
    let (columns, states) =
        pipeline_columns::<F, PCS, WindowState, _>(setup, plan, &blocks, &context_for, producers)?;
    let mut traces = Vec::with_capacity(states.len());
    let mut hots = Vec::with_capacity(states.len());
    for (trace, built) in states {
        traces.push(trace);
        hots.push(built);
    }
    Ok((columns, traces, hots))
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use ark_bn254::G1Projective;
    use ark_ec::{CurveGroup, PrimeGroup};
    use jolt_field::Fr;

    use super::{
        window_columns, ColumnKind, ColumnPlan, CycleWindow, JacobianLimbs, PartBoard, Tier1Part,
    };
    use crate::cuda::commitment::{affine_limbs, fq_from_limbs};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::pack::COLD;
    use jolt_witness::witnesses::RaChunkSelector;
    use std::sync::Arc;

    const CYCLES: usize = 64;
    const ROW_WIDTH: usize = 8;
    const ONE_HOT_K: usize = 4;

    fn point(scalar: u64) -> G1Projective {
        G1Projective::generator() * ark_bn254::Fr::from(scalar)
    }

    fn projective(limbs: JacobianLimbs) -> G1Projective {
        G1Projective::new_unchecked(
            fq_from_limbs(limbs.x),
            fq_from_limbs(limbs.y),
            fq_from_limbs(limbs.z),
        )
    }

    fn as_points(columns: &[Option<Vec<JacobianLimbs>>]) -> Vec<Vec<G1Projective>> {
        columns
            .iter()
            .map(|column| {
                column
                    .as_ref()
                    .expect("every column produced rows")
                    .iter()
                    .copied()
                    .map(projective)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn an_incremental_board_stitches_the_same_columns_in_any_arrival_order() {
        let Some(context) = shared_context() else {
            return;
        };
        let host_bases: Vec<_> = (0..ROW_WIDTH)
            .map(|column| affine_limbs(point(column as u64 + 3).into_affine()))
            .collect();
        let bases = context.upload_g1_bases(&host_bases).expect("upload bases");
        let words: Vec<u32> = (0..CYCLES)
            .map(|cycle| {
                if cycle.is_multiple_of(11) {
                    COLD
                } else {
                    (cycle % ONE_HOT_K) as u32
                }
            })
            .collect();
        let hot = vec![
            Some(Arc::new(
                context.upload_u32_slice(&words).expect("upload hot column"),
            )),
            None,
            Some(Arc::new(
                context.upload_u32_slice(&words).expect("upload hot column"),
            )),
        ];
        let increments: Vec<i128> = (0..CYCLES).map(|cycle| 31 - cycle as i128).collect();
        let dense = |range: std::ops::Range<usize>| {
            vec![
                None,
                Some(
                    context
                        .signed_column(&increments[range])
                        .expect("host increment column"),
                ),
                None,
            ]
        };
        let selector = |chunk: usize| {
            ColumnKind::InstructionRa(
                RaChunkSelector::new(chunk, 2, ONE_HOT_K.ilog2() as usize).expect("chunk selector"),
            )
        };
        let kinds = [selector(0), ColumnKind::RamInc, selector(1)];
        let plan = ColumnPlan {
            kinds: &kinds,
            cycles: CYCLES,
            one_hot_k: ONE_HOT_K,
            row_width: ROW_WIDTH,
        };
        let collect = |window: &CycleWindow| {
            let mut columns: Vec<Option<Vec<JacobianLimbs>>> =
                (0..kinds.len()).map(|_| None).collect();
            window_columns::<Fr>(
                context,
                &bases,
                &hot,
                &dense(window.start..window.end()),
                &plan,
                window,
                &mut |index, rows| {
                    columns[index] = Some(rows);
                    Ok(())
                },
            )
            .expect("window columns");
            columns
        };

        let expected = as_points(&collect(&CycleWindow {
            start: 0,
            len: CYCLES,
        }));

        for devices in [1usize, 2, 4] {
            let windows: Vec<CycleWindow> = (0..devices)
                .map(|device| CycleWindow {
                    start: device * CYCLES / devices,
                    len: CYCLES / devices,
                })
                .collect();
            let blocks: Vec<usize> = windows
                .iter()
                .map(|window| window.len / ROW_WIDTH)
                .collect();
            let parts: Vec<Vec<Option<Vec<JacobianLimbs>>>> = windows.iter().map(collect).collect();

            let mut arrivals: Vec<(usize, usize)> = Vec::new();
            for window in 0..devices {
                for index in 0..kinds.len() {
                    arrivals.push((window, index));
                }
            }
            let orders: [Vec<(usize, usize)>; 3] = [
                arrivals.clone(),
                arrivals.iter().copied().rev().collect(),
                {
                    let mut order = arrivals.clone();
                    order.sort_by_key(|&(window, index)| (index, devices - window));
                    order
                },
            ];

            for (variant, order) in orders.iter().enumerate() {
                let mut board = PartBoard::new(&plan, &blocks);
                let mut columns: Vec<Option<Vec<JacobianLimbs>>> =
                    (0..kinds.len()).map(|_| None).collect();
                for &(window, index) in order {
                    let rows = parts[window][index]
                        .as_ref()
                        .expect("a window part")
                        .clone();
                    let placed = board
                        .place::<Fr>(Tier1Part {
                            window,
                            index,
                            rows,
                        })
                        .expect("place a part");
                    if let Some((index, rows)) = placed {
                        columns[index] = Some(rows);
                    }
                }
                assert_eq!(
                    as_points(&columns),
                    expected,
                    "arrival order {variant} across {devices} windows changed the stitched tier-1 \
                     rows; the board must place a part by its (window, column) coordinates, not \
                     by the order it arrives",
                );
            }
        }
    }

    #[cfg(not(feature = "zk"))]
    #[test]
    fn pipelined_columns_match_reference_dory_under_skewed_arrival() {
        use std::sync::mpsc::Sender;
        use std::sync::Barrier;

        use jolt_dory::DoryScheme;
        use jolt_openings::StreamingCommitment;

        use crate::cuda::commitment::DeviceTier1Commitment;

        use super::{pipeline_columns, Tier1Producer};
        use crate::cuda::dory::CudaDoryScheme;
        use crate::KernelError;

        const NUM_VARS: usize = 10;

        let Some(context) = shared_context() else {
            return;
        };
        let host_bases: Vec<_> = (0..ROW_WIDTH)
            .map(|column| affine_limbs(point(column as u64 + 5).into_affine()))
            .collect();
        let bases = context.upload_g1_bases(&host_bases).expect("upload bases");
        let words = |salt: usize| -> Vec<u32> {
            (0..CYCLES)
                .map(|cycle| {
                    if (cycle + salt).is_multiple_of(9) {
                        COLD
                    } else {
                        ((cycle + salt) % ONE_HOT_K) as u32
                    }
                })
                .collect()
        };
        let upload = |salt: usize| {
            Some(Arc::new(
                context
                    .upload_u32_slice(&words(salt))
                    .expect("upload hot column"),
            ))
        };
        let hot = vec![upload(0), None, upload(1), upload(2), None];
        let increments = |salt: i128| -> Vec<i128> {
            (0..CYCLES)
                .map(|cycle| salt * 7 + cycle as i128 - 23)
                .collect()
        };
        let dense = |range: std::ops::Range<usize>| {
            let column = |salt: i128| {
                Some(
                    context
                        .signed_column(&increments(salt)[range.clone()])
                        .expect("host increment column"),
                )
            };
            vec![None, column(1), None, None, column(-2)]
        };
        let selector = |chunk: usize| {
            ColumnKind::InstructionRa(
                RaChunkSelector::new(chunk, 3, ONE_HOT_K.ilog2() as usize).expect("chunk selector"),
            )
        };
        let kinds = [
            selector(0),
            ColumnKind::RdInc,
            selector(1),
            selector(2),
            ColumnKind::RamInc,
        ];
        let plan = ColumnPlan {
            kinds: &kinds,
            cycles: CYCLES,
            one_hot_k: ONE_HOT_K,
            row_width: ROW_WIDTH,
        };
        let collect = |window: &CycleWindow| {
            let mut columns: Vec<Option<Vec<JacobianLimbs>>> =
                (0..kinds.len()).map(|_| None).collect();
            window_columns::<Fr>(
                context,
                &bases,
                &hot,
                &dense(window.start..window.end()),
                &plan,
                window,
                &mut |index, rows| {
                    columns[index] = Some(rows);
                    Ok(())
                },
            )
            .expect("window columns");
            columns
        };

        let setup = CudaDoryScheme::setup_prover(NUM_VARS);
        let expected: Vec<_> = collect(&CycleWindow {
            start: 0,
            len: CYCLES,
        })
        .iter()
        .map(|rows| {
            let rows = rows.as_ref().expect("every column produced rows");
            let partial = DoryScheme::partial_from_rows(&setup, rows).expect("partial");
            let commitment = DoryScheme::finish_with_hint(partial.clone(), &setup).0;
            (commitment, partial.row_commitments)
        })
        .collect();

        for devices in [1usize, 2, 4] {
            for barrier in [false, true] {
                if barrier && devices != 2 {
                    continue;
                }
                let windows: Vec<CycleWindow> = (0..devices)
                    .map(|device| CycleWindow {
                        start: device * CYCLES / devices,
                        len: CYCLES / devices,
                    })
                    .collect();
                let blocks: Vec<usize> = windows
                    .iter()
                    .map(|window| window.len / ROW_WIDTH)
                    .collect();
                let parts: Vec<Vec<Option<Vec<JacobianLimbs>>>> =
                    windows.iter().map(collect).collect();
                let gate = barrier.then(|| Arc::new(Barrier::new(devices)));
                let producers: Vec<Tier1Producer<'_, Fr, ()>> = parts
                    .into_iter()
                    .map(|part| {
                        let gate = gate.clone();
                        let producer: Tier1Producer<'_, Fr, ()> =
                            Box::new(move |_context, window, tx: &Sender<Tier1Part>| {
                                if window == 0 {
                                    if let Some(gate) = &gate {
                                        let _ = gate.wait();
                                    }
                                }
                                let mut ordered: Vec<(usize, Vec<JacobianLimbs>)> = part
                                    .into_iter()
                                    .enumerate()
                                    .filter_map(|(index, rows)| rows.map(|rows| (index, rows)))
                                    .collect();
                                if !window.is_multiple_of(2) {
                                    ordered.reverse();
                                }
                                for (index, rows) in ordered {
                                    tx.send(Tier1Part {
                                        window,
                                        index,
                                        rows,
                                    })
                                    .map_err(|_| {
                                        KernelError::<Fr>::InvariantViolation {
                                            reason: "the commit pipeline dropped its part channel",
                                        }
                                    })?;
                                }
                                if window == 1 {
                                    if let Some(gate) = &gate {
                                        let _ = gate.wait();
                                    }
                                }
                                Ok(())
                            });
                        producer
                    })
                    .collect();

                let (got, _) = pipeline_columns::<Fr, CudaDoryScheme, (), _>(
                    &setup,
                    &plan,
                    &blocks,
                    &|_| shared_context(),
                    producers,
                )
                .expect("pipelined columns");

                assert_eq!(
                    got.len(),
                    expected.len(),
                    "the pipeline finished {} of {} columns across {devices} windows",
                    got.len(),
                    expected.len(),
                );
                let divergence =
                    got.iter()
                        .zip(&expected)
                        .position(|((commitment, hint), (want, rows))| {
                            commitment != want || &hint.row_commitments != rows
                        });
                assert_eq!(
                    divergence, None,
                    "the pipelined commit diverged from reference Dory at column \
                     {divergence:?} across {devices} windows (barrier: {barrier}); a column's \
                     commitment and its opening hint must depend only on its stitched rows, \
                     never on which device finished it or when",
                );
            }
        }
    }

    #[test]
    fn cycle_windows_cover_the_domain_without_overlap() {
        let windows = crate::cuda::common::devices::committed_windows(CYCLES, ROW_WIDTH);
        let mut next = 0;
        for window in &windows {
            assert_eq!(window.start, next, "windows must be contiguous");
            assert!(
                window.len.is_multiple_of(ROW_WIDTH),
                "a window that splits a row block would mix cycles across row commitments",
            );
            next = window.end();
        }
        assert_eq!(
            next, CYCLES,
            "the windows must cover the whole cycle domain"
        );
    }

    #[test]
    fn unaligned_or_undersized_domains_stay_whole() {
        let single = |windows: Vec<CycleWindow>| {
            assert_eq!(windows.len(), 1);
            assert_eq!(windows[0].start, 0);
        };
        let windows = crate::cuda::common::devices::committed_windows;
        single(windows(CYCLES + 1, ROW_WIDTH));
        single(windows(ROW_WIDTH, ROW_WIDTH));
        single(windows(CYCLES, 0));
    }
}
