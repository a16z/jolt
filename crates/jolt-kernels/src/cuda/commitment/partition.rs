use std::sync::Arc;

use jolt_field::Field;
use jolt_program::execution::TraceRow;
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_witness::backend::cuda::DeviceTrace;

use super::{
    hot_column_views, hot_columns_from_trace, retain_hot_columns, tier1_order, ColumnKind,
    RetainedHotColumns, WindowHotColumns,
};
use crate::cuda::common::context::{context_for, CudaKernelContext};
use crate::cuda::common::devices::{fan_out, CycleWindow, DeviceTask};
use crate::cuda::common::msm::{AffineLimbs, DeviceG1Bases, JacobianLimbs};
use crate::KernelError;

pub(super) struct ColumnPlan<'a> {
    pub(super) kinds: &'a [ColumnKind],
    pub(super) increments: &'a [Vec<i128>],
    pub(super) cycles: usize,
    pub(super) one_hot_k: usize,
    pub(super) row_width: usize,
}

pub(super) struct TraceSource<'a> {
    pub(super) rows: &'a [TraceRow],
    pub(super) preprocessing: &'a JoltProgramPreprocessing,
}

type Columns = Vec<Option<Vec<JacobianLimbs>>>;

pub(super) fn window_columns<F: Field>(
    context: &CudaKernelContext,
    bases: &DeviceG1Bases,
    hot: &RetainedHotColumns,
    plan: &ColumnPlan<'_>,
    window: &CycleWindow,
    hot_offset: usize,
) -> Result<Columns, KernelError<F>> {
    let mut columns: Columns = (0..hot.len()).map(|_| None).collect();
    for index in tier1_order(hot) {
        let rows = if let Some(column) = hot.get(index).and_then(Option::as_ref) {
            context.require_owned(column.ordinal())?;
            let view = column.slice(hot_offset..hot_offset + window.len);
            tracing::info_span!("cuda_commit_tier1_one_hot").in_scope(|| {
                context.one_hot_rows_device(
                    bases,
                    &view,
                    window.len,
                    plan.one_hot_k,
                    plan.row_width,
                )
            })
        } else {
            let increments = plan
                .increments
                .get(index)
                .ok_or(KernelError::InvariantViolation {
                    reason: "the commit pipeline produced a column outside the id list",
                })?;
            let slice = increments.get(window.start..window.end()).ok_or(
                KernelError::InvariantViolation {
                    reason: "a commit cycle window lies outside the collected increments",
                },
            )?;
            tracing::info_span!("cuda_commit_tier1_dense")
                .in_scope(|| context.msm_rows_i128(bases, slice, plan.row_width))
        }?;
        let slot = columns
            .get_mut(index)
            .ok_or(KernelError::InvariantViolation {
                reason: "the commit pipeline produced a column outside the id list",
            })?;
        *slot = Some(rows);
    }
    Ok(columns)
}

fn stitch<F: Field>(
    parts: Vec<Columns>,
    windows: &[CycleWindow],
    hot: &RetainedHotColumns,
    plan: &ColumnPlan<'_>,
) -> Result<Columns, KernelError<F>> {
    let mut columns: Columns = (0..hot.len()).map(|_| None).collect();
    for (index, slot) in columns.iter_mut().enumerate() {
        let one_hot = hot.get(index).is_some_and(Option::is_some);
        let mut rows = Vec::new();
        if one_hot {
            for address in 0..plan.one_hot_k {
                for (part, window) in parts.iter().zip(windows) {
                    let blocks = window.len / plan.row_width;
                    let produced = part.get(index).and_then(Option::as_ref).ok_or(
                        KernelError::InvariantViolation {
                            reason: "a commit cycle window produced no rows for a column",
                        },
                    )?;
                    let segment = produced
                        .get(address * blocks..(address + 1) * blocks)
                        .ok_or(KernelError::InvariantViolation {
                            reason: "a one-hot commit window returned fewer segments than its \
                                     address count",
                        })?;
                    rows.extend_from_slice(segment);
                }
            }
        } else {
            for part in &parts {
                let produced = part.get(index).and_then(Option::as_ref).ok_or(
                    KernelError::InvariantViolation {
                        reason: "a commit cycle window produced no rows for a column",
                    },
                )?;
                rows.extend_from_slice(produced);
            }
        }
        *slot = Some(rows);
    }
    Ok(columns)
}

type WindowResult = (Columns, Option<Arc<DeviceTrace>>, WindowHotColumns);

type SplitColumns = (
    Columns,
    Vec<Option<Arc<DeviceTrace>>>,
    Vec<WindowHotColumns>,
);

pub(super) fn split_columns<F: Field>(
    bases: &DeviceG1Bases,
    hot: &RetainedHotColumns,
    plan: &ColumnPlan<'_>,
    source: &TraceSource<'_>,
    host_bases: &[AffineLimbs],
    windows: &[CycleWindow],
) -> Result<SplitColumns, KernelError<F>> {
    let mut tasks: Vec<DeviceTask<'_, WindowResult, KernelError<F>>> =
        Vec::with_capacity(windows.len());
    for (ordinal, window) in windows.iter().enumerate() {
        if ordinal == 0 {
            tasks.push(Box::new(move || {
                let context = crate::cuda::require_context::<F>()?;
                Ok((
                    window_columns::<F>(context, bases, hot, plan, window, 0)?,
                    None,
                    Vec::new(),
                ))
            }));
            continue;
        }
        tasks.push(Box::new(move || {
            let context = context_for(ordinal).ok_or(KernelError::Unsupported {
                reason: "a CUDA commit window was scheduled onto an absent device",
            })?;
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
            let trace = Arc::new(trace);
            let built =
                tracing::info_span!("cuda_commit_park_hot", device = ordinal).in_scope(|| {
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
            let bases = context.upload_g1_bases(host_bases)?;
            Ok((
                window_columns::<F>(context, &bases, &hot, plan, window, 0)?,
                Some(trace),
                built,
            ))
        }));
    }
    let parts = tracing::info_span!("cuda_commit_tier1_fanout", windows = windows.len())
        .in_scope(|| fan_out(tasks))?;
    let mut columns = Vec::with_capacity(parts.len());
    let mut traces = Vec::with_capacity(parts.len());
    let mut hots = Vec::with_capacity(parts.len());
    for (part, trace, built) in parts {
        columns.push(part);
        traces.push(trace);
        hots.push(built);
    }
    let stitched = tracing::info_span!("cuda_commit_stitch", windows = windows.len())
        .in_scope(|| stitch::<F>(columns, windows, hot, plan))?;
    Ok((stitched, traces, hots))
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

    use super::{stitch, window_columns, ColumnPlan, CycleWindow, JacobianLimbs};
    use crate::cuda::commitment::{affine_limbs, fq_from_limbs};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::pack::COLD;
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
    fn split_cycle_windows_match_the_whole_domain_commitment() {
        let Some(context) = shared_context() else {
            return;
        };
        let host_bases: Vec<_> = (0..ROW_WIDTH)
            .map(|column| affine_limbs(point(column as u64 + 1).into_affine()))
            .collect();
        let bases = context.upload_g1_bases(&host_bases).expect("upload bases");
        let words: Vec<u32> = (0..CYCLES)
            .map(|cycle| {
                if cycle.is_multiple_of(7) {
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
        ];
        let increments = vec![
            Vec::new(),
            (0..CYCLES).map(|cycle| cycle as i128 - 17).collect(),
        ];
        let plan = ColumnPlan {
            kinds: &[],
            increments: &increments,
            cycles: CYCLES,
            one_hot_k: ONE_HOT_K,
            row_width: ROW_WIDTH,
        };

        let expected = as_points(
            &window_columns::<Fr>(
                context,
                &bases,
                &hot,
                &plan,
                &CycleWindow {
                    start: 0,
                    len: CYCLES,
                },
                0,
            )
            .expect("whole-domain columns"),
        );

        for devices in [2usize, 4, 8] {
            let windows: Vec<CycleWindow> = (0..devices)
                .map(|device| CycleWindow {
                    start: device * CYCLES / devices,
                    len: CYCLES / devices,
                })
                .collect();
            let parts: Vec<_> = windows
                .iter()
                .map(|window| {
                    window_columns::<Fr>(context, &bases, &hot, &plan, window, window.start)
                        .expect("window columns")
                })
                .collect();
            let got = as_points(&stitch::<Fr>(parts, &windows, &hot, &plan).expect("stitch"));
            assert_eq!(
                got, expected,
                "splitting the cycle domain across {devices} windows changed the tier-1 row \
                 commitments; a row commitment depends only on the cycles inside its own \
                 row_width-aligned block, so the partition must be exact",
            );
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
