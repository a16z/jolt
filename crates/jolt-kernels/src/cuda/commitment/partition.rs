//! CUDA commitment rows assembled across cycle windows.
//!
//! Dense rows concatenate in cycle order. One-hot rows interleave each window's
//! rows by address before finalization. Optimized streams the column's rows
//! through its PCS consumer in their final order.

use jolt_field::JoltField;

use super::{tier1_order, ColumnKind, RetainedHotColumns};
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::devices::CycleWindow;
use crate::cuda::common::msm::{DeviceG1Bases, JacobianLimbs, SignedColumn};
use crate::KernelError;

pub(super) struct ColumnPlan<'a> {
    pub(super) kinds: &'a [ColumnKind],
    pub(super) one_hot_k: usize,
    pub(super) row_width: usize,
}

type ColumnSink<'a, F> = &'a mut dyn FnMut(usize, Vec<JacobianLimbs>) -> Result<(), KernelError<F>>;

pub(super) fn window_columns<F: JoltField>(
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

fn stitch_column<F: JoltField>(
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

    pub(super) fn place<F: JoltField>(
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

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use ark_bn254::Fr as ArkFr;
    use ark_bn254::G1Projective;
    use ark_ec::{CurveGroup, PrimeGroup};
    use jolt_field::Fr;
    use std::ops::Range;

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
        G1Projective::generator() * ArkFr::from(scalar)
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
        let dense = |range: Range<usize>| {
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
}
