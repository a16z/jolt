use jolt_field::{Fr, JoltField};
use jolt_poly::BindingOrder;

use super::device_rows::DeviceRegisterRows;
use super::PreparedShard;
use crate::cuda::common::context::context_for;
use crate::cuda::common::devices::{plan_witness_windows, CycleWindow};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::read_write_matrix::CycleShard;
use crate::cuda::common::split_eq::DeviceSplitEq;

pub(super) struct RegisterWindow {
    pub ordinal: usize,
    pub window: CycleWindow,
    pub rows: DeviceRegisterRows,
}

impl RegisterWindow {
    pub(super) fn prepare<F: JoltField>(
        self,
        index: usize,
        count: usize,
        point: &[F],
        gamma: Fr,
    ) -> Result<PreparedShard<F>, CudaError> {
        let context = context_for(self.ordinal).ok_or(CudaError::InvariantViolation {
            reason: "a register matrix window names an absent device",
        })?;
        let matrix = self.rows.matrix(context, gamma)?;
        let inc = self.rows.inc(context)?;
        let eq = DeviceSplitEq::new_window(context, point, BindingOrder::LowToHigh, index, count)?;
        Ok((
            CycleShard {
                ordinal: self.ordinal,
                matrix,
                inc,
                eq,
            },
            (self.ordinal, self.rows.into_read_addresses(), self.window),
        ))
    }

    /// Distribute aligned half-windows before expanding their sparse matrices.
    /// Window order, rather than device ordinal, determines equality weights.
    #[tracing::instrument(skip_all, name = "registers_balance_windows")]
    pub(super) fn balance(shards: Vec<Self>, prefix: usize) -> Result<Vec<Self>, CudaError> {
        let count = shards.len();
        if count < 2 {
            return Ok(shards);
        }
        let cycles = shards.iter().map(|shard| shard.window.len).sum();
        let windows = plan_witness_windows(cycles, 2 * count);
        if windows.len() != 2 * count {
            return Ok(shards);
        }
        let retained = |window: &CycleWindow| {
            prefix
                .saturating_sub(window.start)
                .saturating_add(1)
                .min(window.len)
        };
        let old_max = shards
            .iter()
            .map(|shard| retained(&shard.window))
            .max()
            .unwrap_or(0);
        let mut loads = vec![0; count];
        for (index, window) in windows.iter().enumerate() {
            loads[index % count] += retained(window);
        }
        // Full-column transfers and extra dispatches can offset smaller savings.
        // Require at least a quarter less estimated row work on the busiest GPU.
        if loads.into_iter().max().unwrap_or(0) > old_max - old_max / 4 {
            return Ok(shards);
        }

        let mut balanced = Vec::with_capacity(windows.len());
        for (index, window) in windows.into_iter().enumerate() {
            let source = &shards[index / 2];
            let offset = window.start.checked_sub(source.window.start).ok_or(
                CudaError::InvariantViolation {
                    reason: "a register subdivision starts before its resident window",
                },
            )?;
            if window.end() > source.window.end() {
                return Err(CudaError::InvariantViolation {
                    reason: "a register subdivision exceeds its resident window",
                });
            }
            let ordinal = shards[index % count].ordinal;
            let context = context_for(ordinal).ok_or(CudaError::InvariantViolation {
                reason: "a register subdivision names an absent device",
            })?;
            let rows = source.rows.copy_window(context, offset, window.len)?;
            balanced.push(Self {
                ordinal,
                window,
                rows,
            });
        }
        Ok(balanced)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test device operations must succeed")]
mod tests {
    use super::RegisterWindow;
    use crate::cuda::common::context::{context_for, device_count, shared_context};
    use crate::cuda::common::devices::CycleWindow;
    use crate::cuda::registers_read_write::device_rows::DeviceRegisterRows;
    use crate::cuda::registers_read_write::witness::RegistersReadWriteWitness;

    #[test]
    fn balanced_windows_preserve_register_columns() {
        if shared_context().is_none() || device_count() < 2 {
            return;
        }
        const WIDTH: usize = 1 << 14;
        let row = |j: usize| {
            let mut row = RegistersReadWriteWitness::default();
            row.rs1_address.0 = (!j.is_multiple_of(3)).then_some((j % 128) as u8);
            row.rs2_address.0 = (!j.is_multiple_of(5)).then_some(((j + 1) % 128) as u8);
            row.rd_address.0 = (!j.is_multiple_of(7)).then_some(((j + 2) % 128) as u8);
            row.rs1_value.0 = j as u64 * 7;
            row.rs2_value.0 = !(j as u64 * 11);
            row.rd_pre_value.0 = j as u64 * 13;
            row.rd_post_value.0 = !(j as u64 * 17);
            row
        };
        for (prefix, count) in [
            (12_345, 4),
            (WIDTH, 4),
            (WIDTH + WIDTH / 3, 2),
            (2 * WIDTH, 2),
        ] {
            let source: Vec<_> = (0..2 * WIDTH).map(|j| row(j.min(prefix))).collect();
            let windows = (0..2)
                .map(|ordinal| RegisterWindow {
                    ordinal,
                    window: CycleWindow {
                        start: ordinal * WIDTH,
                        len: WIDTH,
                    },
                    rows: DeviceRegisterRows::upload(
                        context_for(ordinal).unwrap(),
                        &source[ordinal * WIDTH..(ordinal + 1) * WIDTH],
                    )
                    .unwrap(),
                })
                .collect();
            let windows = RegisterWindow::balance(windows, prefix).unwrap();
            assert_eq!(windows.len(), count);
            let mut start = 0;
            for window in windows {
                assert_eq!(window.window.start, start);
                let context = context_for(window.ordinal).unwrap();
                let original = &source[start..window.window.end()];
                let columns = &window.rows;
                for (got, expected) in [
                    (
                        columns.rs1_address(),
                        original.iter().map(|r| r.rs1_address.0).collect::<Vec<_>>(),
                    ),
                    (
                        columns.rs2_address(),
                        original.iter().map(|r| r.rs2_address.0).collect(),
                    ),
                    (
                        columns.rd_address(),
                        original.iter().map(|r| r.rd_address.0).collect(),
                    ),
                ] {
                    let expected: Vec<_> = expected
                        .into_iter()
                        .map(|a| a.map_or(u32::MAX, u32::from))
                        .collect();
                    assert_eq!(context.download_u32(got).unwrap(), expected);
                }
                for (got, expected) in [
                    (
                        columns.rs1_value(),
                        original.iter().map(|r| r.rs1_value.0).collect::<Vec<_>>(),
                    ),
                    (
                        columns.rs2_value(),
                        original.iter().map(|r| r.rs2_value.0).collect(),
                    ),
                    (
                        columns.rd_pre_value(),
                        original.iter().map(|r| r.rd_pre_value.0).collect(),
                    ),
                    (
                        columns.rd_post_value(),
                        original.iter().map(|r| r.rd_post_value.0).collect(),
                    ),
                ] {
                    assert_eq!(context.download_u64(got).unwrap(), expected);
                }
                start = window.window.end();
            }
            assert_eq!(start, 2 * WIDTH);
        }
    }
}
