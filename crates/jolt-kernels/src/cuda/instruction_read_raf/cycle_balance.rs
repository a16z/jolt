use std::sync::Arc;

use super::address_phase::DeviceRows;
use super::RowShard;
use crate::cuda::common::context::context_for;
use crate::cuda::common::devices::plan_witness_windows;
use crate::cuda::common::error::CudaError;
use crate::cuda::common::xfer_stats::{self, Phase};

impl RowShard {
    /// Split each logical window in half and distribute the halves cyclically.
    /// Their order remains the cycle order used by the equality polynomial.
    /// Copy compact lookup rows before materializing the cycle's field factors.
    #[tracing::instrument(skip_all, name = "ap_balance_cycle_rows")]
    pub(super) fn balance_cycles(shards: &[Self]) -> Result<Option<Vec<Self>>, CudaError> {
        if shards.len() < 2 {
            return Ok(None);
        }
        let cycles = shards.iter().map(|shard| shard.window.len).sum();
        let windows = plan_witness_windows(cycles, 2 * shards.len());
        if windows.len() != 2 * shards.len() {
            return Ok(None);
        }

        let mut retained = Vec::with_capacity(windows.len());
        let mut loads = vec![0; shards.len()];
        for (index, window) in windows.iter().enumerate() {
            let source = &shards[index / 2];
            let offset = window.start.checked_sub(source.window.start).ok_or(
                CudaError::InvariantViolation {
                    reason: "a cycle subdivision starts before its resident window",
                },
            )?;
            if window.end() > source.window.end() || source.rows.cycles() == 0 {
                return Err(CudaError::InvariantViolation {
                    reason: "a cycle subdivision needs a nonempty covering resident window",
                });
            }
            let len = source
                .rows
                .cycles()
                .saturating_sub(offset)
                .max(1)
                .min(window.len);
            retained.push(len);
            loads[index % shards.len()] += len;
        }
        let old_max = shards.iter().map(|shard| shard.rows.cycles()).max();
        if loads.into_iter().max() >= old_max {
            return Ok(None);
        }

        let mut balanced = Vec::with_capacity(windows.len());
        for (index, (window, len)) in windows.into_iter().zip(retained).enumerate() {
            let source = &shards[index / 2];
            let ordinal = shards[index % shards.len()].ordinal;
            let context = context_for(ordinal).ok_or(CudaError::InvariantViolation {
                reason: "a cycle subdivision names an absent device",
            })?;
            let from = (window.start - source.window.start).min(source.rows.cycles() - 1);
            let mut lookup = context.alloc_u64_unset(2 * len)?;
            let mut tables = context.alloc_u32_unset(len)?;
            let mut flags = context.alloc_u8_unset(len)?;
            let copied = xfer_stats::timed(Phase::D2d, len * (2 * 8 + 4 + 1), || {
                context.stream().memcpy_dtod(
                    &source.rows.lookup_index().slice(2 * from..2 * (from + len)),
                    &mut lookup,
                )?;
                context.stream().memcpy_dtod(
                    &source.rows.table_index().slice(from..from + len),
                    &mut tables,
                )?;
                context
                    .stream()
                    .memcpy_dtod(&source.rows.raf_flag().slice(from..from + len), &mut flags)
            });
            if let Err(error) = copied {
                let _ = context.stream().synchronize();
                return Err(error.into());
            }
            // Finish this destination before a reverse cross-device copy can
            // wait on its stream tail through cudarc's source readiness event.
            context.stream().synchronize()?;
            balanced.push(Self {
                ordinal,
                window,
                rows: Arc::new(DeviceRows::from_device_columns(
                    Arc::new(lookup),
                    tables,
                    flags,
                    len,
                )?),
            });
        }
        Ok(Some(balanced))
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test device operations must succeed")]
mod tests {
    use super::RowShard;
    use crate::cuda::common::context::{context_for, device_count, shared_context};
    use crate::cuda::common::devices::CycleWindow;
    use crate::cuda::instruction_read_raf::address_phase::{DeviceRows, NO_TABLE};
    use std::sync::Arc;

    #[test]
    fn subdivisions_preserve_logical_rows_and_constant_tails() {
        if shared_context().is_none() || device_count() < 2 {
            return;
        }
        const WIDTH: usize = 1 << 14;
        for prefix in [12_345, WIDTH, 17_001] {
            let key = |j: usize| {
                let j = j.min(prefix);
                (
                    (j as u128 * 7919) | ((!(j as u64) as u128) << 64),
                    if j.is_multiple_of(3) {
                        None
                    } else {
                        Some(j % 2)
                    },
                    j.is_multiple_of(2),
                )
            };
            let shards: Vec<_> = (0..2)
                .map(|ordinal| {
                    let start = ordinal * WIDTH;
                    let retained = (prefix.saturating_sub(start) + 1).min(WIDTH);
                    let keys: Vec<_> = (start..start + retained).map(key).collect();
                    RowShard {
                        ordinal,
                        window: CycleWindow { start, len: WIDTH },
                        rows: Arc::new(
                            DeviceRows::new(
                                context_for(ordinal).unwrap(),
                                &keys.iter().map(|k| k.0).collect::<Vec<_>>(),
                                &keys.iter().map(|k| k.1).collect::<Vec<_>>(),
                                &keys.iter().map(|k| k.2).collect::<Vec<_>>(),
                            )
                            .unwrap(),
                        ),
                    }
                })
                .collect();
            let balanced = RowShard::balance_cycles(&shards).unwrap().unwrap();
            let mut start = 0;
            for shard in balanced {
                assert_eq!(shard.window.start, start);
                let stream = context_for(shard.ordinal).unwrap().stream();
                let bits = stream.clone_dtoh(shard.rows.lookup_index()).unwrap();
                let tables = stream.clone_dtoh(shard.rows.table_index()).unwrap();
                let flags = stream.clone_dtoh(shard.rows.raf_flag()).unwrap();
                for offset in 0..shard.window.len {
                    let j = offset.min(shard.rows.cycles() - 1);
                    let expected = key(start + offset);
                    assert_eq!(
                        (bits[2 * j] as u128) | ((bits[2 * j + 1] as u128) << 64),
                        expected.0
                    );
                    assert_eq!(tables[j], expected.1.map_or(NO_TABLE, |t| t as u32));
                    assert_eq!(flags[j], u8::from(expected.2));
                }
                start = shard.window.end();
            }
            assert_eq!(start, 2 * WIDTH);
        }
    }
}
