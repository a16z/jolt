use std::sync::Arc;

use super::address_driver::AddressShard;
use super::address_phase::DeviceRows;
use crate::cuda::common::context::context_for;
use crate::cuda::common::device::LIMBS;
use crate::cuda::common::devices::plan_device_windows;
use crate::cuda::common::error::CudaError;
use crate::cuda::common::xfer_stats::{self, Phase};

impl AddressShard {
    /// Address reductions depend on row/weight pairs, not their cycle positions.
    /// Redistribute the retained pairs once, then reuse them across all phases.
    /// The caller keeps the original rows for the cycle-domain handoff.
    #[tracing::instrument(skip_all, name = "ap_balance_rows")]
    pub(super) fn balance(shards: Vec<Self>) -> Result<Vec<Self>, CudaError> {
        let cycles = shards.iter().map(|shard| shard.rows.cycles()).sum();
        let windows = plan_device_windows(cycles, 1, shards.len());
        if windows.len() != shards.len()
            || windows
                .iter()
                .zip(&shards)
                .all(|(window, shard)| window.len == shard.rows.cycles())
        {
            return Ok(shards);
        }

        let mut balanced = Vec::with_capacity(shards.len());
        for (window, target) in windows.iter().zip(&shards) {
            let context = context_for(target.ordinal).ok_or(CudaError::InvariantViolation {
                reason: "an address-phase window names an absent device",
            })?;
            let mut lookup = context.alloc_u64_unset(2 * window.len)?;
            let mut tables = context.alloc_u32_unset(window.len)?;
            let mut flags = context.alloc_u8_unset(window.len)?;
            let mut weights = context.alloc_fr_unset(window.len)?;
            let mut start = 0;
            for source in &shards {
                let end = start + source.rows.cycles();
                let lo = start.max(window.start);
                let hi = end.min(window.end());
                if lo < hi {
                    let from = lo - start;
                    let to = lo - window.start;
                    let len = hi - lo;
                    let copied =
                        xfer_stats::timed(Phase::D2d, len * (2 * 8 + 4 + 1 + LIMBS * 8), || {
                            context.stream().memcpy_dtod(
                                &source.rows.lookup_index().slice(2 * from..2 * (from + len)),
                                &mut lookup.slice_mut(2 * to..2 * (to + len)),
                            )?;
                            context.stream().memcpy_dtod(
                                &source.rows.table_index().slice(from..from + len),
                                &mut tables.slice_mut(to..to + len),
                            )?;
                            context.stream().memcpy_dtod(
                                &source.rows.raf_flag().slice(from..from + len),
                                &mut flags.slice_mut(to..to + len),
                            )?;
                            context.stream().memcpy_dtod(
                                &source
                                    .u_evals
                                    .limbs()
                                    .slice(LIMBS * from..LIMBS * (from + len)),
                                &mut weights
                                    .limbs_mut()
                                    .slice_mut(LIMBS * to..LIMBS * (to + len)),
                            )
                        });
                    if let Err(error) = copied {
                        let _ = context.stream().synchronize();
                        return Err(error.into());
                    }
                }
                start = end;
            }
            // cudarc records readiness events on the source streams. Finish
            // each destination before another can introduce a reverse wait;
            // the source weights also remain alive until these copies finish.
            context.stream().synchronize()?;
            balanced.push(Self {
                ordinal: target.ordinal,
                rows: Arc::new(DeviceRows::from_device_columns(
                    Arc::new(lookup),
                    tables,
                    flags,
                    window.len,
                )?),
                u_evals: weights,
            });
        }
        Ok(balanced)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test device operations must succeed")]
mod tests {
    use std::sync::Arc;

    use super::AddressShard;
    use crate::cuda::common::context::{context_for, device_count, shared_context};
    use crate::cuda::common::testing::fr;
    use crate::cuda::instruction_read_raf::address_phase::{DeviceRows, NO_TABLE};

    #[test]
    fn balancing_preserves_rows_and_their_weights() {
        if shared_context().is_none() || device_count() < 2 {
            return;
        }
        for lengths in [[33, 1], [1, 33], [19, 16]] {
            let cycles: usize = lengths.iter().sum();
            let bits: Vec<_> = (0..cycles as u64)
                .flat_map(|i| [i.wrapping_mul(0xdead_beef), !i])
                .collect();
            let tables: Vec<_> = (0..cycles)
                .map(|i| if i % 3 == 0 { NO_TABLE } else { (i % 2) as u32 })
                .collect();
            let flags: Vec<_> = (0..cycles).map(|i| (i % 2) as u8).collect();
            let weights: Vec<_> = (0..cycles as u64).map(|i| fr(i * 79 + 13)).collect();
            let mut start = 0;
            let shards = lengths
                .into_iter()
                .enumerate()
                .map(|(ordinal, len)| {
                    let context = context_for(ordinal).unwrap();
                    let end = start + len;
                    let shard = AddressShard {
                        ordinal,
                        rows: Arc::new(
                            DeviceRows::from_encoded(
                                context,
                                &bits[2 * start..2 * end],
                                &tables[start..end],
                                &flags[start..end],
                            )
                            .unwrap(),
                        ),
                        u_evals: context.upload(&weights[start..end]).unwrap(),
                    };
                    start = end;
                    shard
                })
                .collect();
            let balanced = AddressShard::balance(shards).unwrap();
            assert_eq!(balanced.len(), 2);
            assert!(
                balanced[0]
                    .rows
                    .cycles()
                    .abs_diff(balanced[1].rows.cycles())
                    <= 1
            );
            let mut got = Vec::new();
            for shard in balanced {
                let context = context_for(shard.ordinal).unwrap();
                let bits = context
                    .stream()
                    .clone_dtoh(shard.rows.lookup_index())
                    .unwrap();
                let tables = context
                    .stream()
                    .clone_dtoh(shard.rows.table_index())
                    .unwrap();
                let flags = context.stream().clone_dtoh(shard.rows.raf_flag()).unwrap();
                let weights = shard.u_evals.to_host().unwrap();
                got.extend((0..shard.rows.cycles()).map(|i| {
                    (
                        bits[2 * i],
                        bits[2 * i + 1],
                        tables[i],
                        flags[i],
                        weights[i],
                    )
                }));
            }
            let mut expected: Vec<_> = (0..cycles)
                .map(|i| {
                    (
                        bits[2 * i],
                        bits[2 * i + 1],
                        tables[i],
                        flags[i],
                        weights[i],
                    )
                })
                .collect();
            got.sort_by_key(|row| (row.0, row.1));
            expected.sort_by_key(|row| (row.0, row.1));
            assert_eq!(got, expected);
        }
    }
}
