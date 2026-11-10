use std::{
    fmt,
    sync::{Arc, Mutex, Weak},
};

use akita_error::{checked, AkitaError};
use akita_metal::onehot::{DeviceK16Sources, OneHotCommitShape};
use akita_metal::onehot_fold::{OneHotFoldChallenge, OneHotFoldShape, OneHotFoldWorkspace};
use akita_metal::onehot_ntt::{DeviceOneHotNttKey, OneHotNttWorkspace};
use akita_metal::onehot_packing::{OneHotPackingShape, OneHotPackingWorkspace};
use akita_metal::{AkitaMetal, AkitaMetalError};
use akita_pcs::custom_source::CommitInnerPlan;
use akita_types::{AkitaExpandedSetup, RingVec};
use jolt_akita::{
    AkitaField, TraceOneHotComputeKernel, TraceOneHotFoldPlan, TraceOneHotPackingPlan,
    TraceOneHotRows,
};
use jolt_field::Zero;

const D: usize = 512;
const K: usize = 16;
// Target logical window size. Source initialization uses bounded worker scratch.
const STAGING_BYTES: usize = 1 << 28;
// At most 1/16 of the device's recommended working set, capped at 8 GiB.
// This limits proof-lived selectors separately from transient packing buffers.
const MAX_RETAINED_SOURCE_BYTES: u64 = 8 << 30;

/// Trace arithmetic attached to an exact commitment setup owner. The Akita
/// CPU backend retains the source and opening handles.
pub struct MetalTraceCompute {
    shared: Arc<MetalTraceSetup>,
    source: Option<ProofTraceSources>,
}

struct MetalTraceSetup {
    expanded: Arc<AkitaExpandedSetup<AkitaField>>,
    metal: AkitaMetal,
    key: Mutex<Option<(usize, DeviceOneHotNttKey)>>,
}

struct ProofTraceSources {
    owner: Weak<dyn TraceOneHotRows>,
    retained_bytes: Option<usize>,
    windows: Mutex<Option<RetainedWindows>>,
}

struct RetainedWindows {
    rows_per_block: usize,
    windows: Vec<(usize, DeviceK16Sources)>,
}

impl ProofTraceSources {
    fn new(rows: &Arc<dyn TraceOneHotRows>, limit: usize) -> Result<Self, AkitaError> {
        let prefix = rows.nonzero_prefix_len();
        let columns = rows.num_columns();
        let bytes = checked::product([prefix, columns])
            .and_then(|hot| {
                checked::product([prefix.div_ceil(32), columns])
                    .and_then(|kinds| checked::sum([hot, kinds]))
            })
            .ok_or_else(|| compute_error("retained selector size overflows"))?;
        let retained_bytes = (bytes <= limit).then_some(bytes);
        tracing::info!(
            bytes,
            limit,
            retain = retained_bytes.is_some(),
            "Metal trace source retention"
        );
        Ok(Self {
            owner: Arc::downgrade(rows),
            retained_bytes,
            windows: Mutex::new(None),
        })
    }

    fn check_owner(&self, rows: &dyn TraceOneHotRows) -> Result<(), AkitaError> {
        let owner = self
            .owner
            .upgrade()
            .ok_or_else(|| compute_error("trace source owner expired"))?;
        if !std::ptr::eq(owner.as_ref(), rows) {
            return Err(compute_error(
                "trace compute kernel belongs to another source",
            ));
        }
        Ok(())
    }
}

impl fmt::Debug for MetalTraceCompute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalTraceCompute").finish_non_exhaustive()
    }
}

impl MetalTraceCompute {
    pub fn new(expanded: Arc<AkitaExpandedSetup<AkitaField>>) -> Result<Self, AkitaError> {
        Ok(Self {
            shared: Arc::new(MetalTraceSetup {
                expanded,
                metal: AkitaMetal::new().map_err(compute_error)?,
                key: Mutex::new(None),
            }),
            source: None,
        })
    }

    fn pack_rows(
        &self,
        rows: &dyn TraceOneHotRows,
        first_row: usize,
        window_rows: usize,
    ) -> Result<DeviceK16Sources, AkitaError> {
        let columns = rows.num_columns();
        let stored_rows = rows
            .nonzero_prefix_len()
            .saturating_sub(first_row)
            .min(window_rows);
        DeviceK16Sources::from_row_major_producer(
            &self.shared.metal,
            columns,
            stored_rows,
            window_rows,
            |offset, selected, masks| {
                let row = checked::sum([first_row, offset])
                    .ok_or_else(|| AkitaMetalError::Shape("trace row offset overflows".into()))?;
                rows.fill_rows_with_masks(row, selected, masks);
                Ok(())
            },
        )
        .map_err(compute_error)
    }
}

impl TraceOneHotComputeKernel for MetalTraceCompute {
    fn bind_source(
        self: Arc<Self>,
        rows: Arc<dyn TraceOneHotRows>,
    ) -> Result<Arc<dyn TraceOneHotComputeKernel>, AkitaError> {
        let device_limit = self
            .shared
            .metal
            .device()
            .limits()
            .recommended_max_working_set_size
            / 16;
        let limit =
            usize::try_from(MAX_RETAINED_SOURCE_BYTES.min(device_limit)).unwrap_or(usize::MAX);
        Ok(Arc::new(Self {
            source: Some(ProofTraceSources::new(&rows, limit)?),
            shared: Arc::clone(&self.shared),
        }))
    }

    fn coefficient_packing(
        &self,
        rows: &dyn TraceOneHotRows,
        one_hot_k: usize,
        plan: TraceOneHotPackingPlan<'_>,
    ) -> Result<Option<Vec<Vec<AkitaField>>>, AkitaError> {
        let Some(source) = &self.source else {
            return Ok(None);
        };
        source.check_owner(rows)?;
        let shape = OneHotPackingShape {
            num_positions: plan.num_positions(),
            blocks_per_column: plan.blocks_per_column(),
            num_columns: rows.num_columns(),
            subring_dimension: plan.subring_dimension(),
        };
        if one_hot_k != K
            || plan.ring_dimension() != D
            || !shape.is_supported()
            || plan.segment_rings() < plan.num_positions()
            || !plan.segment_rings().is_multiple_of(plan.num_positions())
        {
            return Ok(None);
        }
        let rows_per_block = checked::product([plan.num_positions(), D / K])
            .ok_or_else(|| compute_error("packing block size overflows"))?;
        let cache = source
            .windows
            .lock()
            .map_err(|_| compute_error("source cache lock poisoned"))?;
        let Some(retained) = cache
            .as_ref()
            .filter(|cache| cache.rows_per_block == rows_per_block)
        else {
            return Ok(None);
        };
        let _span = tracing::info_span!(
            "metal_trace_coefficient_packing",
            resident_windows = retained.windows.len()
        )
        .entered();
        let mut workspace = OneHotPackingWorkspace::new(
            &self.shared.metal,
            shape,
            plan.position_weights(),
            plan.packing_weights(),
        )
        .map_err(compute_error)?;
        // Borrow, rather than consume, the exact proof-owned selectors. The
        // root-fold challenge arrives later and still needs these same windows.
        for (first_block, sources) in &retained.windows {
            let dispatch = tracing::info_span!(
                "metal_trace_packing_dispatch",
                gpu_s = tracing::field::Empty
            );
            let _dispatch = dispatch.enter();
            let gpu_time = workspace
                .accumulate(sources, *first_block)
                .map_err(compute_error)?;
            let _ = dispatch.record("gpu_s", gpu_time.as_secs_f64());
        }
        let per_column = checked::product([plan.blocks_per_column(), plan.subring_dimension()])
            .ok_or_else(|| compute_error("packing output column width overflows"))?;
        let output = workspace.read().map_err(compute_error)?;
        Ok(Some(
            output
                .chunks_exact(per_column)
                .map(<[AkitaField]>::to_vec)
                .collect(),
        ))
    }

    fn decompose_fold(
        &self,
        rows: &dyn TraceOneHotRows,
        one_hot_k: usize,
        plan: TraceOneHotFoldPlan<'_>,
    ) -> Result<Option<Vec<i32>>, AkitaError> {
        if let Some(source) = &self.source {
            source.check_owner(rows)?;
        }
        if one_hot_k != K
            || plan.ring_dimension() != D
            || plan.segment_rings() < plan.num_positions()
            || !plan.segment_rings().is_multiple_of(plan.num_positions())
        {
            return Ok(None);
        }
        let columns = rows.num_columns();
        let prefix = rows.nonzero_prefix_len();
        if !(1..=64).contains(&columns) || prefix > rows.num_rows() {
            return Err(compute_error("invalid trace row extent"));
        }
        if plan.challenges().len() >= 1 << 26 {
            return Ok(None);
        }
        // A fixed output coefficient receives each challenge term at most
        // once: the one-hot source exponents within a ring are distinct.
        for range in plan.chunk_ranges() {
            let mut bound = 0u64;
            for column in plan.challenges().chunks_exact(plan.blocks_per_column()) {
                let challenges = column
                    .get(range.clone())
                    .ok_or_else(|| compute_error("validated fold range exceeds column"))?;
                for challenge in challenges {
                    bound += challenge
                        .coeffs
                        .iter()
                        .map(|coefficient| u64::from(coefficient.unsigned_abs()))
                        .sum::<u64>();
                }
            }
            if bound > i32::MAX as u64 {
                return Ok(None);
            }
        }
        let _span = tracing::info_span!("metal_trace_decompose_fold").entered();
        let challenges: Vec<_> = plan
            .challenges()
            .iter()
            .map(|challenge| OneHotFoldChallenge {
                positions: &challenge.positions,
                coefficients: &challenge.coeffs,
            })
            .collect();
        let shape = OneHotFoldShape {
            num_positions: plan.num_positions(),
            blocks_per_column: plan.blocks_per_column(),
            num_columns: columns,
        };
        let rows_per_block = checked::product([plan.num_positions(), D / K])
            .ok_or_else(|| compute_error("fold block size overflows"))?;
        let retained = self
            .source
            .as_ref()
            .map(|source| {
                source
                    .windows
                    .lock()
                    .map(|mut cache| cache.take())
                    .map_err(|_| compute_error("source cache lock poisoned"))
            })
            .transpose()?
            .flatten()
            .filter(|cache| cache.rows_per_block == rows_per_block);
        let mut workspace =
            OneHotFoldWorkspace::new(&self.shared.metal, shape, &challenges, plan.chunk_ranges())
                .map_err(compute_error)?;
        let mut accumulate = |sources: &DeviceK16Sources, first_block| {
            let dispatch =
                tracing::info_span!("metal_trace_fold_dispatch", gpu_s = tracing::field::Empty);
            let _dispatch = dispatch.enter();
            let gpu_time = workspace
                .accumulate(sources, first_block)
                .map_err(compute_error)?;
            let _ = dispatch.record("gpu_s", gpu_time.as_secs_f64());
            Ok::<_, AkitaError>(())
        };
        if let Some(retained) = retained {
            tracing::info!(
                resident_windows = retained.windows.len(),
                "Metal trace fold reuse"
            );
            // Each window drops after its completed dispatch. The setup never
            // owns these buffers; an unopened source also releases them on drop.
            for (first_block, sources) in retained.windows {
                accumulate(&sources, first_block)?;
            }
        } else {
            let blocks_per_window = staging_blocks(rows_per_block, columns)?;
            let active_blocks = prefix.div_ceil(rows_per_block);
            for first_block in (0..active_blocks).step_by(blocks_per_window) {
                let blocks = blocks_per_window.min(active_blocks - first_block);
                let first_row = first_block * rows_per_block;
                let window_rows = (blocks * rows_per_block).min(rows.num_rows() - first_row);
                let pack = tracing::info_span!("metal_trace_fold_pack").entered();
                let sources = self.pack_rows(rows, first_row, window_rows)?;
                drop(pack);
                accumulate(&sources, first_block)?;
            }
        }
        workspace.read().map(Some).map_err(compute_error)
    }

    fn commit(
        &self,
        expanded: &AkitaExpandedSetup<AkitaField>,
        rows: &dyn TraceOneHotRows,
        one_hot_k: usize,
        plan: CommitInnerPlan,
    ) -> Result<Option<Vec<RingVec<AkitaField>>>, AkitaError> {
        if let Some(source) = &self.source {
            source.check_owner(rows)?;
        }
        // Retained sources may be imported by another backend. Only this
        // owner may use the cached key; all other owners use their CPU path.
        if !std::ptr::eq(expanded, self.shared.expanded.as_ref())
            || one_hot_k != K
            || plan.ring_dimension != D
        {
            return Ok(None);
        }
        let _span = tracing::info_span!(
            "metal_trace_commitment",
            columns = rows.num_columns(),
            positions = plan.num_positions_per_block,
            blocks = plan.num_live_blocks,
            n_a = plan.n_a,
            digits = plan.num_digits_inner,
        )
        .entered();
        let columns = rows.num_columns();
        let logical_rows = rows.num_rows();
        let prefix = rows.nonzero_prefix_len();
        if !(1..=64).contains(&columns)
            || prefix > logical_rows
            || !logical_rows.is_multiple_of(D / K)
            || !plan.num_positions_per_block.is_power_of_two()
            || plan.n_a == 0
            || plan.num_digits_inner == 0
        {
            return Err(AkitaError::InvalidInput(
                "invalid Metal trace commitment geometry".into(),
            ));
        }
        let rows_per_block = checked::product([plan.num_positions_per_block, D / K])
            .ok_or_else(|| AkitaError::InvalidInput("trace block size overflows".into()))?;
        if logical_rows.div_ceil(rows_per_block) != plan.num_live_blocks {
            return Err(AkitaError::InvalidInput(
                "trace commitment live-block extent disagrees with its columns".into(),
            ));
        }
        let active_a_cols = checked::product([plan.num_positions_per_block, plan.num_digits_inner])
            .ok_or_else(|| AkitaError::InvalidSetup("active A width overflows".into()))?;
        let key_len = checked::product([plan.n_a, active_a_cols, D])
            .ok_or_else(|| AkitaError::InvalidSetup("active A prefix overflows".into()))?;
        let key_coefficients = expanded
            .shared_matrix()
            .as_field_slice()
            .get(..key_len)
            .ok_or_else(|| AkitaError::InvalidSetup("active A prefix exceeds setup".into()))?;
        let shape = OneHotCommitShape {
            n_a: plan.n_a,
            active_a_cols,
            num_digits_inner: plan.num_digits_inner,
        };
        let block_coefficients = checked::product([plan.n_a, D])
            .ok_or_else(|| AkitaError::InvalidInput("commitment block size overflows".into()))?;
        let output_len = checked::product([plan.num_live_blocks, block_coefficients])
            .ok_or_else(|| AkitaError::InvalidInput("commitment output size overflows".into()))?;
        let mut output = vec![vec![AkitaField::zero(); output_len]; columns];
        let mut retained_cache = self
            .source
            .as_ref()
            .map(|source| {
                source
                    .windows
                    .lock()
                    .map_err(|_| compute_error("source cache lock poisoned"))
            })
            .transpose()?;
        if let Some(cache) = &mut retained_cache {
            **cache = None;
        }
        let mut retained = self
            .source
            .as_ref()
            .and_then(|source| source.retained_bytes)
            .map(|_| RetainedWindows {
                rows_per_block,
                windows: Vec::new(),
            });
        let mut cached = self
            .shared
            .key
            .lock()
            .map_err(|_| compute_error("key cache lock poisoned"))?;
        if cached.as_ref().is_none_or(|(len, _)| *len != key_len) {
            let _prepare = tracing::info_span!("metal_trace_commit_prepare_key").entered();
            *cached = Some((
                key_len,
                DeviceOneHotNttKey::new(&self.shared.metal, key_coefficients)
                    .map_err(compute_error)?,
            ));
        }
        let (_, key) = cached
            .as_ref()
            .ok_or_else(|| compute_error("missing prepared key"))?;
        let blocks_per_window = staging_blocks(rows_per_block, columns)?;
        let active_blocks = prefix.div_ceil(rows_per_block);
        let mut workspace = OneHotNttWorkspace::new(&self.shared.metal);
        for first_block in (0..active_blocks).step_by(blocks_per_window) {
            let blocks = blocks_per_window.min(active_blocks - first_block);
            let first_row = first_block * rows_per_block;
            let window_rows = (blocks * rows_per_block).min(logical_rows - first_row);
            let pack_span = tracing::info_span!("metal_trace_commit_pack").entered();
            let sources = self.pack_rows(rows, first_row, window_rows)?;
            drop(pack_span);
            let dispatch =
                tracing::info_span!("metal_trace_commit_dispatch", gpu_s = tracing::field::Empty);
            let _dispatch = dispatch.enter();
            let (result, gpu_time) = workspace
                .commit(key, &sources, shape)
                .map_err(compute_error)?;
            let _ = dispatch.record("gpu_s", gpu_time.as_secs_f64());
            let result = result.read().map_err(compute_error)?;
            let per_column = blocks * block_coefficients;
            if result.len() != per_column * columns {
                return Err(compute_error(
                    "commitment output length disagrees with plan",
                ));
            }
            let offset = first_block * block_coefficients;
            for (column, coefficients) in output.iter_mut().zip(result.chunks_exact(per_column)) {
                let destination = column
                    .get_mut(offset..offset + per_column)
                    .ok_or_else(|| compute_error("commitment output window exceeds plan"))?;
                destination.copy_from_slice(coefficients);
            }
            if let Some(retained) = &mut retained {
                retained.windows.push((first_block, sources));
            }
        }
        let output = output
            .into_iter()
            .map(|coefficients| RingVec::from_coeffs_with_ring_dim(coefficients, D))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(cache) = &mut retained_cache {
            **cache = retained;
        }
        Ok(Some(output))
    }
}

fn staging_blocks(rows_per_block: usize, columns: usize) -> Result<usize, AkitaError> {
    let bytes_per_block = checked::product([rows_per_block, columns + size_of::<u64>()])
        .ok_or_else(|| compute_error("trace staging size overflows"))?;
    Ok((STAGING_BYTES / bytes_per_block).max(1))
}

fn compute_error(error: impl fmt::Display) -> AkitaError {
    AkitaError::InvalidInput(format!("Metal trace arithmetic: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rows;

    impl TraceOneHotRows for Rows {
        fn num_rows(&self) -> usize {
            64
        }
        fn num_columns(&self) -> usize {
            1
        }
        fn nonzero_prefix_len(&self) -> usize {
            33
        }
        fn fill_row(&self, _row: usize, selected: &mut [u8]) {
            selected.fill(0);
        }
    }

    #[test]
    fn residency_checks_exact_source_and_bounds_retention() -> Result<(), AkitaError> {
        let rows: Arc<dyn TraceOneHotRows> = Arc::new(Rows);
        let other: Arc<dyn TraceOneHotRows> = Arc::new(Rows);
        // 33 selectors and two ring kinds; the implicit suffix costs no bytes.
        let retained = ProofTraceSources::new(&rows, 35)?;
        assert_eq!(retained.retained_bytes, Some(35));
        assert!(retained.check_owner(rows.as_ref()).is_ok());
        assert!(retained.check_owner(other.as_ref()).is_err());
        assert!(ProofTraceSources::new(&rows, 34)?.retained_bytes.is_none());
        drop(rows);
        assert!(retained.owner.upgrade().is_none());
        Ok(())
    }
}
