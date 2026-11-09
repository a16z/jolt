use std::{
    fmt,
    sync::{Arc, Mutex},
};

use akita_error::{checked, AkitaError};
use akita_metal::onehot::{DeviceK16Sources, OneHotCommitShape};
use akita_metal::onehot_ntt::{DeviceOneHotNttKey, OneHotNttWorkspace};
use akita_metal::AkitaMetal;
use akita_pcs::custom_source::CommitInnerPlan;
use akita_types::{AkitaExpandedSetup, RingVec};
use jolt_akita::{AkitaField, TraceOneHotCommitKernel, TraceOneHotRows};
use jolt_field::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

const D: usize = 512;
const K: usize = 16;
// Target host packing size; upload briefly overlaps another selector buffer.
const STAGING_BYTES: usize = 1 << 28;
#[cfg(feature = "parallel")]
const PACK_ROWS: usize = 1 << 13;

/// Commitment arithmetic for an exact setup owner. The existing Akita CPU
/// backend retains the source and handles used by subsequent opening stages.
pub struct MetalTraceCommitment {
    expanded: Arc<AkitaExpandedSetup<AkitaField>>,
    metal: AkitaMetal,
    key: Mutex<Option<(usize, DeviceOneHotNttKey)>>,
}

impl fmt::Debug for MetalTraceCommitment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalTraceCommitment")
            .finish_non_exhaustive()
    }
}

impl MetalTraceCommitment {
    pub fn new(expanded: Arc<AkitaExpandedSetup<AkitaField>>) -> Result<Self, AkitaError> {
        Ok(Self {
            expanded,
            metal: AkitaMetal::new().map_err(compute_error)?,
            key: Mutex::new(None),
        })
    }
}

impl TraceOneHotCommitKernel for MetalTraceCommitment {
    fn commit(
        &self,
        expanded: &AkitaExpandedSetup<AkitaField>,
        rows: &dyn TraceOneHotRows,
        one_hot_k: usize,
        plan: CommitInnerPlan,
    ) -> Result<Option<Vec<RingVec<AkitaField>>>, AkitaError> {
        // Retained sources may be imported by another backend. Only this
        // owner may use the cached key; all other owners use their CPU path.
        if !std::ptr::eq(expanded, self.expanded.as_ref())
            || one_hot_k != K
            || plan.ring_dimension != D
        {
            return Ok(None);
        }
        let _span = tracing::info_span!("metal_trace_commitment").entered();
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
        let mut cached = self
            .key
            .lock()
            .map_err(|_| compute_error("key cache lock poisoned"))?;
        if cached.as_ref().is_none_or(|(len, _)| *len != key_len) {
            let _prepare = tracing::info_span!("metal_trace_commit_prepare_key").entered();
            *cached = Some((
                key_len,
                DeviceOneHotNttKey::new(&self.metal, key_coefficients).map_err(compute_error)?,
            ));
        }
        let (_, key) = cached
            .as_mut()
            .ok_or_else(|| compute_error("missing prepared key"))?;
        let bytes_per_block = checked::product([rows_per_block, columns + size_of::<u64>()])
            .ok_or_else(|| AkitaError::InvalidInput("trace staging size overflows".into()))?;
        let blocks_per_window = (STAGING_BYTES / bytes_per_block).max(1);
        let active_blocks = prefix.div_ceil(rows_per_block);
        let mut workspace = OneHotNttWorkspace::new(&self.metal);
        for first_block in (0..active_blocks).step_by(blocks_per_window) {
            let blocks = blocks_per_window.min(active_blocks - first_block);
            let first_row = first_block * rows_per_block;
            let window_rows = (blocks * rows_per_block).min(logical_rows - first_row);
            let stored_rows = window_rows.min(prefix - first_row);
            let selectors_len = checked::product([stored_rows, columns])
                .ok_or_else(|| AkitaError::InvalidInput("trace selector size overflows".into()))?;
            let mut selected = vec![0; selectors_len];
            let mut masks = vec![0; stored_rows];
            let pack_span = tracing::info_span!("metal_trace_commit_pack").entered();
            #[cfg(feature = "parallel")]
            selected
                .par_chunks_mut(PACK_ROWS * columns)
                .zip(masks.par_chunks_mut(PACK_ROWS))
                .enumerate()
                .for_each(|(chunk, (selected, masks))| {
                    rows.fill_rows_with_masks(first_row + chunk * PACK_ROWS, selected, masks);
                });
            #[cfg(not(feature = "parallel"))]
            rows.fill_rows_with_masks(first_row, &mut selected, &mut masks);
            let sources = DeviceK16Sources::from_row_major(
                &self.metal,
                selected,
                &masks,
                columns,
                window_rows,
            )
            .map_err(compute_error)?;
            drop(masks);
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
        }
        output
            .into_iter()
            .map(|coefficients| RingVec::from_coeffs_with_ring_dim(coefficients, D))
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
    }
}

fn compute_error(error: impl fmt::Display) -> AkitaError {
    AkitaError::InvalidInput(format!("Metal trace commitment: {error}"))
}
