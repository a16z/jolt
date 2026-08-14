use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext as DriverContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use jolt_field::Fr;

use super::device::{fill_staging, DeviceFrVec, LIMBS};
use super::error::CudaError;
use super::staging::StagingPool;
use super::xfer_stats::{self, Phase};

pub const BLOCK: u32 = 256;

const KERNEL_SRC: &str = concat!(
    include_str!("../kernels/prelude.cu"),
    "\n",
    include_str!("../kernels/probe.cu"),
    "\n",
    include_str!("../kernels/arith.cu"),
    "\n",
    include_str!("../kernels/tables.cu"),
    "\n",
    include_str!("../kernels/scan.cu"),
    "\n",
    include_str!("../kernels/lt_poly.cu"),
    "\n",
    include_str!("../kernels/dense_product.cu"),
    "\n",
    include_str!("../kernels/ra_poly.cu"),
    "\n",
    include_str!("../kernels/ram_ra_reduction.cu"),
    "\n",
    include_str!("../kernels/prefix_suffix.cu"),
    "\n",
    include_str!("../kernels/suffixes.cu"),
    "\n",
    include_str!("../kernels/prefixes.cu"),
    "\n",
    include_str!("../kernels/prefix_mle.cu"),
    "\n",
    include_str!("../kernels/combine.cu"),
    "\n",
    include_str!("../kernels/unreduced.cu"),
    "\n",
    include_str!("../kernels/product_accum.cu"),
    "\n",
    include_str!("../kernels/read_write_matrix.cu"),
    "\n",
    include_str!("../kernels/rs2_claim.cu"),
    "\n",
    include_str!("../kernels/address_major_matrix.cu"),
    "\n",
    include_str!("../kernels/address_phase.cu"),
    "\n",
    include_str!("../kernels/cycle_rounds.cu"),
    "\n",
    include_str!("../kernels/ram_read_write.cu"),
    "\n",
    include_str!("../kernels/registers_read_write.cu"),
);

pub struct CudaKernelContext {
    stream: Arc<CudaStream>,
    staging: StagingPool,
    fr_identity_probe: CudaFunction,
    pub(super) add: CudaFunction,
    pub(super) sub: CudaFunction,
    pub(super) mul: CudaFunction,
    pub(super) mul_scalar: CudaFunction,
    pub(super) add_scalar: CudaFunction,
    pub(super) fma: CudaFunction,
    pub(super) bind_low_to_high: CudaFunction,
    pub(super) bind_high_to_low: CudaFunction,
    pub(super) sum_reduce: CudaFunction,
    pub(super) u64_to_mont: CudaFunction,
    pub(super) i128_to_mont: CudaFunction,
    pub(super) eq_double: CudaFunction,
    pub(super) lt_double: CudaFunction,
    pub(super) scan_u32_block: CudaFunction,
    pub(super) scan_u32_add_offsets: CudaFunction,
    dense_product_round: CudaFunction,
    lane_sum_reduce: CudaFunction,
    weighted_combine: CudaFunction,
    ra_split_tables: CudaFunction,
    ra_gather: CudaFunction,
    lt_reconstruct: CudaFunction,
    ram_ra_gather_h: CudaFunction,
    ram_ra_fold_suffix: CudaFunction,
    ram_ra_fold_prefix: CudaFunction,
    ram_ra_phase1_round: CudaFunction,
    ps_init_q_raf: CudaFunction,
    ps_scale_shift: CudaFunction,
    sfx_eval_batch: CudaFunction,
    pfx_eval_batch: CudaFunction,
    pfx_mle_batch: CudaFunction,
    pfx_update_checkpoints: CudaFunction,
    pfx_mle_round: CudaFunction,
    pfx_default_checkpoints: CudaFunction,
    cmb_combine: CudaFunction,
    ap_raf_keys: CudaFunction,
    ap_table_keys: CudaFunction,
    ap_histogram: CudaFunction,
    ap_scatter: CudaFunction,
    ap_raf_reduce: CudaFunction,
    ap_suffix_reduce: CudaFunction,
    ap_scale_shift: CudaFunction,
    ap_condense: CudaFunction,
    ap_raf_prefix: CudaFunction,
    ap_bind_strided: CudaFunction,
    ap_round_message_hinted: CudaFunction,
    ap_combined_val: CudaFunction,
    ap_ra: CudaFunction,
    ap_flag_keys: CudaFunction,
    ap_flag_sums: CudaFunction,
    ap_raf_flag_sum: CudaFunction,
    unr_mul_scatter: CudaFunction,
    unr_reduce: CudaFunction,
    pa_scatter: CudaFunction,
    pa_reduce: CudaFunction,
    rwm_segment_flags: CudaFunction,
    rwm_segment_bounds: CudaFunction,
    rwm_count: CudaFunction,
    rwm_merge: CudaFunction,
    rwm_message: CudaFunction,
    rs2_claim: CudaFunction,
    amm_segment_flags: CudaFunction,
    amm_segment_bounds: CudaFunction,
    amm_count: CudaFunction,
    amm_merge: CudaFunction,
    amm_message: CudaFunction,
    amm_materialize: CudaFunction,
    amm_lift: CudaFunction,
    fr_delta_u64: CudaFunction,
    rrw_flags: CudaFunction,
    rrw_scatter: CudaFunction,
    reg_count: CudaFunction,
    reg_scatter: CudaFunction,
    cr_quotient: CudaFunction,
}

impl CudaKernelContext {
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        let context = DriverContext::new(ordinal)?;
        let stream = context.default_stream();
        let options = CompileOptions {
            options: vec!["--device-int128".to_owned()],
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(KERNEL_SRC, options)?;
        let module = context.load_module(ptx)?;
        Ok(Self {
            stream,
            staging: StagingPool::new(),
            fr_identity_probe: module.load_function("fr_identity_probe")?,
            add: module.load_function("add_kernel")?,
            sub: module.load_function("sub_kernel")?,
            mul: module.load_function("mul_kernel")?,
            mul_scalar: module.load_function("mul_scalar_kernel")?,
            add_scalar: module.load_function("add_scalar_kernel")?,
            fma: module.load_function("fma_kernel")?,
            bind_low_to_high: module.load_function("bind_low_to_high_kernel")?,
            bind_high_to_low: module.load_function("bind_high_to_low_kernel")?,
            sum_reduce: module.load_function("sum_reduce_kernel")?,
            u64_to_mont: module.load_function("u64_to_mont_kernel")?,
            i128_to_mont: module.load_function("i128_to_mont_kernel")?,
            eq_double: module.load_function("eq_double_kernel")?,
            lt_double: module.load_function("lt_double_kernel")?,
            scan_u32_block: module.load_function("scan_u32_block_kernel")?,
            scan_u32_add_offsets: module.load_function("scan_u32_add_offsets_kernel")?,
            dense_product_round: module.load_function("dense_product_round_kernel")?,
            lane_sum_reduce: module.load_function("lane_sum_reduce_kernel")?,
            weighted_combine: module.load_function("weighted_combine_kernel")?,
            ra_split_tables: module.load_function("ra_split_tables_kernel")?,
            ra_gather: module.load_function("ra_gather_kernel")?,
            lt_reconstruct: module.load_function("lt_reconstruct_kernel")?,
            ram_ra_gather_h: module.load_function("ram_ra_gather_h_kernel")?,
            ram_ra_fold_suffix: module.load_function("ram_ra_fold_suffix_kernel")?,
            ram_ra_fold_prefix: module.load_function("ram_ra_fold_prefix_kernel")?,
            ram_ra_phase1_round: module.load_function("ram_ra_phase1_round_kernel")?,
            ps_init_q_raf: module.load_function("ps_init_q_raf_kernel")?,
            ps_scale_shift: module.load_function("ps_scale_shift_kernel")?,
            sfx_eval_batch: module.load_function("sfx_eval_batch_kernel")?,
            pfx_eval_batch: module.load_function("pfx_eval_batch_kernel")?,
            pfx_mle_batch: module.load_function("pfx_mle_batch_kernel")?,
            pfx_update_checkpoints: module.load_function("pfx_update_checkpoints_kernel")?,
            pfx_mle_round: module.load_function("pfx_mle_round_kernel")?,
            pfx_default_checkpoints: module.load_function("pfx_default_checkpoints_kernel")?,
            cmb_combine: module.load_function("cmb_combine_kernel")?,
            ap_raf_keys: module.load_function("ap_raf_keys_kernel")?,
            ap_table_keys: module.load_function("ap_table_keys_kernel")?,
            ap_histogram: module.load_function("ap_histogram_kernel")?,
            ap_scatter: module.load_function("ap_scatter_kernel")?,
            ap_raf_reduce: module.load_function("ap_raf_reduce_kernel")?,
            ap_suffix_reduce: module.load_function("ap_suffix_reduce_kernel")?,
            ap_scale_shift: module.load_function("ap_scale_shift_kernel")?,
            ap_condense: module.load_function("ap_condense_kernel")?,
            ap_raf_prefix: module.load_function("ap_raf_prefix_kernel")?,
            ap_bind_strided: module.load_function("ap_bind_strided_kernel")?,
            ap_round_message_hinted: module.load_function("ap_round_message_hinted_kernel")?,
            ap_combined_val: module.load_function("ap_combined_val_kernel")?,
            ap_ra: module.load_function("ap_ra_kernel")?,
            ap_flag_keys: module.load_function("ap_flag_keys_kernel")?,
            ap_flag_sums: module.load_function("ap_flag_sums_kernel")?,
            ap_raf_flag_sum: module.load_function("ap_raf_flag_sum_kernel")?,
            unr_mul_scatter: module.load_function("unr_mul_scatter_kernel")?,
            unr_reduce: module.load_function("unr_reduce_kernel")?,
            pa_scatter: module.load_function("pa_scatter_kernel")?,
            pa_reduce: module.load_function("pa_reduce_kernel")?,
            rwm_segment_flags: module.load_function("rwm_segment_flags_kernel")?,
            rwm_segment_bounds: module.load_function("rwm_segment_bounds_kernel")?,
            rwm_count: module.load_function("rwm_count_kernel")?,
            rwm_merge: module.load_function("rwm_merge_kernel")?,
            rwm_message: module.load_function("rwm_message_kernel")?,
            rs2_claim: module.load_function("rs2_claim_kernel")?,
            amm_segment_flags: module.load_function("amm_segment_flags_kernel")?,
            amm_segment_bounds: module.load_function("amm_segment_bounds_kernel")?,
            amm_count: module.load_function("amm_count_kernel")?,
            amm_merge: module.load_function("amm_merge_kernel")?,
            amm_message: module.load_function("amm_message_kernel")?,
            amm_materialize: module.load_function("amm_materialize_kernel")?,
            amm_lift: module.load_function("amm_lift_kernel")?,
            fr_delta_u64: module.load_function("fr_delta_u64_kernel")?,
            rrw_flags: module.load_function("rrw_flags_kernel")?,
            rrw_scatter: module.load_function("rrw_scatter_kernel")?,
            reg_count: module.load_function("reg_count_kernel")?,
            reg_scatter: module.load_function("reg_scatter_kernel")?,
            cr_quotient: module.load_function("cr_quotient_kernel")?,
        })
    }

    pub fn upload(&self, values: &[Fr]) -> Result<DeviceFrVec, CudaError> {
        let limbs = values.len() * LIMBS;
        if values.is_empty() {
            let buffer = self.stream.alloc_zeros::<u64>(0)?;
            return Ok(DeviceFrVec::from_parts(
                self.stream.clone(),
                buffer,
                0,
                self.staging.clone(),
            ));
        }
        let buffer = xfer_stats::timed(Phase::H2d, limbs * size_of::<u64>(), || {
            let mut pool = self.staging.lock();
            let staging = pool.ensure(self.stream.context(), limbs)?;
            fill_staging(&mut staging.as_mut_slice()?[..limbs], values);
            let mut buffer = self.stream.alloc_zeros::<u64>(limbs)?;
            self.stream
                .memcpy_htod(&staging.as_slice()?[..limbs], &mut buffer)?;
            self.stream.synchronize()?;
            Ok::<_, CudaError>(buffer)
        })?;
        Ok(DeviceFrVec::from_parts(
            self.stream.clone(),
            buffer,
            values.len(),
            self.staging.clone(),
        ))
    }

    pub fn alloc(&self, len: usize) -> Result<DeviceFrVec, CudaError> {
        let buffer = self.stream.alloc_zeros::<u64>(len * LIMBS)?;
        Ok(DeviceFrVec::from_parts(
            self.stream.clone(),
            buffer,
            len,
            self.staging.clone(),
        ))
    }

    pub(crate) const fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub(crate) fn launch_config(count: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (count.div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    pub(crate) fn count_of(len: usize) -> Result<u32, CudaError> {
        u32::try_from(len).map_err(|_| CudaError::LengthMismatch {
            expected: u32::MAX as usize,
            got: len,
        })
    }

    pub(crate) fn upload_u64_slice(&self, values: &[u64]) -> Result<CudaSlice<u64>, CudaError> {
        xfer_stats::timed(Phase::H2d, size_of_val(values), || {
            Ok(self.stream.clone_htod(values)?)
        })
    }

    pub(crate) fn upload_u32_slice(&self, values: &[u32]) -> Result<CudaSlice<u32>, CudaError> {
        xfer_stats::timed(Phase::H2d, size_of_val(values), || {
            Ok(self.stream.clone_htod(values)?)
        })
    }

    pub(crate) fn upload_u8_slice(&self, values: &[u8]) -> Result<CudaSlice<u8>, CudaError> {
        xfer_stats::timed(Phase::H2d, size_of_val(values), || {
            Ok(self.stream.clone_htod(values)?)
        })
    }

    pub(crate) fn download_u32_range(
        &self,
        buffer: &CudaSlice<u32>,
        start: usize,
        end: usize,
    ) -> Result<Vec<u32>, CudaError> {
        let view = buffer.slice(start..end);
        xfer_stats::timed(Phase::D2h, (end - start) * size_of::<u32>(), || {
            Ok(self.stream.clone_dtoh(&view)?)
        })
    }

    pub(crate) fn clone_u32(&self, buffer: &CudaSlice<u32>) -> Result<CudaSlice<u32>, CudaError> {
        xfer_stats::timed(Phase::D2d, buffer.len() * size_of::<u32>(), || {
            Ok(self.stream.clone_dtod(buffer)?)
        })
    }

    pub(crate) fn download_u32(&self, buffer: &CudaSlice<u32>) -> Result<Vec<u32>, CudaError> {
        xfer_stats::timed(Phase::D2h, buffer.len() * size_of::<u32>(), || {
            Ok(self.stream.clone_dtoh(buffer)?)
        })
    }

    pub(crate) fn alloc_u32(&self, len: usize) -> Result<CudaSlice<u32>, CudaError> {
        Ok(self.stream.alloc_zeros::<u32>(len)?)
    }

    pub(crate) const fn dense_product_round(&self) -> &CudaFunction {
        &self.dense_product_round
    }

    pub(crate) const fn lane_sum_reduce(&self) -> &CudaFunction {
        &self.lane_sum_reduce
    }

    pub(crate) const fn weighted_combine(&self) -> &CudaFunction {
        &self.weighted_combine
    }

    pub(crate) const fn ra_split_tables(&self) -> &CudaFunction {
        &self.ra_split_tables
    }

    pub(crate) const fn ra_gather(&self) -> &CudaFunction {
        &self.ra_gather
    }

    pub(crate) const fn lt_reconstruct(&self) -> &CudaFunction {
        &self.lt_reconstruct
    }

    pub(crate) const fn ram_ra_gather_h(&self) -> &CudaFunction {
        &self.ram_ra_gather_h
    }

    pub(crate) const fn ram_ra_fold_suffix(&self) -> &CudaFunction {
        &self.ram_ra_fold_suffix
    }

    pub(crate) const fn ram_ra_fold_prefix(&self) -> &CudaFunction {
        &self.ram_ra_fold_prefix
    }

    pub(crate) const fn ram_ra_phase1_round(&self) -> &CudaFunction {
        &self.ram_ra_phase1_round
    }

    pub(crate) const fn ps_init_q_raf(&self) -> &CudaFunction {
        &self.ps_init_q_raf
    }

    pub(crate) const fn ps_scale_shift(&self) -> &CudaFunction {
        &self.ps_scale_shift
    }

    pub(crate) const fn sfx_eval_batch(&self) -> &CudaFunction {
        &self.sfx_eval_batch
    }

    pub(crate) const fn pfx_mle_round(&self) -> &CudaFunction {
        &self.pfx_mle_round
    }

    pub(crate) const fn pfx_update_checkpoints(&self) -> &CudaFunction {
        &self.pfx_update_checkpoints
    }

    pub(crate) const fn pfx_mle_batch(&self) -> &CudaFunction {
        &self.pfx_mle_batch
    }

    pub(crate) const fn pfx_eval_batch(&self) -> &CudaFunction {
        &self.pfx_eval_batch
    }

    pub(crate) const fn pfx_default_checkpoints(&self) -> &CudaFunction {
        &self.pfx_default_checkpoints
    }

    pub(crate) const fn cmb_combine(&self) -> &CudaFunction {
        &self.cmb_combine
    }

    pub(crate) const fn ap_raf_keys(&self) -> &CudaFunction {
        &self.ap_raf_keys
    }

    pub(crate) const fn ap_table_keys(&self) -> &CudaFunction {
        &self.ap_table_keys
    }

    pub(crate) const fn ap_histogram(&self) -> &CudaFunction {
        &self.ap_histogram
    }

    pub(crate) const fn ap_scatter(&self) -> &CudaFunction {
        &self.ap_scatter
    }

    pub(crate) const fn ap_raf_reduce(&self) -> &CudaFunction {
        &self.ap_raf_reduce
    }

    pub(crate) const fn ap_suffix_reduce(&self) -> &CudaFunction {
        &self.ap_suffix_reduce
    }

    pub(crate) const fn ap_scale_shift(&self) -> &CudaFunction {
        &self.ap_scale_shift
    }

    pub(crate) const fn ap_condense(&self) -> &CudaFunction {
        &self.ap_condense
    }

    pub(crate) const fn ap_raf_prefix(&self) -> &CudaFunction {
        &self.ap_raf_prefix
    }

    pub(crate) const fn ap_bind_strided(&self) -> &CudaFunction {
        &self.ap_bind_strided
    }

    pub(crate) const fn ap_round_message_hinted(&self) -> &CudaFunction {
        &self.ap_round_message_hinted
    }

    pub(crate) const fn ap_combined_val(&self) -> &CudaFunction {
        &self.ap_combined_val
    }

    pub(crate) const fn ap_ra(&self) -> &CudaFunction {
        &self.ap_ra
    }

    pub(crate) const fn ap_flag_keys(&self) -> &CudaFunction {
        &self.ap_flag_keys
    }

    pub(crate) const fn ap_flag_sums(&self) -> &CudaFunction {
        &self.ap_flag_sums
    }

    pub(crate) const fn ap_raf_flag_sum(&self) -> &CudaFunction {
        &self.ap_raf_flag_sum
    }

    pub(crate) const fn unr_mul_scatter(&self) -> &CudaFunction {
        &self.unr_mul_scatter
    }

    pub(crate) const fn cr_quotient(&self) -> &CudaFunction {
        &self.cr_quotient
    }

    pub(crate) const fn pa_scatter(&self) -> &CudaFunction {
        &self.pa_scatter
    }

    pub(crate) const fn pa_reduce(&self) -> &CudaFunction {
        &self.pa_reduce
    }

    pub(crate) const fn rwm_segment_flags(&self) -> &CudaFunction {
        &self.rwm_segment_flags
    }

    pub(crate) const fn rwm_segment_bounds(&self) -> &CudaFunction {
        &self.rwm_segment_bounds
    }

    pub(crate) const fn rwm_count(&self) -> &CudaFunction {
        &self.rwm_count
    }

    pub(crate) const fn rwm_merge(&self) -> &CudaFunction {
        &self.rwm_merge
    }

    pub(crate) const fn rwm_message(&self) -> &CudaFunction {
        &self.rwm_message
    }

    pub(crate) const fn rs2_claim(&self) -> &CudaFunction {
        &self.rs2_claim
    }

    pub(crate) const fn amm_segment_flags(&self) -> &CudaFunction {
        &self.amm_segment_flags
    }

    pub(crate) const fn amm_segment_bounds(&self) -> &CudaFunction {
        &self.amm_segment_bounds
    }

    pub(crate) const fn amm_count(&self) -> &CudaFunction {
        &self.amm_count
    }

    pub(crate) const fn amm_merge(&self) -> &CudaFunction {
        &self.amm_merge
    }

    pub(crate) const fn amm_message(&self) -> &CudaFunction {
        &self.amm_message
    }

    pub(crate) const fn amm_materialize(&self) -> &CudaFunction {
        &self.amm_materialize
    }

    pub(crate) const fn amm_lift(&self) -> &CudaFunction {
        &self.amm_lift
    }

    pub(crate) const fn fr_delta_u64(&self) -> &CudaFunction {
        &self.fr_delta_u64
    }

    pub(crate) const fn rrw_flags(&self) -> &CudaFunction {
        &self.rrw_flags
    }

    pub(crate) const fn rrw_scatter(&self) -> &CudaFunction {
        &self.rrw_scatter
    }

    pub(crate) const fn reg_count(&self) -> &CudaFunction {
        &self.reg_count
    }

    pub(crate) const fn reg_scatter(&self) -> &CudaFunction {
        &self.reg_scatter
    }

    pub(crate) const fn unr_reduce(&self) -> &CudaFunction {
        &self.unr_reduce
    }

    pub(crate) fn copy_into(
        &self,
        destination: &mut DeviceFrVec,
        offset: usize,
        source: &DeviceFrVec,
    ) -> Result<(), CudaError> {
        if offset + source.len() > destination.len() {
            return Err(CudaError::LengthMismatch {
                expected: destination.len(),
                got: offset + source.len(),
            });
        }
        if source.is_empty() {
            return Ok(());
        }
        let limbs = source.len() * LIMBS;
        let start = offset * LIMBS;
        xfer_stats::timed(Phase::D2d, limbs * size_of::<u64>(), || {
            let source = source.limbs().slice(0..limbs);
            let mut target = destination.limbs_mut().slice_mut(start..start + limbs);
            self.stream.memcpy_dtod(&source, &mut target)?;
            self.stream.synchronize()?;
            Ok::<_, CudaError>(())
        })
    }

    pub(crate) fn alloc_u64(&self, len: usize) -> Result<CudaSlice<u64>, CudaError> {
        Ok(self.stream.alloc_zeros::<u64>(len)?)
    }

    pub(crate) fn download_u64(&self, buffer: &CudaSlice<u64>) -> Result<Vec<u64>, CudaError> {
        xfer_stats::timed(Phase::D2h, buffer.len() * size_of::<u64>(), || {
            Ok(self.stream.clone_dtoh(buffer)?)
        })
    }

    pub(crate) fn device_pointers(
        &self,
        tables: &[&DeviceFrVec],
    ) -> Result<CudaSlice<u64>, CudaError> {
        let pointers: Vec<u64> = tables
            .iter()
            .map(|table| {
                let (pointer, _guard) = table.limbs().device_ptr(&self.stream);
                pointer
            })
            .collect();
        self.upload_u64_slice(&pointers)
    }

    pub fn fr_identity(&self, input: &DeviceFrVec) -> Result<DeviceFrVec, CudaError> {
        let len = input.len();
        let mut output = self.alloc(len)?;
        if len == 0 {
            return Ok(output);
        }
        let count = u32::try_from(len).map_err(|_| CudaError::LengthMismatch {
            expected: u32::MAX as usize,
            got: len,
        })?;
        let config = LaunchConfig {
            grid_dim: (count.div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = self.stream.launch_builder(&self.fr_identity_probe);
        let _ = builder.arg(input.limbs());
        let _ = builder.arg(output.limbs_mut());
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` writes exactly `out[i*4..i*4+4]` and reads
        // exactly `in[i*4..i*4+4]`; both buffers hold `count * 4` u64s (allocated
        // as `len * LIMBS` above and by `upload`), and threads with `i >= count`
        // return before any access. The two buffers are distinct allocations, so
        // no thread aliases another's write.
        let _ = unsafe { builder.launch(config) }?;
        self.stream.synchronize()?;
        Ok(output)
    }
}

pub fn shared_context() -> Option<&'static CudaKernelContext> {
    static CONTEXT: OnceLock<Option<CudaKernelContext>> = OnceLock::new();
    CONTEXT
        .get_or_init(|| match CudaKernelContext::new(0) {
            Ok(context) => Some(context),
            Err(error) => {
                tracing::warn!("CUDA unavailable, falling back to the reference backend: {error}");
                None
            }
        })
        .as_ref()
}
