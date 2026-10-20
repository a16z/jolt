use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext as DriverContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
#[cfg(test)]
use cudarc::driver::{DevicePtr, DevicePtrMut, DeviceRepr};
use cudarc::nvrtc::Ptx;
use jolt_field::Fr;

use super::device::{fill_staging, DeviceFrVec, LIMBS};
use super::error::CudaError;
#[cfg(test)]
use super::half_fold::FoldColumn;
use super::xfer_stats::{self, Phase};

#[cfg(test)]
const POISON_BYTE: u8 = 0xA5;

#[cfg(test)]
fn poison_byte() -> Option<u8> {
    static POISON: OnceLock<Option<u8>> = OnceLock::new();
    *POISON.get_or_init(|| std::env::var_os("JOLT_CUDA_POISON").map(|_| POISON_BYTE))
}

pub const BLOCK: u32 = 256;

const KERNEL_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.cubin"));

pub struct CudaKernelContext {
    ordinal: usize,
    stream: Arc<CudaStream>,
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
    pub(super) twos_i128_to_mont: CudaFunction,
    pub(super) eq_double: CudaFunction,
    pub(super) lt_double: CudaFunction,
    pub(super) scan_u32_block: CudaFunction,
    pub(super) scan_u32_add_offsets: CudaFunction,
    lt_reconstruct: CudaFunction,
    pcr_round: CudaFunction,
    pcr_place_row: CudaFunction,
    #[cfg(test)]
    dense_product_round: CudaFunction,
    lane_sum_total: CudaFunction,
    #[cfg(test)]
    sop_round: CudaFunction,
    #[cfg(test)]
    sopg_round: CudaFunction,
    #[cfg(test)]
    ra_split_tables: CudaFunction,
    #[cfg(test)]
    ra_gather: CudaFunction,
    #[cfg(test)]
    sfx_eval_batch: CudaFunction,
    #[cfg(test)]
    pfx_mle_round: CudaFunction,
    #[cfg(test)]
    ap_bind_lanes: CudaFunction,
    #[cfg(test)]
    rwm_segment_flags: CudaFunction,
    #[cfg(test)]
    rwm_segment_bounds: CudaFunction,
    #[cfg(test)]
    rwm_count: CudaFunction,
    #[cfg(test)]
    rwm_merge: CudaFunction,
    #[cfg(test)]
    rwm_message: CudaFunction,
    #[cfg(test)]
    rwm_square_lut: CudaFunction,
    #[cfg(test)]
    rwm_deref_coeffs: CudaFunction,
    #[cfg(test)]
    rwm_permute_words: CudaFunction,
    #[cfg(test)]
    amm_segment_flags: CudaFunction,
    #[cfg(test)]
    amm_segment_bounds: CudaFunction,
    #[cfg(test)]
    amm_count: CudaFunction,
    #[cfg(test)]
    amm_merge: CudaFunction,
    #[cfg(test)]
    amm_message: CudaFunction,
    #[cfg(test)]
    amm_materialize: CudaFunction,
    #[cfg(test)]
    amm_initial_values: CudaFunction,
    #[cfg(test)]
    amm_lift: CudaFunction,
    #[cfg(test)]
    unr_mul_scatter: CudaFunction,
    #[cfg(test)]
    unr_reduce: CudaFunction,
    #[cfg(test)]
    ohf_fold: CudaFunction,
    #[cfg(test)]
    ohf_reduce: CudaFunction,
    #[cfg(test)]
    ohf_affine: CudaFunction,
    #[cfg(test)]
    msm_fq_add: CudaFunction,
    #[cfg(test)]
    msm_fq_sub: CudaFunction,
    #[cfg(test)]
    msm_fq_mul: CudaFunction,
    #[cfg(test)]
    msm_fq_batch_inverse: CudaFunction,
    #[cfg(test)]
    msm_g1_double: CudaFunction,
    #[cfg(test)]
    msm_g1_add: CudaFunction,
    #[cfg(test)]
    msm_g1_add_affine: CudaFunction,
    #[cfg(test)]
    msm_affine_denominators: CudaFunction,
    #[cfg(test)]
    msm_affine_combine: CudaFunction,
    #[cfg(test)]
    msm_from_montgomery: CudaFunction,
    #[cfg(test)]
    msm_digits: CudaFunction,
    #[cfg(test)]
    msm_bucket_count: CudaFunction,
    #[cfg(test)]
    msm_bucket_scatter: CudaFunction,
    #[cfg(test)]
    msm_window_accumulate: CudaFunction,
    #[cfg(test)]
    msm_one_hot_count: CudaFunction,
    #[cfg(test)]
    msm_one_hot_count_shared: CudaFunction,
    #[cfg(test)]
    msm_one_hot_scatter: CudaFunction,
    #[cfg(test)]
    msm_one_hot_scatter_shared: CudaFunction,
    #[cfg(test)]
    msm_one_hot_complement: CudaFunction,
    #[cfg(test)]
    msm_glv_decompose_2d: CudaFunction,
    #[cfg(test)]
    msm_glv_decompose_4d: CudaFunction,
    #[cfg(test)]
    msm_g2_frobenius: CudaFunction,
    #[cfg(test)]
    msm_g1_endomorphism: CudaFunction,
    #[cfg(test)]
    msm_shared_scalar_rows_glv: CudaFunction,
    #[cfg(test)]
    msm_shared_scalar_windows: CudaFunction,
    #[cfg(test)]
    msm_segment_sum: CudaFunction,
    #[cfg(test)]
    msm_segment_sum_warp: CudaFunction,
    #[cfg(test)]
    msm_segment_sum_classed: CudaFunction,
    #[cfg(test)]
    msm_segment_sum_small: CudaFunction,
    #[cfg(test)]
    msm_g2_segment_sum_small: CudaFunction,
    #[cfg(test)]
    msm_g2_bucket_reduce_chunked: CudaFunction,
    #[cfg(test)]
    msm_g2_point_rows_sum: CudaFunction,
    #[cfg(test)]
    msm_g2_window_fold: CudaFunction,
    #[cfg(test)]
    msm_bucket_reduce_chunked: CudaFunction,
    #[cfg(test)]
    msm_point_rows_sum: CudaFunction,
    #[cfg(test)]
    msm_window_fold: CudaFunction,
    #[cfg(test)]
    msm_jacobian_z: CudaFunction,
    #[cfg(test)]
    msm_jacobian_to_affine: CudaFunction,
    #[cfg(test)]
    pairing_miller: CudaFunction,
    #[cfg(test)]
    pairing_miller_warp: CudaFunction,
    #[cfg(test)]
    pairing_prepare_g2: CudaFunction,
    #[cfg(test)]
    pairing_miller_prepared: CudaFunction,
    #[cfg(test)]
    pairing_fq12_product: CudaFunction,
    #[cfg(test)]
    pub(super) fq6_mul_probe: CudaFunction,
    #[cfg(test)]
    pub(super) fq6_mul_by_01_probe: CudaFunction,
    #[cfg(test)]
    pub(super) fq6_mul_by_fq2_probe: CudaFunction,
    #[cfg(test)]
    pub(super) fq6_mul_by_nonresidue_probe: CudaFunction,
    #[cfg(test)]
    pub(super) fq12_mul_probe: CudaFunction,
    #[cfg(test)]
    pub(super) fq12_sqr_probe: CudaFunction,
    #[cfg(test)]
    pub(super) fq12_mul_by_034_probe: CudaFunction,
    #[cfg(test)]
    pub(super) ell_probe: CudaFunction,
    #[cfg(test)]
    pub(super) g2_double_step_probe: CudaFunction,
    #[cfg(test)]
    pub(super) g2_add_step_probe: CudaFunction,
    #[cfg(test)]
    pub(super) g2_mul_by_char_probe: CudaFunction,
    #[cfg(test)]
    hf_half_fold: CudaFunction,
    #[cfg(test)]
    hf_row_fold: CudaFunction,
}

impl CudaKernelContext {
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        let context = DriverContext::new(ordinal)?;
        // Explicit streams make cudarc bind the owning context before freeing
        // buffers retired by main's background-drop workers.
        let stream = context.new_stream()?;
        let module = context.load_module(Ptx::from_binary(KERNEL_CUBIN.to_vec()))?;
        Ok(Self {
            ordinal,
            stream,
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
            twos_i128_to_mont: module.load_function("twos_i128_to_mont_kernel")?,
            eq_double: module.load_function("eq_double_kernel")?,
            lt_double: module.load_function("lt_double_kernel")?,
            scan_u32_block: module.load_function("scan_u32_block_kernel")?,
            scan_u32_add_offsets: module.load_function("scan_u32_add_offsets_kernel")?,
            lt_reconstruct: module.load_function("lt_reconstruct_kernel")?,
            pcr_round: module.load_function("pcr_round_kernel")?,
            pcr_place_row: module.load_function("pcr_place_row_kernel")?,
            #[cfg(test)]
            dense_product_round: module.load_function("dense_product_round_kernel")?,
            lane_sum_total: module.load_function("lane_sum_total_kernel")?,
            #[cfg(test)]
            sop_round: module.load_function("sop_round_kernel")?,
            #[cfg(test)]
            sopg_round: module.load_function("sopg_round_kernel")?,
            #[cfg(test)]
            ra_split_tables: module.load_function("ra_split_tables_kernel")?,
            #[cfg(test)]
            ra_gather: module.load_function("ra_gather_kernel")?,
            #[cfg(test)]
            sfx_eval_batch: module.load_function("sfx_eval_batch_kernel")?,
            #[cfg(test)]
            pfx_mle_round: module.load_function("pfx_mle_round_kernel")?,
            #[cfg(test)]
            ap_bind_lanes: module.load_function("ap_bind_lanes_kernel")?,
            #[cfg(test)]
            rwm_segment_flags: module.load_function("rwm_segment_flags_kernel")?,
            #[cfg(test)]
            rwm_segment_bounds: module.load_function("rwm_segment_bounds_kernel")?,
            #[cfg(test)]
            rwm_count: module.load_function("rwm_count_kernel")?,
            #[cfg(test)]
            rwm_merge: module.load_function("rwm_merge_kernel")?,
            #[cfg(test)]
            rwm_message: module.load_function("rwm_message_kernel")?,
            #[cfg(test)]
            rwm_square_lut: module.load_function("rwm_square_lut_kernel")?,
            #[cfg(test)]
            rwm_deref_coeffs: module.load_function("rwm_deref_coeffs_kernel")?,
            #[cfg(test)]
            rwm_permute_words: module.load_function("rwm_permute_words_kernel")?,
            #[cfg(test)]
            amm_segment_flags: module.load_function("amm_segment_flags_kernel")?,
            #[cfg(test)]
            amm_segment_bounds: module.load_function("amm_segment_bounds_kernel")?,
            #[cfg(test)]
            amm_count: module.load_function("amm_count_kernel")?,
            #[cfg(test)]
            amm_merge: module.load_function("amm_merge_kernel")?,
            #[cfg(test)]
            amm_message: module.load_function("amm_message_kernel")?,
            #[cfg(test)]
            amm_materialize: module.load_function("amm_materialize_kernel")?,
            #[cfg(test)]
            amm_initial_values: module.load_function("amm_initial_values_kernel")?,
            #[cfg(test)]
            amm_lift: module.load_function("amm_lift_kernel")?,
            #[cfg(test)]
            unr_mul_scatter: module.load_function("unr_mul_scatter_kernel")?,
            #[cfg(test)]
            unr_reduce: module.load_function("unr_reduce_kernel")?,
            #[cfg(test)]
            ohf_fold: module.load_function("ohf_fold_kernel")?,
            #[cfg(test)]
            ohf_reduce: module.load_function("ohf_reduce_kernel")?,
            #[cfg(test)]
            ohf_affine: module.load_function("ohf_affine_kernel")?,
            #[cfg(test)]
            msm_fq_add: module.load_function("msm_fq_add_kernel")?,
            #[cfg(test)]
            msm_fq_sub: module.load_function("msm_fq_sub_kernel")?,
            #[cfg(test)]
            msm_fq_mul: module.load_function("msm_fq_mul_kernel")?,
            #[cfg(test)]
            msm_fq_batch_inverse: module.load_function("msm_fq_batch_inverse_kernel")?,
            #[cfg(test)]
            msm_g1_double: module.load_function("msm_g1_double_kernel")?,
            #[cfg(test)]
            msm_g1_add: module.load_function("msm_g1_add_kernel")?,
            #[cfg(test)]
            msm_g1_add_affine: module.load_function("msm_g1_add_affine_kernel")?,
            #[cfg(test)]
            msm_affine_denominators: module.load_function("msm_affine_denominators_kernel")?,
            #[cfg(test)]
            msm_affine_combine: module.load_function("msm_affine_combine_kernel")?,
            #[cfg(test)]
            msm_from_montgomery: module.load_function("msm_from_montgomery_kernel")?,
            #[cfg(test)]
            msm_digits: module.load_function("msm_digits_kernel")?,
            #[cfg(test)]
            msm_bucket_count: module.load_function("msm_bucket_count_kernel")?,
            #[cfg(test)]
            msm_bucket_scatter: module.load_function("msm_bucket_scatter_kernel")?,
            #[cfg(test)]
            msm_window_accumulate: module.load_function("msm_window_accumulate_kernel")?,
            #[cfg(test)]
            msm_one_hot_count: module.load_function("msm_one_hot_count_kernel")?,
            #[cfg(test)]
            msm_one_hot_count_shared: module.load_function("msm_one_hot_count_shared_kernel")?,
            #[cfg(test)]
            msm_one_hot_scatter: module.load_function("msm_one_hot_scatter_kernel")?,
            #[cfg(test)]
            msm_one_hot_scatter_shared: module
                .load_function("msm_one_hot_scatter_shared_kernel")?,
            #[cfg(test)]
            msm_one_hot_complement: module.load_function("msm_one_hot_complement_kernel")?,
            #[cfg(test)]
            msm_glv_decompose_2d: module.load_function("msm_glv_decompose_2d_kernel")?,
            #[cfg(test)]
            msm_glv_decompose_4d: module.load_function("msm_glv_decompose_4d_kernel")?,
            #[cfg(test)]
            msm_g2_frobenius: module.load_function("msm_g2_frobenius_kernel")?,
            #[cfg(test)]
            msm_g1_endomorphism: module.load_function("msm_g1_endomorphism_kernel")?,
            #[cfg(test)]
            msm_shared_scalar_rows_glv: module
                .load_function("msm_shared_scalar_rows_glv_kernel")?,
            #[cfg(test)]
            msm_shared_scalar_windows: module.load_function("msm_shared_scalar_windows_kernel")?,
            #[cfg(test)]
            msm_segment_sum: module.load_function("msm_segment_sum_kernel")?,
            #[cfg(test)]
            msm_segment_sum_warp: module.load_function("msm_segment_sum_warp_kernel")?,
            #[cfg(test)]
            msm_segment_sum_classed: module.load_function("msm_segment_sum_classed_kernel")?,
            #[cfg(test)]
            msm_segment_sum_small: module.load_function("msm_segment_sum_small_kernel")?,
            #[cfg(test)]
            msm_g2_segment_sum_small: module.load_function("msm_g2_segment_sum_small_kernel")?,
            #[cfg(test)]
            msm_g2_bucket_reduce_chunked: module
                .load_function("msm_g2_bucket_reduce_chunked_kernel")?,
            #[cfg(test)]
            msm_g2_point_rows_sum: module.load_function("msm_g2_point_rows_sum_kernel")?,
            #[cfg(test)]
            msm_g2_window_fold: module.load_function("msm_g2_window_fold_kernel")?,
            #[cfg(test)]
            msm_bucket_reduce_chunked: module.load_function("msm_bucket_reduce_chunked_kernel")?,
            #[cfg(test)]
            msm_point_rows_sum: module.load_function("msm_point_rows_sum_kernel")?,
            #[cfg(test)]
            msm_window_fold: module.load_function("msm_window_fold_kernel")?,
            #[cfg(test)]
            msm_jacobian_z: module.load_function("msm_jacobian_z_kernel")?,
            #[cfg(test)]
            msm_jacobian_to_affine: module.load_function("msm_jacobian_to_affine_kernel")?,
            #[cfg(test)]
            pairing_miller: module.load_function("pairing_miller_kernel")?,
            #[cfg(test)]
            pairing_miller_warp: module.load_function("pairing_miller_warp_kernel")?,
            #[cfg(test)]
            pairing_prepare_g2: module.load_function("pairing_prepare_g2_kernel")?,
            #[cfg(test)]
            pairing_miller_prepared: module.load_function("pairing_miller_prepared_kernel")?,
            #[cfg(test)]
            pairing_fq12_product: module.load_function("pairing_fq12_product_kernel")?,
            #[cfg(test)]
            fq6_mul_probe: module.load_function("fq6_mul_probe")?,
            #[cfg(test)]
            fq6_mul_by_01_probe: module.load_function("fq6_mul_by_01_probe")?,
            #[cfg(test)]
            fq6_mul_by_fq2_probe: module.load_function("fq6_mul_by_fq2_probe")?,
            #[cfg(test)]
            fq6_mul_by_nonresidue_probe: module.load_function("fq6_mul_by_nonresidue_probe")?,
            #[cfg(test)]
            fq12_mul_probe: module.load_function("fq12_mul_probe")?,
            #[cfg(test)]
            fq12_sqr_probe: module.load_function("fq12_sqr_probe")?,
            #[cfg(test)]
            fq12_mul_by_034_probe: module.load_function("fq12_mul_by_034_probe")?,
            #[cfg(test)]
            ell_probe: module.load_function("ell_probe")?,
            #[cfg(test)]
            g2_double_step_probe: module.load_function("g2_double_step_probe")?,
            #[cfg(test)]
            g2_add_step_probe: module.load_function("g2_add_step_probe")?,
            #[cfg(test)]
            g2_mul_by_char_probe: module.load_function("g2_mul_by_char_probe")?,
            #[cfg(test)]
            hf_half_fold: module.load_function("hf_half_fold_kernel")?,
            #[cfg(test)]
            hf_row_fold: module.load_function("hf_row_fold_kernel")?,
        })
    }

    pub fn upload(&self, values: &[Fr]) -> Result<DeviceFrVec, CudaError> {
        let limbs = values.len() * LIMBS;
        if values.is_empty() {
            let buffer = self.stream.alloc_zeros::<u64>(0)?;
            return Ok(DeviceFrVec::from_parts(self.stream.clone(), buffer, 0));
        }
        let buffer = xfer_stats::timed(Phase::H2d, limbs * size_of::<u64>(), || {
            let mut host = vec![0u64; limbs];
            fill_staging(&mut host, values);
            Ok::<_, CudaError>(self.stream.clone_htod(&host)?)
        })?;
        Ok(DeviceFrVec::from_parts(
            self.stream.clone(),
            buffer,
            values.len(),
        ))
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

    #[cfg(test)]
    pub(crate) fn upload_u16_slice(&self, values: &[u16]) -> Result<CudaSlice<u16>, CudaError> {
        xfer_stats::timed(Phase::H2d, size_of_val(values), || {
            Ok(self.stream.clone_htod(values)?)
        })
    }

    #[cfg(test)]
    pub(crate) fn download_u16(&self, buffer: &CudaSlice<u16>) -> Result<Vec<u16>, CudaError> {
        xfer_stats::timed(Phase::D2h, buffer.len() * size_of::<u16>(), || {
            Ok(self.stream.clone_dtoh(buffer)?)
        })
    }

    #[cfg(test)]
    pub(crate) fn alloc_u16_unset(&self, len: usize) -> Result<CudaSlice<u16>, CudaError> {
        self.alloc_unset::<u16>(len)
    }

    #[cfg(test)]
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

    pub(crate) fn upload_u8_slice(&self, values: &[u8]) -> Result<CudaSlice<u8>, CudaError> {
        xfer_stats::timed(Phase::H2d, size_of_val(values), || {
            Ok(self.stream.clone_htod(values)?)
        })
    }

    pub fn alloc(&self, len: usize) -> Result<DeviceFrVec, CudaError> {
        let buffer = self.stream.alloc_zeros::<u64>(len * LIMBS)?;
        Ok(DeviceFrVec::from_parts(self.stream.clone(), buffer, len))
    }

    pub(crate) fn alloc_u32(&self, len: usize) -> Result<CudaSlice<u32>, CudaError> {
        Ok(self.stream.alloc_zeros::<u32>(len)?)
    }

    #[cfg(test)]
    pub(crate) fn alloc_u8(&self, len: usize) -> Result<CudaSlice<u8>, CudaError> {
        Ok(self.stream.alloc_zeros::<u8>(len)?)
    }

    #[cfg(test)]
    fn alloc_unset<T: DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>, CudaError> {
        // SAFETY: `alloc` leaves the elements unset, which every caller of this
        // helper is required to tolerate: each writes all of `len` before any
        // read, or reads only positions it wrote. Nothing here observes the
        // contents. `JOLT_CUDA_POISON` turns that requirement into a test by
        // filling the block with a non-zero pattern instead, so a stale read
        // cannot pass as a zero.
        let mut buffer = unsafe { self.stream.alloc::<T>(len) }?;
        if let Some(byte) = poison_byte() {
            let bytes = len * size_of::<T>();
            let (pointer, _record) = buffer.device_ptr_mut(&self.stream);
            // SAFETY: `pointer` is the freshly allocated block above and `bytes`
            // is exactly its length, so the fill stays inside it; the memset is
            // ordered on the same stream as every later use.
            unsafe {
                cudarc::driver::result::memset_d8_async(
                    pointer,
                    byte,
                    bytes,
                    self.stream.cu_stream(),
                )
            }?;
        }
        Ok(buffer)
    }

    #[cfg(test)]
    pub(crate) fn alloc_u64_unset(&self, len: usize) -> Result<CudaSlice<u64>, CudaError> {
        self.alloc_unset::<u64>(len)
    }

    #[cfg(test)]
    pub(crate) fn alloc_u32_unset(&self, len: usize) -> Result<CudaSlice<u32>, CudaError> {
        self.alloc_unset::<u32>(len)
    }

    #[cfg(test)]
    pub(crate) fn replicate_u8(
        &self,
        buffer: &CudaSlice<u8>,
        copies: usize,
    ) -> Result<CudaSlice<u8>, CudaError> {
        let len = buffer.len();
        let mut out = self.alloc_u8(len * copies)?;
        xfer_stats::timed(Phase::D2d, len * copies, || -> Result<(), CudaError> {
            for copy in 0..copies {
                let source = buffer.slice(..);
                let mut target = out.slice_mut(copy * len..(copy + 1) * len);
                self.stream.memcpy_dtod(&source, &mut target)?;
            }
            Ok(())
        })?;
        Ok(out)
    }

    #[cfg(test)]
    pub(crate) fn alloc_u64(&self, len: usize) -> Result<CudaSlice<u64>, CudaError> {
        Ok(self.stream.alloc_zeros::<u64>(len)?)
    }

    #[cfg(test)]
    pub(crate) fn download_u64(&self, buffer: &CudaSlice<u64>) -> Result<Vec<u64>, CudaError> {
        xfer_stats::timed(Phase::D2h, buffer.len() * size_of::<u64>(), || {
            Ok(self.stream.clone_dtoh(buffer)?)
        })
    }

    #[cfg(test)]
    pub(crate) fn download_u8(&self, buffer: &CudaSlice<u8>) -> Result<Vec<u8>, CudaError> {
        xfer_stats::timed(Phase::D2h, buffer.len(), || {
            Ok(self.stream.clone_dtoh(buffer)?)
        })
    }

    #[cfg(test)]
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

    pub(crate) const fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub const fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn memory_used(&self) -> Result<(usize, usize), CudaError> {
        let (free, total) = self.stream.context().mem_get_info()?;
        Ok((total.saturating_sub(free), total))
    }

    #[cfg(test)]
    pub(crate) fn require_owned(&self, ordinal: usize) -> Result<(), CudaError> {
        if ordinal == self.ordinal {
            return Ok(());
        }
        Err(CudaError::InvariantViolation {
            reason: "a CUDA kernel was launched against a buffer resident on another device",
        })
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

    pub(crate) const fn lt_reconstruct(&self) -> &CudaFunction {
        &self.lt_reconstruct
    }

    #[cfg(test)]
    pub(crate) const fn hf_half_fold(&self) -> &CudaFunction {
        &self.hf_half_fold
    }

    #[cfg(test)]
    pub(crate) const fn hf_row_fold(&self) -> &CudaFunction {
        &self.hf_row_fold
    }

    pub(crate) const fn pcr_round(&self) -> &CudaFunction {
        &self.pcr_round
    }

    pub(crate) const fn pcr_place_row(&self) -> &CudaFunction {
        &self.pcr_place_row
    }

    #[cfg(test)]
    pub(crate) const fn dense_product_round(&self) -> &CudaFunction {
        &self.dense_product_round
    }

    pub(crate) const fn lane_sum_total(&self) -> &CudaFunction {
        &self.lane_sum_total
    }

    #[cfg(test)]
    pub(crate) const fn sop_round(&self) -> &CudaFunction {
        &self.sop_round
    }

    #[cfg(test)]
    pub(crate) const fn sopg_round(&self) -> &CudaFunction {
        &self.sopg_round
    }

    #[cfg(test)]
    pub(crate) const fn ra_split_tables(&self) -> &CudaFunction {
        &self.ra_split_tables
    }

    #[cfg(test)]
    pub(crate) const fn ra_gather(&self) -> &CudaFunction {
        &self.ra_gather
    }

    #[cfg(test)]
    pub(crate) const fn sfx_eval_batch(&self) -> &CudaFunction {
        &self.sfx_eval_batch
    }

    #[cfg(test)]
    pub(crate) const fn pfx_mle_round(&self) -> &CudaFunction {
        &self.pfx_mle_round
    }

    #[cfg(test)]
    pub(crate) const fn ap_bind_lanes(&self) -> &CudaFunction {
        &self.ap_bind_lanes
    }

    #[cfg(test)]
    pub(crate) const fn rwm_segment_flags(&self) -> &CudaFunction {
        &self.rwm_segment_flags
    }

    #[cfg(test)]
    pub(crate) const fn rwm_segment_bounds(&self) -> &CudaFunction {
        &self.rwm_segment_bounds
    }

    #[cfg(test)]
    pub(crate) const fn rwm_count(&self) -> &CudaFunction {
        &self.rwm_count
    }

    #[cfg(test)]
    pub(crate) const fn rwm_merge(&self) -> &CudaFunction {
        &self.rwm_merge
    }

    #[cfg(test)]
    pub(crate) const fn rwm_message(&self) -> &CudaFunction {
        &self.rwm_message
    }

    #[cfg(test)]
    pub(crate) const fn rwm_square_lut(&self) -> &CudaFunction {
        &self.rwm_square_lut
    }

    #[cfg(test)]
    pub(crate) const fn rwm_deref_coeffs(&self) -> &CudaFunction {
        &self.rwm_deref_coeffs
    }

    #[cfg(test)]
    pub(crate) const fn rwm_permute_words(&self) -> &CudaFunction {
        &self.rwm_permute_words
    }

    #[cfg(test)]
    pub(crate) const fn amm_segment_flags(&self) -> &CudaFunction {
        &self.amm_segment_flags
    }

    #[cfg(test)]
    pub(crate) const fn amm_segment_bounds(&self) -> &CudaFunction {
        &self.amm_segment_bounds
    }

    #[cfg(test)]
    pub(crate) const fn amm_count(&self) -> &CudaFunction {
        &self.amm_count
    }

    #[cfg(test)]
    pub(crate) const fn amm_merge(&self) -> &CudaFunction {
        &self.amm_merge
    }

    #[cfg(test)]
    pub(crate) const fn amm_message(&self) -> &CudaFunction {
        &self.amm_message
    }

    #[cfg(test)]
    pub(crate) const fn amm_materialize(&self) -> &CudaFunction {
        &self.amm_materialize
    }

    #[cfg(test)]
    pub(crate) const fn amm_initial_values(&self) -> &CudaFunction {
        &self.amm_initial_values
    }

    #[cfg(test)]
    pub(crate) const fn amm_lift(&self) -> &CudaFunction {
        &self.amm_lift
    }

    #[cfg(test)]
    pub(crate) const fn unr_mul_scatter(&self) -> &CudaFunction {
        &self.unr_mul_scatter
    }

    #[cfg(test)]
    pub(crate) const fn unr_reduce(&self) -> &CudaFunction {
        &self.unr_reduce
    }

    #[cfg(test)]
    pub(crate) const fn ohf_fold(&self) -> &CudaFunction {
        &self.ohf_fold
    }

    #[cfg(test)]
    pub(crate) const fn ohf_reduce(&self) -> &CudaFunction {
        &self.ohf_reduce
    }

    #[cfg(test)]
    pub(crate) const fn ohf_affine(&self) -> &CudaFunction {
        &self.ohf_affine
    }

    #[cfg(test)]
    pub(crate) const fn msm_fq_add(&self) -> &CudaFunction {
        &self.msm_fq_add
    }

    #[cfg(test)]
    pub(crate) const fn msm_fq_sub(&self) -> &CudaFunction {
        &self.msm_fq_sub
    }

    #[cfg(test)]
    pub(crate) const fn msm_fq_mul(&self) -> &CudaFunction {
        &self.msm_fq_mul
    }

    #[cfg(test)]
    pub(crate) const fn msm_fq_batch_inverse(&self) -> &CudaFunction {
        &self.msm_fq_batch_inverse
    }

    #[cfg(test)]
    pub(crate) const fn msm_g1_double(&self) -> &CudaFunction {
        &self.msm_g1_double
    }

    #[cfg(test)]
    pub(crate) const fn msm_g1_add(&self) -> &CudaFunction {
        &self.msm_g1_add
    }

    #[cfg(test)]
    pub(crate) const fn msm_g1_add_affine(&self) -> &CudaFunction {
        &self.msm_g1_add_affine
    }

    #[cfg(test)]
    pub(crate) const fn msm_affine_denominators(&self) -> &CudaFunction {
        &self.msm_affine_denominators
    }

    #[cfg(test)]
    pub(crate) const fn msm_affine_combine(&self) -> &CudaFunction {
        &self.msm_affine_combine
    }

    #[cfg(test)]
    pub(crate) const fn msm_from_montgomery(&self) -> &CudaFunction {
        &self.msm_from_montgomery
    }

    #[cfg(test)]
    pub(crate) const fn msm_digits(&self) -> &CudaFunction {
        &self.msm_digits
    }

    #[cfg(test)]
    pub(crate) const fn msm_bucket_count(&self) -> &CudaFunction {
        &self.msm_bucket_count
    }

    #[cfg(test)]
    pub(crate) const fn msm_bucket_scatter(&self) -> &CudaFunction {
        &self.msm_bucket_scatter
    }

    #[cfg(test)]
    pub(crate) const fn msm_window_accumulate(&self) -> &CudaFunction {
        &self.msm_window_accumulate
    }

    #[cfg(test)]
    pub(crate) const fn msm_one_hot_count(&self) -> &CudaFunction {
        &self.msm_one_hot_count
    }

    #[cfg(test)]
    pub(crate) const fn msm_one_hot_count_shared(&self) -> &CudaFunction {
        &self.msm_one_hot_count_shared
    }

    #[cfg(test)]
    pub(crate) const fn msm_one_hot_scatter(&self) -> &CudaFunction {
        &self.msm_one_hot_scatter
    }

    #[cfg(test)]
    pub(crate) const fn msm_one_hot_scatter_shared(&self) -> &CudaFunction {
        &self.msm_one_hot_scatter_shared
    }

    #[cfg(test)]
    pub(crate) const fn msm_one_hot_complement(&self) -> &CudaFunction {
        &self.msm_one_hot_complement
    }

    #[cfg(test)]
    pub(crate) const fn msm_glv_decompose_2d(&self) -> &CudaFunction {
        &self.msm_glv_decompose_2d
    }

    #[cfg(test)]
    pub(crate) const fn msm_glv_decompose_4d(&self) -> &CudaFunction {
        &self.msm_glv_decompose_4d
    }

    #[cfg(test)]
    pub(crate) const fn msm_g2_frobenius(&self) -> &CudaFunction {
        &self.msm_g2_frobenius
    }

    #[cfg(test)]
    pub(crate) const fn msm_g1_endomorphism(&self) -> &CudaFunction {
        &self.msm_g1_endomorphism
    }

    #[cfg(test)]
    pub(crate) const fn msm_shared_scalar_rows_glv(&self) -> &CudaFunction {
        &self.msm_shared_scalar_rows_glv
    }

    #[cfg(test)]
    pub(crate) const fn msm_shared_scalar_windows(&self) -> &CudaFunction {
        &self.msm_shared_scalar_windows
    }

    #[cfg(test)]
    pub(crate) const fn msm_segment_sum(&self) -> &CudaFunction {
        &self.msm_segment_sum
    }

    #[cfg(test)]
    pub(crate) const fn msm_segment_sum_warp(&self) -> &CudaFunction {
        &self.msm_segment_sum_warp
    }

    #[cfg(test)]
    pub(crate) const fn msm_segment_sum_classed(&self) -> &CudaFunction {
        &self.msm_segment_sum_classed
    }

    #[cfg(test)]
    pub(crate) const fn msm_segment_sum_small(&self) -> &CudaFunction {
        &self.msm_segment_sum_small
    }

    #[cfg(test)]
    pub(crate) const fn msm_g2_segment_sum_small(&self) -> &CudaFunction {
        &self.msm_g2_segment_sum_small
    }

    #[cfg(test)]
    pub(crate) const fn msm_g2_bucket_reduce_chunked(&self) -> &CudaFunction {
        &self.msm_g2_bucket_reduce_chunked
    }

    #[cfg(test)]
    pub(crate) const fn msm_g2_point_rows_sum(&self) -> &CudaFunction {
        &self.msm_g2_point_rows_sum
    }

    #[cfg(test)]
    pub(crate) const fn msm_g2_window_fold(&self) -> &CudaFunction {
        &self.msm_g2_window_fold
    }

    #[cfg(test)]
    pub(crate) const fn msm_bucket_reduce_chunked(&self) -> &CudaFunction {
        &self.msm_bucket_reduce_chunked
    }

    #[cfg(test)]
    pub(crate) const fn msm_point_rows_sum(&self) -> &CudaFunction {
        &self.msm_point_rows_sum
    }

    #[cfg(test)]
    pub(crate) const fn msm_window_fold(&self) -> &CudaFunction {
        &self.msm_window_fold
    }

    #[cfg(test)]
    pub(crate) const fn msm_jacobian_z(&self) -> &CudaFunction {
        &self.msm_jacobian_z
    }

    #[cfg(test)]
    pub(crate) const fn msm_jacobian_to_affine(&self) -> &CudaFunction {
        &self.msm_jacobian_to_affine
    }

    #[cfg(test)]
    pub(crate) const fn pairing_miller(&self) -> &CudaFunction {
        &self.pairing_miller
    }

    #[cfg(test)]
    pub(crate) const fn pairing_miller_warp(&self) -> &CudaFunction {
        &self.pairing_miller_warp
    }

    #[cfg(test)]
    pub(crate) const fn pairing_prepare_g2(&self) -> &CudaFunction {
        &self.pairing_prepare_g2
    }

    #[cfg(test)]
    pub(crate) const fn pairing_miller_prepared(&self) -> &CudaFunction {
        &self.pairing_miller_prepared
    }

    #[cfg(test)]
    pub(crate) const fn pairing_fq12_product(&self) -> &CudaFunction {
        &self.pairing_fq12_product
    }

    #[cfg(test)]
    pub(crate) fn device_columns(
        &self,
        columns: &[FoldColumn<'_>],
    ) -> Result<(CudaSlice<u64>, CudaSlice<u32>), CudaError> {
        let mut pointers = Vec::with_capacity(columns.len());
        let mut descriptors = Vec::with_capacity(columns.len() * 4);
        for column in columns {
            let (pointer, _guard) = column.words().device_ptr(&self.stream);
            pointers.push(pointer);
            descriptors.extend_from_slice(&column.descriptor()?);
        }
        Ok((
            self.upload_u64_slice(&pointers)?,
            self.upload_u32_slice(&descriptors)?,
        ))
    }

    #[cfg(test)]
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

pub const DEVICE_COUNT_VARIABLE: &str = "JOLT_CUDA_GPUS";

thread_local! {
    static CURRENT_DEVICE: Cell<usize> = const { Cell::new(0) };
}

pub struct DeviceGuard {
    previous: usize,
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        CURRENT_DEVICE.set(self.previous);
    }
}

#[must_use]
pub fn enter_device(ordinal: usize) -> DeviceGuard {
    DeviceGuard {
        previous: CURRENT_DEVICE.replace(ordinal),
    }
}

pub fn current_device() -> usize {
    CURRENT_DEVICE.get()
}

static REQUESTED_DEVICES: AtomicUsize = AtomicUsize::new(0);

static POOL: OnceLock<Vec<CudaKernelContext>> = OnceLock::new();

pub fn request_devices(count: usize) {
    REQUESTED_DEVICES.store(count, Ordering::Relaxed);
    if POOL.get().is_some() {
        tracing::warn!(
            "the CUDA device pool was already built; the request for {count} device(s) is ignored"
        );
    }
}

fn requested_devices() -> usize {
    let explicit = REQUESTED_DEVICES.load(Ordering::Relaxed);
    if explicit > 0 {
        return explicit;
    }
    std::env::var(DEVICE_COUNT_VARIABLE)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&count| count > 0)
        .unwrap_or(1)
}

fn pool() -> &'static [CudaKernelContext] {
    POOL.get_or_init(|| {
        let present = DriverContext::device_count().unwrap_or_default().max(0) as usize;
        let requested = requested_devices();
        let wanted = requested.min(present);
        let mut contexts = Vec::with_capacity(wanted);
        for ordinal in 0..wanted {
            match CudaKernelContext::new(ordinal) {
                Ok(context) => contexts.push(context),
                Err(error) => {
                    if ordinal == 0 {
                        tracing::warn!(
                            "CUDA device initialization failed: {error}"
                        );
                    } else {
                        tracing::warn!(
                            "CUDA device {ordinal} did not open, continuing on {ordinal} device(s): {error}"
                        );
                    }
                    break;
                }
            }
        }
        if requested > contexts.len() {
            tracing::warn!(
                "{DEVICE_COUNT_VARIABLE}={requested} but {} CUDA device(s) are usable",
                contexts.len()
            );
        }
        contexts
    })
}

pub fn device_count() -> usize {
    pool().len()
}

pub fn shared_context() -> Option<&'static CudaKernelContext> {
    pool().get(current_device())
}

pub fn context_for(ordinal: usize) -> Option<&'static CudaKernelContext> {
    pool().get(ordinal)
}

pub fn device_memory_used() -> Vec<usize> {
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                pool()
                    .iter()
                    .map(|context| context.memory_used().map_or(0, |(used, _)| used))
                    .collect()
            })
            .join()
            .unwrap_or_default()
    })
}
