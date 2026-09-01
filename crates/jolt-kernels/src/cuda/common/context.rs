use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext as DriverContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use jolt_field::Fr;

use super::device::{fill_staging, DeviceFrVec, LIMBS};
use super::error::CudaError;
use super::staging::StagingPool;
use super::xfer_stats::{self, Phase};

const POISON_BYTE: u8 = 0xA5;

fn poison_byte() -> Option<u8> {
    static POISON: OnceLock<Option<u8>> = OnceLock::new();
    *POISON.get_or_init(|| std::env::var_os("JOLT_CUDA_POISON").map(|_| POISON_BYTE))
}

pub const BLOCK: u32 = 256;

const KERNEL_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.cubin"));

pub struct CudaKernelContext {
    ordinal: usize,
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
    pub(super) u128_to_mont: CudaFunction,
    pub(super) twos_i128_to_mont: CudaFunction,
    pub(super) eq_double: CudaFunction,
    pub(super) lt_double: CudaFunction,
    pub(super) scan_u32_block: CudaFunction,
    pub(super) scan_u32_add_offsets: CudaFunction,
    dense_product_round: CudaFunction,
    lane_sum_total: CudaFunction,
    ra_split_tables: CudaFunction,
    ra_gather: CudaFunction,
    lt_reconstruct: CudaFunction,
    ram_ra_gather_h: CudaFunction,
    ram_ra_fold_suffix: CudaFunction,
    ram_ra_fold_prefix: CudaFunction,
    ram_ra_phase1_round: CudaFunction,
    sfx_eval_batch: CudaFunction,
    pfx_update_checkpoints: CudaFunction,
    pfx_mle_round: CudaFunction,
    pfx_default_checkpoints: CudaFunction,
    ap_raf_keys: CudaFunction,
    ap_table_keys: CudaFunction,
    ap_histogram: CudaFunction,
    ap_scatter: CudaFunction,
    ap_raf_reduce_chunked: CudaFunction,
    ap_suffix_reduce: CudaFunction,
    ap_scale_shift: CudaFunction,
    ap_condense: CudaFunction,
    ap_raf_prefix: CudaFunction,
    ap_bind_lanes: CudaFunction,
    ap_round_message_hinted: CudaFunction,
    ap_combined_val: CudaFunction,
    ap_ra: CudaFunction,
    ap_flag_keys: CudaFunction,
    ap_flag_sums: CudaFunction,
    ap_raf_flag_sum: CudaFunction,
    unr_mul_scatter: CudaFunction,
    unr_reduce: CudaFunction,
    unr_fold_chunks: CudaFunction,
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
    irv_eq_double: CudaFunction,
    irv_tables_split: CudaFunction,
    irv_gather: CudaFunction,
    irv_message_sparse: CudaFunction,
    irv_message_dense: CudaFunction,
    rrv_gather: CudaFunction,
    rrv_message_sparse: CudaFunction,
    rrv_message_dense: CudaFunction,
    brc_tables_init: CudaFunction,
    brc_gather: CudaFunction,
    brc_message_sparse: CudaFunction,
    brc_message_dense: CudaFunction,
    brr_coefficient: CudaFunction,
    brr_gather: CudaFunction,
    brr_message_sparse: CudaFunction,
    brr_message_dense: CudaFunction,
    so_gather: CudaFunction,
    so_shift: CudaFunction,
    so_uniskip: CudaFunction,
    so_factors: CudaFunction,
    gruen_pair_message: CudaFunction,
    so_claims: CudaFunction,
    sop_round: CudaFunction,
    msm_fq_add: CudaFunction,
    msm_fq_sub: CudaFunction,
    msm_fq_mul: CudaFunction,
    msm_fq_batch_inverse: CudaFunction,
    msm_g1_double: CudaFunction,
    msm_g1_add: CudaFunction,
    msm_g1_add_affine: CudaFunction,
    msm_affine_denominators: CudaFunction,
    msm_affine_combine: CudaFunction,
    msm_from_montgomery: CudaFunction,
    msm_digits: CudaFunction,
    msm_bucket_count: CudaFunction,
    msm_bucket_scatter: CudaFunction,
    commit_increment_column: CudaFunction,
    msm_one_hot_count: CudaFunction,
    msm_one_hot_count_shared: CudaFunction,
    msm_one_hot_scatter: CudaFunction,
    msm_one_hot_scatter_shared: CudaFunction,
    msm_segment_sum: CudaFunction,
    msm_segment_sum_warp: CudaFunction,
    msm_segment_sum_classed: CudaFunction,
    msm_segment_sum_small: CudaFunction,
    msm_bucket_reduce_chunked: CudaFunction,
    msm_point_rows_sum: CudaFunction,
    msm_window_fold: CudaFunction,
    msm_g2_segment_sum_small: CudaFunction,
    msm_g2_bucket_reduce_chunked: CudaFunction,
    msm_g2_point_rows_sum: CudaFunction,
    msm_g2_window_fold: CudaFunction,
    msm_glv_decompose_4d: CudaFunction,
    msm_g2_frobenius: CudaFunction,
    msm_g2_axpy_glv: CudaFunction,
    msm_shared_scalar_rows_glv: CudaFunction,
    msm_glv_decompose_2d: CudaFunction,
    msm_g1_endomorphism: CudaFunction,
    msm_jacobian_z: CudaFunction,
    msm_jacobian_to_affine: CudaFunction,
    msm_g1_axpy: CudaFunction,
    msm_g2_fixed_base: CudaFunction,
    pairing_miller: CudaFunction,
    pairing_miller_warp: CudaFunction,
    pairing_fq12_product: CudaFunction,
    msm_window_accumulate: CudaFunction,
    msm_block_embed: CudaFunction,
    msm_scatter_strided: CudaFunction,
    msm_scatter_one_hot: CudaFunction,
    msm_fold_rows: CudaFunction,
    opening_one_hot_embed: CudaFunction,
    opening_one_hot_fold: CudaFunction,
    pcr_round: CudaFunction,
    pcr_scatter: CudaFunction,
    pcr_value_fold: CudaFunction,
    pcr_lane_eq: CudaFunction,
    pcr_shift_eq: CudaFunction,
    pcr_place_row: CudaFunction,
    hf_half_fold: CudaFunction,
    hf_row_fold: CudaFunction,
    hf_bind_low_to_high: CudaFunction,
    sopg_round: CudaFunction,
    ii_flag_words: CudaFunction,
    ss_packed_columns: CudaFunction,
    bap_bind_squared: CudaFunction,
    bap_message: CudaFunction,
    brap_one_hot: CudaFunction,
    brap_term: CudaFunction,
    brap_message: CudaFunction,
    sp_gather: CudaFunction,
    sp_matrix: CudaFunction,
    sp_factors: CudaFunction,
    sp_claims: CudaFunction,
    ohf_fold: CudaFunction,
    ohf_reduce: CudaFunction,
    ohf_affine: CudaFunction,
    hwr_weights: CudaFunction,
    hwr_message: CudaFunction,
    roc_message: CudaFunction,
}

impl CudaKernelContext {
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        let context = DriverContext::new(ordinal)?;
        let stream = context.default_stream();
        let module = context.load_module(Ptx::from_binary(KERNEL_CUBIN.to_vec()))?;
        Ok(Self {
            ordinal,
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
            u128_to_mont: module.load_function("u128_to_mont_kernel")?,
            twos_i128_to_mont: module.load_function("twos_i128_to_mont_kernel")?,
            eq_double: module.load_function("eq_double_kernel")?,
            lt_double: module.load_function("lt_double_kernel")?,
            scan_u32_block: module.load_function("scan_u32_block_kernel")?,
            scan_u32_add_offsets: module.load_function("scan_u32_add_offsets_kernel")?,
            dense_product_round: module.load_function("dense_product_round_kernel")?,
            lane_sum_total: module.load_function("lane_sum_total_kernel")?,
            ra_split_tables: module.load_function("ra_split_tables_kernel")?,
            ra_gather: module.load_function("ra_gather_kernel")?,
            lt_reconstruct: module.load_function("lt_reconstruct_kernel")?,
            ram_ra_gather_h: module.load_function("ram_ra_gather_h_kernel")?,
            ram_ra_fold_suffix: module.load_function("ram_ra_fold_suffix_kernel")?,
            ram_ra_fold_prefix: module.load_function("ram_ra_fold_prefix_kernel")?,
            ram_ra_phase1_round: module.load_function("ram_ra_phase1_round_kernel")?,
            sfx_eval_batch: module.load_function("sfx_eval_batch_kernel")?,
            pfx_update_checkpoints: module.load_function("pfx_update_checkpoints_kernel")?,
            pfx_mle_round: module.load_function("pfx_mle_round_kernel")?,
            pfx_default_checkpoints: module.load_function("pfx_default_checkpoints_kernel")?,
            ap_raf_keys: module.load_function("ap_raf_keys_kernel")?,
            ap_table_keys: module.load_function("ap_table_keys_kernel")?,
            ap_histogram: module.load_function("ap_histogram_kernel")?,
            ap_scatter: module.load_function("ap_scatter_kernel")?,
            ap_raf_reduce_chunked: module.load_function("ap_raf_reduce_chunked_kernel")?,
            ap_suffix_reduce: module.load_function("ap_suffix_reduce_kernel")?,
            ap_scale_shift: module.load_function("ap_scale_shift_kernel")?,
            ap_condense: module.load_function("ap_condense_kernel")?,
            ap_raf_prefix: module.load_function("ap_raf_prefix_kernel")?,
            ap_bind_lanes: module.load_function("ap_bind_lanes_kernel")?,
            ap_round_message_hinted: module.load_function("ap_round_message_hinted_kernel")?,
            ap_combined_val: module.load_function("ap_combined_val_kernel")?,
            ap_ra: module.load_function("ap_ra_kernel")?,
            ap_flag_keys: module.load_function("ap_flag_keys_kernel")?,
            ap_flag_sums: module.load_function("ap_flag_sums_kernel")?,
            ap_raf_flag_sum: module.load_function("ap_raf_flag_sum_kernel")?,
            unr_mul_scatter: module.load_function("unr_mul_scatter_kernel")?,
            unr_reduce: module.load_function("unr_reduce_kernel")?,
            unr_fold_chunks: module.load_function("unr_fold_chunks_kernel")?,
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
            irv_eq_double: module.load_function("irv_eq_double_kernel")?,
            irv_tables_split: module.load_function("irv_tables_split_kernel")?,
            irv_gather: module.load_function("irv_gather_kernel")?,
            irv_message_sparse: module.load_function("irv_message_sparse_kernel")?,
            irv_message_dense: module.load_function("irv_message_dense_kernel")?,
            rrv_gather: module.load_function("rrv_gather_kernel")?,
            rrv_message_sparse: module.load_function("rrv_message_sparse_kernel")?,
            rrv_message_dense: module.load_function("rrv_message_dense_kernel")?,
            brc_tables_init: module.load_function("brc_tables_init_kernel")?,
            brc_gather: module.load_function("brc_gather_kernel")?,
            brc_message_sparse: module.load_function("brc_message_sparse_kernel")?,
            brc_message_dense: module.load_function("brc_message_dense_kernel")?,
            brr_coefficient: module.load_function("brr_coefficient_kernel")?,
            brr_gather: module.load_function("brr_gather_kernel")?,
            brr_message_sparse: module.load_function("brr_message_sparse_kernel")?,
            brr_message_dense: module.load_function("brr_message_dense_kernel")?,
            so_gather: module.load_function("so_gather_kernel")?,
            so_shift: module.load_function("so_shift_kernel")?,
            so_uniskip: module.load_function("so_uniskip_kernel")?,
            so_factors: module.load_function("so_factors_kernel")?,
            gruen_pair_message: module.load_function("gruen_pair_message_kernel")?,
            so_claims: module.load_function("so_claims_kernel")?,
            sop_round: module.load_function("sop_round_kernel")?,
            msm_fq_add: module.load_function("msm_fq_add_kernel")?,
            msm_fq_sub: module.load_function("msm_fq_sub_kernel")?,
            msm_fq_mul: module.load_function("msm_fq_mul_kernel")?,
            msm_fq_batch_inverse: module.load_function("msm_fq_batch_inverse_kernel")?,
            msm_g1_double: module.load_function("msm_g1_double_kernel")?,
            msm_g1_add: module.load_function("msm_g1_add_kernel")?,
            msm_g1_add_affine: module.load_function("msm_g1_add_affine_kernel")?,
            msm_affine_denominators: module.load_function("msm_affine_denominators_kernel")?,
            msm_affine_combine: module.load_function("msm_affine_combine_kernel")?,
            msm_from_montgomery: module.load_function("msm_from_montgomery_kernel")?,
            msm_digits: module.load_function("msm_digits_kernel")?,
            msm_bucket_count: module.load_function("msm_bucket_count_kernel")?,
            msm_bucket_scatter: module.load_function("msm_bucket_scatter_kernel")?,
            commit_increment_column: module.load_function("commit_increment_column_kernel")?,
            msm_one_hot_count: module.load_function("msm_one_hot_count_kernel")?,
            msm_one_hot_count_shared: module.load_function("msm_one_hot_count_shared_kernel")?,
            msm_one_hot_scatter: module.load_function("msm_one_hot_scatter_kernel")?,
            msm_one_hot_scatter_shared: module
                .load_function("msm_one_hot_scatter_shared_kernel")?,
            msm_segment_sum: module.load_function("msm_segment_sum_kernel")?,
            msm_segment_sum_warp: module.load_function("msm_segment_sum_warp_kernel")?,
            msm_segment_sum_classed: module.load_function("msm_segment_sum_classed_kernel")?,
            msm_segment_sum_small: module.load_function("msm_segment_sum_small_kernel")?,
            msm_bucket_reduce_chunked: module.load_function("msm_bucket_reduce_chunked_kernel")?,
            msm_point_rows_sum: module.load_function("msm_point_rows_sum_kernel")?,
            msm_window_fold: module.load_function("msm_window_fold_kernel")?,
            msm_g2_segment_sum_small: module.load_function("msm_g2_segment_sum_small_kernel")?,
            msm_g2_bucket_reduce_chunked: module
                .load_function("msm_g2_bucket_reduce_chunked_kernel")?,
            msm_g2_point_rows_sum: module.load_function("msm_g2_point_rows_sum_kernel")?,
            msm_g2_window_fold: module.load_function("msm_g2_window_fold_kernel")?,
            msm_glv_decompose_4d: module.load_function("msm_glv_decompose_4d_kernel")?,
            msm_g2_frobenius: module.load_function("msm_g2_frobenius_kernel")?,
            msm_g2_axpy_glv: module.load_function("msm_g2_axpy_glv_kernel")?,
            msm_shared_scalar_rows_glv: module
                .load_function("msm_shared_scalar_rows_glv_kernel")?,
            msm_glv_decompose_2d: module.load_function("msm_glv_decompose_2d_kernel")?,
            msm_g1_endomorphism: module.load_function("msm_g1_endomorphism_kernel")?,
            msm_jacobian_z: module.load_function("msm_jacobian_z_kernel")?,
            msm_jacobian_to_affine: module.load_function("msm_jacobian_to_affine_kernel")?,
            msm_g1_axpy: module.load_function("msm_g1_axpy_kernel")?,
            msm_g2_fixed_base: module.load_function("msm_g2_fixed_base_kernel")?,
            pairing_miller: module.load_function("pairing_miller_kernel")?,
            pairing_miller_warp: module.load_function("pairing_miller_warp_kernel")?,
            pairing_fq12_product: module.load_function("pairing_fq12_product_kernel")?,
            msm_window_accumulate: module.load_function("msm_window_accumulate_kernel")?,
            msm_block_embed: module.load_function("msm_block_embed_kernel")?,
            msm_scatter_strided: module.load_function("msm_scatter_strided_kernel")?,
            msm_scatter_one_hot: module.load_function("msm_scatter_one_hot_kernel")?,
            msm_fold_rows: module.load_function("msm_fold_rows_kernel")?,
            opening_one_hot_embed: module.load_function("opening_one_hot_embed_kernel")?,
            opening_one_hot_fold: module.load_function("opening_one_hot_fold_kernel")?,
            pcr_round: module.load_function("pcr_round_kernel")?,
            pcr_scatter: module.load_function("pcr_scatter_kernel")?,
            pcr_value_fold: module.load_function("pcr_value_fold_kernel")?,
            pcr_lane_eq: module.load_function("pcr_lane_eq_kernel")?,
            pcr_shift_eq: module.load_function("pcr_shift_eq_kernel")?,
            pcr_place_row: module.load_function("pcr_place_row_kernel")?,
            hf_half_fold: module.load_function("hf_half_fold_kernel")?,
            hf_row_fold: module.load_function("hf_row_fold_kernel")?,
            hf_bind_low_to_high: module.load_function("hf_bind_low_to_high_kernel")?,
            sopg_round: module.load_function("sopg_round_kernel")?,
            ii_flag_words: module.load_function("ii_flag_words_kernel")?,
            ss_packed_columns: module.load_function("ss_packed_columns_kernel")?,
            bap_bind_squared: module.load_function("bap_bind_squared_kernel")?,
            bap_message: module.load_function("bap_message_kernel")?,
            brap_one_hot: module.load_function("brap_one_hot_kernel")?,
            brap_term: module.load_function("brap_term_kernel")?,
            brap_message: module.load_function("brap_message_kernel")?,
            sp_gather: module.load_function("sp_gather_kernel")?,
            sp_matrix: module.load_function("sp_matrix_kernel")?,
            sp_factors: module.load_function("sp_factors_kernel")?,
            sp_claims: module.load_function("sp_claims_kernel")?,
            ohf_fold: module.load_function("ohf_fold_kernel")?,
            ohf_reduce: module.load_function("ohf_reduce_kernel")?,
            ohf_affine: module.load_function("ohf_affine_kernel")?,
            hwr_weights: module.load_function("hwr_weights_kernel")?,
            hwr_message: module.load_function("hwr_message_kernel")?,
            roc_message: module.load_function("roc_message_kernel")?,
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
            let mut host = vec![0u64; limbs];
            fill_staging(&mut host, values);
            Ok::<_, CudaError>(self.stream.clone_htod(&host)?)
        })?;
        Ok(DeviceFrVec::from_parts(
            self.stream.clone(),
            buffer,
            values.len(),
            self.staging.clone(),
        ))
    }

    pub(crate) fn upload_limbs(&self, limbs: &[u64]) -> Result<DeviceFrVec, CudaError> {
        if !limbs.len().is_multiple_of(LIMBS) {
            return Err(CudaError::LengthMismatch {
                expected: LIMBS,
                got: limbs.len() % LIMBS,
            });
        }
        let len = limbs.len() / LIMBS;
        if limbs.is_empty() {
            let buffer = self.stream.alloc_zeros::<u64>(0)?;
            return Ok(DeviceFrVec::from_parts(
                self.stream.clone(),
                buffer,
                0,
                self.staging.clone(),
            ));
        }
        let buffer = xfer_stats::timed(Phase::H2d, size_of_val(limbs), || {
            Ok::<_, CudaError>(self.stream.clone_htod(limbs)?)
        })?;
        Ok(DeviceFrVec::from_parts(
            self.stream.clone(),
            buffer,
            len,
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

    pub const fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn memory_used(&self) -> Result<(usize, usize), CudaError> {
        let (free, total) = self.stream.context().mem_get_info()?;
        Ok((total.saturating_sub(free), total))
    }

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

    pub(crate) fn download_u8(&self, buffer: &CudaSlice<u8>) -> Result<Vec<u8>, CudaError> {
        xfer_stats::timed(Phase::D2h, buffer.len(), || {
            Ok(self.stream.clone_dtoh(buffer)?)
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

    pub(crate) fn clone_u64(&self, buffer: &CudaSlice<u64>) -> Result<CudaSlice<u64>, CudaError> {
        xfer_stats::timed(Phase::D2d, buffer.len() * size_of::<u64>(), || {
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

    pub(crate) fn alloc_u8(&self, len: usize) -> Result<CudaSlice<u8>, CudaError> {
        Ok(self.stream.alloc_zeros::<u8>(len)?)
    }

    fn alloc_unset<T: cudarc::driver::DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<T>, CudaError> {
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

    pub(crate) fn alloc_u32_unset(&self, len: usize) -> Result<CudaSlice<u32>, CudaError> {
        self.alloc_unset::<u32>(len)
    }

    pub(crate) fn alloc_u8_unset(&self, len: usize) -> Result<CudaSlice<u8>, CudaError> {
        self.alloc_unset::<u8>(len)
    }

    pub(crate) fn alloc_u64_unset(&self, len: usize) -> Result<CudaSlice<u64>, CudaError> {
        self.alloc_unset::<u64>(len)
    }

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

    pub(crate) const fn dense_product_round(&self) -> &CudaFunction {
        &self.dense_product_round
    }

    pub(crate) const fn lane_sum_total(&self) -> &CudaFunction {
        &self.lane_sum_total
    }

    pub(crate) const fn irv_eq_double(&self) -> &CudaFunction {
        &self.irv_eq_double
    }

    pub(crate) const fn irv_tables_split(&self) -> &CudaFunction {
        &self.irv_tables_split
    }

    pub(crate) const fn irv_gather(&self) -> &CudaFunction {
        &self.irv_gather
    }

    pub(crate) const fn irv_message_sparse(&self) -> &CudaFunction {
        &self.irv_message_sparse
    }

    pub(crate) const fn irv_message_dense(&self) -> &CudaFunction {
        &self.irv_message_dense
    }

    pub(crate) const fn rrv_gather(&self) -> &CudaFunction {
        &self.rrv_gather
    }

    pub(crate) const fn rrv_message_sparse(&self) -> &CudaFunction {
        &self.rrv_message_sparse
    }

    pub(crate) const fn rrv_message_dense(&self) -> &CudaFunction {
        &self.rrv_message_dense
    }

    pub(crate) const fn brc_tables_init(&self) -> &CudaFunction {
        &self.brc_tables_init
    }

    pub(crate) const fn brc_gather(&self) -> &CudaFunction {
        &self.brc_gather
    }

    pub(crate) const fn brc_message_sparse(&self) -> &CudaFunction {
        &self.brc_message_sparse
    }

    pub(crate) const fn brr_coefficient(&self) -> &CudaFunction {
        &self.brr_coefficient
    }

    pub(crate) const fn brr_gather(&self) -> &CudaFunction {
        &self.brr_gather
    }

    pub(crate) const fn brr_message_sparse(&self) -> &CudaFunction {
        &self.brr_message_sparse
    }

    pub(crate) const fn brr_message_dense(&self) -> &CudaFunction {
        &self.brr_message_dense
    }

    pub(crate) const fn so_gather(&self) -> &CudaFunction {
        &self.so_gather
    }

    pub(crate) const fn so_shift(&self) -> &CudaFunction {
        &self.so_shift
    }

    pub(crate) const fn so_uniskip(&self) -> &CudaFunction {
        &self.so_uniskip
    }

    pub(crate) const fn so_factors(&self) -> &CudaFunction {
        &self.so_factors
    }

    pub(crate) const fn gruen_pair_message(&self) -> &CudaFunction {
        &self.gruen_pair_message
    }

    pub(crate) const fn so_claims(&self) -> &CudaFunction {
        &self.so_claims
    }

    pub(crate) const fn ohf_fold(&self) -> &CudaFunction {
        &self.ohf_fold
    }

    pub(crate) const fn ohf_reduce(&self) -> &CudaFunction {
        &self.ohf_reduce
    }

    pub(crate) const fn ohf_affine(&self) -> &CudaFunction {
        &self.ohf_affine
    }

    pub(crate) const fn hwr_weights(&self) -> &CudaFunction {
        &self.hwr_weights
    }

    pub(crate) const fn hwr_message(&self) -> &CudaFunction {
        &self.hwr_message
    }

    pub(crate) const fn roc_message(&self) -> &CudaFunction {
        &self.roc_message
    }

    pub(crate) const fn sopg_round(&self) -> &CudaFunction {
        &self.sopg_round
    }

    pub(crate) const fn ii_flag_words(&self) -> &CudaFunction {
        &self.ii_flag_words
    }

    pub(crate) const fn ss_packed_columns(&self) -> &CudaFunction {
        &self.ss_packed_columns
    }

    pub(crate) const fn bap_bind_squared(&self) -> &CudaFunction {
        &self.bap_bind_squared
    }

    pub(crate) const fn bap_message(&self) -> &CudaFunction {
        &self.bap_message
    }

    pub(crate) const fn brap_one_hot(&self) -> &CudaFunction {
        &self.brap_one_hot
    }

    pub(crate) const fn brap_term(&self) -> &CudaFunction {
        &self.brap_term
    }

    pub(crate) const fn brap_message(&self) -> &CudaFunction {
        &self.brap_message
    }

    pub(crate) const fn sop_round(&self) -> &CudaFunction {
        &self.sop_round
    }

    pub(crate) const fn msm_fq_add(&self) -> &CudaFunction {
        &self.msm_fq_add
    }

    pub(crate) const fn msm_fq_sub(&self) -> &CudaFunction {
        &self.msm_fq_sub
    }

    pub(crate) const fn msm_fq_mul(&self) -> &CudaFunction {
        &self.msm_fq_mul
    }

    pub(crate) const fn msm_fq_batch_inverse(&self) -> &CudaFunction {
        &self.msm_fq_batch_inverse
    }

    pub(crate) const fn msm_g1_double(&self) -> &CudaFunction {
        &self.msm_g1_double
    }

    pub(crate) const fn msm_g1_add(&self) -> &CudaFunction {
        &self.msm_g1_add
    }

    pub(crate) const fn msm_g1_add_affine(&self) -> &CudaFunction {
        &self.msm_g1_add_affine
    }

    pub(crate) const fn msm_affine_denominators(&self) -> &CudaFunction {
        &self.msm_affine_denominators
    }

    pub(crate) const fn msm_affine_combine(&self) -> &CudaFunction {
        &self.msm_affine_combine
    }

    pub(crate) const fn msm_from_montgomery(&self) -> &CudaFunction {
        &self.msm_from_montgomery
    }

    pub(crate) const fn msm_digits(&self) -> &CudaFunction {
        &self.msm_digits
    }

    pub(crate) const fn msm_bucket_count(&self) -> &CudaFunction {
        &self.msm_bucket_count
    }

    pub(crate) const fn msm_bucket_scatter(&self) -> &CudaFunction {
        &self.msm_bucket_scatter
    }

    pub(crate) const fn commit_increment_column(&self) -> &CudaFunction {
        &self.commit_increment_column
    }

    pub(crate) const fn msm_one_hot_count(&self) -> &CudaFunction {
        &self.msm_one_hot_count
    }

    pub(crate) const fn msm_one_hot_count_shared(&self) -> &CudaFunction {
        &self.msm_one_hot_count_shared
    }

    pub(crate) const fn msm_one_hot_scatter(&self) -> &CudaFunction {
        &self.msm_one_hot_scatter
    }

    pub(crate) const fn msm_one_hot_scatter_shared(&self) -> &CudaFunction {
        &self.msm_one_hot_scatter_shared
    }

    pub(crate) const fn msm_segment_sum(&self) -> &CudaFunction {
        &self.msm_segment_sum
    }

    pub(crate) const fn msm_segment_sum_warp(&self) -> &CudaFunction {
        &self.msm_segment_sum_warp
    }

    pub(crate) const fn msm_segment_sum_classed(&self) -> &CudaFunction {
        &self.msm_segment_sum_classed
    }

    pub(crate) const fn msm_segment_sum_small(&self) -> &CudaFunction {
        &self.msm_segment_sum_small
    }

    pub(crate) const fn msm_bucket_reduce_chunked(&self) -> &CudaFunction {
        &self.msm_bucket_reduce_chunked
    }

    pub(crate) const fn msm_point_rows_sum(&self) -> &CudaFunction {
        &self.msm_point_rows_sum
    }

    pub(crate) const fn msm_window_fold(&self) -> &CudaFunction {
        &self.msm_window_fold
    }

    pub(crate) const fn msm_g2_segment_sum_small(&self) -> &CudaFunction {
        &self.msm_g2_segment_sum_small
    }

    pub(crate) const fn msm_g2_bucket_reduce_chunked(&self) -> &CudaFunction {
        &self.msm_g2_bucket_reduce_chunked
    }

    pub(crate) const fn msm_g2_point_rows_sum(&self) -> &CudaFunction {
        &self.msm_g2_point_rows_sum
    }

    pub(crate) const fn msm_g2_window_fold(&self) -> &CudaFunction {
        &self.msm_g2_window_fold
    }

    pub(crate) const fn msm_glv_decompose_4d(&self) -> &CudaFunction {
        &self.msm_glv_decompose_4d
    }

    pub(crate) const fn msm_g2_frobenius(&self) -> &CudaFunction {
        &self.msm_g2_frobenius
    }

    pub(crate) const fn msm_g2_axpy_glv(&self) -> &CudaFunction {
        &self.msm_g2_axpy_glv
    }

    pub(crate) const fn msm_shared_scalar_rows_glv(&self) -> &CudaFunction {
        &self.msm_shared_scalar_rows_glv
    }

    pub(crate) const fn msm_glv_decompose_2d(&self) -> &CudaFunction {
        &self.msm_glv_decompose_2d
    }

    pub(crate) const fn msm_g1_endomorphism(&self) -> &CudaFunction {
        &self.msm_g1_endomorphism
    }

    pub(crate) const fn msm_jacobian_z(&self) -> &CudaFunction {
        &self.msm_jacobian_z
    }

    pub(crate) const fn msm_jacobian_to_affine(&self) -> &CudaFunction {
        &self.msm_jacobian_to_affine
    }

    pub(crate) const fn msm_g1_axpy(&self) -> &CudaFunction {
        &self.msm_g1_axpy
    }

    pub(crate) const fn msm_g2_fixed_base(&self) -> &CudaFunction {
        &self.msm_g2_fixed_base
    }

    pub(crate) const fn pairing_miller(&self) -> &CudaFunction {
        &self.pairing_miller
    }

    pub(crate) const fn pairing_miller_warp(&self) -> &CudaFunction {
        &self.pairing_miller_warp
    }

    pub(crate) const fn pairing_fq12_product(&self) -> &CudaFunction {
        &self.pairing_fq12_product
    }

    pub(crate) const fn msm_window_accumulate(&self) -> &CudaFunction {
        &self.msm_window_accumulate
    }

    pub(crate) const fn msm_block_embed(&self) -> &CudaFunction {
        &self.msm_block_embed
    }

    pub(crate) const fn msm_scatter_strided(&self) -> &CudaFunction {
        &self.msm_scatter_strided
    }

    pub(crate) const fn msm_scatter_one_hot(&self) -> &CudaFunction {
        &self.msm_scatter_one_hot
    }

    pub(crate) const fn msm_fold_rows(&self) -> &CudaFunction {
        &self.msm_fold_rows
    }

    pub(crate) const fn opening_one_hot_embed(&self) -> &CudaFunction {
        &self.opening_one_hot_embed
    }

    pub(crate) const fn opening_one_hot_fold(&self) -> &CudaFunction {
        &self.opening_one_hot_fold
    }

    pub(crate) const fn pcr_round(&self) -> &CudaFunction {
        &self.pcr_round
    }

    pub(crate) const fn pcr_scatter(&self) -> &CudaFunction {
        &self.pcr_scatter
    }

    pub(crate) const fn pcr_value_fold(&self) -> &CudaFunction {
        &self.pcr_value_fold
    }

    pub(crate) const fn pcr_lane_eq(&self) -> &CudaFunction {
        &self.pcr_lane_eq
    }

    pub(crate) const fn pcr_shift_eq(&self) -> &CudaFunction {
        &self.pcr_shift_eq
    }

    pub(crate) const fn pcr_place_row(&self) -> &CudaFunction {
        &self.pcr_place_row
    }

    pub(crate) const fn hf_half_fold(&self) -> &CudaFunction {
        &self.hf_half_fold
    }

    pub(crate) const fn hf_row_fold(&self) -> &CudaFunction {
        &self.hf_row_fold
    }

    pub(crate) const fn hf_bind_low_to_high(&self) -> &CudaFunction {
        &self.hf_bind_low_to_high
    }

    pub(crate) const fn sp_gather(&self) -> &CudaFunction {
        &self.sp_gather
    }

    pub(crate) const fn sp_matrix(&self) -> &CudaFunction {
        &self.sp_matrix
    }

    pub(crate) const fn sp_factors(&self) -> &CudaFunction {
        &self.sp_factors
    }

    pub(crate) const fn sp_claims(&self) -> &CudaFunction {
        &self.sp_claims
    }

    pub(crate) const fn brc_message_dense(&self) -> &CudaFunction {
        &self.brc_message_dense
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

    pub(crate) const fn sfx_eval_batch(&self) -> &CudaFunction {
        &self.sfx_eval_batch
    }

    pub(crate) const fn pfx_mle_round(&self) -> &CudaFunction {
        &self.pfx_mle_round
    }

    pub(crate) const fn pfx_update_checkpoints(&self) -> &CudaFunction {
        &self.pfx_update_checkpoints
    }

    pub(crate) const fn pfx_default_checkpoints(&self) -> &CudaFunction {
        &self.pfx_default_checkpoints
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

    pub(crate) const fn ap_raf_reduce_chunked(&self) -> &CudaFunction {
        &self.ap_raf_reduce_chunked
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

    pub(crate) const fn ap_bind_lanes(&self) -> &CudaFunction {
        &self.ap_bind_lanes
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

    pub(crate) const fn unr_fold_chunks(&self) -> &CudaFunction {
        &self.unr_fold_chunks
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

    pub(crate) fn device_columns(
        &self,
        columns: &[super::half_fold::FoldColumn<'_>],
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
                            "CUDA unavailable, falling back to the reference backend: {error}"
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
