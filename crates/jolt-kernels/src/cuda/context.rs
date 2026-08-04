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
    include_str!("kernels/prelude.cu"),
    "\n",
    include_str!("kernels/probe.cu"),
    "\n",
    include_str!("kernels/arith.cu"),
    "\n",
    include_str!("kernels/tables.cu"),
    "\n",
    include_str!("kernels/scan.cu"),
    "\n",
    include_str!("kernels/dense_product.cu"),
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

    pub(super) const fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub(super) fn launch_config(count: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (count.div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    pub(super) fn count_of(len: usize) -> Result<u32, CudaError> {
        u32::try_from(len).map_err(|_| CudaError::LengthMismatch {
            expected: u32::MAX as usize,
            got: len,
        })
    }

    pub(super) fn upload_u64_slice(&self, values: &[u64]) -> Result<CudaSlice<u64>, CudaError> {
        xfer_stats::timed(Phase::H2d, size_of_val(values), || {
            Ok(self.stream.clone_htod(values)?)
        })
    }

    pub(super) fn upload_u32_slice(&self, values: &[u32]) -> Result<CudaSlice<u32>, CudaError> {
        xfer_stats::timed(Phase::H2d, size_of_val(values), || {
            Ok(self.stream.clone_htod(values)?)
        })
    }

    pub(super) fn upload_u8_slice(&self, values: &[u8]) -> Result<CudaSlice<u8>, CudaError> {
        xfer_stats::timed(Phase::H2d, size_of_val(values), || {
            Ok(self.stream.clone_htod(values)?)
        })
    }

    pub(super) fn download_u32(&self, buffer: &CudaSlice<u32>) -> Result<Vec<u32>, CudaError> {
        xfer_stats::timed(Phase::D2h, buffer.len() * size_of::<u32>(), || {
            Ok(self.stream.clone_dtoh(buffer)?)
        })
    }

    pub(super) fn alloc_u32(&self, len: usize) -> Result<CudaSlice<u32>, CudaError> {
        Ok(self.stream.alloc_zeros::<u32>(len)?)
    }

    pub(super) const fn dense_product_round(&self) -> &CudaFunction {
        &self.dense_product_round
    }

    pub(super) const fn lane_sum_reduce(&self) -> &CudaFunction {
        &self.lane_sum_reduce
    }

    pub(super) const fn weighted_combine(&self) -> &CudaFunction {
        &self.weighted_combine
    }

    pub(super) fn device_pointers(
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
