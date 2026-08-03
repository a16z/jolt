use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext as DriverContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg,
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
);

pub struct CudaKernelContext {
    stream: Arc<CudaStream>,
    staging: StagingPool,
    fr_identity_probe: CudaFunction,
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
