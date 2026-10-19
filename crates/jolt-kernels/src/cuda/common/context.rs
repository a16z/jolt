use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

#[cfg(test)]
use cudarc::driver::DevicePtr;
use cudarc::driver::{
    CudaContext as DriverContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use jolt_field::Fr;

use super::device::{fill_staging, DeviceFrVec, LIMBS};
use super::error::CudaError;
#[cfg(test)]
use super::half_fold::FoldColumn;
use super::xfer_stats::{self, Phase};

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
    #[cfg(test)]
    dense_product_round: CudaFunction,
    #[cfg(test)]
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
            #[cfg(test)]
            dense_product_round: module.load_function("dense_product_round_kernel")?,
            #[cfg(test)]
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

    #[cfg(test)]
    pub(crate) const fn dense_product_round(&self) -> &CudaFunction {
        &self.dense_product_round
    }

    #[cfg(test)]
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
