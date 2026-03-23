//! Metal dispatch for field arithmetic over arbitrary limb counts.
//!
//! Compiles the MSL shaders at runtime and provides a typed dispatch API
//! for element-wise field operations. On little-endian ARM64, the CPU's
//! `[u64; N/2]` and Metal's `[u32; N]` Montgomery representations have
//! identical byte layout, so raw buffer uploads work without conversion.
//! [`MetalFieldElement`] provides explicit limb conversion for test and
//! benchmark code that constructs values outside the `ComputeBackend` path.

use std::ffi::c_void;

use metal::{ComputePipelineState, Device, MTLResourceOptions, MTLSize};

use crate::field_config::FieldConfig;
use crate::shaders::{build_source_with_preamble, make_pipeline};

/// N×u32 limbs matching the Metal Fr struct layout.
/// Both CPU and Metal store Montgomery-form values in little-endian limb order.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalFieldElement<const N: usize> {
    pub limbs: [u32; N],
}

impl<const N: usize> MetalFieldElement<N> {
    /// Convert from the CPU's (N/2)×u64 Montgomery limbs.
    ///
    /// # Panics
    /// Panics if `limbs.len() * 2 != N`.
    #[inline]
    pub fn from_u64_limbs(limbs: &[u64]) -> Self {
        assert_eq!(limbs.len() * 2, N, "expected {} u64 limbs, got {}", N / 2, limbs.len());
        let mut out = [0u32; N];
        for (i, &v) in limbs.iter().enumerate() {
            out[2 * i] = v as u32;
            out[2 * i + 1] = (v >> 32) as u32;
        }
        Self { limbs: out }
    }

    /// Convert back to (N/2)×u64 Montgomery limbs.
    #[inline]
    pub fn to_u64_limbs(self) -> Vec<u64> {
        assert_eq!(N % 2, 0, "N must be even for u64 conversion");
        (0..N / 2)
            .map(|i| u64::from(self.limbs[2 * i]) | (u64::from(self.limbs[2 * i + 1]) << 32))
            .collect()
    }
}

/// Compiled shader library with pre-built pipelines for test/benchmark kernels.
pub struct FrKernels {
    pub mul: ComputePipelineState,
    pub add: ComputePipelineState,
    pub sub: ComputePipelineState,
    pub sqr: ComputePipelineState,
    pub neg: ComputePipelineState,
    pub fmadd: ComputePipelineState,
    pub from_u64: ComputePipelineState,
}

impl FrKernels {
    /// Compile all Fr test/benchmark kernels from generated MSL source.
    pub fn new(device: &Device) -> Self {
        let field_config = FieldConfig::from_gpu_field::<jolt_field::Fr>();
        let source = build_source_with_preamble(
            &field_config.msl_preamble,
            &[&field_config.msl_test_kernels],
        );
        let options = metal::CompileOptions::new();
        let library = device
            .new_library_with_source(&source, &options)
            .expect("MSL compilation failed");

        Self {
            mul: make_pipeline(device, &library, "fr_mul_kernel"),
            add: make_pipeline(device, &library, "fr_add_kernel"),
            sub: make_pipeline(device, &library, "fr_sub_kernel"),
            sqr: make_pipeline(device, &library, "fr_sqr_kernel"),
            neg: make_pipeline(device, &library, "fr_neg_kernel"),
            fmadd: make_pipeline(device, &library, "fr_fmadd_kernel"),
            from_u64: make_pipeline(device, &library, "fr_from_u64_kernel"),
        }
    }
}

fn upload_slice<T>(device: &Device, data: &[T]) -> metal::Buffer {
    device.new_buffer_with_data(
        data.as_ptr().cast::<c_void>(),
        std::mem::size_of_val(data) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn alloc_buffer(device: &Device, byte_len: u64) -> metal::Buffer {
    device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared)
}

fn dispatch_and_wait(
    queue: &metal::CommandQueue,
    pipeline: &ComputePipelineState,
    buffers: &[&metal::Buffer],
    n: usize,
) {
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    for (i, buf) in buffers.iter().enumerate() {
        enc.set_buffer(i as u64, Some(buf), 0);
    }

    let threads_per_group = pipeline.max_total_threads_per_threadgroup().min(n as u64);
    enc.dispatch_threads(
        MTLSize::new(n as u64, 1, 1),
        MTLSize::new(threads_per_group, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

fn read_buffer<const N: usize>(buf: &metal::Buffer, n: usize) -> Vec<MetalFieldElement<N>> {
    // SAFETY: buffer was allocated with StorageModeShared and the command
    // buffer has completed, so the contents pointer is valid and coherent.
    unsafe {
        let ptr = buf.contents().cast::<MetalFieldElement<N>>();
        std::slice::from_raw_parts(ptr, n).to_vec()
    }
}

/// Dispatch a binary element-wise kernel (a op b → result).
pub fn dispatch_binary<const N: usize>(
    device: &Device,
    queue: &metal::CommandQueue,
    pipeline: &ComputePipelineState,
    a: &[MetalFieldElement<N>],
    b: &[MetalFieldElement<N>],
) -> Vec<MetalFieldElement<N>> {
    assert_eq!(a.len(), b.len());
    let n = a.len();

    let buf_a = upload_slice(device, a);
    let buf_b = upload_slice(device, b);
    let buf_out = alloc_buffer(device, std::mem::size_of_val(a) as u64);

    dispatch_and_wait(queue, pipeline, &[&buf_a, &buf_b, &buf_out], n);
    read_buffer(&buf_out, n)
}

/// Dispatch a unary element-wise kernel (a → result).
pub fn dispatch_unary<const N: usize>(
    device: &Device,
    queue: &metal::CommandQueue,
    pipeline: &ComputePipelineState,
    a: &[MetalFieldElement<N>],
) -> Vec<MetalFieldElement<N>> {
    let n = a.len();

    let buf_a = upload_slice(device, a);
    let buf_out = alloc_buffer(device, std::mem::size_of_val(a) as u64);

    dispatch_and_wait(queue, pipeline, &[&buf_a, &buf_out], n);
    read_buffer(&buf_out, n)
}

/// Dispatch the fmadd benchmark kernel.
///
/// Each of `n_threads` threads accumulates 256 products from `a` and `b`
/// (which have `stride` elements, indexed cyclically).
pub fn dispatch_fmadd<const N: usize>(
    device: &Device,
    queue: &metal::CommandQueue,
    pipeline: &ComputePipelineState,
    a: &[MetalFieldElement<N>],
    b: &[MetalFieldElement<N>],
    n_threads: usize,
) -> Vec<MetalFieldElement<N>> {
    assert_eq!(a.len(), b.len());
    let stride = a.len() as u32;

    let buf_a = upload_slice(device, a);
    let buf_b = upload_slice(device, b);
    let buf_out = alloc_buffer(device, (n_threads * std::mem::size_of::<MetalFieldElement<N>>()) as u64);
    let buf_params = upload_slice(device, &[stride]);

    dispatch_and_wait(
        queue,
        pipeline,
        &[&buf_a, &buf_b, &buf_out, &buf_params],
        n_threads,
    );
    read_buffer(&buf_out, n_threads)
}

/// Dispatch the fr_from_u64 conversion kernel.
pub fn dispatch_from_u64<const N: usize>(
    device: &Device,
    queue: &metal::CommandQueue,
    pipeline: &ComputePipelineState,
    vals: &[u64],
) -> Vec<MetalFieldElement<N>> {
    let n = vals.len();

    let buf_vals = upload_slice(device, vals);
    let buf_out = alloc_buffer(device, (n * std::mem::size_of::<MetalFieldElement<N>>()) as u64);

    dispatch_and_wait(queue, pipeline, &[&buf_vals, &buf_out], n);
    read_buffer(&buf_out, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_field_element_roundtrip() {
        let limbs = [
            0x0123_4567_89ab_cdef_u64,
            0xfedc_ba98_7654_3210,
            0x1111_2222_3333_4444,
            0x5555_6666_7777_8888,
        ];
        let metal = MetalFieldElement::<8>::from_u64_limbs(&limbs);
        assert_eq!(metal.to_u64_limbs(), limbs);
    }

    #[test]
    fn metal_field_element_4_limb_roundtrip() {
        let limbs = [0xdead_beef_cafe_babe_u64, 0x1234_5678_9abc_def0];
        let metal = MetalFieldElement::<4>::from_u64_limbs(&limbs);
        assert_eq!(metal.to_u64_limbs(), limbs);
    }

    #[test]
    fn shader_source_compiles() {
        let device = Device::system_default().expect("no Metal device");
        let _kernels = FrKernels::new(&device);
    }
}
