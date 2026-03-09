//! Metal dispatch for BN254 Fr field arithmetic.
//!
//! Compiles the MSL shaders at runtime and provides a typed dispatch API
//! for element-wise Fr operations. The dispatch layer handles limb
//! conversion between the CPU's 4×u64 Montgomery representation and
//! Metal's 8×u32 representation.

use std::ffi::c_void;

use metal::{ComputePipelineState, Device, MTLResourceOptions, MTLSize};

use crate::shaders::{
    build_source, make_pipeline, SHADER_BN254_FR, SHADER_TEST_KERNELS, SHADER_WIDE_ACC,
};

/// 8×u32 limbs matching the Metal Fr struct layout.
/// Both CPU and Metal store Montgomery-form values in little-endian limb order.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalFr {
    pub limbs: [u32; 8],
}

impl MetalFr {
    /// Convert from the CPU's 4×u64 Montgomery limbs.
    #[inline]
    pub fn from_u64_limbs(limbs: [u64; 4]) -> Self {
        Self {
            limbs: [
                limbs[0] as u32,
                (limbs[0] >> 32) as u32,
                limbs[1] as u32,
                (limbs[1] >> 32) as u32,
                limbs[2] as u32,
                (limbs[2] >> 32) as u32,
                limbs[3] as u32,
                (limbs[3] >> 32) as u32,
            ],
        }
    }

    /// Convert back to 4×u64 Montgomery limbs.
    #[inline]
    pub fn to_u64_limbs(self) -> [u64; 4] {
        [
            u64::from(self.limbs[0]) | (u64::from(self.limbs[1]) << 32),
            u64::from(self.limbs[2]) | (u64::from(self.limbs[3]) << 32),
            u64::from(self.limbs[4]) | (u64::from(self.limbs[5]) << 32),
            u64::from(self.limbs[6]) | (u64::from(self.limbs[7]) << 32),
        ]
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
    /// Compile all Fr test/benchmark kernels from MSL source.
    pub fn new(device: &Device) -> Self {
        let source = build_source(&[SHADER_BN254_FR, SHADER_WIDE_ACC, SHADER_TEST_KERNELS]);
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

fn read_buffer(buf: &metal::Buffer, n: usize) -> Vec<MetalFr> {
    // SAFETY: buffer was allocated with StorageModeShared and the command
    // buffer has completed, so the contents pointer is valid and coherent.
    unsafe {
        let ptr = buf.contents().cast::<MetalFr>();
        std::slice::from_raw_parts(ptr, n).to_vec()
    }
}

/// Dispatch a binary element-wise kernel (a op b → result).
pub fn dispatch_binary(
    device: &Device,
    queue: &metal::CommandQueue,
    pipeline: &ComputePipelineState,
    a: &[MetalFr],
    b: &[MetalFr],
) -> Vec<MetalFr> {
    assert_eq!(a.len(), b.len());
    let n = a.len();

    let buf_a = upload_slice(device, a);
    let buf_b = upload_slice(device, b);
    let buf_out = alloc_buffer(device, std::mem::size_of_val(a) as u64);

    dispatch_and_wait(queue, pipeline, &[&buf_a, &buf_b, &buf_out], n);
    read_buffer(&buf_out, n)
}

/// Dispatch a unary element-wise kernel (a → result).
pub fn dispatch_unary(
    device: &Device,
    queue: &metal::CommandQueue,
    pipeline: &ComputePipelineState,
    a: &[MetalFr],
) -> Vec<MetalFr> {
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
pub fn dispatch_fmadd(
    device: &Device,
    queue: &metal::CommandQueue,
    pipeline: &ComputePipelineState,
    a: &[MetalFr],
    b: &[MetalFr],
    n_threads: usize,
) -> Vec<MetalFr> {
    assert_eq!(a.len(), b.len());
    let stride = a.len() as u32;

    let buf_a = upload_slice(device, a);
    let buf_b = upload_slice(device, b);
    let buf_out = alloc_buffer(device, (n_threads * std::mem::size_of::<MetalFr>()) as u64);
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
pub fn dispatch_from_u64(
    device: &Device,
    queue: &metal::CommandQueue,
    pipeline: &ComputePipelineState,
    vals: &[u64],
) -> Vec<MetalFr> {
    let n = vals.len();

    let buf_vals = upload_slice(device, vals);
    let buf_out = alloc_buffer(device, (n * std::mem::size_of::<MetalFr>()) as u64);

    dispatch_and_wait(queue, pipeline, &[&buf_vals, &buf_out], n);
    read_buffer(&buf_out, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_fr_roundtrip() {
        let limbs = [
            0x0123_4567_89ab_cdef_u64,
            0xfedc_ba98_7654_3210,
            0x1111_2222_3333_4444,
            0x5555_6666_7777_8888,
        ];
        let metal = MetalFr::from_u64_limbs(limbs);
        assert_eq!(metal.to_u64_limbs(), limbs);
    }

    #[test]
    fn shader_source_compiles() {
        let device = Device::system_default().expect("no Metal device");
        let _kernels = FrKernels::new(&device);
    }
}
