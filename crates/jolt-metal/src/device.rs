use std::ffi::c_void;

use jolt_compute::{BindingOrder, ComputeBackend, Scalar};
use jolt_field::Field;
use metal::{Device, MTLResourceOptions, MTLSize};

use crate::buffer::MetalBuffer;
use crate::kernel::MetalKernel;
use crate::shaders::{ElementwiseKernels, InterpolationKernels};

/// Must match `SUM_GROUP_SIZE` in elementwise.metal.
const REDUCTION_GROUP_SIZE: usize = 256;

/// Maximum threadgroups for reduction dispatches.
/// More groups improve GPU utilization; diminishing returns past ~256 on M-series.
const MAX_REDUCTION_GROUPS: usize = 256;

/// Apple Metal compute backend for Jolt.
///
/// Wraps a `MTLDevice`, command queue, and pre-compiled shader pipelines.
/// All buffer allocations use shared memory mode on Apple Silicon (unified
/// memory architecture), avoiding explicit copies between CPU and Metal
/// address spaces.
///
/// # Field Type Assumption
///
/// The compiled shaders are specialized for BN254 Fr (32-byte Montgomery
/// form). Field operations assert `size_of::<F>() == 32` at runtime. On
/// little-endian ARM64, the CPU's `[u64; 4]` and Metal's `[u32; 8]`
/// representations have identical byte layout, so raw uploads work without
/// conversion.
pub struct MetalBackend {
    device: Device,
    queue: metal::CommandQueue,
    elementwise: ElementwiseKernels,
    interpolation: InterpolationKernels,
}

impl MetalBackend {
    /// Create a backend using the system default Metal device.
    ///
    /// Compiles all shader pipelines on creation.
    ///
    /// # Panics
    ///
    /// Panics if no Metal-capable device is available.
    pub fn new() -> Self {
        let device = Device::system_default().expect("no Metal device available");
        let queue = device.new_command_queue();
        let elementwise = ElementwiseKernels::compile(&device);
        let interpolation = InterpolationKernels::compile(&device);
        Self {
            device,
            queue,
            elementwise,
            interpolation,
        }
    }

    /// The underlying Metal device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The command queue used for dispatches.
    pub fn queue(&self) -> &metal::CommandQueue {
        &self.queue
    }

    /// Upload a single scalar value to a 1-element Metal buffer.
    fn upload_scalar<F>(&self, scalar: &F) -> metal::Buffer {
        self.device.new_buffer_with_data(
            std::ptr::from_ref(scalar).cast::<c_void>(),
            std::mem::size_of::<F>() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Upload a small `[u32]` parameter array to a Metal buffer.
    fn upload_params(&self, params: &[u32]) -> metal::Buffer {
        self.device.new_buffer_with_data(
            params.as_ptr().cast::<c_void>(),
            std::mem::size_of_val(params) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Dispatch a 1D compute kernel over `n` threads, then wait for completion.
    fn dispatch_1d(
        &self,
        pipeline: &metal::ComputePipelineState,
        buffers: &[&metal::Buffer],
        n: usize,
    ) {
        if n == 0 {
            return;
        }
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        for (i, buf) in buffers.iter().enumerate() {
            enc.set_buffer(i as u64, Some(buf), 0);
        }
        let tpg = pipeline.max_total_threads_per_threadgroup().min(n as u64);
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(tpg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Dispatch a parallel reduction and finish partial sums on CPU.
    ///
    /// `input_buffers` are bound at indices `0..k`, then the partials buffer
    /// at index `k` and the params buffer at index `k+1`, matching the MSL
    /// shader buffer layout for `fr_sum_kernel` and `fr_dot_product_kernel`.
    fn reduce<F: Field>(
        &self,
        pipeline: &metal::ComputePipelineState,
        input_buffers: &[&metal::Buffer],
        n: usize,
    ) -> F {
        debug_assert_eq!(
            std::mem::size_of::<F>(),
            32,
            "Metal backend requires BN254 Fr (32 bytes)"
        );
        if n == 0 {
            return F::zero();
        }

        let num_groups = n.div_ceil(REDUCTION_GROUP_SIZE).min(MAX_REDUCTION_GROUPS);

        let partials_buf = self.device.new_buffer(
            (num_groups * std::mem::size_of::<F>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params_buf = self.upload_params(&[n as u32, num_groups as u32]);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);

        for (i, buf) in input_buffers.iter().enumerate() {
            enc.set_buffer(i as u64, Some(buf), 0);
        }
        let k = input_buffers.len() as u64;
        enc.set_buffer(k, Some(&partials_buf), 0);
        enc.set_buffer(k + 1, Some(&params_buf), 0);

        enc.dispatch_thread_groups(
            MTLSize::new(num_groups as u64, 1, 1),
            MTLSize::new(REDUCTION_GROUP_SIZE as u64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // SAFETY: shared memory is coherent after command buffer completion.
        unsafe {
            let ptr = partials_buf.contents().cast::<F>();
            let partials = std::slice::from_raw_parts(ptr, num_groups);
            partials.iter().copied().fold(F::zero(), |acc, x| acc + x)
        }
    }
}

impl Default for MetalBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// SAFETY: Metal device and command queue are thread-safe. MTLDevice is
/// explicitly documented as safe to use from multiple threads. Command
/// buffers created from a queue are independent.
unsafe impl Send for MetalBackend {}
/// SAFETY: See above — MTLDevice and MTLCommandQueue are documented as
/// safe for concurrent use.
unsafe impl Sync for MetalBackend {}

impl ComputeBackend for MetalBackend {
    type Buffer<T: Scalar> = MetalBuffer<T>;
    type CompiledKernel<F: Field> = MetalKernel<F>;

    fn upload<T: Scalar>(&self, data: &[T]) -> Self::Buffer<T> {
        MetalBuffer::from_data(&self.device, data)
    }

    fn download<T: Scalar>(&self, buf: &Self::Buffer<T>) -> Vec<T> {
        buf.to_vec()
    }

    fn alloc<T: Scalar>(&self, len: usize) -> Self::Buffer<T> {
        MetalBuffer::zeroed(&self.device, len)
    }

    fn len<T: Scalar>(&self, buf: &Self::Buffer<T>) -> usize {
        buf.len()
    }

    fn sum<F: Field>(&self, buf: &Self::Buffer<F>) -> F {
        self.reduce(&self.elementwise.sum, &[&buf.raw], buf.len())
    }

    fn dot_product<F: Field>(&self, a: &Self::Buffer<F>, b: &Self::Buffer<F>) -> F {
        debug_assert_eq!(a.len(), b.len());
        self.reduce(&self.elementwise.dot_product, &[&a.raw, &b.raw], a.len())
    }

    fn scale<F: Field>(&self, buf: &mut Self::Buffer<F>, scalar: F) {
        let scalar_buf = self.upload_scalar(&scalar);
        self.dispatch_1d(&self.elementwise.scale, &[&buf.raw, &scalar_buf], buf.len());
    }

    fn add<F: Field>(&self, a: &Self::Buffer<F>, b: &Self::Buffer<F>) -> Self::Buffer<F> {
        debug_assert_eq!(a.len(), b.len());
        let out: MetalBuffer<F> = MetalBuffer::zeroed(&self.device, a.len());
        self.dispatch_1d(&self.elementwise.add, &[&a.raw, &b.raw, &out.raw], a.len());
        out
    }

    fn sub<F: Field>(&self, a: &Self::Buffer<F>, b: &Self::Buffer<F>) -> Self::Buffer<F> {
        debug_assert_eq!(a.len(), b.len());
        let out: MetalBuffer<F> = MetalBuffer::zeroed(&self.device, a.len());
        self.dispatch_1d(&self.elementwise.sub, &[&a.raw, &b.raw, &out.raw], a.len());
        out
    }

    fn accumulate<F: Field>(&self, buf: &mut Self::Buffer<F>, scalar: F, other: &Self::Buffer<F>) {
        debug_assert_eq!(buf.len(), other.len());
        let scalar_buf = self.upload_scalar(&scalar);
        self.dispatch_1d(
            &self.elementwise.accumulate,
            &[&buf.raw, &scalar_buf, &other.raw],
            buf.len(),
        );
    }

    fn interpolate_pairs<T, F>(&self, buf: Self::Buffer<T>, scalar: F) -> Self::Buffer<F>
    where
        T: Scalar,
        F: Field + From<T>,
    {
        let n = buf.len();
        debug_assert!(n % 2 == 0 || n == 0);
        let half = n / 2;

        if std::mem::size_of::<T>() == std::mem::size_of::<F>()
            && std::mem::size_of::<F>() == 32
        {
            // T and F are both 32-byte field elements — dispatch Metal kernel.
            let out: MetalBuffer<F> = MetalBuffer::zeroed(&self.device, half);
            let scalar_buf = self.upload_scalar(&scalar);
            self.dispatch_1d(
                &self.interpolation.interpolate_low,
                &[&buf.raw, &scalar_buf, &out.raw],
                half,
            );
            out
        } else {
            // Compact scalar type (u8, bool, etc.) — fall back to CPU conversion.
            let data = buf.to_vec();
            let result: Vec<F> = (0..half)
                .map(|i| {
                    let lo = F::from(data[2 * i]);
                    let hi = F::from(data[2 * i + 1]);
                    lo + scalar * (hi - lo)
                })
                .collect();
            MetalBuffer::from_data(&self.device, &result)
        }
    }

    fn interpolate_pairs_inplace<F: Field>(
        &self,
        buf: &mut Self::Buffer<F>,
        scalar: F,
        order: BindingOrder,
    ) {
        let n = buf.len();
        debug_assert!(n % 2 == 0 || n == 0);
        let half = n / 2;

        match order {
            BindingOrder::HighToLow => {
                // Safe in-place: thread i reads buf[i] and buf[i+half], writes buf[i].
                let scalar_buf = self.upload_scalar(&scalar);
                let params_buf = self.upload_params(&[half as u32]);
                self.dispatch_1d(
                    &self.interpolation.interpolate_inplace_high,
                    &[&buf.raw, &scalar_buf, &params_buf],
                    half,
                );
                buf.len = half;
            }
            BindingOrder::LowToHigh => {
                // NOT safe in-place (thread i reads buf[2i+1], thread i-1 writes buf[i]).
                // Dispatch to a separate output buffer and replace.
                let out: MetalBuffer<F> = MetalBuffer::zeroed(&self.device, half);
                let scalar_buf = self.upload_scalar(&scalar);
                self.dispatch_1d(
                    &self.interpolation.interpolate_low,
                    &[&buf.raw, &scalar_buf, &out.raw],
                    half,
                );
                *buf = out;
            }
        }
    }

    fn product_table<F: Field>(&self, point: &[F]) -> Self::Buffer<F> {
        let n = point.len();
        let table_size = 1usize << n;

        // Initialize on CPU: table[0] = one, rest unused (will be overwritten
        // by subsequent rounds before being read).
        let mut init: Vec<F> = Vec::with_capacity(table_size);
        init.push(F::one());
        // Fill the rest with zeros to ensure deterministic Metal buffer contents.
        init.resize(table_size, F::zero());
        let table = MetalBuffer::from_data(&self.device, &init);
        drop(init);

        // Each round doubles the active region: table[0..2^k] → table[0..2^(k+1)].
        for (k, r) in point.iter().enumerate() {
            let prev_len = 1usize << k;
            let one_minus_r = F::one() - *r;

            let r_buf = self.upload_scalar(r);
            let omr_buf = self.upload_scalar(&one_minus_r);
            let params_buf = self.upload_params(&[prev_len as u32]);

            self.dispatch_1d(
                &self.interpolation.product_table_round,
                &[&table.raw, &r_buf, &omr_buf, &params_buf],
                prev_len,
            );
        }

        table
    }

    fn pairwise_reduce<F: Field>(
        &self,
        _inputs: &[&Self::Buffer<F>],
        _weights: &Self::Buffer<F>,
        _kernel: &Self::CompiledKernel<F>,
        _num_evals: usize,
        _order: BindingOrder,
    ) -> Vec<F> {
        todo!()
    }

    fn tensor_pairwise_reduce<F: Field>(
        &self,
        _inputs: &[&Self::Buffer<F>],
        _outer_weights: &Self::Buffer<F>,
        _inner_weights: &Self::Buffer<F>,
        _kernel: &Self::CompiledKernel<F>,
        _num_evals: usize,
    ) -> Vec<F> {
        todo!()
    }
}
