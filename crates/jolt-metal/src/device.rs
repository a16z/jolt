use std::ffi::c_void;

use jolt_compute::{BindingOrder, ComputeBackend, Scalar};
use jolt_field::Field;
use jolt_ir::KernelDescriptor;
use metal::{Device, MTLResourceOptions, MTLSize};

use crate::buffer::MetalBuffer;
use crate::compiler::{self, CompileMode};
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
    compile_mode: CompileMode,
}

impl MetalBackend {
    /// Create a backend using the system default Metal device.
    ///
    /// Compiles all shader pipelines on creation with full LLVM inlining.
    ///
    /// # Panics
    ///
    /// Panics if no Metal-capable device is available.
    pub fn new() -> Self {
        Self::with_compile_mode(CompileMode::Performance)
    }

    /// Create a backend with fast shader compilation (noinline on field ops).
    ///
    /// Reduces kernel compilation time from minutes to seconds for large
    /// kernels (D=8+) at the cost of minor GPU runtime overhead.
    /// Ideal for tests where correctness matters but throughput doesn't.
    pub fn new_fast_compile() -> Self {
        Self::with_compile_mode(CompileMode::FastCompile)
    }

    /// Create a backend with an explicit compile mode.
    pub fn with_compile_mode(compile_mode: CompileMode) -> Self {
        let device = Device::system_default().expect("no Metal device available");
        let queue = device.new_command_queue();
        let elementwise = ElementwiseKernels::compile(&device);
        let interpolation = InterpolationKernels::compile(&device);
        Self {
            device,
            queue,
            elementwise,
            interpolation,
            compile_mode,
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

    /// Compile a `KernelDescriptor` into a Metal compute pipeline.
    pub fn compile_kernel<F: Field>(&self, descriptor: &KernelDescriptor) -> MetalKernel<F> {
        compiler::compile_with_mode(&self.device, descriptor, &[], self.compile_mode)
    }

    /// Compile with baked challenge values for Custom expression kernels.
    pub fn compile_kernel_with_challenges<F: Field>(
        &self,
        descriptor: &KernelDescriptor,
        challenges: &[F],
    ) -> MetalKernel<F> {
        compiler::compile_with_mode(&self.device, descriptor, challenges, self.compile_mode)
    }

    /// Dispatch a reduce kernel and finish partial sums on CPU.
    ///
    /// Generic over buffer layout: binds inputs at indices `0..K-1`, then
    /// `weight_buffers` at `K..K+W-1`, partials at `K+W`, params at `K+W+1`.
    /// This handles both standard (1 weight buffer) and tensor (2 weight buffers).
    fn dispatch_reduce<F: Field>(
        &self,
        pipeline: &metal::ComputePipelineState,
        num_evals: usize,
        inputs: &[&MetalBuffer<F>],
        weight_buffers: &[&metal::Buffer],
        params: &[u32],
        n_pairs: usize,
    ) -> Vec<F> {
        if n_pairs == 0 {
            return vec![F::zero(); num_evals];
        }

        let num_groups = n_pairs
            .div_ceil(compiler::REDUCE_GROUP_SIZE)
            .min(compiler::MAX_REDUCE_GROUPS);

        let partials_buf = self.device.new_buffer(
            (num_groups * num_evals * std::mem::size_of::<F>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params_buf = self.upload_params(params);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);

        for (i, buf) in inputs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(&buf.raw), 0);
        }
        let mut idx = inputs.len() as u64;
        for wbuf in weight_buffers {
            enc.set_buffer(idx, Some(wbuf), 0);
            idx += 1;
        }
        enc.set_buffer(idx, Some(&partials_buf), 0);
        enc.set_buffer(idx + 1, Some(&params_buf), 0);

        enc.dispatch_thread_groups(
            MTLSize::new(num_groups as u64, 1, 1),
            MTLSize::new(compiler::REDUCE_GROUP_SIZE as u64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // SAFETY: shared memory is coherent after command buffer completion.
        let partials: &[F] = unsafe {
            let ptr = partials_buf.contents().cast::<F>();
            std::slice::from_raw_parts(ptr, num_groups * num_evals)
        };

        // Sum partials per eval dimension
        let mut result = vec![F::zero(); num_evals];
        for g in 0..num_groups {
            for d in 0..num_evals {
                result[d] += partials[g * num_evals + d];
            }
        }
        result
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

    fn compile_kernel_with_challenges<F: Field>(
        &self,
        desc: &KernelDescriptor,
        challenges: &[F],
    ) -> MetalKernel<F> {
        compiler::compile_with_mode(&self.device, desc, challenges, self.compile_mode)
    }

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

        if std::mem::size_of::<T>() == std::mem::size_of::<F>() && std::mem::size_of::<F>() == 32 {
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
        // Must match CpuBackend's mixed ordering: interleaved for small rounds
        // (prev_len < 1024), split-half for large rounds (prev_len >= 1024).
        const PAR_THRESHOLD: usize = 1024;

        let n = point.len();
        let table_size = 1usize << n;

        let mut table: MetalBuffer<F> = MetalBuffer::zeroed(&self.device, table_size);

        // SAFETY: no Metal commands in flight — we just allocated the buffer.
        unsafe { table.as_mut_slice()[0] = F::one() };

        // CPU rounds: interleaved indexing on shared memory (zero-copy).
        let mut first_gpu_round = n;
        for (k, &r) in point.iter().enumerate() {
            let prev_len = 1usize << k;
            if prev_len >= PAR_THRESHOLD {
                first_gpu_round = k;
                break;
            }
            let one_minus_r = F::one() - r;
            // SAFETY: no Metal commands in flight between CPU rounds.
            let slice = unsafe { table.as_mut_slice() };
            for j in (0..prev_len).rev() {
                let base = slice[j];
                slice[2 * j] = base * one_minus_r;
                slice[2 * j + 1] = base * r;
            }
        }

        // GPU rounds: batched into a single command buffer. Metal guarantees
        // sequential execution with implicit memory barriers between compute
        // command encoders within one command buffer.
        if first_gpu_round < n {
            // Pre-upload scalar/params buffers (must outlive command buffer).
            let gpu_rounds = first_gpu_round..n;
            let r_bufs: Vec<_> = gpu_rounds
                .clone()
                .map(|k| self.upload_scalar(&point[k]))
                .collect();
            let params_bufs: Vec<_> = gpu_rounds
                .clone()
                .map(|k| self.upload_params(&[(1usize << k) as u32]))
                .collect();

            let cmd = self.queue.new_command_buffer();
            for (idx, k) in gpu_rounds.enumerate() {
                let prev_len = 1usize << k;
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.interpolation.product_table_round);
                enc.set_buffer(0, Some(&table.raw), 0);
                enc.set_buffer(1, Some(&r_bufs[idx]), 0);
                enc.set_buffer(2, Some(&params_bufs[idx]), 0);
                let tpg = self
                    .interpolation
                    .product_table_round
                    .max_total_threads_per_threadgroup()
                    .min(prev_len as u64);
                enc.dispatch_threads(MTLSize::new(prev_len as u64, 1, 1), MTLSize::new(tpg, 1, 1));
                enc.end_encoding();
            }
            cmd.commit();
            cmd.wait_until_completed();
        }

        table
    }

    fn pairwise_reduce<F: Field>(
        &self,
        inputs: &[&Self::Buffer<F>],
        weights: &Self::Buffer<F>,
        kernel: &Self::CompiledKernel<F>,
        _num_evals: usize,
        order: BindingOrder,
    ) -> Vec<F> {
        debug_assert!(!inputs.is_empty());
        let n = inputs[0].len();
        let n_pairs = n / 2;
        let pipeline = match order {
            BindingOrder::LowToHigh => &kernel.pipeline_l2h,
            BindingOrder::HighToLow => &kernel.pipeline_h2l,
        };
        self.dispatch_reduce(
            pipeline,
            kernel.num_evals,
            inputs,
            &[&weights.raw],
            &[n_pairs as u32],
            n_pairs,
        )
    }

    fn pairwise_reduce_unweighted<F: Field>(
        &self,
        inputs: &[&Self::Buffer<F>],
        kernel: &Self::CompiledKernel<F>,
        _num_evals: usize,
        order: BindingOrder,
    ) -> Vec<F> {
        debug_assert!(!inputs.is_empty());
        let n = inputs[0].len();
        let n_pairs = n / 2;
        let pipeline = match order {
            BindingOrder::LowToHigh => &kernel.pipeline_l2h_unw,
            BindingOrder::HighToLow => &kernel.pipeline_h2l_unw,
        };
        // Empty weight_buffers — unweighted kernels don't bind a weight buffer,
        // so partials lands at buffer(K) and params at buffer(K+1).
        self.dispatch_reduce(
            pipeline,
            kernel.num_evals,
            inputs,
            &[],
            &[n_pairs as u32],
            n_pairs,
        )
    }

    fn interpolate_pairs_batch<F: Field>(
        &self,
        bufs: Vec<Self::Buffer<F>>,
        scalar: F,
    ) -> Vec<Self::Buffer<F>> {
        if bufs.is_empty() {
            return vec![];
        }

        let scalar_buf = self.upload_scalar(&scalar);
        let cmd = self.queue.new_command_buffer();

        let outputs: Vec<MetalBuffer<F>> = bufs
            .iter()
            .map(|buf| {
                let n = buf.len();
                debug_assert!(n % 2 == 0 || n == 0);
                let half = n / 2;
                let out: MetalBuffer<F> = MetalBuffer::zeroed(&self.device, half);

                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.interpolation.interpolate_low);
                enc.set_buffer(0, Some(&buf.raw), 0);
                enc.set_buffer(1, Some(&scalar_buf), 0);
                enc.set_buffer(2, Some(&out.raw), 0);
                let tpg = self
                    .interpolation
                    .interpolate_low
                    .max_total_threads_per_threadgroup()
                    .min(half.max(1) as u64);
                enc.dispatch_threads(
                    MTLSize::new(half.max(1) as u64, 1, 1),
                    MTLSize::new(tpg, 1, 1),
                );
                enc.end_encoding();

                out
            })
            .collect();

        cmd.commit();
        cmd.wait_until_completed();

        outputs
    }

    fn interpolate_pairs_batch_inplace<F: Field>(
        &self,
        bufs: &mut [Self::Buffer<F>],
        scalar: F,
        order: BindingOrder,
    ) {
        if bufs.is_empty() {
            return;
        }

        let scalar_buf = self.upload_scalar(&scalar);
        let cmd = self.queue.new_command_buffer();

        match order {
            BindingOrder::HighToLow => {
                // Safe in-place: thread i reads buf[i] and buf[i+half], writes buf[i].
                let params_bufs: Vec<_> = bufs
                    .iter()
                    .map(|buf| self.upload_params(&[(buf.len() / 2) as u32]))
                    .collect();

                for (buf, params) in bufs.iter().zip(params_bufs.iter()) {
                    let half = buf.len() / 2;
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&self.interpolation.interpolate_inplace_high);
                    enc.set_buffer(0, Some(&buf.raw), 0);
                    enc.set_buffer(1, Some(&scalar_buf), 0);
                    enc.set_buffer(2, Some(params), 0);
                    let tpg = self
                        .interpolation
                        .interpolate_inplace_high
                        .max_total_threads_per_threadgroup()
                        .min(half.max(1) as u64);
                    enc.dispatch_threads(
                        MTLSize::new(half.max(1) as u64, 1, 1),
                        MTLSize::new(tpg, 1, 1),
                    );
                    enc.end_encoding();
                }

                cmd.commit();
                cmd.wait_until_completed();

                for buf in bufs.iter_mut() {
                    buf.len /= 2;
                }
            }
            BindingOrder::LowToHigh => {
                // NOT safe in-place — dispatch to separate output buffers.
                let outputs: Vec<MetalBuffer<F>> = bufs
                    .iter()
                    .map(|buf| {
                        let half = buf.len() / 2;
                        let out: MetalBuffer<F> = MetalBuffer::zeroed(&self.device, half);
                        let enc = cmd.new_compute_command_encoder();
                        enc.set_compute_pipeline_state(&self.interpolation.interpolate_low);
                        enc.set_buffer(0, Some(&buf.raw), 0);
                        enc.set_buffer(1, Some(&scalar_buf), 0);
                        enc.set_buffer(2, Some(&out.raw), 0);
                        let tpg = self
                            .interpolation
                            .interpolate_low
                            .max_total_threads_per_threadgroup()
                            .min(half.max(1) as u64);
                        enc.dispatch_threads(
                            MTLSize::new(half.max(1) as u64, 1, 1),
                            MTLSize::new(tpg, 1, 1),
                        );
                        enc.end_encoding();
                        out
                    })
                    .collect();

                cmd.commit();
                cmd.wait_until_completed();

                for (buf, out) in bufs.iter_mut().zip(outputs) {
                    *buf = out;
                }
            }
        }
    }

    fn tensor_pairwise_reduce<F: Field>(
        &self,
        inputs: &[&Self::Buffer<F>],
        outer_weights: &Self::Buffer<F>,
        inner_weights: &Self::Buffer<F>,
        kernel: &Self::CompiledKernel<F>,
        _num_evals: usize,
    ) -> Vec<F> {
        debug_assert!(!inputs.is_empty());
        let n = inputs[0].len();
        let n_pairs = n / 2;
        let inner_len = inner_weights.len();
        let inner_log = inner_len.trailing_zeros();
        let inner_mask = (inner_len - 1) as u32;
        self.dispatch_reduce(
            &kernel.pipeline_tensor,
            kernel.num_evals,
            inputs,
            &[&outer_weights.raw, &inner_weights.raw],
            &[n_pairs as u32, inner_log, inner_mask],
            n_pairs,
        )
    }
}
