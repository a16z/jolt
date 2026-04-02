use std::collections::HashMap;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use jolt_compiler::kernel_spec::Iteration;
use jolt_compiler::{Formula, KernelSpec};
use jolt_compute::{BindingOrder, Buf, ComputeBackend, Scalar};
use jolt_field::Field;
use metal::{Device, MTLResourceOptions, MTLSize};

use crate::buffer::MetalBuffer;
use crate::compiler::{self, CompileMode};
use crate::field_config::MslFieldParams;
use crate::kernel::{CachedPipelines, MetalKernel};
use crate::metal_device_config::MetalDeviceConfig;
use crate::shaders::InterpolationKernels;

/// Apple Metal compute backend for Jolt.
///
/// Wraps a `MTLDevice`, command queue, and pre-compiled shader pipelines.
/// All buffer allocations use shared memory mode on Apple Silicon (unified
/// memory architecture), avoiding explicit copies between CPU and Metal
/// address spaces.
///
/// Parameterized over field size via [`MslFieldParams`] and hardware via
/// [`MetalDeviceConfig`]. On little-endian ARM64, the CPU's `[u64; N/2]`
/// and Metal's `[u32; N]` Montgomery representations have identical byte layout.
pub struct MetalBackend {
    device: Device,
    queue: metal::CommandQueue,
    interpolation: InterpolationKernels,
    compile_mode: CompileMode,
    field_config: MslFieldParams,
    device_config: MetalDeviceConfig,
    pipeline_cache: Mutex<HashMap<u64, Arc<CachedPipelines>>>,
    /// Pre-allocated partials buffer for reduce dispatches.
    /// Sized for worst case: device_config.max_reduce_groups × max_evals(32) × field_byte_size.
    reduce_partials: metal::Buffer,
    /// Pre-allocated params buffer for reduce dispatches. 16 bytes (4 × u32).
    reduce_params: metal::Buffer,
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
        Self::with_compile_mode_and_field::<jolt_field::Fr>(compile_mode)
    }

    /// Create a backend with an explicit compile mode and field type.
    pub fn with_compile_mode_and_field<F: jolt_field::MontgomeryConstants>(
        compile_mode: CompileMode,
    ) -> Self {
        let device = Device::system_default().expect("no Metal device available");
        let queue = device.new_command_queue();
        let device_config = MetalDeviceConfig::detect(&device);
        let field_config = MslFieldParams::new::<F>();
        let interpolation = InterpolationKernels::compile(&device, &field_config);

        let reduce_partials = device.new_buffer(
            (device_config.max_reduce_groups * 32 * field_config.byte_size) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let reduce_params = device.new_buffer(16, MTLResourceOptions::StorageModeShared);

        Self {
            device,
            queue,
            interpolation,
            compile_mode,
            field_config,
            device_config,
            pipeline_cache: Mutex::new(HashMap::new()),
            reduce_partials,
            reduce_params,
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

    /// Get cached pipelines or compile and cache them.
    fn get_or_compile(&self, formula: &Formula) -> Arc<CachedPipelines> {
        use std::hash::{Hash, Hasher};
        let msl = compiler::generate_msl(
            formula,
            self.compile_mode,
            &self.field_config,
            &self.device_config,
        );

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        msl.source.hash(&mut hasher);
        let key = hasher.finish();

        let mut cache = self.pipeline_cache.lock().unwrap();
        if let Some(cached) = cache.get(&key) {
            return Arc::clone(cached);
        }

        let pipelines = compiler::compile_msl(&self.device, &msl);
        cache.insert(key, Arc::clone(&pipelines));
        pipelines
    }

    /// Upload a slice of field elements to a shared Metal buffer.
    fn upload_field_buffer<F: Field>(&self, data: &[F]) -> metal::Buffer {
        debug_assert_eq!(std::mem::size_of::<F>(), self.field_config.byte_size);
        self.device.new_buffer_with_data(
            data.as_ptr().cast::<std::ffi::c_void>(),
            std::mem::size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Dispatch a reduce kernel and finish partial sums on CPU.
    ///
    /// Buffer binding layout: `input_0..K-1`, then `weight_buffers` (0–2),
    /// then optional `challenges_buf`, then `partials`, then `params`.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_reduce<F: Field>(
        &self,
        pipeline: &metal::ComputePipelineState,
        num_evals: usize,
        inputs: &[&MetalBuffer<F>],
        weight_buffers: &[&metal::Buffer],
        challenges_buf: Option<&metal::Buffer>,
        params: &[u32],
        n_pairs: usize,
    ) -> Vec<F> {
        if n_pairs == 0 {
            return vec![F::zero(); num_evals];
        }

        let num_groups = n_pairs
            .div_ceil(self.device_config.reduce_group_size)
            .min(self.device_config.max_reduce_groups);

        // SAFETY: `reduce_params` is a 16-byte shared buffer (4 × u32). `params`
        // has at most 3 entries (n_pairs, inner_log, inner_mask). No Metal commands
        // are in flight — previous command buffer completed before this call.
        unsafe {
            let ptr = self.reduce_params.contents().cast::<u32>();
            for (i, &p) in params.iter().enumerate() {
                ptr.add(i).write(p);
            }
        }

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
        if let Some(chal_buf) = challenges_buf {
            enc.set_buffer(idx, Some(chal_buf), 0);
            idx += 1;
        }
        enc.set_buffer(idx, Some(&self.reduce_partials), 0);
        enc.set_buffer(idx + 1, Some(&self.reduce_params), 0);

        enc.dispatch_thread_groups(
            MTLSize::new(num_groups as u64, 1, 1),
            MTLSize::new(self.device_config.reduce_group_size as u64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // SAFETY: shared memory is coherent after command buffer completion.
        let partials: &[F] = unsafe {
            let ptr = self.reduce_partials.contents().cast::<F>();
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

    fn compile<F: Field>(&self, spec: &KernelSpec) -> MetalKernel<F> {
        MetalKernel {
            pipelines: self.get_or_compile(&spec.formula),
            iteration: spec.iteration,
            binding_order: spec.binding_order,
            _marker: PhantomData,
        }
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

    #[tracing::instrument(skip_all, name = "MetalBackend::reduce")]
    fn reduce<F: Field>(
        &self,
        kernel: &MetalKernel<F>,
        inputs: &[&Buf<Self, F>],
        challenges: &[F],
    ) -> Vec<F> {
        let num_formula_inputs = inputs.len()
            - match kernel.iteration {
                Iteration::Dense => 0,
                Iteration::DenseTensor => 2,
                Iteration::Sparse => 1,
            };

        let value_refs: Vec<&MetalBuffer<F>> = inputs[..num_formula_inputs]
            .iter()
            .map(|db| db.as_field())
            .collect();

        debug_assert!(!value_refs.is_empty());
        let n = value_refs[0].len();
        let n_pairs = n / 2;

        let challenges_buf = if kernel.pipelines.has_challenges && !challenges.is_empty() {
            Some(self.upload_field_buffer(challenges))
        } else {
            None
        };
        let challenges_ref = challenges_buf.as_ref();

        match &kernel.iteration {
            Iteration::Dense => {
                let pipeline = match kernel.binding_order {
                    BindingOrder::LowToHigh => kernel.pipeline_l2h_unw(),
                    BindingOrder::HighToLow => kernel.pipeline_h2l_unw(),
                };
                self.dispatch_reduce(
                    pipeline,
                    kernel.num_evals(),
                    &value_refs,
                    &[],
                    challenges_ref,
                    &[n_pairs as u32],
                    n_pairs,
                )
            }
            Iteration::Sparse => todo!("sparse reduce on Metal"),
            Iteration::DenseTensor => {
                let outer = inputs[num_formula_inputs].as_field();
                let inner = inputs[num_formula_inputs + 1].as_field();
                let inner_len = inner.len();
                let inner_log = inner_len.trailing_zeros();
                let inner_mask = (inner_len - 1) as u32;
                self.dispatch_reduce(
                    kernel.pipeline_tensor(),
                    kernel.num_evals(),
                    &value_refs,
                    &[&outer.raw, &inner.raw],
                    challenges_ref,
                    &[n_pairs as u32, inner_log, inner_mask],
                    n_pairs,
                )
            }
        }
    }

    fn bind<F: Field>(&self, kernel: &MetalKernel<F>, inputs: &mut [Buf<Self, F>], scalar: F) {
        let order = kernel.binding_order;
        match &kernel.iteration {
            Iteration::Dense | Iteration::DenseTensor => {
                for buf in inputs.iter_mut() {
                    self.interpolate_inplace(buf.as_field_mut(), scalar, order);
                }
            }
            Iteration::Sparse => todo!("sparse bind on Metal"),
        }
    }

    fn interpolate_inplace<F: Field>(
        &self,
        buf: &mut Self::Buffer<F>,
        scalar: F,
        order: BindingOrder,
    ) {
        let n = buf.len();
        debug_assert!(n.is_multiple_of(2) || n == 0);
        let half = n / 2;

        match order {
            BindingOrder::HighToLow => {
                // Safe in-place: thread i reads buf[i] and buf[i+half], writes buf[i].
                let scalar_buf = self.upload_scalar(&scalar);
                // SAFETY: no Metal commands in flight — reuse pre-allocated params.
                unsafe {
                    self.reduce_params
                        .contents()
                        .cast::<u32>()
                        .write(half as u32);
                }
                self.dispatch_1d(
                    &self.interpolation.interpolate_inplace_high,
                    &[&buf.raw, &scalar_buf, &self.reduce_params],
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

    fn eq_table<F: Field>(&self, point: &[F]) -> Self::Buffer<F> {
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
                enc.set_compute_pipeline_state(&self.interpolation.eq_table_round);
                enc.set_buffer(0, Some(&table.raw), 0);
                enc.set_buffer(1, Some(&r_bufs[idx]), 0);
                enc.set_buffer(2, Some(&params_bufs[idx]), 0);
                let tpg = self
                    .interpolation
                    .eq_table_round
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

    fn lt_table<F: Field>(&self, point: &[F]) -> Self::Buffer<F> {
        let n = point.len();
        let mut evals = vec![F::zero(); 1usize << n];
        for (i, &ri) in point.iter().rev().enumerate() {
            let half = 1usize << i;
            let (left, right) = evals.split_at_mut(half);
            left.iter_mut().zip(right.iter_mut()).for_each(|(x, y)| {
                *y = *x * ri;
                *x += ri - *y;
            });
        }
        self.upload(&evals)
    }

    fn eq_plus_one_table<F: Field>(&self, point: &[F]) -> (Self::Buffer<F>, Self::Buffer<F>) {
        let ell = point.len();
        let size = 1usize << ell;
        let mut eq_evals = vec![F::zero(); size];
        eq_evals[0] = F::one();
        let mut epo_evals = vec![F::zero(); size];

        for i in 0..ell {
            let step = 1usize << (ell - i);
            let half_step = step / 2;

            let mut r_lower_product = F::one();
            for &x in point.iter().skip(i + 1) {
                r_lower_product *= x;
            }
            r_lower_product *= F::one() - point[i];

            let mut idx = half_step;
            while idx < size {
                epo_evals[idx] = eq_evals[idx - half_step] * r_lower_product;
                idx += step;
            }

            let eq_step = 1usize << (ell - i - 1);
            let mut k = 0;
            while k < size {
                let val = eq_evals[k] * point[i];
                eq_evals[k + eq_step] = val;
                eq_evals[k] -= val;
                k += eq_step * 2;
            }
        }

        (self.upload(&eq_evals), self.upload(&epo_evals))
    }
}
