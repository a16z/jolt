use std::collections::HashMap;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use jolt_compiler::kernel_spec::Iteration;
use jolt_compiler::KernelSpec;
use jolt_compute::{BindingOrder, Buf, ComputeBackend, Scalar};
use jolt_field::Field;
use metal::{CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize};

use crate::buffer::MetalBuffer;
use crate::config::MetalDeviceConfig;
use crate::field_params::MslFieldParams;
use crate::kernel::{CompiledPipeline, MetalKernel};
use crate::msl_reduce::{self, CompileMode, KernelVariant};
use crate::pipeline::{build_source_with_preamble, make_pipeline, SHADER_INTERPOLATION};

/// Pre-compiled pipelines for interpolation and eq table operations.
struct InterpolationKernels {
    interpolate_low: ComputePipelineState,
    interpolate_inplace_high: ComputePipelineState,
    sparse_bind: ComputePipelineState,
    eq_table_round: ComputePipelineState,
}

impl InterpolationKernels {
    fn compile(device: &Device, field_config: &MslFieldParams) -> Self {
        let source =
            build_source_with_preamble(&field_config.msl_preamble, &[SHADER_INTERPOLATION], false);
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(&source, &options)
            .expect("interpolation MSL compilation failed");

        Self {
            interpolate_low: make_pipeline(device, &library, "fr_interpolate_low_kernel"),
            interpolate_inplace_high: make_pipeline(
                device,
                &library,
                "fr_interpolate_inplace_high_kernel",
            ),
            sparse_bind: make_pipeline(device, &library, "fr_sparse_bind_kernel"),
            eq_table_round: make_pipeline(device, &library, "fr_eq_table_round_kernel"),
        }
    }
}

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
    pipeline_cache: Mutex<HashMap<u64, Arc<CompiledPipeline>>>,
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
        let device_config = MetalDeviceConfig::default();
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

    /// Get a cached pipeline or compile and cache it for the given variant.
    fn get_or_compile(&self, spec: &KernelSpec) -> Arc<CompiledPipeline> {
        use std::hash::{Hash, Hasher};
        let variant = KernelVariant::from_spec(&spec.iteration, spec.binding_order);
        let msl = msl_reduce::generate_msl(
            &spec.formula,
            variant,
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

        let compiled = msl_reduce::compile_msl(&self.device, &msl);
        cache.insert(key, Arc::clone(&compiled));
        compiled
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
            compiled: self.get_or_compile(spec),
            iteration: spec.iteration.clone(),
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
                Iteration::Dense | Iteration::Domain { .. } => 0,
                Iteration::DenseTensor | Iteration::Gruen => 2,
                Iteration::Sparse => 1,
            };

        let value_refs: Vec<&MetalBuffer<F>> = inputs[..num_formula_inputs]
            .iter()
            .map(|db| db.as_field())
            .collect();

        debug_assert!(!value_refs.is_empty());
        let n = value_refs[0].len();
        let n_pairs = n / 2;

        let challenges_buf = if kernel.compiled.has_challenges && !challenges.is_empty() {
            Some(self.upload_field_buffer(challenges))
        } else {
            None
        };
        let challenges_ref = challenges_buf.as_ref();

        let pipeline = kernel.pipeline();

        match &kernel.iteration {
            Iteration::Dense => self.dispatch_reduce(
                pipeline,
                kernel.num_evals(),
                &value_refs,
                &[],
                challenges_ref,
                &[n_pairs as u32],
                n_pairs,
            ),
            Iteration::Sparse => {
                let keys = inputs[num_formula_inputs].as_u64().to_vec();
                let (pair_indices, _parent_keys) = build_sparse_pairs(&keys);
                let n_sparse_pairs = pair_indices.len() / 2;
                let pair_buf = self.device.new_buffer_with_data(
                    pair_indices.as_ptr().cast::<c_void>(),
                    std::mem::size_of_val(pair_indices.as_slice()) as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                self.dispatch_reduce(
                    pipeline,
                    kernel.num_evals(),
                    &value_refs,
                    &[&pair_buf],
                    challenges_ref,
                    &[n_sparse_pairs as u32],
                    n_sparse_pairs,
                )
            }
            Iteration::DenseTensor => {
                let outer = inputs[num_formula_inputs].as_field();
                let inner = inputs[num_formula_inputs + 1].as_field();
                let inner_len = inner.len();
                let inner_log = inner_len.trailing_zeros();
                let inner_mask = (inner_len - 1) as u32;
                self.dispatch_reduce(
                    pipeline,
                    kernel.num_evals(),
                    &value_refs,
                    &[&outer.raw, &inner.raw],
                    challenges_ref,
                    &[n_pairs as u32, inner_log, inner_mask],
                    n_pairs,
                )
            }
            Iteration::Domain { .. } => {
                panic!("Domain iteration not yet supported on Metal — use CpuBackend")
            }
            Iteration::Gruen => {
                panic!("Gruen iteration not yet supported on Metal — use CpuBackend")
            }
        }
    }

    fn bind<F: Field>(&self, kernel: &MetalKernel<F>, inputs: &mut [Buf<Self, F>], scalar: F) {
        let order = kernel.binding_order;
        match &kernel.iteration {
            Iteration::Dense | Iteration::DenseTensor | Iteration::Gruen => {
                for buf in inputs.iter_mut() {
                    self.interpolate_inplace(buf.as_field_mut(), scalar, order);
                }
            }
            Iteration::Sparse => {
                let num_value_inputs = inputs.len() - 1;
                let keys = inputs[num_value_inputs].as_u64().to_vec();
                let (pair_indices, parent_keys) = build_sparse_pairs(&keys);
                let n_pairs = pair_indices.len() / 2;

                let pair_buf = self.device.new_buffer_with_data(
                    pair_indices.as_ptr().cast::<c_void>(),
                    std::mem::size_of_val(pair_indices.as_slice()) as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                let scalar_buf = self.upload_scalar(&scalar);

                for input in &mut inputs[..num_value_inputs] {
                    let src = input.as_field();
                    let out: MetalBuffer<F> = MetalBuffer::zeroed(&self.device, n_pairs);
                    self.dispatch_1d(
                        &self.interpolation.sparse_bind,
                        &[&src.raw, &out.raw, &pair_buf, &scalar_buf],
                        n_pairs,
                    );
                    *input.as_field_mut() = out;
                }

                *inputs[num_value_inputs].as_u64_mut() = self.upload(&parent_keys);
            }
            Iteration::Domain { .. } => {
                unreachable!("Domain iteration kernels have exactly 1 round and are never bound");
            }
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
        let evals = jolt_poly::LtPolynomial::evaluations(point);
        self.upload(&evals)
    }

    fn eq_plus_one_table<F: Field>(&self, point: &[F]) -> (Self::Buffer<F>, Self::Buffer<F>) {
        let (eq_evals, epo_evals) = jolt_poly::EqPlusOnePolynomial::evals(point, None);
        (self.upload(&eq_evals), self.upload(&epo_evals))
    }

    fn duplicate_interleave<F: Field>(&self, buf: &Self::Buffer<F>) -> Self::Buffer<F> {
        let data = self.download(buf);
        let mut out = Vec::with_capacity(2 * data.len());
        for &val in &data {
            out.push(val);
            out.push(val);
        }
        self.upload(&out)
    }

    fn regroup_constraints<F: Field>(
        &self,
        buf: &Self::Buffer<F>,
        group_indices: &[Vec<usize>],
        old_stride: usize,
        new_stride: usize,
        num_cycles: usize,
    ) -> Self::Buffer<F> {
        let data = self.download(buf);
        let num_groups = group_indices.len();
        let total = num_groups * num_cycles * new_stride;
        let mut out = vec![F::zero(); total];
        for c in 0..num_cycles {
            for (g, indices) in group_indices.iter().enumerate() {
                let dst_row = num_groups * c + g;
                for (k, &src_idx) in indices.iter().enumerate() {
                    out[dst_row * new_stride + k] = data[c * old_stride + src_idx];
                }
            }
        }
        self.upload(&out)
    }

    fn evaluate_claim<F: Field>(
        &self,
        formula: &jolt_compiler::module::ClaimFormula,
        evaluations: &std::collections::HashMap<jolt_compiler::PolynomialId, F>,
        staged_evals: &std::collections::HashMap<(jolt_compiler::PolynomialId, usize), F>,
        challenges: &[F],
    ) -> F {
        use jolt_compiler::module::ClaimFactor;
        let mut sum = F::zero();
        for term in &formula.terms {
            let mut product = F::from_i128(term.coeff);
            for factor in &term.factors {
                product *= match factor {
                    ClaimFactor::Eval(poly) => *evaluations
                        .get(poly)
                        .unwrap_or_else(|| panic!("evaluate_claim: {poly:?} not available")),
                    ClaimFactor::StagedEval { poly, stage } => {
                        *staged_evals.get(&(*poly, *stage)).unwrap_or_else(|| {
                            panic!("evaluate_claim: {poly:?}@stage{stage} not available")
                        })
                    }
                    ClaimFactor::Challenge(i) => challenges[*i],
                    ClaimFactor::EqChallengePair { a, b } => {
                        let (ra, rb) = (challenges[*a], challenges[*b]);
                        ra * rb + (F::one() - ra) * (F::one() - rb)
                    }
                    other => panic!("evaluate_claim: unsupported factor {other:?}"),
                };
            }
            sum += product;
        }
        sum
    }

    fn evaluate_mle<F: Field>(&self, evals: &[F], point: &[F]) -> F {
        let mut buf = evals.to_vec();
        for r in point.iter().rev() {
            let half = buf.len() / 2;
            for i in 0..half {
                buf[i] = buf[2 * i] + *r * (buf[2 * i + 1] - buf[2 * i]);
            }
            buf.truncate(half);
        }
        buf[0]
    }

    fn uniskip_encode<F: Field>(
        &self,
        raw_evals: &mut [F],
        domain_size: usize,
        domain_start: i64,
        tau: F,
        zero_base: bool,
        num_coeffs: usize,
    ) -> Vec<F> {
        let k = domain_size;
        if zero_base {
            for v in raw_evals.iter_mut().take(k) {
                *v = F::zero();
            }
        }
        let t1_coeffs = jolt_poly::lagrange::interpolate_to_coeffs(domain_start, raw_evals);
        let basis_at_tau = jolt_poly::lagrange::lagrange_evals(domain_start, k, tau);
        let lagrange_coeffs =
            jolt_poly::lagrange::interpolate_to_coeffs(domain_start, &basis_at_tau);
        let mut s1 = jolt_poly::lagrange::poly_mul(&t1_coeffs, &lagrange_coeffs);
        s1.resize(num_coeffs, F::zero());
        s1
    }

    fn compressed_encode<F: Field>(&self, evals: &[F]) -> Vec<F> {
        let points: Vec<(F, F)> = evals
            .iter()
            .enumerate()
            .map(|(s, &v)| (F::from_u64(s as u64), v))
            .collect();
        jolt_poly::UnivariatePoly::interpolate(&points).into_coefficients()
    }

    fn interpolate_evaluate<F: Field>(&self, evals: &[F], point: F) -> F {
        let points: Vec<(F, F)> = evals
            .iter()
            .enumerate()
            .map(|(s, &v)| (F::from_u64(s as u64), v))
            .collect();
        jolt_poly::UnivariatePoly::interpolate(&points).evaluate(point)
    }

    fn extend_evals<F: Field>(&self, evals: &[F], target_len: usize) -> Vec<F> {
        let points: Vec<(F, F)> = evals
            .iter()
            .enumerate()
            .map(|(s, &v)| (F::from_u64(s as u64), v))
            .collect();
        let poly = jolt_poly::UnivariatePoly::interpolate(&points);
        let mut result = evals.to_vec();
        for s in evals.len()..target_len {
            result.push(poly.evaluate(F::from_u64(s as u64)));
        }
        result
    }

    fn scale_from_host<F: Field>(&self, data: &[F], scale: F) -> Self::Buffer<F> {
        let scaled: Vec<F> = data.iter().map(|&v| scale * v).collect();
        self.upload(&scaled)
    }

    fn transpose_from_host<F: Field>(
        &self,
        data: &[F],
        rows: usize,
        cols: usize,
    ) -> Self::Buffer<F> {
        let mut out = vec![F::zero(); rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = data[r * cols + c];
            }
        }
        self.upload(&out)
    }

    fn eq_gather<F: Field>(&self, eq_point: &[F], index_data: &[F]) -> Self::Buffer<F> {
        let eq_table = jolt_poly::EqPolynomial::<F>::evals(eq_point, None);
        let gathered: Vec<F> = index_data
            .iter()
            .map(|v| match v.to_u64() {
                Some(k) if (k as usize) < eq_table.len() => eq_table[k as usize],
                _ => F::zero(),
            })
            .collect();
        self.upload(&gathered)
    }

    fn eq_pushforward<F: Field>(
        &self,
        eq_point: &[F],
        index_data: &[F],
        output_size: usize,
    ) -> Self::Buffer<F> {
        let eq_table = jolt_poly::EqPolynomial::<F>::evals(eq_point, None);
        let mut result = vec![F::zero(); output_size];
        for (j, &eq_val) in eq_table.iter().enumerate() {
            if j < index_data.len() {
                if let Some(k) = index_data[j].to_u64() {
                    let k = k as usize;
                    if k < output_size {
                        result[k] += eq_val;
                    }
                }
            }
        }
        self.upload(&result)
    }

    fn eq_project<F: Field>(
        &self,
        source_data: &[F],
        eq_point: &[F],
        inner_size: usize,
        outer_size: usize,
    ) -> Self::Buffer<F> {
        let eq_table = jolt_poly::EqPolynomial::<F>::evals(eq_point, None);
        if eq_table.len() == inner_size {
            let mut projected = vec![F::zero(); outer_size];
            for (t, &eq_val) in eq_table.iter().enumerate() {
                if eq_val.is_zero() {
                    continue;
                }
                let base = t * outer_size;
                for k in 0..outer_size {
                    projected[k] += eq_val * source_data[base + k];
                }
            }
            self.upload(&projected)
        } else {
            let mut projected = vec![F::zero(); inner_size];
            for (t, proj) in projected.iter_mut().enumerate() {
                let base = t * outer_size;
                for (k, &eq_val) in eq_table.iter().enumerate() {
                    if !eq_val.is_zero() {
                        *proj += eq_val * source_data[base + k];
                    }
                }
            }
            self.upload(&projected)
        }
    }

    fn lagrange_project<F: Field>(
        &self,
        buf: &Self::Buffer<F>,
        challenge: F,
        domain_start: i64,
        domain_size: usize,
        stride: usize,
        group_offsets: &[usize],
        scale: F,
    ) -> Self::Buffer<F> {
        let data = self.download(buf);
        let basis = jolt_poly::lagrange::lagrange_evals(domain_start, domain_size, challenge);
        let num_groups = group_offsets.len();
        let num_cycles = data.len() / stride;
        let mut projected = vec![F::zero(); num_cycles * num_groups];
        for c in 0..num_cycles {
            for (g, &offset) in group_offsets.iter().enumerate() {
                let mut acc = F::zero();
                for (k, &lk) in basis.iter().enumerate() {
                    let idx = c * stride + offset + k;
                    if idx < data.len() {
                        acc += lk * data[idx];
                    }
                }
                projected[c * num_groups + g] = acc;
            }
        }
        if !scale.is_one() {
            for v in &mut projected {
                *v *= scale;
            }
        }
        self.upload(&projected)
    }

    fn segmented_reduce<F: Field>(
        &self,
        kernel: &Self::CompiledKernel<F>,
        inputs: &[&Self::Buffer<F>],
        outer_eq: &[F],
        inner_only: &[bool],
        inner_size: usize,
        challenges: &[F],
    ) -> Vec<F> {
        use jolt_compute::DeviceBuffer;
        let input_data: Vec<Vec<F>> = inputs.iter().map(|b| self.download(b)).collect();
        let mut col_buf = vec![F::zero(); inner_size];
        let mut total_evals: Option<Vec<F>> = None;
        for (a, &weight) in outer_eq.iter().enumerate() {
            if weight.is_zero() {
                continue;
            }
            let mut col_bufs: Vec<jolt_compute::Buf<Self, F>> = Vec::with_capacity(inputs.len());
            for (j, data) in input_data.iter().enumerate() {
                if inner_only[j] {
                    col_bufs.push(DeviceBuffer::Field(self.upload(data)));
                } else {
                    let start = a * inner_size;
                    col_buf.copy_from_slice(&data[start..start + inner_size]);
                    col_bufs.push(DeviceBuffer::Field(self.upload(&col_buf)));
                }
            }
            let col_refs: Vec<&jolt_compute::Buf<Self, F>> = col_bufs.iter().collect();
            let evals = self.reduce(kernel, &col_refs, challenges);
            match &mut total_evals {
                Some(total) => {
                    for (t, &e) in total.iter_mut().zip(&evals) {
                        *t += weight * e;
                    }
                }
                None => {
                    total_evals = Some(evals.iter().map(|&e| weight * e).collect());
                }
            }
        }
        total_evals.unwrap_or_default()
    }
}

const SPARSE_SENTINEL: u32 = u32::MAX;

/// Build the sparse pair index from a sorted key column.
///
/// Scans `keys` linearly, merging adjacent `(2k, 2k+1)` entries into a
/// single pair with parent key `k`. Missing lo/hi entries use [`SPARSE_SENTINEL`].
///
/// Returns `(pair_indices, parent_keys)` where `pair_indices` is a flat
/// `[lo_0, hi_0, lo_1, hi_1, ...]` buffer ready for GPU upload.
fn build_sparse_pairs(keys: &[u64]) -> (Vec<u32>, Vec<u64>) {
    let n = keys.len();
    let mut indices = Vec::with_capacity(2 * n);
    let mut parent_keys = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let key = keys[i];
        let parent = key / 2;
        if key.is_multiple_of(2) {
            if i + 1 < n && keys[i + 1] == key + 1 {
                indices.push(i as u32);
                indices.push((i + 1) as u32);
                parent_keys.push(parent);
                i += 2;
            } else {
                indices.push(i as u32);
                indices.push(SPARSE_SENTINEL);
                parent_keys.push(parent);
                i += 1;
            }
        } else {
            indices.push(SPARSE_SENTINEL);
            indices.push(i as u32);
            parent_keys.push(parent);
            i += 1;
        }
    }
    (indices, parent_keys)
}
