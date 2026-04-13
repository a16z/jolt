//! CPU compute backend using Rayon for parallelism.

use std::collections::HashMap;

use jolt_compiler::kernel_spec::Iteration;
use jolt_compiler::module::{ClaimFactor, ClaimFormula};
use jolt_compiler::PolynomialId;
use jolt_field::{Field, FieldAccumulator};

use jolt_compute::{BindingOrder, Buf, ComputeBackend, DeviceBuffer, LookupTraceData, Scalar};

/// Parallelism threshold: buffers smaller than this use sequential loops.
///
/// Below this size the overhead of Rayon work-stealing exceeds the benefit.
const PAR_THRESHOLD: usize = 1024;

/// Composition evaluation function signature for [`CpuKernel`].
///
/// Takes `(lo_values, hi_values, challenges, out)` and writes evaluations into `out`.
pub type EvalFn<F> = dyn Fn(&[F], &[F], &[F], &mut [F]) + Send + Sync;

pub type BoxedEvalFn<F> = Box<EvalFn<F>>;

/// Single-point composition evaluation for domain iteration.
///
/// Takes `(values, challenges)` and returns the scalar formula result.
/// Used by [`reduce_domain`] where inputs are not in lo/hi pairs.
pub type DomainEvalFn<F> = dyn Fn(&[F], &[F]) -> F + Send + Sync;

/// CPU-compiled kernel: an eval closure plus metadata from the [`KernelSpec`].
///
/// The eval closure signature:
/// ```text
/// fn(lo: &[F], hi: &[F], challenges: &[F], out: &mut [F])
/// ```
///
/// where `lo[k]` and `hi[k]` are the even/odd pair for input buffer `k`,
/// `challenges` provides Fiat-Shamir-derived values, and `out` has
/// `num_evals` slots. For Toom-Cook kernels: grid `{1, ..., D-1, ∞}`.
/// For standard-grid kernels: grid `{0, 2, 3, ..., degree}`.
///
/// The [`Iteration`], [`BindingOrder`], and `num_evals` are baked in from
/// the [`KernelSpec`] at compile time. The `reduce`/`bind` dispatch logic
/// uses these to select the right loop structure.
pub struct CpuKernel<F: Field> {
    eval_fn: Box<EvalFn<F>>,
    domain_eval_fn: Option<Box<DomainEvalFn<F>>>,
    pub(crate) num_evals: usize,
    pub(crate) iteration: Iteration,
    pub(crate) binding_order: BindingOrder,
}

impl<F: Field> CpuKernel<F> {
    pub fn new(
        eval_fn: impl Fn(&[F], &[F], &[F], &mut [F]) + Send + Sync + 'static,
        num_evals: usize,
        iteration: Iteration,
        binding_order: BindingOrder,
    ) -> Self {
        Self {
            eval_fn: Box::new(eval_fn),
            domain_eval_fn: None,
            num_evals,
            iteration,
            binding_order,
        }
    }

    pub fn from_boxed(
        eval_fn: BoxedEvalFn<F>,
        num_evals: usize,
        iteration: Iteration,
        binding_order: BindingOrder,
    ) -> Self {
        Self {
            eval_fn,
            domain_eval_fn: None,
            num_evals,
            iteration,
            binding_order,
        }
    }

    pub fn with_domain_eval(mut self, f: Box<DomainEvalFn<F>>) -> Self {
        self.domain_eval_fn = Some(f);
        self
    }

    #[inline]
    pub fn evaluate(&self, lo: &[F], hi: &[F], challenges: &[F], out: &mut [F]) {
        (self.eval_fn)(lo, hi, challenges, out);
    }

    /// Evaluate the composition at a single point (for domain iteration).
    #[inline]
    pub fn evaluate_domain(&self, values: &[F], challenges: &[F]) -> F {
        (self
            .domain_eval_fn
            .as_ref()
            .expect("domain_eval_fn not compiled"))(values, challenges)
    }
}

/// Implements [`ComputeBackend`] with `Buffer<T> = Vec<T>`. All operations
/// use Rayon for parallelism (when the `parallel` feature is enabled and
/// buffers exceed `PAR_THRESHOLD`).
///
/// After monomorphization every trait call compiles to a direct function
/// call with no vtable indirection.
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    type Buffer<T: Scalar> = Vec<T>;
    type CompiledKernel<F: Field> = CpuKernel<F>;

    fn compile<F: Field>(&self, spec: &jolt_compiler::KernelSpec) -> CpuKernel<F> {
        crate::compile(spec)
    }

    fn reduce<F: Field>(
        &self,
        kernel: &CpuKernel<F>,
        inputs: &[&Buf<Self, F>],
        challenges: &[F],
    ) -> Vec<F> {
        let num_evals = kernel.num_evals;
        let order = kernel.binding_order;

        // Composition value columns (excluding iteration-specific extras).
        let num_value_inputs = inputs.len()
            - match kernel.iteration {
                Iteration::Dense
                | Iteration::Domain { .. }
                | Iteration::PrefixSuffix { .. }
                | Iteration::Booleanity { .. }
                | Iteration::HammingWeightReduction { .. } => 0,
                Iteration::DenseTensor => 2,
                Iteration::Sparse => 1,
            };
        let value_refs: Vec<&Vec<F>> = inputs[..num_value_inputs]
            .iter()
            .map(|db| db.as_field())
            .collect();

        match &kernel.iteration {
            Iteration::Dense => reduce_dense(&value_refs, kernel, challenges, num_evals, order),
            Iteration::DenseTensor => {
                let outer = inputs[num_value_inputs].as_field();
                let inner = inputs[num_value_inputs + 1].as_field();
                reduce_tensor(&value_refs, outer, inner, kernel, challenges, num_evals)
            }
            Iteration::Sparse => {
                let keys = inputs[num_value_inputs].as_u64();
                reduce_sparse(&value_refs, keys, kernel, challenges, num_evals)
            }
            Iteration::Domain {
                domain_size,
                stride,
                domain_start,
                domain_indexed,
                ..
            } => reduce_domain(
                &value_refs,
                kernel,
                challenges,
                *domain_size,
                *stride,
                *domain_start,
                domain_indexed,
            ),
            Iteration::PrefixSuffix { .. } => {
                unreachable!("PrefixSuffix reduce is handled by the runtime")
            }
            Iteration::Booleanity { .. } => {
                unreachable!("Booleanity reduce is handled by the runtime")
            }
            Iteration::HammingWeightReduction { .. } => {
                unreachable!("HammingWeightReduction reduce is handled by the runtime")
            }
        }
    }

    fn bind<F: Field>(&self, kernel: &CpuKernel<F>, inputs: &mut [Buf<Self, F>], scalar: F) {
        let order = kernel.binding_order;
        match &kernel.iteration {
            Iteration::Dense | Iteration::DenseTensor => {
                for buf in inputs.iter_mut() {
                    interpolate_vec_inplace(buf.as_field_mut(), scalar, order);
                }
            }
            Iteration::Sparse => {
                bind_sparse(inputs, scalar);
            }
            Iteration::Domain { .. } => {
                unreachable!("Domain iteration kernels have exactly 1 round and are never bound");
            }
            Iteration::PrefixSuffix { .. } => {
                unreachable!("PrefixSuffix bind is handled by the runtime")
            }
            Iteration::Booleanity { .. } => {
                unreachable!("Booleanity bind is handled by the runtime")
            }
            Iteration::HammingWeightReduction { .. } => {
                unreachable!("HammingWeightReduction bind is handled by the runtime")
            }
        }
    }

    fn interpolate_inplace<F: Field>(&self, buf: &mut Vec<F>, scalar: F, order: BindingOrder) {
        interpolate_vec_inplace(buf, scalar, order);
    }

    #[inline]
    fn upload<T: Scalar>(&self, data: &[T]) -> Vec<T> {
        data.to_vec()
    }

    #[inline]
    fn download<T: Scalar>(&self, buf: &Vec<T>) -> Vec<T> {
        buf.clone()
    }

    fn alloc<T: Scalar>(&self, len: usize) -> Vec<T> {
        let mut buf = Vec::with_capacity(len);
        // SAFETY: All Scalar types (integers, bool, field elements) have
        // all-zeros as a valid representation. write_bytes zeroes the
        // allocated capacity, then set_len makes those bytes visible.
        unsafe {
            std::ptr::write_bytes(buf.as_mut_ptr(), 0, len);
            buf.set_len(len);
        }
        buf
    }

    #[inline]
    fn len<T: Scalar>(&self, buf: &Vec<T>) -> usize {
        buf.len()
    }

    #[tracing::instrument(skip_all, name = "CpuBackend::eq_table")]
    fn eq_table<F: Field>(&self, point: &[F]) -> Vec<F> {
        // Interleaved layout: each expansion maps table[j] → table[2j] and table[2j+1].
        // This ensures bit 0 of the index always corresponds to the LAST processed
        // variable, matching the LowToHigh binding convention used by sumcheck.
        //
        // The parallel path uses a separate output buffer to avoid the aliasing issue
        // inherent in in-place interleaved expansion, preserving the same layout as
        // the serial path. This is critical: a mixed serial/split-half layout would
        // cause LowToHigh binding to eliminate variables in the wrong order.
        let n = point.len();
        let size = 1usize << n;
        let mut table = Vec::with_capacity(size);
        table.push(F::one());

        for &r_i in point {
            let one_minus_r_i = F::one() - r_i;
            let prev_len = table.len();

            #[cfg(feature = "parallel")]
            {
                if prev_len >= PAR_THRESHOLD {
                    use rayon::prelude::*;
                    let mut out = vec![F::zero(); prev_len * 2];
                    let src: &[F] = &table;
                    out.par_chunks_mut(2).enumerate().for_each(|(j, pair)| {
                        let base = src[j];
                        pair[0] = base * one_minus_r_i;
                        pair[1] = base * r_i;
                    });
                    table = out;
                    continue;
                }
            }

            table.resize(prev_len * 2, F::zero());
            for j in (0..prev_len).rev() {
                let base = table[j];
                table[2 * j] = base * one_minus_r_i;
                table[2 * j + 1] = base * r_i;
            }
        }

        table
    }

    fn lt_table<F: Field>(&self, point: &[F]) -> Vec<F> {
        jolt_poly::LtPolynomial::evaluations(point)
    }

    fn eq_plus_one_table<F: Field>(&self, point: &[F]) -> (Vec<F>, Vec<F>) {
        jolt_poly::EqPlusOnePolynomial::evals(point, None)
    }

    fn duplicate_interleave<F: Field>(&self, buf: &Vec<F>) -> Vec<F> {
        let n = buf.len();
        let mut out = Vec::with_capacity(2 * n);
        for &val in buf {
            out.push(val);
            out.push(val);
        }
        out
    }

    fn regroup_constraints<F: Field>(
        &self,
        buf: &Vec<F>,
        group_indices: &[Vec<usize>],
        old_stride: usize,
        new_stride: usize,
        num_cycles: usize,
    ) -> Vec<F> {
        let num_groups = group_indices.len();
        let total = num_groups * num_cycles * new_stride;
        let mut out = vec![F::zero(); total];
        for c in 0..num_cycles {
            for (g, indices) in group_indices.iter().enumerate() {
                let dst_row = num_groups * c + g;
                for (k, &src_idx) in indices.iter().enumerate() {
                    out[dst_row * new_stride + k] = buf[c * old_stride + src_idx];
                }
            }
        }
        out
    }

    fn evaluate_claim<F: Field>(
        &self,
        formula: &ClaimFormula,
        evaluations: &HashMap<PolynomialId, F>,
        challenges: &[F],
    ) -> F {
        let mut sum = F::zero();
        for term in &formula.terms {
            let mut product = F::from_i128(term.coeff);
            for factor in &term.factors {
                product *= match factor {
                    ClaimFactor::Eval(poly) => *evaluations
                        .get(poly)
                        .unwrap_or_else(|| panic!("evaluate_claim: {poly:?} not available")),
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

    fn scale_from_host<F: Field>(&self, data: &[F], scale: F) -> Vec<F> {
        data.iter().map(|&v| scale * v).collect()
    }

    fn transpose_from_host<F: Field>(&self, data: &[F], rows: usize, cols: usize) -> Vec<F> {
        debug_assert_eq!(data.len(), rows * cols);
        let mut out = vec![F::zero(); rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = data[r * cols + c];
            }
        }
        out
    }

    fn eq_gather<F: Field>(&self, eq_point: &[F], index_data: &[F]) -> Vec<F> {
        let eq_table = jolt_poly::EqPolynomial::<F>::evals(eq_point, None);
        index_data
            .iter()
            .map(|v| match v.to_u64() {
                Some(k) if (k as usize) < eq_table.len() => eq_table[k as usize],
                _ => F::zero(),
            })
            .collect()
    }

    fn eq_pushforward<F: Field>(
        &self,
        eq_point: &[F],
        index_data: &[F],
        output_size: usize,
    ) -> Vec<F> {
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
        result
    }

    fn eq_project<F: Field>(
        &self,
        source_data: &[F],
        eq_point: &[F],
        inner_size: usize,
        outer_size: usize,
    ) -> Vec<F> {
        let eq_table = jolt_poly::EqPolynomial::<F>::evals(eq_point, None);
        let result = if eq_table.len() == inner_size {
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
            projected
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
            projected
        };
        result
    }

    fn lagrange_project<F: Field>(
        &self,
        buf: &Vec<F>,
        challenge: F,
        domain_start: i64,
        domain_size: usize,
        stride: usize,
        group_offsets: &[usize],
        scale: F,
    ) -> Vec<F> {
        let basis = jolt_poly::lagrange::lagrange_evals(domain_start, domain_size, challenge);
        let num_groups = group_offsets.len();
        let num_cycles = buf.len() / stride;
        let mut projected = vec![F::zero(); num_cycles * num_groups];
        for c in 0..num_cycles {
            for (g, &offset) in group_offsets.iter().enumerate() {
                let mut acc = F::zero();
                for (k, &lk) in basis.iter().enumerate() {
                    let idx = c * stride + offset + k;
                    if idx < buf.len() {
                        acc += lk * buf[idx];
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
        projected
    }

    fn segmented_reduce<F: Field>(
        &self,
        kernel: &CpuKernel<F>,
        inputs: &[&Vec<F>],
        outer_eq: &[F],
        inner_only: &[bool],
        inner_size: usize,
        challenges: &[F],
    ) -> Vec<F> {
        let mut col_buf = vec![F::zero(); inner_size];
        let mut total_evals: Option<Vec<F>> = None;

        for (a, &weight) in outer_eq.iter().enumerate() {
            if weight.is_zero() {
                continue;
            }
            let mut col_bufs: Vec<Buf<Self, F>> = Vec::with_capacity(inputs.len());
            for (j, &data) in inputs.iter().enumerate() {
                if inner_only[j] {
                    col_bufs.push(DeviceBuffer::Field(data.clone()));
                } else {
                    let start = a * inner_size;
                    col_buf.copy_from_slice(&data[start..start + inner_size]);
                    col_bufs.push(DeviceBuffer::Field(col_buf.clone()));
                }
            }
            let col_refs: Vec<&Buf<Self, F>> = col_bufs.iter().collect();
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

        total_evals.unwrap_or_else(|| vec![F::zero(); kernel.num_evals])
    }

    type PrefixSuffixState<F: Field> = crate::prefix_suffix::CpuPrefixSuffixState<F>;

    fn ps_init<F: Field>(
        &self,
        iteration: &Iteration,
        challenges: &[F],
        trace_data: &LookupTraceData,
    ) -> Self::PrefixSuffixState<F> {
        crate::prefix_suffix::CpuPrefixSuffixState::new(iteration, challenges, trace_data)
    }

    fn ps_bind<F: Field>(&self, state: &mut Self::PrefixSuffixState<F>, challenge: F) {
        state.ingest_challenge(challenge);
    }

    fn ps_reduce<F: Field>(&self, state: &Self::PrefixSuffixState<F>) -> [F; 2] {
        state.compute_address_round()
    }

    fn ps_materialize<F: Field>(
        &self,
        state: Self::PrefixSuffixState<F>,
    ) -> Vec<(PolynomialId, Vec<F>)> {
        state.materialize_outputs().into_iter().collect()
    }

    type BooleanityState<F: Field> = crate::booleanity::CpuBooleanityState<F>;

    fn bool_init<F: Field>(
        &self,
        ra_data: Vec<Vec<F>>,
        addr_challenges: &[F],
        cycle_challenges: &[F],
        gamma_powers: Vec<F>,
        gamma_powers_square: Vec<F>,
        log_k_chunk: usize,
        log_t: usize,
    ) -> Self::BooleanityState<F> {
        crate::booleanity::CpuBooleanityState::new(
            ra_data,
            addr_challenges,
            cycle_challenges,
            gamma_powers,
            gamma_powers_square,
            log_k_chunk,
            log_t,
        )
    }

    fn bool_bind<F: Field>(&self, state: &mut Self::BooleanityState<F>, challenge: F) {
        state.ingest_challenge(challenge);
    }

    fn bool_reduce<F: Field>(&self, state: &Self::BooleanityState<F>, previous_claim: F) -> Vec<F> {
        state.compute_round(previous_claim)
    }

    fn bool_final_claims<F: Field>(&self, state: &Self::BooleanityState<F>) -> Vec<F> {
        state.final_ra_claims()
    }

    type HwReductionState<F: Field> = crate::hw_reduction::CpuHwReductionState<F>;

    fn hw_init<F: Field>(
        &self,
        ra_data: &[Vec<F>],
        cycle_ch_be: &[F],
        addr_bool_ch_be: &[F],
        addr_virt_ch_be: &[Vec<F>],
        gamma_powers: Vec<F>,
        hw_claims: Vec<F>,
        bool_claims: Vec<F>,
        virt_claims: Vec<F>,
        log_k_chunk: usize,
        log_t: usize,
    ) -> Self::HwReductionState<F> {
        crate::hw_reduction::CpuHwReductionState::new(
            ra_data,
            cycle_ch_be,
            addr_bool_ch_be,
            addr_virt_ch_be,
            gamma_powers,
            hw_claims,
            bool_claims,
            virt_claims,
            log_k_chunk,
            log_t,
        )
    }

    fn hw_bind<F: Field>(&self, state: &mut Self::HwReductionState<F>, challenge: F) {
        state.bind(challenge);
    }

    fn hw_reduce<F: Field>(&self, state: &Self::HwReductionState<F>, previous_claim: F) -> Vec<F> {
        state.reduce(previous_claim)
    }

    fn hw_final_claims<F: Field>(&self, state: &Self::HwReductionState<F>) -> Vec<F> {
        state.final_g_claims()
    }

    type InstanceState<F: Field> = ();

    fn instance_init<F: Field>(
        &self,
        _config: &jolt_compiler::module::InstanceConfig,
        _challenges: &[F],
        _provider: &mut dyn jolt_compute::BufferProvider<F>,
        _lookup_trace: Option<&LookupTraceData>,
        _kernels: &[jolt_compiler::KernelDef],
    ) -> Self::InstanceState<F> {
        panic!("unified instance API not yet wired")
    }

    fn instance_bind<F: Field>(&self, _state: &mut Self::InstanceState<F>, _challenge: F) {
        panic!("unified instance API not yet wired")
    }

    fn instance_reduce<F: Field>(
        &self,
        _state: &Self::InstanceState<F>,
        _previous_claim: F,
    ) -> Vec<F> {
        panic!("unified instance API not yet wired")
    }

    fn instance_finalize<F: Field>(
        &self,
        _state: Self::InstanceState<F>,
    ) -> jolt_compute::InstanceOutput<Self::Buffer<F>, F> {
        panic!("unified instance API not yet wired")
    }
}

#[tracing::instrument(skip_all, name = "interpolate_inplace")]
fn interpolate_vec_inplace<F: Field>(buf: &mut Vec<F>, scalar: F, order: BindingOrder) {
    match order {
        BindingOrder::HighToLow => jolt_poly::bind_high_to_low(buf, scalar),
        BindingOrder::LowToHigh => jolt_poly::bind_low_to_high(buf, scalar),
    }
}

/// Dense (unit-weighted) pairwise reduce.
///
/// Dispatches to a const-generic inner function for common (num_inputs, num_evals)
/// pairs, using stack-allocated scratch arrays. Falls back to heap-allocated Vecs
/// for uncommon sizes.
#[allow(clippy::ptr_arg)]
#[tracing::instrument(skip_all, name = "reduce_dense")]
fn reduce_dense<F: Field>(
    inputs: &[&Vec<F>],
    kernel: &CpuKernel<F>,
    challenges: &[F],
    num_evals: usize,
    order: BindingOrder,
) -> Vec<F> {
    let n = inputs[0].len();
    debug_assert!(n.is_multiple_of(2), "buffer length must be even");
    let half = n / 2;
    let num_inputs = inputs.len();
    for (idx, inp) in inputs.iter().enumerate() {
        assert_eq!(
            inp.len(),
            n,
            "reduce_dense: input[{idx}].len()={} != inputs[0].len()={n}, num_inputs={num_inputs}, num_evals={num_evals}",
            inp.len()
        );
    }

    match (num_inputs, num_evals) {
        (2, 2) => reduce_dense_fixed::<F, 2, 2>(inputs, kernel, challenges, half, order),
        (3, 3) => reduce_dense_fixed::<F, 3, 3>(inputs, kernel, challenges, half, order),
        (4, 4) => reduce_dense_fixed::<F, 4, 4>(inputs, kernel, challenges, half, order),
        (8, 4) => reduce_dense_fixed::<F, 8, 4>(inputs, kernel, challenges, half, order),
        (8, 8) => reduce_dense_fixed::<F, 8, 8>(inputs, kernel, challenges, half, order),
        (16, 16) => reduce_dense_fixed::<F, 16, 16>(inputs, kernel, challenges, half, order),
        (32, 32) => reduce_dense_fixed::<F, 32, 32>(inputs, kernel, challenges, half, order),
        _ => reduce_dense_dynamic(
            inputs, kernel, challenges, num_inputs, num_evals, half, order,
        ),
    }
}

/// Const-generic dense reduce with stack-allocated scratch arrays.
///
/// Eliminates per-chunk heap allocation in the Rayon fold by using `[F; NI]`
/// for lo/hi and `[F; NE]` for evals/accumulators.
#[inline]
fn reduce_dense_fixed<F: Field, const NI: usize, const NE: usize>(
    inputs: &[&Vec<F>],
    kernel: &CpuKernel<F>,
    challenges: &[F],
    half: usize,
    order: BindingOrder,
) -> Vec<F> {
    debug_assert_eq!(inputs.len(), NI);
    debug_assert_eq!(kernel.num_evals, NE);

    // Pre-compute data pointers as usize to eliminate per-pair indirection
    // through &[&Vec<F>] and to allow capture in Rayon closures (usize is Send+Sync).
    let mut ptrs = [0usize; NI];
    for (k, &input) in inputs.iter().enumerate() {
        ptrs[k] = input.as_ptr() as usize;
    }

    let load_pair = |ptrs: &[usize; NI], k: usize, i: usize| -> (F, F) {
        // SAFETY: ptrs[k] was derived from inputs[k].as_ptr(), valid for
        // inputs[k].len() elements. Indices 2*i, 2*i+1 (L2H) or i, i+half
        // (H2L) are in [0, 2*half) = [0, n), within the allocation.
        unsafe {
            let p = ptrs[k] as *const F;
            match order {
                BindingOrder::LowToHigh => (*p.add(2 * i), *p.add(2 * i + 1)),
                BindingOrder::HighToLow => (*p.add(i), *p.add(i + half)),
            }
        }
    };

    #[cfg(feature = "parallel")]
    {
        if half >= PAR_THRESHOLD {
            use rayon::prelude::*;

            let accs = (0..half)
                .into_par_iter()
                .with_min_len(2048)
                .fold(
                    || [F::Accumulator::default(); NE],
                    |mut acc, i| {
                        let mut lo = [F::zero(); NI];
                        let mut hi = [F::zero(); NI];
                        let mut evals = [F::zero(); NE];
                        for k in 0..NI {
                            let (l, h) = load_pair(&ptrs, k, i);
                            lo[k] = l;
                            hi[k] = h;
                        }
                        kernel.evaluate(&lo, &hi, challenges, &mut evals);
                        for (a, e) in acc.iter_mut().zip(evals.iter()) {
                            a.acc_add(*e);
                        }
                        acc
                    },
                )
                .reduce(
                    || [F::Accumulator::default(); NE],
                    |mut a, b| {
                        for (ai, bi) in a.iter_mut().zip(b) {
                            ai.merge(bi);
                        }
                        a
                    },
                );

            return accs.into_iter().map(FieldAccumulator::reduce).collect();
        }
    }

    let mut accs = [F::Accumulator::default(); NE];
    let mut lo = [F::zero(); NI];
    let mut hi = [F::zero(); NI];
    let mut evals = [F::zero(); NE];

    for i in 0..half {
        for k in 0..NI {
            let (l, h) = load_pair(&ptrs, k, i);
            lo[k] = l;
            hi[k] = h;
        }
        kernel.evaluate(&lo, &hi, challenges, &mut evals);
        for (a, e) in accs.iter_mut().zip(evals.iter()) {
            a.acc_add(*e);
        }
    }

    accs.into_iter().map(FieldAccumulator::reduce).collect()
}

/// Dynamic dense reduce for uncommon input/eval counts. Uses heap-allocated
/// scratch Vecs.
#[allow(clippy::ptr_arg)]
fn reduce_dense_dynamic<F: Field>(
    inputs: &[&Vec<F>],
    kernel: &CpuKernel<F>,
    challenges: &[F],
    num_inputs: usize,
    num_evals: usize,
    half: usize,
    order: BindingOrder,
) -> Vec<F> {
    let new_accs =
        || -> Vec<F::Accumulator> { (0..num_evals).map(|_| F::Accumulator::default()).collect() };

    let pair = |input: &[F], i: usize| -> (F, F) {
        match order {
            BindingOrder::LowToHigh => (input[2 * i], input[2 * i + 1]),
            BindingOrder::HighToLow => (input[i], input[i + half]),
        }
    };

    #[cfg(feature = "parallel")]
    {
        if half >= PAR_THRESHOLD {
            use rayon::prelude::*;

            let accs = (0..half)
                .into_par_iter()
                .with_min_len(4096)
                .fold(
                    || {
                        (
                            new_accs(),
                            vec![F::zero(); num_inputs],
                            vec![F::zero(); num_inputs],
                            vec![F::zero(); num_evals],
                        )
                    },
                    |(mut acc, mut lo, mut hi, mut evals), i| {
                        for (k, &input) in inputs.iter().enumerate() {
                            let (l, h) = pair(input, i);
                            lo[k] = l;
                            hi[k] = h;
                        }
                        kernel.evaluate(&lo, &hi, challenges, &mut evals);
                        for (a, e) in acc.iter_mut().zip(evals.iter()) {
                            a.acc_add(*e);
                        }
                        (acc, lo, hi, evals)
                    },
                )
                .map(|(acc, _, _, _)| acc)
                .reduce(new_accs, |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b) {
                        ai.merge(bi);
                    }
                    a
                });

            return accs.into_iter().map(FieldAccumulator::reduce).collect();
        }
    }

    let mut accs = new_accs();
    let mut lo = vec![F::zero(); num_inputs];
    let mut hi = vec![F::zero(); num_inputs];
    let mut evals = vec![F::zero(); num_evals];

    for i in 0..half {
        for (k, &input) in inputs.iter().enumerate() {
            let (l, h) = pair(input, i);
            lo[k] = l;
            hi[k] = h;
        }
        kernel.evaluate(&lo, &hi, challenges, &mut evals);
        for (a, e) in accs.iter_mut().zip(evals.iter()) {
            a.acc_add(*e);
        }
    }

    accs.into_iter().map(FieldAccumulator::reduce).collect()
}

/// Lagrange-domain reduce for univariate skip rounds.
///
/// Evaluates the composition formula at `2K - 1` domain points. The first K
/// points are base evaluations using direct buffer access; the remaining K - 1
/// are extended evaluations using precomputed Lagrange interpolation weights.
///
/// Returns `Vec<F>` of length `2 * domain_size - 1`.
#[allow(clippy::ptr_arg)]
#[tracing::instrument(skip_all, name = "reduce_domain")]
fn reduce_domain<F: Field>(
    inputs: &[&Vec<F>],
    kernel: &CpuKernel<F>,
    challenges: &[F],
    domain_size: usize,
    stride: usize,
    domain_start: i64,
    domain_indexed: &[bool],
) -> Vec<F> {
    let num_inputs = inputs.len();
    let num_extended = domain_size - 1;
    let num_evals = 2 * domain_size - 1;

    // Determine num_cycles from the first cycle-indexed input.
    let num_cycles = domain_indexed
        .iter()
        .zip(inputs.iter())
        .find_map(
            |(&is_domain, inp)| {
                if !is_domain {
                    Some(inp.len())
                } else {
                    None
                }
            },
        )
        .unwrap_or_else(|| inputs[0].len() / stride);

    // Precompute Lagrange weights for extended evaluation points.
    // Extended points are at {domain_start + K, ..., domain_start + 2K - 2}.
    let ext_weights: Vec<Vec<F>> = (0..num_extended)
        .map(|e| {
            let point = F::from_i64(domain_start + domain_size as i64 + e as i64);
            jolt_poly::lagrange::lagrange_evals(domain_start, domain_size, point)
        })
        .collect();

    let new_accs =
        || -> Vec<F::Accumulator> { (0..num_evals).map(|_| F::Accumulator::default()).collect() };

    let reduce_chunk = |accs: &mut [F::Accumulator], cycle_range: std::ops::Range<usize>| {
        let mut values = vec![F::zero(); num_inputs];

        for c in cycle_range {
            // Base evaluations: formula at each domain point 0..K
            for d in 0..domain_size {
                for (j, val) in values.iter_mut().enumerate() {
                    *val = if domain_indexed[j] {
                        inputs[j][c * stride + d]
                    } else {
                        inputs[j][c]
                    };
                }
                accs[d].acc_add(kernel.evaluate_domain(&values, challenges));
            }

            // Extended evaluations: Lagrange-interpolate each domain-indexed
            // input independently, then evaluate the formula. This matches
            // jolt-core, where Az(y) and Bz(y) are each degree-(K-1)
            // polynomials and their product is degree-2(K-1). Interpolating
            // the product directly would under-sample (K points for a
            // degree-2(K-1) polynomial).
            for (e, weights) in ext_weights.iter().enumerate() {
                for (j, val) in values.iter_mut().enumerate() {
                    if domain_indexed[j] {
                        let base = &inputs[j][c * stride..c * stride + domain_size];
                        let mut interp = F::zero();
                        for (k, &w) in weights.iter().enumerate() {
                            interp += w * base[k];
                        }
                        *val = interp;
                    } else {
                        *val = inputs[j][c];
                    }
                }
                accs[domain_size + e].acc_add(kernel.evaluate_domain(&values, challenges));
            }
        }
    };

    #[cfg(feature = "parallel")]
    {
        if num_cycles >= PAR_THRESHOLD {
            use rayon::prelude::*;

            let accs = (0..num_cycles)
                .into_par_iter()
                .with_min_len(256)
                .fold(&new_accs, |mut acc, c| {
                    reduce_chunk(&mut acc, c..c + 1);
                    acc
                })
                .reduce(new_accs, |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b) {
                        ai.merge(bi);
                    }
                    a
                });

            return accs.into_iter().map(FieldAccumulator::reduce).collect();
        }
    }

    let mut accs = new_accs();
    reduce_chunk(&mut accs, 0..num_cycles);
    accs.into_iter().map(FieldAccumulator::reduce).collect()
}

/// Tensor (split-eq) pairwise reduce. Always uses LowToHigh binding.
#[allow(clippy::ptr_arg)]
#[tracing::instrument(skip_all, name = "reduce_tensor")]
fn reduce_tensor<F: Field>(
    inputs: &[&Vec<F>],
    outer_weights: &[F],
    inner_weights: &[F],
    kernel: &CpuKernel<F>,
    challenges: &[F],
    num_evals: usize,
) -> Vec<F> {
    let inner_len = inner_weights.len();
    let num_inputs = inputs.len();
    let total_pairs = outer_weights.len() * inner_len;

    let new_accs = || -> Vec<F::Accumulator> { vec![F::Accumulator::default(); num_evals] };

    let inner_fold = |outer_acc: &mut Vec<F::Accumulator>,
                      lo: &mut Vec<F>,
                      hi: &mut Vec<F>,
                      evals: &mut Vec<F>,
                      x_out: usize| {
        let mut inner_acc = new_accs();
        for (x_in, &w_in) in inner_weights.iter().enumerate() {
            let g = x_out * inner_len + x_in;
            lo.clear();
            hi.clear();
            for &input in inputs {
                lo.push(input[2 * g]);
                hi.push(input[2 * g + 1]);
            }
            kernel.evaluate(lo, hi, challenges, evals);
            for (a, e) in inner_acc.iter_mut().zip(evals.iter()) {
                a.fmadd(w_in, *e);
            }
        }
        let w_out = outer_weights[x_out];
        for (oa, ia) in outer_acc.iter_mut().zip(inner_acc.iter()) {
            oa.fmadd(w_out, ia.reduce());
        }
    };

    #[cfg(feature = "parallel")]
    {
        if total_pairs >= PAR_THRESHOLD {
            use rayon::prelude::*;

            let accs = (0..outer_weights.len())
                .into_par_iter()
                .fold(
                    || {
                        (
                            new_accs(),
                            Vec::with_capacity(num_inputs),
                            Vec::with_capacity(num_inputs),
                            vec![F::zero(); num_evals],
                        )
                    },
                    |(mut outer_acc, mut lo, mut hi, mut evals), x_out| {
                        inner_fold(&mut outer_acc, &mut lo, &mut hi, &mut evals, x_out);
                        (outer_acc, lo, hi, evals)
                    },
                )
                .map(|(acc, _, _, _)| acc)
                .reduce(new_accs, |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b) {
                        ai.merge(bi);
                    }
                    a
                });

            return accs.into_iter().map(FieldAccumulator::reduce).collect();
        }
    }

    let mut outer_acc = new_accs();
    let mut lo = Vec::with_capacity(num_inputs);
    let mut hi = Vec::with_capacity(num_inputs);
    let mut evals = vec![F::zero(); num_evals];

    for x_out in 0..outer_weights.len() {
        inner_fold(&mut outer_acc, &mut lo, &mut hi, &mut evals, x_out);
    }

    outer_acc
        .into_iter()
        .map(FieldAccumulator::reduce)
        .collect()
}

/// One entry in the sparse pair index.
///
/// `lo_idx` / `hi_idx` are indices into the value columns. `None` means
/// the entry is absent (defaults to zero).
struct SparsePair {
    parent_key: u64,
    lo_idx: Option<usize>,
    hi_idx: Option<usize>,
}

/// Build the pair index from a sorted key column.
///
/// Scans `keys` linearly, merging adjacent `(2k, 2k+1)` entries into a
/// single [`SparsePair`] with parent key `k`. Runs in O(n).
fn build_sparse_pairs(keys: &[u64]) -> Vec<SparsePair> {
    let n = keys.len();
    let mut pairs = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let key = keys[i];
        let parent = key / 2;
        if key.is_multiple_of(2) {
            if i + 1 < n && keys[i + 1] == key + 1 {
                pairs.push(SparsePair {
                    parent_key: parent,
                    lo_idx: Some(i),
                    hi_idx: Some(i + 1),
                });
                i += 2;
            } else {
                pairs.push(SparsePair {
                    parent_key: parent,
                    lo_idx: Some(i),
                    hi_idx: None,
                });
                i += 1;
            }
        } else {
            pairs.push(SparsePair {
                parent_key: parent,
                lo_idx: None,
                hi_idx: Some(i),
            });
            i += 1;
        }
    }
    pairs
}

/// Sparse merge-join reduce over sorted keys.
#[allow(clippy::ptr_arg)]
#[tracing::instrument(skip_all, name = "reduce_sparse")]
fn reduce_sparse<F: Field>(
    value_inputs: &[&Vec<F>],
    keys: &Vec<u64>,
    kernel: &CpuKernel<F>,
    challenges: &[F],
    num_evals: usize,
) -> Vec<F> {
    let pairs = build_sparse_pairs(keys);
    let num_inputs = value_inputs.len();

    let new_accs =
        || -> Vec<F::Accumulator> { (0..num_evals).map(|_| F::Accumulator::default()).collect() };

    let eval_pair = |acc: &mut Vec<F::Accumulator>,
                     lo: &mut Vec<F>,
                     hi: &mut Vec<F>,
                     evals: &mut Vec<F>,
                     p: &SparsePair| {
        for (k, input) in value_inputs.iter().enumerate() {
            lo[k] = p.lo_idx.map_or(F::zero(), |j| input[j]);
            hi[k] = p.hi_idx.map_or(F::zero(), |j| input[j]);
        }
        kernel.evaluate(lo, hi, challenges, evals);
        for (a, e) in acc.iter_mut().zip(evals.iter()) {
            a.acc_add(*e);
        }
    };

    #[cfg(feature = "parallel")]
    {
        if pairs.len() >= PAR_THRESHOLD {
            use rayon::prelude::*;

            let accs = pairs
                .par_iter()
                .with_min_len(4096)
                .fold(
                    || {
                        (
                            new_accs(),
                            vec![F::zero(); num_inputs],
                            vec![F::zero(); num_inputs],
                            vec![F::zero(); num_evals],
                        )
                    },
                    |(mut acc, mut lo, mut hi, mut evals), p| {
                        eval_pair(&mut acc, &mut lo, &mut hi, &mut evals, p);
                        (acc, lo, hi, evals)
                    },
                )
                .map(|(acc, _, _, _)| acc)
                .reduce(new_accs, |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b) {
                        ai.merge(bi);
                    }
                    a
                });

            return accs.into_iter().map(FieldAccumulator::reduce).collect();
        }
    }

    let mut accs = new_accs();
    let mut lo = vec![F::zero(); num_inputs];
    let mut hi = vec![F::zero(); num_inputs];
    let mut evals = vec![F::zero(); num_evals];

    for p in &pairs {
        eval_pair(&mut accs, &mut lo, &mut hi, &mut evals, p);
    }

    accs.into_iter().map(FieldAccumulator::reduce).collect()
}

/// Sparse bind: interpolate paired entries and halve the key space.
#[tracing::instrument(skip_all, name = "bind_sparse")]
fn bind_sparse<F: Field>(inputs: &mut [Buf<CpuBackend, F>], scalar: F) {
    let num_value_inputs = inputs.len() - 1;
    let keys: Vec<u64> = inputs[num_value_inputs].as_u64().clone();
    let pairs = build_sparse_pairs(&keys);

    #[cfg(feature = "parallel")]
    {
        if pairs.len() >= PAR_THRESHOLD {
            use rayon::prelude::*;

            for input in &mut inputs[..num_value_inputs] {
                let values = input.as_field();
                let bound: Vec<F> = pairs
                    .par_iter()
                    .map(|p| {
                        let lo_val = p.lo_idx.map_or(F::zero(), |j| values[j]);
                        let hi_val = p.hi_idx.map_or(F::zero(), |j| values[j]);
                        lo_val + scalar * (hi_val - lo_val)
                    })
                    .collect();
                *input.as_field_mut() = bound;
            }

            let new_keys: Vec<u64> = pairs.iter().map(|p| p.parent_key).collect();
            *inputs[num_value_inputs].as_u64_mut() = new_keys;
            return;
        }
    }

    let out_len = pairs.len();
    for input in &mut inputs[..num_value_inputs] {
        let values = input.as_field();
        let mut bound = Vec::with_capacity(out_len);
        for p in &pairs {
            let lo_val = p.lo_idx.map_or(F::zero(), |j| values[j]);
            let hi_val = p.hi_idx.map_or(F::zero(), |j| values[j]);
            bound.push(lo_val + scalar * (hi_val - lo_val));
        }
        *input.as_field_mut() = bound;
    }

    let new_keys: Vec<u64> = pairs.iter().map(|p| p.parent_key).collect();
    *inputs[num_value_inputs].as_u64_mut() = new_keys;
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_compute::DeviceBuffer;
    use jolt_field::{Field, Fr};
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn backend() -> CpuBackend {
        CpuBackend
    }

    #[test]
    fn upload_download_round_trip() {
        let b = backend();
        let data: Vec<Fr> = vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
        let buf = b.upload(&data);
        let out = b.download(&buf);
        assert_eq!(data, out);
    }

    #[test]
    fn upload_download_compact_u8() {
        let b = backend();
        let data: Vec<u8> = vec![0, 1, 2, 255];
        let buf = b.upload(&data);
        let out = b.download(&buf);
        assert_eq!(data, out);
    }

    #[test]
    fn alloc_is_zeroed() {
        let b = backend();
        let buf: Vec<u64> = b.alloc(16);
        assert_eq!(buf.len(), 16);
        assert!(buf.iter().all(|&x| x == 0));
    }

    #[test]
    fn alloc_field_is_zero() {
        let b = backend();
        let buf: Vec<Fr> = b.alloc(8);
        assert_eq!(buf.len(), 8);
        assert!(buf.iter().all(|x| x.is_zero()));
    }

    #[test]
    fn len() {
        let b = backend();
        let empty: Vec<Fr> = b.alloc(0);
        assert_eq!(b.len(&empty), 0);

        let nonempty = b.upload(&[Fr::one()]);
        assert_eq!(b.len(&nonempty), 1);
    }

    #[test]
    fn interpolate_inplace_low_to_high() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let n = 8;
        let data: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let scalar = Fr::random(&mut rng);

        let mut buf = b.upload(&data);
        b.interpolate_inplace(&mut buf, scalar, BindingOrder::LowToHigh);

        assert_eq!(buf.len(), n / 2);
        for i in 0..n / 2 {
            let expected = data[2 * i] + scalar * (data[2 * i + 1] - data[2 * i]);
            assert_eq!(buf[i], expected);
        }
    }

    #[test]
    fn interpolate_inplace_high_to_low() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(2);
        let n = 8;
        let data: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let scalar = Fr::random(&mut rng);

        let mut buf = b.upload(&data);
        b.interpolate_inplace(&mut buf, scalar, BindingOrder::HighToLow);

        assert_eq!(buf.len(), n / 2);
        for i in 0..n / 2 {
            let expected = data[i] + scalar * (data[i + n / 2] - data[i]);
            assert_eq!(buf[i], expected);
        }
    }

    #[test]
    fn eq_table_sum_is_one() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let n = 5;
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let table = b.eq_table(&point);
        assert_eq!(table.len(), 1 << n);
        let sum: Fr = table.iter().copied().sum();
        assert_eq!(sum, Fr::one());
    }

    #[test]
    fn eq_table_matches_eq_polynomial() {
        use jolt_poly::EqPolynomial;

        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let n = 6;
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let table = b.eq_table(&point);
        let eq_table = EqPolynomial::new(point).evaluations();

        assert_eq!(table, eq_table);
    }

    #[test]
    fn eq_table_parallel_matches_sequential() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 14; // 16384 entries > PAR_THRESHOLD
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let table = b.eq_table(&point);
        assert_eq!(table.len(), 1 << n);

        let sum: Fr = table.iter().copied().sum();
        assert_eq!(sum, Fr::one());
    }

    /// Build a sparse kernel from a simple product `o0 * o1`.
    fn sparse_product_kernel() -> CpuKernel<Fr> {
        use jolt_compiler::{Factor, Formula, Iteration, KernelSpec, ProductTerm};
        let formula = Formula::from_terms(vec![ProductTerm {
            coefficient: 1,
            factors: vec![Factor::Input(0), Factor::Input(1)],
        }]);
        crate::compile(&KernelSpec {
            num_evals: formula.degree(),
            formula,
            iteration: Iteration::Sparse,
            binding_order: BindingOrder::LowToHigh,
        })
    }

    #[test]
    fn sparse_reduce_fully_paired() {
        let b = backend();
        let kernel = sparse_product_kernel();

        // Keys (0,1) and (4,5): two complete pairs with parent keys 0, 2.
        let keys: Buf<CpuBackend, Fr> = DeviceBuffer::U64(b.upload(&[0u64, 1, 4, 5]));
        let col_a: Buf<CpuBackend, Fr> = DeviceBuffer::Field(b.upload(&[
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
        ]));
        let col_b: Buf<CpuBackend, Fr> = DeviceBuffer::Field(b.upload(&[
            Fr::from_u64(11),
            Fr::from_u64(13),
            Fr::from_u64(17),
            Fr::from_u64(19),
        ]));

        let result = b.reduce(&kernel, &[&col_a, &col_b, &keys], &[]);

        // Toom-Cook grid {1, ∞} for d=2:
        // Pair 0: lo=(2,11), hi=(3,13) → P(1)=3*13=39, P(∞)=1*2=2
        // Pair 1: lo=(5,17), hi=(7,19) → P(1)=7*19=133, P(∞)=2*2=4
        // Sums: [39+133, 2+4] = [172, 6]
        assert_eq!(result, vec![Fr::from_u64(172), Fr::from_u64(6)]);
    }

    #[test]
    fn sparse_reduce_unpaired_even() {
        let b = backend();
        let kernel = sparse_product_kernel();

        // Key 6 only (even, no odd sibling): lo=(4,8), hi=(0,0)
        let keys: Buf<CpuBackend, Fr> = DeviceBuffer::U64(b.upload(&[6u64]));
        let col_a: Buf<CpuBackend, Fr> = DeviceBuffer::Field(b.upload(&[Fr::from_u64(4)]));
        let col_b: Buf<CpuBackend, Fr> = DeviceBuffer::Field(b.upload(&[Fr::from_u64(8)]));

        let result = b.reduce(&kernel, &[&col_a, &col_b, &keys], &[]);

        // P(1)=hi_a*hi_b=0*0=0, P(∞)=(0-4)*(0-8)=(-4)*(-8)=32
        assert_eq!(result[0], Fr::zero());
        assert_eq!(result[1], Fr::from_u64(32));
    }

    #[test]
    fn sparse_reduce_unpaired_odd() {
        let b = backend();
        let kernel = sparse_product_kernel();

        // Key 3 only (odd, no even sibling): lo=(0,0), hi=(5,7)
        let keys: Buf<CpuBackend, Fr> = DeviceBuffer::U64(b.upload(&[3u64]));
        let col_a: Buf<CpuBackend, Fr> = DeviceBuffer::Field(b.upload(&[Fr::from_u64(5)]));
        let col_b: Buf<CpuBackend, Fr> = DeviceBuffer::Field(b.upload(&[Fr::from_u64(7)]));

        let result = b.reduce(&kernel, &[&col_a, &col_b, &keys], &[]);

        // P(1)=hi_a*hi_b=5*7=35, P(∞)=(5-0)*(7-0)=35
        assert_eq!(result[0], Fr::from_u64(35));
        assert_eq!(result[1], Fr::from_u64(35));
    }

    #[test]
    fn sparse_reduce_mixed_pairing() {
        let b = backend();
        let kernel = sparse_product_kernel();

        // Keys: 1(odd-only), 4(even-only), 6,7(paired)
        let keys: Buf<CpuBackend, Fr> = DeviceBuffer::U64(b.upload(&[1u64, 4, 6, 7]));
        let col_a: Buf<CpuBackend, Fr> = DeviceBuffer::Field(b.upload(&[
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(8),
        ]));
        let col_b: Buf<CpuBackend, Fr> = DeviceBuffer::Field(b.upload(&[
            Fr::from_u64(10),
            Fr::from_u64(20),
            Fr::from_u64(30),
            Fr::from_u64(40),
        ]));

        let result = b.reduce(&kernel, &[&col_a, &col_b, &keys], &[]);

        // Key 1 (odd-only): lo=(0,0), hi=(2,10) → P(1)=2*10=20, P(∞)=2*10=20
        // Key 4 (even-only): lo=(3,20), hi=(0,0) → P(1)=0, P(∞)=(-3)*(-20)=60
        // Keys (6,7) (paired): lo=(5,30), hi=(8,40) → P(1)=8*40=320, P(∞)=3*10=30
        // Sums: [20+0+320, 20+60+30] = [340, 110]
        assert_eq!(result, vec![Fr::from_u64(340), Fr::from_u64(110)]);
    }

    #[test]
    fn sparse_bind_fully_paired() {
        let b = backend();
        let kernel = sparse_product_kernel();
        let scalar = Fr::from_u64(3);

        let mut inputs: Vec<Buf<CpuBackend, Fr>> = vec![
            DeviceBuffer::Field(b.upload(&[
                Fr::from_u64(10),
                Fr::from_u64(20),
                Fr::from_u64(30),
                Fr::from_u64(40),
            ])),
            DeviceBuffer::Field(b.upload(&[
                Fr::from_u64(1),
                Fr::from_u64(2),
                Fr::from_u64(3),
                Fr::from_u64(4),
            ])),
            DeviceBuffer::U64(b.upload(&[0u64, 1, 4, 5])),
        ];

        b.bind(&kernel, &mut inputs, scalar);

        // Pair (0,1) parent=0: 10+3*(20-10)=40, 1+3*(2-1)=4
        // Pair (4,5) parent=2: 30+3*(40-30)=60, 3+3*(4-3)=6
        assert_eq!(
            *inputs[0].as_field(),
            vec![Fr::from_u64(40), Fr::from_u64(60)]
        );
        assert_eq!(
            *inputs[1].as_field(),
            vec![Fr::from_u64(4), Fr::from_u64(6)]
        );
        assert_eq!(*inputs[2].as_u64(), vec![0u64, 2]);
    }

    #[test]
    fn sparse_bind_unpaired() {
        let b = backend();
        let kernel = sparse_product_kernel();
        let scalar = Fr::from_u64(5);

        // Key 3 (odd-only), key 8 (even-only)
        let mut inputs: Vec<Buf<CpuBackend, Fr>> = vec![
            DeviceBuffer::Field(b.upload(&[Fr::from_u64(7), Fr::from_u64(11)])),
            DeviceBuffer::U64(b.upload(&[3u64, 8])),
        ];

        b.bind(&kernel, &mut inputs, scalar);

        // Key 3 odd-only, parent=1: lo=0, hi=7 → 0+5*(7-0)=35
        // Key 8 even-only, parent=4: lo=11, hi=0 → 11+5*(0-11)=-44
        let neg_44 = Fr::zero() - Fr::from_u64(44);
        assert_eq!(*inputs[0].as_field(), vec![Fr::from_u64(35), neg_44]);
        assert_eq!(*inputs[1].as_u64(), vec![1u64, 4]);
    }

    #[test]
    fn sparse_bind_preserves_sorted_keys() {
        let b = backend();
        let kernel = sparse_product_kernel();
        let scalar = Fr::from_u64(2);

        // Many entries across the key space
        let mut inputs: Vec<Buf<CpuBackend, Fr>> = vec![
            DeviceBuffer::Field(b.upload(&[Fr::one(); 6])),
            DeviceBuffer::U64(b.upload(&[1u64, 2, 3, 10, 11, 20])),
        ];

        b.bind(&kernel, &mut inputs, scalar);

        let keys = inputs[1].as_u64();
        // Keys 1(odd), (2,3)(paired), (10,11)(paired), 20(even)
        assert_eq!(keys, &[0, 1, 5, 10]);
        for w in keys.windows(2) {
            assert!(w[0] < w[1], "keys must remain sorted");
        }
    }

    /// Reference: manually pairs entries and evaluates `a * b` on the
    /// Toom-Cook grid {1, ∞}.
    fn reference_sparse_product(col_a: &[Fr], col_b: &[Fr], keys: &[u64]) -> Vec<Fr> {
        let n = keys.len();
        let mut s1 = Fr::zero();
        let mut s_inf = Fr::zero();
        let mut i = 0;
        while i < n {
            let key = keys[i];
            let (lo_a, lo_b, hi_a, hi_b);
            if key.is_multiple_of(2) {
                lo_a = col_a[i];
                lo_b = col_b[i];
                if i + 1 < n && keys[i + 1] == key + 1 {
                    hi_a = col_a[i + 1];
                    hi_b = col_b[i + 1];
                    i += 2;
                } else {
                    hi_a = Fr::zero();
                    hi_b = Fr::zero();
                    i += 1;
                }
            } else {
                lo_a = Fr::zero();
                lo_b = Fr::zero();
                hi_a = col_a[i];
                hi_b = col_b[i];
                i += 1;
            }
            // Toom-Cook d=2: P(1) = hi*hi, P(∞) = delta*delta
            s1 += hi_a * hi_b;
            s_inf += (hi_a - lo_a) * (hi_b - lo_b);
        }
        vec![s1, s_inf]
    }

    #[test]
    fn sparse_reduce_large_rayon() {
        let b = backend();
        let kernel = sparse_product_kernel();
        let mut rng = ChaCha20Rng::seed_from_u64(11111);

        // 2200 fully-paired entries → 1100 pairs > PAR_THRESHOLD (1024).
        let num_pairs = 1100;
        let keys: Vec<u64> = (0..num_pairs)
            .flat_map(|k| [k as u64 * 2, k as u64 * 2 + 1])
            .collect();
        let n = keys.len();

        let col_a: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let col_b: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let bufs: Vec<Buf<CpuBackend, Fr>> = vec![
            DeviceBuffer::Field(b.upload(&col_a)),
            DeviceBuffer::Field(b.upload(&col_b)),
            DeviceBuffer::U64(b.upload(&keys)),
        ];
        let buf_refs: Vec<&Buf<CpuBackend, Fr>> = bufs.iter().collect();
        let result = b.reduce(&kernel, &buf_refs, &[]);

        let expected = reference_sparse_product(&col_a, &col_b, &keys);
        assert_eq!(result, expected, "large sparse reduce mismatch");
    }

    #[test]
    fn sparse_bind_large_rayon() {
        let b = backend();
        let kernel = sparse_product_kernel();
        let scalar = Fr::from_u64(7);
        let mut rng = ChaCha20Rng::seed_from_u64(22222);

        // 2200 fully-paired entries → 1100 pairs > PAR_THRESHOLD.
        let num_pairs = 1100;
        let keys: Vec<u64> = (0..num_pairs)
            .flat_map(|k| [k as u64 * 2, k as u64 * 2 + 1])
            .collect();
        let n = keys.len();

        let col_a: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let mut inputs: Vec<Buf<CpuBackend, Fr>> = vec![
            DeviceBuffer::Field(b.upload(&col_a)),
            DeviceBuffer::U64(b.upload(&keys)),
        ];

        b.bind(&kernel, &mut inputs, scalar);

        let bound = inputs[0].as_field();
        let new_keys = inputs[1].as_u64();
        assert_eq!(bound.len(), num_pairs);
        assert_eq!(new_keys.len(), num_pairs);

        for i in 0..num_pairs {
            let lo = col_a[2 * i];
            let hi = col_a[2 * i + 1];
            let expected = lo + scalar * (hi - lo);
            assert_eq!(bound[i], expected, "bind mismatch at pair {i}");
            assert_eq!(new_keys[i], i as u64, "key mismatch at pair {i}");
        }
    }
}
