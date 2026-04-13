//! Core trait definitions for compute backends.

use std::collections::HashMap;

use jolt_compiler::module::{ClaimFormula, InstanceConfig};
pub use jolt_compiler::BindingOrder;
use jolt_compiler::PolynomialId;
use jolt_compiler::{KernelDef, KernelSpec};
use jolt_field::Field;
use jolt_instructions::LookupTableKind;

/// Per-cycle data needed for the prefix-suffix decomposition.
///
/// Extracted from the execution trace and passed to the backend's
/// prefix-suffix initialization.
pub struct LookupTraceData {
    /// Per-cycle lookup key (128-bit packed), T entries.
    pub lookup_keys: Vec<u128>,
    /// Per-cycle lookup table kind, T entries. None for cycles with no lookup.
    pub table_kinds: Vec<Option<LookupTableKind>>,
    /// Per-cycle interleaved-operands flag, T entries.
    pub is_interleaved: Vec<bool>,
}

/// Marker trait for types that can be stored in device buffers.
pub trait Scalar: Copy + Send + Sync + 'static {}
impl<T: Copy + Send + Sync + 'static> Scalar for T {}

/// Type-erased device buffer for heterogeneous storage in the runtime.
///
/// The runtime manages a flat array of `Option<DeviceBuffer<BufF, BufU64>>`,
/// indexed by polynomial ID. Different polynomials may have different scalar
/// types: field elements for evaluation tables, u64 for sparse keys/indices.
///
/// Parameterized by concrete buffer types (not the backend trait) to avoid
/// `Self`-in-generic-position issues within trait definitions.
///
/// # Type alias
///
/// Use [`Buf<B, F>`] as shorthand: `DeviceBuffer<B::Buffer<F>, B::Buffer<u64>>`.
pub enum DeviceBuffer<BufF, BufU64 = ()> {
    /// Field element buffer (polynomial evaluations, eq tables, etc.).
    Field(BufF),
    /// 64-bit unsigned integer buffer (sparse keys, indices).
    U64(BufU64),
}

/// Shorthand: `DeviceBuffer` specialized for a backend's buffer types.
pub type Buf<B, F> =
    DeviceBuffer<<B as ComputeBackend>::Buffer<F>, <B as ComputeBackend>::Buffer<u64>>;

impl<BufF, BufU64> DeviceBuffer<BufF, BufU64> {
    /// Borrow the inner field buffer. Panics if this is not `Field`.
    #[inline]
    pub fn as_field(&self) -> &BufF {
        match self {
            DeviceBuffer::Field(buf) => buf,
            DeviceBuffer::U64(_) => panic!("expected DeviceBuffer::Field"),
        }
    }

    /// Mutably borrow the inner field buffer. Panics if this is not `Field`.
    #[inline]
    pub fn as_field_mut(&mut self) -> &mut BufF {
        match self {
            DeviceBuffer::Field(buf) => buf,
            DeviceBuffer::U64(_) => panic!("expected DeviceBuffer::Field"),
        }
    }

    /// Borrow the inner u64 buffer. Panics if this is not `U64`.
    #[inline]
    pub fn as_u64(&self) -> &BufU64 {
        match self {
            DeviceBuffer::U64(buf) => buf,
            DeviceBuffer::Field(_) => panic!("expected DeviceBuffer::U64"),
        }
    }

    /// Mutably borrow the inner u64 buffer. Panics if this is not `U64`.
    #[inline]
    pub fn as_u64_mut(&mut self) -> &mut BufU64 {
        match self {
            DeviceBuffer::U64(buf) => buf,
            DeviceBuffer::Field(_) => panic!("expected DeviceBuffer::U64"),
        }
    }

    /// Returns `true` if this is a `Field` buffer.
    #[inline]
    pub fn is_field(&self) -> bool {
        matches!(self, DeviceBuffer::Field(_))
    }
}

/// Abstraction over a compute device (CPU, Metal GPU, CUDA GPU, WebGPU).
///
/// Provides typed buffer management and kernel compilation/dispatch.
/// The trait is intentionally thin — algorithmic decisions (iteration pattern,
/// evaluation grid, binding order) are captured in [`KernelSpec`] at compile
/// time and baked into the opaque [`CompiledKernel`](Self::CompiledKernel).
///
/// # Direction B: compiler decides algorithm, backend decides codegen
///
/// The compiler produces a [`KernelSpec`] describing WHAT to compute:
/// the composition formula, iteration pattern (dense/tensor),
/// evaluation grid, and binding order. The backend's [`compile`](Self::compile)
/// method produces a [`CompiledKernel`](Self::CompiledKernel) that bakes in
/// HOW to execute it: thread dispatch, memory layout, SIMD strategy, delayed
/// reduction, etc.
///
/// At dispatch time ([`reduce`](Self::reduce), [`bind`](Self::bind)), only the
/// actual buffer data and challenge values are provided. The compiled kernel
/// knows everything else.
///
/// # Associated Types
///
/// - [`Buffer`](Self::Buffer) — typed buffer handle on the device.
///   For CPU: `Vec<T>`. For GPU: wraps device memory.
///
/// - [`CompiledKernel`](Self::CompiledKernel) — opaque compiled kernel.
///   Captures the full [`KernelSpec`] in backend-native form. For CPU:
///   an eval closure + metadata. For GPU: a compute pipeline.
///
/// # Zero Cost for CPU
///
/// `CpuBackend` implements this trait with `Buffer<T> = Vec<T>`. After
/// monomorphization, every trait method compiles to a direct function call
/// with no vtable indirection.
pub trait ComputeBackend: Send + Sync + 'static {
    /// Handle to a typed buffer on the device.
    type Buffer<T: Scalar>: Send + Sync;

    /// Opaque compiled kernel produced from a [`KernelSpec`].
    ///
    /// Captures formula structure, iteration pattern, evaluation grid,
    /// and binding order — everything except actual data and challenge
    /// values, which are provided at dispatch time.
    type CompiledKernel<F: Field>: Send + Sync;

    /// Compile a [`KernelSpec`] into backend-native code.
    ///
    /// Bakes in the formula, iteration pattern, evaluation grid, and
    /// binding order. Challenge values are provided at dispatch time.
    fn compile<F: Field>(&self, spec: &KernelSpec) -> Self::CompiledKernel<F>;

    /// Composition-reduce: evaluate the kernel over all input positions
    /// and return accumulated evaluation sums.
    ///
    /// The compiled kernel determines iteration pattern, grid size, and
    /// binding order. Inputs follow the layout convention defined by the
    /// kernel's [`Iteration`](jolt_compiler::Iteration) variant:
    ///
    /// | Positions | Contents |
    /// |-----------|----------|
    /// | `0 .. formula.num_inputs` | Formula value columns |
    /// | After value columns | Extra inputs (tensor eq weights) |
    ///
    /// Returns `num_evals` field elements (evaluation sums on the grid).
    fn reduce<F: Field>(
        &self,
        kernel: &Self::CompiledKernel<F>,
        inputs: &[&Buf<Self, F>],
        challenges: &[F],
    ) -> Vec<F>;

    /// Bind one variable across all kernel inputs.
    ///
    /// Each buffer is halved via pairwise interpolation at `scalar`.
    /// The compiled kernel determines the binding strategy:
    /// - Dense: standard `lo + scalar × (hi − lo)` interpolation
    /// - Tensor: interpolate value columns AND eq buffers
    fn bind<F: Field>(
        &self,
        kernel: &Self::CompiledKernel<F>,
        inputs: &mut [Buf<Self, F>],
        scalar: F,
    );

    /// Standalone dense interpolation for post-sumcheck variable binding.
    ///
    /// Halves the buffer via `lo + scalar × (hi − lo)`. Used by `Op::Bind`
    /// for survivor polynomials that don't belong to any active kernel.
    fn interpolate_inplace<F: Field>(
        &self,
        buf: &mut Self::Buffer<F>,
        scalar: F,
        order: BindingOrder,
    );

    /// Upload host data to a device buffer.
    fn upload<T: Scalar>(&self, data: &[T]) -> Self::Buffer<T>;

    /// Download device buffer contents to host memory.
    fn download<T: Scalar>(&self, buf: &Self::Buffer<T>) -> Vec<T>;

    /// Allocate a zero-initialized buffer on the device.
    fn alloc<T: Scalar>(&self, len: usize) -> Self::Buffer<T>;

    /// Active element count of a buffer.
    fn len<T: Scalar>(&self, buf: &Self::Buffer<T>) -> usize;

    /// Eq product table: `eq(r, x) = Π(rᵢxᵢ + (1−rᵢ)(1−xᵢ))`.
    fn eq_table<F: Field>(&self, point: &[F]) -> Self::Buffer<F>;

    /// Less-than table: `LT(x, r)` multilinear extension.
    fn lt_table<F: Field>(&self, point: &[F]) -> Self::Buffer<F>;

    /// Eq-plus-one table: `(eq_evals, eq_plus_one_evals)`.
    fn eq_plus_one_table<F: Field>(&self, point: &[F]) -> (Self::Buffer<F>, Self::Buffer<F>);

    /// Interleave-duplicate: `result[2i] = result[2i+1] = buf[i]`.
    ///
    /// Returns a buffer twice the size of the input, representing a
    /// multilinear polynomial that does not depend on a new low-order
    /// variable (streaming extension for outer Spartan).
    fn duplicate_interleave<F: Field>(&self, buf: &Self::Buffer<F>) -> Self::Buffer<F>;

    /// Regroup constraint buffers for group-split uniskip.
    ///
    /// Input: `buf[cycle * old_stride + constraint_idx]` for `num_cycles` cycles.
    /// Output: `result[(2 * cycle + group) * new_stride + k]` (interleaved groups).
    ///
    /// `group_indices[g][k]` maps group `g`, position `k` → original constraint index.
    /// Groups shorter than `new_stride` are zero-padded.
    fn regroup_constraints<F: Field>(
        &self,
        buf: &Self::Buffer<F>,
        group_indices: &[Vec<usize>],
        old_stride: usize,
        new_stride: usize,
        num_cycles: usize,
    ) -> Self::Buffer<F>;

    /// Evaluate a [`ClaimFormula`] against polynomial evaluations and challenges.
    fn evaluate_claim<F: Field>(
        &self,
        formula: &ClaimFormula,
        evaluations: &HashMap<PolynomialId, F>,
        challenges: &[F],
    ) -> F;

    /// Evaluate a multilinear extension at a point via repeated halving.
    fn evaluate_mle<F: Field>(&self, evals: &[F], point: &[F]) -> F;

    /// Encode Uniskip round polynomial into transcript-ready coefficients.
    ///
    /// Computes Lagrange interpolation of `raw_evals` on a domain starting at
    /// `domain_start` with `domain_size` points, multiplies by the Lagrange
    /// basis evaluated at `tau`, and truncates/pads to `num_coeffs`.
    fn uniskip_encode<F: Field>(
        &self,
        raw_evals: &mut [F],
        domain_size: usize,
        domain_start: i64,
        tau: F,
        zero_base: bool,
        num_coeffs: usize,
    ) -> Vec<F>;

    /// Encode round polynomial evaluations into monomial coefficients.
    ///
    /// Interpolates evaluations at `{0, 1, ..., n-1}` to monomial form.
    fn compressed_encode<F: Field>(&self, evals: &[F]) -> Vec<F>;

    /// Interpolate evaluations at `{0, 1, ..., n-1}` and evaluate at `point`.
    fn interpolate_evaluate<F: Field>(&self, evals: &[F], point: F) -> F;

    /// Extend evaluations at `{0, ..., n-1}` to `{0, ..., target_len-1}`.
    fn extend_evals<F: Field>(&self, evals: &[F], target_len: usize) -> Vec<F>;

    /// Element-wise scale host data and produce a device buffer.
    fn scale_from_host<F: Field>(&self, data: &[F], scale: F) -> Self::Buffer<F>;

    /// Transpose a row-major matrix from host data and produce a device buffer.
    fn transpose_from_host<F: Field>(
        &self,
        data: &[F],
        rows: usize,
        cols: usize,
    ) -> Self::Buffer<F>;

    /// Build eq table from `eq_point`, then gather by index from host data.
    ///
    /// For each `i`, output\[i\] = eq_table\[indices\[i\]\] (or zero if out of bounds).
    fn eq_gather<F: Field>(&self, eq_point: &[F], index_data: &[F]) -> Self::Buffer<F>;

    /// Build eq table from `eq_point`, then scatter-add into an output buffer.
    ///
    /// For each `j`, output\[indices\[j\]\] += eq_table\[j\].
    fn eq_pushforward<F: Field>(
        &self,
        eq_point: &[F],
        index_data: &[F],
        output_size: usize,
    ) -> Self::Buffer<F>;

    /// Eq-weighted matrix-vector projection.
    ///
    /// Source layout: `inner_size` rows × `outer_size` cols.
    /// If `eq_point.len() == log2(inner_size)`, projects out inner dim → `outer_size` result.
    /// If `eq_point.len() == log2(outer_size)`, projects out outer dim → `inner_size` result.
    fn eq_project<F: Field>(
        &self,
        source_data: &[F],
        eq_point: &[F],
        inner_size: usize,
        outer_size: usize,
    ) -> Self::Buffer<F>;

    /// Lagrange basis projection with optional kernel tau scaling.
    ///
    /// For each polynomial buffer, projects data through Lagrange basis evaluated
    /// at `challenge`, regrouped by `group_offsets` within `stride`-sized blocks.
    /// Optionally scales by `L(τ, r) = Σ_k L_k(τ) · L_k(r)`.
    ///
    /// Returns `num_cycles × num_groups` elements per buffer.
    #[allow(clippy::too_many_arguments)]
    fn lagrange_project<F: Field>(
        &self,
        buf: &Self::Buffer<F>,
        challenge: F,
        domain_start: i64,
        domain_size: usize,
        stride: usize,
        group_offsets: &[usize],
        scale: F,
    ) -> Self::Buffer<F>;

    /// Fused segmented reduce over mixed-dimensional inputs.
    ///
    /// For each outer position, extracts inner-sized columns from mixed inputs
    /// (inner-only inputs are shared across all positions), runs the kernel,
    /// and accumulates with outer eq weights. Returns `num_evals` field elements.
    #[allow(clippy::too_many_arguments)]
    fn segmented_reduce<F: Field>(
        &self,
        kernel: &Self::CompiledKernel<F>,
        inputs: &[&Self::Buffer<F>],
        outer_eq: &[F],
        inner_only: &[bool],
        inner_size: usize,
        challenges: &[F],
    ) -> Vec<F>;

    /// Opaque state for a stateful sumcheck instance.
    ///
    /// Replaces `PrefixSuffixState`, `BooleanityState`, `HwReductionState`.
    /// The backend internally dispatches based on `InstanceConfig` variant.
    type InstanceState<F: Field>: Send + Sync;

    /// Initialize a stateful sumcheck instance from config + runtime state.
    fn instance_init<F: Field>(
        &self,
        config: &InstanceConfig,
        challenges: &[F],
        provider: &mut dyn BufferProvider<F>,
        lookup_trace: Option<&LookupTraceData>,
        kernels: &[KernelDef],
    ) -> Self::InstanceState<F>;

    /// Bind a sumcheck challenge into the instance state.
    fn instance_bind<F: Field>(&self, state: &mut Self::InstanceState<F>, challenge: F);

    /// Compute round polynomial evaluations. Returns the full eval vector
    /// (e.g., [eval_0, eval_1, eval_2] for degree 2).
    fn instance_reduce<F: Field>(
        &self,
        state: &Self::InstanceState<F>,
        previous_claim: F,
    ) -> Vec<F>;

    /// Finalize the instance: consume state, return buffers and/or evaluations.
    fn instance_finalize<F: Field>(
        &self,
        state: Self::InstanceState<F>,
    ) -> InstanceOutput<Self::Buffer<F>, F>;
}

/// Output from finalizing a stateful sumcheck instance.
///
/// Returns unlabeled buffers and evaluations in order. The runtime maps
/// these to `PolynomialId`s using the `output_buffers` / `output_evals`
/// lists from the `UnifiedInstanceFinalize` Op.
pub struct InstanceOutput<Buf, F> {
    pub buffers: Vec<Buf>,
    pub evaluations: Vec<F>,
}

/// Materializes polynomial data for the prover runtime.
///
/// The runtime calls [`materialize`](BufferProvider::materialize) whenever it
/// needs polynomial data — for device upload (compute ops) or direct host
/// access (PCS ops). The provider is backend-agnostic: it returns raw field
/// data and the runtime decides how to consume it.
///
/// Returns [`Cow::Borrowed`] for stored polynomials (zero-copy) and
/// [`Cow::Owned`] for computed polynomials (R1CS, virtual).
pub trait BufferProvider<F: Field> {
    /// Materialize polynomial data for the given ID.
    fn materialize(&self, poly_id: PolynomialId) -> std::borrow::Cow<'_, [F]>;

    /// Release stored polynomial data to reclaim memory.
    fn release(&mut self, _poly_id: PolynomialId) {}
}
