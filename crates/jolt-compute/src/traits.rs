//! Core trait definitions for compute backends.

pub use jolt_compiler::BindingOrder;
use jolt_compiler::CompositionFormula;
use jolt_field::Field;

/// Round polynomial coefficients in ascending degree order.
///
/// Lightweight representation of a univariate polynomial $p(x) = \sum_i c_i x^i$
/// produced by [`BackendWitness::round_polynomial`]. This avoids a dependency on
/// `jolt-poly` in the compute layer. Conversion to/from `UnivariatePoly<F>` is
/// a zero-cost move of the inner `Vec<F>`.
#[derive(Clone, Debug)]
pub struct RoundCoeffs<F: Field> {
    /// Coefficients in ascending degree order: index `i` holds $c_i$.
    pub coeffs: Vec<F>,
}

impl<F: Field> RoundCoeffs<F> {
    #[inline]
    pub fn new(coeffs: Vec<F>) -> Self {
        Self { coeffs }
    }
}

/// Witness trait for sumcheck, defined in the compute layer.
///
/// Structurally identical to `jolt_sumcheck::SumcheckCompute<F>` but avoids
/// pulling `jolt-poly`, `jolt-openings`, and `jolt-transcript` into
/// `jolt-compute`. Concrete witness types (e.g., `CpuSumcheckWitness`) implement
/// both this trait and `SumcheckCompute` — the latter delegates to the former.
///
/// The prover engine in `jolt-sumcheck` accepts `SumcheckCompute`; the backend
/// factory in `ComputeBackend::make_witness` produces `BackendWitness`. Bridging
/// is a one-line blanket impl in the crate that has both traits in scope.
pub trait BackendWitness<F: Field>: Send + Sync {
    /// Computes the round polynomial $s_i(X)$ for the current round.
    fn round_polynomial(&self) -> RoundCoeffs<F>;

    /// Fixes the current leading variable to `challenge`, reducing the
    /// witness by one variable.
    fn bind(&mut self, challenge: F);

    /// Provides the running sumcheck claim before each round.
    ///
    /// Called with the current running sum. Witnesses that derive evaluation
    /// points from the claim (e.g., `P(1) = claim - P(0)`) should override.
    fn set_claim(&mut self, _claim: F) {}

    /// Optional first-round polynomial override (univariate skip).
    fn first_round_polynomial(&self) -> Option<RoundCoeffs<F>> {
        None
    }

    /// Per-polynomial evaluations at the fully-bound challenge point.
    ///
    /// Returns `(index, eval)` pairs where `index` identifies the polynomial
    /// within this witness.
    fn produced_evaluations(&self) -> Vec<(usize, F)> {
        vec![]
    }
}

/// Marker trait for types that can be stored in device buffers.
///
/// Matches `jolt-poly`'s `Polynomial<T>` philosophy: buffers hold any
/// scalar type, not just field elements. Compact types (`u8`, `bool`)
/// use less device memory and are promoted to `F` inside kernels.
///
/// Blanket-implemented for all `Copy + Send + Sync + 'static` types,
/// covering primitive integers, `bool`, and all [`Field`] types.
pub trait Scalar: Copy + Send + Sync + 'static {}
impl<T: Copy + Send + Sync + 'static> Scalar for T {}

/// Eq-polynomial weighting mode for composition-reduce operations.
///
/// Controls how per-pair weights are applied during
/// [`pairwise_reduce`](ComputeBackend::pairwise_reduce):
///
/// - [`Unit`](EqInput::Unit) — implicit all-ones weights. Avoids allocating
///   a weight buffer and skips per-pair multiply. Used when eq is an input
///   buffer (standard-grid sumchecks).
///
/// - [`Weighted`](EqInput::Weighted) — explicit weight buffer. Used when eq
///   is factored out of the kernel and applied as per-pair scaling (Toom-Cook
///   sumchecks).
///
/// - [`Tensor`](EqInput::Tensor) — split-eq tensor product
///   `w(x_out, x_in) = outer[x_out] · inner[x_in]`. Saves memory and
///   enables cache-friendly nested iteration. Used by Spartan outer sumcheck.
pub enum EqInput<'a, B: ComputeBackend + ?Sized, F: Field> {
    Unit,
    Weighted(&'a B::Buffer<F>),
    Tensor {
        outer: &'a B::Buffer<F>,
        inner: &'a B::Buffer<F>,
    },
}

impl<B: ComputeBackend + ?Sized, F: Field> Clone for EqInput<'_, B, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B: ComputeBackend + ?Sized, F: Field> Copy for EqInput<'_, B, F> {}

/// Abstraction over a compute device (CPU, Metal GPU, CUDA GPU, WebGPU).
///
/// Provides typed buffer management and parallel primitives. All methods
/// are named for what they compute, not what protocol uses them.
///
/// # Associated Types
///
/// - [`Buffer`](Self::Buffer) — handle to a typed, contiguous buffer on
///   the device. For CPU this is `Vec<T>`. For GPU this wraps a device
///   memory allocation.
///
/// - [`CompiledKernel`](Self::CompiledKernel) — opaque compiled kernel for
///   [`pairwise_reduce`](Self::pairwise_reduce). Compiled from a
///   [`CompositionFormula`] via [`compile_kernel`](Self::compile_kernel).
///
/// # Zero Cost for CPU
///
/// `CpuBackend` (in the `jolt-cpu` crate) implements this trait with
/// `Buffer<T> = Vec<T>`. After monomorphization, every trait method call
/// compiles to a direct function call with no indirection — identical
/// codegen to hand-written Rayon code.
///
/// # Delayed Reduction
///
/// Both CPU and GPU backends use delayed modular reduction internally:
/// multiply-add results are accumulated as wide integers and reduced once
/// per accumulation group. This is an implementation detail of
/// [`pairwise_reduce`](Self::pairwise_reduce) and is not exposed in the
/// trait interface.
pub trait ComputeBackend: Send + Sync + 'static {
    /// Handle to a typed buffer on the device.
    ///
    /// For CPU: `Vec<T>`. For Metal: wraps `MTLBuffer`. For CUDA: wraps
    /// `CUdeviceptr` with type and length metadata.
    type Buffer<T: Scalar>: Send + Sync;

    /// Opaque compiled kernel for composition-reduce operations.
    ///
    /// Parameterized by the field type so that each kernel is compiled for
    /// a specific field's arithmetic. For CPU: `CpuKernel<F>` wrapping a
    /// closure. For Metal: `MetalKernel<F>` wrapping pipeline states.
    type CompiledKernel<F: Field>: Send + Sync;

    /// Backend-created sumcheck witness that pairs a compiled kernel with
    /// runtime data. The backend controls the internal representation (dense
    /// buffers on CPU, device-side buffers on GPU) while exposing the
    /// standard round-polynomial / bind interface.
    type SumcheckWitness<F: Field>: BackendWitness<F>;

    /// Sparse buffer for non-dense polynomial representations.
    ///
    /// Stores `(row_index, column_values)` pairs where each entry contributes
    /// to a small number of hypercube positions. For CPU this is
    /// `Vec<(usize, Vec<F>)>`; GPU backends may use CSR-style device buffers.
    type SparseBuffer<F: Field>: Send + Sync;

    /// Compile a [`CompositionFormula`] into a backend-specific kernel.
    fn compile_kernel<F: Field>(&self, formula: &CompositionFormula) -> Self::CompiledKernel<F> {
        self.compile_kernel_with_challenges(formula, &[])
    }

    /// Compile a [`CompositionFormula`] with baked challenge values.
    ///
    /// `challenges[i]` is substituted for `Factor::Challenge(i)` in the
    /// formula. For pure product-sum formulas (no challenge factors),
    /// challenges are ignored.
    fn compile_kernel_with_challenges<F: Field>(
        &self,
        formula: &CompositionFormula,
        challenges: &[F],
    ) -> Self::CompiledKernel<F>;

    /// Upload host data to a device buffer.
    fn upload<T: Scalar>(&self, data: &[T]) -> Self::Buffer<T>;

    /// Download device buffer contents to host memory.
    fn download<T: Scalar>(&self, buf: &Self::Buffer<T>) -> Vec<T>;

    /// Allocate a buffer on the device, initialized to zero bytes.
    fn alloc<T: Scalar>(&self, len: usize) -> Self::Buffer<T>;

    /// Active element count of a buffer.
    fn len<T: Scalar>(&self, buf: &Self::Buffer<T>) -> usize;

    fn is_empty<T: Scalar>(&self, buf: &Self::Buffer<T>) -> bool {
        self.len::<T>(buf) == 0
    }

    /// Pairwise linear interpolation, halving the buffer.
    ///
    /// For each `i` in `[0, n/2)` where `n` is the active buffer length:
    /// $$\text{out}_i = \text{buf}_{2i} + \text{scalar} \cdot (\text{buf}_{2i+1} - \text{buf}_{2i})$$
    ///
    /// When `T` is a compact type (e.g., `u8`), elements are promoted to `F`
    /// during the operation via `F: From<T>`. The returned buffer has element
    /// type `F` and half the original length.
    ///
    /// When `T = F`, the `From` conversion is the identity and the compiler
    /// eliminates it, making this equivalent to an in-place bind.
    fn interpolate_pairs<T, F>(&self, buf: Self::Buffer<T>, scalar: F) -> Self::Buffer<F>
    where
        T: Scalar,
        F: Field + From<T>;

    /// Composition-reduce over paired inputs from multiple buffers.
    ///
    /// For each position `i` in `[0, n/2)` where `n` is the active buffer
    /// length:
    ///
    /// 1. Reads pairs from all `k` input buffers according to `order`:
    ///    - [`LowToHigh`](BindingOrder::LowToHigh): `(inputs[k][2i], inputs[k][2i+1])`
    ///    - [`HighToLow`](BindingOrder::HighToLow): `(inputs[k][i], inputs[k][i + n/2])`
    /// 2. Executes the compiled kernel on those pairs, producing
    ///    `num_evals` values
    /// 3. Multiplies each value by the eq weight for this position
    ///    (see [`EqInput`])
    /// 4. Accumulates into `num_evals` running sums
    ///
    /// Returns `num_evals` field elements after reducing across all
    /// positions. This is the only device-to-host transfer per invocation.
    ///
    /// For Toom-Cook kernels (ProductSum), `num_evals = D` (evaluations on
    /// the grid `{1, ..., D-1, ∞}`). For standard-grid kernels,
    /// `num_evals = degree` (evaluations on `{0, 2, 3, ..., degree}`).
    ///
    /// Both the kernel evaluation and the reduction use delayed modular
    /// reduction internally.
    fn pairwise_reduce<F: Field>(
        &self,
        inputs: &[&Self::Buffer<F>],
        eq: EqInput<'_, Self, F>,
        kernel: &Self::CompiledKernel<F>,
        num_evals: usize,
        order: BindingOrder,
    ) -> Vec<F>;

    /// Batched pairwise linear interpolation across multiple buffers.
    ///
    /// Equivalent to calling [`interpolate_pairs`](Self::interpolate_pairs) on
    /// each buffer individually, but enables backends to parallelize across
    /// all buffers in a single dispatch. This improves work-stealing
    /// granularity when individual buffers are small (e.g., after several
    /// sumcheck rounds).
    fn interpolate_pairs_batch<F: Field>(
        &self,
        bufs: Vec<Self::Buffer<F>>,
        scalar: F,
    ) -> Vec<Self::Buffer<F>> {
        bufs.into_iter()
            .map(|buf| self.interpolate_pairs(buf, scalar))
            .collect()
    }

    /// In-place pairwise linear interpolation, halving the buffer.
    ///
    /// For [`LowToHigh`](BindingOrder::LowToHigh) order (interleaved pairs):
    /// $$\text{buf}_i \leftarrow \text{buf}_{2i} + s \cdot (\text{buf}_{2i+1} - \text{buf}_{2i})$$
    ///
    /// For [`HighToLow`](BindingOrder::HighToLow) order (split-half pairs):
    /// $$\text{buf}_i \leftarrow \text{buf}_i + s \cdot (\text{buf}_{i + n/2} - \text{buf}_i)$$
    ///
    /// The buffer is truncated to half its original length. Only works for
    /// field-to-field binding (no compact type promotion).
    fn interpolate_pairs_inplace<F: Field>(
        &self,
        buf: &mut Self::Buffer<F>,
        scalar: F,
        order: BindingOrder,
    );

    /// In-place batched pairwise interpolation across multiple buffers.
    ///
    /// Equivalent to calling [`interpolate_pairs_inplace`](Self::interpolate_pairs_inplace)
    /// on each buffer, but enables inter-buffer parallelism.
    fn interpolate_pairs_batch_inplace<F: Field>(
        &self,
        bufs: &mut [Self::Buffer<F>],
        scalar: F,
        order: BindingOrder,
    ) {
        for buf in bufs.iter_mut() {
            self.interpolate_pairs_inplace(buf, scalar, order);
        }
    }

    /// Const-generic composition-reduce with stack-allocated scratch.
    ///
    /// Identical to [`pairwise_reduce`](Self::pairwise_reduce) but with
    /// compile-time-known `D`, enabling stack-allocated accumulators and
    /// evaluation scratch arrays. Returns `[F; D]` instead of `Vec<F>`.
    ///
    /// Specialized for D ∈ {4, 8, 16, 32} which cover ~80% of prover time.
    fn pairwise_reduce_fixed<F: Field, const D: usize>(
        &self,
        inputs: &[&Self::Buffer<F>],
        eq: EqInput<'_, Self, F>,
        kernel: &Self::CompiledKernel<F>,
        order: BindingOrder,
    ) -> [F; D] {
        let v = self.pairwise_reduce(inputs, eq, kernel, D, order);
        core::array::from_fn(|i| v[i])
    }

    /// Evaluates multiple kernels over the same inputs and eq weights in a
    /// single pass.
    ///
    /// Each `(kernel, num_evals)` pair shares the same input buffers and
    /// eq weights. The data is read once per position and each kernel is
    /// evaluated, saving cache misses compared to calling
    /// [`pairwise_reduce`](Self::pairwise_reduce) independently per kernel.
    fn pairwise_reduce_multi<F: Field>(
        &self,
        inputs: &[&Self::Buffer<F>],
        eq: EqInput<'_, Self, F>,
        kernels: &[(&Self::CompiledKernel<F>, usize)],
        order: BindingOrder,
    ) -> Vec<Vec<F>> {
        kernels
            .iter()
            .map(|(kernel, num_evals)| self.pairwise_reduce(inputs, eq, kernel, *num_evals, order))
            .collect()
    }

    /// Multiplicative product table over the Boolean hypercube.
    ///
    /// Computes $2^n$ evaluations where $n = \text{point.len()}$:
    /// $$\text{out}_x = \prod_{i=0}^{n-1} \bigl(r_i \cdot x_i + (1 - r_i)(1 - x_i)\bigr)$$
    /// for all `x` in `{0,1}^n`, where `x_i` is the `i`-th bit of `x`.
    ///
    /// On GPU this is constructed on-device, avoiding a $2^n$ field-element
    /// transfer across the bus. On CPU this is equivalent to
    /// `EqPolynomial::evaluations()`.
    fn product_table<F: Field>(&self, point: &[F]) -> Self::Buffer<F>;

    /// Sum all elements in a buffer: $\sum_i \text{buf}_i$.
    ///
    /// Uses delayed modular reduction internally for large buffers.
    fn sum<F: Field>(&self, buf: &Self::Buffer<F>) -> F;

    /// Dot product of two buffers: $\sum_i a_i \cdot b_i$.
    ///
    /// Both buffers must have equal length. Uses delayed modular reduction.
    fn dot_product<F: Field>(&self, a: &Self::Buffer<F>, b: &Self::Buffer<F>) -> F;

    /// Multiply every element by a scalar: `buf[i] *= scalar`.
    fn scale<F: Field>(&self, buf: &mut Self::Buffer<F>, scalar: F);

    /// Element-wise addition: `out[i] = a[i] + b[i]`.
    ///
    /// Both buffers must have equal length. Returns a new buffer.
    fn add<F: Field>(&self, a: &Self::Buffer<F>, b: &Self::Buffer<F>) -> Self::Buffer<F>;

    /// Element-wise subtraction: `out[i] = a[i] - b[i]`.
    ///
    /// Both buffers must have equal length. Returns a new buffer.
    fn sub<F: Field>(&self, a: &Self::Buffer<F>, b: &Self::Buffer<F>) -> Self::Buffer<F>;

    /// Fused multiply-add into buffer: `buf[i] += scalar * other[i]`.
    ///
    /// Both buffers must have equal length.
    fn accumulate<F: Field>(&self, buf: &mut Self::Buffer<F>, scalar: F, other: &Self::Buffer<F>);

    /// Weighted accumulation from multiple buffers:
    /// `buf[i] += Σ_k scalars[k] * inputs[k][i]`.
    ///
    /// All buffers must have equal length. `scalars.len()` must equal
    /// `inputs.len()`. Equivalent to calling [`accumulate`](Self::accumulate)
    /// in a loop, but enables backends to fuse the operation into a single
    /// pass over memory and a single GPU dispatch.
    fn accumulate_weighted<F: Field>(
        &self,
        buf: &mut Self::Buffer<F>,
        scalars: &[F],
        inputs: &[&Self::Buffer<F>],
    ) {
        debug_assert_eq!(scalars.len(), inputs.len());
        for (&s, &input) in scalars.iter().zip(inputs.iter()) {
            self.accumulate(buf, s, input);
        }
    }

    /// Multiply every element of each buffer by the same scalar.
    ///
    /// Equivalent to calling [`scale`](Self::scale) on each buffer, but
    /// enables backends to parallelize across all buffers in a single
    /// dispatch.
    fn scale_batch<F: Field>(&self, bufs: &mut [Self::Buffer<F>], scalar: F) {
        for buf in bufs.iter_mut() {
            self.scale(buf, scalar);
        }
    }

    /// Fused in-place interpolation + weighted composition-reduce (H2L).
    ///
    /// Combines [`interpolate_pairs_batch_inplace`](Self::interpolate_pairs_batch_inplace)
    /// and [`pairwise_reduce`](Self::pairwise_reduce) into a single pass over
    /// memory. After completion, each input buffer and the weight buffer are
    /// halved (interpolated in-place). Returns `num_evals` eval sums.
    ///
    /// Saves one full read pass over all input buffers compared to separate
    /// interpolate + reduce dispatches.
    fn fused_interpolate_reduce<F: Field>(
        &self,
        inputs: &mut [Self::Buffer<F>],
        weights: &mut Self::Buffer<F>,
        interpolation_scalar: F,
        kernel: &Self::CompiledKernel<F>,
        num_evals: usize,
    ) -> Vec<F> {
        self.interpolate_pairs_batch_inplace(inputs, interpolation_scalar, BindingOrder::HighToLow);
        self.interpolate_pairs_inplace(weights, interpolation_scalar, BindingOrder::HighToLow);
        let refs: Vec<_> = inputs.iter().collect();
        self.pairwise_reduce(
            &refs,
            EqInput::Weighted(weights),
            kernel,
            num_evals,
            BindingOrder::HighToLow,
        )
    }

    // -- Witness factory -------------------------------------------------

    /// Creates a sumcheck witness by pairing a pre-compiled kernel with
    /// runtime input buffers and challenge values.
    ///
    /// This replaces the old `build_witness()` + `KernelEvaluator` pattern:
    /// the backend owns both the data layout and the evaluation strategy,
    /// returning an opaque witness that implements [`BackendWitness`].
    fn make_witness<F: Field>(
        &self,
        kernel: &Self::CompiledKernel<F>,
        inputs: Vec<Self::Buffer<F>>,
        challenges: &[F],
    ) -> Self::SumcheckWitness<F>;

    // -- Sparse buffer primitives ----------------------------------------

    /// Upload sparse entries to a device-side buffer.
    ///
    /// Each entry is `(row_index, column_values)` where `row_index`
    /// identifies a hypercube position and `column_values` holds the
    /// polynomial evaluations at that position.
    fn upload_sparse<F: Field>(&self, entries: &[(usize, Vec<F>)]) -> Self::SparseBuffer<F>;

    /// Sparse sumcheck reduction: evaluates the kernel over sparse entries.
    ///
    /// Analogous to [`pairwise_reduce`](Self::pairwise_reduce) but only
    /// iterates over populated positions in the sparse buffer, scaling each
    /// by the corresponding eq weight.
    fn sparse_reduce<F: Field>(
        &self,
        entries: &Self::SparseBuffer<F>,
        eq: &Self::Buffer<F>,
        kernel: &Self::CompiledKernel<F>,
        challenges: &[F],
        num_evals: usize,
    ) -> Vec<F>;

    /// Bind a sparse buffer at a challenge point, halving the index space.
    ///
    /// Analogous to [`interpolate_pairs_inplace`](Self::interpolate_pairs_inplace)
    /// but operates on the sparse representation, merging entries whose indices
    /// differ only in the bound variable.
    fn sparse_bind<F: Field>(
        &self,
        entries: &mut Self::SparseBuffer<F>,
        challenge: F,
        order: BindingOrder,
    );
}
