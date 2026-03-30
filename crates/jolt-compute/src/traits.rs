//! Core trait definitions for compute backends.

pub use jolt_compiler::BindingOrder;
use jolt_compiler::Formula;
use jolt_field::Field;

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
///   [`Formula`] via [`compile_kernel`](Self::compile_kernel). Challenge
///   values are passed at dispatch time, not at compilation.
///
/// # Zero Cost for CPU
///
/// `CpuBackend` (in the `jolt-cpu` crate) implements this trait with
/// `Buffer<T> = Vec<T>`. After monomorphization, every trait method call
/// compiles to a direct function call with no indirection.
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
    /// a specific field's arithmetic. Captures the formula *structure*
    /// (which inputs, which challenge slots) but not challenge *values* —
    /// those are passed at dispatch via [`pairwise_reduce`](Self::pairwise_reduce).
    type CompiledKernel<F: Field>: Send + Sync;

    // -- Kernel compilation ------------------------------------------------

    /// Compile a [`Formula`] into a backend-specific kernel.
    ///
    /// The compiled kernel captures the formula's structure: how many inputs,
    /// which challenge slots are referenced, the sum-of-products shape. Actual
    /// challenge values are provided at dispatch time via `pairwise_reduce`.
    fn compile_kernel<F: Field>(&self, formula: &Formula) -> Self::CompiledKernel<F>;

    // -- Buffer management -------------------------------------------------

    /// Upload host data to a device buffer.
    fn upload<T: Scalar>(&self, data: &[T]) -> Self::Buffer<T>;

    /// Download device buffer contents to host memory.
    fn download<T: Scalar>(&self, buf: &Self::Buffer<T>) -> Vec<T>;

    /// Allocate a buffer on the device, initialized to zero bytes.
    fn alloc<T: Scalar>(&self, len: usize) -> Self::Buffer<T>;

    /// Active element count of a buffer.
    fn len<T: Scalar>(&self, buf: &Self::Buffer<T>) -> usize;

    // -- Interpolation (variable binding) ----------------------------------

    /// Pairwise linear interpolation, halving the buffer.
    ///
    /// For each `i` in `[0, n/2)`:
    /// $$\text{out}_i = \text{buf}_{2i} + \text{scalar} \cdot (\text{buf}_{2i+1} - \text{buf}_{2i})$$
    ///
    /// When `T` is a compact type (e.g., `u8`), elements are promoted to `F`
    /// via `F: From<T>`. The returned buffer has element type `F` and half
    /// the original length.
    fn interpolate_pairs<T, F>(&self, buf: Self::Buffer<T>, scalar: F) -> Self::Buffer<F>
    where
        T: Scalar,
        F: Field + From<T>;

    /// In-place pairwise linear interpolation, halving the buffer.
    ///
    /// For [`LowToHigh`](BindingOrder::LowToHigh) (interleaved pairs):
    /// $$\text{buf}_i \leftarrow \text{buf}_{2i} + s \cdot (\text{buf}_{2i+1} - \text{buf}_{2i})$$
    ///
    /// For [`HighToLow`](BindingOrder::HighToLow) (split-half pairs):
    /// $$\text{buf}_i \leftarrow \text{buf}_i + s \cdot (\text{buf}_{i + n/2} - \text{buf}_i)$$
    ///
    /// The buffer is truncated to half its original length.
    fn interpolate_pairs_inplace<F: Field>(
        &self,
        buf: &mut Self::Buffer<F>,
        scalar: F,
        order: BindingOrder,
    );

    /// Batched pairwise interpolation (LowToHigh, type-promoting).
    ///
    /// Equivalent to calling [`interpolate_pairs`](Self::interpolate_pairs) on
    /// each buffer, but enables inter-buffer parallelism.
    fn interpolate_pairs_batch<F: Field>(
        &self,
        bufs: Vec<Self::Buffer<F>>,
        scalar: F,
    ) -> Vec<Self::Buffer<F>> {
        bufs.into_iter()
            .map(|buf| self.interpolate_pairs(buf, scalar))
            .collect()
    }

    /// In-place batched pairwise interpolation.
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

    // -- Composition-reduce (core dispatch) --------------------------------

    /// Composition-reduce over paired inputs from multiple buffers.
    ///
    /// For each position `i` in `[0, n/2)`:
    ///
    /// 1. Reads pairs from all `k` input buffers according to `order`
    /// 2. Executes the compiled kernel on those pairs with the given
    ///    `challenges`, producing `num_evals` values
    /// 3. Multiplies each value by the eq weight (see [`EqInput`])
    /// 4. Accumulates into `num_evals` running sums
    ///
    /// Returns `num_evals` field elements. Challenge values are resolved
    /// at dispatch time — the compiled kernel only knows the formula shape.
    fn pairwise_reduce<F: Field>(
        &self,
        inputs: &[&Self::Buffer<F>],
        eq: EqInput<'_, Self, F>,
        kernel: &Self::CompiledKernel<F>,
        challenges: &[F],
        num_evals: usize,
        order: BindingOrder,
    ) -> Vec<F>;

    // -- Fused operations --------------------------------------------------

    /// Fused in-place interpolation + weighted composition-reduce (H2L).
    ///
    /// Combines [`interpolate_pairs_batch_inplace`](Self::interpolate_pairs_batch_inplace)
    /// and [`pairwise_reduce`](Self::pairwise_reduce) into a single pass.
    /// After completion, each input buffer and the weight buffer are halved.
    fn fused_interpolate_reduce<F: Field>(
        &self,
        inputs: &mut [Self::Buffer<F>],
        weights: &mut Self::Buffer<F>,
        interpolation_scalar: F,
        kernel: &Self::CompiledKernel<F>,
        challenges: &[F],
        num_evals: usize,
    ) -> Vec<F> {
        self.interpolate_pairs_batch_inplace(inputs, interpolation_scalar, BindingOrder::HighToLow);
        self.interpolate_pairs_inplace(weights, interpolation_scalar, BindingOrder::HighToLow);
        let refs: Vec<_> = inputs.iter().collect();
        self.pairwise_reduce(
            &refs,
            EqInput::Weighted(weights),
            kernel,
            challenges,
            num_evals,
            BindingOrder::HighToLow,
        )
    }

    // -- Table generation -----------------------------------------------------

    /// Eq product table over the Boolean hypercube (big-endian).
    ///
    /// Computes $2^n$ evaluations where $n = \text{point.len()}$:
    /// $$\text{out}_x = \prod_{i=0}^{n-1} \bigl(r_i \cdot x_i + (1 - r_i)(1 - x_i)\bigr)$$
    fn eq_table<F: Field>(&self, point: &[F]) -> Self::Buffer<F>;

    /// Less-than table over the Boolean hypercube (big-endian).
    ///
    /// Computes $2^n$ evaluations of $\text{LT}(x, r)$:
    /// $$\text{LT}(x, r) = \sum_{i} (1 - x_i) \cdot r_i \cdot \text{eq}(x_{>i}, r_{>i})$$
    ///
    /// Result: `out[j] = 1` when `j < r` as integers, extended multilinearly.
    fn lt_table<F: Field>(&self, point: &[F]) -> Self::Buffer<F>;

    /// Eq-plus-one table over the Boolean hypercube (big-endian).
    ///
    /// Computes $2^n$ evaluations of $\text{eq+1}(r, x)$:
    /// the multilinear extension of the indicator `{x : x = r + 1}`.
    ///
    /// Returns `(eq_evals, eq_plus_one_evals)` — both tables are useful
    /// and computed simultaneously with shared intermediate state.
    fn eq_plus_one_table<F: Field>(&self, point: &[F]) -> (Self::Buffer<F>, Self::Buffer<F>);
}

/// Provides polynomial buffers to the runtime on demand.
///
/// The prover runtime calls [`load`](BufferProvider::load) when a kernel input
/// marked [`InputBinding::Provided`] is first needed. The provider uploads host
/// data to the device and returns a device buffer. Implementations are free to
/// cache, compute lazily, or stream from an external source.
///
/// The canonical implementation lives in `jolt-witness`: `Witness<F>` holds
/// host-side tables from trace processing and uploads them on first request.
pub trait BufferProvider<B: ComputeBackend, F: Field> {
    /// Load polynomial data for `poly_index` onto the device.
    ///
    /// Called at most once per poly index during execution. After loading, the
    /// runtime owns the buffer and manages its lifecycle (including
    /// [`Op::Release`](jolt_compiler::Op::Release)).
    fn load(&mut self, poly_index: usize, backend: &B) -> B::Buffer<F>;
}
