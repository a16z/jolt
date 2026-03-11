//! Core trait definitions for compute backends.

use jolt_field::Field;
use jolt_ir::KernelDescriptor;

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

/// Variable binding order for polynomial interpolation.
///
/// Determines how pairs are formed from buffer elements:
///
/// - **LowToHigh**: Interleaved layout. Pairs `(buf[2i], buf[2i+1])`.
///   Binds the least-significant variable first. Default for most sumcheck
///   instances (instruction RA, claim reductions).
///
/// - **HighToLow**: Split-half layout. Pairs `(buf[i], buf[i + n/2])`.
///   Binds the most-significant variable first. Used by Spartan outer
///   sumcheck and RAM read-write checking.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum BindingOrder {
    #[default]
    LowToHigh,
    HighToLow,
}

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
///   [`KernelDescriptor`] via [`compile_kernel`](Self::compile_kernel).
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

    /// Compile a [`KernelDescriptor`] into a backend-specific kernel.
    fn compile_kernel<F: Field>(&self, desc: &KernelDescriptor) -> Self::CompiledKernel<F> {
        self.compile_kernel_with_challenges(desc, &[])
    }

    /// Compile a [`KernelDescriptor`] with baked challenge values.
    ///
    /// For `Custom` expression kernels, `challenges[i]` is substituted for
    /// `Var::Challenge(i)`. For `ProductSum` descriptors, challenges are
    /// ignored.
    fn compile_kernel_with_challenges<F: Field>(
        &self,
        desc: &KernelDescriptor,
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
    /// 3. Multiplies each value by `weights[i]`
    /// 4. Accumulates into `num_evals` running sums
    ///
    /// Returns `num_evals` field elements after reducing across all
    /// positions. This is the only device-to-host transfer per invocation.
    ///
    /// For Toom-Cook kernels (ProductSum), `num_evals = D` (evaluations on
    /// the grid `{1, ..., D-1, ∞}`). For standard-grid kernels (Custom),
    /// `num_evals = degree + 1` (evaluations on `{0, 1, ..., degree}`).
    ///
    /// Both the kernel evaluation and the reduction use delayed modular
    /// reduction internally.
    fn pairwise_reduce<F: Field>(
        &self,
        inputs: &[&Self::Buffer<F>],
        weights: &Self::Buffer<F>,
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
        weights: &Self::Buffer<F>,
        kernel: &Self::CompiledKernel<F>,
        order: BindingOrder,
    ) -> [F; D] {
        let v = self.pairwise_reduce(inputs, weights, kernel, D, order);
        core::array::from_fn(|i| v[i])
    }

    /// Composition-reduce with implicit unit weights (all ones).
    ///
    /// Equivalent to [`pairwise_reduce`](Self::pairwise_reduce) with a weights
    /// buffer of all `F::one()`, but avoids allocating the buffer and skips the
    /// per-element weight multiply. On GPU backends this dispatches a specialized
    /// unweighted kernel that saves one `fr_mul` per pair per evaluation slot.
    fn pairwise_reduce_unweighted<F: Field>(
        &self,
        inputs: &[&Self::Buffer<F>],
        kernel: &Self::CompiledKernel<F>,
        num_evals: usize,
        order: BindingOrder,
    ) -> Vec<F> {
        let n = self.len(inputs[0]);
        let ones = vec![F::one(); n / 2];
        let weights = self.upload(&ones);
        self.pairwise_reduce(inputs, &weights, kernel, num_evals, order)
    }

    /// Split-eq tensor-product composition-reduce.
    ///
    /// Like [`pairwise_reduce`](Self::pairwise_reduce) but with factored
    /// weights. Instead of a flat weight buffer, takes two tables whose
    /// tensor product forms the weights:
    ///
    /// $$w(x_{\text{out}}, x_{\text{in}}) = \text{outer}_{x_{\text{out}}} \cdot \text{inner}_{x_{\text{in}}}$$
    ///
    /// Input buffers have `2 × |outer| × |inner|` elements. Position
    /// $(x_{\text{out}}, x_{\text{in}})$ maps to pair index
    /// $x_{\text{out}} \cdot |\text{inner}| + x_{\text{in}}$.
    ///
    /// On CPU the outer loop stays in L1 while the inner loop streams data.
    /// On GPU the outer factor maps to thread groups, the inner to threads
    /// with the inner weight table in shared memory.
    fn tensor_pairwise_reduce<F: Field>(
        &self,
        inputs: &[&Self::Buffer<F>],
        outer_weights: &Self::Buffer<F>,
        inner_weights: &Self::Buffer<F>,
        kernel: &Self::CompiledKernel<F>,
        num_evals: usize,
    ) -> Vec<F>;

    /// Const-generic split-eq tensor-product composition-reduce.
    ///
    /// Combines the tensor-product weight factoring of
    /// [`tensor_pairwise_reduce`](Self::tensor_pairwise_reduce) with the
    /// stack-allocated scratch of [`pairwise_reduce_fixed`](Self::pairwise_reduce_fixed).
    fn tensor_pairwise_reduce_fixed<F: Field, const D: usize>(
        &self,
        inputs: &[&Self::Buffer<F>],
        outer_weights: &Self::Buffer<F>,
        inner_weights: &Self::Buffer<F>,
        kernel: &Self::CompiledKernel<F>,
    ) -> [F; D] {
        let v = self.tensor_pairwise_reduce(inputs, outer_weights, inner_weights, kernel, D);
        core::array::from_fn(|i| v[i])
    }

    /// Evaluates multiple kernels over the same inputs and weights in a
    /// single pass.
    ///
    /// Each `(kernel, num_evals)` pair shares the same input buffers and
    /// weight buffer. The data is read once per position and each kernel is
    /// evaluated, saving cache misses compared to calling
    /// [`pairwise_reduce`](Self::pairwise_reduce) independently per kernel.
    fn pairwise_reduce_multi<F: Field>(
        &self,
        inputs: &[&Self::Buffer<F>],
        weights: &Self::Buffer<F>,
        kernels: &[(&Self::CompiledKernel<F>, usize)],
        order: BindingOrder,
    ) -> Vec<Vec<F>> {
        kernels
            .iter()
            .map(|(kernel, num_evals)| {
                self.pairwise_reduce(inputs, weights, kernel, *num_evals, order)
            })
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
}
