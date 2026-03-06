//! Core trait definitions for compute backends.

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
///   [`pairwise_reduce`](Self::pairwise_reduce). Each backend compiles from
///   `jolt-ir::KernelDescriptor` via an inherent `compile` method on its
///   concrete type — **not** through this trait (avoids coupling `jolt-compute`
///   to `jolt-ir`).
///
/// # Zero Cost for CPU
///
/// [`CpuBackend`](crate::CpuBackend) implements this trait with
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
    /// closure. For Metal: `MTLComputePipelineState`. For CUDA: compiled
    /// PTX module.
    ///
    /// Compiled from a `jolt-ir::KernelDescriptor` by each backend's
    /// inherent `compile` method at setup time.
    type CompiledKernel<F: Field>: Send + Sync;

    /// Upload host data to a device buffer.
    fn upload<T: Scalar>(&self, data: &[T]) -> Self::Buffer<T>;

    /// Download device buffer contents to host memory.
    fn download<T: Scalar>(&self, buf: &Self::Buffer<T>) -> Vec<T>;

    /// Allocate a buffer on the device, initialized to zero bytes.
    fn alloc<T: Scalar>(&self, len: usize) -> Self::Buffer<T>;

    /// Active element count of a buffer.
    fn len<T: Scalar>(&self, buf: &Self::Buffer<T>) -> usize;

    /// Returns `true` if the buffer has no elements.
    fn is_empty<T: Scalar>(&self, buf: &Self::Buffer<T>) -> bool {
        self.len::<T>(buf) == 0
    }

    /// Pairwise linear interpolation, halving the buffer.
    ///
    /// For each `i` in $[0, n/2)$ where $n$ is the active buffer length:
    /// $$\text{out}[i] = \text{buf}[2i] + \text{scalar} \cdot (\text{buf}[2i+1] - \text{buf}[2i])$$
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
    /// For each position $i$ in $[0, n/2)$ where $n$ is the active buffer
    /// length:
    ///
    /// 1. Reads pairs $(\text{inputs}[k][2i],\; \text{inputs}[k][2i+1])$
    ///    for all $k$ input buffers
    /// 2. Executes the compiled kernel on those pairs, producing
    ///    $\text{degree} + 1$ values
    /// 3. Multiplies each value by $\text{weights}[i]$
    /// 4. Accumulates into $\text{degree} + 1$ running sums
    ///
    /// Returns $\text{degree} + 1$ field elements after reducing across all
    /// positions. This is the only device-to-host transfer per invocation.
    ///
    /// Both the kernel evaluation and the reduction use delayed modular
    /// reduction internally.
    fn pairwise_reduce<F: Field>(
        &self,
        inputs: &[&Self::Buffer<F>],
        weights: &Self::Buffer<F>,
        kernel: &Self::CompiledKernel<F>,
        degree: usize,
    ) -> Vec<F>;

    /// Multiplicative product table over the Boolean hypercube.
    ///
    /// Computes $2^n$ evaluations where $n = \text{point.len()}$:
    /// $$\text{out}[x] = \prod_{i=0}^{n-1} \bigl(r_i \cdot x_i + (1 - r_i)(1 - x_i)\bigr)$$
    /// for all $x \in \{0,1\}^n$, where $x_i$ is the $i$-th bit of $x$.
    ///
    /// On GPU this is constructed on-device, avoiding a $2^n$ field-element
    /// transfer across the bus. On CPU this is equivalent to
    /// `EqPolynomial::evaluations()`.
    fn product_table<F: Field>(&self, point: &[F]) -> Self::Buffer<F>;
}
