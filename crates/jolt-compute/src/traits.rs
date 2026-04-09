//! Core trait definitions for compute backends.

pub use jolt_compiler::BindingOrder;
use jolt_compiler::KernelSpec;
use jolt_compiler::PolynomialId;
use jolt_field::Field;

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

    // ── Kernel compilation ──────────────────────────────────────────────

    /// Compile a [`KernelSpec`] into backend-native code.
    ///
    /// Bakes in the formula, iteration pattern, evaluation grid, and
    /// binding order. Challenge values are provided at dispatch time.
    fn compile<F: Field>(&self, spec: &KernelSpec) -> Self::CompiledKernel<F>;

    // ── Core dispatch ───────────────────────────────────────────────────

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

    // ── Buffer management ───────────────────────────────────────────────

    /// Upload host data to a device buffer.
    fn upload<T: Scalar>(&self, data: &[T]) -> Self::Buffer<T>;

    /// Download device buffer contents to host memory.
    fn download<T: Scalar>(&self, buf: &Self::Buffer<T>) -> Vec<T>;

    /// Allocate a zero-initialized buffer on the device.
    fn alloc<T: Scalar>(&self, len: usize) -> Self::Buffer<T>;

    /// Active element count of a buffer.
    fn len<T: Scalar>(&self, buf: &Self::Buffer<T>) -> usize;

    // ── Table generation ────────────────────────────────────────────────

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
