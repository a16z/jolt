//! Core trait definitions for compute backends.

use std::collections::HashMap;

use jolt_compiler::module::{BufferRef, ClaimFormula, ReduceDestination, ReduceSpec};
pub use jolt_compiler::BindingOrder;
use jolt_compiler::KernelSpec;
use jolt_compiler::PolynomialId;
use jolt_field::Field;

/// Marker trait for types that can be stored in device buffers.
pub trait Scalar: Copy + Send + Sync + 'static {}
impl<T: Copy + Send + Sync + 'static> Scalar for T {}

/// Opaque identifier for a backend-owned stateful resource.
///
/// Handles let the runtime refer to long-lived per-sumcheck-instance state
/// (eq polynomials, scratch buffers, prepared caches) without leaking the
/// backend's concrete state type across the trait boundary. The runtime
/// never inspects the bits; it just passes `HandleId` through.
///
/// # Design rationale
///
/// Production ML runtimes (cuBLAS, cuDNN, PyTorch, TVM, IREE) all use an
/// opaque-handle pattern for per-call resources. This mirrors that shape
/// on the Jolt compute boundary, avoiding `Box<dyn Any>` (which would
/// erase type information and require runtime downcasts).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct HandleId(pub u32);

/// Description of the state a newly-opened handle should hold.
///
/// The backend interprets this to allocate internal state; the runtime
/// just passes it through. New variants may be added over time as
/// additional handle-backed kernels are wired — marked `#[non_exhaustive]`
/// so adding variants is not a breaking change.
#[non_exhaustive]
pub enum HandleShape<'a, F: Field> {
    /// A zero-initialized scratch buffer of the given size.
    ///
    /// Owned by the backend so it can be reused across rounds of the same
    /// sumcheck instance without reallocating on each call.
    Scratch { size: usize },
    /// A Gruen split-eq polynomial over `challenges`, bound with `order`.
    ///
    /// Backend may cache prefix tables + current scalar so per-round
    /// binding runs in O(1) amortized instead of rebuilding the full eq
    /// table each round. Reserved for P76-B wire (not yet live).
    Eq {
        challenges: &'a [F],
        order: BindingOrder,
    },
}

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
    /// Host-side compact integer buffer + encoding metadata.
    ///
    /// Stores small-scalar data as `Vec<i128>` with runtime `bits`/`signed`
    /// metadata; defers promote-to-field until bind/reduce time. Fast paths
    /// in the backend read the encoding and dispatch to the matching
    /// `Field::mul_{u64,i64,u128,i128}` variant (each ~2× faster than
    /// `F::from_i128(n) * x` on BN254). Backend-opaque: the data lives on
    /// host regardless of whether the backend is CPU or GPU, so no
    /// per-backend `Buffer<i128>` generic is needed. P84-A' infra —
    /// no backend method consumes it yet.
    Compact {
        data: Vec<i128>,
        bits: u8,
        signed: bool,
    },
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
            DeviceBuffer::Compact { .. } => panic!("expected DeviceBuffer::Field"),
        }
    }

    /// Mutably borrow the inner field buffer. Panics if this is not `Field`.
    #[inline]
    pub fn as_field_mut(&mut self) -> &mut BufF {
        match self {
            DeviceBuffer::Field(buf) => buf,
            DeviceBuffer::U64(_) => panic!("expected DeviceBuffer::Field"),
            DeviceBuffer::Compact { .. } => panic!("expected DeviceBuffer::Field"),
        }
    }

    /// Borrow the inner u64 buffer. Panics if this is not `U64`.
    #[inline]
    pub fn as_u64(&self) -> &BufU64 {
        match self {
            DeviceBuffer::U64(buf) => buf,
            DeviceBuffer::Field(_) => panic!("expected DeviceBuffer::U64"),
            DeviceBuffer::Compact { .. } => panic!("expected DeviceBuffer::U64"),
        }
    }

    /// Mutably borrow the inner u64 buffer. Panics if this is not `U64`.
    #[inline]
    pub fn as_u64_mut(&mut self) -> &mut BufU64 {
        match self {
            DeviceBuffer::U64(buf) => buf,
            DeviceBuffer::Field(_) => panic!("expected DeviceBuffer::U64"),
            DeviceBuffer::Compact { .. } => panic!("expected DeviceBuffer::U64"),
        }
    }

    /// Borrow the compact i128 data slice. Panics if this is not `Compact`.
    #[inline]
    pub fn as_compact(&self) -> &[i128] {
        match self {
            DeviceBuffer::Compact { data, .. } => data,
            _ => panic!("expected DeviceBuffer::Compact"),
        }
    }

    /// Mutably borrow the compact i128 data. Panics if this is not `Compact`.
    #[inline]
    pub fn as_compact_mut(&mut self) -> &mut Vec<i128> {
        match self {
            DeviceBuffer::Compact { data, .. } => data,
            _ => panic!("expected DeviceBuffer::Compact"),
        }
    }

    /// Return `(bits, signed)` encoding metadata for a `Compact` buffer.
    /// Panics if this is not `Compact`.
    #[inline]
    pub fn compact_encoding(&self) -> (u8, bool) {
        match self {
            DeviceBuffer::Compact { bits, signed, .. } => (*bits, *signed),
            _ => panic!("expected DeviceBuffer::Compact"),
        }
    }

    /// Returns `true` if this is a `Field` buffer.
    #[inline]
    pub fn is_field(&self) -> bool {
        matches!(self, DeviceBuffer::Field(_))
    }

    /// Returns `true` if this is a `Compact` buffer.
    #[inline]
    pub fn is_compact(&self) -> bool {
        matches!(self, DeviceBuffer::Compact { .. })
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
    /// challenge references. Challenge *values* are provided at dispatch time.
    fn compile<F: Field>(&self, spec: &KernelSpec) -> Self::CompiledKernel<F>;

    /// Evaluate a batch of kernel reductions described by [`ReduceSpec`]s.
    ///
    /// Unified reduce surface — covers flat, segmented, Gruen, domain, and
    /// sparse reduces for both standalone sumchecks and batched per-instance
    /// evals. A length-1 `specs` corresponds to a single round; a length-N
    /// `specs` replaces a batch-round window. Backends may group by
    /// `(axes variant, binding order, inner_size)` and dispatch one fused
    /// loop per group — results are returned in the same order as `specs`.
    fn reduce<F: Field>(
        &self,
        specs: &[ReduceSpec],
        inputs: &ReduceInputs<'_, Self, F>,
        challenges: &[F],
    ) -> Vec<Vec<F>>;

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

    /// Bind one variable on a compact small-scalar buffer and return the
    /// promoted field-element output.
    ///
    /// Pairs `data` according to `order` (LowToHigh pairs `(2i, 2i+1)`;
    /// HighToLow pairs `(i, i+half)`) and computes `lo + scalar · (hi − lo)`
    /// directly against the i128 operands — i.e., the caller never has to
    /// materialize a full `Vec<F>` for a one-shot first-round bind.
    ///
    /// The `bits`/`signed` encoding metadata is reserved for backend-specific
    /// fast paths that dispatch to tighter `Field::mul_{u64,i64,u128}` variants
    /// when the range is known to fit; the default implementation ignores it.
    ///
    /// Default: promote `data` via [`Field::from_i128`], upload, and call
    /// [`interpolate_inplace`](Self::interpolate_inplace). Backends override
    /// to skip the full-field promote and use [`Field::mul_i128`] on the
    /// difference `(hi − lo)` — on BN254 this is ~2× faster than the default
    /// because `Fr::from_i128(n) * x` pays a montgomery conversion that the
    /// specialized `mul_i128` avoids (see `arkworks/bn254_ops.rs`).
    ///
    /// P84-B infra: no production hot path wires this yet; iter 89 P84-C
    /// will route small-scalar polynomials (RD_INC, RAM_INC) through
    /// `DeviceBuffer::Compact` and call this method on the first-round bind.
    fn bind_compact<F: Field>(
        &self,
        data: &[i128],
        _bits: u8,
        _signed: bool,
        scalar: F,
        order: BindingOrder,
    ) -> Self::Buffer<F> {
        let promoted: Vec<F> = data.iter().map(|&n| F::from_i128(n)).collect();
        let mut buf = self.upload_vec(promoted);
        self.interpolate_inplace(&mut buf, scalar, order);
        buf
    }

    /// Upload host data to a device buffer.
    fn upload<T: Scalar>(&self, data: &[T]) -> Self::Buffer<T>;

    /// Upload an owned host `Vec<T>` to a device buffer.
    ///
    /// Default forwards to `upload(&data)`. CPU backends override to pass the
    /// `Vec` through without copying — saves a K*T memcpy per materialize
    /// when the source already owns the vec (e.g., `Cow::Owned` returns).
    #[inline]
    fn upload_vec<T: Scalar>(&self, data: Vec<T>) -> Self::Buffer<T> {
        self.upload(&data)
    }

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
        staged_evals: &HashMap<(PolynomialId, usize), F>,
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

    // === Handle API (P76-A infra stub) ===
    //
    // Opaque handles carry backend-owned state (eq polynomials, scratch
    // buffers, prepared caches) across multiple op dispatches. The
    // runtime stores `HandleId` values in its schedule; the backend owns
    // the typed state internally (concrete enum, no `dyn Any`). All
    // methods default to `panic!` so existing backends remain unchanged
    // until they opt in. See `perf/report_tools/kernel_gap_memo.md` §5(B).

    /// Open a new handle holding the state described by `shape`.
    ///
    /// Returns a freshly-minted `HandleId` the runtime stores for
    /// subsequent `bind_handle` / `query_handle` / `close_handle` calls.
    fn open_handle<F: Field>(&self, _shape: HandleShape<'_, F>) -> HandleId {
        panic!("open_handle: backend does not support handles")
    }

    /// Bind variable `round` of the handle's polynomial to challenge `r`.
    ///
    /// Implementations may update cached prefix tables, fold scalars, or
    /// halve the underlying buffer. The runtime does not know which.
    fn bind_handle<F: Field>(&self, _id: HandleId, _round: usize, _r: F) {
        panic!("bind_handle: backend does not support handles")
    }

    /// Read the handle's current value at index `idx`.
    ///
    /// For eq handles, this is `current_scalar × prefix_table[idx]` (the
    /// standard evaluation of the partially-bound eq polynomial at `idx`).
    /// For scratch handles, this is the scratch buffer's slot `idx`.
    fn query_handle<F: Field>(&self, _id: HandleId, _idx: usize) -> F {
        panic!("query_handle: backend does not support handles")
    }

    /// Release the handle; backend may free or recycle the underlying state.
    ///
    /// After `close_handle`, the id may be reused by the backend for a
    /// later `open_handle` call. The runtime must not call any other
    /// handle method with this id after closing.
    fn close_handle(&self, _id: HandleId) {
        panic!("close_handle: backend does not support handles")
    }

    /// Eq-weighted projection using a pre-opened Eq-shape handle.
    ///
    /// Semantically equivalent to [`eq_project`](Self::eq_project) with the
    /// eq table drawn from the handle opened with `HandleShape::Eq`. The
    /// point is resolved once at open time; subsequent calls reuse the
    /// cached table instead of rebuilding it. Default forwards the handle
    /// to `query_handle` slot-by-slot — slow but correct; backends that
    /// keep the eq table contiguously should override.
    fn eq_project_from_handle<F: Field>(
        &self,
        _id: HandleId,
        _source_data: &[F],
        _inner_size: usize,
        _outer_size: usize,
    ) -> Self::Buffer<F> {
        panic!("eq_project_from_handle: backend does not support eq handles")
    }
}

/// Borrow view of runtime state passed to [`ComputeBackend::reduce`].
///
/// The backend resolves each [`BufferRef`] in a [`ReduceSpec`] against
/// this view: `BufferRef::Polynomial` looks up `buffers`, and
/// `BufferRef::SegmentedOuterEq` looks up `outer_eqs`. Kernels are
/// resolved by index via `kernels`. The Gruen prev-claim is resolved
/// from `instance_claims[batch][instance]` using the spec's
/// [`ReduceDestination::Instance`] coordinates. The borrow pattern
/// keeps the backend decoupled from the runtime's concrete state type.
pub struct ReduceInputs<'a, B: ComputeBackend + ?Sized, F: Field> {
    /// Materialized polynomial buffers keyed by `PolynomialId`.
    pub buffers: &'a HashMap<PolynomialId, Buf<B, F>>,
    /// Per-instance segmented outer-eq buffers keyed by `(batch, instance)`.
    pub outer_eqs: &'a HashMap<(usize, usize), Vec<F>>,
    /// Running per-instance claim — `instance_claims[batch][instance]` is the
    /// previous round's claim consumed by Gruen cubic assembly. Mirrors
    /// `RuntimeState::batch_instance_claims`.
    pub instance_claims: &'a [Vec<F>],
    /// Compiled kernels indexed by `ReduceSpec::kernel`.
    pub kernels: &'a [B::CompiledKernel<F>],
}

impl<'a, B, F> ReduceInputs<'a, B, F>
where
    B: ComputeBackend + ?Sized,
    F: Field,
{
    /// Resolve a polynomial [`BufferRef`] to its materialized device buffer.
    /// Panics on [`BufferRef::SegmentedOuterEq`] — those are wired through
    /// `ReduceAxes::Product.outer_eq` instead.
    pub fn buffer(&self, buf_ref: &BufferRef) -> &'a Buf<B, F> {
        match buf_ref {
            BufferRef::Polynomial(id) => self
                .buffers
                .get(id)
                .unwrap_or_else(|| panic!("reduce input: missing polynomial buffer {id:?}")),
            BufferRef::SegmentedOuterEq { .. } => panic!(
                "reduce input: SegmentedOuterEq is not a kernel input — \
                 use ReduceAxes::Product.outer_eq"
            ),
        }
    }

    /// Resolve the segmented outer-eq table referenced by a Product axis.
    /// Panics on [`BufferRef::Polynomial`] — outer eq lives outside the
    /// device buffer map.
    pub fn outer_eq(&self, buf_ref: &BufferRef) -> &'a [F] {
        match buf_ref {
            BufferRef::SegmentedOuterEq { batch, instance } => self
                .outer_eqs
                .get(&(batch.0, instance.0))
                .unwrap_or_else(|| {
                    panic!(
                        "reduce input: missing outer_eq for (batch={}, instance={})",
                        batch.0, instance.0
                    )
                })
                .as_slice(),
            BufferRef::Polynomial(_) => {
                panic!("reduce input: ReduceAxes::Product.outer_eq must be SegmentedOuterEq")
            }
        }
    }

    /// Gruen prev-claim for a Product-axes spec — resolved from the
    /// `(batch, instance)` coordinates of the spec's destination. Panics
    /// on `SumcheckRound` destinations since Gruen only applies to
    /// per-instance reduces.
    pub fn prev_claim(&self, dest: &ReduceDestination) -> F {
        match dest {
            ReduceDestination::Instance { batch, instance } => {
                self.instance_claims[batch.0][instance.0]
            }
            ReduceDestination::SumcheckRound { .. } => panic!(
                "reduce: GruenContext only valid with ReduceDestination::Instance \
                 (got SumcheckRound)"
            ),
        }
    }
}

/// Per-cycle trace data for the instruction lookup sumcheck.
///
/// Provided alongside polynomial data as an input to the prover runtime.
/// Non-field fields (u128 keys, Option<usize>, bool) that can't be expressed
/// as polynomial evaluations. Used by SuffixScatter, QBufferScatter,
/// UpdateInstanceWeights, MaterializeRA, and MaterializeCombinedVal.
pub struct LookupTraceData {
    /// Per-cycle lookup key (128-bit packed), T entries.
    pub lookup_keys: Vec<u128>,
    /// Per-cycle lookup table index, T entries. None for cycles with no lookup.
    pub table_kind_indices: Vec<Option<usize>>,
    /// Per-cycle interleaved-operands flag, T entries.
    pub is_interleaved: Vec<bool>,
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

    /// Per-cycle lookup trace data used by instruction-lookup handlers.
    /// Returns `None` when no lookup sumcheck is in the schedule.
    fn lookup_trace(&self) -> Option<&LookupTraceData> {
        None
    }
}
