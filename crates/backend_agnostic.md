# Backend-Agnostic Compute Design

**Status:** Draft
**Authors:** Markos Georghiades
**Date:** 2026-03-06

## 1. Motivation

Jolt's prover is dominated by sumcheck round polynomial computation and polynomial binding — both embarrassingly parallel map-reduce patterns over field element buffers. Today these run on CPU via Rayon. To reach production proving speeds, Jolt must support GPU backends (Metal, CUDA, WebGPU) without sacrificing CPU performance or introducing protocol-level complexity.

The design follows an ML compiler philosophy: **don't compile individual field operations — compile fused kernels from computation descriptions.** Just as XLA/Triton fuse `relu(matmul(x, w) + bias)` into one GPU kernel, Jolt fuses "for all x in the hypercube, evaluate the product-sum composition weighted by eq(r, x) and reduce" into one kernel dispatch.

### Goals

1. **Zero overhead for CPU** — after monomorphization, the CPU backend compiles to exactly the same code as today's hand-tuned Rayon implementation. No vtables, no indirection, no runtime dispatch.
2. **Full GPU utilization** — the abstraction exposes enough structure for GPU backends to generate optimal kernels (coalesced memory access, shared memory tiling, warp-level reduction).
3. **Protocol-layer ignorance** — `jolt-sumcheck`, `jolt-spartan`, and the Fiat-Shamir transcript are host-only. They never touch device memory or know that a GPU exists.
4. **Kernel compilation from jolt-ir** — most sumcheck compositions are compiled automatically from `KernelDescriptor`. Hand-written kernels are supported as an escape hatch for bespoke cases.
5. **Compact storage on device** — buffers are generic over element type (`u8`, `bool`, `i64`, `F`), matching `jolt-poly`'s `Polynomial<T>` pattern. Compact buffers use up to 32x less device memory.

### Non-Goals

- Rewriting PCS/commitment code for GPU. Dory and other PCS implementations manage their own compute (the `dory-pcs` crate already handles this internally).
- GPU-side Fiat-Shamir. Transcript operations stay on host — hashing on GPU is not worth the complexity for the small amount of data involved.
- Runtime JIT compilation. Kernel parameters (D, composition structure) are known at setup time. AOT compilation with D-specialization is sufficient.

---

## 2. Computational Patterns in Jolt

### What runs on GPU

There are two kernel shapes that cover ~95% of prover compute time:

**Kernel 1: Composition Reduce** (round polynomial computation)

For each pair position `i` in `[0, active_len/2)`:
1. Read pairs `(input_k[2i], input_k[2i+1])` from K input buffers
2. Evaluate the composition (product of K linear interpolants at D+1 grid points, summed across product groups)
3. Multiply by weight `eq[i]`
4. Accumulate into D+1 running sums

Tree-reduce partial sums across all threads. Returns D+1 field elements to host.

**Kernel 2: Paired Interpolation** (polynomial binding)

For each position `i` in `[0, active_len/2)`:
```
out[i] = buf[2i] + scalar * (buf[2i+1] - buf[2i])
```
Halves the active buffer length. When the input element type is compact (e.g., `u8`), the first bind promotes to field elements; subsequent binds are field-to-field.

### What stays on host

- Fiat-Shamir transcript operations (absorb round polynomial coefficients, squeeze challenges)
- Sumcheck protocol orchestration (`SumcheckProver`, `BatchedSumcheckProver`)
- Verifier logic (entirely host-side)
- Opening proof construction and verification (PCS-managed)
- Proof serialization

### The host-device loop

```
setup:
    compiled_kernels = [backend.compile(descriptor) for descriptor in stage_descriptors]
    poly_buffers = [backend.upload(poly_data) for poly_data in witness_polys]
    eq_buffer = backend.product_table(initial_point)

per sumcheck round:
    coeffs = backend.pairwise_reduce(poly_buffers, eq_buffer, kernel, degree)  // GPU → host: D+1 scalars
    transcript.append(coeffs)                                                   // host
    challenge = transcript.challenge()                                          // host
    backend.interpolate_pairs(&mut poly_buffers, challenge)                     // GPU, in-place
    backend.interpolate_pairs(&mut eq_buffer, challenge)                        // GPU, in-place

after all rounds:
    final_evals = [backend.download(buf) for buf in poly_buffers]               // GPU → host
    // ... proceed with opening proofs (host-side PCS)
```

The only data crossing the host-device boundary per round is D+1 field elements (round polynomial coefficients) from device to host, and 1 field element (challenge) from host to device. This is negligible bandwidth.

---

## 3. Instance and Kernel Counts

Each of Jolt's ~7 sumcheck stages is a **batched sumcheck** — a random linear combination of multiple independent sumcheck instances. Each instance has its own `compute_message` implementation that defines a distinct computational kernel.

| Stage | Instances | Kernel Shape | D |
|-------|-----------|-------------|---|
| Spartan outer (uniskip) | 1 | Custom (Lagrange basis + multiquadratic) | Stage-specific |
| Spartan outer (streaming) | 1 | Custom (split-eq + fused R1CS eval) | Stage-specific |
| Instruction RA | 1 per instruction group | ProductSum | 4, 8, or 16 |
| RAM read-write | 1 per memory type | Custom (3 phases: cycle-major, address-major, linear) | 3 |
| Register read-write | Similar to RAM | Custom (phase-based) | 3 |
| Bytecode | 1 | Custom (phase-based) | 3 |
| Claim reductions | Several small instances | ProductSum or simple eval | 2-4 |

Total: ~15-25 distinct `compute_message` implementations, but the majority share the **ProductSum** kernel shape with different D values.

### The ProductSum pattern (dominant cost, ~80% of prover time)

At each pair position `g`:
```
for t in 0..num_products:
    pairs = [(input[t*D+i][2g], input[t*D+i][2g+1]) for i in 0..D]
    grid_evals = product_of_linear_interpolants(pairs, D)   // D+1 values
    sums[k] += eq_weight[g] * grid_evals[k]                 // accumulate
```

This is a map-reduce: the map reads D pairs and produces D+1 grid evaluations, the reduce sums across all positions. Embarrassingly parallel and ideal for GPU.

### The split-eq optimization (tensor product structure)

The eq polynomial weight table has a $\sqrt{N}$-decomposition into outer and inner factors:
$$\text{eq}(r, x) = \text{eq}_{\text{out}}(r_{\text{out}}, x_{\text{out}}) \cdot \text{eq}_{\text{in}}(r_{\text{in}}, x_{\text{in}})$$

This maps naturally to GPU thread hierarchy: outer index → thread groups, inner index → threads within a group. The `TensorSplit` field in `KernelDescriptor` captures this structure.

---

## 4. Crate Architecture

### Dependency Graph

```
jolt-field (leaf)
├── jolt-compute (NEW: ComputeBackend, CpuBackend)
├── jolt-ir      (+ KernelDescriptor, KernelShape, TensorSplit)
├── jolt-poly    (unchanged)
├── jolt-transcript (unchanged)
│
├── jolt-sumcheck (unchanged — host-only protocol)
├── jolt-crypto, jolt-openings, jolt-spartan, jolt-dory (unchanged)
│
├── jolt-zkvm    (+ jolt-compute — wires backend into witnesses)
│
├── jolt-metal   (jolt-compute + jolt-ir + jolt-field)
├── jolt-cuda    (jolt-compute + jolt-ir + jolt-field)
└── jolt-webgpu  (jolt-compute + jolt-ir + jolt-field)
```

**Key properties:**
- `jolt-compute` depends only on `jolt-field`. No awareness of sumcheck, polynomials, protocols, or transcripts.
- `jolt-sumcheck` is unchanged. The protocol layer is host-only.
- GPU backend crates have minimal deps: `jolt-field` + `jolt-compute` + `jolt-ir`. They don't pull in protocol crates.
- `jolt-ir` and `jolt-compute` are independent — no dependency between them. Only the backend crates and `jolt-zkvm` depend on both.

### `jolt-compute` — Parallel Field-Element Primitives

**Purpose:** Backend-agnostic compute device abstraction. Defines buffer management and parallel primitives on field element buffers. No awareness of sumcheck, polynomials, or cryptographic protocols.

**Depends on:** `jolt-field`

**Estimated LOC:** ~400 (traits + CpuBackend)

```rust
/// Marker trait for types storable in device buffers.
///
/// Matches `jolt-poly`'s `Polynomial<T>` philosophy: buffers hold any
/// scalar type (`u8`, `bool`, `i64`, `F`), not just field elements.
/// Compact types use less device memory and are promoted to `F` inside
/// kernels when needed.
pub trait Scalar: Copy + Send + Sync + 'static {}

impl Scalar for u8 {}
impl Scalar for u16 {}
impl Scalar for u32 {}
impl Scalar for u64 {}
impl Scalar for i64 {}
impl Scalar for i128 {}
impl Scalar for bool {}
// Field types implement Scalar via blanket: impl<F: Field> Scalar for F {}

/// Abstraction over a compute device (CPU, Metal GPU, CUDA GPU, WebGPU).
///
/// Provides buffer management and parallel primitives on typed buffers.
/// All methods are named for what they compute, not what protocol uses them.
///
/// # Zero-cost for CPU
///
/// `CpuBackend` implements this trait with `Buffer<T> = Vec<T>`. After
/// monomorphization, all trait method calls compile to direct function
/// calls with no indirection — identical codegen to hand-written Rayon code.
pub trait ComputeBackend: Send + Sync + 'static {
    /// Handle to a typed buffer on the device.
    ///
    /// For CPU: `Vec<T>`. For Metal: wraps `MTLBuffer`. For CUDA: wraps `CUdeviceptr`.
    type Buffer<T: Scalar>: Send + Sync;

    /// Opaque compiled kernel for composition-reduce operations.
    ///
    /// For CPU: function pointer or interpreted Expr.
    /// For Metal: `MTLComputePipelineState`.
    /// For CUDA: compiled PTX module.
    ///
    /// Compiled from a `KernelDescriptor` (jolt-ir) by each backend's
    /// inherent `compile` method — not through this trait (avoids coupling
    /// jolt-compute to jolt-ir).
    type CompiledKernel: Send + Sync;

    // ── Buffer management ──────────────────────────────────

    /// Upload host data to a device buffer.
    fn upload<T: Scalar>(&self, data: &[T]) -> Self::Buffer<T>;

    /// Download device buffer contents to host memory.
    fn download<T: Scalar>(&self, buf: &Self::Buffer<T>) -> Vec<T>;

    /// Allocate a zeroed buffer on the device.
    fn alloc<T: Scalar>(&self, len: usize) -> Self::Buffer<T>;

    /// Active element count of a buffer.
    fn len<T: Scalar>(&self, buf: &Self::Buffer<T>) -> usize;

    // ── Parallel primitives ────────────────────────────────

    /// Pairwise linear interpolation, halving the buffer.
    ///
    /// For each `i` in `[0, active_len / 2)`:
    /// $$\text{out}[i] = \text{buf}[2i] + \text{scalar} \cdot (\text{buf}[2i+1] - \text{buf}[2i])$$
    ///
    /// When `T` is a compact type (e.g., `u8`), elements are promoted to `F`
    /// during the operation. The returned buffer has element type `F` and
    /// half the original length.
    ///
    /// When `T = F`, this is an in-place operation and the `From` conversion
    /// is the identity (compiled away).
    fn interpolate_pairs<T, F>(
        &self,
        buf: Self::Buffer<T>,
        scalar: F,
    ) -> Self::Buffer<F>
    where
        T: Scalar,
        F: Field + From<T>;

    /// Composition-reduce over paired inputs from multiple buffers.
    ///
    /// For each position `i` in `[0, active_len / 2)`:
    /// 1. Reads pairs `(inputs[k][2i], inputs[k][2i+1])` for all `k`
    /// 2. Executes the compiled kernel on those pairs (producing `degree + 1` values)
    /// 3. Multiplies each value by `weights[i]`
    /// 4. Accumulates into `degree + 1` running sums
    ///
    /// Returns `degree + 1` field elements after reducing across all positions.
    ///
    /// Both the kernel evaluation and the reduction use delayed modular
    /// reduction internally — accumulating as wide integers and reducing
    /// once per thread group (CPU) or per warp (GPU).
    fn pairwise_reduce<F: Field>(
        &self,
        inputs: &[&Self::Buffer<F>],
        weights: &Self::Buffer<F>,
        kernel: &Self::CompiledKernel,
        degree: usize,
    ) -> Vec<F>;

    /// Multiplicative product table over the Boolean hypercube.
    ///
    /// Computes $2^n$ evaluations:
    /// $$\text{out}[x] = \prod_{i=0}^{n-1} \bigl(r_i \cdot x_i + (1 - r_i)(1 - x_i)\bigr)$$
    ///
    /// for all $x \in \{0,1\}^n$, where $x_i$ is the $i$-th bit of $x$.
    ///
    /// On GPU this is built on-device (avoids transferring $2^n$ field elements
    /// across the bus). On CPU this is equivalent to `EqPolynomial::evaluations()`.
    fn product_table<F: Field>(&self, point: &[F]) -> Self::Buffer<F>;
}
```

#### CpuBackend

```rust
/// CPU compute backend using Rayon for parallelism.
///
/// `Buffer<T>` is `Vec<T>`. After monomorphization, all trait methods
/// compile to the same code as direct Rayon parallel iterators — zero
/// abstraction overhead.
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    type Buffer<T: Scalar> = Vec<T>;
    type CompiledKernel = CpuKernel;

    fn upload<T: Scalar>(&self, data: &[T]) -> Vec<T> { data.to_vec() }
    fn download<T: Scalar>(&self, buf: &Vec<T>) -> Vec<T> { buf.clone() }
    fn alloc<T: Scalar>(&self, len: usize) -> Vec<T> {
        // SAFETY: Scalar types are Copy + valid for zero-init
        vec![unsafe { std::mem::zeroed() }; len]
    }
    fn len<T: Scalar>(&self, buf: &Vec<T>) -> usize { buf.len() }

    fn interpolate_pairs<T, F>(&self, buf: Vec<T>, scalar: F) -> Vec<F>
    where T: Scalar, F: Field + From<T>
    {
        // Rayon parallel: lo + scalar * (hi - lo), with From<T> promotion
        // When T = F, From is identity — compiler eliminates it.
        // ...
    }

    fn pairwise_reduce<F: Field>(
        &self,
        inputs: &[&Vec<F>],
        weights: &Vec<F>,
        kernel: &CpuKernel,
        degree: usize,
    ) -> Vec<F> {
        // Rayon par_fold with unreduced accumulators + Barrett reduction
        // ...
    }

    fn product_table<F: Field>(&self, point: &[F]) -> Vec<F> {
        // Same as EqPolynomial::evaluations()
        // ...
    }
}
```

### `jolt-ir` — Extended with `KernelDescriptor`

**Existing:** `Expr`, `ExprBuilder`, `SumOfProducts`, visitors, backends (evaluate, R1CS, circuit, lean).

**New types for kernel compilation:**

```rust
/// Describes the computation performed at each pair position in a
/// composition-reduce kernel.
///
/// Two variants: an explicit fast-path for the product-sum pattern
/// (dominant cost in Jolt), and an escape hatch for bespoke compositions.
pub enum KernelShape {
    /// Product of D linear interpolants evaluated at D+1 grid points,
    /// summed across multiple product groups.
    ///
    /// At each pair position `g`, for each group `t` in `0..num_products`:
    /// ```text
    /// pairs = [(input[t*D+i][2g], input[t*D+i][2g+1]) for i in 0..D]
    /// grid[k] += prod_i lerp(pairs[i], grid_point[k])   for k in 0..=D
    /// ```
    ///
    /// This covers instruction RA sumchecks (D=4, 8, 16) and most claim
    /// reductions. ~80% of prover compute time.
    ProductSum {
        /// Number of input buffers per product group.
        num_inputs_per_product: usize,
        /// Number of product groups summed together.
        num_products: usize,
    },

    /// Arbitrary composition defined by an expression tree.
    ///
    /// Escape hatch for RAM read-write checking phases, Spartan outer
    /// sumcheck, and any other bespoke kernel that doesn't fit the
    /// product-sum pattern.
    ///
    /// The Expr's `Opening(i)` variables map to `(input_i[2g], input_i[2g+1])`
    /// pairs. The backend evaluates the expression at D+1 interpolation
    /// points per pair position.
    Custom {
        /// Composition expression. Openings reference input buffer pairs.
        expr: Expr,
        /// Total number of input buffers.
        num_inputs: usize,
    },
}

/// Describes a fused composition-reduce kernel for compilation.
///
/// Constructed from sumcheck instance definitions at setup time.
/// Each backend compiles this into its `CompiledKernel` type.
pub struct KernelDescriptor {
    /// The computation performed at each pair position.
    pub shape: KernelShape,

    /// Degree of the composition. Determines output size (`degree + 1`
    /// field elements).
    pub degree: usize,

    /// Tensor-product decomposition for the weight (eq) buffer.
    ///
    /// When `Some`, the weight buffer has a $\sqrt{N}$ factorization
    /// into outer and inner components. The backend can exploit this
    /// for better thread hierarchy mapping on GPU (outer → thread groups,
    /// inner → threads within a group) and better cache locality on CPU.
    ///
    /// When `None`, the weight buffer is flat (standard reduction).
    pub tensor_split: Option<TensorSplit>,
}

/// Tensor-product decomposition parameters for split-eq optimization.
pub struct TensorSplit {
    /// Number of outer variables. Outer loop size = $2^{\text{outer\_vars}}$.
    pub outer_vars: usize,
    /// Number of inner variables. Inner loop size = $2^{\text{inner\_vars}}$.
    pub inner_vars: usize,
}
```

**Compilation** is an inherent method on each backend's concrete type, not on the `ComputeBackend` trait. This keeps `jolt-compute` independent from `jolt-ir`:

```rust
// In jolt-metal:
impl MetalBackend {
    pub fn compile(&self, desc: &KernelDescriptor) -> MetalKernel { ... }
}

// In CpuBackend (inside jolt-compute, or in a separate jolt-cpu-kernels crate):
impl CpuBackend {
    pub fn compile(&self, desc: &KernelDescriptor) -> CpuKernel { ... }
}
```

For the `ProductSum` shape, GPU backends generate specialized kernels with D unrolled at compile time. For the `Custom` shape, they either interpret the Expr at runtime or generate shader source from it (the `CircuitEmitter` pattern already in jolt-ir provides the visitor infrastructure for this).

### `jolt-sumcheck` — Unchanged

The protocol layer is entirely host-side:

```rust
// These traits and types are UNCHANGED:
pub trait SumcheckWitness<F: Field>: Send + Sync {
    fn round_polynomial(&self) -> UnivariatePoly<F>;
    fn bind(&mut self, challenge: F);
}

pub struct SumcheckProver;   // host-only orchestration
pub struct SumcheckVerifier;  // host-only verification
```

`SumcheckWitness` returns `UnivariatePoly<F>` — a host-side value. The witness implementation internally uses whatever compute backend it holds. The sumcheck protocol layer never touches device memory.

### `jolt-zkvm` — Wiring It Together

Witness implementations are generic over `ComputeBackend`:

```rust
use jolt_compute::{ComputeBackend, Scalar};
use jolt_sumcheck::SumcheckWitness;

/// Instruction RA virtual sumcheck witness.
///
/// Generic over compute backend — same code drives CPU and GPU.
/// After monomorphization with `CpuBackend`, compiles to identical
/// code as today's hand-tuned implementation.
pub struct RaVirtualWitness<F: Field, B: ComputeBackend> {
    backend: B,
    /// D polynomials per product group, stored as device buffers.
    /// Compact (`Buffer<u8>`) until first bind, then `Buffer<F>`.
    poly_buffers: Vec<B::Buffer<F>>,
    /// Eq polynomial evaluations, on device.
    eq_buffer: B::Buffer<F>,
    /// Pre-compiled kernel for product-sum evaluation.
    kernel: B::CompiledKernel,
    degree: usize,
}

impl<F: Field, B: ComputeBackend> SumcheckWitness<F> for RaVirtualWitness<F, B> {
    fn round_polynomial(&self) -> UnivariatePoly<F> {
        let refs: Vec<&B::Buffer<F>> = self.poly_buffers.iter().collect();
        let coeffs = self.backend.pairwise_reduce(
            &refs, &self.eq_buffer, &self.kernel, self.degree,
        );
        UnivariatePoly::new(coeffs)
    }

    fn bind(&mut self, challenge: F) {
        for buf in &mut self.poly_buffers {
            *buf = self.backend.interpolate_pairs(
                std::mem::take(buf), challenge,
            );
        }
        self.eq_buffer = self.backend.interpolate_pairs(
            std::mem::take(&mut self.eq_buffer), challenge,
        );
    }
}
```

**Prover orchestration** is also generic over the backend:

```rust
pub fn prove<F, PCS, B>(
    trace: &Trace,
    pcs_setup: &PCS::ProverSetup,
    backend: &B,
) -> JoltProof<F, PCS>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    B: ComputeBackend,
{
    // Setup: compile kernels from descriptors
    let ra_kernel = backend.compile(&ra_kernel_descriptor());
    let rw_kernels = rw_kernel_descriptors().map(|d| backend.compile(&d));
    // ...

    // Upload witness data to device
    let ra_buffers: Vec<_> = ra_data.iter()
        .map(|p| backend.upload(p))
        .collect();

    // Build eq table on device
    let eq_buffer = backend.product_table(&initial_point);

    // Construct witnesses (hold device buffers + compiled kernels)
    let mut ra_witness = RaVirtualWitness::new(
        backend, ra_buffers, eq_buffer, ra_kernel, D,
    );

    // SumcheckProver is host-only — FS + round_poly + bind
    let proof = SumcheckProver::prove(
        &claim, &mut ra_witness, &mut transcript, challenge_fn,
    );

    // Download final evaluations for opening proofs
    let final_evals: Vec<F> = ra_witness.poly_buffers.iter()
        .map(|b| backend.download(b))
        .collect();

    // Opening proofs (host-side PCS — Dory manages its own compute)
    // ...
}
```

### GPU Backend Crates

Each GPU backend crate implements `ComputeBackend` and provides a `compile` method:

```
jolt-metal/
├── Cargo.toml     # deps: jolt-compute, jolt-ir, jolt-field, metal-rs
├── src/
│   ├── lib.rs
│   ├── backend.rs  # MetalBackend: ComputeBackend impl
│   ├── buffer.rs   # MetalBuffer<T>: device buffer wrapper
│   ├── kernels.rs  # KernelDescriptor → MSL source → MTLComputePipelineState
│   └── shaders/    # Pre-compiled MSL for ProductSum variants (D=4,8,16)
```

GPU backend crates depend only on `jolt-field`, `jolt-compute`, and `jolt-ir`. They don't pull in sumcheck, transcript, openings, or any protocol crate.

---

## 5. Compilation Model

### AOT with D-Specialization

Runtime parameters:
- **D** (polynomials per product group): fixed per sumcheck instance type, known at build time. Values: 4, 8, 16 for instruction RA; 2-4 for claim reductions.
- **`num_vars`**: varies per proof (depends on trace length). Only affects loop bounds, not kernel structure.
- **`num_products`**: varies per instance. Only affects loop iteration count.

Since D is small and enumerable, and num_vars/num_products only affect launch parameters:

**Pre-compile kernels for each D at build time (or first use with caching).** No JIT needed. The compiled kernels are parameterized by num_vars and num_products at dispatch time.

For the `ProductSum` shape, each D value produces a distinct kernel with fully unrolled product evaluation. For D=4, 8, 16 that's 3 kernels covering ~80% of prover time.

For the `Custom` shape, compilation happens at setup time from the Expr. The Expr is small (claim formulas are typically <50 nodes) so compilation is fast (~ms).

### One Compiled Kernel per Composition

Each distinct sumcheck instance type gets its own `CompiledKernel`. A Jolt proof compiles ~15-25 kernels at setup time. This is optimal:

- **CPU:** monomorphization produces distinct, maximally-optimized code paths per kernel. No runtime dispatch overhead.
- **GPU:** 15-25 compiled pipeline objects is trivial — GPUs handle hundreds. Setup cost (~ms each) is negligible vs. proving time (~seconds).
- **Type safety:** each witness holds the correct kernel at compile time. Cannot accidentally dispatch the wrong kernel.

---

## 6. Delayed Reduction

Both CPU and GPU backends use **delayed modular reduction** — accumulating multiply-add results as wide integers and reducing once per accumulation group.

On BN254 (254-bit prime):
- A field multiplication produces a ~508-bit unreduced result
- Adding N unreduced results requires ~508 + log2(N) bits
- Barrett/Montgomery reduction is expensive — doing it once per group instead of per multiplication saves 10-100x reduction overhead

The `ComputeBackend` trait does not expose unreduced types — this is an internal implementation detail of `pairwise_reduce`. Each backend chooses its own accumulator width and reduction strategy:

- **CPU** (`CpuBackend`): Uses `jolt-field`'s `FMAdd`, `SmallAccumU`, `MedAccumS`, `FullAccumS` types with Barrett reduction. Matches the current jolt-core performance.
- **Metal/CUDA**: Uses 512-bit or 768-bit integer types in shader local memory, with warp-level reduction and a single Montgomery reduction per thread group output.
- **WebGPU**: Limited to 32-bit integers in WGSL — uses multi-limb representation with carry propagation. Slower per-op but benefits from massive parallelism.

---

## 7. Compact Buffer Support

Following `jolt-poly`'s `Polynomial<T>` pattern, device buffers are generic over element type:

```rust
type Buffer<T: Scalar>: Send + Sync;
```

A `Buffer<u8>` uses 1 byte per element vs. 32 bytes for `Buffer<Fr>` — 32x memory savings. This matters for GPU where VRAM is limited.

The type transition happens at bind time:

1. **Initial upload:** Compact RA polynomials uploaded as `Buffer<u8>` (small)
2. **First bind:** `interpolate_pairs<u8, Fr>(buf, challenge)` returns `Buffer<Fr>` (promoted)
3. **Subsequent binds:** `interpolate_pairs<Fr, Fr>(buf, challenge)` returns `Buffer<Fr>` (identity `From`, compiled away)

Inside `pairwise_reduce`, input buffers may be compact or dense. The compiled kernel handles promotion internally — for `ProductSum` with `u8` inputs, the kernel reads bytes and promotes to field elements before the product evaluation. This keeps VRAM usage low while avoiding a separate promotion pass.

---

## 8. Escape Hatch for Bespoke Kernels

Not all sumcheck instances fit the `ProductSum` pattern. RAM read-write checking has 3 phases with distinct memory access patterns. The Spartan outer sumcheck has a multiquadratic evaluation structure.

The escape hatch works through `KernelShape::Custom`:

```rust
// RAM read-write phase 1: cycle-major sparse access
let rw_phase1_descriptor = KernelDescriptor {
    shape: KernelShape::Custom {
        expr: rw_phase1_expr(),  // built from jolt-ir
        num_inputs: 6,
    },
    degree: 3,
    tensor_split: Some(TensorSplit { outer_vars: 10, inner_vars: 10 }),
};
```

For truly bespoke cases where even `Custom` doesn't capture the structure (e.g., sparse matrix access patterns that can't be expressed as paired buffer reads), a backend can provide a **hand-written kernel** that implements the same `CompiledKernel` interface:

```rust
// In jolt-metal:
impl MetalBackend {
    /// Compile from a KernelDescriptor (standard path).
    pub fn compile(&self, desc: &KernelDescriptor) -> MetalKernel { ... }

    /// Load a pre-written MSL kernel for bespoke cases.
    pub fn load_custom_kernel(&self, msl_source: &str) -> MetalKernel { ... }
}
```

The witness in jolt-zkvm just holds a `B::CompiledKernel` — it doesn't know or care whether it was auto-compiled or hand-written. The `pairwise_reduce` interface is the same either way.

---

## 9. Open Questions and Future Work

### Immediate open questions

1. **Buffer lifetime on bind:** `interpolate_pairs` currently consumes the input buffer and returns a new one (needed for type transition `T → F`). For `F → F` rounds, should there be an in-place variant that avoids reallocation? The `From<F> for F` identity should allow the compiler to optimize this, but explicit in-place may be clearer for GPU backends.

2. **Mixed-type inputs to `pairwise_reduce`:** If some buffers in a product group are `Buffer<u8>` and others are `Buffer<Fr>`, the current signature requires all `Buffer<F>`. Should promotion happen before `pairwise_reduce` (simpler API) or inside it (less memory)?

3. **Batched sumcheck coordination:** The `BatchedSumcheckProver` combines round polynomials from multiple instances. With GPU backends, should each instance dispatch independently (simple, potentially underutilizing GPU), or should the batched prover fuse multiple instances into one dispatch (complex, better GPU occupancy)?

### Future work

- **MSM engine trait:** If MSM (multi-scalar multiplication) becomes a bottleneck independent of PCS, a similar `msm_reduce` primitive could be added to `ComputeBackend`.
- **NTT primitives:** If Jolt adds NTT-based polynomial operations (e.g., for univariate skip), the same pattern applies.
- **Streaming support:** For polynomials too large to fit in device memory, a `StreamingComputeBackend` extension with chunk-based processing.
- **Multi-device:** Splitting work across multiple GPUs or GPU + CPU hybrid.
- **Kernel caching:** AOT-compiled kernels cached to disk for instant startup on subsequent runs.
