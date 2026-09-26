# Spec: Metal Field Arithmetic (`jolt-metal`)

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @quangvdao                     |
| Created     | 2026-09-25                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

Jolt and Akita each carry Metal prover work: #1848 (Jolt sumcheck kernels and
Akita commitment kernels) and #1733 (BN254 / Dory). Each defines its own GPU
field arithmetic. The Solinas `Fp128` code exists as two near-identical copies
(`jolt-kernels/src/metal/solinas/fp128.metal` and Akita's `onehot.metal`), plus
about six per-kernel lazy-reduction variants. No implementation covers `Fp32`,
`Fp64`, or the extension fields, and none is generic over the modulus.

This spec defines one crate, `jolt-metal`, that owns two things. The first is
the Metal field arithmetic for `jolt_field::solinas`: generic MSL templates
that mirror the CPU types. The second is the small safe runtime every Metal
consumer needs. Jolt's and Akita's Metal kernels build on it instead of
redefining arithmetic. The crate is prover-only and cannot enter a verifier
dependency graph.

## Intent

### Goal

Provide MSL field types and operations for `jolt_field::solinas` whose results
are bit-identical to the CPU implementation. The types are templated over the
modulus, with lazy-reduction accumulators whose capacities are proved. Provide
a safe, typed, non-panicking Rust runtime for compiling, dispatching, and
reading back kernels that use them.

Key abstractions:

- **MSL field templates** (`shaders/jolt/field/*.h`). These define
  `jolt::Fp32<BITS, C>`, `jolt::Fp64<BITS, C>`, `jolt::Fp128<C>`,
  `jolt::Ext2<F, NR>` and `jolt::Ext4<F>`. Each template mirrors one
  `jolt_field` type and uses the same algorithm names (`reduce_product`,
  `mul_unreduced`, `mul_u64_unreduced`, …).
- **MSL accumulators** (`shaders/jolt/field/accum.h`). These mirror
  `jolt_field::WithAccumulator`: `Accumulator`, `SmallScalarAccumulator`, and
  `SignedProductAccumulator`. Each has a `constexpr` `CAPACITY`, the maximum
  number of worst-case terms before `reduce()`, derived in a comment next to
  its definition.
- **`MetalField` trait** (Rust). It is implemented for each supported
  `jolt_field` type. It supplies:
  - the MSL type spelling, for example `jolt::Fp128<0xFFFFA7F7u>`;
  - a stable host-name suffix, for example `fp128_a7f7`;
  - the device word layout;
  - a checked read-back conversion.
- **Runtime** (`jolt_metal::runtime`). This is a thin safe layer over
  `objc2-metal`. Its pieces are `Device`, `ShaderLibrary` (source assembly and
  explicit template instantiation), `Pipeline`, `DeviceBuffer<T>`, `Batch`
  (encode, commit, wait, classify errors), and `MetalError`.

### Invariants

1. **Bit-exact agreement.** For every supported field `F` and every operation
   `op`, the GPU result for canonical inputs equals the `jolt_field` CPU
   result byte for byte. Every GPU output is in canonical form `[0, p)`.
   Field arithmetic is exact, so dispatch order, threadgroup size, and
   reduction tree shape must not change any output.
2. **One arithmetic.** The Solinas reduction, the extension multiplication
   tables, and the lazy accumulators are defined once, in `jolt-metal`.
   Consumers include the headers and do not redefine them. Deleting the
   per-kernel copies in #1848 is part of adopting this crate.
3. **Verifier isolation.** `jolt-metal` is never a dependency, direct or
   transitive, of `jolt-verifier`, `akita-verifier`, or `jolt-field`. It is a
   separate crate, not a feature of `jolt-field`, because Cargo feature
   unification would otherwise compile Metal into the verifier in any build
   that also builds a prover. A dependency check enforces this in both repos.
4. **Platform isolation.** On non-macOS targets the crate compiles. The shader
   text and `MetalField` metadata remain available. `Device::system_default()`
   returns `MetalError::Unavailable`. No Metal symbol is linked and there is no
   `objc2` dependency.
5. **No panics.** Runtime code does not `unwrap`, `expect`, `panic!`,
   `assert!`, or index without a checked bound. Every failure is a typed
   `MetalError`. Arguments are validated before any Objective-C message
   send. As a second line of defence, every message send in `runtime/` goes
   through one helper that wraps it in `objc2::exception::catch` (the `objc2`
   `exception` feature) and maps a caught exception to a `Fault`. Without
   this, an uncaught Objective-C exception unwinding into Rust aborts the
   process. The `catch-all` feature is not used, because it converts
   exceptions into Rust panics.
6. **Checked read-back.** A device buffer becomes `&[F]` or `Vec<F>` only
   through a conversion that verifies canonical form. The check costs one
   comparison per element. Unchecked reinterpretation is not public API.
7. **Accumulator capacity.** Every accumulator's `CAPACITY` is checked by a
   test at exactly `CAPACITY` worst-case terms (all inputs `p − 1`, or
   `u64::MAX` for scalar terms). Any kernel that accumulates more terms than
   `CAPACITY` must reduce first. This is a documented kernel-author
   obligation, and kernels in this crate enforce it with `static_assert`.
8. **Setup-time failure only for compilation.** Every pipeline a consumer
   declares is compiled when its `ShaderLibrary` is built. Shader compile and
   pipeline creation failures surface before any proving work, never mid-proof.

### Non-Goals

- **Kernels above field arithmetic.** Sumcheck, NTT over the Akita CRT primes,
  ring arithmetic, and commitment kernels belong to their consumers
  (`jolt-kernels`, `akita-metal`).
- **BN254 / Montgomery fields.** #1733's `fr.metal` could later move under
  `jolt_metal::field::bn254` on the same runtime. That move is not part of
  this spec.
- **Field inversion on the GPU.** No planned consumer needs it. It will be
  added with its first caller.
- **`FpExt8`, and `Fp32` / `Fp64` before a consumer needs them.** The
  templates are written generically from the start. Instantiations,
  `MetalField` impls, and tests land with their first production caller, per
  the repository rule against speculative API.
- **Fallback policy.** Whether a consumer fails the proof or re-proves on the
  CPU after a GPU error is decided by the consumer. This crate only
  classifies errors (see Error model).
- **CUDA.** A CUDA backend would reuse the conformance vectors and the Rust
  test harness, not the MSL source.

## Evaluation

### Acceptance Criteria

- [ ] The normal and build dependency graphs of `jolt-verifier` and
      `jolt-field`, under `--all-features --target all`, contain no
      `jolt-metal` and no `objc2*` crate
      (`scripts/check-metal-isolation.sh`). The same holds for
      `akita-verifier` in Akita. CI enforces both checks.
- [ ] On `x86_64-unknown-linux-gnu`, `cargo clippy -p jolt-metal
      --all-targets -- -D warnings` passes with no Metal or `objc2` crate in
      the build graph.
- [ ] For each instantiated field, `jolt-metal` passes a conformance suite
      that runs every MSL operation through a generic test kernel and compares
      it with `jolt_field`. The inputs are:
      - fixed edge vectors: `0`, `1`, `p − 1`, `p − C`, `C`, `2^64 − 1`,
        `2^64`, and limb-boundary values;
      - at least 2^20 random pairs per operation from a fixed seed.
- [ ] Accumulator capacity tests run at exactly `CAPACITY` worst-case terms
      and at `CAPACITY + 1` for the documented pre-reduction path.
- [ ] Extension-field conformance against `jolt_field` for every `Ext2` /
      `FpExt4` instantiation a consumer uses (Akita today uses `Ext2` over
      its word fields and `FpExt4<F>` generically): multiply, square,
      and base-by-extension scaling.
- [ ] The conformance suite passes with Metal shader validation and API
      validation enabled (`MTL_SHADER_VALIDATION=1`,
      `MTL_DEBUG_LAYER=1`).
- [ ] `grep` over `crates/jolt-metal/src` (excluding `#[cfg(test)]`) finds no
      `unwrap(`, `expect(`, `panic!`, `assert!`, or `unreachable!`.
- [ ] #1848's `solinas/fp128.metal`, `simd_reduce.metal`, `deferred_sum.metal`,
      and the per-kernel wide accumulators are deleted in favour of
      `jolt-metal` headers. This is tracked in #1848's rebase, not in this
      crate's PRs.

### Testing Strategy

The ground truth is `jolt_field`'s CPU implementation. This follows the
repository's independent-oracle rule: the Metal code is a new
implementation, not a refactor of the CPU code.

- **Conformance.** A generic `#[cfg(test)]` harness is instantiated per
  `MetalField` type. It assembles test kernels (`vec_add`, `vec_mul`,
  `vec_fmadd_accum`, …) from the same headers consumers use.
- **Property tests** with `proptest` on a fixed seed:
  - ring axioms on device outputs;
  - canonical form of all outputs;
  - accumulator reduction equals the sum of fully reduced products.
- **Serialization.** nextest runs each test in its own process, so GPU tests
  take an exclusive file lock (`File::lock` on a file in the temp directory),
  following the `/tmp` flock in #1733. This avoids contention noise and makes
  hang attribution possible.
- **Hang guard.** A test-only watchdog, following `hang_watchdog.rs` in #1848,
  aborts the test process with a diagnostic when one test holds the GPU for
  more than two minutes. #1848 recorded four macOS kernel panics caused by
  GPU hangs, so validation-layer runs come before any unvalidated
  performance run.
- **CI.** The Linux jobs build the uninhabited non-macOS backend and run the
  dependency-graph check. A path-filtered macOS job lints the Objective-C
  backend, probes the runner's device, and runs the GPU tests only when the
  probe finds a supported device; otherwise it runs the host-only tests and
  says so.
- **Local report.** `scripts/metal-report.sh` runs the full suite on a local
  Apple Silicon machine, once normally and once under the API and shader
  validation layers. It emits a report with the device name, macOS version,
  GPU family, git SHA, and test results. Benchmark tables join it with the
  first benchmarks (step 2). Every `jolt-metal` PR description includes that
  report.

### Performance

Criterion benchmarks, one per operation and field:

- elementwise `add`, `mul`, and `square`;
- `fmadd` into an accumulator;
- simdgroup and threadgroup sum reduction.

Each is reported as elements per second at sizes 2^16–2^26, with a CPU
`jolt_field` baseline on the same machine: the NEON packed engine where it
exists, scalar otherwise.

Fp128 limb layout decision gate. The PR introducing `Fp128` benchmarks the
`uint4` (4×u32) representation used by #1848 and Akita against the 2×u64
representation from the earlier `quang/metal-field-kernels` branch. It keeps
the faster one and records both numbers. The expectation, not yet measured,
is that `uint4` wins, because Apple GPU ALUs are 32-bit and 64-bit multiplies
are emulated.

Regression bound: after the first measurement, a PR that changes an MSL
arithmetic header reports the table and a paired comparison against its base
(see Performance model). A regression above 3% on any `mul` or `fmadd` case
needs justification in the PR.

### Performance model

Invariant 1 fixes every output, so a kernel's configuration can change only
its speed. This section fixes how speed is measured and reported. Decisions
then rest on numbers, and per-machine tuning (see Direction) needs no kernel
rewrite.

**Machine limits.** A benchmark, `benches/limits.rs`, measures the resources a
kernel can be bound by, on the machine that runs it:

| Limit | Measured as |
|---|---|
| field multiply | independent `Fp128` multiplies per second, with enough threads to hide latency |
| deferred multiply-accumulate | `fmadd` into an accumulator, reduced once per `CAPACITY` terms |
| memory bandwidth | a streaming copy, at sizes inside and beyond the system-level cache |
| threadgroup memory bandwidth | 16 B loads per second from threadgroup memory |
| round trip | from committing a batch to the host observing its result, for an empty batch and for one reduction to a single element |

The report prints these next to the device descriptor. The machine's ridge,
bandwidth divided by multiply rate, says which kernels are compute-bound. On
an M4 Max, step 2's provisional figures (about 45 G multiplies/s, about
430 GB/s) put the ridge near 1.7 multiplies per 16 B element read. That is
half an RTX 5090's, about 3.3 (327 G multiplies/s at 1.6 TB/s). So on Apple
GPUs any kernel doing more than about two multiplies per element it reads is
compute-bound, and the multiply and multiply-accumulate rates are the limits
that matter most.

**Kernel criteria.** From step 3 on, every kernel PR:

- states the kernel's work per element (multiplies, multiply-accumulates,
  bytes read and written) and, from that, the limit that bounds it;
- reports the measured rate at a representative size as a fraction of that
  limit on the reporting machine. A kernel below half of its limit states why,
  or what would close the gap;
- exposes its tuning knobs (threadgroup size, elements per thread, terms
  accumulated before a reduction, tile sizes) as template parameters or
  function constants. Each knob has a documented valid range that is small and
  finite, and a default that is valid on every supported device;
- runs conformance at every value of every knob in its valid range.

**Measurement hygiene.**

- Absolute rates come from an otherwise idle machine on AC power in high power
  mode. The report records the power source, the energy mode, and the load
  average before and after the benchmarks. A run whose 1-minute load average
  exceeds 1 does not supply absolute rates. Other processes share the chip's
  power budget: at load 32–51 on 16 cores, step 2's GPU multiply chain
  measured 29 G/s instead of 45, and the CPU baseline varied 2.5–20×.
- A choice between variants uses a paired comparison. Each round times every
  variant back to back in alternating order, and the result is the median
  per-round ratio with its 10th–90th percentile spread. The decision rule is
  fixed before the run. Under the load above, step 2's paired ratios
  reproduced within 2%.

**No runtime autotuning.** A kernel's configuration is a pure function of the
kernel, the problem shape, and the device descriptor. It comes from
checked-in data or the default. Nothing is timed during a proof.

## Design

### Architecture

```
jolt-field (CPU types, verifier-reachable)
    ▲
    │ normal dependency (types, OFFSET, ext tables)
jolt-metal (prover-only; macOS runtime, portable shader text)
    ├── shaders/jolt/field/{fp32,fp64,fp128,ext2,ext4,accum,reduce}.h
    ├── src/field.rs        MetalField trait + impls for instantiated types
    ├── src/runtime/        Device, ShaderLibrary, Pipeline, DeviceBuffer, Batch
    └── src/error.rs        MetalError + ErrorClass
    ▲                        ▲
jolt-kernels (feature metal)  akita-metal (Akita `dev`, opt-in)
```

**Shader genericity.** The Metal Shading Language is C++14-based. Field types
are class templates whose non-type parameters are the modulus shape
(`BITS`, `C`). The `C` parameter is a compile-time constant, so a multiply by
a small `C` folds. Consumer kernels are function templates over the field
type:

```metal
template <typename F>
kernel void jolt_vec_mul(device const F* a [[buffer(0)]],
                         device const F* b [[buffer(1)]],
                         device F* out     [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    out[i] = a[i] * b[i];
}
```

`ShaderLibrary` generates the explicit instantiations from the requested
`(kernel template, MetalField)` pairs, and pipelines are looked up by the
generated host name:

```metal
template [[host_name("jolt_vec_mul_fp128_a7f7")]] [[kernel]]
decltype(jolt_vec_mul<jolt::Fp128<0xFFFFA7F7u>>) jolt_vec_mul<jolt::Fp128<0xFFFFA7F7u>>;
```

Kernel source is written once and instantiated per field. No offset value is
hand-written in a consumer, and the `C < 2^32` precondition is a
`static_assert` in `Fp128`, the same one the CPU `Fp128::C` const-asserts.

**Host–device layout.** `Fp128<P>` is `#[repr(transparent)]` over `[u64; 2]`
(little-endian, canonical). On little-endian Apple Silicon its bytes equal
MSL `uint4` little-endian words. Upload is therefore a byte copy, with a
`const` assertion on size and layout. Read-back goes through the checked
conversion (invariant 6). This needs one small `jolt-field` addition: a
checked constructor from canonical limbs or bytes, or a `bytemuck`
`CheckedBitPattern` impl. It is pure, allocation-free, and has no platform
code. Device buffer offsets are required to be multiples of 16 so that
`device uint4*` accesses are aligned.

**Shader packaging.** `ShaderLibrary` compiles embedded source at runtime
(`newLibraryWithSource`), for three reasons:

- it needs no Xcode or Metal toolchain at build time. A machine with only
  the Command Line Tools has no `metal` compiler (`xcrun --find metal`
  fails), but it can still compile shader source at runtime;
- all four existing implementations already do this;
- invariant 8 moves the failure to setup time.

Consumers get the header text through `jolt_metal::shaders::FIELD_HEADERS` and
include it in their own libraries. A precompiled `.metallib` or
`MTLBinaryArchive` cache can be added later as a load-time optimisation
without changing the API.

**Platform floor.** The minimum is Apple GPU family 7 (M1, so every Apple
Silicon Mac) and MSL 3.0 (macOS 13). The runtime uses the classic Metal
command API, not the `MTL4*` API, which requires macOS 26. Metal 4 command
allocators can be evaluated later behind the same runtime types.

**Bindings.** `objc2-metal` (0.3.x), which `wgpu-hal` and #1733 also use.
`metal-rs` describes itself as deprecated in favour of `objc2-metal`.
Nearly all `objc2-metal` calls are `unsafe`. Each `unsafe` block in
`runtime/` carries a `// SAFETY:` comment naming the validated precondition.

### Error model

`MetalError` is a typed enum. Every variant carries an `ErrorClass` that tells
the consumer what it may do:

| Class | Examples | When | Consumer may |
|---|---|---|---|
| `Unavailable` | no device; non-macOS; OS below the minimum GPU family | construction | choose the CPU backend |
| `Setup` | shader compile error; missing entry point; pipeline creation | construction (invariant 8) | choose the CPU backend; report it as a bug |
| `Capacity` | buffer above `maxBufferLength`; working set above `recommendedMaxWorkingSetSize`; allocation returned nil | before encoding | re-plan or use the CPU for this job |
| `Transient` | `Timeout` (2); `OutOfMemory` (8); `AccessRevoked` (4); `NotPermitted` (7) | after commit | retry or fall back at the consumer's granularity |
| `Fault` | `PageFault` (3); `InvalidResource` (9); `StackOverflow` (12); `Internal` (1); an unknown code; a caught Objective-C exception; a dispatch whose bindings or grid do not match the kernel; an undeclared pipeline name; non-canonical read-back | before encoding (misuse) or after commit | treat as a bug: never retry silently, fail loudly |

The numbers are `MTLCommandBufferError` codes from the macOS SDK header
`MTLCommandBuffer.h`. `DeviceRemoved` (11) is deprecated because it "cannot
occur on Apple Silicon", and `Memoryless` (10) applies only to render
targets. Neither is mapped. After a command buffer ends in an error, the
runtime reads `MTLCommandBuffer.error` and maps its code into these classes.
A batch is one command buffer with one serial compute encoder, so Metal's
per-encoder execution status cannot narrow a fault to a dispatch. Instead a
`CommandBuffer` error lists the distinct pipelines the batch dispatched, in
first-use order. A consumer that needs single-dispatch attribution, such as a
diagnostic re-run, splits the batch. #1848 and Akita's `akita-metal` only
compare the status to `Completed`, so they cannot tell a timeout from a page
fault.

Metal's shader validation layer does not change a command buffer's status:
an out-of-bounds access is logged, the access is dropped, and the command
buffer completes. It therefore never produces a `MetalError`. The local
report sets `MTL_SHADER_VALIDATION_ABORT_ON_FAULT=1` so a validation fault
fails the test that caused it.

This crate does not decide fallback. The consumer spec (Akita's Metal
backend) must decide:

- the fallback granularity: operation, stage, or whole proof. It depends on
  whether inputs are still resident on the host;
- whether the prover verifies its own proof before emitting it;
- how fallbacks are surfaced in telemetry.

### Alternatives Considered

- **A `metal` feature on `jolt-field`.** Rejected. Resolver-2 feature
  unification would build Metal into `jolt-verifier` in any workspace or
  `jolt-sdk` host build that also builds a prover.
- **Modulus in a runtime buffer** (the earlier `quang/metal-field-kernels`
  branch). Rejected. The multiply by `C` cannot fold, and one generic
  instantiation hides which field a kernel was validated for.
- **Modulus via a `#define` prefix** (#1848). This works for one field per
  library, but it cannot put two fields in one library. Akita's CRT paths
  mix `Fp128` with word fields. It also does not extend to extension types.
- **Metal function constants.** These specialise values at pipeline creation,
  but not types, so they cannot cover `Fp32` / `Fp64` / `Fp128` / extension
  genericity. They could later select `C` within one width to shrink the
  library. That is not needed now.
- **Slang, CubeCL, rust-gpu, or wgpu/WGSL.** Rejected for this layer:
  - WGSL has no 64-bit integers and no templates.
  - Slang adds a third-party compiler to the trusted path and makes the
    emitted MSL harder to audit and tune.
  - rust-gpu does not target Metal natively.
  - CubeCL is not mature for multi-limb 128-bit arithmetic.

  Hand-written MSL templates are auditable line by line against `jolt_field`.
- **Keeping the runtime per consumer.** Rejected. #1848 (`metal-rs`) and
  #1733 (`objc2-metal`) already diverge on bindings and error handling. The
  runtime is where panics, unsafe code, and error classification live, so it
  should be audited once.

## Documentation

- Add a book page `book/src/how/metal.md` covering the crate boundary, the
  field templates, the kernel-author obligations (accumulator `CAPACITY`,
  aligned buffers, declared pipelines), and the error classes.
- Add a crate README with the local report procedure.

## Execution

Each PR is independently reviewable, carries the local report, and adds no
API without a caller in the same PR or the documented external contract in
this spec. The external contract is Akita's `akita-metal` on Akita `dev`,
together with #1848.

1. **Runtime.** Contents:
   - `Device` and `DeviceLimits`;
   - `LibrarySpec` (sources, plain kernels, and template instances over an
     `MslType`, named by `host_name::<T>(template)`) and `ShaderLibrary`
     (eager pipelines with reflected buffer arguments);
   - `DeviceBuffer<T>` (`from_slice`, `zeroed`, and checked `read`);
   - `Batch`, `Binding`, and `Grid`: every dispatch is checked against the
     kernel's reflected signature before encoding;
   - `MetalError` / `ErrorClass` with `MTLCommandBufferError` mapping;
   - an uninhabited non-macOS backend;
   - the dependency-graph check in Jolt CI, the macOS probe job, and the
     local report script.

   Tested with a vector-add template instantiated for `u32` and `u64`, plus
   a byte-fill kernel for the read-back check. `MslType` is the seam that
   `MetalField` extends in step 2.
2. **`Fp128`.** Contents:
   - `fp128.h`: add, sub, neg, mul, square, `mul_u64`, `mul_i64`,
     `from_u64`, `from_i64`;
   - the `MetalField` impls for `Prime128OffsetA7F7` and `Prime128Offset275`;
   - the checked read-back addition to `jolt-field`;
   - conformance and property suites;
   - benchmarks, including the limb-layout A/B.
3. **Accumulators and machine limits.** `accum.h` mirrors `Fp128Accumulator`
   and `Fp128SignedAccumulator` with proved `CAPACITY`. Also simdgroup and
   threadgroup reductions that are generic over the accumulator, and
   `benches/limits.rs` (Performance model), whose multiply-accumulate limit
   needs the accumulators. The kernel criteria apply from this step.
4. **Extensions.** `Ext2`, and `FpExt4` in the `[1, e1, e2, e3]` cyclotomic
   basis matching `PseudoMersenne::ext4_mul`.
5. **Word fields.** `Fp64<BITS, C>` / `Fp32<BITS, C>` for `Prime64Offset59` /
   `Prime32Offset99`, landed when `akita-metal` first needs them.
6. **Adoption.** #1848 (Jolt) and `akita-metal` (Akita `dev`) switch to these
   headers and delete their copies. The Akita side follows its own spec:
   ring, NTT over CRT primes, commitment, fold, and range sumcheck layers.

## Direction: per-machine plans

This section is a direction, not part of this spec's scope. The Performance
model keeps it possible without rewriting kernels.

The protocol is fixed. A configuration may change how fast the prover runs,
never its output (invariant 1) or the proof bytes. Within that, the aim is a
prover that runs each kernel in the fastest configuration for the machine it
is on, as FFTW's planner and cuBLAS's per-architecture heuristics do.

1. **Device descriptor.** GPU family is not enough. An M4 MacBook Air and an
   M4 Max are both family 9, but by Apple's published figures they differ
   about 4× in GPU cores (8–10 against 40) and about 4.5× in memory
   bandwidth (120 against 546 GB/s). The descriptor adds:
   - the GPU core count, which Metal does not expose but the IORegistry does
     (`gpu-core-count`);
   - the threadgroup memory size and the recommended working set;
   - the measured machine limits.
2. **Plan tables.** Checked-in data maps (kernel, shape class, device class)
   to knob values. It is reviewed like code, and the default applies when
   nothing matches.
3. **Offline tuner.** This generalizes the paired-comparison harness: it
   searches knob values and code variants on one machine, checks each
   candidate's output against `jolt_field`, and emits a table entry for
   review. It runs during development, never inside a proof. Step 2's
   limb-layout and squaring decisions are the manual version. The squaring
   gain (2.06×) came from rewriting the code, not from any parameter, so code
   variants belong in the search space.
4. **Pipeline cache.** `MTLBinaryArchive` stores compiled pipelines per
   device, so specializing for a machine costs compile time once.
5. **Fusion.** The largest remaining gains are structural: fusing the passes
   of a sumcheck round, keeping data resident between rounds, and doing fewer
   reductions. A generator that emits fused kernels from a description of a
   round is the end state. It waits until at least three kernel families
   (sumcheck, NTT, commitment matvec) exist by hand, so that it abstracts
   patterns that have been seen.
6. **CPU and GPU together.** Unified memory lets a plan split one phase
   between the CPU and the GPU without copies.

## Resolved questions

These were checked on 2026-09-25 on an Apple M4 Max running macOS 27.0, with
only the Command Line Tools installed.

- **Templated kernels.** Explicit instantiation with `[[host_name]]` compiles
  under both MSL 2.4 and MSL 3.0. Two `Fp128<C>` instantiations built from one
  template are both listed by the library and both dispatch correctly. A
  `static_assert` on a template parameter fails `newLibraryWithSource` with
  `MTLLibraryErrorDomain` code 3, so an invalid field instantiation is a
  `Setup` error. A small library compiled in about 220 ms; full-library
  compile time will be measured with the `Fp128` library in step 2.
- **Objective-C exceptions.** `objc2` 0.6 lets an uncaught exception unwind
  into Rust, which in practice aborts. `catch-all` wraps every send but
  panics on a caught exception. `exception::catch` returns a `Result`, and
  is used as described in invariant 5. It cannot catch anything under
  `panic = "abort"`. Neither the Jolt nor the Akita workspace sets that for
  host profiles, and the crate README states the dependency.
- **Command-buffer error codes.** These are taken from the SDK header, as
  listed in the error table.
- **Platform floor.** Apple GPU family 7 and MSL 3.0, as described under
  Platform floor.
- **CI.** The GitHub-hosted `macos-latest` runner exposes an `Apple
  Paravirtual device` below Apple GPU family 7. The probe job therefore skips
  the GPU tests there and runs the host-only tests. GPU evidence comes from
  the local report.

## References

- a16z/jolt#1848: Metal prover backend (Jolt sumcheck kernels, Akita commitment).
- a16z/jolt#1733: experimental Metal backend (BN254 / Dory, `objc2-metal`).
- `crates/jolt-field/src/solinas/{fp128.rs,word.rs,ext.rs,unreduced.rs}`,
  `crates/jolt-field/src/fp128_accumulators.rs`.
- LayerZero-Labs/akita `docs/branches.md`: compute backends target `dev`,
  seams target `main`, and backends must produce CPU-identical proof bytes.
- Earlier prototype: `quang/metal-field-kernels` (Akita legacy repository).
