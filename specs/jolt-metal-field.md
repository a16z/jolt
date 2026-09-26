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
  `jolt::Fp128<C>` and `jolt::Fp64<C>`, the fields `2^128 − C` and
  `2^64 − C` for odd `C < 2^32`, and `jolt::Ext2<F>`, the quadratic
  extension with non-residue 2 over either. `Fp32` and `FpExt4` follow the
  same pattern when a consumer needs them (see Non-Goals). Each template
  mirrors one `jolt_field` type and uses the same algorithm names
  (`reduce_product`, `fold2_canonicalize`, …).
- **MSL accumulators.** `shaders/jolt/field/accum.h` states the contract and
  mirrors `jolt_field::WithAccumulator` with `Accumulator` and
  `SmallScalarAccumulator`; each field header specializes it (`fp128_accum.h`
  for `Fp128`). Each accumulator has a `constexpr` `CAPACITY`, the number of
  terms it holds exactly, derived in a comment next to its definition.
  `shaders/jolt/field/reduce.h` sums accumulators over a simdgroup and a
  threadgroup, generically over the contract. `SignedProductAccumulator`
  lands with its first consumer.
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
- **`Fp32`, `FpExt4` and `FpExt8` before a consumer needs them.**
  Instantiations, `MetalField` impls, and tests land with their first
  production caller, per the repository rule against speculative API.
  `Fp64` and `Ext2` landed in step 4 because Akita's `fp64` preset
  (`akita-config`'s `proof_optimized/fp64.rs`: `Field = Prime64Offset59`,
  `ExtensionField = Ext2<Field>`) uses them as its base and extension
  fields. The `fp32` preset's `Prime32Offset99` and `FpExt4` follow in step 5.
- **`Fp64` moduli below `2^63`.** `jolt_field`'s `Fp64<P>` also covers
  sub-word moduli, which fold at a different bit. `MetalField` for such a
  `P` is a build error.
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
      `unwrap(`, `expect(`, `panic!`, `assert!`, or `unreachable!`, except in
      `const fn`s evaluated only in constants, where a failure is a build
      error (`field.rs` spells MSL type names from `P` this way).
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
- **Branch coverage.** Random inputs almost never reach the rare reduction
  branches (the second fold's overflow and its canonicalization), so the
  suite also builds inputs for each branch from the modulus. It recomputes
  each reduction's intermediate values in `u128` arithmetic and asserts that
  every branch of `fold2_canonicalize` is taken for `mul` and `mul_u64`, and
  that `add`'s wrap and canonicalization and `sub`'s borrow occur.
- **`Fp64` and `Ext2`** (`tests/fp64.rs`, `tests/ext2.rs`, step 4). The
  same harness and branch assertions, over the moduli that reach each
  bound at its limit. `Fp64` runs over `Prime64Offset59` and
  `2^64 − 2^32 + 1`, whose offset is the largest `jolt::Fp64` accepts.
  `Ext2` runs over four bases: `Prime64Offset59`; `2^64 − 0x7fffffd3`,
  whose offset is the largest prime offset below `2^31`, the bound of the
  `Fp64` forms that reduce a sum of three products once; and, through the
  generic Karatsuba forms, `2^64 − 2^32 + 1` and `Prime128Offset275`.
  A shared model (`tests/support/fp64.rs`) recomputes the reduction in
  `u128` arithmetic, and the suites assert that every `fold2_canonicalize`
  branch is taken by `mul`, by `mul_u64`, and by each coefficient of the
  `Ext2` multiply, and that one `c1` sum carries into its top word through
  the low word, which random operands reach with probability about
  `2^−64`. Inputs for the rare branches are built from the modulus:
  with `a = 2^63` and `b = 2m`, the product is `m · 2^64`, and `m` is chosen
  so that `C·m` lands just below `(k + 1) · 2^64`. The `2^64 − 0x7fffffd3`
  suite caught a real bug during development: the carry into the second
  fold, up to `3C`, was held in 32 bits, which overflows for `C` near
  `2^31`. Since `Ext2` over non-field bases is exercised (the Goldilocks
  prime has `p ≡ 1 (mod 8)`, so `u^2 − 2` splits), the suites test the
  arithmetic, not the field axioms, which `jolt_field` covers. Of 40
  mutants of `fp64.h` and `ext2.h`, covering each carry, fold, sign step,
  coefficient term and the non-residue, 39 fail the suites. The low-word
  carry was added after its mutant first survived. The other survivor is
  equivalent: narrowing the bound of the reduce-once overloads, which
  changes speed, not results. Widening it past `2^31` fails to compile,
  through `reduce_sum`'s `static_assert`.
- **Mutation testing.** The suite's strength is checked by hand-made
  mutants of each carry, shift, fold, and sign-handling step. In step 2,
  all 23 mutants of the code in the final `fp128.h` fail the suite. Two
  other mutants survived because they were equivalent: shifting a word that
  is always zero, in the loop-form `sqr_wide` since replaced, and reading
  `sub128`'s borrow from bit 32 instead of bit 63, which agree for every
  input. Canonical outputs are enforced by the checked read-back, and
  bit-exact agreement with `jolt_field` implies the ring axioms, so there are
  no separate property tests for base-field operations.
- **Accumulator conformance** (`tests/fp128_accum.rs`, step 3). For both
  instantiated moduli, the accumulator property: every accumulator's
  `reduce()` equals the `jolt_field` sum of its terms. Edge operands and
  2^20 fixed-seed random terms, with every operation interleaved, are summed
  by `threadgroup_merge` at one and at 16 terms per thread, in threadgroups of
  1, 2, 3 and 8 simdgroups and the pipeline's largest whole-simdgroup size.
  Every lane's result is checked. The small-scalar edges are 0, 1, 2,
  2^32 − 1, 2^32, 2^63 − 1, 2^63, 2^63 + 1, 2^64 − 2 and 2^64 − 1, each with
  both signs. Capacity (invariant 7) is checked at exactly `CAPACITY`
  worst-case terms, and at `CAPACITY + 1` both through the documented
  pre-reduction path, which must be exact, and without it, which must
  differ. The signed accumulator is filled with each sign. Of 19 mutants of
  the carry, fold, sign and merge steps in `fp128_accum.h` and `reduce.h`,
  17 fail the suite. The other two are equivalent: negating a zero scalar
  product in `fmadd_i64`, and writing the simdgroup sum from lane 1 instead
  of lane 0, which hold the same value after the butterfly.
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

Criterion benchmarks (`crates/jolt-metal/benches/field.rs`) for `Fp128`
(offset `0xA7F7`), `Fp64` (offset 59) and `Ext2` over that `Fp64`:

- dependent chains of `add`, `mul`, and `square` in registers, plus `mul`
  with four independent chains, reported as operations per second: the ALU
  cost of each operation;
- elementwise `add`, `mul`, and `square` at 2^16–2^26 elements;
- an inner product with a threadgroup reduction at 2^16–2^26 elements, the
  shape of a sumcheck round;
- for `Fp128`, `fmadd`, `fmadd` with four independent accumulators, and
  `fmadd_i64` in registers, and an inner product whose products are
  accumulated unreduced and summed by `threadgroup_merge` (step 3).

GPU samples are GPU execution time from the command buffer's timestamps
(`Batch::commit_and_wait` returns it), which excludes host submission. The
CPU baseline is `jolt_field` on all cores with rayon and the `asm` multiply
Akita's prover uses. The packed NEON `Fp128` multiplies lane by lane through
that same scalar path, so it is not a separate baseline. Every kernel's output
is checked against the CPU before it is timed.
`scripts/metal-report.sh --bench` appends the table to the local report.

**Fp128 limb layout.** The earlier `quang/metal-field-kernels` 2×u64 code
built each 64×64 product from four 32×32 multiplies and read `C` from a
buffer, so comparing it with `uint4` would have measured those choices, not
the layout. The comparison run instead was `uint4` schoolbook against a
`ulong2` port of the same header using MSL's native 64-bit `*` and `mulhi`,
with `C` a template constant in both and the same storage. It lives on the
unmerged branch `metal/fp128-limb-ab` (`benches/limb_ab.rs`), so it can be
rerun on other chips. Each round times every variant in alternating order,
and the result is the median per-round ratio against `uint4`, a paired
comparison. The decision rule, fixed in advance, was to take a variant only
if it is faster on the ALU-bound and inner-product cases by more than the
round-to-round spread, and otherwise to keep the simpler code.

Result on an Apple M4 Max (macOS 27.0, 31 rounds; battery power, high power
mode), time of `ulong2` relative to `uint4`:

| case | `ulong2` / `uint4` (p10–p90) |
|---|---|
| dependent `add` | 1.072 (1.072–1.073) |
| dependent `mul` | 1.495 (1.495–1.495) |
| four independent `mul` chains | 1.625 (1.625–1.629) |
| dependent `square` | 1.910 (1.899–1.918) |
| streaming `mul`, 2^24 | 0.999 (0.995–1.006) |
| inner product, 2^20 | 1.057 (1.033–1.078) |
| inner product, 2^24 | 1.048 (1.024–1.070) |

`uint4` is kept. Streaming `mul` ties because at 2^24 both reach about
430 GB/s, near the memory bandwidth. Every pipeline reported 1024 maximum
threads per threadgroup, so neither layout limits occupancy through register
pressure.

The same run changed `square`. The triangular cross-product loop ported
first ran at 29 G/s, slower than `a * a` at 45 G/s. Written out, with each
square added in one multiply-add step, it runs at 60 G/s (paired ratios
against it: loop 2.055, `a * a` 1.322).

A second run, on AC power with other processes loading the machine (load
average 32–51 on 16 cores), reproduced every ratio within 2%: 1.073, 1.506,
1.643, 1.939, 1.001, 1.076 and 1.040 in the table's order, and 2.090 and 1.330
for `square`. Load moved the absolute rates of both runs, so the rates above
are indicative only; the ratios hold because each round times every variant
back to back.

**Fp64 and Ext2 forms** (step 4). A paired A/B on the unmerged branch
`metal/fp64-ext2-ab` (`benches/fp64_ab.rs`) chose the product, the square,
and the `Ext2` multiply and square. Each variant is the merged header with
exactly one function replaced. The rule, fixed before the first round, is
the limb-layout rule made explicit: the fastest variant on the dependent
chain wins if it beats every other there by at least 3% and is at most 3%
slower than the best on four chains and on the inner product at 2^20;
otherwise the simplest variant within 3% of the best on the chain wins.
Four rounds ran on an M4 Max on AC power, at load 9–37, each with 31
rounds per case. Times relative to the variant named first, median per-round
ratio:

| round | choice | chain | four chains | inner product 2^20 |
|---|---|---|---|---|
| 1 | 64×64 product: row-by-row (as `fp128.h`) / cross products first | 0.848 | 0.838 | 0.994 |
| 1 | the same, MSL `*` and `mulhi` / cross products first | 1.073 | 1.064 | 0.996 |
| 2 | square: `mul_wide(a, a)` / three-product square | 0.847 | — | — |
| 3 | square: row-by-row, cross product once / `mul_wide(a, a)` | 1.001 | — | — |
| 3 | `Ext2` multiply: reduce each coefficient once / Karatsuba | 0.962 | 0.987 | 1.015 |
| 3 | `Ext2` multiply: base-field `dot2` / Karatsuba | 0.992 | 0.992 | 1.004 |
| 3 | `Ext2` multiply: schoolbook / Karatsuba | 1.108 | 1.130 | 1.010 |
| 3 | `Ext2` square: reduce `c0` once / generic | 0.914 | — | — |
| 3 | `Ext2` square: base-field `dot2` / generic | 0.949 | — | — |
| 4 | merged `Ext2` multiply / Karatsuba | 0.960 | 0.985 | 1.009 |
| 4 | merged `Ext2` square / generic | 0.910 | — | — |

The row-by-row product is 15% faster than the four-product form it
replaced, and the three-product square did not beat squaring through it, so
`square(a)` is `a * a`. The `Ext2` multiply over `Fp64<C>` with `C < 2^31`
sums each coefficient's products, `a0 b0 + 2 a1 b1` and `a0 b1 + a1 b0`,
unreduced and reduces once: four base products and two reductions against
Karatsuba's three and three. In round 3 it ran the chain 3.8% faster than
Karatsuba and 3.0% faster than `dot2` (from the two medians against
Karatsuba, just over the rule's margin). `dot2` sums two products and
reduces once, which is valid for every `C < 2^32`, but it must reduce the
doubled `a1` first. On inner products every form ties within the
round-to-round spread, since the kernel is memory-bound; at 2^24 the merged
multiply measured 1.4–1.7% slower than Karatsuba in rounds 2–4, inside that
spread. Round 4 timed forced Karatsuba and generic squaring against the
merged forms (1.042, 1.015, 0.991 and 1.099); the table inverts those
ratios. Offsets from `2^31` up keep Karatsuba: there a sum of three
products can exceed the bound `C (t2 + 1) ≤ p` of `fold2_canonicalize`.

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
| deferred multiply-accumulate | `fmadd` into an accumulator, reduced once per 256 terms |
| memory copy | `out[i] = in[i]` on 16 B words, at sizes inside and beyond the system-level cache |
| memory read | four strided 16 B loads summed per thread, one write, at the same sizes |
| threadgroup memory bandwidth | 16 B loads per second from threadgroup memory |
| round trip | from committing a batch to the host observing its result, for an empty batch and for one reduction to a single element |

The report prints these next to the device descriptor. The machine's ridge,
bandwidth divided by multiply rate, says which kernels are compute-bound.

Copy is not an upper bound for a kernel that only reads. On an M4 Max in
step 3, reads beyond the system-level cache ran at about 440 GB/s and copies
at about 415 (both directions counted), and the inner products measured 1.04
of the copy rate. A read-only kernel is compared with the read limit, and a
kernel that writes as much as it reads with the copy limit.

On that machine, at load about 30, step 3 measured 42.9 G multiplies/s,
46.8 G multiply-accumulates/s and 444 GB/s read at 2^26 elements. That puts
the ridge at about 1.5 multiplies per 16 B element read, under half an
RTX 5090's, about 3.3 (327 G multiplies/s at 1.6 TB/s). So on Apple GPUs any
kernel doing more than about two multiplies per element it reads is
compute-bound, and the multiply and multiply-accumulate rates are the limits
that matter most.

The round trip is dominated by the host. Reducing 1024 elements to one and
reading it back took 135–152 µs from commit to observation (medians of two
runs), of which the GPU spent 8 µs; an empty batch took 33–36 µs. A protocol step that waits on the GPU
between rounds pays that per round, so kernels that end in a host decision
batch as much work as the protocol allows before it.

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

- Benchmarks run on AC power in high power mode. The report records the power
  source, the energy mode, and the load average before and after the
  benchmarks. Other processes share the chip's power budget: at load 32–51 on
  16 cores, from CPU-only work, step 2's GPU multiply chain measured 29 G/s
  instead of 45, and the CPU baseline varied 2.5–20×. An idle development
  machine is rarely available, so absolute rates are reported with the load
  they were measured under, and no comparison is drawn between absolute rates
  from different runs.
- A choice between variants uses a paired comparison. Each round times every
  variant back to back in alternating order, and the result is the median
  per-round ratio with its 10th–90th percentile spread. The decision rule is
  fixed before the run. Under the load above, step 2's paired ratios
  reproduced within 2%.
- A kernel's fraction of its limit is measured the same way: each round times
  the kernel and the benchmark of its bounding limit back to back. That makes
  the fraction a paired ratio. In step 3, runs at load about 30 and about
  100 gave fractions within 3% of each other (stream `mul` 1.028 and 1.032
  of copy; inner product 0.976 and 0.996, and accumulator inner product
  0.959 and 0.989, of read), so fractions are compared across runs the way
  variant ratios are. Unpaired in-cache rates are not: copying 4 MiB ran at
  1880 GB/s in the first run and 460 in the second, while reading 4 MiB ran
  at about 1500 in both. The cause is not known.

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
    ├── shaders/jolt/field/{fp32,fp64,fp128,ext2,ext4,accum,fp128_accum,reduce}.h
    ├── src/field.rs        MetalField trait + impls for instantiated types
    ├── src/runtime/        Device, ShaderLibrary, Pipeline, DeviceBuffer, Batch
    └── src/error.rs        MetalError + ErrorClass
    ▲                        ▲
jolt-kernels (feature metal)  akita-metal (Akita `dev`, opt-in)
```

**Shader genericity.** The Metal Shading Language is C++14-based. Field types
are class templates whose non-type parameter is the offset `C`, and
extensions are templates over their base field. The `C` parameter is a compile-time constant, so a multiply by
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
conversion (invariant 6). This is one small `jolt-field` addition behind an
optional `bytemuck` feature: `Zeroable`, `NoUninit`, and `CheckedBitPattern`
for `Fp128<P>`, whose validity check is `limbs < P`, and likewise for
`Fp64<P>` (over `u64`) and `FpExt2<F, C>` (over `[F; 2]`, valid when both
coefficients are). It is pure,
allocation-free, and has no platform code. Device buffer offsets are required to be multiples of 16 so that
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
   - `fp128.h`: `jolt::Fp128<C>` with add, sub, neg, mul, square, `mul_u64`,
     `mul_i64`, `from_u64`, `from_i64`, ported from #1848's
     `fp128.metal` with the `LONG_MIN` negation fixed and each bound argued;
   - `MetalField`, implemented for every `Fp128<P>`, with the MSL spelling
     and host suffix computed from `P` at compile time;
   - the `bytemuck` feature of `jolt-field` for byte views and checked
     read-back;
   - the conformance suite with asserted branch coverage, checked by
     mutation testing;
   - GPU timestamps on `Batch`, the benchmarks, the limb-layout A/B, and the
     `--bench` option of the local report.
3. **Accumulators and machine limits.** `accum.h` and `fp128_accum.h` mirror
   `Fp128Accumulator` and `Fp128SignedAccumulator` with proved `CAPACITY`.
   Also simdgroup and threadgroup reductions that are generic over the
   accumulator (`reduce.h`), and `benches/limits.rs` (Performance model),
   whose multiply-accumulate limit needs the accumulators. The kernel
   criteria apply from this step. The layouts were chosen by a paired A/B on
   an M4 Max, with the rule fixed in advance: the fastest variant on `fmadd`
   wins if it beats every other by at least 3% and is at most 3% slower than
   the best on `fmadd` in four chains; otherwise the variant with the fewest
   words within 3% of the best on `fmadd` wins, since consumer kernels spend
   registers on other state. A 288-bit carried accumulator in 9 words was
   within 3% of eight `ulong` slots and of eight uncarried column sums, both
   16 words, on `fmadd`, on `fmadd` in four chains and on an inner product at
   2^24, and 3% faster at 2^20. Reducing every product was 1.40× slower on
   `fmadd`. A 224-bit two's-complement signed accumulator in 7 words was
   1.16× faster than a positive and negative pair (14 words), and 1.82×
   faster than reducing every product. The A/B harness is kept on a branch,
   not merged. `SignedProductAccumulator` is deferred until a kernel needs
   it.
4. **`Fp64` and `Ext2`.** Contents:
   - `fp64.h`: `jolt::Fp64<C>` with the operations of `Fp128`, for 64-bit
     moduli with odd `C < 2^32`, and `fp64_detail::reduce_sum`, which
     reduces a sum of up to three 128-bit products once for `C < 2^31`;
   - `ext2.h`: `jolt::Ext2<F>` over either base, with multiplication by a
     base-field element (`mul_base`). Multiply and square are Karatsuba
     forms, overloaded over `Fp64<C>` with `C < 2^31` by forms that reduce
     each coefficient once;
   - `MetalField` for `Fp64<P>` with 64-bit `P` and for `Ext2<F>`, and the
     `bytemuck` impls behind them;
   - the conformance suites, the generic field benchmarks
     (`benches/field.rs`), and the A/B under Performance.
5. **`Fp32` and `FpExt4`.** `Fp32` for `Prime32Offset99`, and `FpExt4` in
   the `[1, e1, e2, e3]` cyclotomic basis matching
   `PseudoMersenne::ext4_mul`, each landed when a consumer first needs it.
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
  `Setup` error. A small library compiled in about 220 ms. The `Fp128`
  conformance library (10 kernels) compiles in 75 ms for one field and
  104 ms for two, the first compile in a process; later compiles in the same
  process take 26 and 43 ms.
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
