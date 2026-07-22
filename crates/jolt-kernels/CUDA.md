# CUDA kernels in jolt-kernels — implementation notes & handoff

This document describes the CUDA kernel layer as it exists in this crate: what kernels
exist, how they compose, how they're driven from Rust, and the lessons learned building
them. It is written for a session that will be **adding new CUDA kernels** (openings,
Dory MSM) to a redesigned jolt-kernels. It deliberately does **not** describe the stage
scheduling or cross-prover performance framing, since those are changing.

The kernels here were built to accelerate the sumcheck/witness math for a set of
"relations" (RAM read/write, register read/write, instruction lookups, bytecode,
booleanity, claim reductions). Every kernel targets the **BN254 scalar field Fr** in
Montgomery form. There is no base-field (Fq) or elliptic-curve arithmetic here yet — that
is exactly what an MSM/openings port would add, and none of the existing kernels help with
it directly (they are all Fr-only). Budget for a new Fq prelude + EC point ops + a
Pippenger/bucket kernel; the reusable pieces below are the *infrastructure* (context,
uploads, residency, the test harness, the build system), not the field ops.

---

## 1. Build & execution model (read this first)

**One translation unit, compiled at runtime via NVRTC.** All `.cu` files are concatenated
in a fixed order into a single `KERNEL_SRC` string (`const KERNEL_SRC: &str = concat!(include_str!(...), ...)`
near the top of `src/cuda.rs`) and compiled once with `compile_ptx_with_opts` when the
`CudaKernelContext` is first created. Consequences:

- **All global symbols share one namespace.** A kernel entry point named `foo` collides
  with any other `foo`, and with CUDA/libm intrinsics. We hit this concretely: a kernel
  named `fma` failed to compile (`more than one instance of overloaded function "fma" has
  "C" linkage`) because `fma` is a built-in. That kernel keeps the CUDA-symbol name
  `fma_kernel` even though its Rust field is `fma`. **When you add kernels, pick
  non-colliding entry-point names** (avoid `add`/`mul`/… only if they'd clash; we verified
  ours don't collide with device helpers, but new ones might).
- **Include order matters.** `.cu` files are not independent — later files call
  `__device__` helpers defined in earlier ones. `prelude.cu` is first and defines the
  field arithmetic everything uses. Some feature `.cu` files *also* export shared helpers
  consumed by later files (see §3). If you add a file that depends on a helper, place its
  `include_str!` after the definer in the `concat!` list.
- **NVRTC options:** compiled with `--device-int128` (the field code uses `unsigned
  __int128`). There are deprecation warnings for `ulonglong4` (harmless).
- **The compile is the real integration test.** `cargo build` only checks Rust; kernel
  name/signature errors surface only when NVRTC runs, which happens the first time any test
  or code creates the context. The cheapest way to trigger it is running any one kernel
  unit test (e.g. `edge_cases`).

**Single process-wide context.** `shared_ctx() -> Option<&'static CudaKernelContext>` is an
`OnceLock`; there is no per-invocation device object. All persistent device state
(resident buffers, the pinned-staging pool) hangs off this singleton. If a device isn't
present, `shared_ctx()` returns `None` and callers fall back to CPU.

**Backend gating.** GPU paths are selected at runtime by a `backend == "cuda"` check that
is threaded through as data, **not** by `#[cfg(feature = "cuda")]` alone. The feature flag
only controls whether the CUDA code is compiled in; whether it *runs* is a per-call
decision. This keeps a genuine CPU-only path for the equivalence tests to compare against
(gating on the feature alone would make the "cpu" backend also run on the GPU and compare
GPU-to-GPU). Preserve this distinction.

---

## 2. Field arithmetic prelude (`prelude.cu`)

Types: `u64 = unsigned long long`, `u128 = unsigned __int128`. An Fr element is **4×u64
limbs, little-endian, in Montgomery form**, stored contiguously (stride 4).

Core device functions (all operate on `u64[4]`):
- `mac/adc/sbb` — multiply-accumulate / add-carry / subtract-borrow primitives.
- `geq_modulus` / `sub_modulus` — conditional final reduction against `MODULUS`.
- `load4(p, r)` / `store4(p, r)` — vectorized `ulonglong4` load/store of one Fr.
- `fr_add`, `fr_sub`, `fr_mul` — field ops. `fr_mul` is CIOS Montgomery mul using the
  `INV` constant. `MODULUS` and `INV` are `__constant__`.
- `cubic_tuple_add` — adds four Fr's pairwise (used by degree-3 tuple reductions).

**Montgomery invariant that simplifies a lot of code:** `mont(0) == 0` (the Montgomery
representation of zero is all-zero limbs). This means conditional/absent terms vanish under
field addition without a branch — e.g. an "increment" `mont(write) - mont(read)`, an
`init_val = mont(initial)`, or a `None`-table sentinel all reduce to a real field zero.
Several kernels exploit this (see `readraf_combined`, the RAM derive kernels).

**Converting integers to Montgomery on device.** Small scalars arrive as raw integers and
must be mapped to Montgomery form. This is done by multiplying by R² (`fr_mul(raw_limbs,
R2)`). Two shared helpers exist, each with its own R² constant defined alongside:
- `raf_to_mont` (defined in `raf_q_scatter.cu`, constant `RAF_R2`) — used by the RAM/i128
  derive kernels and the raf scatter.
- `pc_u64_to_mont` (defined in `prefix_combine.cu`, constant `PC_R2`) — used by the
  prefix/suffix round kernels.

  (These two exist for historical reasons and are functionally the same R² conversion;
  if you consolidate, do it carefully because of the one-TU include ordering.)

---

## 3. Shared device helpers exported by feature files

A few `.cu` files define `__device__` helpers that *other* kernels reuse (relying on
include order). If you touch these, mind their consumers:

- `raf_q_scatter.cu` → `raf_to_mont`, `RAF_R2` (consumed by `ram_derive.cu`,
  `i128_to_mont.cu`, and the scatter/read kernels).
- `prefix_combine.cu` → `pc_u64_to_mont`, `PC_R2`, and `combine_eval` (the prefix×suffix
  MLE combiner used by `read_table_round.cu`).
- `suffix_mle.cu` → `suffix_mle_eval` (evaluates a lookup-table suffix MLE by variant code;
  consumed by `read_suffix_scatter.cu` and the read-table path).

---

## 4. Rust ↔ device abstractions (`src/cuda.rs`)

`CudaKernelContext` owns the CUDA stream, the compiled module (one `CudaFunction` field per
kernel), a **pinned-staging pool**, and the **resident caches** (§5). Each `pub fn`
wrapper: clones the `CudaFunction`, builds a `LaunchConfig`, binds args via
`launch_builder`, launches `unsafe`, and (for scalar results) does one `clone_dtoh` +
`synchronize`.

- **`DeviceFrVec`** — an owned device buffer of Fr (holds `buf: CudaSlice<u64>`, `len`,
  and a handle to the staging pool). `.to_host()` downloads; `.first()` reads element 0;
  `.len()`. This is the currency type passed between kernels to keep data device-resident.
- **`CudaSlice<T>`** — cudarc raw device buffer for non-Fr scalar columns (`u16`, `u32`,
  `u64`, `u8`, `i32`). Used for index/flag columns that feed kernels (lookup indices,
  table indices, interleave flags, schedules).
- **Uploads:** `upload(&[Fr]) -> DeviceFrVec` (goes through the pinned pool),
  `upload_many(&[&[Fr]])` (batches several uploads behind one staging fill — use this when
  uploading N factor vectors at once), and `upload_uNN_slice` for raw columns.
  `download_uNN` / `.to_host()` for readback. `concat_device(&[&DeviceFrVec])` concatenates
  device buffers without a host round-trip.
- **Pinned staging pool.** Pageable `clone_htod`/`clone_dtoh` transfer at only a few GB/s;
  cacheable-pinned staging is ~1000× faster. But `cuMemHostAlloc` is itself expensive and
  scales with size, so the pool **reuses** pinned buffers rather than allocating per call.
  Route all uploads through `upload*`; don't hand-roll `clone_htod` for large data.
- **`as_fr_slice::<F>(&[F]) -> Option<&[Fr]>`, `into_fr::<F>`, `fr_into::<F>`** — zero-copy
  reinterpret between the generic prover field `F` and the concrete `Fr` the kernels use.
  Wrappers are generic over `F` at the boundary, concrete `Fr` inside.

**Launch conventions used throughout** (worth matching for consistency):
- Block size constant `BLOCK = 256`; `LIMBS = 4`.
- Elementwise kernels: `grid = ceil(n / BLOCK)`, one thread per element, `if i >= n return`.
- Reductions: a per-block shared-memory tree reduction writes one partial per block, then a
  separate `*_reduce` kernel folds partials (see `round_poly_reduce`, `reduce.cu`). Shared
  memory is sized `block * tuple * sizeof(u64)`.
- Every `unsafe { launch }` has a `// SAFETY:` comment stating each thread's exclusive
  write region and what it reads (buffer lengths). Keep this — it's the only comment kind
  allowed in this code (see §8).

---

## 5. Device residency (the singleton caches)

Because there is no per-prove object, **data shared across phases lives as caches on the
context**, not as handles threaded through calls. Pattern: a field
`resident_X: Arc<Mutex<Option<...>>>` (or a `Vec` for content-keyed multi-entry caches),
populated once, read opportunistically with a length/content filter and a CPU fallback on
miss.

Existing residents: `resident_witness`, `resident_committed` (content-keyed),
`resident_stage3`, `resident_stage2_product`, `resident_ram_state`,
`resident_ram_addresses`. There are `set_shared_resident_*` / `clear_shared_resident_*`
free functions the caller uses to populate/evict them around a prove.

**Keying:** use a full hash over *every* limb, not a sparse sample. A sparse-sample key
silently reused a stale buffer that differed only at unsampled indices — a correctness bug.
The O(n) host fold is negligible next to the PCIe upload it guards.

**Why residency matters for correctness of your work too:** the general principle we
converged on is *on-device construction pays exactly when the consumer is already on
device* — then building a buffer on device deletes both the host build and the H2D upload
and adds no D2H (the consumer reads it in place). If you build something on device but its
consumer is on CPU, you pay build + D2H + CPU-consume and lose. Keep producer and consumer
on the same side.

---

## 6. Kernel catalog

Grouped by role. Names are the CUDA entry points; the Rust wrapper method usually matches.

### Elementwise / primitives
- `add`, `sub`, `mul`, `fma_kernel` (io = io*b + c), `add_scalar`, `mul_scalar`,
  `add_scalar_at` (add a scalar to one indexed element, 1 thread) — vectorized Fr ops.
  These are DRAM-bandwidth-bound at large n (see §7); `fma` halves traffic vs mul+add.
- `bind`, `bind_many` (bind many equal-length polys in one launch),
  `bind_high_to_low`, `batched_bind_high_to_low` — sumcheck variable binding
  (fold a poly by a challenge: `out[i] = lo + r*(hi-lo)`), the innermost per-round op.
- `eq_double`, `lt_double` — build `EqPolynomial::evals` / the `lt` doubling recurrence on
  device (each round doubles length: `right = left*r; left += r - right` etc.). Used to
  construct eq/lt factor vectors device-side instead of uploading them.
- `sum_reduce`, `product_reduce`, `round_poly_reduce` — tree reductions (the last folds the
  3-lane round-poly partials).
- `u64_to_mont`, `i128_to_mont` (+ their `_dev` device-input variants) — batch integer →
  Montgomery conversion.
- `scan_u32_block` + `scan_u32_add_offsets` — an exclusive prefix-sum over u32 (block scan
  + offset fixup). **This is a keystone primitive**: it enables variable-length-output
  kernels (compaction), which is what makes on-device sparse index construction possible.

### Gather / scatter
- `rd_wa_gather`, `ram_ra_gather`, `gather8_materialize` — gather factor values by
  per-cycle index (register/RAM address columns; the 8-way collapse-to-dense).
- `scatter_add_eq` — scatter-add eq weights into address bins.
- `raf_q_scatter` (+ `raf_q_scatter_reduce`), `read_suffix_scatter` — scatter lookup
  weights into prefix/suffix banks for the instruction read-RAF address phase.

### Round-poly kernels (compute one sumcheck round's message)
These come in **two output bases** — getting this wrong silently breaks transcript parity:
- **Evaluation form** (`{0, 2, 3, …}`, recombined host-side via `from_evals_and_hint`):
  `round_poly_pairs`, `eq_round_poly_pairs`, `uniskip_pairs`, and the read-table /
  prefix-suffix evals.
- **Monomial-coefficient form** (`[c0, c1, …]`): `dense_product_pairs` (plain product,
  incremental convolution) and the **Gruen split-eq** kernels `gruen_round_poly_pairs`
  (computes only bottom `q_constant = Σ w·Π lo` and top `q_top = Σ w·Π Δ`; host assembles
  the cubic/quadratic). The Gruen eq weight is *pair-summed* and branches on which of
  `e_in`/`e_out` is still >1 element.
- `cubic_pairs` + `cubic_tuple_reduce` — degree-3 tuple round poly.

### Relation-specific round/bind kernels
Each mirrors a specific relation's per-round math; most have a dense form plus a **sparse**
form used for the first few rounds (rounds 1–3) that then **collapses to dense**:
- Booleanity: `core_booleanity_gather`, `core_booleanity_sparse_pairs` /
  `core_booleanity_sparse_collapse8` / `core_booleanity_sparse_bind`,
  `core_booleanity_cycle_pairs`, `core_booleanity_address_pairs`.
- Hamming: `hamming_pairs`, `hamming_booleanity_pairs`.
- Registers (sparse): `sparse_register_round_pairs`, `sparse_register_bind`.
- Instruction RA-virtual (degree-4): `ra_virtual_d4_pairs`,
  `ra_virtual_d4_sparse_pairs` / `_bind` / `_collapse`.
- Instruction read-RAF cycle (degree-9): `instruction_raf_cycle_pairs`,
  `instruction_raf_cycle_sparse_pairs` / `_collapse`.
- Bytecode: `bytecode_cycle_sparse_pairs`.
- RAM read/write: `ram_rw_cycle_round_pairs` / `_bind`, `ram_rw_address_round_pairs` /
  `_bind`.

The **sparse→dense** shape is important and recurring: a `Sparse` enum variant holds a
`round` counter, small per-round tables, and the sparse `values`/`combined` buffers; its
`bind` advances rounds and at a fixed round (3) calls a `*_collapse` kernel to materialize a
dense state, replacing itself with a `Dense` variant. Final evaluations read from the dense
side (`chunk_first`). New sparse relations should follow this state-machine.

### On-device factor / index construction (deletes host build + upload)
- `row_dots_kernel` (CSR weighted matvec), `dense_outer_kernel` / `dense_outer_fused_kernel`
  — build outer-product/CSR structures on device.
- `register_merge_count` / `register_merge_scatter` and `schedule_round_count` /
  `schedule_round_emit` — the **on-device sparse-schedule builder**: a 3-way register merge
  and the per-round segmented merge-union + `scan_u32` compaction that replaces a host
  `build_schedules` (log_T sequential passes) entirely. This was one of the biggest wins:
  keeping the sparse index construction on device rather than building it host-side and
  uploading.
- `stage2_product_factors`, `ram_output_factors`, `instruction_lookup_q`,
  `raf_weight_phase_update` — build specific relation factors from resident columns.
- `readraf_chunk_values`, `readraf_combined` — build the instruction read-RAF cycle-phase
  inputs on device from resident columns (lookup indices, interleave flags): the `u16`
  chunk-index array (bit-extraction) and the per-cycle `combined` factor
  (`table_values[table_index] + (interleaved ? raf_i : raf_id)`, with `table_index ==
  u32::MAX` → field zero via `mont(0)`). Consumer (the cycle rounds) is already on device,
  so this deleted ~200 MB of uploads.

### Instruction read-RAF address phase (prefix/suffix)
- `read_table_round_pairs` — per-(table,row) work-item gathers 46 prefix evals + suffix
  values and calls `combine_eval` to form the 3-lane round tuple. Note: it derives
  `table = i / half`, `row = i % half` from the thread index (an earlier version uploaded
  explicit `item_table`/`item_row` index arrays *every round* — deriving them in-kernel
  removed that per-round build + 2 uploads).
- `prefix_suffix_round_pairs` and `prefix_suffix_round_pairs3` — the single- and
  3-fused-triple prefix×suffix round evaluators. The `…3` variant runs all three
  (left/right/identity) triples in one launch via `blockIdx.y ∈ {0,1,2}`, cutting per-round
  launch/sync count (only beneficial because each sub-launch was tiny — see §7 on when
  launch fusion does vs doesn't pay).

### Test-only / probe wrappers
Some wrappers (`prefix_combine_probe`, `suffix_mle_probe`, host-input variants like
`compute_row_dots` vs `compute_row_dots_dev`, `i128_to_mont` vs `_dev`, `sum_dev`/
`product_dev`) exist so the equivalence tests can drive a kernel directly. These are *not*
dead code — they are the correctness harness. The production path calls the device-input
(`_dev`) variant; the host-input variant is the readable test entry.

---

## 7. Performance lessons (kernel-level; still valid regardless of stage layout)

- **Most kernels are DRAM-bandwidth-bound, not compute-bound.** On the L4s (~300 GB/s), the
  map/bind kernels run at 80–89% of peak bandwidth. `add` is *slower* than `mul` despite
  less arithmetic — confirming memory-bound. **Corollary:** micro-optimizing the field
  arithmetic (Montgomery mul, removing a div/mod, uninit-alloc instead of alloc_zeros) buys
  ~nothing at large n; we measured several such tweaks as neutral. The only lever that moves
  bandwidth-bound kernels is **reducing traffic** (fewer bytes moved — algorithmic), e.g.
  `fma` folding mul+add, or not re-uploading a buffer.
- **Launch/sync fusion pays only when launches are small and numerous.** Fusing the 3
  prefix-suffix launches helped a little because each was tiny (one block, dominated by
  launch+sync overhead). Fusing large kernels wouldn't help — the work, not the launch,
  dominates. Measure which regime you're in before fusing.
- **Host-blocked ≠ GPU-idle.** "Host waiting in `cuStreamSynchronize`/`cuMemcpyDtoH`" is not
  the same as recoverable GPU stall. For a sequentially-dependent chain, the GPU critical
  path is unchanged whether the host blocks or not; removing mid-chain downloads frees the
  host but not the GPU. Don't assume host-wait time is reclaimable.
- **Pinned + pooled host memory is the biggest transfer lever, but pool it** (per-call
  `cuMemHostAlloc` is a net loss). Use cacheable pinned (flag 0), not write-combined.
- **On-device construction pays iff the consumer is on device** (§5). This is the single
  most important rule for deciding whether a new "build X on GPU" kernel is worth it.
- Profiling here used **nsys**, not ncu (ncu hits `ERR_NVGPUCTRPERM` on the shared box).

---

## 8. Working conventions (please keep)

- **Stub + equivalence test FIRST, then body.** For every new kernel mirroring CPU
  functionality: write the CUDA stub (args voided / `Err(Unsupported)` wrapper) **and** a
  proptest equivalence test against the real CPU reference, and get them reviewed *before*
  writing the kernel body. This catches interface/semantics mistakes on crypto code where a
  vacuously-passing test is dangerous. Applies even to trivial kernels.
- **Equivalence tests are the correctness gate.** They live in `mod tests` in `src/cuda.rs`
  (per-kernel, `<name>_matches_cpu`) and as stage-level parity tests in jolt-equivalence
  (`<relation>_cuda_backend_matches_cpu_backend`, which run a full prove on both backends
  and assert byte-identical transcripts). Both are slow (each forces an NVRTC compile +
  GPU work) but they are the real signal. Run at least one after any kernel rename/signature
  change — the Rust build won't catch NVRTC-level errors.
- **Comments:** the only comments in kernel/wrapper code are `// SAFETY:` blocks on `unsafe`
  launches (documenting exclusive write regions + read lengths) and genuine gotchas.
  No section banners, no restating-the-name doc comments.
- **CPU code stays untouched.** Adding a GPU path must not restructure the CPU
  implementation. Allowed: making items `pub(crate)`, and "backend piping" (threading a
  `backend` param). Anything else in CPU logic should be cleared first.
- **Naming:** kernel/wrapper base names match; device-input variants use the `_dev` suffix
  (not `_device`; the two prepositional names `from_device`/`reduce_to_device` are left as
  prose). Reductions use a `*_reduce` companion; sparse relations use
  `*_sparse_pairs`/`_bind`/`_collapse`.
- **Transfer profiling** is available behind `JOLT_CUDA_XFER_STATS=1` via the
  `xfer_stats` module (H2D/D2H byte + call + phase-ns counters; `snapshot()`/`reset()`).
  Per-relation timing behind `JOLT_KERNEL_TIMING_LOG` and the `JOLT_STAGE*_KERNEL_TIMINGS`
  env vars. The `examples/cuda_single_prove.rs` harness prints the breakdown.

---

## 9. For the MSM / openings port specifically

- **None of the existing kernels transfer to MSM.** They are all Fr-only. MSM needs a new
  **Fq base-field** prelude (BN254 base field, different modulus/R²/INV), **EC point
  arithmetic** (G1 add/double in projective/Jacobian, and G2 for tier-2 pairing inputs),
  and a **bucket/Pippenger** kernel. Reuse the *infrastructure* (context singleton, pinned
  upload pool, `DeviceFrVec`/`CudaSlice` currency, residency caches, the one-TU build, the
  stub-test-first harness), not the math.
- The scalars are `ark_bn254::Fr` (same field these kernels use — you can reuse the Fr
  prelude for scalar handling); the points are `ark_bn254` G1/G2. Watch the arkworks fork
  (`a16z/arkworks-algebra`, branch `dev/twist-shout`) for the exact point representations.
- The MSM inputs are many independent row-MSMs (commit) plus a few folding-round MSMs
  (open). That's naturally parallel; residency of the fixed generator bases across all row
  MSMs is the obvious analogue of the resident-buffer pattern here.
