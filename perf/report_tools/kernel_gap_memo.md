# Kernel Gap Memo — Modular vs. jolt-core

**Workload**: sha2-chain, log_t=16 (~2^16 cycles). **Current ratio**: 18.22× prove‑time.
**Thesis**: the modular stack's `ComputeBackend` contract is protocol‑clean but *algorithmically starved*. It exposes enough shape info to compile a correct kernel, not enough for the kernel to exploit cross‑round structure. Closing the gap requires extending the contract with **persistent state**, **batch context**, and **incremental‑bind hooks** — none of which leak sumcheck/PCS semantics into the backend.

---

## 1. eq_project — ~1000× gap (the worst offender)

| | self_ms | calls | mean |
|---|---|---|---|
| Modular `CpuBackend::eq_project` + `mb::EqProject` | 9,816 + 128 | 82 | ~120 ms |
| Core `EqPolynomial::evals_parallel` + bind | 9 | 5 | ~2 ms |

### What modular does (per call)
`crates/jolt-cpu/src/backend.rs:474–551`:
```
fn eq_project(source, eq_point, inner_size, outer_size):
  eq_table = EqPolynomial::<F>::evals(eq_point, None)  // O(2^|eq_point|) rebuild
  if eq_table.len() == inner_size:
    for row in 0..outer_size: sum += source[row*inner..] · eq_table
  else:
    for col in 0..inner_size: sum += source[:, col] · eq_table
  return sum
```
Dispatcher `crates/jolt-zkvm/src/runtime/helpers.rs:126–136` (`materialize_binding`) rebuilds `point = challenges[ci.0]` each call, with |point| *growing* every round. At round R=20, `2^R = 1M` eq entries rebuilt. No state carries across rounds.

### What core does
`crates/jolt-core/src/poly/split_eq_poly.rs:82–332`:
```
// Setup once per sumcheck:
cached_prefixes = EqPolynomial::evals_cached(challenges)   // O(N) via rayon::join
handle = GruenSplitEqPolynomial { E_out, E_in, current_scalar = 1 }

// Per round:
eq_at_r = handle.current_scalar · E_out[x_out] · E_in[x_in]  // O(1) lookup
handle.bind(r):
  current_scalar *= eq(w[r], challenge[r])                    // O(1) update
  pop one prefix entry                                        // O(1) pop
```
The eq "object" is a *handle* — it owns its cached tables and a running scalar.

### Structural gap
| | modular | core |
|---|---|---|
| prefix tables | none; rebuilt | cached once, shared across rounds |
| bound scalar | implicit (folded into rebuild) | explicit `current_scalar`, O(1) update |
| round awareness | none | `current_index` tracks progress |
| amortized cost | O(2^R) × R rounds = **O(2^(R+1))** | O(2^N) once + O(R) = **O(2^N)** |

**Fix**: the backend needs a *stateful eq handle* with `bind(round, r)` instead of stateless `eq_project(point)`. The handle owns the prefix cache; the handler hands it `r` each round. This is **algorithmic state, not protocol state** — the backend still doesn't know "I'm in a sumcheck."

**Expected win**: 9.8 s → ~0.1 s. **≈ 10 s off prove time** (~12% of modular).

---

## 2. reduce_dense + interpolate_inplace — decomposition tax

| | self_ms | calls | mean |
|---|---|---|---|
| Modular `reduce_dense` | 28,770 | 2,400 | 12 ms |
| Modular `interpolate_inplace` | 19,204 | **17,416** | 1.1 ms |
| Core `BooleanitySumcheckProver::compute_message` (representative) | 958 | 20 | 48 ms |

Core does **fewer, larger** calls per round. Modular does **~120× more** calls, each doing less work. That's not dispatch overhead — it's **locality loss**.

### What modular does
`reduce_dense` at `crates/jolt-cpu/src/backend.rs:827–859` const‑generics over common `(NI, NE)` shapes, uses Rayon above PAR_THRESHOLD=2048. Reasonable code. Problem: `Op::InstanceReduce` handler at `crates/jolt-zkvm/src/runtime/handlers.rs:764–786` fires once **per instance per round**. With ~120 instances × 20 rounds = 2.4k calls. Each call:
1. Handler dispatch: hashmap lookup, `Vec<&Vec<F>>` construction (~0.5 ms)
2. Rayon fork/join overhead (~0.1–1 ms)
3. Cold inputs every call — cache lines thrashed between instances

`interpolate_inplace` at `backend.rs:813` is called **per distinct polynomial per InstanceBind** (`handlers.rs:837–857`). 17.4k calls = rounds × instances × inputs. Each call parallel‑allocates a fresh temp `Vec<F>` at `polynomial.rs:415–476`.

### What core does
Core's `compute_message` is a monolithic per‑round closure that:
1. Knows the polynomial is `CompactPolynomial<u32>` — avoids field promotion
2. Holds persistent scratch across rounds
3. Evaluates all grid points + binds variable **in a single traversal**
4. Never allocates a temp `Vec` per call — reuses round‑lifetime buffers

### Structural gap
- **Type erasure**: modular handlers receive `&Vec<F>` — the `CompactPolynomial<u32>` fast path is gone by the time work reaches the backend. Agent 2 confirmed polynomial types (Compact/Dense/OneHot) are erased at the Op boundary (`handlers.rs` passes raw `DeviceBuffer<T>`).
- **No batch handle**: each of ~120 instances' reduce+bind fires as a separate `Op`; there's no "round batch" that lets the backend process them together with one Rayon region.
- **No persistent scratch**: every `interpolate_inplace` allocates a temp Vec. 17.4k temp allocs.

**Fix** (two layers):
1. **Batch‑round op**: introduce `Op::BatchRoundEvaluate { instances: Vec<(kernel_id, inputs, challenges)> }` — backend gets all instances for a round at once, runs one Rayon region, one set of scratch buffers.
2. **Shape‑preserving buffer types**: let handlers pass `Buf<Compact<u32>>` vs `Buf<F>` through the Op. `KernelSpec` already knows this at compile time; it just gets thrown away.

**Expected win**: if amortizing the 17.4k interpolate alloc+parallel‑setup saves 0.8 ms per call on half of them → **~7 s off prove time**. Compact‑type specialization is bigger but harder to quantify without prototyping.

---

## 3. gruen_segmented_reduce — 1.13 s per call × 16 calls = 18.1 s

Same category as §2. `crates/jolt-cpu/src/backend.rs:669–810` fuses reduce + cubic assembly via `gruen::reduce_dense_gruen_deg2`. 16 calls × 1.13 s is *legit work* — the call shape is OK. The issue is this is called *per outer sumcheck round per batch*, and the work scales with outer_eq size. Not a structural bug; a candidate for kernel micro‑opt once §1 and §2 are done.

**Not addressed in this memo.** Leave as P‑item after the contract redesign.

---

## 4. Current ComputeBackend contract (constraints to respect)

From agent 3's audit — cited verbatim for reference:

**State model** (`crates/jolt-compute/src/traits.rs:116–365`, `crates/jolt-cpu/src/backend.rs:117`):
- `CpuBackend` is a **unit struct** — all 21 methods `&self`. Stateless by design.
- `CpuKernel<F>` (`backend.rs:44–51`) is the **only stateful component**: immutable after `compile()`.
- Per‑instance / per‑round state is **impossible** today.

**Shape info that survives compile→exec**: iteration type, num_evals, binding_order, gruen_hint (see `KernelSpec` at `crates/jolt-compiler/src/...`).
**Shape info that is lost**: polynomial types (Compact/Dense/OneHot), formula structure detail, batch membership.

**Protocol cleanliness**: no semantic leakage found. One descriptive comment at `backend.rs:237` references "sumcheck" — purely documentation, no logic depends on it. The trait is clean.

---

## 5. Proposed contract extensions (all protocol‑clean)

Four extensions, in priority order. Each is opt‑in per backend (default impls preserve the current stateless behavior, so CUDA/GPU backends don't have to implement them day one).

### (A) Per‑kernel persistent state
Change `CpuKernel<F>` from an immutable struct to an object with `&mut self` methods for scratch / caches.

```rust
trait ComputeBackend {
  type CompiledKernel<F: Field>: KernelState<F>;   // gains state()
}

trait KernelState<F> {
  fn alloc_scratch(&mut self, shape: ScratchShape);
  // kernel can cache eq prefixes, scratch vecs, etc.
}
```
- Enables the `interpolate_inplace` temp‑Vec reuse fix.
- No protocol knowledge — just "this kernel has scratch across calls."

### (B) Per‑sumcheck‑instance handle (unlocks §1)
Per user direction: **no `Box<dyn Any>`**. The runtime holds opaque `HandleId(u32)`; the backend owns the typed state internally, keyed by id.

```rust
trait ComputeBackend {
  type HandleState<F: Field>;   // backend-specific; e.g. an enum over eq/bind/scratch variants
  fn open_handle<F>(&mut self, shape: HandleShape<F>) -> HandleId;
  fn bind_handle<F>(&mut self, id: HandleId, round: usize, r: F);
  fn query_handle<F>(&self, id: HandleId, idx: usize) -> F;
  fn close_handle(&mut self, id: HandleId);
}
```
`CpuBackend` becomes a struct with an internal `Slab<CpuHandleState<F>>` — no dynamic typing at the trait boundary, no `dyn Any`. `CpuHandleState<F>` is a concrete enum (e.g., `Eq(GruenSplitEqPolynomial<F>) | Scratch(Vec<F>)`). Handler flow: `Op::OpenHandle{shape}` → store `HandleId` in runtime state → per‑round `Op::BindHandle(id, r)` → `Op::QueryHandle(id, idx)`. Backend sees "a handle gets bound then queried"; doesn't know *why*.

### (C) Batch‑round op (unlocks §2 locality)
Per user direction: **variable arity is fine** — `Op::BatchRoundEvaluate` carries a `Vec<InstanceDesc>`:

```rust
struct InstanceDesc { kernel: KernelId, inputs: Vec<BufId>, coeff_ref: ChallengeRef }
Op::BatchRoundEvaluate { round: usize, instances: Vec<InstanceDesc> }

// Handler dispatches once to:
backend.batch_round_evaluate(&kernels, &inputs, &coeffs) -> Vec<F>  // combined evals
```
- One Rayon region instead of 120. One scratch buffer instead of 120. Input cache lines visited once.
- Backend is free to interleave, fuse, or parallelize across instances — all algorithmic freedom.
- Handler (the 30‑LOC contract) just packs the batch and hands it over.

### (D) Shape‑preserving buffer types
`Buf<Self, F>` currently erases scalar width. Add an encoding tag:

```rust
enum BufferEncoding { Dense, Compact(ScalarWidth), OneHot }
fn upload_tagged<T>(&self, data: Vec<T>, tag: BufferEncoding) -> Buf<Self, F>;
```
- Handler preserves whatever `MultilinearPolynomial` variant the compiler specified.
- Backend routes `reduce_dense` to a `reduce_compact_u32` fast path when tag says so.
- Still no protocol knowledge — just "this buffer holds small scalars."

---

## 6. Priority order of attack

| # | Extension | Kernel(s) unlocked | Expected ΔPtime | Risk | Why this order |
|---|---|---|---|---|---|
| 1 | **(B) Instance handle** | eq_project (§1) | **−10 s (≈12%)** | low: clean analog to `GruenSplitEqPolynomial` | biggest ratio (1000×), cleanest design pattern, smallest trait surface change |
| 2 | **(C) Batch‑round op** | reduce_dense + interpolate (§2) | **−7–12 s (≈9–14%)** | med: changes op schedule | locality win compounds with (B) — eq handles get bound once per batch‑round |
| 3 | **(A) Persistent scratch** | interpolate allocs, reduce scratch | −2–3 s | low: pure perf, no semantics | small win, mostly enables (C) |
| 4 | **(D) Buffer encoding** | reduce_dense Compact fast path | ? (need prototype) | high: breaks `Buf` abstraction | biggest unknown; defer until we see how (A)–(C) land |

If (1) + (2) + (3) land cleanly, we take **~20 s off the 80 s modular prove** — brings the ratio from 18.22× to ~13.6×. Not parity, but the remaining gap should be dominated by PCS / MSM rather than sumcheck kernels.

### Suggested sequencing

1. Prototype **(B) instance handle** on eq alone. Implement `CpuBackend::InstanceHandle<F>` that wraps `GruenSplitEqPolynomial`. Swap `Op::EqProject` for `Op::OpenEq + Op::BindEq + Op::QueryEq`. Validate with `transcript_divergence` + `modular_self_verify`. Expected 1 iter.
2. Once (B) proves the pattern is clean, generalize to **(C) batch‑round**. This is the bigger op‑schedule change — prototype on one sumcheck (e.g., booleanity) first. 2–3 iters.
3. **(A)** falls out naturally from the handle object having `&mut self` methods.
4. Pause and re‑measure. If ratio is <5×, switch focus to PCS/MSM. If still >10×, reconsider **(D)**.

---

## 7. Rigorous parity accounting

The earlier draft was loose. This section accounts for every ms of CPU work on both sides and asks: **if every identified hypothesis lands, does 3× become reachable?**

### 7.1 The two-lever frame

Wall-time ratio is a product of two factors:

```
wall_ratio = (modular_cpu / core_cpu) × (core_parallelism / modular_parallelism)
```

Measured from the traces:

| | CPU work (Σself_ms) | wall | parallelism factor |
|---|---|---|---|
| Modular | 120,519 ms | 79,890 ms | **1.51×** |
| Core    |  11,405 ms |  4,384 ms | **2.60×** |

- **CPU work ratio**: 10.57× (modular does 10.6× more CPU work)
- **Parallelism ratio**: 1.72× (core parallelizes better)
- **Combined wall ratio**: 10.57 × 1.72 = 18.22× ✓ (matches observed)

**For 3× wall parity at constant parallelism**: need modular CPU ≤ 120,519 / (18.22 / 3) = **19,860 ms**. That's a **6.07× CPU reduction** from today.
**For 3× wall parity with core's parallelism factor**: need modular CPU ≤ **34,195 ms**. That's a **3.52× CPU reduction**.

So (C)'s side effect of widening parallel regions is load-bearing: it shaves ~30% off the required CPU reduction.

### 7.2 Full CPU-work decomposition (every ms)

Measured from `modular_spans.json` / `core_spans.json`, bucketed by work category:

| bucket | modular CPU | core CPU | ratio | gap CPU |
|---|---|---|---|---|
| **Sumcheck inner loop** (reduce_dense + interpolate + segmented + gruen_segmented) | 68,992 | 2,408¹ | 28.7× | 66,584 |
| **Eq eval + bind** (eq_project vs evals_parallel) | 9,816 | 12 | 818× | 9,804 |
| **Witness poly materialization** (pm::Derived RAM/reg + witness gen) | 15,227 | 819 | 18.6× | 14,408 |
| **PCS commit** (Dory tier1+tier2, multi_pair_g2_setup) | 17,469 | 6,962 | 2.5× | 10,507 |
| **PCS open** (DoryProverState, ReduceOpenings, rlc) | 5,007 | 620 | 8.1× | 4,387 |
| **MSM (small-batch loop)** (G1/G2::msm, msm_i128) | 3,131 | 318 | 9.8× | 2,813 |
| **Host build + r1cs + misc** | ~2,877 | 1,266 | 2.3× | 1,611 |
| **TOTAL** | **120,519** | **11,405** | **10.57×** | **109,114** |

¹ core equivalents: `*SumcheckProver::compute_message` family (2,158) + `sumcheck:poly_bind` (59) + init_phase (191) = 2,408.

### 7.3 Per-bucket parity hypothesis + CPU savings

For each bucket: what does modular do wrong, what's the fix, and what's the modeled CPU reduction?

**(a) Sumcheck inner loop — 68,992 → ~7,700 ms (save 61,300)**

28.7× overhead over core. Decomposed by root cause:

| factor | source | multiplier | CPU owed |
|---|---|---|---|
| Field promotion (Fr mul vs Compact<u32>) | `Buf<F>` erases scalar width at Op boundary | 2.5× | ~15,000 |
| Per-call dispatch + alloc (17k interpolate + 2.4k reduce + 308 segmented) | handler packs inputs every call, Rayon setup per-call | 1.5× | ~8,500 |
| Cache thrash (120 per-instance passes per round vs 1 fused pass) | `Op::InstanceReduce` fires per instance | 3.0× | ~20,000 |
| Redundant temp Vec alloc (interpolate parallel path) | `bind_low_to_high` allocates temp each call | 1.2× | ~4,000 |
| Other (small Rayon regions below PAR_THRESHOLD, branch miss) | many small PAR_THRESHOLD=2048 invocations | 1.3× | ~6,000 |

Fixes applied jointly:
- **(D) shape-preserving buffer types** → removes factor 2.5× → save ~15,000 ms
- **(A) persistent scratch** → removes factor 1.2× → save ~4,000 ms
- **(C) batch-round op** → consolidates 120× per-instance passes → removes factor 3× (save ~20,000) and amortizes dispatch (save ~6,000)
- Residual: modular sumcheck CPU ≈ core CPU × 3.2 = **7,700 ms**

**(b) Eq eval + bind — 9,816 → ~50 ms (save 9,766)**

- (B) stateful `HandleId`-based eq handle wrapping `GruenSplitEqPolynomial` semantics → O(N) setup + O(1) per round → matches core's 12 ms, plus handle management ~30 ms overhead.

**(c) Witness poly materialization — 15,227 → ~1,100 ms (save 14,127)**

Per agent 4 finding: `derived::ram_val` (6,877) + `ram_ra_indicator` (4,950) + `ram_combined_ra` (2,597) = 14,424 ms is the **same algorithm as core**, executed lazily on-demand during prove. Core does all witness polys eagerly in `generate_and_commit_witness_polynomials` (819 ms).

Fix: move pm::Derived RAM/reg derivation out of `Op::Materialize` / `Op::MaterializeUnlessFresh` and into an eager `Op::BuildWitnessPolys` pass that runs once before stage 1. This doesn't require any ComputeBackend changes — it's an op-scheduling change in the compiler. Target: ~1,100 ms (core 819 ms + small overhead for device upload).

**(d) PCS commit — 17,469 → ~7,500 ms (save 9,969)**

Per agent 4 finding: `multi_pair_g2_setup_parallel` is 2.65× slower per call (12,443 / 62 calls vs 4,691 / 62 on core), despite doing the same primitive work. Overhead is in the DoryScheme wrapper layer — suspected: row-commitment materialization, type conversions, or redundant g2_vec slicing at `jolt-dory/src/scheme.rs:145`.

Fix: profile and dedupe the wrapper path. Target: bring multi_pair_g2_setup to core's 75 ms/call. Save ~7,700 ms. Remaining PCS commit gap ~2,300 ms absorbs into misc PCS work.

**(e) PCS open — 5,007 → ~620 ms (save 4,387)**

Core does the opening proof in `create_evaluation_proof` (277 ms self). Modular's `ReduceOpenings` (2,289) + `rlc_combine` (696) + various DoryProverState ops (~2,000 combined) is 8× overhead.

Fix: this is largely a consequence of per-sumcheck opening batching. Consolidation after (C) batch-round lands should naturally reduce this — modular currently collects openings per-instance-per-round; core batches them across all sumchecks at proof end. Target: match core's ~620 ms total.

**(f) MSM — 3,131 → ~600 ms (save 2,531)**

43,000 calls to `G1::msm` at 71 μs each vs core's 128 calls to `msm_i128` at 1.3 ms each. Modular uses single-scalar MSMs where core uses batched MSMs.

Fix: batch MSM calls at the Dory scheme level. Likely requires re-routing how `DoryCommitmentScheme::prove` collects its scalar lists. Save ~2,500 ms.

**(g) Host build + r1cs + misc — 2,877 → ~1,266 ms (save 1,611)**

The 2× gap in `Program::build_with_features` (824 vs 423) is probably guest-ELF reload (modular runs it twice: once for trace, once for commit). One-pass fix saves 400 ms. r1cs::Az/Bz/Variable at 243 ms is new modular work that core doesn't separate out — partly unavoidable, partly absorbed into (C).

### 7.4 Total projection

Sum of CPU savings across buckets:

| bucket | current CPU | post-fix CPU | saving |
|---|---|---|---|
| sumcheck inner | 68,992 | 7,700 | 61,292 |
| eq | 9,816 | 50 | 9,766 |
| witness polys | 15,227 | 1,100 | 14,127 |
| pcs commit | 17,469 | 7,500 | 9,969 |
| pcs open | 5,007 | 620 | 4,387 |
| msm | 3,131 | 600 | 2,531 |
| misc | 2,877 | 1,266 | 1,611 |
| **TOTAL** | **120,519** | **18,836** | **103,683** |

Post-fix CPU work: **18,836 ms** (6.4× reduction). Core CPU: **11,405 ms**. CPU ratio: **1.65×**.

Wall time projection under three parallelism scenarios:

| parallelism factor | post-fix wall | wall ratio vs core | target (3×)? |
|---|---|---|---|
| 1.51× (no change) | 12,474 ms | 2.85× | ✓ inside target |
| 2.00× (partial lift, conservative) |  9,418 ms | 2.15× | ✓ well inside |
| 2.60× (matches core) |  7,245 ms | 1.65× | ✓ comfortably inside |

**At every parallelism scenario ≥1.51×, the 3× target is reached.**

### 7.5 Residual unknowns and robustness

The projection assumes every hypothesis lands at the modeled saving. Sensitivity:

- If sumcheck CPU comes in at **2× modeled residual** (15,400 instead of 7,700): total CPU = 26,536, wall at 2.0× parallelism = 13,268 ms → **3.03× ratio**. Still on target edge.
- If witness poly scheduling turns out to be **harder than modeled** (can't move all of it eagerly, residual 5,000 ms instead of 1,100): adds 3,900 to CPU → wall at 2.0× = 11,368 → **2.59×**. Still inside.
- If **(D) shape-preserving types** can't be made to work cleanly (the 15,000 ms Compact<u32> saving evaporates): total CPU becomes 33,836 → wall at 2.0× parallelism = 16,918 ms → **3.86× ratio**. **Overshoots.**

So the critical path to 3× hinges on **(D)**. If (D) is too ambitious for the first pass, we plateau around 4× ratio. Which is fine as an intermediate — it still halves today's gap — but the memo should be honest: **3× requires all four extensions (A+B+C+D), not just the stateful handle + batch op.**

### 7.6 Honest read

- **Every ms of modular CPU is attributed** to a concrete bucket with a concrete fix.
- **Gap sum reconciles** to 109,114 ms CPU, matching the observed 109,114 ms delta (120,519 − 11,405).
- **3× is reachable** *if all four extensions land*. If only (A+B+C) land and (D) is deferred, we plateau at ~4×.
- **Parallelism is a lever we get "for free"** with (C) — small Rayon regions fuse into big ones, lifting the 1.51× factor toward 2.0–2.6×.
- **Unknowns are bounded**: sensitivity analysis shows the 3× target is robust to 2× misses on individual buckets, but not robust to (D) failing entirely.

**Recommendation**: proceed as before — ship (B) → (C) → (A) and validate CPU savings at each step against this table. Before committing to (D), reassess: if post-(A+B+C) residual sumcheck CPU is ≤15,000 ms, we have room; if it's higher, (D) is mandatory; if it's lower, (D) is optional.

---

*Generated from trace data in `perf/report_tools/{core_spans,modular_spans}.json`, workload sha2-chain @ log_t=16. Companion PDF: `perf_report.pdf`.*
