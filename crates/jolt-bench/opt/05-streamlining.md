# Ticket 5 — Streamline the Op pipeline (rails before perf)

**Status:** proposed. Rails-first architectural overhaul. No perf iterations
inside this ticket; perf loop resumes after O7 lands.
**Est. perf impact:** zero during the overhaul (O1–O7 are correctness-only).
The rails unlock every future perf ticket — after O8 graduates, backend-specific
fusion rules are tractable instead of accreted hacks.
**Est. effort:** ~4–5 weeks of focused work; ~15–25 commits spanning 8 phases.
**Philosophy:** make the canonical primitive-emission layer + pass pipeline
durable *before* any more perf golf. Without it, every fusion rule compounds
IR-fragmentation tech debt (see Ticket 1 post-mortem).

## Why this exists

Tickets 0–4 operated on the assumption that the compiler emits unrolled
primitives and backends add their own fusion rules. In practice:

- The `Op` enum has **9+ protocol-specific variants** (`ReadCheckingReduce`,
  `RafReduce`, `SuffixScatter`, `QBufferScatter`, `MaterializeRA`,
  `MaterializeCombinedVal`, `MaterializePBuffers`, `InitInstanceWeights`,
  `UpdateInstanceWeights`, `MaterializeSegmentedOuterEq`) that every handler
  arm has to know about. The runtime is not protocol-unaware; it just
  pretends to be.
- `ComputeBackend::fuse_ops` (ticket 0) was an approximation: fusion on the
  backend trait. Per current direction, fusion belongs in the compiler as
  per-target passes, not on the backend trait.
- There are **three emission sites** (`compiler/emit.rs`, `builder.rs`,
  `examples/jolt_core_module.rs`) with overlapping logic; only the builder
  is the canonical API.
- `BufferProvider` + `LookupTraceData` live in `jolt-compute` even though
  they're data-pipeline (runtime) concepts; the "backend-agnostic compute
  device abstraction" crate documents itself incorrectly.
- There is **no pass pipeline**. `link()` clones ops as-is. There is no way
  to express "CPU target does these 5 rewrites, Metal does those 3."

The plan below fixes these in eight phases that ship the right rails, then
hands control to a per-target pass pipeline where future perf tickets live.

## Target architecture

```
┌─────────────────────── jolt-compiler ────────────────────────┐
│ Protocol IR → Module (maximally-unrolled primitive ops)      │
│              → Pass pipeline (per-target, in jolt-compiler)  │
│              → Module (optimized for that target)            │
│                                                              │
│ Owns: Op, Module, KernelSpec, Formula, PolynomialId          │
│ Owns: all passes (MergeBatchRoundReduces, BindCoalesce, …)   │
│ Owns: eval_scalar_expr (ScalarExpr IR + interpreter)         │
│ Emits: canonical primitive stream, always                    │
└──────────────────────────────────────────────────────────────┘
                          │   Module
                          ▼
┌────────────────────── jolt-compute ──────────────────────────┐
│ "Platform to execute ops." Only this.                        │
│                                                              │
│ Owns: ComputeBackend trait, DeviceBuffer, Buf, ReduceInputs  │
│       HandleId / HandleShape                                 │
│ Does NOT own: BufferProvider, LookupTraceData, Executable,   │
│                link(), anything protocol-aware               │
└──────────────────────────────────────────────────────────────┘
                          │   backend trait
                          ▼
┌─────────────────────── jolt-zkvm ────────────────────────────┐
│ Dumb walker + data pipeline + PCS orchestration.             │
│                                                              │
│ Owns: runtime/ (for op in ops { dispatch }), BufferProvider, │
│       PCS orchestration (commit/open/opening proofs),        │
│       prove() / preprocess() / proving_key                   │
│ Handler arm count: ~20 primitive ops, zero protocol-specific │
│ Grep-test: no arm name matches Lookup|Ram|Instruction|RA|RAF │
└──────────────────────────────────────────────────────────────┘
```

### Three invariants the rails enforce

1. **Compiler emits primitives only.** An `Op::is_primitive()` classifier
   returns `true` iff the variant is the canonical primitive form. A
   compiler-side `debug_assert!` runs post-emit (before any pass runs) and
   fails if a non-primitive op made it out of the emission layer.
2. **Runtime is protocol-unaware.** Every `handlers.rs` arm is ≤ 30 LOC
   and its name describes compute, not protocol. Grep-test for names
   containing `Lookup|Ram|Instruction|RA|RAF|Booleanity` returns zero hits.
3. **Passes are pure rewrites.** `fn(Module) -> Module`, deterministic,
   idempotent on their own output, testable in isolation, never required
   for correctness. The empty pipeline (`debug_pipeline()`) must produce
   a valid proof — it is the permanent reference for transcript equivalence.

## Phase overview

Eight phases. Each ends with a durable piece of infrastructure the next
phase relies on. Phases are individually shippable and reversible. No perf
gate until O8 — correctness only throughout O1–O7.

| # | Name | Exit |
|---|---|---|
| O1 | Crate boundary cleanup | `BufferProvider`, `LookupTraceData`, `eval_scalar_expr` move out of `jolt-compute`; crate doc tightened |
| O2 | Emission consolidation | `grep 'ops.push' crates/jolt-compiler/` returns only `ModuleBuilder`-internal hits |
| O3 | OPS.md inventory + `is_primitive()` classifier | Inventory doc exists; classifier + post-emit assertion compile and run |
| O4 | Collapse redundant variants (bind + materialize) | 4 bind ops → 1; 3 materialize ops → 1; runtime has no `bound_this_round` set |
| O5 | Lower protocol-specific compute ops | 9+ protocol ops gone; runtime arms match generic names only |
| O6 | Pass infrastructure | `CompilerPass` trait + per-target pipelines + CI cross-verification under `JOLT_COMPILE_TARGET=debug\|cpu` |
| O7 | First rewrite rules (proof of concept) | `cpu_pipeline()` has ≥3 rules, all tested individually + transcript-match `debug_pipeline` on muldiv |
| O8 | Backend-specific pipelines + perf iteration | Perf loop resumed; each new rule follows the pass-infra + transcript-match protocol |

Full per-phase detail below.

---

## O1 — Crate boundary cleanup — **LANDED** (`445d583c5`)

**Why now**: pre-req for every other phase. Tiny commit, unblocks correctly-scoped
imports for O2+.

**What landed**:
- `pub trait BufferProvider` + `pub struct LookupTraceData` moved from
  `crates/jolt-compute/src/traits.rs` to a new file
  `crates/jolt-compiler/src/buffer_provider.rs`, re-exported at the
  `jolt_compiler` crate root.
- `fn eval_scalar_expr` + helpers moved from `crates/jolt-zkvm/src/scalar_expr.rs`
  to `crates/jolt-compiler/src/scalar_expr.rs`, alongside the IR types
  (`ScalarExpr`, `Monomial`, `ValueSource`, `DefaultVal`) that were
  already in `jolt-compiler::module`. `runtime/scalar_expr.rs` deleted.
- `jolt-compute/src/lib.rs` crate-level docstring tightened to explicitly
  state **"ComputeBackend trait + buffer / reduce-input types. Only
  that."** — lists what lives elsewhere (BufferProvider in jolt-compiler,
  scalar_expr in jolt-compiler, runtime walker in jolt-zkvm).
- `jolt-field` added as a production dep of `jolt-compiler` to support
  the `F: Field` bound on `BufferProvider`. No other jolt-* production
  deps added.
- `Executable` + `link()` stayed in `jolt-compute/src/linker.rs` — they
  bridge compiler output to backend-compiled kernels, which is a
  compute-adjacent concern. Optional move deferred.

**Deviation from original plan**: BufferProvider targeted jolt-zkvm in
the original plan, but `jolt-witness` implements the trait and can't
depend on `jolt-zkvm` without a crate cycle (jolt-zkvm already depends
on jolt-witness). The trait must live upstream of any implementor, so
jolt-compiler — the only common ancestor of both `jolt-witness`
(implementor) and `jolt-zkvm` (consumer) — is the right home. The
semantic framing is arguably cleaner too: the compiler defines the IR
contract, including the witness-shape the runtime will need to resolve
polynomial references.

**Importers updated**: jolt-witness (provider.rs, polynomials.rs),
jolt-zkvm (prove.rs, runtime/{mod, handlers, helpers, prefix_suffix}.rs),
jolt-bench/stacks/modular.rs, jolt-equivalence/tests/checkpoint_eval_parity.rs.

**Correctness gate**: jolt-equivalence 50/50 PASS. Clippy clean on
jolt-core (host + host,zk), jolt-cpu, jolt-compute, jolt-compiler,
jolt-zkvm under `-D warnings`.

**Exit achieved**: `grep -r BufferProvider crates/jolt-compute/` returns
zero hits. `jolt-compute::traits` exports only backend abstractions.

**Actual scope**: 1 commit, ~700 LOC of moves + import fixes + plan doc.

---

## O2 — Emission consolidation

**Why now**: every downstream phase wants a single emission surface. Without
it, O4/O5 changes have to be duplicated across three files.

**What changes**:
- Extend `crates/jolt-compiler/src/builder.rs` `ModuleBuilder` with primitive
  emitters for every `Op` variant used by `emit.rs` and `jolt_core_module.rs`:
  - `bind_kernel_inputs(kernel, challenge)` → emits `Op::Bind` with deduped
    kernel inputs (already exists as free helper; move into `ModuleBuilder`)
  - `reduce_flat(kernel, destination)`
  - `reduce_product(kernel, seg, round_within_phase, destination)`
  - `materialize(binding)`, `materialize_if_absent(binding)`, etc.
  - Every compute-op primitive currently emitted directly to `ctx.ops`.
- Replace every `ctx.ops.push(Op::X { … })` in
  `crates/jolt-compiler/src/compiler/emit.rs` with a `builder.emit_x(…)`
  call. Same for `crates/jolt-compiler/examples/jolt_core_module.rs`.
- Delete now-unused emission scaffolding in `emit.rs` (direct access to
  `ctx.ops`).

**Correctness rail**: replace each push with a builder call, one site at a
time. After each file's conversion, run the full correctness gate. Any
mismatch in the emitted op stream fails `transcript_divergence` immediately.
Commit per file or per logical section.

**Exit**: `grep -rn 'ops.push' crates/jolt-compiler/` returns only
`ModuleBuilder`-internal hits. `jolt_core_module.rs` and `emit.rs` hold
no direct `ctx.ops.push(Op::…)` calls.

**Scope**: 3–5 commits, ~800 LOC churn.

---

## O3 — OPS.md inventory + `is_primitive()` classifier

**Why now**: pins the target-primitive form before O4/O5 start lowering.
Without a pinned target, every lowering becomes a design debate.

**What changes**:
- Write `crates/jolt-compiler/OPS.md` — one section per `Op` variant,
  classifying as one of:
  - **Primitive** — one logical step, no runtime conditional, backend-agnostic.
    Lowering target for others.
  - **Redundant** — different name for a primitive action (e.g.,
    `Op::InstanceBind` is a `Op::Bind` with compile-time dedup).
  - **Conditional** — runtime branch that compile-time tracking eliminates
    (e.g., `Op::MaterializeIfAbsent`).
  - **Protocol-specific** — names a protocol concept; must lower to
    primitives in O5.
  - **Batch-scaffold** — compiler-emitted per-window markers
    (`BatchRoundBegin`, `BatchRoundFinalize`, etc.). Keep.
  For each non-primitive, document the target lowering.
- Add an `impl Op` method:
  ```rust
  impl Op {
      /// True iff this variant is the canonical primitive form we want
      /// the compiler to emit pre-pass-pipeline. Enforced via assertion
      /// in the post-emit phase.
      pub fn is_primitive(&self) -> bool { /* match */ }
  }
  ```
- Add an assertion in the compiler entrypoint after emission, before any
  pass runs:
  ```rust
  debug_assert!(
      module.prover.ops.iter().all(Op::is_primitive),
      "compiler emitted non-primitive op: {:?}",
      module.prover.ops.iter().find(|op| !op.is_primitive()),
  );
  ```
- Initially `is_primitive()` returns `true` for every variant so the
  assertion is a no-op. As O4/O5 lower each non-primitive, the classifier
  flips — the assertion then catches every emission site that still uses
  the old form.

**Correctness rail**: classification is declarative. The assertion is a
no-op initially; its value compounds as O4/O5 land.

**Exit**: `crates/jolt-compiler/OPS.md` exists with a target form documented
for every non-primitive variant. `Op::is_primitive` compiles. Assertion
runs in debug builds without firing.

**Scope**: 1 commit, ~400 LOC doc + ~80 LOC classifier + assertion.

---

## O4 — Collapse redundant variants

**Why now**: the cheapest structural simplification. O4.a was compile-time
bookkeeping; O4.b wants producer analysis that's the same analysis O6/O7's
`DeadMaterializeElim` pass needs, so it's deferred there.

### O4.a Bind collapse — **LANDED** (`51cd49cf0`)

Four ops → one:
- `Op::InstanceBind { batch, instance, kernel, challenge }` — **deleted**
- `Op::InstanceBindPreviousPhase { batch, instance, kernel, challenge }` — **deleted**
- `Op::BindCarryBuffers { polys, challenge, order }` — **deleted**
- `Op::Bind { polys, challenge, order }` ← **sole bind primitive**

Compiler tracks per-batch-round bound-poly sets via an `emit_bind(...,
bound_this_round: &mut HashSet)` helper scoped to each `for round in ..`
loop. The runtime's `state.bound_this_round: HashSet<PolynomialId>` and
its `.clear()` site are deleted; every emitted `Op::Bind` carries a
pre-deduped poly list and the runtime arm binds unconditionally.

### O4.b Materialize collapse — **DEFERRED to O6/O7**

Two ops would collapse into `Op::Materialize`:
- `Op::MaterializeUnlessFresh { binding, expected_size }`
- `Op::MaterializeIfAbsent { binding }`

The **reason for deferral** (learned from the O4.b attempt in this branch):
compile-time producer analysis is required to decide emit-vs-elide at each
conditional site, and that analysis must enumerate the outputs of
`Op::MaterializeRA`, `Op::MaterializeCombinedVal`, `Op::MaterializePBuffers`,
`Op::WeightedSum`, `Op::BuildSegmentedEq`, plus track which
polys got bound-down (so `MaterializeUnlessFresh`'s size check can be
computed at compile time). **That's the same reaching-definitions analysis
O6/O7's `DeadMaterializeElim` pass needs**, so it lives there — once as a
reusable pass, not scattered inline at each emission site.

Until that pass lands, both variants remain in the Op enum with
`is_primitive()` returning `true` (the O3 classifier scaffold is
unchanged for these two). The runtime handler arms remain load-bearing.

Retry path for O4.b: it becomes a rewrite pass inside O7 that consumes a
producer-analysis result built in O6's pass infra. Input is the un-collapsed
stream with both conditional ops; output is the primitive form with only
`Op::Materialize` (or elided). The pass unit-tests against a hand-built
fixture module just like every other O7 rule.

**Correctness rail for both sub-phases**: dual-path validation (the
T2-C pattern). During the landing window, keep both arms running — old
conditional arms AND new primitive arms — and assert byte-equal
`device_buffers` state after each op. Delete the legacy arm only after
jolt-equivalence full suite greens on the new path.

**Exit (O4.a achieved)**: four bind ops → one. Runtime has no
`bound_this_round` set. O4.b's "three materialize ops → one" exit
deferred per above. `is_primitive()` returns `true` for `Op::Bind`
among the
collapsed families. Muldiv + `transcript_divergence` + `modular_self_verify`
green.

**Scope**: 2 commits (one per sub-phase), ~800 LOC delta.

---

## O5 — Lower protocol-specific compute ops

**Why now**: the big win. Every symptom of "runtime knows about protocols"
comes from these ops.

**Status snapshot**:

**Rename + field-slim wave** (no variant removal; field-set hygiene
that matches the future primitive signatures):

| Sub-phase | Status | Commit |
|---|---|---|
| `S5.rename` — `CheckpointEvalBatch` → `InstanceScalarUpdate` | landed | `4d5e36832` |
| `S5.build_segmented_eq` — `MaterializeSegmentedOuterEq` → `BuildSegmentedEq` (rename + field slim) | landed | `91dc8a2f6` |
| `S5.field_slim` — `UpdateInstanceWeights {num_phases, phase}` → `{suffix_len}` | landed | `52125c160` |
| `S5.scatter_field_slim` — `{SuffixScatter, QBufferScatter} phase → suffix_len` | landed | `0b8dd37b7` |
| `S5.init_weights_slim` — `InitInstanceWeights` drop `num_prefixes` (pre-size `state.instance_scalars` at runtime init) | landed | `51b00b7a3` |
| Docs refresh — revised S5 targets with accurate blocker classes | landed | `9c4f4d39a` |
| Docs sync — this file brought current with OPS.md | landed | `d012ed0ae` |
| Stale-ref cleanup — TASKS.md V9, module.rs, plan docs | landed | `f5186fd57` |

**Handler-extraction wave** (not in the original plan, but part of the
same rails-before-perf goal — every protocol-specific handler arm now
≤ 30 LOC per CLAUDE.md / ARCHITECTURE.md invariant). The extracted
helpers' signatures mirror what the Group A/B/C primitives will need,
so they become reference implementations for the eventual lowering:

| Handler | Prior | After | Helper | Commit |
|---|---|---|---|---|
| `Op::WeightedSum` | ~35 | 11 | `compute_weighted_sum` | `fff334402` |
| `Op::QBufferScatter` | ~64 | 14 | `compute_q_buffer_scatter` | `04942324d` |
| `Op::RafReduce` (pass 1) | ~57 | 24 | `compute_raf_reduce` | `9d825057d` |
| `Op::SuffixScatter` | ~37 | 22 | `compute_suffix_scatter` | `472c95097` |
| `Op::MaterializePBuffers` | ~40 | 17 | `compute_p_buffers` | `cdc3109a6` |
| `Op::MaterializeRA` | ~37 | 18 | `compute_ra_chunk` | `42016d772` |
| `Op::MaterializeCombinedVal` | ~40 | 22 | `compute_combined_val` | `cf1165a8b` |
| `Op::ReadCheckingReduce` | ~32 | 22 | `download_suffix_polys` | `9db25f266` |
| `Op::RafReduce` (pass 2) | 34 | 18 | (Q/P downloads absorbed) | `d79579a21` |

Plus `c9452ab54` removed a dead `RuntimeState.current_batch_round`
field (written but never read).

**Variant-removal wave** — lowered 4 protocol-specific ops (Group A +
Group C):

| Sub-phase | Status | Commit |
|---|---|---|
| Group A prerequisite — `state.instance_weights` → `device_buffers[PolynomialId::InstanceWeights]` | landed | `0b0ca8e10` |
| `S5.materialize_p_buffers` — `MaterializePBuffers` → 3× `WeightedSum` + 4 derived polys | landed | `0a89be18c` |
| `S5.init_instance_weights` — `InitInstanceWeights` → `InitExpandingTable` + N × `ExpandingTableUpdate` + `InstanceScalarUpdate` | landed | `795ef8f74` |
| `S5.update_instance_weights` — `UpdateInstanceWeights` → new `TraceGatherMultiply` primitive | landed | `186a9c4b4` |
| `S5.materialize_ra` — `MaterializeRA` → new `TraceGatherProduct` primitive | landed | `a97679d43` |

Op count: 48 → 46 (one net variant removed in Group C; three Group A
renames to new generic primitives keep count but move variants from
"lower" to "primitive — compute"). Five protocol-specific variants
still emit. See `crates/jolt-compiler/OPS.md` §"Protocol-specific —
lower to primitives" for the authoritative target table — this section
records the plan-level framing.

**Blocker groups** (from reading each remaining handler — the original
targets conflated three distinct classes under "simple reroute"):

- **(A) State-in-host + trace-driven**: `state.instance_weights: Vec<F>`
  is host-allocated and accessed by trace cycle index; the ops that
  read/write it while also consuming `provider.lookup_trace()` have no
  existing primitive analog. Relocating `instance_weights` to a device
  buffer (`PolynomialId::InstanceWeights`) plus introducing
  `Op::TraceGatherMultiply` / `Op::TraceGatherProduct` /
  `Op::TraceGatherIndexed` / `Op::TraceScatter` primitives is a
  prerequisite for 6 ops: `InitInstanceWeights`, `UpdateInstanceWeights`,
  `MaterializeRA`, `MaterializeCombinedVal`, `SuffixScatter`,
  `QBufferScatter`.
- **(B) Kernel formula lowering**: `ReadCheckingReduce` / `RafReduce`
  lower to `Op::Reduce` once `KernelSpec.formula` can express
  combine-entries + gamma-weighted products. Affects 2 ops.
- **(C) `WeightedSum` shape mismatch**: `MaterializePBuffers` produces
  `base_scalar + index_fn(i)` where `index_fn ∈ {i, lo(i), ro(i)}`.
  Feasible standalone with new preprocessed polys (`ChunkSizeConst`,
  `HalfChunkSizeConst`, `UninterleaveLo`, `UninterleaveRo`) + 3×
  `Op::WeightedSum`. Affects 1 op.

**Revised graduation order** (subsequent sub-phases):

1. `S5.materialize_p_buffers` (Group C, standalone): preprocessed-poly
   infrastructure + 3× `Op::WeightedSum` emission. One commit.
2. `S5.instance_weights_device` (Group A prerequisite, multi-commit):
   relocate `state.instance_weights` to a device buffer; introduce
   `Op::TraceGatherMultiply` + `Op::TraceScatter` primitives. Unblocks
   6 ops.
3. `S5.kernel_formula` (Group B prerequisite): extend `KernelSpec` for
   combine-entry + gamma-weighted product shapes. Unblocks 2 ops.
4. Remaining ops lower one-by-one post-prerequisites.

**Correctness rail**: dual-path validation per op family. For each op X:
1. Compiler emits BOTH the old `Op::X` AND the new primitive-lowered sequence.
2. Runtime executes the old `Op::X` arm and skips the new primitives (or vice
   versa behind an env toggle, like `JOLT_PROTOCOL_LOWERED=1`).
3. Shadow-assert that the runtime state after the new sequence equals state
   after the old op (`device_buffers`, `instance_scalars`, etc.).
4. Flip to new-only after jolt-equivalence full suite greens.
5. Delete the old `Op::X` variant + its handler arm.

Same pattern that unified-reduce (T2-C) validated. Re-use that bridge.

**Also deleted in O5**: `crates/jolt-zkvm/src/runtime/prefix_suffix.rs` — a
131-LOC file that exists only to implement the protocol-specific
`Op::SuffixScatter` et al. Once those ops are lowered, the file's callers
disappear and it deletes itself.

**Exit**:
- 9 protocol-specific ops removed from the `Op` enum.
- Runtime `handlers.rs` arm count drops from ~41 to ~23.
- `grep -E 'Op::(Lookup|Ram|Instruction|RA|RAF|Booleanity|Checkpoint|Read|Raf|Suffix|QBuffer)' crates/jolt-zkvm/src/runtime/handlers.rs` returns zero hits.
- `Op::is_primitive()` returns `true` for every remaining variant.
- `runtime/prefix_suffix.rs` is deleted.
- Full jolt-equivalence suite green; all three handler-size grep tests pass.

**Scope** (revised after blocker analysis): ~10–15 commits, ~3000–4500
LOC churn including the Group A / Group B prerequisite infrastructure.
Original estimate of 5–10 commits / 2000–3000 LOC was based on the
optimistic "simple reroute" framing; the actual infrastructure for
device-side `instance_weights` and `KernelSpec.formula` extension is
larger than the original plan anticipated.

---

## O6 — Pass infrastructure

**Why now**: with primitive IR stable (post-O5), fusion rules have a clean
surface to rewrite against.

**What changes** — new module `crates/jolt-compiler/src/passes/`:

```rust
// crates/jolt-compiler/src/passes/mod.rs
pub trait CompilerPass {
    /// Rewrite a module. Must be pure: same input ⇒ same output. Must
    /// preserve observable proof-producing behavior (no-op at worst).
    fn run(&self, module: Module) -> Module;

    /// Stable identifier for logging and test fixtures.
    fn name(&self) -> &'static str;
}

/// Apply passes in order. Each pass sees the output of the previous.
pub fn run_pipeline(module: Module, passes: &[Box<dyn CompilerPass>]) -> Module { … }

pub mod pipelines {
    /// Empty pipeline — reference implementation. Always identity.
    /// Every other pipeline must transcript-match this one on fixtures.
    pub fn debug_pipeline() -> Vec<Box<dyn CompilerPass>> { vec![] }

    /// CPU target pipeline. Populated in O7 with the proof-of-concept rules.
    pub fn cpu_pipeline() -> Vec<Box<dyn CompilerPass>> { vec![] }

    /// Metal target pipeline. Empty for now; lands in post-O8 perf work.
    pub fn metal_pipeline() -> Vec<Box<dyn CompilerPass>> { vec![] }
}
```

Compilation becomes:

```rust
pub fn compile_for(protocol: &Protocol, target: CompileTarget) -> Module {
    let module = emit(protocol);              // canonical primitive stream
    run_pipeline(module, &target.pipeline())  // target's passes
}
```

Target selection comes from either (a) an explicit `CompileTarget` parameter
or (b) an env toggle `JOLT_COMPILE_TARGET=debug|cpu|metal` that CI sets.
The toggle mirrors `JOLT_FUSE_DEBUG` from T0 and gives the same
belt-and-suspenders dual-path property.

**Test infrastructure** lives in `crates/jolt-equivalence/tests/passes.rs`:

```rust
/// Unit test: assert a pass rewrites `input` to exactly `expected`.
/// Op-stream equality is checked element-by-element.
pub fn assert_pass_rewrites<P: CompilerPass>(pass: P, input: Module, expected: Module);

/// Integration test: assert `cpu_pipeline()` and `debug_pipeline()` both
/// produce valid proofs with byte-identical Fiat-Shamir transcripts on a
/// small muldiv / sha2-chain fixture.
#[test]
fn cpu_pipeline_matches_debug_pipeline_muldiv() { … }
```

**Correctness rail**: two levels.
- Per-pass unit tests pin pass behavior on tiny fixtures.
- Pipeline integration tests enforce transcript equivalence between any
  non-empty pipeline and `debug_pipeline()` on the muldiv gate.
- CI runs the muldiv gate twice: once under `JOLT_COMPILE_TARGET=debug`,
  once under `JOLT_COMPILE_TARGET=cpu`. Both must pass. Difference in
  transcripts fails the build.

**Exit**:
- `CompilerPass` trait + `run_pipeline` + three named pipelines exist.
- The `debug_pipeline` integration test runs the muldiv gate to green.
- `jolt-equivalence` has ≥ 1 `assert_pass_rewrites`-based test as a
  smoke demonstration (using a no-op pass).

**Scope**: 2 commits, ~500 LOC infra + ~300 LOC tests.

---

## O7 — First rewrite rules (proof of concept)

**Why now**: proves the pass pipeline is usable. Three rules is enough to
validate the test infra and the target-pipeline composition.

**Rules to land**:

1. **`MergeBatchRoundReduces`** — collapse N adjacent length-1 `Op::Reduce`
   inside a `BatchRoundBegin..BatchRoundFinalize` window into one length-N
   `Op::Reduce { specs: Vec<ReduceSpec> }`. Pure reordering. Unlocks the
   single-rayon-region dispatch for batch rounds.

2. **`BindCoalesce`** — adjacent
   `Op::Bind { polys: A, challenge: c, order: o }` +
   `Op::Bind { polys: B, challenge: c, order: o }`
   → `Op::Bind { polys: A ∪ B, challenge: c, order: o }`.
   Pure reordering. Common at phase transitions where carry-bindings and
   kernel-input binds emit separately.

3. **`DeadMaterializeElim`** — drop `Op::Materialize { binding }` for
   polys that are re-produced by a later `Op::WeightedSum` or overwritten
   before read. Pure deletion, driven by the producer-map analysis that
   O4.b already built.

Each rule:
- Gets a file `crates/jolt-compiler/src/passes/<rule>.rs` (≤ 200 LOC).
- Implements `CompilerPass`.
- Ships with a unit test fixture in `crates/jolt-equivalence/tests/passes.rs`
  that hand-constructs a tiny `Module`, runs the pass, asserts the output
  `Vec<Op>` matches expected.
- Gets added to `cpu_pipeline()`.

**Correctness rail**:
- Per-rule unit tests pin behavior.
- The muldiv + modular_self_verify gate runs once with `JOLT_COMPILE_TARGET=cpu`
  (all three rules active). Transcript must match `JOLT_COMPILE_TARGET=debug`.

**Exit**: `cpu_pipeline()` contains 3 named passes. Each has a standalone
unit test. Full jolt-equivalence suite green under both pipelines.

**Scope**: ~1 commit per rule + 1 rollup, ~600 LOC total.

---

## O8 — Backend-specific pipelines + perf iteration

**Why now**: the payoff. Rails are good; now we play perf golf with
confidence.

**What changes**: the Perf Loop Protocol resumes. Each new perf iter adds a
single rewrite rule to the target pipeline, following the O7 protocol:

1. Pick a hypothesis from a fresh Perfetto trace.
2. Write the rule as a `CompilerPass`.
3. Unit-test it (input op stream → expected output) in jolt-equivalence.
4. Add it to `cpu_pipeline()` (or the relevant target pipeline).
5. Run the muldiv + modular_self_verify gate under both `JOLT_COMPILE_TARGET=debug`
   and `JOLT_COMPILE_TARGET=cpu`. Both green → commit.
6. Re-measure; accept/reject per the standard Perf Loop band thresholds.

Candidate rules for the first few iters, depending on what the profile
shows post-O7:

- `FuseBindReduce` — if we introduce `Op::BindReduce` (a fused in-place
  Gruen-style bind + next-round-poly eval), a pass that rewrites adjacent
  `Op::Bind + Op::Reduce` into the fused form for matching kernels.
- `ShareOuterEq` — identify adjacent `Op::Reduce` whose `ReduceAxes::Product`
  specs share an outer-eq buffer; fuse into one multi-spec call that the
  backend can prefetch once.
- `FoldInactiveClaim` — precompute inactive-instance claim updates at
  compile time (the claim/2 pattern) instead of per-op runtime arithmetic.
- `MergeAdjacentAbsorb` — some transcript ops can combine safely.

Each becomes its own perf-loop iter with the standard journal commit format.

**Exit**: open-ended. The loop runs until Phase 3 stop condition
(modular ≤ core at log_T ∈ {18, 20}) or the user explicitly halts.

---

## Dependencies DAG

```
        O1 ──┐
             ├──► O2 ──► O3 ──► O4 ──► O5 ──► O6 ──► O7 ──► O8
(independent; O1, O2, O3 can be any
 order but running them topologically
 makes every downstream phase cleaner)
```

Parallelism within phases:
- O4.a and O4.b are independent; can land in parallel commits.
- O5's sub-phases by op family are largely independent; order within
  dependency groups (InitInstanceWeights before MaterializeRA, etc.).
- O6 can start any time after O3. Pass infra doesn't need simplified IR
  but benefits from it.
- O7 depends only on O6, not O5-complete. The batch-round-merge rule
  works on today's `Op::Reduce { specs: Vec<ReduceSpec> }` shape.

## Correctness philosophy

Single mechanism across the whole overhaul: **the muldiv +
modular_self_verify + transcript_divergence + zkvm_proof_accepted_by_core_verifier
gate must pass at every commit.**

Two layered rails bridge structural changes:

1. **Dual-path validation** for every O4/O5 sub-phase. New primitive path
   runs alongside the legacy op; byte-equal state assertion catches
   regressions before the legacy path is deleted.
2. **`debug_pipeline` as permanent reference.** Once O6 lands, every
   pipeline change (O7, O8) must transcript-match `debug_pipeline` on the
   muldiv gate. CI runs both sides.

No commit lands without:
- `cargo nextest run -p jolt-equivalence` — full suite green
- `cargo clippy -D warnings` — on jolt-core (host + host,zk), jolt-cpu,
  jolt-compute, jolt-compiler, jolt-zkvm
- Post-O6: both `JOLT_COMPILE_TARGET=debug` and `JOLT_COMPILE_TARGET=cpu`
  runs of the muldiv gate green

## Out of scope (deliberately excluded)

- **Perf improvements.** O1–O7 are correctness-only. O8 opens the perf door.
- **GPU backend full wire-up.** Metal stays minimal (Gruen not implemented,
  `MetalBackend::reduce` handles Flat + Product-Dense only). Expanding
  Metal is post-rails.
- **New proof protocols.** No Plonk, Groth16, Halo2. Rails first.
- **New RISC-V instructions.** Instruction extension is an orthogonal axis.
- **Verifier-side changes.** The `VerifierSchedule` IR and verifier runtime
  are stable; only the prover's op pipeline changes.

## Size estimate

| Phase | Commits | Rough effort |
|---|---|---|
| O1 | 1 | ½ day |
| O2 | 3–5 | 2–3 days |
| O3 | 1 | 1 day |
| O4 | 2 | 2–3 days |
| O5 | 5–10 | 2–3 weeks |
| O6 | 2 | 2–3 days |
| O7 | 4 | 3–4 days |
| O8 | open | perpetual |

**Total to get from "rails being built" → "rails ready, perf loop resumed":**
~4–5 weeks of focused work. O5 dominates. O1–O3 could ship in a few days
and give immediate compounding benefit (every subsequent phase is easier
to test / reason about).

## Commit discipline

Per the Perf Loop Protocol (CLAUDE.md), commits fall into two classes:

- **Structural commits** (O1–O7): message prefix `refactor(streamline):`
  followed by `S<phase>.<sub> <short summary>`. E.g.:
  - `refactor(streamline): S1 move BufferProvider + LookupTraceData to jolt-zkvm`
  - `refactor(streamline): S4.a collapse Bind variants into Op::Bind`
  - `refactor(streamline): S5 lower Op::SuffixScatter to Op::WeightedSum + Op::Reduce`
- **Perf commits** (O8 onward): standard Perf Loop format
  (`perf(<scope>):` or `journal:`).

No cross-phase commits. No "while I'm here" unrelated changes.

Reverts get a `journal:` bookkeeping commit so dead ends aren't rediscovered.

## How to resume

1. Read this file plus `CLAUDE.md` ("Task Loop Protocol" section).
2. Read `crates/jolt-compiler/OPS.md` once it exists (post-O3).
3. Check `git log --grep='refactor(streamline):'` to see what's landed and
   what's next in sequence.
4. Pick up at the first unchecked phase. Every phase is self-contained
   (inputs: the prior phase's exit state; outputs: this phase's exit state).
5. Correctness gate every commit. No exceptions.

## Cross-reference

- **CLAUDE.md "Task Loop Protocol"** — the cycle used for O4/O5 dual-path
  validation (step-by-step).
- **CLAUDE.md "Perf Loop Protocol"** — the cycle O8 returns to.
- **`crates/jolt-zkvm/ARCHITECTURE.md`** — the current-state architectural
  doc; this ticket's exit state lands there after O5 completes.
- **`crates/jolt-bench/opt/02-unified-reduce.md`** — the Ticket 2
  dual-path pattern, re-used in O4 and O5.
- **`crates/jolt-bench/opt/00-validation-harness.md`** — Ticket 0's
  shadow-stream pattern; conceptual ancestor of O6's `JOLT_COMPILE_TARGET`
  dual-pipeline CI.
