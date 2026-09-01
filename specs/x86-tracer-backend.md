# Spec: x86 Tracer Backend (AOT-Transpiled Trace Generation)

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @moodlezoup, Claude            |
| Created     | 2026-07-28                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

Trace generation is the one part of proving that does not parallelize: the first execution pass is inherently sequential, and every downstream consumer waits on it. It is already a latency bottleneck for multi-GPU prover deployments, and it becomes a *throughput* bottleneck under streaming sumcheck, which regenerates the trace on-the-fly multiple times per proof to avoid holding all of it in memory (the legacy prover already re-executes the entire guest a second time during the streaming commit — `crates/jolt-prover-legacy/src/zkvm/prover.rs:749-767` — and `crates/jolt-witness` re-walks the trace from row 0 for every oracle materialization). Our tracer is a decode-dispatch interpreter forked from `takahirox/riscv-rust`: it re-fetches and re-decodes every instruction on every tick (`tracer/src/emulator/cpu.rs:497-541`), backs guest RAM with a `HashMap<usize, u64>` (`tracer/src/emulator/memory.rs:7-16`), and allocates a fresh inline-expansion `Vec` per executed instruction (`cpu.rs:531`). No benchmark anywhere in the workspace measures tracing speed in isolation. Meanwhile the field has converged on transpilation: ZisK reports **1.5 GHz** trace generation from ahead-of-time (AOT) compilation of RISC-V to x86 (3–4 x86 instructions per RISC-V instruction), roughly 10× the ~150 MHz interpreter state of the art, and SP1 ships `sp1-jit`, an in-process whole-program AOT transpiler built on dynasm-rs.

This spec commits to four things: **(1)** formalize the chunked, checkpointed execution contract as a first-class trait in `jolt-program`, so trace generation is plug-and-play per backend and per architecture; **(2)** build `crates/jolt-tracer-x86`, an AOT transpiler that compiles a `JoltProgram`'s expanded bytecode to native x86-64 once and executes it in two modes (a fast checkpointing pass and a parallel per-chunk recording pass); **(3)** build the backend-generic benchmark and differential-test infrastructure first — per-instruction unit tests and iai-callgrind microbenchmarks, whole-guest throughput and peak-memory benchmarks — and record the reference baseline before optimizing; **(4)** keep the existing interpreter as the untouched reference implementation, default backend, and correctness oracle. The x86 backend must produce a `TraceRow` stream bit-identical to the reference on every supported guest, be ≥10× faster end-to-end, and use no more peak memory.

## Intent

### Goal

Build an x86-64 AOT-transpiling implementation of the existing `ExecutionBackend` seam that generates traces ≥10× faster than the reference interpreter with bit-identical output and no additional peak memory, behind a new chunked-execution trait that any future backend (e.g. ARM64) can implement for parallel, checkpointed re-tracing.

Key abstractions:

- **`ChunkedExecutionBackend`** (new, `crates/jolt-program/src/execution/backend.rs`) — extends the existing `ExecutionBackend` trait (`backend.rs:3-11`) with the two-pass contract that streaming consumers need. The associated `Checkpoint` type is deliberately opaque: each backend chooses its own resume-state representation.

  ```rust
  pub trait ChunkedExecutionBackend: ExecutionBackend {
      /// Everything needed to deterministically re-execute one chunk,
      /// independent of every other chunk.
      type Checkpoint: Send + Sync;

      /// Fast pass: run the program to completion WITHOUT materializing trace
      /// rows. `checkpoints[i]` resumes at trace cycle `i * chunk_size`;
      /// `checkpoints.len() == trace_len.div_ceil(chunk_size)`.
      fn execute(
          &mut self,
          program: &JoltProgram,
          inputs: TraceInputs,
          chunk_size: usize,
      ) -> Result<ExecutionSummary<Self::Checkpoint>, TraceError>;

      /// Recording pass: re-execute one chunk, materializing exactly
      /// `chunk_size` trace rows (fewer for the final chunk — replay emits no
      /// padding rows; padding stays the consumer's job, as in the existing
      /// RowSource contract). Takes `&self` so disjoint chunks can be
      /// replayed in parallel, in any order.
      fn replay_chunk(&self, checkpoint: &Self::Checkpoint)
          -> Result<Self::Trace, TraceError>;
  }

  pub struct ExecutionSummary<C> {
      pub checkpoints: Vec<C>,
      pub trace_len: usize,
      pub device: JoltDevice,
      pub final_memory: Option<MemoryImage>,
      pub advice_tape: Option<Vec<u8>>,
  }
  ```

  Neither backend enforces `max_trace_length` — parity with today, where enforcement happens at prove time (`crates/jolt-prover-legacy/src/zkvm/prover.rs:354-360`); `execute` runs to guest termination.

- **Advice-tape plumbing** (extension, `crates/jolt-program/src/execution/trace.rs`) — the modular seam cannot express the SDK's two-pass advice flow today (`jolt-sdk/macros/src/lib.rs:905-938`): `TracerBackend` neither seeds a tape (hardcodes `None`, `tracer/src/execution_backend.rs:48`) nor captures the populated tape that pass 1 must harvest (discarded as `_advice_tape`, `execution_backend.rs:41`). Add `pub advice_tape: Option<Vec<u8>>` to **both** `TraceInputs` (seeding; the tape is plain bytes plus a cursor that always starts at 0, `tracer/src/emulator/cpu.rs:22-25`) and `TraceOutput`/`ExecutionSummary` (capture), populated by both backends, so either backend can run either pass.

- **`X86TracerBackend`** (new crate, `crates/jolt-tracer-x86`) — implements `ExecutionBackend` and `ChunkedExecutionBackend` with `Trace = OwnedTrace`. On first use for a given `JoltProgram`, compiles the program to native x86-64 via dynasm-rs and caches the artifact. Native codegen is gated to `cfg(all(target_arch = "x86_64", target_os = "linux"))` (the SP1/ZisK precedent). The crate additionally exports a cfg-selected alias `pub type NativeBackend` — `X86TracerBackend` on x86-64 Linux, `tracer::TracerBackend` elsewhere — re-exported by the SDK next to the existing `TracerBackend` re-export (`jolt-sdk/src/host_utils.rs:25`); existing call sites are already generic over `B: ExecutionBackend`, so the alias is the entire selection API.

- **Row templates** — the load-bearing codegen insight: the expansion of every source instruction is *static per PC*. Runtime tracing materializes inline sequences through the same `jolt_program::expand` pipeline used to build the committed bytecode (`tracer/src/instruction/mod.rs:750-771`, `tracer/src/instruction/inline.rs:239-258`), including the `rd = x0` rewrite applied at expansion level (`crates/jolt-program/src/expand/mod.rs:129-146`); recipes "may only expose advice row positions", never runtime-dependent shapes (`expand/mod.rs:106-113`); and the parity is mechanically tested by the `source_to_jolt_expansion_equivalence` invariant. So for each expanded bytecode row, the `instruction: JoltInstructionRow` field and row skeleton are compile-time constants; only register values, RAM values, and advice values are dynamic. Record mode copies a pre-built `TraceRow` template and patches the dynamic fields — no per-row construction logic at runtime. (The row *value-shape* contracts — e.g. load value equals rd post-value — are separately checked by `cycle_to_trace_row`, `tracer/src/trace_row.rs:53-118`, on the legacy path.)

- **`InlineAdviceContext`** (refactor, `tracer` + `jolt-inlines/sdk`) — registered inline advice builders currently take the interpreter's `&mut Cpu` (`build_advice: AdviceFn`, `tracer/src/instruction/inline.rs:43-51`). `specs/inline-expansion-grammar.md` already specifies the replacement seam — `InlineAdviceContext`, a minimal operands + CPU/memory-read view with binding rules at its lines 364-375 — and this spec adopts that name and contract rather than redesigning it: the x86 backend's state becomes the second implementor, so inline advice (sha2, bigint, secp256k1, …) is computed by the same Rust code under both backends.

### Invariants

1. **Row-stream equivalence.** For every supported execution (defined below), `X86TracerBackend` produces a `TraceRow` stream bit-identical to `TracerBackend`'s: same row count, and per-row equality of the full `TraceRow` (`crates/jolt-program/src/execution/trace.rs:232-238`; the `cfg(field-inline)` fourth field is trivially `None` — see Non-Goals). A *supported execution* is one whose every dynamic PC lands on the start of a statically decoded instruction (equivalently: an execution consistent with the committed bytecode). The interpreter will happily fetch and decode a jump target in the middle of a 4-byte instruction whose tail bytes decode (`cpu.rs:505-524`) — such executions are unprovable anyway (the PC has no committed bytecode row) and are out of scope: the x86 backend surfaces them as runtime errors, and the equivalence harness treats interpreter-executes/x86-errors on such targets as conforming.
2. **Device and final-memory equivalence.** `TraceOutput::device` (inputs, outputs, panic flag), `final_memory`, and the captured `advice_tape` are equal across backends for the same program and inputs.
3. **Determinism and chunking-independence.** The same program + inputs yield identical `ExecutionSummary` artifacts across runs, and the concatenation of `replay_chunk` outputs equals the eager `trace()` row stream for *every* chunk size — including degenerate ones (1, larger than the trace) — regardless of replay order or parallelism.
4. **Proof equivalence.** A proof generated by the modular prover from an `X86TracerBackend` trace is byte-identical to one generated from a `TracerBackend` trace (same preprocessing, same transcript), extending the existing byte-diff methodology (`crates/jolt-prover/tests/byte_diff.rs`).
5. **Reference backend untouched.** `TracerBackend` remains the default backend everywhere; its observable behavior does not change. The only tracer-side changes are additive (advice-tape plumbing, `InlineAdviceContext`, `ChunkedExecutionBackend` impl wrapping the existing `CheckpointingTracer`).
6. **Host-memory safety.** Generated code never reads or writes host memory outside the backend-owned guest-memory plane, trace buffers, and helper-call ABI. Code buffers are never writable and executable simultaneously (W^X: assemble RW, `mprotect` to RX). Every `unsafe` block carries a `SAFETY:` comment (workspace denies `undocumented_unsafe_blocks`).
7. **Fail-fast coverage.** Any bytecode row the transpiler does not support returns `TraceError` at compile time (backend setup) — never a silent wrong-semantics execution and never a partial trace. Runtime faults (indirect jump to a non-decoded PC, out-of-bounds or region-violating access) surface as `Err(TraceError)` from `trace`/`execute`/`replay_chunk`; the reference interpreter panics in these situations today and keeps doing so (invariant 5), an asymmetry the equivalence harness explicitly accepts as conforming. Partial traces are never compared.

`jolt-eval` plan:

- Existing invariants must keep passing unchanged, in particular `soundness` (whose prove path exercises the default reference tracer via `tracer::trace`) and `source_to_jolt_expansion_equivalence` (whose static-expansion guarantee this design leans on).
- New, via `/new-invariant` during implementation:
  - `tracer_backend_equivalence` (Test, Fuzz, RedTeam) — invariants 1–2 above. Modeled on `split_eq_bind_low_high` (reference-vs-optimized lockstep comparison, `jolt-eval/src/invariant/split_eq_bind.rs:48`) for the Test/Fuzz targets: `Setup` compiles the **equivalence corpus** — the six profile guests (fibonacci, sha2, sha3, sha2-chain, sha3-chain, btreemap), `muldiv-guest`, `examples/advice-demo` (advice tape), and inline-exercising guests covering the sha2, keccak256, blake3, bigint, and secp256k1 inlines (from `jolt-inlines/fixtures`, adding fixture guests there where none exists) — `Input` selects a guest and fuzzes its input bytes, `check` runs both backends and compares row streams (first divergent row index in the violation details; matching failure outcomes per invariant 7 are conforming). The RedTeam target follows the `soundness` sandbox pattern (`jolt-eval/src/invariant/soundness.rs:113`): the agent writes an adversarial guest in `guest-sandbox/` to make the backends diverge.
  - `tracer_chunk_composition` (Test, Fuzz) — invariant 3 above, checked for **both** backends: `Input` is a (guest, input bytes, chunk size) triple; `check` compares `execute` + `replay_chunk` concatenation against the eager row stream.

### Non-Goals

- **An ARM64/Apple-Silicon backend.** This spec delivers x86-64 Linux only. The trait seam, differential-test harness, and benchmarks are backend-generic by construction so that a future `jolt-tracer-arm64` is "implement codegen, inherit the test suite" — but no ARM codegen is in scope here. (Alternative 11 records the likely codegen route for that follow-up and the trigger it flips on.)
- **Rewriting or optimizing the reference interpreter.** The interpreter's known inefficiencies (per-tick re-decode, HashMap memory, per-instruction expansion allocs) stay as-is; it is the stable correctness oracle and portability fallback. Interpreter-side changes are limited to the additive items in invariant 5.
- **Rewiring the legacy prover.** `JoltCpuProver` keeps consuming `LazyTraceIterator` + `Arc<Vec<Cycle>>` (`crates/jolt-prover-legacy/src/zkvm/prover.rs:202-203`) through the default tracer. The x86 backend targets the modular seam (`TraceRow`); it does not emit the legacy `Cycle` type.
- **Changing any proof-facing format or protocol.** `TraceRow`, `Cycle`, `JoltInstructionRow`, bytecode preprocessing, padding rules, and Fiat-Shamir inputs are all untouched.
- **Non-Linux native codegen.** macOS x86_64 (MAP_JIT/W^X entitlements) and Windows are out of scope; those targets get the reference backend via `NativeBackend`. This mirrors SP1 and ZisK, both of which gate their native paths to Linux x86-64.
- **RV32 or ISA extensions beyond the `RV64IMAC_JOLT` profile** (plus registered inlines). The tracer itself is RV64-only (`tracer/src/lib.rs:656`).
- **The `field-inline` feature.** `crates/jolt-tracer-x86` does not enable it: the feature adds a fourth `TraceRow` field (`trace.rs:236-237`) and `FIELD_*` instruction kinds (`crates/jolt-riscv/src/lib.rs:161-177`); a field-inline `JoltProgram` handed to the x86 backend fails fast per invariant 7. Row equivalence is over the full `TraceRow`, with `field_inline` trivially `None` in supported builds.
- **Self-modifying guest code.** The AOT model assumes static bytecode — the same assumption the proof system itself makes (bytecode is committed at preprocessing). The reference interpreter technically re-fetches from RAM each tick, but a guest that rewrites its own instructions could not be proven anyway. The backend may cheaply detect writes to the program image and fail fast; supporting SMC is explicitly out of scope.
- **Parallelizing the first pass.** The fast pass is inherently sequential (that is the premise); parallelism comes only from chunk replay.
- **GPU trace generation, and prover-side adoption of chunked re-tracing.** Wiring `ChunkedExecutionBackend` into the streaming sumcheck prover (replacing `jolt-witness`'s from-row-0 re-walks in `backend/trace/cycle.rs:73-139` with checkpoint-based `visit_chunks`) is enabled by this spec but delivered by the streaming-prover work that consumes it.

## Evaluation

### Acceptance Criteria

- [ ] **AC1 — Baseline first.** Backend-generic trace-generation benchmarks exist and reference-backend baseline numbers (MHz and peak RSS per profile guest) are recorded in the PR description before any x86 codegen lands (Execution slice 0).
- [ ] **AC2 — Chunked seam.** `ChunkedExecutionBackend` is implemented by both `TracerBackend` (wrapping the existing `CheckpointingTracer`/`Checkpoint` machinery, `tracer/src/lib.rs:377-588`) and `X86TracerBackend`; chunk-composition tests pass for both backends with chunk sizes {1, 100, 2^18, > trace length}.
- [ ] **AC3 — Per-instruction coverage.** Every instruction kind enumerated by `jolt_riscv::for_each_instruction_kind!` (the same macro that generates the `Instruction`/`Cycle` enums, `tracer/src/instruction/mod.rs:862`) plus `INLINE` has: (a) a macro-generated differential unit test comparing x86 execution against the reference `Cpu` over ≥1000 random operand/state instances (modeled on the existing execute-vs-trace harness, `tracer/src/instruction/test.rs:103-194`), and (b) an iai-callgrind microbenchmark. Macro-driven enumeration makes coverage exhaustive by construction: a new kind without a test fails to compile; one without a committed baseline fails the nightly iai lane.
- [ ] **AC4 — Architecture tests.** Every RISC-V ACT4 test (`tests/arch-tests/` + `third-party/riscv-arch-test`, root `Makefile` target `arch-tests-64imac`) that the `JoltProgram` decode/expand pipeline accepts passes under the x86 backend via the new `jolt-emu-x86` runner bin, with signatures matching the reference `jolt-emu`; pipeline-rejected tests are skipped and listed (the `skip.txt` precedent).
- [ ] **AC5 — Equivalence invariant.** `tracer_backend_equivalence` (Test, Fuzz, RedTeam) is added via `/new-invariant`, registered in `JoltInvariants`, and green over the equivalence corpus defined in the Invariants section.
- [ ] **AC6 — Composition invariant.** `tracer_chunk_composition` (Test, Fuzz) is added via `/new-invariant`, registered, and green for both backends.
- [ ] **AC7 — Proof byte-equality.** A `byte_diff`-style test proves the same statement from both backends' traces and asserts byte-identical proofs, 10/10 runs.
- [ ] **AC8 — Throughput gate.** Steady-state eager `ExecutionBackend::trace` via `X86TracerBackend` (compiled artifact cached; compile time excluded and reported separately) is ≥10× faster than `TracerBackend` on sha2-chain (≈15M cycles) and fibonacci (≈4.8M cycles), median of ≥5 runs on the slice-0 benchmarks, evaluated at PR acceptance on a linux-x86_64 workstation-class machine (≥16 cores).
- [ ] **AC9 — Fast-pass gate.** `ChunkedExecutionBackend::execute` at `chunk_size = 2^18` is ≥25× faster than reference eager tracing on the same guests, same procedure as AC8; absolute MHz is reported (tracked, not gated).
- [ ] **AC10 — Memory gate.** Peak RSS (`VmHWM`) of eager x86 tracing ≤ reference eager tracing on the AC8 guests; peak RSS of the chunked pipeline (fast pass, then replay at `chunk_size = 2^18` with N = 8 workers whose chunk buffers are held concurrently) < eager tracing of either backend.
- [ ] **AC11 — Safety checks.** A test asserts the finalized code mapping is non-writable (via `/proc/self/maps`) and that an out-of-bounds/region-violating guest access surfaces as the defined `TraceError`, not UB (invariant 6; the `SAFETY:`-comment clause is enforced by the workspace `undocumented_unsafe_blocks` deny).
- [ ] **AC12 — Portability.** Non-x86_64 / non-Linux builds compile the crate with native codegen cfg'd out and `NativeBackend = TracerBackend` (verified by `cargo check -p jolt-tracer-x86 --target aarch64-unknown-linux-gnu`; requires `rustup target add aarch64-unknown-linux-gnu` on the pinned toolchain, or adding the target to `rust-toolchain.toml`).
- [ ] **AC13 — Hygiene.** `cargo fmt -q` and `cargo clippy --all --features host -q --all-targets -- -D warnings` (and `host,zk`) pass; `cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host` and `--features host,zk` still pass unchanged (default paths untouched).

### Testing Strategy

**Existing tests that must keep passing unchanged:** the full `cargo nextest run` suite; the muldiv e2e test in both `host` and `host,zk` modes; tracer's own tests (execute-vs-trace equivalence, decode parity, checkpoint round-trip, Cycle size assertion); the ACT4 suite on the reference backend; all registered jolt-eval invariants. ZK mode does not touch trace generation, but the muldiv both-modes gate stays as the canary that default prover paths are unaffected.

**New tests:**
- Per-instruction differential unit tests and per-instruction fuzzing (random instruction words + random pre-state) in `crates/jolt-tracer-x86`, macro-generated for exhaustiveness as described in AC3.
- Whole-guest equivalence tests over the equivalence corpus, run as ordinary `#[test]`s (guests built via the `jolt` CLI, following the pattern in `tracer/src/lib.rs:782-816`).
- Chunk-size sweep tests for both backends (invariant 3), including chunk size 1, which forces checkpoint offsets inside multi-row groups (e.g. SC.D expands to ~15 rows).
- Advice-tape round-trip test: seed a tape through `TraceInputs::advice_tape`, verify both backends consume it identically **and** capture the populated output tape identically (covers both halves of the SDK two-pass flow).
- Fail-fast tests: a synthetic `JoltProgram` containing an unsupported row makes backend setup return `TraceError`; an indirect jump to a non-decoded PC returns the runtime `TraceError` (invariant 7).
- Safety tests per AC11.
- Fuzz targets are synthesized from the two new invariants via the `#[invariant(..., Fuzz)]` macro and `./jolt-eval/sync_targets.sh`.

New-crate tests are field-free and mode-independent; they run once (no `host,zk` matrix needed). CI runs unit/equivalence/composition tests; iai-callgrind microbenchmarks run in a nightly/manual lane (they require valgrind), with committed baselines checked by `--regression` thresholds.

### Performance

Posture: hard **relative** gates against the reference backend, **tracked** absolute numbers, measured by infrastructure this spec itself delivers — no in-repo baseline exists today. Gates are evaluated once at PR acceptance (procedure and parameters pinned in AC8–AC10: median of ≥5 subprocess runs, `chunk_size = 2^18`, N = 8 replay workers, guest inputs pinned to the `e2e_profiling.rs` defaults — fibonacci n = 400000, sha2-chain iterations sized to ≈15M cycles); the iai lane is the recurring regression net.

| Metric | Gate | Tracked |
|---|---|---|
| Eager trace throughput (record mode, steady-state) | ≥10× reference on sha2-chain, fibonacci (AC8) | absolute MHz per profile guest; one-time AOT compile wall-clock (not gated) |
| Fast pass throughput (no rows, chunk_size 2^18) | ≥25× reference eager (AC9) | absolute MHz; SOTA context: interpreters ~150 MHz, ZisK AOT 1.5 GHz |
| Peak RSS, eager | ≤ reference eager (AC10) | absolute bytes |
| Peak RSS, chunked pipeline (2^18, N=8) | < eager (either backend) (AC10) | bytes vs. (checkpoints + N chunk buffers + memory plane) |
| Parallel replay scaling | — | wall-clock of full replay at 1/4/8 workers |
| Per-instruction cost | no committed-baseline regression >10% (iai, nightly lane) | callgrind instruction counts per kind |

Measurement methodology: throughput = trace rows / wall-clock of the backend call, in a dedicated bench binary that does nothing but trace (guest build and `JoltProgram` construction excluded). AOT compilation is cached at backend setup and reused across iterations, so steady-state Criterion timing excludes it by construction; compile time is reported as its own tracked number and must be amortizable across re-traces, which the streaming use case guarantees. Peak RSS is read from `/proc/self/status` `VmHWM` in a subprocess-per-measurement harness.

Why these numbers are conservative-plausible: record mode is bounded by row-buffer bandwidth (~96 B/row templates+patches; 50 MHz ≈ 5 GB/s, well within DDR5), and the fast pass by ZisK's demonstrated 3–4 x86 instructions per RISC-V instruction plus ~2 for chunk accounting.

`jolt-eval` plan:
- No existing objective is expected to move: the default backend is unchanged, so `prover_time_*` and all `PerformanceObjective::all()` entries are unaffected (any drift beyond noise is a bug in invariant-5 terms).
- Add, via `/new-objective`, standalone Criterion bench targets in jolt-eval — registered via `bench_objective!` + `./jolt-eval/sync_targets.sh` but deliberately **skipping** the optimizer enum/`all()` registration, exactly the `prover_time_*` precedent (`jolt-eval/README.md:51`): bench targets `trace_gen_fibonacci` and `trace_gen_sha2_chain`, each containing Criterion benchmark ids for the `reference`, `x86`, and `x86_fast` variants (mirroring how target `prover_time_fibonacci` contains id `prover_time_fibonacci_100`). Slice 0 lands the targets with the `reference` ids; the x86 ids join in slice 5.
- Per-instruction iai-callgrind benches live in `crates/jolt-tracer-x86/benches/` (workspace already ships `iai-callgrind = "0.16.1"`, root `Cargo.toml:386`, with the attribute-macro style precedent in `crates/jolt-prover-legacy/benches/iai.rs`); whole-guest throughput benches live only in jolt-eval (they must exist in slice 0, before the crate does).

## Design

### Architecture

The plug-and-play seam already exists; this spec extends it downward (chunked contract) and adds one implementation:

```text
                    crates/jolt-program (execution seam)
        ExecutionBackend ── TraceSource ── TraceRow / TraceOutput
        ChunkedExecutionBackend (NEW) ── ExecutionSummary (NEW)
                 ▲ impl                          ▲ impl
   ┌─────────────┴─────────────┐   ┌─────────────┴──────────────┐
   │ tracer::TracerBackend     │   │ jolt-tracer-x86::           │
   │ (reference interpreter,   │   │   X86TracerBackend (NEW)    │
   │  portable, default)       │   │ (AOT x86-64, linux-only)    │
   └───────────────────────────┘   └─────────────────────────────┘
                 ▲ consumed via TraceSource / rows
        crates/jolt-witness  TraceBackend ── RowSource::visit_chunks
                 ▲
        crates/jolt-prover   prove(…, W: JoltWitnessPlane, …)
```

Consumers are already trace-source-agnostic: the modular prover is generic over `JoltWitnessPlane` (`crates/jolt-prover/src/prover.rs:47-63`), and `jolt-witness` deliberately makes random access to trace rows inexpressible so that "a checkpointed, re-emulating trace source can implement every signature honestly" (`crates/jolt-witness/src/lib.rs:15-19`). `ChunkedExecutionBackend` is the missing producer-side contract for that design: `visit_chunks(range, …)` maps to "replay the checkpoints covering `range`", in parallel, instead of today's clone-cursor-and-drain-from-row-0 (`crates/jolt-witness/src/backend/trace/cycle.rs:95-139`). Prover-side adoption is a non-goal here, but the trait is shaped for it.

**New crate layout** (`crates/jolt-tracer-x86`, `[lints] workspace = true`):

```text
src/
├── lib.rs          X86TracerBackend, NativeBackend alias, portable fallback
├── compile/
│   ├── mod.rs      CompiledProgram { code: ExecutableBuffer, entry, pc_table, row_templates }
│   ├── emitter.rs  internal RowEmitter seam the per-kind emitters implement
│   ├── emit.rs     per-row-kind dynasm templates (one per final bytecode row kind)
│   ├── groups.rs   source-instruction grouping, advice-slot assignment, fused fast-mode templates
│   └── layout.rs   host register pinning, helper-call ABI, state offsets
├── state.rs        guest state plane: registers [u64; 128], flat memory, cursors
├── memory.rs       mmap'd memory plane, region checks, device-region routing
├── record.rs       row-template patching, chunk trace buffers
├── checkpoint.rs   X86Checkpoint, fast-pass logging, replay driver
├── helpers.rs      extern "C" helpers: JoltDevice I/O, inline advice, faults
└── bin/
    └── jolt_emu_x86.rs  ACT4 runner: HTIF tohost halt, signature dump
benches/            per-instruction iai-callgrind
tests/              differential per-instruction tests, guest equivalence, chunk sweeps
```

**Compilation model.** The transpiler consumes two inputs, both derived from the same `JoltProgram`: **(a)** `expanded_bytecode` (`Vec<JoltInstructionRow>`, already inline- and virtual-expanded — `crates/jolt-program/src/execution/trace.rs:20`), which supplies every row's static skeleton and PC mapping, so the backend never re-implements *expansion*, decompression, or the `rd = x0` rewrite; and **(b)** the decoded *source* instructions, which `JoltInstructionRow` erases (it carries no source kind or inline key) but which are required to select per-group advice computations (DIV-family quotient/remainder, SC success bit), LR/SC reservation side effects, and inline advice builders keyed by `(opcode, funct3, funct7)` (`tracer/src/instruction/inline.rs:266`). Source rows are recovered by re-running the shared ELF decoder (`jolt_program::image::decode_elf`) over `program.elf_bytes()` and keying by address — or, equivalently, by extending `JoltProgram` to retain the image's source rows (additive; implementer's choice). Compilation happens once per program at backend setup and is cached; all subsequent traces and chunk replays reuse the artifact (this is what makes repeated re-tracing cheap). Per expanded row the emitter produces straight-line x86 for the row's semantics; per *source-instruction group* (delimited by `is_first_in_sequence` / `virtual_sequence_remaining`) it emits the group's advice computation ahead of its `VirtualAdvice` rows. Two code bodies are emitted from the same templates: **fast** (no row stores; maintains the checkpoint log and chunk counter) and **record** (additionally memcpy's the row template and patches dynamic fields). Emission is isolated behind a small internal `RowEmitter` seam (`compile/emitter.rs`; the dynasm templates are its primary implementor), so alternate emitters — notably the copy-and-patch stencil route recorded in Alternatives §11 — can be A/B-tested per row kind without touching the harness, tests, or benches, and can coexist per kind if a hybrid ever wins. One implementation note: `SB/SH/LB/LH/LW/SW` are source-only kinds that expand to `LD`/`SD` sequences, so the interpreter's byte/halfword store witness paths (`mmu.rs:548-627`) are unreachable from final bytecode and need no x86 equivalents.

**State plane.** All 128 guest registers (32 architectural + 96 virtual, `common/src/constants.rs:2-5`) live in a host-memory array; a handful of host registers are pinned (guest-memory base, register-array base, trace cursor, cycle counter — the SP1 convention). Guest RAM is a single `mmap` of `MemoryLayout::get_total_memory_size()` (`common/src/jolt_device.rs:466-468`), addressed as `guest_addr - 0x80000000` with explicit region checks mirroring `assert_effective_address` (`tracer/src/emulator/mmu.rs:139-213`). Untouched pages cost no physical memory, so for the dense access patterns of realistic guests the flat plane is smaller than the interpreter's per-doubleword `HashMap` (~4 KiB per touched page vs. tens of bytes per touched doubleword — a sparse adversarial guest can invert this, which is why the memory gate is measured on the named guests, not assumed). Addresses below `RAM_START_ADDRESS` (JoltDevice I/O) route to `extern "C"` Rust helpers, as do inline-advice computation and fault paths.

**Control flow.** Direct branches and jumps compile to direct x86 jumps. Indirect jumps (`JALR`) go through a bounds-checked table indexed by `(guest_pc - text_base) / 2` — halfword-granular, because compressed instructions make 2-byte-aligned targets legal — with one entry per halfword pointing at the code offset of the group whose `is_first_in_sequence` row has that address; all other entries route to the runtime-fault helper (invariant 7). Termination reproduces the interpreter's PC-stall heuristic (`tracer/src/lib.rs:216-226`) with three specified cases: (a) a direct jump or *taken* branch whose target equals the row's own source PC executes and records its group once, then exits; (b) the indirect-dispatch path adds one `target == current source PC` compare that likewise exits after recording; (c) detection must look through the `rd = x0` rewrite — the guest's `j .` idiom is a `JAL x0` (`jolt-platform/src/exit.rs:13`) which expansion rewrites to target a virtual register, so the self-loop test is on source PC, not on the literal row operands. The terminal group's rows are included in `trace_len` and the row stream, exactly as the interpreter emits them once before stalling.

**Checkpoints and replay.** `X86Checkpoint` = the full register file + pc + cycle count + advice-tape/device cursors + LR/SC reservation, snapshotted at a *group boundary*, plus `skip_rows: u16` and a fast-pass log slice. Because chunk marks can fall mid-group (chunk size 1 lands inside SC.D's ~15-row expansion) while compiled code has entry points only at group starts, the fast pass snapshots at the last group boundary at or before each mark — group row counts are static, so the boundary check is a compare against a per-group constant (~2 instructions) — and `replay_chunk` re-enters at that boundary, discards the first `skip_rows` rows, then emits exactly `chunk_size` rows, stopping mid-group at the far end by truncating the buffer. The log makes replay **memoryless** (no per-worker memory image; chunks replay in parallel — the SP1 `TraceChunk`/ZisK `EmuTrace` design). Its records are: **(a)** per RAM-touching row, the pre-access value encoded exactly as the reference records it — the normative encoding is `trace_load`/`trace_store` (`tracer/src/emulator/mmu.rs:517-663`): 8 bytes LE read from the 4-byte-aligned floor of the effective address for reads and from the raw effective address for `SD` writes; device-region accesses produce real Read/Write rows in the reference and are logged identically; **(b)** per group with advice, the *computed advice values* (one u64 per `VirtualAdvice` slot) — necessary because inline advice builders read arbitrary guest memory (e.g. secp256k1 reads operands at `cpu.x[rs1]`, `jolt-inlines/secp256k1/src/sequence_builder.rs:31-36`) that the per-row log does not cover, so replay patches advice from the log instead of recomputing it. During replay, loads and store-pre-values are answered from the log, post-values are computed natively, `VirtualHostIO` side effects (advice writes, prints, cycle markers) are suppressed, and device writes patch rows only — device state was fully captured in the fast pass. Replay workers read the captured advice tape through the checkpoint's cursor (the tape is append-only, so cursors into the final tape are sound). The tracer's existing first-touch-delta `Checkpoint` (`tracer/src/emulator/memory.rs:42-63`) is the address-keyed equivalent and backs `TracerBackend`'s impl of the same trait — which is exactly why `Checkpoint` is an opaque associated type.

**Fused fast-mode groups.** In fast mode, a source-instruction group need not execute its expansion row by row: the emitter may compile the group as its *architectural semantics plus an observability epilogue*, mirroring the reference interpreter's own non-tracing path (`instr.execute` vs `instr.trace`, `tracer/src/emulator/cpu.rs:526-531`) — e.g. the DIV group becomes a native `idiv` with the two RISC-V edge-case guards (x86 `idiv` faults on divisor-zero and `MIN / -1`, both of which RISC-V defines results for) instead of a translation of its multi-row virtual sequence. What keeps this honest is naming exactly which state is observable across group boundaries:
- **Virtual temporaries (registers 40–47) are not fully dead.** Every rd-write row records the destination's *pre*-value (`RegisterWrite`, `crates/jolt-program/src/execution/trace.rs:167-171`), which feeds the committed `RdInc` polynomial — and fast-pass checkpoints seed bit-exact replay. A fused template must therefore end by storing the recipe's *final* temp values (deterministic functions of the group's inputs, 1–2 instructions each). The interpreter's `execute` path skips this, and its own harness accordingly compares only the 32 architectural registers between the two paths (`tracer/src/instruction/test.rs:103-194`); the fast pass does not get that luxury because its register snapshots feed replay.
- Persistent virtual state (registers 32–39: CSR mirrors, LR/SC reservation) is architectural and always maintained.
- Advice values are appended to the advice log exactly as the row-by-row path would compute them (`idiv` conveniently yields both quotient and remainder — the DIV group's two advice slots).
- Memory-op groups append their access-log entries in row order; the cycle counter advances by the group's static row count, so chunk accounting is unchanged.

Record mode is never fused — every intermediate register value is observable in its rows — though the computation *between* row stores may be. Fusion couples the fast pass to each expansion recipe's internals (a recipe change in `jolt-program/src/expand` must touch the fused template); this is acceptable because any divergence, including a wrong temp final, surfaces deterministically as a checkpoint-register mismatch caught by `tracer_chunk_composition`. Implementation is phased accordingly: the fast pass comes up row-by-row (correct by construction), and fusion lands kind-by-kind as slice-5 performance work — memory-op and M-extension groups first (loads/stores are the most frequent expansions and the largest aggregate win) — each fusion landing only behind a green chunk-composition and equivalence sweep.

**Safety.** dynasm-rs assembles into an anonymous mapping finalized RW→RX (`ExecutableBuffer`); no RWX window. Generated code addresses only the state plane (pinned base + bounded offsets, explicit checks) and the current chunk's buffers. All `unsafe` (mmap, transmute-to-fn, raw cursor stores) is concentrated in `memory.rs`/`checkpoint.rs`/`compile/mod.rs` with `SAFETY:` comments; the workspace's `undocumented_unsafe_blocks`/`unwrap_used` denies apply via `[lints] workspace = true`. AC11 tests the mapping permissions and fault behavior.

**ACT4 runner.** `jolt-emu` is a bin of the `tracer` crate (`tracer/Cargo.toml:20-22`), and `jolt-tracer-x86` depends on `tracer` (for the inline registry and `InlineAdviceContext`), so backend selection inside `jolt-emu` would be a dependency cycle. Instead the crate ships `jolt-emu-x86`: it builds a `JoltProgram` from an ACT4 test ELF, executes via `X86TracerBackend`, implements the HTIF halt convention (a store to the `tohost` symbol address terminates execution, mirroring `Emulator::run_test`, `tracer/src/emulator/mod.rs:143-192`), and dumps the signature region from the memory plane using ELF symbol addresses. The arch-test runner (`tests/arch-tests/run.sh`) gains an emulator-binary override; tests whose instruction stream the decode/expand pipeline rejects are skipped and listed (AC4).

**Backend selection.** `pub type NativeBackend` (cfg-selected as described in Goal) is the entire public selection surface; nothing switches by default. Call sites opt in (`Program::trace_with_backend`, `trace_{fn}_with_backend`, benches) — they are already generic over `B: ExecutionBackend`, so no new plumbing is needed.

### Alternatives Considered

1. **Optimize the interpreter in place** (decode cache, threaded dispatch, flat memory, expansion caching). Rejected as the primary strategy: the industry data is unambiguous that interpreters plateau around 150 MHz (OpenVM's heavily optimized interpreter) while AOT reaches 1.5 GHz, and Jolt's trace-recording obligations make interpreter dispatch overhead proportionally worse. Also rejected tactically: the interpreter is the correctness oracle, and churning it undermines the differential-testing story. Individual interpreter wins remain possible follow-ups; nothing here precludes them.
2. **Out-of-process AOT assembly emulator (ZisK's design).** Rejected: a separate build product and shm/semaphore IPC protocol for modest gains over an in-process transpiler; much harder to unit-test per instruction. SP1's in-process `sp1-jit` demonstrates the same technique without the process boundary. Revisit only if the in-process ceiling is hit.
3. **Hot-path JIT with interpreter tiering (rv8's design).** Rejected: profiling tiers exist to handle unknown/dynamic code, but a `JoltProgram` is fully static and fully expanded before execution begins. One-shot whole-program AOT is simpler and strictly better here.
4. **Cranelift as the codegen backend.** Rejected: Cranelift buys an IR, register allocation, and multi-target portability — none of which this workload needs (fixed template per row kind, fixed register plane, one target). dynasm-rs is what SP1 uses in production for the identical problem, and it also supports aarch64 for the future ARM backend. iced-x86 (encoder-only) and a handwritten encoder were rejected as more code for no benefit over dynasm's assembler macros.
5. **First-touch memory-delta checkpoints instead of access-value logs** (the tracer's existing `Checkpoint` design). Not chosen for the x86 fast pass: dedup tracking costs a map probe per access in the hot loop, vs. an append per RAM access; and the log natively supplies store pre-values which `TraceRow` needs anyway. The trait's opaque `Checkpoint` keeps both: `TracerBackend` uses its existing deltas, `X86TracerBackend` uses logs. Flip condition (recorded): if log volume on RAM-heavy guests dominates replay bandwidth, switch the x86 backend to address-keyed deltas.
6. **Group-aligned (variable-length) chunks instead of exact `chunk_size` chunks.** Rejected: it would push variable-chunk handling into every consumer (`jolt-witness` streams fixed `1 << 12`-row bundles) and forfeit the chunk-size-1 stress test. The snapshot-at-boundary + `skip_rows` scheme keeps exact chunk semantics for the cost of replaying at most one partial group per chunk.
7. **Emit legacy `Cycle` rows for `JoltCpuProver` integration.** Rejected: `Cycle` is a 96-byte instruction-keyed enum whose layout exists for the legacy monolith that is actively being decomposed; `TraceRow` is the modular contract, and the byte-diff harness already proves the two pipelines equivalent at the proof level.
8. **A fresh advice-callback design instead of adopting `InlineAdviceContext`.** Rejected: `specs/inline-expansion-grammar.md` already specifies that seam with binding rules; two live specs redesigning `build_advice` independently would fork the contract. This spec adds the second implementor and defers the contract's shape to that spec.
9. **Offline AOT to a shared object (compile at guest-build time via build.rs/LLVM).** Rejected: programs arrive as ELF bytes at runtime in several flows (SDK, fixtures, eval harnesses), and runtime AOT with dynasm is fast enough to amortize; an extra offline toolchain step complicates every consumer for no measured need.
10. **Hybrid per-instruction fallback to the interpreter for unsupported rows.** Rejected: bridging two register/memory state layouts mid-trace is a correctness hazard precisely where scrutiny is weakest. Fail-fast at compile time (invariant 7) plus exhaustive per-kind coverage (macro-enforced) is safer and forces coverage honesty.
11. **Copy-and-patch stencils as the emitter** (the CPython 3.13+ `Tools/jit` model: per-row-kind C stencils compiled by a *pinned* clang at build time under `preserve_none` + `musttail`, relocations extracted as patchable holes, runtime emission = memcpy + hole-patch). This captures the "inherit compiler optimizations" benefit that OpenVM's portable-native-recompilation demonstrated, while preserving the constraint that ruled out runtime C emission: no toolchain on the proving machine — clang runs only in a CI stencil-regeneration lane, and the generated `(bytes, holes)` tables are committed with embedded disassembly, so `cargo build` stays toolchain-free. Not chosen as the primary emitter: dynasm has the in-domain production precedent (SP1's `sp1-jit`) and zero scaffolding cost, while stencils require an extraction tool, `extern "C"` entry/exit trampolines, a pinned-LLVM lane (CPython re-pins each release), and compiler-generated bytes are harder to hand-audit for invariant 6. Expected performance is a wash — record mode is store-bandwidth-bound and hot fast-path kinds compile to the same 3–4 instructions either way — so the slice-2 spike measures it rather than argues it. Flip conditions, recorded: (a) **the aarch64 backend moves from non-goal to goal** — the decisive trigger, since C stencils recompile with `--target=aarch64` plus a small relocation patcher (CPython ships this across seven triples including Apple Silicon) versus re-hand-writing every dynasm template in A64 assembly; for the same reason, if stencils are adopted the C sources become the single source of truth *before* the x86 template set is complete, with hot kinds optionally hand-overridden in dynasm behind the `RowEmitter` seam. If adopted, stencil granularity is the source-instruction *group*, authored as semantics-plus-observability-epilogue per the Fused fast-mode groups design — program-independent (expansion shapes are static per source kind) while giving clang cross-row scope, so the fused `idiv` form falls out of the compiler automatically; (b) the spike shows dynasm losing >25% on hot kinds for reasons not fixable in the templates; (c) dynasm-rs maintenance becomes disqualifying (healthy as of 2026-07: v5.1.0 current, active, no known soundness issues — but single-maintainer). Endgame note: once Rust's `explicit_tail_calls` plus a register-freeing tail calling convention settle (`become` + `extern "rust-preserve-none"` work together on nightly today, x86-64 and aarch64; owners target `become` stabilization in 2027, the calling-convention half is unsequenced — track rust-project-goals#634), the stencil sources can be ported from C to Rust compiled by a pinned nightly in the same regen lane. The extraction pipeline is source-language-agnostic, so that port replaces only the stencil bodies — and eliminates the C-side mirror of `TraceRow`, letting stencils share types and helpers with the interpreter's `exec()` reference semantics.

## Documentation

- Update `book/src/how/architecture/emulation.md`: it currently documents only the riscv-rust-derived interpreter; add the backend seam and the two-pass chunked model.
- New page `book/src/how/architecture/execution-backends.md`: the `ExecutionBackend`/`ChunkedExecutionBackend` contracts, backend selection (`NativeBackend`), checkpoint/replay semantics, and guidance for adding a backend (ARM64 as the worked example).
- `crates/jolt-tracer-x86/README.md`: platform support matrix, safety model, how to run the differential tests and benches.
- Update the crate list in `CLAUDE.md` and the benchmark section if the trace-gen bench commands belong there.

## Execution

Slices in dependency order; every slice ends with: `cargo fmt -q`, `cargo clippy --all --features host -q --all-targets -- -D warnings` (and `host,zk`), `cargo nextest run --cargo-quiet` for touched packages, plus the slice's own gate.

0. **Slice 0 — measure before optimizing.** Backend-generic bench harness in jolt-eval: `trace_gen_fibonacci` and `trace_gen_sha2_chain` targets with `reference` Criterion ids, VmHWM subprocess harness, throughput reporting (rows/s). Record reference baselines for the six profile guests in the PR. Gate: `cargo bench -p jolt-eval --bench trace_gen_fibonacci` (and `trace_gen_sha2_chain`) run; baseline table committed (AC1).
1. **Slice 1 — seam.** `ChunkedExecutionBackend` + `ExecutionSummary` + advice-tape input/output plumbing in `jolt-program`; `TracerBackend` impl over the existing `CheckpointingTracer` (stops discarding the populated tape); `InlineAdviceContext` refactor in tracer + `jolt-inlines/sdk` per `specs/inline-expansion-grammar.md`; chunk-composition and advice round-trip tests for the reference backend. Gate: muldiv e2e both modes; tracer + jolt-inlines suites green; chunk sweep green on reference.
2. **Slice 2 — x86 bring-up (fast mode, base ISA).** Crate skeleton with `NativeBackend` alias and cfg gating, memory/state planes, helper ABI, jump table, chunk counter, the `RowEmitter` seam, and dynasm templates for the base RV64 ALU/branch/load/store row kinds; per-instruction differential tests and iai benches land *with each kind*; unsupported kinds fail fast. Includes a time-boxed **emitter spike**: implement ~5 representative kinds (ADD, BEQ, LD, SD, the DIV group) as copy-and-patch stencils behind the same `RowEmitter` seam and compare iai instruction counts and fast-pass wall-clock against the dynasm templates; record the outcome (with disassembly for any >25% delta) in the PR as input to Alternative 11's flip conditions. The spike is informative, not a gate. Gate: fibonacci guest runs to completion in fast mode with correct `JoltDevice` outputs; differential tests green for implemented kinds; `cargo check -p jolt-tracer-x86 --target aarch64-unknown-linux-gnu` passes (AC12).
3. **Slice 3 — full coverage + record mode.** Remaining row kinds (M/A, virtual kinds, ECALL sequence, CSR-mirror kinds, INLINE + advice helpers); source-row recovery for group/advice codegen; row-template record mode; `jolt-emu-x86` runner with HTIF halt + signature dump and the arch-test runner override; ACT4 green (AC4); `tracer_backend_equivalence` via `/new-invariant` green over the full corpus (AC5); proof byte-equality test green (AC7); safety tests (AC11). Gate: AC3, AC4, AC5, AC7, AC11.
4. **Slice 4 — chunked execution.** Fast-pass access-value + advice-value logging, `X86Checkpoint` with `skip_rows`, parallel replay, `tracer_chunk_composition` via `/new-invariant`, memory gates measured. Gate: AC2, AC6, AC10.
5. **Slice 5 — performance to gate.** Fused fast-mode groups (memory-op and M-extension kinds first, each fusion landing only behind green `tracer_chunk_composition` and `tracer_backend_equivalence` runs); pinned-register tuning, template specialization, group-boundary chunk checks, compile-time reporting; `x86`/`x86_fast` Criterion ids added to the jolt-eval bench targets; final benchmark table (throughput, memory, scaling, compile time) in the PR; docs. Gate: AC8, AC9; iai baselines committed (AC3 bench half).

## References

- eth_proofs tweet (2025-06-23): ZisK 1.5 GHz AOT trace generation, sequential-first-pass framing — https://x.com/eth_proofs/status/1937083157519458687
- ZisK: AOT x86 ASM emulator, shared-memory minimal traces, 2^18-step chunks — https://github.com/0xPolygonHermez/zisk (`core/src/zisk_definitions.rs`, `common/src/emu_minimal_trace.rs`)
- SP1 `sp1-jit`: in-process whole-program AOT via dynasm-rs; `TraceChunkHeader` (register checkpoint) + `MemValue` read log — https://github.com/succinctlabs/sp1 (`crates/core/jit`); pre-JIT interpreter baseline ~9.35 MHz — https://github.com/succinctlabs/riscv-emulator-challenge
- OpenVM distributed proving (metered → pure → preflight three-pass; 150 MHz interpreter) — https://docs.openvm.dev/specs/architecture/distributed-proving/
- RISC Zero continuations (segment = sparse memory image + I/O records, re-executed preflight) — https://risczero.com/blog/continuations
- rv8: hybrid interpreter/JIT, 2.6× native vs QEMU's 4.6× (CARRV 2017) — https://carrv.github.io/2017/papers/clark-rv8-carrv2017.pdf
- QEMU TCG design (translation blocks, direct chaining) — https://www.qemu.org/docs/master/devel/tcg.html
- dynasm-rs — https://github.com/CensoredUsername/dynasm-rs; iai-callgrind — https://github.com/iai-callgrind/iai-callgrind
- Copy-and-patch compilation (Xu & Kjolstad, OOPSLA 2021) — http://fredrikbk.com/copy-and-patch.html; CPython JIT stencil pipeline — https://github.com/python/cpython/blob/main/Tools/jit/README.md
- OpenVM "portable native recompilation" (`rvr`: compiler-emitted native executor, C-at-runtime variant) — https://github.com/openvm-org/openvm/tree/develop-v2.1.0/crates/rvr
- Rust tail-calls project goal (tracks the pure-Rust stencil endgame in Alternative 11) — https://github.com/rust-lang/rust-project-goals/issues/634
- In-repo anchors: `ExecutionBackend` seam (`crates/jolt-program/src/execution/backend.rs:3`), `RowSource` contract and design note (`crates/jolt-witness/src/consumer.rs:103`, `crates/jolt-witness/src/lib.rs:15-19`), legacy double-execution (`crates/jolt-prover-legacy/src/zkvm/prover.rs:749-767`), interpreter hot loop (`tracer/src/emulator/cpu.rs:480-541`), checkpoint machinery (`tracer/src/lib.rs:377-588`), RAM witness encoding (`tracer/src/emulator/mmu.rs:517-663`), static expansion pipeline (`crates/jolt-program/src/expand/mod.rs:87-160`), per-instruction test harness (`tracer/src/instruction/test.rs:103-194`), byte-diff methodology (`crates/jolt-prover/tests/byte_diff.rs`), ACT4 harness (`tests/arch-tests/`, root `Makefile`)
- Sibling specs: `specs/inline-expansion-grammar.md` (`InlineAdviceContext` contract), `specs/clean-slate-prover.md` (backend seam precedent), `specs/witness-redesign.md` (RowSource design), `specs/proof-trace-row-layout.md`, `specs/act4-tests.md`
- [`jolt-eval` framework](../jolt-eval/README.md)
