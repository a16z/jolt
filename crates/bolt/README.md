# bolt

This crate is the Bolt-shaped compiler prototype for Jolt v2. `melior::ir::Module` is the IR source of truth; local Rust types only provide phase/role guardrails, generic schema validation, and small builders for the registered Bolt IRDL dialects. IRDL definitions live under `irdl/`, split by dialect.

Protocol-specific facts live under `src/protocols/`. The generic compiler layers understand Bolt dialect operations (`protocol`, `piop`, `commit`, `pcs`, `transcript`, `party`, `compute`, `cpu`) but do not know Jolt-only parameters such as `ram_k`, oracle families, or transcript ordering. The Rust path lowers through explicit per-party, `compute.*`, and `cpu.*` ops before Rust emission, so generated Rust is only the final target representation.

The commitment pipeline uses SSA values for commitment artifacts and transcript state from protocol/concrete IR through party, compute, and CPU IR. Metadata remains in attributes, but execution dataflow is carried by operands/results so MLIR verifies def-use structure before Rust emission.

Sumcheck and PCS opening rails follow the same rule. `piop.sumcheck` models the protocol contract, including grouped-round schedules for uniskip-style first rounds. `pcs.opening_claim` and `pcs.opening_batch` keep opening obligations as typed SSA values with explicit ordering; any mutable opening accumulator is only a later CPU/Rust implementation detail.

The first CPU target is intentionally coarse. Protocol IR names sumcheck relations; a compute-to-compute kernel-resolution pass maps those relations to CPU kernel symbols. Only after that pass does `cpu.sumcheck_driver` carry `kernel = @...`. The IR still owns stage order, transcript labels, batching policy, proof slots, and PCS opening obligations.

Stage 1 CPU kernel extraction is typed in `emit::rust::stage1_cpu_program`. It does not infer Jolt protocol semantics from Rust code; it checks that prover-side `cpu.kernel`, `cpu.sumcheck_claim`, `cpu.sumcheck_batch`, `cpu.sumcheck_driver`, `cpu.sumcheck_eval`, and virtual opening ops form the explicit SSA contract that real CPU kernels will implement. Verifier-side Stage 1 uses `cpu.sumcheck_verify_claim` and `cpu.sumcheck_verify`, carrying the relation directly and rejecting kernel ops in verifier CPU IR.

Stage 1 Rust emission serializes that checked CPU contract into static plan constants. The generated prover still calls `jolt_kernels::stage1::execute_stage1_program`; `jolt-kernels` owns the temporary coarse CPU ABI until those kernels are progressively replaced by finer compute lowerings. The generated verifier is kernel-free and verifies Stage 1 with audit-scope modular crates such as `jolt-sumcheck` and `jolt-transcript`.

`Stage1ShapeKernelExecutor` remains the lightweight prover-side structural runner. It dispatches only the known Stage 1 kernel ABI symbols and returns proof/evaluation artifacts with the expected shape; the generated verifier consumes those artifacts through its own kernel-free verifier path.

`Stage1ProverKernelExecutor` is the real-arithmetic prover dispatch rail. The uniskip ABI builds the Jolt-core-shaped first-round polynomial from explicit prover evaluations or an R1CS-backed data source, and the remaining outer sumcheck ABI runs the generic degree-3 transcript/proof mechanics over the same typed evaluator boundary. The generated verifier replays the corresponding sumcheck verification directly through modular verifier APIs. The current coarse CPU data source consumes the modular `jolt-r1cs` witness layout; replacing its temporary evaluator ABI with a stricter Stage 1 data object is the next hardening slice.

Stage 2 follows the same boundary with more relations wired. The Jolt protocol module defines product-virtual uniskip, RAM read-write, product-virtual remainder, instruction lookup claim reduction, RAM RAF evaluation, and RAM output check as `piop`/`pcs` obligations. Prover lowering resolves those relations into coarse CPU kernels only after `compute` IR owns the SSA claim, point, transcript, and opening flow. Verifier lowering keeps the relation language intact and generated verifier Rust rejects `jolt-kernels`, so verifier code remains thin modular-crate glue.

The Stage 2 coarse kernels are deliberately below the dialect boundary. `jolt-kernels` now shares reusable dense-binding and split-equality state helpers for product/RAM sumcheck kernels, but those helpers are implementation details of the current CPU backend rather than protocol semantics. Future lowering can replace each coarse kernel with finer `compute`/`cpu` ops without changing the protocol-stage ordering or verifier contract.

`tests/fixtures/jolt_protocol_chain_commitment_stage1.yaml` tracks the current full-protocol chain from commitment into Stage 1. It sits next to the per-component MLIR/Rust fixtures so internal prover/verifier parity and later jolt-core parity can grow as one ordered phase chain rather than isolated tests.

Commitment compute and CPU lowerings build IR with `melior` operation builders rather than formatting full MLIR modules as strings. Textual MLIR remains only where the prototype intentionally phase-casts whole modules while preserving existing SSA bodies.

See `GOAL.md` for the end-to-end Jolt-on-Bolt target,
`JOLT_PROTOCOL_IMPLEMENTATION.md` for the stage-addition playbook, and
`TESTING.md` for the compiler-v2 parity gates. Each implemented protocol stage
must keep the same stage-local oracles green before it is treated as done: Bolt
prover output accepted by the Bolt verifier on real trace data, Bolt
prover/verifier transcript-state equality, Bolt artifacts accepted by the
jolt-core verifier when spliced into the matching proof prefix, Bolt
transcript/artifact parity against jolt-core through that stage, generated
verifier tamper rejection, and the stage prover within 20% of jolt-core. This
matrix is intentionally per-stage so commitment, Stage 1, Stage 2, Stage 3, and
later phases can be added without waiting for a full end-to-end prover before
finding semantic drift.

Generated Jolt Rust artifacts are organized as two role crates. Prover modules
target `crates/jolt-prover/src/stages/<stage>.rs` and may import coarse CPU
kernels from `jolt-kernels`; verifier modules target
`crates/jolt-verifier/src/stages/<stage>.rs` and must stay kernel-free, using
only audit-scope modular crates and local generated modules. The artifact rail
now assembles full generated crates, including manifests, stage registries, and
stage module graphs. Compiler tests check both temp standalone crates and the
checked-in workspace crates so `jolt-equivalence` and `jolt-bench` can import
the same generated prefix.

The checked-in role crates are generated artifacts, not hand-maintained code.
Regenerate them through the Rust artifact rail with
`JOLT_UPDATE_GOLDENS=1 cargo nextest run -p bolt generated_jolt_artifacts_have_uniform_crate_layout_and_import_rules --cargo-quiet`.
The same generator emits `prover.rs` and `verifier.rs` from role artifacts:
`jolt-verifier` owns `JoltProof` and verification, while `jolt-prover` may only
import verifier-owned proof types and never verifier stage internals.

## Local MLIR Toolchain

On macOS with Homebrew LLVM:

```bash
brew install llvm
export MLIR_SYS_220_PREFIX=/opt/homebrew/opt/llvm
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
export SDKROOT="$(xcrun --show-sdk-path)"
export BINDGEN_EXTRA_CLANG_ARGS="-isysroot $(xcrun --show-sdk-path)"
```

Do not set `MLIR_SYS_LINK_SHARED=1` with the Homebrew LLVM 22 bottle; it does not ship `libMLIR-C.dylib`, so `mlir-sys` needs its default static MLIR link path.
