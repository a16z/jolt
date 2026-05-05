# bolt

This crate is the Bolt-shaped compiler prototype for the full-field, non-zk
Jolt implementation. `melior::ir::Module` is the IR source of truth; Rust types
provide phase/role guardrails, schema validation, builders, analysis results,
and final Rust emission.

## Active Goal

The first Jolt-on-Bolt implementation is semantically complete enough that the
active work is verifier cleanup and hardening, not stage bring-up. See
`GOAL.md` for the long-haul target:

```text
make the generated jolt-verifier compact, human-readable, auditable,
security-hardened, and driven by explicit MLIR-derived plan data
```

`GENERIC_PROTOCOL_GOAL.md` describes the parallel cleanup track that makes Bolt
generic over protocol packages instead of Jolt-shaped. `JOLT_PROTOCOL_IMPLEMENTATION.md`
keeps the durable compiler-boundary rules. `TESTING.md` lists the LOC,
readability, equivalence, import, MLIR, and tamper gates for this cleanup track.
`CODE_QUALITY.md` captures the Rust compiler-project idioms Bolt should borrow
as the implementation is tightened and de-duplicated.

## Compiler Shape

Protocol-specific facts live under `src/protocols/`. Generic compiler layers
understand Bolt dialect operations but should not learn Jolt-only protocol
semantics except as ordinary typed attrs, SSA values, or typed plan data carried
by a protocol definition.

The intended lowering path is:

```text
protocol -> concrete -> party -> compute -> cpu -> Rust
```

The dialect split matters:

- `protocol`, `piop`, `poly`, `field`, `transcript`, `commit`, and `pcs` model
  protocol obligations.
- `party` projects prover/verifier visibility.
- `compute` represents role-specific executable structure while preserving
  semantic dataflow.
- `cpu` is the final MLIR target before Rust emission.
- Rust is generated output, not the place where protocol meaning should be
  inferred.

## Verifier Boundary

The generated verifier must remain audit-stable:

```text
no jolt-prover dependency
no jolt-kernels dependency
no jolt-core dependency
no jolt-equivalence dependency
no jolt-bench dependency
no tracer internals
```

Verifier CPU IR must stay kernel-free. Prover code may still call coarse
`jolt-kernels` CPU kernels while performance work continues, but those kernels
are below the dialect boundary and must not become verifier infrastructure.

The cleanup target is for generated verifier modules to become mostly
declarative plan data, with generic mechanics factored into named verifier
runtime modules.

## Generated Artifacts

Generated Jolt Rust artifacts are organized as two role crates:

```text
crates/jolt-prover
crates/jolt-verifier
```

The checked-in role crates are generated artifacts, not hand-maintained code.
Regenerate them through the Rust artifact rail with:

```bash
JOLT_UPDATE_GOLDENS=1 cargo nextest run -p bolt generated_jolt_artifacts_have_uniform_crate_layout_and_import_rules --cargo-quiet
```

The generator emits manifests, stage registries, `src/stages/*.rs`, and the
top-level `prover.rs`/`verifier.rs` APIs. `jolt-verifier` owns proof types and
verification. `jolt-prover` may construct verifier-owned proof types, but must
not import verifier stage internals.

## Local MLIR Toolchain

On macOS with Homebrew LLVM:

```bash
brew install llvm
export MLIR_SYS_220_PREFIX=/opt/homebrew/opt/llvm
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
export SDKROOT="$(xcrun --show-sdk-path)"
export BINDGEN_EXTRA_CLANG_ARGS="-isysroot $(xcrun --show-sdk-path)"
```

Do not set `MLIR_SYS_LINK_SHARED=1` with the Homebrew LLVM 22 bottle; it does
not ship `libMLIR-C.dylib`, so `mlir-sys` needs its default static MLIR link
path.
