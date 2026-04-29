# jolt-compiler-v2

This crate is the Bolt-shaped compiler prototype for Jolt v2. `melior::ir::Module` is the IR source of truth; local Rust types only provide phase/role guardrails, generic schema validation, and small builders for the registered Bolt IRDL dialects. IRDL definitions live under `irdl/`, split by dialect.

Protocol-specific facts live under `src/protocols/`. The generic compiler layers understand Bolt dialect operations (`protocol`, `piop`, `commit`, `pcs`, `transcript`, `party`, `compute`, `cpu`) but do not know Jolt-only parameters such as `ram_k`, oracle families, or transcript ordering. The Rust path lowers through explicit per-party, `compute.*`, and `cpu.*` ops before Rust emission, so generated Rust is only the final target representation.

The commitment pipeline uses SSA values for commitment artifacts and transcript state from protocol/concrete IR through party, compute, and CPU IR. Metadata remains in attributes, but execution dataflow is carried by operands/results so MLIR verifies def-use structure before Rust emission.

Commitment compute and CPU lowerings build IR with `melior` operation builders rather than formatting full MLIR modules as strings. Textual MLIR remains only where the prototype intentionally phase-casts whole modules while preserving existing SSA bodies.

See `TESTING.md` for the compiler-v2 parity gates: IR/golden checks, generated prover/verifier self-parity, modular self-verify, transcript parity against jolt-core, and jolt-core proof acceptance.

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
