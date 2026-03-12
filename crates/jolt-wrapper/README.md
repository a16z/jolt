# jolt-wrapper

Verifier wrapping infrastructure for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate provides symbolic execution, AST capture, and pluggable codegen backends for transpiling the Jolt verifier into external proof systems (gnark/Groth16, Spartan+HyperKZG, etc.).

The pipeline works in four stages:

1. **Symbolic execution** -- `SymbolicField` implements the `Field` trait, recording operations into a thread-local arena instead of performing real arithmetic.
2. **AST capture** -- The arena graph is collected into an `AstBundle` (constraints + inputs), a backend-agnostic intermediate representation.
3. **Codegen** -- An `AstEmitter` backend walks the bundle and emits target code.
4. **Transcript tunneling** -- `PoseidonSymbolicTranscript` implements `Transcript` for symbolic fields.

## Public API

### Core Types

- **`SymbolicField`** -- A `Field` implementation that records operations as AST nodes. Supports constant folding.
- **`ArenaSession`** -- RAII guard for the thread-local arena.
- **`AstBundle`** -- Serializable constraint bundle: nodes, inputs, and assertions.
- **`VarAllocator`** -- Builder for constructing an `AstBundle` from symbolic execution results.

### Codegen

- **`AstEmitter`** -- Trait for pluggable codegen backends.
- **`GnarkAstEmitter`** -- Gnark Go code emitter (arena-based AST) with CSE and `api.AssertIsEqual` support.
- **`GnarkEmitter`** -- Gnark Go code emitter for `jolt-ir` expressions (`CircuitEmitter` impl).
- **`MemoizedCodeGen`** -- Reference-counting CSE pass for efficient code generation.
- **`generate_go_file`** -- Convenience function producing a complete Go source file from a bundle.
- **`GoFileConfig`** -- Configuration for `generate_go_file`.
- **`sanitize_go_name`** -- Converts Rust identifiers to valid Go exported names.

### Modules

- **`spartan`** -- Spartan AST emitter backend for R1CS constraint generation.
- **`transcript`** -- `PoseidonSymbolicTranscript` for symbolic Fiat-Shamir.
- **`scalar_ops`** -- BN254 scalar field arithmetic over `[u64; 4]` limb representation.
- **`tunneling`** -- Thread-local channel for passing symbolic values through `from_u128` boundaries.

## Dependency Position

```
jolt-field ─┐
jolt-ir    ─┼─> jolt-wrapper
jolt-transcript ─┘
```

## Feature Flags

This crate has no feature flags.

## License

MIT
