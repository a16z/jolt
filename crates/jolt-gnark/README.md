# jolt-gnark

Gnark Go code generation from `jolt-ir` expressions.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate provides `GnarkEmitter`, an implementation of `jolt-ir`'s `CircuitEmitter` trait that produces gnark `frontend.API` calls as Go code strings. Each arithmetic operation becomes a Go assignment with a CSE variable name, and the final result is the root wire.

## Public API

- **`GnarkEmitter`** — Gnark Go code emitter. Builder methods: `with_cse_prefix`, `with_opening_name`, `with_challenge_name`. Output methods: `finish`, `finish_with_assert_zero`, `lines`.
- **`sanitize_go_name`** — Converts a Rust identifier to a valid Go exported name (PascalCase with underscore separators).

## Usage

```rust
use jolt_ir::{ExprBuilder, CircuitEmitter};
use jolt_gnark::GnarkEmitter;

let b = ExprBuilder::new();
let h = b.opening(0);
let gamma = b.challenge(0);
let expr = b.build(gamma * (h * h - h));

let mut emitter = GnarkEmitter::new();
let root = expr.to_circuit(&mut emitter);

let code = emitter.finish_with_assert_zero(&root);
// Produces: api.Mul, api.Sub, api.AssertIsEqual calls
```

## Dependency Position

```
jolt-ir ─► jolt-gnark
```

## Feature Flags

This crate has no feature flags.

## License

MIT
