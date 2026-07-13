# AI Agent Workflow

Jolt ships an agent skill for Claude Code, Cursor, Codex, and other coding agents:

```bash
npx skills add a16z/jolt
```

Use the skill when you want an agent to wrap an existing Rust computation in a Jolt proof. The best requests name the function to prove, the expected inputs, whether any inputs should be private, and whether the result must run in `no_std` or may use `std`.

## What to Ask For

Good agent requests are specific about the computation and the proof boundary:

```text
Make `crate::matching::score_order(order, book)` provable with Jolt.
The verifier should know `order.id` and the final score, but not the private book state.
Use std mode if it avoids unnecessary signature rewrites.
```

If the function is not pure and deterministic, ask the agent to extract a pure function first. Guest code cannot depend on I/O, networking, wall-clock time, host randomness, or process state. Pass those values in as explicit inputs instead.

## Scaffold First

Always start from the CLI scaffold:

```bash
jolt new proof-demo
```

This creates a host crate and a nested `guest` crate. The generated `guest/src/main.rs` contains the `#![no_main]` binary stub needed to produce a RISC-V ELF. Put provable functions in `guest/src/lib.rs`; do not move the binary stub into the library.

For full zero-knowledge mode, scaffold with:

```bash
jolt new proof-demo --zk
```

This configures the host for BlindFold verifier preprocessing and imports `PrivateInput`.

## Signature Adaptation

Jolt guests have heap allocation, so `Vec`, `String`, and other allocation-backed data structures can be used inside the function body. The main constraint is the parameter and return boundary.

For `no_std` guests:

- Prefer fixed-width integers such as `u32`, `u64`, and `u128`.
- Replace `usize` with a fixed-width integer so host and guest agree on layout. The guest target is 64-bit RISC-V (RV64IMAC).
- Replace `Vec<T>` parameters with `[T; N]` plus a length, or switch to std mode.
- Keep array parameters at 32 elements or fewer when using serde_core.
- Rewrite floating-point logic as fixed-point integer arithmetic.

For std-mode guests, enable `guest-std` in `guest/Cargo.toml` and keep ordinary serde-compatible inputs when that makes the proof boundary clearer.

## Host Pipeline

For ordinary proofs, the generated host pipeline is:

```rust
let target_dir = "/tmp/jolt-guest-targets";
let mut program = guest::compile_fib(target_dir);

let shared_preprocessing = guest::preprocess_shared_fib(&mut program).unwrap();
let prover_preprocessing = guest::preprocess_prover_fib(shared_preprocessing.clone());
let verifier_setup = prover_preprocessing.generators.to_verifier_setup();
let verifier_preprocessing =
    guest::preprocess_verifier_fib(shared_preprocessing, verifier_setup, None);

let prove_fib = guest::build_prover_fib(program, prover_preprocessing);
let verify_fib = guest::build_verifier_fib(verifier_preprocessing);
```

The last verifier-preprocessing argument is `None` unless the host is proving with the `zk` feature. In ZK mode, pass `Some(prover_preprocessing.blindfold_setup())`.

## Public, Advice, and Private Inputs

Use public inputs when the verifier should see the value. Use advice when the prover needs witness data that is not part of the verifier's API.

- `jolt::UntrustedAdvice<T>` is prover-only but not cryptographically hidden by itself.
- `jolt::PrivateInput<T>` marks a prover-only value that should be hidden by BlindFold. Use `jolt new --zk` or enable the host `zk` feature.
- `TrustedAdvice<T>` is for data committed by a third party and requires a host commitment call.

When in doubt, start with public inputs until the proof runs, then switch specific witness values to advice or private inputs.

## Agent Checklist

Before proving, the agent should verify:

- The target function and any helper APIs it imports are `pub` when they live outside the guest crate.
- The guest has no I/O, networking, floating point, host randomness, or implicit time dependency.
- The host calls `preprocess_shared_*(&mut program).unwrap()`.
- The host passes `None` or `Some(blindfold_setup)` to `preprocess_verifier_*`.
- The generated `guest/src/main.rs` binary stub still exists.
- `max_trace_length` is tight enough for performance but large enough for the analyzed cycle count.

## Fast Debug Loop

Use `analyze_*` before proving expensive programs:

```rust
guest::analyze_fib(50)
    .write_to_file("fib_50.txt".into())
    .unwrap();
```

This catches guest panics and gives cycle counts without running the full prover. Once the proof works, lower `max_trace_length` to the smallest power of two above the analyzed cycle count. Proving time and peak memory scale with this value.

For guest panics, set:

```bash
JOLT_BACKTRACE=1 cargo run --release
```

Use `JOLT_BACKTRACE=full` when function names are not enough and you also need register snapshots and cycle counts per frame.

