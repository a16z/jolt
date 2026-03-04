---
name: jolt
description: |
  Jolt zkVM SDK expert. Activates for: jolt-sdk, #[jolt::provable], guest/host
  RISC-V programs, advice (TrustedAdvice/UntrustedAdvice/#[jolt::advice]),
  jolt-inlines-*, cycle tracking, proving/verifying pipelines.
---

# Jolt zkVM SDK

You are an expert on the Jolt zkVM SDK for building provable RISC-V programs.
Jolt proves correct execution of Rust programs compiled to RV64IMAC using
sumcheck-based protocols and the Dory polynomial commitment scheme.

## Project Structure

Every Jolt project has two crates: a **guest** library (compiled to RISC-V) and
a **host** binary (runs natively, drives proving/verification).

```
my-project/
  Cargo.toml              # host (workspace root)
  rust-toolchain.toml
  src/main.rs             # host code
  guest/
    Cargo.toml            # guest lib
    src/lib.rs            # provable functions
```

Both crates depend on `jolt-sdk`. The host uses `features = ["host"]`.
A `[patch.crates-io]` block for arkworks is **required**.

### rust-toolchain.toml

```toml
[toolchain]
channel = "1.88"
targets = ["riscv32imac-unknown-none-elf", "riscv64imac-unknown-none-elf"]
profile = "minimal"
components = ["cargo", "rustc", "clippy", "rustfmt"]
```

### Guest Cargo.toml

```toml
[package]
name = "guest"
version = "0.1.0"
edition = "2021"

[features]
guest = []

[dependencies]
jolt = { package = "jolt-sdk", git = "https://github.com/a16z/jolt" }
```

Variants:
- **std**: add `features = ["guest-std"]` to jolt dep, remove `no_std` cfg
- **Runtime advice**: add `compute_advice = []` to `[features]`
- **Inlines**: add e.g. `jolt-inlines-sha2 = { git = "https://github.com/a16z/jolt" }`
- **Threading/random**: add `"thread"` / `"random"` to jolt features (requires `guest-std`)

For `alloc` in `no_std`, use `extern crate alloc;` and import from `alloc::`.

### Host Cargo.toml

```toml
[package]
name = "my-project"
version = "0.1.0"
edition = "2021"

[dependencies]
guest = { path = "./guest" }
jolt-sdk = { git = "https://github.com/a16z/jolt", features = ["host"] }

[patch.crates-io]
ark-bn254 = { git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout" }
ark-ff = { git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout" }
ark-ec = { git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout" }
ark-serialize = { git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout" }
allocative = { git = "https://github.com/facebookexperimental/allocative", rev = "85b773d85d526d068ce94724ff7a7b81203fc95e" }

[profile.release]
lto = "fat"
```

For **inlines**: add e.g. `jolt-inlines-sha2 = { git = "...", features = ["host"] }`
to host deps and `[package.metadata.cargo-machete] ignored = ["jolt-inlines-sha2"]`
(host doesn't reference the crate directly but it's needed for compilation).

## Guest Programs

Mark functions with `#[jolt::provable]` to make them provable. The guest crate
is `no_std` by default (use `guest-std` feature for std support).

```rust
#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable]
fn fib(n: u32) -> u128 {
    let mut a: u128 = 0;
    let mut b: u128 = 1;
    for _ in 1..n {
        let sum = a + b;
        a = b;
        b = sum;
    }
    b
}
```

### Macro Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `heap_size` | `u64` | 134217728 (128 MB) | Guest heap allocation in bytes |
| `stack_size` | `u64` | 4096 (4 KB) | Guest stack size in bytes |
| `max_input_size` | `u64` | 4096 | Max serialized input bytes |
| `max_output_size` | `u64` | 4096 | Max serialized output bytes |
| `max_trusted_advice_size` | `u64` | 4096 | Max trusted advice bytes |
| `max_untrusted_advice_size` | `u64` | 4096 | Max untrusted advice bytes |
| `max_trace_length` | `u64` | 16777216 (16M) | Max RISC-V trace cycles |
| `guest_only` | flag | — | Skip host-architecture compilation |
| `wasm` | flag | — | Enable WASM bindings |
| `nightly` | flag | — | Use nightly features |

Inputs must be `serde::Serialize`, outputs must be `serde::Deserialize`.

### Standard Library Mode

Enable `guest-std` in the guest's jolt-sdk dependency and remove the
`#![cfg_attr(feature = "guest", no_std)]` directive.

| Feature | Enables |
|---------|---------|
| `guest-std` | Standard library support |
| `thread` | Threading / rayon (requires `guest-std`) |
| `random` | `getrandom` crate support (requires `guest-std`) |
| `debug` | Syscall logging to stdout |

### Multiple Functions Per Crate

A guest crate can contain multiple `#[jolt::provable]` functions. Each
generates its own set of host-side functions.

### `guest_only` Flag

Use when the guest code cannot compile on the host (e.g., inline RISC-V
assembly):

```rust
#[jolt::provable(guest_only)]
fn asm_example() -> u32 {
    let result: u32;
    unsafe { core::arch::asm!("li {}, 42", out(reg) result); }
    result
}
```

### Optimization Level

Control guest compilation via `JOLT_GUEST_OPT` env var: `0`, `1`, `2`, `3`
(default), `s`, `z`. Use `z` to optimize for code size.

## Host API

The `#[jolt::provable]` macro generates these functions for each guest function
`foo`:

| Function | Signature |
|----------|-----------|
| `compile_foo(target_dir)` | `&str → Program` |
| `preprocess_shared_foo(program)` | `&mut Program → JoltSharedPreprocessing` |
| `preprocess_prover_foo(shared)` | `JoltSharedPreprocessing → JoltProverPreprocessing` |
| `preprocess_verifier_foo(shared, setup)` | `(JoltSharedPreprocessing, VerifierSetup) → JoltVerifierPreprocessing` |
| `build_prover_foo(program, pp)` | `→ impl Fn(args...) → (output, proof, JoltDevice)` |
| `build_verifier_foo(pp)` | `→ impl Fn(pub_args..., output, panic, proof) → bool` |
| `analyze_foo(args...)` | `→ ProgramSummary` |
| `trace_foo_to_file(path, args...)` | Dumps execution trace |

### Preprocessing

Preprocessing is split into three steps and only depends on the program (not
inputs). It can be reused across multiple proofs.

1. `preprocess_shared_foo` — shared preprocessing consumed by both prover and
   verifier. **Clone it** since both steps consume it.
2. `preprocess_prover_foo` — prover-specific, includes commitment generators.
3. `preprocess_verifier_foo` — verifier-specific. Obtain `VerifierSetup` from
   `prover_pp.generators.to_verifier_setup()`.

### Prover / Verifier Closures

`build_prover_foo` returns a closure with the same input signature as the guest
function. Returns `(output, proof, program_io)` where `program_io.panic` is a
`bool`.

`build_verifier_foo` returns a closure taking public inputs (excludes advice
inputs), claimed output, `panic` flag, and the proof. Returns `bool`.

### Full Pipeline Example

```rust
use std::time::Instant;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_fib(target_dir);

    let shared = guest::preprocess_shared_fib(&mut program);
    let prover_pp = guest::preprocess_prover_fib(shared.clone());
    let verifier_setup = prover_pp.generators.to_verifier_setup();
    let verifier_pp = guest::preprocess_verifier_fib(shared, verifier_setup);

    let prove_fib = guest::build_prover_fib(program, prover_pp);
    let verify_fib = guest::build_verifier_fib(verifier_pp);

    // Analyze trace without proving
    let summary = guest::analyze_fib(50);
    summary.write_to_file("fib_50.txt".into()).unwrap();

    // Prove
    let now = Instant::now();
    let (output, proof, program_io) = prove_fib(50);
    println!("Proved in {:.2}s", now.elapsed().as_secs_f64());

    // Verify
    let is_valid = verify_fib(50, output, program_io.panic, proof);
    println!("output: {output}, valid: {is_valid}");
}
```

### Trusted Advice Host Flow

When the guest function has `TrustedAdvice<T>` parameters, commit before proving:

```rust
use jolt_sdk::{TrustedAdvice, UntrustedAdvice};

let (commitment, hint) = guest::commit_trusted_advice_merkle_tree(
    TrustedAdvice::new(leaf2),
    TrustedAdvice::new(leaf3),
    &prover_pp,
);

let prove = guest::build_prover_merkle_tree(program, prover_pp.clone());
let verify = guest::build_verifier_merkle_tree(verifier_pp);

// Prover: all inputs + commitment + hint
let (output, proof, program_io) = prove(
    leaf1,
    TrustedAdvice::new(leaf2),
    TrustedAdvice::new(leaf3),
    UntrustedAdvice::new(leaf4),
    commitment,
    hint,
);

// Verifier: only public inputs + commitment (no advice values)
let is_valid = verify(leaf1, output, program_io.panic, commitment, proof);
```

### Serialization

```rust
use jolt_sdk::serialize_and_print_size;
serialize_and_print_size("Proof", "/tmp/proof.bin", &proof).unwrap();
```

## Advice System

Two distinct mechanisms for auxiliary prover data:

| | Advice Inputs | Runtime Advice |
|---|---|---|
| **Source** | Host supplies before execution | Guest computes in first-pass |
| **Annotation** | `TrustedAdvice<T>` / `UntrustedAdvice<T>` params | `#[jolt::advice]` function |
| **Serialization** | `serde` | `AdviceTapeIO` trait |
| **Guest feature** | None | `compute_advice = []` in Cargo.toml |
| **Use case** | Host already knows the data | Checking cheaper than computing |

### Advice Inputs (host-provided, pre-execution)

Parameters on `#[jolt::provable]` wrapped in `TrustedAdvice<T>` or
`UntrustedAdvice<T>`. The guest accesses them via `.deref()`.

- **TrustedAdvice**: verifier gets a commitment (not data). Host calls
  `commit_trusted_advice_*` before proving.
- **UntrustedAdvice**: verifier sees neither data nor commitment. Guest must
  verify correctness.

```rust
use core::ops::Deref;

#[jolt::provable]
fn verify_path(
    leaf: &[u8],
    sibling: jolt::TrustedAdvice<[u8; 32]>,
    proof_node: jolt::UntrustedAdvice<[u8; 32]>,
) -> [u8; 32] {
    let s = sibling.deref();   // trusted: verifier has commitment
    let p = proof_node.deref(); // untrusted: guest must verify
    // ...
}
```

### Runtime Advice (guest-computed, two-pass)

`#[jolt::advice]` functions compute values in a first pass; the prover replays
them from an advice tape in the proving pass.

Two-pass model:
1. **First pass** (`compute_advice` feature active): advice function body
   executes, writes results to tape
2. **Second pass** (proving): reads precomputed results from tape

```rust
use jolt::AdviceTapeIO;

#[jolt::advice]
fn modinv(a: u64, m: u64) -> jolt::UntrustedAdvice<(u64, u64)> {
    let inv = extended_gcd(a, m);
    let quo = (a as u128 * inv as u128 / m as u128) as u64;
    (inv, quo)
}
```

Requirements:
- Return type must be `jolt::UntrustedAdvice<T>`
- `T` must implement `AdviceTapeIO`
- Arguments must be immutable

### Verifying Advice

Advice is untrusted — the guest must verify. Use `check_advice!` or
`check_advice_eq!` (prover-enforced assertions; proof fails if false):

```rust
#[jolt::provable]
fn modinv_verified(a: u64, m: u64) -> u64 {
    let adv = modinv(a, m);
    let (inv, quo) = *adv;
    let product = (a as u128) * (inv as u128) - (quo as u128) * (m as u128);
    jolt::check_advice!(product == 1u128 && inv < m);
    inv
}
```

`check_advice_eq!(a, b)` — optimized register-sized equality check. Both macros
accept optional trailing string for debug messages.

### `AdviceTapeIO` Trait

Built-in implementations:

| Type | Notes |
|------|-------|
| `u8`, `u16`, `u32`, `u64`, `usize` | Primitive unsigned |
| `i8`, `i16`, `i32`, `i64` | Signed |
| `[T; N]` where `T: Pod` | Fixed-size arrays |
| `(A, B)` ... `(A, B, C, D, E, F, G)` | Tuples up to 7 |
| `Vec<T>` where `T: Pod` | Requires `guest-std` |

Manual implementation:

```rust
struct Pair { x: u64, y: u64 }

impl jolt::AdviceTapeIO for Pair {
    fn write_to_advice_tape(&self) {
        self.x.write_to_advice_tape();
        self.y.write_to_advice_tape();
    }
    fn new_from_advice_tape() -> Self {
        Pair { x: u64::new_from_advice_tape(), y: u64::new_from_advice_tape() }
    }
}
```

### JoltPod: Automatic AdviceTapeIO via bytemuck

For `#[repr(C)]` plain-old-data structs:

```rust
use bytemuck_derive::{Pod, Zeroable};
use jolt::JoltPod;

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct Point { x: u32, y: u32 }
impl JoltPod for Point {}
```

Requires `bytemuck = "1.23"` and `bytemuck_derive = "1.10"` in guest deps.

### Full Advice Example: Factoring

Guest:

```rust
use jolt::AdviceTapeIO;

#[jolt::advice]
fn factor(n: u32) -> jolt::UntrustedAdvice<[u32; 2]> {
    for i in 2..=n {
        if n % i == 0 { return [i, n / i]; }
    }
    [1, n]
}

#[jolt::provable]
fn verify_composite(n: u32) {
    let adv = factor(n);
    let [a, b] = *adv;
    jolt::check_advice_eq!((a as u64) * (b as u64), n as u64);
    jolt::check_advice!(1 < a && a <= b && b < n);
}
```

Host:

```rust
pub fn main() {
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_verify_composite(target_dir);

    let shared = guest::preprocess_shared_verify_composite(&mut program);
    let prover_pp = guest::preprocess_prover_verify_composite(shared.clone());
    let verifier_setup = prover_pp.generators.to_verifier_setup();
    let verifier_pp =
        guest::preprocess_verifier_verify_composite(shared, verifier_setup);

    let prove = guest::build_prover_verify_composite(program, prover_pp);
    let verify = guest::build_verifier_verify_composite(verifier_pp);

    let n = 221u32;  // 13 * 17
    let (output, proof, program_io) = prove(n);
    let is_valid = verify(n, output, program_io.panic, proof);
    assert!(is_valid);
}
```

## Crypto Inlines

Optimized cryptographic primitives replacing guest-side computation with
efficient constraint-native implementations.

| Crate | Public API |
|-------|-----------|
| `jolt-inlines-sha2` | `Sha256::digest(input) -> [u8; 32]`, `new/update/finalize` |
| `jolt-inlines-keccak256` | `Keccak256::digest(input) -> [u8; 32]`, `new/update/finalize` |
| `jolt-inlines-blake3` | `Blake3::digest(input) -> [u8; 32]` (max 64 bytes), `blake3_keyed64()` |
| `jolt-inlines-blake2` | Low-level compression, constants (`IV`, `SIGMA`) |
| `jolt-inlines-secp256k1` | `ecdsa_verify(z, r, s, q)`, `Secp256k1Fq`/`Fr` field ops, `Secp256k1Point` affine ops |
| `jolt-inlines-bigint` | 256-bit integer multiplication |
| `jolt-inlines-grumpkin` | Base/scalar field division |

Dual-dependency setup: guest uses no features, host uses `features = ["host"]`.
Add `[package.metadata.cargo-machete] ignored = [...]` to host crate.

### SHA-256 Inline Example

```rust
#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable(heap_size = 32768, max_trace_length = 65536)]
fn sha2(input: &[u8]) -> [u8; 32] {
    jolt_inlines_sha2::Sha256::digest(input)
}
```

Host follows the same pipeline as above (compile → preprocess → prove → verify).

### secp256k1 ECDSA Verify Example

```rust
#![cfg_attr(feature = "guest", no_std)]
use jolt_inlines_secp256k1::{ecdsa_verify, Secp256k1Fq, Secp256k1Fr, Secp256k1Point};

#[jolt::provable(heap_size = 32768, max_trace_length = 1048576)]
fn verify_sig(
    msg_hash: [u8; 32], r: [u8; 32], s: [u8; 32],
    pk_x: [u8; 32], pk_y: [u8; 32],
) -> bool {
    let z = Secp256k1Fr::from_be_bytes(msg_hash);
    let r_scalar = Secp256k1Fr::from_be_bytes(r);
    let s_scalar = Secp256k1Fr::from_be_bytes(s);
    let q = Secp256k1Point {
        x: Secp256k1Fq::from_be_bytes(pk_x),
        y: Secp256k1Fq::from_be_bytes(pk_y),
    };
    ecdsa_verify(z, r_scalar, s_scalar, q).is_ok()
}
```

## Jolt CLI

Install: `cargo install --path .` from the Jolt repo root, or run via
`cargo run -p jolt -- <subcommand>`.

### `jolt new <NAME> [--wasm]`

Scaffold a new project (host + guest crates, toolchain, arkworks patches).
`--wasm` adds WASM-compatible files.

### `jolt build`

Build a guest program for Jolt zkVM. Wraps the toolchain with Jolt-specific
RUSTFLAGS (lower-atomic, panic=abort, medany code model, machine outliner
disabled).

```bash
jolt build -p guest                    # no_std (default)
jolt build -p guest --mode std         # std mode (auto-installs musl toolchain)
jolt build -p guest --mode std --fully # build musl toolchain from source if missing
```

| Flag | Default | Description |
|------|---------|-------------|
| `-p, --package` | required | Package name to build |
| `--mode` | `no-std` | `std` or `no-std` |
| `--backtrace` | `auto` | `auto`/`enable`/`disable` — controls symbol stripping |
| `--memory-origin` | `0x80000000` | Memory origin address |
| `--memory-size` | `128Mi` | Total memory (human-readable sizes) |
| `--stack-size` | `8Mi` | Stack size |
| `--heap-size` | `64Mi` | Heap size |
| `--fully` | false | Auto-build musl toolchain if missing |
| `--musl-lib-path` | env `RISCV_MUSL_PATH` | Path to musl lib |
| `--gcc-lib-path` | env `RISCV_GCC_PATH` | Path to gcc lib |
| `-- <CARGO_ARGS>` | — | Forwarded to `cargo build` |

Env vars: `JOLT_GUEST_OPT` (0/1/2/3/s/z, default 3), `JOLT_BACKTRACE=1`
(preserve symbols for panic backtraces).

### `jolt run <BINARY> [--jolt-emu <PATH>] [-- <EMU_ARGS>...]`

Run an ELF binary on the Jolt emulator (`jolt-emu`). Searches PATH, then
common locations (`target/release/jolt-emu`, `target/debug/jolt-emu`).
Override with `--jolt-emu` or `JOLT_EMU_PATH` env var.

```bash
jolt build -p guest && jolt run target/riscv64imac-unknown-none-elf/release/guest
```

### `jolt generate target`

Generate a custom RISC-V target specification JSON. Requires `--profile` or
`--target`.

```bash
jolt generate target --profile jolt-rv64 -o my-target.json
```

### `jolt generate linker`

Generate a linker script with custom memory layout.

```bash
jolt generate linker --ram-size 256Mi --heap-size 128Mi --stack-size 4Mi -o linker.ld
```

| Flag | Default | Description |
|------|---------|-------------|
| `--ram-start` | `0x80000000` | RAM start address |
| `--ram-size` | `128Mi` | RAM size |
| `--heap-size` | `64Mi` | Heap size |
| `--stack-size` | `2Mi` | Stack size |
| `--backtrace` | `true` | Include backtrace support |
| `--entry-point` | `_start` | Entry point symbol |
| `-o` | `linker.ld` | Output path |

### `jolt build-wasm`

Preprocess all `#[jolt::provable]` functions, generate `index.html`, and run
`wasm-pack build`. Project must have been created with `jolt new --wasm`.

### `jolt-emu` (RISC-V Emulator)

Standalone emulator binary (crate: `tracer`). Build with
`cargo build -p tracer --release`.

```bash
jolt-emu ./guest.elf              # execute
jolt-emu ./guest.elf -d           # disassemble each instruction
jolt-emu ./guest.elf -t true      # trace mode
```

### Profiling (jolt-core binary)

```bash
cargo run --release -p jolt-core -- profile --name sha3 -f chrome
cargo run --release -p jolt-core -- benchmark --name sha3 -s 20 -f chrome
```

Benchmark names: `sha2`, `sha3`, `sha2-chain`, `sha3-chain`, `fibonacci`,
`btreemap`. Add `--features monitor` for CPU/memory counter tracks, or
`--features allocative` for memory flamegraphs.

## Debugging & Profiling

### Trace Analysis

```rust
let summary = guest::analyze_fib(50);  // dry-run, returns ProgramSummary
guest::trace_fib_to_file("/tmp/trace.bin", 50);  // dump full trace
```

### Cycle Tracking

```rust
use jolt::{start_cycle_tracking, end_cycle_tracking};

start_cycle_tracking("my_span");
// ... code to measure ...
end_cycle_tracking("my_span");
```

Use `core::hint::black_box()` to prevent compiler reordering across boundaries.

### Environment Variables

| Variable | Effect |
|----------|--------|
| `JOLT_BACKTRACE=1` | Panic backtraces from guest |
| `JOLT_GUEST_OPT=0\|1\|2\|3\|s\|z` | Guest optimization level (default: 3) |
| `RUST_LOG=info` | Enable Jolt prover logs |

### Debug Feature

Log every syscall the guest makes:
```toml
jolt = { package = "jolt-sdk", features = ["guest-std", "debug"] }
```

## Troubleshooting

### Stack Overflow / Unpredictable Errors

Default `stack_size` is 4096 bytes. Increase for recursive code:
```rust
#[jolt::provable(stack_size = 65536)]
```

### Input/Output Too Large

Default 4096 bytes. Increase via macro attributes:
```rust
#[jolt::provable(max_input_size = 65536, max_output_size = 65536)]
```

### Null Pointer Write / Unknown Memory Mapping

Guest crashed — usually a missing jolt-sdk feature:

| Guest uses | Required feature |
|------------|-----------------|
| Threading / rayon | `"thread"` |
| Randomness (getrandom) | `"random"` |

### Standard Library Compilation Errors

Ensure `rust-toolchain.toml` is present and RISC-V target is installed:
```bash
rustup target add riscv64imac-unknown-none-elf
```

## Common Pitfalls

- **Missing `[patch.crates-io]`**: arkworks fork is required; builds fail
  with cryptic trait errors without it
- **Not cloning shared preprocessing**: `preprocess_prover_*` and
  `preprocess_verifier_*` both consume `JoltSharedPreprocessing` — clone it
- **Missing `compute_advice` feature**: runtime advice functions silently
  return zeros without the feature in guest `Cargo.toml`
- **Advice sizes not power-of-two**: `max_trusted_advice_size` and
  `max_untrusted_advice_size` should be powers of two
- **Input/output too large**: increase `max_input_size` / `max_output_size`
  in macro attributes (default 4096 bytes)
- **Stack overflow in guest**: increase `stack_size` (default 4096 — very
  small for recursive code)
- **Null pointer write**: missing `"thread"` or `"random"` jolt-sdk feature
  when guest uses threading/randomness
- **Guest fails to compile on host**: use `#[jolt::provable(guest_only)]`
  for code with inline RISC-V assembly
- **Inline crate not in host deps**: host must depend on `jolt-inlines-*`
  with `features = ["host"]`
