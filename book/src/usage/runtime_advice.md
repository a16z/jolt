# Runtime Advice

Runtime advice allows guest programs to offload expensive computations to the prover and receive the results via an **advice tape**. The guest then verifies the result cheaply, rather than recomputing it from scratch. This can dramatically reduce cycle counts for operations like modular inversion, factoring, or any computation where checking a result is cheaper than computing it.

Runtime advice is distinct from the [advice inputs](./guests_hosts/guests.md#advice-inputs-vs-runtime-advice) (`TrustedAdvice<T>` / `UntrustedAdvice<T>` parameters on `#[jolt::provable]`), which are host-provided data fixed before execution begins. With runtime advice, the guest itself computes the values during a first-pass execution, and those values are replayed from the advice tape during the proving pass. Use advice inputs when the host already knows the auxiliary data; use runtime advice when the data depends on the guest's own logic and checking is cheaper than computing.

## How it works

Jolt uses a **two-pass** execution model for advice:

1. **First pass (compute_advice)**: The guest is compiled with the `compute_advice` feature flag. Advice functions execute their body and write results to the advice tape.
2. **Second pass (proving)**: The guest is compiled without `compute_advice`. Advice functions read precomputed results from the advice tape instead of recomputing them.

The prover handles both passes automatically. The guest code only needs to define advice functions and verify their outputs.

## Defining advice functions

Annotate a function with `#[jolt::advice]`. The function must return `jolt::UntrustedAdvice<T>`, where `T` implements `AdviceTapeIO`.

```rust
use jolt::AdviceTapeIO;

#[jolt::advice]
fn modinv_advice(a: u64, m: u64) -> jolt::UntrustedAdvice<(u64, u64)> {
    // This body only runs during the compute_advice pass.
    // During proving, the result is read from the advice tape.
    let inv = extended_gcd(a, m);
    let quo = (a as u128 * inv as u128 / m as u128) as u64;
    (inv, quo)
}
```

Requirements:
- The return type **must** be `jolt::UntrustedAdvice<T>`
- Arguments must be immutable (no `mut` or `&mut`)
- `T` must implement `AdviceTapeIO` (see [Supported types](#supported-types) below)

## Verifying advice

Advice values are **untrusted** -- the prover could supply arbitrary data. The guest must verify correctness using `check_advice!` or `check_advice_eq!`. These macros emit prover-enforced assertions: if the condition is false, proof generation fails.

```rust
#[jolt::provable]
fn modinv(a: u64, m: u64) -> u64 {
    let adv = modinv_advice(a, m);
    let (inv, quo) = *adv;  // Deref to access the inner value

    // Verify: a * inv ≡ 1 (mod m)
    let product = (a as u128) * (inv as u128) - (quo as u128) * (m as u128);
    jolt::check_advice!(product == 1u128 && inv < m);

    inv
}
```

`check_advice_eq!` is a specialization for equality checks that directly compares two register-sized values, saving a few instructions compared to `check_advice!`:

```rust
jolt::check_advice_eq!(a * b, n, "incorrect factors");
jolt::check_advice!(1 < a && a <= b && b < n, "factors out of range");
```

Both macros accept an optional error message string as the last argument. The message is used in `assert!`/`assert_eq!` on non-RISC-V targets (useful for debugging) but is stripped from the guest binary.

## Supported types

The `AdviceTapeIO` trait controls how values are serialized to and from the advice tape. Built-in implementations are provided for:

| Type | Notes |
|------|-------|
| `u8`, `u16`, `u32`, `u64`, `usize` | Primitive integers |
| `i8`, `i16`, `i32`, `i64` | Signed integers |
| `[T; N]` where `T: Pod` | Fixed-size arrays |
| `(A, B)`, ..., `(A, B, C, D, E, F, G)` | Tuples up to 7 elements |
| `Vec<T>` where `T: Pod` | Requires `guest-std` feature |

### Custom structs

For structs composed of supported types, you can implement `AdviceTapeIO` manually:

```rust
struct Pair {
    x: u64,
    y: u64,
}

impl jolt::AdviceTapeIO for Pair {
    fn write_to_advice_tape(&self) {
        self.x.write_to_advice_tape();
        self.y.write_to_advice_tape();
    }
    fn new_from_advice_tape() -> Self {
        Pair {
            x: u64::new_from_advice_tape(),
            y: u64::new_from_advice_tape(),
        }
    }
}
```

### JoltPod: automatic AdviceTapeIO via bytemuck

For `#[repr(C)]` structs where all fields are plain-old-data, you can derive `AdviceTapeIO` automatically using `bytemuck` and the `JoltPod` marker trait:

```rust
use bytemuck_derive::{Pod, Zeroable};
use jolt::JoltPod;

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct Point {
    x: u32,
    y: u32,
}
impl JoltPod for Point {}
```

`JoltPod` types get a blanket `AdviceTapeIO` implementation that uses `bytemuck` for zero-copy serialization. This works for nested structs too, as long as all types in the hierarchy are `Pod`.

## Guest setup

The guest `Cargo.toml` must include a `compute_advice` feature:

```toml
[package]
name = "guest"
version = "0.1.0"
edition = "2021"

[features]
guest = []
compute_advice = []

[dependencies]
jolt = { package = "jolt-sdk", git = "https://github.com/a16z/jolt" }
```

The `compute_advice` feature is used by the SDK to build the first-pass binary. You do not need to activate it yourself.

## Full example

Guest (`guest/src/lib.rs`):

```rust
use jolt::AdviceTapeIO;

/// Compute factors of n via advice (expensive trial division runs outside the proof)
#[jolt::advice]
fn factor(n: u32) -> jolt::UntrustedAdvice<[u32; 2]> {
    for i in 2..=n {
        if n % i == 0 {
            return [i, n / i];
        }
    }
    [1, n]
}

/// Prove that n is composite by obtaining and verifying its factors via advice
#[jolt::provable]
fn verify_composite(n: u32) {
    let adv = factor(n);
    let [a, b] = *adv;

    // Verify the factors are correct and non-trivial
    jolt::check_advice_eq!((a as u64) * (b as u64), n as u64);
    jolt::check_advice!(1 < a && a <= b && b < n);
}
```

Host (`src/main.rs`):

```rust
pub fn main() {
    let target_dir = "/tmp/jolt-guest-targets";

    let mut program = guest::compile_verify_composite(target_dir);
    let shared_preprocessing = guest::preprocess_shared_verify_composite(&mut program);
    let prover_preprocessing =
        guest::preprocess_prover_verify_composite(shared_preprocessing.clone());
    let verifier_setup = prover_preprocessing.generators.to_verifier_setup();
    let verifier_preprocessing =
        guest::preprocess_verifier_verify_composite(shared_preprocessing, verifier_setup);

    let prove = guest::build_prover_verify_composite(program, prover_preprocessing);
    let verify = guest::build_verifier_verify_composite(verifier_preprocessing);

    let n = 221u32;  // 13 * 17
    let (output, proof, program_io) = prove(n);
    let is_valid = verify(n, output, program_io.panic, proof);
    assert!(is_valid);
}
```

The host code is identical to any other Jolt program. The two-pass advice mechanism is handled transparently by the `compile_*` and `build_prover_*` functions generated by the `#[jolt::provable]` macro.
