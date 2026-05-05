---
name: jolt
description: Wrap a Rust function in a Jolt zero-knowledge proof
allowed-tools: Bash, Read, Edit, Write, Glob, Grep, Task
---

**Invoke when** the user says: "make this Jolt provable", "wrap this in Jolt", "prove this with Jolt", "add ZK proofs to this", "make this zero-knowledge", "make this provable", "jolt-ify this".

#### Step 1 — Identify the computation to prove

Look for a **pure, deterministic Rust function** — inputs in, result out, no I/O or side effects. If not obvious, ask:

> "What function should I make provable? It needs to be a pure Rust function with no I/O or side effects."

Before writing any guest code, verify the target function and its entire module path are `pub`. If not, make it `pub` in the library source (preferred — we're proving the library) and confirm with the user, noting that inlining is an alternative if they'd rather not modify the library.

#### Step 2 — Analyze and adapt the signature

The guest has a real heap — `Vec`, `String`, alloc types work freely inside the body. The constraint is at the **parameter boundary**: std mode uses full serde (Vec/String as params fine); no_std uses serde_core (no Vec params, arrays capped at size 32). Only adapt what's necessary:

| Issue | Resolution |
|-------|-----------|
| `Vec<T>` param in no_std | `[T; N], len: u32` — or switch to std mode |
| `[T; N]` where N > 32 in no_std | Split across multiple params (serde_core array size limit) |
| `usize` | `u32` (guest is 32-bit) |
| `f32` / `f64` | Fixed-point integer (e.g. `i64 * 1_000_000`) — RV64IMAC has no FPU |
| `std::io`, `std::net` | Cannot run in guest — explain and stop |
| Non-determinism | Pass seed/timestamp as explicit input |

**Build mode**: read the library's `Cargo.toml`. Use std mode if the library requires std, or if it makes the example simpler (e.g. Vec/String as params). No_std is a choice, not the default.

#### Step 3 — Install Jolt

```bash
jolt --version  # check if installed
cargo install --git https://github.com/a16z/jolt --force jolt  # if not
```

#### Step 4 — Scaffold

If inside an existing Rust library repo, propose:
> "I'll create `<library-name>-jolt/` here with the proof scaffold and import your library as a path dependency. Sound good?"

```bash
jolt new <project-name>        # standard mode
jolt new <project-name> --zk   # with PrivateInput + BlindFold support
```

This generates a workspace with a `fib` example — replace it by renaming `fib` → `<fn>` throughout `src/main.rs` and `guest/src/lib.rs`. Preserve the `[patch.crates-io]` block in the root `Cargo.toml` (required arkworks patches).

**Critical**: always use `jolt new` to scaffold. The generated `guest/src/main.rs` (`#![no_main]` binary stub that re-exports the lib) is required for Jolt to produce a RISC-V ELF binary. Without it, `cargo` builds only a `.rlib` and the host panics with "Built ELF not found." If creating the guest manually, always include:

```rust
// guest/src/main.rs
#![cfg_attr(feature = "guest", no_std)]
#![no_main]

#[allow(unused_imports)]
use guest::*;
```

#### Step 5 — Write the guest (`guest/src/lib.rs`)

**no_std mode** (default):
```rust
#![cfg_attr(feature = "guest", no_std)]
extern crate alloc;  // heap always available

#[jolt::provable]
fn <fn>(<params>) -> <ret> { ... }
```

**std mode** — in `guest/Cargo.toml`:
```toml
jolt = { package = "jolt-sdk", git = "https://github.com/a16z/jolt", features = ["guest-std", "thread", "stdout"] }
```
Include `"thread"` for rayon/parallel, `"stdout"` for `println!`. No `cfg_attr` needed in the lib file.

**Macro parameters** — use `#[jolt::provable]` bare; only add parameters when you have a reason:

| Parameter | Default | When to change | How to pick a value |
|-----------|---------|----------------|---------------------|
| `stack_size` | 4096 | `stack overflow` | Start at 8388608 (8 MB, matches Linux default); reduce in the optimization pass. |
| `max_trace_length` | 2^22 | `max_trace_length exceeded` | Run `analyze_<fn>` to get actual cycle count, round up to next power of 2. **Proving time and memory scale with this** — tighten in Step 9. |
| `heap_size` | 32 MB | `heap allocation failed` | Estimate peak live allocations; halve until it fails, then double back. |

**Prover-only inputs** — two options depending on whether you need cryptographic privacy:

- `jolt::UntrustedAdvice<T>` — prover-only; excluded from the verifier API but values may be recoverable from the proof
- `jolt::PrivateInput<T>` — same underlying type, signals that values should be cryptographically hidden via BlindFold (requires `zk` on the host, not the guest)

```rust
#[jolt::provable]
fn my_fn(public: u64, secret: jolt::UntrustedAdvice<[u8; 32]>) -> bool {
    let secret = *secret;
    // ...
}
```
Host prove call: `prove(..., UntrustedAdvice::new(val))`. The generated verifier signature omits the advice entirely. Add `use jolt_sdk::UntrustedAdvice;` to the host.

For `PrivateInput<T>`, enable `zk` on the host only (see Step 7). The macro enforces this at compile time.

`TrustedAdvice<T>` is the alternative for data committed by a third party — it requires a `commit_trusted_advice_<fn>(...)` host call and the commitment is passed to the verifier.

**Dependencies** — add to `guest/Cargo.toml`. When wrapping an existing repo, add `<library> = { path = "../.." }`. Avoid `default-features = false` unless you know the library supports it — disabled default features can expose conditionally-compiled modules that still reference missing optional deps. For crypto, prefer `jolt-inlines-sha2`, `jolt-inlines-keccak256`, `jolt-inlines-secp256k1`.

**Multiple functions** — each `#[jolt::provable]` generates independent `compile_*`, `preprocess_*`, `build_prover_*`, `build_verifier_*` APIs.

**Advice functions** — for expensive witness computation that should run outside the proof, use `#[jolt::advice]` in the guest. The function runs on the host/prover; the guest verifies the result cheaply with `jolt::check_advice!(bool_expr)` or `jolt::check_advice_eq!(a, b)`.

**Advice-based modular exponentiation** — for RSA/bigint modexp, the host computes `(quotient, remainder)` for each modular multiply via `#[jolt::advice]`; the guest only verifies `a*b == q*n + r` (two schoolbook wide multiplies + equality check) and `r < n`. This drops RSA-2048 verify from ~12M to ~461K cycles (26x). For e=65537 (2^16+1): 16 squarings + 1 multiply = 17 advice-verified steps. This pattern works for any modular exponentiation.

**Hash inline performance** — cycle counts at 64-byte input (with inline acceleration):

| Hash | Cycles | vs SHA-256 | Max input |
|------|--------|-----------|-----------|
| BLAKE3 | 863 | 8.2x faster | 64 bytes (single block only) |
| Blake2b | 1,264 | 5.6x faster | unlimited |
| Keccak-256 | 3,680 | 1.9x faster | unlimited |
| SHA-256 | 7,134 | baseline | unlimited |

BLAKE3 is fastest but capped at 64 bytes. For variable-length hashing, Blake2b is the best choice. SHA-256 must be used when the hash is dictated by an external protocol (e.g., JWT RSA-SHA256 signatures).

**Cycle tracking** — instrument sections of the guest to measure per-section cycle counts (visible in the prover log):
```rust
use jolt::{start_cycle_tracking, end_cycle_tracking};

start_cycle_tracking("my section");
// ... code to measure ...
end_cycle_tracking("my section");
```

#### Step 6 — Write the host (`src/main.rs`)

```rust
use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt().with_env_filter(
        tracing_subscriber::EnvFilter::from_default_env()
    ).init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_<fn>(target_dir);
    let shared = guest::preprocess_shared_<fn>(&mut program);
    let prover_prep = guest::preprocess_prover_<fn>(shared.clone());
    let verifier_setup = prover_prep.generators.to_verifier_setup();
    let verifier_prep = guest::preprocess_verifier_<fn>(shared, verifier_setup, None);
    let prove = guest::build_prover_<fn>(program, prover_prep);
    let verify = guest::build_verifier_<fn>(verifier_prep);

    let t = Instant::now();
    let (output, proof, io) = prove(<inputs>);
    info!("Prover runtime: {} s", t.elapsed().as_secs_f64());

    // io.panic is true if the guest panicked; the verifier checks it matches the proof
    let is_valid = verify(<inputs>, output, io.panic, proof);
    info!("output: {:?}", output);
    info!("valid: {is_valid}");
    assert!(is_valid);
}
```

For multiple functions, replicate the block per function. To measure cycles before proving: `guest::analyze_<fn>(<inputs>).write_to_file("summary.txt".into()).unwrap()`.

#### Step 7 — Run

Before running, estimate peak memory from `max_trace_length` (conservative worst-case):

| max_trace_length | Peak memory |
|-----------------|-------------|
| ≤ 2^23 | < 10 GB |
| 2^24 | ~15 GB |
| 2^25 | ~32 GB |
| 2^26 | ~42 GB |
| 2^27 | ~81 GB |
| 2^28 | ~99 GB |

If `max_trace_length` is 2^24 or above, warn the user and ask how to proceed:
> "This may require ~X GB of RAM. I can: (a) run `analyze_<fn>` first to get the actual cycle count — if it's well below `max_trace_length` we can lower it and reduce memory significantly, or (b) proceed directly. Which do you prefer?"

```bash
RUST_LOG=info cargo run --release
```

For **full zero-knowledge** (hides witness via BlindFold protocol), enable `zk` in **both** crates. Use `jolt new --zk` to scaffold a ZK project, or add manually:

Host `Cargo.toml`:
```toml
jolt-sdk = { git = "https://github.com/a16z/jolt", features = ["host", "zk"] }
```

Guest `Cargo.toml`:
```toml
jolt = { package = "jolt-sdk", git = "https://github.com/a16z/jolt", features = ["zk"] }
```

In the host, pass `BlindfoldSetup` to verifier preprocessing:
```rust
let blindfold_setup = prover_prep.blindfold_setup();
let verifier_prep = guest::preprocess_verifier_<fn>(shared, verifier_setup, Some(blindfold_setup));
```

Preprocessing runs once on first invocation and is not included in "Prover runtime". Diagnose failures:

| Error | Fix |
|-------|-----|
| `max_trace_length exceeded` | Add `max_trace_length = N` (tight power of 2 — proving time scales with this) |
| `heap allocation failed` | Add `heap_size = N` |
| `stack overflow` | Increase `stack_size`; start at 8388608 (8 MB) if not already set |
| `Illegal instruction` | Rewrite floats as fixed-point |
| `could not find crate` | Find no_std alternative or switch to std mode |
| `does not implement Serialize` | Add `#[derive(serde::Serialize, serde::Deserialize)]` |

#### Step 8 — Summarize

Tell the user: what function was made provable, what type adaptations were applied and why, std or no_std mode, and how to run it.

Once the proof runs end-to-end, **always offer a performance optimization pass**:

> "The proof works! Want me to optimize it? I can tighten `max_trace_length` to reduce memory and proving time, profile which sections dominate cycle count, and offload expensive witness computation."

#### Step 9 — Optimize (offer after Step 8 succeeds)

Work through these in order:

**1. Tighten `max_trace_length`** — run `guest::analyze_<fn>(<inputs>)`, find the actual cycle count, set `max_trace_length` to the smallest power of 2 above it. Proving time and peak memory are both proportional — a 2× reduction is a 2× speedup.

**2. Find the bottleneck** — add `start_cycle_tracking` / `end_cycle_tracking` (see Step 5) around major sections and run `analyze_<fn>` again. Focus on whichever section consumes >50% of cycles.

**3. Offload expensive witness computation** — if a section is expensive to compute but cheap to verify (sorting, hashing, witness generation), convert it to `#[jolt::advice]`. The advice function runs on the host outside the proof; the guest only verifies the result:
```rust
#[jolt::advice]
fn sort_array(input: &[u64]) -> jolt::UntrustedAdvice<Vec<u64>> {
    let mut v = input.to_vec();
    v.sort_unstable();
    v
}

#[jolt::provable]
fn my_fn(input: &[u64]) -> bool {
    let adv = sort_array(input);
    let sorted = &*adv;
    // O(n) verification: sorted order + length
    jolt::check_advice!(sorted.windows(2).all(|w| w[0] <= w[1]));
    jolt::check_advice!(sorted.len() == input.len());
    true
}
```

**4. Use crypto inlines** — for SHA-2, Keccak, secp256k1, replace standard crate calls with `jolt-inlines-*` (constraint-native, fraction of the cycle cost):
```toml
jolt-inlines-sha2 = { git = "https://github.com/a16z/jolt" }
```

**5. Trim `stack_size` and `heap_size`** — over-allocation doesn't cost cycles but does increase peak prover memory. Lower to actual usage once `max_trace_length` is tight.
