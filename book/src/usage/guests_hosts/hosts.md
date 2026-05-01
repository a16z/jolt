# Hosts

Hosts are where we can invoke the Jolt prover to prove functions defined within the guest.

The host imports the guest package, and will have automatically generated functions to build each of the Jolt functions. For the SHA3 example we looked at in the [guest](./guests.md) section, the `jolt::provable` procedural macro generates several functions that can be invoked from the host (shown below):

- `compile_sha3(target_dir)` compiles the SHA3 guest to RISC-V.
- `preprocess_shared_sha3(&mut program)` produces shared preprocessing from the compiled program. This only needs to be generated once and can be reused across proofs.
- `preprocess_prover_sha3(shared)` and `preprocess_verifier_sha3(shared, verifier_setup)` produce prover and verifier preprocessing from the shared preprocessing. The verifier setup is obtained from `prover_pp.generators.to_verifier_setup()`.
- `build_prover_sha3` returns a closure for the prover, which takes in the same input types as the original function and returns the output, a proof, and a `program_io` device.
- `build_verifier_sha3` returns a closure for the verifier. The verifier closure's parameters comprise of the program input, the claimed output, a `bool` value claiming whether the guest panicked, and the proof.

```rust
pub fn main() {
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_sha3(target_dir);

    let shared = guest::preprocess_shared_sha3(&mut program);
    let prover_pp = guest::preprocess_prover_sha3(shared.clone());
    let verifier_pp = guest::preprocess_verifier_sha3(
        shared,
        prover_pp.generators.to_verifier_setup(),
    );

    let prove_sha3 = guest::build_prover_sha3(program, prover_pp);
    let verify_sha3 = guest::build_verifier_sha3(verifier_pp);

    let input: &[u8] = &[5u8; 32];
    let now = Instant::now();
    let (output, proof, program_io) = prove_sha3(input);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_sha3(input, output, program_io.panic, proof);

    println!("output: {}", hex::encode(output));
    println!("valid: {is_valid}");
}
```

## Memory management

To reduce peak memory usage during proving, replace the default system allocator with [jemalloc](https://github.com/jemalloc/jemalloc). The default system allocator retains freed pages in the process's address space for reuse, which inflates RSS beyond actual live memory. jemalloc returns freed pages to the OS more aggressively via `madvise`, keeping RSS close to true usage at a modest proving time cost.

```toml
# your-host/Cargo.toml
[dependencies]
tikv-jemallocator = "0.6"
```

```rust
// your-host/src/main.rs
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
```

`#[global_allocator]` must be declared in your final binary crate — Jolt SDK is a library and cannot set it for you.

This is most useful in containers, on shared machines where RSS determines whether you get OOM-killed, or for client-side proving where the prover runs on consumer hardware with limited memory. If you have plenty of memory and want the fastest possible proving, stick with the default allocator.
