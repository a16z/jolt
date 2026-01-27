# Hosts

Hosts are where we can invoke the Jolt prover to prove functions defined within the guest.

The host imports the guest package, and will have automatically generated functions to build each of the Jolt functions. For the SHA3 example we looked at in the [guest](./guests.md) section, the `jolt::provable` procedural macro generates several functions that can be invoked from the host (shown below):

- `compile_sha3(target_dir)` to compile the SHA3 guest to RISC-V
- `preprocess_sha3` and `verifier_preprocessing_from_prover_sha3` to generate the prover and verifier preprocessing. Note that the preprocessing only needs to be generated once for a given guest program, and can subsequently be reused to prove multiple invocations of the guest.
- `build_prover_sha3` returns a closure for the prover, which takes in the same input types as the original function and modifies the output to additionally include a proof.
- `build_verifier_sha3` returns a closure for the verifier, which verifies the proof. The verifier closure's parameters comprise of the program input, the claimed output, a `bool` value claiming whether the guest panicked, and the proof.

```rust
pub fn main() {
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_sha3(target_dir);

    let prover_preprocessing = guest::preprocess_sha3(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_sha3(&prover_preprocessing);

    let prove_sha3 = guest::build_prover_sha3(program, prover_preprocessing);
    let verify_sha3 = guest::build_verifier_sha3(verifier_preprocessing);

    let input: &[u8] = &[5u8; 32];
    let now = Instant::now();
    let (output, proof, program_io) = prove_sha3(input);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_sha3(input, output, program_io.panic, proof);

    println!("output: {}", hex::encode(output));
    println!("valid: {is_valid}");
}
```
