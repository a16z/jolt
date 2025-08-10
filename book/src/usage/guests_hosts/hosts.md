# Hosts
Hosts are where we can invoke the Jolt prover to prove functions defined within the guest. Hosts do not have the `no_std` requirement, and are free to use the Rust standard library.

The host imports the guest package, and will have automatically generated functions to build each of the Jolt functions. For the sha2 and sha3 example guest we looked at in the [guest](./guests.md) section, these functions would be called `build_sha2` and `build_sha3` respectively. Each returns two results, a prover function and a verifier function. The prover function takes in the same input types as the original function and modifies the output to additionally include a proof. The verifier can then take this proof and verify it.


```rust
pub fn main() {
    let (prove_sha2, verify_sha2) = guest::build_sha2();
    let (prove_sha3, verify_sha3) = guest::build_sha3();

    let input = &[5u8; 32];

    let (output, proof) = prove_sha2(input);
    let is_valid = verify_sha2(proof);

    println!("sha2 output: {output}");
    println!("sha2 valid: {is_valid}");

    let (output, proof) = prove_sha3(input);
    let is_valid = verify_sha3(proof);

    println!("sha3 output: {output}");
    println!("sha3 valid: {is_valid}");
}
```
