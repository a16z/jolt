# WASM Support

Jolt supports WebAssembly compatibility for the verification side of provable functions. This allows developers to run proof verification in WASM environments such as web browsers. Note that currently, only the verification process is supported in WASM; the proving process is not WASM-compatible at this time.

## Creating a WASM-Compatible Project

To create a new project with WASM verification support, use the following command:

```bash
jolt new <PROJECT_NAME> --wasm
```

This creates a new project with the necessary structure and configuration for WASM-compatible verification.

## Marking Functions for WASM Verification

For functions whose verification process should be WASM-compatible, extend the `#[jolt::provable]` macro with the `wasm` attribute:

```rust
#[jolt::provable(wasm)]
fn my_wasm_verifiable_function() {
    // function implementation
}
```

## Building for WASM Verification

To compile your project for WASM verification, use the following command:

```bash
jolt build-wasm
```

This command performs several actions:

1. It extracts all functions marked with `#[jolt::provable(wasm)` from your `guest/src/lib.rs` file.
2. For each WASM-verifiable function, it preprocesses the function and saves the necessary verification data.
3. It creates an `index.html` file as an example of how to use your WASM-compiled verification functions in a web environment.
4. It uses wasm-pack to build your project, targeting web environments.

> **Important:** The build process only compiles the verification functions for WASM. The proving process must still be performed outside of the WASM environment.

## Adding Dependencies

When adding new dependencies for WASM-compatible projects, note that they must be added to both guest/Cargo.toml and the root Cargo.toml. The build-wasm process will automatically add necessary WASM-related dependencies to your project.

## Switching Between WASM and Normal Compilation

The build process modifies the `Cargo.toml` files to enable WASM compilation. If you need to switch back to normal compilation, you may need to manually adjust these files. In the root `Cargo.toml`, you can remove the following lines:

```toml
[lib]
crate-type = ["cdylib"]
path = "guest/src/lib.rs"
```

Remember to restore these lines before attempting another WASM build.

## Example Project Structure

After running `jolt build-wasm`, your project will include:

- An `index.html` file in the root directory, providing a basic interface to verify proofs for each of your WASM-verifiable functions.
- A `pkg` directory containing the WASM-compiled version of your project's verification functions.
- Preprocessed data files for each WASM-verifiable function in the `target/wasm32-unknown-unknown/release/` directory.

You can use this example as a starting point and customize it to fit your specific requirements.

### Example: Modifying the Quickstart Project

You can use the example from the [Quickstart](./quickstart.md#project-tour) chapter and modify the `/src/main.rs` file as follows:

```rust
use jolt::Serializable;
pub fn main() {
    let (prove_fib, _verify_fib) = guest::build_fib();

    let (_output, proof) = prove_fib(50);

    proof
        .save_to_file("proof.bin")
        .expect("Failed to save proof to file");
}
```

Remember to modify the `Cargo.toml` as described [before](#switching-between-wasm-and-normal-compilation) for normal compilation, and then you can generate and save the proof with the following command:

```bash
cargo run -r
```

This will create a `proof.bin` file in the root directory. You can then verify this proof using the example `index.html`. Before doing this, change the crate type back to `cdylib` in the `Cargo.toml` and ensure that your `/guest/src/lib.rs` looks like this:

```rust
#![cfg_attr(feature = "guest", no_std)]
#![no_main]

#[jolt::provable(wasm)]
fn fib(n: u32) -> u128 {
    let mut a: u128 = 0;
    let mut b: u128 = 1;
    let mut sum: u128;
    for _ in 1..n {
        sum = a + b;
        a = b;
        b = sum;
    }

    b
}
```

Then run `jolt build-wasm`. After successful compilation, you can verify the proof using the `index.html` file. Start a local server from the project directory, for example with:

```bash
npx http-server .
```

> Note: Make sure you have `npx` installed to use the `http-server` command.
