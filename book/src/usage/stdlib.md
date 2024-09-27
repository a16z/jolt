# Standard Library
Jolt supports the full Rust standard library. To enable support, simply add the `guest-std` feature to the Jolt import in the guest's `Cargo.toml` file and remove the `#![cfg_attr(feature = "guest", no_std)]` directive from the guest code.

## Example
```rust
[package]
name = "guest"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "guest"
path = "./src/lib.rs"

[features]
guest = []

[dependencies]
jolt = { package = "jolt-sdk", git = "https://github.com/a16z/jolt", features = ["guest-std"] }
```

```rust
#![no_main]

#[jolt::provable]
fn int_to_string(n: i32) -> String {
    n.to_string()
}
```
