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


## no_std

Jolt provides an allocator which supports most containers such as `Vec` and `Box`. This is useful for Jolt users who would like to write `no_std` code rather than using Jolt's standard library support. To use these containers, they must be explicitly imported from `alloc`. The `alloc` crate is automatically provided by rust and does not need to be added to the `Cargo.toml` file.

### Example
```rust
#![cfg_attr(feature = "guest", no_std)]
#![no_main]

extern crate alloc;
use alloc::vec::Vec;

#[jolt::provable]
fn alloc(n: u32) -> u32 {
    let mut v = Vec::<u32>::new();
    for i in 0..100 {
        v.push(i);
    }

    v[n as usize]
}
```
