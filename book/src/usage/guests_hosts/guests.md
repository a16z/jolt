# Guests

Guests contain functions for Jolt to prove. Making a function provable is as easy as ensuring it is inside the `guest` package and adding the `jolt::provable` macro above it.

Let's take a look at a simple guest program to better understand it.
```rust
#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable]
fn add(x: u32, y: u32) -> u32 {
    x + y
}
```

As we can see, the guest looks like a normal `no_std` Rust library. The only major change is the addition of the `jolt::provable` macro, which lets Jolt know of the function's existence. The only requirement of these functions is that its inputs are serializable and outputs are deserializable with `serde`. Fortunately `serde` is prevalent throughout the Rust ecosystem, so most types will support it by default.

There is no requirement that just a single function lives within the guest, and we are free to add as many as we need. Additionally, we can import any `no_std` compatible library just as we normally would in Rust.
```rust
#![cfg_attr(feature = "guest", no_std)]

use sha2::{Sha256, Digest};
use sha3::{Keccak256, Digest};

#[jolt::provable]
fn sha2(input: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(input);
    let result = hasher.finalize();
    Into::<[u8; 32]>::into(result)
}

#[jolt::provable]
fn sha3(input: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    hasher.update(input);
    let result = hasher.finalize();
    Into::<[u8; 32]>::into(result)
}
```

## Standard Library
Jolt supports the Rust standard library. To enable support, simply add the `guest-std` feature to the Jolt import in the guest's `Cargo.toml` file and remove the `#![cfg_attr(feature = "guest", no_std)]` directive from the guest code.

### Example
```rust
[package]
name = "guest"
version = "0.1.0"
edition = "2021"

[features]
guest = []

[dependencies]
jolt = { package = "jolt-sdk", git = "https://github.com/a16z/jolt", features = ["guest-std"] }
```

```rust

#[jolt::provable]
fn int_to_string(n: i32) -> String {
    n.to_string()
}
```

## alloc

Jolt provides an allocator which supports most containers such as `Vec` and `Box`. This is useful for Jolt users who would like to write `no_std` code rather than using Jolt's standard library support. To use these containers, they must be explicitly imported from `alloc`. The `alloc` crate is automatically provided by rust and does not need to be added to the `Cargo.toml` file.

### Example
```rust
#![cfg_attr(feature = "guest", no_std)]

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

## Print statements
Jolt provides utilities emulating `print!` and `println!` in a guest program.

### Example

```rust
use jolt::{jolt_print, jolt_println};

#[jolt::provable(memory_size = 10240, max_trace_length = 65536)]
fn int_to_string(n: i32) -> String {
    jolt_print!("Hello, ")
    jolt_println!("from int_to_string({n})!");
    n.to_string()
}
```

Note that `jolt_print` and `jolt_println` support format strings. The printed strings are written to stdout during RISC-V emulation of the guest.
