# Guests
Guests contain functions for Jolt to prove. Currently, these functions must be written using `no_std` Rust. If you have a function that does not require the standard library, making it provable is as easy as ensuring it is inside the `guest` package and adding the `jolt::provable` macro above it.

Let's take a look at a simple guest program to better understand it.
```rust
#![cfg_attr(feature = "guest", no_std)]
#![no_main]

#[jolt::provable]
fn add(x: u32, y: u32) -> u32 {
    x + y
}
```

As we can see, the guest looks like a normal `no_std` Rust library. The only major change is the addition of the `jolt::provable` macro, which lets Jolt know of the function's existence. Other than `no_std`, the only requirement of these functions is that its inputs are serializable and outputs are deserializable with `serde`. Fortunately `serde` is prevalent throughout the Rust ecosystem, so most types will support it by default.

There is no requirement that just a single function lives within the guest, and we are free to add as many as we need. Additionally, we can import any `no_std` compatible library just as we normally would in Rust.
```rust
#![cfg_attr(feature = "guest", no_std)]
#![no_main]

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
