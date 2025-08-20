# Guest Profiling
This section details the available tools to profile guest programs. There is currently a single utility available: `Cycle-Tracking`, which is detailed below.
## Cycle-Tracking in Jolt

Measure **real** (RV32IM) and **virtual** cycles inside your RISC-V guest code with zero-overhead markers. This is useful when analyzing the mapping between the high-level guest program and the equivalent compiled program to be proven by Jolt.


> **Note:** The Rust compiler will often shuffle around your implementation for optimization purposes, which can affect cycle tracking.
> If you suspect the compiler is interfering with profiling, then use the [Hint module](https://doc.rust-lang.org/core/hint/index.html).
> For example, wrap values in `core::hint::black_box()` (see below) to help keep your measurements honest.

### API

| Function                              | Effect                                   |
|---------------------------------------|-------------------------------------------|
| `start_cycle_tracking(label: &str)`   | Begin a span tagged with `label`          |
| `end_cycle_tracking(label: &str)`     | End the span started with the same `label`  |


Under the hood each marker emits an **ECALL** signaling the emulator to track your code segment.

### Example

~~~rust
#![cfg_attr(feature = "guest", no_std)]
use jolt::{start_cycle_tracking, end_cycle_tracking};

#[jolt::provable]
fn fib(n: u32) -> u128 {
    let (mut a, mut b) = (0, 1);

    start_cycle_tracking("fib_loop");
    for _ in 1..n {
        let sum = a + b;
        a = b;
        b = sum;
    }
    end_cycle_tracking("fib_loop");

    b
}
~~~

#### Hinting the compiler

~~~rust
#![cfg_attr(feature = "guest", no_std)]
use core::hint::black_box;

#[jolt::provable]
fn muldiv(a: u32, b: u32, c: u32) -> u32 {
    use jolt::{end_cycle_tracking, start_cycle_tracking};

   start_cycle_tracking("muldiv");
   let result = black_box(a * b / c); // use black_box to keep code in place
   end_cycle_tracking("muldiv");

   result
}
~~~

Wrap inputs *or* outputs that must stay observable during the span. In the above example, `a * b / c` gets moved to the return line without `black_box()`, causing inaccurate measurements. To run the above example, use `cargo run --release -p muldiv`.

###  Expected Output

~~~text
"muldiv": 9 RV32IM cycles, 16 virtual cycles
Trace length: 533
Prover runtime: 0.487551667 s
output: 2223173
valid: true
~~~
