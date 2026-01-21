# Issue Log

This document tracks encountered problems, including the original error messages and their root causes. Append new issues chronologically.

---

## 2026-01-20 — `cargo run --release -p stdlib`

**Error message**
```
thread 'main' panicked at /rustc/6b00bc3880198600130e1cf62b8f8a93494488cc/library/alloc/src/raw_vec/mod.rs:558:17:
capacity overflow
```

**Root cause**

The `parallel_sum_of_squares` guest uses Rayon, which attempts to perform OS-level threading. Inside the tracer, this leads to a store targeting physical address `0x0`, which the MMU treats as device I/O. `JoltDevice::store` assumes every write hits the output buffer, so subtracting the output base from `0x0` underflows, forcing `Vec::resize` to request an impossibly large capacity and triggering `alloc::raw_vec::capacity_overflow`.

**Suggested fix / workaround**

Avoid Rayon (or any thread-spawning APIs) inside provable guests until the runtime supports them. Reimplement `parallel_sum_of_squares` single-threaded or add proper syscall/threading support so device writes stay within the valid output region.
