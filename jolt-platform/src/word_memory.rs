//! RV64 memory routines using only existing integer instructions.
//!
//! The assembly copies object representations, including uninitialized bytes,
//! without materializing them as Rust integer values. Every load and store is
//! contained in the caller's ranges; the shifted path never rounds a read
//! outside those ranges. `memcpy` retains its non-overlap precondition.

core::arch::global_asm!(include_str!("word_memory.S"));
