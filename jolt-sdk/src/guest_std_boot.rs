//! Jolt guest std boot implementation.
//!
//! For std-mode guests, ZeroOS provides the complete boot infrastructure:
//! - `_start`, `__runtime_bootstrap`, `_init`, `_fini` from arch-riscv/runtime-musl
//! - `_default_trap_handler` from arch-riscv
//! - `trap_handler` from jolt-platform/trap.rs (routes syscalls via foundation::kfn::trap::ksyscall)
//!
//! This module is currently empty as jolt-platform provides everything needed.
//! It exists to maintain the module structure for potential future Jolt-specific overrides.
