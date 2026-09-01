//! AOT x86-64 transpiling execution backend for Jolt trace generation.
//!
//! Implements the `jolt-program` execution seam (`ExecutionBackend`,
//! `ChunkedExecutionBackend`) by compiling a `JoltProgram`'s expanded bytecode
//! to native x86-64 once per program (via dynasm-rs) and executing it in two
//! modes: a fast checkpointing pass and a per-chunk recording pass.
//!
//! Native codegen is gated to `x86_64`-Linux (the SP1/ZisK precedent). On
//! every other target this crate still compiles, and [`NativeBackend`]
//! resolves to the reference interpreter, so call sites can select the
//! fastest available backend unconditionally:
//!
//! ```ignore
//! let mut backend = jolt_tracer_x86::NativeBackend::default();
//! let output = program.trace_with(&mut backend, inputs)?;
//! ```
//!
//! See `specs/x86-tracer-backend.md` for the design (row templates over the
//! statically expanded bytecode; fail-fast on unsupported rows; bit-identical
//! `TraceRow` streams vs. the reference `TracerBackend`).

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod native;

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
pub use native::X86TracerBackend;

#[doc(hidden)]
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
pub use native::harness;

/// The fastest execution backend available on this target: the AOT x86-64
/// transpiler on `x86_64`-Linux, the reference interpreter everywhere else.
///
/// This alias is the entire backend-selection surface; nothing switches by
/// default.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
pub type NativeBackend = X86TracerBackend;

/// The fastest execution backend available on this target: the AOT x86-64
/// transpiler on `x86_64`-Linux, the reference interpreter everywhere else.
///
/// This alias is the entire backend-selection surface; nothing switches by
/// default.
#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
pub type NativeBackend = tracer::TracerBackend;
