//! Runtime advice system constants and utilities.
//!
//! The advice system allows guest programs to provide non-deterministic witness data
//! during execution that will be verified by the proof system.
//!
//! In this context, *non-deterministic witness data* is any value that:
//! - Is known to the guest program at run time (during emulation/proving),
//! - Is not uniquely determined by the public inputs alone, and
//! - Must still be constrained and checked by the proof system.
//!
//! Typical use cases include:
//! - Providing Merkle authentication paths or other commitment openings that are
//!   checked inside the proof but are too large or inconvenient to recompute from
//!   public inputs.
//! - Supplying private randomness or secrets (e.g., keys, nonces) that influence
//!   execution, while allowing the prover to convince the verifier that the
//!   computation used those values consistently.
//! - Streaming lookups into large external tables or datasets that live outside
//!   the core circuit, where the advice tape records the looked-up values and the
//!   proof enforces that the guest used exactly those values.
//!
//! Practically, the guest writes this data during the first emulation pass using
//! the advice interface. During the second (proving) pass, advice instructions
//! read the recorded values, and the proof system ensures that the execution is
//! consistent with the supplied advice.

/// Identifier for writing advice data during emulation.
/// The advice tape stores data from the first emulation pass that can be read
/// during the second (proving) pass via advice instructions.
pub const JOLT_ADVICE_WRITE_CALL_ID: u32 = 0xADBABE;
