//! IR-based claim definitions for all Jolt sumcheck instances.
//!
//! Each submodule corresponds to a subsystem of the Jolt zkVM. Every
//! function returns a [`ClaimDefinition`](crate::ClaimDefinition) encoding the
//! mathematical formula as a symbolic expression. These definitions are the
//! **single source of truth** — evaluation, BlindFold R1CS, Lean4, and circuit
//! backends all derive from them.
//!
//! # Subsystems
//!
//! | Module | Claims | Description |
//! |--------|--------|-------------|
//! | [`ram`] | 6 | Hamming booleanity, RW checking, output, val, RAF |
//! | [`registers`] | 2 | Register RW checking, val evaluation |
//! | [`spartan`] | 2 | Shift, instruction input |
//! | [`instruction`] | 1 | RA virtual decomposition (parameterized) |
//! | [`reductions`] | 5 | Registers, instruction lookups, RAM RA, increment, Hamming weight |
//!
//! # Design
//!
//! Output claims use pre-computed challenge values (eq evaluations, γ-powers)
//! as `Var::Challenge` entries. This keeps the IR expression simple — degree-2
//! or degree-3 at most — while the challenge computation logic lives in the
//! stage implementations that supply concrete values at proving time.
//!
//! Input claims follow the same pattern for complex cases (RAM val check with
//! advice contributions). Simple input claims (weighted sum of prior openings)
//! are computed directly in stage code without a ClaimDefinition.

pub mod instruction;
pub mod r1cs;
pub mod ram;
pub mod reductions;
pub mod registers;
pub mod spartan;

// The following claims are computed at runtime and do not have static
// ClaimDefinitions:
//
// - **Outer uni-skip**: Input is zero, output is a direct polynomial opening.
// - **Product virtual uni-skip**: Input is a Lagrange kernel evaluation over
//   the uni-skip domain — depends on runtime `tau` values and domain size.
// - **Outer remaining**: Output involves `UniformSpartanKey::evaluate_inner_sum_product_at_point`
//   — a lazy R1CS evaluation that depends on the full constraint system.
// - **Advice cycle phase**: Output is a direct polynomial opening (no formula).
//
// These are handled directly in stage implementations.
