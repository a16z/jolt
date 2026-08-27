//! Packed (Akita) mode facts for the field-register extension, per
//! `specs/field-inline-portability.md` (Axis 1): the u64 limb decomposition
//! of `FieldRdInc`'s canonical representative and its linear recomposition
//! identity. The packed commitment treatment of the limbs is being reworked
//! onto the dense-group batch opening (`field-inline x akita` is
//! compile-error gated in `jolt-verifier` meanwhile).
//!
//! This module names facts only, mirroring `protocols::jolt::lattice`; the
//! balanced-digit algebra itself is the id-free [`crate::lattice`] module.

pub mod geometry;

pub use geometry::{canonical_limbs, field_inc_limb_count, limb_place_value, FIELD_INC_LIMB_BITS};
