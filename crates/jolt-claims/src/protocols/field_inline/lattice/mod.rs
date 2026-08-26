//! Packed (Akita) mode facts for the field-register extension, per
//! `specs/field-inline-portability.md` (Axis 1): the limb-column
//! decomposition of `FieldRdInc`'s canonical representative, its linear
//! recomposition identity, and the canonical prefix-packed layout of the
//! committed limb columns.
//!
//! This module names facts only, mirroring `protocols::jolt::lattice`; the
//! balanced-digit algebra itself is the id-free [`crate::lattice`] module,
//! ridden verbatim. Stage wiring lands with the packed FR fixtures (the
//! `field-inline x akita` verifier gate comes out only in that change).

pub mod geometry;
pub mod packing;

pub use geometry::{
    canonical_limbs, field_inc_limb_count, limb_place_value, recomposition_coefficient,
    FieldIncLimbGeometryError, FIELD_INC_LIMB_BITS,
};
pub use packing::{field_inc_limb_columns, FieldIncLimbPackingPlan, FieldIncLimbShape};
