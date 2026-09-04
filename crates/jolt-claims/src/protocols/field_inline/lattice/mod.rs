//! Packed (Akita) mode facts for the field-register extension, per
//! `specs/field-inline-portability.md` (Axis 1): the u64 limb decomposition
//! of `FieldRdInc`'s canonical representative, its linear recomposition
//! identity, and the dense prefix packing that commits the limb-word columns
//! as one independent Akita group beside advice.
//!
//! This module names facts only, mirroring `protocols::jolt::lattice`; the
//! balanced-digit algebra itself is the id-free [`crate::lattice`] module.

pub mod geometry;
pub mod packing;

pub use geometry::{
    canonical_limbs, canonical_limbs_into, field_inc_limb_count, limb_place_value, recompose_limbs,
    FIELD_INC_LIMB_BITS,
};
#[cfg(feature = "akita")]
pub use packing::field_inc_limbs_precommitted_role;
pub use packing::{FieldIncLimbPackingPlan, FieldIncLimbShape, FieldIncLimbWord};
