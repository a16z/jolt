pub mod inputs;
#[cfg(feature = "akita")]
mod lattice;
pub mod outputs;
mod precommitted;
mod verify;

pub use inputs::{deps, Deps};
#[cfg(feature = "akita")]
pub use lattice::{
    build_lattice_packed_validity_batch, derive_lattice_packed_validity_requirements,
    derive_lattice_packed_validity_statements, derive_lattice_packed_witness_layout,
    jolt_lattice_physical_manifest, jolt_lattice_physical_manifest_with_validity,
    jolt_lattice_view_formula, jolt_lattice_view_formulas, lattice_packed_validity_claims,
    lattice_packed_validity_opening_count, lattice_protocol_config_for_packed_witness_layout,
    lattice_validity_requirements_for_packed_witness_layout,
    sample_lattice_packed_validity_eq_points, validate_lattice_packed_witness_layout_config,
    validate_lattice_packed_witness_validity_config, validate_lattice_view_validity_coverage,
    verify_lattice_packed_validity_proof, JoltLatticeViewFormulaWithRowPoint,
    LatticePackedValidityBatch, LatticePackedValidityBatchStatement,
    LatticePackedValidityStatement, LatticePackedValidityStatementKind,
};
#[cfg(feature = "akita")]
pub(crate) use lattice::{
    field_element_canonical_factors, field_element_canonical_value_from_openings,
    FieldCanonicalFactor,
};
pub use outputs::{
    Stage8BatchStatement, Stage8ClaimMode, Stage8ClearBatchStatement, Stage8ClearOutput,
    Stage8LogicalManifest, Stage8LogicalOpening, Stage8OpeningId, Stage8OpeningStatement,
    Stage8Output, Stage8PhysicalManifest, Stage8PhysicalOpening, Stage8ZkBatchStatement,
    Stage8ZkOutput,
};
pub use verify::{batch_statement, verify, verify_clear};
