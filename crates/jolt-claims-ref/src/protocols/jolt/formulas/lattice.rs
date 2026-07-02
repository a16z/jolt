mod families;
mod layout;
mod openings;
mod relations;
#[cfg(test)]
mod tests;
mod types;
mod validity;
mod views;

pub const UNSIGNED_INC_BITS: usize = 64;

pub use families::{jolt_advice_kind, packing_advice_kind, JoltPackingFamilyId};
pub use layout::{
    derive_jolt_lattice_packed_validity_requirements, derive_jolt_lattice_packed_witness_layout,
    lattice_validity_requirements_for_packed_witness_layout, layout_has_advice,
    layout_has_field_rd_inc, packed_alphabet_with_size, packed_family_is_precommitted,
    FieldRdIncPacking, JoltLatticeLayoutError, JoltLatticePackingInputs, JoltLatticeValidityInputs,
};
pub use openings::{
    final_opening_lattice_requirement, inc_virtualization_inc_opening,
    inc_virtualization_input_openings, inc_virtualization_output_openings,
    inc_virtualization_ram_read_write_opening, inc_virtualization_ram_val_check_opening,
    inc_virtualization_rd_read_write_opening, inc_virtualization_rd_val_evaluation_opening,
    inc_virtualization_relation, inc_virtualization_store_opening, unsigned_inc_chunk_opening,
    unsigned_inc_chunk_reconstruction_relation, unsigned_inc_claim_reduction_relation,
    unsigned_inc_input_opening, unsigned_inc_lower_chunk_count, unsigned_inc_msb_opening,
    unsigned_inc_opening, LatticeFinalOpeningRequirement,
};
pub use relations::{
    inc_virtualization_claim, unsigned_inc_chunk_reconstruction_claim,
    unsigned_inc_claim_reduction_claim, unsigned_inc_msb_booleanity_claim,
};
pub use types::{
    packing_validity_digest, PackingAdviceKind, PackingAlphabet, PackingFactDomain,
    PackingFamilyId, PackingFamilySpec, PackingLayoutError, PackingLayoutFamily,
    PackingValidityDigest, PackingValidityKind, PackingValidityRequirement, PackingViewFormula,
    PackingViewKind, PackingViewTerm, PackingViewValidity, PackingWitnessLayout,
};
pub use validity::{
    advice_bytes_validity_requirement, bytecode_imm_canonical_bytes_requirement,
    bytecode_validity_requirements, program_image_validity_requirement,
    unsigned_inc_validity_requirements,
};
pub use views::{
    byte_decode_terms, bytecode_chunk_lattice_view_formula,
    bytecode_rd_present_lattice_view_formula, bytecode_store_flag_lattice_view_formula,
    little_endian_byte_decode_terms, symbol_decode_terms,
    unsigned_inc_lower_value_lattice_view_formula, unsigned_inc_lower_value_terms,
    unsigned_inc_msb_lattice_view_formula, weighted_byte_decode_terms, weighted_symbol_terms,
};
