const FIELD_SOURCE: &str = include_str!("fp128.metal");
const SIMD_REDUCE_SOURCE: &str = include_str!("simd_reduce.metal");
const BYTECODE_READ_RAF_OFFSET: u32 = super::AKITA_OFFSET_FFFFA7F7;
const BYTECODE_READ_RAF_ADDRESS_SOURCE: &str = super::bytecode_read_raf_address::SOURCE;
const REGISTERS_CLAIM_REDUCTION_SOURCE: &str = super::registers_claim_reduction::SOURCE;
const DEFERRED_SUM_SOURCE: &str = include_str!("deferred_sum.metal");
const INSTRUCTION_READ_RAF_SOURCE: &str = super::instruction_read_raf::SOURCE;
const SPARTAN_OUTER_COMMON_SOURCE: &str = include_str!("spartan_outer_common.metal");
const BOOLEANITY_COMMON_SOURCE: &str = include_str!("booleanity_common.metal");
const INSTRUCTION_RA_COMMON_SOURCE: &str = include_str!("instruction_ra_common.metal");
const INSTRUCTION_CLAIM_REDUCTION_SOURCE: &str = super::instruction_claim_reduction::SOURCE;
const ADDRESS_RAF_DIRECT_SOURCE: &str = include_str!("address_raf_direct/shader.metal");
const ADDRESS_SUFFIX_FULL_SOURCE: &str = include_str!("address_suffix_full/shader.metal");
const ADDRESS_CYCLE_SOURCE: &str = include_str!("address_sequence/shader.metal");
const PRODUCT5_SOURCE: &str = include_str!("product5/shader.metal");
const PRODUCT_REMAINDER_SOURCE: &str = super::product_remainder::SOURCE;
const PRODUCT_INSTRUCTION_SERVICE_SOURCE: &str =
    super::instruction_claim_reduction_successor::SOURCE;
const PRODUCT_UNISKIP_SOURCE: &str = super::product_uniskip::SOURCE;
const RAM_RAF_EVALUATION_SOURCE: &str = super::ram_raf_evaluation::SOURCE;
const RAM_READ_WRITE_SOURCE: &str = super::ram_read_write::SOURCE;
const REGISTERS_READ_WRITE_SOURCE: &str = super::registers_read_write::SOURCE;
const REGISTERS_VAL_SOURCE: &str = include_str!("registers_val/shader.metal");
const BOOLEANITY_SOURCE: &str = include_str!("booleanity/shader.metal");
const BOOLEANITY_ADDRESS_SOURCE: &str = include_str!("booleanity_address/shader.metal");
const INSTRUCTION_RA_SEQUENCE_SOURCE: &str = include_str!("instruction_ra_sequence/shader.metal");
const RAM_RA_SEQUENCE_SOURCE: &str = include_str!("ram_ra_sequence/shader.metal");
const RAM_RA_CLAIM_REDUCTION_SOURCE: &str = super::ram_ra_claim_reduction::SOURCE;
const RAM_HAMMING_SEQUENCE_SOURCE: &str = include_str!("ram_hamming_sequence/shader.metal");
const RAM_VAL_SEQUENCE_SOURCE: &str = include_str!("ram_val_sequence/shader.metal");
const INSTRUCTION_INPUT_SOURCE: &str = include_str!("instruction_input/shader.metal");
const BYTECODE_CYCLE_SOURCE: &str = include_str!("bytecode_cycle/shader.metal");
const BYTECODE_ROW_SOURCE: &str = include_str!("bytecode_row/shader.metal");
const SPARTAN_OUTER_UNISKIP_SOURCE: &str = include_str!("spartan_outer_uniskip/shader.metal");
const SPARTAN_SHIFT_SOURCE: &str = super::spartan_shift::SOURCE;
const OUTER_REMAINDER_SOURCE: &str = super::outer_remainder::SOURCE;

struct SourceFragment {
    #[cfg(test)]
    id: &'static str,
    source: &'static str,
    required_offset: Option<u32>,
}

impl SourceFragment {
    const fn new(_id: &'static str, source: &'static str) -> Self {
        Self {
            #[cfg(test)]
            id: _id,
            source,
            required_offset: None,
        }
    }

    const fn for_offset(_id: &'static str, source: &'static str, offset: u32) -> Self {
        Self {
            #[cfg(test)]
            id: _id,
            source,
            required_offset: Some(offset),
        }
    }

    fn applies_to(&self, offset: u32) -> bool {
        match self.required_offset {
            Some(required) => required == offset,
            None => true,
        }
    }
}

const LIBRARY_SOURCE_FRAGMENTS: &[SourceFragment] = &[
    SourceFragment::new("fp128", FIELD_SOURCE),
    SourceFragment::new("simd_reduce", SIMD_REDUCE_SOURCE),
    SourceFragment::new("booleanity_common", BOOLEANITY_COMMON_SOURCE),
    SourceFragment::for_offset(
        "bytecode_read_raf_address",
        BYTECODE_READ_RAF_ADDRESS_SOURCE,
        BYTECODE_READ_RAF_OFFSET,
    ),
    SourceFragment::new(
        "registers_claim_reduction",
        REGISTERS_CLAIM_REDUCTION_SOURCE,
    ),
    SourceFragment::new("deferred_sum", DEFERRED_SUM_SOURCE),
    SourceFragment::new("instruction_read_raf", INSTRUCTION_READ_RAF_SOURCE),
    SourceFragment::new("spartan_outer_common", SPARTAN_OUTER_COMMON_SOURCE),
    SourceFragment::new("instruction_ra_common", INSTRUCTION_RA_COMMON_SOURCE),
    SourceFragment::new(
        "instruction_claim_reduction",
        INSTRUCTION_CLAIM_REDUCTION_SOURCE,
    ),
    SourceFragment::new("address_raf_direct", ADDRESS_RAF_DIRECT_SOURCE),
    SourceFragment::new("address_suffix_full", ADDRESS_SUFFIX_FULL_SOURCE),
    SourceFragment::new("product5", PRODUCT5_SOURCE),
    SourceFragment::new("product_remainder", PRODUCT_REMAINDER_SOURCE),
    SourceFragment::new(
        "product_instruction_round_service",
        PRODUCT_INSTRUCTION_SERVICE_SOURCE,
    ),
    SourceFragment::new("product_uniskip", PRODUCT_UNISKIP_SOURCE),
    SourceFragment::new("ram_raf_evaluation", RAM_RAF_EVALUATION_SOURCE),
    SourceFragment::new("ram_read_write", RAM_READ_WRITE_SOURCE),
    SourceFragment::new("registers_read_write", REGISTERS_READ_WRITE_SOURCE),
    SourceFragment::new("registers_val", REGISTERS_VAL_SOURCE),
    SourceFragment::new("booleanity", BOOLEANITY_SOURCE),
    SourceFragment::new("booleanity_address", BOOLEANITY_ADDRESS_SOURCE),
    SourceFragment::new("instruction_ra_sequence", INSTRUCTION_RA_SEQUENCE_SOURCE),
    SourceFragment::new("ram_ra_sequence", RAM_RA_SEQUENCE_SOURCE),
    SourceFragment::new("ram_ra_claim_reduction", RAM_RA_CLAIM_REDUCTION_SOURCE),
    SourceFragment::new("ram_hamming_sequence", RAM_HAMMING_SEQUENCE_SOURCE),
    SourceFragment::new("ram_val_sequence", RAM_VAL_SEQUENCE_SOURCE),
    SourceFragment::new("bytecode_cycle", BYTECODE_CYCLE_SOURCE),
    SourceFragment::new("bytecode_row", BYTECODE_ROW_SOURCE),
    SourceFragment::new("spartan_outer_uniskip", SPARTAN_OUTER_UNISKIP_SOURCE),
    SourceFragment::new("spartan_shift", SPARTAN_SHIFT_SOURCE),
    SourceFragment::new("instruction_input", INSTRUCTION_INPUT_SOURCE),
    SourceFragment::new("address_cycle", ADDRESS_CYCLE_SOURCE),
    SourceFragment::new("outer_remainder", OUTER_REMAINDER_SOURCE),
];

pub(super) fn library_source(offset: u32) -> String {
    assemble_library_source(offset, LIBRARY_SOURCE_FRAGMENTS)
}

fn assemble_library_source(offset: u32, source_fragments: &[SourceFragment]) -> String {
    let fragments = source_fragments
        .iter()
        .filter(|fragment| fragment.applies_to(offset))
        .map(|fragment| fragment.source)
        .collect::<Vec<_>>();
    let tables = super::instruction_read_raf::INSTRUCTION_READ_RAF_TABLES;
    format!(
        "#define SOLINAS_OFFSET {offset}u\n#define INSTRUCTION_READ_RAF_TABLE_COUNT {tables}u\n{}",
        fragments.join("\n")
    )
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "production manifest test fixture")]
mod tests {
    use super::*;
    use crate::metal::solinas::AKITA_OFFSET_FFFFA7F7;

    fn manifest_fragment_ids(field: &str) -> Vec<String> {
        let manifest: serde_json::Value =
            serde_json::from_str(include_str!("../production_manifest.json")).unwrap();
        manifest[field]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect()
    }

    #[test]
    fn source_assembly_puts_the_outer_fragment_last() {
        let generic = library_source(275);
        assert!(!generic.contains(BYTECODE_READ_RAF_ADDRESS_SOURCE));
        assert!(generic.ends_with(OUTER_REMAINDER_SOURCE));

        let akita = library_source(AKITA_OFFSET_FFFFA7F7);
        assert!(akita.contains(BYTECODE_READ_RAF_ADDRESS_SOURCE));
        assert!(akita.ends_with(OUTER_REMAINDER_SOURCE));
    }

    #[test]
    fn source_pins_the_lookup_table_count_for_the_read_raf_shader() {
        let source = library_source(AKITA_OFFSET_FFFFA7F7);
        let define = format!(
            "#define INSTRUCTION_READ_RAF_TABLE_COUNT {}u\n",
            crate::metal::solinas::instruction_read_raf::INSTRUCTION_READ_RAF_TABLES
        );
        assert!(source.contains(&define));
        assert!(!INSTRUCTION_READ_RAF_SOURCE.contains("INSTRUCTION_READ_RAF_TABLES = 5"));
    }

    #[test]
    fn source_excludes_rejected_kernels() {
        let source = library_source(AKITA_OFFSET_FFFFA7F7);

        for rejected in [
            "solinas_ram_output_check_",
            "solinas_address_raf_histogram",
            "solinas_address_suffix_one_tile",
            "solinas_product_uniskip_reduce2",
            "solinas_registers_claim_build_linear_q",
            "solinas_registers_claim_fold_direct",
            "solinas_ram_ra_claim_build_q_compact",
            "solinas_ram_ra_claim_build_q_single_term",
            "solinas_ram_ra_claim_build_q_vector",
        ] {
            assert!(!source.contains(rejected), "rejected kernel {rejected}");
        }
        for required in [
            PRODUCT_REMAINDER_SOURCE,
            PRODUCT_INSTRUCTION_SERVICE_SOURCE,
            REGISTERS_VAL_SOURCE,
        ] {
            assert!(source.contains(required));
        }
    }

    #[test]
    fn production_manifest_matches_source_assembly() {
        let fragments = LIBRARY_SOURCE_FRAGMENTS
            .iter()
            .map(|fragment| fragment.id.to_owned())
            .collect::<Vec<_>>();

        assert_eq!(manifest_fragment_ids("metal_source_fragments"), fragments);
        assert_eq!(
            manifest_fragment_ids("cpu_delegated_slots"),
            ["ram_output_check"]
        );
    }
}
