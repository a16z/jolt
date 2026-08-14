const FIELD_SOURCE: &str = include_str!("fp128.metal");
const SIMD_REDUCE_SOURCE: &str = include_str!("simd_reduce.metal");
const BYTECODE_READ_RAF_OFFSET: u32 = super::AKITA_OFFSET_FFFFA7F7;
const BYTECODE_READ_RAF_ADDRESS_SOURCE: &str = super::bytecode_read_raf_address::SOURCE;
const REGISTERS_CLAIM_REDUCTION_SOURCE: &str = super::registers_claim_reduction::SOURCE;
const RAM_RA_CLAIM_REDUCTION_SOURCE: &str = super::ram_ra_claim_reduction::SOURCE;
const DEFERRED_SUM_SOURCE: &str = include_str!("deferred_sum.metal");
const INSTRUCTION_READ_RAF_ADDRESS_SOURCE: &str = super::instruction_read_raf_v3::SOURCE;
const INSTRUCTION_READ_RAF_SOURCE: &str = super::instruction_read_raf::SOURCE;
const SPARTAN_OUTER_COMMON_SOURCE: &str = include_str!("spartan_outer_common.metal");
const BOOLEANITY_COMMON_SOURCE: &str = include_str!("booleanity_common.metal");
const INSTRUCTION_RA_COMMON_SOURCE: &str = include_str!("instruction_ra_common.metal");
const INSTRUCTION_CLAIM_REDUCTION_SOURCE: &str = super::instruction_claim_reduction::SOURCE;
const ADDRESS_RAF_SOURCE: &str = include_str!("address_raf/shader.metal");
const ADDRESS_RAF_DIRECT_SOURCE: &str = include_str!("address_raf_direct/shader.metal");
const ADDRESS_SUFFIX_SOURCE: &str = include_str!("address_suffix/shader.metal");
const ADDRESS_SUFFIX_FULL_SOURCE: &str = include_str!("address_suffix_full/shader.metal");
const ADDRESS_CYCLE_SOURCE: &str = include_str!("address_sequence/shader.metal");
const PROBE_SOURCE: &str = include_str!("probes.metal");
const PRODUCT5_SOURCE: &str = include_str!("product5/shader.metal");
const PRODUCT_REMAINDER_SOURCE: &str = super::product_remainder::SOURCE;
const PRODUCT_INSTRUCTION_SERVICE_SOURCE: &str =
    super::instruction_claim_reduction_successor::SOURCE;
const PRODUCT_UNISKIP_SOURCE: &str = super::product_uniskip::SOURCE;
const RAM_RAF_EVALUATION_SOURCE: &str = super::ram_raf_evaluation::SOURCE;
const RAM_VAL_CHECK_SOURCE: &str = super::ram_val_check::SOURCE;
const REGISTERS_VAL_SOURCE: &str = include_str!("registers_val/shader.metal");
const BOOLEANITY_SOURCE: &str = include_str!("booleanity/shader.metal");
const BOOLEANITY_ADDRESS_SOURCE: &str = include_str!("booleanity_address/shader.metal");
const INSTRUCTION_RA_SOURCE: &str = include_str!("instruction_ra_virtualization/shader.metal");
const INSTRUCTION_RA_SEQUENCE_SOURCE: &str = include_str!("instruction_ra_sequence/shader.metal");
const INSTRUCTION_INPUT_SOURCE: &str = include_str!("instruction_input/shader.metal");
const INSTRUCTION_INPUT_DENSE_SOURCE: &str = super::instruction_input_successor::SOURCE;
const BYTECODE_CYCLE_SOURCE: &str = include_str!("bytecode_cycle/shader.metal");
const BYTECODE_ROW_SOURCE: &str = include_str!("bytecode_row/shader.metal");
const SPARTAN_OUTER_UNISKIP_SOURCE: &str = include_str!("spartan_outer_uniskip/shader.metal");
const SPARTAN_SHIFT_SOURCE: &str = super::spartan_shift::SOURCE;
const OUTER_REMAINDER_SOURCE: &str = super::outer_remainder::SOURCE;
const OUTER_REMAINDER_PADDED_56_SOURCE: &str = super::outer_remainder::PADDED_56_SOURCE;

struct SourceFragment {
    id: &'static str,
    source: &'static str,
    required_offset: Option<u32>,
    production: bool,
}

impl SourceFragment {
    const fn new(id: &'static str, source: &'static str) -> Self {
        Self {
            id,
            source,
            required_offset: None,
            production: true,
        }
    }

    const fn diagnostic(id: &'static str, source: &'static str) -> Self {
        Self {
            id,
            source,
            required_offset: None,
            production: false,
        }
    }

    const fn for_offset(id: &'static str, source: &'static str, offset: u32) -> Self {
        Self {
            id,
            source,
            required_offset: Some(offset),
            production: true,
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
    SourceFragment::new("ram_ra_claim_reduction", RAM_RA_CLAIM_REDUCTION_SOURCE),
    SourceFragment::new("deferred_sum", DEFERRED_SUM_SOURCE),
    SourceFragment::new(
        "instruction_read_raf_address",
        INSTRUCTION_READ_RAF_ADDRESS_SOURCE,
    ),
    SourceFragment::new("instruction_read_raf", INSTRUCTION_READ_RAF_SOURCE),
    SourceFragment::new("spartan_outer_common", SPARTAN_OUTER_COMMON_SOURCE),
    SourceFragment::new("instruction_ra_common", INSTRUCTION_RA_COMMON_SOURCE),
    SourceFragment::new(
        "instruction_claim_reduction",
        INSTRUCTION_CLAIM_REDUCTION_SOURCE,
    ),
    SourceFragment::new("address_raf", ADDRESS_RAF_SOURCE),
    SourceFragment::new("address_raf_direct", ADDRESS_RAF_DIRECT_SOURCE),
    SourceFragment::new("address_suffix", ADDRESS_SUFFIX_SOURCE),
    SourceFragment::new("address_suffix_full", ADDRESS_SUFFIX_FULL_SOURCE),
    SourceFragment::diagnostic("probes", PROBE_SOURCE),
    SourceFragment::new("product5", PRODUCT5_SOURCE),
    SourceFragment::new("product_remainder", PRODUCT_REMAINDER_SOURCE),
    SourceFragment::new(
        "product_instruction_round_service",
        PRODUCT_INSTRUCTION_SERVICE_SOURCE,
    ),
    SourceFragment::new("product_uniskip", PRODUCT_UNISKIP_SOURCE),
    SourceFragment::new("ram_raf_evaluation", RAM_RAF_EVALUATION_SOURCE),
    SourceFragment::new("ram_val_check", RAM_VAL_CHECK_SOURCE),
    SourceFragment::new("registers_val", REGISTERS_VAL_SOURCE),
    SourceFragment::new("booleanity", BOOLEANITY_SOURCE),
    SourceFragment::new("booleanity_address", BOOLEANITY_ADDRESS_SOURCE),
    SourceFragment::new("instruction_ra_virtualization", INSTRUCTION_RA_SOURCE),
    SourceFragment::new("instruction_ra_sequence", INSTRUCTION_RA_SEQUENCE_SOURCE),
    SourceFragment::new("bytecode_cycle", BYTECODE_CYCLE_SOURCE),
    SourceFragment::new("bytecode_row", BYTECODE_ROW_SOURCE),
    SourceFragment::new("spartan_outer_uniskip", SPARTAN_OUTER_UNISKIP_SOURCE),
    SourceFragment::new("spartan_shift", SPARTAN_SHIFT_SOURCE),
    SourceFragment::new("instruction_input", INSTRUCTION_INPUT_SOURCE),
    SourceFragment::new("instruction_input_dense", INSTRUCTION_INPUT_DENSE_SOURCE),
    SourceFragment::new("address_cycle", ADDRESS_CYCLE_SOURCE),
    SourceFragment::new("outer_remainder", OUTER_REMAINDER_SOURCE),
    SourceFragment::new(
        "outer_remainder_padded_56",
        OUTER_REMAINDER_PADDED_56_SOURCE,
    ),
];

#[cfg(any(test, feature = "test-utils"))]
const OUTER_LIBRARY_SOURCE_FRAGMENTS: &[SourceFragment] = &[
    SourceFragment::new("fp128", FIELD_SOURCE),
    SourceFragment::new("simd_reduce", SIMD_REDUCE_SOURCE),
    SourceFragment::new("spartan_outer_common", SPARTAN_OUTER_COMMON_SOURCE),
    SourceFragment::new("outer_remainder", OUTER_REMAINDER_SOURCE),
    SourceFragment::new(
        "outer_remainder_padded_56",
        OUTER_REMAINDER_PADDED_56_SOURCE,
    ),
];

pub(super) fn library_source(offset: u32) -> String {
    assemble_library_source(offset, LIBRARY_SOURCE_FRAGMENTS, None)
}

pub(super) fn production_library_source(offset: u32) -> String {
    assemble_library_source_filtered(offset, LIBRARY_SOURCE_FRAGMENTS, None, true)
}

#[cfg(any(test, feature = "test-utils"))]
pub(super) fn library_source_with_outer(offset: u32, outer_source: &str) -> String {
    assemble_library_source(
        offset,
        LIBRARY_SOURCE_FRAGMENTS,
        Some(("outer_remainder_padded_56", outer_source)),
    )
}

#[cfg(any(test, feature = "test-utils"))]
pub(super) fn outer_library_source_with_outer(offset: u32, outer_source: &str) -> String {
    assemble_library_source(
        offset,
        OUTER_LIBRARY_SOURCE_FRAGMENTS,
        Some(("outer_remainder_padded_56", outer_source)),
    )
}

fn assemble_library_source(
    offset: u32,
    source_fragments: &[SourceFragment],
    replacement: Option<(&str, &str)>,
) -> String {
    assemble_library_source_filtered(offset, source_fragments, replacement, false)
}

fn assemble_library_source_filtered(
    offset: u32,
    source_fragments: &[SourceFragment],
    replacement: Option<(&str, &str)>,
    production_only: bool,
) -> String {
    let fragments = source_fragments
        .iter()
        .filter(|fragment| fragment.applies_to(offset) && (!production_only || fragment.production))
        .map(|fragment| match replacement {
            Some((id, source)) if fragment.id == id => source,
            _ => fragment.source,
        })
        .collect::<Vec<_>>();
    format!("#define SOLINAS_OFFSET {offset}u\n{}", fragments.join("\n"))
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

    fn expected_outer_library_source(offset: u32, outer_source: &str) -> String {
        format!(
            "#define SOLINAS_OFFSET {offset}u\n{FIELD_SOURCE}\n{SIMD_REDUCE_SOURCE}\n{SPARTAN_OUTER_COMMON_SOURCE}\n{OUTER_REMAINDER_SOURCE}\n{outer_source}"
        )
    }

    #[test]
    fn source_assembly_puts_the_runtime_fragment_last() {
        let generic = library_source(275);
        assert!(!generic.contains(BYTECODE_READ_RAF_ADDRESS_SOURCE));
        assert!(generic.ends_with(OUTER_REMAINDER_PADDED_56_SOURCE));

        let akita = library_source(AKITA_OFFSET_FFFFA7F7);
        assert!(akita.contains(BYTECODE_READ_RAF_ADDRESS_SOURCE));
        assert!(akita.ends_with(OUTER_REMAINDER_PADDED_56_SOURCE));
    }

    #[test]
    fn production_source_excludes_diagnostic_and_rejected_kernels() {
        let source = production_library_source(AKITA_OFFSET_FFFFA7F7);

        assert!(!source.contains(PROBE_SOURCE));
        assert!(!source.contains("solinas_ram_output_check_"));
        for required in [
            PRODUCT_REMAINDER_SOURCE,
            PRODUCT_INSTRUCTION_SERVICE_SOURCE,
            INSTRUCTION_READ_RAF_ADDRESS_SOURCE,
            REGISTERS_VAL_SOURCE,
        ] {
            assert!(source.contains(required));
        }
    }

    #[test]
    fn production_manifest_matches_source_assembly() {
        let production = LIBRARY_SOURCE_FRAGMENTS
            .iter()
            .filter(|fragment| fragment.production)
            .map(|fragment| fragment.id.to_owned())
            .collect::<Vec<_>>();
        let diagnostic = LIBRARY_SOURCE_FRAGMENTS
            .iter()
            .filter(|fragment| !fragment.production)
            .map(|fragment| fragment.id.to_owned())
            .collect::<Vec<_>>();

        assert_eq!(manifest_fragment_ids("metal_source_fragments"), production);
        assert_eq!(
            manifest_fragment_ids("diagnostic_source_fragments"),
            diagnostic
        );
        assert_eq!(
            manifest_fragment_ids("cpu_delegated_slots"),
            ["registers_read_write", "ram_output_check"]
        );
    }

    #[test]
    fn outer_source_assembly_closes_the_minimal_dependency_set() {
        let replacement = "kernel void replacement_outer() {}";
        let source = outer_library_source_with_outer(275, replacement);

        assert_eq!(source, expected_outer_library_source(275, replacement));
        assert!(source.contains(OUTER_REMAINDER_SOURCE));
        assert!(!source.contains(OUTER_REMAINDER_PADDED_56_SOURCE));
        assert!(!source.contains(INSTRUCTION_INPUT_SOURCE));
    }

    #[test]
    fn full_source_assembly_replaces_only_the_outer_fragment() {
        let replacement = "kernel void replacement_outer() {}";
        let source = library_source_with_outer(275, replacement);

        assert!(source.contains(replacement));
        assert!(source.contains(OUTER_REMAINDER_SOURCE));
        assert!(!source.contains(OUTER_REMAINDER_PADDED_56_SOURCE));
        assert!(source.contains(INSTRUCTION_INPUT_SOURCE));
        assert!(source.ends_with(replacement));
    }
}
