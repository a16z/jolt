const FIELD_SOURCE: &str = include_str!("fp128.metal");
#[cfg(feature = "test-utils")]
const HAMMING_WEIGHT_CLAIM_REDUCTION_SOURCE: &str = super::hamming_weight_claim_reduction::SOURCE;
const HALF_WIDTH_PROBE_SOURCE: &str = super::half_width_probe::SOURCE;
const SIMD_REDUCE_SOURCE: &str = include_str!("simd_reduce.metal");
const REGISTERS_CLAIM_REDUCTION_SOURCE: &str = super::registers_claim_reduction::SOURCE;
const RAM_RA_CLAIM_REDUCTION_SOURCE: &str = super::ram_ra_claim_reduction::SOURCE;
const DEFERRED_SUM_SOURCE: &str = include_str!("deferred_sum.metal");
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
const INSTRUCTION_CLAIM_SUCCESSOR_SOURCE: &str =
    include_str!("instruction_claim_reduction_successor/shader.metal");
const PRODUCT_UNISKIP_SOURCE: &str = super::product_uniskip::SOURCE;
const RAM_OUTPUT_CHECK_SOURCE: &str = super::ram_output_check::SOURCE;
const RAM_RAF_EVALUATION_SOURCE: &str = super::ram_raf_evaluation::SOURCE;
const RAM_VAL_CHECK_SOURCE: &str = super::ram_val_check::SOURCE;
const REGISTERS_READ_WRITE_SOURCE: &str = include_str!("registers_read_write/shader.metal");
const REGISTERS_READ_WRITE_DENSE_SOURCE: &str = super::registers_read_write_dense::SOURCE;
const REGISTERS_VAL_SOURCE: &str = include_str!("registers_val/shader.metal");
const BOOLEANITY_SOURCE: &str = include_str!("booleanity/shader.metal");
const BOOLEANITY_ADDRESS_SOURCE: &str = include_str!("booleanity_address/shader.metal");
const INSTRUCTION_RA_SOURCE: &str = include_str!("instruction_ra_virtualization/shader.metal");
const INSTRUCTION_RA_SEQUENCE_SOURCE: &str = include_str!("instruction_ra_sequence/shader.metal");
const INSTRUCTION_INPUT_SOURCE: &str = include_str!("instruction_input/shader.metal");
const INSTRUCTION_INPUT_SUCCESSOR_SOURCE: &str = super::instruction_input_successor::SOURCE;
const BYTECODE_CYCLE_SOURCE: &str = include_str!("bytecode_cycle/shader.metal");
const BYTECODE_ROW_SOURCE: &str = include_str!("bytecode_row/shader.metal");
const SPARTAN_OUTER_UNISKIP_SOURCE: &str = include_str!("spartan_outer_uniskip/shader.metal");
const SPARTAN_SHIFT_SOURCE: &str = super::spartan_shift::SOURCE;
const OUTER_REMAINDER_SOURCE: &str = super::outer_remainder::SOURCE;
const OUTER_REMAINDER_PADDED_56_SOURCE: &str = super::outer_remainder::PADDED_56_SOURCE;

struct SourceFragment {
    id: &'static str,
    source: &'static str,
}

impl SourceFragment {
    const fn new(id: &'static str, source: &'static str) -> Self {
        Self { id, source }
    }
}

const LIBRARY_SOURCE_FRAGMENTS: &[SourceFragment] = &[
    SourceFragment::new("fp128", FIELD_SOURCE),
    SourceFragment::new("half_width_probe", HALF_WIDTH_PROBE_SOURCE),
    SourceFragment::new("simd_reduce", SIMD_REDUCE_SOURCE),
    SourceFragment::new(
        "registers_claim_reduction",
        REGISTERS_CLAIM_REDUCTION_SOURCE,
    ),
    SourceFragment::new("ram_ra_claim_reduction", RAM_RA_CLAIM_REDUCTION_SOURCE),
    SourceFragment::new("deferred_sum", DEFERRED_SUM_SOURCE),
    SourceFragment::new("spartan_outer_common", SPARTAN_OUTER_COMMON_SOURCE),
    SourceFragment::new("booleanity_common", BOOLEANITY_COMMON_SOURCE),
    SourceFragment::new("instruction_ra_common", INSTRUCTION_RA_COMMON_SOURCE),
    SourceFragment::new(
        "instruction_claim_reduction",
        INSTRUCTION_CLAIM_REDUCTION_SOURCE,
    ),
    SourceFragment::new("address_raf", ADDRESS_RAF_SOURCE),
    SourceFragment::new("address_raf_direct", ADDRESS_RAF_DIRECT_SOURCE),
    SourceFragment::new("address_suffix", ADDRESS_SUFFIX_SOURCE),
    SourceFragment::new("address_suffix_full", ADDRESS_SUFFIX_FULL_SOURCE),
    SourceFragment::new("probes", PROBE_SOURCE),
    SourceFragment::new("product5", PRODUCT5_SOURCE),
    SourceFragment::new("product_remainder", PRODUCT_REMAINDER_SOURCE),
    SourceFragment::new(
        "instruction_claim_reduction_successor",
        INSTRUCTION_CLAIM_SUCCESSOR_SOURCE,
    ),
    SourceFragment::new("product_uniskip", PRODUCT_UNISKIP_SOURCE),
    SourceFragment::new("ram_output_check", RAM_OUTPUT_CHECK_SOURCE),
    SourceFragment::new("ram_raf_evaluation", RAM_RAF_EVALUATION_SOURCE),
    SourceFragment::new("ram_val_check", RAM_VAL_CHECK_SOURCE),
    SourceFragment::new("registers_read_write", REGISTERS_READ_WRITE_SOURCE),
    SourceFragment::new(
        "registers_read_write_dense",
        REGISTERS_READ_WRITE_DENSE_SOURCE,
    ),
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
    SourceFragment::new(
        "instruction_input_successor",
        INSTRUCTION_INPUT_SUCCESSOR_SOURCE,
    ),
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

#[cfg(feature = "test-utils")]
pub(super) fn hamming_weight_claim_reduction_probe_source(offset: u32) -> String {
    assemble_library_source(
        offset,
        &[
            SourceFragment::new("fp128", FIELD_SOURCE),
            SourceFragment::new(
                "hamming_weight_claim_reduction",
                HAMMING_WEIGHT_CLAIM_REDUCTION_SOURCE,
            ),
        ],
        None,
    )
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
    let fragments = source_fragments
        .iter()
        .map(|fragment| match replacement {
            Some((id, source)) if fragment.id == id => source,
            _ => fragment.source,
        })
        .collect::<Vec<_>>();
    format!("#define SOLINAS_OFFSET {offset}u\n{}", fragments.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected_library_source(offset: u32) -> String {
        format!(
            "#define SOLINAS_OFFSET {offset}u\n{FIELD_SOURCE}\n{HALF_WIDTH_PROBE_SOURCE}\n{SIMD_REDUCE_SOURCE}\n{REGISTERS_CLAIM_REDUCTION_SOURCE}\n{RAM_RA_CLAIM_REDUCTION_SOURCE}\n{DEFERRED_SUM_SOURCE}\n{SPARTAN_OUTER_COMMON_SOURCE}\n{BOOLEANITY_COMMON_SOURCE}\n{INSTRUCTION_RA_COMMON_SOURCE}\n{INSTRUCTION_CLAIM_REDUCTION_SOURCE}\n{ADDRESS_RAF_SOURCE}\n{ADDRESS_RAF_DIRECT_SOURCE}\n{ADDRESS_SUFFIX_SOURCE}\n{ADDRESS_SUFFIX_FULL_SOURCE}\n{PROBE_SOURCE}\n{PRODUCT5_SOURCE}\n{PRODUCT_REMAINDER_SOURCE}\n{INSTRUCTION_CLAIM_SUCCESSOR_SOURCE}\n{PRODUCT_UNISKIP_SOURCE}\n{RAM_OUTPUT_CHECK_SOURCE}\n{RAM_RAF_EVALUATION_SOURCE}\n{RAM_VAL_CHECK_SOURCE}\n{REGISTERS_READ_WRITE_SOURCE}\n{REGISTERS_READ_WRITE_DENSE_SOURCE}\n{REGISTERS_VAL_SOURCE}\n{BOOLEANITY_SOURCE}\n{BOOLEANITY_ADDRESS_SOURCE}\n{INSTRUCTION_RA_SOURCE}\n{INSTRUCTION_RA_SEQUENCE_SOURCE}\n{BYTECODE_CYCLE_SOURCE}\n{BYTECODE_ROW_SOURCE}\n{SPARTAN_OUTER_UNISKIP_SOURCE}\n{SPARTAN_SHIFT_SOURCE}\n{INSTRUCTION_INPUT_SOURCE}\n{INSTRUCTION_INPUT_SUCCESSOR_SOURCE}\n{ADDRESS_CYCLE_SOURCE}\n{OUTER_REMAINDER_SOURCE}\n{OUTER_REMAINDER_PADDED_56_SOURCE}"
        )
    }

    fn expected_outer_library_source(offset: u32, outer_source: &str) -> String {
        format!(
            "#define SOLINAS_OFFSET {offset}u\n{FIELD_SOURCE}\n{SIMD_REDUCE_SOURCE}\n{SPARTAN_OUTER_COMMON_SOURCE}\n{OUTER_REMAINDER_SOURCE}\n{outer_source}"
        )
    }

    #[test]
    fn source_assembly_puts_the_runtime_fragment_last() {
        for offset in [275, 0xffff_a7f7] {
            assert_eq!(
                library_source(offset).as_bytes(),
                expected_library_source(offset).as_bytes()
            );
        }
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
    fn hamming_weight_probe_source_has_only_its_field_dependency() {
        let source = hamming_weight_claim_reduction_probe_source(0xffff_a7f7);

        assert!(source.starts_with("#define SOLINAS_OFFSET 4294944759u\n"));
        assert!(source.contains(FIELD_SOURCE));
        assert!(source.ends_with(HAMMING_WEIGHT_CLAIM_REDUCTION_SOURCE));
        assert!(!source.contains(HALF_WIDTH_PROBE_SOURCE));
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
