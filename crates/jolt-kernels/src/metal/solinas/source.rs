const FIELD_SOURCE: &str = include_str!("fp128.metal");
const DEFERRED_SUM_SOURCE: &str = include_str!("deferred_sum.metal");
const ADDRESS_RAF_SOURCE: &str = include_str!("address_raf.metal");
const ADDRESS_RAF_DIRECT_SOURCE: &str = include_str!("address_raf_direct.metal");
const ADDRESS_SUFFIX_SOURCE: &str = include_str!("address_suffix.metal");
const ADDRESS_SUFFIX_FULL_SOURCE: &str = include_str!("address_suffix_full.metal");
const ADDRESS_CYCLE_SOURCE: &str = include_str!("address_cycle.metal");
const PROBE_SOURCE: &str = include_str!("probes.metal");
const PRODUCT5_SOURCE: &str = include_str!("product5.metal");
const BOOLEANITY_SOURCE: &str = include_str!("booleanity.metal");
const BOOLEANITY_ADDRESS_SOURCE: &str = include_str!("booleanity_address.metal");
const INSTRUCTION_RA_SOURCE: &str = include_str!("instruction_ra_virtualization.metal");
const INSTRUCTION_RA_SEQUENCE_SOURCE: &str = include_str!("instruction_ra_sequence.metal");
const INSTRUCTION_INPUT_SOURCE: &str = include_str!("instruction_input.metal");
const BYTECODE_CYCLE_SOURCE: &str = include_str!("bytecode_cycle.metal");
const BYTECODE_ROW_SOURCE: &str = include_str!("bytecode_row.metal");
const SPARTAN_OUTER_UNISKIP_SOURCE: &str = include_str!("spartan_outer_uniskip.metal");
const OUTER_REMAINDER_SOURCE: &str = super::outer_remainder::SOURCE;

const LIBRARY_SOURCE_FRAGMENTS: &[&str] = &[
    FIELD_SOURCE,
    DEFERRED_SUM_SOURCE,
    ADDRESS_RAF_SOURCE,
    ADDRESS_RAF_DIRECT_SOURCE,
    ADDRESS_SUFFIX_SOURCE,
    ADDRESS_SUFFIX_FULL_SOURCE,
    PROBE_SOURCE,
    PRODUCT5_SOURCE,
    BOOLEANITY_SOURCE,
    BOOLEANITY_ADDRESS_SOURCE,
    INSTRUCTION_RA_SOURCE,
    INSTRUCTION_RA_SEQUENCE_SOURCE,
    BYTECODE_CYCLE_SOURCE,
    BYTECODE_ROW_SOURCE,
    SPARTAN_OUTER_UNISKIP_SOURCE,
    OUTER_REMAINDER_SOURCE,
    INSTRUCTION_INPUT_SOURCE,
    ADDRESS_CYCLE_SOURCE,
];

pub(super) fn library_source(offset: u32) -> String {
    format!(
        "#define SOLINAS_OFFSET {offset}u\n{}",
        LIBRARY_SOURCE_FRAGMENTS.join("\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_library_source(offset: u32) -> String {
        format!(
            "#define SOLINAS_OFFSET {offset}u\n{FIELD_SOURCE}\n{DEFERRED_SUM_SOURCE}\n{ADDRESS_RAF_SOURCE}\n{ADDRESS_RAF_DIRECT_SOURCE}\n{ADDRESS_SUFFIX_SOURCE}\n{ADDRESS_SUFFIX_FULL_SOURCE}\n{PROBE_SOURCE}\n{PRODUCT5_SOURCE}\n{BOOLEANITY_SOURCE}\n{BOOLEANITY_ADDRESS_SOURCE}\n{INSTRUCTION_RA_SOURCE}\n{INSTRUCTION_RA_SEQUENCE_SOURCE}\n{BYTECODE_CYCLE_SOURCE}\n{BYTECODE_ROW_SOURCE}\n{SPARTAN_OUTER_UNISKIP_SOURCE}\n{OUTER_REMAINDER_SOURCE}\n{INSTRUCTION_INPUT_SOURCE}\n{ADDRESS_CYCLE_SOURCE}"
        )
    }

    #[test]
    fn source_assembly_is_byte_equivalent_to_the_legacy_builder() {
        for offset in [275, 0xffff_a7f7] {
            assert_eq!(
                library_source(offset).as_bytes(),
                legacy_library_source(offset).as_bytes()
            );
        }
    }
}
