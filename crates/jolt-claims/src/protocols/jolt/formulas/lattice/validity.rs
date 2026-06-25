use jolt_lookup_tables::{LookupTableKind, XLEN};
use jolt_openings::{PackingFamilyId, PackingValidityRequirement};
use jolt_riscv::{CircuitFlags, NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};

use crate::protocols::jolt::JoltAdviceKind;

use super::super::dimensions::REGISTER_ADDRESS_BITS;
use super::families::JoltPackingFamilyId;
use super::openings::unsigned_inc_chunking;

pub fn unsigned_inc_validity_requirements(
    log_k_chunk: usize,
) -> Option<Vec<PackingValidityRequirement>> {
    let chunking = unsigned_inc_chunking(log_k_chunk)?;
    let mut requirements = (0..chunking.chunk_count)
        .map(|index| {
            PackingValidityRequirement::exact_one_hot(
                JoltPackingFamilyId::UnsignedIncChunk { index }.into(),
                1,
                chunking.alphabet_size,
            )
        })
        .collect::<Vec<_>>();
    requirements.push(PackingValidityRequirement::boolean_indicator(
        JoltPackingFamilyId::UnsignedIncMsb.into(),
        1,
        2,
        1,
    ));
    Some(requirements)
}

pub fn advice_bytes_validity_requirement(kind: JoltAdviceKind) -> PackingValidityRequirement {
    byte_validity_requirement(
        JoltPackingFamilyId::AdviceBytes { kind, index: 0 }.into(),
        1,
    )
}

pub fn program_image_validity_requirement() -> PackingValidityRequirement {
    byte_validity_requirement(JoltPackingFamilyId::ProgramImageInit.into(), 8)
}

pub fn bytecode_validity_requirements(
    chunk: usize,
    field_byte_width: usize,
) -> Vec<PackingValidityRequirement> {
    let register_count = 1usize << REGISTER_ADDRESS_BITS;
    let mut requirements = Vec::new();
    for selector in 0..3 {
        requirements.push(PackingValidityRequirement::optional_one_hot(
            JoltPackingFamilyId::BytecodeRegisterSelector { chunk, selector }.into(),
            1,
            register_count,
        ));
    }
    for flag in 0..NUM_CIRCUIT_FLAGS {
        requirements.push(PackingValidityRequirement::boolean_indicator(
            JoltPackingFamilyId::BytecodeCircuitFlag { chunk, flag }.into(),
            1,
            2,
            1,
        ));
    }
    for flag in 0..NUM_INSTRUCTION_FLAGS {
        requirements.push(PackingValidityRequirement::boolean_indicator(
            JoltPackingFamilyId::BytecodeInstructionFlag { chunk, flag }.into(),
            1,
            2,
            1,
        ));
    }
    requirements.push(PackingValidityRequirement::optional_one_hot(
        JoltPackingFamilyId::BytecodeLookupSelector { chunk }.into(),
        1,
        LookupTableKind::<XLEN>::COUNT.next_power_of_two(),
    ));
    requirements.push(PackingValidityRequirement::boolean_indicator(
        JoltPackingFamilyId::BytecodeRafFlag { chunk }.into(),
        1,
        2,
        1,
    ));
    requirements.push(byte_validity_requirement(
        JoltPackingFamilyId::BytecodeUnexpandedPcBytes { chunk }.into(),
        8,
    ));
    requirements.push(byte_validity_requirement(
        JoltPackingFamilyId::BytecodeImmBytes { chunk }.into(),
        field_byte_width,
    ));
    requirements.push(PackingValidityRequirement::bytecode_store_rd_disjoint(
        JoltPackingFamilyId::BytecodeCircuitFlag {
            chunk,
            flag: CircuitFlags::Store as usize,
        }
        .into(),
    ));
    requirements
}

pub fn bytecode_imm_canonical_bytes_requirement(
    chunk: usize,
    byte_width: usize,
    modulus: u128,
) -> PackingValidityRequirement {
    PackingValidityRequirement::field_element_canonical_bytes(
        JoltPackingFamilyId::BytecodeImmBytes { chunk }.into(),
        byte_width,
        modulus,
    )
}

fn byte_validity_requirement(family: PackingFamilyId, limbs: usize) -> PackingValidityRequirement {
    PackingValidityRequirement::exact_one_hot(family, limbs, 256)
}
