use super::types::{
    PackingAdviceKind, PackingAlphabet, PackingFactDomain, PackingFamilyId, PackingFamilySpec,
    PackingLayoutError, PackingValidityKind, PackingValidityRequirement, PackingWitnessLayout,
};
use jolt_riscv::CircuitFlags;
use thiserror::Error;

use crate::protocols::field_inline::formulas::lattice as field_lattice;
use crate::protocols::jolt::{AdviceClaimReductionLayout, JoltAdviceKind};

use super::super::ra::JoltRaPolynomialLayout;
use super::families::{packing_advice_kind, JoltPackingFamilyId};
use super::{advice_bytes_validity_requirement, unsigned_inc_validity_requirements};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldRdIncPacking {
    pub byte_width: usize,
    pub canonical_modulus: Option<u128>,
}

#[derive(Clone, Copy, Debug)]
pub struct JoltLatticePackingInputs<'a> {
    pub log_t: usize,
    pub log_k_chunk: usize,
    pub ra_layout: JoltRaPolynomialLayout,
    pub field_rd_inc: Option<FieldRdIncPacking>,
    pub untrusted_advice: Option<&'a AdviceClaimReductionLayout>,
}

#[derive(Clone, Copy, Debug)]
pub struct JoltLatticeValidityInputs<'a> {
    pub log_k_chunk: usize,
    pub field_rd_inc: Option<FieldRdIncPacking>,
    pub untrusted_advice: Option<&'a AdviceClaimReductionLayout>,
}

impl<'a> From<JoltLatticePackingInputs<'a>> for JoltLatticeValidityInputs<'a> {
    fn from(inputs: JoltLatticePackingInputs<'a>) -> Self {
        Self {
            log_k_chunk: inputs.log_k_chunk,
            field_rd_inc: inputs.field_rd_inc,
            untrusted_advice: inputs.untrusted_advice,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum JoltLatticeLayoutError {
    #[error(
        "unsigned increment chunk reconstruction requires log_k_chunk to divide 64, got {log_k_chunk}"
    )]
    InvalidUnsignedIncChunking { log_k_chunk: usize },
    #[error("lattice one-hot chunk size must be nonzero")]
    ZeroOneHotChunkSize,
    #[error("lattice one-hot chunk size {log_k_chunk} is too large")]
    OneHotChunkSizeTooLarge { log_k_chunk: usize },
    #[error("packed validity requirement alphabet size must be nonzero")]
    ZeroAlphabetSize,
    #[error("packed witness family {family:?} is not a Jolt lattice family")]
    UnsupportedPackingFamily { family: PackingFamilyId },
    #[error(transparent)]
    PackingLayout(#[from] PackingLayoutError),
}

pub fn derive_jolt_lattice_packed_witness_layout(
    inputs: JoltLatticePackingInputs<'_>,
) -> Result<PackingWitnessLayout, JoltLatticeLayoutError> {
    let trace = PackingFactDomain::TraceRows {
        log_t: inputs.log_t,
    };
    let ra_alphabet = one_hot_alphabet(inputs.log_k_chunk)?;
    let mut specs = Vec::new();
    specs.extend((0..inputs.ra_layout.instruction()).map(|index| {
        PackingFamilySpec::direct(
            JoltPackingFamilyId::InstructionRa { index }.into(),
            trace,
            1,
            ra_alphabet,
        )
    }));
    specs.extend((0..inputs.ra_layout.bytecode()).map(|index| {
        PackingFamilySpec::direct(
            JoltPackingFamilyId::BytecodeRa { index }.into(),
            trace,
            1,
            ra_alphabet,
        )
    }));
    specs.extend((0..inputs.ra_layout.ram()).map(|index| {
        PackingFamilySpec::direct(
            JoltPackingFamilyId::RamRa { index }.into(),
            trace,
            1,
            ra_alphabet,
        )
    }));

    let unsigned_inc_requirements = unsigned_inc_validity_requirements(inputs.log_k_chunk).ok_or(
        JoltLatticeLayoutError::InvalidUnsignedIncChunking {
            log_k_chunk: inputs.log_k_chunk,
        },
    )?;
    extend_validity_requirement_families(&mut specs, &unsigned_inc_requirements, trace)?;

    if let Some(field_rd_inc) = inputs.field_rd_inc {
        extend_validity_requirement_families(
            &mut specs,
            &field_rd_inc_validity_requirements(field_rd_inc),
            trace,
        )?;
    }

    if let Some(layout) = inputs.untrusted_advice {
        specs.push(advice_family(JoltAdviceKind::Untrusted, layout)?);
    }

    Ok(PackingWitnessLayout::new(specs)?)
}

pub fn derive_jolt_lattice_packed_validity_requirements(
    inputs: JoltLatticeValidityInputs<'_>,
) -> Result<Vec<PackingValidityRequirement>, JoltLatticeLayoutError> {
    let mut requirements = unsigned_inc_validity_requirements(inputs.log_k_chunk).ok_or(
        JoltLatticeLayoutError::InvalidUnsignedIncChunking {
            log_k_chunk: inputs.log_k_chunk,
        },
    )?;
    if let Some(field_rd_inc) = inputs.field_rd_inc {
        requirements.extend(field_rd_inc_validity_requirements(field_rd_inc));
    }
    if inputs.untrusted_advice.is_some() {
        requirements.push(advice_bytes_validity_requirement(JoltAdviceKind::Untrusted));
    }
    Ok(requirements)
}

pub fn lattice_validity_requirements_for_packed_witness_layout(
    layout: &PackingWitnessLayout,
) -> Result<Vec<PackingValidityRequirement>, JoltLatticeLayoutError> {
    let mut requirements = Vec::new();
    for family in &layout.families {
        let limbs = family.limbs;
        let alphabet_size = family.alphabet.size();
        let jolt_family = JoltPackingFamilyId::from_physical_id(&family.id)
            .ok_or(JoltLatticeLayoutError::UnsupportedPackingFamily { family: family.id })?;
        let requirement = match jolt_family {
            JoltPackingFamilyId::UnsignedIncChunk { index } => {
                Some(PackingValidityRequirement::exact_one_hot(
                    JoltPackingFamilyId::UnsignedIncChunk { index }.into(),
                    limbs,
                    alphabet_size,
                ))
            }
            JoltPackingFamilyId::UnsignedIncMsb => {
                Some(PackingValidityRequirement::boolean_indicator(
                    JoltPackingFamilyId::UnsignedIncMsb.into(),
                    limbs,
                    alphabet_size,
                    1,
                ))
            }
            JoltPackingFamilyId::FieldRdIncByte { index } => {
                Some(PackingValidityRequirement::exact_one_hot(
                    JoltPackingFamilyId::FieldRdIncByte { index }.into(),
                    limbs,
                    alphabet_size,
                ))
            }
            JoltPackingFamilyId::AdviceBytes { kind, index } => {
                Some(PackingValidityRequirement::exact_one_hot(
                    JoltPackingFamilyId::AdviceBytes { kind, index }.into(),
                    limbs,
                    alphabet_size,
                ))
            }
            JoltPackingFamilyId::BytecodeRegisterSelector { chunk, selector } => {
                Some(PackingValidityRequirement::optional_one_hot(
                    JoltPackingFamilyId::BytecodeRegisterSelector { chunk, selector }.into(),
                    limbs,
                    alphabet_size,
                ))
            }
            JoltPackingFamilyId::BytecodeCircuitFlag { chunk, flag } => {
                Some(PackingValidityRequirement::boolean_indicator(
                    JoltPackingFamilyId::BytecodeCircuitFlag { chunk, flag }.into(),
                    limbs,
                    alphabet_size,
                    1,
                ))
            }
            JoltPackingFamilyId::BytecodeInstructionFlag { chunk, flag } => {
                Some(PackingValidityRequirement::boolean_indicator(
                    JoltPackingFamilyId::BytecodeInstructionFlag { chunk, flag }.into(),
                    limbs,
                    alphabet_size,
                    1,
                ))
            }
            JoltPackingFamilyId::BytecodeLookupSelector { chunk } => {
                Some(PackingValidityRequirement::optional_one_hot(
                    JoltPackingFamilyId::BytecodeLookupSelector { chunk }.into(),
                    limbs,
                    alphabet_size,
                ))
            }
            JoltPackingFamilyId::BytecodeRafFlag { chunk } => {
                Some(PackingValidityRequirement::boolean_indicator(
                    JoltPackingFamilyId::BytecodeRafFlag { chunk }.into(),
                    limbs,
                    alphabet_size,
                    1,
                ))
            }
            JoltPackingFamilyId::BytecodeUnexpandedPcBytes { chunk } => {
                Some(PackingValidityRequirement::exact_one_hot(
                    JoltPackingFamilyId::BytecodeUnexpandedPcBytes { chunk }.into(),
                    limbs,
                    alphabet_size,
                ))
            }
            JoltPackingFamilyId::BytecodeImmBytes { chunk } => {
                Some(PackingValidityRequirement::exact_one_hot(
                    JoltPackingFamilyId::BytecodeImmBytes { chunk }.into(),
                    limbs,
                    alphabet_size,
                ))
            }
            JoltPackingFamilyId::ProgramImageInit => {
                Some(PackingValidityRequirement::exact_one_hot(
                    JoltPackingFamilyId::ProgramImageInit.into(),
                    limbs,
                    alphabet_size,
                ))
            }
            JoltPackingFamilyId::InstructionRa { .. }
            | JoltPackingFamilyId::BytecodeRa { .. }
            | JoltPackingFamilyId::RamRa { .. }
            | JoltPackingFamilyId::FieldRdIncSign
            | JoltPackingFamilyId::BytecodeChunk { .. } => None,
        };
        if let Some(requirement) = requirement {
            requirements.push(requirement);
        }
    }

    for family in &layout.families {
        let jolt_family = JoltPackingFamilyId::from_physical_id(&family.id)
            .ok_or(JoltLatticeLayoutError::UnsupportedPackingFamily { family: family.id })?;
        if let JoltPackingFamilyId::BytecodeCircuitFlag { chunk, flag } = jolt_family {
            if flag == CircuitFlags::Store as usize
                && layout
                    .family(
                        &JoltPackingFamilyId::BytecodeRegisterSelector { chunk, selector: 2 }
                            .into(),
                    )
                    .is_some()
            {
                requirements.push(PackingValidityRequirement::bytecode_store_rd_disjoint(
                    JoltPackingFamilyId::BytecodeCircuitFlag {
                        chunk,
                        flag: CircuitFlags::Store as usize,
                    }
                    .into(),
                ));
            }
        }
    }
    Ok(requirements)
}

pub fn packed_family_is_precommitted(family: &PackingFamilyId) -> bool {
    matches!(
        JoltPackingFamilyId::from_physical_id(family),
        Some(
            JoltPackingFamilyId::AdviceBytes {
                kind: JoltAdviceKind::Trusted,
                ..
            } | JoltPackingFamilyId::BytecodeChunk { .. }
                | JoltPackingFamilyId::BytecodeRegisterSelector { .. }
                | JoltPackingFamilyId::BytecodeCircuitFlag { .. }
                | JoltPackingFamilyId::BytecodeInstructionFlag { .. }
                | JoltPackingFamilyId::BytecodeLookupSelector { .. }
                | JoltPackingFamilyId::BytecodeRafFlag { .. }
                | JoltPackingFamilyId::BytecodeUnexpandedPcBytes { .. }
                | JoltPackingFamilyId::BytecodeImmBytes { .. }
                | JoltPackingFamilyId::ProgramImageInit
        )
    )
}

pub fn layout_has_field_rd_inc(layout: &PackingWitnessLayout) -> bool {
    layout.families.iter().any(|family| {
        matches!(
            JoltPackingFamilyId::from_physical_id(&family.id),
            Some(JoltPackingFamilyId::FieldRdIncByte { .. })
        )
    })
}

pub fn layout_has_advice(layout: &PackingWitnessLayout, kind: PackingAdviceKind) -> bool {
    layout.families.iter().any(|family| {
        matches!(
            JoltPackingFamilyId::from_physical_id(&family.id),
            Some(JoltPackingFamilyId::AdviceBytes {
                kind: family_kind,
                ..
            }) if packing_advice_kind(family_kind) == kind
        )
    })
}

pub fn packed_alphabet_with_size(size: usize) -> Result<PackingAlphabet, JoltLatticeLayoutError> {
    match size {
        0 => Err(JoltLatticeLayoutError::ZeroAlphabetSize),
        2 => Ok(PackingAlphabet::Bit),
        256 => Ok(PackingAlphabet::Byte),
        size => Ok(PackingAlphabet::Fixed { size }),
    }
}

fn advice_family(
    kind: JoltAdviceKind,
    layout: &AdviceClaimReductionLayout,
) -> Result<PackingFamilySpec, JoltLatticeLayoutError> {
    let packed_kind = packing_advice_kind(kind);
    let requirement = advice_bytes_validity_requirement(kind);
    Ok(PackingFamilySpec::direct(
        requirement.family,
        PackingFactDomain::AdviceBytes {
            kind: packed_kind,
            log_bytes: layout.advice_shape().total_vars() + 3,
        },
        requirement.limbs,
        packed_alphabet_with_size(requirement.alphabet_size)?,
    ))
}

fn extend_validity_requirement_families(
    specs: &mut Vec<PackingFamilySpec>,
    requirements: &[PackingValidityRequirement],
    domain: PackingFactDomain,
) -> Result<(), JoltLatticeLayoutError> {
    for requirement in requirements {
        if matches!(
            requirement.kind,
            PackingValidityKind::BytecodeStoreRdDisjoint
                | PackingValidityKind::FieldElementCanonicalBytes { .. }
        ) {
            continue;
        }
        specs.push(PackingFamilySpec::direct(
            requirement.family,
            domain,
            requirement.limbs,
            packed_alphabet_with_size(requirement.alphabet_size)?,
        ));
    }
    Ok(())
}

fn field_rd_inc_validity_requirements(
    field_rd_inc: FieldRdIncPacking,
) -> Vec<PackingValidityRequirement> {
    let mut requirements =
        field_lattice::field_rd_inc_validity_requirements(field_rd_inc.byte_width);
    if let Some(modulus) = field_rd_inc.canonical_modulus {
        requirements.push(field_lattice::field_rd_inc_canonical_bytes_requirement(
            field_rd_inc.byte_width,
            modulus,
        ));
    }
    requirements
}

fn one_hot_alphabet(log_k_chunk: usize) -> Result<PackingAlphabet, JoltLatticeLayoutError> {
    match log_k_chunk {
        0 => Err(JoltLatticeLayoutError::ZeroOneHotChunkSize),
        1 => Ok(PackingAlphabet::Bit),
        8 => Ok(PackingAlphabet::Byte),
        bits if bits < usize::BITS as usize => Ok(PackingAlphabet::Fixed {
            size: 1usize << bits,
        }),
        _ => Err(JoltLatticeLayoutError::OneHotChunkSizeTooLarge { log_k_chunk }),
    }
}
