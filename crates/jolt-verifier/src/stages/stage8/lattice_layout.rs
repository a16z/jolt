use crate::{
    config::{
        validate_protocol_config, AdviceLatticeConfig, FieldInlineLatticeConfig,
        IncrementCommitmentMode, JoltProtocolConfig, LatticeConfig, PackedWitnessConfig, PcsFamily,
        ProgramMode,
    },
    stages::PrecommittedSchedule,
    VerifierError,
};
use jolt_akita::AkitaField;
#[cfg(feature = "field-inline")]
use jolt_akita::AKITA_FIELD_MODULUS;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::formulas::lattice as field_lattice;
use jolt_claims::protocols::jolt::{
    advice_bytes_validity_requirement, formulas::ra::JoltRaPolynomialLayout,
    lattice_packed_validity_digest, unsigned_inc_validity_requirements, AdviceClaimReductionLayout,
    JoltAdviceKind, LatticePackedFamilyId, LatticePackedValidityKind,
    LatticePackedValidityRequirement,
};
use jolt_field::FixedByteSize;
use jolt_openings::{
    PackingAdviceKind, PackingAlphabet, PackingFactDomain, PackingFamilyId, PackingFamilySpec,
    PackingWitnessLayout,
};
use jolt_riscv::CircuitFlags;

use super::{
    invalid_lattice_config, invalid_precommitted_schedule, lattice_packing_advice_kind,
    lattice_packing_family_id,
};

pub fn derive_lattice_packed_witness_layout(
    config: &JoltProtocolConfig,
    log_t: usize,
    log_k_chunk: usize,
    ra_layout: JoltRaPolynomialLayout,
    precommitted: &PrecommittedSchedule,
) -> Result<PackingWitnessLayout, VerifierError> {
    if validate_protocol_config(config)? != PcsFamily::Lattice {
        return Err(invalid_lattice_config(
            "lattice packing witness layout derivation requires lattice PCS mode",
        ));
    }

    let trace = PackingFactDomain::TraceRows { log_t };
    let ra_alphabet = one_hot_alphabet(log_k_chunk)?;
    let mut specs = Vec::new();
    specs.extend((0..ra_layout.instruction()).map(|index| {
        PackingFamilySpec::direct(
            PackingFamilyId::InstructionRa { index },
            trace,
            1,
            ra_alphabet,
        )
    }));
    specs.extend((0..ra_layout.bytecode()).map(|index| {
        PackingFamilySpec::direct(PackingFamilyId::BytecodeRa { index }, trace, 1, ra_alphabet)
    }));
    specs.extend((0..ra_layout.ram()).map(|index| {
        PackingFamilySpec::direct(PackingFamilyId::RamRa { index }, trace, 1, ra_alphabet)
    }));

    let unsigned_inc_requirements =
        unsigned_inc_validity_requirements(log_k_chunk).ok_or_else(|| {
            invalid_lattice_config(format!(
                "unsigned increment chunk reconstruction requires log_k_chunk to divide 64, got {log_k_chunk}",
            ))
        })?;
    extend_validity_requirement_families(&mut specs, &unsigned_inc_requirements, trace)?;

    if config.lattice.field_inline.enabled {
        extend_field_rd_inc_families(&mut specs, trace)?;
    }

    if config.lattice.advice.untrusted {
        specs.push(advice_family(
            JoltAdviceKind::Untrusted,
            require_advice_layout(precommitted, JoltAdviceKind::Untrusted)?,
        )?);
    }

    let _ = precommitted.bytecode.as_ref().ok_or_else(|| {
        invalid_precommitted_schedule(
            "lattice committed-program mode requires a bytecode claim-reduction layout",
        )
    })?;

    let _ = precommitted.program_image.as_ref().ok_or_else(|| {
        invalid_precommitted_schedule(
            "lattice committed-program mode requires a program-image claim-reduction layout",
        )
    })?;

    PackingWitnessLayout::new(specs).map_err(|error| invalid_lattice_config(error.to_string()))
}

pub fn derive_lattice_packed_validity_requirements(
    config: &JoltProtocolConfig,
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
) -> Result<Vec<LatticePackedValidityRequirement>, VerifierError> {
    if validate_protocol_config(config)? != PcsFamily::Lattice {
        return Err(invalid_lattice_config(
            "lattice packed validity derivation requires lattice PCS mode",
        ));
    }

    let mut requirements = unsigned_inc_validity_requirements(log_k_chunk).ok_or_else(|| {
        invalid_lattice_config(format!(
            "unsigned increment chunk reconstruction requires log_k_chunk to divide 64, got {log_k_chunk}",
        ))
    })?;
    if config.lattice.field_inline.enabled {
        requirements.extend(field_rd_inc_validity_requirements());
    }
    if config.lattice.advice.untrusted {
        let _ = require_advice_layout(precommitted, JoltAdviceKind::Untrusted)?;
        requirements.push(advice_bytes_validity_requirement(JoltAdviceKind::Untrusted));
    }

    let _ = precommitted.bytecode.as_ref().ok_or_else(|| {
        invalid_precommitted_schedule(
            "lattice committed-program mode requires a bytecode claim-reduction layout",
        )
    })?;

    let _ = precommitted.program_image.as_ref().ok_or_else(|| {
        invalid_precommitted_schedule(
            "lattice committed-program mode requires a program-image claim-reduction layout",
        )
    })?;

    Ok(requirements)
}

pub fn lattice_protocol_config_for_packed_witness_layout(
    layout: &PackingWitnessLayout,
) -> JoltProtocolConfig {
    let validity_requirements = lattice_validity_requirements_for_packed_witness_layout(layout);
    let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
    config.lattice = LatticeConfig {
        program_mode: ProgramMode::Committed,
        increment_mode: IncrementCommitmentMode::FusedOneHot,
        packed_witness: PackedWitnessConfig {
            layout_digest: Some(layout.digest),
            d_pack: Some(layout.dimension),
            validity_digest: Some(lattice_packed_validity_digest(&validity_requirements)),
        },
        field_inline: FieldInlineLatticeConfig {
            enabled: layout_has_field_rd_inc(layout),
        },
        advice: AdviceLatticeConfig {
            trusted: false,
            untrusted: layout_has_advice(layout, PackingAdviceKind::Untrusted),
        },
    };
    config
}

pub fn lattice_validity_requirements_for_packed_witness_layout(
    layout: &PackingWitnessLayout,
) -> Vec<LatticePackedValidityRequirement> {
    let mut requirements = layout
        .families
        .iter()
        .filter_map(|family| {
            let limbs = family.limbs;
            let alphabet_size = family.alphabet.size();
            match family.id {
                PackingFamilyId::UnsignedIncChunk { index } => {
                    Some(LatticePackedValidityRequirement::exact_one_hot(
                        LatticePackedFamilyId::UnsignedIncChunk { index },
                        limbs,
                        alphabet_size,
                    ))
                }
                PackingFamilyId::UnsignedIncMsb => {
                    Some(LatticePackedValidityRequirement::boolean_indicator(
                        LatticePackedFamilyId::UnsignedIncMsb,
                        limbs,
                        alphabet_size,
                        1,
                    ))
                }
                PackingFamilyId::FieldRdIncByte { index } => {
                    Some(LatticePackedValidityRequirement::exact_one_hot(
                        LatticePackedFamilyId::FieldRdIncByte { index },
                        limbs,
                        alphabet_size,
                    ))
                }
                PackingFamilyId::AdviceBytes { kind, index } => {
                    Some(LatticePackedValidityRequirement::exact_one_hot(
                        LatticePackedFamilyId::AdviceBytes {
                            kind: jolt_advice_kind(kind),
                            index,
                        },
                        limbs,
                        alphabet_size,
                    ))
                }
                PackingFamilyId::BytecodeRegisterSelector { chunk, selector } => {
                    Some(LatticePackedValidityRequirement::optional_one_hot(
                        LatticePackedFamilyId::BytecodeRegisterSelector { chunk, selector },
                        limbs,
                        alphabet_size,
                    ))
                }
                PackingFamilyId::BytecodeCircuitFlag { chunk, flag } => {
                    Some(LatticePackedValidityRequirement::boolean_indicator(
                        LatticePackedFamilyId::BytecodeCircuitFlag { chunk, flag },
                        limbs,
                        alphabet_size,
                        1,
                    ))
                }
                PackingFamilyId::BytecodeInstructionFlag { chunk, flag } => {
                    Some(LatticePackedValidityRequirement::boolean_indicator(
                        LatticePackedFamilyId::BytecodeInstructionFlag { chunk, flag },
                        limbs,
                        alphabet_size,
                        1,
                    ))
                }
                PackingFamilyId::BytecodeLookupSelector { chunk } => {
                    Some(LatticePackedValidityRequirement::optional_one_hot(
                        LatticePackedFamilyId::BytecodeLookupSelector { chunk },
                        limbs,
                        alphabet_size,
                    ))
                }
                PackingFamilyId::BytecodeRafFlag { chunk } => {
                    Some(LatticePackedValidityRequirement::boolean_indicator(
                        LatticePackedFamilyId::BytecodeRafFlag { chunk },
                        limbs,
                        alphabet_size,
                        1,
                    ))
                }
                PackingFamilyId::BytecodeUnexpandedPcBytes { chunk } => {
                    Some(LatticePackedValidityRequirement::exact_one_hot(
                        LatticePackedFamilyId::BytecodeUnexpandedPcBytes { chunk },
                        limbs,
                        alphabet_size,
                    ))
                }
                PackingFamilyId::BytecodeImmBytes { chunk } => {
                    Some(LatticePackedValidityRequirement::exact_one_hot(
                        LatticePackedFamilyId::BytecodeImmBytes { chunk },
                        limbs,
                        alphabet_size,
                    ))
                }
                PackingFamilyId::ProgramImageInit => {
                    Some(LatticePackedValidityRequirement::exact_one_hot(
                        LatticePackedFamilyId::ProgramImageInit,
                        limbs,
                        alphabet_size,
                    ))
                }
                PackingFamilyId::InstructionRa { .. }
                | PackingFamilyId::BytecodeRa { .. }
                | PackingFamilyId::RamRa { .. }
                | PackingFamilyId::FieldRdIncSign
                | PackingFamilyId::BytecodeChunk { .. }
                | PackingFamilyId::Custom { .. } => None,
            }
        })
        .collect::<Vec<_>>();
    for family in &layout.families {
        let PackingFamilyId::BytecodeCircuitFlag { chunk, flag } = &family.id else {
            continue;
        };
        let chunk = *chunk;
        if *flag == CircuitFlags::Store as usize
            && layout
                .family(&PackingFamilyId::BytecodeRegisterSelector { chunk, selector: 2 })
                .is_some()
        {
            requirements.push(LatticePackedValidityRequirement::bytecode_store_rd_disjoint(chunk));
        }
    }
    requirements
}

pub fn validate_lattice_packed_witness_validity_config(
    config: &JoltProtocolConfig,
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
) -> Result<(), VerifierError> {
    let requirements =
        derive_lattice_packed_validity_requirements(config, log_k_chunk, precommitted)?;
    let digest = lattice_packed_validity_digest(&requirements);
    if config.lattice.packed_witness.validity_digest != Some(digest) {
        return Err(invalid_lattice_config(
            "configured lattice packed validity digest does not match derived requirements",
        ));
    }
    Ok(())
}

pub fn validate_lattice_packed_witness_layout_config(
    config: &JoltProtocolConfig,
    layout: &PackingWitnessLayout,
) -> Result<(), VerifierError> {
    if validate_protocol_config(config)? != PcsFamily::Lattice {
        return Err(invalid_lattice_config(
            "lattice packing witness layout validation requires lattice PCS mode",
        ));
    }
    if config.lattice.packed_witness.layout_digest != Some(layout.digest) {
        return Err(invalid_lattice_config(
            "configured lattice packing witness layout digest does not match derived layout",
        ));
    }
    if config.lattice.packed_witness.d_pack != Some(layout.dimension) {
        return Err(invalid_lattice_config(
            "configured lattice packing witness D_pack does not match derived layout",
        ));
    }
    for family in &layout.families {
        if packed_family_is_precommitted(&family.id) {
            return Err(invalid_lattice_config(format!(
                "precommitted family {:?} cannot be included in the lattice packing witness layout",
                family.id
            )));
        }
    }
    Ok(())
}

fn packed_family_is_precommitted(family: &PackingFamilyId) -> bool {
    matches!(
        family,
        PackingFamilyId::AdviceBytes {
            kind: PackingAdviceKind::Trusted,
            ..
        } | PackingFamilyId::BytecodeChunk { .. }
            | PackingFamilyId::BytecodeRegisterSelector { .. }
            | PackingFamilyId::BytecodeCircuitFlag { .. }
            | PackingFamilyId::BytecodeInstructionFlag { .. }
            | PackingFamilyId::BytecodeLookupSelector { .. }
            | PackingFamilyId::BytecodeRafFlag { .. }
            | PackingFamilyId::BytecodeUnexpandedPcBytes { .. }
            | PackingFamilyId::BytecodeImmBytes { .. }
            | PackingFamilyId::ProgramImageInit
    )
}

fn advice_family(
    kind: JoltAdviceKind,
    layout: &AdviceClaimReductionLayout,
) -> Result<PackingFamilySpec, VerifierError> {
    let packed_kind = lattice_packing_advice_kind(kind);
    let requirement = advice_bytes_validity_requirement(kind);
    Ok(PackingFamilySpec::direct(
        lattice_packing_family_id(&requirement.family),
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
    requirements: &[LatticePackedValidityRequirement],
    domain: PackingFactDomain,
) -> Result<(), VerifierError> {
    for requirement in requirements {
        if matches!(
            requirement.kind,
            LatticePackedValidityKind::BytecodeStoreRdDisjoint
                | LatticePackedValidityKind::FieldElementCanonicalBytes { .. }
        ) {
            continue;
        }
        specs.push(PackingFamilySpec::direct(
            lattice_packing_family_id(&requirement.family),
            domain,
            requirement.limbs,
            packed_alphabet_with_size(requirement.alphabet_size)?,
        ));
    }
    Ok(())
}

#[cfg(feature = "field-inline")]
fn field_rd_inc_validity_requirements() -> Vec<LatticePackedValidityRequirement> {
    let mut requirements = field_lattice::field_rd_inc_validity_requirements(AkitaField::NUM_BYTES);
    requirements.push(field_lattice::field_rd_inc_canonical_bytes_requirement(
        AkitaField::NUM_BYTES,
        AKITA_FIELD_MODULUS,
    ));
    requirements
}

#[cfg(not(feature = "field-inline"))]
fn field_rd_inc_validity_requirements() -> Vec<LatticePackedValidityRequirement> {
    (0..AkitaField::NUM_BYTES)
        .map(|index| {
            LatticePackedValidityRequirement::exact_one_hot(
                LatticePackedFamilyId::FieldRdIncByte { index },
                1,
                256,
            )
        })
        .collect()
}

#[cfg(feature = "field-inline")]
fn extend_field_rd_inc_families(
    specs: &mut Vec<PackingFamilySpec>,
    domain: PackingFactDomain,
) -> Result<(), VerifierError> {
    extend_validity_requirement_families(specs, &field_rd_inc_validity_requirements(), domain)
}

#[cfg(not(feature = "field-inline"))]
fn extend_field_rd_inc_families(
    specs: &mut Vec<PackingFamilySpec>,
    domain: PackingFactDomain,
) -> Result<(), VerifierError> {
    extend_validity_requirement_families(specs, &field_rd_inc_validity_requirements(), domain)
}

fn require_advice_layout(
    precommitted: &PrecommittedSchedule,
    kind: JoltAdviceKind,
) -> Result<&AdviceClaimReductionLayout, VerifierError> {
    precommitted.advice(kind).ok_or_else(|| {
        invalid_precommitted_schedule(format!(
            "lattice advice mode requires a {kind:?} advice claim-reduction layout",
        ))
    })
}

fn jolt_advice_kind(kind: PackingAdviceKind) -> JoltAdviceKind {
    match kind {
        PackingAdviceKind::Trusted => JoltAdviceKind::Trusted,
        PackingAdviceKind::Untrusted => JoltAdviceKind::Untrusted,
    }
}

fn layout_has_field_rd_inc(layout: &PackingWitnessLayout) -> bool {
    layout
        .families
        .iter()
        .any(|family| matches!(family.id, PackingFamilyId::FieldRdIncByte { .. }))
}

fn layout_has_advice(layout: &PackingWitnessLayout, kind: PackingAdviceKind) -> bool {
    layout.families.iter().any(|family| {
        matches!(
            family.id,
            PackingFamilyId::AdviceBytes {
                kind: family_kind,
                ..
            } if family_kind == kind
        )
    })
}

fn one_hot_alphabet(log_k_chunk: usize) -> Result<PackingAlphabet, VerifierError> {
    match log_k_chunk {
        0 => Err(invalid_lattice_config(
            "lattice one-hot chunk size must be nonzero",
        )),
        1 => Ok(PackingAlphabet::Bit),
        8 => Ok(PackingAlphabet::Byte),
        bits if bits < usize::BITS as usize => Ok(PackingAlphabet::Fixed {
            size: 1usize << bits,
        }),
        _ => Err(invalid_lattice_config(
            "lattice one-hot chunk size is too large",
        )),
    }
}

pub(super) fn packed_alphabet_with_size(size: usize) -> Result<PackingAlphabet, VerifierError> {
    match size {
        0 => Err(invalid_lattice_config(
            "packed validity requirement alphabet size must be nonzero",
        )),
        2 => Ok(PackingAlphabet::Bit),
        256 => Ok(PackingAlphabet::Byte),
        size => Ok(PackingAlphabet::Fixed { size }),
    }
}
