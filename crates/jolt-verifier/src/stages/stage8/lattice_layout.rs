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
    derive_jolt_lattice_packed_validity_requirements, derive_jolt_lattice_packed_witness_layout,
    formulas::ra::JoltRaPolynomialLayout,
    lattice_validity_requirements_for_packed_witness_layout as claims_lattice_validity_requirements_for_packed_witness_layout,
    layout_has_advice, packed_family_is_precommitted, AdviceClaimReductionLayout,
    FieldRdIncPacking, JoltAdviceKind, JoltLatticeLayoutError, JoltLatticePackingInputs,
    JoltLatticeValidityInputs, JoltPackingFamilyId,
};
use jolt_field::FixedByteSize;
use jolt_openings::{
    packing_validity_digest, PackingAdviceKind, PackingValidityRequirement, PackingWitnessLayout,
};

use super::{invalid_lattice_config, invalid_precommitted_schedule};

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

    let inputs = lattice_packing_inputs(config, log_t, log_k_chunk, ra_layout, precommitted)?;
    require_committed_program_schedule(precommitted)?;
    derive_jolt_lattice_packed_witness_layout(inputs).map_err(map_lattice_layout_error)
}

pub fn derive_lattice_packed_validity_requirements(
    config: &JoltProtocolConfig,
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
) -> Result<Vec<PackingValidityRequirement>, VerifierError> {
    if validate_protocol_config(config)? != PcsFamily::Lattice {
        return Err(invalid_lattice_config(
            "lattice packed validity derivation requires lattice PCS mode",
        ));
    }

    let inputs = lattice_validity_inputs(config, log_k_chunk, precommitted)?;
    require_committed_program_schedule(precommitted)?;
    derive_jolt_lattice_packed_validity_requirements(inputs).map_err(map_lattice_layout_error)
}

pub fn lattice_protocol_config_for_packed_witness_layout(
    layout: &PackingWitnessLayout,
) -> Result<JoltProtocolConfig, VerifierError> {
    let field_rd_inc_byte_width = field_rd_inc_byte_width(layout)?;
    let mut validity_requirements =
        lattice_validity_requirements_for_packed_witness_layout(layout)?;
    extend_layout_derived_field_rd_inc_validity(
        field_rd_inc_byte_width,
        &mut validity_requirements,
    )?;
    let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
    config.lattice = LatticeConfig {
        program_mode: ProgramMode::Committed,
        increment_mode: IncrementCommitmentMode::FusedOneHot,
        packed_witness: PackedWitnessConfig {
            layout_digest: Some(layout.digest),
            d_pack: Some(layout.dimension),
            validity_digest: Some(packing_validity_digest(&validity_requirements)),
        },
        field_inline: FieldInlineLatticeConfig {
            enabled: field_rd_inc_byte_width.is_some(),
        },
        advice: AdviceLatticeConfig {
            trusted: false,
            untrusted: layout_has_advice(layout, PackingAdviceKind::Untrusted),
        },
    };
    Ok(config)
}

pub fn lattice_validity_requirements_for_packed_witness_layout(
    layout: &PackingWitnessLayout,
) -> Result<Vec<PackingValidityRequirement>, VerifierError> {
    claims_lattice_validity_requirements_for_packed_witness_layout(layout)
        .map_err(|error| invalid_lattice_config(error.to_string()))
}

pub fn validate_lattice_packed_witness_validity_config(
    config: &JoltProtocolConfig,
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
) -> Result<(), VerifierError> {
    let requirements =
        derive_lattice_packed_validity_requirements(config, log_k_chunk, precommitted)?;
    let digest = packing_validity_digest(&requirements);
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

fn lattice_packing_inputs<'a>(
    config: &JoltProtocolConfig,
    log_t: usize,
    log_k_chunk: usize,
    ra_layout: JoltRaPolynomialLayout,
    precommitted: &'a PrecommittedSchedule,
) -> Result<JoltLatticePackingInputs<'a>, VerifierError> {
    let untrusted_advice = if config.lattice.advice.untrusted {
        Some(require_advice_layout(
            precommitted,
            JoltAdviceKind::Untrusted,
        )?)
    } else {
        None
    };
    Ok(JoltLatticePackingInputs {
        log_t,
        log_k_chunk,
        ra_layout,
        field_rd_inc: field_rd_inc_packing(config),
        untrusted_advice,
    })
}

fn field_rd_inc_packing(config: &JoltProtocolConfig) -> Option<FieldRdIncPacking> {
    config
        .lattice
        .field_inline
        .enabled
        .then(field_rd_inc_packing_config)
}

fn field_rd_inc_byte_width(layout: &PackingWitnessLayout) -> Result<Option<usize>, VerifierError> {
    let mut indices = layout
        .families
        .iter()
        .filter_map(
            |family| match JoltPackingFamilyId::from_physical_id(&family.id) {
                Some(JoltPackingFamilyId::FieldRdIncByte { index }) => Some(index),
                _ => None,
            },
        )
        .collect::<Vec<_>>();
    if indices.is_empty() {
        return Ok(None);
    }
    indices.sort_unstable();
    for (expected, index) in indices.iter().copied().enumerate() {
        if index != expected {
            return Err(invalid_lattice_config(
                "lattice field-inline byte families must be contiguous from zero",
            ));
        }
    }
    Ok(Some(indices.len()))
}

#[cfg(feature = "field-inline")]
fn extend_layout_derived_field_rd_inc_validity(
    byte_width: Option<usize>,
    requirements: &mut Vec<PackingValidityRequirement>,
) -> Result<(), VerifierError> {
    if let Some(byte_width) = byte_width {
        if byte_width != AkitaField::NUM_BYTES {
            return Err(invalid_lattice_config(
                "lattice field-inline byte family count must match the Akita field byte width",
            ));
        }
        requirements.push(field_lattice::field_rd_inc_canonical_bytes_requirement(
            byte_width,
            AKITA_FIELD_MODULUS,
        ));
    }
    Ok(())
}

#[cfg(not(feature = "field-inline"))]
fn extend_layout_derived_field_rd_inc_validity(
    byte_width: Option<usize>,
    _requirements: &mut Vec<PackingValidityRequirement>,
) -> Result<(), VerifierError> {
    if byte_width.is_some() {
        return Err(invalid_lattice_config(
            "lattice field-inline families require the field-inline feature",
        ));
    }
    Ok(())
}

fn lattice_validity_inputs<'a>(
    config: &JoltProtocolConfig,
    log_k_chunk: usize,
    precommitted: &'a PrecommittedSchedule,
) -> Result<JoltLatticeValidityInputs<'a>, VerifierError> {
    let untrusted_advice = if config.lattice.advice.untrusted {
        Some(require_advice_layout(
            precommitted,
            JoltAdviceKind::Untrusted,
        )?)
    } else {
        None
    };
    Ok(JoltLatticeValidityInputs {
        log_k_chunk,
        field_rd_inc: field_rd_inc_packing(config),
        untrusted_advice,
    })
}

#[cfg(feature = "field-inline")]
fn field_rd_inc_packing_config() -> FieldRdIncPacking {
    FieldRdIncPacking {
        byte_width: AkitaField::NUM_BYTES,
        canonical_modulus: Some(AKITA_FIELD_MODULUS),
    }
}

#[cfg(not(feature = "field-inline"))]
fn field_rd_inc_packing_config() -> FieldRdIncPacking {
    FieldRdIncPacking {
        byte_width: AkitaField::NUM_BYTES,
        canonical_modulus: None,
    }
}

fn require_committed_program_schedule(
    precommitted: &PrecommittedSchedule,
) -> Result<(), VerifierError> {
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

    Ok(())
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

fn map_lattice_layout_error(error: JoltLatticeLayoutError) -> VerifierError {
    invalid_lattice_config(error.to_string())
}
