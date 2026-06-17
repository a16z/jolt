use crate::{
    config::{validate_protocol_config, JoltProtocolConfig, PcsFamily},
    stages::PrecommittedSchedule,
    VerifierError,
};
use jolt_akita::{
    PackedAdviceKind, PackedAlphabet, PackedFactDomain, PackedFamilyId, PackedFamilySpec,
    PackedWitnessLayout,
};
use jolt_claims::protocols::jolt::{
    formulas::{claim_reductions::bytecode, ra::JoltRaPolynomialLayout},
    AdviceClaimReductionLayout, JoltAdviceKind, ProgramImageClaimReductionLayout,
};

pub fn derive_akita_packed_witness_layout(
    config: &JoltProtocolConfig,
    log_t: usize,
    log_k_chunk: usize,
    ra_layout: JoltRaPolynomialLayout,
    precommitted: &PrecommittedSchedule,
) -> Result<PackedWitnessLayout, VerifierError> {
    if validate_protocol_config(config)? != PcsFamily::Lattice {
        return Err(invalid_lattice_config(
            "Akita packed witness layout derivation requires lattice PCS mode",
        ));
    }

    let trace = PackedFactDomain::TraceRows { log_t };
    let ra_alphabet = one_hot_alphabet(log_k_chunk)?;
    let mut specs = Vec::new();
    specs.extend((0..ra_layout.instruction()).map(|index| {
        PackedFamilySpec::direct(
            PackedFamilyId::InstructionRa { index },
            trace,
            1,
            ra_alphabet,
        )
    }));
    specs.extend((0..ra_layout.bytecode()).map(|index| {
        PackedFamilySpec::direct(PackedFamilyId::BytecodeRa { index }, trace, 1, ra_alphabet)
    }));
    specs.extend((0..ra_layout.ram()).map(|index| {
        PackedFamilySpec::direct(PackedFamilyId::RamRa { index }, trace, 1, ra_alphabet)
    }));

    specs.extend((0..8).map(|index| {
        PackedFamilySpec::direct(
            PackedFamilyId::IncByte { index },
            trace,
            1,
            PackedAlphabet::Byte,
        )
    }));
    specs.push(PackedFamilySpec::direct(
        PackedFamilyId::IncSign,
        trace,
        1,
        PackedAlphabet::Bit,
    ));

    if config.lattice.field_inline.enabled {
        specs.extend((0..8).map(|index| {
            PackedFamilySpec::direct(
                PackedFamilyId::FieldRdIncByte { index },
                trace,
                1,
                PackedAlphabet::Byte,
            )
        }));
        specs.push(PackedFamilySpec::direct(
            PackedFamilyId::FieldRdIncSign,
            trace,
            1,
            PackedAlphabet::Bit,
        ));
    }

    if config.lattice.advice.trusted {
        specs.push(advice_family(
            PackedAdviceKind::Trusted,
            require_advice_layout(precommitted, JoltAdviceKind::Trusted)?,
        ));
    }
    if config.lattice.advice.untrusted {
        specs.push(advice_family(
            PackedAdviceKind::Untrusted,
            require_advice_layout(precommitted, JoltAdviceKind::Untrusted)?,
        ));
    }

    let bytecode_layout = precommitted.bytecode.as_ref().ok_or_else(|| {
        invalid_precommitted_schedule(
            "lattice committed-program mode requires a bytecode claim-reduction layout",
        )
    })?;
    for index in 0..bytecode_layout.chunk_count() {
        specs.push(PackedFamilySpec::direct(
            PackedFamilyId::BytecodeChunk { index },
            PackedFactDomain::BytecodeRows {
                log_bytecode: bytecode_layout.log_bytecode_chunk_size(),
            },
            bytecode::committed_lanes(),
            PackedAlphabet::Byte,
        ));
    }

    let program_image_layout = precommitted.program_image.as_ref().ok_or_else(|| {
        invalid_precommitted_schedule(
            "lattice committed-program mode requires a program-image claim-reduction layout",
        )
    })?;
    specs.push(program_image_family(program_image_layout)?);

    PackedWitnessLayout::new(specs).map_err(|error| invalid_lattice_config(error.to_string()))
}

pub fn validate_akita_packed_witness_layout_config(
    config: &JoltProtocolConfig,
    layout: &PackedWitnessLayout,
) -> Result<(), VerifierError> {
    if validate_protocol_config(config)? != PcsFamily::Lattice {
        return Err(invalid_lattice_config(
            "Akita packed witness layout validation requires lattice PCS mode",
        ));
    }
    if config.lattice.packed_witness.layout_digest != Some(layout.digest) {
        return Err(invalid_lattice_config(
            "configured Akita packed witness layout digest does not match derived layout",
        ));
    }
    if config.lattice.packed_witness.d_pack != Some(layout.dimension) {
        return Err(invalid_lattice_config(
            "configured Akita packed witness D_pack does not match derived layout",
        ));
    }
    Ok(())
}

fn advice_family(kind: PackedAdviceKind, layout: &AdviceClaimReductionLayout) -> PackedFamilySpec {
    PackedFamilySpec::direct(
        PackedFamilyId::AdviceBytes { kind, index: 0 },
        PackedFactDomain::AdviceBytes {
            kind,
            log_bytes: layout.advice_shape().total_vars() + 3,
        },
        1,
        PackedAlphabet::Byte,
    )
}

fn program_image_family(
    layout: &ProgramImageClaimReductionLayout,
) -> Result<PackedFamilySpec, VerifierError> {
    Ok(PackedFamilySpec::direct(
        PackedFamilyId::ProgramImageInit,
        PackedFactDomain::ProgramImageWords {
            log_words: power_of_two_log(layout.padded_len_words(), "program image length")?,
        },
        8,
        PackedAlphabet::Byte,
    ))
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

fn one_hot_alphabet(log_k_chunk: usize) -> Result<PackedAlphabet, VerifierError> {
    match log_k_chunk {
        0 => Err(invalid_lattice_config(
            "lattice one-hot chunk size must be nonzero",
        )),
        1 => Ok(PackedAlphabet::Bit),
        8 => Ok(PackedAlphabet::Byte),
        bits if bits < usize::BITS as usize => Ok(PackedAlphabet::Fixed {
            size: 1usize << bits,
        }),
        _ => Err(invalid_lattice_config(
            "lattice one-hot chunk size is too large",
        )),
    }
}

fn power_of_two_log(value: usize, name: &'static str) -> Result<usize, VerifierError> {
    if value.is_power_of_two() {
        Ok(value.trailing_zeros() as usize)
    } else {
        Err(invalid_lattice_config(format!(
            "{name} must be a power of two, got {value}",
        )))
    }
}

fn invalid_lattice_config(reason: impl Into<String>) -> VerifierError {
    VerifierError::InvalidProtocolConfig {
        reason: reason.into(),
    }
}

fn invalid_precommitted_schedule(reason: impl Into<String>) -> VerifierError {
    VerifierError::InvalidPrecommittedSchedule {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::panic, reason = "tests fail loudly on setup errors")]

    use super::*;
    use crate::{
        config::{IncrementCommitmentMode, PackedWitnessConfig, ProgramMode},
        stages::CommittedProgramSchedule,
    };
    use jolt_claims::protocols::jolt::TracePolynomialOrder;

    fn lattice_config() -> JoltProtocolConfig {
        let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
        config.lattice.program_mode = ProgramMode::Committed;
        config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
        config.lattice.packed_witness = PackedWitnessConfig {
            layout_digest: Some([0; 32]),
            d_pack: Some(0),
            field_rd_inc_family: false,
            trusted_advice_family: false,
            untrusted_advice_family: false,
        };
        config
    }

    fn precommitted_schedule(trusted_max_advice_bytes: Option<usize>) -> PrecommittedSchedule {
        PrecommittedSchedule::new(
            TracePolynomialOrder::CycleMajor,
            2,
            8,
            trusted_max_advice_bytes,
            None,
            Some(CommittedProgramSchedule {
                bytecode_len: 16,
                bytecode_chunk_count: 2,
                program_image_len_words: 4,
                program_image_start_index: 0,
            }),
        )
        .unwrap_or_else(|error| panic!("precommitted schedule should build: {error}"))
    }

    fn ra_layout() -> JoltRaPolynomialLayout {
        JoltRaPolynomialLayout::new(2, 1, 1)
            .unwrap_or_else(|error| panic!("RA layout should build: {error}"))
    }

    #[test]
    fn derive_layout_includes_base_lattice_families() {
        let config = lattice_config();
        let layout = derive_akita_packed_witness_layout(
            &config,
            2,
            8,
            ra_layout(),
            &precommitted_schedule(None),
        )
        .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));

        assert!(layout
            .family(&PackedFamilyId::InstructionRa { index: 0 })
            .is_some());
        assert!(layout.family(&PackedFamilyId::RamRa { index: 0 }).is_some());
        assert!(layout
            .family(&PackedFamilyId::IncByte { index: 7 })
            .is_some());
        assert!(layout.family(&PackedFamilyId::IncSign).is_some());
        assert!(layout
            .family(&PackedFamilyId::BytecodeChunk { index: 0 })
            .is_some());
        assert!(layout.family(&PackedFamilyId::ProgramImageInit).is_some());
        assert_eq!(layout.audit().d_pack, layout.dimension);

        let mut matching_config = lattice_config();
        matching_config.lattice.packed_witness.layout_digest = Some(layout.digest);
        matching_config.lattice.packed_witness.d_pack = Some(layout.dimension);

        validate_akita_packed_witness_layout_config(&matching_config, &layout)
            .unwrap_or_else(|error| panic!("layout config should validate: {error}"));
    }

    #[test]
    fn layout_config_mismatch_rejects() {
        let layout = derive_akita_packed_witness_layout(
            &lattice_config(),
            2,
            8,
            ra_layout(),
            &precommitted_schedule(None),
        )
        .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));

        assert!(matches!(
            validate_akita_packed_witness_layout_config(&lattice_config(), &layout),
            Err(VerifierError::InvalidProtocolConfig { .. })
        ));
    }

    #[test]
    fn advice_layout_requires_precommitted_schedule() {
        let mut config = lattice_config();
        config.lattice.advice.trusted = true;
        config.lattice.packed_witness.trusted_advice_family = true;

        assert!(matches!(
            derive_akita_packed_witness_layout(
                &config,
                2,
                8,
                ra_layout(),
                &precommitted_schedule(None),
            ),
            Err(VerifierError::InvalidPrecommittedSchedule { .. })
        ));

        let layout = derive_akita_packed_witness_layout(
            &config,
            2,
            8,
            ra_layout(),
            &precommitted_schedule(Some(64)),
        )
        .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));
        assert!(layout
            .family(&PackedFamilyId::AdviceBytes {
                kind: PackedAdviceKind::Trusted,
                index: 0,
            })
            .is_some());
    }
}
