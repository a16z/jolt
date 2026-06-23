//! Prover-facing helpers for assembling Akita verifier artifacts.

use crate::{
    akita_packing::AkitaPackingScheme,
    akita_validation::{
        validate_akita_advice_commitment_aliases,
        validate_akita_packing_opening_proof_payload_shape,
        validate_akita_precommitted_commitment_aliases,
        validate_akita_precommitted_opening_proof_payload_shapes,
        validate_akita_proof_payload_shape, validate_akita_verifier_setup_config,
    },
    config::{
        AdviceLatticeConfig, FieldInlineLatticeConfig, IncrementCommitmentMode, JoltProtocolConfig,
        LatticeConfig, PackedWitnessConfig, PcsFamily, ProgramMode,
    },
    preprocessing::JoltVerifierPreprocessing,
    proof::{ClearOnlyVectorCommitment, CommitmentPayload, JoltProof, LatticeCommitmentPayload},
    stages::stage8::validate_lattice_packed_witness_layout_config,
    VerifierError,
};
use common::jolt_device::JoltDevice;
use jolt_akita::{AkitaBatchProof, AkitaCommitment, AkitaField, AkitaProverHint};
use jolt_claims::protocols::jolt::{
    lattice_packed_validity_digest, JoltAdviceKind, LatticePackedFamilyId,
    LatticePackedValidityRequirement,
};
use jolt_field::{RingAccumulator, WithAccumulator};
use jolt_openings::{
    CommitmentScheme, PackingAdviceKind, PackingBatchProof, PackingFamilyId, PackingWitnessLayout,
    PackingWitnessSource,
};
use jolt_riscv::CircuitFlags;
use jolt_transcript::Transcript;

#[cfg(test)]
use crate::stages::stage8::{
    derive_lattice_packed_validity_requirements, derive_lattice_packed_validity_statements,
    lattice_packing_family_id, LatticePackedValidityStatement,
};
#[cfg(test)]
use jolt_claims::protocols::jolt::LatticePackedValidityKind;

pub use crate::akita_openings::{
    prove_akita_jolt_final_openings, prove_akita_jolt_final_openings_with_precommitted,
    prove_akita_packing_openings, prove_akita_stage8_clear_openings,
    prove_akita_stage8_clear_openings_with_precommitted, prove_and_attach_akita_opening_proofs,
    prove_and_attach_akita_opening_proofs_with_precommitted, AkitaPrecommittedOpeningInput,
    AkitaStage8ClearOpeningProofs,
};
pub use crate::akita_validity::{
    attach_akita_packing_validity_proof, prove_akita_jolt_packed_validity,
    prove_akita_packing_validity, AkitaPackingValidityProofArtifacts,
};
pub use crate::akita_witness::{build_akita_packing_jolt_witness, AkitaPackingJoltWitnessInput};

pub type AkitaClearVectorCommitment = ClearOnlyVectorCommitment<AkitaField>;
pub type AkitaPackingBatchProof = PackingBatchProof<AkitaBatchProof>;
pub type AkitaPackingProverSetup = <AkitaPackingScheme as CommitmentScheme>::ProverSetup;
pub type AkitaPackingVerifierSetup = <AkitaPackingScheme as CommitmentScheme>::VerifierSetup;
pub type AkitaVerifierPreprocessing =
    JoltVerifierPreprocessing<AkitaPackingScheme, AkitaClearVectorCommitment>;
pub type AkitaJoltProof = JoltProof<AkitaPackingScheme, AkitaClearVectorCommitment>;

#[derive(Clone, Debug)]
pub struct AkitaPackingWitnessArtifacts {
    pub protocol: JoltProtocolConfig,
    pub layout: PackingWitnessLayout,
    pub commitments: CommitmentPayload<AkitaCommitment>,
    pub hint: AkitaProverHint,
}

#[derive(Clone, Debug)]
pub struct AkitaCommittedPackedJoltWitness {
    pub artifacts: AkitaPackingWitnessArtifacts,
    pub witness: jolt_openings::SparsePackingWitness<AkitaField>,
}

impl AkitaPackingWitnessArtifacts {
    pub fn payload(&self) -> Option<&LatticeCommitmentPayload<AkitaCommitment>> {
        self.commitments.as_lattice()
    }
}

pub fn commit_akita_packing_jolt_witness(
    setup: &AkitaPackingProverSetup,
    input: AkitaPackingJoltWitnessInput<'_>,
) -> Result<AkitaCommittedPackedJoltWitness, VerifierError> {
    let witness = build_akita_packing_jolt_witness(input)?;
    let artifacts = commit_akita_packing_witness(setup, &witness)?;
    Ok(AkitaCommittedPackedJoltWitness { artifacts, witness })
}

pub fn akita_lattice_protocol_config_for_layout(
    layout: &PackingWitnessLayout,
) -> JoltProtocolConfig {
    let validity_requirements = akita_lattice_validity_requirements_for_layout(layout);
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

pub fn akita_lattice_validity_requirements_for_layout(
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

pub fn commit_akita_packing_witness<S>(
    setup: &AkitaPackingProverSetup,
    source: &S,
) -> Result<AkitaPackingWitnessArtifacts, VerifierError>
where
    S: PackingWitnessSource<AkitaField>,
{
    let protocol = akita_lattice_protocol_config_for_layout(source.layout());
    commit_akita_packing_witness_with_config(protocol, setup, source)
}

pub fn commit_akita_packing_witness_with_config<S>(
    protocol: JoltProtocolConfig,
    setup: &AkitaPackingProverSetup,
    source: &S,
) -> Result<AkitaPackingWitnessArtifacts, VerifierError>
where
    S: PackingWitnessSource<AkitaField>,
{
    let layout = source.layout().clone();
    validate_lattice_packed_witness_layout_config(&protocol, &layout)?;
    let (commitment, hint) =
        AkitaPackingScheme::commit_packing_source(setup, source).map_err(|error| {
            VerifierError::LatticePackingCommitmentFailed {
                reason: error.to_string(),
            }
        })?;
    let payload = LatticeCommitmentPayload::new(commitment, layout.digest, layout.dimension);
    crate::proof::validate_lattice_commitment_payload_config(&protocol, &payload)?;

    Ok(AkitaPackingWitnessArtifacts {
        protocol,
        layout,
        commitments: CommitmentPayload::Lattice(payload),
        hint,
    })
}

pub fn verify_akita_clear<T>(
    preprocessing: &AkitaVerifierPreprocessing,
    public_io: &JoltDevice,
    proof: &AkitaJoltProof,
    trusted_advice_commitment: Option<&AkitaCommitment>,
    config: &JoltProtocolConfig,
) -> Result<(), VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    <AkitaField as WithAccumulator>::Accumulator: RingAccumulator<Element = AkitaField>,
{
    validate_akita_verifier_setup_config(&preprocessing.pcs_setup, config)?;
    validate_akita_proof_payload_shape(&preprocessing.pcs_setup, &proof.commitments)?;
    validate_akita_packing_opening_proof_payload_shape(
        &proof.commitments,
        &proof.joint_opening_proof,
        "Akita joint opening proof",
    )?;
    if let Some(opening_proof) = &proof.lattice_packed_validity_opening_proof {
        validate_akita_packing_opening_proof_payload_shape(
            &proof.commitments,
            opening_proof,
            "Akita lattice packed validity opening proof",
        )?;
    }
    validate_akita_precommitted_opening_proof_payload_shapes(
        &proof.commitments,
        &proof.lattice_precommitted_opening_proofs,
    )?;
    validate_akita_advice_commitment_aliases(
        &proof.commitments,
        proof.untrusted_advice_commitment.as_ref(),
        trusted_advice_commitment,
    )?;
    validate_akita_precommitted_commitment_aliases(
        preprocessing,
        &proof.commitments,
        trusted_advice_commitment,
    )?;
    crate::verifier::verify_clear_with_config::<
        AkitaField,
        AkitaPackingScheme,
        AkitaClearVectorCommitment,
        T,
    >(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
        config,
    )
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

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "tests assert successful artifact construction"
    )]

    use super::*;
    use crate::proof::ClearOnlyCommitment;
    use crate::stages::{CommittedProgramSchedule, PrecommittedSchedule};
    use jolt_akita::{AkitaSetupParams, AKITA_FIELD_MODULUS};
    use jolt_claims::protocols::jolt::{
        bytecode_imm_canonical_bytes_requirement,
        formulas::{
            dimensions::{TracePolynomialOrder, REGISTER_ADDRESS_BITS},
            ra::JoltRaPolynomialLayout,
        },
    };
    use jolt_field::FixedByteSize;
    use jolt_openings::{
        CommitmentScheme, PackingAlphabet, PackingCellAddress, PackingFactDomain,
        PackingFamilySpec, PackingSetupParams, SparsePackingWitness,
    };
    use jolt_riscv::CircuitFlags;
    use jolt_transcript::{Blake2bTranscript, Transcript};

    fn run_on_large_stack(test: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn(test)
            .expect("failed to spawn test thread")
            .join()
            .expect("test thread panicked");
    }

    fn akita_packing_params(
        layout: &PackingWitnessLayout,
        max_num_polys_per_commitment_group: usize,
    ) -> PackingSetupParams<AkitaSetupParams, PackingWitnessLayout> {
        PackingSetupParams {
            pcs: AkitaSetupParams::new(
                layout.dimension,
                max_num_polys_per_commitment_group,
                layout.digest,
            ),
            layout: layout.clone(),
        }
    }

    #[test]
    #[cfg_attr(
        feature = "field-inline",
        ignore = "field-inline canonical-byte validity makes the real Akita proof fixture expensive; run explicitly with --run-ignored"
    )]
    fn packed_validity_helper_proves_real_akita_opening_proof() {
        run_on_large_stack(|| {
            let log_t = 0;
            let log_k_chunk = 1;
            let precommitted = PrecommittedSchedule::new(
                TracePolynomialOrder::CycleMajor,
                log_t,
                log_k_chunk,
                None,
                None,
                Some(CommittedProgramSchedule {
                    bytecode_len: 1,
                    bytecode_chunk_count: 1,
                    program_image_len_words: 1,
                    program_image_start_index: 0,
                }),
            )
            .expect("precommitted schedule should build");
            let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
            config.lattice.program_mode = ProgramMode::Committed;
            config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
            config.lattice.packed_witness.layout_digest = Some([0; 32]);
            config.lattice.packed_witness.d_pack = Some(0);
            config.lattice.packed_witness.validity_digest = Some([0; 32]);
            #[cfg(feature = "field-inline")]
            {
                config.lattice.field_inline.enabled = true;
            }

            let layout = crate::stages::stage8::derive_lattice_packed_witness_layout(
                &config,
                log_t,
                log_k_chunk,
                JoltRaPolynomialLayout::new(1, 1, 1).expect("RA layout should build"),
                &precommitted,
            )
            .expect("layout should derive");
            config.lattice.packed_witness.layout_digest = Some(layout.digest);
            config.lattice.packed_witness.d_pack = Some(layout.dimension);
            let requirements =
                derive_lattice_packed_validity_requirements(&config, log_k_chunk, &precommitted)
                    .expect("validity requirements should derive");
            config.lattice.packed_witness.validity_digest =
                Some(lattice_packed_validity_digest(&requirements));
            let source = validity_default_source(&layout, &requirements);
            let params = akita_packing_params(&layout, 1);
            let (prover_setup, verifier_setup) = AkitaPackingScheme::setup(params);
            let artifacts =
                commit_akita_packing_witness_with_config(config, &prover_setup, &source)
                    .expect("valid packed witness should commit");

            let mut prover_transcript = Blake2bTranscript::new(b"akita-validity");
            let validity = prove_akita_packing_validity(
                &prover_setup,
                &mut prover_transcript,
                &artifacts,
                &source,
                log_k_chunk,
                &precommitted,
            )
            .expect("validity proof should prove");

            let mut verifier_transcript = Blake2bTranscript::new(b"akita-validity");
            verify_validity_artifacts(
                &verifier_setup,
                &mut verifier_transcript,
                &artifacts,
                log_k_chunk,
                &precommitted,
                &validity,
            )
            .expect("validity proof should verify");
            assert_eq!(prover_transcript.state(), verifier_transcript.state());

            let mut tampered = validity.clone();
            tampered.opening_claims.opening_claims[0] += AkitaField::one();
            let mut tampered_transcript = Blake2bTranscript::new(b"akita-validity");
            let error = verify_validity_artifacts(
                &verifier_setup,
                &mut tampered_transcript,
                &artifacts,
                log_k_chunk,
                &precommitted,
                &tampered,
            )
            .expect_err("tampered validity opening claim should reject");
            assert!(matches!(
                error,
                VerifierError::LatticePackedValidityOutputMismatch
                    | VerifierError::LatticePackedValidityOpeningVerificationFailed { .. }
            ));
        });
    }

    #[cfg(feature = "field-inline")]
    #[test]
    #[ignore = "real Akita negative canonical-byte proof takes over two minutes; run explicitly with --run-ignored"]
    fn packed_validity_rejects_noncanonical_field_rd_inc_bytes() {
        run_on_large_stack(|| {
            let log_t = 0;
            let log_k_chunk = 1;
            let precommitted = PrecommittedSchedule::new(
                TracePolynomialOrder::CycleMajor,
                log_t,
                log_k_chunk,
                None,
                None,
                Some(CommittedProgramSchedule {
                    bytecode_len: 1,
                    bytecode_chunk_count: 1,
                    program_image_len_words: 1,
                    program_image_start_index: 0,
                }),
            )
            .expect("precommitted schedule should build");
            let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
            config.lattice.program_mode = ProgramMode::Committed;
            config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
            config.lattice.field_inline.enabled = true;
            config.lattice.packed_witness.layout_digest = Some([0; 32]);
            config.lattice.packed_witness.d_pack = Some(0);
            config.lattice.packed_witness.validity_digest = Some([0; 32]);

            let layout = crate::stages::stage8::derive_lattice_packed_witness_layout(
                &config,
                log_t,
                log_k_chunk,
                JoltRaPolynomialLayout::new(1, 1, 1).expect("RA layout should build"),
                &precommitted,
            )
            .expect("layout should derive");
            config.lattice.packed_witness.layout_digest = Some(layout.digest);
            config.lattice.packed_witness.d_pack = Some(layout.dimension);
            let requirements =
                derive_lattice_packed_validity_requirements(&config, log_k_chunk, &precommitted)
                    .expect("validity requirements should derive");
            config.lattice.packed_witness.validity_digest =
                Some(lattice_packed_validity_digest(&requirements));

            let modulus_bytes = AKITA_FIELD_MODULUS.to_le_bytes();
            let source =
                validity_source_with_field_rd_inc_bytes(&layout, &requirements, &modulus_bytes);
            let params = akita_packing_params(&layout, 1);
            let (prover_setup, verifier_setup) = AkitaPackingScheme::setup(params);
            let artifacts =
                commit_akita_packing_witness_with_config(config, &prover_setup, &source)
                    .expect("packed witness should commit");

            let mut prover_transcript = Blake2bTranscript::new(b"akita-validity");
            let validity = prove_akita_packing_validity(
                &prover_setup,
                &mut prover_transcript,
                &artifacts,
                &source,
                log_k_chunk,
                &precommitted,
            )
            .expect("invalid packed witness can still produce a proof transcript");

            let mut verifier_transcript = Blake2bTranscript::new(b"akita-validity");
            let error = verify_validity_artifacts(
                &verifier_setup,
                &mut verifier_transcript,
                &artifacts,
                log_k_chunk,
                &precommitted,
                &validity,
            )
            .expect_err("noncanonical field bytes should reject");
            assert!(matches!(
                error,
                VerifierError::LatticePackedValidityOutputMismatch
                    | VerifierError::LatticePackedValiditySumcheckFailed { .. }
                    | VerifierError::LatticePackedValidityOpeningVerificationFailed { .. }
            ));
        });
    }

    #[test]
    fn packed_validity_rejects_precommitted_bytecode_layout_config() {
        let (layout, _, requirements) = small_bytecode_validity_context();
        let source = validity_source_with_bytecode_imm_bytes(
            &layout,
            &requirements,
            &AKITA_FIELD_MODULUS.to_le_bytes(),
        );
        let mut config = akita_lattice_protocol_config_for_layout(&layout);
        config.lattice.packed_witness.validity_digest =
            Some(lattice_packed_validity_digest(&requirements));
        let params = akita_packing_params(&layout, 1);
        let (prover_setup, _) = AkitaPackingScheme::setup(params);

        let error = commit_akita_packing_witness_with_config(config, &prover_setup, &source)
            .expect_err("precommitted bytecode families should reject");

        assert!(matches!(
            error,
            VerifierError::InvalidProtocolConfig { reason }
                if reason.contains("precommitted family")
        ));
    }

    fn validity_default_source(
        layout: &PackingWitnessLayout,
        requirements: &[LatticePackedValidityRequirement],
    ) -> SparsePackingWitness<AkitaField> {
        validity_source_with_symbols(layout, requirements, |_, _| 0)
    }

    fn small_bytecode_validity_context() -> (
        PackingWitnessLayout,
        Vec<LatticePackedValidityStatement>,
        Vec<LatticePackedValidityRequirement>,
    ) {
        let specs = vec![
            PackingFamilySpec::direct(
                PackingFamilyId::BytecodeRegisterSelector {
                    chunk: 0,
                    selector: 0,
                },
                PackingFactDomain::BytecodeRows { log_bytecode: 0 },
                1,
                PackingAlphabet::Fixed {
                    size: 1 << REGISTER_ADDRESS_BITS,
                },
            ),
            PackingFamilySpec::direct(
                PackingFamilyId::BytecodeRegisterSelector {
                    chunk: 0,
                    selector: 2,
                },
                PackingFactDomain::BytecodeRows { log_bytecode: 0 },
                1,
                PackingAlphabet::Fixed {
                    size: 1 << REGISTER_ADDRESS_BITS,
                },
            ),
            PackingFamilySpec::direct(
                PackingFamilyId::BytecodeCircuitFlag {
                    chunk: 0,
                    flag: CircuitFlags::Store as usize,
                },
                PackingFactDomain::BytecodeRows { log_bytecode: 0 },
                1,
                PackingAlphabet::Bit,
            ),
            PackingFamilySpec::direct(
                PackingFamilyId::BytecodeImmBytes { chunk: 0 },
                PackingFactDomain::BytecodeRows { log_bytecode: 0 },
                AkitaField::NUM_BYTES,
                PackingAlphabet::Byte,
            ),
        ];
        #[cfg(feature = "field-inline")]
        let specs = {
            let mut specs = specs;
            specs.extend((0..AkitaField::NUM_BYTES).map(|index| {
                PackingFamilySpec::direct(
                    PackingFamilyId::FieldRdIncByte { index },
                    PackingFactDomain::TraceRows { log_t: 0 },
                    1,
                    PackingAlphabet::Byte,
                )
            }));
            specs
        };
        let layout =
            PackingWitnessLayout::new(specs).expect("manual bytecode validity layout should build");
        let mut requirements = akita_lattice_validity_requirements_for_layout(&layout);
        requirements.push(bytecode_imm_canonical_bytes_requirement(
            0,
            AkitaField::NUM_BYTES,
            AKITA_FIELD_MODULUS,
        ));
        let statements = derive_lattice_packed_validity_statements(&layout, &requirements)
            .expect("manual bytecode validity statements should derive");
        (layout, statements, requirements)
    }

    #[cfg(feature = "field-inline")]
    fn validity_source_with_field_rd_inc_bytes(
        layout: &PackingWitnessLayout,
        requirements: &[LatticePackedValidityRequirement],
        bytes: &[u8],
    ) -> SparsePackingWitness<AkitaField> {
        validity_source_with_symbols(layout, requirements, |family, _| match family {
            LatticePackedFamilyId::FieldRdIncByte { index } => bytes[*index] as usize,
            _ => 0,
        })
    }

    fn validity_source_with_bytecode_imm_bytes(
        layout: &PackingWitnessLayout,
        requirements: &[LatticePackedValidityRequirement],
        bytes: &[u8],
    ) -> SparsePackingWitness<AkitaField> {
        validity_source_with_symbols(layout, requirements, |family, limb| match family {
            LatticePackedFamilyId::BytecodeImmBytes { .. } => bytes[limb] as usize,
            _ => 0,
        })
    }

    fn validity_source_with_symbols(
        layout: &PackingWitnessLayout,
        requirements: &[LatticePackedValidityRequirement],
        mut symbol_for: impl FnMut(&LatticePackedFamilyId, usize) -> usize,
    ) -> SparsePackingWitness<AkitaField> {
        let mut cells = Vec::new();
        for requirement in requirements {
            let family_id = lattice_packing_family_id(&requirement.family);
            let family = layout
                .family(&family_id)
                .expect("validity family should exist");
            let rows = family.domain.rows().expect("family rows should derive");
            if !matches!(requirement.kind, LatticePackedValidityKind::ExactOneHot) {
                continue;
            }
            for row in 0..rows {
                for limb in 0..requirement.limbs {
                    let symbol = symbol_for(&requirement.family, limb);
                    cells.push((
                        PackingCellAddress {
                            family: family_id.clone(),
                            row,
                            limb,
                            symbol,
                        },
                        AkitaField::one(),
                    ));
                }
            }
        }
        SparsePackingWitness::try_from_cells(layout.clone(), cells)
            .expect("validity source should build")
    }

    fn verify_validity_artifacts<T>(
        setup: &AkitaPackingVerifierSetup,
        transcript: &mut T,
        artifacts: &AkitaPackingWitnessArtifacts,
        log_k_chunk: usize,
        precommitted: &PrecommittedSchedule,
        validity: &AkitaPackingValidityProofArtifacts,
    ) -> Result<(), VerifierError>
    where
        T: Transcript<Challenge = AkitaField>,
    {
        crate::stages::stage8::verify_lattice_packed_validity_proof::<
            AkitaField,
            AkitaPackingScheme,
            T,
            ClearOnlyCommitment,
        >(
            setup,
            transcript,
            &artifacts.protocol,
            log_k_chunk,
            precommitted,
            &artifacts.layout,
            artifacts
                .payload()
                .expect("artifact should carry lattice payload")
                .packed_witness
                .clone(),
            &validity.sumcheck_proof,
            &validity.opening_claims.opening_claims,
            &validity.opening_proof,
        )
    }
}
