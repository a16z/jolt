//! Prover-facing helpers for assembling Akita verifier artifacts.

use std::collections::BTreeMap;

use crate::{
    config::{
        AdviceLatticeConfig, FieldInlineLatticeConfig, IncrementCommitmentMode, JoltProtocolConfig,
        LatticeConfig, PackedWitnessConfig, PcsFamily, ProgramMode,
    },
    preprocessing::JoltVerifierPreprocessing,
    proof::{AkitaCommitmentPayload, ClearOnlyVectorCommitment, CommitmentPayload, JoltProof},
    stages::stage8::validate_akita_packed_witness_layout_config,
    VerifierError,
};
use common::jolt_device::JoltDevice;
use jolt_akita::{
    AkitaCommitment, AkitaField, AkitaPackedBatchProof, AkitaPackedScheme, AkitaProverHint,
    AkitaProverSetup, JoltPackedWitnessBuilder, PackedAdviceKind, PackedFactDomain, PackedFamilyId,
    PackedWitnessLayout, PackedWitnessSource, SparsePackedWitness,
};
use jolt_field::{RingAccumulator, WithAccumulator};
use jolt_openings::BatchOpeningStatement;
use jolt_riscv::{JoltInstructionRow, JoltTraceRow};
use jolt_transcript::Transcript;

pub type AkitaClearVectorCommitment = ClearOnlyVectorCommitment<AkitaField>;
pub type AkitaVerifierPreprocessing =
    JoltVerifierPreprocessing<AkitaPackedScheme, AkitaClearVectorCommitment>;
pub type AkitaJoltProof = JoltProof<AkitaPackedScheme, AkitaClearVectorCommitment>;

#[derive(Clone, Debug)]
pub struct AkitaPackedWitnessArtifacts {
    pub protocol: JoltProtocolConfig,
    pub layout: PackedWitnessLayout,
    pub commitments: CommitmentPayload<AkitaCommitment>,
    pub hint: AkitaProverHint,
}

#[derive(Clone, Debug)]
pub struct AkitaPackedJoltWitnessInput<'a> {
    pub layout: PackedWitnessLayout,
    pub trace_rows: &'a [JoltTraceRow],
    pub log_k_chunk: usize,
    pub instruction_lookup_indices: &'a [u128],
    pub bytecode_rows: &'a [JoltInstructionRow],
    pub program_image_words: &'a [u64],
    pub trusted_advice: Option<&'a [u8]>,
    pub untrusted_advice: Option<&'a [u8]>,
}

#[derive(Clone, Debug)]
pub struct AkitaCommittedPackedJoltWitness {
    pub artifacts: AkitaPackedWitnessArtifacts,
    pub witness: SparsePackedWitness<AkitaField>,
}

impl AkitaPackedWitnessArtifacts {
    pub fn payload(&self) -> Option<&AkitaCommitmentPayload<AkitaCommitment>> {
        self.commitments.as_akita()
    }
}

pub fn build_akita_packed_jolt_witness(
    input: AkitaPackedJoltWitnessInput<'_>,
) -> Result<SparsePackedWitness<AkitaField>, VerifierError> {
    if input.instruction_lookup_indices.len() != input.trace_rows.len() {
        return Err(akita_witness_error(format!(
            "instruction lookup index count {} does not match trace row count {}",
            input.instruction_lookup_indices.len(),
            input.trace_rows.len()
        )));
    }

    let mut builder = JoltPackedWitnessBuilder::new(input.layout.clone());
    builder
        .pack_trace_rows(
            input.trace_rows,
            input.log_k_chunk,
            |row, _| input.instruction_lookup_indices[row],
            |_, row| (row.is_load() || row.is_store()).then(|| row.ram_address()),
        )
        .map(|_| ())
        .map_err(akita_witness_error)?;

    pack_bytecode_rows(&mut builder, input.bytecode_rows)?;
    pack_program_image_words(&mut builder, input.program_image_words)?;
    pack_advice_bytes(
        &mut builder,
        PackedAdviceKind::Trusted,
        input.trusted_advice,
    )?;
    pack_advice_bytes(
        &mut builder,
        PackedAdviceKind::Untrusted,
        input.untrusted_advice,
    )?;

    builder.finish().map_err(akita_witness_error)
}

pub fn commit_akita_packed_jolt_witness(
    setup: &AkitaProverSetup,
    input: AkitaPackedJoltWitnessInput<'_>,
) -> Result<AkitaCommittedPackedJoltWitness, VerifierError> {
    let witness = build_akita_packed_jolt_witness(input)?;
    let artifacts = commit_akita_packed_witness(setup, &witness)?;
    Ok(AkitaCommittedPackedJoltWitness { artifacts, witness })
}

pub fn akita_lattice_protocol_config_for_layout(
    layout: &PackedWitnessLayout,
) -> JoltProtocolConfig {
    let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
    config.lattice = LatticeConfig {
        program_mode: ProgramMode::Committed,
        increment_mode: IncrementCommitmentMode::FusedOneHot,
        packed_witness: PackedWitnessConfig {
            layout_digest: Some(layout.digest),
            d_pack: Some(layout.dimension),
            field_rd_inc_family: layout_has_field_rd_inc(layout),
            trusted_advice_family: layout_has_advice(layout, PackedAdviceKind::Trusted),
            untrusted_advice_family: layout_has_advice(layout, PackedAdviceKind::Untrusted),
        },
        field_inline: FieldInlineLatticeConfig {
            enabled: layout_has_field_rd_inc(layout),
        },
        advice: AdviceLatticeConfig {
            trusted: layout_has_advice(layout, PackedAdviceKind::Trusted),
            untrusted: layout_has_advice(layout, PackedAdviceKind::Untrusted),
        },
        zk: false,
    };
    config
}

pub fn commit_akita_packed_witness<S>(
    setup: &AkitaProverSetup,
    source: &S,
) -> Result<AkitaPackedWitnessArtifacts, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let protocol = akita_lattice_protocol_config_for_layout(source.layout());
    commit_akita_packed_witness_with_config(protocol, setup, source)
}

pub fn commit_akita_packed_witness_with_config<S>(
    protocol: JoltProtocolConfig,
    setup: &AkitaProverSetup,
    source: &S,
) -> Result<AkitaPackedWitnessArtifacts, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let layout = source.layout().clone();
    validate_akita_packed_witness_layout_config(&protocol, &layout)?;
    let (commitment, hint) =
        AkitaPackedScheme::commit_packed_source(setup, source).map_err(|error| {
            VerifierError::AkitaCommitmentFailed {
                reason: error.to_string(),
            }
        })?;
    let payload = AkitaCommitmentPayload::new(commitment, layout.digest, layout.dimension);
    crate::proof::validate_akita_commitment_payload_config(&protocol, &payload)?;

    Ok(AkitaPackedWitnessArtifacts {
        protocol,
        layout,
        commitments: CommitmentPayload::Akita(payload),
        hint,
    })
}

pub fn prove_akita_packed_openings<T, OpeningId, RelationId, S>(
    setup: &AkitaProverSetup,
    transcript: &mut T,
    artifacts: &AkitaPackedWitnessArtifacts,
    source: &S,
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
) -> Result<AkitaPackedBatchProof, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackedWitnessSource<AkitaField>,
{
    if source.layout() != &artifacts.layout {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: "Akita packed opening source layout does not match committed artifact"
                .to_string(),
        });
    }
    if statement.layout_digest != artifacts.layout.digest {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason:
                "Akita packed opening statement layout digest does not match committed artifact"
                    .to_string(),
        });
    }
    let payload = artifacts
        .payload()
        .ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
            reason: "Akita packed opening artifacts do not carry an Akita payload".to_string(),
        })?;
    for claim in &statement.claims {
        if claim.commitment != payload.packed_witness {
            return Err(VerifierError::FinalOpeningBatchFailed {
                reason: "Akita packed opening statement references a non-artifact commitment"
                    .to_string(),
            });
        }
    }

    AkitaPackedScheme::prove_packed_source_batch(
        setup,
        transcript,
        statement,
        source,
        artifacts.hint.clone(),
    )
    .map_err(|error| VerifierError::FinalOpeningBatchFailed {
        reason: error.to_string(),
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
    crate::verifier::verify_clear_with_config::<
        AkitaField,
        AkitaPackedScheme,
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

fn pack_bytecode_rows(
    builder: &mut JoltPackedWitnessBuilder,
    rows: &[JoltInstructionRow],
) -> Result<(), VerifierError> {
    let expected = expected_bytecode_rows(builder.layout())?;
    let Some(expected) = expected else {
        if rows.is_empty() {
            return Ok(());
        }
        return Err(akita_witness_error(
            "bytecode rows were supplied but the packed layout has no bytecode families",
        ));
    };
    let padded = padded_slice(rows, expected, "bytecode rows")?;
    builder
        .pack_bytecode_rows(&padded)
        .map(|_| ())
        .map_err(akita_witness_error)
}

fn pack_program_image_words(
    builder: &mut JoltPackedWitnessBuilder,
    words: &[u64],
) -> Result<(), VerifierError> {
    let expected = expected_rows_for_family(
        builder.layout(),
        |id| matches!(id, PackedFamilyId::ProgramImageInit),
        "program image words",
    )?;
    let Some(expected) = expected else {
        if words.is_empty() {
            return Ok(());
        }
        return Err(akita_witness_error(
            "program image words were supplied but the packed layout has no program image family",
        ));
    };
    let padded = padded_slice(words, expected, "program image words")?;
    builder
        .pack_program_image_words(&padded)
        .map(|_| ())
        .map_err(akita_witness_error)
}

fn pack_advice_bytes(
    builder: &mut JoltPackedWitnessBuilder,
    kind: PackedAdviceKind,
    bytes: Option<&[u8]>,
) -> Result<(), VerifierError> {
    let expected = expected_rows_for_family(
        builder.layout(),
        |id| matches!(id, PackedFamilyId::AdviceBytes { kind: family_kind, index: 0 } if *family_kind == kind),
        advice_domain_name(kind),
    )?;
    let Some(expected) = expected else {
        if bytes.is_none_or(<[u8]>::is_empty) {
            return Ok(());
        }
        return Err(akita_witness_error(format!(
            "{} were supplied but the packed layout has no matching advice family",
            advice_domain_name(kind)
        )));
    };
    let padded = padded_slice(
        bytes.unwrap_or_default(),
        expected,
        advice_domain_name(kind),
    )?;
    builder
        .pack_advice_bytes(kind, &padded)
        .map(|_| ())
        .map_err(akita_witness_error)
}

fn expected_bytecode_rows(layout: &PackedWitnessLayout) -> Result<Option<usize>, VerifierError> {
    let mut chunks = BTreeMap::<usize, usize>::new();
    for family in &layout.families {
        let chunk = match family.id {
            PackedFamilyId::BytecodeChunk { index }
            | PackedFamilyId::BytecodeRegisterSelector { chunk: index, .. }
            | PackedFamilyId::BytecodeCircuitFlag { chunk: index, .. }
            | PackedFamilyId::BytecodeInstructionFlag { chunk: index, .. }
            | PackedFamilyId::BytecodeLookupSelector { chunk: index }
            | PackedFamilyId::BytecodeRafFlag { chunk: index }
            | PackedFamilyId::BytecodeUnexpandedPcBytes { chunk: index }
            | PackedFamilyId::BytecodeImmBytes { chunk: index } => index,
            _ => continue,
        };
        let rows = packed_domain_rows(family.domain)?;
        match chunks.insert(chunk, rows) {
            Some(existing) if existing != rows => {
                return Err(akita_witness_error(format!(
                    "bytecode rows layout chunk {chunk} has inconsistent row counts {existing} and {rows}"
                )));
            }
            _ => {}
        }
    }
    let Some((&max_chunk, &chunk_rows)) = chunks.last_key_value() else {
        return Ok(None);
    };
    for chunk in 0..=max_chunk {
        if !chunks.contains_key(&chunk) {
            return Err(akita_witness_error(format!(
                "bytecode rows layout is missing chunk {chunk}",
            )));
        }
    }
    max_chunk
        .checked_add(1)
        .and_then(|chunk_count| chunk_count.checked_mul(chunk_rows))
        .ok_or_else(|| akita_witness_error("bytecode rows layout size overflow"))
        .map(Some)
}

fn expected_rows_for_family(
    layout: &PackedWitnessLayout,
    mut matches_family: impl FnMut(&PackedFamilyId) -> bool,
    domain: &'static str,
) -> Result<Option<usize>, VerifierError> {
    let mut rows = None;
    for family in &layout.families {
        if !matches_family(&family.id) {
            continue;
        }
        let got = packed_domain_rows(family.domain)?;
        match rows {
            Some(expected) if expected != got => {
                return Err(akita_witness_error(format!(
                    "{domain} layout row count mismatch: expected {expected}, got {got}"
                )));
            }
            Some(_) => {}
            None => rows = Some(got),
        }
    }
    Ok(rows)
}

fn packed_domain_rows(domain: PackedFactDomain) -> Result<usize, VerifierError> {
    let log_rows = match domain {
        PackedFactDomain::TraceRows { log_t } => log_t,
        PackedFactDomain::BytecodeRows { log_bytecode } => log_bytecode,
        PackedFactDomain::ProgramImageWords { log_words } => log_words,
        PackedFactDomain::AdviceBytes { log_bytes, .. } => log_bytes,
    };
    1usize
        .checked_shl(log_rows as u32)
        .ok_or_else(|| akita_witness_error("packed witness domain row count overflow"))
}

fn padded_slice<T: Clone + Default>(
    values: &[T],
    expected: usize,
    domain: &'static str,
) -> Result<Vec<T>, VerifierError> {
    if values.len() > expected {
        return Err(akita_witness_error(format!(
            "{domain} length {} exceeds packed layout size {expected}",
            values.len()
        )));
    }
    let mut padded = values.to_vec();
    padded.resize_with(expected, T::default);
    Ok(padded)
}

fn advice_domain_name(kind: PackedAdviceKind) -> &'static str {
    match kind {
        PackedAdviceKind::Trusted => "trusted advice bytes",
        PackedAdviceKind::Untrusted => "untrusted advice bytes",
    }
}

fn akita_witness_error(reason: impl ToString) -> VerifierError {
    VerifierError::AkitaCommitmentFailed {
        reason: format!(
            "Akita packed witness packing failed: {}",
            reason.to_string()
        ),
    }
}

fn layout_has_field_rd_inc(layout: &PackedWitnessLayout) -> bool {
    layout
        .families
        .iter()
        .any(|family| matches!(family.id, PackedFamilyId::FieldRdIncByte { .. }))
}

fn layout_has_advice(layout: &PackedWitnessLayout, kind: PackedAdviceKind) -> bool {
    layout.families.iter().any(|family| {
        matches!(
            family.id,
            PackedFamilyId::AdviceBytes {
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
    use jolt_akita::{
        AkitaSetupParams, PackedAlphabet, PackedCellAddress, PackedFactDomain, PackedFamilySpec,
        SparsePackedWitness,
    };
    use jolt_field::FixedByteSize;
    use jolt_openings::{
        BatchOpeningClaim, BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme,
        PackedLinearTerm, PhysicalView,
    };
    use jolt_riscv::{
        CapturedState, JoltInstructionKind, JoltInstructionRow, JoltTraceRow, NonMemoryState,
        NormalizedOperands, StoreState,
    };
    use jolt_transcript::{Blake2bTranscript, Transcript};

    fn tiny_layout() -> PackedWitnessLayout {
        PackedWitnessLayout::new([
            PackedFamilySpec::direct(
                PackedFamilyId::InstructionRa { index: 0 },
                PackedFactDomain::TraceRows { log_t: 0 },
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::IncSign,
                PackedFactDomain::TraceRows { log_t: 0 },
                1,
                PackedAlphabet::Bit,
            ),
        ])
        .expect("layout should build")
    }

    fn packed_cell(family: PackedFamilyId, symbol: usize) -> PackedCellAddress {
        packed_cell_at(family, 0, 0, symbol)
    }

    fn packed_cell_at(
        family: PackedFamilyId,
        row: usize,
        limb: usize,
        symbol: usize,
    ) -> PackedCellAddress {
        PackedCellAddress {
            family,
            row,
            limb,
            symbol,
        }
    }

    fn instruction(
        kind: JoltInstructionKind,
        address: usize,
        operands: NormalizedOperands,
    ) -> JoltInstructionRow {
        JoltInstructionRow {
            instruction_kind: kind,
            address,
            operands,
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        }
    }

    fn trace_row(
        kind: JoltInstructionKind,
        operands: NormalizedOperands,
        state: CapturedState,
        bytecode_pc: u32,
    ) -> JoltTraceRow {
        JoltTraceRow::from_components(
            state,
            &instruction(kind, 0x8000_0000 + (bytecode_pc as usize * 4), operands),
            bytecode_pc,
        )
        .expect("trace row should build")
    }

    #[test]
    fn protocol_config_binds_layout_digest_and_dimension() {
        let layout = tiny_layout();

        let config = akita_lattice_protocol_config_for_layout(&layout);

        assert_eq!(
            config.lattice.packed_witness.layout_digest,
            Some(layout.digest)
        );
        assert_eq!(config.lattice.packed_witness.d_pack, Some(layout.dimension));
        assert_eq!(config.lattice.program_mode, ProgramMode::Committed);
        assert_eq!(
            config.lattice.increment_mode,
            IncrementCommitmentMode::FusedOneHot
        );
    }

    #[test]
    fn commits_packed_witness_and_returns_verifier_payload() {
        let layout = tiny_layout();
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, _) = AkitaPackedScheme::setup(params);
        let source = SparsePackedWitness::try_new(
            layout.clone(),
            vec![(0, AkitaField::from_u64(1)), (256, AkitaField::from_u64(1))],
        )
        .expect("source should build");

        let artifact = commit_akita_packed_witness(&prover_setup, &source)
            .expect("packed witness should commit");

        assert_eq!(artifact.layout, layout);
        let payload = artifact
            .payload()
            .expect("artifact should carry Akita payload");
        assert_eq!(payload.layout_digest, layout.digest);
        assert_eq!(payload.d_pack, layout.dimension);
        assert_eq!(payload.packed_witness.layout_digest, layout.digest);
        assert_eq!(payload.packed_witness.num_vars, layout.dimension);
        assert_eq!(
            artifact.protocol.lattice.packed_witness.layout_digest,
            Some(layout.digest)
        );
    }

    #[test]
    fn commits_jolt_packed_witness_inputs_with_padding() {
        let layout = PackedWitnessLayout::new([
            PackedFamilySpec::direct(
                PackedFamilyId::InstructionRa { index: 0 },
                PackedFactDomain::TraceRows { log_t: 1 },
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeRa { index: 0 },
                PackedFactDomain::TraceRows { log_t: 1 },
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::RamRa { index: 0 },
                PackedFactDomain::TraceRows { log_t: 1 },
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::IncByte { index: 0 },
                PackedFactDomain::TraceRows { log_t: 1 },
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeUnexpandedPcBytes { chunk: 0 },
                PackedFactDomain::BytecodeRows { log_bytecode: 1 },
                8,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeImmBytes { chunk: 0 },
                PackedFactDomain::BytecodeRows { log_bytecode: 1 },
                AkitaField::NUM_BYTES,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::ProgramImageInit,
                PackedFactDomain::ProgramImageWords { log_words: 1 },
                8,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::AdviceBytes {
                    kind: PackedAdviceKind::Trusted,
                    index: 0,
                },
                PackedFactDomain::AdviceBytes {
                    kind: PackedAdviceKind::Trusted,
                    log_bytes: 2,
                },
                1,
                PackedAlphabet::Byte,
            ),
        ])
        .expect("layout should build");
        let rows = [
            trace_row(
                JoltInstructionKind::ADD,
                NormalizedOperands {
                    rs1: Some(1),
                    rs2: Some(2),
                    rd: Some(3),
                    imm: 0,
                },
                CapturedState::NonMemory(NonMemoryState {
                    rs1_value: 1,
                    rs2_value: 2,
                    rd_pre_value: 4,
                    rd_write_value: 7,
                }),
                0,
            ),
            trace_row(
                JoltInstructionKind::SD,
                NormalizedOperands {
                    rs1: Some(1),
                    rs2: Some(2),
                    rd: None,
                    imm: 8,
                },
                CapturedState::Store(StoreState {
                    rs1_value: 1,
                    rs2_value: 11,
                    ram_read_value: 10,
                    ram_address: 0x34,
                }),
                1,
            ),
        ];
        let bytecode = [instruction(
            JoltInstructionKind::ADDI,
            0x8000_0000,
            NormalizedOperands {
                rs1: Some(1),
                rs2: None,
                rd: Some(5),
                imm: 7,
            },
        )];
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, _) = AkitaPackedScheme::setup(params);

        let committed = commit_akita_packed_jolt_witness(
            &prover_setup,
            AkitaPackedJoltWitnessInput {
                layout: layout.clone(),
                trace_rows: &rows,
                log_k_chunk: 8,
                instruction_lookup_indices: &[0xaa, 0xbb],
                bytecode_rows: &bytecode,
                program_image_words: &[0x0201],
                trusted_advice: Some(&[7, 8]),
                untrusted_advice: None,
            },
        )
        .expect("Jolt packed witness should build and commit");

        assert_eq!(committed.artifacts.layout, layout);
        let payload = committed
            .artifacts
            .payload()
            .expect("artifact should carry Akita payload");
        assert_eq!(payload.layout_digest, layout.digest);

        let witness = &committed.witness;
        assert_eq!(
            witness
                .eval_direct_fact(&packed_cell_at(
                    PackedFamilyId::InstructionRa { index: 0 },
                    0,
                    0,
                    0xaa,
                ))
                .expect("instruction RA cell should exist"),
            AkitaField::one()
        );
        assert_eq!(
            witness
                .eval_direct_fact(&packed_cell_at(
                    PackedFamilyId::BytecodeRa { index: 0 },
                    1,
                    0,
                    1,
                ))
                .expect("bytecode RA cell should exist"),
            AkitaField::one()
        );
        assert_eq!(
            witness
                .eval_direct_fact(&packed_cell_at(
                    PackedFamilyId::RamRa { index: 0 },
                    1,
                    0,
                    0x34
                ))
                .expect("RAM RA cell should exist"),
            AkitaField::one()
        );
        assert_eq!(
            witness
                .eval_direct_fact(&packed_cell_at(
                    PackedFamilyId::IncByte { index: 0 },
                    0,
                    0,
                    3
                ))
                .expect("increment cell should exist"),
            AkitaField::one()
        );
        assert_eq!(
            witness
                .eval_direct_fact(&packed_cell_at(
                    PackedFamilyId::BytecodeImmBytes { chunk: 0 },
                    0,
                    0,
                    7,
                ))
                .expect("bytecode immediate cell should exist"),
            AkitaField::one()
        );
        assert_eq!(
            witness
                .eval_direct_fact(&packed_cell_at(
                    PackedFamilyId::BytecodeUnexpandedPcBytes { chunk: 0 },
                    1,
                    0,
                    0,
                ))
                .expect("padded bytecode row should exist"),
            AkitaField::one()
        );
        assert_eq!(
            witness
                .eval_direct_fact(&packed_cell_at(PackedFamilyId::ProgramImageInit, 1, 0, 0))
                .expect("padded program image word should exist"),
            AkitaField::one()
        );
        assert_eq!(
            witness
                .eval_direct_fact(&packed_cell_at(
                    PackedFamilyId::AdviceBytes {
                        kind: PackedAdviceKind::Trusted,
                        index: 0,
                    },
                    2,
                    0,
                    0,
                ))
                .expect("padded trusted advice byte should exist"),
            AkitaField::one()
        );
    }

    #[test]
    fn packed_witness_artifacts_feed_akita_packed_batch_verifier() {
        let layout = tiny_layout();
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, verifier_setup) = AkitaPackedScheme::setup(params);
        let instruction_family = PackedFamilyId::InstructionRa { index: 0 };
        let sign_family = PackedFamilyId::IncSign;
        let source = SparsePackedWitness::try_from_cells(
            layout.clone(),
            [
                (
                    packed_cell(instruction_family.clone(), 7),
                    AkitaField::from_u64(11),
                ),
                (packed_cell(sign_family.clone(), 1), AkitaField::from_u64(5)),
            ],
        )
        .expect("source should build");
        let artifact = commit_akita_packed_witness(&prover_setup, &source)
            .expect("packed witness should commit");
        let commitment = artifact
            .payload()
            .expect("artifact should carry Akita payload")
            .packed_witness
            .clone();
        let instruction_claim = AkitaField::from_u64(22);
        let sign_claim = AkitaField::from_u64(15);
        let statement = BatchOpeningStatement {
            logical_point: Vec::new(),
            pcs_point: Vec::new(),
            layout_digest: layout.digest,
            claims: vec![
                BatchOpeningClaim {
                    id: 0usize,
                    relation: 0usize,
                    commitment: commitment.clone(),
                    claim: instruction_claim,
                    view: PhysicalView::PackedLinear {
                        layout_digest: layout.digest,
                        terms: vec![PackedLinearTerm::new(
                            AkitaField::from_u64(2),
                            instruction_family.physical_ref(),
                            0,
                            7,
                        )
                        .with_row_point(Vec::new())],
                    },
                    scale: AkitaField::from_u64(3),
                },
                BatchOpeningClaim {
                    id: 1usize,
                    relation: 1usize,
                    commitment: commitment.clone(),
                    claim: sign_claim,
                    view: PhysicalView::PackedLinear {
                        layout_digest: layout.digest,
                        terms: vec![PackedLinearTerm::new(
                            AkitaField::from_u64(3),
                            sign_family.physical_ref(),
                            0,
                            1,
                        )
                        .with_row_point(Vec::new())],
                    },
                    scale: AkitaField::from_u64(7),
                },
            ],
        };

        let mut prover_transcript = Blake2bTranscript::new(b"verifier-akita-packed");
        let proof = prove_akita_packed_openings(
            &prover_setup,
            &mut prover_transcript,
            &artifact,
            &source,
            &statement,
        )
        .expect("packed batch proof should be produced");

        let mut wrong_statement = statement.clone();
        wrong_statement.claims[0].commitment.layout_digest = [9; 32];
        let mut wrong_transcript = Blake2bTranscript::new(b"verifier-akita-packed");
        let error = prove_akita_packed_openings(
            &prover_setup,
            &mut wrong_transcript,
            &artifact,
            &source,
            &wrong_statement,
        )
        .expect_err("non-artifact commitment should reject");
        assert!(matches!(
            error,
            VerifierError::FinalOpeningBatchFailed { .. }
        ));

        let mut verifier_transcript = Blake2bTranscript::new(b"verifier-akita-packed");
        let result = <AkitaPackedScheme as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &statement,
            &proof,
        )
        .expect("packed batch proof should verify");

        assert_eq!(result.joint_commitment, commitment);
        assert_eq!(result.coefficients.len(), 2);
        assert_eq!(
            result.reduced_opening,
            result.coefficients[0] * instruction_claim + result.coefficients[1] * sign_claim
        );
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    }

    #[test]
    fn akita_clear_verifier_surface_is_nameable() {
        type TestTranscript = Blake2bTranscript<AkitaField>;
        type VerifyFn = fn(
            &AkitaVerifierPreprocessing,
            &JoltDevice,
            &AkitaJoltProof,
            Option<&AkitaCommitment>,
            &JoltProtocolConfig,
        ) -> Result<(), VerifierError>;
        let _verify: VerifyFn = verify_akita_clear::<TestTranscript>;
    }

    #[test]
    fn configured_layout_mismatch_rejects_before_commit() {
        let layout = tiny_layout();
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, _) = AkitaPackedScheme::setup(params);
        let source = SparsePackedWitness::try_new(layout.clone(), Vec::new())
            .expect("empty sparse source should build");
        let mut config = akita_lattice_protocol_config_for_layout(&layout);
        config.lattice.packed_witness.layout_digest = Some([9; 32]);

        let error = commit_akita_packed_witness_with_config(config, &prover_setup, &source)
            .expect_err("layout mismatch should reject");

        assert!(matches!(error, VerifierError::InvalidProtocolConfig { .. }));
    }
}
