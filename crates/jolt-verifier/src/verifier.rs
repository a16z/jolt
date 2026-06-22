//! Top-level verifier entry point.

use common::jolt_device::JoltDevice;
#[cfg(feature = "akita")]
use jolt_claims::protocols::jolt::formulas::dimensions::JoltFormulaDimensions;
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_field::{Field, RingAccumulator, WithAccumulator};
#[cfg(feature = "akita")]
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_openings::{
    BatchOpeningScheme, CommitmentLayoutDigest, CommitmentScheme, ZkBatchOpeningScheme,
};
use jolt_program::preprocess::{compute_max_ram_k, compute_min_ram_k};
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};

use crate::{
    config::{
        validate_proof_config, IncrementCommitmentMode, JoltProtocolConfig, PcsFamily, ProgramMode,
        ZkConfig,
    },
    preprocessing::JoltVerifierPreprocessing,
    proof::{CommitmentPayload, JoltProof},
    stages::{
        stage1, stage2, stage3, stage4, stage5, stage5_increment, stage6, stage7, stage8,
        zk::{blindfold, committed, inputs::BlindFoldInputs, outputs::zk_stage_outputs},
        CommittedProgramSchedule, PrecommittedSchedule,
    },
    VerifierError,
};

pub fn verify<F, PCS, VC, T>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC>,
    trusted_advice_commitment: Option<&PCS::Output>,
    zk: bool,
) -> Result<(), VerifierError>
where
    F: Field + AppendToTranscript,
    PCS: CommitmentScheme<Field = F>
        + BatchOpeningScheme
        + ZkBatchOpeningScheme<HidingCommitment = VC::Output>,
    PCS::Output: AppendToTranscript + CommitmentLayoutDigest,
    VC: VectorCommitment<Field = F>,
    VC::Output: Copy + HomomorphicCommitment<F> + AppendToTranscript,
    T: Transcript<Challenge = F>,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    verify_with_config::<F, PCS, VC, T>(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
        &JoltProtocolConfig::for_zk(zk),
    )
}

pub fn verify_clear<F, PCS, VC, T>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC>,
    trusted_advice_commitment: Option<&PCS::Output>,
) -> Result<(), VerifierError>
where
    F: Field + AppendToTranscript,
    PCS: CommitmentScheme<Field = F> + BatchOpeningScheme,
    PCS::Output: Clone + AppendToTranscript + CommitmentLayoutDigest,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    verify_clear_with_config::<F, PCS, VC, T>(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
        &JoltProtocolConfig::for_zk(false),
    )
}

pub fn verify_with_config<F, PCS, VC, T>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC>,
    trusted_advice_commitment: Option<&PCS::Output>,
    config: &JoltProtocolConfig,
) -> Result<(), VerifierError>
where
    F: Field + AppendToTranscript,
    PCS: CommitmentScheme<Field = F>
        + BatchOpeningScheme
        + ZkBatchOpeningScheme<HidingCommitment = VC::Output>,
    PCS::Output: AppendToTranscript + CommitmentLayoutDigest,
    VC: VectorCommitment<Field = F>,
    VC::Output: Copy + HomomorphicCommitment<F> + AppendToTranscript,
    T: Transcript<Challenge = F>,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    let zk = config_zk(config);
    let checked = validate_inputs(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.is_some(),
        zk,
    )?;
    validate_proof_consistency(proof, checked.zk)?;
    validate_proof_config(config, proof)?;
    validate_lattice_layout_binding(config, preprocessing, proof, &checked)?;
    validate_lattice_validity_proof_surface(config, preprocessing, proof, &checked)?;

    let mut transcript = T::new(b"Jolt");
    absorb_preamble(&checked, proof, &mut transcript);
    absorb_commitments(
        preprocessing,
        proof,
        trusted_advice_commitment,
        &mut transcript,
    )?;

    let stage1 = stage1::verify(&checked, preprocessing, proof, &mut transcript)?;
    let stage2 = stage2::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage2::deps(&stage1),
    )?;
    let stage3 = stage3::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage3::deps(&stage1, &stage2)?,
    )?;
    let stage4 = stage4::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage4::deps(&stage2, &stage3)?,
    )?;
    let stage5 = stage5::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage5::deps(&stage2, &stage4)?,
    )?;
    let _stage5_increment = stage5_increment::verify(
        &checked,
        proof,
        &mut transcript,
        stage5_increment::deps(&stage2, &stage4, &stage5),
    )?;
    let stage6 = stage6::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage6::deps(
            &stage1,
            &stage2,
            &stage3,
            &stage4,
            &stage5,
            _stage5_increment.as_ref(),
        )?,
    )?;
    let stage7 = stage7::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage7::deps(&stage4, &stage6)?,
    )?;
    verify_lattice_packed_validity(config, preprocessing, proof, &checked, &mut transcript)?;
    let stage8 = stage8::verify(
        &checked,
        preprocessing,
        proof,
        trusted_advice_commitment,
        &mut transcript,
        stage8::deps(&stage6, &stage7)?,
    )?;

    if checked.zk {
        let zk_stages = zk_stage_outputs::<PCS, VC>(
            &stage1, &stage2, &stage3, &stage4, &stage5, &stage6, &stage7, &stage8,
        )?;
        let blindfold = blindfold::build(BlindFoldInputs {
            checked: &checked,
            preprocessing,
            proof,
            stage1: zk_stages.stage1,
            stage2: zk_stages.stage2,
            stage3: zk_stages.stage3,
            stage4: zk_stages.stage4,
            stage5: zk_stages.stage5,
            stage6: zk_stages.stage6,
            stage7: zk_stages.stage7,
            stage8: zk_stages.stage8,
        })?;
        let vc_setup = preprocessing
            .vc_setup
            .as_ref()
            .ok_or(VerifierError::MissingVectorCommitmentSetup)?;
        transcript.append(&Label(b"BlindFold"));
        blindfold
            .protocol
            .verify::<VC, T>(proof.blindfold_proof()?, vc_setup, &mut transcript)
            .map_err(|error| VerifierError::BlindFoldVerificationFailed {
                reason: error.to_string(),
            })?;
        return Ok(());
    }

    let stage8::Stage8Output::Clear(_stage8) = stage8 else {
        return Err(VerifierError::ExpectedClearProof { field: "stage8" });
    };

    Ok(())
}

pub fn verify_clear_with_config<F, PCS, VC, T>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC>,
    trusted_advice_commitment: Option<&PCS::Output>,
    config: &JoltProtocolConfig,
) -> Result<(), VerifierError>
where
    F: Field + AppendToTranscript,
    PCS: CommitmentScheme<Field = F> + BatchOpeningScheme,
    PCS::Output: Clone + AppendToTranscript + CommitmentLayoutDigest,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    if config_zk(config) {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "verify_clear requires a transparent protocol config".to_string(),
        });
    }

    let checked = validate_inputs(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.is_some(),
        false,
    )?;
    validate_proof_consistency(proof, false)?;
    validate_proof_config(config, proof)?;
    validate_lattice_layout_binding(config, preprocessing, proof, &checked)?;
    validate_lattice_validity_proof_surface(config, preprocessing, proof, &checked)?;

    let mut transcript = T::new(b"Jolt");
    absorb_preamble(&checked, proof, &mut transcript);
    absorb_commitments(
        preprocessing,
        proof,
        trusted_advice_commitment,
        &mut transcript,
    )?;

    let stage1 = stage1::verify(&checked, preprocessing, proof, &mut transcript)?;
    let stage2 = stage2::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage2::deps(&stage1),
    )?;
    let stage3 = stage3::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage3::deps(&stage1, &stage2)?,
    )?;
    let stage4 = stage4::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage4::deps(&stage2, &stage3)?,
    )?;
    let stage5 = stage5::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage5::deps(&stage2, &stage4)?,
    )?;
    let _stage5_increment = stage5_increment::verify(
        &checked,
        proof,
        &mut transcript,
        stage5_increment::deps(&stage2, &stage4, &stage5),
    )?;
    let stage6 = stage6::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage6::deps(
            &stage1,
            &stage2,
            &stage3,
            &stage4,
            &stage5,
            _stage5_increment.as_ref(),
        )?,
    )?;
    let stage7 = stage7::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage7::deps(&stage4, &stage6)?,
    )?;
    verify_lattice_packed_validity(config, preprocessing, proof, &checked, &mut transcript)?;
    let _stage8 = stage8::verify_clear(
        &checked,
        preprocessing,
        proof,
        trusted_advice_commitment,
        &mut transcript,
        stage8::deps(&stage6, &stage7)?,
    )?;

    Ok(())
}

pub fn stage8_batch_statement<F, PCS, VC, T, ZkProof>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment: Option<&PCS::Output>,
    zk: bool,
) -> Result<stage8::Stage8BatchStatement<F, PCS::Output>, VerifierError>
where
    F: Field + AppendToTranscript,
    PCS: CommitmentScheme<Field = F> + BatchOpeningScheme,
    PCS::Output: Clone + AppendToTranscript + CommitmentLayoutDigest,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    stage8_batch_statement_with_config::<F, PCS, VC, T, ZkProof>(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
        &JoltProtocolConfig::for_zk(zk),
    )
}

pub fn stage8_batch_statement_with_config<F, PCS, VC, T, ZkProof>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment: Option<&PCS::Output>,
    config: &JoltProtocolConfig,
) -> Result<stage8::Stage8BatchStatement<F, PCS::Output>, VerifierError>
where
    F: Field + AppendToTranscript,
    PCS: CommitmentScheme<Field = F> + BatchOpeningScheme,
    PCS::Output: Clone + AppendToTranscript + CommitmentLayoutDigest,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    stage8_batch_statement_with_config_and_transcript::<F, PCS, VC, T, ZkProof>(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
        config,
    )
    .map(|(statement, _transcript)| statement)
}

pub fn stage8_batch_statement_with_config_and_transcript<F, PCS, VC, T, ZkProof>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment: Option<&PCS::Output>,
    config: &JoltProtocolConfig,
) -> Result<(stage8::Stage8BatchStatement<F, PCS::Output>, T), VerifierError>
where
    F: Field + AppendToTranscript,
    PCS: CommitmentScheme<Field = F> + BatchOpeningScheme,
    PCS::Output: Clone + AppendToTranscript + CommitmentLayoutDigest,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    let (checked, mut transcript, stage6, stage7) =
        stage7_transcript_with_config_impl::<F, PCS, VC, T, ZkProof>(
            preprocessing,
            public_io,
            proof,
            trusted_advice_commitment,
            config,
            true,
        )?;
    verify_lattice_packed_validity(config, preprocessing, proof, &checked, &mut transcript)?;
    let statement = stage8::batch_statement(
        &checked,
        preprocessing,
        proof,
        trusted_advice_commitment,
        stage8::deps(&stage6, &stage7)?,
    )?;
    Ok((statement, transcript))
}

pub fn lattice_packed_validity_transcript_with_config<F, PCS, VC, T, ZkProof>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment: Option<&PCS::Output>,
    config: &JoltProtocolConfig,
) -> Result<(CheckedInputs, T), VerifierError>
where
    F: Field + AppendToTranscript,
    PCS: CommitmentScheme<Field = F> + BatchOpeningScheme,
    PCS::Output: Clone + AppendToTranscript,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    let (checked, transcript, _, _) = stage7_transcript_with_config_impl::<F, PCS, VC, T, ZkProof>(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
        config,
        false,
    )?;
    Ok((checked, transcript))
}

type Stage7TranscriptContext<F, C, T> = (
    CheckedInputs,
    T,
    stage6::Stage6Output<F, C>,
    stage7::Stage7Output<F, C>,
);

fn stage7_transcript_with_config_impl<F, PCS, VC, T, ZkProof>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment: Option<&PCS::Output>,
    config: &JoltProtocolConfig,
    require_lattice_validity_surface: bool,
) -> Result<Stage7TranscriptContext<F, VC::Output, T>, VerifierError>
where
    F: Field + AppendToTranscript,
    PCS: CommitmentScheme<Field = F> + BatchOpeningScheme,
    PCS::Output: Clone + AppendToTranscript,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    let zk = config_zk(config);
    let checked = validate_inputs(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.is_some(),
        zk,
    )?;
    validate_proof_consistency(proof, checked.zk)?;
    validate_proof_config(config, proof)?;
    validate_lattice_layout_binding(config, preprocessing, proof, &checked)?;
    if require_lattice_validity_surface {
        validate_lattice_validity_proof_surface(config, preprocessing, proof, &checked)?;
    }

    let mut transcript = T::new(b"Jolt");
    absorb_preamble(&checked, proof, &mut transcript);
    absorb_commitments(
        preprocessing,
        proof,
        trusted_advice_commitment,
        &mut transcript,
    )?;

    let stage1 = stage1::verify(&checked, preprocessing, proof, &mut transcript)?;
    let stage2 = stage2::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage2::deps(&stage1),
    )?;
    let stage3 = stage3::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage3::deps(&stage1, &stage2)?,
    )?;
    let stage4 = stage4::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage4::deps(&stage2, &stage3)?,
    )?;
    let stage5 = stage5::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage5::deps(&stage2, &stage4)?,
    )?;
    let _stage5_increment = stage5_increment::verify(
        &checked,
        proof,
        &mut transcript,
        stage5_increment::deps(&stage2, &stage4, &stage5),
    )?;
    let stage6 = stage6::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage6::deps(
            &stage1,
            &stage2,
            &stage3,
            &stage4,
            &stage5,
            _stage5_increment.as_ref(),
        )?,
    )?;
    let stage7 = stage7::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage7::deps(&stage4, &stage6)?,
    )?;
    Ok((checked, transcript, stage6, stage7))
}

fn verify_lattice_packed_validity<F, PCS, VC, T, ZkProof>(
    config: &JoltProtocolConfig,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    checked: &CheckedInputs,
    transcript: &mut T,
) -> Result<(), VerifierError>
where
    F: Field,
    PCS: CommitmentScheme<Field = F> + BatchOpeningScheme,
    PCS::Output: Clone,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    if proof.commitments.family() != PcsFamily::Lattice {
        return Ok(());
    }

    #[cfg(not(feature = "akita"))]
    {
        let _ = (config, preprocessing, proof, checked, transcript);
        Err(VerifierError::InvalidProtocolConfig {
            reason: "lattice packed validity verification requires the jolt-verifier akita feature"
                .to_string(),
        })
    }

    #[cfg(feature = "akita")]
    {
        if checked.zk {
            return Err(VerifierError::InvalidProtocolConfig {
                reason:
                    "lattice packed validity verification currently requires transparent claims"
                        .to_string(),
            });
        }
        let payload =
            proof
                .commitments
                .as_akita()
                .ok_or_else(|| VerifierError::InvalidProtocolConfig {
                    reason: "lattice packed validity verification requires Akita commitments"
                        .to_string(),
                })?;
        let validity_claims = proof
            .clear_claims()?
            .stage7
            .lattice_packed_validity
            .as_ref()
            .ok_or(VerifierError::MissingAkitaPackedValidityProof {
                field: "opening_claims",
            })?;
        let sumcheck_proof = proof
            .stages
            .lattice_packed_validity_sumcheck_proof
            .as_ref()
            .ok_or(VerifierError::MissingAkitaPackedValidityProof {
                field: "sumcheck_proof",
            })?;
        let opening_proof = proof.lattice_packed_validity_opening_proof.as_ref().ok_or(
            VerifierError::MissingAkitaPackedValidityProof {
                field: "opening_proof",
            },
        )?;

        let log_t = checked.trace_length.ilog2() as usize;
        let formula_dimensions = JoltFormulaDimensions::try_from(proof.one_hot_config.dimensions(
            log_t,
            2 * RISCV_XLEN,
            preprocessing.program.bytecode_len(),
            checked.ram_K,
        ))
        .map_err(|error| VerifierError::InvalidProtocolConfig {
            reason: format!("invalid lattice formula dimensions: {error}"),
        })?;
        let layout = stage8::derive_akita_packed_witness_layout(
            config,
            log_t,
            proof.one_hot_config.committed_chunk_bits(),
            formula_dimensions.ra_layout,
            &checked.precommitted,
        )?;
        stage8::verify_lattice_packed_validity_proof::<F, PCS, T, _>(
            &preprocessing.pcs_setup,
            transcript,
            config,
            proof.one_hot_config.committed_chunk_bits(),
            &checked.precommitted,
            &layout,
            payload.packed_witness.clone(),
            sumcheck_proof,
            &validity_claims.opening_claims,
            opening_proof,
        )
    }
}

fn config_zk(config: &JoltProtocolConfig) -> bool {
    matches!(config.zk, ZkConfig::BlindFold)
}

#[expect(non_snake_case, reason = "Matches current jolt-core proof field name.")]
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedInputs {
    pub public_io: JoltDevice,
    pub zk: bool,
    pub trace_length: usize,
    pub ram_K: usize,
    pub entry_address: u64,
    pub preprocessing_digest: [u8; 32],
    pub trusted_advice_commitment_present: bool,
    pub vc_capacity: Option<usize>,
    pub precommitted: PrecommittedSchedule,
}

pub fn validate_inputs<PCS, VC, ZkProof>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment_present: bool,
    zk: bool,
) -> Result<CheckedInputs, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let memory_layout = preprocessing.program.memory_layout();
    if &public_io.memory_layout != memory_layout {
        return Err(VerifierError::MemoryLayoutMismatch);
    }

    let max_input_size = memory_layout.max_input_size as usize;
    if public_io.inputs.len() > max_input_size {
        return Err(VerifierError::InputTooLarge {
            got: public_io.inputs.len(),
            max: max_input_size,
        });
    }

    let max_output_size = memory_layout.max_output_size as usize;
    if public_io.outputs.len() > max_output_size {
        return Err(VerifierError::OutputTooLarge {
            got: public_io.outputs.len(),
            max: max_output_size,
        });
    }

    if !proof.trace_length.is_power_of_two()
        || proof.trace_length > preprocessing.program.max_padded_trace_length()
    {
        return Err(VerifierError::InvalidTraceLength {
            got: proof.trace_length,
            max: preprocessing.program.max_padded_trace_length(),
        });
    }

    let min_ram_k = compute_min_ram_k(
        preprocessing.program.min_bytecode_address(),
        preprocessing.program.program_image_len_words(),
        memory_layout,
    )
    .map_err(|error| VerifierError::InvalidMemoryLayout {
        reason: error.to_string(),
    })?;
    let max_ram_k =
        compute_max_ram_k(memory_layout).map_err(|error| VerifierError::InvalidMemoryLayout {
            reason: error.to_string(),
        })?;
    if !proof.ram_K.is_power_of_two() || proof.ram_K < min_ram_k || proof.ram_K > max_ram_k {
        return Err(VerifierError::InvalidRamK {
            got: proof.ram_K,
            min: min_ram_k,
            max: max_ram_k,
        });
    }

    let vc_capacity = if zk {
        Some(validate_zk_vector_commitment_setup::<PCS, VC>(
            preprocessing,
        )?)
    } else {
        None
    };

    let mut normalized_public_io = public_io.clone();
    normalized_public_io.outputs.truncate(
        normalized_public_io
            .outputs
            .iter()
            .rposition(|&byte| byte != 0)
            .map_or(0, |position| position + 1),
    );

    let committed_program = preprocessing
        .program
        .committed()
        .map(|committed| {
            let program_image_start_index = memory_layout
                .remapped_word_address(committed.meta.min_bytecode_address)
                .map_err(|error| VerifierError::InvalidCommittedProgram {
                    reason: error.to_string(),
                })?;
            if committed.meta.entry_bytecode_index >= committed.meta.bytecode_len {
                return Err(VerifierError::InvalidCommittedProgram {
                    reason: format!(
                        "entry bytecode index {} is out of range for bytecode length {}",
                        committed.meta.entry_bytecode_index, committed.meta.bytecode_len
                    ),
                });
            }
            Ok(CommittedProgramSchedule {
                bytecode_len: committed.meta.bytecode_len,
                bytecode_chunk_count: committed.bytecode_chunk_count(),
                program_image_len_words: committed.meta.program_image_len_words,
                program_image_start_index: program_image_start_index as usize,
            })
        })
        .transpose()?;
    let precommitted = PrecommittedSchedule::new(
        proof.trace_polynomial_order,
        proof.trace_length.ilog2() as usize,
        proof.one_hot_config.committed_chunk_bits(),
        trusted_advice_commitment_present.then_some(memory_layout.max_trusted_advice_size as usize),
        proof
            .untrusted_advice_commitment
            .is_some()
            .then_some(memory_layout.max_untrusted_advice_size as usize),
        committed_program,
    )
    .map_err(|error| VerifierError::InvalidPrecommittedSchedule {
        reason: error.to_string(),
    })?;

    Ok(CheckedInputs {
        public_io: normalized_public_io,
        zk,
        trace_length: proof.trace_length,
        ram_K: proof.ram_K,
        entry_address: preprocessing.program.entry_address(),
        preprocessing_digest: preprocessing.preprocessing_digest,
        trusted_advice_commitment_present,
        vc_capacity,
        precommitted,
    })
}

fn validate_lattice_layout_binding<PCS, VC, ZkProof>(
    config: &JoltProtocolConfig,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    checked: &CheckedInputs,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    if proof.commitments.family() != PcsFamily::Lattice {
        return Ok(());
    }
    validate_lattice_precommitted_surface(config, checked)?;

    #[cfg(not(feature = "akita"))]
    {
        let _ = (config, preprocessing, checked);
        Err(VerifierError::InvalidProtocolConfig {
            reason: "lattice PCS mode requires the jolt-verifier akita feature".to_string(),
        })
    }

    #[cfg(feature = "akita")]
    {
        let log_t = checked.trace_length.ilog2() as usize;
        let formula_dimensions = JoltFormulaDimensions::try_from(proof.one_hot_config.dimensions(
            log_t,
            2 * RISCV_XLEN,
            preprocessing.program.bytecode_len(),
            checked.ram_K,
        ))
        .map_err(|error| VerifierError::InvalidProtocolConfig {
            reason: format!("invalid lattice formula dimensions: {error}"),
        })?;
        let layout = stage8::derive_akita_packed_witness_layout(
            config,
            log_t,
            proof.one_hot_config.committed_chunk_bits(),
            formula_dimensions.ra_layout,
            &checked.precommitted,
        )?;
        stage8::validate_akita_packed_witness_layout_config(config, &layout)?;
        stage8::validate_akita_packed_witness_validity_config(
            config,
            proof.one_hot_config.committed_chunk_bits(),
            &checked.precommitted,
        )
    }
}

fn validate_lattice_precommitted_surface(
    config: &JoltProtocolConfig,
    checked: &CheckedInputs,
) -> Result<(), VerifierError> {
    let trusted_advice_present = checked.precommitted.trusted_advice.is_some();
    if config.lattice.advice.trusted != trusted_advice_present {
        let reason = if trusted_advice_present {
            "trusted advice precommitted schedule requires trusted advice lattice mode"
        } else {
            "trusted advice lattice mode requires a trusted advice precommitted schedule"
        };
        return Err(VerifierError::InvalidProtocolConfig {
            reason: reason.to_string(),
        });
    }

    let untrusted_advice_present = checked.precommitted.untrusted_advice.is_some();
    if config.lattice.advice.untrusted != untrusted_advice_present {
        let reason = if untrusted_advice_present {
            "untrusted advice precommitted schedule requires untrusted advice lattice mode"
        } else {
            "untrusted advice lattice mode requires an untrusted advice precommitted schedule"
        };
        return Err(VerifierError::InvalidProtocolConfig {
            reason: reason.to_string(),
        });
    }

    Ok(())
}

fn validate_lattice_validity_proof_surface<PCS, VC, ZkProof>(
    config: &JoltProtocolConfig,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    checked: &CheckedInputs,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let lattice = proof.commitments.family() == PcsFamily::Lattice;
    let validity_claims = match &proof.claims {
        crate::proof::JoltProofClaims::Clear(claims) => {
            claims.stage7.lattice_packed_validity.as_ref()
        }
        crate::proof::JoltProofClaims::Zk { .. } => None,
    };

    if !lattice {
        if proof
            .stages
            .lattice_packed_validity_sumcheck_proof
            .is_some()
        {
            return Err(VerifierError::UnexpectedAkitaPackedValidityProof {
                field: "sumcheck_proof",
            });
        }
        if proof.lattice_packed_validity_opening_proof.is_some() {
            return Err(VerifierError::UnexpectedAkitaPackedValidityProof {
                field: "opening_proof",
            });
        }
        if validity_claims.is_some() {
            return Err(VerifierError::UnexpectedAkitaPackedValidityProof {
                field: "opening_claims",
            });
        }
        return Ok(());
    }

    if proof
        .stages
        .lattice_packed_validity_sumcheck_proof
        .is_none()
    {
        return Err(VerifierError::MissingAkitaPackedValidityProof {
            field: "sumcheck_proof",
        });
    }
    if proof.lattice_packed_validity_opening_proof.is_none() {
        return Err(VerifierError::MissingAkitaPackedValidityProof {
            field: "opening_proof",
        });
    }
    let validity_claims =
        validity_claims.ok_or(VerifierError::MissingAkitaPackedValidityProof {
            field: "opening_claims",
        })?;

    #[cfg(not(feature = "akita"))]
    {
        let _ = (config, preprocessing, checked, validity_claims);
        Err(VerifierError::InvalidProtocolConfig {
            reason: "lattice packed validity proof requires the jolt-verifier akita feature"
                .to_string(),
        })
    }

    #[cfg(feature = "akita")]
    {
        let log_t = checked.trace_length.ilog2() as usize;
        let formula_dimensions = JoltFormulaDimensions::try_from(proof.one_hot_config.dimensions(
            log_t,
            2 * RISCV_XLEN,
            preprocessing.program.bytecode_len(),
            checked.ram_K,
        ))
        .map_err(|error| VerifierError::InvalidProtocolConfig {
            reason: format!("invalid lattice formula dimensions: {error}"),
        })?;
        let layout = stage8::derive_akita_packed_witness_layout(
            config,
            log_t,
            proof.one_hot_config.committed_chunk_bits(),
            formula_dimensions.ra_layout,
            &checked.precommitted,
        )?;
        let requirements = stage8::derive_akita_packed_validity_requirements(
            config,
            proof.one_hot_config.committed_chunk_bits(),
            &checked.precommitted,
        )?;
        let statements = stage8::derive_akita_packed_validity_statements(&layout, &requirements)?;
        let expected_opening_claims = stage8::lattice_packed_validity_opening_count(&statements);
        if validity_claims.opening_claims.len() != expected_opening_claims {
            return Err(VerifierError::AkitaPackedValidityClaimCountMismatch {
                expected: expected_opening_claims,
                got: validity_claims.opening_claims.len(),
            });
        }
        Ok(())
    }
}

fn validate_zk_vector_commitment_setup<PCS, VC>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
) -> Result<usize, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let setup = preprocessing
        .vc_setup
        .as_ref()
        .ok_or(VerifierError::MissingVectorCommitmentSetup)?;
    let required = committed::zk_vector_commitment_capacity_requirement();
    let got = VC::capacity(setup);
    if got < required {
        return Err(VerifierError::InvalidVectorCommitmentCapacity { required, got });
    }

    Ok(got)
}

pub(crate) fn absorb_preamble<PCS, VC, ZkProof, T>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
) where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let public_io = &checked.public_io;
    absorb_labeled_bytes(
        transcript,
        b"preprocessing_digest",
        &checked.preprocessing_digest,
    );
    absorb_labeled_u64(
        transcript,
        b"max_input_size",
        public_io.memory_layout.max_input_size,
    );
    absorb_labeled_u64(
        transcript,
        b"max_output_size",
        public_io.memory_layout.max_output_size,
    );
    absorb_labeled_u64(transcript, b"heap_size", public_io.memory_layout.heap_size);
    absorb_labeled_bytes(transcript, b"inputs", &public_io.inputs);
    absorb_labeled_bytes(transcript, b"outputs", &public_io.outputs);
    absorb_labeled_u64(transcript, b"panic", public_io.panic as u64);
    absorb_labeled_u64(transcript, b"ram_K", checked.ram_K as u64);
    absorb_labeled_u64(transcript, b"trace_length", checked.trace_length as u64);
    absorb_labeled_u64(transcript, b"entry_address", checked.entry_address);
    absorb_labeled_u64(
        transcript,
        b"ram_rw_phase1_num_rounds",
        proof.rw_config.ram_rw_phase1_num_rounds as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"ram_rw_phase2_num_rounds",
        proof.rw_config.ram_rw_phase2_num_rounds as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"registers_rw_phase1_num_rounds",
        proof.rw_config.registers_rw_phase1_num_rounds as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"registers_rw_phase2_num_rounds",
        proof.rw_config.registers_rw_phase2_num_rounds as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"log_k_chunk",
        proof.one_hot_config.log_k_chunk as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"lookups_ra_virtual_log_k_chunk",
        proof.one_hot_config.lookups_ra_virtual_log_k_chunk as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"dory_layout",
        proof.trace_polynomial_order.transcript_scalar(),
    );
}

pub(crate) fn absorb_commitments<PCS, VC, ZkProof, T>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment: Option<&PCS::Output>,
    transcript: &mut T,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    PCS::Output: AppendToTranscript,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    match &proof.commitments {
        CommitmentPayload::Dory(commitments) => {
            let mut absorb_commitment = |commitment: &PCS::Output| {
                append_payload_label(transcript, b"commitment", commitment);
                transcript.append(commitment);
            };
            absorb_commitment(&commitments.rd_inc);
            absorb_commitment(&commitments.ram_inc);
            for commitment in &commitments.ra.instruction {
                absorb_commitment(commitment);
            }
            for commitment in &commitments.ra.ram {
                absorb_commitment(commitment);
            }
            for commitment in &commitments.ra.bytecode {
                absorb_commitment(commitment);
            }
            if let Some(untrusted_advice_commitment) = &proof.untrusted_advice_commitment {
                append_payload_label(transcript, b"untrusted_advice", untrusted_advice_commitment);
                transcript.append(untrusted_advice_commitment);
            }
            if let Some(trusted_advice_commitment) = trusted_advice_commitment {
                append_payload_label(transcript, b"trusted_advice", trusted_advice_commitment);
                transcript.append(trusted_advice_commitment);
            }
            if let Some(committed) = preprocessing.program.committed() {
                for commitment in &committed.bytecode_chunk_commitments {
                    append_payload_label(transcript, b"bytecode_chunk_commit", commitment);
                    transcript.append(commitment);
                }
                append_payload_label(
                    transcript,
                    b"program_image_commitment",
                    &committed.program_image_commitment,
                );
                transcript.append(&committed.program_image_commitment);
            }
        }
        CommitmentPayload::Akita(payload) => {
            absorb_akita_protocol_header(transcript, &proof.protocol);
            append_payload_label(transcript, b"akita_packed_witness", &payload.packed_witness);
            transcript.append(&payload.packed_witness);
            absorb_labeled_bytes(transcript, b"akita_layout_digest", &payload.layout_digest);
            absorb_labeled_u64(transcript, b"akita_d_pack", payload.d_pack as u64);
            if let Some(validity_digest) = proof.protocol.lattice.packed_witness.validity_digest {
                absorb_labeled_bytes(transcript, b"akita_validity_digest", &validity_digest);
            }
        }
    }
    Ok(())
}

fn absorb_akita_protocol_header<T: Transcript>(transcript: &mut T, protocol: &JoltProtocolConfig) {
    transcript.append(&Label(b"akita_protocol_header"));
    absorb_labeled_u64(
        transcript,
        b"akita_pcs_curve",
        (protocol.pcs == PcsFamily::Curve) as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"akita_pcs_lattice",
        (protocol.pcs == PcsFamily::Lattice) as u64,
    );
    absorb_labeled_u64(transcript, b"akita_zk", zk_config_tag(protocol.zk));
    absorb_labeled_u64(
        transcript,
        b"akita_program_mode",
        program_mode_tag(protocol.lattice.program_mode),
    );
    absorb_labeled_u64(
        transcript,
        b"akita_increment_mode",
        increment_mode_tag(protocol.lattice.increment_mode),
    );
    absorb_labeled_u64(transcript, b"akita_lattice_zk", protocol.lattice.zk as u64);
    absorb_labeled_u64(
        transcript,
        b"akita_field_inline",
        protocol.lattice.field_inline.enabled as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"akita_advice_trusted",
        protocol.lattice.advice.trusted as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"akita_advice_untrusted",
        protocol.lattice.advice.untrusted as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"akita_field_rd_inc_family",
        protocol.lattice.packed_witness.field_rd_inc_family as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"akita_trusted_advice_family",
        protocol.lattice.packed_witness.trusted_advice_family as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"akita_untrusted_advice_family",
        protocol.lattice.packed_witness.untrusted_advice_family as u64,
    );
}

const fn zk_config_tag(zk: ZkConfig) -> u64 {
    match zk {
        ZkConfig::Transparent => 0,
        ZkConfig::BlindFold => 1,
    }
}

const fn program_mode_tag(mode: ProgramMode) -> u64 {
    match mode {
        ProgramMode::Full => 0,
        ProgramMode::Committed => 1,
    }
}

const fn increment_mode_tag(mode: IncrementCommitmentMode) -> u64 {
    match mode {
        IncrementCommitmentMode::Dense => 0,
        IncrementCommitmentMode::SeparateOneHot => 1,
        IncrementCommitmentMode::FusedOneHot => 2,
    }
}

fn append_payload_label<T, A>(transcript: &mut T, label: &'static [u8], payload: &A)
where
    T: Transcript,
    A: AppendToTranscript,
{
    if let Some(len) = payload.transcript_payload_len() {
        transcript.append(&LabelWithCount(label, len));
    } else {
        transcript.append(&Label(label));
    }
}

fn absorb_labeled_bytes<T: Transcript>(transcript: &mut T, label: &'static [u8], bytes: &[u8]) {
    transcript.append(&LabelWithCount(label, bytes.len() as u64));
    transcript.append_bytes(bytes);
}

fn absorb_labeled_u64<T: Transcript>(transcript: &mut T, label: &'static [u8], value: u64) {
    transcript.append(&Label(label));
    transcript.append(&U64Word(value));
}

pub fn validate_proof_consistency<PCS, VC, ZkProof>(
    proof: &JoltProof<PCS, VC, ZkProof>,
    zk: bool,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    validate_sumcheck_representation(
        &proof.stages.stage1_uni_skip_first_round_proof,
        "stage1_uni_skip_first_round_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage1_sumcheck_proof,
        "stage1_sumcheck_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage2_uni_skip_first_round_proof,
        "stage2_uni_skip_first_round_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage2_sumcheck_proof,
        "stage2_sumcheck_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage3_sumcheck_proof,
        "stage3_sumcheck_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage4_sumcheck_proof,
        "stage4_sumcheck_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage5_sumcheck_proof,
        "stage5_sumcheck_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage6a_sumcheck_proof,
        "stage6a_sumcheck_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage6b_sumcheck_proof,
        "stage6b_sumcheck_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage7_sumcheck_proof,
        "stage7_sumcheck_proof",
        zk,
    )?;

    match (&proof.claims, zk) {
        (crate::proof::JoltProofClaims::Clear(_), false)
        | (crate::proof::JoltProofClaims::Zk { .. }, true) => {}
        (crate::proof::JoltProofClaims::Clear(_), true) => {
            return Err(VerifierError::UnexpectedOpeningClaims);
        }
        (crate::proof::JoltProofClaims::Zk { .. }, false) => {
            return Err(VerifierError::UnexpectedBlindFoldProof);
        }
    }
    Ok(())
}

fn validate_sumcheck_representation<F, RoundCommitment>(
    proof: &SumcheckProof<F, RoundCommitment>,
    field: &'static str,
    zk: bool,
) -> Result<(), VerifierError>
where
    F: Field,
{
    if proof.is_committed() == zk {
        return Ok(());
    }

    if zk {
        Err(VerifierError::ExpectedCommittedProof { field })
    } else {
        Err(VerifierError::ExpectedClearProof { field })
    }
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::panic,
        reason = "verifier unit tests fail loudly on setup errors"
    )]

    use super::*;
    use crate::proof::{
        AkitaCommitmentPayload, ClearProofClaims, CommitmentPayload, JoltProofClaims,
        JoltStageProofs,
    };
    use common::jolt_device::{JoltDevice, MemoryConfig};
    #[cfg(feature = "akita")]
    use jolt_claims::protocols::jolt::{
        formulas::{
            claim_reductions::bytecode, dimensions::JoltFormulaDimensions,
            ra::JoltRaPolynomialLayout,
        },
        lattice_packed_validity_digest, JoltCommittedPolynomial, JoltOneHotConfig, JoltOpeningId,
        JoltReadWriteConfig, JoltRelationId,
    };
    #[cfg(not(feature = "akita"))]
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig};
    use jolt_crypto::{Bn254G1, Commitment, Pedersen, PedersenSetup, VectorCommitmentOpening};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_openings::{
        BatchOpeningResult, BatchOpeningScheme, BatchOpeningStatement, CommitmentLayoutDigest,
        CommitmentScheme, OpeningsError,
    };
    use jolt_poly::{MultilinearPoly, Polynomial};
    use jolt_program::preprocess::{
        BytecodePreprocessing, JoltProgramPreprocessing, ProgramMetadata, RAMPreprocessing,
    };
    use jolt_sumcheck::{
        ClearProof, ClearSumcheckProof, CommittedSumcheckProof, CompressedSumcheckProof,
    };
    use jolt_transcript::Transcript;
    use num_traits::Zero;

    #[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct TestPcs;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct TestCommitment;

    impl CommitmentLayoutDigest for TestCommitment {
        fn layout_digest(&self) -> Option<[u8; 32]> {
            None
        }
    }

    impl Commitment for TestPcs {
        type Output = TestCommitment;
    }

    impl CommitmentScheme for TestPcs {
        type Field = Fr;
        type Proof = ();
        type ProverSetup = ();
        type VerifierSetup = ();
        type Polynomial = Polynomial<Fr>;
        type OpeningHint = ();
        type SetupParams = ();

        fn setup(_params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup) {
            ((), ())
        }

        fn verifier_setup(_prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {}

        fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
            _poly: &P,
            _setup: &Self::ProverSetup,
        ) -> (Self::Output, Self::OpeningHint) {
            (TestCommitment, ())
        }

        fn open(
            _poly: &Self::Polynomial,
            _point: &[Self::Field],
            _eval: Self::Field,
            _setup: &Self::ProverSetup,
            _hint: Option<Self::OpeningHint>,
            _transcript: &mut impl Transcript<Challenge = Self::Field>,
        ) -> Self::Proof {
        }

        fn verify(
            _commitment: &Self::Output,
            _point: &[Self::Field],
            _eval: Self::Field,
            _proof: &Self::Proof,
            _setup: &Self::VerifierSetup,
            _transcript: &mut impl Transcript<Challenge = Self::Field>,
        ) -> Result<(), OpeningsError> {
            Ok(())
        }

        fn bind_opening_inputs(
            _transcript: &mut impl Transcript<Challenge = Self::Field>,
            _point: &[Self::Field],
            _eval: &Self::Field,
        ) {
        }
    }

    impl BatchOpeningScheme for TestPcs {
        fn prove_batch<T, OpeningId, RelationId>(
            _setup: &Self::ProverSetup,
            _transcript: &mut T,
            _statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
            _polynomials: &[Self::Polynomial],
            _hints: Vec<Self::OpeningHint>,
        ) -> Result<Self::Proof, OpeningsError>
        where
            T: Transcript<Challenge = Self::Field>,
        {
            Ok(())
        }

        fn verify_batch<T, OpeningId, RelationId>(
            _setup: &Self::VerifierSetup,
            _transcript: &mut T,
            statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
            _proof: &Self::Proof,
        ) -> Result<BatchOpeningResult<Self::Field, Self::Output>, OpeningsError>
        where
            T: Transcript<Challenge = Self::Field>,
        {
            let coefficients = vec![Fr::from_u64(1); statement.claims.len()];
            let reduced_opening = statement
                .claims
                .iter()
                .map(|claim| claim.claim * claim.scale)
                .sum();
            Ok(BatchOpeningResult {
                coefficients,
                joint_commitment: TestCommitment,
                reduced_opening,
            })
        }
    }

    #[cfg(feature = "akita")]
    type AkitaStage8StatementFixture = (
        JoltVerifierPreprocessing<TestPcs, Pedersen<Bn254G1>>,
        CheckedInputs,
        TestProof,
        stage6::Stage6ClearOutput<Fr>,
        stage7::Stage7ClearOutput<Fr>,
    );

    impl jolt_transcript::AppendToTranscript for TestCommitment {
        fn append_to_transcript<T: Transcript>(&self, _transcript: &mut T) {}
    }

    #[derive(Default)]
    struct RecordingTranscript {
        bytes: Vec<u8>,
    }

    impl Transcript for RecordingTranscript {
        type Challenge = Fr;

        fn new(label: &'static [u8]) -> Self {
            let mut transcript = Self::default();
            transcript.append_bytes(label);
            transcript
        }

        fn append_bytes(&mut self, bytes: &[u8]) {
            self.bytes.extend_from_slice(bytes);
        }

        fn challenge(&mut self) -> Self::Challenge {
            Fr::zero()
        }

        fn state(&self) -> [u8; 32] {
            [0; 32]
        }
    }

    type TestProof = JoltProof<TestPcs, Pedersen<Bn254G1>>;
    type TestClaims = JoltProofClaims<Fr, jolt_blindfold::BlindFoldProof<Fr, Bn254G1>>;

    #[test]
    fn proof_wrapper_uses_modular_trait_bounds() {
        fn assert_proof_traits<T>()
        where
            T: Clone
                + std::fmt::Debug
                + PartialEq
                + Eq
                + Send
                + Sync
                + 'static
                + serde::Serialize
                + serde::de::DeserializeOwned,
        {
        }

        assert_proof_traits::<TestProof>();
    }

    #[test]
    fn accepts_standard_proof_consistency() {
        let proof = proof_with_zk(false, clear_claims());

        assert!(validate_proof_consistency(&proof, false).is_ok());
    }

    #[test]
    fn accepts_zk_proof_consistency() {
        let proof = proof_with_zk(true, zk_claims());

        assert!(validate_proof_consistency(&proof, true).is_ok());
    }

    #[test]
    fn rejects_wrong_stage_representation() {
        let mut proof = proof_with_zk(false, clear_claims());
        proof.stages.stage5_sumcheck_proof =
            SumcheckProof::Committed(CommittedSumcheckProof::default());

        assert!(matches!(
            validate_proof_consistency(&proof, false),
            Err(VerifierError::ExpectedClearProof {
                field: "stage5_sumcheck_proof",
            })
        ));
    }

    #[test]
    fn rejects_wrong_verifier_zk_flag() {
        let proof = proof_with_zk(false, clear_claims());

        assert!(matches!(
            validate_proof_consistency(&proof, true),
            Err(VerifierError::ExpectedCommittedProof {
                field: "stage1_uni_skip_first_round_proof",
            })
        ));
    }

    #[test]
    fn checks_payload_for_selected_zk_flag() {
        assert!(matches!(
            validate_proof_consistency(&proof_with_zk(false, zk_claims()), false),
            Err(VerifierError::UnexpectedBlindFoldProof)
        ));
        assert!(matches!(
            validate_proof_consistency(&proof_with_zk(true, clear_claims()), true),
            Err(VerifierError::UnexpectedOpeningClaims)
        ));
    }

    #[test]
    #[expect(
        clippy::expect_used,
        reason = "test fixture mutation should fail loudly if the serialized shape changes"
    )]
    fn proof_model_rejects_unknown_serialized_fields() {
        fn with_extra_field(
            mut value: serde_json::Value,
            path: &[&str],
            field: &str,
        ) -> serde_json::Value {
            let mut cursor = &mut value;
            for segment in path {
                cursor = cursor
                    .as_object_mut()
                    .expect("proof segment should be an object")
                    .get_mut(*segment)
                    .expect("proof segment should exist");
            }
            let previous = cursor
                .as_object_mut()
                .expect("target proof segment should be an object")
                .insert(field.to_string(), serde_json::Value::Bool(true));
            assert!(previous.is_none());
            value
        }

        let clear = serde_json::to_value(proof_with_zk(false, clear_claims()))
            .expect("clear proof should serialize");
        let zk = serde_json::to_value(proof_with_zk(true, zk_claims()))
            .expect("ZK proof should serialize");

        for (value, path, field) in [
            (clear.clone(), &[][..], "extra_proof"),
            (clear.clone(), &["stages"][..], "extra_stage_proof"),
            (
                clear.clone(),
                &["claims", "Clear"][..],
                "extra_clear_claims",
            ),
            (
                clear.clone(),
                &["claims", "Clear", "stage1"][..],
                "extra_stage1",
            ),
            (
                clear.clone(),
                &["claims", "Clear", "stage1", "outer"][..],
                "extra_outer",
            ),
            (
                clear.clone(),
                &["claims", "Clear", "stage6", "inc_claim_reduction"][..],
                "extra_inc_claim_reduction",
            ),
            (
                clear,
                &[
                    "claims",
                    "Clear",
                    "stage7",
                    "hamming_weight_claim_reduction",
                ][..],
                "extra_hamming_weight",
            ),
            (zk, &["claims", "Zk"][..], "extra_zk_claims"),
        ] {
            assert!(
                serde_json::from_value::<TestProof>(with_extra_field(value, path, field)).is_err(),
                "unknown proof field {field} at path {path:?} must reject"
            );
        }
    }

    #[cfg(feature = "akita")]
    #[test]
    #[expect(
        clippy::expect_used,
        reason = "test fixture mutation should fail loudly if the serialized shape changes"
    )]
    fn proof_model_rejects_unknown_akita_nested_fields() {
        fn with_extra_field(
            mut value: serde_json::Value,
            path: &[&str],
            field: &str,
        ) -> serde_json::Value {
            let mut cursor = &mut value;
            for segment in path {
                cursor = cursor
                    .as_object_mut()
                    .expect("proof segment should be an object")
                    .get_mut(*segment)
                    .expect("proof segment should exist");
            }
            let previous = cursor
                .as_object_mut()
                .expect("target proof segment should be an object")
                .insert(field.to_string(), serde_json::Value::Bool(true));
            assert!(previous.is_none());
            value
        }

        let (_, _, _, proof) = lattice_validity_surface_fixture();
        let value = serde_json::to_value(proof).expect("Akita proof should serialize");

        for (path, field) in [
            (
                &["claims", "Clear", "stage7", "lattice_packed_validity"][..],
                "extra_validity_claim",
            ),
            (&["claims", "Clear", "stage6"][..], "extra_stage6_claim"),
        ] {
            assert!(
                serde_json::from_value::<TestProof>(with_extra_field(value.clone(), path, field))
                    .is_err(),
                "unknown Akita proof field {field} at path {path:?} must reject"
            );
        }
    }

    #[test]
    fn validate_proof_config_checks_akita_payload_layout() {
        let mut config =
            JoltProtocolConfig::for_zk(false).with_pcs_family(crate::config::PcsFamily::Lattice);
        config.lattice.program_mode = crate::config::ProgramMode::Committed;
        config.lattice.increment_mode = crate::config::IncrementCommitmentMode::FusedOneHot;
        config.lattice.packed_witness.layout_digest = Some([7; 32]);
        config.lattice.packed_witness.d_pack = Some(43);
        config.lattice.packed_witness.validity_digest = Some([11; 32]);
        #[cfg(feature = "field-inline")]
        {
            config.lattice.field_inline.enabled = true;
            config.lattice.packed_witness.field_rd_inc_family = true;
        }

        let mut proof = proof_with_zk(false, clear_claims());
        proof.protocol = config;
        proof.commitments = crate::proof::CommitmentPayload::Akita(
            crate::proof::AkitaCommitmentPayload::new(TestCommitment, [8; 32], 43),
        );

        assert!(matches!(
            validate_proof_config(&config, &proof),
            Err(VerifierError::AkitaPayloadLayoutDigestMismatch {
                expected,
                got,
            }) if expected == [7; 32] && got == [8; 32]
        ));
    }

    #[test]
    fn stage8_statement_with_config_uses_supplied_protocol_config() {
        let mut config =
            JoltProtocolConfig::for_zk(false).with_pcs_family(crate::config::PcsFamily::Lattice);
        config.lattice.program_mode = crate::config::ProgramMode::Committed;
        config.lattice.increment_mode = crate::config::IncrementCommitmentMode::FusedOneHot;
        config.lattice.packed_witness.layout_digest = Some([7; 32]);
        config.lattice.packed_witness.d_pack = Some(43);
        config.lattice.packed_witness.validity_digest = Some([11; 32]);
        #[cfg(feature = "field-inline")]
        {
            config.lattice.field_inline.enabled = true;
            config.lattice.packed_witness.field_rd_inc_family = true;
        }

        let preprocessing = test_preprocessing();
        let public_io = JoltDevice {
            memory_layout: preprocessing.program.memory_layout().clone(),
            ..JoltDevice::default()
        };
        let proof = proof_with_zk(false, clear_claims());

        let result = stage8_batch_statement_with_config::<
            Fr,
            TestPcs,
            Pedersen<Bn254G1>,
            jolt_transcript::Blake2bTranscript,
            _,
        >(&preprocessing, &public_io, &proof, None, &config);

        assert!(matches!(
            result,
            Err(VerifierError::ProtocolConfigMismatch { expected, got })
                if expected == config && got == JoltProtocolConfig::for_zk(false)
        ));

        let result_with_transcript = stage8_batch_statement_with_config_and_transcript::<
            Fr,
            TestPcs,
            Pedersen<Bn254G1>,
            jolt_transcript::Blake2bTranscript,
            _,
        >(&preprocessing, &public_io, &proof, None, &config);

        assert!(matches!(
            result_with_transcript,
            Err(VerifierError::ProtocolConfigMismatch { expected, got })
                if expected == config && got == JoltProtocolConfig::for_zk(false)
        ));
    }

    #[cfg(feature = "akita")]
    #[test]
    fn lattice_layout_binding_accepts_derived_layout_config() {
        let preprocessing = committed_test_preprocessing();
        let public_io = public_io_for_preprocessing(&preprocessing);
        let placeholder_config = lattice_config([0; 32], 0);
        let mut proof = proof_with_lattice_payload(&placeholder_config, [0; 32], 0);
        let checked = validate_inputs(&preprocessing, &public_io, &proof, false, false)
            .unwrap_or_else(|error| panic!("inputs should validate: {error}"));
        let layout = expected_lattice_layout(&placeholder_config, &preprocessing, &proof, &checked);
        let config = lattice_config_with_derived_validity(
            layout.digest,
            layout.dimension,
            &checked.precommitted,
        );
        proof.protocol = config;
        proof.commitments = CommitmentPayload::Akita(AkitaCommitmentPayload::new(
            TestCommitment,
            layout.digest,
            layout.dimension,
        ));

        validate_proof_config(&config, &proof)
            .unwrap_or_else(|error| panic!("proof config should validate: {error}"));
        validate_lattice_layout_binding(&config, &preprocessing, &proof, &checked)
            .unwrap_or_else(|error| panic!("layout binding should validate: {error}"));
    }

    #[cfg(feature = "akita")]
    #[test]
    fn lattice_layout_binding_rejects_derived_layout_mismatch() {
        let preprocessing = committed_test_preprocessing();
        let public_io = public_io_for_preprocessing(&preprocessing);
        let config = lattice_config([0; 32], 0);
        let proof = proof_with_lattice_payload(&config, [0; 32], 0);
        let checked = validate_inputs(&preprocessing, &public_io, &proof, false, false)
            .unwrap_or_else(|error| panic!("inputs should validate: {error}"));

        validate_proof_config(&config, &proof)
            .unwrap_or_else(|error| panic!("proof config should validate: {error}"));
        assert!(matches!(
            validate_lattice_layout_binding(&config, &preprocessing, &proof, &checked),
            Err(VerifierError::InvalidProtocolConfig { .. })
        ));
    }

    #[cfg(feature = "akita")]
    #[test]
    fn lattice_layout_binding_rejects_trusted_advice_schedule_without_config() {
        let preprocessing = committed_test_preprocessing_with_advice(64, 0);
        let public_io = public_io_for_preprocessing(&preprocessing);
        let config = lattice_config([0; 32], 0);
        let mut proof = proof_with_lattice_payload(&config, [0; 32], 0);
        proof.ram_K = 16;
        let checked = validate_inputs(&preprocessing, &public_io, &proof, true, false)
            .unwrap_or_else(|error| panic!("inputs should validate: {error}"));

        assert!(matches!(
            validate_lattice_layout_binding(&config, &preprocessing, &proof, &checked),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("trusted advice precommitted schedule")
        ));
    }

    #[cfg(feature = "akita")]
    #[test]
    fn lattice_layout_binding_rejects_untrusted_advice_schedule_without_config() {
        let preprocessing = committed_test_preprocessing_with_advice(0, 64);
        let public_io = public_io_for_preprocessing(&preprocessing);
        let config = lattice_config([0; 32], 0);
        let mut proof = proof_with_lattice_payload(&config, [0; 32], 0);
        proof.ram_K = 16;
        proof.untrusted_advice_commitment = Some(TestCommitment);
        let checked = validate_inputs(&preprocessing, &public_io, &proof, false, false)
            .unwrap_or_else(|error| panic!("inputs should validate: {error}"));

        assert!(matches!(
            validate_lattice_layout_binding(&config, &preprocessing, &proof, &checked),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("untrusted advice precommitted schedule")
        ));
    }

    #[cfg(feature = "akita")]
    #[test]
    fn lattice_validity_surface_accepts_derived_statement_count() {
        let (preprocessing, checked, config, proof) = lattice_validity_surface_fixture();

        validate_lattice_validity_proof_surface(&config, &preprocessing, &proof, &checked)
            .unwrap_or_else(|error| panic!("validity proof surface should validate: {error}"));
    }

    #[cfg(feature = "akita")]
    #[test]
    fn lattice_validity_surface_requires_all_validity_material() {
        let (preprocessing, checked, config, mut proof) = lattice_validity_surface_fixture();
        proof.stages.lattice_packed_validity_sumcheck_proof = None;

        assert!(matches!(
            validate_lattice_validity_proof_surface(&config, &preprocessing, &proof, &checked),
            Err(VerifierError::MissingAkitaPackedValidityProof {
                field: "sumcheck_proof"
            })
        ));

        let (preprocessing, checked, config, mut proof) = lattice_validity_surface_fixture();
        proof.lattice_packed_validity_opening_proof = None;

        assert!(matches!(
            validate_lattice_validity_proof_surface(&config, &preprocessing, &proof, &checked),
            Err(VerifierError::MissingAkitaPackedValidityProof {
                field: "opening_proof"
            })
        ));

        let (preprocessing, checked, config, mut proof) = lattice_validity_surface_fixture();
        let JoltProofClaims::Clear(claims) = &mut proof.claims else {
            panic!("fixture should be a clear proof");
        };
        claims.stage7.lattice_packed_validity = None;

        assert!(matches!(
            validate_lattice_validity_proof_surface(&config, &preprocessing, &proof, &checked),
            Err(VerifierError::MissingAkitaPackedValidityProof {
                field: "opening_claims"
            })
        ));
    }

    #[cfg(feature = "akita")]
    #[test]
    fn lattice_validity_surface_rejects_wrong_opening_claim_count() {
        let (preprocessing, checked, config, mut proof) = lattice_validity_surface_fixture();
        let JoltProofClaims::Clear(claims) = &mut proof.claims else {
            panic!("fixture should be a clear proof");
        };
        let _ = claims
            .stage7
            .lattice_packed_validity
            .as_mut()
            .unwrap_or_else(|| panic!("fixture should include validity claims"))
            .opening_claims
            .pop();

        assert!(matches!(
            validate_lattice_validity_proof_surface(&config, &preprocessing, &proof, &checked),
            Err(VerifierError::AkitaPackedValidityClaimCountMismatch { expected, got })
                if expected == got + 1
        ));
    }

    #[cfg(feature = "akita")]
    #[test]
    fn stage8_statement_helper_requires_lattice_validity_surface() {
        let (preprocessing, _checked, config, mut proof) = lattice_validity_surface_fixture();
        let public_io = public_io_for_preprocessing(&preprocessing);
        proof.stages.lattice_packed_validity_sumcheck_proof = None;

        let result = stage8_batch_statement_with_config_and_transcript::<
            Fr,
            TestPcs,
            Pedersen<Bn254G1>,
            jolt_transcript::Blake2bTranscript,
            _,
        >(&preprocessing, &public_io, &proof, None, &config);

        assert!(matches!(
            result,
            Err(VerifierError::MissingAkitaPackedValidityProof {
                field: "sumcheck_proof"
            })
        ));
    }

    #[cfg(feature = "akita")]
    #[test]
    fn akita_stage8_rejects_missing_committed_program_final_openings() {
        let (preprocessing, checked, proof, stage6, stage7) =
            akita_stage8_statement_fixture(committed_test_preprocessing(), false, |_| {});

        for missing in [
            JoltCommittedPolynomial::BytecodeChunk(0),
            JoltCommittedPolynomial::ProgramImageInit,
        ] {
            let mut stage7 = stage7.clone();
            stage7
                .precommitted_final_openings
                .retain(|opening| opening.polynomial != missing);

            let result = stage8::batch_statement(
                &checked,
                &preprocessing,
                &proof,
                None,
                stage8::Deps::Clear {
                    stage6: &stage6,
                    stage7: &stage7,
                },
            );

            assert!(matches!(
                result,
                Err(VerifierError::MissingOpeningClaim { id })
                    if id == final_opening_id_for_test(missing)
            ));
        }
    }

    #[cfg(feature = "akita")]
    #[test]
    fn akita_stage8_rejects_missing_trusted_advice_final_opening() {
        let (preprocessing, checked, proof, stage6, stage7) = akita_stage8_statement_fixture(
            committed_test_preprocessing_with_advice(64, 0),
            true,
            |config| {
                config.lattice.advice.trusted = true;
            },
        );

        let result = stage8::batch_statement(
            &checked,
            &preprocessing,
            &proof,
            Some(&TestCommitment),
            stage8::Deps::Clear {
                stage6: &stage6,
                stage7: &stage7,
            },
        );

        assert!(matches!(
            result,
            Err(VerifierError::MissingOpeningClaim { id })
                if id == JoltOpeningId::trusted_advice(
                    JoltRelationId::AdviceClaimReduction,
                )
        ));
    }

    #[test]
    fn curve_validity_surface_rejects_unexpected_lattice_material() {
        let preprocessing = test_preprocessing();
        let public_io = public_io_for_preprocessing(&preprocessing);
        let config = JoltProtocolConfig::for_zk(false);
        let mut proof = proof_with_zk(false, clear_claims());
        proof.stages.lattice_packed_validity_sumcheck_proof = Some(sumcheck_proof(false));
        let checked = validate_inputs(&preprocessing, &public_io, &proof, false, false)
            .unwrap_or_else(|error| panic!("inputs should validate: {error}"));

        assert!(matches!(
            validate_lattice_validity_proof_surface(&config, &preprocessing, &proof, &checked),
            Err(VerifierError::UnexpectedAkitaPackedValidityProof {
                field: "sumcheck_proof"
            })
        ));

        let mut proof = proof_with_zk(false, clear_claims());
        proof.lattice_packed_validity_opening_proof = Some(());
        let checked = validate_inputs(&preprocessing, &public_io, &proof, false, false)
            .unwrap_or_else(|error| panic!("inputs should validate: {error}"));

        assert!(matches!(
            validate_lattice_validity_proof_surface(&config, &preprocessing, &proof, &checked),
            Err(VerifierError::UnexpectedAkitaPackedValidityProof {
                field: "opening_proof"
            })
        ));

        let mut proof = proof_with_zk(false, clear_claims());
        let JoltProofClaims::Clear(claims) = &mut proof.claims else {
            panic!("fixture should be a clear proof");
        };
        claims.stage7.lattice_packed_validity =
            Some(stage7::inputs::LatticePackedValidityOutputClaims {
                opening_claims: Vec::new(),
            });
        let checked = validate_inputs(&preprocessing, &public_io, &proof, false, false)
            .unwrap_or_else(|error| panic!("inputs should validate: {error}"));

        assert!(matches!(
            validate_lattice_validity_proof_surface(&config, &preprocessing, &proof, &checked),
            Err(VerifierError::UnexpectedAkitaPackedValidityProof {
                field: "opening_claims"
            })
        ));
    }

    #[cfg(feature = "akita")]
    #[test]
    fn akita_stage8_batch_statement_uses_precommitted_bytecode_source_components() {
        let preprocessing = committed_test_preprocessing();
        let public_io = public_io_for_preprocessing(&preprocessing);
        let placeholder_config = lattice_config([0; 32], 0);
        let mut proof = proof_with_lattice_payload(&placeholder_config, [0; 32], 0);
        let checked = validate_inputs(&preprocessing, &public_io, &proof, false, false)
            .unwrap_or_else(|error| panic!("inputs should validate: {error}"));
        let log_t = checked.trace_length.ilog2() as usize;
        let formula_dimensions = JoltFormulaDimensions::try_from(proof.one_hot_config.dimensions(
            log_t,
            2 * RISCV_XLEN,
            preprocessing.program.bytecode_len(),
            checked.ram_K,
        ))
        .unwrap_or_else(|error| panic!("formula dimensions should derive: {error}"));
        let layout = expected_lattice_layout(&placeholder_config, &preprocessing, &proof, &checked);
        let config = lattice_config_with_derived_validity(
            layout.digest,
            layout.dimension,
            &checked.precommitted,
        );
        proof.protocol = config;
        proof.commitments = CommitmentPayload::Akita(AkitaCommitmentPayload::new(
            TestCommitment,
            layout.digest,
            layout.dimension,
        ));
        let bytecode_layout = checked
            .precommitted
            .bytecode
            .as_ref()
            .unwrap_or_else(|| panic!("committed bytecode schedule should exist"));
        let bytecode_address_point = field_zeros(bytecode_layout.log_bytecode_chunk_size() + 1);
        let stage6 = akita_snapshot_stage6_output(
            formula_dimensions.ra_layout.bytecode(),
            log_t,
            proof.one_hot_config.committed_chunk_bits(),
            bytecode_layout.chunk_count(),
            bytecode_address_point,
        );
        let stage7 = akita_snapshot_stage7_output(
            formula_dimensions.ra_layout,
            log_t,
            proof.one_hot_config.committed_chunk_bits(),
            &checked.precommitted,
        );

        let result = stage8::batch_statement(
            &checked,
            &preprocessing,
            &proof,
            None,
            stage8::Deps::Clear {
                stage6: &stage6,
                stage7: &stage7,
            },
        );

        let stage8::Stage8BatchStatement::Clear(batch) =
            result.unwrap_or_else(|error| panic!("Stage 8 statement should build: {error}"))
        else {
            panic!("fixture is clear mode");
        };
        let bytecode_chunk_count = checked
            .precommitted
            .bytecode
            .as_ref()
            .unwrap_or_else(|| panic!("committed bytecode schedule should exist"))
            .chunk_count();
        assert_eq!(
            batch.precommitted_statements.len(),
            bytecode_chunk_count + 1
        );
        assert!(batch.precommitted_statements.iter().all(|statement| {
            statement.claims.len() == 1
                && statement
                    .claims
                    .iter()
                    .all(|claim| matches!(claim.view, jolt_openings::PhysicalView::Direct))
        }));
        let precommitted_ids = batch
            .precommitted_statements
            .iter()
            .map(|statement| statement.claims[0].id)
            .collect::<Vec<_>>();
        let count_id = |id| precommitted_ids.iter().filter(|&&got| got == id).count();
        for chunk in 0..bytecode_chunk_count {
            assert_eq!(
                count_id(stage8::Stage8OpeningId::from(final_opening_id_for_test(
                    JoltCommittedPolynomial::BytecodeChunk(chunk),
                ))),
                1
            );
        }
        assert_eq!(
            count_id(stage8::Stage8OpeningId::from(final_opening_id_for_test(
                JoltCommittedPolynomial::ProgramImageInit,
            ))),
            1
        );
        let packed_ids = batch
            .statement
            .claims
            .iter()
            .map(|claim| claim.id)
            .collect::<Vec<_>>();
        let packed_count_id = |id| packed_ids.iter().filter(|&&got| got == id).count();
        assert_eq!(
            packed_count_id(stage8::Stage8OpeningId::from(final_opening_id_for_test(
                JoltCommittedPolynomial::RamInc,
            ))),
            0
        );
        assert_eq!(
            packed_count_id(stage8::Stage8OpeningId::from(final_opening_id_for_test(
                JoltCommittedPolynomial::RdInc,
            ))),
            0
        );
        let unsigned_inc_chunk_count =
            jolt_claims::protocols::jolt::unsigned_inc_lower_chunk_count(
                proof.one_hot_config.committed_chunk_bits(),
            )
            .unwrap_or_else(|| panic!("test unsigned increment chunks should derive"));
        for index in 0..unsigned_inc_chunk_count {
            assert_eq!(
                packed_count_id(stage8::Stage8OpeningId::from(
                    jolt_claims::protocols::jolt::unsigned_inc_chunk_opening(index),
                )),
                1
            );
        }
        assert_eq!(
            packed_count_id(stage8::Stage8OpeningId::from(
                jolt_claims::protocols::jolt::unsigned_inc_msb_opening(),
            )),
            1
        );
    }

    #[cfg(feature = "akita")]
    #[test]
    fn akita_stage8_rejects_missing_unsigned_increment_final_sources() {
        let (preprocessing, checked, proof, stage6, stage7) =
            akita_stage8_statement_fixture(committed_test_preprocessing(), false, |_| {});

        let assert_missing_unsigned_source =
            |stage6: stage6::Stage6ClearOutput<Fr>, stage7: stage7::Stage7ClearOutput<Fr>| {
                let result = stage8::batch_statement(
                    &checked,
                    &preprocessing,
                    &proof,
                    None,
                    stage8::Deps::Clear {
                        stage6: &stage6,
                        stage7: &stage7,
                    },
                );
                assert!(matches!(
                    result,
                    Err(VerifierError::MissingOpeningClaim { id })
                        if id == jolt_claims::protocols::jolt::unsigned_inc_chunk_opening(0)
                ));
            };

        let mut missing_msb_sumcheck = stage6.clone();
        missing_msb_sumcheck.batch.unsigned_inc_msb_booleanity = None;
        assert_missing_unsigned_source(missing_msb_sumcheck, stage7.clone());

        let mut missing_msb_claim = stage6.clone();
        missing_msb_claim.output_claims.unsigned_inc_claim_reduction = None;
        assert_missing_unsigned_source(missing_msb_claim, stage7.clone());

        let mut missing_chunk_sumcheck = stage7.clone();
        missing_chunk_sumcheck
            .batch
            .unsigned_inc_chunk_reconstruction = None;
        assert_missing_unsigned_source(stage6.clone(), missing_chunk_sumcheck);

        let mut missing_chunk_claims = stage7;
        missing_chunk_claims
            .output_claims
            .unsigned_inc_chunk_reconstruction = None;
        assert_missing_unsigned_source(stage6, missing_chunk_claims);
    }

    #[cfg(feature = "akita")]
    #[test]
    fn akita_stage8_rejects_wrong_unsigned_increment_final_chunk_count() {
        let (preprocessing, checked, proof, stage6, mut stage7) =
            akita_stage8_statement_fixture(committed_test_preprocessing(), false, |_| {});
        let _ = stage7
            .output_claims
            .unsigned_inc_chunk_reconstruction
            .as_mut()
            .unwrap_or_else(|| panic!("test fixture should include unsigned increment chunks"))
            .chunks
            .pop();

        let result = stage8::batch_statement(
            &checked,
            &preprocessing,
            &proof,
            None,
            stage8::Deps::Clear {
                stage6: &stage6,
                stage7: &stage7,
            },
        );

        assert!(matches!(
            result,
            Err(VerifierError::FinalOpeningBatchFailed { reason })
                if reason.contains("unsigned increment final chunk opening count mismatch")
        ));
    }

    #[cfg(feature = "akita")]
    #[test]
    fn akita_stage8_verify_rejects_missing_precommitted_bytecode_source_proofs() {
        let (preprocessing, checked, proof, stage6, stage7) =
            akita_stage8_statement_fixture(committed_test_preprocessing(), false, |_| {});
        let mut transcript = jolt_transcript::Blake2bTranscript::new(b"akita-stage8-test");

        let result = stage8::verify_clear::<
            Fr,
            TestPcs,
            Pedersen<Bn254G1>,
            jolt_transcript::Blake2bTranscript,
            _,
        >(
            &checked,
            &preprocessing,
            &proof,
            None,
            &mut transcript,
            stage8::Deps::Clear {
                stage6: &stage6,
                stage7: &stage7,
            },
        );

        assert!(matches!(
            result,
            Err(VerifierError::FinalOpeningVerificationFailed { reason })
                if reason.contains("precommitted opening proofs")
        ));
    }

    #[cfg(feature = "akita")]
    #[test]
    fn akita_stage8_verify_rejects_missing_committed_program_precommitted_proofs() {
        let (preprocessing, checked, mut proof, stage6, stage7) =
            akita_stage8_statement_fixture(committed_test_preprocessing(), false, |_| {});
        let statement = stage8::batch_statement(
            &checked,
            &preprocessing,
            &proof,
            None,
            stage8::Deps::Clear {
                stage6: &stage6,
                stage7: &stage7,
            },
        )
        .unwrap_or_else(|error| panic!("Stage 8 statement should build: {error}"));
        let stage8::Stage8BatchStatement::Clear(batch) = statement else {
            panic!("fixture is clear mode");
        };
        let bytecode_chunk_count = checked
            .precommitted
            .bytecode
            .as_ref()
            .unwrap_or_else(|| panic!("committed bytecode schedule should exist"))
            .chunk_count();
        let expected_precommitted_count = bytecode_chunk_count + 1;
        assert_eq!(
            batch.precommitted_statements.len(),
            expected_precommitted_count
        );
        proof.lattice_precommitted_opening_proofs =
            vec![(); expected_precommitted_count.saturating_sub(1)];
        let mut transcript = jolt_transcript::Blake2bTranscript::new(b"akita-stage8-test");

        let result = stage8::verify_clear::<
            Fr,
            TestPcs,
            Pedersen<Bn254G1>,
            jolt_transcript::Blake2bTranscript,
            _,
        >(
            &checked,
            &preprocessing,
            &proof,
            None,
            &mut transcript,
            stage8::Deps::Clear {
                stage6: &stage6,
                stage7: &stage7,
            },
        );

        assert!(matches!(
            result,
            Err(VerifierError::FinalOpeningVerificationFailed { reason })
                if reason.contains(&format!(
                    "expected {} precommitted opening proofs, got {}",
                    expected_precommitted_count,
                    expected_precommitted_count.saturating_sub(1)
                ))
        ));
    }

    #[cfg(not(feature = "akita"))]
    #[test]
    fn lattice_layout_binding_requires_akita_feature() {
        let preprocessing = committed_test_preprocessing();
        let public_io = public_io_for_preprocessing(&preprocessing);
        let config = lattice_config([0; 32], 0);
        let proof = proof_with_lattice_payload(&config, [0; 32], 0);
        let checked = validate_inputs(&preprocessing, &public_io, &proof, false, false)
            .unwrap_or_else(|error| panic!("inputs should validate: {error}"));

        validate_proof_config(&config, &proof)
            .unwrap_or_else(|error| panic!("proof config should validate: {error}"));
        assert!(matches!(
            validate_lattice_layout_binding(&config, &preprocessing, &proof, &checked),
            Err(VerifierError::InvalidProtocolConfig { .. })
        ));
    }

    #[test]
    fn absorb_commitments_accepts_akita_payload_and_binds_layout_metadata() {
        let preprocessing = committed_test_preprocessing();
        let layout_digest = [9; 32];
        let d_pack = 44;
        let validity_digest = [11; 32];
        let proof = proof_with_lattice_payload(
            &lattice_config(layout_digest, d_pack),
            layout_digest,
            d_pack,
        );
        let mut transcript = RecordingTranscript::new(b"test");

        absorb_commitments(&preprocessing, &proof, None, &mut transcript)
            .unwrap_or_else(|error| panic!("Akita commitment absorption should succeed: {error}"));

        assert!(contains_subslice(
            &transcript.bytes,
            b"akita_protocol_header"
        ));
        assert!(contains_subslice(&transcript.bytes, b"akita_program_mode"));
        assert!(contains_subslice(
            &transcript.bytes,
            b"akita_increment_mode"
        ));
        assert!(contains_subslice(
            &transcript.bytes,
            b"akita_advice_trusted"
        ));
        assert!(contains_subslice(&transcript.bytes, &layout_digest));
        assert!(contains_subslice(&transcript.bytes, b"akita_d_pack"));
        assert!(contains_subslice(
            &transcript.bytes,
            &u64_word_bytes(d_pack as u64)
        ));
        assert!(contains_subslice(&transcript.bytes, &validity_digest));
    }

    #[test]
    fn absorb_commitments_binds_akita_protocol_header_flags() {
        let preprocessing = committed_test_preprocessing();
        let layout_digest = [9; 32];
        let d_pack = 44;
        let config = lattice_config(layout_digest, d_pack);
        let proof = proof_with_lattice_payload(&config, layout_digest, d_pack);
        let mut transcript = RecordingTranscript::new(b"test");
        absorb_commitments(&preprocessing, &proof, None, &mut transcript)
            .unwrap_or_else(|error| panic!("Akita commitment absorption should succeed: {error}"));

        let mut changed_proof = proof;
        changed_proof.protocol.lattice.advice.trusted = true;
        let mut changed_transcript = RecordingTranscript::new(b"test");
        absorb_commitments(
            &preprocessing,
            &changed_proof,
            None,
            &mut changed_transcript,
        )
        .unwrap_or_else(|error| panic!("Akita commitment absorption should succeed: {error}"));

        assert_ne!(transcript.bytes, changed_transcript.bytes);
    }

    #[test]
    fn validate_inputs_normalizes_public_output() {
        let preprocessing = test_preprocessing();
        let mut public_io = JoltDevice {
            memory_layout: preprocessing.program.memory_layout().clone(),
            inputs: vec![1, 2],
            outputs: vec![3, 0, 0],
            ..JoltDevice::default()
        };
        let proof = proof_with_zk(false, clear_claims());

        let checked = validate_inputs(&preprocessing, &public_io, &proof, false, false);
        assert!(checked.is_ok());
        let Ok(checked) = checked else {
            return;
        };

        assert_eq!(checked.public_io.inputs, vec![1, 2]);
        assert_eq!(checked.public_io.outputs, vec![3]);
        assert_eq!(checked.trace_length, proof.trace_length);
        assert_eq!(checked.ram_K, proof.ram_K);

        public_io.outputs = vec![0, 0];
        let checked = validate_inputs(&preprocessing, &public_io, &proof, false, false);
        assert!(checked.is_ok());
        let Ok(checked) = checked else {
            return;
        };
        assert!(checked.public_io.outputs.is_empty());
    }

    #[test]
    fn validate_inputs_rejects_public_io_layout_mismatch() {
        let preprocessing = test_preprocessing();
        let public_io = JoltDevice::default();
        let proof = proof_with_zk(false, clear_claims());

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &proof, false, false),
            Err(VerifierError::MemoryLayoutMismatch)
        ));
    }

    #[test]
    fn validate_inputs_rejects_ram_domain_below_layout_minimum() {
        let preprocessing = test_preprocessing();
        let public_io = JoltDevice {
            memory_layout: preprocessing.program.memory_layout().clone(),
            ..JoltDevice::default()
        };
        let mut proof = proof_with_zk(false, clear_claims());
        proof.ram_K = 2;

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &proof, false, false),
            Err(VerifierError::InvalidRamK { got: 2, min: 4, .. })
        ));
    }

    #[test]
    fn validate_inputs_rejects_ram_domain_above_layout_maximum() {
        let preprocessing = test_preprocessing();
        let public_io = JoltDevice {
            memory_layout: preprocessing.program.memory_layout().clone(),
            ..JoltDevice::default()
        };
        let mut proof = proof_with_zk(false, clear_claims());
        proof.ram_K = 1 << 20;

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &proof, false, false),
            Err(VerifierError::InvalidRamK {
                got,
                min: 4,
                max,
            }) if got == 1 << 20 && max < got
        ));
    }

    #[test]
    fn validate_inputs_rejects_missing_zk_vector_commitment_setup() {
        let preprocessing = test_preprocessing();
        let public_io = JoltDevice {
            memory_layout: preprocessing.program.memory_layout().clone(),
            ..JoltDevice::default()
        };
        let proof = proof_with_zk(true, zk_claims());

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &proof, false, true),
            Err(VerifierError::MissingVectorCommitmentSetup)
        ));
    }

    #[test]
    fn validate_inputs_rejects_small_zk_vector_commitment_setup() {
        let mut preprocessing = test_preprocessing();
        preprocessing.vc_setup = Some(PedersenSetup::new(
            vec![Bn254G1::default()],
            Bn254G1::default(),
        ));
        let public_io = JoltDevice {
            memory_layout: preprocessing.program.memory_layout().clone(),
            ..JoltDevice::default()
        };
        let proof = proof_with_zk(true, zk_claims());

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &proof, false, true),
            Err(VerifierError::InvalidVectorCommitmentCapacity { got: 1, .. })
        ));
    }

    fn proof_with_zk(is_zk: bool, claims: TestClaims) -> TestProof {
        JoltProof::new(
            crate::proof::JoltCommitments::new(
                TestCommitment,
                TestCommitment,
                crate::proof::JoltRaCommitments::new(
                    Vec::<TestCommitment>::new(),
                    Vec::<TestCommitment>::new(),
                    Vec::<TestCommitment>::new(),
                ),
                #[cfg(feature = "field-inline")]
                crate::proof::FieldInlineCommitments::new(
                    crate::proof::FieldRegistersCommitments::new(TestCommitment),
                ),
            ),
            stage_proofs(is_zk),
            (),
            None,
            claims,
            1,
            4,
            JoltReadWriteConfig {
                ram_rw_phase1_num_rounds: 0,
                ram_rw_phase2_num_rounds: 0,
                registers_rw_phase1_num_rounds: 0,
                registers_rw_phase2_num_rounds: 0,
            },
            JoltOneHotConfig {
                log_k_chunk: 0,
                lookups_ra_virtual_log_k_chunk: 0,
            },
            crate::proof::TracePolynomialOrder::CycleMajor,
        )
    }

    fn clear_claims() -> TestClaims {
        let zero = Fr::zero();

        JoltProofClaims::Clear(ClearProofClaims {
            stage1: stage1::inputs::Stage1Claims {
                uniskip_output_claim: zero,
                outer: empty_spartan_outer_claims(),
                #[cfg(feature = "field-inline")]
                field_inline: stage1::inputs::FieldInlineStage1Claims::zero(),
            },
            stage2: stage2::inputs::Stage2Claims {
                product_uniskip_output_claim: zero,
                batch_outputs: stage2::inputs::Stage2BatchOutputOpeningClaims {
                    ram_read_write: stage2::inputs::RamReadWriteOutputOpeningClaims {
                        val: zero,
                        ra: zero,
                        inc: zero,
                    },
                    product_remainder: stage2::inputs::ProductRemainderOutputOpeningClaims {
                        left_instruction_input: zero,
                        right_instruction_input: zero,
                        jump_flag: zero,
                        write_lookup_output_to_rd: zero,
                        lookup_output: zero,
                        branch_flag: zero,
                        next_is_noop: zero,
                        virtual_instruction: zero,
                    },
                    #[cfg(feature = "field-inline")]
                    field_inline: stage2::inputs::FieldInlineStage2OutputOpeningClaims {
                        product: stage2::inputs::FieldInlineProductOutputOpeningClaims {
                            field_rs1_value: zero,
                            field_rs2_value: zero,
                            field_rd_value: zero,
                        },
                    },
                    instruction_claim_reduction:
                        stage2::inputs::InstructionClaimReductionOutputOpeningClaims {
                            lookup_output: None,
                            left_lookup_operand: zero,
                            right_lookup_operand: zero,
                            left_instruction_input: None,
                            right_instruction_input: None,
                        },
                    ram_raf_evaluation: zero,
                    ram_output_check: zero,
                },
            },
            stage3: stage3::inputs::Stage3Claims {
                shift: stage3::inputs::SpartanShiftOutputOpeningClaims {
                    unexpanded_pc: zero,
                    pc: zero,
                    is_virtual: zero,
                    is_first_in_sequence: zero,
                    is_noop: zero,
                },
                instruction_input: stage3::inputs::InstructionInputOutputOpeningClaims {
                    left_operand_is_rs1: zero,
                    rs1_value: zero,
                    left_operand_is_pc: zero,
                    unexpanded_pc: zero,
                    right_operand_is_rs2: zero,
                    rs2_value: zero,
                    right_operand_is_imm: zero,
                    imm: zero,
                },
                registers_claim_reduction:
                    stage3::inputs::RegistersClaimReductionOutputOpeningClaims {
                        rd_write_value: zero,
                        rs1_value: zero,
                        rs2_value: zero,
                    },
            },
            stage4: stage4::inputs::Stage4Claims {
                advice: stage4::inputs::RamValCheckAdviceOpeningClaims {
                    untrusted: None,
                    trusted: None,
                },
                program_image_contribution: None,
                registers_read_write: stage4::inputs::RegistersReadWriteOutputOpeningClaims {
                    registers_val: zero,
                    rs1_ra: zero,
                    rs2_ra: zero,
                    rd_wa: zero,
                    rd_inc: zero,
                },
                #[cfg(feature = "field-inline")]
                field_inline: stage4::inputs::FieldInlineStage4Claims {
                    field_registers_read_write:
                        stage4::inputs::FieldRegistersReadWriteOutputOpeningClaims {
                            field_registers_val: zero,
                            field_rs1_ra: zero,
                            field_rs2_ra: zero,
                            field_rd_wa: zero,
                            field_rd_inc: zero,
                        },
                },
                ram_val_check: stage4::inputs::RamValCheckOutputOpeningClaims {
                    ram_ra: zero,
                    ram_inc: zero,
                },
            },
            stage5: stage5::inputs::Stage5Claims {
                instruction_read_raf: stage5::inputs::InstructionReadRafOutputOpeningClaims {
                    lookup_table_flags: Vec::new(),
                    instruction_ra: Vec::new(),
                    instruction_raf_flag: zero,
                },
                ram_ra_claim_reduction: Some(
                    stage5::inputs::RamRaClaimReductionOutputOpeningClaims { ram_ra: zero },
                ),
                registers_val_evaluation:
                    stage5::inputs::RegistersValEvaluationOutputOpeningClaims {
                        rd_inc: zero,
                        rd_wa: zero,
                    },
                #[cfg(feature = "field-inline")]
                field_inline: stage5::inputs::FieldInlineStage5Claims {
                    field_registers_val_evaluation:
                        stage5::inputs::FieldRegistersValEvaluationOutputOpeningClaims {
                            field_rd_inc: zero,
                            field_rd_wa: zero,
                        },
                },
            },
            stage5_increment: None,
            stage6: stage6::inputs::Stage6Claims {
                address_phase: stage6::inputs::Stage6AddressPhaseClaims {
                    bytecode_read_raf: zero,
                    booleanity: zero,
                    bytecode_val_stages: None,
                },
                bytecode_read_raf: stage6::inputs::BytecodeReadRafOutputOpeningClaims {
                    bytecode_ra: Vec::new(),
                },
                booleanity: stage6::inputs::BooleanityOutputOpeningClaims {
                    instruction_ra: Vec::new(),
                    bytecode_ra: Vec::new(),
                    ram_ra: Vec::new(),
                    unsigned_inc_chunks: Vec::new(),
                },
                ram_hamming_booleanity: stage6::inputs::RamHammingBooleanityOutputOpeningClaims {
                    ram_hamming_weight: zero,
                },
                ram_ra_virtualization: stage6::inputs::RamRaVirtualizationOutputOpeningClaims {
                    ram_ra: Vec::new(),
                },
                instruction_ra_virtualization:
                    stage6::inputs::InstructionRaVirtualizationOutputOpeningClaims {
                        committed_instruction_ra: Vec::new(),
                    },
                inc_claim_reduction: Some(stage6::inputs::IncClaimReductionOutputOpeningClaims {
                    ram_inc: zero,
                    rd_inc: zero,
                }),
                unsigned_inc_claim_reduction: None,
                #[cfg(feature = "field-inline")]
                field_inline: stage6::inputs::FieldInlineStage6Claims {
                    field_registers_inc_claim_reduction:
                        stage6::inputs::FieldRegistersIncClaimReductionOutputOpeningClaims {
                            field_rd_inc: zero,
                        },
                },
                advice_cycle_phase: stage6::inputs::Stage6AdviceCyclePhaseClaims {
                    trusted: None,
                    untrusted: None,
                },
                bytecode_claim_reduction: None,
                program_image_claim_reduction: None,
            },
            stage7: stage7::inputs::Stage7Claims {
                hamming_weight_claim_reduction:
                    stage7::inputs::HammingWeightClaimReductionOutputOpeningClaims {
                        instruction_ra: Vec::new(),
                        bytecode_ra: Vec::new(),
                        ram_ra: Vec::new(),
                    },
                advice_address_phase: stage7::inputs::Stage7AdviceAddressPhaseClaims {
                    trusted: None,
                    untrusted: None,
                },
                bytecode_address_phase: None,
                program_image_address_phase: None,
                unsigned_inc_chunk_reconstruction: None,
                lattice_packed_validity: None,
            },
        })
    }

    fn empty_spartan_outer_claims() -> stage1::inputs::SpartanOuterClaims<Fr> {
        let zero = Fr::zero();

        stage1::inputs::SpartanOuterClaims {
            left_instruction_input: zero,
            right_instruction_input: zero,
            product: zero,
            should_branch: zero,
            pc: zero,
            unexpanded_pc: zero,
            imm: zero,
            ram_address: zero,
            rs1_value: zero,
            rs2_value: zero,
            rd_write_value: zero,
            ram_read_value: zero,
            ram_write_value: zero,
            left_lookup_operand: zero,
            right_lookup_operand: zero,
            next_unexpanded_pc: zero,
            next_pc: zero,
            next_is_virtual: zero,
            next_is_first_in_sequence: zero,
            lookup_output: zero,
            should_jump: zero,
            flags: stage1::inputs::SpartanOuterFlagClaims {
                add_operands: zero,
                subtract_operands: zero,
                multiply_operands: zero,
                load: zero,
                store: zero,
                jump: zero,
                write_lookup_output_to_rd: zero,
                virtual_instruction: zero,
                assert: zero,
                do_not_update_unexpanded_pc: zero,
                advice: zero,
                is_compressed: zero,
                is_first_in_sequence: zero,
                is_last_in_sequence: zero,
            },
        }
    }

    fn zk_claims() -> TestClaims {
        JoltProofClaims::Zk {
            blindfold_proof: empty_blindfold_proof(),
        }
    }

    fn empty_blindfold_proof() -> jolt_blindfold::BlindFoldProof<Fr, Bn254G1> {
        jolt_blindfold::BlindFoldProof {
            auxiliary_row_commitments: Vec::new(),
            random_round_commitments: Vec::new(),
            random_output_claim_row_commitments: Vec::new(),
            random_auxiliary_row_commitments: Vec::new(),
            random_error_row_commitments: Vec::new(),
            random_eval_commitments: Vec::new(),
            random_u: Fr::zero(),
            cross_term_error_row_commitments: Vec::new(),
            outer_sumcheck: CompressedSumcheckProof::default(),
            az_rx: Fr::zero(),
            bz_rx: Fr::zero(),
            cz_rx: Fr::zero(),
            inner_sumcheck: CompressedSumcheckProof::default(),
            witness_opening: VectorCommitmentOpening {
                combined_vector: Vec::new(),
                combined_blinding: Fr::zero(),
            },
            error_opening: VectorCommitmentOpening {
                combined_vector: Vec::new(),
                combined_blinding: Fr::zero(),
            },
            folded_eval_outputs: Vec::new(),
            folded_eval_blindings: Vec::new(),
            folded_eval_output_openings: Vec::new(),
            folded_eval_blinding_openings: Vec::new(),
        }
    }

    fn stage_proofs(is_zk: bool) -> JoltStageProofs<Fr, Pedersen<Bn254G1>> {
        JoltStageProofs {
            stage1_uni_skip_first_round_proof: uniskip_proof(is_zk),
            stage1_sumcheck_proof: sumcheck_proof(is_zk),
            stage2_uni_skip_first_round_proof: uniskip_proof(is_zk),
            stage2_sumcheck_proof: sumcheck_proof(is_zk),
            stage3_sumcheck_proof: sumcheck_proof(is_zk),
            stage4_sumcheck_proof: sumcheck_proof(is_zk),
            stage5_sumcheck_proof: sumcheck_proof(is_zk),
            stage5_increment_sumcheck_proof: None,
            stage6a_sumcheck_proof: sumcheck_proof(is_zk),
            stage6b_sumcheck_proof: sumcheck_proof(is_zk),
            stage7_sumcheck_proof: sumcheck_proof(is_zk),
            lattice_packed_validity_sumcheck_proof: None,
        }
    }

    fn uniskip_proof(is_zk: bool) -> SumcheckProof<Fr, Bn254G1> {
        if is_zk {
            SumcheckProof::Committed(CommittedSumcheckProof::default())
        } else {
            SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof::default()))
        }
    }

    fn sumcheck_proof(is_zk: bool) -> SumcheckProof<Fr, Bn254G1> {
        if is_zk {
            SumcheckProof::Committed(CommittedSumcheckProof::default())
        } else {
            SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof::default()))
        }
    }

    fn test_preprocessing() -> JoltVerifierPreprocessing<TestPcs, Pedersen<Bn254G1>> {
        let memory_layout = common::jolt_device::MemoryLayout::new(&MemoryConfig {
            program_size: Some(1024),
            max_trusted_advice_size: 0,
            max_untrusted_advice_size: 0,
            max_input_size: 8,
            max_output_size: 8,
            stack_size: 8,
            heap_size: 8,
        });
        JoltVerifierPreprocessing::new(
            crate::preprocessing::ProgramPreprocessing::Full(JoltProgramPreprocessing {
                bytecode: BytecodePreprocessing::default(),
                ram: RAMPreprocessing::default(),
                memory_layout,
                max_padded_trace_length: 16,
            }),
            [7; 32],
            (),
            None,
        )
    }

    fn committed_test_preprocessing() -> JoltVerifierPreprocessing<TestPcs, Pedersen<Bn254G1>> {
        committed_test_preprocessing_with_advice(0, 0)
    }

    fn committed_test_preprocessing_with_advice(
        max_trusted_advice_size: u64,
        max_untrusted_advice_size: u64,
    ) -> JoltVerifierPreprocessing<TestPcs, Pedersen<Bn254G1>> {
        let memory_layout = common::jolt_device::MemoryLayout::new(&MemoryConfig {
            program_size: Some(1024),
            max_trusted_advice_size,
            max_untrusted_advice_size,
            max_input_size: 8,
            max_output_size: 8,
            stack_size: 8,
            heap_size: 8,
        });
        let bytecode_address = memory_layout.get_lowest_address();
        JoltVerifierPreprocessing::new(
            crate::preprocessing::ProgramPreprocessing::Committed(
                crate::preprocessing::CommittedProgramPreprocessing {
                    meta: ProgramMetadata {
                        entry_address: bytecode_address,
                        min_bytecode_address: bytecode_address,
                        entry_bytecode_index: 0,
                        program_image_len_words: 4,
                        bytecode_len: 16,
                    },
                    memory_layout,
                    max_padded_trace_length: 16,
                    bytecode_chunk_commitments: vec![TestCommitment, TestCommitment],
                    program_image_commitment: TestCommitment,
                },
            ),
            [7; 32],
            (),
            None,
        )
    }

    fn public_io_for_preprocessing<PCS, VC>(
        preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    ) -> JoltDevice
    where
        PCS: CommitmentScheme,
        VC: VectorCommitment<Field = PCS::Field>,
    {
        JoltDevice {
            memory_layout: preprocessing.program.memory_layout().clone(),
            ..JoltDevice::default()
        }
    }

    fn lattice_config(layout_digest: [u8; 32], d_pack: usize) -> JoltProtocolConfig {
        let mut config =
            JoltProtocolConfig::for_zk(false).with_pcs_family(crate::config::PcsFamily::Lattice);
        config.lattice.program_mode = crate::config::ProgramMode::Committed;
        config.lattice.increment_mode = crate::config::IncrementCommitmentMode::FusedOneHot;
        config.lattice.packed_witness.layout_digest = Some(layout_digest);
        config.lattice.packed_witness.d_pack = Some(d_pack);
        config.lattice.packed_witness.validity_digest = Some([11; 32]);
        #[cfg(feature = "field-inline")]
        {
            config.lattice.field_inline.enabled = true;
            config.lattice.packed_witness.field_rd_inc_family = true;
        }
        config
    }

    #[cfg(feature = "akita")]
    fn lattice_config_with_derived_validity(
        layout_digest: [u8; 32],
        d_pack: usize,
        precommitted: &crate::stages::PrecommittedSchedule,
    ) -> JoltProtocolConfig {
        let mut config = lattice_config(layout_digest, d_pack);
        let requirements =
            stage8::derive_akita_packed_validity_requirements(&config, 8, precommitted)
                .unwrap_or_else(|error| panic!("validity requirements should derive: {error}"));
        config.lattice.packed_witness.validity_digest =
            Some(lattice_packed_validity_digest(&requirements));
        config
    }

    fn proof_with_lattice_payload(
        config: &JoltProtocolConfig,
        layout_digest: [u8; 32],
        d_pack: usize,
    ) -> TestProof {
        let mut proof = proof_with_zk(false, clear_claims());
        proof.protocol = *config;
        proof.trace_length = 4;
        proof.ram_K = 4;
        proof.one_hot_config = JoltOneHotConfig {
            log_k_chunk: 8,
            lookups_ra_virtual_log_k_chunk: 8,
        };
        proof.commitments = CommitmentPayload::Akita(AkitaCommitmentPayload::new(
            TestCommitment,
            layout_digest,
            d_pack,
        ));
        #[cfg(feature = "akita")]
        attach_lattice_validity_surface(&mut proof, config);
        proof
    }

    #[cfg(feature = "akita")]
    fn attach_lattice_validity_surface(proof: &mut TestProof, config: &JoltProtocolConfig) {
        let precommitted = test_lattice_precommitted_schedule();
        let log_t = proof.trace_length.ilog2() as usize;
        let formula_dimensions = JoltFormulaDimensions::try_from(proof.one_hot_config.dimensions(
            log_t,
            2 * RISCV_XLEN,
            16,
            proof.ram_K,
        ))
        .unwrap_or_else(|error| panic!("formula dimensions should derive: {error}"));
        let layout = stage8::derive_akita_packed_witness_layout(
            config,
            log_t,
            proof.one_hot_config.committed_chunk_bits(),
            formula_dimensions.ra_layout,
            &precommitted,
        )
        .unwrap_or_else(|error| panic!("packed witness layout should derive: {error}"));
        let requirements = stage8::derive_akita_packed_validity_requirements(
            config,
            proof.one_hot_config.committed_chunk_bits(),
            &precommitted,
        )
        .unwrap_or_else(|error| panic!("validity requirements should derive: {error}"));
        let statements = stage8::derive_akita_packed_validity_statements(&layout, &requirements)
            .unwrap_or_else(|error| panic!("validity statements should derive: {error}"));

        proof.stages.lattice_packed_validity_sumcheck_proof = Some(sumcheck_proof(false));
        proof.lattice_packed_validity_opening_proof = Some(());
        let JoltProofClaims::Clear(claims) = &mut proof.claims else {
            panic!("test lattice proof should be transparent");
        };
        claims.stage7.lattice_packed_validity =
            Some(stage7::inputs::LatticePackedValidityOutputClaims {
                opening_claims: vec![
                    Fr::zero();
                    stage8::lattice_packed_validity_opening_count(&statements)
                ],
            });
    }

    #[cfg(feature = "akita")]
    fn test_lattice_precommitted_schedule() -> PrecommittedSchedule {
        PrecommittedSchedule::new(
            crate::proof::TracePolynomialOrder::CycleMajor,
            2,
            8,
            None,
            None,
            Some(CommittedProgramSchedule {
                bytecode_len: 16,
                bytecode_chunk_count: 2,
                program_image_len_words: 4,
                program_image_start_index: 0,
            }),
        )
        .unwrap_or_else(|error| panic!("test lattice precommitted schedule should build: {error}"))
    }

    #[cfg(feature = "akita")]
    fn lattice_validity_surface_fixture() -> (
        JoltVerifierPreprocessing<TestPcs, Pedersen<Bn254G1>>,
        CheckedInputs,
        JoltProtocolConfig,
        TestProof,
    ) {
        let preprocessing = committed_test_preprocessing();
        let public_io = public_io_for_preprocessing(&preprocessing);
        let placeholder_config = lattice_config([0; 32], 0);
        let mut proof = proof_with_lattice_payload(&placeholder_config, [0; 32], 0);
        let checked = validate_inputs(&preprocessing, &public_io, &proof, false, false)
            .unwrap_or_else(|error| panic!("inputs should validate: {error}"));
        let layout = expected_lattice_layout(&placeholder_config, &preprocessing, &proof, &checked);
        let config = lattice_config_with_derived_validity(
            layout.digest,
            layout.dimension,
            &checked.precommitted,
        );
        proof.protocol = config;
        proof.commitments = CommitmentPayload::Akita(AkitaCommitmentPayload::new(
            TestCommitment,
            layout.digest,
            layout.dimension,
        ));

        (preprocessing, checked, config, proof)
    }

    #[cfg(feature = "akita")]
    fn akita_stage8_statement_fixture(
        preprocessing: JoltVerifierPreprocessing<TestPcs, Pedersen<Bn254G1>>,
        trusted_advice_commitment_present: bool,
        configure_lattice: impl FnOnce(&mut JoltProtocolConfig),
    ) -> AkitaStage8StatementFixture {
        let public_io = public_io_for_preprocessing(&preprocessing);
        let placeholder_config = lattice_config([0; 32], 0);
        let mut proof = proof_with_lattice_payload(&placeholder_config, [0; 32], 0);
        if trusted_advice_commitment_present {
            proof.ram_K = 16;
        }
        let checked = validate_inputs(
            &preprocessing,
            &public_io,
            &proof,
            trusted_advice_commitment_present,
            false,
        )
        .unwrap_or_else(|error| panic!("inputs should validate: {error}"));
        let mut config = lattice_config([0; 32], 0);
        configure_lattice(&mut config);
        let layout = expected_lattice_layout(&config, &preprocessing, &proof, &checked);
        config.lattice.packed_witness.layout_digest = Some(layout.digest);
        config.lattice.packed_witness.d_pack = Some(layout.dimension);
        let requirements = stage8::derive_akita_packed_validity_requirements(
            &config,
            proof.one_hot_config.committed_chunk_bits(),
            &checked.precommitted,
        )
        .unwrap_or_else(|error| panic!("validity requirements should derive: {error}"));
        config.lattice.packed_witness.validity_digest =
            Some(lattice_packed_validity_digest(&requirements));
        proof.protocol = config;
        proof.commitments = CommitmentPayload::Akita(AkitaCommitmentPayload::new(
            TestCommitment,
            layout.digest,
            layout.dimension,
        ));

        let log_t = checked.trace_length.ilog2() as usize;
        let formula_dimensions = JoltFormulaDimensions::try_from(proof.one_hot_config.dimensions(
            log_t,
            2 * RISCV_XLEN,
            preprocessing.program.bytecode_len(),
            checked.ram_K,
        ))
        .unwrap_or_else(|error| panic!("formula dimensions should derive: {error}"));
        let bytecode_layout = checked
            .precommitted
            .bytecode
            .as_ref()
            .unwrap_or_else(|| panic!("committed bytecode schedule should exist"));
        let bytecode_address_point = field_zeros(bytecode_layout.log_bytecode_chunk_size() + 1);
        let stage6 = akita_snapshot_stage6_output(
            formula_dimensions.ra_layout.bytecode(),
            log_t,
            proof.one_hot_config.committed_chunk_bits(),
            bytecode_layout.chunk_count(),
            bytecode_address_point,
        );
        let stage7 = akita_snapshot_stage7_output(
            formula_dimensions.ra_layout,
            log_t,
            proof.one_hot_config.committed_chunk_bits(),
            &checked.precommitted,
        );

        (preprocessing, checked, proof, stage6, stage7)
    }

    #[cfg(feature = "akita")]
    fn akita_snapshot_stage6_output(
        _bytecode_ra_count: usize,
        log_t: usize,
        log_k_chunk: usize,
        _bytecode_chunk_count: usize,
        _bytecode_address_point: Vec<Fr>,
    ) -> stage6::Stage6ClearOutput<Fr> {
        let zero = Fr::zero();
        let trace_point = field_zeros(log_t);
        let unsigned_inc_chunk_count =
            jolt_claims::protocols::jolt::unsigned_inc_lower_chunk_count(log_k_chunk)
                .unwrap_or_else(|| panic!("test unsigned increment chunks should derive"));
        let mut output_claims = clear_claim_payload().stage6;
        output_claims.inc_claim_reduction = None;
        output_claims.booleanity.unsigned_inc_chunks = field_zeros(unsigned_inc_chunk_count);
        output_claims.unsigned_inc_claim_reduction = Some(
            stage6::inputs::UnsignedIncClaimReductionOutputOpeningClaims {
                unsigned_inc: zero,
                unsigned_inc_msb: zero,
            },
        );

        stage6::Stage6ClearOutput {
            public: stage6::outputs::Stage6PublicOutput {
                address_phase_challenges: Vec::new(),
                address_phase_batching_coefficients: Vec::new(),
                challenges: Vec::new(),
                batching_coefficients: Vec::new(),
                bytecode_gamma_powers: Vec::new(),
                stage1_gammas: Vec::new(),
                stage2_gammas: Vec::new(),
                stage3_gammas: Vec::new(),
                stage4_gammas: Vec::new(),
                stage5_gammas: Vec::new(),
                booleanity_reference_address: Vec::new(),
                booleanity_reference_cycle: Vec::new(),
                booleanity_gamma: zero,
                instruction_ra_gamma_powers: Vec::new(),
                inc_gamma: zero,
                #[cfg(feature = "field-inline")]
                field_inline: stage6::outputs::FieldInlineStage6PublicOutput {
                    field_inc_gamma: zero,
                },
                bytecode_reduction_eta: None,
            },
            output_claims,
            batch: stage6::outputs::VerifiedStage6Batch {
                address_phase_batching_coefficients: Vec::new(),
                address_phase_sumcheck_point: jolt_poly::Point::high_to_low(Vec::new()),
                address_phase_sumcheck_final_claim: zero,
                address_phase_expected_final_claim: zero,
                bytecode_read_raf_address: verified_stage6_address_sumcheck(Vec::new()),
                booleanity_address: verified_stage6_address_sumcheck(Vec::new()),
                batching_coefficients: Vec::new(),
                sumcheck_point: jolt_poly::Point::high_to_low(Vec::new()),
                sumcheck_final_claim: zero,
                expected_final_claim: zero,
                bytecode_read_raf: verified_bytecode_read_raf(Vec::new(), Vec::new(), Vec::new()),
                booleanity: stage6::outputs::VerifiedBooleanitySumcheck {
                    input_claim: zero,
                    sumcheck_point: Vec::new(),
                    r_address: Vec::new(),
                    r_cycle: Vec::new(),
                    opening_point: Vec::new(),
                    reference_address: Vec::new(),
                    reference_cycle: Vec::new(),
                    expected_output_claim: zero,
                },
                ram_hamming_booleanity: verified_stage6_sumcheck(Vec::new()),
                ram_ra_virtualization: stage6::outputs::VerifiedRamRaVirtualizationSumcheck {
                    input_claim: zero,
                    sumcheck_point: Vec::new(),
                    opening_point: Vec::new(),
                    ram_ra_opening_points: Vec::new(),
                    expected_output_claim: zero,
                },
                instruction_ra_virtualization:
                    stage6::outputs::VerifiedInstructionRaVirtualizationSumcheck {
                        input_claim: zero,
                        sumcheck_point: Vec::new(),
                        opening_point: Vec::new(),
                        instruction_ra_opening_points: Vec::new(),
                        expected_output_claim: zero,
                    },
                inc_claim_reduction: None,
                unsigned_inc_claim_reduction: Some(verified_stage6_sumcheck(trace_point.clone())),
                unsigned_inc_msb_booleanity: Some(verified_stage6_sumcheck(trace_point.clone())),
                #[cfg(feature = "field-inline")]
                field_registers_inc_claim_reduction: verified_stage6_sumcheck(trace_point.clone()),
                trusted_advice_cycle_phase: None,
                untrusted_advice_cycle_phase: None,
                bytecode_cycle_phase: None,
                program_image_cycle_phase: None,
            },
        }
    }

    #[cfg(feature = "akita")]
    fn akita_snapshot_stage7_output(
        ra_layout: JoltRaPolynomialLayout,
        log_t: usize,
        log_k_chunk: usize,
        precommitted: &PrecommittedSchedule,
    ) -> stage7::Stage7ClearOutput<Fr> {
        let zero = Fr::zero();
        let ra_point = field_zeros(log_k_chunk + log_t);
        let unsigned_inc_chunk_count =
            jolt_claims::protocols::jolt::unsigned_inc_lower_chunk_count(log_k_chunk)
                .unwrap_or_else(|| panic!("test unsigned increment chunks should derive"));
        let mut output_claims = clear_claim_payload().stage7;
        output_claims.hamming_weight_claim_reduction.instruction_ra =
            field_zeros(ra_layout.instruction());
        output_claims.hamming_weight_claim_reduction.bytecode_ra =
            field_zeros(ra_layout.bytecode());
        output_claims.hamming_weight_claim_reduction.ram_ra = field_zeros(ra_layout.ram());
        output_claims.unsigned_inc_chunk_reconstruction =
            Some(stage7::inputs::UnsignedIncChunkReconstructionOutputClaims {
                chunks: field_zeros(unsigned_inc_chunk_count),
            });

        let bytecode_layout = precommitted
            .bytecode
            .as_ref()
            .unwrap_or_else(|| panic!("committed bytecode schedule should exist"));
        let bytecode_point = field_zeros(
            bytecode::committed_lane_vars() + bytecode_layout.log_bytecode_chunk_size(),
        );
        let mut precommitted_final_openings = (0..bytecode_layout.chunk_count())
            .map(|index| stage7::outputs::PrecommittedFinalOpening {
                polynomial: JoltCommittedPolynomial::BytecodeChunk(index),
                point: bytecode_point.clone(),
                opening_claim: Some(zero),
            })
            .collect::<Vec<_>>();
        let program_image_layout = precommitted
            .program_image
            .as_ref()
            .unwrap_or_else(|| panic!("program image schedule should exist"));
        precommitted_final_openings.push(stage7::outputs::PrecommittedFinalOpening {
            polynomial: JoltCommittedPolynomial::ProgramImageInit,
            point: field_zeros(program_image_layout.image_shape().row_vars()),
            opening_claim: Some(zero),
        });

        stage7::Stage7ClearOutput {
            public: stage7::outputs::Stage7PublicOutput {
                challenges: Vec::new(),
                batching_coefficients: Vec::new(),
                hamming_gamma: zero,
            },
            output_claims,
            batch: stage7::outputs::VerifiedStage7Batch {
                batching_coefficients: Vec::new(),
                sumcheck_point: jolt_poly::Point::high_to_low(Vec::new()),
                sumcheck_final_claim: zero,
                expected_final_claim: zero,
                hamming_weight_claim_reduction:
                    stage7::outputs::VerifiedHammingWeightClaimReductionSumcheck {
                        input_claim: zero,
                        sumcheck_point: Vec::new(),
                        opening_point: ra_point.clone(),
                        instruction_ra_opening_points: vec![
                            ra_point.clone();
                            ra_layout.instruction()
                        ],
                        bytecode_ra_opening_points: vec![ra_point.clone(); ra_layout.bytecode()],
                        ram_ra_opening_points: vec![ra_point.clone(); ra_layout.ram()],
                        expected_output_claim: zero,
                    },
                trusted_advice_address_phase: None,
                untrusted_advice_address_phase: None,
                bytecode_address_phase: None,
                program_image_address_phase: None,
                unsigned_inc_chunk_reconstruction: Some(
                    stage7::outputs::VerifiedUnsignedIncChunkReconstructionSumcheck {
                        input_claim: zero,
                        sumcheck_point: field_zeros(log_k_chunk),
                        opening_point: ra_point.clone(),
                        expected_output_claim: zero,
                    },
                ),
            },
            precommitted_final_openings,
        }
    }

    #[cfg(feature = "akita")]
    fn final_opening_id_for_test(polynomial: JoltCommittedPolynomial) -> JoltOpeningId {
        match polynomial {
            JoltCommittedPolynomial::TrustedAdvice => {
                JoltOpeningId::trusted_advice(JoltRelationId::AdviceClaimReduction)
            }
            JoltCommittedPolynomial::UntrustedAdvice => {
                JoltOpeningId::untrusted_advice(JoltRelationId::AdviceClaimReduction)
            }
            JoltCommittedPolynomial::BytecodeChunk(_) => {
                JoltOpeningId::committed(polynomial, JoltRelationId::BytecodeClaimReduction)
            }
            JoltCommittedPolynomial::ProgramImageInit => {
                JoltOpeningId::committed(polynomial, JoltRelationId::ProgramImageClaimReduction)
            }
            JoltCommittedPolynomial::RamInc | JoltCommittedPolynomial::RdInc => {
                JoltOpeningId::committed(polynomial, JoltRelationId::IncClaimReduction)
            }
            JoltCommittedPolynomial::InstructionRa(_)
            | JoltCommittedPolynomial::BytecodeRa(_)
            | JoltCommittedPolynomial::RamRa(_) => {
                JoltOpeningId::committed(polynomial, JoltRelationId::HammingWeightClaimReduction)
            }
        }
    }

    #[cfg(feature = "akita")]
    fn verified_stage6_address_sumcheck(
        opening_point: Vec<Fr>,
    ) -> stage6::outputs::VerifiedStage6AddressPhaseSumcheck<Fr> {
        stage6::outputs::VerifiedStage6AddressPhaseSumcheck {
            input_claim: Fr::zero(),
            sumcheck_point: Vec::new(),
            opening_point,
            expected_output_claim: Fr::zero(),
        }
    }

    #[cfg(feature = "akita")]
    fn verified_stage6_sumcheck(
        opening_point: Vec<Fr>,
    ) -> stage6::outputs::VerifiedStage6Sumcheck<Fr> {
        stage6::outputs::VerifiedStage6Sumcheck {
            input_claim: Fr::zero(),
            sumcheck_point: Vec::new(),
            opening_point,
            expected_output_claim: Fr::zero(),
        }
    }

    #[cfg(feature = "akita")]
    fn verified_bytecode_read_raf(
        r_address: Vec<Fr>,
        r_cycle: Vec<Fr>,
        bytecode_ra_opening_points: Vec<Vec<Fr>>,
    ) -> stage6::outputs::VerifiedBytecodeReadRafSumcheck<Fr> {
        stage6::outputs::VerifiedBytecodeReadRafSumcheck {
            input_claim: Fr::zero(),
            sumcheck_point: Vec::new(),
            full_opening_point: [r_address.as_slice(), r_cycle.as_slice()].concat(),
            r_address,
            r_cycle,
            bytecode_ra_opening_points,
            expected_output_claim: Fr::zero(),
        }
    }

    #[cfg(feature = "akita")]
    fn clear_claim_payload() -> ClearProofClaims<Fr> {
        match clear_claims() {
            JoltProofClaims::Clear(claims) => claims,
            JoltProofClaims::Zk { .. } => panic!("test clear claims helper returned ZK claims"),
        }
    }

    #[cfg(feature = "akita")]
    fn field_zeros(len: usize) -> Vec<Fr> {
        vec![Fr::zero(); len]
    }

    fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
        haystack
            .windows(needle.len())
            .any(|window| window == needle)
    }

    fn u64_word_bytes(value: u64) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[24..].copy_from_slice(&value.to_be_bytes());
        bytes
    }

    #[cfg(feature = "akita")]
    fn expected_lattice_layout(
        config: &JoltProtocolConfig,
        preprocessing: &JoltVerifierPreprocessing<TestPcs, Pedersen<Bn254G1>>,
        proof: &TestProof,
        checked: &CheckedInputs,
    ) -> jolt_openings::PackedWitnessLayout {
        let log_t = checked.trace_length.ilog2() as usize;
        let formula_dimensions = JoltFormulaDimensions::try_from(proof.one_hot_config.dimensions(
            log_t,
            2 * RISCV_XLEN,
            preprocessing.program.bytecode_len(),
            checked.ram_K,
        ))
        .unwrap_or_else(|error| panic!("formula dimensions should derive: {error}"));
        stage8::derive_akita_packed_witness_layout(
            config,
            log_t,
            proof.one_hot_config.committed_chunk_bits(),
            formula_dimensions.ra_layout,
            &checked.precommitted,
        )
        .unwrap_or_else(|error| panic!("Akita layout should derive: {error}"))
    }
}
