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
                .as_lattice()
                .ok_or_else(|| VerifierError::InvalidProtocolConfig {
                    reason: "lattice packed validity verification requires lattice commitments"
                        .to_string(),
                })?;
        let validity_claims = proof
            .clear_claims()?
            .stage7
            .lattice_packed_validity
            .as_ref()
            .ok_or(VerifierError::MissingLatticePackedValidityProof {
                field: "opening_claims",
            })?;
        let sumcheck_proof = proof
            .stages
            .lattice_packed_validity_sumcheck_proof
            .as_ref()
            .ok_or(VerifierError::MissingLatticePackedValidityProof {
                field: "sumcheck_proof",
            })?;
        let opening_proof = proof.lattice_packed_validity_opening_proof.as_ref().ok_or(
            VerifierError::MissingLatticePackedValidityProof {
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
        let layout = stage8::derive_lattice_packed_witness_layout(
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
        let layout = stage8::derive_lattice_packed_witness_layout(
            config,
            log_t,
            proof.one_hot_config.committed_chunk_bits(),
            formula_dimensions.ra_layout,
            &checked.precommitted,
        )?;
        stage8::validate_lattice_packed_witness_layout_config(config, &layout)?;
        stage8::validate_lattice_packed_witness_validity_config(
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
            return Err(VerifierError::UnexpectedLatticePackedValidityProof {
                field: "sumcheck_proof",
            });
        }
        if proof.lattice_packed_validity_opening_proof.is_some() {
            return Err(VerifierError::UnexpectedLatticePackedValidityProof {
                field: "opening_proof",
            });
        }
        if validity_claims.is_some() {
            return Err(VerifierError::UnexpectedLatticePackedValidityProof {
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
        return Err(VerifierError::MissingLatticePackedValidityProof {
            field: "sumcheck_proof",
        });
    }
    if proof.lattice_packed_validity_opening_proof.is_none() {
        return Err(VerifierError::MissingLatticePackedValidityProof {
            field: "opening_proof",
        });
    }
    let validity_claims =
        validity_claims.ok_or(VerifierError::MissingLatticePackedValidityProof {
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
        let layout = stage8::derive_lattice_packed_witness_layout(
            config,
            log_t,
            proof.one_hot_config.committed_chunk_bits(),
            formula_dimensions.ra_layout,
            &checked.precommitted,
        )?;
        let requirements = stage8::derive_lattice_packed_validity_requirements(
            config,
            proof.one_hot_config.committed_chunk_bits(),
            &checked.precommitted,
        )?;
        let statements = stage8::derive_lattice_packed_validity_statements(&layout, &requirements)?;
        let expected_opening_claims = stage8::lattice_packed_validity_opening_count(&statements);
        if validity_claims.opening_claims.len() != expected_opening_claims {
            return Err(VerifierError::LatticePackedValidityClaimCountMismatch {
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
        CommitmentPayload::Lattice(payload) => {
            absorb_lattice_protocol_header(transcript, &proof.protocol);
            append_payload_label(
                transcript,
                b"lattice_packed_witness",
                &payload.packed_witness,
            );
            transcript.append(&payload.packed_witness);
            absorb_labeled_bytes(transcript, b"lattice_layout_digest", &payload.layout_digest);
            absorb_labeled_u64(transcript, b"lattice_d_pack", payload.d_pack as u64);
            if let Some(validity_digest) = proof.protocol.lattice.packed_witness.validity_digest {
                absorb_labeled_bytes(transcript, b"lattice_validity_digest", &validity_digest);
            }
        }
    }
    Ok(())
}

fn absorb_lattice_protocol_header<T: Transcript>(
    transcript: &mut T,
    protocol: &JoltProtocolConfig,
) {
    transcript.append(&Label(b"lattice_protocol_header"));
    absorb_labeled_u64(
        transcript,
        b"lattice_pcs_curve",
        (protocol.pcs == PcsFamily::Curve) as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"lattice_pcs_lattice",
        (protocol.pcs == PcsFamily::Lattice) as u64,
    );
    absorb_labeled_u64(transcript, b"lattice_zk", zk_config_tag(protocol.zk));
    absorb_labeled_u64(
        transcript,
        b"lattice_program_mode",
        program_mode_tag(protocol.lattice.program_mode),
    );
    absorb_labeled_u64(
        transcript,
        b"lattice_increment_mode",
        increment_mode_tag(protocol.lattice.increment_mode),
    );
    absorb_labeled_u64(
        transcript,
        b"lattice_blindfold_zk",
        (protocol.zk == ZkConfig::BlindFold) as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"lattice_field_inline",
        protocol.lattice.field_inline.enabled as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"lattice_advice_trusted",
        protocol.lattice.advice.trusted as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"lattice_advice_untrusted",
        protocol.lattice.advice.untrusted as u64,
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
#[path = "verifier_tests.rs"]
mod tests;
