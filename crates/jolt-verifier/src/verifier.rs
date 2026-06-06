//! Top-level verifier entry point.

use common::jolt_device::JoltDevice;
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_field::{Field, RingAccumulator, WithAccumulator};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, ZkOpeningScheme};
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};

#[cfg(feature = "field-inline")]
use jolt_claims::protocols::{
    field_inline::formulas::bytecode as field_bytecode, jolt::JoltRelationId,
};

use crate::{
    config::{validate_proof_config, JoltProtocolConfig},
    pcs_assist::PcsProofAssist,
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{
        stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8,
        zk::{blindfold, committed, inputs::BlindFoldInputs, outputs::zk_stage_outputs},
    },
    VerifierError,
};

#[derive(Debug)]
pub struct PreStage1VerifierState<T> {
    pub checked: CheckedInputs,
    pub transcript: T,
}

pub fn verify<F, PCS, VC, T, PcsAssist>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<
        PCS,
        VC,
        jolt_blindfold::BlindFoldProof<F, <VC as jolt_crypto::Commitment>::Output>,
        PcsAssist,
    >,
    trusted_advice_commitment: Option<&PCS::Output>,
    zk: bool,
) -> Result<(), VerifierError>
where
    F: Field + AppendToTranscript,
    PCS: CommitmentScheme<Field = F>
        + AdditivelyHomomorphic
        + ZkOpeningScheme<HidingCommitment = VC::Output>,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: Copy + HomomorphicCommitment<F> + AppendToTranscript,
    PcsAssist: PcsProofAssist<PCS>,
    T: Transcript<Challenge = F>,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    let config = JoltProtocolConfig::selected_for_zk::<PCS, PcsAssist>(zk);
    validate_proof_config(&config, proof)?;

    let checked = validate_inputs(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.is_some(),
        zk,
    )?;
    validate_proof_consistency(proof, checked.zk)?;

    let mut transcript = T::new(b"Jolt");
    absorb_preamble(&checked, proof, &mut transcript);
    absorb_commitments(proof, trusted_advice_commitment, &mut transcript);

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
    let stage6 = stage6::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage6::deps(&stage1, &stage2, &stage3, &stage4, &stage5)?,
    )?;
    let stage7 = stage7::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage7::deps(&stage4, &stage6)?,
    )?;
    let stage8 = stage8::verify(
        &checked,
        &config,
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
        jolt_blindfold::verify::<F, VC, T>(
            &blindfold.protocol,
            proof.blindfold_proof()?,
            vc_setup,
            &mut transcript,
        )
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

pub fn verify_until_stage1<PCS, VC, T, ZkProof, PcsAssist>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC, ZkProof, PcsAssist>,
    trusted_advice_commitment: Option<&PCS::Output>,
    zk: bool,
) -> Result<PreStage1VerifierState<T>, VerifierError>
where
    PCS: CommitmentScheme,
    PCS::Output: AppendToTranscript,
    VC: VectorCommitment<Field = PCS::Field>,
    PcsAssist: PcsProofAssist<PCS>,
    T: Transcript<Challenge = PCS::Field>,
{
    let config = JoltProtocolConfig::selected_for_zk::<PCS, PcsAssist>(zk);
    validate_proof_config(&config, proof)?;

    let checked = validate_inputs(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.is_some(),
        zk,
    )?;
    validate_proof_consistency(proof, checked.zk)?;

    let mut transcript = T::new(b"Jolt");
    absorb_preamble(&checked, proof, &mut transcript);
    absorb_commitments(proof, trusted_advice_commitment, &mut transcript);

    Ok(PreStage1VerifierState {
        checked,
        transcript,
    })
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
    #[cfg(feature = "field-inline")]
    pub field_inline_bytecode_transcript: Vec<u8>,
}

pub fn validate_inputs<PCS, VC, ZkProof, PcsAssist>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC, ZkProof, PcsAssist>,
    trusted_advice_commitment_present: bool,
    zk: bool,
) -> Result<CheckedInputs, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    PcsAssist: PcsProofAssist<PCS>,
{
    if public_io.memory_layout != preprocessing.program.memory_layout {
        return Err(VerifierError::MemoryLayoutMismatch);
    }

    let max_input_size = preprocessing.program.memory_layout.max_input_size as usize;
    if public_io.inputs.len() > max_input_size {
        return Err(VerifierError::InputTooLarge {
            got: public_io.inputs.len(),
            max: max_input_size,
        });
    }

    let max_output_size = preprocessing.program.memory_layout.max_output_size as usize;
    if public_io.outputs.len() > max_output_size {
        return Err(VerifierError::OutputTooLarge {
            got: public_io.outputs.len(),
            max: max_output_size,
        });
    }

    if !proof.trace_length.is_power_of_two()
        || proof.trace_length > preprocessing.program.max_padded_trace_length
    {
        return Err(VerifierError::InvalidTraceLength {
            got: proof.trace_length,
            max: preprocessing.program.max_padded_trace_length,
        });
    }

    if !proof.ram_K.is_power_of_two() {
        return Err(VerifierError::InvalidRamK { got: proof.ram_K });
    }

    let vc_capacity = if zk {
        Some(validate_zk_vector_commitment_setup::<PCS, VC>(
            preprocessing,
        )?)
    } else {
        None
    };

    #[cfg(feature = "field-inline")]
    let field_inline_bytecode_transcript = {
        let field_inline_bytecode =
            preprocessing
                .field_inline_bytecode
                .as_deref()
                .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::BytecodeReadRaf,
                    reason: "field-inline bytecode metadata is missing".to_string(),
                })?;
        field_bytecode::validate_bytecode_rows(
            field_inline_bytecode,
            preprocessing.program.bytecode.code_size,
            proof.protocol.field_inline.field_register_log_k,
        )
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: error.to_string(),
        })?;
        field_bytecode::bytecode_transcript_bytes(
            field_inline_bytecode,
            proof.protocol.field_inline.field_register_log_k,
        )
    };

    let mut normalized_public_io = public_io.clone();
    normalized_public_io.outputs.truncate(
        normalized_public_io
            .outputs
            .iter()
            .rposition(|&byte| byte != 0)
            .map_or(0, |position| position + 1),
    );

    Ok(CheckedInputs {
        public_io: normalized_public_io,
        zk,
        trace_length: proof.trace_length,
        ram_K: proof.ram_K,
        entry_address: preprocessing.program.bytecode.entry_address,
        preprocessing_digest: preprocessing.preprocessing_digest,
        trusted_advice_commitment_present,
        vc_capacity,
        #[cfg(feature = "field-inline")]
        field_inline_bytecode_transcript,
    })
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

pub(crate) fn absorb_preamble<PCS, VC, ZkProof, PcsAssist, T>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof, PcsAssist>,
    transcript: &mut T,
) where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    PcsAssist: PcsProofAssist<PCS>,
    T: Transcript<Challenge = PCS::Field>,
{
    let public_io = &checked.public_io;
    absorb_labeled_bytes(
        transcript,
        b"preprocessing_digest",
        &checked.preprocessing_digest,
    );
    #[cfg(feature = "field-inline")]
    absorb_labeled_bytes(
        transcript,
        b"field_inline_bytecode",
        &checked.field_inline_bytecode_transcript,
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

pub(crate) fn absorb_commitments<PCS, VC, ZkProof, PcsAssist, T>(
    proof: &JoltProof<PCS, VC, ZkProof, PcsAssist>,
    trusted_advice_commitment: Option<&PCS::Output>,
    transcript: &mut T,
) where
    PCS: CommitmentScheme,
    PCS::Output: AppendToTranscript,
    VC: VectorCommitment<Field = PCS::Field>,
    PcsAssist: PcsProofAssist<PCS>,
    T: Transcript<Challenge = PCS::Field>,
{
    let mut absorb_commitment = |commitment: &PCS::Output| {
        append_payload_label(transcript, b"commitment", commitment);
        transcript.append(commitment);
    };
    absorb_commitment(&proof.commitments.rd_inc);
    absorb_commitment(&proof.commitments.ram_inc);
    for commitment in &proof.commitments.ra.instruction {
        absorb_commitment(commitment);
    }
    for commitment in &proof.commitments.ra.ram {
        absorb_commitment(commitment);
    }
    for commitment in &proof.commitments.ra.bytecode {
        absorb_commitment(commitment);
    }
    #[cfg(feature = "field-inline")]
    {
        absorb_commitment(&proof.commitments.field_inline.field_registers.rd_inc);
    }
    if let Some(untrusted_advice_commitment) = &proof.untrusted_advice_commitment {
        append_payload_label(transcript, b"untrusted_advice", untrusted_advice_commitment);
        transcript.append(untrusted_advice_commitment);
    }
    if let Some(trusted_advice_commitment) = trusted_advice_commitment {
        append_payload_label(transcript, b"trusted_advice", trusted_advice_commitment);
        transcript.append(trusted_advice_commitment);
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

pub fn validate_proof_consistency<PCS, VC, ZkProof, PcsAssist>(
    proof: &JoltProof<PCS, VC, ZkProof, PcsAssist>,
    zk: bool,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    PcsAssist: PcsProofAssist<PCS>,
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
        &proof.stages.stage6_sumcheck_proof,
        "stage6_sumcheck_proof",
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
    use super::*;
    use crate::pcs_assist::{NoPcsAssist, NoPcsAssistProof};
    use crate::proof::{ClearProofClaims, JoltProofClaims, JoltStageProofs};
    #[cfg(feature = "field-inline")]
    use crate::proof::{FieldInlineCommitments, FieldRegistersCommitments};
    use common::jolt_device::{JoltDevice, MemoryLayout};
    use jolt_claims::protocols::field_inline::FieldInlineConfig;
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig};
    use jolt_crypto::{Bn254G1, Commitment, Pedersen, PedersenSetup, VectorCommitmentOpening};
    use jolt_field::Fr;
    use jolt_openings::{CommitmentScheme, OpeningsError};
    use jolt_poly::{MultilinearPoly, Polynomial};
    use jolt_program::preprocess::{
        BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing,
    };
    use jolt_sumcheck::{
        ClearProof, ClearSumcheckProof, CommittedSumcheckProof, CompressedSumcheckProof,
    };
    use jolt_transcript::{Transcript, U64Word};
    use num_traits::Zero;

    #[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct TestPcs;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct TestCommitment(u64);

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
            (TestCommitment::default(), ())
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

    impl jolt_transcript::AppendToTranscript for TestCommitment {
        fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
            transcript.append(&U64Word(self.0));
        }

        fn transcript_payload_len(&self) -> Option<u64> {
            Some(32)
        }
    }

    #[derive(Clone, Default)]
    struct RecordingTranscript {
        chunks: Vec<Vec<u8>>,
        state: [u8; 32],
    }

    impl Transcript for RecordingTranscript {
        type Challenge = Fr;

        fn new(_label: &'static [u8]) -> Self {
            Self::default()
        }

        fn append_bytes(&mut self, bytes: &[u8]) {
            self.chunks.push(bytes.to_vec());
        }

        fn challenge(&mut self) -> Self::Challenge {
            Fr::zero()
        }

        fn state(&self) -> &[u8; 32] {
            &self.state
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
    fn validate_proof_config_accepts_default_disabled_field_inline() {
        let mut proof = proof_with_zk(false, clear_claims());
        let config = protocol_config(false, FieldInlineConfig::disabled());
        proof.protocol = config.clone();
        set_selected_pcs_assist_payload(&mut proof);

        assert!(validate_proof_config(&config, &proof).is_ok());
    }

    #[test]
    fn validate_proof_config_rejects_protocol_mismatch() {
        let mut proof = proof_with_zk(false, clear_claims());
        proof.protocol = JoltProtocolConfig::for_zk(true);

        assert!(matches!(
            validate_proof_config(&JoltProtocolConfig::for_zk(false), &proof),
            Err(VerifierError::ProtocolConfigMismatch { .. })
        ));
    }

    #[test]
    fn validate_proof_config_accepts_enabled_field_inline_config() {
        let mut proof = proof_with_zk(false, clear_claims());
        let config = protocol_config(false, FieldInlineConfig::native_v1());
        proof.protocol = config.clone();
        set_selected_pcs_assist_payload(&mut proof);

        assert!(validate_proof_config(&config, &proof).is_ok());
    }

    #[test]
    fn validate_proof_config_rejects_field_inline_config_mismatch() {
        let mut proof = proof_with_zk(false, clear_claims());
        let config = protocol_config(false, FieldInlineConfig::native_v1());
        proof.protocol = config;
        set_selected_pcs_assist_payload(&mut proof);

        assert!(matches!(
            validate_proof_config(
                &protocol_config(false, FieldInlineConfig::disabled()),
                &proof
            ),
            Err(VerifierError::ProtocolConfigMismatch { .. })
        ));
    }

    #[cfg(not(feature = "pcs-assist"))]
    #[test]
    fn validate_proof_config_rejects_unexpected_pcs_assist_payload() {
        let mut proof = proof_with_zk(false, clear_claims());
        let config = protocol_config(false, FieldInlineConfig::disabled());
        proof.protocol = config.clone();
        proof.pcs_assist = Some(NoPcsAssistProof);

        assert!(matches!(
            validate_proof_config(&config, &proof),
            Err(VerifierError::UnexpectedPcsAssistProof)
        ));
    }

    #[cfg(feature = "pcs-assist")]
    #[test]
    fn validate_proof_config_rejects_missing_required_pcs_assist_payload() {
        let mut proof = proof_with_zk(false, clear_claims());
        let config = protocol_config(false, FieldInlineConfig::disabled());
        proof.protocol = config.clone();

        assert!(matches!(
            validate_proof_config(&config, &proof),
            Err(VerifierError::MissingPcsAssistProof)
        ));
    }

    #[test]
    fn validate_inputs_normalizes_public_output() {
        let preprocessing = test_preprocessing();
        let mut public_io = JoltDevice {
            memory_layout: preprocessing.program.memory_layout.clone(),
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

    #[cfg(feature = "field-inline")]
    #[test]
    fn validate_inputs_rejects_missing_field_inline_bytecode() {
        let mut preprocessing = test_preprocessing();
        preprocessing.field_inline_bytecode = None;
        let public_io = JoltDevice {
            memory_layout: preprocessing.program.memory_layout.clone(),
            ..JoltDevice::default()
        };
        let proof = proof_with_zk(false, clear_claims());

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &proof, false, false),
            Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                ..
            })
        ));
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn validate_inputs_rejects_malformed_field_inline_bytecode() {
        let mut preprocessing = test_preprocessing();
        preprocessing.program.bytecode.code_size = 1;
        preprocessing.field_inline_bytecode = Some(vec![field_bytecode::FieldInlineBytecodeRow {
            flags: field_bytecode::FieldInlineBytecodeFlags {
                mul: true,
                ..field_bytecode::FieldInlineBytecodeFlags::default()
            },
            ..field_bytecode::FieldInlineBytecodeRow::default()
        }]);
        let public_io = JoltDevice {
            memory_layout: preprocessing.program.memory_layout.clone(),
            ..JoltDevice::default()
        };
        let proof = proof_with_zk(false, clear_claims());

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &proof, false, false),
            Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                ..
            })
        ));
    }

    #[test]
    fn validate_inputs_rejects_missing_zk_vector_commitment_setup() {
        let preprocessing = test_preprocessing();
        let public_io = JoltDevice {
            memory_layout: preprocessing.program.memory_layout.clone(),
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
            memory_layout: preprocessing.program.memory_layout.clone(),
            ..JoltDevice::default()
        };
        let proof = proof_with_zk(true, zk_claims());

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &proof, false, true),
            Err(VerifierError::InvalidVectorCommitmentCapacity { got: 1, .. })
        ));
    }

    #[test]
    fn absorb_commitments_follows_selected_commitment_order() {
        let mut proof = proof_with_zk(false, clear_claims());
        proof.commitments = selected_commitments(
            TestCommitment(1),
            TestCommitment(2),
            crate::proof::JoltRaCommitments::new(
                vec![TestCommitment(3), TestCommitment(4)],
                vec![TestCommitment(5)],
                vec![TestCommitment(6)],
            ),
        );
        proof.untrusted_advice_commitment = Some(TestCommitment(10));

        let trusted_advice_commitment = TestCommitment(11);
        let mut transcript = RecordingTranscript::new(b"commitments");
        absorb_commitments(&proof, Some(&trusted_advice_commitment), &mut transcript);

        let mut expected = vec![1, 2, 3, 4, 5, 6];
        #[cfg(feature = "field-inline")]
        expected.push(7);
        expected.extend([10, 11]);
        assert_eq!(commitment_payload_values(&transcript), expected);
    }

    fn commitment_payload_values(transcript: &RecordingTranscript) -> Vec<u64> {
        transcript
            .chunks
            .iter()
            .filter_map(|chunk| {
                let (prefix, suffix) = chunk.split_at(24);
                if prefix.iter().all(|byte| *byte == 0) {
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(suffix);
                    Some(u64::from_be_bytes(bytes))
                } else {
                    None
                }
            })
            .collect()
    }

    fn proof_with_zk(is_zk: bool, claims: TestClaims) -> TestProof {
        JoltProof::new(
            selected_commitments(
                TestCommitment::default(),
                TestCommitment::default(),
                crate::proof::JoltRaCommitments::new(
                    Vec::<TestCommitment>::new(),
                    Vec::<TestCommitment>::new(),
                    Vec::<TestCommitment>::new(),
                ),
            ),
            stage_proofs(is_zk),
            (),
            None,
            claims,
            1,
            1,
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

    #[cfg(not(feature = "field-inline"))]
    fn selected_commitments(
        rd_inc: TestCommitment,
        ram_inc: TestCommitment,
        ra: crate::proof::JoltRaCommitments<TestCommitment>,
    ) -> crate::proof::JoltCommitments<TestCommitment> {
        crate::proof::JoltCommitments::new(rd_inc, ram_inc, ra)
    }

    #[cfg(feature = "field-inline")]
    fn selected_commitments(
        rd_inc: TestCommitment,
        ram_inc: TestCommitment,
        ra: crate::proof::JoltRaCommitments<TestCommitment>,
    ) -> crate::proof::JoltCommitments<TestCommitment> {
        crate::proof::JoltCommitments::new(
            rd_inc,
            ram_inc,
            ra,
            FieldInlineCommitments::new(FieldRegistersCommitments::new(TestCommitment(7))),
        )
    }

    fn protocol_config(zk: bool, field_inline: FieldInlineConfig) -> JoltProtocolConfig {
        let mut config = JoltProtocolConfig::selected_for_zk::<TestPcs, NoPcsAssist>(zk);
        config.field_inline = field_inline;
        config
    }

    #[cfg(feature = "pcs-assist")]
    fn set_selected_pcs_assist_payload(proof: &mut TestProof) {
        proof.pcs_assist = Some(NoPcsAssistProof);
    }

    #[cfg(not(feature = "pcs-assist"))]
    fn set_selected_pcs_assist_payload(_proof: &mut TestProof) {}

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
                ram_ra_claim_reduction: stage5::inputs::RamRaClaimReductionOutputOpeningClaims {
                    ram_ra: zero,
                },
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
            stage6: stage6::inputs::Stage6Claims {
                bytecode_read_raf: stage6::inputs::BytecodeReadRafOutputOpeningClaims {
                    bytecode_ra: Vec::new(),
                },
                booleanity: stage6::inputs::BooleanityOutputOpeningClaims {
                    instruction_ra: Vec::new(),
                    bytecode_ra: Vec::new(),
                    ram_ra: Vec::new(),
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
                inc_claim_reduction: stage6::inputs::IncClaimReductionOutputOpeningClaims {
                    ram_inc: zero,
                    rd_inc: zero,
                },
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
            stage6_sumcheck_proof: sumcheck_proof(is_zk),
            stage7_sumcheck_proof: sumcheck_proof(is_zk),
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
        let memory_layout = MemoryLayout {
            max_input_size: 8,
            max_output_size: 8,
            heap_size: 8,
            ..MemoryLayout::default()
        };
        let preprocessing = JoltVerifierPreprocessing::new(
            JoltProgramPreprocessing {
                bytecode: BytecodePreprocessing::default(),
                ram: RAMPreprocessing::default(),
                memory_layout,
                max_padded_trace_length: 16,
            },
            [7; 32],
            (),
            None,
        );
        #[cfg(feature = "field-inline")]
        let preprocessing = preprocessing.with_field_inline_bytecode(Vec::new());
        preprocessing
    }
}
