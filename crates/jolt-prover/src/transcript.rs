use common::jolt_device::JoltDevice;
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig, TracePolynomialOrder};
use jolt_crypto::VectorCommitment;
use jolt_openings::CommitmentScheme;
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};
use jolt_verifier::{
    config::{validate_proof_config, JoltProtocolConfig},
    proof::JoltCommitments,
    verifier::{validate_inputs, validate_proof_consistency, CheckedInputs},
    JoltProof, JoltVerifierPreprocessing,
};

use crate::ProverError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage0TranscriptContext {
    pub rw_config: JoltReadWriteConfig,
    pub one_hot_config: JoltOneHotConfig,
    pub trace_polynomial_order: TracePolynomialOrder,
}

impl Stage0TranscriptContext {
    pub const fn new(
        rw_config: JoltReadWriteConfig,
        one_hot_config: JoltOneHotConfig,
        trace_polynomial_order: TracePolynomialOrder,
    ) -> Self {
        Self {
            rw_config,
            one_hot_config,
            trace_polynomial_order,
        }
    }
}

pub fn initialize_proof_transcript<PCS, VC, ZkProof, T>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment: Option<&PCS::Output>,
    zk: bool,
    transcript: &mut T,
) -> Result<CheckedInputs, ProverError>
where
    PCS: CommitmentScheme,
    PCS::Output: AppendToTranscript,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let config = JoltProtocolConfig::for_zk(zk);
    validate_proof_config(&config, proof)?;
    let checked = validate_inputs(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.is_some(),
        zk,
    )?;
    validate_proof_consistency(proof, checked.zk)?;

    absorb_stage0_transcript(
        &checked,
        Stage0TranscriptContext::new(
            proof.rw_config,
            proof.one_hot_config,
            proof.trace_polynomial_order,
        ),
        &proof.commitments,
        proof.untrusted_advice_commitment.as_ref(),
        trusted_advice_commitment,
        transcript,
    );
    Ok(checked)
}

pub fn absorb_stage0_transcript<C, T>(
    checked: &CheckedInputs,
    context: Stage0TranscriptContext,
    commitments: &JoltCommitments<C>,
    untrusted_advice_commitment: Option<&C>,
    trusted_advice_commitment: Option<&C>,
    transcript: &mut T,
) where
    C: AppendToTranscript,
    T: Transcript,
{
    absorb_preamble(checked, context, transcript);
    absorb_commitments(
        commitments,
        untrusted_advice_commitment,
        trusted_advice_commitment,
        transcript,
    );
}

fn absorb_preamble<T>(checked: &CheckedInputs, context: Stage0TranscriptContext, transcript: &mut T)
where
    T: Transcript,
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
        context.rw_config.ram_rw_phase1_num_rounds as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"ram_rw_phase2_num_rounds",
        context.rw_config.ram_rw_phase2_num_rounds as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"registers_rw_phase1_num_rounds",
        context.rw_config.registers_rw_phase1_num_rounds as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"registers_rw_phase2_num_rounds",
        context.rw_config.registers_rw_phase2_num_rounds as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"log_k_chunk",
        context.one_hot_config.log_k_chunk as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"lookups_ra_virtual_log_k_chunk",
        context.one_hot_config.lookups_ra_virtual_log_k_chunk as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"dory_layout",
        context.trace_polynomial_order.transcript_scalar(),
    );
}

fn absorb_commitments<C, T>(
    commitments: &JoltCommitments<C>,
    untrusted_advice_commitment: Option<&C>,
    trusted_advice_commitment: Option<&C>,
    transcript: &mut T,
) where
    C: AppendToTranscript,
    T: Transcript,
{
    let mut absorb_commitment = |commitment: &C| {
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
    #[cfg(feature = "field-inline")]
    {
        absorb_commitment(&commitments.field_inline.field_registers.rd_inc);
    }
    if let Some(untrusted_advice_commitment) = untrusted_advice_commitment {
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

#[cfg(test)]
mod tests {
    use common::jolt_device::{JoltDevice, MemoryConfig};
    use jolt_field::Fr;
    use jolt_transcript::AppendToTranscript;
    use jolt_verifier::proof::{JoltCommitments, JoltRaCommitments};

    use super::*;

    #[derive(Clone, Debug, Default)]
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
            Self::Challenge::default()
        }

        fn state(&self) -> &[u8; 32] {
            &self.state
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct TestCommitment(&'static [u8]);

    impl AppendToTranscript for TestCommitment {
        fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
            transcript.append_bytes(self.0);
        }

        fn transcript_payload_len(&self) -> Option<u64> {
            Some(self.0.len() as u64)
        }
    }

    fn checked_inputs() -> CheckedInputs {
        let memory_config = MemoryConfig {
            max_input_size: 16,
            max_trusted_advice_size: 16,
            max_untrusted_advice_size: 16,
            max_output_size: 16,
            stack_size: 16,
            heap_size: 16,
            program_size: Some(64),
        };
        let mut public_io = JoltDevice::new(&memory_config);
        public_io.inputs = vec![1, 2];
        public_io.outputs = vec![3, 4];
        CheckedInputs {
            public_io,
            zk: true,
            trace_length: 8,
            ram_K: 4,
            entry_address: 0x8000_0000,
            preprocessing_digest: [7; 32],
            trusted_advice_commitment_present: true,
            vc_capacity: Some(16),
            #[cfg(feature = "field-inline")]
            field_inline_bytecode_transcript: vec![9, 10],
        }
    }

    fn context() -> Stage0TranscriptContext {
        Stage0TranscriptContext::new(
            JoltReadWriteConfig {
                ram_rw_phase1_num_rounds: 1,
                ram_rw_phase2_num_rounds: 2,
                registers_rw_phase1_num_rounds: 3,
                registers_rw_phase2_num_rounds: 4,
            },
            JoltOneHotConfig {
                log_k_chunk: 2,
                lookups_ra_virtual_log_k_chunk: 1,
            },
            TracePolynomialOrder::AddressMajor,
        )
    }

    #[cfg(not(feature = "field-inline"))]
    fn commitments() -> JoltCommitments<TestCommitment> {
        JoltCommitments::new(
            TestCommitment(b"payload:rd_inc"),
            TestCommitment(b"payload:ram_inc"),
            JoltRaCommitments::new(
                vec![
                    TestCommitment(b"payload:instruction_0"),
                    TestCommitment(b"payload:instruction_1"),
                ],
                vec![TestCommitment(b"payload:ram_0")],
                vec![TestCommitment(b"payload:bytecode_0")],
            ),
        )
    }

    #[cfg(feature = "field-inline")]
    fn commitments() -> JoltCommitments<TestCommitment> {
        use jolt_verifier::proof::{FieldInlineCommitments, FieldRegistersCommitments};

        JoltCommitments::new(
            TestCommitment(b"payload:rd_inc"),
            TestCommitment(b"payload:ram_inc"),
            JoltRaCommitments::new(
                vec![
                    TestCommitment(b"payload:instruction_0"),
                    TestCommitment(b"payload:instruction_1"),
                ],
                vec![TestCommitment(b"payload:ram_0")],
                vec![TestCommitment(b"payload:bytecode_0")],
            ),
            FieldInlineCommitments::new(FieldRegistersCommitments::new(TestCommitment(
                b"payload:field_rd_inc",
            ))),
        )
    }

    #[test]
    fn stage0_transcript_boundary_absorbs_commitments_in_verifier_order() {
        let commitments = commitments();
        let untrusted = TestCommitment(b"payload:untrusted_advice");
        let trusted = TestCommitment(b"payload:trusted_advice");
        let mut transcript = RecordingTranscript::new(b"stage0-test");

        absorb_stage0_transcript(
            &checked_inputs(),
            context(),
            &commitments,
            Some(&untrusted),
            Some(&trusted),
            &mut transcript,
        );

        let payloads = transcript
            .chunks
            .iter()
            .filter_map(|chunk| {
                let bytes = chunk.as_slice();
                if bytes.starts_with(b"payload:") {
                    std::str::from_utf8(bytes).ok()
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        #[cfg(not(feature = "field-inline"))]
        assert_eq!(
            payloads,
            vec![
                "payload:rd_inc",
                "payload:ram_inc",
                "payload:instruction_0",
                "payload:instruction_1",
                "payload:ram_0",
                "payload:bytecode_0",
                "payload:untrusted_advice",
                "payload:trusted_advice",
            ]
        );
        #[cfg(feature = "field-inline")]
        assert_eq!(
            payloads,
            vec![
                "payload:rd_inc",
                "payload:ram_inc",
                "payload:instruction_0",
                "payload:instruction_1",
                "payload:ram_0",
                "payload:bytecode_0",
                "payload:field_rd_inc",
                "payload:untrusted_advice",
                "payload:trusted_advice",
            ]
        );
    }
}
