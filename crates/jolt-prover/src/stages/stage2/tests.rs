#[cfg(feature = "zk")]
use common::jolt_device::{JoltDevice, MemoryConfig};
#[cfg(feature = "zk")]
use jolt_backends::cpu::CpuBackend;
#[cfg(feature = "zk")]
use jolt_crypto::{
    Bn254, Bn254G1, Commitment, JoltGroup, Pedersen, PedersenSetup, VectorCommitment,
};
#[cfg(feature = "zk")]
use jolt_field::{Fr, FromPrimitiveInt};
#[cfg(feature = "zk")]
use jolt_openings::CommitmentScheme;
#[cfg(feature = "zk")]
use jolt_poly::Polynomial;
#[cfg(feature = "zk")]
use jolt_sumcheck::{ClearProof, ClearSumcheckProof, SumcheckProof};
#[cfg(feature = "zk")]
use jolt_transcript::{Blake2bTranscript, Transcript};
#[cfg(all(feature = "zk", feature = "field-inline"))]
use jolt_witness::protocols::jolt_vm::field_inline::{
    FieldInlineNamespace, FieldInlineRegisterReadWriteRow, FieldInlineRegisterReadWriteRows,
};
#[cfg(feature = "zk")]
use jolt_witness::protocols::jolt_vm::{
    JoltVmNamespace, JoltVmProductUniskipRow, JoltVmProductUniskipRows, JoltVmStage2Rows,
    JoltVmStage2TraceRow,
};
#[cfg(feature = "zk")]
use jolt_witness::protocols::jolt_vm::{JoltVmSpartanOuterRow, JoltVmSpartanOuterRows};
#[cfg(feature = "zk")]
use jolt_witness::{
    MaterializationPolicy, OracleDescriptor, OracleKind, OracleRef, OracleViewRequest,
    PolynomialEncoding, PolynomialView, RetentionHint, ViewRequirement, WitnessDimensions,
    WitnessError, WitnessProvider,
};

#[cfg(feature = "zk")]
use super::input::{Stage2BatchProverConfig, Stage2ProverInput};
#[cfg(feature = "zk")]
use super::prove::prove_committed_boundary as prove_stage2_committed_boundary;
#[cfg(feature = "zk")]
use crate::stages::stage1::{
    input::{Stage1ProverConfig, Stage1ProverInput},
    prove::prove_committed_boundary as prove_stage1_committed_boundary,
};

#[test]
#[cfg(all(feature = "zk", not(feature = "field-inline")))]
fn stage2_committed_boundary_produces_native_verifier_output(
) -> Result<(), Box<dyn std::error::Error>> {
    let public_io = test_public_io();
    let trace_length = 16;
    let ram_k = 16;
    let witness = SatisfyingStage2Witness {
        final_ram_state: public_io_ram_state(&public_io, ram_k)?,
    };
    let mut backend = CpuBackend::default();
    let vc_setup = pedersen_setup(64);
    let checked = checked_inputs(public_io.clone(), trace_length, ram_k, &vc_setup);
    let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage2-zk-test");

    let stage1 = prove_stage1_committed_boundary::<Fr, _, _, _, Pedersen<Bn254G1>>(
        Stage1ProverInput::new(Stage1ProverConfig::new(4), &witness),
        &mut backend,
        &mut prover_transcript,
        &vc_setup,
    )?;
    let stage2 = prove_stage2_committed_boundary::<Fr, _, _, _, Pedersen<Bn254G1>>(
        Stage2ProverInput::new(stage2_config(), &checked, &stage1.verifier_output, &witness),
        &mut backend,
        &vc_setup,
        &mut prover_transcript,
    )?;

    assert_eq!(stage2.product_uniskip_output_claim_values.len(), 1);
    assert_eq!(stage2.batch_output_claim_values.len(), 15);
    assert_eq!(
        stage2
            .product_uniskip_committed_witness
            .round_coefficients
            .len(),
        1
    );
    assert_eq!(
        stage2.batch_committed_witness.round_coefficients.len(),
        stage2_config().log_t + stage2_config().log_k
    );

    let proof = stage2_zk_proof(
        stage1.uniskip_proof.clone(),
        stage1.remainder_proof.clone(),
        stage2.product_uniskip_proof.clone(),
        stage2.regular_batch_proof.clone(),
        trace_length,
        ram_k,
    );
    let preprocessing = verifier_preprocessing(public_io, trace_length, Some(vc_setup));
    let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage2-zk-test");
    let stage1_native = jolt_verifier::stages::stage1::verify(
        &checked,
        &preprocessing,
        &proof,
        &mut verifier_transcript,
    )?;
    let stage2_native = jolt_verifier::stages::stage2::verify(
        &checked,
        &preprocessing,
        &proof,
        &mut verifier_transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1_native),
    )?;
    let jolt_verifier::stages::stage2::Stage2Output::Zk(stage2_native) = stage2_native else {
        return Err("Stage 2 verifier did not return ZK output".into());
    };

    assert_eq!(stage2_native.public, stage2.public);
    assert_eq!(
        stage2_native
            .product_uniskip_output_claims
            .shape
            .output_claim_count,
        stage2.product_uniskip_output_claim_values.len()
    );
    assert_eq!(
        stage2_native.batch_output_claims.shape.output_claim_count,
        stage2.batch_output_claim_values.len()
    );
    assert_eq!(verifier_transcript.state(), prover_transcript.state());

    Ok(())
}

#[test]
#[cfg(all(feature = "zk", feature = "field-inline"))]
fn stage2_field_inline_committed_boundary_produces_native_verifier_output(
) -> Result<(), Box<dyn std::error::Error>> {
    let public_io = test_public_io();
    let trace_length = 16;
    let ram_k = 16;
    let witness = SatisfyingStage2Witness {
        final_ram_state: public_io_ram_state(&public_io, ram_k)?,
    };
    let field_inline_witness = FieldInlineWitness;
    let mut backend = CpuBackend::default();
    let vc_setup = pedersen_setup(64);
    let checked = checked_inputs(public_io.clone(), trace_length, ram_k, &vc_setup);
    let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage2-field-inline-zk-test");

    let stage1 = prove_stage1_committed_boundary::<Fr, _, _, _, _, Pedersen<Bn254G1>>(
        Stage1ProverInput::new(Stage1ProverConfig::new(4), &witness, &field_inline_witness),
        &mut backend,
        &mut prover_transcript,
        &vc_setup,
    )?;
    let stage2 = prove_stage2_committed_boundary::<Fr, _, _, _, _, Pedersen<Bn254G1>>(
        Stage2ProverInput::new(
            stage2_config(),
            &checked,
            &stage1.verifier_output,
            &witness,
            &field_inline_witness,
        ),
        &mut backend,
        &vc_setup,
        &mut prover_transcript,
    )?;

    assert_eq!(stage2.product_uniskip_output_claim_values.len(), 1);
    assert_eq!(stage2.batch_output_claim_values.len(), 18);
    assert_eq!(
        stage2.batch_committed_witness.round_coefficients.len(),
        stage2_config().log_t + stage2_config().log_k
    );

    let proof = stage2_zk_proof(
        stage1.uniskip_proof.clone(),
        stage1.remainder_proof.clone(),
        stage2.product_uniskip_proof.clone(),
        stage2.regular_batch_proof.clone(),
        trace_length,
        ram_k,
    );
    let preprocessing = verifier_preprocessing(public_io, trace_length, Some(vc_setup))
        .with_field_inline_bytecode(Vec::new());
    let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage2-field-inline-zk-test");
    let stage1_native = jolt_verifier::stages::stage1::verify(
        &checked,
        &preprocessing,
        &proof,
        &mut verifier_transcript,
    )?;
    let stage2_native = jolt_verifier::stages::stage2::verify(
        &checked,
        &preprocessing,
        &proof,
        &mut verifier_transcript,
        jolt_verifier::stages::stage2::inputs::deps(&stage1_native),
    )?;
    let jolt_verifier::stages::stage2::Stage2Output::Zk(stage2_native) = stage2_native else {
        return Err("Stage 2 verifier did not return field-inline ZK output".into());
    };

    assert_eq!(stage2_native.public, stage2.public);
    assert_eq!(
        stage2_native.batch_output_claims.shape.output_claim_count,
        stage2.batch_output_claim_values.len()
    );
    assert_eq!(verifier_transcript.state(), prover_transcript.state());

    Ok(())
}

#[cfg(feature = "zk")]
fn stage2_config() -> Stage2BatchProverConfig {
    Stage2BatchProverConfig::new(
        4,
        4,
        jolt_claims::protocols::jolt::JoltReadWriteConfig {
            ram_rw_phase1_num_rounds: 1,
            ram_rw_phase2_num_rounds: 1,
            registers_rw_phase1_num_rounds: 1,
            registers_rw_phase2_num_rounds: 1,
        },
    )
}

#[cfg(feature = "zk")]
fn test_public_io() -> JoltDevice {
    JoltDevice::new(&MemoryConfig {
        program_size: Some(1024),
        max_trusted_advice_size: 0,
        max_untrusted_advice_size: 0,
        max_input_size: 0,
        max_output_size: 0,
        ..Default::default()
    })
}

#[cfg(feature = "zk")]
fn public_io_ram_state(
    public_io: &JoltDevice,
    ram_k: usize,
) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
    let public_memory = jolt_program::preprocess::PublicIoMemory::new(public_io)
        .map_err(|error| format!("{error:?}"))?;
    let mut state = vec![0; ram_k];
    for segment in public_memory.segments {
        let end = segment.start_index + segment.words.len();
        state[segment.start_index..end].copy_from_slice(&segment.words);
    }
    Ok(state)
}

#[cfg(all(feature = "zk", not(feature = "field-inline")))]
fn checked_inputs(
    public_io: JoltDevice,
    trace_length: usize,
    ram_k: usize,
    vc_setup: &PedersenSetup<Bn254G1>,
) -> jolt_verifier::CheckedInputs {
    jolt_verifier::CheckedInputs {
        public_io,
        zk: true,
        trace_length,
        ram_K: ram_k,
        entry_address: 0,
        preprocessing_digest: [0; 32],
        trusted_advice_commitment_present: false,
        vc_capacity: Some(Pedersen::<Bn254G1>::capacity(vc_setup)),
    }
}

#[cfg(all(feature = "zk", feature = "field-inline"))]
fn checked_inputs(
    public_io: JoltDevice,
    trace_length: usize,
    ram_k: usize,
    vc_setup: &PedersenSetup<Bn254G1>,
) -> jolt_verifier::CheckedInputs {
    jolt_verifier::CheckedInputs {
        public_io,
        zk: true,
        trace_length,
        ram_K: ram_k,
        entry_address: 0,
        preprocessing_digest: [0; 32],
        trusted_advice_commitment_present: false,
        vc_capacity: Some(Pedersen::<Bn254G1>::capacity(vc_setup)),
        field_inline_bytecode_transcript: Vec::new(),
    }
}

#[cfg(feature = "zk")]
fn verifier_preprocessing(
    public_io: JoltDevice,
    trace_length: usize,
    vc_setup: Option<PedersenSetup<Bn254G1>>,
) -> jolt_verifier::JoltVerifierPreprocessing<
    jolt_openings::mock::MockCommitmentScheme<Fr>,
    Pedersen<Bn254G1>,
> {
    jolt_verifier::JoltVerifierPreprocessing::new(
        jolt_program::preprocess::JoltProgramPreprocessing {
            bytecode: Default::default(),
            ram: Default::default(),
            memory_layout: public_io.memory_layout,
            max_padded_trace_length: trace_length,
        },
        [0; 32],
        (),
        vc_setup,
    )
}

#[cfg(all(feature = "zk", not(feature = "field-inline")))]
fn stage2_zk_proof(
    stage1_uniskip_proof: SumcheckProof<Fr, Bn254G1>,
    stage1_remainder_proof: SumcheckProof<Fr, Bn254G1>,
    stage2_uniskip_proof: SumcheckProof<Fr, Bn254G1>,
    stage2_batch_proof: SumcheckProof<Fr, Bn254G1>,
    trace_length: usize,
    ram_k: usize,
) -> jolt_verifier::proof::JoltProof<
    jolt_openings::mock::MockCommitmentScheme<Fr>,
    Pedersen<Bn254G1>,
    (),
> {
    type MockPcs = jolt_openings::mock::MockCommitmentScheme<Fr>;
    let commitment = <MockPcs as Commitment>::Output::default();
    let commitments = jolt_verifier::proof::JoltCommitments::new(
        commitment.clone(),
        commitment,
        jolt_verifier::proof::JoltRaCommitments::new(Vec::new(), Vec::new(), Vec::new()),
    );
    stage2_zk_proof_with_commitments(
        commitments,
        stage1_uniskip_proof,
        stage1_remainder_proof,
        stage2_uniskip_proof,
        stage2_batch_proof,
        trace_length,
        ram_k,
    )
}

#[cfg(all(feature = "zk", feature = "field-inline"))]
fn stage2_zk_proof(
    stage1_uniskip_proof: SumcheckProof<Fr, Bn254G1>,
    stage1_remainder_proof: SumcheckProof<Fr, Bn254G1>,
    stage2_uniskip_proof: SumcheckProof<Fr, Bn254G1>,
    stage2_batch_proof: SumcheckProof<Fr, Bn254G1>,
    trace_length: usize,
    ram_k: usize,
) -> jolt_verifier::proof::JoltProof<
    jolt_openings::mock::MockCommitmentScheme<Fr>,
    Pedersen<Bn254G1>,
    (),
> {
    type MockPcs = jolt_openings::mock::MockCommitmentScheme<Fr>;
    let commitment = <MockPcs as Commitment>::Output::default();
    let commitments = jolt_verifier::proof::JoltCommitments::new(
        commitment.clone(),
        commitment.clone(),
        jolt_verifier::proof::JoltRaCommitments::new(Vec::new(), Vec::new(), Vec::new()),
        jolt_verifier::proof::FieldInlineCommitments::new(
            jolt_verifier::proof::FieldRegistersCommitments::new(commitment),
        ),
    );
    stage2_zk_proof_with_commitments(
        commitments,
        stage1_uniskip_proof,
        stage1_remainder_proof,
        stage2_uniskip_proof,
        stage2_batch_proof,
        trace_length,
        ram_k,
    )
}

#[cfg(feature = "zk")]
fn stage2_zk_proof_with_commitments(
    commitments: jolt_verifier::proof::JoltCommitments<
        <jolt_openings::mock::MockCommitmentScheme<Fr> as Commitment>::Output,
    >,
    stage1_uniskip_proof: SumcheckProof<Fr, Bn254G1>,
    stage1_remainder_proof: SumcheckProof<Fr, Bn254G1>,
    stage2_uniskip_proof: SumcheckProof<Fr, Bn254G1>,
    stage2_batch_proof: SumcheckProof<Fr, Bn254G1>,
    trace_length: usize,
    ram_k: usize,
) -> jolt_verifier::proof::JoltProof<
    jolt_openings::mock::MockCommitmentScheme<Fr>,
    Pedersen<Bn254G1>,
    (),
> {
    type MockPcs = jolt_openings::mock::MockCommitmentScheme<Fr>;
    let empty = SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof::default()));
    let stages = jolt_verifier::proof::JoltStageProofs {
        stage1_uni_skip_first_round_proof: stage1_uniskip_proof,
        stage1_sumcheck_proof: stage1_remainder_proof,
        stage2_uni_skip_first_round_proof: stage2_uniskip_proof,
        stage2_sumcheck_proof: stage2_batch_proof,
        stage3_sumcheck_proof: empty.clone(),
        stage4_sumcheck_proof: empty.clone(),
        stage5_sumcheck_proof: empty.clone(),
        stage6_sumcheck_proof: empty.clone(),
        stage7_sumcheck_proof: empty,
    };
    let poly = Polynomial::new(vec![Fr::from_u64(0)]);
    let mut transcript = Blake2bTranscript::<Fr>::new(b"stage2-zk-opening");
    let opening_proof = MockPcs::open(&poly, &[], Fr::from_u64(0), &(), None, &mut transcript);
    jolt_verifier::proof::JoltProof::<MockPcs, Pedersen<Bn254G1>, ()>::new(
        commitments,
        stages,
        opening_proof,
        None,
        jolt_verifier::proof::JoltProofClaims::Zk {
            blindfold_proof: (),
        },
        trace_length,
        ram_k,
        stage2_config().rw_config,
        jolt_claims::protocols::jolt::JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
        },
        jolt_verifier::proof::TracePolynomialOrder::AddressMajor,
    )
}

#[cfg(feature = "zk")]
fn pedersen_setup(capacity: usize) -> PedersenSetup<Bn254G1> {
    let generator = Bn254::g1_generator();
    let message_generators = (1..=capacity)
        .map(|index| generator.scalar_mul(&Fr::from_u64(index as u64)))
        .collect();
    PedersenSetup::new(message_generators, generator.scalar_mul(&Fr::from_u64(99)))
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug)]
struct SatisfyingStage2Witness {
    final_ram_state: Vec<u64>,
}

#[cfg(feature = "zk")]
impl WitnessProvider<Fr, JoltVmNamespace> for SatisfyingStage2Witness {
    fn describe_oracle(
        &self,
        oracle: OracleRef<JoltVmNamespace>,
    ) -> Result<OracleDescriptor<JoltVmNamespace>, WitnessError> {
        let dimensions = match oracle.kind {
            OracleKind::Virtual(
                jolt_claims::protocols::jolt::JoltVirtualPolynomial::RamVal
                | jolt_claims::protocols::jolt::JoltVirtualPolynomial::RamRa,
            ) => WitnessDimensions::new(256, 8),
            OracleKind::Committed(_) | OracleKind::Virtual(_) => WitnessDimensions::new(16, 4),
        };
        Ok(OracleDescriptor::new(
            oracle,
            dimensions,
            PolynomialEncoding::Dense,
        ))
    }

    fn view_requirements(
        &self,
        oracle: OracleRef<JoltVmNamespace>,
    ) -> Result<Vec<ViewRequirement<JoltVmNamespace>>, WitnessError> {
        let descriptor = self.describe_oracle(oracle)?;
        Ok(vec![ViewRequirement::new(
            descriptor.reference,
            descriptor.encoding,
            MaterializationPolicy::BackendChoice,
            RetentionHint::ThroughStage8,
        )])
    }

    fn oracle_view(
        &self,
        request: OracleViewRequest<JoltVmNamespace>,
    ) -> Result<PolynomialView<'_, Fr, JoltVmNamespace>, WitnessError> {
        let descriptor = self.describe_oracle(request.oracle())?;
        let rows = descriptor.dimensions.rows;
        let values = match request.oracle().kind {
            OracleKind::Virtual(
                jolt_claims::protocols::jolt::JoltVirtualPolynomial::NextUnexpandedPC,
            ) => vec![Fr::from_u64(4); rows],
            OracleKind::Virtual(
                jolt_claims::protocols::jolt::JoltVirtualPolynomial::RamValFinal,
            ) => self
                .final_ram_state
                .iter()
                .copied()
                .map(Fr::from_u64)
                .collect(),
            _ => vec![Fr::from_u64(0); rows],
        };
        Ok(PolynomialView::owned(descriptor, values))
    }
}

#[cfg(feature = "zk")]
impl JoltVmSpartanOuterRows for SatisfyingStage2Witness {
    fn spartan_outer_rows(&self) -> Result<Vec<JoltVmSpartanOuterRow>, WitnessError> {
        Ok(vec![
            JoltVmSpartanOuterRow {
                left_instruction_input: 0,
                right_instruction_input: 0,
                product_magnitude: 0,
                product_is_positive: false,
                should_branch: false,
                pc: 0,
                unexpanded_pc: 0,
                imm: 0,
                ram_address: 0,
                rs1_value: 0,
                rs2_value: 0,
                rd_write_value: 0,
                ram_read_value: 0,
                ram_write_value: 0,
                left_lookup_operand: 0,
                right_lookup_operand: 0,
                next_unexpanded_pc: 4,
                next_pc: 0,
                next_is_virtual: false,
                next_is_first_in_sequence: false,
                lookup_output: 0,
                should_jump: false,
                flag_add_operands: false,
                flag_subtract_operands: false,
                flag_multiply_operands: false,
                flag_load: false,
                flag_store: false,
                flag_jump: false,
                flag_write_lookup_output_to_rd: false,
                flag_virtual_instruction: false,
                flag_assert: false,
                flag_do_not_update_unexpanded_pc: false,
                flag_advice: false,
                flag_is_compressed: false,
                flag_is_first_in_sequence: false,
                flag_is_last_in_sequence: false,
            };
            16
        ])
    }
}

#[cfg(feature = "zk")]
impl JoltVmProductUniskipRows for SatisfyingStage2Witness {
    fn product_uniskip_rows(&self) -> Result<Vec<JoltVmProductUniskipRow>, WitnessError> {
        Ok(vec![
            JoltVmProductUniskipRow {
                right_instruction: 0,
                left_instruction: 0,
                lookup_output: 0,
                jump_flag: false,
                branch_flag: false,
                next_is_noop: false,
            };
            16
        ])
    }
}

#[cfg(feature = "zk")]
impl JoltVmStage2Rows for SatisfyingStage2Witness {
    fn stage2_rows(&self) -> Result<Vec<JoltVmStage2TraceRow>, WitnessError> {
        Ok(vec![
            JoltVmStage2TraceRow {
                remapped_ram_address: None,
                ram_read_value: 0,
                ram_write_value: 0,
                ram_increment: 0,
                left_instruction_input: 0,
                right_instruction_input: 0,
                lookup_output: 0,
                left_lookup_operand: 0,
                right_lookup_operand: 0,
                branch_flag: false,
                jump_flag: false,
                write_lookup_output_to_rd_flag: false,
                virtual_instruction_flag: false,
                next_is_noop: false,
            };
            16
        ])
    }

    fn initial_ram_state_words(&self) -> Result<Vec<u64>, WitnessError> {
        Ok(vec![0; 16])
    }

    fn final_ram_state_words(&self) -> Result<Vec<u64>, WitnessError> {
        Ok(self.final_ram_state.clone())
    }
}

#[cfg(all(feature = "zk", feature = "field-inline"))]
#[derive(Clone, Copy, Debug)]
struct FieldInlineWitness;

#[cfg(all(feature = "zk", feature = "field-inline"))]
impl WitnessProvider<Fr, FieldInlineNamespace> for FieldInlineWitness {
    fn describe_oracle(
        &self,
        oracle: OracleRef<FieldInlineNamespace>,
    ) -> Result<OracleDescriptor<FieldInlineNamespace>, WitnessError> {
        Ok(OracleDescriptor::new(
            oracle,
            WitnessDimensions::new(16, 4),
            PolynomialEncoding::Dense,
        ))
    }

    fn view_requirements(
        &self,
        oracle: OracleRef<FieldInlineNamespace>,
    ) -> Result<Vec<ViewRequirement<FieldInlineNamespace>>, WitnessError> {
        let descriptor = self.describe_oracle(oracle)?;
        Ok(vec![ViewRequirement::new(
            descriptor.reference,
            descriptor.encoding,
            MaterializationPolicy::BackendChoice,
            RetentionHint::ThroughStage8,
        )])
    }

    fn oracle_view(
        &self,
        request: OracleViewRequest<FieldInlineNamespace>,
    ) -> Result<PolynomialView<'_, Fr, FieldInlineNamespace>, WitnessError> {
        let descriptor = self.describe_oracle(request.oracle())?;
        Ok(PolynomialView::owned(descriptor, vec![Fr::from_u64(0); 16]))
    }
}

#[cfg(all(feature = "zk", feature = "field-inline"))]
impl FieldInlineRegisterReadWriteRows<Fr> for FieldInlineWitness {
    fn field_inline_register_read_write_rows(
        &self,
    ) -> Result<Vec<FieldInlineRegisterReadWriteRow<Fr>>, WitnessError> {
        Ok(vec![FieldInlineRegisterReadWriteRow::default(); 16])
    }
}
