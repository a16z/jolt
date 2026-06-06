#[cfg(any(
    feature = "field-inline",
    all(feature = "zk", not(feature = "field-inline"))
))]
use jolt_backends::cpu::CpuBackend;
use jolt_backends::{
    Backend, BackendError, BackendRelationId, SumcheckBackend, SumcheckEvaluationOutput,
    SumcheckEvaluationRequest, SumcheckProofOutput, SumcheckRequest, SumcheckResult,
};
#[cfg(not(feature = "field-inline"))]
use jolt_backends::{
    SumcheckLinearProductOutput, SumcheckLinearProductRequest, SumcheckMaterializationOutput,
    SumcheckMaterializationRequest, SumcheckPrefixProductSumRequest,
    SumcheckSpartanOuterRemainderRound, SumcheckSpartanOuterRemainderState,
    SumcheckSpartanOuterRemainderStateRequest,
};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::{
    formulas::spartan::FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS, FieldInlineOpFlag,
    FieldInlineVirtualPolynomial,
};
use jolt_claims::protocols::jolt::{
    formulas::spartan::{SpartanOuterDimensions, SPARTAN_OUTER_R1CS_INPUTS},
    JoltVirtualPolynomial,
};
#[cfg(feature = "zk")]
use jolt_crypto::{Bn254, JoltGroup, PedersenSetup, VectorCommitment};
#[cfg(any(feature = "field-inline", feature = "zk"))]
use jolt_crypto::{Bn254G1, Commitment, Pedersen};
use jolt_field::{Fr, FromPrimitiveInt};
#[cfg(any(feature = "field-inline", feature = "zk"))]
use jolt_openings::CommitmentScheme;
#[cfg(feature = "field-inline")]
use jolt_poly::Polynomial;
#[cfg(not(feature = "field-inline"))]
use jolt_poly::{MultilinearPoly, TensorEqTable};
use jolt_r1cs::constraints::jolt::{
    JoltSpartanOuterRemainder, JoltSpartanOuterRemainderChallenges, SPARTAN_OUTER_REMAINDER_DEGREE,
    SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
};
use jolt_sumcheck::{
    BatchedSumcheckVerifier, CenteredIntegerDomain, SumcheckClaim, UNISKIP_ROUND_TRANSCRIPT_LABEL,
};
#[cfg(any(feature = "field-inline", feature = "zk"))]
use jolt_sumcheck::{ClearProof, ClearSumcheckProof, SumcheckProof};
use jolt_transcript::{Blake2bTranscript, Transcript};
#[cfg(feature = "field-inline")]
use jolt_witness::protocols::jolt_vm::field_inline::{
    FieldInlineNamespace, FieldInlineRegisterReadWriteRow, FieldInlineRegisterReadWriteRows,
};
use jolt_witness::protocols::jolt_vm::{JoltVmSpartanOuterRow, JoltVmSpartanOuterRows};
use jolt_witness::{
    protocols::jolt_vm::JoltVmNamespace, MaterializationPolicy, OracleDescriptor, OracleKind,
    OracleRef, OracleViewRequest, PolynomialEncoding, PolynomialView, RetentionHint,
    ViewRequirement, WitnessDimensions, WitnessError, WitnessNamespace, WitnessProvider,
};

#[cfg(any(feature = "field-inline", feature = "zk"))]
use super::input::Stage1ProverInput;
#[cfg(feature = "field-inline")]
use super::prove::prove as prove_stage1_public;
#[cfg(all(feature = "zk", not(feature = "field-inline")))]
use super::prove::prove_committed_boundary as prove_stage1_committed_boundary;
#[cfg(not(feature = "field-inline"))]
use super::prove::prove_stage1_transparent_sumchecks;
use super::{
    input::Stage1ProverConfig,
    output::{spartan_outer_claims_from_r1cs_inputs, Stage1R1csInputClaim, Stage1SumcheckOutput},
    prove::{evaluate_stage1_r1cs_inputs, prove_stage1_sumchecks},
    request::{
        build_stage1_request, r1cs_input_slot, SPARTAN_OUTER_REMAINDER_RELATION,
        SPARTAN_OUTER_UNISKIP_RELATION, STAGE1_REMAINDER_OUTPUT_SLOT, STAGE1_REMAINDER_SLOT,
        STAGE1_SPARTAN_OUTER_OPTIMIZATION_IDS, STAGE1_UNISKIP_INPUT_SLOT,
        STAGE1_UNISKIP_OUTPUT_SLOT, STAGE1_UNISKIP_SLOT,
    },
};
#[cfg(feature = "field-inline")]
use super::{
    output::{
        field_inline_stage1_claims_from_r1cs_inputs, stage1_claims_from_r1cs_inputs,
        Stage1FieldInlineR1csInputClaim,
    },
    request::{build_stage1_field_inline_r1cs_evaluation_request, field_inline_r1cs_input_slot},
};

#[test]
fn stage1_request_matches_verifier_spartan_outer_shape() -> Result<(), Box<dyn std::error::Error>> {
    let witness = Stage1Witness;
    let config = Stage1ProverConfig::new(4);
    let request = build_stage1_request::<Fr, _>(config, &witness)?;
    let dimensions = SpartanOuterDimensions::rv64(config.log_t);

    assert_eq!(request.sumchecks.label, "stage1.spartan_outer");
    assert_eq!(request.sumchecks.instances.len(), 2);
    assert_eq!(request.r1cs_inputs.len(), dimensions.variables().len());
    assert_eq!(request.r1cs_inputs.len(), SPARTAN_OUTER_R1CS_INPUTS.len());

    let uniskip = &request.sumchecks.instances[0];
    let uniskip_spec = dimensions.uniskip_sumcheck();
    assert_eq!(uniskip.slot, STAGE1_UNISKIP_SLOT);
    assert_eq!(uniskip.relation, SPARTAN_OUTER_UNISKIP_RELATION);
    assert_eq!(
        uniskip.optimization_ids,
        STAGE1_SPARTAN_OUTER_OPTIMIZATION_IDS
    );
    assert_eq!(uniskip.rounds, uniskip_spec.rounds);
    assert_eq!(uniskip.degree, uniskip_spec.degree);
    assert_eq!(uniskip.input_claim, STAGE1_UNISKIP_INPUT_SLOT);
    assert_eq!(uniskip.output_claim, STAGE1_UNISKIP_OUTPUT_SLOT);

    let remainder = &request.sumchecks.instances[1];
    let remainder_spec = dimensions.remainder_sumcheck();
    assert_eq!(remainder.slot, STAGE1_REMAINDER_SLOT);
    assert_eq!(remainder.relation, SPARTAN_OUTER_REMAINDER_RELATION);
    assert_eq!(
        remainder.optimization_ids,
        STAGE1_SPARTAN_OUTER_OPTIMIZATION_IDS
    );
    assert_eq!(remainder.rounds, remainder_spec.rounds);
    assert_eq!(remainder.degree, remainder_spec.degree);
    assert_eq!(remainder.input_claim, STAGE1_UNISKIP_OUTPUT_SLOT);
    assert_eq!(remainder.output_claim, STAGE1_REMAINDER_OUTPUT_SLOT);

    let expected_views = request
        .r1cs_inputs
        .iter()
        .map(|input| input.view)
        .collect::<Vec<_>>();
    assert_eq!(uniskip.witness_views, expected_views);
    assert_eq!(remainder.witness_views, expected_views);

    for (index, input) in request.r1cs_inputs.iter().enumerate() {
        assert_eq!(input.variable, SPARTAN_OUTER_R1CS_INPUTS[index]);
        assert_eq!(input.slot, r1cs_input_slot(index));
        assert_eq!(
            input.view.oracle,
            OracleRef::virtual_polynomial(input.variable)
        );
        assert_eq!(input.view.encoding, PolynomialEncoding::Dense);
        assert_eq!(
            input.view.materialization,
            MaterializationPolicy::BackendChoice
        );
        assert_eq!(input.view.retention, RetentionHint::ThroughStage8);
    }

    Ok(())
}

#[test]
#[cfg(feature = "zk")]
fn stage1_request_marks_committed_rounds_in_zk_mode() -> Result<(), Box<dyn std::error::Error>> {
    let witness = Stage1Witness;
    let request = build_stage1_request::<Fr, _>(
        Stage1ProverConfig::new(4).with_committed_rounds(true),
        &witness,
    )?;

    assert!(request
        .sumchecks
        .instances
        .iter()
        .all(|instance| instance.committed_rounds));
    Ok(())
}

#[test]
fn stage1_prove_maps_backend_outputs_by_slot() -> Result<(), Box<dyn std::error::Error>> {
    let witness = Stage1Witness;
    let config = Stage1ProverConfig::new(3);
    let mut backend = RecordingSumcheckBackend::default();

    let output = prove_stage1_sumchecks::<Fr, _, _>(config, &witness, &mut backend)?;

    assert_eq!(backend.labels, vec!["stage1.spartan_outer"]);
    assert_eq!(backend.relations[0], SPARTAN_OUTER_UNISKIP_RELATION);
    assert_eq!(backend.relations[1], SPARTAN_OUTER_REMAINDER_RELATION);
    assert_eq!(output.uniskip_proof, "uniskip");
    assert_eq!(output.remainder_proof, "remainder");
    assert_eq!(output.uniskip_output_claim, Fr::from_u64(11));
    assert_eq!(output.remainder_output_claim, Fr::from_u64(17));
    assert_eq!(
        output.r1cs_input_claims.len(),
        SPARTAN_OUTER_R1CS_INPUTS.len()
    );
    assert_eq!(
        output.r1cs_input_claims[0].variable,
        JoltVirtualPolynomial::LeftInstructionInput
    );
    assert_eq!(output.r1cs_input_claims[0].value, Fr::from_u64(100));

    Ok(())
}

#[test]
fn stage1_evaluation_helper_runs_backend_owned_view_evaluations(
) -> Result<(), Box<dyn std::error::Error>> {
    let witness = Stage1Witness;
    let config = Stage1ProverConfig::new(3);
    let mut backend = RecordingSumcheckBackend::default();

    let claims = evaluate_stage1_r1cs_inputs(
        config,
        &witness,
        &mut backend,
        vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)],
    )?;

    assert_eq!(
        backend.evaluation_labels,
        vec!["stage1.spartan_outer.r1cs_inputs"]
    );
    assert_eq!(
        backend.evaluation_points[0],
        vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)]
    );
    assert_eq!(claims.len(), SPARTAN_OUTER_R1CS_INPUTS.len());
    assert_eq!(
        claims[0].variable,
        JoltVirtualPolynomial::LeftInstructionInput
    );
    assert_eq!(claims[0].value, Fr::from_u64(200));
    assert_eq!(
        claims.last().map(|claim| claim.value),
        Some(Fr::from_u64(
            200 + SPARTAN_OUTER_R1CS_INPUTS.len() as u64 - 1
        ))
    );
    Ok(())
}

#[test]
#[cfg(not(feature = "field-inline"))]
fn stage1_transparent_helper_produces_verifier_compatible_sumchecks(
) -> Result<(), Box<dyn std::error::Error>> {
    let witness = SatisfyingStage1Witness;
    let config = Stage1ProverConfig::new(4);
    let mut backend = EvaluationBackend;
    let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage1-test");

    let output = prove_stage1_transparent_sumchecks::<Fr, _, _, _, Fr>(
        config,
        &witness,
        &mut backend,
        &mut prover_transcript,
    )?;

    let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage1-test");
    let tau = verifier_transcript.challenge_vector(config.log_t + 2);
    let uniskip_reduction = output.uniskip_proof.verify(
        &SumcheckClaim::new(1, SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE, Fr::from_u64(0)),
        CenteredIntegerDomain::new(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE),
        UNISKIP_ROUND_TRANSCRIPT_LABEL,
        &mut verifier_transcript,
    )?;
    assert_eq!(uniskip_reduction.value, output.uniskip_output_claim);

    verifier_transcript.append_labeled(b"opening_claim", &output.uniskip_output_claim);
    let remainder_batch = BatchedSumcheckVerifier::verify_compressed_boolean(
        &[SumcheckClaim::new(
            config.log_t + 1,
            SPARTAN_OUTER_REMAINDER_DEGREE,
            output.uniskip_output_claim,
        )],
        &output.remainder_proof,
        &mut verifier_transcript,
    )?;
    let r1cs_input_claims = output
        .r1cs_input_claims
        .iter()
        .map(|claim| claim.value)
        .collect::<Vec<_>>();
    let expected_remainder = JoltSpartanOuterRemainder::new(JoltSpartanOuterRemainderChallenges {
        tau: &tau,
        uniskip: uniskip_reduction.point[0],
        remainder: &remainder_batch.reduction.point,
    })?
    .expected_output_claim(&r1cs_input_claims)?
        * remainder_batch.batching_coefficients[0];
    assert_eq!(remainder_batch.reduction.value, expected_remainder);
    for opening_claim in &r1cs_input_claims {
        verifier_transcript.append_labeled(b"opening_claim", opening_claim);
    }
    assert_eq!(prover_transcript.state(), verifier_transcript.state());

    Ok(())
}

#[test]
#[cfg(all(feature = "zk", not(feature = "field-inline")))]
fn stage1_committed_boundary_produces_native_verifier_output(
) -> Result<(), Box<dyn std::error::Error>> {
    let witness = SatisfyingStage1Witness;
    let config = Stage1ProverConfig::new(4);
    let mut backend = CpuBackend::default();
    let vc_setup = pedersen_setup(64);
    let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage1-zk-test");

    let output = prove_stage1_committed_boundary::<Fr, _, _, _, Pedersen<Bn254G1>>(
        Stage1ProverInput::new(config, &witness),
        &mut backend,
        &mut prover_transcript,
        &vc_setup,
    )?;

    assert_eq!(output.uniskip_output_claim_values.len(), 1);
    assert_eq!(
        output.remainder_output_claim_values.len(),
        SpartanOuterDimensions::rv64(config.log_t).variables().len()
    );
    assert_eq!(output.uniskip_committed_witness.round_coefficients.len(), 1);
    assert_eq!(
        output.remainder_committed_witness.round_coefficients.len(),
        config.log_t + 1
    );

    let trace_length = 1 << config.log_t;
    let proof = stage1_zk_proof(
        output.uniskip_proof.clone(),
        output.remainder_proof.clone(),
        trace_length,
    );
    let checked = jolt_verifier::CheckedInputs {
        public_io: common::jolt_device::JoltDevice::default(),
        zk: true,
        trace_length,
        ram_K: 16,
        entry_address: 0,
        preprocessing_digest: [0; 32],
        trusted_advice_commitment_present: false,
        vc_capacity: Some(Pedersen::<Bn254G1>::capacity(&vc_setup)),
    };
    let preprocessing = jolt_verifier::JoltVerifierPreprocessing::<
        jolt_openings::mock::MockCommitmentScheme<Fr>,
        Pedersen<Bn254G1>,
    >::new(
        empty_program_preprocessing(trace_length),
        [0; 32],
        (),
        Some(vc_setup),
    );
    let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage1-zk-test");
    let native_output = jolt_verifier::stages::stage1::verify(
        &checked,
        &preprocessing,
        &proof,
        &mut verifier_transcript,
    )?;
    let jolt_verifier::stages::stage1::Stage1Output::Zk(native_output) = native_output else {
        return Err("Stage 1 verifier did not return ZK output".into());
    };

    assert_eq!(native_output.public, output.public);
    assert_eq!(
        native_output.uniskip_output_claims.shape.output_claim_count,
        output.uniskip_output_claim_values.len()
    );
    assert_eq!(
        native_output
            .remainder_output_claims
            .shape
            .output_claim_count,
        output.remainder_output_claim_values.len()
    );
    assert_eq!(verifier_transcript.state(), prover_transcript.state());

    Ok(())
}

#[test]
#[cfg(feature = "field-inline")]
fn stage1_field_inline_prove_produces_verifier_compatible_sumchecks(
) -> Result<(), Box<dyn std::error::Error>> {
    let witness = SatisfyingStage1Witness;
    let field_inline_witness = Stage1FieldInlineWitness;
    let config = Stage1ProverConfig::new(4);
    let mut backend = CpuBackend::default();
    let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage1-field-inline-test");

    let output = prove_stage1_public::<Fr, _, _, _, _, Bn254G1>(
        Stage1ProverInput::new(config, &witness, &field_inline_witness),
        &mut backend,
        &mut prover_transcript,
    )?;
    let prover_verifier_output = output
        .verifier_output
        .clone()
        .ok_or("field-inline Stage 1 prover did not return verifier output")?;

    assert_eq!(
        output.field_inline_r1cs_input_claims.len(),
        FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS.len()
    );

    let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage1-field-inline-test");
    let tau = verifier_transcript.challenge_vector(config.log_t + 2);
    let uniskip_reduction = output.uniskip_proof.verify(
        &SumcheckClaim::new(1, SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE, Fr::from_u64(0)),
        CenteredIntegerDomain::new(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE),
        UNISKIP_ROUND_TRANSCRIPT_LABEL,
        &mut verifier_transcript,
    )?;
    assert_eq!(uniskip_reduction.value, output.uniskip_output_claim);

    verifier_transcript.append_labeled(b"opening_claim", &output.uniskip_output_claim);
    let remainder_batch = BatchedSumcheckVerifier::verify_compressed_boolean(
        &[SumcheckClaim::new(
            config.log_t + 1,
            SPARTAN_OUTER_REMAINDER_DEGREE,
            output.uniskip_output_claim,
        )],
        &output.remainder_proof,
        &mut verifier_transcript,
    )?;
    let mut r1cs_input_claims = output
        .r1cs_input_claims
        .iter()
        .map(|claim| claim.value)
        .collect::<Vec<_>>();
    r1cs_input_claims.extend(
        output
            .field_inline_r1cs_input_claims
            .iter()
            .map(|claim| claim.value),
    );
    let expected_remainder = JoltSpartanOuterRemainder::new(JoltSpartanOuterRemainderChallenges {
        tau: &tau,
        uniskip: uniskip_reduction.point[0],
        remainder: &remainder_batch.reduction.point,
    })?
    .expected_output_claim(&r1cs_input_claims)?
        * remainder_batch.batching_coefficients[0];
    assert_eq!(remainder_batch.reduction.value, expected_remainder);
    for opening_claim in &r1cs_input_claims {
        verifier_transcript.append_labeled(b"opening_claim", opening_claim);
    }
    assert_eq!(prover_transcript.state(), verifier_transcript.state());

    let stage_claims = stage1_claims_from_r1cs_inputs(
        output.uniskip_output_claim,
        &output.r1cs_input_claims,
        &output.field_inline_r1cs_input_claims,
    )?;
    let proof = field_inline_stage1_proof(output, stage_claims, 1 << config.log_t);
    let checked = jolt_verifier::CheckedInputs {
        public_io: common::jolt_device::JoltDevice::default(),
        zk: false,
        trace_length: 1 << config.log_t,
        ram_K: 16,
        entry_address: 0,
        preprocessing_digest: [0; 32],
        trusted_advice_commitment_present: false,
        vc_capacity: None,
        field_inline_bytecode_transcript: Vec::new(),
    };
    let preprocessing = jolt_verifier::JoltVerifierPreprocessing::<
        jolt_openings::mock::MockCommitmentScheme<Fr>,
        Pedersen<Bn254G1>,
    >::new(
        empty_program_preprocessing(1 << config.log_t),
        [0; 32],
        (),
        None,
    );
    let mut native_transcript = Blake2bTranscript::<Fr>::new(b"stage1-field-inline-test");
    let native_output = jolt_verifier::stages::stage1::verify(
        &checked,
        &preprocessing,
        &proof,
        &mut native_transcript,
    )?;
    let jolt_verifier::stages::stage1::Stage1Output::Clear(native_output) = native_output else {
        return Err("field-inline Stage 1 verifier did not return clear output".into());
    };
    assert_eq!(prover_verifier_output, native_output);
    assert_eq!(native_transcript.state(), prover_transcript.state());

    Ok(())
}

#[test]
#[cfg(all(feature = "zk", feature = "field-inline"))]
fn stage1_field_inline_committed_boundary_produces_native_verifier_output(
) -> Result<(), Box<dyn std::error::Error>> {
    let witness = SatisfyingStage1Witness;
    let field_inline_witness = Stage1FieldInlineWitness;
    let config = Stage1ProverConfig::new(4);
    let mut backend = CpuBackend::default();
    let vc_setup = pedersen_setup(64);
    let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage1-field-inline-zk-test");

    let output = super::prove::prove_committed_boundary::<Fr, _, _, _, _, Pedersen<Bn254G1>>(
        Stage1ProverInput::new(config, &witness, &field_inline_witness),
        &mut backend,
        &mut prover_transcript,
        &vc_setup,
    )?;

    assert_eq!(output.uniskip_output_claim_values.len(), 1);
    assert_eq!(
        output.remainder_output_claim_values.len(),
        SPARTAN_OUTER_R1CS_INPUTS.len() + FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS.len()
    );
    assert_eq!(
        output.remainder_committed_witness.round_coefficients.len(),
        config.log_t + 1
    );

    let trace_length = 1 << config.log_t;
    let proof = field_inline_stage1_zk_proof(
        output.uniskip_proof.clone(),
        output.remainder_proof.clone(),
        trace_length,
    );
    let checked = jolt_verifier::CheckedInputs {
        public_io: common::jolt_device::JoltDevice::default(),
        zk: true,
        trace_length,
        ram_K: 16,
        entry_address: 0,
        preprocessing_digest: [0; 32],
        trusted_advice_commitment_present: false,
        vc_capacity: Some(Pedersen::<Bn254G1>::capacity(&vc_setup)),
        field_inline_bytecode_transcript: Vec::new(),
    };
    let preprocessing = jolt_verifier::JoltVerifierPreprocessing::<
        jolt_openings::mock::MockCommitmentScheme<Fr>,
        Pedersen<Bn254G1>,
    >::new(
        empty_program_preprocessing(trace_length),
        [0; 32],
        (),
        Some(vc_setup),
    )
    .with_field_inline_bytecode(Vec::new());
    let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage1-field-inline-zk-test");
    let native_output = jolt_verifier::stages::stage1::verify(
        &checked,
        &preprocessing,
        &proof,
        &mut verifier_transcript,
    )?;
    let jolt_verifier::stages::stage1::Stage1Output::Zk(native_output) = native_output else {
        return Err("field-inline Stage 1 verifier did not return ZK output".into());
    };

    assert_eq!(native_output.public, output.public);
    assert_eq!(
        native_output
            .remainder_output_claims
            .shape
            .output_claim_count,
        output.remainder_output_claim_values.len()
    );
    assert_eq!(verifier_transcript.state(), prover_transcript.state());

    Ok(())
}

#[cfg(any(
    feature = "field-inline",
    all(feature = "zk", not(feature = "field-inline"))
))]
fn empty_program_preprocessing(
    trace_length: usize,
) -> jolt_program::preprocess::JoltProgramPreprocessing {
    jolt_program::preprocess::JoltProgramPreprocessing {
        bytecode: Default::default(),
        ram: Default::default(),
        memory_layout: Default::default(),
        max_padded_trace_length: trace_length,
    }
}

#[test]
fn stage1_output_rejects_missing_backend_values() -> Result<(), Box<dyn std::error::Error>> {
    let witness = Stage1Witness;
    let request = build_stage1_request::<Fr, _>(Stage1ProverConfig::new(2), &witness)?;
    let result = SumcheckResult::new(
        vec![
            SumcheckProofOutput::new(STAGE1_UNISKIP_SLOT, "uniskip"),
            SumcheckProofOutput::new(STAGE1_REMAINDER_SLOT, "remainder"),
        ],
        vec![SumcheckEvaluationOutput::new(
            STAGE1_UNISKIP_OUTPUT_SLOT,
            Fr::from_u64(11),
        )],
    );

    assert!(Stage1SumcheckOutput::from_backend_result(&request, result).is_err());
    Ok(())
}

#[test]
fn stage1_r1cs_inputs_assemble_verifier_spartan_outer_claims(
) -> Result<(), Box<dyn std::error::Error>> {
    let claims = r1cs_claims();
    let outer = spartan_outer_claims_from_r1cs_inputs(&claims)?;

    assert_eq!(
        outer.left_instruction_input,
        value_for(JoltVirtualPolynomial::LeftInstructionInput)
    );
    assert_eq!(outer.product, value_for(JoltVirtualPolynomial::Product));
    assert_eq!(
        outer.next_is_first_in_sequence,
        value_for(JoltVirtualPolynomial::NextIsFirstInSequence)
    );
    assert_eq!(
        outer.flags.add_operands,
        value_for(JoltVirtualPolynomial::OpFlags(
            jolt_riscv::CircuitFlags::AddOperands
        ))
    );
    assert_eq!(
        outer.flags.is_last_in_sequence,
        value_for(JoltVirtualPolynomial::OpFlags(
            jolt_riscv::CircuitFlags::IsLastInSequence
        ))
    );
    Ok(())
}

#[test]
#[cfg(not(feature = "field-inline"))]
fn stage1_r1cs_inputs_assemble_verifier_stage_claims() -> Result<(), Box<dyn std::error::Error>> {
    let claims = r1cs_claims();
    let stage_claims = super::output::stage1_claims_from_r1cs_inputs(Fr::from_u64(77), &claims)?;

    assert_eq!(stage_claims.uniskip_output_claim, Fr::from_u64(77));
    assert_eq!(
        stage_claims.outer.lookup_output,
        value_for(JoltVirtualPolynomial::LookupOutput)
    );
    Ok(())
}

#[test]
fn stage1_r1cs_claim_assembly_rejects_duplicate_inputs() {
    let duplicate = JoltVirtualPolynomial::LeftInstructionInput;
    let claims = vec![
        Stage1R1csInputClaim {
            variable: duplicate,
            slot: r1cs_input_slot(0),
            value: Fr::from_u64(1),
        },
        Stage1R1csInputClaim {
            variable: duplicate,
            slot: r1cs_input_slot(1),
            value: Fr::from_u64(2),
        },
    ];

    assert!(spartan_outer_claims_from_r1cs_inputs(&claims).is_err());
}

#[test]
#[cfg(feature = "field-inline")]
fn stage1_field_inline_evaluation_request_matches_verifier_opening_order(
) -> Result<(), Box<dyn std::error::Error>> {
    let witness = Stage1FieldInlineWitness;
    let config = Stage1ProverConfig::new(4);
    let request = build_stage1_field_inline_r1cs_evaluation_request(
        config,
        &witness,
        vec![
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
        ],
    )?;

    assert_eq!(
        request.evaluations.label,
        "stage1.field_inline.spartan_outer.r1cs_inputs"
    );
    assert_eq!(
        request.r1cs_inputs.len(),
        FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS.len()
    );
    assert_eq!(
        request.evaluations.views.len(),
        FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS.len()
    );
    for (index, input) in request.r1cs_inputs.iter().enumerate() {
        assert_eq!(
            input.variable,
            FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS[index]
        );
        assert_eq!(input.slot, field_inline_r1cs_input_slot(index));
        assert_eq!(
            input.view.oracle,
            OracleRef::virtual_polynomial(input.variable)
        );
        assert_eq!(input.view.retention, RetentionHint::ThroughBlindFold);
    }
    Ok(())
}

#[test]
#[cfg(feature = "field-inline")]
fn stage1_field_inline_r1cs_inputs_assemble_verifier_stage_claims(
) -> Result<(), Box<dyn std::error::Error>> {
    let base_claims = r1cs_claims();
    let field_claims = field_inline_r1cs_claims();
    let field_inline = field_inline_stage1_claims_from_r1cs_inputs(&field_claims)?;

    assert_eq!(
        field_inline.field_rs1_value,
        field_inline_value_for(FieldInlineVirtualPolynomial::FieldRs1Value)
    );
    assert_eq!(
        field_inline.field_inv_product,
        field_inline_value_for(FieldInlineVirtualPolynomial::FieldInvProduct)
    );
    assert_eq!(
        field_inline.flags.mul,
        field_inline_value_for(FieldInlineVirtualPolynomial::FieldOpFlag(
            FieldInlineOpFlag::Mul
        ))
    );

    let stage_claims = super::output::stage1_claims_from_r1cs_inputs(
        Fr::from_u64(77),
        &base_claims,
        &field_claims,
    )?;
    assert_eq!(stage_claims.uniskip_output_claim, Fr::from_u64(77));
    assert_eq!(
        stage_claims.field_inline.flags.load_imm,
        field_inline_value_for(FieldInlineVirtualPolynomial::FieldOpFlag(
            FieldInlineOpFlag::LoadImm
        ))
    );
    assert_eq!(
        stage_claims
            .spartan_outer_claims(&SpartanOuterDimensions::rv64(4))?
            .len(),
        SPARTAN_OUTER_R1CS_INPUTS.len() + FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS.len()
    );
    Ok(())
}

fn value_for(variable: JoltVirtualPolynomial) -> Fr {
    SPARTAN_OUTER_R1CS_INPUTS
        .iter()
        .position(|candidate| *candidate == variable)
        .map_or(Fr::from_u64(0), |index| Fr::from_u64(100 + index as u64))
}

fn r1cs_claims() -> Vec<Stage1R1csInputClaim<Fr>> {
    SPARTAN_OUTER_R1CS_INPUTS
        .iter()
        .copied()
        .enumerate()
        .map(|(index, variable)| Stage1R1csInputClaim {
            variable,
            slot: r1cs_input_slot(index),
            value: Fr::from_u64(100 + index as u64),
        })
        .collect()
}

#[cfg(feature = "field-inline")]
fn field_inline_value_for(variable: FieldInlineVirtualPolynomial) -> Fr {
    FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS
        .iter()
        .position(|candidate| *candidate == variable)
        .map_or(Fr::from_u64(0), |index| Fr::from_u64(200 + index as u64))
}

#[cfg(feature = "field-inline")]
fn field_inline_r1cs_claims() -> Vec<Stage1FieldInlineR1csInputClaim<Fr>> {
    FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS
        .iter()
        .copied()
        .enumerate()
        .map(|(index, variable)| Stage1FieldInlineR1csInputClaim {
            variable,
            slot: field_inline_r1cs_input_slot(index),
            value: Fr::from_u64(200 + index as u64),
        })
        .collect()
}

#[cfg(feature = "field-inline")]
fn field_inline_stage1_proof(
    output: Stage1SumcheckOutput<Fr, SumcheckProof<Fr, Bn254G1>>,
    stage1: jolt_verifier::stages::stage1::inputs::Stage1Claims<Fr>,
    trace_length: usize,
) -> jolt_verifier::proof::JoltProof<jolt_openings::mock::MockCommitmentScheme<Fr>, Pedersen<Bn254G1>>
{
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
    let empty = SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof::default()));
    let stages = jolt_verifier::proof::JoltStageProofs {
        stage1_uni_skip_first_round_proof: output.uniskip_proof,
        stage1_sumcheck_proof: output.remainder_proof,
        stage2_uni_skip_first_round_proof: empty.clone(),
        stage2_sumcheck_proof: empty.clone(),
        stage3_sumcheck_proof: empty.clone(),
        stage4_sumcheck_proof: empty.clone(),
        stage5_sumcheck_proof: empty.clone(),
        stage6_sumcheck_proof: empty.clone(),
        stage7_sumcheck_proof: empty,
    };
    let mut claims = jolt_verifier::compat::claims::empty_clear_opening_claims(trace_length);
    claims.stage1 = stage1;
    let poly = Polynomial::new(vec![Fr::from_u64(0)]);
    let mut transcript = Blake2bTranscript::<Fr>::new(b"stage1-field-inline-opening");
    let opening_proof = MockPcs::open(&poly, &[], Fr::from_u64(0), &(), None, &mut transcript);
    jolt_verifier::proof::JoltProof::<MockPcs, Pedersen<Bn254G1>>::new(
        commitments,
        stages,
        opening_proof,
        None,
        jolt_verifier::proof::JoltProofClaims::Clear(claims),
        trace_length,
        16,
        jolt_claims::protocols::jolt::JoltReadWriteConfig {
            ram_rw_phase1_num_rounds: 1,
            ram_rw_phase2_num_rounds: 1,
            registers_rw_phase1_num_rounds: 1,
            registers_rw_phase2_num_rounds: 1,
        },
        jolt_claims::protocols::jolt::JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
        },
        jolt_verifier::proof::TracePolynomialOrder::AddressMajor,
    )
}

#[cfg(all(feature = "zk", feature = "field-inline"))]
fn field_inline_stage1_zk_proof(
    uniskip_proof: SumcheckProof<Fr, Bn254G1>,
    remainder_proof: SumcheckProof<Fr, Bn254G1>,
    trace_length: usize,
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
    let empty = SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof::default()));
    let stages = jolt_verifier::proof::JoltStageProofs {
        stage1_uni_skip_first_round_proof: uniskip_proof,
        stage1_sumcheck_proof: remainder_proof,
        stage2_uni_skip_first_round_proof: empty.clone(),
        stage2_sumcheck_proof: empty.clone(),
        stage3_sumcheck_proof: empty.clone(),
        stage4_sumcheck_proof: empty.clone(),
        stage5_sumcheck_proof: empty.clone(),
        stage6_sumcheck_proof: empty.clone(),
        stage7_sumcheck_proof: empty,
    };
    let poly = jolt_poly::Polynomial::new(vec![Fr::from_u64(0)]);
    let mut transcript = Blake2bTranscript::<Fr>::new(b"stage1-field-inline-zk-opening");
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
        16,
        jolt_claims::protocols::jolt::JoltReadWriteConfig {
            ram_rw_phase1_num_rounds: 1,
            ram_rw_phase2_num_rounds: 1,
            registers_rw_phase1_num_rounds: 1,
            registers_rw_phase2_num_rounds: 1,
        },
        jolt_claims::protocols::jolt::JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
        },
        jolt_verifier::proof::TracePolynomialOrder::AddressMajor,
    )
}

#[cfg(all(feature = "zk", not(feature = "field-inline")))]
fn stage1_zk_proof(
    uniskip_proof: SumcheckProof<Fr, Bn254G1>,
    remainder_proof: SumcheckProof<Fr, Bn254G1>,
    trace_length: usize,
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
    let empty = SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof::default()));
    let stages = jolt_verifier::proof::JoltStageProofs {
        stage1_uni_skip_first_round_proof: uniskip_proof,
        stage1_sumcheck_proof: remainder_proof,
        stage2_uni_skip_first_round_proof: empty.clone(),
        stage2_sumcheck_proof: empty.clone(),
        stage3_sumcheck_proof: empty.clone(),
        stage4_sumcheck_proof: empty.clone(),
        stage5_sumcheck_proof: empty.clone(),
        stage6_sumcheck_proof: empty.clone(),
        stage7_sumcheck_proof: empty,
    };
    let poly = jolt_poly::Polynomial::new(vec![Fr::from_u64(0)]);
    let mut transcript = Blake2bTranscript::<Fr>::new(b"stage1-zk-opening");
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
        16,
        jolt_claims::protocols::jolt::JoltReadWriteConfig {
            ram_rw_phase1_num_rounds: 1,
            ram_rw_phase2_num_rounds: 1,
            registers_rw_phase1_num_rounds: 1,
            registers_rw_phase2_num_rounds: 1,
        },
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

#[derive(Clone, Copy, Debug)]
struct Stage1Witness;

impl WitnessProvider<Fr, JoltVmNamespace> for Stage1Witness {
    fn describe_oracle(
        &self,
        oracle: OracleRef<JoltVmNamespace>,
    ) -> Result<OracleDescriptor<JoltVmNamespace>, WitnessError> {
        let OracleKind::Virtual(_) = oracle.kind else {
            return Err(WitnessError::UnknownOracle {
                namespace: JoltVmNamespace::ID.name,
            });
        };
        Ok(OracleDescriptor::new(
            oracle,
            WitnessDimensions::new(16, 4),
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
        Ok(PolynomialView::owned(descriptor, vec![Fr::from_u64(0); 16]))
    }
}

#[derive(Clone, Copy, Debug)]
struct SatisfyingStage1Witness;

impl WitnessProvider<Fr, JoltVmNamespace> for SatisfyingStage1Witness {
    fn describe_oracle(
        &self,
        oracle: OracleRef<JoltVmNamespace>,
    ) -> Result<OracleDescriptor<JoltVmNamespace>, WitnessError> {
        let OracleKind::Virtual(_) = oracle.kind else {
            return Err(WitnessError::UnknownOracle {
                namespace: JoltVmNamespace::ID.name,
            });
        };
        Ok(OracleDescriptor::new(
            oracle,
            WitnessDimensions::new(16, 4),
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
        let OracleKind::Virtual(variable) = request.oracle().kind else {
            return Err(WitnessError::UnknownOracle {
                namespace: JoltVmNamespace::ID.name,
            });
        };
        Ok(PolynomialView::owned(
            descriptor,
            vec![satisfying_stage1_value(variable); 16],
        ))
    }
}

impl JoltVmSpartanOuterRows for SatisfyingStage1Witness {
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

#[cfg(feature = "field-inline")]
impl FieldInlineRegisterReadWriteRows<Fr> for Stage1FieldInlineWitness {
    fn field_inline_register_read_write_rows(
        &self,
    ) -> Result<Vec<FieldInlineRegisterReadWriteRow<Fr>>, WitnessError> {
        Ok(vec![FieldInlineRegisterReadWriteRow::default(); 16])
    }
}

fn satisfying_stage1_value(variable: JoltVirtualPolynomial) -> Fr {
    match variable {
        JoltVirtualPolynomial::NextUnexpandedPC => Fr::from_u64(4),
        _ => Fr::from_u64(0),
    }
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Copy, Debug)]
struct Stage1FieldInlineWitness;

#[cfg(feature = "field-inline")]
impl WitnessProvider<Fr, FieldInlineNamespace> for Stage1FieldInlineWitness {
    fn describe_oracle(
        &self,
        oracle: OracleRef<FieldInlineNamespace>,
    ) -> Result<OracleDescriptor<FieldInlineNamespace>, WitnessError> {
        let OracleKind::Virtual(_) = oracle.kind else {
            return Err(WitnessError::UnknownOracle {
                namespace: FieldInlineNamespace::ID.name,
            });
        };
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
            RetentionHint::ThroughBlindFold,
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

#[derive(Default)]
struct RecordingSumcheckBackend {
    labels: Vec<&'static str>,
    relations: Vec<BackendRelationId>,
    evaluation_labels: Vec<&'static str>,
    evaluation_points: Vec<Vec<Fr>>,
}

impl Backend for RecordingSumcheckBackend {
    fn name(&self) -> &'static str {
        "recording"
    }
}

impl SumcheckBackend<Fr, JoltVmNamespace> for RecordingSumcheckBackend {
    type Proof = &'static str;

    fn prove_sumcheck<W>(
        &mut self,
        request: &SumcheckRequest<JoltVmNamespace>,
        _witness: &W,
    ) -> Result<SumcheckResult<Fr, Self::Proof>, BackendError>
    where
        W: WitnessProvider<Fr, JoltVmNamespace>,
    {
        self.labels.push(request.label);
        self.relations
            .extend(request.instances.iter().map(|instance| instance.relation));

        let proofs = vec![
            SumcheckProofOutput::new(STAGE1_UNISKIP_SLOT, "uniskip"),
            SumcheckProofOutput::new(STAGE1_REMAINDER_SLOT, "remainder"),
        ];
        let mut evaluations = vec![
            SumcheckEvaluationOutput::new(STAGE1_UNISKIP_OUTPUT_SLOT, Fr::from_u64(11)),
            SumcheckEvaluationOutput::new(STAGE1_REMAINDER_OUTPUT_SLOT, Fr::from_u64(17)),
        ];
        evaluations.extend((0..SPARTAN_OUTER_R1CS_INPUTS.len()).map(|index| {
            SumcheckEvaluationOutput::new(r1cs_input_slot(index), Fr::from_u64(100 + index as u64))
        }));

        Ok(SumcheckResult::new(proofs, evaluations))
    }

    fn evaluate_sumcheck_views<W>(
        &mut self,
        request: &SumcheckEvaluationRequest<Fr, JoltVmNamespace>,
        _witness: &W,
    ) -> Result<Vec<SumcheckEvaluationOutput<Fr>>, BackendError>
    where
        W: WitnessProvider<Fr, JoltVmNamespace>,
    {
        self.evaluation_labels.push(request.label);
        self.evaluation_points
            .extend(request.views.iter().map(|view| view.point.clone()));
        Ok(request
            .views
            .iter()
            .enumerate()
            .map(|(index, view)| {
                SumcheckEvaluationOutput::new(view.slot, Fr::from_u64(200 + index as u64))
            })
            .collect())
    }
}

#[cfg(not(feature = "field-inline"))]
struct EvaluationBackend;

#[cfg(not(feature = "field-inline"))]
impl Backend for EvaluationBackend {
    fn name(&self) -> &'static str {
        "evaluation"
    }
}

#[cfg(not(feature = "field-inline"))]
impl SumcheckBackend<Fr, JoltVmNamespace> for EvaluationBackend {
    type Proof = ();

    fn materialize_sumcheck_views<W>(
        &mut self,
        request: &SumcheckMaterializationRequest<JoltVmNamespace>,
        witness: &W,
    ) -> Result<Vec<SumcheckMaterializationOutput<Fr>>, BackendError>
    where
        W: WitnessProvider<Fr, JoltVmNamespace>,
    {
        request
            .views
            .iter()
            .map(|view| {
                let materialized = witness.oracle_view(OracleViewRequest::new(view.requirement))?;
                let Some(values) = materialized.as_slice() else {
                    return Err(BackendError::InvalidRequest {
                        backend: self.name(),
                        task: "test materialization",
                        reason: "deferred test view".to_owned(),
                    });
                };
                Ok(SumcheckMaterializationOutput::new(
                    view.slot,
                    values.to_vec(),
                ))
            })
            .collect()
    }

    fn evaluate_sumcheck_views<W>(
        &mut self,
        request: &SumcheckEvaluationRequest<Fr, JoltVmNamespace>,
        witness: &W,
    ) -> Result<Vec<SumcheckEvaluationOutput<Fr>>, BackendError>
    where
        W: WitnessProvider<Fr, JoltVmNamespace>,
    {
        request
            .views
            .iter()
            .map(|view| {
                let materialized = witness.oracle_view(OracleViewRequest::new(view.requirement))?;
                let Some(values) = materialized.as_slice() else {
                    return Err(BackendError::InvalidRequest {
                        backend: self.name(),
                        task: "test evaluation",
                        reason: "deferred test view".to_owned(),
                    });
                };
                Ok(SumcheckEvaluationOutput::new(
                    view.slot,
                    values.evaluate(&view.point),
                ))
            })
            .collect()
    }

    fn evaluate_sumcheck_linear_products(
        &mut self,
        request: &SumcheckLinearProductRequest<Fr>,
    ) -> Result<Vec<SumcheckLinearProductOutput<Fr>>, BackendError> {
        request
            .queries
            .iter()
            .map(|query| {
                let witness_evaluations = request
                    .witness_polynomials
                    .iter()
                    .map(|polynomial| polynomial.evaluate(&query.point))
                    .collect::<Vec<_>>();
                let left = test_weighted_linear_rows(
                    request.left_rows,
                    &query.row_weights,
                    request.input_columns,
                    request.constant_column,
                    &witness_evaluations,
                )?;
                let right = test_weighted_linear_rows(
                    request.right_rows,
                    &query.row_weights,
                    request.input_columns,
                    request.constant_column,
                    &witness_evaluations,
                )?;
                Ok(SumcheckLinearProductOutput::new(
                    query.slot,
                    query.scale * left * right,
                ))
            })
            .collect()
    }

    fn evaluate_sumcheck_prefix_product_sums(
        &mut self,
        request: &SumcheckPrefixProductSumRequest<Fr>,
    ) -> Result<Vec<SumcheckLinearProductOutput<Fr>>, BackendError> {
        let Some(first) = request.witness_polynomials.first() else {
            return Err(BackendError::InvalidRequest {
                backend: self.name(),
                task: "test prefix product sum",
                reason: "missing witness polynomials".to_owned(),
            });
        };
        let log_rows = first.len().trailing_zeros() as usize;
        let total_vars = log_rows + 1;
        request
            .queries
            .iter()
            .map(|query| {
                let suffix_count = 1usize << query.suffix_vars;
                let mut total = Fr::from_u64(0);
                for suffix_index in 0..suffix_count {
                    let value = |position: usize| {
                        if position < query.fixed_prefix.len() {
                            query.fixed_prefix[position]
                        } else {
                            let suffix_position = position - query.fixed_prefix.len();
                            let shift = query.suffix_vars - suffix_position - 1;
                            Fr::from_bool(((suffix_index >> shift) & 1) == 1)
                        }
                    };
                    let stream = value(0);
                    let row_weights = test_blend_row_weights(
                        stream,
                        &query.row_weights_at_zero,
                        &query.row_weights_at_one,
                    );
                    let cycle_point = (1..total_vars).rev().map(value).collect::<Vec<_>>();
                    let witness_evaluations = request
                        .witness_polynomials
                        .iter()
                        .map(|polynomial| polynomial.evaluate(&cycle_point))
                        .collect::<Vec<_>>();
                    let left = test_weighted_linear_rows(
                        request.left_rows,
                        &row_weights,
                        request.input_columns,
                        request.constant_column,
                        &witness_evaluations,
                    )?;
                    let right = test_weighted_linear_rows(
                        request.right_rows,
                        &row_weights,
                        request.input_columns,
                        request.constant_column,
                        &witness_evaluations,
                    )?;
                    let eq = (0..total_vars)
                        .map(|position| {
                            let challenge = query.eq_point[total_vars - position - 1];
                            let point = value(position);
                            challenge * point
                                + (Fr::from_u64(1) - challenge) * (Fr::from_u64(1) - point)
                        })
                        .product::<Fr>();
                    total += query.scale * eq * left * right;
                }
                Ok(SumcheckLinearProductOutput::new(query.slot, total))
            })
            .collect()
    }

    fn materialize_sumcheck_spartan_outer_remainder_state(
        &mut self,
        request: &SumcheckSpartanOuterRemainderStateRequest<Fr>,
    ) -> Result<SumcheckSpartanOuterRemainderState<Fr>, BackendError> {
        let Some(first) = request.witness_polynomials.first() else {
            return Err(BackendError::InvalidRequest {
                backend: self.name(),
                task: "test remainder state",
                reason: "missing witness polynomials".to_owned(),
            });
        };
        let log_rows = first.len().trailing_zeros() as usize;
        if request.eq_point.len() != log_rows + 1 {
            return Err(BackendError::InvalidRequest {
                backend: self.name(),
                task: "test remainder state",
                reason: format!(
                    "equality point has {} variables, expected {}",
                    request.eq_point.len(),
                    log_rows + 1
                ),
            });
        }
        let stream_eq = test_eq_factor(request.eq_point[log_rows], request.stream_challenge);
        let row_weights = test_blend_row_weights(
            request.stream_challenge,
            &request.row_weights_at_zero,
            &request.row_weights_at_one,
        );
        let mut left = Vec::with_capacity(first.len());
        let mut right = Vec::with_capacity(first.len());
        for row_index in 0..first.len() {
            left.push(test_weighted_linear_rows_at_row(
                request.left_rows,
                &row_weights,
                request.input_columns,
                request.constant_column,
                request.witness_polynomials,
                row_index,
            )?);
            right.push(test_weighted_linear_rows_at_row(
                request.right_rows,
                &row_weights,
                request.input_columns,
                request.constant_column,
                request.witness_polynomials,
                row_index,
            )?);
        }
        Ok(SumcheckSpartanOuterRemainderState::new(
            request.label,
            request.eq_point[..log_rows].to_vec(),
            left,
            right,
            first.len(),
            request.scale * stream_eq,
        ))
    }

    fn evaluate_sumcheck_spartan_outer_remainder_round(
        &mut self,
        state: &SumcheckSpartanOuterRemainderState<Fr>,
    ) -> Result<SumcheckSpartanOuterRemainderRound<Fr>, BackendError> {
        let active_log = state.active_len.trailing_zeros() as usize;
        let remaining_vars = active_log - 1;
        let eq_tensor = TensorEqTable::<Fr>::new(&state.eq_point[..remaining_vars]);
        let (q_at_zero, q_at_infinity) = test_sum_bound_pair_endpoints(
            &eq_tensor,
            &state.left[..state.active_len],
            &state.right[..state.active_len],
        );
        Ok(SumcheckSpartanOuterRemainderRound::new(
            q_at_zero,
            q_at_infinity,
        ))
    }

    fn bind_sumcheck_spartan_outer_remainder_state(
        &mut self,
        state: &mut SumcheckSpartanOuterRemainderState<Fr>,
        challenge: Fr,
    ) -> Result<(), BackendError> {
        let active_log = state.active_len.trailing_zeros() as usize;
        state.scale *= test_eq_factor(state.eq_point[active_log - 1], challenge);
        state.active_len = test_bind_low_variable(&mut state.left, state.active_len, challenge);
        let right_len = test_bind_low_variable(&mut state.right, state.active_len * 2, challenge);
        assert_eq!(state.active_len, right_len);
        Ok(())
    }
}

#[cfg(not(feature = "field-inline"))]
fn test_blend_row_weights(selector: Fr, zero: &[Fr], one: &[Fr]) -> Vec<Fr> {
    if selector == Fr::from_u64(0) {
        return zero.to_vec();
    }
    if selector == Fr::from_u64(1) {
        return one.to_vec();
    }
    zero.iter()
        .zip(one)
        .map(|(&at_zero, &at_one)| at_zero + selector * (at_one - at_zero))
        .collect()
}

#[cfg(not(feature = "field-inline"))]
fn test_weighted_linear_rows(
    rows: &[Vec<(usize, Fr)>],
    row_weights: &[Fr],
    input_columns: &[usize],
    constant_column: usize,
    witness_evaluations: &[Fr],
) -> Result<Fr, BackendError> {
    rows.iter()
        .zip(row_weights)
        .map(|(row, &weight)| {
            if weight == Fr::from_u64(0) {
                Ok(Fr::from_u64(0))
            } else {
                test_linear_row(row, input_columns, constant_column, witness_evaluations)
                    .map(|value| weight * value)
            }
        })
        .sum()
}

#[cfg(not(feature = "field-inline"))]
fn test_weighted_linear_rows_at_row(
    rows: &[Vec<(usize, Fr)>],
    row_weights: &[Fr],
    input_columns: &[usize],
    constant_column: usize,
    witness_polynomials: &[Vec<Fr>],
    row_index: usize,
) -> Result<Fr, BackendError> {
    rows.iter()
        .zip(row_weights)
        .map(|(row, &weight)| {
            if weight == Fr::from_u64(0) {
                Ok(Fr::from_u64(0))
            } else {
                test_linear_row_at_row(
                    row,
                    input_columns,
                    constant_column,
                    witness_polynomials,
                    row_index,
                )
                .map(|value| weight * value)
            }
        })
        .sum()
}

#[cfg(not(feature = "field-inline"))]
fn test_linear_row(
    row: &[(usize, Fr)],
    input_columns: &[usize],
    constant_column: usize,
    witness_evaluations: &[Fr],
) -> Result<Fr, BackendError> {
    row.iter()
        .map(|&(column, coefficient)| {
            if column == constant_column {
                return Ok(coefficient);
            }
            let Some(index) = input_columns
                .iter()
                .position(|candidate| *candidate == column)
            else {
                return Err(BackendError::InvalidRequest {
                    backend: "evaluation",
                    task: "test linear product",
                    reason: format!("unsupported column {column}"),
                });
            };
            Ok(coefficient * witness_evaluations[index])
        })
        .sum()
}

#[cfg(not(feature = "field-inline"))]
fn test_linear_row_at_row(
    row: &[(usize, Fr)],
    input_columns: &[usize],
    constant_column: usize,
    witness_polynomials: &[Vec<Fr>],
    row_index: usize,
) -> Result<Fr, BackendError> {
    row.iter()
        .map(|&(column, coefficient)| {
            if column == constant_column {
                return Ok(coefficient);
            }
            let Some(index) = input_columns
                .iter()
                .position(|candidate| *candidate == column)
            else {
                return Err(BackendError::InvalidRequest {
                    backend: "evaluation",
                    task: "test linear product row",
                    reason: format!("unsupported column {column}"),
                });
            };
            Ok(coefficient * witness_polynomials[index][row_index])
        })
        .sum()
}

#[cfg(not(feature = "field-inline"))]
fn test_eq_factor(challenge: Fr, value: Fr) -> Fr {
    challenge * value + (Fr::from_u64(1) - challenge) * (Fr::from_u64(1) - value)
}

#[cfg(not(feature = "field-inline"))]
fn test_sum_bound_pair_endpoints(
    eq_tensor: &TensorEqTable<Fr>,
    left: &[Fr],
    right: &[Fr],
) -> (Fr, Fr) {
    let [q_at_zero, q_at_infinity] = eq_tensor.par_fold_out_in(
        || [Fr::from_u64(0); 2],
        |inner, row_index, _x_in, e_in| {
            let low_index = 2 * row_index;
            let high_index = low_index + 1;
            inner[0] += e_in * left[low_index] * right[low_index];
            inner[1] += e_in
                * (left[high_index] - left[low_index])
                * (right[high_index] - right[low_index]);
        },
        |_x_out, e_out, inner| [e_out * inner[0], e_out * inner[1]],
        |mut left, right| {
            left[0] += right[0];
            left[1] += right[1];
            left
        },
    );
    (q_at_zero, q_at_infinity)
}

#[cfg(not(feature = "field-inline"))]
fn test_bind_low_variable(values: &mut [Fr], active_len: usize, point: Fr) -> usize {
    let next_len = active_len / 2;
    let one_minus_point = Fr::from_u64(1) - point;
    for index in 0..next_len {
        values[index] = one_minus_point * values[2 * index] + point * values[2 * index + 1];
    }
    next_len
}
