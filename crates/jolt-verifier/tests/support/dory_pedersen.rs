#![expect(
    clippy::panic,
    reason = "test fixture construction should fail loudly if the synthetic proof is malformed"
)]

use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
use jolt_claims::protocols::jolt::{
    formulas::{
        ram,
        spartan::{
            product_uniskip_opening, SpartanOuterDimensions, SpartanOuterLinearForms,
            SpartanOuterRemainderPlan,
        },
    },
    JoltOneHotConfig, JoltReadWriteConfig, JoltStageId, JoltVirtualPolynomial,
};
use jolt_crypto::{Bn254G1, DeriveSetup, Pedersen, PedersenSetup, VectorCommitment};
use jolt_dory::{DoryCommitment, DoryProof, DoryScheme, DoryVerifierSetup};
use jolt_field::{Fr, FromPrimitiveInt, Invertible};
use jolt_openings::{CommitmentScheme, ZkOpeningScheme};
use jolt_poly::{sparse_segments_mle_msb, CompressedPoly, Polynomial, UnivariatePoly};
use jolt_program::preprocess::{
    BytecodePreprocessing, JoltProgramPreprocessing, PublicIoMemory, RAMPreprocessing,
};
use jolt_r1cs::constraints::rv64::{const_column, input_column, rv64_eq_constraints};
use jolt_sumcheck::{
    append_sumcheck_claim, BatchedSumcheckVerifier, ClearProof, ClearSumcheckProof,
    CommittedOutputClaims, CommittedRound, CommittedSumcheckProof, CompressedSumcheckProof,
    LabeledRoundPoly, SumcheckClaim, SumcheckProof, SumcheckVerifier,
    SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, LabelWithCount, Transcript, U64Word};
use jolt_verifier::{
    compat::claims::attach_opening_claims,
    proof::{JoltProofClaims, JoltStageProofs, TracePolynomialOrder, TransparentProofClaims},
    stages::{stage1, stage2},
    verify, JoltProof, JoltVerifierPreprocessing, VerifierError,
};

const DORY_NUM_VARS: usize = 4;
const VC_CAPACITY: usize = 4;
const SYNTHETIC_RAM_K: usize = 32;

pub type TestVectorCommitment = Pedersen<Bn254G1>;
pub type TestProof = JoltProof<DoryScheme, TestVectorCommitment, ()>;
pub type TestPreprocessing = JoltVerifierPreprocessing<DoryScheme, TestVectorCommitment>;

pub struct DoryPedersenVerifierCase {
    pub preprocessing: TestPreprocessing,
    pub public_io: JoltDevice,
    pub proof: TestProof,
    pub zk: bool,
    pub trusted_advice_commitment: Option<DoryCommitment>,
}

impl DoryPedersenVerifierCase {
    pub fn verify(&self) -> Result<(), VerifierError> {
        verify::<Fr, DoryScheme, TestVectorCommitment, Blake2bTranscript, ()>(
            &self.preprocessing,
            &self.public_io,
            &self.proof,
            self.trusted_advice_commitment.as_ref(),
            self.zk,
        )
    }
}

pub fn standard_case() -> DoryPedersenVerifierCase {
    let artifacts = dory_artifacts(false);
    let preprocessing = preprocessing(artifacts.pcs_setup.clone(), None);
    let public_io = public_io();
    let mut proof = proof_with_payload(false, None, &artifacts);
    let opening_claims = standard_opening_claims(&preprocessing, &public_io, &proof);
    assert!(attach_opening_claims(&mut proof, opening_claims).is_ok());

    DoryPedersenVerifierCase {
        preprocessing,
        public_io,
        proof,
        zk: false,
        trusted_advice_commitment: None,
    }
}

pub fn zk_case() -> DoryPedersenVerifierCase {
    let artifacts = dory_artifacts(true);
    DoryPedersenVerifierCase {
        preprocessing: preprocessing(
            artifacts.pcs_setup.clone(),
            Some(artifacts.vc_setup.clone()),
        ),
        public_io: public_io(),
        proof: proof_with_payload(true, Some(()), &artifacts),
        zk: true,
        trusted_advice_commitment: None,
    }
}

pub fn preprocessing(
    pcs_setup: DoryVerifierSetup,
    vc_setup: Option<PedersenSetup<Bn254G1>>,
) -> TestPreprocessing {
    let memory_layout = memory_layout();
    JoltVerifierPreprocessing::new(
        JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing::default(),
            ram: RAMPreprocessing::default(),
            memory_layout,
            max_padded_trace_length: 16,
        },
        [7; 32],
        pcs_setup,
        vc_setup,
    )
}

pub fn public_io() -> JoltDevice {
    JoltDevice {
        memory_layout: memory_layout(),
        inputs: vec![1, 2, 3],
        outputs: vec![4],
        ..JoltDevice::default()
    }
}

pub fn proof_with_payload(
    is_zk: bool,
    blindfold_proof: Option<()>,
    artifacts: &DoryArtifacts,
) -> TestProof {
    JoltProof::new(
        vec![artifacts.commitment.clone()],
        stage_proofs(is_zk, &artifacts.vc_setup),
        artifacts.opening_proof.clone(),
        None,
        proof_claims(is_zk, blindfold_proof),
        1,
        SYNTHETIC_RAM_K,
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
        TracePolynomialOrder::CycleMajor,
    )
}

fn proof_claims(is_zk: bool, blindfold_proof: Option<()>) -> JoltProofClaims<Fr, ()> {
    if is_zk {
        return JoltProofClaims::Zk {
            blindfold_proof: blindfold_proof.unwrap_or(()),
        };
    }

    JoltProofClaims::Transparent(empty_transparent_claims())
}

fn empty_transparent_claims() -> TransparentProofClaims<Fr> {
    let zero = Fr::default();

    TransparentProofClaims {
        stage1: stage1::inputs::Stage1Claims {
            uniskip_output_claim: zero,
            outer: empty_spartan_outer_claims(),
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
    }
}

fn empty_spartan_outer_claims() -> stage1::inputs::SpartanOuterClaims<Fr> {
    let zero = Fr::default();

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

#[derive(Clone)]
pub struct DoryArtifacts {
    pub pcs_setup: DoryVerifierSetup,
    pub vc_setup: PedersenSetup<Bn254G1>,
    pub commitment: DoryCommitment,
    pub opening_proof: DoryProof,
}

fn dory_artifacts(is_zk: bool) -> DoryArtifacts {
    let prover_setup = DoryScheme::setup_prover(DORY_NUM_VARS);
    let pcs_setup = DoryScheme::verifier_setup(&prover_setup);
    let vc_setup = PedersenSetup::<Bn254G1>::derive(&prover_setup, VC_CAPACITY);
    let poly = Polynomial::new(vec![Fr::from_u64(2), Fr::from_u64(7)]);
    let point = vec![Fr::from_u64(3)];
    let eval = poly.evaluate(&point);
    let mut transcript = Blake2bTranscript::new(b"jolt-verifier-dory-pedersen-test");

    let (commitment, opening_proof) = if is_zk {
        let (commitment, hint) = DoryScheme::commit_zk(poly.evaluations(), &prover_setup);
        let (proof, _, _) =
            DoryScheme::open_zk(&poly, &point, eval, &prover_setup, hint, &mut transcript);
        (commitment, proof)
    } else {
        let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);
        let proof = DoryScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut transcript,
        );
        (commitment, proof)
    };

    DoryArtifacts {
        pcs_setup,
        vc_setup,
        commitment,
        opening_proof,
    }
}

fn memory_layout() -> MemoryLayout {
    MemoryLayout::new(&MemoryConfig {
        program_size: Some(8),
        max_trusted_advice_size: 0,
        max_untrusted_advice_size: 0,
        max_input_size: 8,
        max_output_size: 8,
        stack_size: 0,
        heap_size: 8,
    })
}

fn stage_proofs(
    is_zk: bool,
    vc_setup: &PedersenSetup<Bn254G1>,
) -> JoltStageProofs<Fr, TestVectorCommitment> {
    JoltStageProofs {
        stage1_uni_skip_first_round_proof: uniskip_proof(is_zk, vc_setup),
        stage1_sumcheck_proof: sumcheck_proof_with_rounds(is_zk, vc_setup, 1),
        stage2_uni_skip_first_round_proof: uniskip_proof(is_zk, vc_setup),
        stage2_sumcheck_proof: sumcheck_proof_with_rounds(
            is_zk,
            vc_setup,
            SYNTHETIC_RAM_K.ilog2() as usize,
        ),
        stage3_sumcheck_proof: sumcheck_proof_with_rounds(is_zk, vc_setup, 1),
        stage4_sumcheck_proof: sumcheck_proof_with_rounds(is_zk, vc_setup, 1),
        stage5_sumcheck_proof: sumcheck_proof_with_rounds(is_zk, vc_setup, 1),
        stage6_sumcheck_proof: sumcheck_proof_with_rounds(is_zk, vc_setup, 1),
        stage7_sumcheck_proof: sumcheck_proof_with_rounds(is_zk, vc_setup, 1),
    }
}

fn uniskip_proof(is_zk: bool, vc_setup: &PedersenSetup<Bn254G1>) -> SumcheckProof<Fr, Bn254G1> {
    if is_zk {
        SumcheckProof::Committed(committed_sumcheck_proof(vc_setup))
    } else {
        SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof {
            round_polynomials: vec![UnivariatePoly::zero()],
        }))
    }
}

fn sumcheck_proof_with_rounds(
    is_zk: bool,
    vc_setup: &PedersenSetup<Bn254G1>,
    rounds: usize,
) -> SumcheckProof<Fr, Bn254G1> {
    if is_zk {
        SumcheckProof::Committed(committed_sumcheck_proof(vc_setup))
    } else {
        SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof {
            round_polynomials: (0..rounds)
                .map(|_| CompressedPoly::new(vec![Fr::from_u64(0)]))
                .collect(),
        }))
    }
}

fn committed_sumcheck_proof(vc_setup: &PedersenSetup<Bn254G1>) -> CommittedSumcheckProof<Bn254G1> {
    CommittedSumcheckProof {
        rounds: vec![CommittedRound {
            commitment: pedersen_commit(vc_setup, &[1, 2, 3], 4),
            degree: 2,
        }],
        output_claims: CommittedOutputClaims {
            commitments: vec![pedersen_commit(vc_setup, &[5, 6], 7)],
        },
    }
}

fn pedersen_commit(setup: &PedersenSetup<Bn254G1>, values: &[u64], blinding: u64) -> Bn254G1 {
    let values: Vec<_> = values.iter().copied().map(Fr::from_u64).collect();
    Pedersen::commit(setup, &values, &Fr::from_u64(blinding))
}

fn standard_opening_claims(
    preprocessing: &TestPreprocessing,
    public_io: &JoltDevice,
    proof: &TestProof,
) -> Vec<(jolt_claims::protocols::jolt::JoltOpeningId, Fr)> {
    let mut transcript = transcript_after_commitments(preprocessing, public_io, proof);
    let log_t = proof.trace_length.ilog2() as usize;
    let dimensions = SpartanOuterDimensions::rv64(log_t);
    let _tau = transcript.challenge_vector(log_t + 2);

    let uniskip_challenge = verify_fixture_uniskip(proof, &mut transcript);
    transcript.append_labeled(b"opening_claim", &Fr::from_u64(0));
    append_sumcheck_claim(&mut transcript, &Fr::from_u64(0));
    let _batching_coefficient = transcript.challenge_scalar();
    let remainder_challenges = verify_fixture_remainder(log_t, proof, &mut transcript);
    let variable_openings =
        zeroing_spartan_outer_openings(&dimensions, uniskip_challenge, &remainder_challenges);

    let mut claims = vec![(
        spartan_outer_opening(JoltVirtualPolynomial::UnivariateSkip),
        Fr::from_u64(0),
    )];
    for (variable, opening_claim) in dimensions
        .variables()
        .iter()
        .copied()
        .zip(variable_openings)
    {
        claims.push((spartan_outer_opening(variable), opening_claim));
        transcript.append_labeled(b"opening_claim", &opening_claim);
    }
    claims.push((product_uniskip_opening(), Fr::from_u64(0)));
    let ram_output_check_claim =
        zeroing_stage2_output_check_claim(proof, public_io, &mut transcript);
    claims.push((
        ram::output_check_output_openings()[0],
        ram_output_check_claim,
    ));

    claims
}

fn zeroing_stage2_output_check_claim(
    proof: &TestProof,
    public_io: &JoltDevice,
    transcript: &mut Blake2bTranscript,
) -> Fr {
    let log_t = proof.trace_length.ilog2() as usize;
    let log_k = proof.ram_K.ilog2() as usize;
    let _tau_high = transcript.challenge();
    let _product_uniskip_challenge = verify_fixture_stage2_uniskip(proof, transcript);
    transcript.append_labeled(b"opening_claim", &Fr::from_u64(0));
    let _ram_read_write_gamma = transcript.challenge_scalar();
    let _instruction_gamma = transcript.challenge_scalar();
    let _output_address_challenges = (0..log_k)
        .map(|_| transcript.challenge())
        .collect::<Vec<_>>();

    let SumcheckProof::Clear(ClearProof::Compressed(proof)) = &proof.stages.stage2_sumcheck_proof
    else {
        panic!("standard fixture must use a clear compressed Stage 2 proof");
    };
    let reduction = BatchedSumcheckVerifier::verify_compressed(
        &[
            SumcheckClaim::new(log_t + log_k, 3, Fr::from_u64(0)),
            SumcheckClaim::new(log_t, 3, Fr::from_u64(0)),
            SumcheckClaim::new(log_t, 2, Fr::from_u64(0)),
            SumcheckClaim::new(log_t + log_k, 2, Fr::from_u64(0)),
            SumcheckClaim::new(log_t + log_k, 3, Fr::from_u64(0)),
        ],
        proof,
        transcript,
    )
    .unwrap_or_else(|error| panic!("fixture Stage 2 batch proof must verify: {error}"));

    let mut r_address = reduction.reduction.point.as_slice().to_vec();
    r_address.reverse();
    let public_memory = PublicIoMemory::new(public_io)
        .unwrap_or_else(|error| panic!("fixture public IO memory should materialize: {error}"));
    let io_num_vars = public_memory.io_num_vars();
    let (r_hi, r_lo) = r_address.split_at(log_k - io_num_vars);
    let hi_scale = r_hi.iter().fold(Fr::from_u64(1), |acc, challenge| {
        acc * (Fr::from_u64(1) - *challenge)
    });
    hi_scale
        * sparse_segments_mle_msb(
            public_memory
                .segments
                .iter()
                .map(|segment| (segment.start_index, segment.words.as_slice())),
            r_lo,
        )
}

fn verify_fixture_stage2_uniskip(proof: &TestProof, transcript: &mut Blake2bTranscript) -> Fr {
    let SumcheckProof::Clear(ClearProof::Full(proof)) =
        &proof.stages.stage2_uni_skip_first_round_proof
    else {
        panic!("standard fixture must use a clear full Stage 2 uni-skip proof");
    };

    let reduction = SumcheckVerifier::verify(
        &SumcheckClaim::new(1, 6, Fr::from_u64(0)),
        &proof
            .round_polynomials
            .iter()
            .map(LabeledRoundPoly::uniskip)
            .collect::<Vec<_>>(),
        jolt_sumcheck::CenteredIntegerDomain::new(3),
        transcript,
    )
    .unwrap_or_else(|error| panic!("fixture Stage 2 uni-skip proof must verify: {error}"));

    reduction.point[0]
}

fn verify_fixture_uniskip(proof: &TestProof, transcript: &mut Blake2bTranscript) -> Fr {
    let SumcheckProof::Clear(ClearProof::Full(proof)) =
        &proof.stages.stage1_uni_skip_first_round_proof
    else {
        panic!("standard fixture must use a clear full Stage 1 uni-skip proof");
    };

    let reduction = SumcheckVerifier::verify(
        &SumcheckClaim::new(1, 27, Fr::from_u64(0)),
        &proof
            .round_polynomials
            .iter()
            .map(LabeledRoundPoly::uniskip)
            .collect::<Vec<_>>(),
        jolt_sumcheck::CenteredIntegerDomain::new(10),
        transcript,
    )
    .unwrap_or_else(|error| panic!("fixture Stage 1 uni-skip proof must verify: {error}"));

    reduction.point[0]
}

fn verify_fixture_remainder(
    log_t: usize,
    proof: &TestProof,
    transcript: &mut Blake2bTranscript,
) -> Vec<Fr> {
    let SumcheckProof::Clear(ClearProof::Compressed(proof)) = &proof.stages.stage1_sumcheck_proof
    else {
        panic!("standard fixture must use a clear compressed Stage 1 remainder proof");
    };

    let reduction = SumcheckVerifier::verify_compressed(
        &SumcheckClaim::new(1 + log_t, 3, Fr::from_u64(0)),
        proof,
        jolt_sumcheck::BooleanHypercube,
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        transcript,
    )
    .unwrap_or_else(|error| panic!("fixture Stage 1 remainder proof must verify: {error}"));

    reduction.point.into_vec()
}

fn zeroing_spartan_outer_openings(
    dimensions: &SpartanOuterDimensions,
    uniskip_challenge: Fr,
    remainder_challenges: &[Fr],
) -> Vec<Fr> {
    let plan = SpartanOuterRemainderPlan::from_dimensions(dimensions);
    let row_weights = plan
        .row_weights(uniskip_challenge, remainder_challenges[0])
        .unwrap_or_else(|error| panic!("fixture row weights must be valid: {error}"));
    let columns: Vec<_> = plan
        .r1cs_input_indices()
        .unwrap_or_else(|error| panic!("fixture R1CS input map must be valid: {error}"))
        .into_iter()
        .map(|index| {
            input_column(index)
                .unwrap_or_else(|| panic!("fixture R1CS input {index} must have a column"))
        })
        .collect();

    let matrices = rv64_eq_constraints::<Fr>();
    let weighted = matrices
        .weighted_columns(&row_weights, &columns)
        .unwrap_or_else(|error| panic!("fixture weighted columns must be valid: {error}"));
    let (az_constant, bz_constant, _) = matrices
        .public_column_contributions(&row_weights, const_column(), Fr::from_u64(1))
        .unwrap_or_else(|error| panic!("fixture constant columns must be valid: {error}"));

    let mut openings = vec![Fr::from_u64(0); dimensions.variables().len()];
    if az_constant != Fr::from_u64(0) {
        set_opening_to_cancel_linear_form(
            &mut openings,
            az_constant,
            &weighted.a,
            dimensions.variables(),
        );
    } else if bz_constant != Fr::from_u64(0) {
        set_opening_to_cancel_linear_form(
            &mut openings,
            bz_constant,
            &weighted.b,
            dimensions.variables(),
        );
    }

    let linear_forms = SpartanOuterLinearForms {
        az_coefficients: weighted.a,
        bz_coefficients: weighted.b,
        az_constant,
        bz_constant,
    };
    let az = eval_linear_form(
        &linear_forms.az_coefficients,
        linear_forms.az_constant,
        &openings,
    );
    let bz = eval_linear_form(
        &linear_forms.bz_coefficients,
        linear_forms.bz_constant,
        &openings,
    );
    assert!(
        az == Fr::from_u64(0) || bz == Fr::from_u64(0),
        "fixture Stage 1 openings must zero one Spartan outer linear form"
    );

    openings
}

fn set_opening_to_cancel_linear_form(
    openings: &mut [Fr],
    constant: Fr,
    coefficients: &[Fr],
    variables: &[JoltVirtualPolynomial],
) {
    for (index, (&coefficient, &variable)) in coefficients.iter().zip(variables).enumerate() {
        if matches!(
            variable,
            JoltVirtualPolynomial::Product
                | JoltVirtualPolynomial::ShouldBranch
                | JoltVirtualPolynomial::ShouldJump
                | JoltVirtualPolynomial::RamReadValue
                | JoltVirtualPolynomial::RamWriteValue
                | JoltVirtualPolynomial::RamAddress
                | JoltVirtualPolynomial::LookupOutput
                | JoltVirtualPolynomial::LeftLookupOperand
                | JoltVirtualPolynomial::RightLookupOperand
                | JoltVirtualPolynomial::LeftInstructionInput
                | JoltVirtualPolynomial::RightInstructionInput
        ) {
            continue;
        }
        if coefficient == Fr::from_u64(0) {
            continue;
        }
        let Some(inverse) = coefficient.inverse() else {
            continue;
        };
        openings[index] = -constant * inverse;
        return;
    }
    panic!("fixture could not cancel Spartan outer linear form");
}

fn eval_linear_form(coefficients: &[Fr], constant: Fr, openings: &[Fr]) -> Fr {
    coefficients
        .iter()
        .zip(openings)
        .fold(constant, |acc, (&coefficient, &opening_claim)| {
            acc + coefficient * opening_claim
        })
}

fn spartan_outer_opening(
    polynomial: JoltVirtualPolynomial,
) -> jolt_claims::protocols::jolt::JoltOpeningId {
    jolt_claims::protocols::jolt::JoltOpeningId::virtual_polynomial(
        polynomial,
        JoltStageId::SpartanOuter,
    )
}

fn transcript_after_commitments(
    preprocessing: &TestPreprocessing,
    public_io: &JoltDevice,
    proof: &TestProof,
) -> Blake2bTranscript {
    let mut transcript = Blake2bTranscript::new(b"Jolt");

    append_labeled_bytes(
        &mut transcript,
        b"preprocessing_digest",
        &preprocessing.preprocessing_digest,
    );
    append_labeled_u64(
        &mut transcript,
        b"max_input_size",
        public_io.memory_layout.max_input_size,
    );
    append_labeled_u64(
        &mut transcript,
        b"max_output_size",
        public_io.memory_layout.max_output_size,
    );
    append_labeled_u64(
        &mut transcript,
        b"heap_size",
        public_io.memory_layout.heap_size,
    );
    append_labeled_bytes(&mut transcript, b"inputs", &public_io.inputs);
    append_labeled_bytes(&mut transcript, b"outputs", &public_io.outputs);
    append_labeled_u64(&mut transcript, b"panic", public_io.panic as u64);
    append_labeled_u64(&mut transcript, b"ram_K", proof.ram_K as u64);
    append_labeled_u64(&mut transcript, b"trace_length", proof.trace_length as u64);
    append_labeled_u64(
        &mut transcript,
        b"entry_address",
        preprocessing.program.bytecode.entry_address,
    );
    append_labeled_u64(
        &mut transcript,
        b"ram_rw_phase1_num_rounds",
        proof.rw_config.ram_rw_phase1_num_rounds as u64,
    );
    append_labeled_u64(
        &mut transcript,
        b"ram_rw_phase2_num_rounds",
        proof.rw_config.ram_rw_phase2_num_rounds as u64,
    );
    append_labeled_u64(
        &mut transcript,
        b"registers_rw_phase1_num_rounds",
        proof.rw_config.registers_rw_phase1_num_rounds as u64,
    );
    append_labeled_u64(
        &mut transcript,
        b"registers_rw_phase2_num_rounds",
        proof.rw_config.registers_rw_phase2_num_rounds as u64,
    );
    append_labeled_u64(
        &mut transcript,
        b"log_k_chunk",
        proof.one_hot_config.log_k_chunk as u64,
    );
    append_labeled_u64(
        &mut transcript,
        b"lookups_ra_virtual_log_k_chunk",
        proof.one_hot_config.lookups_ra_virtual_log_k_chunk as u64,
    );
    append_labeled_u64(
        &mut transcript,
        b"dory_layout",
        proof.trace_polynomial_order.transcript_scalar(),
    );

    for commitment in &proof.commitments {
        append_payload_label(&mut transcript, b"commitment", commitment);
        transcript.append(commitment);
    }

    transcript
}

fn append_labeled_bytes(transcript: &mut Blake2bTranscript, label: &'static [u8], bytes: &[u8]) {
    transcript.append(&LabelWithCount(label, bytes.len() as u64));
    transcript.append_bytes(bytes);
}

fn append_labeled_u64(transcript: &mut Blake2bTranscript, label: &'static [u8], value: u64) {
    transcript.append(&jolt_transcript::Label(label));
    transcript.append(&U64Word(value));
}

fn append_payload_label<A>(transcript: &mut Blake2bTranscript, label: &'static [u8], payload: &A)
where
    A: AppendToTranscript,
{
    if let Some(len) = payload.transcript_payload_len() {
        transcript.append(&LabelWithCount(label, len));
    } else {
        transcript.append(&jolt_transcript::Label(label));
    }
}
