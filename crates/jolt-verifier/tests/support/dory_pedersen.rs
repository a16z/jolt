#![expect(
    clippy::panic,
    reason = "test fixture construction should fail loudly if the synthetic proof is malformed"
)]

use common::jolt_device::{JoltDevice, MemoryLayout};
use jolt_claims::protocols::jolt::{
    formulas::spartan::{
        SpartanOuterDimensions, SpartanOuterLinearForms, SpartanOuterRemainderPlan,
    },
    JoltOneHotConfig, JoltReadWriteConfig, JoltStageId, JoltVirtualPolynomial,
};
use jolt_crypto::{Bn254G1, DeriveSetup, Pedersen, PedersenSetup, VectorCommitment};
use jolt_dory::{DoryCommitment, DoryProof, DoryScheme, DoryVerifierSetup};
use jolt_field::{Fr, FromPrimitiveInt, Invertible};
use jolt_openings::{append_opening_claim, CommitmentScheme, ZkOpeningScheme};
use jolt_poly::{CompressedPoly, Polynomial, UnivariatePoly};
use jolt_program::preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing};
use jolt_r1cs::constraints::rv64::{const_column, input_column, rv64_eq_constraints};
use jolt_sumcheck::{
    append_sumcheck_claim, ClearProof, ClearSumcheckProof, CommittedOutputClaims, CommittedRound,
    CommittedSumcheckProof, CompressedSumcheckProof, LabeledRoundPoly, SumcheckClaim,
    SumcheckProof, SumcheckVerifier, SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, LabelWithCount, Transcript, U64Word};
use jolt_verifier::{
    compat::claims::attach_opening_claims,
    proof::{JoltStageProofs, TracePolynomialOrder},
    verify, JoltProof, JoltVerifierPreprocessing, VerifierError,
};

const DORY_NUM_VARS: usize = 4;
const VC_CAPACITY: usize = 4;

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
    let opening_claims = stage1_opening_claims(&preprocessing, &public_io, &proof);
    attach_opening_claims(&mut proof, opening_claims);

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
        blindfold_proof,
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
        TracePolynomialOrder::CycleMajor,
    )
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
    MemoryLayout {
        max_input_size: 8,
        max_output_size: 8,
        heap_size: 8,
        ..MemoryLayout::default()
    }
}

fn stage_proofs(
    is_zk: bool,
    vc_setup: &PedersenSetup<Bn254G1>,
) -> JoltStageProofs<Fr, TestVectorCommitment> {
    JoltStageProofs {
        stage1_uni_skip_first_round_proof: uniskip_proof(is_zk, vc_setup),
        stage1_sumcheck_proof: sumcheck_proof(is_zk, vc_setup),
        stage2_uni_skip_first_round_proof: uniskip_proof(is_zk, vc_setup),
        stage2_sumcheck_proof: sumcheck_proof(is_zk, vc_setup),
        stage3_sumcheck_proof: sumcheck_proof(is_zk, vc_setup),
        stage4_sumcheck_proof: sumcheck_proof(is_zk, vc_setup),
        stage5_sumcheck_proof: sumcheck_proof(is_zk, vc_setup),
        stage6_sumcheck_proof: sumcheck_proof(is_zk, vc_setup),
        stage7_sumcheck_proof: sumcheck_proof(is_zk, vc_setup),
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

fn sumcheck_proof(is_zk: bool, vc_setup: &PedersenSetup<Bn254G1>) -> SumcheckProof<Fr, Bn254G1> {
    if is_zk {
        SumcheckProof::Committed(committed_sumcheck_proof(vc_setup))
    } else {
        SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof {
            round_polynomials: vec![CompressedPoly::new(vec![Fr::from_u64(0)])],
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

fn stage1_opening_claims(
    preprocessing: &TestPreprocessing,
    public_io: &JoltDevice,
    proof: &TestProof,
) -> Vec<(jolt_claims::protocols::jolt::JoltOpeningId, Fr)> {
    let mut transcript = transcript_after_commitments(preprocessing, public_io, proof);
    let log_t = proof.trace_length.ilog2() as usize;
    let dimensions = SpartanOuterDimensions::rv64(log_t);
    let _tau = transcript.challenge_vector(log_t + 2);

    let uniskip_challenge = verify_fixture_uniskip(proof, &mut transcript);
    append_opening_claim(&mut transcript, &Fr::from_u64(0));
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
    }

    claims
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
        set_opening_to_cancel_linear_form(&mut openings, az_constant, &weighted.a);
    } else if bz_constant != Fr::from_u64(0) {
        set_opening_to_cancel_linear_form(&mut openings, bz_constant, &weighted.b);
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

fn set_opening_to_cancel_linear_form(openings: &mut [Fr], constant: Fr, coefficients: &[Fr]) {
    for (index, &coefficient) in coefficients.iter().enumerate() {
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
        u8::from(proof.trace_polynomial_order) as u64,
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
